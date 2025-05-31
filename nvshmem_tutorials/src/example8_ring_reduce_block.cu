#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <ctype.h>

#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>


#undef CUDA_CHECK
#define CUDA_CHECK(expr)    \
    do {                    \
        cudaError_t err = (expr);   \
        if (err != cudaSuccess) {   \
            fprintf(stderr, "[%s:%d] Cuda failed with the error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(-1);   \
        } \
    } while(0)


#define NVSHMEM_CHECK(expr) \
    do {    \
        int result = (expr);    \
        if (result != NVSHMEMX_SUCCESS) {   \
            fprintf(stderr, "[%s:%d] Nvshmem failed with the error: %d\n", __FILE__, __LINE__, result);   \
        }   \
    } while(0)


#define CHUNK_SIZE 512
#define BLOCK_SIZE 256
#define NUM_BLOCKS 4
#define NUM_ELEMS 8192
#define WARMUP 0
#define REPS 1


__global__ void ring_reduce_block(int* nvs_dst, int* nvs_src, int nelems, int chunk_size, uint64_t* signal) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    int tidx = threadIdx.x; int bidx = blockIdx.x; 
    int block_size = blockDim.x; int num_blocks = gridDim.x;
    int nelems_per_block = (nelems + num_blocks - 1) / num_blocks;
    int num_chunks = (nelems_per_block + chunk_size - 1) / chunk_size;

    // move to the head ptr of this block
    int ptr_offs_this_block = bidx * nelems_per_block;
    // avoid the remaining SMs from working on out-of-bound data
    if (ptr_offs_this_block + nelems_per_block > nelems) return;
    nvs_src += ptr_offs_this_block;
    nvs_dst += ptr_offs_this_block;
    signal += bidx; // already init to 0 outside

    // ring-reduce phase
    for (int i = 0; i < num_chunks; i++) {
        if (mype != 0) { // pe0 does not need to wait for the cumulative reduced results
            // wait for the cumulative reduced results from the prev peer
            // to be ready by setting my signal to i + 1, for ith chunk iter
            if (tidx == 0) {
                nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, i + 1);
            }; __syncthreads();

            // when the cumulative reduced results are ready,
            // each thread in the block does the reduction
            // which adds my original source to the cumulative reduced results
            // and becomes the send buffer for the next peer
            for (int j = tidx; j < chunk_size; j += block_size) {
                // NOTE: thread j moves the data with a strided idx list: j, j + block_size, ...
                nvs_dst[j] += nvs_src[j];
            }; __syncthreads();
        }
        if (tidx == 0) {
            // only thread 0 in each block calls the nvshmem put
            // when the send buffer is ready
            // NOTE: the put is non-blocking, due to each chunk is independent in the PGAS,
            // thus chunk(i+1) does not wait for chunk(i) to finish
            nvshmem_int_put_signal_nbi(
                nvs_dst,
                // if mype == 0, the send buffer is directly my original source
                // otherwise, the send buffer is the summation of my original source 
                // and the cumulative reduced results recved from the prev peer
                (mype == 0) ? nvs_src : nvs_dst,
                chunk_size, // put to next peer for the ith chunk
                signal,
                1,
                // add the signal of next peer by 1
                // i.e., turn its signal from i to i + 1
                NVSHMEM_SIGNAL_ADD,
                peer
            );
        }

        // move to the head ptr of next chunk
        nvs_dst += chunk_size;
        nvs_src += chunk_size;
    }

    // ring-broadcast phase
    // now, both the pe0 and pe(n-1) have the final reduced results
    // so we need to broadcast the final reduced results 
    // started from pe0, and ended at pe(n-2)
    if (tidx > 0) return; // only let the thread 0 do the broadcast
    nvs_dst -= num_chunks * chunk_size;
    for (int i = 0; i < num_chunks; i++) {
        // both pe0 and pe1 do not need to wait for the prev peer
        if (mype < npes - 1) {
            nvshmem_signal_wait_until(
                signal,
                NVSHMEM_CMP_GE,
                (mype == 0) ? i + 1 : num_chunks + i + 1
            );
        }

        // both pe(n-1) and pe(n-2) do not need to put to the next peer
        if (mype < npes - 2) {
            nvshmem_int_put_signal_nbi(
                nvs_dst,
                nvs_dst,
                chunk_size,
                signal,
                1,
                NVSHMEM_SIGNAL_ADD,
                peer
            );
        }

        // move to the head ptr of next chunk
        nvs_dst += chunk_size;
    }
    *signal = 0; // reset for next
}


void run_iter(void** kernel_args, dim3 dimGrid, dim3 dimBlock, cudaStream_t& stream) {
    // run kernel
    nvshmemx_collective_launch(
        (const void*) ring_reduce_block,
        dimGrid,
        dimBlock,
        kernel_args,
        0,
        stream
    );
    nvshmemx_barrier_all_on_stream(stream);
}


int main(int argc, char** argv) {
    // init nvshmem
    nvshmem_init();

    // get my PE number and number of PEs
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // set device and stream
    cudaStream_t stream;
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // create cuda event for timing
    float ms;
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    // allocate memory
    int *nvs_dst = (int *) nvshmem_malloc(NUM_ELEMS * sizeof(int));
    int *nvs_src = (int *) nvshmem_malloc(NUM_ELEMS * sizeof(int));
    int *host_buf = (int *) malloc(NUM_ELEMS * sizeof(int));
    uint64_t* signal = (uint64_t *) nvshmem_calloc(NUM_BLOCKS, sizeof(uint64_t)); // each block (sm) has one signal, init to 0

    // init host buffer and move to src on device
    for (int i = 0; i < NUM_ELEMS; i++)
        host_buf[i] = i;
    CUDA_CHECK(cudaMemcpyAsync(nvs_src, host_buf, NUM_ELEMS * sizeof(int), cudaMemcpyHostToDevice, stream));
    nvshmemx_barrier_all_on_stream(stream);

    // prepare kernel args
    int nelems = NUM_ELEMS; int chunk_size = CHUNK_SIZE;
    assert (NUM_ELEMS % NUM_BLOCKS == 0); // per block takes over equal number of elements
    assert ((NUM_ELEMS / NUM_BLOCKS) % chunk_size == 0); // each chunk iteration takes over equal number of elements
    // NOTE: in this example, the number of blocks is fixed, 
    // not set automatically as the number of elements
    // i.e. the number of SMs is limited for ring-allreduce
    dim3 dimGrid(NUM_BLOCKS), dimBlock(BLOCK_SIZE);
    void* kernel_args[] = {&nvs_dst, &nvs_src, &nelems, &chunk_size, &signal};

    nvshmemx_collective_launch(
        (const void*) ring_reduce_block,
        dimGrid,
        dimBlock,
        kernel_args,
        0,
        stream
    );
    nvshmemx_barrier_all_on_stream(stream);

    // warmpup
    for (int i = 0; i < WARMUP; i++)
        run_iter(kernel_args, dimGrid, dimBlock, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // main loop
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < REPS; i++)
        run_iter(kernel_args, dimGrid, dimBlock, stream);
    CUDA_CHECK(cudaEventRecord(end, stream));

    // print timing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    printf("[RANK%d]: run all-reduce for %d elements takes %f ms\n", mype, NUM_ELEMS, ms / REPS);

    // results validation
    CUDA_CHECK(cudaMemcpyAsync(host_buf, nvs_dst, NUM_ELEMS * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bool success = true;
    for (size_t i = 0; i < NUM_ELEMS; i++) {
        if (host_buf[i] != (int) i * npes) {
            printf("Error at idx[%d] of rank[%d]: actual: %d | expect: %d\n", (int) i, mype, host_buf[i], (int) i * npes);
            success = false;
        }
    }

    if (success) {
        printf("[%d of %d] run complete \n", mype, npes);
    } else {
        printf("[%d of %d] run failure \n", mype, npes);
    }

    // destroy event
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    // destroy stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // free memory
    free(host_buf);
    nvshmem_free(nvs_dst);
    nvshmem_free(nvs_src);
    nvshmem_free(signal);

    // finalize nvshmem
    nvshmem_finalize();

    return 0;
}