#include <stdio.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define THREADS_PER_BLOCK 1024
#define NUM_ELEMS 8000

__global__ void set_and_shift_kernel(float *send_data, float *recv_data, int num_elems, int mype,
                                     int npes) {
    int block_offset = blockIdx.x * blockDim.x;
    int thread_idx = block_offset + threadIdx.x;
    if (thread_idx < num_elems) send_data[thread_idx] = mype;

    int peer = (mype + 1) % npes;
    /* Every thread in block 0 calls nvshmemx_float_put_block. Alternatively,
       every thread can call shmem_float_p, but shmem_float_p has a disadvantage
       that when the destination GPU is connected via IB, there will be one rma
       message for every single element which can be detrimental to performance.
       And the disadvantage with shmem_float_put is that when the destination GPU is p2p
       connected, it cannot leverage multiple threads to copy the data to the destination
       GPU. */
    
    // NOTE: this is a non-blocking nvshmem op
    // thus we need to additional nvshmem synchronize to wait it all compeleted
    nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                             min(blockDim.x, num_elems - block_offset),
                             peer); /* All threads in a block call the API
                                       with the same arguments */
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    float *host; float *send_data, *recv_data;
    int num_elems = NUM_ELEMS; int num_blocks;
    cudaStream_t stream;

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    float ref = (mype + npes - 1) % npes;

    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

    host = (float *)malloc(sizeof(float) * num_elems);
    send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    num_blocks = (num_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, recv_data, num_elems, mype,
                                                            npes);
    nvshmemx_barrier_all_on_stream(stream);

    /* Do data validation */
    CUDA_CHECK(cudaMemcpyAsync(host, recv_data, num_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool success = true;
    for (int i = 0; i < num_elems; ++i) {
        if (host[i] != ref) {
            printf("Error at idx[%d] of rank[%d]: actual: %f | expect: %f\n", i, mype, host[i], ref);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[%d of %d] run complete \n", mype, npes);
    } else {
        printf("[%d of %d] run failure \n", mype, npes);
    }

    cudaStreamDestroy(stream);

    free(host);
    nvshmem_free(send_data);
    nvshmem_free(recv_data);

    nvshmem_finalize();

    return 0;
}