#include <stdio.h>
#include <stdint.h>

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


__global__ void ring_broadcast(int* nvs_data, uint64_t data_len, int src, uint64_t* psync) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    int last_peer = (src + npes - 1) % npes;

    // source pe set the sync flag to 1 directly
    if (mype == src)
        *psync = 1;
    
    // other pes wait until the sync flag is set by the previous pe
    nvshmem_signal_wait_until(psync, NVSHMEM_CMP_NE, 0);

    // last pe does not need to broadcast to next, thus just return
    if (mype == last_peer)
        return;
    
    // put my data to the one of next peer's
    nvshmem_int_put(nvs_data, nvs_data, data_len, peer);
    // ensure the `put` op is issured ahead of next `signal` op
    nvshmem_fence();
    // notify the next pe to start by setting the sync flag
    nvshmemx_signal_op(psync, 1, NVSHMEM_SIGNAL_SET, peer);

    *psync = 0;
}


int main(int argc, char** argv) {
    size_t data_len = 8192;
    cudaStream_t stream;

    // init nvshmem
    nvshmem_init();

    // get my PE number (global and local)
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // set device and stream
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // allocate data memory on the symmetric heap and host
    int* nvs_data = (int*) nvshmem_malloc(data_len * sizeof(int));
    int* host_data = (int *) malloc(data_len * sizeof(int));
    uint64_t* psync = (uint64_t*) nvshmem_malloc(sizeof(uint64_t));

    // init host data and move to device
    for (size_t i = 0; i < data_len; i++)
        host_data[i] = mype + i;
    CUDA_CHECK(cudaMemcpyAsync(nvs_data, host_data, data_len * sizeof(int), cudaMemcpyHostToDevice, stream));

    // prepare kernel args
    int src = 1; assert (src < npes);
    dim3 gridDim(1), blockDim(1);
    void* kernel_args[] = {&nvs_data, &data_len, &src, &psync};

    // launch ring-broadcast kernel
    // `nvshmemx_collective_launch` function must be used to launch CUDA kernels on the GPU 
    // when the CUDA kernels use NVSHMEM synchronization or collective APIs 
    // (e.g., nvshmem_wait, nvshmem_barrier, nvshmem_barrier_all, or any other collective operation).
    NVSHMEM_CHECK(
        nvshmemx_collective_launch(
            (const void*) ring_broadcast,
            gridDim,
            blockDim,
            kernel_args,
            0,
            stream
        )
    );
    // barrier the stream to ensure all nvshmem ops are completed
    nvshmemx_barrier_all_on_stream(stream);

    // copy device data to host data
    // wait for all operations on current stream to be finished
    CUDA_CHECK(cudaMemcpyAsync(host_data, nvs_data, data_len * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // results validation
    bool success = true;
    for (size_t i = 0; i < data_len; i++)
        if (host_data[i] != i + src) {
            printf("[RANK%d] error: host_data[%zu] = %d, but expected %d\n", mype, i, host_data[i], (int) i + src);
            success = false;
        }

    if (success) {
        printf("[%d of %d] run complete \n", mype, nvshmem_n_pes());
    } else {
        printf("[%d of %d] run failure \n", mype, nvshmem_n_pes());
    }

    // destroy stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // free memory
    nvshmem_free(nvs_data); nvshmem_free(psync);
    free(host_data);
    
    // finalize nvshmem
    nvshmem_finalize();
    
    return 0;
}