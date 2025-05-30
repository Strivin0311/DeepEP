#include <stdio.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#ifdef NVSHMEM_MPI_SUPPORT
#include <mpi.h>
#endif

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


__global__ void ring_reduce(int* nvs_data, int mype, int npes) {
    int peer = (mype + 1) % npes; // global pe number of next peer
    int reduce_value = mype; // init current reduced value to mype number
    
    // (npes-1)x ring p2p to perform ring all-reduce
    for (int i = 1; i < npes; i++) {
        // send my current reduced value to next peer into its nvs_data
        // and recv the current reduced value from prev peer into my nvs_data
        nvshmem_int_p(nvs_data, reduce_value, peer);
        // barrier all pes to wait `nvshmem_int_p` is finished
        // REVIEW: why barrier is necessary 
        // and can not be relaxed by neither `nvshmem_fence` nor `nvshmem_quiet`?
        nvshmem_barrier_all();
        // add `mype` to the reduced value from prev peer
        // and use it as my current reduced value in next iteration
        reduce_value = *nvs_data + mype;
        // barrier all pes to wait for reduced value to be updated
        // before launching next `nvshmem_int_p`
        // REVIEW: why barrier is necessary here ?
        nvshmem_barrier_all();
    }

    // store my final reduced value to nvs_data
    *nvs_data = reduce_value;
}


int main(int argc, char** argv) {
    int mype, npes, mype_node;
    cudaStream_t stream;

#ifdef NVSHMEM_MPI_SUPPORT
    bool use_mpi = false;
    char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef NVSHMEM_MPI_SUPPORT
    if (use_mpi) { 
        // init mpi first before nvshmem
        MPI_Init(&argc, &argv);
        int rank, nranks;
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &nranks);

        printf("[RANK%d] Hello, World! And the number of MPI ranks: %d\n", rank, nranks);

        // init nvshmem with mpi
        nvshmemx_init_attr_t nvs_attr;
        nvs_attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &nvs_attr);
    } else
        nvshmem_init();
#else
    nvshmem_init();
#endif

    // get my PE number (global and local) and number of PEs
    mype = nvshmem_my_pe(); // global pe number
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE); // local pe number

    printf("[PE%d] Hello, World! And the number of PEs: %d\n", mype, npes);

    // set device
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // allocate memory on the host and symmetric heap
    // and initialize to zero
    int *host_data = (int *)calloc(1, sizeof(int));
    int *nvs_data = (int *)nvshmem_calloc(1, sizeof(int));

    // prepare kernel args and meta info
    void* kernel_args[] = {&nvs_data, &mype, &npes};
    dim3 dimGrid(1); dim3 dimBlock(1);

    // launch kernel
    NVSHMEM_CHECK(
        nvshmemx_collective_launch(
            (const void*) ring_reduce,
            dimGrid,
            dimBlock,
            kernel_args,
            0,
            stream
        )
    );

    // wait the final reduced value is stored
    // CUDA_CHECK(cudaDeviceSynchronize());
    nvshmemx_barrier_all_on_stream(stream);

    // print results
    CUDA_CHECK(cudaMemcpyAsync(host_data, nvs_data, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("reduced value on device [%d] is %d \n", mype_node, *host_data);

    // destroy stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // free memory
    nvshmem_free(nvs_data);
    free(host_data);

    // finalize nvshmem
    nvshmem_finalize();

#ifdef NVSHMEM_MPI_SUPPORT
    // finalize mpi
    if (use_mpi) MPI_Finalize();
#endif

    return 0;
}