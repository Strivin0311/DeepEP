#include <stdio.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#ifdef NVSHMEM_MPI_SUPPORT
#include <mpi.h>
#endif

#define N 512
#define N_REDUCE 1
#define BLOCK_SIZE 32
#define THRESHOLD 42
#define CORRECTION 7

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


__device__ void accumulate_inner(int *nvs_input, int *nvs_partial_sum) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // thread0 initializes partial_sum to 0
    // note: since we only 
    if (idx == 0)
        *nvs_partial_sum = 0;
    __syncthreads();

    // atomic add nvs input to partial_sum
    atomicAdd(nvs_partial_sum, nvs_input[idx] == 0 ? idx : nvs_input[idx]);
}

__global__ void accumulate(int *nvs_input, int *nvs_partial_sum) {
    accumulate_inner(nvs_input, nvs_partial_sum);
}

__global__ void correct_accumulate(int *nvs_input, int *nvs_partial_sum, int *nvs_reduce_sum) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // correct each partial sum
    if (*nvs_reduce_sum > THRESHOLD)
        nvs_input[idx] -= CORRECTION;
    
    // re-accumulate
    accumulate_inner(nvs_input, nvs_partial_sum);
}


int main(int argc, char** argv) {
    int mype, npes, mype_node;
    int *nvs_input;
    int *nvs_partial_sum; int *host_partial_sum;
    int *nvs_reduce_sum; int *host_reduce_sum;
    int *nvs_correct_partial_sum; int *host_correct_partial_sum;
    
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
    nvs_input = (int *)nvshmem_malloc(N * sizeof(int));
    nvs_partial_sum = (int *)nvshmem_malloc(sizeof(int)); host_partial_sum = (int *)malloc(sizeof(int));
    nvs_reduce_sum = (int *)nvshmem_malloc(sizeof(int)); host_reduce_sum = (int *)malloc(sizeof(int));
    nvs_correct_partial_sum = (int *)nvshmem_malloc(sizeof(int)); host_correct_partial_sum = (int *)malloc(sizeof(int));

    // launch accumulate kernel on current stream
    accumulate<<<1, N, 0, stream>>>(nvs_input, nvs_partial_sum);

    // reduce all partial sum across all pes on current stream
    NVSHMEM_CHECK(
        nvshmemx_int_sum_reduce_on_stream(
            NVSHMEMX_TEAM_NODE,
            nvs_reduce_sum,
            nvs_partial_sum,
            N_REDUCE,
            stream
        )
    );

    // launch correct accumulate kernel on current stream
    correct_accumulate<<<1, N, 0, stream>>>(nvs_input, nvs_correct_partial_sum, nvs_reduce_sum);

    // launch D2H operation on current stream
    CUDA_CHECK(cudaMemcpyAsync(host_partial_sum, nvs_partial_sum, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_reduce_sum, nvs_reduce_sum, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_correct_partial_sum, nvs_correct_partial_sum, sizeof(int), cudaMemcpyDeviceToHost, stream));

    // wait for the operations issued to current stream to be finished
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("original partial sum on device [%d] is %d \n", mype, *host_partial_sum);
    printf("reduce sum on device [%d] is %d \n", mype, *host_reduce_sum);
    printf("correct partial sum on device [%d] is %d \n", mype, *host_correct_partial_sum);

    // destroy stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // free memory
    nvshmem_free(nvs_input); 
    nvshmem_free(nvs_partial_sum); free(host_partial_sum);
    nvshmem_free(nvs_reduce_sum); free(host_reduce_sum);
    nvshmem_free(nvs_correct_partial_sum); free(host_correct_partial_sum);

    // finalize nvshmem
    nvshmem_finalize();

#ifdef NVSHMEM_MPI_SUPPORT
    // finalize mpi
    if (use_mpi) MPI_Finalize();
#endif

    return 0;
}