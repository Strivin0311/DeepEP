#include <iostream>
#include <curand_kernel.h>

#include <nvshmem.h>
#include <nvshmemx.h>


inline void CUDA_CHECK  (cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Stopping..." << std::endl;
        exit(EXIT_FAILURE);
    }
}


# define N 1024*1024


__global__ void monte_carlo_kernel(int* d_hits, int seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // init curand state
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // generate (x,y) uniformly in [0.,1.]
    float x = curand_uniform(&state);
    float y = curand_uniform(&state);

    // check if (x,y) is in unit circle
    if (x*x + y*y <= 1.) {
        atomicAdd(d_hits, 1);
    }
}


int main(int argc, char** argv) {
    // init nvshmem
    nvshmem_init();

    // get my PE number and number of PEs
    int my_pe = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
    int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    std::cout << "[PE" << my_pe << "] " << "Hello, World! And the number of PEs: " << npes << std::endl;

    // set device id to my pe number
    int device = my_pe;
    CUDA_CHECK(cudaSetDevice(device));

    // allocate hits on host
    int* hits = (int*) malloc(sizeof(int));
    int* hits_total = (int*) malloc(sizeof(int));
    // allocate hits on symmetric heap
    int* d_hits = (int*) nvshmem_malloc(sizeof(int));
    int* d_hits_total = (int*) nvshmem_malloc(sizeof(int));

    // initialize hits to zero
    *hits = 0;
    CUDA_CHECK(cudaMemcpy(d_hits, hits, sizeof(int), cudaMemcpyHostToDevice));

    // get kernel info
    int num_threads_per_pe = N / npes;
    int block_size = 256;
    int num_blocks = (num_threads_per_pe + block_size - 1) / block_size;

    // launch kernel
    int seed = my_pe;
    monte_carlo_kernel<<<num_blocks, block_size>>>(d_hits, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // reduce hits with `nvshmem_int_sum_reduce`
    nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, d_hits_total, d_hits, 1);

    // copy hits from device to host
    CUDA_CHECK(cudaMemcpy(hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hits_total, d_hits_total, sizeof(int), cudaMemcpyDeviceToHost));


    // approximate pi
    float pi_approx = (float) *hits_total / (float) (N) * 4.0f;

    // print results
    std::cout << "[PE" << my_pe << "] " << "Hits: " << *hits << std::endl;

    if (my_pe == 0) {
        std::cout << "[PE" << my_pe << "] " << "Total Hits: " << *hits_total << std::endl;
        std::cout << "[PE" << my_pe << "] " << "Approximated PI: " << pi_approx << std::endl;
    }

    // free memory
    free(hits); free(hits_total);
    nvshmem_free(d_hits); nvshmem_free(d_hits_total);

    // finalize nvshmem
    nvshmem_finalize();

    return 0;
}