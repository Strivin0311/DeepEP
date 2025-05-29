#include <stdio.h>
#include <iostream>
#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <mpi.h>


__global__ void ring_shift(int* nvs_msg) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    // send `mype` id as message to my next peer
    nvshmem_int_p(nvs_msg, mype, peer);
}


int main(int argc, char** argv) {
    int rank, num_ranks, num_devices;
    int mype_node, npes_node, msg;
    cudaStream_t stream;

    // init mpi first
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_ranks);

    std::cout << "[RANK" << rank << "] " << "Hello, World! And the number of MPI ranks: " << num_ranks << std::endl;
    
    // init nvshmem with mpi
    nvshmemx_init_attr_t nvs_attr;
    nvs_attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &nvs_attr);

    // get my PE number and number of PEs
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    std::cout << "[PE" << mype_node << "] " << "Hello, World! And the number of PEs: " << npes_node << std::endl;

    // set device and stream
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
    cudaStreamCreate(&stream);

    // allocate memory on the symmetric heap
    int* nvs_msg = (int*) nvshmem_malloc(sizeof(int));

    // launch kernel
    ring_shift<<<1, 1, 0, stream>>>(nvs_msg);

    // barrier the stream to ensure all nvshmem ops are completed
    // i.e. nvs_msg is prepared
    nvshmemx_barrier_all_on_stream(stream);

    // copy msg to host
    cudaMemcpyAsync(&msg, nvs_msg, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // cpu sync all ops on stream to be finished
    // i.e. msg is prepared
    cudaStreamSynchronize(stream);

    // print msg
    printf("[PE%d] received message: %d\n", mype_node, msg);

    // free memory on the symmetric heap
    nvshmem_free(nvs_msg);

    // finalize nvshmem
    nvshmem_finalize();

    // finalize mpi last
    MPI_Finalize();

    return 0;
}