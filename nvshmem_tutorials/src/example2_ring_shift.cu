#include <stdio.h>
#include <iostream>
#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>


__global__ void ring_shift(int* nvs_msg) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    // send `mype` to my next peer and recv `mype-1` from my prev peer
    // since this (one-sided) api is launched on device-side:
    // 1. if peer is in this node, then we use cuda kernel to copy message
    // 2. if peer is in other node, then:
    //  2-1. if ibgda enabled, cuda core will call `ibgda_post_send` to let nic transport message
    //  2-2. otherwise, cuda core will notify the proxy thread,
    //                      who will call `ibv_post_send` to let nic transport message through second qp

    // while, if this api is launched on host-side:
    // 1. if peer is in this node, then we use cudaMemcpyAsync to copy message
    // 2. if peer is in other node, then cpu will call `ibv_post_send` to let nic transport message through first qp
    nvshmem_int_p(nvs_msg, mype, peer);
}


int main(int argc, char** argv) {
    int mype_node, npes_node, msg;
    cudaStream_t stream;

    // init nvshmem before any nvshmem op
    nvshmem_init();

    // get my PE number and number of PEs
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    std::cout << "[PE" << mype_node << "] " << "Hello, World! And the number of PEs: " << npes_node << std::endl;

    // set device and stream
    cudaSetDevice(mype_node);
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

    // destroy stream
    cudaStreamDestroy(stream);

    // free memory on the symmetric heap
    nvshmem_free(nvs_msg);

    // finalize nvshmem
    // which adds an implicit collective synchronization across PEs
    // to complete all pending communication and release all the resources
    nvshmem_finalize();

    return 0;
}