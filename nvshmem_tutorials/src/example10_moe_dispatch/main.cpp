#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdint>
#include <assert.h>
#include <fstream>
#include "buffer.h"
#include "intranode.h"
#include "api.h"

#ifndef SHIP_LOG_PREFIX
#define SHIP_LOG_PREFIX "log/rank_"
#endif

using namespace ship;

void testDispatch(
    // cudaStream_t stream,
    unsigned rank,
    unsigned world_size,
    uint32_t localTokens = 4,
    uint32_t hiddenDim = 3,
    uint32_t numExperts = 8,
    uint32_t expertsPerToken = 2,
    uint32_t maxNumTokens = 10
) {
    // check
    Assert(
        numExperts / world_size == expertsPerToken, 
        "Just for test, rank[i] and rank[i+1] buterfly transfer the same token"
    );
    assert(numExperts % world_size == 0);
    uint32_t numLocalExperts = numExperts / world_size;

    // init tokens
    std::vector<uint32_t> tokens_h(localTokens * hiddenDim);
    for (int i = 0; i < localTokens; i++) {
        for (int j = 0; j < hiddenDim; j++) {
            tokens_h[i * hiddenDim + j] = i + rank * localTokens + 10;
        }
    }
    
    // print expected information
    std::ofstream logFile(SHIP_LOG_PREFIX + std::to_string(rank) + ".log");
    logFile << "Total ranks: " << world_size << "\n";
    logFile << "Each rank will transfer tokens num: " << localTokens << "\n";
    logFile << "Each rank have experts num: " << numLocalExperts << "\n";
    logFile << "Each token have experts num: " << expertsPerToken << "\n";

    // init transfer indices
    std::vector<uint32_t> indices_h(localTokens * expertsPerToken, 0);
    for (int i = 0; i < localTokens; i ++) {
        // For each token, assign it to other rank
        for (int j = 0; j < expertsPerToken; j ++) {
            // indices_h[i * expertsPerToken + j] = j;

            // every odd/even rank assign all tokens to the experts which in the paired even/odd rank
            // indices_h[i * expertsPerToken + j] = (rank ^ 0x1) * numLocalExperts + j;

            // assign each token to the experts which in the same rank
            // indices_h[i * expertsPerToken + j] = (rank ^ i) * numLocalExperts + j;

            // mixed assign
            indices_h[i * expertsPerToken + j] = (rank * 17 + i * 11 + j * 13 + 23) % numExperts;
        }
    }

    // sanity check
    for (int i = 0; i < localTokens; i ++) {
        for (int j = 1; j < expertsPerToken; j ++) {
            Assert(
                indices_h[i * expertsPerToken] != indices_h[i * expertsPerToken + j], 
                "The same token should not be assigned to the same expert"
            );
        }
    }
    
    // print transfer information
    print_transfer_information(tokens_h, indices_h, localTokens, hiddenDim, expertsPerToken, rank, logFile);

    // Device buffers
    DeviceBuffer<uint32_t> tokens_d(tokens_h);
    DeviceBuffer<uint32_t> indices_d(indices_h);
    const uint32_t hiddenDimBytes = hiddenDim * sizeof(tokens_d.get()[0]);

    AllToAllIntraNode allToAllIntranode(
        rank,
        world_size,
        localTokens,
        hiddenDim,
        hiddenDimBytes,
        numExperts,
        expertsPerToken,
        maxNumTokens
    );

    logFile << "\n\n\n--------------Dispatch start----------------\n\n\n";

    allToAllIntranode.dispatch(
        tokens_d,
        indices_d,
        logFile
    );

    logFile << "\n\n\n--------------Dispatch end----------------\n\n\n";

    logFile.close();
}

int main(int argc, char **argv) {
    // init nvshmem
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // set device
    int deviceId = my_pe % 8;
    cudaSetDevice(deviceId);

    // run dispatch test
    testDispatch(my_pe, n_pes);

    // finalize nvshmem
    nvshmem_finalize();

    return 0;
}