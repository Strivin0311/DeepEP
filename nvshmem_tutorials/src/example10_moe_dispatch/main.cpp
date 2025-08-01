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
    uint32_t localTokens = 4, /* local seqlen */
    uint32_t hiddenDim = 3,
    uint32_t numExperts = 8,
    uint32_t expertsPerToken = 2, /* topk = 2 */
    uint32_t maxNumTokens = 10 /* capacity = 10 */
) {
    // check
    assert(numExperts % world_size == 0);
    assert(numExperts / world_size == expertsPerToken);
    uint32_t numLocalExperts = numExperts / world_size;

    // init tokens on host
    // e.g. rank0 will have tokens (shape=[localTokens * hiddenDim,]):
    //      Token 0: 10 10 10 
    //      Token 1: 11 11 11 
    //      Token 2: 12 12 12 
    //      Token 3: 13 13 13 
    // and rank1 will have tokens:
    //      Token 0: 14 14 14 
    //      Token 1: 15 15 15 
    //      Token 2: 16 16 16 
    //      Token 3: 17 17 17
    std::vector<uint32_t> tokens_h(localTokens * hiddenDim);
    for (int i = 0; i < localTokens; ++i) {
        for (int j = 0; j < hiddenDim; ++j) {
            tokens_h[i * hiddenDim + j] = i + rank * localTokens + 10;
        }
    }
    
    // print expected information
    std::ofstream logFile(SHIP_LOG_PREFIX + std::to_string(rank) + ".log");
    logFile << "Total ranks (world size): " << world_size << "\n";
    logFile << "Each rank will transfer unique tokens num (local seqlen): " << localTokens << "\n";
    logFile << "Each rank have experts num (num of local experts): " << numLocalExperts << "\n";
    logFile << "Each token have experts num (topk): " << expertsPerToken << "\n";
    logFile << "Each rank will transfer repeated tokens num (local seqlen x topk): " << localTokens * expertsPerToken << "\n";
    logFile << "\n";

    // init transfer indices on host
    // where indices_h[i * expertsPerToken + j] indicates the jth expert index for local token i to send to
    // e.g. for rank0, its first token (i=0) will be send to 7th expert and 4th expert
    // and its last token (i=3) will be send to 0th expert and 5th expert
    // then indices_h[0] = 7, indices_h[1] = 4, indices_h[6] = 0, indices_h[7] = 5
    std::vector<uint32_t> indices_h(localTokens * expertsPerToken, 0);
    for (int i = 0; i < localTokens; ++i) {
        // For each token, assign it to other rank
        for (int j = 0; j < expertsPerToken; ++j) {
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
    for (int i = 0; i < localTokens; ++i) {
        for (int j = 1; j < expertsPerToken; ++j) {
            Assert(
                indices_h[i * expertsPerToken] != indices_h[i * expertsPerToken + j], 
                "The same token should not be assigned to the same expert"
            );
        }
    }
    
    // print transfer information
    print_transfer_information(tokens_h, indices_h, localTokens, hiddenDim, expertsPerToken, rank, logFile);

    // init device buffers from host
    DeviceBuffer<uint32_t> tokens_d(tokens_h);
    DeviceBuffer<uint32_t> indices_d(indices_h);
    const uint32_t hiddenDimBytes = hiddenDim * sizeof(tokens_d.getElementSize());

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