#ifndef SHIP_INTRANODE_H 
#define SHIP_INTRANODE_H
#include <cstdint>
#include <nvshmem.h>
#include <cuda_runtime.h>
#include <fstream>
#include "api.h"
#include "buffer.h"
#include "api.h"

namespace {
    template <typename T> T *mallocZeroBuffer(size_t size) {
      T *ptr;
      cudaMalloc(&ptr, size * sizeof(T));
      cudaMemset(ptr, 0, size * sizeof(T));
      return ptr;
    }
} // namespace

namespace ship {
    enum {
        SEND,
        RECV,
    };

    // For each rank
    struct AllToAllIntraNode {
        uint32_t rank;
        uint32_t world_size;
        uint32_t localTokens;
        uint32_t hiddenDim;
        uint32_t perTokenBytes; // The number of bytes for each token
        uint32_t numExperts;      // For the whole world
        uint32_t expertsPerToken; // The number of experts per token.
        uint32_t maxNumTokens;    // Each rank be allowed to send maxNumTokens tokens
        uint32_t numLocalExperts;

        AllToAllIntraNode(
            uint32_t rank,
            uint32_t world_size,
            uint32_t localTokens, /* local seqlen */
            uint32_t hiddenDim,
            uint32_t perTokenBytes,
            uint32_t numExperts,
            uint32_t expertsPerToken, /* topk */
            uint32_t maxNumTokens /* capacity */
        ): 
            rank(rank),
            world_size(world_size),
            localTokens(localTokens),
            hiddenDim(hiddenDim),
            perTokenBytes(perTokenBytes),
            numExperts(numExperts),
            expertsPerToken(expertsPerToken),
            maxNumTokens(maxNumTokens)
        {
            Assert(numExperts % world_size == 0, "numExperts should be divisible by world_size");
            numLocalExperts = numExperts / world_size;

            // numTokensBuffer[i * numLocalExperts + j]: the num of tokens plus 1 received from ranki for jth local expert
            // as the signal
            numTokensBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numExperts);
            Assert(numTokensBuffer != nullptr, "Failed to allocate numTokensBuffer");
            cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numExperts);

            // numDispatchRecvBuffer[i * numLocalExperts + j]: the num of tokens received from ranki for jth local expert
            // as both the signal and the data
            numDispatchRecvBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numExperts);
            Assert(numDispatchRecvBuffer != nullptr, "Failed to allocate numDispatchRecvBuffer");
            cudaMemset(numDispatchRecvBuffer, 0, sizeof(uint64_t) * numExperts);

            // xDispatchOut[i * numLocalExperts * maxNumTokens + j * maxNumTokens + k]:
            // the kth token received from ranki for jth local expert
            xDispatchOut = (std::byte *)nvshmem_malloc(numExperts * maxNumTokens * perTokenBytes);
            Assert(xDispatchOut != nullptr, "Failed to allocate xDispatchOut");
        }
        
        // Storage the number of tokens for each local expert
        // Each rank will receive its tokens to the local experts
        // 64bit type for nvshmemx_signal_op
        uint64_t *numTokensBuffer = nullptr;

        // Size is similar to numTokensBuffer
        // Each rank will receive its tokens from the local experts
        uint64_t *numDispatchRecvBuffer = nullptr;

        std::byte *xDispatchOut = nullptr;

        void dispatch(
            const DeviceBuffer<uint32_t> &tokens_d,
	        const DeviceBuffer<uint32_t> &indices_d,
            std::ofstream &logFile
        );

        ~AllToAllIntraNode() {
            nvshmem_free(numTokensBuffer);
            nvshmem_free(numDispatchRecvBuffer);
            nvshmem_free(xDispatchOut);
        }
    };
}

#endif // SHIP_uint32_tRANODE_H