#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <sys/types.h>
#include "intranode.h"
#include "cuda_utils.h"

using namespace kato;

template <bool isSend, bool isRecv>
__global__ void dispatchKernel (
	uint32_t rank,
	uint32_t worldSize,
	uint32_t localTokens,
	uint32_t hiddenDim,
	uint32_t perTokenBytes,
	uint32_t numLocalExperts,
	uint32_t expertsPerToken,
	uint32_t numExperts,
	uint32_t maxNumTokens,

	uint32_t *tokens,
	uint32_t *indices,

	uint64_t *numTokensBuffer,
	uint64_t *numRecvBuffer,
	std::byte *xDispatchOut
) {
	const unsigned WARP_SIZE = 32;
	const unsigned NUM_WARPS = blockDim.x / WARP_SIZE; /* num_warps = 10 */
	const unsigned blockId = blockIdx.x;
	const unsigned warpId = threadIdx.x / WARP_SIZE;
	const unsigned laneId = threadIdx.x % WARP_SIZE;

	// for send
	if constexpr (isSend) {
		// init tokenCount with size `numExperts` for each block
		// where tokenCount[i] identifies the number of tokens assigned to expert i of this rank.
		extern __shared__ uint32_t tokenCount[];
		for (uint32_t i = threadIdx.x; i < numExperts; i += blockDim.x) {
			tokenCount[i] = 0;
		} __syncthreads();

		// Dispatch local tokens to experts.
		if (warpId == NUM_WARPS - 1) { // warp9 in each block counts the number of tokens assigned to each expert.
			// each block's warp9 handle one dst expert
			for (int dstExpert = blockId; dstExpert < numExperts; dstExpert += gridDim.x) {
				const uint32_t dstRank = dstExpert / numLocalExperts;
				const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

				// each thread in warp9 handle one (repeated) token in indices
				unsigned dstExpertCount = 0;
				for (int i = laneId; i < localTokens * expertsPerToken; i += WARP_SIZE) {
					unsigned expert = __ldg(indices + i);
					if (expert == dstExpert) {
						dstExpertCount++;
					}
				}

				// get the sum of dstExpertCount across all threads in warp9
				// which is the total number of tokens assigned to dstExpert
				unsigned dstExpertCountSum = device::warp_sum(dstExpertCount);

				// the first thread in warp9 will set numTokensBuffer signal
				// to the value of dstExpertCountSum + 1
				// for the peer pe `dstRank` w.r.t. the dstLocalExpert
				// NOTE: +1 because we need to wait the signal to be updated at least once
				// thus if we send 0 token to the dstExpert, the signal will not be updated
				// hence we never know if the signal is ready for receving 0 token 
				// or the signal for receiving non-zero tokens is not ready
				if (laneId == 0) {
					nvshmemx_signal_op(
						// for the peer pe `dstRank`, numTokenBuffer[rank][dstLocalExpert]: 
						// this `rank` transfer dstExpertCountSum tokens to `dstLocalExpert`-th local expert within `dstRank`
						numTokensBuffer + rank * numLocalExperts + dstLocalExpert,
						dstExpertCountSum + 1,
						NVSHMEM_SIGNAL_SET,
						dstRank
					);
				}
			}
		} 
		else { // warp0-8 are responsible for sending each token to the local experts.
			const unsigned numGroupWarps = NUM_WARPS - 1; /* warp_group_size = 9 */
			const unsigned numGroupThreads = numGroupWarps * WARP_SIZE;
			for (unsigned i = 0; i < localTokens; i++) {
				// for each token, the warp0 will load the topk dst expert indices of this token
				// and increment tokenCount to count the number of tokens assigned to these dst experts
				// NOTE: for each token, in order to be saw by all blocks,
				// though only one block transmit this token, each block should count it.
				if (warpId == 0) {
					for (unsigned j=laneId; j < expertsPerToken; j += WARP_SIZE) {
						uint32_t dstExpert = __ldg(indices + i * expertsPerToken + j);
						tokenCount[dstExpert]++;
					}
				}

				// Synchronize warp0-8 within this warp group.
          		asm volatile("bar.sync 1, %0;" ::"r"(numGroupThreads));

				// If the token is assigned to this block, handle it. (Each block handles one token)
 				if (i % gridDim.x == blockIdx.x) {
					// Each warp in block transmit the token to one expert.
					 for (unsigned j = warpId; j < expertsPerToken; j += numGroupWarps) {
						// get the info about current dst expert
						const uint32_t dstExpert = __ldg(indices + i * expertsPerToken + j);
						const uint32_t dstRank = dstExpert / numLocalExperts;
						const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

						// get the token index within the dst expert's receive buffer
						const uint32_t tokenIdx = tokenCount[dstExpert] - 1;

						// send this token parallel in warp-level
						nvshmemx_putmem_signal_nbi_warp(
							// for the peer pe `dstRank`, xDispatchOut[rank][dstLocalExpert][tokenIdx]:
							// this `rank` transfer `tokenIdx`-th token to `dstLocalExpert`-th local expert within `dstRank`
							xDispatchOut + (
								rank * numLocalExperts * maxNumTokens + 
								dstLocalExpert * maxNumTokens + tokenIdx
							) * perTokenBytes, /* dst addr in bytes of peer pe */
							(std::byte *)tokens + i * perTokenBytes, /* ith token src addr in bytes of this pe */
							perTokenBytes, // data size in bytes of this token
							// for the peer pe `dstRank`, numRecvBuffer[rank][dstLocalExpert]:
							// this `rank` now transfer 1 token to `dstLocalExpert`-th local expert within `dstRank`
							numRecvBuffer + rank * numLocalExperts + dstLocalExpert, /* signal addr of peer pe */
							1,
							NVSHMEM_SIGNAL_ADD, /* signal+1 */
							dstRank /* peer pe id */
						);
					}
				}
			} 
		}
	}
	
	// for recv
	if constexpr (isRecv) {
		// each thread of the whole grid handle one expert
		for (int i = blockId * blockDim.x + threadIdx.x; i < numExperts; i += gridDim.x * blockDim.x) {
			// wait for the numTokenBuffer[i] signal to be ready (non-zero)
			// where i // numLocalExperts is the sender pe id
			// i % numLocalExperts is the local expert id
			nvshmem_uint64_wait_until(
				numTokensBuffer + i,
				NVSHMEM_CMP_NE,
			    0
			);

			// when the number of tokens received from expert i is ready (counter reached expected_num)
			// then we need to wait for the signal in numRecvBuffer[i] to be incremented to numTokens
			uint64_t numTokens = numTokensBuffer[i] - 1;
			nvshmem_uint64_wait_until(
				numRecvBuffer + i, 
				NVSHMEM_CMP_EQ,
				numTokens
			);
		}
	}
}


void AllToAllIntraNode::dispatch (
	const DeviceBuffer<uint32_t> &tokens_d,
	const DeviceBuffer<uint32_t> &indices_d,
	std::ofstream &logFile
) {
	// prepare kernel args
	constexpr unsigned NUM_WRAPS = 10;
	constexpr unsigned numThreadsperBlock = 32 * NUM_WRAPS; /* block size = 320 */
	const unsigned numBlocks = std::min((uint32_t)132, numExperts); /* grid size = num_experts, meaning one block per expert */

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsperBlock);

	void *args[] = {
		const_cast<uint32_t*>(&rank),
		const_cast<uint32_t*>(&world_size),
		const_cast<uint32_t*>(&localTokens),
		const_cast<uint32_t*>(&hiddenDim),
		const_cast<uint32_t*>(&perTokenBytes),
		const_cast<uint32_t*>(&numLocalExperts),
		const_cast<uint32_t*>(&expertsPerToken),
		const_cast<uint32_t*>(&numExperts),
		const_cast<uint32_t*>(&maxNumTokens),
		const_cast<uint32_t**>(&tokens_d.data),
		const_cast<uint32_t**>(&indices_d.data),
		&numTokensBuffer,
		&numDispatchRecvBuffer,
		&xDispatchOut
	};

	// launch kernel in cooperative mode
	// cudaLaunchCooperativeKernel(
	nvshmemx_collective_launch(
        (void *)&dispatchKernel<true, true>,
        dimGrid,
        dimBlock,
        args,
		sizeof(uint32_t) * numExperts, /* shared memory size = num_experts */
		0
    );

	// wait for kernel to finish
	cudaDeviceSynchronize();

	// copy device results to host
	uint64_t *numTokensBuffer_h = new uint64_t[numExperts];
	cudaMemcpy(
		numTokensBuffer_h,
		numTokensBuffer,
		numExperts * sizeof(uint64_t),
		cudaMemcpyDeviceToHost
	);
	uint64_t *numDispatchRecvBuffer_h = new uint64_t[numExperts];
	cudaMemcpy(
		numDispatchRecvBuffer_h,
		numDispatchRecvBuffer,
		numExperts * sizeof(uint64_t),
		cudaMemcpyDeviceToHost
	);
	std::byte *xDispatchOut_h = new std::byte[numExperts * maxNumTokens * perTokenBytes];
	cudaMemcpy(
		xDispatchOut_h,
		xDispatchOut,
		numExperts * maxNumTokens * perTokenBytes * sizeof(std::byte),
		cudaMemcpyDeviceToHost
	);

	// sanity check
	for (int i = 0; i < numExperts; i++) {
		assert(numTokensBuffer_h[i] -1 == numDispatchRecvBuffer_h[i]);
	}

	// stats and print the result
	// recvTokens_h[j]: the number of tokens received by jth local expert
	std::vector<int> recvTokens_h(numLocalExperts, 0);
	for (int i = 0; i < world_size; i++) {
		for (int j = 0; j < numLocalExperts; j++) {
			int recvTokens = numDispatchRecvBuffer_h[i * numLocalExperts + j];
			recvTokens_h[j] += recvTokens;
		}
	}

	for (int j = 0; j < numLocalExperts; j++) {
		int idxExpert = rank * numLocalExperts + j;
		logFile << "\nExpert " << idxExpert << " (Local Expert " << j << ")" << ": received " << recvTokens_h[j] << " tokens in total, details as follows:\n\n";
		for (int i = 0; i < world_size; i++) {
			for (int k = 0; k < maxNumTokens; k++) {
				// xDispatchOut_h[i * numLocalExperts * maxNumTokens * perTokenBytes + j * maxNumTokens * perTokenBytes + k * perTokenBytes]:
				// the kth token received from ranki for jth local expert
				uint32_t* first_val_ptr = (uint32_t*)(
					xDispatchOut_h + 
					i * numLocalExperts * maxNumTokens * perTokenBytes + 
					j * maxNumTokens * perTokenBytes + 
					k * perTokenBytes
				);
				if (*first_val_ptr == 0) continue;

				logFile << "> Received token from rank " << i << ": [";

				for (int h = 0; h < hiddenDim; ++h) {
					uint32_t val = *(first_val_ptr + h);
					logFile << val;
					if (h < hiddenDim - 1) {
						logFile << " ";
					}
				}
				logFile << "]\n";
			}
		}
		logFile << "\n";
	}

	// free host memory
	delete[] numTokensBuffer_h;
	delete[] numDispatchRecvBuffer_h;
	delete[] xDispatchOut_h;
}