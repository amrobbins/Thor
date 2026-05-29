#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include "Utilities/Expression/CudaHelpers.h"

namespace {
[[maybe_unused]] constexpr uint32_t DEFAULT_EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX = 16U;
[[maybe_unused]] constexpr uint32_t DEFAULT_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN = 1024U;
}  // namespace

struct EmbeddingSparseGradientReduceGridUpdateConfig {
    uint32_t reduceRowsPerGridX = 1U;
    uint32_t reduceGridDimY = 1U;
    uint32_t minReduceGridDimX = 1U;
    uint32_t maxReduceGridDimX = 1U;
    uint32_t maxReduceGridDimY = 1U;
};

// This needs to be done by hand and fixed.
//  scan with buffering and coalescing
// This will be a single block running all by itself, so need to maximize bandwidth. Always 1024 threads in 1 block, unless less that 1024
// seq len, in which case prune whole warps only.
//      read 4 per thread. have a shared view of 4 lines as uint4[32]. After reading all write to shared. Then will process one at a time.
// have 3 buffer offsets and 3 double length buffers, shared mem banks are 32 bits. What when I need to write 3 sequential on one thread?
//    optimize by not writing out lowrunrows, just high and ultra, and not_low_run_rows
//    well, output a sequence of offsets for high run counts, a sequence of offsets for ultra
//    then iterate over the offsets, one per thread in the high/ultra reducer
//    but how to handle the low reducer? Low receives reads 1 not-mine flag at every runOffset, not-mine threads are skipped.
//      I could pack them as bit-flags, but then I need to launch 1 low thread group for every token, some will just do nothing.
// load 4 elements
// figure out the bucket and then scan to see where to fill in shared buffer
// fill in
// When to write out? When previous iteration finished a buffer line, write it out using all threads
// update buffer base: bottom vs top
// when fully done, flush all non-empty buffers

namespace {

enum : uint32_t {
    FINALIZE_BUCKETIZE_RUN_COUNT_LINES = 2U,
    FINALIZE_BUCKETIZE_RUN_COUNT_LINE_WORDS = 4096U,

    FINALIZE_BUCKETIZE_WARP_COUNT_WORDS = 32U,
    FINALIZE_BUCKETIZE_SCALAR_WORDS = 4U,

    FINALIZE_BUCKETIZE_LOW_WRITE_BUFFER_WORDS = 2048U,
    FINALIZE_BUCKETIZE_HIGH_WRITE_BUFFER_WORDS = 2048U,
    FINALIZE_BUCKETIZE_ULTRA_WRITE_BUFFER_WORDS = 2048U,

    FINALIZE_BUCKETIZE_BUFFER_DUMP_CONTROL_WORDS = 1U,
};

constexpr size_t FINALIZE_BUCKETIZE_SHARED_WORDS =
    FINALIZE_BUCKETIZE_RUN_COUNT_LINES * FINALIZE_BUCKETIZE_RUN_COUNT_LINE_WORDS + FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +  // ultra
    FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +                                                                                 // high
    FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +                                                                                 // low
    FINALIZE_BUCKETIZE_SCALAR_WORDS + FINALIZE_BUCKETIZE_LOW_WRITE_BUFFER_WORDS + FINALIZE_BUCKETIZE_HIGH_WRITE_BUFFER_WORDS +
    FINALIZE_BUCKETIZE_ULTRA_WRITE_BUFFER_WORDS + FINALIZE_BUCKETIZE_BUFFER_DUMP_CONTROL_WORDS;

constexpr size_t FINALIZE_BUCKETIZE_SHARED_BYTES = FINALIZE_BUCKETIZE_SHARED_WORDS * sizeof(uint32_t);

static_assert(FINALIZE_BUCKETIZE_SHARED_BYTES == 57748);

}  // namespace

namespace cg = cooperative_groups;

// Takes in 1 runCount per token, indicating the number of times that token was present in the sequence.
// I should probably add a 2-stage version of this after.
__global__ void finalizeAndBucketizeEmbeddingSparseGradientRowsKernel(
    const uint32_t* __restrict__ outputRows,
    const uint32_t* __restrict__ numRuns,
    uint32_t* __restrict__ outputNumRows,
    const uint32_t* __restrict__ runCounts,
    uint32_t* __restrict__ lowRunRows,
    uint32_t* __restrict__ highRunRows,
    uint32_t* __restrict__ ultraHighRunRows,
    uint32_t* __restrict__ ultraHighRunPartialCounts,
    uint32_t* __restrict__ ultraHighRunPartialOffsets,
    uint32_t* __restrict__ numUltraHighPartials,
    uint32_t* __restrict__ numLowRunRows,
    uint32_t* __restrict__ numHighRunRows,
    uint32_t* __restrict__ numUltraHighRunRows,
    uint64_t vocabularySize,
    uint32_t lowRunMax,
    uint32_t ultraHighRunMin,
    uint32_t ultraHighTokensPerPartial,
    const cudaGraphDeviceNode_t* lowReduceNode,
    const cudaGraphDeviceNode_t* highReduceNode,
    const cudaGraphDeviceNode_t* ultraHighPartialReduceNode,
    const cudaGraphDeviceNode_t* ultraHighReduceNode,
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig) {
    extern __shared__ __align__(16) uint32_t sharedMem[];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warpGroup = cg::tiled_partition<32>(block);

    uint32_t sharedOffset = 0;

    uint32_t* runCountsLineShared[2] = {sharedMem + sharedOffset, sharedMem + sharedOffset + FINALIZE_BUCKETIZE_RUN_COUNT_LINE_WORDS};
    sharedOffset += FINALIZE_BUCKETIZE_RUN_COUNT_LINES * FINALIZE_BUCKETIZE_RUN_COUNT_LINE_WORDS;

    uint32_t* warpBaseCountsUltraShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    uint32_t* warpBaseCountsHighShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    uint32_t* warpBaseCountsLowShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    // Kept as a distinct carved region, like the old standalone shared allocation.
    uint32_t* scalarUint32Shared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_SCALAR_WORDS;

    uint32_t* lowWriteBuffer = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_LOW_WRITE_BUFFER_WORDS;

    uint32_t* highWriteBuffer = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_HIGH_WRITE_BUFFER_WORDS;

    uint32_t* ultraWriteBuffer = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_ULTRA_WRITE_BUFFER_WORDS;

    uint32_t* bufferDumpControlShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_BUFFER_DUMP_CONTROL_WORDS;

    uint32_t* validRunsShared = &(scalarUint32Shared[0]);
    // uint32_t* lowWriteBufferNextIndexShared = &(scalarUint32Shared[1]);
    // uint32_t* highWriteBufferNextIndexShared = &(scalarUint32Shared[2]);
    // uint32_t* ultraWriteBufferNextIndexShared = &(scalarUint32Shared[3]);

    uint4* runCountsVec4 = (uint4*)runCounts;

    uint32_t readLineIndex = 0;
    uint32_t writeLineIndex = 1;

    // uint32_t lowBufferWriteLine = 0;
    // uint32_t highBufferWriteLine = 0;
    // uint32_t ultraBufferWriteLine = 0;

    uint32_t myRunIndexVec4 = threadIdx.x;
    uint32_t myRunIndex = (myRunIndexVec4 << 2);
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31U;
    const bool highestThread = (threadIdx.x == blockDim.x - 1);

    if (warp == 0) {
        warpBaseCountsLowShared[lane] = 0;  // FIXME: Later check if these first three are even needed, when pruning warps
        warpBaseCountsHighShared[lane] = 0;
        warpBaseCountsUltraShared[lane] = 0;
        if (threadIdx.x == 0U) {
            // numRuns is device mem, so read as a pointer
            uint4 initialShared;
            initialShared.x = numRuns[0];  // i.e. validRunsShared
            initialShared.y = 0;
            initialShared.z = 0;
            initialShared.w = 0;
            *(uint4*)scalarUint32Shared = initialShared;
        }
    }
    __syncthreads();
    const uint32_t validRuns = validRunsShared[0];

    // Pre-fetch
    uint4* runCountsLineSharedVec4[2] = {(uint4*)runCountsLineShared[0], (uint4*)runCountsLineShared[1]};
    if (myRunIndex <= validRuns + 3) {
        runCountsLineSharedVec4[0][threadIdx.x] = runCountsVec4[myRunIndexVec4];
    }

    // const uint32_t lLaneMaskLt = (lane == 0U) ? 0U : ((1U << lane) - 1U);
    const uint32_t leLaneMaskLt = (1ULL << (lane + 1)) - 1U;

    __syncthreads();

    for (uint32_t i = 0; i < validRuns; i += 4096) {
        if (myRunIndex <= validRuns + 3) {
            // Read ahead
            runCountsLineSharedVec4[writeLineIndex][threadIdx.x] = runCountsVec4[myRunIndexVec4];
        }

        // Iterate all warps until the first thread of the first warp is past the last valid run index.
        uint32_t blockFirstRunIndex = myRunIndex & ~0x3FF;
        for (uint32_t j = threadIdx.x; j < 4096 && blockFirstRunIndex + j < validRuns; j += 1024) {
            const uint32_t myRunCount = runCountsLineShared[readLineIndex][j];

            bool isUltra;
            bool isHigh;
            bool isLow;
            if (myRunIndex < validRuns) {
                isUltra = (myRunCount >= DEFAULT_EMBEDDING_SPARSE_GRADIENT_ULTRA_HIGH_RUN_MIN);
                isHigh = (myRunCount > DEFAULT_EMBEDDING_SPARSE_GRADIENT_LOW_RUN_MAX) && !isUltra;
                isLow = !(isHigh || isUltra);
            } else {
                isUltra = false;
                isHigh = false;
                isLow = false;
            }

            const uint32_t ultraMask = __ballot_sync(0xFFFFFFFF, isUltra);
            const uint32_t highMask = __ballot_sync(0xFFFFFFFF, isHigh);

            uint32_t numUltraWithThreadIdxLessThanIncludingMineInMyWarp = __popc(ultraMask & leLaneMaskLt);
            uint32_t numHighWithThreadIdxLessThanIncludingMineInMyWarp = __popc(highMask & leLaneMaskLt);
            const uint32_t numLowWithThreadIdxLessThanIncludingMineInMyWarp =
                lane - (numUltraWithThreadIdxLessThanIncludingMineInMyWarp + numHighWithThreadIdxLessThanIncludingMineInMyWarp);

            // Then thread31 know the stats for the whole warp
            // so the pattern is like this. Scan intra-warp, write to shared.
            if (lane == 31) {
                warpBaseCountsUltraShared[warp] = numUltraWithThreadIdxLessThanIncludingMineInMyWarp;
                warpBaseCountsHighShared[warp] = numHighWithThreadIdxLessThanIncludingMineInMyWarp;
                warpBaseCountsLowShared[warp] = numLowWithThreadIdxLessThanIncludingMineInMyWarp;
            }
            uint4 sharedValues = *(uint4*)scalarUint32Shared;
            uint32_t lowBufferNextIndex = sharedValues.y;
            uint32_t highBufferNextIndex = sharedValues.z;
            uint32_t ultraBufferNextIndex = sharedValues.w;
            __syncthreads();

            // Transform from per-warp count to cumulative ordered counts from prior warps.
            if (warp == 0) {
                uint32_t buffer;
                buffer = cooperative_groups::exclusive_scan(warpGroup, warpBaseCountsUltraShared[lane]);
                warpBaseCountsUltraShared[lane] = buffer;
                buffer = cooperative_groups::exclusive_scan(warpGroup, warpBaseCountsHighShared[lane]);
                warpBaseCountsHighShared[lane] = buffer;
                buffer = cooperative_groups::exclusive_scan(warpGroup, warpBaseCountsLowShared[lane]);
                warpBaseCountsLowShared[lane] = buffer;
            }
            __syncthreads();

            // Now all threads know where in the output buffer to write their entry
            // all threads write their entries
            // going to need to track the current next open slot per output buffer
            // once more than 1024 entries in low buffer, dump section of double buffer as vector4 writes from lowest 8 warps
            // once more than 1024 entries in high buffer, dump section of double buffer as vector4 writes from warps 8 - 15 warps
            // once more than 1024 entries in ultra buffer, dump section of double buffer as vector4 writes from warps 16 - 23 warps
            //    But how do I know the globally highest buffer slot that was written into?
            //      Highest numbered thread knows.
            // FIXME double output buffer, rollover, maybe make single dimensional, and roll from 2047 -> 0
            if (isLow) {
                uint32_t writeBufferIndex = (warpBaseCountsLowShared[warp] + numLowWithThreadIdxLessThanIncludingMineInMyWarp - 1) & 0x7FF;
                lowWriteBuffer[writeBufferIndex] = myRunIndex;
            } else if (isHigh) {
                uint32_t writeBufferIndex =
                    (warpBaseCountsHighShared[warp] + numHighWithThreadIdxLessThanIncludingMineInMyWarp - 1) & 0x7FF;
                highWriteBuffer[writeBufferIndex] = myRunIndex;
            } else if (isUltra) {
                uint32_t writeBufferIndex =
                    (warpBaseCountsUltraShared[warp] + numUltraWithThreadIdxLessThanIncludingMineInMyWarp - 1) & 0x7FF;
                ultraWriteBuffer[writeBufferIndex] = myRunIndex;
            }

            // If crossed the boundary from [1023] to [1024] via buffer write, dump left buffer.
            // Else if crossed the boundary from [2047] to [0] via buffer write, dump right buffer.
            if (highestThread) {
                uint4 sharedUpdate;
                sharedUpdate.x = validRuns;
                sharedUpdate.y =
                    (lowBufferNextIndex + (warpBaseCountsLowShared[warp] + numLowWithThreadIdxLessThanIncludingMineInMyWarp)) & 0x7FF;
                sharedUpdate.z =
                    (highBufferNextIndex + (warpBaseCountsHighShared[warp] + numHighWithThreadIdxLessThanIncludingMineInMyWarp)) & 0x7FF;
                sharedUpdate.w =
                    (ultraBufferNextIndex + (warpBaseCountsUltraShared[warp] + numUltraWithThreadIdxLessThanIncludingMineInMyWarp)) & 0x7FF;
                *(uint4*)scalarUint32Shared = sharedUpdate;

                // Here I know the new span of the buffers that were written to.
                signed char dumpLow = 0;
                signed char dumpHigh = 0;
                signed char dumpUltra = 0;
                if (lowBufferNextIndex <= 1023 && sharedUpdate.y > 1023) {
                    // When index 1023 was just written.
                    dumpLow = 1;
                } else if (lowBufferNextIndex >= 1024 && sharedUpdate.y < 1024) {
                    // When index 2047 was just written.
                    dumpLow = 2;
                }
                if (highBufferNextIndex <= 1023 && sharedUpdate.z > 1023) {
                    dumpHigh = 1;
                } else if (highBufferNextIndex >= 1024 && sharedUpdate.z < 1024) {
                    dumpHigh = 2;
                }
                if (ultraBufferNextIndex <= 1023 && sharedUpdate.y > 1023) {
                    dumpUltra = 1;
                } else if (ultraBufferNextIndex >= 1024 && sharedUpdate.y < 1024) {
                    dumpUltra = 2;
                }
                char4 dumpControl{dumpLow, dumpHigh, dumpUltra, 0};
                ((char4*)bufferDumpControlShared)[0] = dumpControl;
            }

            myRunIndex += 1024;
        }
        myRunIndexVec4 = myRunIndex >> 2;
        writeLineIndex = !writeLineIndex;
        readLineIndex = !readLineIndex;
        __syncthreads();
    }

    // Remember when done looping dump all non-empty buffers
}

void launch() {
    static bool didSetSharedMemoryAttribute = false;
    if (!didSetSharedMemoryAttribute) {
        for (uint32_t i = 0; i < MachineEvaluator::instance().getNumGpus(); ++i) {
            ScopedGpu scopedGpu(i);
            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(FINALIZE_BUCKETIZE_SHARED_BYTES)));
        }
        didSetSharedMemoryAttribute = true;
    }

    // ...
}
