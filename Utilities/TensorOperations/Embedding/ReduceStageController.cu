#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {

namespace cg = cooperative_groups;

constexpr uint32_t FINALIZE_BUCKETIZE_THREADS = 1024U;
constexpr uint32_t FINALIZE_BUCKETIZE_WARPS = FINALIZE_BUCKETIZE_THREADS / 32U;
constexpr uint32_t FINALIZE_BUCKETIZE_LINE_WORDS = 4096U;
constexpr uint32_t FINALIZE_BUCKETIZE_BUFFER_WORDS = 2048U;
constexpr uint32_t FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS = 1024U;
constexpr uint32_t FINALIZE_BUCKETIZE_BUFFER_HALF_VEC4S = FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS / 4U;
constexpr uint32_t FINALIZE_BUCKETIZE_BUFFER_MASK = FINALIZE_BUCKETIZE_BUFFER_WORDS - 1U;

struct FinalizeBucketizeSharedState {
    uint32_t validRuns;
    uint32_t useTwoStage;
    uint32_t lowTotal;
    uint32_t highTotal;
    uint32_t ultraTotal;
    uint32_t ultraPartialTotal;
};

constexpr uint32_t FINALIZE_BUCKETIZE_RUN_COUNT_LINES = 2U;
constexpr uint32_t FINALIZE_BUCKETIZE_WARP_COUNT_WORDS = FINALIZE_BUCKETIZE_WARPS + 1U;
constexpr uint32_t FINALIZE_BUCKETIZE_SHARED_STATE_WORDS = 8U;
constexpr uint32_t FINALIZE_BUCKETIZE_DUMP_CONTROL_WORDS = 1U;

constexpr uint32_t FINALIZE_BUCKETIZE_SHARED_WORDS =
    FINALIZE_BUCKETIZE_RUN_COUNT_LINES * FINALIZE_BUCKETIZE_LINE_WORDS + FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +  // ultra run counts
    FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +                                                                       // high run counts
    FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +                                                                       // low run counts
    FINALIZE_BUCKETIZE_WARP_COUNT_WORDS +                                                                       // ultra partial counts
    FINALIZE_BUCKETIZE_SHARED_STATE_WORDS + FINALIZE_BUCKETIZE_BUFFER_WORDS +                                   // low circular write buffer
    FINALIZE_BUCKETIZE_BUFFER_WORDS +  // high circular write buffer
    FINALIZE_BUCKETIZE_BUFFER_WORDS +  // ultra circular write buffer
    FINALIZE_BUCKETIZE_DUMP_CONTROL_WORDS;

constexpr uint32_t FINALIZE_BUCKETIZE_SHARED_BYTES = FINALIZE_BUCKETIZE_SHARED_WORDS * sizeof(uint32_t);
static_assert(FINALIZE_BUCKETIZE_THREADS == 1024U);
static_assert(FINALIZE_BUCKETIZE_WARPS == 32U);
static_assert(FINALIZE_BUCKETIZE_WARP_COUNT_WORDS == 33U);
static_assert(FINALIZE_BUCKETIZE_LINE_WORDS == FINALIZE_BUCKETIZE_THREADS * 4U);
static_assert(FINALIZE_BUCKETIZE_BUFFER_WORDS == 2U * FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS);
static_assert((FINALIZE_BUCKETIZE_BUFFER_WORDS & FINALIZE_BUCKETIZE_BUFFER_MASK) == 0U);
static_assert(sizeof(FinalizeBucketizeSharedState) == 6U * sizeof(uint32_t));
static_assert(FINALIZE_BUCKETIZE_SHARED_STATE_WORDS % 4U == 0U);
static_assert(FINALIZE_BUCKETIZE_SHARED_BYTES == 57908U);
static_assert(EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK <= 0xffffffffULL,
              "Two-stage embedding sparse-gradient finalize block size must fit in uint32_t.");

constexpr uint32_t TWO_STAGE_FINALIZE_THREADS = 1024U;
constexpr uint32_t TWO_STAGE_FINALIZE_WARPS = TWO_STAGE_FINALIZE_THREADS / 32U;
constexpr uint32_t TWO_STAGE_FINALIZE_WARP_COUNT_WORDS = TWO_STAGE_FINALIZE_WARPS + 1U;
constexpr uint32_t TWO_STAGE_FINALIZE_RUNS_PER_BLOCK = static_cast<uint32_t>(EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK);
static_assert(TWO_STAGE_FINALIZE_THREADS == 1024U);
static_assert(TWO_STAGE_FINALIZE_WARPS == 32U);
static_assert(TWO_STAGE_FINALIZE_WARP_COUNT_WORDS == 33U);

#ifndef NDEBUG
#define THOR_DEVICE_TRAP_IF(cond) \
    do {                          \
        if (cond) {               \
            asm("trap;");         \
        }                         \
    } while (0)
#else
#define THOR_DEVICE_TRAP_IF(cond) \
    do {                          \
        if (cond) {               \
            asm("trap;");         \
        }                         \
    } while (0)
#endif

// FIXME: Put back
// #define THOR_DEVICE_TRAP_IF(cond)
//     do {
//     } while (0)

__device__ __forceinline__ cudaGraphDeviceNode_t loadEmbeddingSparseGradientTargetNode(const cudaGraphDeviceNode_t* targetNode) {
    THOR_DEVICE_TRAP_IF(targetNode == nullptr);
    cudaGraphDeviceNode_t node = *targetNode;
    THOR_DEVICE_TRAP_IF(node == nullptr);
    return node;
}

__device__ __forceinline__ uint32_t checkedEmbeddingSparseGradientGridDim(uint64_t value, uint32_t minGrid, uint32_t maxGrid) {
    THOR_DEVICE_TRAP_IF(minGrid == 0U || maxGrid == 0U || minGrid > maxGrid);
    uint64_t clamped = value;
    if (clamped < static_cast<uint64_t>(minGrid)) {
        clamped = static_cast<uint64_t>(minGrid);
    }
    THOR_DEVICE_TRAP_IF(clamped > static_cast<uint64_t>(maxGrid) || clamped > 0xffffffffULL);
    return static_cast<uint32_t>(clamped);
}

__device__ __forceinline__ void setEmbeddingSparseGradientKernelNodeEnabled(const cudaGraphDeviceNode_t* node, bool enabled) {
    if (node == nullptr) {
        return;
    }
    cudaError_t status = cudaGraphKernelNodeSetEnabled(loadEmbeddingSparseGradientTargetNode(node), enabled);
    if (status != cudaSuccess) {
        asm("trap;");
    }
}

__device__ __forceinline__ void setEmbeddingSparseGradientKernelNodeGridDim(const cudaGraphDeviceNode_t* node, dim3 gridDim) {
    THOR_DEVICE_TRAP_IF(node == nullptr);
    cudaError_t status = cudaGraphKernelNodeSetGridDim(loadEmbeddingSparseGradientTargetNode(node), gridDim);
    if (status != cudaSuccess) {
        asm("trap;");
    }
}

__device__ __forceinline__ void updateEmbeddingSparseGradientBucketReduceGrid(const cudaGraphDeviceNode_t* reduceNode,
                                                                              uint64_t rows,
                                                                              EmbeddingSparseGradientReduceGridUpdateConfig config) {
    if (reduceNode == nullptr) {
        return;
    }
    THOR_DEVICE_TRAP_IF(config.reduceRowsPerGridX == 0U || config.reduceGridDimY == 0U || config.reduceGridDimY > config.maxReduceGridDimY);
    const uint64_t packedGridX =
        (rows + static_cast<uint64_t>(config.reduceRowsPerGridX) - 1ULL) / static_cast<uint64_t>(config.reduceRowsPerGridX);
    const uint32_t gridX = checkedEmbeddingSparseGradientGridDim(packedGridX, config.minReduceGridDimX, config.maxReduceGridDimX);
    setEmbeddingSparseGradientKernelNodeGridDim(reduceNode, dim3(gridX, config.reduceGridDimY, 1U));
}

__device__ __forceinline__ uint32_t ceilDivU32(uint32_t x, uint32_t y) { return (x + y - 1U) / y; }

__device__ __forceinline__ void dumpFullHalf(uint32_t totalBeforeAppend,
                                             uint32_t* __restrict__ output,
                                             const uint32_t* __restrict__ buffer,
                                             uint32_t tid) {
    if (tid >= FINALIZE_BUCKETIZE_BUFFER_HALF_VEC4S) {
        return;
    }
    const uint32_t outputScalarBase = totalBeforeAppend & ~(FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS - 1U);
    const uint32_t bufferScalarBase = outputScalarBase & FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS;
    reinterpret_cast<uint4*>(output + outputScalarBase)[tid] = reinterpret_cast<const uint4*>(buffer + bufferScalarBase)[tid];
}

__device__ __forceinline__ void dumpTail(uint32_t total, uint32_t* __restrict__ output, const uint32_t* __restrict__ buffer, uint32_t tid) {
    const uint32_t tailWords = total & (FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS - 1U);
    if (tailWords == 0U) {
        return;
    }
    const uint32_t outputScalarBase = total & ~(FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS - 1U);
    const uint32_t bufferScalarBase = outputScalarBase & FINALIZE_BUCKETIZE_BUFFER_HALF_WORDS;
    const uint32_t fullVec4s = tailWords >> 2U;
    if (tid < fullVec4s) {
        reinterpret_cast<uint4*>(output + outputScalarBase)[tid] = reinterpret_cast<const uint4*>(buffer + bufferScalarBase)[tid];
    }
    const uint32_t tailScalars = tailWords & 3U;
    if (tailScalars != 0U && tid < tailScalars) {
        const uint32_t scalarOffset = (fullVec4s << 2U) + tid;
        output[outputScalarBase + scalarOffset] = buffer[bufferScalarBase + scalarOffset];
    }
}

struct BucketAppendCounts {
    uint32_t low;
    uint32_t high;
    uint32_t ultra;
    uint32_t ultraPartials;
};

template <typename RowT>
__global__ void finalizeAndBucketizeEmbeddingSparseGradientRowsKernel(
    const RowT* __restrict__ outputRows,
    const uint32_t* __restrict__ numRuns,
    RowT* __restrict__ outputNumRows,
    const uint32_t* __restrict__ runCounts,
    uint32_t* __restrict__ lowRunRowIndices,
    uint32_t* __restrict__ highRunRowIndices,
    uint32_t* __restrict__ ultraRunRowIndices,
    uint32_t* __restrict__ ultraRunPartialCounts,
    uint32_t* __restrict__ ultraRunPartialOffsets,
    uint32_t* __restrict__ numUltraPartials,
    uint32_t* __restrict__ numLowRunRows,
    uint32_t* __restrict__ numHighRunRows,
    uint32_t* __restrict__ numUltraRunRows,
    uint64_t vocabularySize,
    uint32_t lowRunMax,
    uint32_t ultraRunMin,
    uint32_t ultraTokensPerPartial,
    const cudaGraphDeviceNode_t* lowReduceNode,
    const cudaGraphDeviceNode_t* highReduceNode,
    const cudaGraphDeviceNode_t* ultraPartialReduceNode,
    const cudaGraphDeviceNode_t* ultraReduceNode,
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraPartialReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraReduceGridConfig,
    const cudaGraphDeviceNode_t* twoStageClassifyNode,
    const cudaGraphDeviceNode_t* twoStageAccumulateNode,
    uint32_t runtimeTwoStageRunThreshold) {
    extern __shared__ __align__(16) uint32_t sharedMem[];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warpGroup = cg::tiled_partition<32>(block);

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5U;
    const uint32_t lane = tid & 31U;
    const uint32_t leLaneMask = (1U << lane) | ((1U << lane) - 1U);
    const uint32_t activeThreads = blockDim.x;
    const uint32_t activeWarps = activeThreads >> 5U;
    const uint32_t lineWords = activeThreads << 2U;

    uint32_t sharedOffset = 0;

    uint32_t* runCountsLineShared[2] = {sharedMem + sharedOffset, sharedMem + sharedOffset + FINALIZE_BUCKETIZE_LINE_WORDS};
    sharedOffset += FINALIZE_BUCKETIZE_RUN_COUNT_LINES * FINALIZE_BUCKETIZE_LINE_WORDS;

    uint32_t* warpBaseCountsUltraShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    uint32_t* warpBaseCountsHighShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    uint32_t* warpBaseCountsLowShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    uint32_t* warpBaseCountsUltraPartialShared = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_WARP_COUNT_WORDS;

    FinalizeBucketizeSharedState* sharedState = reinterpret_cast<FinalizeBucketizeSharedState*>(sharedMem + sharedOffset);
    sharedOffset += FINALIZE_BUCKETIZE_SHARED_STATE_WORDS;

    uint32_t* lowWriteBuffer = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_BUFFER_WORDS;

    uint32_t* highWriteBuffer = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_BUFFER_WORDS;

    uint32_t* ultraWriteBuffer = sharedMem + sharedOffset;
    sharedOffset += FINALIZE_BUCKETIZE_BUFFER_WORDS;

    char4* bufferDumpControlShared = reinterpret_cast<char4*>(sharedMem + sharedOffset);
    sharedOffset += FINALIZE_BUCKETIZE_DUMP_CONTROL_WORDS;

    if (tid == 0U) {
        uint32_t validRuns = numRuns[0];
        if (validRuns != 0U && static_cast<uint64_t>(outputRows[validRuns - 1U]) == vocabularySize) {
            --validRuns;
        }
        const bool useTwoStage = runtimeTwoStageRunThreshold != 0U && validRuns > runtimeTwoStageRunThreshold;
        sharedState->validRuns = validRuns;
        sharedState->useTwoStage = useTwoStage ? 1U : 0U;
        sharedState->lowTotal = 0U;
        sharedState->highTotal = 0U;
        sharedState->ultraTotal = 0U;
        sharedState->ultraPartialTotal = 0U;

        if (twoStageClassifyNode != nullptr || twoStageAccumulateNode != nullptr) {
            THOR_DEVICE_TRAP_IF(twoStageClassifyNode == nullptr || twoStageAccumulateNode == nullptr);
            if (useTwoStage) {
                const uint32_t stageBlocks = ceilDivU32(validRuns, TWO_STAGE_FINALIZE_RUNS_PER_BLOCK);
                setEmbeddingSparseGradientKernelNodeGridDim(twoStageClassifyNode, dim3(stageBlocks, 1U, 1U));
                setEmbeddingSparseGradientKernelNodeGridDim(twoStageAccumulateNode, dim3(stageBlocks, 1U, 1U));
                setEmbeddingSparseGradientKernelNodeEnabled(twoStageClassifyNode, true);
                setEmbeddingSparseGradientKernelNodeEnabled(twoStageAccumulateNode, true);
            } else {
                setEmbeddingSparseGradientKernelNodeEnabled(twoStageClassifyNode, false);
                setEmbeddingSparseGradientKernelNodeEnabled(twoStageAccumulateNode, false);
                outputNumRows[0] = static_cast<RowT>(validRuns);
            }
        } else {
            THOR_DEVICE_TRAP_IF(useTwoStage);
            outputNumRows[0] = static_cast<RowT>(validRuns);
        }
    }

    if (warp == 0U) {
        warpBaseCountsLowShared[lane] = 0U;
        warpBaseCountsHighShared[lane] = 0U;
        warpBaseCountsUltraShared[lane] = 0U;
        warpBaseCountsUltraPartialShared[lane] = 0U;
    }
    __syncthreads();

    const uint32_t validRuns = sharedState->validRuns;
    if (sharedState->useTwoStage != 0U) {
        return;
    }

    const uint32_t numRunVec4 = (validRuns + 3U) >> 2U;
    const uint4* __restrict__ runCountsVec4 = reinterpret_cast<const uint4*>(runCounts);
    uint4* runCountsLineSharedVec4[2] = {reinterpret_cast<uint4*>(runCountsLineShared[0]),
                                         reinterpret_cast<uint4*>(runCountsLineShared[1])};

    if (tid < numRunVec4) {
        runCountsLineSharedVec4[0][tid] = runCountsVec4[tid];
    }

    uint32_t readLineIndex = 0U;
    uint32_t writeLineIndex = 1U;

    for (uint32_t blockRunBase = 0U; blockRunBase < validRuns; blockRunBase += lineWords) {
        __syncthreads();

        const uint32_t nextVecBase = (blockRunBase + lineWords) >> 2U;
        const uint32_t nextVecIndex = nextVecBase + tid;
        if (nextVecIndex < numRunVec4) {
            runCountsLineSharedVec4[writeLineIndex][tid] = runCountsVec4[nextVecIndex];
        }

        for (uint32_t j = 0U; j < lineWords && blockRunBase + j < validRuns; j += activeThreads) {
            const uint32_t runIndex = blockRunBase + j + tid;
            const bool validRun = runIndex < validRuns;
            const uint32_t myRunCount = validRun ? runCountsLineShared[readLineIndex][j + tid] : 0U;
            const bool isUltra = validRun && (myRunCount >= ultraRunMin);
            const bool isHigh = validRun && (myRunCount > lowRunMax) && !isUltra;
            const bool isLow = validRun && !(isHigh || isUltra);
            const uint32_t myUltraPartialCount = isUltra ? ceilDivU32(myRunCount, ultraTokensPerPartial) : 0U;

            const uint32_t ultraMask = __ballot_sync(0xffffffffU, isUltra);
            const uint32_t highMask = __ballot_sync(0xffffffffU, isHigh);
            const uint32_t lowMask = __ballot_sync(0xffffffffU, isLow);

            const uint32_t ultraBeforeIncludingMe = __popc(ultraMask & leLaneMask);
            const uint32_t highBeforeIncludingMe = __popc(highMask & leLaneMask);
            const uint32_t lowBeforeIncludingMe = __popc(lowMask & leLaneMask);
            const uint32_t ultraPartialBeforeInWarp = cg::exclusive_scan(warpGroup, myUltraPartialCount);
            const uint32_t ultraPartialIncludingMe = ultraPartialBeforeInWarp + myUltraPartialCount;

            if (lane == 31U) {
                warpBaseCountsUltraShared[warp] = ultraBeforeIncludingMe;
                warpBaseCountsHighShared[warp] = highBeforeIncludingMe;
                warpBaseCountsLowShared[warp] = lowBeforeIncludingMe;
                warpBaseCountsUltraPartialShared[warp] = ultraPartialIncludingMe;
            }

            const uint32_t lowTotalBefore = sharedState->lowTotal;
            const uint32_t highTotalBefore = sharedState->highTotal;
            const uint32_t ultraTotalBefore = sharedState->ultraTotal;
            const uint32_t ultraPartialTotalBefore = sharedState->ultraPartialTotal;
            __syncthreads();

            if (warp == 0U) {
                const uint32_t ultraCount = lane < activeWarps ? warpBaseCountsUltraShared[lane] : 0U;
                const uint32_t highCount = lane < activeWarps ? warpBaseCountsHighShared[lane] : 0U;
                const uint32_t lowCount = lane < activeWarps ? warpBaseCountsLowShared[lane] : 0U;
                const uint32_t ultraPartialCount = lane < activeWarps ? warpBaseCountsUltraPartialShared[lane] : 0U;

                const uint32_t ultraBase = cg::exclusive_scan(warpGroup, ultraCount);
                const uint32_t highBase = cg::exclusive_scan(warpGroup, highCount);
                const uint32_t lowBase = cg::exclusive_scan(warpGroup, lowCount);
                const uint32_t ultraPartialBase = cg::exclusive_scan(warpGroup, ultraPartialCount);

                if (lane < activeWarps) {
                    warpBaseCountsUltraShared[lane] = ultraBase;
                    warpBaseCountsHighShared[lane] = highBase;
                    warpBaseCountsLowShared[lane] = lowBase;
                    warpBaseCountsUltraPartialShared[lane] = ultraPartialBase;
                }
                if (lane == activeWarps - 1U) {
                    warpBaseCountsUltraShared[FINALIZE_BUCKETIZE_WARPS] = ultraBase + ultraCount;
                    warpBaseCountsHighShared[FINALIZE_BUCKETIZE_WARPS] = highBase + highCount;
                    warpBaseCountsLowShared[FINALIZE_BUCKETIZE_WARPS] = lowBase + lowCount;
                    warpBaseCountsUltraPartialShared[FINALIZE_BUCKETIZE_WARPS] = ultraPartialBase + ultraPartialCount;
                }
            }
            __syncthreads();

            if (isLow) {
                const uint32_t writeBufferIndex =
                    (lowTotalBefore + warpBaseCountsLowShared[warp] + lowBeforeIncludingMe - 1U) & FINALIZE_BUCKETIZE_BUFFER_MASK;
                lowWriteBuffer[writeBufferIndex] = runIndex;
            } else if (isHigh) {
                const uint32_t writeBufferIndex =
                    (highTotalBefore + warpBaseCountsHighShared[warp] + highBeforeIncludingMe - 1U) & FINALIZE_BUCKETIZE_BUFFER_MASK;
                highWriteBuffer[writeBufferIndex] = runIndex;
            } else if (isUltra) {
                const uint32_t ultraBucketIndex = ultraTotalBefore + warpBaseCountsUltraShared[warp] + ultraBeforeIncludingMe - 1U;
                const uint32_t writeBufferIndex = ultraBucketIndex & FINALIZE_BUCKETIZE_BUFFER_MASK;
                ultraWriteBuffer[writeBufferIndex] = runIndex;
                ultraRunPartialCounts[ultraBucketIndex] = myUltraPartialCount;
                ultraRunPartialOffsets[ultraBucketIndex] =
                    ultraPartialTotalBefore + warpBaseCountsUltraPartialShared[warp] + ultraPartialBeforeInWarp;
            }

            if (tid == activeThreads - 1U) {
                const BucketAppendCounts appendCounts{warpBaseCountsLowShared[FINALIZE_BUCKETIZE_WARPS],
                                                      warpBaseCountsHighShared[FINALIZE_BUCKETIZE_WARPS],
                                                      warpBaseCountsUltraShared[FINALIZE_BUCKETIZE_WARPS],
                                                      warpBaseCountsUltraPartialShared[FINALIZE_BUCKETIZE_WARPS]};
                const uint32_t lowTotalAfter = lowTotalBefore + appendCounts.low;
                const uint32_t highTotalAfter = highTotalBefore + appendCounts.high;
                const uint32_t ultraTotalAfter = ultraTotalBefore + appendCounts.ultra;
                const uint32_t ultraPartialTotalAfter = ultraPartialTotalBefore + appendCounts.ultraPartials;

                sharedState->lowTotal = lowTotalAfter;
                sharedState->highTotal = highTotalAfter;
                sharedState->ultraTotal = ultraTotalAfter;
                sharedState->ultraPartialTotal = ultraPartialTotalAfter;

                const signed char dumpLow = (lowTotalBefore >> 10U) != (lowTotalAfter >> 10U) ? static_cast<signed char>(1) : 0;
                const signed char dumpHigh = (highTotalBefore >> 10U) != (highTotalAfter >> 10U) ? static_cast<signed char>(1) : 0;
                const signed char dumpUltra = (ultraTotalBefore >> 10U) != (ultraTotalAfter >> 10U) ? static_cast<signed char>(1) : 0;
                bufferDumpControlShared[0] = char4{dumpLow, dumpHigh, dumpUltra, 0};
            }

            __syncthreads();
            const char4 dumpControl = bufferDumpControlShared[0];
            if (dumpControl.x != 0) {
                dumpFullHalf(lowTotalBefore, lowRunRowIndices, lowWriteBuffer, tid);
            }
            if (dumpControl.y != 0) {
                dumpFullHalf(highTotalBefore, highRunRowIndices, highWriteBuffer, tid);
            }
            if (dumpControl.z != 0) {
                dumpFullHalf(ultraTotalBefore, ultraRunRowIndices, ultraWriteBuffer, tid);
            }
        }

        readLineIndex ^= 1U;
        writeLineIndex ^= 1U;
    }

    __syncthreads();

    const uint32_t lowTotal = sharedState->lowTotal;
    const uint32_t highTotal = sharedState->highTotal;
    const uint32_t ultraTotal = sharedState->ultraTotal;
    const uint32_t ultraPartialTotal = sharedState->ultraPartialTotal;

    dumpTail(lowTotal, lowRunRowIndices, lowWriteBuffer, tid);
    dumpTail(highTotal, highRunRowIndices, highWriteBuffer, tid);
    dumpTail(ultraTotal, ultraRunRowIndices, ultraWriteBuffer, tid);

    if (tid == 0U) {
        numLowRunRows[0] = lowTotal;
        numHighRunRows[0] = highTotal;
        numUltraRunRows[0] = ultraTotal;
        numUltraPartials[0] = ultraPartialTotal;

        updateEmbeddingSparseGradientBucketReduceGrid(lowReduceNode, lowTotal, lowReduceGridConfig);
        updateEmbeddingSparseGradientBucketReduceGrid(highReduceNode, highTotal, highReduceGridConfig);
        updateEmbeddingSparseGradientBucketReduceGrid(ultraPartialReduceNode, ultraPartialTotal, ultraPartialReduceGridConfig);
        updateEmbeddingSparseGradientBucketReduceGrid(ultraReduceNode, ultraTotal, ultraReduceGridConfig);
    }
}

struct TwoStageFinalizeSharedState {
    uint32_t lowTotal;
    uint32_t highTotal;
    uint32_t ultraTotal;
    uint32_t ultraPartialTotal;
};

__device__ __forceinline__ uint32_t sumEmbeddingSparseGradientStageCounts(const uint32_t* __restrict__ counts, uint32_t blocks) {
    uint32_t total = 0U;
    for (uint32_t i = 0U; i < blocks; ++i) {
        total += counts[i];
    }
    return total;
}

template <typename RowT>
__global__ void finalizeAndBucketizeEmbeddingSparseGradientRowsTwoStageClassifyKernel(const RowT* __restrict__ outputRows,
                                                                                      const uint32_t* __restrict__ numRuns,
                                                                                      RowT* __restrict__ outputNumRows,
                                                                                      const uint32_t* __restrict__ runCounts,
                                                                                      uint32_t* __restrict__ lowRunRowsScratch,
                                                                                      uint32_t* __restrict__ highRunRowsScratch,
                                                                                      uint32_t* __restrict__ ultraRunRowsScratch,
                                                                                      uint32_t* __restrict__ ultraRunPartialCountsScratch,
                                                                                      uint32_t* __restrict__ ultraRunPartialOffsetsScratch,
                                                                                      uint32_t* __restrict__ lowRunRowCounts,
                                                                                      uint32_t* __restrict__ highRunRowCounts,
                                                                                      uint32_t* __restrict__ ultraRunRowCounts,
                                                                                      uint32_t* __restrict__ ultraPartialCounts,
                                                                                      uint64_t vocabularySize,
                                                                                      uint32_t lowRunMax,
                                                                                      uint32_t ultraRunMin,
                                                                                      uint32_t ultraTokensPerPartial) {
    __shared__ uint32_t warpBaseCountsLowShared[TWO_STAGE_FINALIZE_WARP_COUNT_WORDS];
    __shared__ uint32_t warpBaseCountsHighShared[TWO_STAGE_FINALIZE_WARP_COUNT_WORDS];
    __shared__ uint32_t warpBaseCountsUltraShared[TWO_STAGE_FINALIZE_WARP_COUNT_WORDS];
    __shared__ uint32_t warpBaseCountsUltraPartialShared[TWO_STAGE_FINALIZE_WARP_COUNT_WORDS];
    __shared__ TwoStageFinalizeSharedState sharedState;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warpGroup = cg::tiled_partition<32>(block);

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5U;
    const uint32_t lane = tid & 31U;
    const uint32_t leLaneMask = (1U << lane) | ((1U << lane) - 1U);
    const uint32_t activeThreads = blockDim.x;
    const uint32_t activeWarps = activeThreads >> 5U;
    const uint64_t blockRunBase = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TWO_STAGE_FINALIZE_RUNS_PER_BLOCK);

    uint32_t validRuns = numRuns[0];
    if (validRuns != 0U && static_cast<uint64_t>(outputRows[validRuns - 1U]) == vocabularySize) {
        --validRuns;
    }

    if (blockIdx.x == 0U && tid == 0U) {
        outputNumRows[0] = static_cast<RowT>(validRuns);
    }

    if (tid == 0U) {
        sharedState.lowTotal = 0U;
        sharedState.highTotal = 0U;
        sharedState.ultraTotal = 0U;
        sharedState.ultraPartialTotal = 0U;
    }
    if (warp == 0U) {
        warpBaseCountsLowShared[lane] = 0U;
        warpBaseCountsHighShared[lane] = 0U;
        warpBaseCountsUltraShared[lane] = 0U;
        warpBaseCountsUltraPartialShared[lane] = 0U;
    }
    __syncthreads();

    uint32_t blockRunCount = 0U;
    if (blockRunBase < static_cast<uint64_t>(validRuns)) {
        const uint64_t remainingRuns = static_cast<uint64_t>(validRuns) - blockRunBase;
        blockRunCount = static_cast<uint32_t>(remainingRuns < static_cast<uint64_t>(TWO_STAGE_FINALIZE_RUNS_PER_BLOCK)
                                                  ? remainingRuns
                                                  : static_cast<uint64_t>(TWO_STAGE_FINALIZE_RUNS_PER_BLOCK));
    }

    for (uint32_t localRunBase = 0U; localRunBase < blockRunCount; localRunBase += activeThreads) {
        const uint32_t localRun = localRunBase + tid;
        const bool validRun = localRun < blockRunCount;
        const uint32_t runIndex = static_cast<uint32_t>(blockRunBase + static_cast<uint64_t>(localRun));
        const uint32_t myRunCount = validRun ? runCounts[runIndex] : 0U;
        const bool isUltra = validRun && myRunCount >= ultraRunMin;
        const bool isHigh = validRun && myRunCount > lowRunMax && !isUltra;
        const bool isLow = validRun && !(isHigh || isUltra);
        const uint32_t myUltraPartialCount = isUltra ? ceilDivU32(myRunCount, ultraTokensPerPartial) : 0U;

        const uint32_t ultraMask = __ballot_sync(0xffffffffU, isUltra);
        const uint32_t highMask = __ballot_sync(0xffffffffU, isHigh);
        const uint32_t lowMask = __ballot_sync(0xffffffffU, isLow);

        const uint32_t ultraBeforeIncludingMe = __popc(ultraMask & leLaneMask);
        const uint32_t highBeforeIncludingMe = __popc(highMask & leLaneMask);
        const uint32_t lowBeforeIncludingMe = __popc(lowMask & leLaneMask);
        const uint32_t ultraPartialBeforeInWarp = cg::exclusive_scan(warpGroup, myUltraPartialCount);
        const uint32_t ultraPartialIncludingMe = ultraPartialBeforeInWarp + myUltraPartialCount;

        if (lane == 31U) {
            warpBaseCountsUltraShared[warp] = ultraBeforeIncludingMe;
            warpBaseCountsHighShared[warp] = highBeforeIncludingMe;
            warpBaseCountsLowShared[warp] = lowBeforeIncludingMe;
            warpBaseCountsUltraPartialShared[warp] = ultraPartialIncludingMe;
        }

        const uint32_t lowTotalBefore = sharedState.lowTotal;
        const uint32_t highTotalBefore = sharedState.highTotal;
        const uint32_t ultraTotalBefore = sharedState.ultraTotal;
        const uint32_t ultraPartialTotalBefore = sharedState.ultraPartialTotal;
        __syncthreads();

        if (warp == 0U) {
            const uint32_t ultraCount = lane < activeWarps ? warpBaseCountsUltraShared[lane] : 0U;
            const uint32_t highCount = lane < activeWarps ? warpBaseCountsHighShared[lane] : 0U;
            const uint32_t lowCount = lane < activeWarps ? warpBaseCountsLowShared[lane] : 0U;
            const uint32_t ultraPartialCount = lane < activeWarps ? warpBaseCountsUltraPartialShared[lane] : 0U;

            const uint32_t ultraBase = cg::exclusive_scan(warpGroup, ultraCount);
            const uint32_t highBase = cg::exclusive_scan(warpGroup, highCount);
            const uint32_t lowBase = cg::exclusive_scan(warpGroup, lowCount);
            const uint32_t ultraPartialBase = cg::exclusive_scan(warpGroup, ultraPartialCount);

            if (lane < activeWarps) {
                warpBaseCountsUltraShared[lane] = ultraBase;
                warpBaseCountsHighShared[lane] = highBase;
                warpBaseCountsLowShared[lane] = lowBase;
                warpBaseCountsUltraPartialShared[lane] = ultraPartialBase;
            }
            if (lane == activeWarps - 1U) {
                warpBaseCountsUltraShared[TWO_STAGE_FINALIZE_WARPS] = ultraBase + ultraCount;
                warpBaseCountsHighShared[TWO_STAGE_FINALIZE_WARPS] = highBase + highCount;
                warpBaseCountsLowShared[TWO_STAGE_FINALIZE_WARPS] = lowBase + lowCount;
                warpBaseCountsUltraPartialShared[TWO_STAGE_FINALIZE_WARPS] = ultraPartialBase + ultraPartialCount;
            }
        }
        __syncthreads();

        if (isLow) {
            const uint32_t localOut = lowTotalBefore + warpBaseCountsLowShared[warp] + lowBeforeIncludingMe - 1U;
            lowRunRowsScratch[blockRunBase + static_cast<uint64_t>(localOut)] = runIndex;
        } else if (isHigh) {
            const uint32_t localOut = highTotalBefore + warpBaseCountsHighShared[warp] + highBeforeIncludingMe - 1U;
            highRunRowsScratch[blockRunBase + static_cast<uint64_t>(localOut)] = runIndex;
        } else if (isUltra) {
            const uint32_t localOut = ultraTotalBefore + warpBaseCountsUltraShared[warp] + ultraBeforeIncludingMe - 1U;
            const uint64_t scratchOut = blockRunBase + static_cast<uint64_t>(localOut);
            ultraRunRowsScratch[scratchOut] = runIndex;
            ultraRunPartialCountsScratch[scratchOut] = myUltraPartialCount;
            ultraRunPartialOffsetsScratch[scratchOut] =
                ultraPartialTotalBefore + warpBaseCountsUltraPartialShared[warp] + ultraPartialBeforeInWarp;
        }

        if (tid == activeThreads - 1U) {
            sharedState.lowTotal = lowTotalBefore + warpBaseCountsLowShared[TWO_STAGE_FINALIZE_WARPS];
            sharedState.highTotal = highTotalBefore + warpBaseCountsHighShared[TWO_STAGE_FINALIZE_WARPS];
            sharedState.ultraTotal = ultraTotalBefore + warpBaseCountsUltraShared[TWO_STAGE_FINALIZE_WARPS];
            sharedState.ultraPartialTotal = ultraPartialTotalBefore + warpBaseCountsUltraPartialShared[TWO_STAGE_FINALIZE_WARPS];
        }
        __syncthreads();
    }

    if (tid == 0U) {
        lowRunRowCounts[blockIdx.x] = sharedState.lowTotal;
        highRunRowCounts[blockIdx.x] = sharedState.highTotal;
        ultraRunRowCounts[blockIdx.x] = sharedState.ultraTotal;
        ultraPartialCounts[blockIdx.x] = sharedState.ultraPartialTotal;
    }
}

template <typename RowT>
__global__ void finalizeAndBucketizeEmbeddingSparseGradientRowsTwoStageAccumulateKernel(
    const uint32_t* __restrict__ lowRunRowsScratch,
    const uint32_t* __restrict__ highRunRowsScratch,
    const uint32_t* __restrict__ ultraRunRowsScratch,
    const uint32_t* __restrict__ ultraRunPartialCountsScratch,
    const uint32_t* __restrict__ ultraRunPartialOffsetsScratch,
    const uint32_t* __restrict__ lowRunRowCounts,
    const uint32_t* __restrict__ highRunRowCounts,
    const uint32_t* __restrict__ ultraRunRowCounts,
    const uint32_t* __restrict__ ultraPartialCounts,
    uint32_t* __restrict__ lowRunRowIndices,
    uint32_t* __restrict__ highRunRowIndices,
    uint32_t* __restrict__ ultraRunRowIndices,
    uint32_t* __restrict__ ultraRunPartialCounts,
    uint32_t* __restrict__ ultraRunPartialOffsets,
    uint32_t* __restrict__ numUltraPartials,
    uint32_t* __restrict__ numLowRunRows,
    uint32_t* __restrict__ numHighRunRows,
    uint32_t* __restrict__ numUltraRunRows,
    const cudaGraphDeviceNode_t* lowReduceNode,
    const cudaGraphDeviceNode_t* highReduceNode,
    const cudaGraphDeviceNode_t* ultraPartialReduceNode,
    const cudaGraphDeviceNode_t* ultraReduceNode,
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraPartialReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraReduceGridConfig) {
    __shared__ uint32_t lowBaseShared;
    __shared__ uint32_t highBaseShared;
    __shared__ uint32_t ultraBaseShared;
    __shared__ uint32_t ultraPartialBaseShared;

    const uint32_t stageBlock = blockIdx.x;
    const uint32_t stageBlocks = gridDim.x;
    const uint32_t tid = threadIdx.x;

    if (tid == 0U) {
        uint32_t lowBase = 0U;
        uint32_t highBase = 0U;
        uint32_t ultraBase = 0U;
        uint32_t ultraPartialBase = 0U;
        for (uint32_t i = 0U; i < stageBlock; ++i) {
            lowBase += lowRunRowCounts[i];
            highBase += highRunRowCounts[i];
            ultraBase += ultraRunRowCounts[i];
            ultraPartialBase += ultraPartialCounts[i];
        }
        lowBaseShared = lowBase;
        highBaseShared = highBase;
        ultraBaseShared = ultraBase;
        ultraPartialBaseShared = ultraPartialBase;
    }
    __syncthreads();

    const uint32_t lowBase = lowBaseShared;
    const uint32_t highBase = highBaseShared;
    const uint32_t ultraBase = ultraBaseShared;
    const uint32_t ultraPartialBase = ultraPartialBaseShared;
    const uint32_t lowCount = lowRunRowCounts[stageBlock];
    const uint32_t highCount = highRunRowCounts[stageBlock];
    const uint32_t ultraCount = ultraRunRowCounts[stageBlock];
    const uint64_t scratchBase = static_cast<uint64_t>(stageBlock) * static_cast<uint64_t>(TWO_STAGE_FINALIZE_RUNS_PER_BLOCK);

    for (uint32_t i = tid; i < lowCount; i += blockDim.x) {
        lowRunRowIndices[lowBase + i] = lowRunRowsScratch[scratchBase + static_cast<uint64_t>(i)];
    }
    for (uint32_t i = tid; i < highCount; i += blockDim.x) {
        highRunRowIndices[highBase + i] = highRunRowsScratch[scratchBase + static_cast<uint64_t>(i)];
    }
    for (uint32_t i = tid; i < ultraCount; i += blockDim.x) {
        const uint64_t scratchIndex = scratchBase + static_cast<uint64_t>(i);
        const uint32_t dst = ultraBase + i;
        ultraRunRowIndices[dst] = ultraRunRowsScratch[scratchIndex];
        ultraRunPartialCounts[dst] = ultraRunPartialCountsScratch[scratchIndex];
        ultraRunPartialOffsets[dst] = ultraPartialBase + ultraRunPartialOffsetsScratch[scratchIndex];
    }

    if (stageBlock == 0U && tid == 0U) {
        const uint32_t lowTotal = sumEmbeddingSparseGradientStageCounts(lowRunRowCounts, stageBlocks);
        const uint32_t highTotal = sumEmbeddingSparseGradientStageCounts(highRunRowCounts, stageBlocks);
        const uint32_t ultraTotal = sumEmbeddingSparseGradientStageCounts(ultraRunRowCounts, stageBlocks);
        const uint32_t ultraPartialTotal = sumEmbeddingSparseGradientStageCounts(ultraPartialCounts, stageBlocks);

        numLowRunRows[0] = lowTotal;
        numHighRunRows[0] = highTotal;
        numUltraRunRows[0] = ultraTotal;
        numUltraPartials[0] = ultraPartialTotal;

        updateEmbeddingSparseGradientBucketReduceGrid(lowReduceNode, lowTotal, lowReduceGridConfig);
        updateEmbeddingSparseGradientBucketReduceGrid(highReduceNode, highTotal, highReduceGridConfig);
        updateEmbeddingSparseGradientBucketReduceGrid(ultraPartialReduceNode, ultraPartialTotal, ultraPartialReduceGridConfig);
        updateEmbeddingSparseGradientBucketReduceGrid(ultraReduceNode, ultraTotal, ultraReduceGridConfig);
    }
}

DeviceUpdatableKernelNode captureDeviceUpdatableRuntimeKernel(
    const void* kernel, dim3 gridDim, dim3 blockDim, uint32_t sharedMemBytes, void** args, Stream stream) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attr.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    attr.val.deviceUpdatableKernelNode.devNode = nullptr;

    cudaLaunchConfig_t config{};
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = sharedMemBytes;
    config.stream = stream.getStream();
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelExC(&config, kernel, args));
    return DeviceUpdatableKernelNode(reinterpret_cast<CUgraphDeviceNode>(attr.val.deviceUpdatableKernelNode.devNode));
}

template <typename RowT>
void launchFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped(
    const void* outputRows,
    const uint32_t* numRuns,
    void* outputNumRows,
    const uint32_t* runCounts,
    uint32_t* lowRunRowIndices,
    uint32_t* highRunRowIndices,
    uint32_t* ultraHighRunRowIndices,
    uint32_t* ultraHighRunPartialCounts,
    uint32_t* ultraHighRunPartialOffsets,
    uint32_t* numUltraHighPartials,
    uint32_t* numLowRunRows,
    uint32_t* numHighRunRows,
    uint32_t* numUltraHighRunRows,
    uint64_t vocabularySize,
    uint32_t maxPossibleRuns,
    uint32_t lowRunMax,
    uint32_t ultraHighRunMin,
    uint32_t ultraHighTokensPerPartial,
    const cudaGraphDeviceNode_t* lowReduceNodePtr,
    const cudaGraphDeviceNode_t* highReduceNodePtr,
    const cudaGraphDeviceNode_t* ultraHighPartialReduceNodePtr,
    const cudaGraphDeviceNode_t* ultraHighReduceNodePtr,
    const cudaGraphDeviceNode_t* twoStageClassifyNodePtr,
    const cudaGraphDeviceNode_t* twoStageAccumulateNodePtr,
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
    bool runtimeTwoStageDelegate,
    uint32_t runtimeTwoStageRunThreshold,
    Stream stream) {
    if (runtimeTwoStageDelegate && !useTwoStageEmbeddingSparseGradientFinalize(maxPossibleRuns)) {
        throw std::invalid_argument(
            "runtime-delegated embedding sparse-gradient finalizer selected below the two-stage capacity threshold.");
    }
    if (runtimeTwoStageDelegate && (twoStageClassifyNodePtr == nullptr || twoStageAccumulateNodePtr == nullptr)) {
        throw std::invalid_argument("runtime-delegated embedding sparse-gradient finalizer selected without two-stage graph-node handles.");
    }

    const uint32_t meaningfulThreads =
        maxPossibleRuns == 0U ? 32U : (maxPossibleRuns < FINALIZE_BUCKETIZE_THREADS ? maxPossibleRuns : FINALIZE_BUCKETIZE_THREADS);
    const uint32_t blockThreads = ((meaningfulThreads + 31U) / 32U) * 32U;

    finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<RowT><<<1, blockThreads, FINALIZE_BUCKETIZE_SHARED_BYTES, stream.getStream()>>>(
        static_cast<const RowT*>(outputRows),
        numRuns,
        static_cast<RowT*>(outputNumRows),
        runCounts,
        lowRunRowIndices,
        highRunRowIndices,
        ultraHighRunRowIndices,
        ultraHighRunPartialCounts,
        ultraHighRunPartialOffsets,
        numUltraHighPartials,
        numLowRunRows,
        numHighRunRows,
        numUltraHighRunRows,
        vocabularySize,
        lowRunMax,
        ultraHighRunMin,
        ultraHighTokensPerPartial,
        lowReduceNodePtr,
        highReduceNodePtr,
        ultraHighPartialReduceNodePtr,
        ultraHighReduceNodePtr,
        lowReduceGridConfig,
        highReduceGridConfig,
        ultraHighPartialReduceGridConfig,
        ultraHighReduceGridConfig,
        runtimeTwoStageDelegate ? twoStageClassifyNodePtr : nullptr,
        runtimeTwoStageDelegate ? twoStageAccumulateNodePtr : nullptr,
        runtimeTwoStageDelegate ? runtimeTwoStageRunThreshold : 0U);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename RowT>
EmbeddingSparseGradientTwoStageFinalizeCapturedNodes captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped(
    const void* outputRows,
    const uint32_t* numRuns,
    void* outputNumRows,
    const uint32_t* runCounts,
    uint32_t* lowRunRowIndices,
    uint32_t* highRunRowIndices,
    uint32_t* ultraHighRunRowIndices,
    uint32_t* ultraHighRunPartialCounts,
    uint32_t* ultraHighRunPartialOffsets,
    uint32_t* numUltraHighPartials,
    uint32_t* numLowRunRows,
    uint32_t* numHighRunRows,
    uint32_t* numUltraHighRunRows,
    uint32_t* twoStageLowRunRowsScratch,
    uint32_t* twoStageHighRunRowsScratch,
    uint32_t* twoStageUltraHighRunRowsScratch,
    uint32_t* twoStageUltraHighRunPartialCountsScratch,
    uint32_t* twoStageUltraHighRunPartialOffsetsScratch,
    uint32_t* twoStageLowRunRowCounts,
    uint32_t* twoStageHighRunRowCounts,
    uint32_t* twoStageUltraHighRunRowCounts,
    uint32_t* twoStageUltraHighPartialCounts,
    uint64_t vocabularySize,
    uint32_t maxPossibleRuns,
    uint32_t lowRunMax,
    uint32_t ultraHighRunMin,
    uint32_t ultraHighTokensPerPartial,
    const cudaGraphDeviceNode_t* lowReduceNodePtr,
    const cudaGraphDeviceNode_t* highReduceNodePtr,
    const cudaGraphDeviceNode_t* ultraHighPartialReduceNodePtr,
    const cudaGraphDeviceNode_t* ultraHighReduceNodePtr,
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
    Stream stream) {
    if (!useTwoStageEmbeddingSparseGradientFinalize(maxPossibleRuns)) {
        throw std::invalid_argument(
            "two-stage embedding sparse-gradient finalizer capture requested below the two-stage capacity threshold.");
    }
    if (twoStageLowRunRowsScratch == nullptr || twoStageHighRunRowsScratch == nullptr || twoStageUltraHighRunRowsScratch == nullptr ||
        twoStageUltraHighRunPartialCountsScratch == nullptr || twoStageUltraHighRunPartialOffsetsScratch == nullptr ||
        twoStageLowRunRowCounts == nullptr || twoStageHighRunRowCounts == nullptr || twoStageUltraHighRunRowCounts == nullptr ||
        twoStageUltraHighPartialCounts == nullptr) {
        throw std::invalid_argument("two-stage embedding sparse-gradient finalizer capture requires all scratch buffers.");
    }

    const uint32_t stageBlocks =
        static_cast<uint32_t>(twoStageEmbeddingSparseGradientFinalizeBlockCount(static_cast<uint64_t>(maxPossibleRuns)));

    const void* outputRowsPtr = outputRows;
    const void* numRunsPtr = numRuns;
    void* outputNumRowsPtr = outputNumRows;
    const void* runCountsPtr = runCounts;
    void* lowScratchPtr = twoStageLowRunRowsScratch;
    void* highScratchPtr = twoStageHighRunRowsScratch;
    void* ultraScratchPtr = twoStageUltraHighRunRowsScratch;
    void* ultraPartialCountsScratchPtr = twoStageUltraHighRunPartialCountsScratch;
    void* ultraPartialOffsetsScratchPtr = twoStageUltraHighRunPartialOffsetsScratch;
    void* lowCountsPtr = twoStageLowRunRowCounts;
    void* highCountsPtr = twoStageHighRunRowCounts;
    void* ultraCountsPtr = twoStageUltraHighRunRowCounts;
    void* ultraPartialCountsPtr = twoStageUltraHighPartialCounts;
    uint64_t vocabularySizeArg = vocabularySize;
    uint32_t lowRunMaxArg = lowRunMax;
    uint32_t ultraHighRunMinArg = ultraHighRunMin;
    uint32_t ultraHighTokensPerPartialArg = ultraHighTokensPerPartial;

    void* classifyArgs[] = {(void*)&outputRowsPtr,
                            (void*)&numRunsPtr,
                            (void*)&outputNumRowsPtr,
                            (void*)&runCountsPtr,
                            (void*)&lowScratchPtr,
                            (void*)&highScratchPtr,
                            (void*)&ultraScratchPtr,
                            (void*)&ultraPartialCountsScratchPtr,
                            (void*)&ultraPartialOffsetsScratchPtr,
                            (void*)&lowCountsPtr,
                            (void*)&highCountsPtr,
                            (void*)&ultraCountsPtr,
                            (void*)&ultraPartialCountsPtr,
                            (void*)&vocabularySizeArg,
                            (void*)&lowRunMaxArg,
                            (void*)&ultraHighRunMinArg,
                            (void*)&ultraHighTokensPerPartialArg};

    EmbeddingSparseGradientTwoStageFinalizeCapturedNodes nodes;
    nodes.classifyNode = captureDeviceUpdatableRuntimeKernel(
        reinterpret_cast<const void*>(finalizeAndBucketizeEmbeddingSparseGradientRowsTwoStageClassifyKernel<RowT>),
        dim3(stageBlocks, 1U, 1U),
        dim3(TWO_STAGE_FINALIZE_THREADS, 1U, 1U),
        0U,
        classifyArgs,
        stream);
    CUDA_CHECK(cudaPeekAtLastError());

    const void* lowScratchConstPtr = twoStageLowRunRowsScratch;
    const void* highScratchConstPtr = twoStageHighRunRowsScratch;
    const void* ultraScratchConstPtr = twoStageUltraHighRunRowsScratch;
    const void* ultraPartialCountsScratchConstPtr = twoStageUltraHighRunPartialCountsScratch;
    const void* ultraPartialOffsetsScratchConstPtr = twoStageUltraHighRunPartialOffsetsScratch;
    const void* lowCountsConstPtr = twoStageLowRunRowCounts;
    const void* highCountsConstPtr = twoStageHighRunRowCounts;
    const void* ultraCountsConstPtr = twoStageUltraHighRunRowCounts;
    const void* ultraPartialCountsConstPtr = twoStageUltraHighPartialCounts;
    void* lowRunRowsPtr = lowRunRowIndices;
    void* highRunRowsPtr = highRunRowIndices;
    void* ultraRunRowsPtr = ultraHighRunRowIndices;
    void* ultraRunPartialCountsPtr = ultraHighRunPartialCounts;
    void* ultraRunPartialOffsetsPtr = ultraHighRunPartialOffsets;
    void* numUltraHighPartialsPtr = numUltraHighPartials;
    void* numLowRunRowsPtr = numLowRunRows;
    void* numHighRunRowsPtr = numHighRunRows;
    void* numUltraHighRunRowsPtr = numUltraHighRunRows;
    const void* lowReduceNode = lowReduceNodePtr;
    const void* highReduceNode = highReduceNodePtr;
    const void* ultraHighPartialReduceNode = ultraHighPartialReduceNodePtr;
    const void* ultraHighReduceNode = ultraHighReduceNodePtr;
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfigArg = lowReduceGridConfig;
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfigArg = highReduceGridConfig;
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfigArg = ultraHighPartialReduceGridConfig;
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfigArg = ultraHighReduceGridConfig;

    void* accumulateArgs[] = {(void*)&lowScratchConstPtr,
                              (void*)&highScratchConstPtr,
                              (void*)&ultraScratchConstPtr,
                              (void*)&ultraPartialCountsScratchConstPtr,
                              (void*)&ultraPartialOffsetsScratchConstPtr,
                              (void*)&lowCountsConstPtr,
                              (void*)&highCountsConstPtr,
                              (void*)&ultraCountsConstPtr,
                              (void*)&ultraPartialCountsConstPtr,
                              (void*)&lowRunRowsPtr,
                              (void*)&highRunRowsPtr,
                              (void*)&ultraRunRowsPtr,
                              (void*)&ultraRunPartialCountsPtr,
                              (void*)&ultraRunPartialOffsetsPtr,
                              (void*)&numUltraHighPartialsPtr,
                              (void*)&numLowRunRowsPtr,
                              (void*)&numHighRunRowsPtr,
                              (void*)&numUltraHighRunRowsPtr,
                              (void*)&lowReduceNode,
                              (void*)&highReduceNode,
                              (void*)&ultraHighPartialReduceNode,
                              (void*)&ultraHighReduceNode,
                              (void*)&lowReduceGridConfigArg,
                              (void*)&highReduceGridConfigArg,
                              (void*)&ultraHighPartialReduceGridConfigArg,
                              (void*)&ultraHighReduceGridConfigArg};

    nodes.accumulateNode = captureDeviceUpdatableRuntimeKernel(
        reinterpret_cast<const void*>(finalizeAndBucketizeEmbeddingSparseGradientRowsTwoStageAccumulateKernel<RowT>),
        dim3(stageBlocks, 1U, 1U),
        dim3(TWO_STAGE_FINALIZE_THREADS, 1U, 1U),
        0U,
        accumulateArgs,
        stream);
    CUDA_CHECK(cudaPeekAtLastError());

    return nodes;
}

}  // namespace

void initializeEmbeddingKernelsSharedAttributes() {
    static std::once_flag setSharedMemoryAttributesOnce;
    std::call_once(setSharedMemoryAttributesOnce, [] {
        for (uint32_t i = 0; i < MachineEvaluator::instance().getNumGpus(); ++i) {
            ScopedGpu scopedGpu(i);

            int maxOptInShared = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&maxOptInShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, static_cast<int>(i)));
            if (FINALIZE_BUCKETIZE_SHARED_BYTES > static_cast<uint32_t>(maxOptInShared)) {
                throw std::runtime_error("finalize/bucketize embedding sparse-gradient kernel requires too much shared memory for gpu " +
                                         std::to_string(i));
            }

            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<uint16_t>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(FINALIZE_BUCKETIZE_SHARED_BYTES)));
            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<uint16_t>,
                                            cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<uint32_t>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(FINALIZE_BUCKETIZE_SHARED_BYTES)));
            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<uint32_t>,
                                            cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<uint64_t>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(FINALIZE_BUCKETIZE_SHARED_BYTES)));
            CUDA_CHECK(cudaFuncSetAttribute(finalizeAndBucketizeEmbeddingSparseGradientRowsKernel<uint64_t>,
                                            cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
        }
    });
}

void launchFinalizeAndBucketizeEmbeddingSparseGradientRows(const void* outputRows,
                                                           const uint32_t* numRuns,
                                                           void* outputNumRows,
                                                           const uint32_t* runCounts,
                                                           uint32_t* lowRunRowIndices,
                                                           uint32_t* highRunRowIndices,
                                                           uint32_t* ultraHighRunRowIndices,
                                                           uint32_t* ultraHighRunPartialCounts,
                                                           uint32_t* ultraHighRunPartialOffsets,
                                                           uint32_t* numUltraHighPartials,
                                                           uint32_t* numLowRunRows,
                                                           uint32_t* numHighRunRows,
                                                           uint32_t* numUltraHighRunRows,
                                                           uint64_t vocabularySize,
                                                           uint32_t maxPossibleRuns,
                                                           DataType rowDataType,
                                                           uint32_t lowRunMax,
                                                           uint32_t ultraHighRunMin,
                                                           uint32_t ultraHighTokensPerPartial,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* lowReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* highReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* ultraHighPartialReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* ultraHighReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* twoStageClassifyNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* twoStageAccumulateNodeHandle,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
                                                           bool runtimeTwoStageDelegate,
                                                           uint32_t runtimeTwoStageRunThreshold,
                                                           Stream stream) {
    initializeEmbeddingKernelsSharedAttributes();

    const cudaGraphDeviceNode_t* lowReduceNodePtr = lowReduceNodeHandle != nullptr ? lowReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* highReduceNodePtr = highReduceNodeHandle != nullptr ? highReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* ultraHighPartialReduceNodePtr =
        ultraHighPartialReduceNodeHandle != nullptr ? ultraHighPartialReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* ultraHighReduceNodePtr =
        ultraHighReduceNodeHandle != nullptr ? ultraHighReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* twoStageClassifyNodePtr =
        twoStageClassifyNodeHandle != nullptr ? twoStageClassifyNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* twoStageAccumulateNodePtr =
        twoStageAccumulateNodeHandle != nullptr ? twoStageAccumulateNodeHandle->devicePtr() : nullptr;

    switch (rowDataType) {
        case DataType::UINT16:
            launchFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped<uint16_t>(outputRows,
                                                                                 numRuns,
                                                                                 outputNumRows,
                                                                                 runCounts,
                                                                                 lowRunRowIndices,
                                                                                 highRunRowIndices,
                                                                                 ultraHighRunRowIndices,
                                                                                 ultraHighRunPartialCounts,
                                                                                 ultraHighRunPartialOffsets,
                                                                                 numUltraHighPartials,
                                                                                 numLowRunRows,
                                                                                 numHighRunRows,
                                                                                 numUltraHighRunRows,
                                                                                 vocabularySize,
                                                                                 maxPossibleRuns,
                                                                                 lowRunMax,
                                                                                 ultraHighRunMin,
                                                                                 ultraHighTokensPerPartial,
                                                                                 lowReduceNodePtr,
                                                                                 highReduceNodePtr,
                                                                                 ultraHighPartialReduceNodePtr,
                                                                                 ultraHighReduceNodePtr,
                                                                                 twoStageClassifyNodePtr,
                                                                                 twoStageAccumulateNodePtr,
                                                                                 lowReduceGridConfig,
                                                                                 highReduceGridConfig,
                                                                                 ultraHighPartialReduceGridConfig,
                                                                                 ultraHighReduceGridConfig,
                                                                                 runtimeTwoStageDelegate,
                                                                                 runtimeTwoStageRunThreshold,
                                                                                 stream);
            break;
        case DataType::UINT32:
            launchFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped<uint32_t>(outputRows,
                                                                                 numRuns,
                                                                                 outputNumRows,
                                                                                 runCounts,
                                                                                 lowRunRowIndices,
                                                                                 highRunRowIndices,
                                                                                 ultraHighRunRowIndices,
                                                                                 ultraHighRunPartialCounts,
                                                                                 ultraHighRunPartialOffsets,
                                                                                 numUltraHighPartials,
                                                                                 numLowRunRows,
                                                                                 numHighRunRows,
                                                                                 numUltraHighRunRows,
                                                                                 vocabularySize,
                                                                                 maxPossibleRuns,
                                                                                 lowRunMax,
                                                                                 ultraHighRunMin,
                                                                                 ultraHighTokensPerPartial,
                                                                                 lowReduceNodePtr,
                                                                                 highReduceNodePtr,
                                                                                 ultraHighPartialReduceNodePtr,
                                                                                 ultraHighReduceNodePtr,
                                                                                 twoStageClassifyNodePtr,
                                                                                 twoStageAccumulateNodePtr,
                                                                                 lowReduceGridConfig,
                                                                                 highReduceGridConfig,
                                                                                 ultraHighPartialReduceGridConfig,
                                                                                 ultraHighReduceGridConfig,
                                                                                 runtimeTwoStageDelegate,
                                                                                 runtimeTwoStageRunThreshold,
                                                                                 stream);
            break;
        case DataType::UINT64:
            launchFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped<uint64_t>(outputRows,
                                                                                 numRuns,
                                                                                 outputNumRows,
                                                                                 runCounts,
                                                                                 lowRunRowIndices,
                                                                                 highRunRowIndices,
                                                                                 ultraHighRunRowIndices,
                                                                                 ultraHighRunPartialCounts,
                                                                                 ultraHighRunPartialOffsets,
                                                                                 numUltraHighPartials,
                                                                                 numLowRunRows,
                                                                                 numHighRunRows,
                                                                                 numUltraHighRunRows,
                                                                                 vocabularySize,
                                                                                 maxPossibleRuns,
                                                                                 lowRunMax,
                                                                                 ultraHighRunMin,
                                                                                 ultraHighTokensPerPartial,
                                                                                 lowReduceNodePtr,
                                                                                 highReduceNodePtr,
                                                                                 ultraHighPartialReduceNodePtr,
                                                                                 ultraHighReduceNodePtr,
                                                                                 twoStageClassifyNodePtr,
                                                                                 twoStageAccumulateNodePtr,
                                                                                 lowReduceGridConfig,
                                                                                 highReduceGridConfig,
                                                                                 ultraHighPartialReduceGridConfig,
                                                                                 ultraHighReduceGridConfig,
                                                                                 runtimeTwoStageDelegate,
                                                                                 runtimeTwoStageRunThreshold,
                                                                                 stream);
            break;
        default:
            throw std::runtime_error("Embedding sparse-gradient finalize controller has unsupported row dtype.");
    }
}

EmbeddingSparseGradientTwoStageFinalizeCapturedNodes captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRows(
    const void* outputRows,
    const uint32_t* numRuns,
    void* outputNumRows,
    const uint32_t* runCounts,
    uint32_t* lowRunRowIndices,
    uint32_t* highRunRowIndices,
    uint32_t* ultraHighRunRowIndices,
    uint32_t* ultraHighRunPartialCounts,
    uint32_t* ultraHighRunPartialOffsets,
    uint32_t* numUltraHighPartials,
    uint32_t* numLowRunRows,
    uint32_t* numHighRunRows,
    uint32_t* numUltraHighRunRows,
    uint32_t* twoStageLowRunRowsScratch,
    uint32_t* twoStageHighRunRowsScratch,
    uint32_t* twoStageUltraHighRunRowsScratch,
    uint32_t* twoStageUltraHighRunPartialCountsScratch,
    uint32_t* twoStageUltraHighRunPartialOffsetsScratch,
    uint32_t* twoStageLowRunRowCounts,
    uint32_t* twoStageHighRunRowCounts,
    uint32_t* twoStageUltraHighRunRowCounts,
    uint32_t* twoStageUltraHighPartialCounts,
    uint64_t vocabularySize,
    uint32_t maxPossibleRuns,
    DataType rowDataType,
    uint32_t lowRunMax,
    uint32_t ultraHighRunMin,
    uint32_t ultraHighTokensPerPartial,
    const DeviceUpdatableKernelNodeDeviceHandle* lowReduceNodeHandle,
    const DeviceUpdatableKernelNodeDeviceHandle* highReduceNodeHandle,
    const DeviceUpdatableKernelNodeDeviceHandle* ultraHighPartialReduceNodeHandle,
    const DeviceUpdatableKernelNodeDeviceHandle* ultraHighReduceNodeHandle,
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
    EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
    Stream stream) {
    const cudaGraphDeviceNode_t* lowReduceNodePtr = lowReduceNodeHandle != nullptr ? lowReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* highReduceNodePtr = highReduceNodeHandle != nullptr ? highReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* ultraHighPartialReduceNodePtr =
        ultraHighPartialReduceNodeHandle != nullptr ? ultraHighPartialReduceNodeHandle->devicePtr() : nullptr;
    const cudaGraphDeviceNode_t* ultraHighReduceNodePtr =
        ultraHighReduceNodeHandle != nullptr ? ultraHighReduceNodeHandle->devicePtr() : nullptr;

    switch (rowDataType) {
        case DataType::UINT16:
            return captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped<uint16_t>(outputRows,
                                                                                                 numRuns,
                                                                                                 outputNumRows,
                                                                                                 runCounts,
                                                                                                 lowRunRowIndices,
                                                                                                 highRunRowIndices,
                                                                                                 ultraHighRunRowIndices,
                                                                                                 ultraHighRunPartialCounts,
                                                                                                 ultraHighRunPartialOffsets,
                                                                                                 numUltraHighPartials,
                                                                                                 numLowRunRows,
                                                                                                 numHighRunRows,
                                                                                                 numUltraHighRunRows,
                                                                                                 twoStageLowRunRowsScratch,
                                                                                                 twoStageHighRunRowsScratch,
                                                                                                 twoStageUltraHighRunRowsScratch,
                                                                                                 twoStageUltraHighRunPartialCountsScratch,
                                                                                                 twoStageUltraHighRunPartialOffsetsScratch,
                                                                                                 twoStageLowRunRowCounts,
                                                                                                 twoStageHighRunRowCounts,
                                                                                                 twoStageUltraHighRunRowCounts,
                                                                                                 twoStageUltraHighPartialCounts,
                                                                                                 vocabularySize,
                                                                                                 maxPossibleRuns,
                                                                                                 lowRunMax,
                                                                                                 ultraHighRunMin,
                                                                                                 ultraHighTokensPerPartial,
                                                                                                 lowReduceNodePtr,
                                                                                                 highReduceNodePtr,
                                                                                                 ultraHighPartialReduceNodePtr,
                                                                                                 ultraHighReduceNodePtr,
                                                                                                 lowReduceGridConfig,
                                                                                                 highReduceGridConfig,
                                                                                                 ultraHighPartialReduceGridConfig,
                                                                                                 ultraHighReduceGridConfig,
                                                                                                 stream);
        case DataType::UINT32:
            return captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped<uint32_t>(outputRows,
                                                                                                 numRuns,
                                                                                                 outputNumRows,
                                                                                                 runCounts,
                                                                                                 lowRunRowIndices,
                                                                                                 highRunRowIndices,
                                                                                                 ultraHighRunRowIndices,
                                                                                                 ultraHighRunPartialCounts,
                                                                                                 ultraHighRunPartialOffsets,
                                                                                                 numUltraHighPartials,
                                                                                                 numLowRunRows,
                                                                                                 numHighRunRows,
                                                                                                 numUltraHighRunRows,
                                                                                                 twoStageLowRunRowsScratch,
                                                                                                 twoStageHighRunRowsScratch,
                                                                                                 twoStageUltraHighRunRowsScratch,
                                                                                                 twoStageUltraHighRunPartialCountsScratch,
                                                                                                 twoStageUltraHighRunPartialOffsetsScratch,
                                                                                                 twoStageLowRunRowCounts,
                                                                                                 twoStageHighRunRowCounts,
                                                                                                 twoStageUltraHighRunRowCounts,
                                                                                                 twoStageUltraHighPartialCounts,
                                                                                                 vocabularySize,
                                                                                                 maxPossibleRuns,
                                                                                                 lowRunMax,
                                                                                                 ultraHighRunMin,
                                                                                                 ultraHighTokensPerPartial,
                                                                                                 lowReduceNodePtr,
                                                                                                 highReduceNodePtr,
                                                                                                 ultraHighPartialReduceNodePtr,
                                                                                                 ultraHighReduceNodePtr,
                                                                                                 lowReduceGridConfig,
                                                                                                 highReduceGridConfig,
                                                                                                 ultraHighPartialReduceGridConfig,
                                                                                                 ultraHighReduceGridConfig,
                                                                                                 stream);
        case DataType::UINT64:
            return captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRowsTyped<uint64_t>(outputRows,
                                                                                                 numRuns,
                                                                                                 outputNumRows,
                                                                                                 runCounts,
                                                                                                 lowRunRowIndices,
                                                                                                 highRunRowIndices,
                                                                                                 ultraHighRunRowIndices,
                                                                                                 ultraHighRunPartialCounts,
                                                                                                 ultraHighRunPartialOffsets,
                                                                                                 numUltraHighPartials,
                                                                                                 numLowRunRows,
                                                                                                 numHighRunRows,
                                                                                                 numUltraHighRunRows,
                                                                                                 twoStageLowRunRowsScratch,
                                                                                                 twoStageHighRunRowsScratch,
                                                                                                 twoStageUltraHighRunRowsScratch,
                                                                                                 twoStageUltraHighRunPartialCountsScratch,
                                                                                                 twoStageUltraHighRunPartialOffsetsScratch,
                                                                                                 twoStageLowRunRowCounts,
                                                                                                 twoStageHighRunRowCounts,
                                                                                                 twoStageUltraHighRunRowCounts,
                                                                                                 twoStageUltraHighPartialCounts,
                                                                                                 vocabularySize,
                                                                                                 maxPossibleRuns,
                                                                                                 lowRunMax,
                                                                                                 ultraHighRunMin,
                                                                                                 ultraHighTokensPerPartial,
                                                                                                 lowReduceNodePtr,
                                                                                                 highReduceNodePtr,
                                                                                                 ultraHighPartialReduceNodePtr,
                                                                                                 ultraHighReduceNodePtr,
                                                                                                 lowReduceGridConfig,
                                                                                                 highReduceGridConfig,
                                                                                                 ultraHighPartialReduceGridConfig,
                                                                                                 ultraHighReduceGridConfig,
                                                                                                 stream);
        default:
            throw std::runtime_error("Embedding sparse-gradient two-stage finalize capture has unsupported row dtype.");
    }
}

}  // namespace ThorImplementation
