#pragma once

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RunStats.h"

#include <cublasLt.h>

#include <cstdint>

namespace ThorImplementation {

struct CublasKernelOptions {
    CublasKernelOptions(cublasLtMatmulAlgo_t algorithm,
                        int algorithmId,
                        cublasLtMatmulTile_t tileSize,
                        uint32_t splitK,
                        uint32_t reductionFlag,
                        uint32_t swizzleType,
                        uint32_t customOptionValue,
                        uint32_t stagesId,
                        uint16_t innerShapeId,
                        uint16_t clusterShapeId,
                        uint64_t workspaceSizeInBytes,
                        float wavesCount)
        : algorithm(algorithm),
          algorithmId(algorithmId),
          tileSize(tileSize),
          splitK(splitK),
          reductionFlag(reductionFlag),
          swizzleType(swizzleType),
          customOptionValue(customOptionValue),
          stagesId(stagesId),
          innerShapeId(innerShapeId),
          clusterShapeId(clusterShapeId),
          workspaceSizeInBytes(workspaceSizeInBytes),
          wavesCount(wavesCount) {}

    const cublasLtMatmulAlgo_t algorithm;
    const int algorithmId;
    const cublasLtMatmulTile_t tileSize;
    const uint32_t splitK;
    const uint32_t reductionFlag;
    const uint32_t swizzleType;
    const uint32_t customOptionValue;
    const uint32_t stagesId;
    const uint16_t innerShapeId;
    const uint16_t clusterShapeId;
    const uint64_t workspaceSizeInBytes;
    const float wavesCount;

    RunStats runStats;

    inline bool operator<(CublasKernelOptions &rhs) { return runStats < rhs.runStats; }

    inline bool operator==(const CublasKernelOptions &other) const {
        return algorithmId == other.algorithmId && splitK == other.splitK && reductionFlag == other.reductionFlag &&
               swizzleType == other.swizzleType && customOptionValue == other.customOptionValue && stagesId == other.stagesId &&
               innerShapeId == other.innerShapeId && clusterShapeId == other.clusterShapeId;
    }
};

/**
 * Immutable value recipe for a measured cuBLASLt GEMM choice.
 *
 * This is the only form of an optimal ordinary GEMM that may live in the
 * process-global cache. It deliberately contains no matmul descriptors, matrix
 * layouts, handles, workspace allocations, or mutable RunStats. The measured
 * timing fields are immutable tuning facts copied out of the local contest
 * kernel after selection.
 */
struct CublasKernelSelection : AcceleratorBackendSelectionRecipeTag {
    cublasLtMatmulAlgo_t algorithm{};
    int algorithmId = -1;
    cublasLtMatmulTile_t tileSize = CUBLASLT_MATMUL_TILE_UNDEFINED;
    uint32_t splitK = 0;
    uint32_t reductionFlag = 0;
    uint32_t swizzleType = 0;
    uint32_t customOptionValue = 0;
    uint32_t stagesId = 0;
    uint16_t innerShapeId = 0;
    uint16_t clusterShapeId = 0;
    uint64_t workspaceSizeInBytes = 0;
    float wavesCount = 0.0f;
    int measuredRunCount = 0;
    double measuredTotalExecutionTimeMilliseconds = 0.0;

    [[nodiscard]] double getAverageRunTimeMilliseconds() const {
        THOR_THROW_IF_FALSE(measuredRunCount > 0);
        return measuredTotalExecutionTimeMilliseconds / measuredRunCount;
    }

    [[nodiscard]] CublasKernelOptions makeKernelOptions() const {
        return CublasKernelOptions(algorithm,
                                   algorithmId,
                                   tileSize,
                                   splitK,
                                   reductionFlag,
                                   swizzleType,
                                   customOptionValue,
                                   stagesId,
                                   innerShapeId,
                                   clusterShapeId,
                                   workspaceSizeInBytes,
                                   wavesCount);
    }

    bool operator==(const CublasKernelSelection &other) const {
        return algorithmId == other.algorithmId && tileSize == other.tileSize && splitK == other.splitK &&
               reductionFlag == other.reductionFlag && swizzleType == other.swizzleType &&
               customOptionValue == other.customOptionValue && stagesId == other.stagesId &&
               innerShapeId == other.innerShapeId && clusterShapeId == other.clusterShapeId &&
               workspaceSizeInBytes == other.workspaceSizeInBytes && wavesCount == other.wavesCount &&
               measuredRunCount == other.measuredRunCount &&
               measuredTotalExecutionTimeMilliseconds == other.measuredTotalExecutionTimeMilliseconds;
    }
};

}  // namespace ThorImplementation
