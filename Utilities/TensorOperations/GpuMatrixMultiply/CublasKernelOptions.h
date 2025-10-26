#pragma once

#include "Utilities/TensorOperations/GpuMatrixMultiply/RunStats.h"

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

}  // namespace ThorImplementation
