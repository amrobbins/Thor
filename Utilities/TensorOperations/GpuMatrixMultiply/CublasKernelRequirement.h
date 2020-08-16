#pragma once

#include "Utilities/TensorOperations/GpuMatrixMultiply/KernelSpec.h"

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

class OperationType {
   public:
    OperationType(cublasComputeType_t computeDataType,
                  cudaDataType_t scaleDataType,
                  cudaDataType_t ADataType,
                  cudaDataType_t BDataType,
                  cudaDataType_t CDataType,
                  cudaDataType_t DDataType,
                  bool transposeA,
                  bool transposeB)
        : computeDataType(computeDataType),
          scaleDataType(scaleDataType),
          ADataType(ADataType),
          BDataType(BDataType),
          CDataType(CDataType),
          DDataType(DDataType),
          transposeA(transposeA),
          transposeB(transposeB) {
        // Below is exactly the bounds of current support
        assert(computeDataType == CUBLAS_COMPUTE_32F);
        assert(scaleDataType == CUDA_R_32F);
        assert(ADataType == CUDA_R_32F || ADataType == CUDA_R_16F);
        assert(BDataType == CUDA_R_32F || BDataType == CUDA_R_16F);
        assert(CDataType == CUDA_R_32F || CDataType == CUDA_R_16F);
        assert(DDataType == CUDA_R_32F || DDataType == CUDA_R_16F);
        assert(ADataType == BDataType);
        assert(ADataType == CDataType);
        assert(ADataType == DDataType);
    }

    cublasComputeType_t getComputeDataType() const { return computeDataType; }
    cudaDataType_t getScaleDataType() const { return scaleDataType; }
    cudaDataType_t getADataType() const { return ADataType; }
    cudaDataType_t getBDataType() const { return BDataType; }
    cudaDataType_t getCDataType() const { return CDataType; }
    cudaDataType_t getDDataType() const { return DDataType; }
    bool getTransposeA() const { return transposeA; }
    bool getTransposeB() const { return transposeB; }

    inline bool operator==(const OperationType &other) const {
        return computeDataType == other.computeDataType && scaleDataType == other.scaleDataType && ADataType == other.ADataType &&
               BDataType == other.BDataType && CDataType == other.CDataType && DDataType == other.DDataType &&
               transposeA == other.transposeA && transposeB == other.transposeB;
    }

   private:
    const cublasComputeType_t computeDataType;
    const cudaDataType_t scaleDataType;
    const cudaDataType_t ADataType;
    const cudaDataType_t BDataType;
    const cudaDataType_t CDataType;
    const cudaDataType_t DDataType;
    const bool transposeA;
    const bool transposeB;
};

struct CublasKernelRequirement {
   public:
    CublasKernelRequirement() = delete;

    CublasKernelRequirement(KernelRequirement kernelRequirement, OperationType operationType)
        : kernelRequirement(kernelRequirement), operationType(operationType) {}

    KernelRequirement kernelRequirement;
    OperationType operationType;

    inline bool operator==(const CublasKernelRequirement &other) const {
        return kernelRequirement == other.kernelRequirement && operationType == other.operationType;
    }
};

namespace std {

template <>
struct hash<OperationType> {
    std::size_t operator()(const OperationType &o) const {
        size_t hashValue = 0;
        hashValue |= o.getComputeDataType() == CUBLAS_COMPUTE_32F ? 7 : 13;
        hashValue |= o.getScaleDataType() == CUDA_R_32F ? 50 : 70;
        hashValue |= o.getADataType() == CUDA_R_32F ? 500 : 900;
        hashValue |= o.getBDataType() == CUDA_R_32F ? 2000 : 3400;
        hashValue |= o.getCDataType() == CUDA_R_32F ? 10000 : 30000;
        hashValue |= o.getDDataType() == CUDA_R_32F ? 40000 : 55000;
        return hashValue;
    }
};

template <>
struct hash<CublasKernelRequirement> {
    std::size_t operator()(const CublasKernelRequirement &k) const {
        hash<KernelRequirement> hashK;
        hash<OperationType> hashO;
        return hashK(k.kernelRequirement) ^ hashO(k.operationType);
    }
};

}  // namespace std
