#pragma once

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <atomic>
#include <utility>

struct KernelRequirement {
    KernelRequirement(const std::string gpuType,
                      const int rowsA,
                      const int colsA,
                      const int rowsB,
                      const int colsB,
                      const bool transposeA,
                      const bool transposeB,
                      const bool transposeC,
                      const int ldA,
                      const int ldB,
                      const int ldC,
                      const int ldD,
                      const bool allowWorkspace)
        : gpuType(gpuType),
          rowsA(rowsA),
          colsA(colsA),
          rowsB(rowsB),
          colsB(colsB),
          transposeA(transposeA),
          transposeB(transposeB),
          transposeC(transposeC),
          ldA(ldA),
          ldB(ldB),
          ldC(ldC),
          ldD(ldD),
          allowWorkspace(allowWorkspace) {
        assert(rowsA > 0);
        assert(colsA > 0);
        assert(rowsB > 0);
        assert(colsB > 0);
        assert(ldA >= colsA);
        assert(ldB >= colsB);

        int finalRowsA = transposeA == false ? rowsA : colsA;
        int finalColsA = transposeA == false ? colsA : rowsA;
        int finalRowsB = transposeB == false ? rowsB : colsB;
        int finalColsB = transposeB == false ? colsB : rowsB;

        /*
        printf("rowsA %d colsA %d rowsB %d colsB %d transposeA %d transposeB %d ldA %d ldB %d ldC %d\n",
               rowsA,
               colsA,
               rowsB,
               colsB,
               transposeA,
               transposeB,
               ldA,
               ldB,
               ldC);
        */

        assert(finalColsA == finalRowsB);
        if (transposeC)
            assert(ldC >= finalRowsA);
        else
            assert(ldC >= finalColsB);
        assert(ldD >= finalColsB);
    }

    const std::string gpuType;
    const int rowsA;
    const int colsA;
    const int rowsB;
    const int colsB;
    const bool transposeA;
    const bool transposeB;
    const bool transposeC;
    const int ldA;
    const int ldB;
    const int ldC;
    const int ldD;
    const bool allowWorkspace;

    bool operator==(const KernelRequirement &other) const {
        return gpuType == other.gpuType && rowsA == other.rowsA && colsA == other.colsA && rowsB == other.rowsB && colsB == other.colsB &&
               transposeA == other.transposeA && transposeB == other.transposeB && transposeC == other.transposeC && ldA == other.ldA &&
               ldB == other.ldB && ldC == other.ldC && ldD == other.ldD && allowWorkspace == other.allowWorkspace;
    }

    bool operator<(const KernelRequirement &other) const {
        long total = rowsA + colsA + rowsB + colsB + ldA + ldB + ldC + ldD + (int)transposeA + (int)transposeB + (int)transposeC +
                     (int)allowWorkspace;
        long otherTotal = other.rowsA + other.colsA + other.rowsB + other.colsB + other.ldA + other.ldB + other.ldC + ldD +
                          (int)other.transposeA + (int)other.transposeB + (int)other.transposeC + (int)other.allowWorkspace;
        return total < otherTotal;
    }
};

namespace std {
template <>
struct hash<KernelRequirement> {
    std::size_t operator()(const KernelRequirement &k) const {
        size_t hashValue;
        hashValue = (hash<int>()(k.allowWorkspace)) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.rowsA))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.colsA))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.rowsB))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.colsB))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.ldA))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.ldB))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.ldC))) << 1;
        hashValue = (hashValue ^ (hash<int>()(k.ldD))) << 1;
        hashValue = hashValue ^ hash<std::string>()(k.gpuType);

        return hashValue;
    }
};
}  // namespace std

struct OperationType {
    OperationType(cublasComputeType_t computeDataType,
                  cudaDataType_t scaleDataType,
                  cudaDataType_t ADataType,
                  cudaDataType_t BDataType,
                  cudaDataType_t CDataType,
                  cudaDataType_t DDataType)
        : computeDataType(computeDataType),
          scaleDataType(scaleDataType),
          ADataType(ADataType),
          BDataType(BDataType),
          CDataType(CDataType),
          DDataType(DDataType) {
        // Below is exactly the bounds of current support - well it was at one time
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

    const cublasComputeType_t computeDataType;
    const cudaDataType_t scaleDataType;
    const cudaDataType_t ADataType;
    const cudaDataType_t BDataType;
    const cudaDataType_t CDataType;
    const cudaDataType_t DDataType;

    inline bool operator==(const OperationType &other) const {
        return computeDataType == other.computeDataType && scaleDataType == other.scaleDataType && ADataType == other.ADataType &&
               BDataType == other.BDataType && CDataType == other.CDataType && DDataType == other.DDataType;
    }
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
        hashValue |= o.computeDataType == CUBLAS_COMPUTE_32F ? 7 : 13;
        hashValue |= o.scaleDataType == CUDA_R_32F ? 50 : 70;
        hashValue |= o.ADataType == CUDA_R_32F ? 500 : 900;
        hashValue |= o.BDataType == CUDA_R_32F ? 2000 : 3400;
        hashValue |= o.CDataType == CUDA_R_32F ? 10000 : 30000;
        hashValue |= o.DDataType == CUDA_R_32F ? 40000 : 55000;
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
