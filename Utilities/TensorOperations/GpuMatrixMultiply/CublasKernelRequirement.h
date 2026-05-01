#pragma once

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <atomic>
#include <cstddef>
#include <functional>
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

inline bool isCublasLtFp8CudaType(cudaDataType_t dataType) { return dataType == CUDA_R_8F_E4M3 || dataType == CUDA_R_8F_E5M2; }

inline bool isCublasLtFloatingCudaType(cudaDataType_t dataType) {
    return dataType == CUDA_R_32F || dataType == CUDA_R_16BF || dataType == CUDA_R_16F || isCublasLtFp8CudaType(dataType);
}

struct CublasFp8MatmulScales {
    const float *AScaleDevicePointer = nullptr;
    const float *BScaleDevicePointer = nullptr;
    const float *CScaleDevicePointer = nullptr;
    const float *DScaleDevicePointer = nullptr;
    float *DAmaxDevicePointer = nullptr;

    bool hasAScale() const { return AScaleDevicePointer != nullptr; }
    bool hasBScale() const { return BScaleDevicePointer != nullptr; }
    bool hasCScale() const { return CScaleDevicePointer != nullptr; }
    bool hasDScale() const { return DScaleDevicePointer != nullptr; }
    bool hasDAmax() const { return DAmaxDevicePointer != nullptr; }

    bool hasAnyScalePointer() const { return hasAScale() || hasBScale() || hasCScale() || hasDScale() || hasDAmax(); }

    static CublasFp8MatmulScales none() { return CublasFp8MatmulScales{}; }

    static CublasFp8MatmulScales tensorwide(const float *AScaleDevicePointer,
                                            const float *BScaleDevicePointer,
                                            const float *CScaleDevicePointer = nullptr,
                                            const float *DScaleDevicePointer = nullptr,
                                            float *DAmaxDevicePointer = nullptr) {
        return CublasFp8MatmulScales{
            AScaleDevicePointer, BScaleDevicePointer, CScaleDevicePointer, DScaleDevicePointer, DAmaxDevicePointer};
    }
};

inline bool isCublasLtFp8DTypeAllowed(cudaDataType_t ADataType,
                                      cudaDataType_t BDataType,
                                      cudaDataType_t CDataType,
                                      cudaDataType_t DDataType) {
    if (!isCublasLtFp8CudaType(ADataType) || !isCublasLtFp8CudaType(BDataType)) {
        return false;
    }

    // The cublasLt FP8 table lists E4M3/E4M3, E4M3/E5M2, and E5M2/E4M3 inputs.
    // E5M2/E5M2 is intentionally not accepted here.
    if (ADataType == CUDA_R_8F_E5M2 && BDataType == CUDA_R_8F_E5M2) {
        return false;
    }

    if (CDataType == CUDA_R_32F) {
        return DDataType == CUDA_R_32F;
    }

    if (CDataType == CUDA_R_16BF || CDataType == CUDA_R_16F) {
        if (DDataType == CDataType) {
            return true;
        }
        if (DDataType == CUDA_R_8F_E4M3) {
            return true;
        }
        if (DDataType == CUDA_R_8F_E5M2) {
            return ADataType != CUDA_R_8F_E4M3 || BDataType != CUDA_R_8F_E4M3;
        }
    }

    return false;
}

inline bool isSupportedCublasLtOperationType(cublasComputeType_t computeDataType,
                                             cudaDataType_t scaleDataType,
                                             cudaDataType_t ADataType,
                                             cudaDataType_t BDataType,
                                             cudaDataType_t CDataType,
                                             cudaDataType_t DDataType) {
    // Keep this wrapper float-scale compatible. The public CublasMatrixMultiply API passes alpha/beta as float pointers,
    // so do not admit 16F scale or 32I scale operation types here.
    if (scaleDataType != CUDA_R_32F) {
        return false;
    }

    if (computeDataType == CUBLAS_COMPUTE_32F || computeDataType == CUBLAS_COMPUTE_32F_PEDANTIC ||
        computeDataType == CUBLAS_COMPUTE_32F_FAST_16F || computeDataType == CUBLAS_COMPUTE_32F_FAST_16BF ||
        computeDataType == CUBLAS_COMPUTE_32F_FAST_TF32) {
        if (ADataType == CUDA_R_32F && BDataType == CUDA_R_32F && CDataType == CUDA_R_32F && DDataType == CUDA_R_32F) {
            return true;
        }

        if ((ADataType == CUDA_R_16F || ADataType == CUDA_R_16BF) && ADataType == BDataType &&
            (CDataType == ADataType || CDataType == CUDA_R_32F) && CDataType == DDataType) {
            return true;
        }

        if (ADataType == CUDA_R_8I && BDataType == CUDA_R_8I && CDataType == CUDA_R_32F && DDataType == CUDA_R_32F) {
            return true;
        }

        if (isCublasLtFp8CudaType(ADataType) || isCublasLtFp8CudaType(BDataType) || isCublasLtFp8CudaType(DDataType)) {
            return computeDataType == CUBLAS_COMPUTE_32F && isCublasLtFp8DTypeAllowed(ADataType, BDataType, CDataType, DDataType);
        }
    }

    if (computeDataType == CUBLAS_COMPUTE_32I || computeDataType == CUBLAS_COMPUTE_32I_PEDANTIC) {
        // Regular-layout IMMA, float alpha/beta variant: int8 input and int8 C/D.
        return ADataType == CUDA_R_8I && BDataType == CUDA_R_8I && CDataType == CUDA_R_8I && DDataType == CUDA_R_8I;
    }

    return false;
}

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
        assert(isSupportedCublasLtOperationType(computeDataType, scaleDataType, ADataType, BDataType, CDataType, DDataType));
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

inline bool isCublasLtFp8OperationType(const OperationType &operationType) {
    return isCublasLtFp8CudaType(operationType.ADataType) || isCublasLtFp8CudaType(operationType.BDataType) ||
           isCublasLtFp8CudaType(operationType.CDataType) || isCublasLtFp8CudaType(operationType.DDataType);
}

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
        std::size_t seed = 0;
        auto hashCombine = [&seed](auto value) {
            std::size_t h = std::hash<int>()(static_cast<int>(value));
            seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        hashCombine(o.computeDataType);
        hashCombine(o.scaleDataType);
        hashCombine(o.ADataType);
        hashCombine(o.BDataType);
        hashCombine(o.CDataType);
        hashCombine(o.DDataType);
        return seed;
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
