#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelOptions.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelRequirement.h"
#include "Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h"

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

enum class CublasScalarPointerMode { Host, Device };

class CublasKernel : private ReferenceCounted {
   public:
    CublasKernel() : ReferenceCounted() {}

    CublasKernel(CublasKernelRequirement cublasKernelRequirement, CublasKernelOptions cublasKernelOptions, std::string gpuType) {
        construct(cublasKernelRequirement, cublasKernelOptions, gpuType);
    }

    CublasKernel(const CublasKernel &other) {
        // implemented using operator=
        *this = other;
    }

    CublasKernel &operator=(const CublasKernel &other) {
        copyFrom(other);
        return *this;
    }

    virtual ~CublasKernel() {
        bool shouldDestroy = ReferenceCounted::removeReference();
        if (shouldDestroy)
            destroy();
    }

    inline bool operator==(const CublasKernel &other) const {
        return cublasKernelRequirement == other.cublasKernelRequirement && cublasKernelOptions == other.cublasKernelOptions;
    }

    void setErrorFlag() {
        THOR_THROW_IF_FALSE(!uninitialized());
        cublasKernelOptions->runStats.errorFlag = true;
    }

    bool getErrorFlag() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return cublasKernelOptions->runStats.errorFlag;
    }

    void recordRun(double executionTimeOfRun) { cublasKernelOptions->runStats.recordRun(executionTimeOfRun); }

    double getAverageRunTimeMilliseconds() { return cublasKernelOptions->runStats.getAverageRunTimeMilliseconds(); }

    void stashRunStats() { cublasKernelOptions->runStats.stashRunStats(); }

    void unstashRunStats() { cublasKernelOptions->runStats.unstashRunStats(); }

    cublasLtMatmulDesc_t getOperationDesc(CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host) {
        THOR_THROW_IF_FALSE(!uninitialized());
        return (pointerMode == CublasScalarPointerMode::Device) ? *operationDescDevice : *operationDescHost;
    }

    cublasLtMatrixLayout_t getADesc() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return *ADesc;
    }

    cublasLtMatrixLayout_t getBDesc() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return *BDesc;
    }

    cublasLtMatrixLayout_t getCDesc() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return *CDesc;
    }

    cublasLtMatrixLayout_t getDDesc() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return *DDesc;
    }

    float getWavesCount(int gpuNum) const {
        THOR_THROW_IF_FALSE(!uninitialized());

        return cublasKernelOptions->wavesCount;
    }

    static inline bool executionTimeComparison(CublasKernel &lhs, CublasKernel &rhs) {
        THOR_THROW_IF_FALSE(!lhs.uninitialized());
        THOR_THROW_IF_FALSE(!rhs.uninitialized());
        return lhs.cublasKernelOptions->runStats < rhs.cublasKernelOptions->runStats;
    }

    void executeKernel(Tensor A,
                       Tensor B,
                       Tensor C,
                       Tensor D,
                       std::optional<Tensor> workspace,
                       const float *alpha,
                       const float *beta,
                       Stream stream,
                       CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host,
                       CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) {
        executeKernel(A,
                      B,
                      C,
                      D,
                      A.getDescriptor().getDimensions()[1],
                      B.getDescriptor().getDimensions()[1],
                      C.getDescriptor().getDimensions()[1],
                      D.getDescriptor().getDimensions()[1],
                      workspace,
                      alpha,
                      beta,
                      stream,
                      pointerMode,
                      fp8Scales);
    }

    void executeKernel(Tensor A,
                       Tensor B,
                       Tensor C,
                       Tensor D,
                       size_t ldA,
                       size_t ldB,
                       size_t ldC,
                       size_t ldD,
                       std::optional<Tensor> workspace,
                       const float *alpha,
                       const float *beta,
                       Stream stream,
                       CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host,
                       CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) {
        THOR_THROW_IF_FALSE(!uninitialized());

        uint64_t rowsC = cublasKernelRequirement->kernelRequirement.transposeA == false ? cublasKernelRequirement->kernelRequirement.rowsA
                                                                                        : cublasKernelRequirement->kernelRequirement.colsA;

        // Check that everything matches up
        std::vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(ADimensions.size() == 2);
        THOR_THROW_IF_FALSE(ADimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsA);
        THOR_THROW_IF_FALSE(ADimensions[1] == ldA);
        THOR_THROW_IF_FALSE(mapTensorDataTypeToCublasDataType(A.getDescriptor().getDataType()) ==
                            cublasKernelRequirement->operationType.ADataType);

        std::vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(BDimensions.size() == 2);
        THOR_THROW_IF_FALSE(BDimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsB);
        THOR_THROW_IF_FALSE(BDimensions[1] == ldB);
        THOR_THROW_IF_FALSE(mapTensorDataTypeToCublasDataType(B.getDescriptor().getDataType()) ==
                            cublasKernelRequirement->operationType.BDataType);

        std::vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(CDimensions.size() == 2);
        THOR_THROW_IF_FALSE(CDimensions[0] == rowsC);
        THOR_THROW_IF_FALSE(CDimensions[1] == ldC);
        THOR_THROW_IF_FALSE(mapTensorDataTypeToCublasDataType(C.getDescriptor().getDataType()) ==
                            cublasKernelRequirement->operationType.CDataType);

        std::vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(DDimensions.size() == 2);
        THOR_THROW_IF_FALSE(DDimensions[0] == rowsC);
        THOR_THROW_IF_FALSE(DDimensions[1] == ldD);
        THOR_THROW_IF_FALSE(mapTensorDataTypeToCublasDataType(D.getDescriptor().getDataType()) ==
                            cublasKernelRequirement->operationType.DDataType);

        // FIXME: Why was this there? What is the current support surface?
        // THOR_THROW_IF_FALSE(C.getMemPtr() != A.getMemPtr());
        // THOR_THROW_IF_FALSE(C.getMemPtr() != B.getMemPtr());

        THOR_THROW_IF_FALSE(runWithoutChecks(A, B, C, D, workspace, alpha, beta, stream, pointerMode, fp8Scales) == CUBLAS_STATUS_SUCCESS);
    }

    inline cublasStatus_t launchUncheckedPrevalidated(Tensor A,
                                                       Tensor B,
                                                       Tensor C,
                                                       Tensor D,
                                                       std::optional<Tensor> workspace,
                                                       const float *alpha,
                                                       const float *beta,
                                                       Stream stream,
                                                       CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host,
                                                       CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) {
        THOR_THROW_IF_FALSE(!uninitialized());
        ScopedGpu scopedGpu(stream.getGpuNum());

        // The stamped expression path has already validated tensor shape/dtype and workspace capacity in the
        // build/stamp phase.  Keep the common FP16/FP32/BF16 launch path as close to submit-only as possible:
        // no descriptor/dimension validation, no workspace-size recomputation, and no descriptor mutation.
        // FP8 still needs the older path for per-launch scale pointers and optional row-major transpose workspace.
        if (usesFp8ColumnMajorLtPath() || fp8Scales.hasAnyScalePointer()) {
            return runWithoutChecks(A, B, C, D, workspace, alpha, beta, stream, pointerMode, fp8Scales);
        }

        void *ltWorkspace = cublasKernelOptions->workspaceSizeInBytes > 0 ? workspace.value().getMemPtr() : nullptr;

        return cublasLtMatmul(stream.getCublasLtHandleUnchecked(),
                              getOperationDesc(pointerMode),
                              alpha,
                              A.getMemPtr(),
                              *ADesc,
                              B.getMemPtr(),
                              *BDesc,
                              beta,
                              C.getMemPtr(),
                              *CDesc,
                              D.getMemPtr(),
                              *DDesc,
                              &cublasKernelOptions->algorithm,
                              ltWorkspace,
                              cublasKernelOptions->workspaceSizeInBytes,
                              stream);
    }

    inline cublasStatus_t runWithoutChecks(Tensor A,
                                           Tensor B,
                                           Tensor C,
                                           Tensor D,
                                           std::optional<Tensor> workspace,
                                           const float *alpha,
                                           const float *beta,
                                           Stream stream,
                                           CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host,
                                           CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) {
        THOR_THROW_IF_FALSE(!uninitialized());
        ScopedGpu scopedGpu(stream.getGpuNum());

        const size_t requiredWorkspaceSize = getWorkspaceSizeInBytes(stream.getGpuNum(), fp8Scales);

        if (requiredWorkspaceSize > 0 && !workspace.has_value()) {
            throw std::runtime_error("CublasKernel::runWithoutChecks requires a workspace tensor for this cuBLASLt kernel.");
        }
        if (workspace.has_value()) {
            THOR_THROW_IF_FALSE(workspace.value().getDescriptor().getArraySizeInBytes() >= requiredWorkspaceSize);
        }

        cublasLtMatmulDesc_t operationDesc = getOperationDesc(pointerMode);
        configureTensorwideFp8Scales(operationDesc, fp8Scales);

        const void *ltA = A.getMemPtr();
        const void *ltB = B.getMemPtr();
        void *ltWorkspace = nullptr;
        size_t ltWorkspaceSizeInBytes = cublasKernelOptions->workspaceSizeInBytes;

        if (usesFp8ColumnMajorLtPath()) {
            validateFp8RowMajorGemmShapeAndLayoutOrThrow("CublasKernel::runWithoutChecks");

            if (fp8NeedsTransposedAWorkspace() &&
                cublasKernelRequirement->kernelRequirement.ldA != cublasKernelRequirement->kernelRequirement.colsA) {
                throw std::runtime_error(
                    "CublasKernel FP8 row-major TN path currently requires contiguous A rows when A must be materialized transposed.");
            }
            if (fp8NeedsTransposedBWorkspace() &&
                cublasKernelRequirement->kernelRequirement.ldB != cublasKernelRequirement->kernelRequirement.colsB) {
                throw std::runtime_error(
                    "CublasKernel FP8 row-major TN path currently requires contiguous B rows when B must be materialized transposed.");
            }

            void *workspaceBase = requiredWorkspaceSize > 0 ? workspace.value().getMemPtr() : nullptr;

            // Internal FP8 cuBLASLt uses column-major TN.  The first cuBLASLt operand is derived from external B,
            // and the second cuBLASLt operand is derived from external A, so the row-major public contract still computes
            // D = alpha * op(A) * op(B) + beta * C.
            if (fp8NeedsTransposedBWorkspace()) {
                void *transposedB = addBytes(workspaceBase, fp8TransposedBWorkspaceOffsetInBytes());
                launchMatrixTransposeByType(transposedB,
                                            B.getMemPtr(),
                                            static_cast<uint32_t>(cublasKernelRequirement->kernelRequirement.rowsB),
                                            static_cast<uint32_t>(cublasKernelRequirement->kernelRequirement.colsB),
                                            thorDataTypeForCudaDataType(cublasKernelRequirement->operationType.BDataType),
                                            thorDataTypeForCudaDataType(cublasKernelRequirement->operationType.BDataType),
                                            stream);
                ltA = transposedB;
            } else {
                ltA = B.getMemPtr();
            }

            if (fp8NeedsTransposedAWorkspace()) {
                void *transposedA = addBytes(workspaceBase, fp8TransposedAWorkspaceOffsetInBytes());
                launchMatrixTransposeByType(transposedA,
                                            A.getMemPtr(),
                                            static_cast<uint32_t>(cublasKernelRequirement->kernelRequirement.rowsA),
                                            static_cast<uint32_t>(cublasKernelRequirement->kernelRequirement.colsA),
                                            thorDataTypeForCudaDataType(cublasKernelRequirement->operationType.ADataType),
                                            thorDataTypeForCudaDataType(cublasKernelRequirement->operationType.ADataType),
                                            stream);
                ltB = transposedA;
            } else {
                ltB = A.getMemPtr();
            }

            ltWorkspace = ltWorkspaceSizeInBytes > 0 ? addBytes(workspaceBase, cublasWorkspaceOffsetInBytes()) : nullptr;
        } else {
            ltWorkspace = ltWorkspaceSizeInBytes > 0 ? workspace.value().getMemPtr() : nullptr;
        }

        cublasStatus_t cublasStatus;
        cublasStatus = cublasLtMatmul(stream.getCublasLtHandleUnchecked(),
                                      operationDesc,
                                      alpha,
                                      ltA,
                                      *ADesc,
                                      ltB,
                                      *BDesc,
                                      beta,
                                      C.getMemPtr(),
                                      *CDesc,
                                      D.getMemPtr(),
                                      *DDesc,
                                      &cublasKernelOptions->algorithm,
                                      ltWorkspace,
                                      ltWorkspaceSizeInBytes,
                                      stream);
        return cublasStatus;
    }

    std::string toString(int gpuNum) {
        THOR_THROW_IF_FALSE(!uninitialized());

        std::string description;
        description += "AlgoId " + std::to_string(cublasKernelOptions->algorithmId);

        // CUDA/cuBLASLt may add new tile enum values before Thor's debug-name map
        // is updated.  This string is used for diagnostics and kernel-contest
        // printing, so it must never turn an otherwise valid kernel into a
        // runtime failure.  Preserve the friendly name when known and fall back
        // to the numeric enum value when it is not.
        const auto tileName = tileEnumToString.find(cublasKernelOptions->tileSize);
        if (tileName != tileEnumToString.end()) {
            description += " " + tileName->second;
        } else {
            description += " CUBLASLT_MATMUL_TILE_UNKNOWN(" + std::to_string(static_cast<int>(cublasKernelOptions->tileSize)) + ")";
        }

        description += " error: " + std::to_string(cublasKernelOptions->runStats.errorFlag);
        description += " waves: " + std::to_string(getWavesCount(gpuNum));
        description += " splitK: " + std::to_string(cublasKernelOptions->splitK);
        description += " reductionFlag: " + std::to_string(cublasKernelOptions->reductionFlag);
        description += " swizzleType: " + std::to_string(cublasKernelOptions->swizzleType);
        description += " customOption: " + std::to_string(cublasKernelOptions->customOptionValue);
        description += " stagesId: " + std::to_string(cublasKernelOptions->stagesId);
        description += " innerShapeId: " + std::to_string(cublasKernelOptions->innerShapeId);
        description += " clusterShapeId: " + std::to_string(cublasKernelOptions->clusterShapeId);
        int workspaceSize = getWorkspaceSizeInBytes(gpuNum);
        description += " workspace: " + std::to_string(workspaceSize);

        if (cublasKernelOptions->runStats.runCount > 0) {
            double timePerKernelMs = cublasKernelOptions->runStats.getAverageRunTimeMilliseconds();
            std::string timePerKernelMsString = std::to_string(timePerKernelMs);

            int finalRowsA = cublasKernelRequirement->kernelRequirement.transposeA == false
                                 ? cublasKernelRequirement->kernelRequirement.rowsA
                                 : cublasKernelRequirement->kernelRequirement.colsA;
            int finalColsA = cublasKernelRequirement->kernelRequirement.transposeA == false
                                 ? cublasKernelRequirement->kernelRequirement.colsA
                                 : cublasKernelRequirement->kernelRequirement.rowsA;
            int finalColsB = cublasKernelRequirement->kernelRequirement.transposeB == false
                                 ? cublasKernelRequirement->kernelRequirement.colsB
                                 : cublasKernelRequirement->kernelRequirement.rowsB;
            double TFLOPS = (2.0 * finalRowsA * finalColsA * finalColsB) / (timePerKernelMs * 1.0e9);
            std::string TFLOPSString = std::to_string(TFLOPS);

            description += " kernelTime: " + timePerKernelMsString + "ms";
            description += " TFLOPS: " + TFLOPSString + "\n";
        }

        return description;
    }

    unsigned long getWorkspaceSizeInBytes(int gpuNum,
                                          bool &kernelWillRunOnGpu,
                                          CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) const {
        (void)gpuNum;
        (void)fp8Scales;
        THOR_THROW_IF_FALSE(!uninitialized());
        kernelWillRunOnGpu = true;
        return totalWorkspaceSizeInBytes();
    }

    unsigned long getWorkspaceSizeInBytes(int gpuNum,
                                          CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) const {
        bool kernelWillRunOnGpu = false;
        unsigned long workspaceSize = getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales);
        THOR_THROW_IF_FALSE(kernelWillRunOnGpu);
        return workspaceSize;
    }

    bool validateAlgorithmForBuild(cublasLtHandle_t ltHandle,
                                   CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none()) {
        THOR_THROW_IF_FALSE(!uninitialized());

        cublasLtMatmulHeuristicResult_t result;
        cublasLtMatmulDesc_t operationDesc = getOperationDesc(CublasScalarPointerMode::Host);
        configureTensorwideFp8Scales(operationDesc, fp8Scales);

        cublasStatus_t cublasStatus = cublasLtMatmulAlgoCheck(ltHandle,
                                                              operationDesc,
                                                              *ADesc,
                                                              *BDesc,
                                                              *CDesc,
                                                              *DDesc,
                                                              &cublasKernelOptions->algorithm,
                                                              &result);
        return cublasStatus == CUBLAS_STATUS_SUCCESS;
    }

    CublasKernelRequirement getCublasKernelRequirement() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return *cublasKernelRequirement;
    }

    CublasKernelOptions getCublasKernelOptions() {
        THOR_THROW_IF_FALSE(!uninitialized());
        return *cublasKernelOptions;
    }

    cublasLtMatmulAlgo_t getAlgorithm(int gpuNum) { return cublasKernelOptions->algorithm; }

   private:
    CublasKernelRequirement *cublasKernelRequirement;
    CublasKernelOptions *cublasKernelOptions;

    cublasLtMatmulDesc_t *operationDescHost;
    cublasLtMatmulDesc_t *operationDescDevice;
    cublasLtMatrixLayout_t *ADesc;
    cublasLtMatrixLayout_t *BDesc;
    cublasLtMatrixLayout_t *CDesc;
    cublasLtMatrixLayout_t *DDesc;

    std::string gpuType;

    static cudaDataType_t mapTensorDataTypeToCublasDataType(DataType dataType) {
        switch (dataType) {
            case DataType::FP32:
                return CUDA_R_32F;
            case DataType::BF16:
                return CUDA_R_16BF;
            case DataType::FP16:
                return CUDA_R_16F;
            case DataType::FP8_E4M3:
                return CUDA_R_8F_E4M3;
            case DataType::FP8_E5M2:
                return CUDA_R_8F_E5M2;
            case DataType::INT8:
                return CUDA_R_8I;
            default:
                THOR_UNREACHABLE();
                return CUDA_R_32F;
        }
    }

    static std::map<cublasLtMatmulTile_t, std::string> tileEnumToString;

    static void setTensorwideFp8ScaleMode(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attribute) {
        const cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasStatus_t cublasStatus = cublasLtMatmulDescSetAttribute(desc, attribute, &scaleMode, sizeof(scaleMode));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    static void setFp8ScalePointerIfPresent(cublasLtMatmulDesc_t desc,
                                            cublasLtMatmulDescAttributes_t attribute,
                                            const float *scaleDevicePointer) {
        if (scaleDevicePointer != nullptr) {
            cublasStatus_t cublasStatus = cublasLtMatmulDescSetAttribute(desc, attribute, &scaleDevicePointer, sizeof(scaleDevicePointer));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        }
    }

    static void setFp8AmaxPointerIfPresent(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attribute, float *amaxDevicePointer) {
        if (amaxDevicePointer != nullptr) {
            cublasStatus_t cublasStatus = cublasLtMatmulDescSetAttribute(desc, attribute, &amaxDevicePointer, sizeof(amaxDevicePointer));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        }
    }

    static constexpr size_t WORKSPACE_ALIGNMENT_BYTES = 256;

    static size_t alignWorkspaceSize(size_t value) { return (value + WORKSPACE_ALIGNMENT_BYTES - 1) & ~(WORKSPACE_ALIGNMENT_BYTES - 1); }

    static void *addBytes(void *ptr, size_t byteOffset) { return static_cast<void *>(static_cast<unsigned char *>(ptr) + byteOffset); }

    static const void *addBytes(const void *ptr, size_t byteOffset) {
        return static_cast<const void *>(static_cast<const unsigned char *>(ptr) + byteOffset);
    }

    static size_t cudaDataTypeSizeInBytes(cudaDataType_t dataType) {
        switch (dataType) {
            case CUDA_R_32F:
                return 4;
            case CUDA_R_16BF:
            case CUDA_R_16F:
                return 2;
            case CUDA_R_8F_E4M3:
            case CUDA_R_8F_E5M2:
            case CUDA_R_8I:
                return 1;
            default:
                THOR_UNREACHABLE();
                return 1;
        }
    }

    static DataType thorDataTypeForCudaDataType(cudaDataType_t dataType) {
        switch (dataType) {
            case CUDA_R_32F:
                return DataType::FP32;
            case CUDA_R_16BF:
                return DataType::BF16;
            case CUDA_R_16F:
                return DataType::FP16;
            case CUDA_R_8F_E4M3:
                return DataType::FP8_E4M3;
            case CUDA_R_8F_E5M2:
                return DataType::FP8_E5M2;
            case CUDA_R_8I:
                return DataType::INT8;
            default:
                THOR_UNREACHABLE();
                return DataType::UINT8;
        }
    }

    bool usesFp8ColumnMajorLtPath() const { return isCublasLtFp8OperationType(cublasKernelRequirement->operationType); }

    cudaDataType_t getLtADescDataType() const {
        return usesFp8ColumnMajorLtPath() ? cublasKernelRequirement->operationType.BDataType
                                          : cublasKernelRequirement->operationType.ADataType;
    }

    cudaDataType_t getLtBDescDataType() const {
        return usesFp8ColumnMajorLtPath() ? cublasKernelRequirement->operationType.ADataType
                                          : cublasKernelRequirement->operationType.BDataType;
    }

    CublasFp8MatmulScales getLtFp8Scales(CublasFp8MatmulScales fp8Scales) const {
        if (!usesFp8ColumnMajorLtPath()) {
            return fp8Scales;
        }
        return CublasFp8MatmulScales::tensorwide(fp8Scales.BScaleDevicePointer,
                                                 fp8Scales.AScaleDevicePointer,
                                                 fp8Scales.CScaleDevicePointer,
                                                 fp8Scales.DScaleDevicePointer,
                                                 fp8Scales.DAmaxDevicePointer);
    }

    bool fp8NeedsTransposedAWorkspace() const {
        return usesFp8ColumnMajorLtPath() && cublasKernelRequirement->kernelRequirement.transposeA;
    }

    bool fp8NeedsTransposedBWorkspace() const {
        return usesFp8ColumnMajorLtPath() && !cublasKernelRequirement->kernelRequirement.transposeB;
    }

    void validateFp8RowMajorGemmShapeAndLayoutOrThrow(const std::string &context) const {
        if (!usesFp8ColumnMajorLtPath()) {
            return;
        }

        const KernelRequirement &kr = cublasKernelRequirement->kernelRequirement;
        const int n = kr.transposeB ? kr.rowsB : kr.colsB;
        const int k = kr.transposeA ? kr.rowsA : kr.colsA;

        if ((n % 2) != 0) {
            throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires even N.");
        }
        if ((k % 2) != 0) {
            throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires even K.");
        }
        if (kr.ldA != kr.colsA) {
            throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed A: ldA must equal colsA.");
        }
        if (kr.ldB != kr.colsB) {
            throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed B: ldB must equal colsB.");
        }
        if (kr.ldC != n) {
            throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed C: ldC must equal N.");
        }
        if (kr.ldD != n) {
            throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed D: ldD must equal N.");
        }
    }

    size_t fp8TransposedAWorkspaceSizeInBytes() const {
        if (!fp8NeedsTransposedAWorkspace()) {
            return 0;
        }
        return static_cast<size_t>(cublasKernelRequirement->kernelRequirement.rowsA) *
               static_cast<size_t>(cublasKernelRequirement->kernelRequirement.colsA) *
               cudaDataTypeSizeInBytes(cublasKernelRequirement->operationType.ADataType);
    }

    size_t fp8TransposedBWorkspaceSizeInBytes() const {
        if (!fp8NeedsTransposedBWorkspace()) {
            return 0;
        }
        return static_cast<size_t>(cublasKernelRequirement->kernelRequirement.rowsB) *
               static_cast<size_t>(cublasKernelRequirement->kernelRequirement.colsB) *
               cudaDataTypeSizeInBytes(cublasKernelRequirement->operationType.BDataType);
    }

    size_t fp8TransposedAWorkspaceOffsetInBytes() const { return 0; }

    size_t fp8TransposedBWorkspaceOffsetInBytes() const { return alignWorkspaceSize(fp8TransposedAWorkspaceSizeInBytes()); }

    size_t cublasWorkspaceOffsetInBytes() const {
        return fp8TransposedBWorkspaceOffsetInBytes() + alignWorkspaceSize(fp8TransposedBWorkspaceSizeInBytes());
    }

    size_t totalWorkspaceSizeInBytes() const {
        return cublasWorkspaceOffsetInBytes() + static_cast<size_t>(cublasKernelOptions->workspaceSizeInBytes);
    }

    void configureTensorwideFp8Scales(cublasLtMatmulDesc_t desc, CublasFp8MatmulScales fp8Scales) {
        const OperationType &operationType = cublasKernelRequirement->operationType;
        const cudaDataType_t ltADataType = getLtADescDataType();
        const cudaDataType_t ltBDataType = getLtBDescDataType();
        const CublasFp8MatmulScales ltFp8Scales = getLtFp8Scales(fp8Scales);

        if (!ltFp8Scales.hasAnyScalePointer() && !isCublasLtFp8CudaType(ltADataType) && !isCublasLtFp8CudaType(ltBDataType) &&
            !isCublasLtFp8CudaType(operationType.CDataType) && !isCublasLtFp8CudaType(operationType.DDataType)) {
            return;
        }

        if (isCublasLtFp8CudaType(ltADataType)) {
            setTensorwideFp8ScaleMode(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE);
        }
        if (isCublasLtFp8CudaType(ltBDataType)) {
            setTensorwideFp8ScaleMode(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE);
        }
        if (isCublasLtFp8CudaType(operationType.CDataType) || ltFp8Scales.hasCScale()) {
            setTensorwideFp8ScaleMode(desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE);
        }
        if (isCublasLtFp8CudaType(operationType.DDataType)) {
            setTensorwideFp8ScaleMode(desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE);
        }

        setFp8ScalePointerIfPresent(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, ltFp8Scales.AScaleDevicePointer);
        setFp8ScalePointerIfPresent(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, ltFp8Scales.BScaleDevicePointer);
        setFp8ScalePointerIfPresent(desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, ltFp8Scales.CScaleDevicePointer);

        if (isCublasLtFp8CudaType(operationType.DDataType)) {
            setFp8ScalePointerIfPresent(desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, ltFp8Scales.DScaleDevicePointer);
            setFp8AmaxPointerIfPresent(desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, ltFp8Scales.DAmaxDevicePointer);
        }
    }

    void allocateCublasResources() {
        THOR_THROW_IF_FALSE(!uninitialized());

        cublasStatus_t cublasStatus;

        auto createOperationDesc = [&](cublasLtPointerMode_t pointerMode, cublasLtMatmulDesc_t *desc) {
            cublasStatus = cublasLtMatmulDescCreate(
                desc, cublasKernelRequirement->operationType.computeDataType, cublasKernelRequirement->operationType.scaleDataType);
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
            cublasStatus = cublasLtMatmulDescSetAttribute(*desc, pointerModeAttribute, &pointerMode, sizeof(pointerMode));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

            if (usesFp8ColumnMajorLtPath()) {
                // For FP8, cuBLASLt exposes the usable kernels as column-major TN.  CublasKernel keeps Thor's
                // external row-major API by making cuBLASLt compute D^T = (op(B))^T * (op(A))^T.
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
                THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
                return;
            }

            if (cublasKernelRequirement->kernelRequirement.transposeA) {
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
                THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            }
            if (cublasKernelRequirement->kernelRequirement.transposeB) {
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose));
                THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            }
            if (cublasKernelRequirement->kernelRequirement.transposeC) {
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSC, &transpose, sizeof(transpose));
                THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            }
        };

        operationDescHost = new cublasLtMatmulDesc_t;
        operationDescDevice = new cublasLtMatmulDesc_t;
        createOperationDesc(CUBLASLT_POINTER_MODE_HOST, operationDescHost);
        createOperationDesc(CUBLASLT_POINTER_MODE_DEVICE, operationDescDevice);

        int64_t ld;

        ADesc = new cublasLtMatrixLayout_t;
        BDesc = new cublasLtMatrixLayout_t;
        CDesc = new cublasLtMatrixLayout_t;
        DDesc = new cublasLtMatrixLayout_t;

        if (usesFp8ColumnMajorLtPath()) {
            const KernelRequirement &kr = cublasKernelRequirement->kernelRequirement;
            const cublasLtOrder_t columnMajorOrder = CUBLASLT_ORDER_COL;

            // Internal cuBLASLt A operand is the row-major matrix X=(op(B))^T presented as column-major X^T.
            const int internalARowMajorRows = kr.transposeB ? kr.rowsB : kr.colsB;
            const int internalARowMajorCols = kr.transposeB ? kr.colsB : kr.rowsB;
            const int internalALd = kr.transposeB ? kr.ldB : kr.rowsB;

            cublasStatus =
                cublasLtMatrixLayoutCreate(ADesc, getLtADescDataType(), internalARowMajorCols, internalARowMajorRows, internalALd);
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            cublasStatus =
                cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            ld = internalALd;
            cublasStatus = cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

            // Internal cuBLASLt B operand is the row-major matrix Y=op(A) presented as column-major Y^T.
            const int internalBRowMajorRows = kr.transposeA ? kr.colsA : kr.rowsA;
            const int internalBRowMajorCols = kr.transposeA ? kr.rowsA : kr.colsA;
            const int internalBLd = kr.transposeA ? kr.rowsA : kr.ldA;

            cublasStatus =
                cublasLtMatrixLayoutCreate(BDesc, getLtBDescDataType(), internalBRowMajorCols, internalBRowMajorRows, internalBLd);
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            cublasStatus =
                cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            ld = internalBLd;
            cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

            const int rowsD = kr.transposeA ? kr.colsA : kr.rowsA;
            const int colsD = kr.transposeB ? kr.rowsB : kr.colsB;

            cublasStatus = cublasLtMatrixLayoutCreate(CDesc, cublasKernelRequirement->operationType.CDataType, colsD, rowsD, kr.ldC);
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            cublasStatus =
                cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            ld = kr.ldC;
            cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

            cublasStatus = cublasLtMatrixLayoutCreate(DDesc, cublasKernelRequirement->operationType.DDataType, colsD, rowsD, kr.ldD);
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            cublasStatus =
                cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            ld = kr.ldD;
            cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

            return;
        }

        cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;

        cublasStatus = cublasLtMatrixLayoutCreate(ADesc,
                                                  cublasKernelRequirement->operationType.ADataType,
                                                  cublasKernelRequirement->kernelRequirement.rowsA,
                                                  cublasKernelRequirement->kernelRequirement.colsA,
                                                  cublasKernelRequirement->kernelRequirement.ldA);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldA;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

        cublasStatus = cublasLtMatrixLayoutCreate(BDesc,
                                                  cublasKernelRequirement->operationType.BDataType,
                                                  cublasKernelRequirement->kernelRequirement.rowsB,
                                                  cublasKernelRequirement->kernelRequirement.colsB,
                                                  cublasKernelRequirement->kernelRequirement.ldB);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldB;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

        int rowsC = cublasKernelRequirement->kernelRequirement.transposeA == false ? cublasKernelRequirement->kernelRequirement.rowsA
                                                                                   : cublasKernelRequirement->kernelRequirement.colsA;
        int colsC = cublasKernelRequirement->kernelRequirement.transposeB == false ? cublasKernelRequirement->kernelRequirement.colsB
                                                                                   : cublasKernelRequirement->kernelRequirement.rowsB;

        cublasStatus = cublasLtMatrixLayoutCreate(
            CDesc, cublasKernelRequirement->operationType.CDataType, rowsC, colsC, cublasKernelRequirement->kernelRequirement.ldC);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldC;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);

        cublasStatus = cublasLtMatrixLayoutCreate(
            DDesc, cublasKernelRequirement->operationType.DDataType, rowsC, colsC, cublasKernelRequirement->kernelRequirement.ldD);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldD;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    void construct(CublasKernelRequirement cublasKernelRequirement, CublasKernelOptions cublasKernelOptions, std::string gpuType) {
        ReferenceCounted::initialize();

        this->cublasKernelRequirement = new CublasKernelRequirement(cublasKernelRequirement);
        this->cublasKernelOptions = new CublasKernelOptions(cublasKernelOptions);
        this->gpuType = gpuType;

        validateFp8RowMajorGemmShapeAndLayoutOrThrow("CublasKernel::construct");
        allocateCublasResources();
    }

    void copyFrom(const CublasKernel &other) {
        *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

        cublasKernelRequirement = other.cublasKernelRequirement;
        cublasKernelOptions = other.cublasKernelOptions;
        operationDescHost = other.operationDescHost;
        operationDescDevice = other.operationDescDevice;
        ADesc = other.ADesc;
        BDesc = other.BDesc;
        CDesc = other.CDesc;
        DDesc = other.DDesc;
    }

    void destroy() {
        delete cublasKernelRequirement;
        delete cublasKernelOptions;

        cublasStatus_t cublasStatus;

        cublasStatus = cublasLtMatmulDescDestroy(*operationDescHost);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete operationDescHost;

        cublasStatus = cublasLtMatmulDescDestroy(*operationDescDevice);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete operationDescDevice;

        cublasStatus = cublasLtMatrixLayoutDestroy(*ADesc);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete ADesc;

        cublasStatus = cublasLtMatrixLayoutDestroy(*BDesc);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete BDesc;

        cublasStatus = cublasLtMatrixLayoutDestroy(*CDesc);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete CDesc;

        cublasStatus = cublasLtMatrixLayoutDestroy(*DDesc);
        THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete DDesc;
    }
};

}  // namespace ThorImplementation
