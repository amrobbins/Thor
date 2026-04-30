#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelOptions.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelRequirement.h"

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <atomic>
#include <stdexcept>
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
        assert(!uninitialized());
        cublasKernelOptions->runStats.errorFlag = true;
    }

    bool getErrorFlag() const {
        assert(!uninitialized());
        return cublasKernelOptions->runStats.errorFlag;
    }

    void recordRun(double executionTimeOfRun) { cublasKernelOptions->runStats.recordRun(executionTimeOfRun); }

    double getAverageRunTimeMilliseconds() { return cublasKernelOptions->runStats.getAverageRunTimeMilliseconds(); }

    void stashRunStats() { cublasKernelOptions->runStats.stashRunStats(); }

    void unstashRunStats() { cublasKernelOptions->runStats.unstashRunStats(); }

    cublasLtMatmulDesc_t getOperationDesc(CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host) {
        assert(!uninitialized());
        return (pointerMode == CublasScalarPointerMode::Device) ? *operationDescDevice : *operationDescHost;
    }

    cublasLtMatrixLayout_t getADesc() {
        assert(!uninitialized());
        return *ADesc;
    }

    cublasLtMatrixLayout_t getBDesc() {
        assert(!uninitialized());
        return *BDesc;
    }

    cublasLtMatrixLayout_t getCDesc() {
        assert(!uninitialized());
        return *CDesc;
    }

    cublasLtMatrixLayout_t getDDesc() {
        assert(!uninitialized());
        return *DDesc;
    }

    float getWavesCount(int gpuNum) const {
        assert(!uninitialized());

        return cublasKernelOptions->wavesCount;
    }

    static inline bool executionTimeComparison(CublasKernel &lhs, CublasKernel &rhs) {
        assert(!lhs.uninitialized());
        assert(!rhs.uninitialized());
        return lhs.cublasKernelOptions->runStats < rhs.cublasKernelOptions->runStats;
    }

    void executeKernel(Tensor A,
                       Tensor B,
                       Tensor C,
                       Tensor D,
                       Optional<Tensor> workspace,
                       const float *alpha,
                       const float *beta,
                       Stream stream,
                       CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host) {
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
                      pointerMode);
    }

    void executeKernel(Tensor A,
                       Tensor B,
                       Tensor C,
                       Tensor D,
                       size_t ldA,
                       size_t ldB,
                       size_t ldC,
                       size_t ldD,
                       Optional<Tensor> workspace,
                       const float *alpha,
                       const float *beta,
                       Stream stream,
                       CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host) {
        assert(!uninitialized());

        uint64_t rowsC = cublasKernelRequirement->kernelRequirement.transposeA == false ? cublasKernelRequirement->kernelRequirement.rowsA
                                                                                        : cublasKernelRequirement->kernelRequirement.colsA;

        // Check that everything matches up
        std::vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
        assert(ADimensions.size() == 2);
        assert(ADimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsA);
        assert(ADimensions[1] == ldA);
        assert(mapTensorDataTypeToCublasDataType(A.getDescriptor().getDataType()) == cublasKernelRequirement->operationType.ADataType);

        std::vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
        assert(BDimensions.size() == 2);
        assert(BDimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsB);
        assert(BDimensions[1] == ldB);
        assert(mapTensorDataTypeToCublasDataType(B.getDescriptor().getDataType()) == cublasKernelRequirement->operationType.BDataType);

        std::vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
        assert(CDimensions.size() == 2);
        assert(CDimensions[0] == rowsC);
        assert(CDimensions[1] == ldC);
        assert(mapTensorDataTypeToCublasDataType(C.getDescriptor().getDataType()) == cublasKernelRequirement->operationType.CDataType);

        std::vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
        assert(DDimensions.size() == 2);
        assert(DDimensions[0] == rowsC);
        assert(DDimensions[1] == ldD);
        assert(mapTensorDataTypeToCublasDataType(D.getDescriptor().getDataType()) == cublasKernelRequirement->operationType.DDataType);

        assert(C.getMemPtr() != A.getMemPtr());
        assert(C.getMemPtr() != B.getMemPtr());

        assert(runWithoutChecks(A, B, C, D, workspace, alpha, beta, stream, pointerMode) == CUBLAS_STATUS_SUCCESS);
    }

    inline cublasStatus_t runWithoutChecks(Tensor A,
                                           Tensor B,
                                           Tensor C,
                                           Tensor D,
                                           Optional<Tensor> workspace,
                                           const float *alpha,
                                           const float *beta,
                                           Stream stream,
                                           CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host) {
        assert(!uninitialized());
        ScopedGpu scopedGpu(stream.getGpuNum());

        size_t workspaceSizeInBytes = 0;
        bool kernelWillRunOnGpu;
        size_t requiredWorkspaceSize = getWorkspaceSizeInBytes(stream.getGpuNum(), kernelWillRunOnGpu);
        assert(kernelWillRunOnGpu);
        if (workspace.isPresent() && requiredWorkspaceSize > 0) {
            workspaceSizeInBytes = workspace.get().getDescriptor().getArraySizeInBytes();
        }
        assert(workspaceSizeInBytes >= requiredWorkspaceSize);

        cublasStatus_t cublasStatus;
        cublasStatus = cublasLtMatmul(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
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
                                      requiredWorkspaceSize > 0 ? workspace.get().getMemPtr() : nullptr,
                                      requiredWorkspaceSize > 0 ? workspace.get().getDescriptor().getArraySizeInBytes() : 0,
                                      stream);
        return cublasStatus;
    }

    std::string toString(int gpuNum) {
        assert(!uninitialized());

        std::string description;
        description += "AlgoId " + std::to_string(cublasKernelOptions->algorithmId);
        assert(tileEnumToString.count(cublasKernelOptions->tileSize) == 1);
        description += " " + tileEnumToString[cublasKernelOptions->tileSize];
        description += " error: " + std::to_string(cublasKernelOptions->runStats.errorFlag);
        description += " waves: " + std::to_string(getWavesCount(gpuNum));
        description += " splitK: " + std::to_string(cublasKernelOptions->splitK);
        description += " reductionFlag: " + std::to_string(cublasKernelOptions->reductionFlag);
        description += " swizzleType: " + std::to_string(cublasKernelOptions->swizzleType);
        description += " customOption: " + std::to_string(cublasKernelOptions->customOptionValue);
        description += " stagesId: " + std::to_string(cublasKernelOptions->stagesId);
        description += " innerShapeId: " + std::to_string(cublasKernelOptions->innerShapeId);
        description += " clusterShapeId: " + std::to_string(cublasKernelOptions->clusterShapeId);
        bool kernelWillRunOnGpu;
        int workspaceSize = getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu);
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

    unsigned long getWorkspaceSizeInBytes(int gpuNum, bool &kernelWillRunOnGpu) {
        assert(!uninitialized());

        cublasStatus_t cublasStatus;
        cublasLtMatmulHeuristicResult_t result;

        cublasStatus = cublasLtMatmulAlgoCheck(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                               getOperationDesc(CublasScalarPointerMode::Host),
                                               *ADesc,
                                               *BDesc,
                                               *CDesc,
                                               *DDesc,
                                               &cublasKernelOptions->algorithm,
                                               &result);
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            kernelWillRunOnGpu = false;
        } else {
            kernelWillRunOnGpu = true;
        }

        return cublasKernelOptions->workspaceSizeInBytes;
    }

    CublasKernelRequirement getCublasKernelRequirement() {
        assert(!uninitialized());
        return *cublasKernelRequirement;
    }

    CublasKernelOptions getCublasKernelOptions() {
        assert(!uninitialized());
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

    static cudaDataType_t mapTensorDataTypeToCublasDataType(TensorDescriptor::DataType dataType) {
        switch (dataType) {
            case TensorDescriptor::DataType::FP32:
                return CUDA_R_32F;
            case TensorDescriptor::DataType::BF16:
                return CUDA_R_16BF;
            case TensorDescriptor::DataType::FP16:
                return CUDA_R_16F;
            case TensorDescriptor::DataType::FP8_E4M3:
                return CUDA_R_8F_E4M3;
            case TensorDescriptor::DataType::FP8_E5M2:
                return CUDA_R_8F_E5M2;
            case TensorDescriptor::DataType::INT8:
                return CUDA_R_8I;
            default:
                assert(false);
                return CUDA_R_32F;
        }
    }

    static std::map<cublasLtMatmulTile_t, std::string> tileEnumToString;

    void allocateCublasResources() {
        assert(!uninitialized());

        cublasStatus_t cublasStatus;

        auto createOperationDesc = [&](cublasLtPointerMode_t pointerMode, cublasLtMatmulDesc_t *desc) {
            cublasStatus = cublasLtMatmulDescCreate(
                desc, cublasKernelRequirement->operationType.computeDataType, cublasKernelRequirement->operationType.scaleDataType);
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
            cublasStatus = cublasLtMatmulDescSetAttribute(*desc, pointerModeAttribute, &pointerMode, sizeof(pointerMode));
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            if (cublasKernelRequirement->kernelRequirement.transposeA) {
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
                assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            }
            if (cublasKernelRequirement->kernelRequirement.transposeB) {
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose));
                assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            }
            if (cublasKernelRequirement->kernelRequirement.transposeC) {
                cublasOperation_t transpose = CUBLAS_OP_T;
                cublasStatus = cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSC, &transpose, sizeof(transpose));
                assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            }
        };

        operationDescHost = new cublasLtMatmulDesc_t;
        operationDescDevice = new cublasLtMatmulDesc_t;
        createOperationDesc(CUBLASLT_POINTER_MODE_HOST, operationDescHost);
        createOperationDesc(CUBLASLT_POINTER_MODE_DEVICE, operationDescDevice);

        cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
        int64_t ld;

        ADesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(ADesc,
                                                  cublasKernelRequirement->operationType.ADataType,
                                                  cublasKernelRequirement->kernelRequirement.rowsA,
                                                  cublasKernelRequirement->kernelRequirement.colsA,
                                                  cublasKernelRequirement->kernelRequirement.ldA);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldA;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        BDesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(BDesc,
                                                  cublasKernelRequirement->operationType.BDataType,
                                                  cublasKernelRequirement->kernelRequirement.rowsB,
                                                  cublasKernelRequirement->kernelRequirement.colsB,
                                                  cublasKernelRequirement->kernelRequirement.ldB);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldB;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        int rowsC = cublasKernelRequirement->kernelRequirement.transposeA == false ? cublasKernelRequirement->kernelRequirement.rowsA
                                                                                   : cublasKernelRequirement->kernelRequirement.colsA;
        int colsC = cublasKernelRequirement->kernelRequirement.transposeB == false ? cublasKernelRequirement->kernelRequirement.colsB
                                                                                   : cublasKernelRequirement->kernelRequirement.rowsB;

        CDesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(
            CDesc, cublasKernelRequirement->operationType.CDataType, rowsC, colsC, cublasKernelRequirement->kernelRequirement.ldC);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldC;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        DDesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(
            DDesc, cublasKernelRequirement->operationType.DDataType, rowsC, colsC, cublasKernelRequirement->kernelRequirement.ldD);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldD;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    void construct(CublasKernelRequirement cublasKernelRequirement, CublasKernelOptions cublasKernelOptions, std::string gpuType) {
        ReferenceCounted::initialize();

        this->cublasKernelRequirement = new CublasKernelRequirement(cublasKernelRequirement);
        this->cublasKernelOptions = new CublasKernelOptions(cublasKernelOptions);
        this->gpuType = gpuType;

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
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete operationDescHost;

        cublasStatus = cublasLtMatmulDescDestroy(*operationDescDevice);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete operationDescDevice;

        cublasStatus = cublasLtMatrixLayoutDestroy(*ADesc);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete ADesc;

        cublasStatus = cublasLtMatrixLayoutDestroy(*BDesc);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete BDesc;

        cublasStatus = cublasLtMatrixLayoutDestroy(*CDesc);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete CDesc;

        cublasStatus = cublasLtMatrixLayoutDestroy(*DDesc);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        delete DDesc;
    }
};

}  // namespace ThorImplementation
