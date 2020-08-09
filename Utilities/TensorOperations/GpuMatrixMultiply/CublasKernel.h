#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelRequirement.h"

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

struct RunStats {
    RunStats() {
        errorFlag = false;
        runCount = 0;
        totalExecutionTimeMilliseconds = 0.0;
        stashedRunCount = 0;
        stashedExecutionTimeMilliseconds = 0.0;
    }

    RunStats(const RunStats &other) {
        // implemented using operator=
        *this = other;
    }

    RunStats &operator=(const RunStats &other) {
        errorFlag = other.errorFlag;
        runCount = other.runCount;
        totalExecutionTimeMilliseconds = other.totalExecutionTimeMilliseconds;
        stashedRunCount = other.stashedRunCount;
        stashedExecutionTimeMilliseconds = other.stashedExecutionTimeMilliseconds;
        return *this;
    }

    bool errorFlag;
    int runCount;
    double totalExecutionTimeMilliseconds;

    int stashedRunCount;
    double stashedExecutionTimeMilliseconds;

    void recordRun(double executionTimeOfRun) {
        mtx.lock();
        runCount += 1;
        totalExecutionTimeMilliseconds += executionTimeOfRun;
        mtx.unlock();
    }

    inline double getAverageRunTimeMilliseconds() {
        // Updates should not be concurrently ongoing when running this function.
        assert(runCount > 0);
        return totalExecutionTimeMilliseconds / runCount;
    }

    void stashRunStats() {
        mtx.lock();
        stashedRunCount += runCount;
        stashedExecutionTimeMilliseconds += totalExecutionTimeMilliseconds;

        runCount = 0;
        totalExecutionTimeMilliseconds = 0.0;
        mtx.unlock();
    }

    void unstashRunStats() {
        mtx.lock();
        runCount += stashedRunCount;
        totalExecutionTimeMilliseconds += stashedExecutionTimeMilliseconds;

        stashedRunCount = 0;
        stashedExecutionTimeMilliseconds = 0.0;
        mtx.unlock();
    }

    inline bool operator<(RunStats &rhs) {
        assert(!errorFlag);
        assert(!rhs.errorFlag);
        return getAverageRunTimeMilliseconds() < rhs.getAverageRunTimeMilliseconds();
    }

   private:
    std::mutex mtx;
};

struct CublasKernelOptions {
    CublasKernelOptions(int algorithmId,
                        cublasLtMatmulTile_t tileSize,
                        uint32_t splitK,
                        uint32_t reductionFlag,
                        uint32_t swizzleType,
                        uint32_t customOptionValue)
        : algorithmId(algorithmId),
          tileSize(tileSize),
          splitK(splitK),
          reductionFlag(reductionFlag),
          swizzleType(swizzleType),
          customOptionValue(customOptionValue) {}

    int algorithmId;
    cublasLtMatmulTile_t tileSize;
    uint32_t splitK;
    uint32_t reductionFlag;
    uint32_t swizzleType;
    uint32_t customOptionValue;

    RunStats runStats;

    inline bool operator<(CublasKernelOptions &rhs) { return runStats < rhs.runStats; }

    inline bool operator==(const CublasKernelOptions &other) const {
        return algorithmId == other.algorithmId && splitK == other.splitK && reductionFlag == other.reductionFlag &&
               swizzleType == other.swizzleType && customOptionValue == other.customOptionValue;
    }
};

class CublasKernel {
   public:
    static const float ALPHA_NO_SCALE;
    static const float BETA_ACCUMULATE;
    static const float BETA_CLEAR;

    CublasKernel() {
        uninitialized = true;
        referenceCount = nullptr;
    }

    CublasKernel(CublasKernelRequirement cublasKernelRequirement, CublasKernelOptions cublasKernelOptions, string gpuType) {
        referenceCount = new atomic<int>(1);

        uninitialized = false;

        this->cublasKernelRequirement = new CublasKernelRequirement(cublasKernelRequirement);
        this->cublasKernelOptions = new CublasKernelOptions(cublasKernelOptions);
        this->gpuType = gpuType;
        this->algorithmPerGpu = new vector<cublasLtMatmulAlgo_t>(MachineEvaluator::instance().getNumGpus());

        allocateCublasResources();
    }

    CublasKernel(const CublasKernel &other) {
        uninitialized = true;
        referenceCount = nullptr;

        // implemented using operator=
        *this = other;
    }

    CublasKernel &operator=(const CublasKernel &other) {
        // Do not reorder the increment/decrement of refCount here or object may be destroyed prematurely
        if (!other.uninitialized) {
            // other stream is initialized
            other.referenceCount->fetch_add(1);
            if (!uninitialized) {
                // this stream was previously initialized
                removeReference();
            }
            uninitialized = false;
            referenceCount = other.referenceCount;

            cublasKernelRequirement = other.cublasKernelRequirement;
            cublasKernelOptions = other.cublasKernelOptions;
            operationDesc = other.operationDesc;
            ADesc = other.ADesc;
            BDesc = other.BDesc;
            CDesc = other.CDesc;
            DDesc = other.DDesc;
            algorithmPerGpu = other.algorithmPerGpu;

            return *this;
        } else {
            // other stream is not initialized
            if (!uninitialized) {
                // this stream was previously initialized
                removeReference();
            }
            uninitialized = true;
            referenceCount = nullptr;
            return *this;
        }
    }

    virtual ~CublasKernel() { removeReference(); }

    inline bool operator==(const CublasKernel &other) const {
        return cublasKernelRequirement == other.cublasKernelRequirement && cublasKernelOptions == other.cublasKernelOptions;
    }

    void setErrorFlag() {
        assert(!uninitialized);
        cublasKernelOptions->runStats.errorFlag = true;
    }

    bool getErrorFlag() const {
        assert(!uninitialized);
        return cublasKernelOptions->runStats.errorFlag;
    }

    void recordRun(double executionTimeOfRun) { cublasKernelOptions->runStats.recordRun(executionTimeOfRun); }

    double getAverageRunTimeMilliseconds() { return cublasKernelOptions->runStats.getAverageRunTimeMilliseconds(); }

    void stashRunStats() { cublasKernelOptions->runStats.stashRunStats(); }

    void unstashRunStats() { cublasKernelOptions->runStats.unstashRunStats(); }

    cublasLtMatmulDesc_t getOperationDesc() {
        assert(!uninitialized);
        return *operationDesc;
    }

    cublasLtMatrixLayout_t getADesc() {
        assert(!uninitialized);
        return *ADesc;
    }

    cublasLtMatrixLayout_t getBDesc() {
        assert(!uninitialized);
        return *BDesc;
    }

    cublasLtMatrixLayout_t getCDesc() {
        assert(!uninitialized);
        return *CDesc;
    }

    cublasLtMatrixLayout_t getDDesc() {
        assert(!uninitialized);
        return *DDesc;
    }

    float getWavesCount(int gpuNum) const {
        assert(!uninitialized);

        cublasStatus_t cublasStatus;
        cublasLtMatmulHeuristicResult_t result;

        cublasStatus = cublasLtMatmulAlgoCheck(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                               *operationDesc,
                                               *ADesc,
                                               *BDesc,
                                               *CDesc,
                                               *DDesc,
                                               &((*algorithmPerGpu)[gpuNum]),
                                               &result);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        return result.wavesCount;
    }

    static inline bool executionTimeComparison(CublasKernel &lhs, CublasKernel &rhs) {
        assert(!lhs.uninitialized);
        assert(!rhs.uninitialized);
        return lhs.cublasKernelOptions->runStats < rhs.cublasKernelOptions->runStats;
    }

    void executeKernel(Tensor A, Tensor B, Tensor C, Tensor D, Optional<Tensor> workspace, bool accumulate, Stream stream) {
        executeKernel(A,
                      B,
                      C,
                      D,
                      A.getDescriptor().getDimensions()[1],
                      B.getDescriptor().getDimensions()[1],
                      C.getDescriptor().getDimensions()[1],
                      D.getDescriptor().getDimensions()[1],
                      workspace,
                      accumulate,
                      stream);
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
                       bool accumulate,
                       Stream stream) {
        assert(!uninitialized);

        // Check that everything matches up
        vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
        assert(ADimensions.size() == 2);
        assert(ADimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsA);
        assert(ADimensions[1] == ldA);
        if (A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
            assert(cublasKernelRequirement->operationType.getADataType() == CUDA_R_32F);
        else
            assert(cublasKernelRequirement->operationType.getADataType() == CUDA_R_16F);

        vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
        assert(BDimensions.size() == 2);
        assert(BDimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.colsA);
        assert(BDimensions[1] == ldB);
        if (B.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
            assert(cublasKernelRequirement->operationType.getBDataType() == CUDA_R_32F);
        else
            assert(cublasKernelRequirement->operationType.getBDataType() == CUDA_R_16F);

        vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
        assert(CDimensions.size() == 2);
        assert(CDimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsA);
        assert(CDimensions[1] == ldC);
        if (C.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
            assert(cublasKernelRequirement->operationType.getCDataType() == CUDA_R_32F);
        else
            assert(cublasKernelRequirement->operationType.getCDataType() == CUDA_R_16F);

        vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
        assert(DDimensions.size() == 2);
        assert(DDimensions[0] == (uint64_t)cublasKernelRequirement->kernelRequirement.rowsA);
        assert(DDimensions[1] == ldD);
        if (D.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32)
            assert(cublasKernelRequirement->operationType.getDDataType() == CUDA_R_32F);
        else
            assert(cublasKernelRequirement->operationType.getDDataType() == CUDA_R_16F);

        assert(C.getMemPtr() != A.getMemPtr());
        assert(C.getMemPtr() != B.getMemPtr());
        assert(C.getMemPtr() == D.getMemPtr());

        assert(runWithoutChecks(A, B, C, D, workspace, accumulate, stream) == CUBLAS_STATUS_SUCCESS);
    }

    inline cublasStatus_t runWithoutChecks(
        Tensor A, Tensor B, Tensor C, Tensor D, Optional<Tensor> workspace, bool accumulate, Stream stream) {
        assert(!uninitialized);
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
                                      *operationDesc,
                                      &ALPHA_NO_SCALE,
                                      A.getMemPtr(),
                                      *ADesc,
                                      B.getMemPtr(),
                                      *BDesc,
                                      accumulate ? &BETA_ACCUMULATE : &BETA_CLEAR,
                                      C.getMemPtr(),
                                      *CDesc,
                                      D.getMemPtr(),
                                      *DDesc,
                                      &((*algorithmPerGpu)[stream.getGpuNum()]),
                                      requiredWorkspaceSize > 0 ? workspace.get().getMemPtr() : nullptr,
                                      requiredWorkspaceSize > 0 ? workspace.get().getDescriptor().getArraySizeInBytes() : 0,
                                      stream);
        return cublasStatus;
    }

    string toString(int gpuNum) {
        assert(!uninitialized);

        string description;
        description += "AlgoId " + std::to_string(cublasKernelOptions->algorithmId);
        assert(tileEnumToString.count(cublasKernelOptions->tileSize) == 1);
        description += " " + tileEnumToString[cublasKernelOptions->tileSize];
        description += " error: " + std::to_string(cublasKernelOptions->runStats.errorFlag);
        description += " waves: " + std::to_string(getWavesCount(gpuNum));
        description += " splitK: " + std::to_string(cublasKernelOptions->splitK);
        description += " reductionFlag: " + std::to_string(cublasKernelOptions->reductionFlag);
        description += " swizzleType: " + std::to_string(cublasKernelOptions->swizzleType);
        description += " customOption: " + std::to_string(cublasKernelOptions->customOptionValue);
        bool kernelWillRunOnGpu;
        int workspaceSize = getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu);
        description += " workspace: " + std::to_string(workspaceSize);

        double timePerKernelMs = cublasKernelOptions->runStats.getAverageRunTimeMilliseconds();
        string timePerKernelMsString = std::to_string(timePerKernelMs);
        double TFLOPS = (2.0 * cublasKernelRequirement->kernelRequirement.rowsA * cublasKernelRequirement->kernelRequirement.colsA *
                         cublasKernelRequirement->kernelRequirement.colsB) /
                        (timePerKernelMs * 1.0e9);
        string TFLOPSString = std::to_string(TFLOPS);

        description += " kernelTime: " + timePerKernelMsString + "ms";
        description += " TFLOPS: " + TFLOPSString + "\n";

        return description;
    }

    unsigned long getWorkspaceSizeInBytes(int gpuNum, bool &kernelWillRunOnGpu) {
        assert(!uninitialized);

        cublasStatus_t cublasStatus;
        cublasLtMatmulHeuristicResult_t result;

        cublasStatus = cublasLtMatmulAlgoCheck(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                               *operationDesc,
                                               *ADesc,
                                               *BDesc,
                                               *CDesc,
                                               *DDesc,
                                               &((*algorithmPerGpu)[gpuNum]),
                                               &result);
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            kernelWillRunOnGpu = false;
            return 0;
        }

        kernelWillRunOnGpu = true;
        return result.workspaceSize;
    }

    CublasKernelRequirement getCublasKernelRequirement() {
        assert(!uninitialized);
        return *cublasKernelRequirement;
    }

    CublasKernelOptions getCublasKernelOptions() {
        assert(!uninitialized);
        return *cublasKernelOptions;
    }

    cublasLtMatmulAlgo_t getAlgorithm(int gpuNum) { return (*algorithmPerGpu)[gpuNum]; }

   private:
    CublasKernelRequirement *cublasKernelRequirement;
    CublasKernelOptions *cublasKernelOptions;

    cublasLtMatmulDesc_t *operationDesc;
    cublasLtMatrixLayout_t *ADesc;
    cublasLtMatrixLayout_t *BDesc;
    cublasLtMatrixLayout_t *CDesc;
    cublasLtMatrixLayout_t *DDesc;

    string gpuType;

    vector<cublasLtMatmulAlgo_t> *algorithmPerGpu;

    atomic<int> *referenceCount;

    bool uninitialized;

    static map<cublasLtMatmulTile_t, string> tileEnumToString;

    void allocateCublasResources() {
        assert(!uninitialized);

        cublasStatus_t cublasStatus;

        operationDesc = new cublasLtMatmulDesc_t;
        cublasStatus = cublasLtMatmulDescCreate(operationDesc,
                                                cublasKernelRequirement->operationType.getComputeDataType(),
                                                cublasKernelRequirement->operationType.getScaleDataType());
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
        const cublasLtPointerMode_t hostPointerMode = CUBLASLT_POINTER_MODE_HOST;
        cublasStatus = cublasLtMatmulDescSetAttribute(*operationDesc, pointerModeAttribute, &hostPointerMode, sizeof(hostPointerMode));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
        int64_t ld;

        ADesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(ADesc,
                                                  cublasKernelRequirement->operationType.getADataType(),
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
                                                  cublasKernelRequirement->operationType.getBDataType(),
                                                  cublasKernelRequirement->kernelRequirement.colsA,
                                                  cublasKernelRequirement->kernelRequirement.colsB,
                                                  cublasKernelRequirement->kernelRequirement.ldB);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldB;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        CDesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(CDesc,
                                                  cublasKernelRequirement->operationType.getCDataType(),
                                                  cublasKernelRequirement->kernelRequirement.rowsA,
                                                  cublasKernelRequirement->kernelRequirement.colsB,
                                                  cublasKernelRequirement->kernelRequirement.ldC);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldC;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        DDesc = new cublasLtMatrixLayout_t;
        cublasStatus = cublasLtMatrixLayoutCreate(DDesc,
                                                  cublasKernelRequirement->operationType.getDDataType(),
                                                  cublasKernelRequirement->kernelRequirement.rowsA,
                                                  cublasKernelRequirement->kernelRequirement.colsB,
                                                  cublasKernelRequirement->kernelRequirement.ldC);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        ld = cublasKernelRequirement->kernelRequirement.ldC;
        cublasStatus = cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        for (unsigned int gpuNum = 0; gpuNum < MachineEvaluator::instance().getNumGpus(); ++gpuNum) {
            // Instantiate an algorithm object for each gpu of the proper type
            // Note that to initialize an algo, you must pass the per-gpu cublas handle
            if (MachineEvaluator::instance().getGpuType(gpuNum) != gpuType)
                continue;

            cublasStatus = cublasLtMatmulAlgoInit(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                                  cublasKernelRequirement->operationType.getComputeDataType(),
                                                  cublasKernelRequirement->operationType.getScaleDataType(),
                                                  cublasKernelRequirement->operationType.getADataType(),
                                                  cublasKernelRequirement->operationType.getBDataType(),
                                                  cublasKernelRequirement->operationType.getCDataType(),
                                                  cublasKernelRequirement->operationType.getDDataType(),
                                                  cublasKernelOptions->algorithmId,
                                                  &((*algorithmPerGpu)[gpuNum]));
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

            cublasLtMatmulAlgoConfigSetAttribute(&((*algorithmPerGpu)[gpuNum]),
                                                 CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                                 &cublasKernelOptions->customOptionValue,
                                                 sizeof(cublasKernelOptions->customOptionValue));
            cublasLtMatmulAlgoConfigSetAttribute(&((*algorithmPerGpu)[gpuNum]),
                                                 CUBLASLT_ALGO_CONFIG_TILE_ID,
                                                 &cublasKernelOptions->tileSize,
                                                 sizeof(cublasKernelOptions->tileSize));
            cublasLtMatmulAlgoConfigSetAttribute(&((*algorithmPerGpu)[gpuNum]),
                                                 CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                 &cublasKernelOptions->splitK,
                                                 sizeof(cublasKernelOptions->splitK));
            cublasLtMatmulAlgoConfigSetAttribute(&((*algorithmPerGpu)[gpuNum]),
                                                 CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                                 &cublasKernelOptions->swizzleType,
                                                 sizeof(cublasKernelOptions->swizzleType));
            cublasLtMatmulAlgoConfigSetAttribute(&((*algorithmPerGpu)[gpuNum]),
                                                 CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                 &cublasKernelOptions->reductionFlag,
                                                 sizeof(cublasKernelOptions->reductionFlag));
        }
    }

    void removeReference() {
        if (uninitialized) {
            assert(referenceCount == nullptr);
            return;
        }

        int refCountBeforeDecrement = referenceCount->fetch_sub(1);
        if (refCountBeforeDecrement == 1) {
            delete referenceCount;
            referenceCount = nullptr;

            delete cublasKernelRequirement;
            delete cublasKernelOptions;

            cublasStatus_t cublasStatus;

            cublasStatus = cublasLtMatmulDescDestroy(*operationDesc);
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            delete operationDesc;

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

            delete algorithmPerGpu;
        }
    }
};
