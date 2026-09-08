#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <climits>
#include <limits>
#include <utility>

using namespace std;

namespace ThorImplementation {

#if !defined(CUDNN_VERSION) || CUDNN_VERSION < 9000
#error "Thor CTC requires cuDNN 9+ so cudnnSetCTCLossDescriptor_v9 and cudnnCTCGradMode_t are available. No older cuDNN fallback is provided."
#endif

namespace {

cudnnCTCLossAlgo_t toCudnnAlgo(CtcLossAlgorithm algorithm) {
    switch (algorithm) {
        case CtcLossAlgorithm::DETERMINISTIC:
            return CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
    }
    THOR_UNREACHABLE();
    return CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
}

cudnnLossNormalizationMode_t toCudnnNormalization(CtcLossNormalization normalization) {
    switch (normalization) {
        case CtcLossNormalization::SOFTMAX:
            return CUDNN_LOSS_NORMALIZATION_SOFTMAX;
    }
    THOR_UNREACHABLE();
    return CUDNN_LOSS_NORMALIZATION_SOFTMAX;
}

cudnnCTCGradMode_t toCudnnOobGradientMode(CtcLossOobGradientMode mode) {
    switch (mode) {
        case CtcLossOobGradientMode::ZERO:
            return CUDNN_CTC_ZERO_OOB_GRADIENTS;
        case CtcLossOobGradientMode::SKIP:
            return CUDNN_CTC_SKIP_OOB_GRADIENTS;
    }
    THOR_UNREACHABLE();
    return CUDNN_CTC_ZERO_OOB_GRADIENTS;
}

int checkedInt(uint32_t value, const char *what) {
    (void)what;
    THOR_THROW_IF_FALSE(value <= static_cast<uint32_t>(numeric_limits<int>::max()));
    return static_cast<int>(value);
}

void setBatchMajorPhysicalCtcTensorDescriptor(cudnnTensorDescriptor_t descriptor,
                                              const CudnnCtcLossConfig &config,
                                              uint32_t batchSize) {
    THOR_THROW_IF_FALSE(descriptor != nullptr);
    THOR_THROW_IF_FALSE(batchSize >= 1 && batchSize <= config.batchSize);

    const int dimA[3] = {checkedInt(config.maxTimeSteps, "maxTimeSteps"),
                         checkedInt(batchSize, "batchSize"),
                         checkedInt(config.numClasses, "numClasses")};
    // Thor physical memory is contiguous [B, T, C]. cuDNN logical indices are [T, B, C], so:
    //   offset(t,b,c) = b * T * C + t * C + c
    // The batch stride is the physical row width and therefore does not change for a partial batch.
    const int strideA[3] = {checkedInt(config.numClasses, "numClasses"),
                            checkedInt(config.maxTimeSteps * config.numClasses, "maxTimeSteps * numClasses"),
                            1};

    const cudnnStatus_t cudnnStatus =
        cudnnSetTensorNdDescriptor(descriptor, CUDNN_DATA_FLOAT, 3, dimA, strideA);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

cudnnTensorDescriptor_t createBatchMajorPhysicalCtcTensorDescriptor(const CudnnCtcLossConfig &config) {
    cudnnTensorDescriptor_t descriptor = nullptr;
    cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

    try {
        setBatchMajorPhysicalCtcTensorDescriptor(descriptor, config, config.batchSize);
    } catch (...) {
        (void)cudnnDestroyTensorDescriptor(descriptor);
        throw;
    }
    return descriptor;
}

}  // namespace

void CudnnCtcLossPlan::validateConfig(const CudnnCtcLossConfig &config) {
    THOR_THROW_IF_FALSE(config.maxTimeSteps > 0);
    THOR_THROW_IF_FALSE(config.batchSize > 0);
    THOR_THROW_IF_FALSE(config.numClasses > 1);
    THOR_THROW_IF_FALSE(config.dataType == DataType::FP32);

    THOR_THROW_IF_FALSE(config.maxTimeSteps <= static_cast<uint32_t>(numeric_limits<int>::max()));
    THOR_THROW_IF_FALSE(config.batchSize <= static_cast<uint32_t>(numeric_limits<int>::max()));
    THOR_THROW_IF_FALSE(config.numClasses <= static_cast<uint32_t>(numeric_limits<int>::max()));
    THOR_THROW_IF_FALSE(config.maxLabelLength <= static_cast<uint32_t>(numeric_limits<int>::max()));
    THOR_THROW_IF_FALSE(config.batchSize <= static_cast<uint32_t>(numeric_limits<int>::max()) / config.numClasses);
    THOR_THROW_IF_FALSE(config.maxTimeSteps <= static_cast<uint32_t>(numeric_limits<int>::max()) / config.numClasses);

    switch (config.algorithm) {
        case CtcLossAlgorithm::DETERMINISTIC:
            // cuDNN reports NOT_SUPPORTED for deterministic CTC when maxLabelLength >= 256.
            // Thor rejects it before graph construction rather than falling back to another implementation.
            THOR_THROW_IF_FALSE(config.maxLabelLength < 256);
            break;
    }

    switch (config.normalization) {
        case CtcLossNormalization::SOFTMAX:
            break;
    }

    switch (config.oobGradientMode) {
        case CtcLossOobGradientMode::ZERO:
        case CtcLossOobGradientMode::SKIP:
            break;
    }
}

CudnnCtcLossPlan::CudnnCtcLossPlan(const CudnnCtcLossConfig &config, Stream stream) : config(config) {
    THOR_THROW_IF_FALSE(stream.isInitialized());
    validateConfig(config);
    ScopedGpu scopedGpu(stream.getGpuNum());

    probabilitiesDesc = createBatchMajorPhysicalCtcTensorDescriptor(config);
    gradientsDesc = createBatchMajorPhysicalCtcTensorDescriptor(config);

    cudnnStatus_t cudnnStatus = cudnnCreateCTCLossDescriptor(&ctcLossDesc);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

    cudnnStatus = cudnnSetCTCLossDescriptor_v9(ctcLossDesc,
                                               CUDNN_DATA_FLOAT,
                                               toCudnnNormalization(config.normalization),
                                               toCudnnOobGradientMode(config.oobGradientMode),
                                               checkedInt(config.maxLabelLength, "maxLabelLength"));
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

    cudnnStatus = cudnnGetCTCLossWorkspaceSize_v8(stream.getCudnnHandle(),
                                                  toCudnnAlgo(config.algorithm),
                                                  ctcLossDesc,
                                                  probabilitiesDesc,
                                                  gradientsDesc,
                                                  &workspaceSizeInBytes);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    currentWorkspaceSizeInBytes = workspaceSizeInBytes;
    currentBatchSize = config.batchSize;
    workspaceSizeByBatchSize.emplace(config.batchSize, workspaceSizeInBytes);
}

CudnnCtcLossPlan::~CudnnCtcLossPlan() { destroy(); }

CudnnCtcLossPlan::CudnnCtcLossPlan(CudnnCtcLossPlan &&other) noexcept {
    *this = std::move(other);
}

CudnnCtcLossPlan &CudnnCtcLossPlan::operator=(CudnnCtcLossPlan &&other) noexcept {
    if (this == &other)
        return *this;

    destroy();

    config = other.config;
    probabilitiesDesc = other.probabilitiesDesc;
    gradientsDesc = other.gradientsDesc;
    ctcLossDesc = other.ctcLossDesc;
    workspaceSizeInBytes = other.workspaceSizeInBytes;
    currentWorkspaceSizeInBytes = other.currentWorkspaceSizeInBytes;
    currentBatchSize = other.currentBatchSize;
    workspaceSizeByBatchSize = std::move(other.workspaceSizeByBatchSize);

    other.probabilitiesDesc = nullptr;
    other.gradientsDesc = nullptr;
    other.ctcLossDesc = nullptr;
    other.workspaceSizeInBytes = 0;
    other.currentWorkspaceSizeInBytes = 0;
    other.currentBatchSize = 0;
    other.workspaceSizeByBatchSize.clear();

    return *this;
}

void CudnnCtcLossPlan::destroy() noexcept {
    if (ctcLossDesc != nullptr) {
        (void)cudnnDestroyCTCLossDescriptor(ctcLossDesc);
        ctcLossDesc = nullptr;
    }
    if (gradientsDesc != nullptr) {
        (void)cudnnDestroyTensorDescriptor(gradientsDesc);
        gradientsDesc = nullptr;
    }
    if (probabilitiesDesc != nullptr) {
        (void)cudnnDestroyTensorDescriptor(probabilitiesDesc);
        probabilitiesDesc = nullptr;
    }
    workspaceSizeInBytes = 0;
    currentWorkspaceSizeInBytes = 0;
    currentBatchSize = 0;
    workspaceSizeByBatchSize.clear();
}

void CudnnCtcLossPlan::run(void *probabilities,
                           const int *labels,
                           const int *labelLengths,
                           const int *inputLengths,
                           void *costs,
                           void *gradients,
                           void *workspace,
                           size_t workspaceSizeBytes,
                           uint32_t activeBatchSize,
                           Stream stream) {
    THOR_THROW_IF_FALSE(stream.isInitialized());
    THOR_THROW_IF_FALSE(probabilitiesDesc != nullptr);
    THOR_THROW_IF_FALSE(gradientsDesc != nullptr);
    THOR_THROW_IF_FALSE(ctcLossDesc != nullptr);
    THOR_THROW_IF_FALSE(probabilities != nullptr);
    THOR_THROW_IF_FALSE(labels != nullptr);
    THOR_THROW_IF_FALSE(labelLengths != nullptr);
    THOR_THROW_IF_FALSE(inputLengths != nullptr);
    THOR_THROW_IF_FALSE(costs != nullptr);
    THOR_THROW_IF_FALSE(gradients != nullptr);
    THOR_THROW_IF_FALSE(activeBatchSize >= 1 && activeBatchSize <= config.batchSize);
    THOR_THROW_IF_FALSE(workspaceSizeBytes >= workspaceSizeInBytes);
    THOR_THROW_IF_FALSE(workspaceSizeInBytes == 0 || workspace != nullptr);

    ScopedGpu scopedGpu(stream.getGpuNum());

    if (currentBatchSize != activeBatchSize) {
        setBatchMajorPhysicalCtcTensorDescriptor(probabilitiesDesc, config, activeBatchSize);
        setBatchMajorPhysicalCtcTensorDescriptor(gradientsDesc, config, activeBatchSize);

        const auto cachedWorkspace = workspaceSizeByBatchSize.find(activeBatchSize);
        if (cachedWorkspace != workspaceSizeByBatchSize.end()) {
            currentWorkspaceSizeInBytes = cachedWorkspace->second;
        } else {
            size_t requiredWorkspaceSizeInBytes = 0;
            const cudnnStatus_t workspaceStatus =
                cudnnGetCTCLossWorkspaceSize_v8(stream.getCudnnHandle(),
                                                toCudnnAlgo(config.algorithm),
                                                ctcLossDesc,
                                                probabilitiesDesc,
                                                gradientsDesc,
                                                &requiredWorkspaceSizeInBytes);
            THOR_THROW_IF_FALSE(workspaceStatus == CUDNN_STATUS_SUCCESS);
            // The full-capacity plan owns the reusable workspace. A smaller active
            // batch is expected to fit inside it; fail before launch if a backend
            // ever violates that assumption rather than risking an overwrite.
            THOR_THROW_IF_FALSE(requiredWorkspaceSizeInBytes <= workspaceSizeInBytes);
            workspaceSizeByBatchSize.emplace(activeBatchSize, requiredWorkspaceSizeInBytes);
            currentWorkspaceSizeInBytes = requiredWorkspaceSizeInBytes;
        }
        currentBatchSize = activeBatchSize;
    }
    THOR_THROW_IF_FALSE(workspaceSizeBytes >= currentWorkspaceSizeInBytes);
    THOR_THROW_IF_FALSE(currentWorkspaceSizeInBytes == 0 || workspace != nullptr);

    cudnnStatus_t cudnnStatus = cudnnCTCLoss_v8(stream.getCudnnHandle(),
                                                toCudnnAlgo(config.algorithm),
                                                ctcLossDesc,
                                                probabilitiesDesc,
                                                probabilities,
                                                labels,
                                                labelLengths,
                                                inputLengths,
                                                costs,
                                                gradientsDesc,
                                                gradients,
                                                currentWorkspaceSizeInBytes,
                                                workspace);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

}  // namespace ThorImplementation
