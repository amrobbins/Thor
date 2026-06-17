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

cudnnTensorDescriptor_t createBatchMajorPhysicalCtcTensorDescriptor(const CudnnCtcLossConfig &config) {
    cudnnTensorDescriptor_t descriptor = nullptr;
    cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

    const int dimA[3] = {checkedInt(config.maxTimeSteps, "maxTimeSteps"),
                         checkedInt(config.batchSize, "batchSize"),
                         checkedInt(config.numClasses, "numClasses")};
    // Thor physical memory is contiguous [B, T, C]. cuDNN logical indices are [T, B, C], so:
    //   offset(t,b,c) = b * T * C + t * C + c
    const int strideA[3] = {checkedInt(config.numClasses, "numClasses"),
                            checkedInt(config.maxTimeSteps * config.numClasses, "maxTimeSteps * numClasses"),
                            1};

    cudnnStatus = cudnnSetTensorNdDescriptor(descriptor, CUDNN_DATA_FLOAT, 3, dimA, strideA);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        (void)cudnnDestroyTensorDescriptor(descriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
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

    other.probabilitiesDesc = nullptr;
    other.gradientsDesc = nullptr;
    other.ctcLossDesc = nullptr;
    other.workspaceSizeInBytes = 0;

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
}

void CudnnCtcLossPlan::run(void *probabilities,
                           const int *labels,
                           const int *labelLengths,
                           const int *inputLengths,
                           void *costs,
                           void *gradients,
                           void *workspace,
                           size_t workspaceSizeBytes,
                           Stream stream) const {
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
    THOR_THROW_IF_FALSE(workspaceSizeBytes >= workspaceSizeInBytes);
    THOR_THROW_IF_FALSE(workspaceSizeInBytes == 0 || workspace != nullptr);

    ScopedGpu scopedGpu(stream.getGpuNum());

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
                                                workspaceSizeInBytes,
                                                workspace);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

}  // namespace ThorImplementation
