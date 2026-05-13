#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include <optional>
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DeepLearning/Implementation/ThorError.h"
using namespace std;

namespace ThorImplementation {

namespace {
using DataType = TensorDescriptor::DataType;


class BNParameter final : public PhysicalParameter {
   public:
    BNParameter(const string& name, const std::optional<TensorDescriptor::DataType>& storageDataType, bool trainable)
        : PhysicalParameter(name, trainable), storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        THOR_THROW_IF_FALSE(inputTensor.getDimensions().size() == 2 || inputTensor.getDimensions().size() == 4);
        const uint64_t channels = inputTensor.getDimensions()[1];
        TensorDescriptor::DataType resolvedDataType;
        if (storageDataType.has_value())
            resolvedDataType = storageDataType.value();
        else
            resolvedDataType = inputTensor.getDataType();

        storage = Tensor(inputTensor.getPlacement(), TensorDescriptor(resolvedDataType, {channels}));
    }

   private:
    const std::optional<TensorDescriptor::DataType> storageDataType;
};

class ScopedCudnnTensorDescriptor final {
   public:
    ScopedCudnnTensorDescriptor() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptor)); }

    ~ScopedCudnnTensorDescriptor() {
        if (descriptor != nullptr) {
            (void)cudnnDestroyTensorDescriptor(descriptor);
        }
    }

    ScopedCudnnTensorDescriptor(const ScopedCudnnTensorDescriptor&) = delete;
    ScopedCudnnTensorDescriptor& operator=(const ScopedCudnnTensorDescriptor&) = delete;

    ScopedCudnnTensorDescriptor(ScopedCudnnTensorDescriptor&& other) noexcept : descriptor(std::exchange(other.descriptor, nullptr)) {}

    ScopedCudnnTensorDescriptor& operator=(ScopedCudnnTensorDescriptor&& other) noexcept {
        if (this != &other) {
            if (descriptor != nullptr) {
                (void)cudnnDestroyTensorDescriptor(descriptor);
            }
            descriptor = std::exchange(other.descriptor, nullptr);
        }
        return *this;
    }

    cudnnTensorDescriptor_t get() const { return descriptor; }

   private:
    cudnnTensorDescriptor_t descriptor = nullptr;
};

static bool isCudnnBatchNormDataType(DataType dtype) { return dtype == DataType::FP32 || dtype == DataType::FP16 || dtype == DataType::BF16; }

static cudnnDataType_t toCudnnBatchNormDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return CUDNN_DATA_FLOAT;
        case DataType::FP16:
            return CUDNN_DATA_HALF;
        case DataType::BF16:
            return CUDNN_DATA_BFLOAT16;
        default:
            throw std::runtime_error("BatchNormalization supports FP32, FP16, and BF16 inputs with the cuDNN batch-normalization API; got " +
                                     TensorDescriptor::getElementTypeName(dtype));
    }
}

static int checkedCudnnDim(uint64_t value, const char* name) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("BatchNormalization ") + name + " dimension exceeds cuDNN int descriptor range.");
    }
    return static_cast<int>(value);
}

static std::vector<int> packedNchwDimsMin4(const Tensor& tensor) {
    const auto dims = tensor.getDimensions();
    if (dims.size() == 2) {
        return {checkedCudnnDim(dims[0], "N"), checkedCudnnDim(dims[1], "C"), 1, 1};
    }
    if (dims.size() == 4) {
        return {checkedCudnnDim(dims[0], "N"),
                checkedCudnnDim(dims[1], "C"),
                checkedCudnnDim(dims[2], "H"),
                checkedCudnnDim(dims[3], "W")};
    }
    throw std::runtime_error("BatchNormalization requires rank-2 [N,C] or rank-4 [N,C,H,W] input tensors.");
}

static std::vector<int> packedStridesForDims(const std::vector<int>& dims) {
    THOR_THROW_IF_FALSE(dims.size() == 4);
    std::vector<int> strides(dims.size(), 1);
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return strides;
}

static ScopedCudnnTensorDescriptor makePackedNchwTensorDescriptor(const Tensor& tensor) {
    if (!isCudnnBatchNormDataType(tensor.getDataType())) {
        throw std::runtime_error("BatchNormalization supports FP32, FP16, and BF16 inputs with the cuDNN batch-normalization API; got " +
                                 TensorDescriptor::getElementTypeName(tensor.getDataType()));
    }

    const std::vector<int> dims = packedNchwDimsMin4(tensor);
    const std::vector<int> strides = packedStridesForDims(dims);
    ScopedCudnnTensorDescriptor descriptor;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        descriptor.get(), toCudnnBatchNormDataType(tensor.getDataType()), static_cast<int>(dims.size()), dims.data(), strides.data()));
    return descriptor;
}

static ScopedCudnnTensorDescriptor makeBatchNormStatsDescriptor(uint64_t channels) {
    const int c = checkedCudnnDim(channels, "C");
    const std::vector<int> dims = {1, c, 1, 1};
    const std::vector<int> strides = {c, 1, 1, 1};
    ScopedCudnnTensorDescriptor descriptor;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        descriptor.get(), CUDNN_DATA_FLOAT, static_cast<int>(dims.size()), dims.data(), strides.data()));
    return descriptor;
}

static void validateBatchNormIoTensors(const Tensor& input, const Tensor& output) {
    if (input.getDimensions() != output.getDimensions()) {
        throw std::runtime_error("BatchNormalization input and output dimensions must match.");
    }
    if (input.getDataType() != output.getDataType()) {
        throw std::runtime_error("BatchNormalization input and output dtypes must match.");
    }
    (void)packedNchwDimsMin4(input);
    if (!isCudnnBatchNormDataType(input.getDataType())) {
        throw std::runtime_error("BatchNormalization supports FP32, FP16, and BF16 inputs with the cuDNN batch-normalization API; got " +
                                 TensorDescriptor::getElementTypeName(input.getDataType()));
    }
}

static void validateRunningAverageFactor(double exponentialRunningAverageFactor) {
    if (exponentialRunningAverageFactor <= 0.0 || exponentialRunningAverageFactor > 1.0) {
        throw std::runtime_error("BatchNormalization exponential running average factor must be in the interval (0, 1].");
    }
}

static double computeEffectiveRunningAverageFactor(uint64_t itemsObserved, double exponentialRunningAverageFactor) {
    validateRunningAverageFactor(exponentialRunningAverageFactor);

    // cuDNN's exponentialAverageFactor is the new-batch weight in:
    //   running = running * (1 - factor) + batch_stat * factor
    // Use the exact cumulative moving-average factor while it is larger than the configured EMA floor, then keep the configured
    // factor once EMA would adapt faster than 1/N. This gives unbiased early running statistics without changing the steady-state EMA.
    if (itemsObserved == 0) {
        return 1.0;
    }
    return std::max(1.0 / static_cast<double>(itemsObserved), exponentialRunningAverageFactor);
}

}  // namespace

const float BatchNormalization::ALPHA_NO_SCALE = 1.0f;
const float BatchNormalization::BETA_CLEAR = 0.0f;
const float BatchNormalization::BETA_ACCUMULATE = 1.0f;

BatchNormalization::BatchNormalization(const TensorPlacement& placement,
                                       bool inferenceOnly,
                                       uint64_t numItemsObserved,
                                       std::optional<double> exponentialRunningAverageFactor,
                                       std::optional<double> epsilon,
                                       std::optional<TensorDescriptor::DataType> storageDataType,
                                       int64_t stampedId)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      exponentialRunningAverageFactor(exponentialRunningAverageFactor.has_value() ? exponentialRunningAverageFactor.value() : 0.05),
      epsilon(epsilon.has_value() ? epsilon.value() : 0.0001) {
    addParameter(make_shared<BNParameter>("weights", DataType::FP32, true));
    addParameter(make_shared<BNParameter>("biases", DataType::FP32, true));
    addParameter(make_shared<BNParameter>("running_mean", DataType::FP32, false));
    addParameter(make_shared<BNParameter>("running_variance", DataType::FP32, false));

    itemsObserved = numItemsObserved;
}

BatchNormalization::~BatchNormalization() { cleanup(); }

std::optional<Tensor> BatchNormalization::createFeatureOutputTensor() {
    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    return maybeInput.value().clone();
}

std::optional<Tensor> BatchNormalization::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInputs.size() > connectionNumber);
        THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
        return featureInputs[connectionNumber].value().clone();
    }
    return std::nullopt;
}

uint64_t BatchNormalization::flopCountForward() {
    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 8;
}

uint64_t BatchNormalization::flopCountBackward() {
    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (!maybeInput.has_value())
        return 0;
    return maybeInput.value().getTotalNumElements() * 16;
}

void BatchNormalization::compileImpl() {
    TrainableLayer::compileImpl();

    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(!featureOutputs.empty());
    THOR_THROW_IF_FALSE(featureInputs.size() == featureOutputs.size());

    std::optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(maybeInput.has_value());
    const Tensor& input = maybeInput.value();

    placement = input.getPlacement();
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    ensureNoDeviceCrossing(placement);

    attachGradientUpdateStream();

    for (const auto& parameter : parameters) {
        if (!parameter->isStorageInitialized()) {
            parameter->compileStorage(input);
            parameter->compileInitializer(getFanIn(), getFanOut());
        }
        if (parameter->isTrainable()) {
            parameter->compileOptimizer(gradientUpdateStream, isInferenceOnly());
        }
    }

    weights = getParameter("weights")->getStorage().value();
    biases = getParameter("biases")->getStorage().value();
    resultRunningMean = getParameter("running_mean")->getStorage().value();
    resultRunningVariance = getParameter("running_variance")->getStorage().value();
    THOR_THROW_IF_FALSE(weights.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(biases.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(resultRunningMean.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(resultRunningVariance.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(weights.getDimensions() == biases.getDimensions());
    THOR_THROW_IF_FALSE(weights.getDimensions() == resultRunningMean.getDimensions());
    THOR_THROW_IF_FALSE(weights.getDimensions() == resultRunningVariance.getDimensions());

    const vector<uint64_t> inputDimensions = input.getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(inputDimensions.size() == 2 || inputDimensions.size() == 4);
    batchSize = inputDimensions[0];
    numChannels = inputDimensions[1];
    if (inputDimensions.size() == 2) {
        height = 1;
        width = 1;
    } else {
        height = inputDimensions[2];
        width = inputDimensions[3];
    }

    if (!isCudnnBatchNormDataType(input.getDataType())) {
        throw std::runtime_error("BatchNormalization supports FP32, FP16, and BF16 inputs with the cuDNN batch-normalization API; got " +
                                 TensorDescriptor::getElementTypeName(input.getDataType()));
    }
    if (epsilon < CUDNN_BN_MIN_EPSILON) {
        throw std::runtime_error("BatchNormalization epsilon must be >= CUDNN_BN_MIN_EPSILON for cuDNN batch normalization.");
    }

    resultSaveMean.clear();
    resultSaveInvVariance.clear();
    scratchErrorOutput.reset();

    resultSaveMean.reserve(featureInputs.size());
    resultSaveInvVariance.reserve(featureInputs.size());
    for (unsigned int i = 0; i < featureInputs.size(); ++i) {
        resultSaveMean.push_back(weights.clone());
        resultSaveInvVariance.push_back(weights.clone());
        if (errorInputs.size() > i && errorInputs[i].has_value() && (errorOutputs.size() <= i || !errorOutputs[i].has_value())) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            // We may need a single, right sized, chunk of scratch memory for back prop pruned paths.
            if (!scratchErrorOutput.has_value())
                scratchErrorOutput = featureInputs[i].value().clone();
        }
    }

    validateRunningAverageFactor(exponentialRunningAverageFactor);
}

void BatchNormalization::cleanup() {
    resultSaveMean.clear();
    resultSaveInvVariance.clear();
    scratchErrorOutput.reset();

    Layer::cleanup();
}

void BatchNormalization::runForward(std::optional<Tensor> inputTensor,
                                    std::optional<Tensor> outputTensor,
                                    Stream stream,
                                    unsigned int connectionNumber,
                                    Tensor weights,
                                    std::optional<Tensor> biases) {}

void BatchNormalization::computeFeatureOut(uint32_t connectionNumber) {
    std::optional<Tensor> inputTensor = featureInputs[connectionNumber];
    std::optional<Tensor> outputTensor = featureOutputs[connectionNumber];
    Stream stream = streams[connectionNumber];
    THOR_THROW_IF_FALSE(inputTensor.has_value());
    THOR_THROW_IF_FALSE(outputTensor.has_value());
    validateBatchNormIoTensors(inputTensor.value(), outputTensor.value());

    double effectiveRunningAverageFactor = exponentialRunningAverageFactor;
    if (!isInferenceOnly()) {
        if (itemsObserved != UINT64_MAX) {
            itemsObserved += 1;
        }
        effectiveRunningAverageFactor = computeEffectiveRunningAverageFactor(itemsObserved, exponentialRunningAverageFactor);
    }

    const ScopedCudnnTensorDescriptor xDesc = makePackedNchwTensorDescriptor(inputTensor.value());
    const ScopedCudnnTensorDescriptor yDesc = makePackedNchwTensorDescriptor(outputTensor.value());
    const ScopedCudnnTensorDescriptor bnDesc = makeBatchNormStatsDescriptor(numChannels);

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (!isInferenceOnly()) {
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(stream.getCudnnHandle(),
                                                           CUDNN_BATCHNORM_SPATIAL,
                                                           &ALPHA_NO_SCALE,
                                                           &BETA_CLEAR,
                                                           xDesc.get(),
                                                           inputTensor.value().getMemPtr(),
                                                           yDesc.get(),
                                                           outputTensor.value().getMemPtr(),
                                                           bnDesc.get(),
                                                           weights.getMemPtr(),
                                                           biases.getMemPtr(),
                                                           effectiveRunningAverageFactor,
                                                           resultRunningMean.getMemPtr(),
                                                           resultRunningVariance.getMemPtr(),
                                                           epsilon,
                                                           resultSaveMean[connectionNumber].getMemPtr(),
                                                           resultSaveInvVariance[connectionNumber].getMemPtr()));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(stream.getCudnnHandle(),
                                                            CUDNN_BATCHNORM_SPATIAL,
                                                            &ALPHA_NO_SCALE,
                                                            &BETA_CLEAR,
                                                            xDesc.get(),
                                                            inputTensor.value().getMemPtr(),
                                                            yDesc.get(),
                                                            outputTensor.value().getMemPtr(),
                                                            bnDesc.get(),
                                                            weights.getMemPtr(),
                                                            biases.getMemPtr(),
                                                            resultRunningMean.getMemPtr(),
                                                            resultRunningVariance.getMemPtr(),
                                                            epsilon));
    }
}
// Error in is up-to-date by the end of the gradient stream.
// Gradient accumulation must be performed on the gradient stream, for serialization.
std::optional<Event> BatchNormalization::computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                                      bool clearWeightsGradientFirstIfFused) {
    if (!errorInputs[connectionNumber].has_value())
        return std::nullopt;
    if (isInferenceOnly())
        return std::nullopt;

    auto weightsParameter = getParameter("weights");
    auto biasesParameter = getParameter("biases");
    THOR_THROW_IF_FALSE(weightsParameter->hasOptimizer());
    THOR_THROW_IF_FALSE(biasesParameter->hasOptimizer());

    shared_ptr<Optimizer> weightsOptimizer = weightsParameter->getOptimizer();
    shared_ptr<Optimizer> biasesOptimizer = biasesParameter->getOptimizer();
    THOR_THROW_IF_FALSE(weightsOptimizer != nullptr);
    THOR_THROW_IF_FALSE(biasesOptimizer != nullptr);
    THOR_THROW_IF_FALSE(weightsOptimizer->getWeightsGradient().has_value());
    THOR_THROW_IF_FALSE(biasesOptimizer->getWeightsGradient().has_value());

    std::optional<Tensor> errorOut = std::nullopt;
    if (errorOutputs.size() > connectionNumber && errorOutputs[connectionNumber].has_value()) {
        errorOut = errorOutputs[connectionNumber];
    } else {
        errorOut = scratchErrorOutput;
    }
    THOR_THROW_IF_FALSE(errorOut.has_value());
    THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
    validateBatchNormIoTensors(featureInputs[connectionNumber].value(), errorOut.value());
    validateBatchNormIoTensors(featureInputs[connectionNumber].value(), errorInputs[connectionNumber].value());

    Tensor dscaleOutput = weightsOptimizer->getWeightsGradient().value();
    Tensor dbiasOutput = biasesOptimizer->getWeightsGradient().value();

    const ScopedCudnnTensorDescriptor xDesc = makePackedNchwTensorDescriptor(featureInputs[connectionNumber].value());
    const ScopedCudnnTensorDescriptor dyDesc = makePackedNchwTensorDescriptor(errorInputs[connectionNumber].value());
    const ScopedCudnnTensorDescriptor dxDesc = makePackedNchwTensorDescriptor(errorOut.value());
    const ScopedCudnnTensorDescriptor bnDesc = makeBatchNormStatsDescriptor(numChannels);

    const float betaParamDiff = clearWeightsGradientFirstIfFused ? BETA_CLEAR : BETA_ACCUMULATE;

    ScopedGpu scopedGpu(gradientUpdateStream.value().getGpuNum());
    CUDNN_CHECK(cudnnBatchNormalizationBackward(gradientUpdateStream.value().getCudnnHandle(),
                                                CUDNN_BATCHNORM_SPATIAL,
                                                &ALPHA_NO_SCALE,
                                                &BETA_CLEAR,
                                                &ALPHA_NO_SCALE,
                                                &betaParamDiff,
                                                xDesc.get(),
                                                featureInputs[connectionNumber].value().getMemPtr(),
                                                dyDesc.get(),
                                                errorInputs[connectionNumber].value().getMemPtr(),
                                                dxDesc.get(),
                                                errorOut.value().getMemPtr(),
                                                bnDesc.get(),
                                                weights.getMemPtr(),
                                                dscaleOutput.getMemPtr(),
                                                dbiasOutput.getMemPtr(),
                                                epsilon,
                                                resultSaveMean[connectionNumber].getMemPtr(),
                                                resultSaveInvVariance[connectionNumber].getMemPtr()));

    return gradientUpdateStream.value().putEvent();
}
void BatchNormalization::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: the cuDNN batchnorm backward call already produced dscale/dbias on the gradient stream.
}

}  // namespace ThorImplementation
