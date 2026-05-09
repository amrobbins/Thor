#include <optional>
#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <stdexcept>

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

    cudnnDataType_t cudnnDataType;
    if (input.getDataType() == DataType::FP16)
        cudnnDataType = CUDNN_DATA_HALF;
    else if (input.getDataType() == DataType::BF16)
        cudnnDataType = CUDNN_DATA_BFLOAT16;
    else if (input.getDataType() == DataType::FP32)
        cudnnDataType = CUDNN_DATA_FLOAT;
    else
        throw runtime_error(
            "BatchNormalization only supports input tensors of FP32, FP16 and BF16."
            " Please convert the input tensor to one of those before connecting it as the input to a BatchNormalization layer.");

    cleanup();

    featureInputDescriptor = cudnnTensorDescriptor_t();
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&featureInputDescriptor.value()));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(featureInputDescriptor.value(), CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width));

    featureOutputDescriptor = cudnnTensorDescriptor_t();
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&featureOutputDescriptor.value()));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(featureOutputDescriptor.value(), CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width));

    derivedBnDescriptor = cudnnTensorDescriptor_t();
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&derivedBnDescriptor.value()));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(derivedBnDescriptor.value(),
                                              featureInputDescriptor.value(),
                                              inputDimensions.size() == 2 ? CUDNN_BATCHNORM_PER_ACTIVATION : CUDNN_BATCHNORM_SPATIAL));

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

    THOR_THROW_IF_FALSE(exponentialRunningAverageFactor > 0.0);
    THOR_THROW_IF_FALSE(exponentialRunningAverageFactor <= 1.0);
    itemsObserved = 0;
    currentExponentialRunningAverageFactor.assign(featureInputs.size(), 0.0);
}

void BatchNormalization::cleanup() {
    if (derivedBnDescriptor.has_value()) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(derivedBnDescriptor.value()));
        derivedBnDescriptor.reset();
    }

    if (featureInputDescriptor.has_value()) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(featureInputDescriptor.value()));
        featureInputDescriptor.reset();
    }

    if (featureOutputDescriptor.has_value()) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(featureOutputDescriptor.value()));
        featureOutputDescriptor.reset();
    }

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

    if (itemsObserved != UINT64_MAX)
        itemsObserved += 1;
    if (currentExponentialRunningAverageFactor[connectionNumber] != exponentialRunningAverageFactor) {
        currentExponentialRunningAverageFactor[connectionNumber] = 1.0 / itemsObserved;
        if (currentExponentialRunningAverageFactor[connectionNumber] < exponentialRunningAverageFactor)
            currentExponentialRunningAverageFactor[connectionNumber] = exponentialRunningAverageFactor;
    }

    if (!isInferenceOnly()) {
        CUDNN_CHECK(
            cudnnBatchNormalizationForwardTraining(stream.getCudnnHandle(),
                                                   height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                   &ALPHA_NO_SCALE,
                                                   &BETA_CLEAR,
                                                   featureInputDescriptor.value(),
                                                   inputTensor.value().getMemPtr(),
                                                   featureOutputDescriptor.value(),
                                                   outputTensor.value().getMemPtr(),
                                                   derivedBnDescriptor.value(),
                                                   weights.getMemPtr(),
                                                   biases.getMemPtr(),
                                                   currentExponentialRunningAverageFactor[connectionNumber],
                                                   resultRunningMean.getMemPtr(),
                                                   resultRunningVariance.getMemPtr(),
                                                   epsilon,
                                                   resultSaveMean[connectionNumber].getMemPtr(),
                                                   resultSaveInvVariance[connectionNumber].getMemPtr()));

    } else {
        CUDNN_CHECK(
            cudnnBatchNormalizationForwardInference(stream.getCudnnHandle(),
                                                    height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                    &ALPHA_NO_SCALE,
                                                    &BETA_CLEAR,
                                                    featureInputDescriptor.value(),
                                                    inputTensor.value().getMemPtr(),
                                                    featureOutputDescriptor.value(),
                                                    outputTensor.value().getMemPtr(),
                                                    derivedBnDescriptor.value(),
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

    CUDNN_CHECK(cudnnBatchNormalizationBackward(gradientUpdateStream.value().getCudnnHandle(),
                                                height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                &ALPHA_NO_SCALE,
                                                &BETA_CLEAR,
                                                &ALPHA_NO_SCALE,
                                                clearWeightsGradientFirstIfFused ? &BETA_CLEAR : &BETA_ACCUMULATE,
                                                featureInputDescriptor.value(),
                                                featureInputs[connectionNumber].value().getMemPtr(),
                                                featureOutputDescriptor.value(),
                                                errorInputs[connectionNumber].value().getMemPtr(),
                                                featureInputDescriptor.value(),
                                                errorOut.value().getMemPtr(),
                                                derivedBnDescriptor.value(),
                                                weights.getMemPtr(),
                                                weightsOptimizer->getWeightsGradient().value().getMemPtr(),
                                                biasesOptimizer->getWeightsGradient().value().getMemPtr(),
                                                epsilon,
                                                resultSaveMean[connectionNumber].getMemPtr(),
                                                resultSaveInvVariance[connectionNumber].getMemPtr()));

    return gradientUpdateStream.value().putEvent();
}

void BatchNormalization::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: cudnnBatchNormalizationBackward() already produced dscale/dbias on the data stream.
}

}  // namespace ThorImplementation
