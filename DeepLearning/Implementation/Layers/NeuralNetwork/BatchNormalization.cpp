#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

namespace {
using DataType = TensorDescriptor::DataType;

class BNParameter final : public Parameter {
   public:
    BNParameter(const string& name, const Optional<TensorDescriptor::DataType>& storageDataType, bool trainable)
        : Parameter(name, trainable), storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        assert(inputTensor.getDimensions().size() == 2 || inputTensor.getDimensions().size() == 4);
        const uint64_t channels = inputTensor.getDimensions()[1];
        TensorDescriptor::DataType resolvedDataType;
        if (storageDataType.isPresent())
            resolvedDataType = storageDataType.get();
        else
            resolvedDataType = inputTensor.getDataType();

        storage = Tensor(inputTensor.getPlacement(), TensorDescriptor(resolvedDataType, {channels}));
    }

   private:
    const Optional<TensorDescriptor::DataType> storageDataType;
};

}  // namespace

const float BatchNormalization::ALPHA_NO_SCALE = 1.0f;
const float BatchNormalization::BETA_CLEAR = 0.0f;
const float BatchNormalization::BETA_ACCUMULATE = 1.0f;

BatchNormalization::BatchNormalization(const TensorPlacement& placement,
                                       bool inferenceOnly,
                                       uint64_t numItemsObserved,
                                       Optional<double> exponentialRunningAverageFactor,
                                       Optional<double> epsilon,
                                       Optional<TensorDescriptor::DataType> storageDataType,
                                       int64_t stampedId)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      exponentialRunningAverageFactor(exponentialRunningAverageFactor.isPresent() ? exponentialRunningAverageFactor.get() : 0.05),
      epsilon(epsilon.isPresent() ? epsilon.get() : 0.0001) {
    addParameter(make_shared<BNParameter>("weights", storageDataType, true));
    addParameter(make_shared<BNParameter>("biases", storageDataType, true));
    addParameter(make_shared<BNParameter>("running_mean", storageDataType, false));
    addParameter(make_shared<BNParameter>("running_variance", storageDataType, false));

    itemsObserved = numItemsObserved;
}

BatchNormalization::~BatchNormalization() { cleanup(); }

Optional<Tensor> BatchNormalization::createFeatureOutputTensor() {
    Optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    assert(maybeInput.isPresent());
    return maybeInput.get().clone();
}

Optional<Tensor> BatchNormalization::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    if (backPropagateError && !isInferenceOnly()) {
        assert(featureInputs.size() > connectionNumber);
        assert(featureInputs[connectionNumber].isPresent());
        return featureInputs[connectionNumber].get().clone();
    }
    return Optional<Tensor>::empty();
}

uint64_t BatchNormalization::flopCountForward() {
    Optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (maybeInput.isEmpty())
        return 0;
    return maybeInput.get().getTotalNumElements() * 8;
}

uint64_t BatchNormalization::flopCountBackward() {
    Optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    if (maybeInput.isEmpty())
        return 0;
    return maybeInput.get().getTotalNumElements() * 16;
}

void BatchNormalization::compileImpl() {
    TrainableLayer::compileImpl();

    assert(!featureInputs.empty());
    assert(!featureOutputs.empty());
    assert(featureInputs.size() == featureOutputs.size());

    Optional<Tensor> maybeInput = getFirstPresentTensor(featureInputs);
    assert(maybeInput.isPresent());
    const Tensor& input = maybeInput.get();

    placement = input.getPlacement();
    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    ensureNoDeviceCrossing(placement);

    attachGradientUpdateStream();

    for (const auto& parameter : parameters) {
        if (!parameter->isStorageInitialized()) {
            string paramName = parameter->getName();
            bool trainable = paramName == "weights" || paramName == "biases";
            bool inferenceOnly = isInferenceOnly();
            parameter->compileStorageAndOptimizer(input, gradientUpdateStream, inferenceOnly || !trainable);
        }
        parameter->compileInitializer(getFanIn(), getFanOut());
    }

    // FIXME: Check what cudnn batch norm supports, it may be only fp32
    weights = getParameter("weights")->getStorage();
    biases = getParameter("biases")->getStorage();
    resultRunningMean = getParameter("running_mean")->getStorage();
    resultRunningVariance = getParameter("running_variance")->getStorage();
    assert(weights.getDataType() == DataType::FP32);
    assert(biases.getDataType() == DataType::FP32);
    assert(resultRunningMean.getDataType() == DataType::FP32);
    assert(resultRunningVariance.getDataType() == DataType::FP32);
    assert(weights.getDimensions() == biases.getDimensions());
    assert(weights.getDimensions() == resultRunningMean.getDimensions());
    assert(weights.getDimensions() == resultRunningVariance.getDimensions());

    const vector<uint64_t> inputDimensions = input.getDescriptor().getDimensions();
    assert(inputDimensions.size() == 2 || inputDimensions.size() == 4);
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
    else if (input.getDataType() == DataType::FP32)
        cudnnDataType = CUDNN_DATA_FLOAT;
    else
        assert(false);

    cleanup();

    featureInputDescriptor = cudnnTensorDescriptor_t();
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&featureInputDescriptor.get()));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(featureInputDescriptor, CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width));

    featureOutputDescriptor = cudnnTensorDescriptor_t();
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&featureOutputDescriptor.get()));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(featureOutputDescriptor, CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width));

    derivedBnDescriptor = cudnnTensorDescriptor_t();
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&derivedBnDescriptor.get()));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(derivedBnDescriptor,
                                              featureInputDescriptor,
                                              inputDimensions.size() == 2 ? CUDNN_BATCHNORM_PER_ACTIVATION : CUDNN_BATCHNORM_SPATIAL));

    resultSaveMean.clear();
    resultSaveInvVariance.clear();
    scratchErrorOutput.clear();
    resultSaveMean.reserve(featureInputs.size());
    resultSaveInvVariance.reserve(featureInputs.size());
    for (unsigned int i = 0; i < featureInputs.size(); ++i) {
        resultSaveMean.push_back(weights.clone());
        resultSaveInvVariance.push_back(weights.clone());
        if (errorInputs.size() > i && errorInputs[i].isPresent() && (errorOutputs.size() <= i || errorOutputs[i].isEmpty())) {
            assert(featureInputs[i].isPresent());
            // We may need a single, right sized, chunk of scratch memory for back prop pruned paths.
            if (scratchErrorOutput.isEmpty())
                scratchErrorOutput = featureInputs[i].get().clone();
        }
    }

    assert(exponentialRunningAverageFactor > 0.0);
    assert(exponentialRunningAverageFactor <= 1.0);
    itemsObserved = 0;
    currentExponentialRunningAverageFactor.assign(featureInputs.size(), 0.0);
}

void BatchNormalization::cleanup() {
    if (derivedBnDescriptor.isPresent()) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(derivedBnDescriptor.get()));
        derivedBnDescriptor.clear();
    }

    if (featureInputDescriptor.isPresent()) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(featureInputDescriptor.get()));
        featureInputDescriptor.clear();
    }

    if (featureOutputDescriptor.isPresent()) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(featureOutputDescriptor.get()));
        featureOutputDescriptor.clear();
    }

    Layer::cleanup();
}

void BatchNormalization::runForward(Optional<Tensor> inputTensor,
                                    Optional<Tensor> outputTensor,
                                    Stream stream,
                                    unsigned int connectionNumber,
                                    Tensor weights,
                                    Optional<Tensor> biases) {}

void BatchNormalization::computeFeatureOut(uint32_t connectionNumber) {
    Optional<Tensor> inputTensor = featureInputs[connectionNumber];
    Optional<Tensor> outputTensor = featureOutputs[connectionNumber];
    Stream stream = streams[connectionNumber];
    assert(inputTensor.isPresent());
    assert(outputTensor.isPresent());

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
                                                   featureInputDescriptor,
                                                   inputTensor.get().getMemPtr(),
                                                   featureOutputDescriptor,
                                                   outputTensor.get().getMemPtr(),
                                                   derivedBnDescriptor,
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
                                                    featureInputDescriptor,
                                                    inputTensor.get().getMemPtr(),
                                                    featureOutputDescriptor,
                                                    outputTensor.get().getMemPtr(),
                                                    derivedBnDescriptor,
                                                    weights.getMemPtr(),
                                                    biases.getMemPtr(),
                                                    resultRunningMean.getMemPtr(),
                                                    resultRunningVariance.getMemPtr(),
                                                    epsilon));
    }
}

// Error in is up-to-date by the end of the data stream.
// Gradient update stream must wait for that, and gradient accumulation must be performed on the gradient stream, for serialization.
Optional<Event> BatchNormalization::computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber,
                                                                                 bool clearWeightsGradientFirstIfFused) {
    if (errorInputs[connectionNumber].isEmpty())
        return Optional<Event>::empty();
    if (isInferenceOnly())
        return Optional<Event>::empty();

    auto weightsParameter = getParameter("weights");
    auto biasesParameter = getParameter("biases");
    assert(weightsParameter->hasOptimizer());
    assert(biasesParameter->hasOptimizer());

    shared_ptr<Optimizer> weightsOptimizer = weightsParameter->getOptimizer();
    shared_ptr<Optimizer> biasesOptimizer = biasesParameter->getOptimizer();
    assert(weightsOptimizer != nullptr);
    assert(biasesOptimizer != nullptr);
    assert(weightsOptimizer->getWeightsGradient().isPresent());
    assert(biasesOptimizer->getWeightsGradient().isPresent());

    Optional<Tensor> errorOut = Optional<Tensor>::empty();
    if (errorOutputs.size() > connectionNumber && errorOutputs[connectionNumber].isPresent()) {
        errorOut = errorOutputs[connectionNumber];
    } else {
        errorOut = scratchErrorOutput;
    }
    assert(errorOut.isPresent());
    assert(gradientUpdateStream.isPresent());

    CUDNN_CHECK(cudnnBatchNormalizationBackward(gradientUpdateStream.get().getCudnnHandle(),
                                                height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                &ALPHA_NO_SCALE,
                                                &BETA_CLEAR,
                                                &ALPHA_NO_SCALE,
                                                clearWeightsGradientFirstIfFused ? &BETA_CLEAR : &BETA_ACCUMULATE,
                                                featureInputDescriptor,
                                                featureInputs[connectionNumber].get().getMemPtr(),
                                                featureOutputDescriptor,
                                                errorInputs[connectionNumber].get().getMemPtr(),
                                                featureInputDescriptor,
                                                errorOut.get().getMemPtr(),
                                                derivedBnDescriptor,
                                                weights.getMemPtr(),
                                                weightsOptimizer->getWeightsGradient().get().getMemPtr(),
                                                biasesOptimizer->getWeightsGradient().get().getMemPtr(),
                                                epsilon,
                                                resultSaveMean[connectionNumber].getMemPtr(),
                                                resultSaveInvVariance[connectionNumber].getMemPtr()));

    return gradientUpdateStream.get().putEvent();
}

void BatchNormalization::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: cudnnBatchNormalizationBackward() already produced dscale/dbias on the data stream.
}

}  // namespace ThorImplementation
