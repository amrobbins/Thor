#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"

#include <stdexcept>

using namespace std;

namespace ThorImplementation {

namespace {
using DataType = TensorDescriptor::DataType;

class BNParameter final : public Parameter {
   public:
    BNParameter(string name, Optional<Tensor> sharedStorage)
        : Parameter(std::move(name), /*trainable=*/true, /*trainingEnabled=*/true), sharedStorage(sharedStorage) {}

    void createStorage(const Tensor& inputTensor) override {
        if (sharedStorage.isPresent()) {
            storage = sharedStorage.get();
            return;
        }

        assert(inputTensor.getDimensions().size() == 2 || inputTensor.getDimensions().size() == 4);
        const uint64_t channels = inputTensor.getDimensions()[1];
        storage = Tensor(inputTensor.getPlacement(), TensorDescriptor(DataType::FP32, {channels}));
    }

   private:
    Optional<Tensor> sharedStorage;
};

}  // namespace

const float BatchNormalization::ALPHA_NO_SCALE = 1.0f;
const float BatchNormalization::BETA_CLEAR = 0.0f;
const float BatchNormalization::BETA_ACCUMULATE = 1.0f;

BatchNormalization::BatchNormalization(bool training,
                                       int64_t stampedId,
                                       Optional<double> exponentialRunningAverageFactor,
                                       Optional<double> epsilon)
    : TrainableLayer(TensorPlacement(TensorPlacement::MemDevices::CPU), /*inferenceOnly=*/false, stampedId),
      training(training),
      exponentialRunningAverageFactor(exponentialRunningAverageFactor.isPresent() ? exponentialRunningAverageFactor.get() : 0.05),
      epsilon(epsilon.isPresent() ? epsilon.get() : 0.0001) {
    addParameter(make_shared<BNParameter>("weights", Optional<Tensor>::empty()));
    addParameter(make_shared<BNParameter>("biases", Optional<Tensor>::empty()));
}

BatchNormalization::~BatchNormalization() { cleanup(); }

void BatchNormalization::setTrainingMode(bool training) {
    assert(running == false);
    assert(isInferenceOnly() == false);
    this->training = training;
}

Optional<Stream> BatchNormalization::resolveStateStream(Optional<Stream> stream) const {
    if (stream.isPresent())
        return stream;
    if (gradientUpdateStream.isPresent())
        return gradientUpdateStream;
    if (!streams.empty())
        return streams[0];
    return Optional<Stream>::empty();
}

void BatchNormalization::setCurrentExponentialRunningAverageFactor(double value) {
    currentExponentialRunningAverageFactor = vector<double>(featureInputs.size(), value);
}

void BatchNormalization::setInitializer(Tensor target, shared_ptr<Initializer> initializer) {
    if (target == getWeights()) {
        getParameter("weights")->setInitializer(initializer);
    } else if (target == getBiases().get()) {
        getParameter("biases")->setInitializer(initializer);
    } else if (target == resultRunningMean) {
        resultRunningMeanInitializer = initializer;
    } else if (target == resultRunningVariance) {
        resultRunningVarianceInitializer = initializer;
    } else {
        assert(false);
    }
}

bool BatchNormalization::hasInitializer(Tensor target) {
    if (target == getWeights()) {
        return getParameter("weights")->hasInitializer();
    } else if (target == getBiases().get()) {
        return getParameter("biases")->hasInitializer();
    } else if (target == resultRunningMean) {
        return resultRunningMeanInitializer != nullptr;
    } else if (target == resultRunningVariance) {
        return resultRunningVarianceInitializer != nullptr;
    } else {
        assert(false);
        return false;
    }
}

Event BatchNormalization::initializeTensor(Tensor target) {
    if (target == getWeights()) {
        return getParameter("weights")->initialize();
    } else if (target == getBiases().get()) {
        return getParameter("biases")->initialize();
    } else if (target == resultRunningMean) {
        assert(resultRunningMeanInitializer != nullptr);
        return resultRunningMeanInitializer->initialize();
    } else if (target == resultRunningVariance) {
        assert(resultRunningVarianceInitializer != nullptr);
        return resultRunningVarianceInitializer->initialize();
    } else {
        assert(false);
        return Event();
    }
}

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

void BatchNormalization::compileRunningStatInitializer(const shared_ptr<Initializer>& initializer, const Tensor& tensor) {
    if (initializer == nullptr)
        return;

    Stream initStream = gradientUpdateStream.isPresent() ? gradientUpdateStream.get()
                                                         : Stream::getMostRecentGradientUpdateStream(tensor.getPlacement().getDeviceNum());
    initializer->compile(tensor, initStream, getFanIn(), getFanOut());
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

    attachGradientUpdateStream();
    wGradFusedWithEOutGrad = true;

    for (const auto& parameter : parameters) {
        if (!parameter->isStorageInitialized()) {
            parameter->compileStorageAndOptimizer(input, gradientUpdateStream, isInferenceOnly());
        }
        parameter->compileInitializer(getFanIn(), getFanOut());
    }

    const Tensor weights = getWeights();
    const Tensor biases = getBiases().get();
    assert(weights.getDataType() == DataType::FP32);
    assert(biases.getDataType() == DataType::FP32);
    assert(weights.getDimensions() == biases.getDimensions());

    const uint64_t channels = input.getDimensions()[1];
    if (!resultRunningMean.isInitialized()) {
        resultRunningMean = Tensor(placement, TensorDescriptor(DataType::FP32, {channels}));
    }
    if (!resultRunningVariance.isInitialized()) {
        resultRunningVariance = Tensor(placement, TensorDescriptor(DataType::FP32, {channels}));
    }
    assert(resultRunningMean.getDimensions() == vector<uint64_t>({channels}));
    assert(resultRunningVariance.getDimensions() == vector<uint64_t>({channels}));
    assert(resultRunningMean.getDataType() == DataType::FP32);
    assert(resultRunningVariance.getDataType() == DataType::FP32);

    compileRunningStatInitializer(resultRunningMeanInitializer, resultRunningMean);
    compileRunningStatInitializer(resultRunningVarianceInitializer, resultRunningVariance);

    cudnnStatus_t cudnnStatus;

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
    cudnnStatus = cudnnCreateTensorDescriptor(&featureInputDescriptor.get());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    cudnnStatus =
        cudnnSetTensor4dDescriptor(featureInputDescriptor, CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    featureOutputDescriptor = cudnnTensorDescriptor_t();
    cudnnStatus = cudnnCreateTensorDescriptor(&featureOutputDescriptor.get());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    cudnnStatus =
        cudnnSetTensor4dDescriptor(featureOutputDescriptor, CUDNN_TENSOR_NCHW, cudnnDataType, batchSize, numChannels, height, width);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    derivedBnDescriptor = cudnnTensorDescriptor_t();
    cudnnStatus = cudnnCreateTensorDescriptor(&derivedBnDescriptor.get());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    cudnnStatus = cudnnDeriveBNTensorDescriptor(derivedBnDescriptor,
                                                featureInputDescriptor,
                                                inputDimensions.size() == 2 ? CUDNN_BATCHNORM_PER_ACTIVATION : CUDNN_BATCHNORM_SPATIAL);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    resultSaveMean.clear();
    resultSaveInvVariance.clear();
    scratchErrorOutputs.clear();
    resultSaveMean.reserve(featureInputs.size());
    resultSaveInvVariance.reserve(featureInputs.size());
    scratchErrorOutputs.reserve(featureInputs.size());
    for (unsigned int i = 0; i < featureInputs.size(); ++i) {
        resultSaveMean.push_back(weights.clone());
        resultSaveInvVariance.push_back(weights.clone());
        if (errorInputs.size() > i && errorInputs[i].isPresent() && (errorOutputs.size() <= i || errorOutputs[i].isEmpty())) {
            assert(featureInputs[i].isPresent());
            scratchErrorOutputs.push_back(featureInputs[i].get().clone());
        } else {
            scratchErrorOutputs.push_back(Optional<Tensor>::empty());
        }
    }

    assert(exponentialRunningAverageFactor > 0.0);
    assert(exponentialRunningAverageFactor <= 1.0);
    itemsObserved = 0;
    currentExponentialRunningAverageFactor.assign(featureInputs.size(), 0.0);
}

void BatchNormalization::cleanup() {
    cudnnStatus_t cudnnStatus;

    if (derivedBnDescriptor.isPresent()) {
        cudnnStatus = cudnnDestroyTensorDescriptor(derivedBnDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        derivedBnDescriptor.clear();
    }

    if (featureInputDescriptor.isPresent()) {
        cudnnStatus = cudnnDestroyTensorDescriptor(featureInputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        featureInputDescriptor.clear();
    }

    if (featureOutputDescriptor.isPresent()) {
        cudnnStatus = cudnnDestroyTensorDescriptor(featureOutputDescriptor.get());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        featureOutputDescriptor.clear();
    }

    Layer::cleanup();
}

void BatchNormalization::runForward(Optional<Tensor> inputTensor,
                                    Optional<Tensor> outputTensor,
                                    Stream stream,
                                    unsigned int connectionNumber,
                                    Tensor weights,
                                    Optional<Tensor> biases) {
    assert(inputTensor.isPresent());
    assert(outputTensor.isPresent());
    assert(biases.isPresent());

    cudnnStatus_t cudnnStatus;

    if (currentExponentialRunningAverageFactor[connectionNumber] != exponentialRunningAverageFactor) {
        ++itemsObserved;
        currentExponentialRunningAverageFactor[connectionNumber] = 1.0 / itemsObserved;
        if (currentExponentialRunningAverageFactor[connectionNumber] < exponentialRunningAverageFactor)
            currentExponentialRunningAverageFactor[connectionNumber] = exponentialRunningAverageFactor;
    }

    if (training) {
        cudnnStatus =
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
                                                   biases.get().getMemPtr(),
                                                   currentExponentialRunningAverageFactor[connectionNumber],
                                                   resultRunningMean.getMemPtr(),
                                                   resultRunningVariance.getMemPtr(),
                                                   epsilon,
                                                   resultSaveMean[connectionNumber].getMemPtr(),
                                                   resultSaveInvVariance[connectionNumber].getMemPtr());
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    } else {
        cudnnStatus =
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
                                                    biases.get().getMemPtr(),
                                                    resultRunningMean.getMemPtr(),
                                                    resultRunningVariance.getMemPtr(),
                                                    epsilon);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }
}

void BatchNormalization::computeFeatureOut(uint32_t connectionNumber) {
    runForward(featureInputs[connectionNumber],
               featureOutputs[connectionNumber],
               streams[connectionNumber],
               connectionNumber,
               getWeights(),
               getBiases());
}

// FIXME: Add accumulateWeightsIfFused: bool
Optional<Event> BatchNormalization::computeErrorOut(uint32_t connectionNumber) {
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
    } else if (scratchErrorOutputs.size() > connectionNumber && scratchErrorOutputs[connectionNumber].isPresent()) {
        errorOut = scratchErrorOutputs[connectionNumber];
    }
    assert(errorOut.isPresent());

    const bool accumulateGradient = (numBackwardConnectionsMade != 0);

    cudnnStatus_t cudnnStatus =
        cudnnBatchNormalizationBackward(streams[connectionNumber].getCudnnHandle(),
                                        height > 1 || width > 1 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                        &ALPHA_NO_SCALE,
                                        &BETA_CLEAR,
                                        &ALPHA_NO_SCALE,
                                        accumulateGradient ? &BETA_ACCUMULATE : &BETA_CLEAR,
                                        featureInputDescriptor,
                                        featureInputs[connectionNumber].get().getMemPtr(),
                                        featureOutputDescriptor,
                                        errorInputs[connectionNumber].get().getMemPtr(),
                                        featureInputDescriptor,
                                        errorOut.get().getMemPtr(),
                                        derivedBnDescriptor,
                                        getWeights().getMemPtr(),
                                        weightsOptimizer->getWeightsGradient().get().getMemPtr(),
                                        biasesOptimizer->getWeightsGradient().get().getMemPtr(),
                                        epsilon,
                                        resultSaveMean[connectionNumber].getMemPtr(),
                                        resultSaveInvVariance[connectionNumber].getMemPtr());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    return streams[connectionNumber].putEvent();
}

void BatchNormalization::accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
    (void)connectionNumber;
    (void)clearGradientFirst;
    // No-op: cudnnBatchNormalizationBackward() already produced dscale/dbias on the data stream.
}

}  // namespace ThorImplementation
