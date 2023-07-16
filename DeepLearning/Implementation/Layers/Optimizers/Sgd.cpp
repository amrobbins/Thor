#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"

using namespace ThorImplementation;
using namespace std;

Sgd::Sgd(shared_ptr<TrainableWeightsBiasesLayer> trainableLayer,
         float initialLearningRate,
         float decay,
         float momentum,
         bool useNesterovMomentum,
         Optional<Tensor> errorInput,
         Optional<Tensor> errorOutput) {
    assert(trainableLayer != nullptr);
    TensorPlacement layerPlacement = trainableLayer->getPlacement();
    assert(layerPlacement == TensorPlacement::MemDevices::GPU);
    gpuNum = layerPlacement.getDeviceNum();
    gradientUpdateStream = Stream::getNextGradientUpdateStream(gpuNum);
    if (errorInput.isEmpty())
        return;
    assert(initialLearningRate > 0.0f);
    assert(decay >= 0.0f);
    assert(momentum >= 0.0f);
    assert(gpuNum < (uint32_t)MachineEvaluator::instance().getNumGpus());

    this->trainableLayerShared = trainableLayer;
    this->trainableLayer = trainableLayer.get();
    assert(gradientUpdateStream.isInitialized());

    this->initialLearningRate = initialLearningRate;
    this->decay = decay;
    this->momentum = momentum;
    this->useNesterovMomentum = useNesterovMomentum;

    epoch = -1;

    assert(errorInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (errorOutput.isPresent()) {
        assert(errorInput.get().getPlacement().getMemDevice() == errorOutput.get().getPlacement().getMemDevice());
        assert(errorInput.get().getPlacement().getDeviceNum() == errorOutput.get().getPlacement().getDeviceNum());
    }
    assert(errorInput.get().getDimensions().size() > 0);

    Tensor weights = trainableLayer->getWeights();
    weightsGradient = weights.clone();

    Optional<Tensor> biases = trainableLayer->getBiases();
    if (biases.isPresent())
        biasesGradient = biases.get().clone();
}

void Sgd::computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) {
    if (errorIn.isEmpty())
        return;
    assert(featureIn.isPresent());
    assert(epoch > 0);  // Negative epoch indicates that optimizer has not been initialized

    trainableLayer->computeWeightsGradient(weightsGradient, biasesGradient, featureIn, errorIn, gradientUpdateStream, accumulateValues);
}

void Sgd::updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) {
    assert(weights.getDataType() == TensorDescriptor::DataType::FP16 || weights.getDataType() == TensorDescriptor::DataType::FP32);

    if (momentum > 0.0f) {
        if (useNesterovMomentum) {
            assert(false);
        } else {
            // Need to save of previous weights update, scale this by momentum, add to scaled gradient and add that to weights:
            // weights += previousWeightsUpdate * momentum - weightsGradient * scaleFromBelow
            // There is unfortunately no cublasHgeam: grep geam /usr/local/cuda-11.4/targets/x86_64-linux/include/cublas_api.h
            // but this function does implement this: C = α*A + β*B for fp32
            // so create a custom one for fp16 and use cublas for fp32. Wrap this in a function called addMatrixMatrix
            assert(false);
        }
    } else {
        // subtract the gradient, scaled by the learning rate, from the weights
        // Note: learning rate is divided by batchSize because learning rate is the total update to the weights when
        // processing a batch, and each item in the batch provides a share of that update.
        float alpha = (-1.0f * currentLearningRate) / (Loss::getLossScalingFactor() * batchSize);
        float beta = 1.0f;

        accumulateScale(weights, weightsGradient, &alpha, &beta, gradientUpdateStream);
        if (biases.isPresent()) {
            assert(biasesGradient.isPresent());
            accumulateScale(biases, biasesGradient, &alpha, &beta, gradientUpdateStream);
        }
    }
}

unordered_map<string, float> Sgd::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    unordered_map<string, float> updatedParameters;

    currentEpoch = epoch;

    currentLearningRate = initialLearningRate * pow(1.0 - (double)decay, (double)epoch);

    updatedParameters["currentLearningRate"] = currentLearningRate;

    return updatedParameters;
}

unordered_map<string, float> Sgd::getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    float currentLearningRate = initialLearningRate * pow(1.0 - (double)decay, (double)epoch);
    unordered_map<string, float> parameters;
    parameters["currentLearningRate"] = currentLearningRate;
    parameters["initialLearningRate"] = initialLearningRate;
    parameters["decay"] = decay;
    parameters["momentum"] = momentum;
    parameters["useNesterovMomentum"] = useNesterovMomentum ? 1.0f : 0.0f;
    return parameters;
}

void Sgd::setInitialLearningRate(float initialLearningRate) { this->initialLearningRate = initialLearningRate; }

void Sgd::setDecay(float decay) { this->decay = decay; }

void Sgd::setMomentum(float momentum) {
    assert(momentum >= 0.0f);
    this->momentum = momentum;
}

void Sgd::setUseNesterovMomentum(bool useNesterovMomentum) { this->useNesterovMomentum = useNesterovMomentum; }

float Sgd::getInitialLearningRate() { return initialLearningRate; }

float Sgd::getDecay() { return decay; }

float Sgd::getMomentum() { return momentum; }

bool Sgd::getUseNesterovMomentum() { return useNesterovMomentum; }
