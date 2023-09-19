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
    if (biases.isPresent()) {
        biasesGradient = biases.get().clone();
    }

    if (momentum > 0.0f) {
        weightsUpdate = weightsGradient.clone();
        weightsUpdate.memset(0);
        if (biases.isPresent()) {
            assert(biasesGradient.isPresent());
            biasesUpdate = biasesGradient.get().clone();
            biasesUpdate.get().memset(0);

            if (useNesterovMomentum) {
                projectedWeights = weights.clone();
                if (biases.isPresent())
                    projectedBiases = biases.get().clone();
                this->trainableLayer->assignWeightsParameterizationTensor(projectedWeights, projectedBiases);
            }
        } else {
        }
    }
}

// This function just accumulates the gradient
void Sgd::computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) {
    if (errorIn.isEmpty())
        return;
    assert(featureIn.isPresent());
    assert(epoch > 0);  // Negative epoch indicates that optimizer has not been initialized

    trainableLayer->computeWeightsGradient(weightsGradient, biasesGradient, featureIn, errorIn, gradientUpdateStream, accumulateValues);
}

// Now having the full gradient the weight update computation is completed in this function
void Sgd::updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) {
    assert(weights.getDataType() == TensorDescriptor::DataType::FP16 || weights.getDataType() == TensorDescriptor::DataType::FP32);

    if (momentum > 0.0f) {
        if (useNesterovMomentum) {
            // WeightUpdate_t = momentum * weightUpdate_t-1 - (lr * gradient(weights_t + momentum * weightUpdate_t-1)) / batch_size

            // Note that the gradients below are computed wrt the projected weights, computed further below
            float alpha = momentum;
            float beta = (-1.0f * currentLearningRate) / batchSize;
            weightsUpdate.add(weightsUpdate, weightsGradient, alpha, beta, gradientUpdateStream);
            if (biases.isPresent()) {
                biasesUpdate.get().add(biasesUpdate, biasesGradient, alpha, beta, gradientUpdateStream);
            }

            // Next line increments t from these equations:
            Optimizer::updateWeightsWithScale(weights, biases, 1.0f);

            // projectedWeights_t+1 = weights_t+1 + momentum * weightUpdate_t
            // Note updateWeightsWithScale automatially applies the loss scaling factor, since I am projecting the update to the weights
            // I need to apply to it the projection here.
            projectedWeights.get().add(weights, weightsUpdate, 1.0f, momentum / Loss::getLossScalingFactor(), gradientUpdateStream);
            if (projectedBiases.isPresent()) {
                assert(biases.isPresent());
                projectedBiases.get().add(biases, biasesUpdate, 1.0f, momentum, gradientUpdateStream);
            }

            Optimizer::updateWeightsWithScale(weights, biases, 1.0f);
        } else {
            // WeightUpdate_t = WeightUpdate_t-1 * momentum - (lr * gradient_t) / batchSize
            float alpha = momentum;
            float beta = (-1.0f * currentLearningRate) / batchSize;

            weightsUpdate.add(weightsUpdate, weightsGradient, alpha, beta, gradientUpdateStream);
            if (biases.isPresent())
                biasesUpdate.get().add(biasesUpdate, biasesGradient, alpha, beta, gradientUpdateStream);

            Optimizer::updateWeightsWithScale(weights, biases, 1.0f);
        }
    } else {
        // subtract the gradient, scaled by the learning rate, from the weights
        // Note: learning rate is divided by batchSize because learning rate is the total update to the weights when
        // processing a batch, and each item in the batch provides a share of that update.
        float weightUpdateScalingFactor = (-1.0f * currentLearningRate) / batchSize;
        weightsUpdate = weightsGradient;
        biasesUpdate = biasesGradient;
        Optimizer::updateWeightsWithScale(weights, biases, weightUpdateScalingFactor);
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
