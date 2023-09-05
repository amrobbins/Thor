#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"

using namespace ThorImplementation;
using namespace std;

atomic<int64_t> Optimizer::nextId(2);

void Optimizer::updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) {
    updateWeightsWithScale(weights, biases, 1.0f / batchSize);
}

void Optimizer::updateWeightsWithScale(Tensor weights, Optional<Tensor> biases, float weightUpdateScalingFactor) {
    // The optimizer base takes care of reverse loss scaling so that the individual optimizers don't need to worry about it.
    const float ALPHA = weightUpdateScalingFactor / Loss::getLossScalingFactor();
    const float BETA = 1.0f;

    accumulateScale(weights, weightsUpdate, &ALPHA, &BETA, gradientUpdateStream);
    if (biases.isPresent()) {
        assert(biasesUpdate.isPresent());
        accumulateScale(biases, biasesUpdate, &ALPHA, &BETA, gradientUpdateStream);
    }
}