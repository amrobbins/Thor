#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"

using namespace ThorImplementation;
using namespace std;

atomic<int64_t> Optimizer::nextId(2);

void Optimizer::updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) {
    // Note: the algorithm describes the update for a single example, since this computation is done for
    // each example in a minibatch and summed, the update must be divided by the batch size to get the average update.
    // Note also that this summation is part of the matrix multiply of weightsGradient = featureInput * errorInput in the case of FC layer.
    const float ALPHA = 1.0f / (Loss::getLossScalingFactor() * batchSize);
    const float BETA = 1.0f;

    accumulateScale(weights, weightsUpdate, &ALPHA, &BETA, gradientUpdateStream);
    if (biases.isPresent()) {
        assert(biasesUpdate.isPresent());
        accumulateScale(biases, biasesUpdate, &ALPHA, &BETA, gradientUpdateStream);
    }
}