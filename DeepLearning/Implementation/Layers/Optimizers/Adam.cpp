#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

using namespace ThorImplementation;

Adam::Adam(float alpha, float beta1, float beta2, float epsilon) {
    this->alpha = alpha;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
}

void Adam::computeWeightsUpdate(Tensor featureInput, Tensor errorInput, Tensor weightsUpdate, Stream stream) {}