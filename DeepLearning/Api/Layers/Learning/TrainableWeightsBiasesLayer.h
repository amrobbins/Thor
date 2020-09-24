#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"

namespace Thor {

class TrainableWeightsBiasesLayer : public MultiConnectionLayer {
   public:
    virtual ~TrainableWeightsBiasesLayer() {}

    Tensor getWeights() const { return weights; }
    Optional<Tensor> getBiases() const { return biases; }
    Optional<Tensor> getWeightsGradient() const { return weightsGradient; }
    Optional<Tensor> getBiasesGradient() const { return biasesGradient; }

   protected:
    Tensor weights;
    Optional<Tensor> biases;
    Optional<Tensor> weightsGradient;
    Optional<Tensor> biasesGradient;
};

}  // namespace Thor
