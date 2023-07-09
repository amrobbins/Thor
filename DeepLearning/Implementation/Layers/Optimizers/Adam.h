#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"

namespace ThorImplementation {

class Adam : public Optimizer {
   public:
    Adam(float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 10e-8);
    virtual void computeWeightsUpdate(Tensor featureInput, Tensor errorInput, Tensor weightsUpdate, Stream stream);

   protected:
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
};

}  // namespace ThorImplementation