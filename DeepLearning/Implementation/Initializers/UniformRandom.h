#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

namespace ThorImplementation {

class UniformRandom : public Initializer {
   public:
    UniformRandom(double maxValue, double minValue);

    virtual void initialize(Layer *layer, Tensor tensorToInitialize);

    virtual shared_ptr<Initializer> clone();

   protected:
    const double maxValue;
    const double minValue;

    virtual void initialize(Layer *layer, Tensor tensorToInitialize, vector<Stream> streams);
};

}  // namespace ThorImplementation
