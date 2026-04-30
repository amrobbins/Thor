#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

namespace ThorImplementation {

class UniformRandom : public Initializer {
   public:
    UniformRandom(float maxValue, float minValue);

    void initialize(Stream initStream) override;

    std::shared_ptr<Initializer> clone() override;

   protected:
    const float maxValue;
    const float minValue;
};

}  // namespace ThorImplementation
