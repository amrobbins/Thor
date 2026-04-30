#pragma once

#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

namespace ThorImplementation {

class ConstantInitializer : public UniformRandom {
   public:
    explicit ConstantInitializer(float value) : UniformRandom(value, value) {}

    std::shared_ptr<Initializer> clone() override { return std::make_shared<ConstantInitializer>(*this); }
};

}  // namespace ThorImplementation
