#pragma once

#include "DeepLearning/Implementation/Initializers/ConstantInitializer.h"

namespace ThorImplementation {

class ZerosInitializer : public ConstantInitializer {
   public:
    ZerosInitializer() : ConstantInitializer(0.0f) {}

    std::shared_ptr<Initializer> clone() override { return std::make_shared<ZerosInitializer>(*this); }
};

}  // namespace ThorImplementation
