#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"

#include <assert.h>

namespace Thor {

class UniformRandomInitializer : public Initializer {
   public:
    virtual ~UniformRandomInitializer() {}
};

}  // namespace Thor
