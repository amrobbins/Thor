#pragma once

#include "DeepLearning/Api/Initializers/InitializerBase.h"

#include <assert.h>

namespace Thor {

class UniformRandomInitializer : public InitializerBase {
   public:
    virtual ~UniformRandomInitializer() {}
};

}  // namespace Thor
