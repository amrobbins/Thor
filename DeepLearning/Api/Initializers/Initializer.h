#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Initializer {
   public:
    Initializer() {}

    virtual ~Initializer() {}
};

}  // namespace Thor
