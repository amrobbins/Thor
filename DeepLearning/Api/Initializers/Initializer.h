#pragma once

#include "DeepLearning/Api/Initializers/InitializerBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Initializer {
   public:
    Initializer() {}
    Initializer(InitializerBase *initializerBase) { initializer = shared_ptr<InitializerBase>(initializerBase); }

    virtual ~Initializer() {}

   private:
    shared_ptr<InitializerBase> initializer;
};

}  // namespace Thor
