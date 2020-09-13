#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class ActivationBase : public LayerBase {
   public:
    ActivationBase() {}
    virtual ~ActivationBase() {}
};

}  // namespace Thor
