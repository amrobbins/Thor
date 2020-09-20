#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class LossBase : public LayerBase {
   public:
    LossBase() {}
    virtual ~LossBase() {}

    Tensor getLossTensor();
};

}  // namespace Thor
