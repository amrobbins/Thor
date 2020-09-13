#pragma once

#include "DeepLearning/Api/Layers/Loss/LossBase.h"

namespace Thor {

class CategoricalCrossEntropy : public LossBase {
   public:
    virtual ~CategoricalCrossEntropy();
};

}  // namespace Thor
