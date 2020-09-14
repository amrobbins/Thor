#pragma once

#include "DeepLearning/Api/Layers/Loss/LossBase.h"

namespace Thor {

class CategoricalCrossEntropyLoss : public LossBase {
   public:
    class Builder;
    CategoricalCrossEntropyLoss() : initialized(false) {}

    virtual ~CategoricalCrossEntropyLoss();

   private:
    bool initialized;
    Optional<float> lossScalingFactor;
};

class CategoricalCrossEntropyLoss::Builder {
   public:
    virtual Layer build() {
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss();
        categoricalCrossEntropyLoss->lossScalingFactor = _lossScalingFactor;
        categoricalCrossEntropyLoss->initialized = true;
        return Layer(categoricalCrossEntropyLoss);
    }

    CategoricalCrossEntropyLoss::Builder exponentialRunningAverageFactor(float lossScalingFactor) {
        assert(!_lossScalingFactor.isPresent());
        assert(lossScalingFactor > 0.0);
        this->_lossScalingFactor = lossScalingFactor;
        return *this;
    }

   private:
    Optional<float> _lossScalingFactor;
};

}  // namespace Thor
