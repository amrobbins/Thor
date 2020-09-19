#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
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
    // FIXME: Add feature input
};

class CategoricalCrossEntropyLoss::Builder {
   public:
    virtual Loss build() {
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss();
        categoricalCrossEntropyLoss->lossScalingFactor = _lossScalingFactor;
        categoricalCrossEntropyLoss->initialized = true;
        return Loss(categoricalCrossEntropyLoss);
    }

    CategoricalCrossEntropyLoss::Builder exponentialRunningAverageFactor(float lossScalingFactor) {
        assert(!_lossScalingFactor.isPresent());
        assert(lossScalingFactor > 0.0);
        this->_lossScalingFactor = lossScalingFactor;
        return *this;
    }

    // FIXME: implement these options:
    enum class LossFormat { PER_BATCH = 5, PER_BATCH_ITEM, PER_CLASS, PER_BATCH_ITEM_PER_CLASS };

   private:
    Optional<float> _lossScalingFactor;
};

}  // namespace Thor
