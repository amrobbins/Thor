#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"

namespace Thor {

class CategoricalCrossEntropyLoss : public Loss {
   public:
    class Builder;
    CategoricalCrossEntropyLoss() : initialized(false) {}

    virtual ~CategoricalCrossEntropyLoss() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<CategoricalCrossEntropyLoss>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    Tensor featureInput;
    Optional<float> lossScalingFactor;
    bool initialized;
};

class CategoricalCrossEntropyLoss::Builder {
   public:
    virtual CategoricalCrossEntropyLoss build() {
        assert(_featureInput.isPresent());

        CategoricalCrossEntropyLoss categoricalCrossEntropyLoss;
        categoricalCrossEntropyLoss.lossScalingFactor = _lossScalingFactor;
        categoricalCrossEntropyLoss.featureInput = _featureInput;
        categoricalCrossEntropyLoss.featureOutput = _featureInput.get().clone();
        categoricalCrossEntropyLoss.lossTensor = Tensor(Tensor::DataType::FP32, vector<uint64_t>());
        categoricalCrossEntropyLoss.initialized = true;
        return categoricalCrossEntropyLoss;
    }

    virtual BatchNormalization::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
    }

    CategoricalCrossEntropyLoss::Builder &lossScalingFactor(float lossScalingFactor) {
        assert(!_lossScalingFactor.isPresent());
        assert(lossScalingFactor > 0.0);
        this->_lossScalingFactor = lossScalingFactor;
        return *this;
    }

    // FIXME: implement these options:
    enum class LossFormat { PER_BATCH = 5, PER_BATCH_ITEM, PER_CLASS, PER_BATCH_ITEM_PER_CLASS };

   private:
    Optional<Tensor> _featureInput;
    Optional<float> _lossScalingFactor;
};

}  // namespace Thor
