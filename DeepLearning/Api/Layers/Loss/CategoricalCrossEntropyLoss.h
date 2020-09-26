#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"

namespace Thor {

class CategoricalCrossEntropyLoss : public Loss {
   public:
    class Builder;
    CategoricalCrossEntropyLoss() {}

    virtual ~CategoricalCrossEntropyLoss() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<CategoricalCrossEntropyLoss>(*this); }

    Optional<float> getLossScalingFactor() { return lossScalingFactor; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    Optional<float> lossScalingFactor;
};

class CategoricalCrossEntropyLoss::Builder {
   public:
    virtual CategoricalCrossEntropyLoss build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());

        CategoricalCrossEntropyLoss categoricalCrossEntropyLoss;
        categoricalCrossEntropyLoss.lossScalingFactor = _lossScalingFactor;
        categoricalCrossEntropyLoss.featureInput = _featureInput;
        categoricalCrossEntropyLoss.featureOutput = _featureInput.get().clone();
        categoricalCrossEntropyLoss.lossTensor = Tensor(Tensor::DataType::FP32, vector<uint64_t>());
        categoricalCrossEntropyLoss.initialized = true;
        categoricalCrossEntropyLoss.addToNetwork(_network.get());
        return categoricalCrossEntropyLoss;
    }

    virtual CategoricalCrossEntropyLoss::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalCrossEntropyLoss::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
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
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _lossScalingFactor;
};

}  // namespace Thor
