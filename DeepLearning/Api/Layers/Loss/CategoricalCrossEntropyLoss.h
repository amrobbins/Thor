#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CategoricalCrossEntropyLoss.h"

namespace Thor {

class CategoricalCrossEntropyLoss : public Loss {
   public:
    class Builder;
    CategoricalCrossEntropyLoss() {}

    virtual ~CategoricalCrossEntropyLoss() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<CategoricalCrossEntropyLoss>(*this); }

    virtual Optional<float> getLossScalingFactor() { return lossScalingFactor; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer = nullptr,
                                             Thor::Tensor connectingApiTensor = Thor::Tensor()) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::CategoricalCrossEntropyLoss *categoricalCrossEntropy =
            new ThorImplementation::CategoricalCrossEntropyLoss(lossScalingFactor);
        Thor::Layer::connectTwoLayers(drivingLayer, categoricalCrossEntropy, drivingApiLayer, this, connectingApiTensor);
        return categoricalCrossEntropy;
    }

   private:
    Optional<float> lossScalingFactor;
};

class CategoricalCrossEntropyLoss::Builder {
   public:
    virtual CategoricalCrossEntropyLoss build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_labels.isPresent());

        CategoricalCrossEntropyLoss categoricalCrossEntropyLoss;
        categoricalCrossEntropyLoss.lossScalingFactor = _lossScalingFactor;
        categoricalCrossEntropyLoss.featureInput = _featureInput;
        categoricalCrossEntropyLoss.labelsTensor = _labels;
        categoricalCrossEntropyLoss.predictionsTensor = _featureInput.get().clone();
        // Loss is per batch item currently so this is an empty tensor
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

    virtual CategoricalCrossEntropyLoss::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual CategoricalCrossEntropyLoss::Builder &lossScalingFactor(float lossScalingFactor) {
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
    Optional<Tensor> _labels;
    Optional<float> _lossScalingFactor;
};

}  // namespace Thor
