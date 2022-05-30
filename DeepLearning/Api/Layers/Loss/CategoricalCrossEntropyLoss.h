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

    virtual string getLayerType() const { return "CategoricalCrossEntropyLoss"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput() || connectingApiTensor == labelsTensor);

        ThorImplementation::CategoricalCrossEntropyLoss *categoricalCrossEntropy = new ThorImplementation::CategoricalCrossEntropyLoss();
        Thor::Layer::connectTwoLayers(drivingLayer, categoricalCrossEntropy, drivingApiLayer, this, connectingApiTensor);
        return categoricalCrossEntropy;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);

        uint64_t lossWorkspaceBytes = featureInput.get().getTotalNumElements() * 4;
        uint64_t inverseSumOfExponentials = 4;  // 1 per batch item, FP32

        return standardLossBytes + batchSize * (lossWorkspaceBytes + inverseSumOfExponentials);
    }
};

class CategoricalCrossEntropyLoss::Builder {
   public:
    virtual CategoricalCrossEntropyLoss build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_labels.isPresent());
        assert(_featureInput.get() != _labels.get());
        assert(_lossType.isPresent());

        CategoricalCrossEntropyLoss categoricalCrossEntropyLoss;
        categoricalCrossEntropyLoss.featureInput = _featureInput;
        categoricalCrossEntropyLoss.labelsTensor = _labels;
        categoricalCrossEntropyLoss.predictionsTensor = _featureInput.get().clone(Tensor::DataType::FP32);
        categoricalCrossEntropyLoss.lossType = _lossType;
        if (_lossType == ThorImplementation::Loss::ConnectionType::BATCH_LOSS) {
            categoricalCrossEntropyLoss.lossTensor = Tensor(Tensor::DataType::FP32, {1});
        } else if (_lossType == ThorImplementation::Loss::ConnectionType::ELEMENTWISE_LOSS) {
            uint32_t batchSize = _featureInput.get().getDimensions()[0];
            categoricalCrossEntropyLoss.lossTensor = Tensor(Tensor::DataType::FP32, {batchSize});
        }
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

    virtual CategoricalCrossEntropyLoss::Builder &lossType(ThorImplementation::Loss::ConnectionType _lossType) {
        // FIXME: temp
        assert(_lossType == ThorImplementation::Loss::ConnectionType::BATCH_LOSS ||
               _lossType == ThorImplementation::Loss::ConnectionType::ELEMENTWISE_LOSS);

        assert(!this->_lossType.isPresent());
        this->_lossType = _lossType;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<Tensor> _labels;
    Optional<ThorImplementation::Loss::ConnectionType> _lossType;
};

}  // namespace Thor
