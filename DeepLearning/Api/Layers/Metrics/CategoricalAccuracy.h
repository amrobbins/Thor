#pragma once

#include "DeepLearning/Api/Layers/Metrics/Metric.h"

namespace Thor {

class CategoricalAccuracy : public Metric {
   public:
    class Builder;
    CategoricalAccuracy() {}

    virtual ~CategoricalAccuracy() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<CategoricalAccuracy>(*this); }

    virtual string getLayerType() const { return "CategoricalAccuracy"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput() || connectingApiTensor == labelsTensor);

        ThorImplementation::CategoricalAccuracy *categoricalAccuracy = new ThorImplementation::CategoricalAccuracy();
        Thor::Layer::connectTwoLayers(drivingLayer, categoricalAccuracy, drivingApiLayer, this, connectingApiTensor);
        return categoricalAccuracy;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        uint64_t workspaceSize = 4 * batchSize;
        uint64_t metricOutputSize = 4;

        return workspaceSize + metricOutputSize;
    }
};

class CategoricalAccuracy::Builder {
   public:
    virtual CategoricalAccuracy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());

        CategoricalAccuracy CategoricalAccuracy;
        CategoricalAccuracy.featureInput = _predictions;
        CategoricalAccuracy.labelsTensor = _labels;
        CategoricalAccuracy.metricTensor = Tensor(Tensor::DataType::FP32, {1});
        CategoricalAccuracy.initialized = true;
        CategoricalAccuracy.addToNetwork(_network.get());
        return CategoricalAccuracy;
    }

    virtual CategoricalAccuracy::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalAccuracy::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual CategoricalAccuracy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
};

}  // namespace Thor