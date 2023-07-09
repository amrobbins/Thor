#pragma once

#include "DeepLearning/Api/Layers/Metrics/Metric.h"

namespace Thor {

class BinaryAccuracy : public Metric {
   public:
    class Builder;
    BinaryAccuracy() {}

    virtual ~BinaryAccuracy() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<BinaryAccuracy>(*this); }

    virtual std::string getLayerType() const { return "BinaryAccuracy"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput() || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::BinaryAccuracy> BinaryAccuracy = std::make_shared<ThorImplementation::BinaryAccuracy>();
        return BinaryAccuracy;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        uint64_t workspaceSize = 2 * batchSize;
        uint64_t metricOutputSize = 4;

        return workspaceSize + metricOutputSize;
    }

    uint32_t numClasses;
};

class BinaryAccuracy::Builder {
   public:
    virtual BinaryAccuracy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
        std::vector<uint64_t> predictionDimensions = _predictions.get().getDimensions();
        assert(labelDimensions.size() == 1 && labelDimensions[0] == 1);
        assert(predictionDimensions.size() == 1 && predictionDimensions[0] == 1);

        BinaryAccuracy binaryAccuracy;
        binaryAccuracy.featureInput = _predictions;
        binaryAccuracy.labelsTensor = _labels;
        binaryAccuracy.metricTensor = Tensor(Tensor::DataType::FP32, {1});
        binaryAccuracy.initialized = true;
        binaryAccuracy.addToNetwork(_network.get());
        return binaryAccuracy;
    }

    virtual BinaryAccuracy::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryAccuracy::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryAccuracy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<uint32_t> _numClasses;
};

}  // namespace Thor