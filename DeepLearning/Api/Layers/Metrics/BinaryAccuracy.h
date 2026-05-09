#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/BinaryAccuracy.h"

namespace Thor {

class BinaryAccuracy : public Metric {
   public:
    class Builder;
    BinaryAccuracy() {}

    ~BinaryAccuracy() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<BinaryAccuracy>(*this); }

    std::string getLayerType() const override { return "BinaryAccuracy"; }

    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput() || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::BinaryAccuracy> BinaryAccuracy = std::make_shared<ThorImplementation::BinaryAccuracy>();
        return BinaryAccuracy;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t workspaceSize = 2 * batchSize;
        uint64_t metricOutputSize = 4;

        return workspaceSize + metricOutputSize;
    }

    uint32_t numClasses;
};

class BinaryAccuracy::Builder {
   public:
    virtual BinaryAccuracy build() {
        THOR_THROW_IF_FALSE(_network.isPresent());
        THOR_THROW_IF_FALSE(_predictions.isPresent());
        THOR_THROW_IF_FALSE(_labels.isPresent());
        THOR_THROW_IF_FALSE(_predictions.get() != _labels.get());
        std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
        std::vector<uint64_t> predictionDimensions = _predictions.get().getDimensions();
        THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);
        THOR_THROW_IF_FALSE(predictionDimensions.size() == 1 && predictionDimensions[0] == 1);

        BinaryAccuracy binaryAccuracy;
        binaryAccuracy.featureInput = _predictions;
        binaryAccuracy.labelsTensor = _labels;
        binaryAccuracy.metricTensor = Tensor(Tensor::DataType::FP32, {1});
        binaryAccuracy.initialized = true;
        binaryAccuracy.addToNetwork(_network.get());
        return binaryAccuracy;
    }

    virtual BinaryAccuracy::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryAccuracy::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.isPresent());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryAccuracy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.isPresent());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
};

}  // namespace Thor
