#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class ListwiseSoftmaxCrossEntropyLoss : public Loss {
   public:
    class Builder;
    ListwiseSoftmaxCrossEntropyLoss() {}

    ~ListwiseSoftmaxCrossEntropyLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<ListwiseSoftmaxCrossEntropyLoss>(*this); }

    std::string getLayerType() const override { return "ListwiseSoftmaxCrossEntropyLoss"; }

    float getTemperature() const { return temperature; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual bool isMultiLayer() const { return true; }

    virtual void buildSupportLayersAndAddToNetwork();

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)connectingApiTensor;
        (void)inferenceOnly;
        throw std::runtime_error("ListwiseSoftmaxCrossEntropyLoss is a compound API loss and should not be stamped directly.");
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        return standardLossBytes + lossShaperBytes;
    }

    float temperature = 1.0f;
};

class ListwiseSoftmaxCrossEntropyLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual ListwiseSoftmaxCrossEntropyLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions()[0] > 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float temperature = _temperature.value_or(1.0f);
        THOR_THROW_IF_FALSE(temperature > 0.0f);

        ListwiseSoftmaxCrossEntropyLoss listwiseSoftmaxCrossEntropyLoss;
        listwiseSoftmaxCrossEntropyLoss.predictionsTensor = _predictions.value();
        listwiseSoftmaxCrossEntropyLoss.labelsTensor = _labels.value();
        listwiseSoftmaxCrossEntropyLoss.lossDataType = _lossDataType.value();
        listwiseSoftmaxCrossEntropyLoss.lossShape = _lossShape.value();
        listwiseSoftmaxCrossEntropyLoss.temperature = temperature;
        listwiseSoftmaxCrossEntropyLoss.network = _network.value();
        listwiseSoftmaxCrossEntropyLoss.initialized = true;

        listwiseSoftmaxCrossEntropyLoss.buildSupportLayersAndAddToNetwork();

        return listwiseSoftmaxCrossEntropyLoss;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &temperature(float _temperature) {
        THOR_THROW_IF_FALSE(!this->_temperature.has_value());
        THOR_THROW_IF_FALSE(_temperature > 0.0f);
        this->_temperature = _temperature;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual ListwiseSoftmaxCrossEntropyLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _temperature;
};

}  // namespace Thor
