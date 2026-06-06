#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace Thor {

class ListNetLoss : public Loss {
   public:
    class Builder;
    ListNetLoss() {}

    ~ListNetLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<ListNetLoss>(*this); }

    std::string getLayerType() const override { return "ListNetLoss"; }

    float getScoreTemperature() const { return scoreTemperature; }
    float getLabelTemperature() const { return labelTemperature; }
    std::optional<Tensor> getMask() const { return maskTensor; }

    std::vector<Tensor> getLossInputTensors() const override {
        std::vector<Tensor> tensors = {predictionsTensor, labelsTensor};
        if (maskTensor.has_value())
            tensors.push_back(maskTensor.value());
        return tensors;
    }

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
        throw std::runtime_error("ListNetLoss is a compound API loss and should not be stamped directly.");
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
        uint64_t maskBytes = maskTensor.has_value() ? batchSize * maskTensor.value().getTotalSizeInBytes() : 0;
        return standardLossBytes + maskBytes + lossShaperBytes;
    }

    float scoreTemperature = 1.0f;
    float labelTemperature = 1.0f;
    std::optional<Tensor> maskTensor = std::nullopt;
};

class ListNetLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual ListNetLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions()[0] > 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());
        if (_mask.has_value()) {
            THOR_THROW_IF_FALSE(_mask.value() != _predictions.value());
            THOR_THROW_IF_FALSE(_mask.value() != _labels.value());
            THOR_THROW_IF_FALSE(_mask.value().getDimensions() == _predictions.value().getDimensions());
        }

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float scoreTemperature = _scoreTemperature.value_or(1.0f);
        float labelTemperature = _labelTemperature.value_or(1.0f);
        THOR_THROW_IF_FALSE(scoreTemperature > 0.0f);
        THOR_THROW_IF_FALSE(labelTemperature > 0.0f);

        ListNetLoss listNetLoss;
        listNetLoss.predictionsTensor = _predictions.value();
        listNetLoss.labelsTensor = _labels.value();
        listNetLoss.lossDataType = _lossDataType.value();
        listNetLoss.lossShape = _lossShape.value();
        listNetLoss.scoreTemperature = scoreTemperature;
        listNetLoss.labelTemperature = labelTemperature;
        listNetLoss.maskTensor = _mask;
        listNetLoss.network = _network.value();
        listNetLoss.initialized = true;

        listNetLoss.buildSupportLayersAndAddToNetwork();

        return listNetLoss;
    }

    virtual ListNetLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual ListNetLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual ListNetLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual ListNetLoss::Builder &scoreTemperature(float _scoreTemperature) {
        THOR_THROW_IF_FALSE(!this->_scoreTemperature.has_value());
        THOR_THROW_IF_FALSE(_scoreTemperature > 0.0f);
        this->_scoreTemperature = _scoreTemperature;
        return *this;
    }

    virtual ListNetLoss::Builder &labelTemperature(float _labelTemperature) {
        THOR_THROW_IF_FALSE(!this->_labelTemperature.has_value());
        THOR_THROW_IF_FALSE(_labelTemperature > 0.0f);
        this->_labelTemperature = _labelTemperature;
        return *this;
    }

    virtual ListNetLoss::Builder &mask(Tensor _mask) {
        THOR_THROW_IF_FALSE(!this->_mask.has_value());
        THOR_THROW_IF_FALSE(!_mask.getDimensions().empty());
        this->_mask = _mask;
        return *this;
    }

    virtual ListNetLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual ListNetLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual ListNetLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual ListNetLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual ListNetLoss::Builder &lossDataType(DataType _lossDataType) {
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
    std::optional<float> _scoreTemperature;
    std::optional<float> _labelTemperature;
    std::optional<Tensor> _mask;
};

}  // namespace Thor
