#pragma once
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class QuantileLoss : public Loss {
   public:
    class Builder;
    QuantileLoss() {}

    ~QuantileLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<QuantileLoss>(*this); }

    std::string getLayerType() const override { return "QuantileLoss"; }

    float getQuantile() const { return quantile; }

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
        throw std::runtime_error("QuantileLoss is a compound API loss and should not be stamped directly.");
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

    float quantile = 0.5f;
};

class QuantileLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual QuantileLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = ThorImplementation::RegressionLossDType::defaultLossDType(_predictions.value().getDataType());
        ThorImplementation::RegressionLossDType::validateLossDType("QuantileLoss", _lossDataType.value());

        float quantile = _quantile.value_or(0.5f);
        THOR_THROW_IF_FALSE(quantile > 0.0f && quantile < 1.0f);

        QuantileLoss quantileLoss;
        quantileLoss.predictionsTensor = _predictions.value();
        quantileLoss.labelsTensor = _labels.value();
        quantileLoss.exampleWeightsTensor = _exampleWeights;
        quantileLoss.lossDataType = _lossDataType.value();

        quantileLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        quantileLoss.lossShape = _lossShape.value();
        quantileLoss.quantile = quantile;
        quantileLoss.network = _network.value();
        quantileLoss.initialized = true;

        quantileLoss.buildSupportLayersAndAddToNetwork();

        return quantileLoss;
    }

    virtual QuantileLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual QuantileLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual QuantileLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual QuantileLoss::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual QuantileLoss::Builder &quantile(float _quantile) {
        THOR_THROW_IF_FALSE(!this->_quantile.has_value());
        THOR_THROW_IF_FALSE(_quantile > 0.0f && _quantile < 1.0f);
        this->_quantile = _quantile;
        return *this;
    }

    virtual QuantileLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual QuantileLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual QuantileLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual QuantileLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual QuantileLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual QuantileLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        ThorImplementation::RegressionLossDType::validateLossDType("QuantileLoss", _lossDataType);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _exampleWeights;
    std::optional<float> _quantile;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};


}  // namespace Thor
