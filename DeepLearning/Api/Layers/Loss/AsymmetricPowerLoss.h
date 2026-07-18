#pragma once
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace Thor {

// AsymmetricPowerLoss applies different weights to positive and negative
// residuals before raising their absolute value to a configurable power:
//
//   2 * level       * abs(label - prediction)^exponent, label > prediction
//   2 * (1 - level) * abs(label - prediction)^exponent, otherwise
//
// level=0.5 exactly matches MeanPowerError(exponent), and exponent=2 exactly
// matches ExpectileLoss(level).
class AsymmetricPowerLoss : public Loss {
   public:
    class Builder;
    AsymmetricPowerLoss() {}

    ~AsymmetricPowerLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<AsymmetricPowerLoss>(*this); }

    std::string getLayerType() const override { return "AsymmetricPowerLoss"; }

    float getLevel() const { return level; }
    float getExponent() const { return exponent; }

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
        throw std::runtime_error("AsymmetricPowerLoss is a compound API loss and should not be stamped directly.");
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

    float level = 0.5f;
    float exponent = 1.5f;
};

class AsymmetricPowerLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual AsymmetricPowerLoss build() {
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
        ThorImplementation::RegressionLossDType::validateLossDType("AsymmetricPowerLoss", _lossDataType.value());

        const float level = _level.value_or(0.5f);
        const float exponent = _exponent.value_or(1.5f);
        THOR_THROW_IF_FALSE(std::isfinite(level) && level > 0.0f && level < 1.0f);
        THOR_THROW_IF_FALSE(std::isfinite(exponent) && exponent >= 1.0f);

        AsymmetricPowerLoss asymmetricPowerLoss;
        asymmetricPowerLoss.predictionsTensor = _predictions.value();
        asymmetricPowerLoss.labelsTensor = _labels.value();
        asymmetricPowerLoss.exampleWeightsTensor = _exampleWeights;
        asymmetricPowerLoss.lossDataType = _lossDataType.value();
        asymmetricPowerLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        asymmetricPowerLoss.lossShape = _lossShape.value();
        asymmetricPowerLoss.level = level;
        asymmetricPowerLoss.exponent = exponent;
        asymmetricPowerLoss.network = _network.value();
        asymmetricPowerLoss.initialized = true;

        asymmetricPowerLoss.buildSupportLayersAndAddToNetwork();
        return asymmetricPowerLoss;
    }

    virtual AsymmetricPowerLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &level(float _level) {
        THOR_THROW_IF_FALSE(!this->_level.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_level) && _level > 0.0f && _level < 1.0f);
        this->_level = _level;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &exponent(float _exponent) {
        THOR_THROW_IF_FALSE(!this->_exponent.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_exponent) && _exponent >= 1.0f);
        this->_exponent = _exponent;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual AsymmetricPowerLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        ThorImplementation::RegressionLossDType::validateLossDType("AsymmetricPowerLoss", _lossDataType);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _exampleWeights;
    std::optional<float> _level;
    std::optional<float> _exponent;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};

}  // namespace Thor
