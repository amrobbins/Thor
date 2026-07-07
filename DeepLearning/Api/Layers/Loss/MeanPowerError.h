#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanPowerError.h"

#include <cmath>
#include <optional>

namespace Thor {

// MeanPowerError computes mean(abs(prediction - label) ^ exponent).
// Useful ordinary-regression exponents are typically in [1, 2]:
//   exponent = 1 is MAE.
//   exponent = 2 is MSE.
// Exponents greater than 2 are allowed but increasingly outlier-sensitive.
class MeanPowerError : public Loss {
   public:
    class Builder;
    MeanPowerError() {}

    ~MeanPowerError() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<MeanPowerError>(*this); }

    std::string getLayerType() const override { return "MeanPowerError"; }

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
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::MeanPowerError> meanPowerError =
            std::make_shared<ThorImplementation::MeanPowerError>(lossDataType, exponent);

        return meanPowerError;
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

    float exponent = 1.5f;
};

class MeanPowerError::Builder {
   public:
    virtual ~Builder() = default;

    virtual MeanPowerError build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        const float exponent = _exponent.value_or(1.5f);
        THOR_THROW_IF_FALSE(std::isfinite(exponent) && exponent >= 1.0f);

        MeanPowerError meanPowerError;
        meanPowerError.predictionsTensor = _predictions.value();
        meanPowerError.labelsTensor = _labels.value();
        meanPowerError.exampleWeightsTensor = _exampleWeights;
        meanPowerError.lossDataType = _lossDataType.value();

        meanPowerError.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        meanPowerError.lossShape = _lossShape.value();
        meanPowerError.exponent = exponent;
        meanPowerError.network = _network.value();
        meanPowerError.initialized = true;

        meanPowerError.buildSupportLayersAndAddToNetwork();

        return meanPowerError;
    }

    virtual MeanPowerError::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual MeanPowerError::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual MeanPowerError::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual MeanPowerError::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual MeanPowerError::Builder &exponent(float _exponent) {
        THOR_THROW_IF_FALSE(!this->_exponent.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_exponent) && _exponent >= 1.0f);
        this->_exponent = _exponent;
        return *this;
    }

    virtual MeanPowerError::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual MeanPowerError::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual MeanPowerError::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual MeanPowerError::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual MeanPowerError::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual MeanPowerError::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<float> _exponent;
};

}  // namespace Thor
