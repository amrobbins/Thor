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

class ExpectileLoss : public Loss {
   public:
    class Builder;
    ExpectileLoss() {}

    ~ExpectileLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<ExpectileLoss>(*this); }

    std::string getLayerType() const override { return "ExpectileLoss"; }

    float getExpectile() const { return expectile; }

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
        throw std::runtime_error("ExpectileLoss is a compound API loss and should not be stamped directly.");
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

    float expectile = 0.5f;
};

class ExpectileLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual ExpectileLoss build() {
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
        ThorImplementation::RegressionLossDType::validateLossDType("ExpectileLoss", _lossDataType.value());

        float expectile = _expectile.value_or(0.5f);
        THOR_THROW_IF_FALSE(expectile > 0.0f && expectile < 1.0f);

        ExpectileLoss expectileLoss;
        expectileLoss.predictionsTensor = _predictions.value();
        expectileLoss.labelsTensor = _labels.value();
        expectileLoss.exampleWeightsTensor = _exampleWeights;
        expectileLoss.lossDataType = _lossDataType.value();

        expectileLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        expectileLoss.lossShape = _lossShape.value();
        expectileLoss.expectile = expectile;
        expectileLoss.network = _network.value();
        expectileLoss.initialized = true;

        expectileLoss.buildSupportLayersAndAddToNetwork();

        return expectileLoss;
    }

    virtual ExpectileLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual ExpectileLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual ExpectileLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual ExpectileLoss::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual ExpectileLoss::Builder &expectile(float _expectile) {
        THOR_THROW_IF_FALSE(!this->_expectile.has_value());
        THOR_THROW_IF_FALSE(_expectile > 0.0f && _expectile < 1.0f);
        this->_expectile = _expectile;
        return *this;
    }

    virtual ExpectileLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual ExpectileLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual ExpectileLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual ExpectileLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual ExpectileLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual ExpectileLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        ThorImplementation::RegressionLossDType::validateLossDType("ExpectileLoss", _lossDataType);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _exampleWeights;
    std::optional<float> _expectile;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};


}  // namespace Thor
