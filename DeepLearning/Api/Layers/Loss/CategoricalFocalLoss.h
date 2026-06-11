#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class CategoricalFocalLoss : public Loss {
   public:
    class Builder;
    CategoricalFocalLoss() {}

    ~CategoricalFocalLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CategoricalFocalLoss>(*this); }

    std::string getLayerType() const override { return "CategoricalFocalLoss"; }

    float getGamma() const { return gamma; }
    float getAlpha() const { return alpha; }

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
        throw std::runtime_error("CategoricalFocalLoss is a compound API loss and should not be stamped directly.");
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

    float gamma = 2.0f;
    float alpha = 1.0f;
};

class CategoricalFocalLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual CategoricalFocalLoss build() {
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

        float gamma = _gamma.value_or(2.0f);
        float alpha = _alpha.value_or(1.0f);
        THOR_THROW_IF_FALSE(gamma >= 0.0f);
        THOR_THROW_IF_FALSE(alpha >= 0.0f);

        CategoricalFocalLoss categoricalFocalLoss;
        categoricalFocalLoss.predictionsTensor = _predictions.value();
        categoricalFocalLoss.labelsTensor = _labels.value();
        categoricalFocalLoss.lossDataType = _lossDataType.value();

        categoricalFocalLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        categoricalFocalLoss.lossShape = _lossShape.value();
        categoricalFocalLoss.gamma = gamma;
        categoricalFocalLoss.alpha = alpha;
        categoricalFocalLoss.network = _network.value();
        categoricalFocalLoss.initialized = true;

        categoricalFocalLoss.buildSupportLayersAndAddToNetwork();

        return categoricalFocalLoss;
    }

    virtual CategoricalFocalLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &focusingParameter(float _gamma) {
        THOR_THROW_IF_FALSE(!this->_gamma.has_value());
        THOR_THROW_IF_FALSE(_gamma >= 0.0f);
        this->_gamma = _gamma;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        THOR_THROW_IF_FALSE(_alpha >= 0.0f);
        this->_alpha = _alpha;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual CategoricalFocalLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual CategoricalFocalLoss::Builder &lossDataType(DataType _lossDataType) {
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
    std::optional<float> _lossWeight;
    std::optional<float> _gamma;
    std::optional<float> _alpha;
};

}  // namespace Thor
