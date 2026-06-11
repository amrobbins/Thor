#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class BinaryFocalLoss : public Loss {
   public:
    class Builder;
    BinaryFocalLoss() {}

    ~BinaryFocalLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<BinaryFocalLoss>(*this); }

    std::string getLayerType() const override { return "BinaryFocalLoss"; }

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
        throw std::runtime_error("BinaryFocalLoss is a compound API loss and should not be stamped directly.");
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
    float alpha = 0.25f;
};

class BinaryFocalLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual BinaryFocalLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1 && _predictions.value().getDimensions()[0] == 1);
        THOR_THROW_IF_FALSE(_labels.value().getDimensions().size() == 1 && _labels.value().getDimensions()[0] == 1);

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::BATCH || _lossShape.value() == LossShape::ELEMENTWISE ||
                            _lossShape.value() == LossShape::RAW);
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float gamma = _gamma.value_or(2.0f);
        float alpha = _alpha.value_or(0.25f);
        THOR_THROW_IF_FALSE(gamma >= 0.0f);
        THOR_THROW_IF_FALSE(alpha >= 0.0f && alpha <= 1.0f);

        BinaryFocalLoss binaryFocalLoss;
        binaryFocalLoss.predictionsTensor = _predictions.value();
        binaryFocalLoss.labelsTensor = _labels.value();
        binaryFocalLoss.lossDataType = _lossDataType.value();

        binaryFocalLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        binaryFocalLoss.lossShape = _lossShape.value();
        binaryFocalLoss.gamma = gamma;
        binaryFocalLoss.alpha = alpha;
        binaryFocalLoss.network = _network.value();
        binaryFocalLoss.initialized = true;

        binaryFocalLoss.buildSupportLayersAndAddToNetwork();

        return binaryFocalLoss;
    }

    virtual BinaryFocalLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(_predictions.getDimensions().size() == 1 && _predictions.getDimensions()[0] == 1);
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(_labels.getDimensions().size() == 1 && _labels.getDimensions()[0] == 1);
        this->_labels = _labels;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &focusingParameter(float _gamma) {
        THOR_THROW_IF_FALSE(!this->_gamma.has_value());
        THOR_THROW_IF_FALSE(_gamma >= 0.0f);
        this->_gamma = _gamma;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        THOR_THROW_IF_FALSE(_alpha >= 0.0f && _alpha <= 1.0f);
        this->_alpha = _alpha;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual BinaryFocalLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual BinaryFocalLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual BinaryFocalLoss::Builder &lossDataType(DataType _lossDataType) {
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
