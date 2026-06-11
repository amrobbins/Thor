#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class TripletLoss : public Loss {
   public:
    class Builder;
    TripletLoss() {}

    ~TripletLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<TripletLoss>(*this); }

    std::string getLayerType() const override { return "TripletLoss"; }

    Tensor getAnchor() const { return anchorTensor; }
    Tensor getPositive() const { return positiveTensor; }
    Tensor getNegative() const { return negativeTensor; }
    Tensor getPredictions() const override { return anchorTensor; }
    Tensor getLabels() const override { throw std::runtime_error("TripletLoss does not have labels."); }
    std::optional<Tensor> getFeatureInput() const override { return anchorTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {anchorTensor, positiveTensor, negativeTensor}; }

    float getMargin() const { return margin; }
    float getEps() const { return eps; }

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
        throw std::runtime_error("TripletLoss is a compound API loss and should not be stamped directly.");
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t bytes = 4;
        bytes += batchSize * anchorTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * positiveTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * negativeTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    Tensor anchorTensor;
    Tensor positiveTensor;
    Tensor negativeTensor;
    float margin = 1.0f;
    float eps = 1.0e-6f;
};

class TripletLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual TripletLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_anchor.has_value());
        THOR_THROW_IF_FALSE(_positive.has_value());
        THOR_THROW_IF_FALSE(_negative.has_value());
        THOR_THROW_IF_FALSE(_anchor.value() != _positive.value());
        THOR_THROW_IF_FALSE(_anchor.value() != _negative.value());
        THOR_THROW_IF_FALSE(_positive.value() != _negative.value());
        THOR_THROW_IF_FALSE(_anchor.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_anchor.value().getDimensions()[0] > 0);
        THOR_THROW_IF_FALSE(_anchor.value().getDimensions() == _positive.value().getDimensions());
        THOR_THROW_IF_FALSE(_anchor.value().getDimensions() == _negative.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _anchor.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float margin = _margin.value_or(1.0f);
        float eps = _eps.value_or(1.0e-6f);
        THOR_THROW_IF_FALSE(margin > 0.0f);
        THOR_THROW_IF_FALSE(eps > 0.0f);

        TripletLoss tripletLoss;
        tripletLoss.anchorTensor = _anchor.value();
        tripletLoss.positiveTensor = _positive.value();
        tripletLoss.negativeTensor = _negative.value();
        tripletLoss.lossDataType = _lossDataType.value();

        tripletLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        tripletLoss.lossShape = _lossShape.value();
        tripletLoss.margin = margin;
        tripletLoss.eps = eps;
        tripletLoss.network = _network.value();
        tripletLoss.initialized = true;

        tripletLoss.buildSupportLayersAndAddToNetwork();

        return tripletLoss;
    }

    virtual TripletLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual TripletLoss::Builder &anchor(Tensor _anchor) {
        THOR_THROW_IF_FALSE(!this->_anchor.has_value());
        THOR_THROW_IF_FALSE(!_anchor.getDimensions().empty());
        this->_anchor = _anchor;
        return *this;
    }

    virtual TripletLoss::Builder &positive(Tensor _positive) {
        THOR_THROW_IF_FALSE(!this->_positive.has_value());
        THOR_THROW_IF_FALSE(!_positive.getDimensions().empty());
        this->_positive = _positive;
        return *this;
    }

    virtual TripletLoss::Builder &negative(Tensor _negative) {
        THOR_THROW_IF_FALSE(!this->_negative.has_value());
        THOR_THROW_IF_FALSE(!_negative.getDimensions().empty());
        this->_negative = _negative;
        return *this;
    }

    virtual TripletLoss::Builder &margin(float _margin) {
        THOR_THROW_IF_FALSE(!this->_margin.has_value());
        THOR_THROW_IF_FALSE(_margin > 0.0f);
        this->_margin = _margin;
        return *this;
    }

    virtual TripletLoss::Builder &eps(float _eps) {
        THOR_THROW_IF_FALSE(!this->_eps.has_value());
        THOR_THROW_IF_FALSE(_eps > 0.0f);
        this->_eps = _eps;
        return *this;
    }

    virtual TripletLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual TripletLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual TripletLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual TripletLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual TripletLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual TripletLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _anchor;
    std::optional<Tensor> _positive;
    std::optional<Tensor> _negative;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<float> _margin;
    std::optional<float> _eps;
};

}  // namespace Thor
