#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class WassersteinGANCriticGradientPenaltyLoss : public Loss {
   public:
    class Builder;
    WassersteinGANCriticGradientPenaltyLoss() {}

    ~WassersteinGANCriticGradientPenaltyLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<WassersteinGANCriticGradientPenaltyLoss>(*this); }

    std::string getLayerType() const override { return "WassersteinGANCriticGradientPenaltyLoss"; }

    Tensor getRealScores() const { return realScoresTensor; }
    Tensor getFakeScores() const { return fakeScoresTensor; }
    Tensor getSampleGradients() const { return sampleGradientsTensor; }
    float getGradientPenaltyWeight() const { return gradientPenaltyWeight; }
    float getTargetGradientNorm() const { return targetGradientNorm; }
    float getEps() const { return eps; }

    Tensor getPredictions() const override { return realScoresTensor; }
    Tensor getLabels() const override { return fakeScoresTensor; }
    std::optional<Tensor> getFeatureInput() const override { return realScoresTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {realScoresTensor, fakeScoresTensor, sampleGradientsTensor}; }

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
        throw std::runtime_error(
            "WassersteinGANCriticGradientPenaltyLoss is a compound API loss and should not be stamped directly.");
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
        bytes += batchSize * realScoresTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * fakeScoresTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * sampleGradientsTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    Tensor realScoresTensor;
    Tensor fakeScoresTensor;
    Tensor sampleGradientsTensor;
    float gradientPenaltyWeight = 10.0f;
    float targetGradientNorm = 1.0f;
    float eps = 1.0e-12f;
};

class WassersteinGANCriticGradientPenaltyLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual WassersteinGANCriticGradientPenaltyLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_realScores.has_value());
        THOR_THROW_IF_FALSE(_fakeScores.has_value());
        THOR_THROW_IF_FALSE(_sampleGradients.has_value());
        THOR_THROW_IF_FALSE(_realScores.value() != _fakeScores.value());
        THOR_THROW_IF_FALSE(_realScores.value() != _sampleGradients.value());
        THOR_THROW_IF_FALSE(_fakeScores.value() != _sampleGradients.value());
        THOR_THROW_IF_FALSE(_realScores.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_realScores.value().getDimensions()[0] == 1);
        THOR_THROW_IF_FALSE(_realScores.value().getDimensions() == _fakeScores.value().getDimensions());
        THOR_THROW_IF_FALSE(!_sampleGradients.value().getDimensions().empty());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _realScores.value().getDataType();
        if (!_gradientPenaltyWeight.has_value())
            _gradientPenaltyWeight = 10.0f;
        if (!_targetGradientNorm.has_value())
            _targetGradientNorm = 1.0f;
        if (!_eps.has_value())
            _eps = 1.0e-12f;

        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        THOR_THROW_IF_FALSE(_gradientPenaltyWeight.value() >= 0.0f);
        THOR_THROW_IF_FALSE(_targetGradientNorm.value() > 0.0f);
        THOR_THROW_IF_FALSE(_eps.value() > 0.0f);

        WassersteinGANCriticGradientPenaltyLoss loss;
        loss.realScoresTensor = _realScores.value();
        loss.fakeScoresTensor = _fakeScores.value();
        loss.sampleGradientsTensor = _sampleGradients.value();
        loss.gradientPenaltyWeight = _gradientPenaltyWeight.value();
        loss.targetGradientNorm = _targetGradientNorm.value();
        loss.eps = _eps.value();
        loss.lossDataType = _lossDataType.value();

        loss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        loss.lossShape = _lossShape.value();
        loss.network = _network.value();
        loss.initialized = true;

        loss.buildSupportLayersAndAddToNetwork();

        return loss;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &realScores(Tensor _realScores) {
        THOR_THROW_IF_FALSE(!this->_realScores.has_value());
        THOR_THROW_IF_FALSE(!_realScores.getDimensions().empty());
        this->_realScores = _realScores;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &fakeScores(Tensor _fakeScores) {
        THOR_THROW_IF_FALSE(!this->_fakeScores.has_value());
        THOR_THROW_IF_FALSE(!_fakeScores.getDimensions().empty());
        this->_fakeScores = _fakeScores;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &sampleGradients(Tensor _sampleGradients) {
        THOR_THROW_IF_FALSE(!this->_sampleGradients.has_value());
        THOR_THROW_IF_FALSE(!_sampleGradients.getDimensions().empty());
        this->_sampleGradients = _sampleGradients;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &gradientPenaltyWeight(float _gradientPenaltyWeight) {
        THOR_THROW_IF_FALSE(!this->_gradientPenaltyWeight.has_value());
        THOR_THROW_IF_FALSE(_gradientPenaltyWeight >= 0.0f);
        this->_gradientPenaltyWeight = _gradientPenaltyWeight;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &targetGradientNorm(float _targetGradientNorm) {
        THOR_THROW_IF_FALSE(!this->_targetGradientNorm.has_value());
        THOR_THROW_IF_FALSE(_targetGradientNorm > 0.0f);
        this->_targetGradientNorm = _targetGradientNorm;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &epsilon(float _eps) {
        THOR_THROW_IF_FALSE(!this->_eps.has_value());
        THOR_THROW_IF_FALSE(_eps > 0.0f);
        this->_eps = _eps;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &eps(float _eps) { return epsilon(_eps); }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual WassersteinGANCriticGradientPenaltyLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _realScores;
    std::optional<Tensor> _fakeScores;
    std::optional<Tensor> _sampleGradients;
    std::optional<float> _gradientPenaltyWeight;
    std::optional<float> _targetGradientNorm;
    std::optional<float> _eps;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};

}  // namespace Thor
