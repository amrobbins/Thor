#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class WassersteinGANCriticLoss : public Loss {
   public:
    class Builder;
    WassersteinGANCriticLoss() {}

    ~WassersteinGANCriticLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<WassersteinGANCriticLoss>(*this); }

    std::string getLayerType() const override { return "WassersteinGANCriticLoss"; }

    Tensor getRealScores() const { return realScoresTensor; }
    Tensor getFakeScores() const { return fakeScoresTensor; }
    Tensor getPredictions() const override { return realScoresTensor; }
    Tensor getLabels() const override { return fakeScoresTensor; }
    std::optional<Tensor> getFeatureInput() const override { return realScoresTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {realScoresTensor, fakeScoresTensor}; }

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
        throw std::runtime_error("WassersteinGANCriticLoss is a compound API loss and should not be stamped directly.");
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
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    Tensor realScoresTensor;
    Tensor fakeScoresTensor;
};

class WassersteinGANCriticLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual WassersteinGANCriticLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_realScores.has_value());
        THOR_THROW_IF_FALSE(_fakeScores.has_value());
        THOR_THROW_IF_FALSE(_realScores.value() != _fakeScores.value());
        THOR_THROW_IF_FALSE(_realScores.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_realScores.value().getDimensions()[0] > 0);
        THOR_THROW_IF_FALSE(_realScores.value().getDimensions() == _fakeScores.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _realScores.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        WassersteinGANCriticLoss wassersteinGANCriticLoss;
        wassersteinGANCriticLoss.realScoresTensor = _realScores.value();
        wassersteinGANCriticLoss.fakeScoresTensor = _fakeScores.value();
        wassersteinGANCriticLoss.lossDataType = _lossDataType.value();

        wassersteinGANCriticLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        wassersteinGANCriticLoss.lossShape = _lossShape.value();
        wassersteinGANCriticLoss.network = _network.value();
        wassersteinGANCriticLoss.initialized = true;

        wassersteinGANCriticLoss.buildSupportLayersAndAddToNetwork();

        return wassersteinGANCriticLoss;
    }

    virtual WassersteinGANCriticLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &realScores(Tensor _realScores) {
        THOR_THROW_IF_FALSE(!this->_realScores.has_value());
        THOR_THROW_IF_FALSE(!_realScores.getDimensions().empty());
        this->_realScores = _realScores;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &fakeScores(Tensor _fakeScores) {
        THOR_THROW_IF_FALSE(!this->_fakeScores.has_value());
        THOR_THROW_IF_FALSE(!_fakeScores.getDimensions().empty());
        this->_fakeScores = _fakeScores;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual WassersteinGANCriticLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _realScores;
    std::optional<Tensor> _fakeScores;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};

}  // namespace Thor
