#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class HingeGANGeneratorLoss : public Loss {
   public:
    class Builder;
    HingeGANGeneratorLoss() {}

    ~HingeGANGeneratorLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<HingeGANGeneratorLoss>(*this); }

    std::string getLayerType() const override { return "HingeGANGeneratorLoss"; }

    Tensor getFakeScores() const { return fakeScoresTensor; }
    Tensor getPredictions() const override { return fakeScoresTensor; }
    Tensor getLabels() const override { return Tensor(); }
    std::optional<Tensor> getFeatureInput() const override { return fakeScoresTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {fakeScoresTensor}; }

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
        throw std::runtime_error("HingeGANGeneratorLoss is a compound API loss and should not be stamped directly.");
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
        bytes += batchSize * fakeScoresTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    Tensor fakeScoresTensor;
};

class HingeGANGeneratorLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual HingeGANGeneratorLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_fakeScores.has_value());
        THOR_THROW_IF_FALSE(_fakeScores.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_fakeScores.value().getDimensions()[0] > 0);

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _fakeScores.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        HingeGANGeneratorLoss hingeGANGeneratorLoss;
        hingeGANGeneratorLoss.fakeScoresTensor = _fakeScores.value();
        hingeGANGeneratorLoss.lossDataType = _lossDataType.value();
        hingeGANGeneratorLoss.lossShape = _lossShape.value();
        hingeGANGeneratorLoss.network = _network.value();
        hingeGANGeneratorLoss.initialized = true;

        hingeGANGeneratorLoss.buildSupportLayersAndAddToNetwork();

        return hingeGANGeneratorLoss;
    }

    virtual HingeGANGeneratorLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual HingeGANGeneratorLoss::Builder &fakeScores(Tensor _fakeScores) {
        THOR_THROW_IF_FALSE(!this->_fakeScores.has_value());
        THOR_THROW_IF_FALSE(!_fakeScores.getDimensions().empty());
        this->_fakeScores = _fakeScores;
        return *this;
    }

    virtual HingeGANGeneratorLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual HingeGANGeneratorLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual HingeGANGeneratorLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual HingeGANGeneratorLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual HingeGANGeneratorLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _fakeScores;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
};

}  // namespace Thor
