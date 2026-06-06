#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class LSGANGeneratorLoss : public Loss {
   public:
    class Builder;
    LSGANGeneratorLoss() {}

    ~LSGANGeneratorLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<LSGANGeneratorLoss>(*this); }

    std::string getLayerType() const override { return "LSGANGeneratorLoss"; }

    Tensor getFakeScores() const { return fakeScoresTensor; }
    Tensor getPredictions() const override { return fakeScoresTensor; }
    Tensor getLabels() const override { return Tensor(); }
    std::optional<Tensor> getFeatureInput() const override { return fakeScoresTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {fakeScoresTensor}; }
    float getTarget() const { return target; }

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
        throw std::runtime_error("LSGANGeneratorLoss is a compound API loss and should not be stamped directly.");
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
    float target = 1.0f;
};

class LSGANGeneratorLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual LSGANGeneratorLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_fakeScores.has_value());
        THOR_THROW_IF_FALSE(_fakeScores.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_fakeScores.value().getDimensions()[0] > 0);

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _fakeScores.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        LSGANGeneratorLoss loss;
        loss.fakeScoresTensor = _fakeScores.value();
        loss.target = _target;
        loss.lossDataType = _lossDataType.value();
        loss.lossShape = _lossShape.value();
        loss.network = _network.value();
        loss.initialized = true;

        loss.buildSupportLayersAndAddToNetwork();

        return loss;
    }

    virtual LSGANGeneratorLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &fakeScores(Tensor _fakeScores) {
        THOR_THROW_IF_FALSE(!this->_fakeScores.has_value());
        THOR_THROW_IF_FALSE(!_fakeScores.getDimensions().empty());
        this->_fakeScores = _fakeScores;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &target(float _target) {
        this->_target = _target;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual LSGANGeneratorLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _fakeScores;
    float _target = 1.0f;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
};

}  // namespace Thor
