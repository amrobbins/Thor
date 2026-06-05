#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class MarginRankingLoss : public Loss {
   public:
    class Builder;
    MarginRankingLoss() {}

    ~MarginRankingLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<MarginRankingLoss>(*this); }

    std::string getLayerType() const override { return "MarginRankingLoss"; }

    Tensor getInput1() const { return input1Tensor; }
    Tensor getInput2() const { return input2Tensor; }
    Tensor getTarget() const { return targetTensor; }
    Tensor getPredictions() const override { return input1Tensor; }
    Tensor getLabels() const override { return targetTensor; }
    std::optional<Tensor> getFeatureInput() const override { return input1Tensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {input1Tensor, input2Tensor, targetTensor}; }

    float getMargin() const { return margin; }

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
        throw std::runtime_error("MarginRankingLoss is a compound API loss and should not be stamped directly.");
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
        bytes += batchSize * input1Tensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * input2Tensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * targetTensor.getTotalSizeInBytes();
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    Tensor input1Tensor;
    Tensor input2Tensor;
    Tensor targetTensor;
    float margin = 0.0f;
};

class MarginRankingLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual MarginRankingLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_input1.has_value());
        THOR_THROW_IF_FALSE(_input2.has_value());
        THOR_THROW_IF_FALSE(_target.has_value());
        THOR_THROW_IF_FALSE(_input1.value() != _input2.value());
        THOR_THROW_IF_FALSE(_input1.value() != _target.value());
        THOR_THROW_IF_FALSE(_input2.value() != _target.value());
        THOR_THROW_IF_FALSE(!_input1.value().getDimensions().empty());
        THOR_THROW_IF_FALSE(_input1.value().getDimensions() == _input2.value().getDimensions());
        THOR_THROW_IF_FALSE(_input1.value().getDimensions() == _target.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _input1.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float margin = _margin.value_or(0.0f);
        THOR_THROW_IF_FALSE(margin >= 0.0f);

        MarginRankingLoss marginRankingLoss;
        marginRankingLoss.input1Tensor = _input1.value();
        marginRankingLoss.input2Tensor = _input2.value();
        marginRankingLoss.targetTensor = _target.value();
        marginRankingLoss.lossDataType = _lossDataType.value();
        marginRankingLoss.lossShape = _lossShape.value();
        marginRankingLoss.margin = margin;
        marginRankingLoss.network = _network.value();
        marginRankingLoss.initialized = true;

        marginRankingLoss.buildSupportLayersAndAddToNetwork();

        return marginRankingLoss;
    }

    virtual MarginRankingLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual MarginRankingLoss::Builder &input1(Tensor _input1) {
        THOR_THROW_IF_FALSE(!this->_input1.has_value());
        THOR_THROW_IF_FALSE(!_input1.getDimensions().empty());
        this->_input1 = _input1;
        return *this;
    }

    virtual MarginRankingLoss::Builder &input2(Tensor _input2) {
        THOR_THROW_IF_FALSE(!this->_input2.has_value());
        THOR_THROW_IF_FALSE(!_input2.getDimensions().empty());
        this->_input2 = _input2;
        return *this;
    }

    virtual MarginRankingLoss::Builder &target(Tensor _target) {
        THOR_THROW_IF_FALSE(!this->_target.has_value());
        THOR_THROW_IF_FALSE(!_target.getDimensions().empty());
        this->_target = _target;
        return *this;
    }

    virtual MarginRankingLoss::Builder &margin(float _margin) {
        THOR_THROW_IF_FALSE(!this->_margin.has_value());
        THOR_THROW_IF_FALSE(_margin >= 0.0f);
        this->_margin = _margin;
        return *this;
    }

    virtual MarginRankingLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual MarginRankingLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual MarginRankingLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual MarginRankingLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual MarginRankingLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _input1;
    std::optional<Tensor> _input2;
    std::optional<Tensor> _target;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _margin;
};

}  // namespace Thor
