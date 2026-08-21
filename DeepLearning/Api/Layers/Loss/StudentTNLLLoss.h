#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace Thor {

class StudentTNLLLoss : public Loss {
   public:
    class Builder;
    StudentTNLLLoss() {}

    ~StudentTNLLLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<StudentTNLLLoss>(*this); }

    std::string getLayerType() const override { return "StudentTNLLLoss"; }

    Tensor getLocation() const { return predictionsTensor; }
    Tensor getTarget() const { return labelsTensor; }
    Tensor getLogScale() const { return logScaleTensor; }
    std::optional<float> getDegreesOfFreedom() const {
        if (logDegreesOfFreedomTensor.has_value())
            return std::nullopt;
        return degreesOfFreedom;
    }
    std::optional<Tensor> getLearnedLogDegreesOfFreedom() const { return logDegreesOfFreedomTensor; }
    float getMinimumDegreesOfFreedom() const { return minimumDegreesOfFreedom; }

    std::vector<Tensor> getLossInputTensors() const override {
        std::vector<Tensor> inputs{predictionsTensor, labelsTensor, logScaleTensor};
        if (logDegreesOfFreedomTensor.has_value())
            inputs.push_back(logDegreesOfFreedomTensor.value());
        if (exampleWeightsTensor.has_value())
            inputs.push_back(exampleWeightsTensor.value());
        return inputs;
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == logScaleTensor)
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        if (logDegreesOfFreedomTensor.has_value() && connectingTensor == logDegreesOfFreedomTensor.value())
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        return Loss::getConnectionType(connectingTensor);
    }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (logScaleTensor.isInitialized() && inputTensor == logScaleTensor)
            return "log_scale";
        if (logDegreesOfFreedomTensor.has_value() && inputTensor == logDegreesOfFreedomTensor.value())
            return "log_degrees_of_freedom";
        return Loss::getInputPortName(inputTensor);
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

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
        throw std::runtime_error("StudentTNLLLoss is a compound API loss and should not be stamped directly.");
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
        uint64_t logScaleBytes = batchSize * logScaleTensor.getTotalSizeInBytes() * 2;
        uint64_t degreesOfFreedomBytes = logDegreesOfFreedomTensor.has_value()
                                             ? batchSize * logDegreesOfFreedomTensor.value().getTotalSizeInBytes() * 2
                                             : 0;
        return standardLossBytes + logScaleBytes + degreesOfFreedomBytes + lossShaperBytes;
    }

    Tensor logScaleTensor;
    std::optional<Tensor> logDegreesOfFreedomTensor;
    float degreesOfFreedom = 3.0f;
    float minimumDegreesOfFreedom = 0.0f;
};

class StudentTNLLLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual StudentTNLLLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_location.has_value());
        THOR_THROW_IF_FALSE(_logScale.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_location.value() != _logScale.value());
        THOR_THROW_IF_FALSE(_location.value() != _labels.value());
        THOR_THROW_IF_FALSE(_logScale.value() != _labels.value());
        THOR_THROW_IF_FALSE(!(_degreesOfFreedom.has_value() && _logDegreesOfFreedom.has_value()));
        if (_logDegreesOfFreedom.has_value()) {
            THOR_THROW_IF_FALSE(_logDegreesOfFreedom.value() != _location.value());
            THOR_THROW_IF_FALSE(_logDegreesOfFreedom.value() != _logScale.value());
            THOR_THROW_IF_FALSE(_logDegreesOfFreedom.value() != _labels.value());
        }
        if (_exampleWeights.has_value()) {
            THOR_THROW_IF_FALSE(_exampleWeights.value() != _location.value());
            THOR_THROW_IF_FALSE(_exampleWeights.value() != _logScale.value());
            THOR_THROW_IF_FALSE(_exampleWeights.value() != _labels.value());
            if (_logDegreesOfFreedom.has_value())
                THOR_THROW_IF_FALSE(_exampleWeights.value() != _logDegreesOfFreedom.value());
        }
        THOR_THROW_IF_FALSE(!_location.value().getDimensions().empty());
        THOR_THROW_IF_FALSE(_location.value().getDimensions() == _logScale.value().getDimensions());
        THOR_THROW_IF_FALSE(_location.value().getDimensions() == _labels.value().getDimensions());
        if (_logDegreesOfFreedom.has_value())
            THOR_THROW_IF_FALSE(_location.value().getDimensions() == _logDegreesOfFreedom.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _location.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float degreesOfFreedom = _degreesOfFreedom.value_or(3.0f);
        THOR_THROW_IF_FALSE(std::isfinite(degreesOfFreedom) && degreesOfFreedom > 0.0f);
        THOR_THROW_IF_FALSE(std::isfinite(_minimumDegreesOfFreedom) && _minimumDegreesOfFreedom >= 0.0f);
        if (!_logDegreesOfFreedom.has_value())
            THOR_THROW_IF_FALSE(degreesOfFreedom > _minimumDegreesOfFreedom);

        StudentTNLLLoss loss;
        loss.predictionsTensor = _location.value();
        loss.logScaleTensor = _logScale.value();
        loss.labelsTensor = _labels.value();
        loss.logDegreesOfFreedomTensor = _logDegreesOfFreedom;
        loss.degreesOfFreedom = degreesOfFreedom;
        loss.minimumDegreesOfFreedom = _minimumDegreesOfFreedom;
        loss.exampleWeightsTensor = _exampleWeights;
        loss.lossDataType = _lossDataType.value();
        loss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        loss.lossShape = _lossShape.value();
        loss.network = _network.value();
        loss.initialized = true;
        loss.buildSupportLayersAndAddToNetwork();
        return loss;
    }

    virtual StudentTNLLLoss::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& location(Tensor _location) {
        THOR_THROW_IF_FALSE(!this->_location.has_value());
        THOR_THROW_IF_FALSE(!_location.getDimensions().empty());
        this->_location = _location;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& predictions(Tensor _location) { return location(_location); }

    virtual StudentTNLLLoss::Builder& logScale(Tensor _logScale) {
        THOR_THROW_IF_FALSE(!this->_logScale.has_value());
        THOR_THROW_IF_FALSE(!_logScale.getDimensions().empty());
        this->_logScale = _logScale;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& target(Tensor _target) { return labels(_target); }

    virtual StudentTNLLLoss::Builder& degreesOfFreedom(float _degreesOfFreedom) {
        THOR_THROW_IF_FALSE(!this->_degreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_logDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_degreesOfFreedom) && _degreesOfFreedom > 0.0f);
        this->_degreesOfFreedom = _degreesOfFreedom;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& logDegreesOfFreedom(Tensor _logDegreesOfFreedom) {
        THOR_THROW_IF_FALSE(!this->_logDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_degreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!_logDegreesOfFreedom.getDimensions().empty());
        this->_logDegreesOfFreedom = _logDegreesOfFreedom;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& minimumDegreesOfFreedom(float _minimumDegreesOfFreedom) {
        THOR_THROW_IF_FALSE(std::isfinite(_minimumDegreesOfFreedom) && _minimumDegreesOfFreedom >= 0.0f);
        this->_minimumDegreesOfFreedom = _minimumDegreesOfFreedom;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& reportsNoLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual StudentTNLLLoss::Builder& lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _location;
    std::optional<Tensor> _logScale;
    std::optional<Tensor> _labels;
    std::optional<float> _degreesOfFreedom;
    std::optional<Tensor> _logDegreesOfFreedom;
    float _minimumDegreesOfFreedom = 0.0f;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};

}  // namespace Thor
