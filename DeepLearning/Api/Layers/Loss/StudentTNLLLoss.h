#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"

#include <utility>

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
    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!isRagged()) throw std::runtime_error("StudentTNLLLoss location is dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!isRagged()) throw std::runtime_error("StudentTNLLLoss target is dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLogScale() const {
        if (!isRagged()) throw std::runtime_error("StudentTNLLLoss log_scale is dense.");
        return raggedLogScaleTensor.value();
    }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedLearnedLogDegreesOfFreedom() const {
        if (!isRagged()) throw std::runtime_error("StudentTNLLLoss learned degrees of freedom are dense.");
        return raggedLogDegreesOfFreedomTensor;
    }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("StudentTNLLLoss raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("StudentTNLLLoss does not expose a ragged reported loss for this LossShape.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] LossShape getLossShape() const { return lossShape; }

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
    std::optional<RaggedTensor> raggedPredictionsTensor;
    std::optional<RaggedTensor> raggedLabelsTensor;
    std::optional<RaggedTensor> raggedLogScaleTensor;
    std::optional<RaggedTensor> raggedLogDegreesOfFreedomTensor;
    std::optional<RaggedTensor> raggedRawLossTensor;

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
        const bool hasDenseLocation = _location.has_value();
        const bool hasDenseLogScale = _logScale.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedLocation = _raggedLocation.has_value();
        const bool hasRaggedLogScale = _raggedLogScale.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDenseLocation == hasDenseLogScale && hasDenseLogScale == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedLocation == hasRaggedLogScale && hasRaggedLogScale == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDenseLocation != hasRaggedLocation);
        THOR_THROW_IF_FALSE(!(_degreesOfFreedom.has_value() && (_logDegreesOfFreedom.has_value() || _raggedLogDegreesOfFreedom.has_value())));
        THOR_THROW_IF_FALSE(!(_logDegreesOfFreedom.has_value() && _raggedLogDegreesOfFreedom.has_value()));
        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;

        float degreesOfFreedom = _degreesOfFreedom.value_or(3.0f);
        THOR_THROW_IF_FALSE(std::isfinite(degreesOfFreedom) && degreesOfFreedom > 0.0f);
        THOR_THROW_IF_FALSE(std::isfinite(_minimumDegreesOfFreedom) && _minimumDegreesOfFreedom >= 0.0f);

        StudentTNLLLoss loss;
        if (hasDenseLocation) {
            THOR_THROW_IF_FALSE(_location.value() != _logScale.value());
            THOR_THROW_IF_FALSE(_location.value() != _labels.value());
            THOR_THROW_IF_FALSE(_logScale.value() != _labels.value());
            THOR_THROW_IF_FALSE(!_raggedLogDegreesOfFreedom.has_value());
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
            if (!_lossDataType.has_value()) _lossDataType = _location.value().getDataType();
            loss.predictionsTensor = _location.value();
            loss.logScaleTensor = _logScale.value();
            loss.labelsTensor = _labels.value();
            loss.logDegreesOfFreedomTensor = _logDegreesOfFreedom;
            loss.exampleWeightsTensor = _exampleWeights;
        } else {
            THOR_THROW_IF_FALSE(!_logDegreesOfFreedom.has_value());
            const RaggedTensor& location = _raggedLocation.value();
            const RaggedTensor& logScale = _raggedLogScale.value();
            const RaggedTensor& target = _raggedLabels.value();
            THOR_THROW_IF_FALSE(location.isInitialized() && logScale.isInitialized() && target.isInitialized());
            THOR_THROW_IF_FALSE(location.getValues() != logScale.getValues());
            THOR_THROW_IF_FALSE(location.getValues() != target.getValues());
            THOR_THROW_IF_FALSE(logScale.getValues() != target.getValues());
            if (location.getOffsets() != logScale.getOffsets() || location.getOffsets() != target.getOffsets())
                throw std::invalid_argument("StudentTNLLLoss ragged location, log_scale, and target must use the exact same row partition tensor.");
            if (location.getBatchSize() != logScale.getBatchSize() || location.getBatchSize() != target.getBatchSize() ||
                location.getMaxTotalValues() != logScale.getMaxTotalValues() || location.getMaxTotalValues() != target.getMaxTotalValues() ||
                location.getTrailingDimensions() != logScale.getTrailingDimensions() || location.getTrailingDimensions() != target.getTrailingDimensions())
                throw std::invalid_argument("StudentTNLLLoss ragged location, log_scale, and target must have identical value geometry.");
            if (_raggedLogDegreesOfFreedom.has_value()) {
                const RaggedTensor& logDof = _raggedLogDegreesOfFreedom.value();
                if (location.getOffsets() != logDof.getOffsets())
                    throw std::invalid_argument("StudentTNLLLoss ragged learned log degrees of freedom must use the exact same row partition tensor.");
                if (location.getBatchSize() != logDof.getBatchSize() || location.getMaxTotalValues() != logDof.getMaxTotalValues() ||
                    location.getTrailingDimensions() != logDof.getTrailingDimensions())
                    throw std::invalid_argument("StudentTNLLLoss ragged learned log degrees of freedom must have identical value geometry.");
                THOR_THROW_IF_FALSE(logDof.getValues() != location.getValues());
                THOR_THROW_IF_FALSE(logDof.getValues() != logScale.getValues());
                THOR_THROW_IF_FALSE(logDof.getValues() != target.getValues());
            }
            if (_exampleWeights.has_value() && _exampleWeights->getDimensions() != std::vector<uint64_t>{1})
                throw std::invalid_argument("StudentTNLLLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            if (_lossShape.value() == LossShape::PER_OUTPUT)
                throw std::invalid_argument("StudentTNLLLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");
            if (!_lossDataType.has_value()) _lossDataType = location.getValuesDataType();
            loss.predictionsTensor = location.getValues();
            loss.logScaleTensor = logScale.getValues();
            loss.labelsTensor = target.getValues();
            loss.raggedPredictionsTensor = location;
            loss.raggedLogScaleTensor = logScale;
            loss.raggedLabelsTensor = target;
            if (_raggedLogDegreesOfFreedom.has_value()) {
                loss.logDegreesOfFreedomTensor = _raggedLogDegreesOfFreedom->getValues();
                loss.raggedLogDegreesOfFreedomTensor = _raggedLogDegreesOfFreedom;
            }
            loss.exampleWeightsTensor = _exampleWeights;
        }

        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        if (!loss.logDegreesOfFreedomTensor.has_value())
            THOR_THROW_IF_FALSE(degreesOfFreedom > _minimumDegreesOfFreedom);
        loss.degreesOfFreedom = degreesOfFreedom;
        loss.minimumDegreesOfFreedom = _minimumDegreesOfFreedom;
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

    virtual StudentTNLLLoss::Builder& location(RaggedTensor location) {
        THOR_THROW_IF_FALSE(!this->_raggedLocation.has_value());
        THOR_THROW_IF_FALSE(location.isInitialized());
        this->_raggedLocation = std::move(location);
        return *this;
    }

    virtual StudentTNLLLoss::Builder& predictions(Tensor _location) { return location(_location); }
    virtual StudentTNLLLoss::Builder& predictions(RaggedTensor location) { return this->location(std::move(location)); }

    virtual StudentTNLLLoss::Builder& logScale(Tensor _logScale) {
        THOR_THROW_IF_FALSE(!this->_logScale.has_value());
        THOR_THROW_IF_FALSE(!_logScale.getDimensions().empty());
        this->_logScale = _logScale;
        return *this;
    }
    virtual StudentTNLLLoss::Builder& logScale(RaggedTensor logScale) {
        THOR_THROW_IF_FALSE(!this->_raggedLogScale.has_value());
        THOR_THROW_IF_FALSE(logScale.isInitialized());
        this->_raggedLogScale = std::move(logScale);
        return *this;
    }

    virtual StudentTNLLLoss::Builder& labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }
    virtual StudentTNLLLoss::Builder& labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    virtual StudentTNLLLoss::Builder& target(Tensor _target) { return labels(_target); }
    virtual StudentTNLLLoss::Builder& target(RaggedTensor target) { return labels(std::move(target)); }

    virtual StudentTNLLLoss::Builder& degreesOfFreedom(float _degreesOfFreedom) {
        THOR_THROW_IF_FALSE(!this->_degreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_logDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedLogDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_degreesOfFreedom) && _degreesOfFreedom > 0.0f);
        this->_degreesOfFreedom = _degreesOfFreedom;
        return *this;
    }

    virtual StudentTNLLLoss::Builder& logDegreesOfFreedom(Tensor _logDegreesOfFreedom) {
        THOR_THROW_IF_FALSE(!this->_logDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedLogDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_degreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!_logDegreesOfFreedom.getDimensions().empty());
        this->_logDegreesOfFreedom = _logDegreesOfFreedom;
        return *this;
    }
    virtual StudentTNLLLoss::Builder& logDegreesOfFreedom(RaggedTensor logDegreesOfFreedom) {
        THOR_THROW_IF_FALSE(!this->_raggedLogDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_logDegreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(!this->_degreesOfFreedom.has_value());
        THOR_THROW_IF_FALSE(logDegreesOfFreedom.isInitialized());
        this->_raggedLogDegreesOfFreedom = std::move(logDegreesOfFreedom);
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
    std::optional<RaggedTensor> _raggedLocation;
    std::optional<RaggedTensor> _raggedLogScale;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<float> _degreesOfFreedom;
    std::optional<Tensor> _logDegreesOfFreedom;
    std::optional<RaggedTensor> _raggedLogDegreesOfFreedom;
    float _minimumDegreesOfFreedom = 0.0f;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};

}  // namespace Thor
