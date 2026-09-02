#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>
#include <utility>

namespace Thor {

class LaplaceNLLLoss : public Loss {
   public:
    class Builder;
    LaplaceNLLLoss() {}

    ~LaplaceNLLLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<LaplaceNLLLoss>(*this); }

    std::string getLayerType() const override { return "LaplaceNLLLoss"; }

    Tensor getLocation() const { return predictionsTensor; }
    Tensor getTarget() const { return labelsTensor; }
    Tensor getScale() const { return scaleTensor; }
    bool getLogScale() const { return logScale; }
    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!isRagged()) throw std::runtime_error("LaplaceNLLLoss location is dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!isRagged()) throw std::runtime_error("LaplaceNLLLoss target is dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedScale() const {
        if (!isRagged()) throw std::runtime_error("LaplaceNLLLoss scale is dense.");
        return raggedScaleTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("LaplaceNLLLoss raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("LaplaceNLLLoss does not expose a ragged reported loss for this LossShape.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] LossShape getLossShape() const { return lossShape; }
    float getEps() const { return eps; }

    std::vector<Tensor> getLossInputTensors() const override {
        std::vector<Tensor> inputs{predictionsTensor, labelsTensor, scaleTensor};
        if (exampleWeightsTensor.has_value())
            inputs.push_back(exampleWeightsTensor.value());
        return inputs;
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == scaleTensor)
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        return Loss::getConnectionType(connectingTensor);
    }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (scaleTensor.isInitialized() && inputTensor == scaleTensor)
            return "scale";
        return Loss::getInputPortName(inputTensor);
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::optional<RaggedTensor> raggedPredictionsTensor;
    std::optional<RaggedTensor> raggedLabelsTensor;
    std::optional<RaggedTensor> raggedScaleTensor;
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
        throw std::runtime_error("LaplaceNLLLoss is a compound API loss and should not be stamped directly.");
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
        uint64_t scaleBytes = batchSize * scaleTensor.getTotalSizeInBytes() * 2;
        return standardLossBytes + scaleBytes + lossShaperBytes;
    }

    Tensor scaleTensor;
    bool logScale = true;
    float eps = 1.0e-8f;
};

class LaplaceNLLLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual LaplaceNLLLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        const bool hasDenseLocation = _location.has_value();
        const bool hasDenseScale = _scale.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedLocation = _raggedLocation.has_value();
        const bool hasRaggedScale = _raggedScale.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDenseLocation == hasDenseScale && hasDenseScale == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedLocation == hasRaggedScale && hasRaggedScale == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDenseLocation != hasRaggedLocation);
        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;

        const float eps = _eps.value_or(1.0e-8f);
        THOR_THROW_IF_FALSE(eps > 0.0f);

        LaplaceNLLLoss loss;
        if (hasDenseLocation) {
            THOR_THROW_IF_FALSE(_location.value() != _scale.value());
            THOR_THROW_IF_FALSE(_location.value() != _labels.value());
            THOR_THROW_IF_FALSE(_scale.value() != _labels.value());
            if (_exampleWeights.has_value()) {
                THOR_THROW_IF_FALSE(_exampleWeights.value() != _location.value());
                THOR_THROW_IF_FALSE(_exampleWeights.value() != _scale.value());
                THOR_THROW_IF_FALSE(_exampleWeights.value() != _labels.value());
            }
            THOR_THROW_IF_FALSE(!_location.value().getDimensions().empty());
            THOR_THROW_IF_FALSE(_location.value().getDimensions() == _scale.value().getDimensions());
            THOR_THROW_IF_FALSE(_location.value().getDimensions() == _labels.value().getDimensions());
            if (!_lossDataType.has_value()) _lossDataType = _location.value().getDataType();
            loss.predictionsTensor = _location.value();
            loss.scaleTensor = _scale.value();
            loss.labelsTensor = _labels.value();
            loss.exampleWeightsTensor = _exampleWeights;
        } else {
            const RaggedTensor& location = _raggedLocation.value();
            const RaggedTensor& scale = _raggedScale.value();
            const RaggedTensor& target = _raggedLabels.value();
            THOR_THROW_IF_FALSE(location.isInitialized() && scale.isInitialized() && target.isInitialized());
            THOR_THROW_IF_FALSE(location.getValues() != scale.getValues());
            THOR_THROW_IF_FALSE(location.getValues() != target.getValues());
            THOR_THROW_IF_FALSE(scale.getValues() != target.getValues());
            if (location.getOffsets() != scale.getOffsets() || location.getOffsets() != target.getOffsets())
                throw std::invalid_argument("LaplaceNLLLoss ragged location, scale, and target must use the exact same row partition tensor.");
            if (location.getBatchSize() != scale.getBatchSize() || location.getBatchSize() != target.getBatchSize() ||
                location.getMaxTotalValues() != scale.getMaxTotalValues() || location.getMaxTotalValues() != target.getMaxTotalValues() ||
                location.getTrailingDimensions() != scale.getTrailingDimensions() || location.getTrailingDimensions() != target.getTrailingDimensions())
                throw std::invalid_argument("LaplaceNLLLoss ragged location, scale, and target must have identical value geometry.");
            if (_exampleWeights.has_value() && _exampleWeights->getDimensions() != std::vector<uint64_t>{1})
                throw std::invalid_argument("LaplaceNLLLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            if (_lossShape.value() == LossShape::PER_OUTPUT)
                throw std::invalid_argument("LaplaceNLLLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");
            if (!_lossDataType.has_value()) _lossDataType = location.getValuesDataType();
            loss.predictionsTensor = location.getValues();
            loss.scaleTensor = scale.getValues();
            loss.labelsTensor = target.getValues();
            loss.raggedPredictionsTensor = location;
            loss.raggedScaleTensor = scale;
            loss.raggedLabelsTensor = target;
            loss.exampleWeightsTensor = _exampleWeights;
        }
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        loss.lossDataType = _lossDataType.value();
        loss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        loss.lossShape = _lossShape.value();
        loss.logScale = _logScale.value_or(true);
        loss.eps = eps;
        loss.network = _network.value();
        loss.initialized = true;
        loss.buildSupportLayersAndAddToNetwork();
        return loss;
    }

    virtual LaplaceNLLLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &location(Tensor _location) {
        THOR_THROW_IF_FALSE(!this->_location.has_value());
        THOR_THROW_IF_FALSE(!_location.getDimensions().empty());
        this->_location = _location;
        return *this;
    }
    virtual LaplaceNLLLoss::Builder &location(RaggedTensor location) {
        THOR_THROW_IF_FALSE(!this->_raggedLocation.has_value());
        THOR_THROW_IF_FALSE(location.isInitialized());
        this->_raggedLocation = std::move(location);
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &predictions(Tensor _location) { return location(_location); }
    virtual LaplaceNLLLoss::Builder &predictions(RaggedTensor location) { return this->location(std::move(location)); }

    virtual LaplaceNLLLoss::Builder &scale(Tensor _scale) {
        THOR_THROW_IF_FALSE(!this->_scale.has_value());
        THOR_THROW_IF_FALSE(!_scale.getDimensions().empty());
        this->_scale = _scale;
        return *this;
    }
    virtual LaplaceNLLLoss::Builder &scale(RaggedTensor scale) {
        THOR_THROW_IF_FALSE(!this->_raggedScale.has_value());
        THOR_THROW_IF_FALSE(scale.isInitialized());
        this->_raggedScale = std::move(scale);
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }
    virtual LaplaceNLLLoss::Builder &labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &target(Tensor _target) { return labels(_target); }
    virtual LaplaceNLLLoss::Builder &target(RaggedTensor target) { return labels(std::move(target)); }

    virtual LaplaceNLLLoss::Builder &logScale(bool _logScale) {
        THOR_THROW_IF_FALSE(!this->_logScale.has_value());
        this->_logScale = _logScale;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &eps(float _eps) {
        THOR_THROW_IF_FALSE(!this->_eps.has_value());
        THOR_THROW_IF_FALSE(_eps > 0.0f);
        this->_eps = _eps;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &reportsNoLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual LaplaceNLLLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _location;
    std::optional<Tensor> _scale;
    std::optional<Tensor> _labels;
    std::optional<RaggedTensor> _raggedLocation;
    std::optional<RaggedTensor> _raggedScale;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<bool> _logScale;
    std::optional<float> _eps;
};

}  // namespace Thor
