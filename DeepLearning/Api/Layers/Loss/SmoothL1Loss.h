#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <utility>
#include <stdexcept>

namespace Thor {

class SmoothL1Loss : public Loss {
   public:
    class Builder;
    SmoothL1Loss() {}

    ~SmoothL1Loss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<SmoothL1Loss>(*this); }

    std::string getLayerType() const override { return "SmoothL1Loss"; }

    float getBeta() const { return beta; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!raggedPredictionsTensor.has_value()) throw std::runtime_error("SmoothL1Loss predictions are dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!raggedLabelsTensor.has_value()) throw std::runtime_error("SmoothL1Loss labels are dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("SmoothL1Loss raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("SmoothL1Loss does not expose a ragged reported loss for this LossShape.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] LossShape getLossShape() const { return lossShape; }

   protected:
    std::optional<RaggedTensor> raggedPredictionsTensor;
    std::optional<RaggedTensor> raggedLabelsTensor;
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
        throw std::runtime_error("SmoothL1Loss is a compound API loss and should not be stamped directly.");
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        if (isMultiLayer() && !isRagged()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        return standardLossBytes + lossShaperBytes;
    }

    float beta = 1.0f;
};

class SmoothL1Loss::Builder {
   public:
    virtual ~Builder() = default;

    virtual SmoothL1Loss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        const bool hasDensePredictions = _predictions.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedPredictions = _raggedPredictions.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDensePredictions == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedPredictions == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDensePredictions != hasRaggedPredictions);

        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;
        const float parameter = _beta.value_or(1.0f);
        THOR_THROW_IF_FALSE(parameter > 0.0f);

        SmoothL1Loss loss;
        if (hasDensePredictions) {
            THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
            THOR_THROW_IF_FALSE(!_predictions.value().getDimensions().empty());
            THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());
            if (!_lossDataType.has_value()) _lossDataType = _predictions.value().getDataType();
            loss.predictionsTensor = _predictions.value();
            loss.labelsTensor = _labels.value();
        } else {
            const RaggedTensor& predictions = _raggedPredictions.value();
            const RaggedTensor& labels = _raggedLabels.value();
            THOR_THROW_IF_FALSE(predictions.isInitialized() && labels.isInitialized());
            THOR_THROW_IF_FALSE(predictions.getValues() != labels.getValues());
            if (predictions.getOffsets() != labels.getOffsets())
                throw std::invalid_argument("SmoothL1Loss ragged predictions and labels must use the exact same row partition tensor.");
            if (predictions.getBatchSize() != labels.getBatchSize() ||
                predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                throw std::invalid_argument("SmoothL1Loss ragged predictions and labels must have identical value geometry.");
            if (_lossShape.value() == LossShape::PER_OUTPUT)
                throw std::invalid_argument("SmoothL1Loss LossShape::PER_OUTPUT is undefined for ragged sequences.");
            if (!_lossDataType.has_value()) _lossDataType = predictions.getValuesDataType();
            loss.predictionsTensor = predictions.getValues();
            loss.labelsTensor = labels.getValues();
            loss.raggedPredictionsTensor = predictions;
            loss.raggedLabelsTensor = labels;
        }

        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        loss.lossDataType = _lossDataType.value();
        loss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        loss.lossShape = _lossShape.value();
        loss.beta = parameter;
        loss.network = _network.value();
        loss.initialized = true;
        loss.buildSupportLayersAndAddToNetwork();
        return loss;
    }

    virtual Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Builder &predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!predictions.getDimensions().empty());
        this->_predictions = std::move(predictions);
        return *this;
    }

    virtual Builder &predictions(RaggedTensor predictions) {
        THOR_THROW_IF_FALSE(!this->_raggedPredictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_raggedPredictions = std::move(predictions);
        return *this;
    }

    virtual Builder &labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!labels.getDimensions().empty());
        this->_labels = std::move(labels);
        return *this;
    }

    virtual Builder &labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    virtual Builder &beta(float value) {
        THOR_THROW_IF_FALSE(!this->_beta.has_value());
        THOR_THROW_IF_FALSE(value > 0.0f);
        this->_beta = value;
        return *this;
    }

    virtual Builder &reportsBatchLoss() { THOR_THROW_IF_FALSE(!_lossShape.has_value()); _lossShape = LossShape::BATCH; return *this; }
    virtual Builder &reportsPerExampleLoss() { THOR_THROW_IF_FALSE(!_lossShape.has_value()); _lossShape = LossShape::PER_EXAMPLE; return *this; }
    virtual Builder &reportsPerOutputLoss() { THOR_THROW_IF_FALSE(!_lossShape.has_value()); _lossShape = LossShape::PER_OUTPUT; return *this; }
    virtual Builder &reportsNoLoss() { THOR_THROW_IF_FALSE(!_lossShape.has_value()); _lossShape = LossShape::NONE; return *this; }
    virtual Builder &reportsRawLoss() { THOR_THROW_IF_FALSE(!_lossShape.has_value()); _lossShape = LossShape::RAW; return *this; }

    virtual Builder &lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        _lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual Builder &lossDataType(DataType lossDataType) {
        THOR_THROW_IF_FALSE(!_lossDataType.has_value());
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
        _lossDataType = lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<RaggedTensor> _raggedPredictions;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<float> _beta;
};

}  // namespace Thor
