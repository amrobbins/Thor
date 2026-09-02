#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropy.h"
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class BinaryCrossEntropy : public Loss {
   public:
    class Builder;
    BinaryCrossEntropy() {}

    ~BinaryCrossEntropy() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<BinaryCrossEntropy>(*this); }

    std::string getLayerType() const override { return "BinaryCrossEntropy"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!raggedPredictionsTensor.has_value()) throw std::runtime_error("BinaryCrossEntropy predictions are dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!raggedLabelsTensor.has_value()) throw std::runtime_error("BinaryCrossEntropy labels are dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("BinaryCrossEntropy raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("BinaryCrossEntropy does not expose a ragged reported loss for this LossShape.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] LossShape getLossShape() const { return lossShape; }

   protected:
    std::optional<RaggedTensor> raggedPredictionsTensor;
    std::optional<RaggedTensor> raggedLabelsTensor;
    std::optional<RaggedTensor> raggedRawLossTensor;

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        return std::make_shared<ThorImplementation::BinaryCrossEntropy>(lossDataType);
    }

    virtual bool isMultiLayer() const { return true; }

    virtual void buildSupportLayersAndAddToNetwork();

    bool rawLossAddedToNetwork = false;
};

class BinaryCrossEntropy::Builder {
   public:
    virtual BinaryCrossEntropy build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        const bool hasDensePredictions = _predictions.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedPredictions = _raggedPredictions.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDensePredictions == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedPredictions == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDensePredictions != hasRaggedPredictions);
        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;

        BinaryCrossEntropy binaryCrossEntropy;
        binaryCrossEntropy.rawLossAddedToNetwork = _rawLossAddedToNetwork.value_or(false);
        if (hasDensePredictions) {
            THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
            const std::vector<uint64_t> predictionDimensions = _predictions->getDimensions();
            const std::vector<uint64_t> labelDimensions = _labels->getDimensions();
            THOR_THROW_IF_FALSE(!predictionDimensions.empty());
            THOR_THROW_IF_FALSE(predictionDimensions == labelDimensions);
            binaryCrossEntropy.predictionsTensor = _predictions.value();
            binaryCrossEntropy.labelsTensor = _labels.value();
        } else {
            const RaggedTensor& predictions = _raggedPredictions.value();
            const RaggedTensor& labels = _raggedLabels.value();
            THOR_THROW_IF_FALSE(predictions.isInitialized() && labels.isInitialized());
            THOR_THROW_IF_FALSE(predictions.getValues() != labels.getValues());
            THOR_THROW_IF_FALSE(predictions.getValuesDataType() == DataType::FP16 || predictions.getValuesDataType() == DataType::FP32);
            if (predictions.getOffsets() != labels.getOffsets())
                throw std::invalid_argument("BinaryCrossEntropy ragged predictions and labels must use the exact same row partition tensor.");
            if (predictions.getBatchSize() != labels.getBatchSize() ||
                predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                throw std::invalid_argument("BinaryCrossEntropy ragged predictions and labels must have identical value geometry.");
            if (_lossShape.value() == LossShape::PER_OUTPUT)
                throw std::invalid_argument("BinaryCrossEntropy LossShape::PER_OUTPUT is undefined for ragged sequences.");
            THOR_THROW_IF_FALSE(!binaryCrossEntropy.rawLossAddedToNetwork);
            binaryCrossEntropy.predictionsTensor = predictions.getValues();
            binaryCrossEntropy.labelsTensor = labels.getValues();
            binaryCrossEntropy.raggedPredictionsTensor = predictions;
            binaryCrossEntropy.raggedLabelsTensor = labels;
        }

        if (!_lossDataType.has_value()) _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        binaryCrossEntropy.lossDataType = _lossDataType.value();
        binaryCrossEntropy.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::NONE || _lossShape.value() == LossShape::BATCH ||
                            _lossShape.value() == LossShape::PER_EXAMPLE || _lossShape.value() == LossShape::PER_OUTPUT ||
                            _lossShape.value() == LossShape::RAW);
        binaryCrossEntropy.lossShape = _lossShape.value();
        binaryCrossEntropy.initialized = true;
        binaryCrossEntropy.network = _network.value();

        if (binaryCrossEntropy.rawLossAddedToNetwork) {
            THOR_THROW_IF_FALSE(hasDensePredictions);
            THOR_THROW_IF_FALSE(binaryCrossEntropy.lossShape == LossShape::PER_EXAMPLE);
            binaryCrossEntropy.lossTensor = Tensor(_lossDataType.value(), _predictions->getDimensions());
            binaryCrossEntropy.lossShaperInput = binaryCrossEntropy.lossTensor;
            binaryCrossEntropy.addToNetwork(_network.value());
        } else {
            binaryCrossEntropy.buildSupportLayersAndAddToNetwork();
        }
        return binaryCrossEntropy;
    }

    virtual BinaryCrossEntropy::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &predictions(RaggedTensor predictions) {
        THOR_THROW_IF_FALSE(!this->_raggedPredictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_raggedPredictions = std::move(predictions);
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    /**
     * Reports one scalar equal to the sum of all non-batch loss values averaged over the batch.
     * Note that this setting affects reporting only, not the loss used to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    /**
     * Reports one scalar per example by summing every non-batch loss dimension.
     * Note that this setting affects reporting only, not the loss used to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    /**
     * Reports the loss averaged over the batch while preserving every non-batch loss dimension.
     * Note that this setting affects reporting only, not the loss used to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    /**
     * Does not expose a reported loss tensor. The raw loss remains available internally as the training objective.
     */
    virtual BinaryCrossEntropy::Builder &reportsNoLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    /**
     * Reports the unreduced pointwise loss tensor.
     * Note that this setting affects reporting only, not the loss used to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP32 || _lossDataType == DataType::FP16);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   protected:
    /**
     * Legacy/internal path for reconstructing the historical raw BCE layer. Public BCE construction now routes through CustomLoss.
     */
    virtual BinaryCrossEntropy::Builder &rawLossAddedToNetwork() {
        THOR_THROW_IF_FALSE(!_rawLossAddedToNetwork.has_value());
        _rawLossAddedToNetwork = true;
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
    std::optional<bool> _rawLossAddedToNetwork;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
