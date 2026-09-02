#pragma once
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.h"
#include <optional>

namespace Thor {

class MAE : public Loss {
   public:
    class Builder;
    MAE() {}

    ~MAE() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<MAE>(*this); }

    std::string getLayerType() const override { return "MAE"; }

    nlohmann::json architectureJson() const override;

    static void deserialize(const nlohmann::json &j, Network *network);

    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!raggedPredictionsTensor.has_value()) throw std::runtime_error("MAE predictions are dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!raggedLabelsTensor.has_value()) throw std::runtime_error("MAE labels are dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("MAE raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("MAE does not expose a ragged reported loss for this LossShape.");
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
        // FIXME: How to prune backward then.
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::MeanAbsoluteError> meanAbsoluteError =
            std::make_shared<ThorImplementation::MeanAbsoluteError>(lossDataType);

        return meanAbsoluteError;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        // Loss will be reported either element-wise or batch-wise, the shaper is only required when loss is batch-wise.
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        return standardLossBytes + lossShaperBytes;
    }
};

class MAE::Builder {
   public:
    virtual ~Builder() = default;

    virtual MAE build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        const bool hasDensePredictions = _predictions.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedPredictions = _raggedPredictions.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDensePredictions == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedPredictions == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDensePredictions != hasRaggedPredictions);

        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;

        MAE meanAbsoluteError;
        if (hasDensePredictions) {
            THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
            THOR_THROW_IF_FALSE(!_predictions.value().getDimensions().empty());
            THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());
            if (!_lossDataType.has_value())
                _lossDataType = ThorImplementation::RegressionLossDType::defaultLossDType(_predictions.value().getDataType());
            meanAbsoluteError.predictionsTensor = _predictions.value();
            meanAbsoluteError.labelsTensor = _labels.value();
            meanAbsoluteError.exampleWeightsTensor = _exampleWeights;
        } else {
            const RaggedTensor& predictions = _raggedPredictions.value();
            const RaggedTensor& labels = _raggedLabels.value();
            THOR_THROW_IF_FALSE(predictions.isInitialized() && labels.isInitialized());
            THOR_THROW_IF_FALSE(predictions.getValues() != labels.getValues());
            ThorImplementation::RegressionLossDType::validatePredictionsDType("MAE", predictions.getValuesDataType());
            ThorImplementation::RegressionLossDType::validateLabelsDType("MAE", labels.getValuesDataType());
            if (predictions.getOffsets() != labels.getOffsets())
                throw std::invalid_argument("MAE ragged predictions and labels must use the exact same row partition tensor.");
            if (predictions.getBatchSize() != labels.getBatchSize() ||
                predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                throw std::invalid_argument("MAE ragged predictions and labels must have identical value geometry.");
            if (_exampleWeights.has_value()) {
                if (_exampleWeights.value() == predictions.getValues() || _exampleWeights.value() == labels.getValues())
                    throw std::invalid_argument("MAE ragged example_weights must be distinct from predictions and labels values.");
                ThorImplementation::RegressionLossDType::validateExampleWeightDType(
                    "MAE", _exampleWeights->getDataType());
                if (_exampleWeights->getDimensions() != std::vector<uint64_t>{1})
                    throw std::invalid_argument("MAE ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            }
            if (_lossShape.value() == LossShape::PER_OUTPUT)
                throw std::invalid_argument("MAE LossShape::PER_OUTPUT is undefined for ragged sequences.");
            if (!_lossDataType.has_value())
                _lossDataType = ThorImplementation::RegressionLossDType::defaultLossDType(predictions.getValuesDataType());
            meanAbsoluteError.predictionsTensor = predictions.getValues();
            meanAbsoluteError.labelsTensor = labels.getValues();
            meanAbsoluteError.raggedPredictionsTensor = predictions;
            meanAbsoluteError.raggedLabelsTensor = labels;
            meanAbsoluteError.exampleWeightsTensor = _exampleWeights;
        }

        ThorImplementation::RegressionLossDType::validateLossDType("MAE", _lossDataType.value());
        meanAbsoluteError.lossDataType = _lossDataType.value();
        meanAbsoluteError.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        meanAbsoluteError.lossShape = _lossShape.value();
        meanAbsoluteError.network = _network.value();
        meanAbsoluteError.initialized = true;
        meanAbsoluteError.buildSupportLayersAndAddToNetwork();
        return meanAbsoluteError;
    }

    virtual MAE::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual MAE::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual MAE::Builder &predictions(RaggedTensor predictions) {
        THOR_THROW_IF_FALSE(!this->_raggedPredictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_raggedPredictions = std::move(predictions);
        return *this;
    }

    virtual MAE::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual MAE::Builder &labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    virtual MAE::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual MAE::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual MAE::Builder &reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    virtual MAE::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    virtual MAE::Builder &reportsNoLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    virtual MAE::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual MAE::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual MAE::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        ThorImplementation::RegressionLossDType::validateLossDType("MAE", _lossDataType);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<RaggedTensor> _raggedPredictions;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
};


}  // namespace Thor
