#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Metrics/CategoricalAccuracy.h"
#include "DeepLearning/Implementation/Layers/Metrics/RaggedAccuracy.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class CategoricalAccuracy : public Metric {
   public:
    class Builder;
    CategoricalAccuracy() = default;
    ~CategoricalAccuracy() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CategoricalAccuracy>(*this); }

    std::string getLayerType() const override { return "CategoricalAccuracy"; }

    [[nodiscard]] ThorImplementation::RaggedPartitionRequirement
    getRaggedPartitionRequirementForInput(const Tensor& inputTensor) const override {
        if (raggedPredictions.has_value() && inputTensor == raggedPredictions->getOffsets())
            return ThorImplementation::kRaggedAccuracyPartitionRequirement;
        return Layer::getRaggedPartitionRequirementForInput(inputTensor);
    }
    MetricAggregation getAggregation() const override {
        return raggedPredictions.has_value() ? MetricAggregation::RATIO : MetricAggregation::MEAN_BY_EXAMPLE;
    }

    enum class LabelType { INDEX = 5, ONE_HOT };

    std::optional<RaggedTensor> getRaggedPredictions() const { return raggedPredictions; }
    std::optional<RaggedTensor> getRaggedLabels() const { return raggedLabels; }
    bool getUseRagged() const { return raggedPredictions.has_value(); }

    std::vector<Tensor> getAllInputTensors() const override {
        if (!getUseRagged())
            return Metric::getAllInputTensors();
        THOR_THROW_IF_FALSE(raggedLabels.has_value());
        return {raggedPredictions->getValues(), raggedLabels->getValues(), raggedPredictions->getOffsets()};
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (!getUseRagged())
            return Metric::getConnectionType(connectingTensor);
        if (connectingTensor == raggedPredictions->getValues())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::FORWARD);
        if (connectingTensor == raggedLabels->getValues())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::LABELS);
        if (connectingTensor == raggedPredictions->getOffsets())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::STRUCTURAL);
        if (connectingTensor == getMetric())
            return static_cast<int>(ThorImplementation::Metric::ConnectionType::METRIC);
        THOR_UNREACHABLE();
    }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (!getUseRagged())
            return Metric::getInputPortName(inputTensor);
        if (inputTensor == raggedPredictions->getValues())
            return "predictions";
        if (inputTensor == raggedLabels->getValues())
            return "labels";
        if (inputTensor == raggedPredictions->getOffsets())
            return "offsets";
        return std::nullopt;
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override {
        if (!getUseRagged())
            return Metric::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        (void)batchSize;
        (void)tensorPlacement;
        const uint64_t classes = raggedPredictions->getTrailingDimensions().at(0);
        const uint64_t workspaceBytes =
            ThorImplementation::raggedAccuracyStatisticsWorkspaceDescriptor(
                raggedPredictions->getMaxTotalValues(), classes)
                .getArraySizeInBytes();
        return metricTensor.getTotalSizeInBytes() + 4 * sizeof(float) + workspaceBytes;
    }

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
        if (!getUseRagged()) {
            THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value() || connectingApiTensor == labelsTensor);
            return std::make_shared<ThorImplementation::CategoricalAccuracy>();
        }
        THOR_THROW_IF_FALSE(raggedLabels.has_value());
        if (connectingApiTensor != raggedPredictions->getValues() &&
            connectingApiTensor != raggedLabels->getValues() &&
            connectingApiTensor != raggedPredictions->getOffsets()) {
            throw std::invalid_argument("CategoricalAccuracy ragged stamp received an unrelated tensor.");
        }
        const ThorImplementation::RaggedCategoricalLabelFormat format =
            labelType == LabelType::INDEX ? ThorImplementation::RaggedCategoricalLabelFormat::CLASS_INDEX
                                          : ThorImplementation::RaggedCategoricalLabelFormat::PER_CLASS;
        return std::make_shared<ThorImplementation::RaggedAccuracyMetric>(
            ThorImplementation::RaggedAccuracyDetail::Kind::CATEGORICAL,
            raggedPredictions->getBatchSize(),
            raggedPredictions->getMaxTotalValues(),
            numClasses,
            format);
    }

    LabelType labelType = LabelType::ONE_HOT;
    uint32_t numClasses = 0;
    std::optional<RaggedTensor> raggedPredictions;
    std::optional<RaggedTensor> raggedLabels;
};

class CategoricalAccuracy::Builder {
   public:
    virtual CategoricalAccuracy build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_labelType.has_value());
        THOR_THROW_IF_FALSE(_labelType.value() == LabelType::INDEX || _labelType.value() == LabelType::ONE_HOT);
        const bool dense = _predictions.has_value() || _labels.has_value();
        const bool ragged = _raggedPredictions.has_value() || _raggedLabels.has_value();
        if (dense == ragged)
            throw std::invalid_argument(
                "CategoricalAccuracy requires exactly one dense Tensor pair or RaggedTensor pair for predictions/labels.");

        CategoricalAccuracy metric;
        metric.labelType = _labelType.value();
        if (ragged) {
            if (!_raggedPredictions.has_value() || !_raggedLabels.has_value())
                throw std::invalid_argument("CategoricalAccuracy ragged predictions and labels must both be provided.");
            validateRaggedPair(_raggedPredictions.value(), _raggedLabels.value(), _labelType.value(), _numClasses);
            metric.raggedPredictions = _raggedPredictions.value();
            metric.raggedLabels = _raggedLabels.value();
            metric.featureInput = _raggedPredictions->getValues();
            metric.labelsTensor = _raggedLabels->getValues();
            metric.numClasses = static_cast<uint32_t>(_raggedPredictions->getTrailingDimensions().at(0));
        } else {
            THOR_THROW_IF_FALSE(_predictions.has_value());
            THOR_THROW_IF_FALSE(_labels.has_value());
            THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
            if (_labelType.value() == LabelType::ONE_HOT) {
                const std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
                THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] > 1);
                THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());
                metric.numClasses = static_cast<uint32_t>(_predictions.value().getDimensions()[0]);
            } else {
                const std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
                const std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
                THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);
                const DataType labelsDataType = _labels.value().getDataType();
                THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                                    labelsDataType == DataType::UINT32);
                THOR_THROW_IF_FALSE(_numClasses.has_value());
                THOR_THROW_IF_FALSE(predictionDimensions.size() == 1 && predictionDimensions[0] == _numClasses.value());
                metric.numClasses = _numClasses.value();
            }
            metric.featureInput = _predictions.value();
            metric.labelsTensor = _labels.value();
        }

        metric.metricTensor = Tensor(DataType::FP32, {1});
        metric.initialized = true;
        metric.addToNetwork(_network.value());
        return metric;
    }

    virtual CategoricalAccuracy::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual CategoricalAccuracy::Builder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value() && !this->_raggedPredictions.has_value());
        THOR_THROW_IF_FALSE(!predictions.getDimensions().empty());
        this->_predictions = std::move(predictions);
        return *this;
    }

    virtual CategoricalAccuracy::Builder& predictions(RaggedTensor predictions) {
        if (this->_predictions.has_value() || this->_raggedPredictions.has_value())
            throw std::invalid_argument("CategoricalAccuracy predictions were already provided.");
        if (!predictions.isInitialized())
            throw std::invalid_argument("CategoricalAccuracy ragged predictions must be initialized.");
        this->_raggedPredictions = std::move(predictions);
        return *this;
    }

    virtual CategoricalAccuracy::Builder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value() && !this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(!labels.getDimensions().empty());
        this->_labels = std::move(labels);
        return *this;
    }

    virtual CategoricalAccuracy::Builder& labels(RaggedTensor labels) {
        if (this->_labels.has_value() || this->_raggedLabels.has_value())
            throw std::invalid_argument("CategoricalAccuracy labels were already provided.");
        if (!labels.isInitialized())
            throw std::invalid_argument("CategoricalAccuracy ragged labels must be initialized.");
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    virtual CategoricalAccuracy::Builder& receivesClassIndexLabels(uint32_t numClasses) {
        THOR_THROW_IF_FALSE(!_labelType.has_value());
        THOR_THROW_IF_FALSE(numClasses > 1);
        _labelType = LabelType::INDEX;
        this->_numClasses = numClasses;
        return *this;
    }

    virtual CategoricalAccuracy::Builder& receivesOneHotLabels() {
        THOR_THROW_IF_FALSE(!_labelType.has_value());
        _labelType = LabelType::ONE_HOT;
        return *this;
    }

   private:
    static bool perClassLabelDTypeSupported(DataType dtype) {
        return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
               dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
               dtype == DataType::FP16 || dtype == DataType::FP32;
    }

    static bool classIndexLabelDTypeSupported(DataType dtype) {
        // Preserve the existing public CategoricalAccuracy index-label contract.
        return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32;
    }

    static void validateSamePartition(const RaggedTensor& predictions, const RaggedTensor& labels) {
        if (!predictions.isInitialized() || !labels.isInitialized())
            throw std::invalid_argument("CategoricalAccuracy ragged predictions and labels must be initialized.");
        if (predictions.getValues() == labels.getValues())
            throw std::invalid_argument("CategoricalAccuracy predictions and labels values must be distinct graph tensors.");
        if (!predictions.sharesPartitionWith(labels))
            throw std::invalid_argument(
                "CategoricalAccuracy ragged predictions and labels must use the exact same row partition.");
        if (predictions.getBatchSize() != labels.getBatchSize() ||
            predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
            predictions.getOffsetsDataType() != labels.getOffsetsDataType() ||
            predictions.hasMaxValuesPerRow() != labels.hasMaxValuesPerRow() ||
            (predictions.hasMaxValuesPerRow() && predictions.getMaxValuesPerRow() != labels.getMaxValuesPerRow())) {
            throw std::invalid_argument("CategoricalAccuracy ragged predictions and labels partition metadata must match.");
        }
    }

    static void validateRaggedPair(const RaggedTensor& predictions,
                                   const RaggedTensor& labels,
                                   LabelType labelType,
                                   const std::optional<uint32_t>& requestedNumClasses) {
        validateSamePartition(predictions, labels);
        if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
            throw std::invalid_argument("CategoricalAccuracy ragged predictions must be FP16 or FP32.");
        const std::vector<uint64_t> predictionTrailing = predictions.getTrailingDimensions();
        if (predictionTrailing.size() != 1 || predictionTrailing[0] < 2)
            throw std::invalid_argument(
                "CategoricalAccuracy ragged predictions must have one trailing class dimension with at least two classes.");
        const uint64_t classes = predictionTrailing[0];
        if (classes > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("CategoricalAccuracy ragged class width must fit uint32.");
        if (requestedNumClasses.has_value() && requestedNumClasses.value() != classes)
            throw std::invalid_argument("CategoricalAccuracy num_classes must match ragged prediction class width.");

        if (labelType == LabelType::INDEX) {
            if (!requestedNumClasses.has_value())
                throw std::invalid_argument("CategoricalAccuracy class-index labels require num_classes.");
            if (labels.getTrailingDimensions() != std::vector<uint64_t>{1})
                throw std::invalid_argument(
                    "CategoricalAccuracy ragged class-index labels must contain one scalar per token.");
            if (!classIndexLabelDTypeSupported(labels.getValuesDataType()))
                throw std::invalid_argument(
                    "CategoricalAccuracy ragged class-index labels must use UINT8, UINT16, or UINT32.");
        } else {
            if (labels.getTrailingDimensions() != predictionTrailing)
                throw std::invalid_argument(
                    "CategoricalAccuracy ragged one-hot/per-class labels must match prediction class width.");
            if (!perClassLabelDTypeSupported(labels.getValuesDataType()))
                throw std::invalid_argument("CategoricalAccuracy ragged per-class labels use an unsupported dtype.");
        }
    }

    std::optional<Network*> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<RaggedTensor> _raggedPredictions;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<LabelType> _labelType;
    std::optional<uint32_t> _numClasses;
};

NLOHMANN_JSON_SERIALIZE_ENUM(CategoricalAccuracy::LabelType,
                             {
                                 {CategoricalAccuracy::LabelType::ONE_HOT, "one_hot"},
                                 {CategoricalAccuracy::LabelType::INDEX, "index"},
                             })

}  // namespace Thor
