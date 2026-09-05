#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Implementation/Layers/Metrics/RaggedAccuracy.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class BinaryAccuracy : public Metric {
   public:
    class Builder;
    BinaryAccuracy() = default;
    ~BinaryAccuracy() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<BinaryAccuracy>(*this); }

    std::string getLayerType() const override { return "BinaryAccuracy"; }
    MetricAggregation getAggregation() const override {
        return raggedPredictions.has_value() ? MetricAggregation::RATIO : MetricAggregation::MEAN_BY_EXAMPLE;
    }

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
        // metric + ratio numerator/denominator + one default slot's device buffers.
        return metricTensor.getTotalSizeInBytes() + 4 * sizeof(float);
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
            return std::make_shared<ThorImplementation::BinaryAccuracy>();
        }
        THOR_THROW_IF_FALSE(raggedLabels.has_value());
        if (connectingApiTensor != raggedPredictions->getValues() &&
            connectingApiTensor != raggedLabels->getValues() &&
            connectingApiTensor != raggedPredictions->getOffsets()) {
            throw std::invalid_argument("BinaryAccuracy ragged stamp received an unrelated tensor.");
        }
        return std::make_shared<ThorImplementation::RaggedAccuracyMetric>(
            ThorImplementation::RaggedAccuracyDetail::Kind::BINARY,
            raggedPredictions->getBatchSize(),
            raggedPredictions->getMaxTotalValues());
    }

    std::optional<RaggedTensor> raggedPredictions;
    std::optional<RaggedTensor> raggedLabels;
};

class BinaryAccuracy::Builder {
   public:
    virtual BinaryAccuracy build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        const bool dense = _predictions.has_value() || _labels.has_value();
        const bool ragged = _raggedPredictions.has_value() || _raggedLabels.has_value();
        if (dense == ragged)
            throw std::invalid_argument(
                "BinaryAccuracy requires exactly one dense Tensor pair or RaggedTensor pair for predictions/labels.");

        BinaryAccuracy binaryAccuracy;
        if (ragged) {
            if (!_raggedPredictions.has_value() || !_raggedLabels.has_value())
                throw std::invalid_argument("BinaryAccuracy ragged predictions and labels must both be provided.");
            validateRaggedPair(_raggedPredictions.value(), _raggedLabels.value());
            binaryAccuracy.raggedPredictions = _raggedPredictions.value();
            binaryAccuracy.raggedLabels = _raggedLabels.value();
            binaryAccuracy.featureInput = _raggedPredictions->getValues();
            binaryAccuracy.labelsTensor = _raggedLabels->getValues();
        } else {
            THOR_THROW_IF_FALSE(_predictions.has_value());
            THOR_THROW_IF_FALSE(_labels.has_value());
            THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
            const std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
            const std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);
            THOR_THROW_IF_FALSE(predictionDimensions.size() == 1 && predictionDimensions[0] == 1);
            binaryAccuracy.featureInput = _predictions.value();
            binaryAccuracy.labelsTensor = _labels.value();
        }

        binaryAccuracy.metricTensor = Tensor(DataType::FP32, {1});
        binaryAccuracy.initialized = true;
        binaryAccuracy.addToNetwork(_network.value());
        return binaryAccuracy;
    }

    virtual BinaryAccuracy::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual BinaryAccuracy::Builder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value() && !this->_raggedPredictions.has_value());
        THOR_THROW_IF_FALSE(!predictions.getDimensions().empty());
        this->_predictions = std::move(predictions);
        return *this;
    }

    virtual BinaryAccuracy::Builder& predictions(RaggedTensor predictions) {
        if (this->_predictions.has_value() || this->_raggedPredictions.has_value())
            throw std::invalid_argument("BinaryAccuracy predictions were already provided.");
        if (!predictions.isInitialized())
            throw std::invalid_argument("BinaryAccuracy ragged predictions must be initialized.");
        this->_raggedPredictions = std::move(predictions);
        return *this;
    }

    virtual BinaryAccuracy::Builder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value() && !this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(!labels.getDimensions().empty());
        this->_labels = std::move(labels);
        return *this;
    }

    virtual BinaryAccuracy::Builder& labels(RaggedTensor labels) {
        if (this->_labels.has_value() || this->_raggedLabels.has_value())
            throw std::invalid_argument("BinaryAccuracy labels were already provided.");
        if (!labels.isInitialized())
            throw std::invalid_argument("BinaryAccuracy ragged labels must be initialized.");
        this->_raggedLabels = std::move(labels);
        return *this;
    }

   private:
    static bool labelsDTypeSupported(DataType dtype) {
        return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
               dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
               dtype == DataType::FP16 || dtype == DataType::FP32;
    }

    static void validateRaggedPair(const RaggedTensor& predictions, const RaggedTensor& labels) {
        if (!predictions.isInitialized() || !labels.isInitialized())
            throw std::invalid_argument("BinaryAccuracy ragged predictions and labels must be initialized.");
        if (predictions.getValues() == labels.getValues())
            throw std::invalid_argument("BinaryAccuracy predictions and labels values must be distinct graph tensors.");
        if (!predictions.sharesPartitionWith(labels))
            throw std::invalid_argument("BinaryAccuracy ragged predictions and labels must use the exact same row partition.");
        if (predictions.getBatchSize() != labels.getBatchSize() ||
            predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
            predictions.getOffsetsDataType() != labels.getOffsetsDataType() ||
            predictions.hasMaxValuesPerRow() != labels.hasMaxValuesPerRow() ||
            (predictions.hasMaxValuesPerRow() && predictions.getMaxValuesPerRow() != labels.getMaxValuesPerRow())) {
            throw std::invalid_argument("BinaryAccuracy ragged predictions and labels partition metadata must match.");
        }
        if (predictions.getTrailingDimensions() != std::vector<uint64_t>{1} ||
            labels.getTrailingDimensions() != std::vector<uint64_t>{1})
            throw std::invalid_argument("BinaryAccuracy ragged predictions and labels must contain one scalar per token.");
        if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
            throw std::invalid_argument("BinaryAccuracy ragged predictions must be FP16 or FP32.");
        if (!labelsDTypeSupported(labels.getValuesDataType()))
            throw std::invalid_argument("BinaryAccuracy ragged labels use an unsupported dtype.");
    }

    std::optional<Network*> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<RaggedTensor> _raggedPredictions;
    std::optional<RaggedTensor> _raggedLabels;
};

}  // namespace Thor
