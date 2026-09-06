#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"
#include "DeepLearning/Implementation/Layers/Loss/SparseCategoricalCrossEntropyWithLogits.h"
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace Thor {

class SparseCategoricalCrossEntropy;

class CategoricalCrossEntropy : public Loss {
   public:
    class Builder;
    CategoricalCrossEntropy() {}

    ~CategoricalCrossEntropy() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CategoricalCrossEntropy>(*this); }

    std::string getLayerType() const override { return "CategoricalCrossEntropy"; }

    Tensor getPredictions() const override { return softmaxOutput.isInitialized() ? softmaxOutput : predictionsTensor; }

    std::vector<Tensor> getLossInputTensors() const override {
        std::vector<Tensor> inputs{predictionsTensor, labelsTensor};
        if (maskTensor.has_value())
            inputs.push_back(maskTensor.value());
        if (isRagged() && labelType == LabelType::SPARSE)
            inputs.push_back(raggedPredictionsTensor->getRowPartitionToken());
        return inputs;
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (maskTensor.has_value() && connectingTensor == maskTensor.value())
            return ThorImplementation::SparseCategoricalCrossEntropyWithLogits::MASK_CONNECTION_TYPE;
        if (isRagged() && labelType == LabelType::SPARSE &&
            connectingTensor == raggedPredictionsTensor->getRowPartitionToken())
            return ThorImplementation::SparseCategoricalCrossEntropyWithLogits::ACTIVE_COUNT_CONNECTION_TYPE;
        return Loss::getConnectionType(connectingTensor);
    }

    [[nodiscard]] ThorImplementation::RaggedPartitionRequirement
    getRaggedPartitionRequirementForInput(const Tensor& inputTensor) const override {
        if (isRagged() && labelType == LabelType::SPARSE &&
            inputTensor == raggedPredictionsTensor->getRowPartitionToken())
            return ThorImplementation::kRaggedSparseCategoricalCrossEntropyWithLogitsPartitionRequirement;
        return Layer::getRaggedPartitionRequirementForInput(inputTensor);
    }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (maskTensor.has_value() && inputTensor == maskTensor.value()) return "mask";
        if (isRagged() && labelType == LabelType::SPARSE &&
            inputTensor == raggedPredictionsTensor->getRowPartitionToken())
            return "active_count";
        return Loss::getInputPortName(inputTensor);
    }

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        if (isRagged() && labelType == LabelType::SPARSE && logitsNativeLossAddedToNetwork) {
            THOR_THROW_IF_FALSE(raggedRawLossTensor.has_value());
            THOR_THROW_IF_FALSE(outputTensor == raggedRawLossTensor->getValues());
            return true;
        }
        return Layer::outputTensorDimensionsIncludeBatch(outputTensor);
    }

    [[nodiscard]] uint64_t getOutputTensorBytes(uint32_t batchSize) const override {
        if (isRagged() && labelType == LabelType::SPARSE && logitsNativeLossAddedToNetwork) {
            (void)batchSize;
            THOR_THROW_IF_FALSE(raggedRawLossTensor.has_value());
            return raggedRawLossTensor->getValues().getTotalSizeInBytes();
        }
        return Layer::getOutputTensorBytes(batchSize);
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!raggedPredictionsTensor.has_value()) throw std::runtime_error("CategoricalCrossEntropy predictions are dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!raggedLabelsTensor.has_value()) throw std::runtime_error("CategoricalCrossEntropy labels are dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedMask() const { return raggedMaskTensor; }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("CategoricalCrossEntropy raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("CategoricalCrossEntropy does not expose a ragged reported loss for this LossShape.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] LossShape getLossShape() const { return lossShape; }

   protected:
    static void deserializeInto(const nlohmann::json &j,
                                Network *network,
                                CategoricalCrossEntropy &categoricalCrossEntropy,
                                LabelType labelType,
                                const std::string &expectedLayerType);

    virtual bool isMultiLayer() const {
        if (isRagged()) {
            if (labelType == LabelType::SPARSE && lossShape == LossShape::RAW && logitsNativeLossAddedToNetwork)
                return false;
            return true;
        }
        if (labelType == LabelType::SPARSE) {
            if (lossShape != LossShape::RAW || !logitsNativeLossAddedToNetwork)
                return true;
            return false;
        }
        if (lossShape != LossShape::RAW || !softmaxAddedToNetwork)
            return true;
        return false;
    }

    virtual void buildSupportLayersAndAddToNetwork();

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
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor ||
                            (maskTensor.has_value() && connectingApiTensor == maskTensor.value()) ||
                            (isRagged() && labelType == LabelType::SPARSE &&
                             connectingApiTensor == raggedPredictionsTensor->getRowPartitionToken()));

        // Softmax and LossShaper are connected during multi-layer flattening.
        if (labelType == LabelType::SPARSE && logitsNativeLossAddedToNetwork) {
            std::optional<uint32_t> raggedBatchSize;
            if (isRagged()) {
                THOR_THROW_IF_FALSE(raggedPredictionsTensor->getBatchSize() <= std::numeric_limits<uint32_t>::max());
                raggedBatchSize = static_cast<uint32_t>(raggedPredictionsTensor->getBatchSize());
            }
            return std::make_shared<ThorImplementation::SparseCategoricalCrossEntropyWithLogits>(
                lossDataType, lossWeight, ignoreIndex, raggedBatchSize);
        }

        return std::make_shared<ThorImplementation::CrossEntropy>(
            CrossEntropyLossType::CATEGORICAL, lossDataType, labelType == LabelType::SPARSE, lossWeight);
    }

    std::optional<RaggedTensor> raggedPredictionsTensor;
    std::optional<RaggedTensor> raggedLabelsTensor;
    std::optional<RaggedTensor> raggedMaskTensor;
    std::optional<RaggedTensor> raggedRawLossTensor;

    LabelType labelType = LabelType::DENSE;
    uint32_t numClasses = 0;
    bool softmaxAddedToNetwork = false;
    bool logitsNativeLossAddedToNetwork = false;
    Tensor softmaxOutput;
    std::optional<uint32_t> ignoreIndex;
    std::optional<Tensor> maskTensor;

    uint64_t getFirstInstanceMemRequirementInBytes(
        uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        if (isRagged() && labelType == LabelType::SPARSE && logitsNativeLossAddedToNetwork) {
            (void)batchSize;
            (void)tensorPlacement;
            THOR_THROW_IF_FALSE(raggedRawLossTensor.has_value());
            // Inputs and the managed active-count carrier are owned by their
            // upstream producers. The sparse CE layer owns raw-loss storage and,
            // for training, a dense logits-gradient buffer with prediction geometry.
            return raggedRawLossTensor->getValues().getTotalSizeInBytes() +
                   raggedPredictionsTensor->getValues().getTotalSizeInBytes();
        }
        return Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }
};

class CategoricalCrossEntropy::Builder {
   public:
    CategoricalCrossEntropy build() {
        CategoricalCrossEntropy categoricalCrossEntropy;
        populateAndAdd(categoricalCrossEntropy, LabelType::DENSE, std::nullopt);
        return categoricalCrossEntropy;
    }

    virtual CategoricalCrossEntropy::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &predictions(RaggedTensor predictions) {
        THOR_THROW_IF_FALSE(!this->_raggedPredictions.has_value());
        THOR_THROW_IF_FALSE(predictions.isInitialized());
        this->_raggedPredictions = std::move(predictions);
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    /**
     * Reports loss to the user as a single scalar that represents the total loss of the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     * Batch [b][c] -> [1]
     * Per-output [b][c] -> [c]
     * Per-example [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    /**
     * Reports loss to the user as one value per output position, averaged across the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     * Batch [b][c] -> [1]
     * Per-output [b][c] -> [c]
     * Per-example [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per example in the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     * Batch [b][c] -> [1]
     * Per-output [b][c] -> [c]
     * Per-example [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    /**
     * Does not expose a reported loss tensor. The raw loss remains available internally as the training objective.
     */
    virtual CategoricalCrossEntropy::Builder &reportsNoLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    /**
     * Reports loss to the user in its raw form: one scalar per class per example in the batch.
     * Batch [b][c] -> [1]
     * Per-output [b][c] -> [c]
     * Per-example [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP32 || _lossDataType == DataType::FP16);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   protected:

    static bool sparseLabelDimensionsMatchPredictionPrefix(const std::vector<uint64_t>& predictionDimensions,
                                                           const std::vector<uint64_t>& labelDimensions) {
        THOR_THROW_IF_FALSE(!predictionDimensions.empty());
        const size_t prefixRank = predictionDimensions.size() - 1;
        if (prefixRank == 0) {
            return labelDimensions.size() == 1 && labelDimensions[0] == 1;
        }
        if (labelDimensions.size() == prefixRank) {
            for (size_t i = 0; i < prefixRank; ++i) {
                if (labelDimensions[i] != predictionDimensions[i])
                    return false;
            }
            return true;
        }
        if (labelDimensions.size() == prefixRank + 1 && labelDimensions.back() == 1) {
            for (size_t i = 0; i < prefixRank; ++i) {
                if (labelDimensions[i] != predictionDimensions[i])
                    return false;
            }
            return true;
        }
        return false;
    }

    static bool sparseRaggedValueIsScalar(const std::vector<uint64_t>& trailingDimensions) {
        return trailingDimensions.empty() ||
               (trailingDimensions.size() == 1 && trailingDimensions.front() == 1);
    }

    void populateAndAdd(CategoricalCrossEntropy &categoricalCrossEntropy,
                        LabelType labelType,
                        std::optional<uint32_t> sparseNumClasses) {
        THOR_THROW_IF_FALSE(_network.has_value());
        const bool hasDensePredictions = _predictions.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedPredictions = _raggedPredictions.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDensePredictions == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedPredictions == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDensePredictions != hasRaggedPredictions);
        THOR_THROW_IF_FALSE(labelType == LabelType::SPARSE || labelType == LabelType::DENSE);
        if (hasDensePredictions && _raggedMask.has_value())
            throw std::invalid_argument("SparseCategoricalCrossEntropy dense predictions require a dense mask tensor.");
        if (hasRaggedPredictions && _mask.has_value())
            throw std::invalid_argument("SparseCategoricalCrossEntropy ragged predictions require a ragged mask tensor.");
        std::vector<uint64_t> predictionDimensions;
        std::vector<uint64_t> labelDimensions;
        if (hasDensePredictions) {
            THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
            // API layer does not have a batch dimension. The final prediction
            // dimension is the class dimension; any preceding dimensions are
            // per-example positions (for example, LM tokens).
            predictionDimensions = _predictions.value().getDimensions();
            labelDimensions = _labels.value().getDimensions();
            THOR_THROW_IF_FALSE(!predictionDimensions.empty() && predictionDimensions.back() > 1);
        } else {
            const RaggedTensor& predictions = _raggedPredictions.value();
            const RaggedTensor& labels = _raggedLabels.value();
            THOR_THROW_IF_FALSE(predictions.isInitialized() && labels.isInitialized());
            THOR_THROW_IF_FALSE(predictions.getValues() != labels.getValues());
            predictionDimensions = predictions.getTrailingDimensions();
            labelDimensions = labels.getTrailingDimensions();
            if (predictionDimensions.size() != 1 || predictionDimensions.back() <= 1)
                throw std::invalid_argument(
                    labelType == LabelType::SPARSE
                        ? "SparseCategoricalCrossEntropy ragged predictions must have exactly one trailing class dimension greater than one."
                        : "CategoricalCrossEntropy ragged predictions must have exactly one trailing class dimension greater than one.");
            if (!predictions.sharesPartitionWith(labels))
                throw std::invalid_argument(labelType == LabelType::SPARSE
                                                ? "SparseCategoricalCrossEntropy ragged predictions and labels must use the exact same row partition."
                                                : "CategoricalCrossEntropy ragged predictions and labels must use the exact same row partition.");
            if (predictions.getBatchSize() != labels.getBatchSize() ||
                predictions.getMaxTotalValues() != labels.getMaxTotalValues())
                throw std::invalid_argument(labelType == LabelType::SPARSE
                                                ? "SparseCategoricalCrossEntropy ragged predictions and labels must have the same batch size and packed capacity."
                                                : "CategoricalCrossEntropy ragged predictions and labels must have identical value geometry.");
            if (labelType == LabelType::DENSE && predictionDimensions != labelDimensions)
                throw std::invalid_argument("CategoricalCrossEntropy ragged predictions and labels must have identical value geometry.");
        }

        if (labelType == LabelType::DENSE) {
            THOR_THROW_IF_FALSE(predictionDimensions == labelDimensions);
            categoricalCrossEntropy.numClasses = predictionDimensions.back();
        } else {
            if (hasDensePredictions) {
                THOR_THROW_IF_FALSE(sparseLabelDimensionsMatchPredictionPrefix(predictionDimensions, labelDimensions));
            } else if (!sparseRaggedValueIsScalar(labelDimensions)) {
                throw std::invalid_argument(
                    "SparseCategoricalCrossEntropy ragged labels must be scalar per active token (trailing shape [] or [1]).");
            }
            DataType labelsDataType = hasDensePredictions ? _labels.value().getDataType() : _raggedLabels->getValuesDataType();
            THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                                labelsDataType == DataType::UINT32);
            THOR_THROW_IF_FALSE(sparseNumClasses.has_value());
            THOR_THROW_IF_FALSE(sparseNumClasses.value() > 1);
            THOR_THROW_IF_FALSE(predictionDimensions.back() == sparseNumClasses.value());
            if (hasDensePredictions && _mask.has_value()) {
                THOR_THROW_IF_FALSE(sparseLabelDimensionsMatchPredictionPrefix(predictionDimensions, _mask.value().getDimensions()));
                DataType maskDataType = _mask.value().getDataType();
                THOR_THROW_IF_FALSE(maskDataType == DataType::BOOLEAN || maskDataType == DataType::UINT8 ||
                                    maskDataType == DataType::FP16 || maskDataType == DataType::FP32);
            } else if (hasRaggedPredictions && _raggedMask.has_value()) {
                const RaggedTensor& predictions = _raggedPredictions.value();
                const RaggedTensor& mask = _raggedMask.value();
                if (!predictions.sharesPartitionWith(mask))
                    throw std::invalid_argument(
                        "SparseCategoricalCrossEntropy ragged predictions and mask must use the exact same row partition.");
                if (predictions.getBatchSize() != mask.getBatchSize() ||
                    predictions.getMaxTotalValues() != mask.getMaxTotalValues())
                    throw std::invalid_argument(
                        "SparseCategoricalCrossEntropy ragged predictions and mask must have the same batch size and packed capacity.");
                if (!sparseRaggedValueIsScalar(mask.getTrailingDimensions()))
                    throw std::invalid_argument(
                        "SparseCategoricalCrossEntropy ragged mask must be scalar per active token (trailing shape [] or [1]).");
                DataType maskDataType = mask.getValuesDataType();
                THOR_THROW_IF_FALSE(maskDataType == DataType::BOOLEAN || maskDataType == DataType::UINT8 ||
                                    maskDataType == DataType::FP16 || maskDataType == DataType::FP32);
            }
            categoricalCrossEntropy.numClasses = sparseNumClasses.value();
        }

        categoricalCrossEntropy.softmaxAddedToNetwork = _softmaxAddedToNetwork.value_or(false);
        categoricalCrossEntropy.logitsNativeLossAddedToNetwork = _logitsNativeLossAddedToNetwork.value_or(false);
        categoricalCrossEntropy.ignoreIndex = _ignoreIndex;
        categoricalCrossEntropy.maskTensor = _mask;
        if (hasDensePredictions) {
            categoricalCrossEntropy.predictionsTensor = _predictions.value();
            categoricalCrossEntropy.labelsTensor = _labels.value();
        } else {
            const RaggedTensor& predictions = _raggedPredictions.value();
            const RaggedTensor& labels = _raggedLabels.value();
            categoricalCrossEntropy.predictionsTensor = predictions.getValues();
            categoricalCrossEntropy.labelsTensor = labels.getValues();
            categoricalCrossEntropy.raggedPredictionsTensor = predictions;
            categoricalCrossEntropy.raggedLabelsTensor = labels;
            if (_raggedMask.has_value()) {
                categoricalCrossEntropy.raggedMaskTensor = _raggedMask.value();
                categoricalCrossEntropy.maskTensor = _raggedMask->getValues();
            }
            if (_lossShape.value_or(LossShape::BATCH) == LossShape::PER_OUTPUT)
                throw std::invalid_argument(labelType == LabelType::SPARSE
                                                ? "SparseCategoricalCrossEntropy LossShape::PER_OUTPUT is undefined for ragged sequences."
                                                : "CategoricalCrossEntropy LossShape::PER_OUTPUT is undefined for ragged sequences.");
        }
        if (categoricalCrossEntropy.softmaxAddedToNetwork || categoricalCrossEntropy.logitsNativeLossAddedToNetwork)
            categoricalCrossEntropy.softmaxOutput = categoricalCrossEntropy.predictionsTensor;
        if (!_lossDataType.has_value()) _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        categoricalCrossEntropy.lossDataType = _lossDataType.value();

        categoricalCrossEntropy.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);

        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;
        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::NONE || _lossShape.value() == LossShape::BATCH ||
                            _lossShape.value() == LossShape::PER_OUTPUT || _lossShape.value() == LossShape::PER_EXAMPLE ||
                            _lossShape.value() == LossShape::RAW);
        categoricalCrossEntropy.lossShape = _lossShape.value();
        categoricalCrossEntropy.labelType = labelType;
        categoricalCrossEntropy.initialized = true;
        categoricalCrossEntropy.network = _network.value();

        if (categoricalCrossEntropy.isMultiLayer()) {
            categoricalCrossEntropy.buildSupportLayersAndAddToNetwork();
        } else {
            THOR_THROW_IF_FALSE(categoricalCrossEntropy.lossShape == LossShape::RAW);
            std::vector<uint64_t> rawLossDimensions = categoricalCrossEntropy.isRagged()
                ? std::vector<uint64_t>{categoricalCrossEntropy.raggedPredictionsTensor->getMaxTotalValues()}
                : predictionDimensions;
            if (categoricalCrossEntropy.labelType == LabelType::SPARSE && categoricalCrossEntropy.logitsNativeLossAddedToNetwork) {
                if (categoricalCrossEntropy.isRagged()) {
                    // One scalar loss per packed token. The leading packed-capacity
                    // dimension is already physical and must not be batch-expanded.
                } else if (rawLossDimensions.size() == 1)
                    rawLossDimensions = {1};
                else
                    rawLossDimensions.pop_back();
            }
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType.value(), rawLossDimensions);
            categoricalCrossEntropy.lossShaperInput = categoricalCrossEntropy.lossTensor;
            if (categoricalCrossEntropy.isRagged())
                categoricalCrossEntropy.raggedRawLossTensor =
                    categoricalCrossEntropy.raggedPredictionsTensor->withValues(categoricalCrossEntropy.lossTensor);
            categoricalCrossEntropy.addToNetwork(_network.value());
        }
    }

    /**
     * CategoricalCrossEntropy is a softmax activation followed by a cross entropy loss.
     * During multi-layer flattening this flag marks the internal raw cross-entropy layer whose input is already softmax output.
     */
    virtual CategoricalCrossEntropy::Builder &softmaxAddedToNetwork() {
        THOR_THROW_IF_FALSE(!_softmaxAddedToNetwork.has_value());
        _softmaxAddedToNetwork = true;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &logitsNativeLossAddedToNetwork() {
        THOR_THROW_IF_FALSE(!_logitsNativeLossAddedToNetwork.has_value());
        _logitsNativeLossAddedToNetwork = true;
        return *this;
    }

    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<RaggedTensor> _raggedPredictions;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<RaggedTensor> _raggedMask;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<bool> _softmaxAddedToNetwork;
    std::optional<bool> _logitsNativeLossAddedToNetwork;
    std::optional<uint32_t> _ignoreIndex;
    std::optional<Tensor> _mask;

    friend class CategoricalCrossEntropy;
    friend class SparseCategoricalCrossEntropy;
};

class SparseCategoricalCrossEntropy : public CategoricalCrossEntropy {
   public:
    class Builder;
    SparseCategoricalCrossEntropy() {}

    ~SparseCategoricalCrossEntropy() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<SparseCategoricalCrossEntropy>(*this); }

    std::string getLayerType() const override { return "SparseCategoricalCrossEntropy"; }

    static void deserialize(const nlohmann::json &j, Network *network);
};

class SparseCategoricalCrossEntropy::Builder : public CategoricalCrossEntropy::Builder {
   public:
    SparseCategoricalCrossEntropy build() {
        SparseCategoricalCrossEntropy sparseCategoricalCrossEntropy;
        populateAndAdd(sparseCategoricalCrossEntropy, LabelType::SPARSE, _numClasses);
        return sparseCategoricalCrossEntropy;
    }

    virtual SparseCategoricalCrossEntropy::Builder &network(Network &_network) {
        CategoricalCrossEntropy::Builder::network(_network);
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &predictions(Tensor _predictions) {
        CategoricalCrossEntropy::Builder::predictions(_predictions);
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &predictions(RaggedTensor _predictions) {
        CategoricalCrossEntropy::Builder::predictions(std::move(_predictions));
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &labels(Tensor _labels) {
        CategoricalCrossEntropy::Builder::labels(_labels);
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &labels(RaggedTensor _labels) {
        CategoricalCrossEntropy::Builder::labels(std::move(_labels));
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &numClasses(uint32_t _numClasses) {
        THOR_THROW_IF_FALSE(!this->_numClasses.has_value());
        THOR_THROW_IF_FALSE(_numClasses > 1);
        this->_numClasses = _numClasses;
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &ignoreIndex(uint32_t _ignoreIndex) {
        THOR_THROW_IF_FALSE(!this->_ignoreIndex.has_value());
        this->_ignoreIndex = _ignoreIndex;
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &mask(Tensor _mask) {
        THOR_THROW_IF_FALSE(!this->_raggedMask.has_value());
        THOR_THROW_IF_FALSE(!this->_mask.has_value());
        THOR_THROW_IF_FALSE(!_mask.getDimensions().empty());
        this->_mask = _mask;
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &mask(RaggedTensor _mask) {
        THOR_THROW_IF_FALSE(!this->_mask.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedMask.has_value());
        THOR_THROW_IF_FALSE(_mask.isInitialized());
        this->_raggedMask = std::move(_mask);
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsBatchLoss() {
        CategoricalCrossEntropy::Builder::reportsBatchLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsPerOutputLoss() {
        CategoricalCrossEntropy::Builder::reportsPerOutputLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsPerExampleLoss() {
        CategoricalCrossEntropy::Builder::reportsPerExampleLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsNoLoss() {
        CategoricalCrossEntropy::Builder::reportsNoLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsRawLoss() {
        CategoricalCrossEntropy::Builder::reportsRawLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &lossDataType(DataType _lossDataType) {
        CategoricalCrossEntropy::Builder::lossDataType(_lossDataType);
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &lossWeight(float lossWeight) {
        CategoricalCrossEntropy::Builder::lossWeight(lossWeight);
        return *this;
    }

   protected:
    virtual SparseCategoricalCrossEntropy::Builder &softmaxAddedToNetwork() {
        CategoricalCrossEntropy::Builder::softmaxAddedToNetwork();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &logitsNativeLossAddedToNetwork() {
        CategoricalCrossEntropy::Builder::logitsNativeLossAddedToNetwork();
        return *this;
    }

   private:
    std::optional<uint32_t> _numClasses;

    friend class CategoricalCrossEntropy;
    friend class SparseCategoricalCrossEntropy;
};

}  // namespace Thor
