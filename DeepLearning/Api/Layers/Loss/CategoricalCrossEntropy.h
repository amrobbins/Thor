#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"
#include "DeepLearning/Implementation/Layers/Loss/SparseCategoricalCrossEntropyWithLogits.h"
#include <optional>

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
        if (maskTensor.has_value())
            return {predictionsTensor, labelsTensor, maskTensor.value()};
        return {predictionsTensor, labelsTensor};
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (maskTensor.has_value() && connectingTensor == maskTensor.value())
            return ThorImplementation::SparseCategoricalCrossEntropyWithLogits::MASK_CONNECTION_TYPE;
        return Loss::getConnectionType(connectingTensor);
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    static void deserializeInto(const nlohmann::json &j,
                                Network *network,
                                CategoricalCrossEntropy &categoricalCrossEntropy,
                                LabelType labelType,
                                const std::string &expectedLayerType);

    virtual bool isMultiLayer() const {
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
                            (maskTensor.has_value() && connectingApiTensor == maskTensor.value()));

        // Softmax and LossShaper are connected during multi-layer flattening.
        if (labelType == LabelType::SPARSE && logitsNativeLossAddedToNetwork) {
            return std::make_shared<ThorImplementation::SparseCategoricalCrossEntropyWithLogits>(
                lossDataType, lossWeight, ignoreIndex);
        }

        return std::make_shared<ThorImplementation::CrossEntropy>(
            CrossEntropyLossType::CATEGORICAL, lossDataType, labelType == LabelType::SPARSE, lossWeight);
    }

    LabelType labelType = LabelType::DENSE;
    uint32_t numClasses = 0;
    bool softmaxAddedToNetwork = false;
    bool logitsNativeLossAddedToNetwork = false;
    Tensor softmaxOutput;
    std::optional<uint32_t> ignoreIndex;
    std::optional<Tensor> maskTensor;
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

    virtual CategoricalCrossEntropy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    /**
     * Reports loss to the user as a single scalar that represents the total loss of the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     * Batch [b][c] -> [1]
     * Classwise [b][c] -> [c]
     * Elementwise [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per class that indicates the loss attributed to that class across the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     * Batch [b][c] -> [1]
     * Classwise [b][c] -> [c]
     * Elementwise [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsClasswiseLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per example in the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     * Batch [b][c] -> [1]
     * Classwise [b][c] -> [c]
     * Elementwise [b][c] -> [b]
     * Raw [b][c] -> [b][c]
     */
    virtual CategoricalCrossEntropy::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    /**
     * Reports loss to the user in its raw form: one scalar per class per example in the batch.
     * Batch [b][c] -> [1]
     * Classwise [b][c] -> [c]
     * Elementwise [b][c] -> [b]
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

    void populateAndAdd(CategoricalCrossEntropy &categoricalCrossEntropy,
                        LabelType labelType,
                        std::optional<uint32_t> sparseNumClasses) {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        // API layer does not have a batch dimension.  The final prediction
        // dimension is the class dimension; any preceding dimensions are
        // per-example positions (for example, LM tokens).
        THOR_THROW_IF_FALSE(labelType == LabelType::SPARSE || labelType == LabelType::DENSE);

        std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
        std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
        THOR_THROW_IF_FALSE(!predictionDimensions.empty() && predictionDimensions.back() > 1);

        if (labelType == LabelType::DENSE) {
            THOR_THROW_IF_FALSE(predictionDimensions == labelDimensions);
            categoricalCrossEntropy.numClasses = predictionDimensions.back();
        } else {
            THOR_THROW_IF_FALSE(sparseLabelDimensionsMatchPredictionPrefix(predictionDimensions, labelDimensions));
            DataType labelsDataType = _labels.value().getDataType();
            THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                                labelsDataType == DataType::UINT32);
            THOR_THROW_IF_FALSE(sparseNumClasses.has_value());
            THOR_THROW_IF_FALSE(sparseNumClasses.value() > 1);
            THOR_THROW_IF_FALSE(predictionDimensions.back() == sparseNumClasses.value());
            if (_mask.has_value()) {
                THOR_THROW_IF_FALSE(sparseLabelDimensionsMatchPredictionPrefix(predictionDimensions, _mask.value().getDimensions()));
                DataType maskDataType = _mask.value().getDataType();
                THOR_THROW_IF_FALSE(maskDataType == DataType::BOOLEAN || maskDataType == DataType::UINT8 ||
                                    maskDataType == DataType::FP16 || maskDataType == DataType::FP32);
            }
            categoricalCrossEntropy.numClasses = sparseNumClasses.value();
        }

        categoricalCrossEntropy.softmaxAddedToNetwork = _softmaxAddedToNetwork.value_or(false);
        categoricalCrossEntropy.logitsNativeLossAddedToNetwork = _logitsNativeLossAddedToNetwork.value_or(false);
        categoricalCrossEntropy.ignoreIndex = _ignoreIndex;
        categoricalCrossEntropy.maskTensor = _mask;
        categoricalCrossEntropy.predictionsTensor = _predictions.value();
        categoricalCrossEntropy.labelsTensor = _labels.value();
        if (categoricalCrossEntropy.softmaxAddedToNetwork || categoricalCrossEntropy.logitsNativeLossAddedToNetwork)
            categoricalCrossEntropy.softmaxOutput = categoricalCrossEntropy.predictionsTensor;
        if (!_lossDataType.has_value())
            _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        categoricalCrossEntropy.lossDataType = _lossDataType.value();

        categoricalCrossEntropy.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::BATCH || _lossShape.value() == LossShape::CLASSWISE ||
                            _lossShape.value() == LossShape::ELEMENTWISE || _lossShape.value() == LossShape::RAW);
        categoricalCrossEntropy.lossShape = _lossShape.value();
        categoricalCrossEntropy.labelType = labelType;
        categoricalCrossEntropy.initialized = true;
        categoricalCrossEntropy.network = _network.value();

        if (categoricalCrossEntropy.isMultiLayer()) {
            categoricalCrossEntropy.buildSupportLayersAndAddToNetwork();
        } else {
            THOR_THROW_IF_FALSE(categoricalCrossEntropy.lossShape == LossShape::RAW);
            std::vector<uint64_t> rawLossDimensions = predictionDimensions;
            if (categoricalCrossEntropy.labelType == LabelType::SPARSE && categoricalCrossEntropy.logitsNativeLossAddedToNetwork) {
                if (rawLossDimensions.size() == 1)
                    rawLossDimensions = {1};
                else
                    rawLossDimensions.pop_back();
            }
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType.value(), rawLossDimensions);
            categoricalCrossEntropy.lossShaperInput = categoricalCrossEntropy.lossTensor;
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

    virtual SparseCategoricalCrossEntropy::Builder &labels(Tensor _labels) {
        CategoricalCrossEntropy::Builder::labels(_labels);
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
        THOR_THROW_IF_FALSE(!this->_mask.has_value());
        THOR_THROW_IF_FALSE(!_mask.getDimensions().empty());
        this->_mask = _mask;
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsBatchLoss() {
        CategoricalCrossEntropy::Builder::reportsBatchLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsClasswiseLoss() {
        CategoricalCrossEntropy::Builder::reportsClasswiseLoss();
        return *this;
    }

    virtual SparseCategoricalCrossEntropy::Builder &reportsElementwiseLoss() {
        CategoricalCrossEntropy::Builder::reportsElementwiseLoss();
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
