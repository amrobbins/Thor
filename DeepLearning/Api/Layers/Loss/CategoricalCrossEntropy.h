#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"
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

    Tensor getPredictions() const override { return softmaxOutput; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    static void deserializeInto(const nlohmann::json &j,
                                Network *network,
                                CategoricalCrossEntropy &categoricalCrossEntropy,
                                LabelType labelType,
                                const std::string &expectedLayerType);

    virtual bool isMultiLayer() const {
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
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        // Softmax and LossShaper are connected during multi-layer flattening.
        return std::make_shared<ThorImplementation::CrossEntropy>(
            CrossEntropyLossType::CATEGORICAL, lossDataType, labelType == LabelType::SPARSE);
    }

    LabelType labelType = LabelType::DENSE;
    uint32_t numClasses = 0;
    bool softmaxAddedToNetwork = false;
    Tensor softmaxOutput;
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
        THOR_THROW_IF_FALSE(_labels.getDimensions().size() == 1);
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
    void populateAndAdd(CategoricalCrossEntropy &categoricalCrossEntropy,
                        LabelType labelType,
                        std::optional<uint32_t> sparseNumClasses) {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        // API layer does not have a batch dimension.
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(labelType == LabelType::SPARSE || labelType == LabelType::DENSE);

        std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
        std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
        THOR_THROW_IF_FALSE(predictionDimensions.size() == 1 && predictionDimensions[0] > 1);

        if (labelType == LabelType::DENSE) {
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] > 1);
            THOR_THROW_IF_FALSE(predictionDimensions == labelDimensions);
            categoricalCrossEntropy.numClasses = predictionDimensions[0];
        } else {
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);
            DataType labelsDataType = _labels.value().getDataType();
            THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                                labelsDataType == DataType::UINT32);
            THOR_THROW_IF_FALSE(sparseNumClasses.has_value());
            THOR_THROW_IF_FALSE(sparseNumClasses.value() > 1);
            THOR_THROW_IF_FALSE(predictionDimensions[0] == sparseNumClasses.value());
            categoricalCrossEntropy.numClasses = sparseNumClasses.value();
        }

        categoricalCrossEntropy.softmaxAddedToNetwork = _softmaxAddedToNetwork.value_or(false);
        categoricalCrossEntropy.predictionsTensor = _predictions.value();
        categoricalCrossEntropy.labelsTensor = _labels.value();
        if (categoricalCrossEntropy.softmaxAddedToNetwork)
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
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType.value(), {categoricalCrossEntropy.numClasses});
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

    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<bool> _softmaxAddedToNetwork;

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

   protected:
    virtual SparseCategoricalCrossEntropy::Builder &softmaxAddedToNetwork() {
        CategoricalCrossEntropy::Builder::softmaxAddedToNetwork();
        return *this;
    }

   private:
    std::optional<uint32_t> _numClasses;

    friend class CategoricalCrossEntropy;
    friend class SparseCategoricalCrossEntropy;
};

}  // namespace Thor
