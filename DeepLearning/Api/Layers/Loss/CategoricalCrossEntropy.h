#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"
#include <optional>

namespace Thor {

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
        // FIXME: How to prune backward then.
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        // Softmax and LossShaper are connected during multi-layer flattening
        std::shared_ptr<ThorImplementation::CrossEntropy> crossEntropy = std::make_shared<ThorImplementation::CrossEntropy>(
            CrossEntropyLossType::CATEGORICAL, lossDataType, labelType == LabelType::INDEX);
        return crossEntropy;
    }

    LabelType labelType;
    uint32_t numClasses;
    bool softmaxAddedToNetwork;
    Tensor softmaxOutput;
};

class CategoricalCrossEntropy::Builder {
   public:
    virtual CategoricalCrossEntropy build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        // API layer does not have a batch dimension:
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_labelType.has_value());
        THOR_THROW_IF_FALSE(_labelType.value() == LabelType::INDEX || _labelType.value() == LabelType::ONE_HOT);

        CategoricalCrossEntropy categoricalCrossEntropy;
        if (_labelType.value() == LabelType::ONE_HOT) {
            std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] > 1);
            THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == labelDimensions);
            std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
            THOR_THROW_IF_FALSE(predictionDimensions.size() == 1);
            categoricalCrossEntropy.numClasses = predictionDimensions[0];
        } else {
            std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
            std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);
            Tensor::DataType labelsDataType = _labels.value().getDataType();
            THOR_THROW_IF_FALSE(labelsDataType == Tensor::DataType::UINT8 || labelsDataType == Tensor::DataType::UINT16 ||
                   labelsDataType == Tensor::DataType::UINT32);
            THOR_THROW_IF_FALSE(_numClasses.has_value());
            THOR_THROW_IF_FALSE(predictionDimensions.size() == 1);
            categoricalCrossEntropy.numClasses = _numClasses.value();
        }

        if (_softmaxAddedToNetwork.has_value()) {
            THOR_THROW_IF_FALSE(_softmaxAddedToNetwork.value() == true);
            categoricalCrossEntropy.softmaxAddedToNetwork = true;
        } else {
            categoricalCrossEntropy.softmaxAddedToNetwork = false;
        }
        categoricalCrossEntropy.predictionsTensor = _predictions.value();
        categoricalCrossEntropy.labelsTensor = _labels.value();
        if (!_lossDataType.has_value())
            _lossDataType = Tensor::DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == Tensor::DataType::FP16 || _lossDataType.value() == Tensor::DataType::FP32);
        categoricalCrossEntropy.lossDataType = _lossDataType.value();

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (_lossShape.value() == LossShape::BATCH) {
            categoricalCrossEntropy.lossShape = LossShape::BATCH;
        } else if (_lossShape.value() == LossShape::CLASSWISE) {
            // This type is batch-reduced by the implemenation layer
            categoricalCrossEntropy.lossShape = LossShape::CLASSWISE;
        } else if (_lossShape.value() == LossShape::ELEMENTWISE) {
            categoricalCrossEntropy.lossShape = LossShape::ELEMENTWISE;
        } else {
            // This type is *not* batch-reduced by the implemenation layer
            THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::RAW);
            categoricalCrossEntropy.lossShape = LossShape::RAW;
        }
        categoricalCrossEntropy.labelType = _labelType.value();
        categoricalCrossEntropy.initialized = true;
        categoricalCrossEntropy.network = _network.value();

        if (categoricalCrossEntropy.isMultiLayer()) {
            categoricalCrossEntropy.buildSupportLayersAndAddToNetwork();
        } else {
            THOR_THROW_IF_FALSE(categoricalCrossEntropy.lossShape == LossShape::RAW);
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType.value(), {categoricalCrossEntropy.numClasses});
            categoricalCrossEntropy.addToNetwork(_network.value());
        }

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

    virtual CategoricalCrossEntropy::Builder &lossDataType(Tensor::DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == Tensor::DataType::FP32 || _lossDataType == Tensor::DataType::FP16);
        this->_lossDataType = _lossDataType;
        return *this;
    }

    /*
     * A numerical index is passed as the label. The value of the label is the number of the true class.
     * One number is passed per item in the batch.
     * Soft labels are not supported in this case.
     */
    virtual CategoricalCrossEntropy::Builder &receivesClassIndexLabels(uint32_t numClasses) {
        THOR_THROW_IF_FALSE(!_labelType.has_value());
        THOR_THROW_IF_FALSE(numClasses > 1);
        _labelType = LabelType::INDEX;
        this->_numClasses = numClasses;
        return *this;
    }

    /**
     * A vector of labels. One label per class per example in the batch.
     * The label can be a one-hot vector, but soft labels are also supported,
     * so for example two classes may both have a label of 0.5.
     */
    virtual CategoricalCrossEntropy::Builder &receivesOneHotLabels() {
        THOR_THROW_IF_FALSE(!_labelType.has_value());
        _labelType = LabelType::ONE_HOT;
        return *this;
    }

   protected:
    /**
     * CategoricalCrossEntropy is a softmax activation followed by a cross entropy loss.
     * When the layer is stamped, an external softmax will also be stamped and this will be recorded so that next attempt to stamp will
     * result in a single layer that can be stamped.
     */
    virtual CategoricalCrossEntropy::Builder &softmaxAddedToNetwork() {
        THOR_THROW_IF_FALSE(!_softmaxAddedToNetwork.has_value());
        _softmaxAddedToNetwork = true;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<LabelType> _labelType;
    std::optional<uint32_t> _numClasses;
    std::optional<LossShape> _lossShape;
    std::optional<Tensor::DataType> _lossDataType;
    std::optional<bool> _softmaxAddedToNetwork;

    friend class CategoricalCrossEntropy;
};

NLOHMANN_JSON_SERIALIZE_ENUM(CategoricalCrossEntropy::LabelType,
                             {
                                 {CategoricalCrossEntropy::LabelType::INDEX, "index"},
                                 {CategoricalCrossEntropy::LabelType::ONE_HOT, "one_hot"},
                             })

}  // namespace Thor
