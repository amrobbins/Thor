#pragma once

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

namespace Thor {

class CategoricalCrossEntropy : public Loss {
   public:
    class Builder;
    CategoricalCrossEntropy() {}

    virtual ~CategoricalCrossEntropy() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<CategoricalCrossEntropy>(*this); }

    virtual std::string getLayerType() const { return "CategoricalCrossEntropy"; }

    virtual Tensor getPredictions() const { return softmaxOutput; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual bool isMultiLayer() const {
        if (lossType != LossType::RAW || !softmaxStamped)
            return true;
        return false;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        // Softmax and LossShaper are connected during multi-layer flattening
        std::shared_ptr<ThorImplementation::CrossEntropy> crossEntropy = std::make_shared<ThorImplementation::CrossEntropy>(
            CrossEntropyLossType::CATEGORICAL, Tensor::convertToImplementationDataType(lossDataType), labelType == LabelType::INDEX);
        return crossEntropy;
    }

    Network *network;
    LabelType labelType;
    Tensor::DataType lossDataType;
    uint32_t numClasses;
    bool softmaxStamped;
    Tensor softmaxOutput;
};

class CategoricalCrossEntropy::Builder {
   public:
    virtual CategoricalCrossEntropy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        // API layer does not have a batch dimension:
        assert(_predictions.get().getDimensions().size() == 1);
        assert(_labelType.isPresent());
        assert(_labelType == LabelType::INDEX || _labelType == LabelType::ONE_HOT);

        CategoricalCrossEntropy categoricalCrossEntropy;
        if (_labelType == LabelType::ONE_HOT) {
            std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            assert(labelDimensions.size() == 1 && labelDimensions[0] > 1);
            assert(_predictions.get().getDimensions() == _labels.get().getDimensions());
            categoricalCrossEntropy.numClasses = _predictions.get().getDimensions()[0];
        } else {
            std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            std::vector<uint64_t> predictionDimensions = _predictions.get().getDimensions();
            assert(labelDimensions.size() == 1 && labelDimensions[0] == 1);
            Tensor::DataType labelsDataType = _labels.get().getDataType();
            assert(labelsDataType == Tensor::DataType::UINT8 || labelsDataType == Tensor::DataType::UINT16 ||
                   labelsDataType == Tensor::DataType::UINT32);
            assert(_numClasses.isPresent());
            assert(predictionDimensions.size() == 1 && predictionDimensions[0] == _numClasses);
            categoricalCrossEntropy.numClasses = _numClasses;
        }

        if (_softmaxStamped.isPresent()) {
            assert(_softmaxStamped.get() == true);
            categoricalCrossEntropy.softmaxStamped = true;
        } else {
            categoricalCrossEntropy.softmaxStamped = false;
        }
        categoricalCrossEntropy.predictionsTensor = _predictions.get();
        categoricalCrossEntropy.labelsTensor = _labels.get();
        if (_lossDataType.isEmpty())
            _lossDataType = Tensor::DataType::FP32;
        assert(_lossDataType == Tensor::DataType::FP16 || _lossDataType == Tensor::DataType::FP32);
        categoricalCrossEntropy.lossDataType = _lossDataType;

        if (_lossType.isEmpty())
            _lossType = LossType::BATCH;
        if (_lossType == LossType::BATCH) {
            categoricalCrossEntropy.lossType = LossType::BATCH;
        } else if (_lossType == LossType::CLASSWISE) {
            // This type is batch-reduced by the implemenation layer
            categoricalCrossEntropy.lossType = LossType::CLASSWISE;
        } else if (_lossType == LossType::ELEMENTWISE) {
            categoricalCrossEntropy.lossType = LossType::ELEMENTWISE;
        } else {
            // This type is *not* batch-reduced by the implemenation layer
            assert(_lossType == LossType::RAW);
            categoricalCrossEntropy.lossType = LossType::RAW;
        }
        categoricalCrossEntropy.labelType = _labelType;
        categoricalCrossEntropy.initialized = true;
        categoricalCrossEntropy.network = _network;

        if (categoricalCrossEntropy.isMultiLayer()) {
            categoricalCrossEntropy.buildSupportLayersAndAddToNetwork();
        } else {
            assert(categoricalCrossEntropy.lossType == LossType::RAW);
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType, {categoricalCrossEntropy.numClasses});
            categoricalCrossEntropy.addToNetwork(_network.get());
        }

        return categoricalCrossEntropy;
    }

    virtual CategoricalCrossEntropy::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        assert(_labels.getDimensions().size() == 1);
        this->_labels = _labels;
        return *this;
    }

    /**
     * Reports loss to the user as a single scalar that represents the total loss of the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual CategoricalCrossEntropy::Builder &reportsBatchLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::BATCH;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per class that indicates the loss attributed to that class across the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual CategoricalCrossEntropy::Builder &reportsClasswiseLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::CLASSWISE;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per class per example in the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual CategoricalCrossEntropy::Builder &reportsElementwiseLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::ELEMENTWISE;
        return *this;
    }

    /**
     * Reports loss to the user in its raw form: one scalar per class per example in the batch.
     */
    virtual CategoricalCrossEntropy::Builder &reportsRawLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::RAW;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &lossDataType(Tensor::DataType _lossDataType) {
        assert(this->_lossDataType.isEmpty());
        assert(_lossDataType == Tensor::DataType::FP32 || _lossDataType == Tensor::DataType::FP16);
        this->_lossDataType = _lossDataType;
        return *this;
    }

    /*
     * A numerical index is passed as the label. The value of the label is the number of the true class.
     * One number is passed per item in the batch.
     * Soft labels are not supported in this case.
     */
    virtual CategoricalCrossEntropy::Builder &receivesClassIndexLabels(uint32_t numClasses) {
        assert(!_labelType.isPresent());
        assert(numClasses > 1);
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
        assert(!_labelType.isPresent());
        _labelType = LabelType::ONE_HOT;
        return *this;
    }

   protected:
    /**
     * CategoricalCrossEntropy is a softmax activation followed by a cross entropy loss.
     * When the layer is stamped, an external softmax will also be stamped and this will be recorded so that next attempt to stamp will
     * result in a single layer that can be stamped.
     */
    virtual CategoricalCrossEntropy::Builder &softmaxStamped() {
        assert(!_softmaxStamped.isPresent());
        _softmaxStamped = true;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<LabelType> _labelType;
    Optional<uint32_t> _numClasses;
    Optional<LossType> _lossType;
    Optional<Tensor::DataType> _lossDataType;
    Optional<bool> _softmaxStamped;

    friend class CategoricalCrossEntropy;
};

NLOHMANN_JSON_SERIALIZE_ENUM(CategoricalCrossEntropy::LabelType,
                             {
                                 {CategoricalCrossEntropy::LabelType::INDEX, "index"},
                                 {CategoricalCrossEntropy::LabelType::ONE_HOT, "one_hot"},
                             })

}  // namespace Thor
