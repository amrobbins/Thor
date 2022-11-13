#pragma once

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

namespace Thor {

class CategoricalCrossEntropy : public Loss {
   public:
    class Builder;
    CategoricalCrossEntropy() {}

    virtual ~CategoricalCrossEntropy() {}

    virtual bool isMultiLayer() const {
        if (lossType != ThorImplementation::Loss::LossType::RAW || !softmaxStamped)
            return true;
        return false;
    }

    virtual void convertToSingleLayersAndAddToNetwork();

    virtual shared_ptr<Layer> clone() const { return make_shared<CategoricalCrossEntropy>(*this); }

    virtual string getLayerType() const { return "CategoricalCrossEntropy"; }

   private:
    enum class LabelType { INDEX = 5, ONE_HOT };
    enum class LossType { BATCH = 9, ELEMENTWISE, CLASSWISE, RAW };

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        // Softmax and LossShaper are connected during multi-layer flattening
        ThorImplementation::CrossEntropy *crossEntropy =
            new ThorImplementation::CrossEntropy(CrossEntropyLossType::CATEGORICAL, labelType == LabelType::INDEX);
        Thor::Layer::connectTwoLayers(drivingLayer, crossEntropy, drivingApiLayer, this, connectingApiTensor);
        return crossEntropy;
    }

    Network *network;
    LabelType labelType;
    bool softmaxStamped;
};

class CategoricalCrossEntropy::Builder {
   public:
    virtual CategoricalCrossEntropy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_predictions.get().getDimensions().size() == 2);
        assert(_labelType.isPresent());
        assert(_labelType == LabelType::INDEX || _labelType == LabelType::ONE_HOT);
        if (_labelType == LabelType::ONE_HOT) {
            assert(_predictions.get().getDimensions() == _labels.get().getDimensions());
            vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            assert(labelDimensions.size() == 2);
            assert(labelDimensions[1] > 1);
        } else {
            vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            vector<uint64_t> predictionDimensions = _predictions.get().getDimensions();
            assert(labelDimensions.size() == 1 || labelDimensions.size() == 2);
            assert(predictionDimensions[0] == labelDimensions[0]);
            if (labelDimensions.size() == 2)
                assert(_labels.get().getDimensions()[1] == 1);
            else
                _labels.get().reshape({predictionDimensions[0], 1});
            Tensor::DataType labelsDataType = _labels.get().getDataType();
            assert(labelsDataType == Tensor::DataType::UINT8 || labelsDataType == Tensor::DataType::UINT16 ||
                   labelsDataType == Tensor::DataType::UINT32);
        }

        CategoricalCrossEntropy categoricalCrossEntropy;
        if (_softmaxStamped.isPresent()) {
            assert(_softmaxStamped.get() == true);
            categoricalCrossEntropy.softmaxStamped = true;
        } else {
            categoricalCrossEntropy.softmaxStamped = false;
        }
        categoricalCrossEntropy.predictionsTensor = _predictions.get().clone();
        categoricalCrossEntropy.labelsTensor = _labels.get().clone();
        if (_lossDataType.isEmpty()) {
            _lossDataType = Tensor::DataType::FP32;
        } else {
            assert(_lossDataType == Tensor::DataType::FP16 || _lossDataType == Tensor::DataType::FP32);
        }
        if (_lossType.isEmpty())
            _lossType = LossType::BATCH;
        if (_lossType == LossType::BATCH) {
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::BATCH;
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType, {1});
        } else if (_lossType == LossType::CLASSWISE) {
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::CLASSWISE;
            uint32_t numClasses = _predictions.get().getDimensions()[1];
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType, {numClasses});
        } else if (_lossType == LossType::ELEMENTWISE) {
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::ELEMENTWISE;
            uint32_t batchSize = _predictions.get().getDimensions()[0];
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType, {batchSize});
        } else {
            assert(_lossType == LossType::RAW);
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::RAW;
            categoricalCrossEntropy.lossTensor = Tensor(_lossDataType, _predictions.get().getDimensions());
        }
        categoricalCrossEntropy.labelType = _labelType;
        categoricalCrossEntropy.initialized = true;
        categoricalCrossEntropy.network = _network;
        categoricalCrossEntropy.addToNetwork(_network.get());
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
    virtual CategoricalCrossEntropy::Builder &receivesClassIndexLabels() {
        assert(!_labelType.isPresent());
        _labelType = LabelType::INDEX;
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
     * When the layer is stamped an external softmax will also be stamped and this will be recorded so that next attempt to stamp will
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
    Optional<LossType> _lossType;
    Optional<Tensor::DataType> _lossDataType;
    Optional<bool> _softmaxStamped;

    friend class CategoricalCrossEntropy;
};

}  // namespace Thor
