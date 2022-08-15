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
        if (lossType == ThorImplementation::Loss::LossType::RAW && softmaxRemoved)
            return false;
        return true;
    }

    virtual void convertToSingleLayersAndAddToNetwork();

    virtual shared_ptr<Layer> clone() const { return make_shared<CategoricalCrossEntropy>(*this); }

    virtual string getLayerType() const { return "CategoricalCrossEntropy"; }

   private:
    enum class LabelType { INDEX = 5, VECTOR };
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
        ThorImplementation::CrossEntropy *crossEntropy = new ThorImplementation::CrossEntropy(labelType == LabelType::INDEX);
        Thor::Layer::connectTwoLayers(drivingLayer, crossEntropy, drivingApiLayer, this, connectingApiTensor);
        return crossEntropy;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);

        uint64_t lossWorkspaceBytes = featureInput.get().getTotalNumElements() * 4;
        uint64_t inverseSumOfExponentials = 4;  // 1 per batch item, FP32

        return standardLossBytes + batchSize * (lossWorkspaceBytes + inverseSumOfExponentials);
    }

    Network *network;
    LabelType labelType;
    bool softmaxRemoved;
};

class CategoricalCrossEntropy::Builder {
   public:
    virtual CategoricalCrossEntropy build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_labels.isPresent());
        assert(_featureInput.get() != _labels.get());
        assert(_labelType.isPresent());
        assert(_lossType.isPresent());

        CategoricalCrossEntropy categoricalCrossEntropy;
        if (_softmaxRemoved.isPresent() && _softmaxRemoved.get() == true)
            categoricalCrossEntropy.softmaxRemoved = true;
        else
            categoricalCrossEntropy.softmaxRemoved = false;
        categoricalCrossEntropy.featureInput = _featureInput;
        categoricalCrossEntropy.labelsTensor = _labels;
        categoricalCrossEntropy.predictionsTensor = _featureInput.get().clone(Tensor::DataType::FP32);
        if (_lossType == LossType::BATCH) {
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::BATCH;
            categoricalCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {1});
        } else if (_lossType == LossType::CLASSWISE) {
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::CLASSWISE;
            uint32_t batchSize = _featureInput.get().getDimensions()[0];
            categoricalCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {batchSize});
        } else {
            assert(_lossType == LossType::RAW);
            categoricalCrossEntropy.lossType = ThorImplementation::Loss::LossType::RAW;
            categoricalCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, _featureInput.get().getDimensions());
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

    virtual CategoricalCrossEntropy::Builder &predictions(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual CategoricalCrossEntropy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    /**
     * CategoricalCrossEntropy is a softmax activation followed by a cross entropy loss.
     * If for whatever reason the softmax is not wanted it can be removed so only the cross entropy loss function will be applied.
     */
    virtual CategoricalCrossEntropy::Builder &removeSoftmax() {
        assert(!_softmaxRemoved.isPresent());
        _softmaxRemoved = true;
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
    virtual CategoricalCrossEntropy::Builder &receivesPerClassLabels() {
        assert(!_labelType.isPresent());
        _labelType = LabelType::VECTOR;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<Tensor> _labels;
    Optional<LabelType> _labelType;
    Optional<LossType> _lossType;
    Optional<bool> _softmaxRemoved;
};

}  // namespace Thor
