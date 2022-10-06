#pragma once

#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

namespace Thor {

class BinaryCrossEntropy : public Loss {
   public:
    class Builder;
    BinaryCrossEntropy() {}

    virtual ~BinaryCrossEntropy() {}

    virtual bool isMultiLayer() const {
        if (lossType == ThorImplementation::Loss::LossType::RAW && softmaxRemoved)
            return false;
        return true;
    }

    virtual void convertToSingleLayersAndAddToNetwork();

    virtual shared_ptr<Layer> clone() const { return make_shared<BinaryCrossEntropy>(*this); }

    virtual string getLayerType() const { return "BinaryCrossEntropy"; }

   private:
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
        ThorImplementation::CrossEntropy *crossEntropy = new ThorImplementation::CrossEntropy(CrossEntropyLossType::BINARY);
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
    bool softmaxRemoved;
};

class BinaryCrossEntropy::Builder {
   public:
    virtual BinaryCrossEntropy build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_labels.isPresent());
        assert(_featureInput.get() != _labels.get());
        assert(_lossType.isPresent());

        BinaryCrossEntropy binaryCrossEntropy;
        if (_softmaxRemoved.isPresent() && _softmaxRemoved.get() == true)
            binaryCrossEntropy.softmaxRemoved = true;
        else
            binaryCrossEntropy.softmaxRemoved = false;
        binaryCrossEntropy.featureInput = _featureInput;
        binaryCrossEntropy.labelsTensor = _labels;
        binaryCrossEntropy.predictionsTensor = _featureInput.get().clone(Tensor::DataType::FP32);
        if (_lossType == LossType::BATCH) {
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::BATCH;
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {1});
        } else if (_lossType == LossType::CLASSWISE) {
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::CLASSWISE;
            uint32_t batchSize = _featureInput.get().getDimensions()[0];
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {batchSize});
        } else {
            assert(_lossType == LossType::RAW);
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::RAW;
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, _featureInput.get().getDimensions());
        }
        binaryCrossEntropy.initialized = true;
        binaryCrossEntropy.network = _network;
        binaryCrossEntropy.addToNetwork(_network.get());
        return binaryCrossEntropy;
    }

    virtual BinaryCrossEntropy::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &predictions(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    /**
     * BinaryCrossEntropy is a softmax activation followed by a cross entropy loss.
     * If for whatever reason the softmax is not wanted it can be removed so only the cross entropy loss function will be applied.
     */
    virtual BinaryCrossEntropy::Builder &removeSoftmax() {
        assert(!_softmaxRemoved.isPresent());
        _softmaxRemoved = true;
        return *this;
    }

    /**
     * Reports loss to the user as a single scalar that represents the total loss of the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsBatchLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::BATCH;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per class that indicates the loss attributed to that class across the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsClasswiseLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::CLASSWISE;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per class per example in the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsElementwiseLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::ELEMENTWISE;
        return *this;
    }

    /**
     * Reports loss to the user in its raw form: one scalar per class per example in the batch.
     */
    virtual BinaryCrossEntropy::Builder &reportsRawLoss() {
        assert(!_lossType.isPresent());
        _lossType = LossType::RAW;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<Tensor> _labels;
    Optional<LossType> _lossType;
    Optional<bool> _softmaxRemoved;
};

}  // namespace Thor
