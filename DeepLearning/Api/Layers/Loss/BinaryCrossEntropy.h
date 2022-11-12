#pragma once

#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Activation/Sigmoid.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/CrossEntropy.h"

namespace Thor {

class BinaryCrossEntropy : public Loss {
   public:
    class Builder;
    BinaryCrossEntropy() {}

    virtual ~BinaryCrossEntropy() {}

    virtual bool isMultiLayer() const {
        if (lossType != ThorImplementation::Loss::LossType::RAW || !sigmoidStamped)
            return true;
        return false;
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

    Network *network;
    bool sigmoidStamped;
};

class BinaryCrossEntropy::Builder {
   public:
    virtual BinaryCrossEntropy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_lossType.isPresent());

        vector<uint64_t> labelDimensions = _labels.get().getDimensions();
        assert(labelDimensions.size() == 1);
        assert(labelDimensions[0] == _predictions.get().getDimensions()[0]);

        BinaryCrossEntropy binaryCrossEntropy;
        if (_sigmoidStamped.isPresent()) {
            assert(_sigmoidStamped.get() == true);
            binaryCrossEntropy.sigmoidStamped = true;
        } else {
            binaryCrossEntropy.sigmoidStamped = false;
        }
        binaryCrossEntropy.predictionsTensor = _predictions;
        binaryCrossEntropy.labelsTensor = _labels;
        binaryCrossEntropy.predictionsTensor = _predictions.get().clone(Tensor::DataType::FP32);
        if (_lossType == LossType::BATCH) {
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::BATCH;
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {1});
        } else if (_lossType == LossType::CLASSWISE) {
            // FIXME: For this and CCE index labels, I will need to do something special
            // One solution is to convert whatever the user sends to CCE one hot labels, think I should do that.
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::CLASSWISE;
            uint32_t numClasses = 2;
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {numClasses});
        } else if (_lossType == LossType::ELEMENTWISE) {
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::ELEMENTWISE;
            uint32_t batchSize = _predictions.get().getDimensions()[0];
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, {batchSize});
        } else {
            assert(_lossType == LossType::RAW);
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::RAW;
            binaryCrossEntropy.lossTensor = Tensor(Tensor::DataType::FP32, _predictions.get().getDimensions());
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

    virtual BinaryCrossEntropy::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
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

   protected:
    /**
     * BinaryCrossEntropy is a sigmoid activation followed by a cross entropy loss.
     * When the layer is stamped an external sigmoid will also be stamped and this will be recorded so that next attempt to stamp will
     * result in a single layer that can be stamped.
     */
    virtual BinaryCrossEntropy::Builder &sigmoidStamped() {
        assert(!_sigmoidStamped.isPresent());
        _sigmoidStamped = true;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<LossType> _lossType;
    Optional<bool> _sigmoidStamped;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
