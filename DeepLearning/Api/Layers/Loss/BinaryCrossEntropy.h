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

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<BinaryCrossEntropy>(*this); }

    virtual std::string getLayerType() const { return "BinaryCrossEntropy"; }

   private:
    // Binary cross entropy only supports batch and elementwise loss.
    // For classwise loss or raw loss use CategoricalCrossEntropy with 2 classes instead of BinaryCrossEntropy
    enum class LossType { BATCH = 9, ELEMENTWISE };

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        // Softmax and LossShaper are connected during multi-layer flattening
        ThorImplementation::CrossEntropy *crossEntropy = new ThorImplementation::CrossEntropy(CrossEntropyLossType::BINARY);
        Thor::Layer::connectTwoLayers(drivingLayer, crossEntropy, drivingApiLayer, this, connectingApiTensor);
        return crossEntropy;
    }

    virtual bool isMultiLayer() const {
        if (lossType != ThorImplementation::Loss::LossType::ELEMENTWISE || !sigmoidStamped)
            return true;
        return false;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    Network *network;
    bool sigmoidStamped;
    Tensor::DataType lossDataType;
};

class BinaryCrossEntropy::Builder {
   public:
    virtual BinaryCrossEntropy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_lossType.isPresent());

        std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
        // API layer does not have a batch dimension:
        assert(labelDimensions.size() == 1 && labelDimensions[0] == 1);

        BinaryCrossEntropy binaryCrossEntropy;
        if (_sigmoidStamped.isPresent()) {
            assert(_sigmoidStamped.get() == true);
            binaryCrossEntropy.sigmoidStamped = true;
        } else {
            binaryCrossEntropy.sigmoidStamped = false;
        }
        binaryCrossEntropy.predictionsTensor = _predictions.get();
        binaryCrossEntropy.labelsTensor = _labels.get();
        if (_lossDataType.isEmpty())
            _lossDataType = Tensor::DataType::FP32;
        assert(_lossDataType == Tensor::DataType::FP16 || _lossDataType == Tensor::DataType::FP32);
        binaryCrossEntropy.lossDataType = _lossDataType;

        if (_lossType == LossType::BATCH) {
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::BATCH;
        } else if (_lossType == LossType::ELEMENTWISE) {
            binaryCrossEntropy.lossType = ThorImplementation::Loss::LossType::ELEMENTWISE;
        }
        binaryCrossEntropy.initialized = true;
        binaryCrossEntropy.network = _network;

        if (binaryCrossEntropy.isMultiLayer()) {
            binaryCrossEntropy.buildSupportLayersAndAddToNetwork();
        } else {
            assert(binaryCrossEntropy.lossType == ThorImplementation::Loss::LossType::ELEMENTWISE);
            binaryCrossEntropy.lossTensor = Tensor(_lossDataType, {1});
            binaryCrossEntropy.addToNetwork(_network.get());
        }

        return binaryCrossEntropy;
    }

    virtual BinaryCrossEntropy::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(_predictions.getDimensions().size() == 1 && _predictions.getDimensions()[0] == 1);
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(_labels.getDimensions().size() == 1 && _labels.getDimensions()[0] == 1);
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

    virtual BinaryCrossEntropy::Builder &lossDataType(Tensor::DataType _lossDataType) {
        assert(this->_lossDataType.isEmpty());
        assert(_lossDataType == Tensor::DataType::FP32 || _lossDataType == Tensor::DataType::FP16);
        this->_lossDataType = _lossDataType;
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
    Optional<Tensor::DataType> _lossDataType;
    Optional<bool> _sigmoidStamped;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
