#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanSquaredError.h"

namespace Thor {

class MeanSquaredError : public Loss {
   public:
    class Builder;
    MeanSquaredError() {}

    virtual ~MeanSquaredError() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<MeanSquaredError>(*this); }

    virtual string getLayerType() const { return "MeanSquaredError"; }

   protected:
    virtual bool isMultiLayer() const {
        if (lossType == ThorImplementation::Loss::LossType::RAW)
            return false;
        return true;
    }

    virtual void convertToSingleLayersAndAddToNetwork();

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        ThorImplementation::MeanSquaredError *meanSquaredError = new ThorImplementation::MeanSquaredError();
        Thor::Layer::connectTwoLayers(drivingLayer, meanSquaredError, drivingApiLayer, this, connectingApiTensor);

        return meanSquaredError;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        uint64_t lossShaperBytes = 0;
        // Loss will be reported either element-wise or batch-wise, the shaper is only required when loss is batch-wise.
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        return standardLossBytes + lossShaperBytes;
    }

    Network *network;
};

class MeanSquaredError::Builder {
   public:
    virtual MeanSquaredError build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_predictions.get().getDimensions().size() == 2);
        assert(_predictions.get().getDimensions() == _labels.get().getDimensions());

        if (_lossType.isEmpty())
            _lossType = ThorImplementation::Loss::LossType::BATCH;
        if (_lossDataType.isEmpty())
            _lossDataType = _predictions.get().getDataType();
        uint32_t batchSize = _predictions.get().getDimensions()[0];
        uint32_t numPredictions = _predictions.get().getDimensions()[1];

        MeanSquaredError meanSquaredError;
        meanSquaredError.predictionsTensor = _predictions;
        meanSquaredError.labelsTensor = _labels;
        meanSquaredError.lossType = _lossType;
        // lossTensor is the one that comes directly out of MeanSquaredError, that may be replaced by a loss shaper.
        meanSquaredError.lossTensor = Tensor(_lossDataType, {batchSize, numPredictions});
        meanSquaredError.network = _network;
        meanSquaredError.initialized = true;

        meanSquaredError.addToNetwork(_network.get());

        return meanSquaredError;
    }

    virtual MeanSquaredError::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual MeanSquaredError::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual MeanSquaredError::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsBatchLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::BATCH;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsElementwiseLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::ELEMENTWISE;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsPerOutputLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::CLASSWISE;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsRawLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::RAW;
        return *this;
    }

    virtual MeanSquaredError::Builder &lossDataType(Tensor::DataType _lossDataType) {
        assert(this->_lossDataType.isEmpty());
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<ThorImplementation::Loss::LossType> _lossType;
    Optional<Tensor::DataType> _lossDataType;
};

}  // namespace Thor
