#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.h"

namespace Thor {

class MeanAbsolutePercentageError : public Loss {
   public:
    class Builder;
    MeanAbsolutePercentageError() {}

    virtual ~MeanAbsolutePercentageError() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<MeanAbsolutePercentageError>(*this); }

    virtual std::string getLayerType() const { return "MeanAbsolutePercentageError"; }

   protected:
    virtual bool isMultiLayer() const {
        if (lossType == ThorImplementation::Loss::LossType::RAW)
            return false;
        return true;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        ThorImplementation::MeanAbsolutePercentageError *meanAbsolutePercentageError =
            new ThorImplementation::MeanAbsolutePercentageError();

        return meanAbsolutePercentageError;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
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

    Tensor::DataType lossDataType;
    Network *network;
};

class MeanAbsolutePercentageError::Builder {
   public:
    virtual MeanAbsolutePercentageError build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_predictions.get().getDimensions().size() == 1);
        assert(_predictions.get().getDimensions() == _labels.get().getDimensions());

        if (_lossType.isEmpty())
            _lossType = ThorImplementation::Loss::LossType::BATCH;
        if (_lossDataType.isEmpty())
            _lossDataType = _predictions.get().getDataType();
        uint32_t batchSize = _predictions.get().getDimensions()[0];

        MeanAbsolutePercentageError meanAbsolutePercentageError;
        meanAbsolutePercentageError.predictionsTensor = _predictions;
        meanAbsolutePercentageError.labelsTensor = _labels;
        meanAbsolutePercentageError.lossDataType = _lossDataType;
        meanAbsolutePercentageError.lossType = _lossType;
        meanAbsolutePercentageError.network = _network;
        meanAbsolutePercentageError.initialized = true;

        if (meanAbsolutePercentageError.isMultiLayer()) {
            meanAbsolutePercentageError.buildSupportLayersAndAddToNetwork();
        } else {
            // lossTensor is the one that comes directly out of MeanAbsolutePercentageError, that may be replaced by a loss shaper.
            meanAbsolutePercentageError.lossTensor = Tensor(_lossDataType, {batchSize});
            meanAbsolutePercentageError.addToNetwork(_network.get());
        }

        return meanAbsolutePercentageError;
    }

    virtual MeanAbsolutePercentageError::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &reportsBatchLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::BATCH;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &reportsElementwiseLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::ELEMENTWISE;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &reportsPerOutputLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::CLASSWISE;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &reportsRawLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = ThorImplementation::Loss::LossType::RAW;
        return *this;
    }

    virtual MeanAbsolutePercentageError::Builder &lossDataType(Tensor::DataType _lossDataType) {
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
