#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanAbsoluteError.h"

namespace Thor {

class MeanAbsoluteError : public Loss {
   public:
    class Builder;
    MeanAbsoluteError() {}

    virtual ~MeanAbsoluteError() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<MeanAbsoluteError>(*this); }

    virtual std::string getLayerType() const { return "MeanAbsoluteError"; }

    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual bool isMultiLayer() const {
        if (lossShape == LossShape::RAW)
            return false;
        return true;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::MeanAbsoluteError> meanAbsoluteError =
            std::make_shared<ThorImplementation::MeanAbsoluteError>(Tensor::convertToImplementationDataType(lossDataType));

        return meanAbsoluteError;
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
};

class MeanAbsoluteError::Builder {
   public:
    virtual ~Builder() = default;

    virtual MeanAbsoluteError build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_predictions.get().getDimensions().size() == 1);
        assert(_predictions.get().getDimensions() == _labels.get().getDimensions());

        if (_lossType.isEmpty())
            _lossType = LossShape::BATCH;
        if (_lossDataType.isEmpty())
            _lossDataType = _predictions.get().getDataType();
        uint32_t batchSize = _predictions.get().getDimensions()[0];

        MeanAbsoluteError meanAbsoluteError;
        meanAbsoluteError.predictionsTensor = _predictions;
        meanAbsoluteError.labelsTensor = _labels;
        meanAbsoluteError.lossDataType = _lossDataType;
        meanAbsoluteError.lossShape = _lossType;
        meanAbsoluteError.network = _network;
        meanAbsoluteError.initialized = true;

        if (meanAbsoluteError.isMultiLayer()) {
            meanAbsoluteError.buildSupportLayersAndAddToNetwork();
        } else {
            // lossTensor is the one that comes directly out of MeanAbsoluteError, that may be replaced by a loss shaper.
            meanAbsoluteError.lossTensor = Tensor(_lossDataType, {batchSize});
            meanAbsoluteError.lossShaperInput = meanAbsoluteError.lossTensor;
            meanAbsoluteError.addToNetwork(_network.get());
        }

        return meanAbsoluteError;
    }

    virtual MeanAbsoluteError::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &reportsBatchLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::BATCH;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &reportsElementwiseLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &reportsPerOutputLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::CLASSWISE;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &reportsRawLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::RAW;
        return *this;
    }

    virtual MeanAbsoluteError::Builder &lossDataType(Tensor::DataType _lossDataType) {
        assert(this->_lossDataType.isEmpty());
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<LossShape> _lossType;
    Optional<Tensor::DataType> _lossDataType;
};

}  // namespace Thor
