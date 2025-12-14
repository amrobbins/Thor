#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanSquaredError.h"

namespace Thor {

class MeanSquaredError : public Loss {
   public:
    class Builder;
    MeanSquaredError() {}

    virtual ~MeanSquaredError() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<MeanSquaredError>(*this); }

    virtual std::string getLayerType() const { return "MeanSquaredError"; }

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

        std::shared_ptr<ThorImplementation::MeanSquaredError> meanSquaredError =
            std::make_shared<ThorImplementation::MeanSquaredError>(Tensor::convertToImplementationDataType(lossDataType));

        return meanSquaredError;
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

class MeanSquaredError::Builder {
   public:
    virtual MeanSquaredError build() {
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

        MeanSquaredError meanSquaredError;
        meanSquaredError.predictionsTensor = _predictions;
        meanSquaredError.labelsTensor = _labels;
        meanSquaredError.lossDataType = _lossDataType;
        meanSquaredError.lossShape = _lossType;
        meanSquaredError.network = _network;
        meanSquaredError.initialized = true;

        if (meanSquaredError.isMultiLayer()) {
            meanSquaredError.buildSupportLayersAndAddToNetwork();
        } else {
            // lossTensor is the one that comes directly out of MeanSquaredError, that may be replaced by a loss shaper.
            meanSquaredError.lossTensor = Tensor(_lossDataType, {batchSize});
            meanSquaredError.addToNetwork(_network.get());
        }

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
        _lossType = LossShape::BATCH;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsElementwiseLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsPerOutputLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::CLASSWISE;
        return *this;
    }

    virtual MeanSquaredError::Builder &reportsRawLoss() {
        assert(this->_lossType.isEmpty());
        _lossType = LossShape::RAW;
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
    Optional<LossShape> _lossType;
    Optional<Tensor::DataType> _lossDataType;
};

}  // namespace Thor
