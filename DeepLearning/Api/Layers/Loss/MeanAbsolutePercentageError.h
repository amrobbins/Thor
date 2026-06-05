#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/MeanAbsolutePercentageError.h"
#include <optional>

namespace Thor {

class MAPE : public Loss {
   public:
    class Builder;
    MAPE() {}

    ~MAPE() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<MAPE>(*this); }

    std::string getLayerType() const override { return "MAPE"; }

    nlohmann::json architectureJson() const override;

    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual bool isMultiLayer() const {
        if (lossShape == LossShape::RAW)
            return false;
        return true;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        // FIXME: How to prune backward then.
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::MeanAbsolutePercentageError> meanAbsolutePercentageError =
            std::make_shared<ThorImplementation::MeanAbsolutePercentageError>(lossDataType);

        return meanAbsolutePercentageError;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
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

class MAPE::Builder {
   public:
    virtual MAPE build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        uint32_t batchSize = _predictions.value().getDimensions()[0];

        MAPE meanAbsolutePercentageError;
        meanAbsolutePercentageError.predictionsTensor = _predictions.value();
        meanAbsolutePercentageError.labelsTensor = _labels.value();
        meanAbsolutePercentageError.lossDataType = _lossDataType.value();
        meanAbsolutePercentageError.lossShape = _lossShape.value();
        meanAbsolutePercentageError.network = _network.value();
        meanAbsolutePercentageError.initialized = true;

        if (meanAbsolutePercentageError.isMultiLayer()) {
            meanAbsolutePercentageError.buildSupportLayersAndAddToNetwork();
        } else {
            // lossTensor is the one that comes directly out of MAPE, that may be replaced by a loss shaper.
            meanAbsolutePercentageError.lossTensor = Tensor(_lossDataType.value(), {batchSize});
            meanAbsolutePercentageError.lossShaperInput = meanAbsolutePercentageError.lossTensor;
            meanAbsolutePercentageError.addToNetwork(_network.value());
        }

        return meanAbsolutePercentageError;
    }

    virtual MAPE::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual MAPE::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual MAPE::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual MAPE::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual MAPE::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual MAPE::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual MAPE::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual MAPE::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
};


}  // namespace Thor
