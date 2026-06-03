#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropy.h"
#include <optional>

namespace Thor {

class BinaryCrossEntropy : public Loss {
   public:
    class Builder;
    BinaryCrossEntropy() {}

    ~BinaryCrossEntropy() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<BinaryCrossEntropy>(*this); }

    std::string getLayerType() const override { return "BinaryCrossEntropy"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        // FIXME: How to prune backward then.
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        return std::make_shared<ThorImplementation::BinaryCrossEntropy>(lossDataType);
    }

    virtual bool isMultiLayer() const {
        if (lossShape != LossShape::ELEMENTWISE || !rawLossAddedToNetwork)
            return true;
        return false;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    bool rawLossAddedToNetwork;
};

class BinaryCrossEntropy::Builder {
   public:
    virtual BinaryCrossEntropy build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;

        std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
        // API layer does not have a batch dimension:
        THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);

        BinaryCrossEntropy binaryCrossEntropy;
        if (_rawLossAddedToNetwork.has_value()) {
            THOR_THROW_IF_FALSE(_rawLossAddedToNetwork.value() == true);
            binaryCrossEntropy.rawLossAddedToNetwork = true;
        } else {
            binaryCrossEntropy.rawLossAddedToNetwork = false;
        }
        binaryCrossEntropy.predictionsTensor = _predictions.value();
        binaryCrossEntropy.labelsTensor = _labels.value();
        if (!_lossDataType.has_value())
            _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        binaryCrossEntropy.lossDataType = _lossDataType.value();

        if (_lossShape.value() == LossShape::BATCH) {
            binaryCrossEntropy.lossShape = LossShape::BATCH;
        } else if (_lossShape.value() == LossShape::ELEMENTWISE) {
            binaryCrossEntropy.lossShape = LossShape::ELEMENTWISE;
        }
        binaryCrossEntropy.initialized = true;
        binaryCrossEntropy.network = _network.value();

        if (binaryCrossEntropy.isMultiLayer()) {
            binaryCrossEntropy.buildSupportLayersAndAddToNetwork();
        } else {
            THOR_THROW_IF_FALSE(binaryCrossEntropy.lossShape == LossShape::ELEMENTWISE);
            binaryCrossEntropy.lossTensor = Tensor(_lossDataType.value(), {1});
            binaryCrossEntropy.lossShaperInput = binaryCrossEntropy.lossTensor;
            binaryCrossEntropy.addToNetwork(_network.value());
        }

        return binaryCrossEntropy;
    }

    virtual BinaryCrossEntropy::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(_predictions.getDimensions().size() == 1 && _predictions.getDimensions()[0] == 1);
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(_labels.getDimensions().size() == 1 && _labels.getDimensions()[0] == 1);
        this->_labels = _labels;
        return *this;
    }

    /**
     * Reports loss to the user as a single scalar that represents the total loss of the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    /**
     * Reports loss to the user as a scalar per class per example in the batch.
     * Note that is only for reporting, this setting does not affect the form of loss used in the math to train the network.
     */
    virtual BinaryCrossEntropy::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP32 || _lossDataType == DataType::FP16);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   protected:
    /**
     * BinaryCrossEntropy is implemented as BCE-with-logits.
     * When the public compound layer is built a raw elementwise BCE layer will be built and this will be recorded so that the
     * next build is the single raw layer that can be directly stamped.
     */
    virtual BinaryCrossEntropy::Builder &rawLossAddedToNetwork() {
        THOR_THROW_IF_FALSE(!_rawLossAddedToNetwork.has_value());
        _rawLossAddedToNetwork = true;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<bool> _rawLossAddedToNetwork;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
