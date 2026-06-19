#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
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
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

        return std::make_shared<ThorImplementation::BinaryCrossEntropy>(lossDataType);
    }

    virtual bool isMultiLayer() const { return true; }

    virtual void buildSupportLayersAndAddToNetwork();

    bool rawLossAddedToNetwork = false;
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

        std::vector<uint64_t> predictionDimensions = _predictions.value().getDimensions();
        std::vector<uint64_t> labelDimensions = _labels.value().getDimensions();
        // API layer does not have a batch dimension. Allow vector-valued BCE for multi-output/multi-label heads.
        THOR_THROW_IF_FALSE(predictionDimensions.size() == 1);
        THOR_THROW_IF_FALSE(labelDimensions.size() == 1);
        THOR_THROW_IF_FALSE(predictionDimensions == labelDimensions);

        BinaryCrossEntropy binaryCrossEntropy;
        binaryCrossEntropy.rawLossAddedToNetwork = _rawLossAddedToNetwork.value_or(false);
        binaryCrossEntropy.predictionsTensor = _predictions.value();
        binaryCrossEntropy.labelsTensor = _labels.value();
        if (!_lossDataType.has_value())
            _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
        binaryCrossEntropy.lossDataType = _lossDataType.value();

        binaryCrossEntropy.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);

        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::BATCH || _lossShape.value() == LossShape::ELEMENTWISE);
        binaryCrossEntropy.lossShape = _lossShape.value();
        binaryCrossEntropy.initialized = true;
        binaryCrossEntropy.network = _network.value();

        if (binaryCrossEntropy.rawLossAddedToNetwork) {
            // Legacy/deserialization-only path: build the single raw BCE layer itself. New public BCE construction
            // builds a raw CustomLoss support layer instead.
            THOR_THROW_IF_FALSE(binaryCrossEntropy.lossShape == LossShape::ELEMENTWISE);
            binaryCrossEntropy.lossTensor = Tensor(_lossDataType.value(), predictionDimensions);
            binaryCrossEntropy.lossShaperInput = binaryCrossEntropy.lossTensor;
            binaryCrossEntropy.addToNetwork(_network.value());
        } else {
            binaryCrossEntropy.buildSupportLayersAndAddToNetwork();
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
        THOR_THROW_IF_FALSE(_predictions.getDimensions().size() == 1);
        this->_predictions = _predictions;
        return *this;
    }

    virtual BinaryCrossEntropy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(_labels.getDimensions().size() == 1);
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

    virtual BinaryCrossEntropy::Builder & lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
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
     * Legacy/internal path for reconstructing the historical raw BCE layer. Public BCE construction now routes through CustomLoss.
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
    std::optional<float> _lossWeight;
    std::optional<bool> _rawLossAddedToNetwork;

    friend class BinaryCrossEntropy;
};

}  // namespace Thor
