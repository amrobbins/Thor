#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"
#include "Utilities/TensorOperations/Loss/CtcLoss.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <optional>
#include <vector>

namespace Thor {

// Public cuDNN-backed CTC loss.
//
// v1 API contract:
//   * logits/activations tensor has API dimensions [T, C] and FP32 dtype.
//     Thor's physical batch dimension makes the implementation tensor [B, T, C].
//   * labels tensor has API dimensions [maxLabelLength] and INT32 dtype. It is
//     a padded per-sample target row; the implementation layer compacts it to
//     cuDNN's packed labels list on device using label_lengths.
//   * label_lengths and input_lengths have API dimensions [1] and INT32 dtype.
//   * blank label is cuDNN's fixed blank convention: class 0.
//   * cuDNN deterministic CTC only; no native/CPU fallback.
class CtcLoss : public Loss {
   public:
    class Builder;

    CtcLoss() = default;
    ~CtcLoss() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CtcLoss>(*this); }
    std::string getLayerType() const override { return "CtcLoss"; }

    Tensor getLabelLengths() const { return labelLengthsTensor; }
    Tensor getInputLengths() const { return inputLengthsTensor; }
    uint32_t getMaxLabelLength() const { return maxLabelLength; }
    ThorImplementation::CtcLossOobGradientMode getOobGradientMode() const { return oobGradientMode; }

    std::vector<Tensor> getLossInputTensors() const override {
        return {predictionsTensor, labelsTensor, labelLengthsTensor, inputLengthsTensor};
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == predictionsTensor)
            return static_cast<int>(ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD);
        if (connectingTensor == labelsTensor)
            return static_cast<int>(ThorImplementation::Loss::ConnectionType::LABELS);
        if (connectingTensor == labelLengthsTensor)
            return ThorImplementation::CtcLoss::LABEL_LENGTHS_CONNECTION_TYPE;
        if (connectingTensor == inputLengthsTensor)
            return ThorImplementation::CtcLoss::INPUT_LENGTHS_CONNECTION_TYPE;
        if (connectingTensor == lossTensor)
            return 0;
        throw std::runtime_error("Tensor is not connected to this CtcLoss.");
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    bool isMultiLayer() const { return lossShape != LossShape::RAW || !rawLossAddedToNetwork; }
    void buildSupportLayersAndAddToNetwork();

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
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor ||
                            connectingApiTensor == labelLengthsTensor || connectingApiTensor == inputLengthsTensor);
        return std::make_shared<ThorImplementation::CtcLoss>(maxLabelLength, oobGradientMode, lossWeight);
    }

    bool rawLossAddedToNetwork = false;
    Tensor labelLengthsTensor;
    Tensor inputLengthsTensor;
    uint32_t maxLabelLength = 0;
    ThorImplementation::CtcLossOobGradientMode oobGradientMode = ThorImplementation::CtcLossOobGradientMode::ZERO;
};

class CtcLoss::Builder {
   public:
    CtcLoss build() {
        CtcLoss ctcLoss;
        populateAndAdd(ctcLoss);
        return ctcLoss;
    }

    CtcLoss::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    CtcLoss::Builder& logits(Tensor logits) { return predictions(logits); }

    CtcLoss::Builder& predictions(Tensor predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(predictions.getDataType() == DataType::FP32);
        const std::vector<uint64_t>& dims = predictions.getDimensions();
        THOR_THROW_IF_FALSE(dims.size() == 2);
        THOR_THROW_IF_FALSE(dims[0] > 0);
        THOR_THROW_IF_FALSE(dims[1] > 1);
        this->_predictions = predictions;
        return *this;
    }

    CtcLoss::Builder& labels(Tensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(labels.getDataType() == DataType::INT32);
        const std::vector<uint64_t>& dims = labels.getDimensions();
        THOR_THROW_IF_FALSE(dims.size() == 1);
        THOR_THROW_IF_FALSE(dims[0] > 0);
        THOR_THROW_IF_FALSE(dims[0] < 256);
        this->_labels = labels;
        return *this;
    }

    CtcLoss::Builder& labelLengths(Tensor labelLengths) {
        THOR_THROW_IF_FALSE(!this->_labelLengths.has_value());
        THOR_THROW_IF_FALSE(ThorImplementation::isCudnnCtcLengthDataType(labelLengths.getDataType()));
        THOR_THROW_IF_FALSE(labelLengths.getDimensions() == std::vector<uint64_t>{1});
        this->_labelLengths = labelLengths;
        return *this;
    }

    CtcLoss::Builder& inputLengths(Tensor inputLengths) {
        THOR_THROW_IF_FALSE(!this->_inputLengths.has_value());
        THOR_THROW_IF_FALSE(ThorImplementation::isCudnnCtcLengthDataType(inputLengths.getDataType()));
        THOR_THROW_IF_FALSE(inputLengths.getDimensions() == std::vector<uint64_t>{1});
        this->_inputLengths = inputLengths;
        return *this;
    }

    CtcLoss::Builder& reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    CtcLoss::Builder& reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    CtcLoss::Builder& reportsRawLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    CtcLoss::Builder& lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    CtcLoss::Builder& lossDataType(DataType lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP32);
        this->_lossDataType = lossDataType;
        return *this;
    }

    CtcLoss::Builder& zeroOutOfBoundsGradients() {
        THOR_THROW_IF_FALSE(!_oobGradientMode.has_value());
        _oobGradientMode = ThorImplementation::CtcLossOobGradientMode::ZERO;
        return *this;
    }

    CtcLoss::Builder& skipOutOfBoundsGradients() {
        THOR_THROW_IF_FALSE(!_oobGradientMode.has_value());
        _oobGradientMode = ThorImplementation::CtcLossOobGradientMode::SKIP;
        return *this;
    }

   protected:
    CtcLoss::Builder& rawLossAddedToNetwork() {
        THOR_THROW_IF_FALSE(!_rawLossAddedToNetwork.has_value());
        _rawLossAddedToNetwork = true;
        return *this;
    }

    void populateAndAdd(CtcLoss& ctcLoss) {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_labelLengths.has_value());
        THOR_THROW_IF_FALSE(_inputLengths.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labelLengths.value());
        THOR_THROW_IF_FALSE(_predictions.value() != _inputLengths.value());
        THOR_THROW_IF_FALSE(_labels.value() != _labelLengths.value());
        THOR_THROW_IF_FALSE(_labels.value() != _inputLengths.value());
        THOR_THROW_IF_FALSE(_labelLengths.value() != _inputLengths.value());

        const std::vector<uint64_t>& predictionDims = _predictions.value().getDimensions();
        const std::vector<uint64_t>& labelDims = _labels.value().getDimensions();
        THOR_THROW_IF_FALSE(predictionDims.size() == 2);
        THOR_THROW_IF_FALSE(labelDims.size() == 1);
        THOR_THROW_IF_FALSE(labelDims[0] > 0 && labelDims[0] < 256);
        THOR_THROW_IF_FALSE(predictionDims[0] >= labelDims[0]);

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::BATCH || _lossShape.value() == LossShape::ELEMENTWISE ||
                            _lossShape.value() == LossShape::RAW);

        if (!_lossDataType.has_value())
            _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP32);

        ctcLoss.rawLossAddedToNetwork = _rawLossAddedToNetwork.value_or(false);
        ctcLoss.predictionsTensor = _predictions.value();
        ctcLoss.labelsTensor = _labels.value();
        ctcLoss.labelLengthsTensor = _labelLengths.value();
        ctcLoss.inputLengthsTensor = _inputLengths.value();
        ctcLoss.maxLabelLength = static_cast<uint32_t>(labelDims[0]);
        ctcLoss.lossDataType = _lossDataType.value();
        ctcLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        ctcLoss.lossShape = _lossShape.value();
        ctcLoss.oobGradientMode = _oobGradientMode.value_or(ThorImplementation::CtcLossOobGradientMode::ZERO);
        ctcLoss.initialized = true;
        ctcLoss.network = _network.value();

        if (ctcLoss.isMultiLayer()) {
            ctcLoss.buildSupportLayersAndAddToNetwork();
        } else {
            THOR_THROW_IF_FALSE(ctcLoss.lossShape == LossShape::RAW);
            ctcLoss.lossTensor = Tensor(DataType::FP32, {1});
            ctcLoss.lossShaperInput = ctcLoss.lossTensor;
            ctcLoss.addToNetwork(_network.value());
        }
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _labelLengths;
    std::optional<Tensor> _inputLengths;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<ThorImplementation::CtcLossOobGradientMode> _oobGradientMode;
    std::optional<bool> _rawLossAddedToNetwork;

    friend class CtcLoss;
};

}  // namespace Thor
