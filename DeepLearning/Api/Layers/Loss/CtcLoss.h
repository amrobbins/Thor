#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"
#include "Utilities/TensorOperations/Loss/CtcLoss.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <optional>
#include <vector>

namespace Thor {

// Public cuDNN-backed CTC loss.
//
// Canonical API contract:
//   * logits/activations tensor has API dimensions [T, C] and FP32 dtype.
//     Thor's physical batch dimension makes the implementation tensor [B, T, C].
//   * labels are a rank-1 RaggedTensor whose packed values are INT32 and whose
//     canonical UINT32/UINT64 offsets describe one target sequence per batch row.
//   * input_lengths has API dimensions [1] and INT32 dtype.
//   * label lengths are derived from labels.offsets on device. There is no
//     padded-label or separately supplied label-length compatibility path.
//   * blank label is cuDNN's fixed blank convention: class 0.
//   * cuDNN deterministic CTC only; no native/CPU fallback.
class CtcLoss : public Loss {
   public:
    class Builder;

    CtcLoss() = default;
    ~CtcLoss() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CtcLoss>(*this); }
    std::string getLayerType() const override { return "CtcLoss"; }
    std::string getLayerVersion() const override { return "2.0.0"; }

    RaggedTensor getRaggedLabels() const {
        THOR_THROW_IF_FALSE(labelsRaggedTensor.isInitialized());
        return labelsRaggedTensor;
    }
    Tensor getInputLengths() const { return inputLengthsTensor; }
    ThorImplementation::CtcLossOobGradientMode getOobGradientMode() const { return oobGradientMode; }

    std::vector<Tensor> getLossInputTensors() const override {
        THOR_THROW_IF_FALSE(labelsRaggedTensor.isInitialized());
        return {predictionsTensor, labelsRaggedTensor.getValues(), labelsRaggedTensor.getOffsets(), inputLengthsTensor};
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == predictionsTensor)
            return static_cast<int>(ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD);
        if (labelsRaggedTensor.isInitialized() && connectingTensor == labelsRaggedTensor.getValues())
            return static_cast<int>(ThorImplementation::Loss::ConnectionType::LABELS);
        if (labelsRaggedTensor.isInitialized() && connectingTensor == labelsRaggedTensor.getOffsets())
            return ThorImplementation::CtcLoss::LABEL_OFFSETS_CONNECTION_TYPE;
        if (connectingTensor == inputLengthsTensor)
            return ThorImplementation::CtcLoss::INPUT_LENGTHS_CONNECTION_TYPE;
        if (connectingTensor == getRawLoss())
            return 0;
        throw std::runtime_error("Tensor is not connected to this CtcLoss.");
    }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (predictionsTensor.isInitialized() && inputTensor == predictionsTensor)
            return "predictions";
        if (labelsRaggedTensor.isInitialized() && inputTensor == labelsRaggedTensor.getValues())
            return "labels.values";
        if (labelsRaggedTensor.isInitialized() && inputTensor == labelsRaggedTensor.getOffsets())
            return "labels.offsets";
        if (inputLengthsTensor.isInitialized() && inputTensor == inputLengthsTensor)
            return "input_lengths";
        return std::nullopt;
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
        THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsRaggedTensor.getValues() ||
                            connectingApiTensor == labelsRaggedTensor.getOffsets() || connectingApiTensor == inputLengthsTensor);
        return std::make_shared<ThorImplementation::CtcLoss>(oobGradientMode, lossWeight);
    }

    bool rawLossAddedToNetwork = false;
    RaggedTensor labelsRaggedTensor;
    Tensor inputLengthsTensor;
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

    CtcLoss::Builder& labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        THOR_THROW_IF_FALSE(labels.getValuesDataType() == DataType::INT32);
        THOR_THROW_IF_FALSE(labels.getTrailingDimensions().empty());
        THOR_THROW_IF_FALSE(labels.getBatchSize() > 0);
        THOR_THROW_IF_FALSE(labels.getMaxTotalValues() > 0);
        THOR_THROW_IF_FALSE(ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(labels.getOffsetsDataType()));
        this->_labels = labels;
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

    CtcLoss::Builder& reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    CtcLoss::Builder& reportsNoLoss() {
        THOR_THROW_IF_FALSE(!_lossShape.has_value());
        _lossShape = LossShape::NONE;
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
        THOR_THROW_IF_FALSE(_inputLengths.has_value());

        const Tensor labelValues = _labels->getValues();
        const Tensor labelOffsets = _labels->getOffsets();
        THOR_THROW_IF_FALSE(_predictions.value() != labelValues);
        THOR_THROW_IF_FALSE(_predictions.value() != labelOffsets);
        THOR_THROW_IF_FALSE(_predictions.value() != _inputLengths.value());
        THOR_THROW_IF_FALSE(labelValues != labelOffsets);
        THOR_THROW_IF_FALSE(labelValues != _inputLengths.value());
        THOR_THROW_IF_FALSE(labelOffsets != _inputLengths.value());

        const std::vector<uint64_t>& predictionDims = _predictions.value().getDimensions();
        THOR_THROW_IF_FALSE(predictionDims.size() == 2);
        THOR_THROW_IF_FALSE(predictionDims[0] > 0);

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        THOR_THROW_IF_FALSE(_lossShape.value() == LossShape::NONE || _lossShape.value() == LossShape::BATCH ||
                            _lossShape.value() == LossShape::PER_EXAMPLE || _lossShape.value() == LossShape::RAW);

        if (!_lossDataType.has_value())
            _lossDataType = DataType::FP32;
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP32);

        ctcLoss.rawLossAddedToNetwork = _rawLossAddedToNetwork.value_or(false);
        ctcLoss.predictionsTensor = _predictions.value();
        ctcLoss.labelsRaggedTensor = _labels.value();
        // Loss's base graph bookkeeping is Tensor-valued, so use the packed
        // values edge there. The CTC API itself exposes labels as RaggedTensor.
        ctcLoss.labelsTensor = labelValues;
        ctcLoss.inputLengthsTensor = _inputLengths.value();
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
    std::optional<RaggedTensor> _labels;
    std::optional<Tensor> _inputLengths;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<ThorImplementation::CtcLossOobGradientMode> _oobGradientMode;
    std::optional<bool> _rawLossAddedToNetwork;

    friend class CtcLoss;
};

}  // namespace Thor
