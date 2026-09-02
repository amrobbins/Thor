#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"

#include <stdexcept>
#include <utility>

using json = nlohmann::json;
using namespace std;

namespace Thor {
namespace {

void requireSamePartitionAndShape(const RaggedTensor& predictions, const RaggedTensor& labels) {
    if (!predictions.isInitialized() || !labels.isInitialized())
        throw invalid_argument("RaggedCustomLoss predictions and labels must be initialized RaggedTensor objects.");
    if (predictions.getValues() == labels.getValues())
        throw invalid_argument("RaggedCustomLoss predictions and labels values must be distinct graph tensors.");
    if (predictions.getOffsets() != labels.getOffsets())
        throw invalid_argument("RaggedCustomLoss predictions and labels must use the exact same row partition tensor.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
        predictions.getTrailingDimensions() != labels.getTrailingDimensions()) {
        throw invalid_argument("RaggedCustomLoss predictions and labels must have identical ragged value geometry.");
    }
}

void requireSecondaryCompatible(const RaggedTensor& predictions, const RaggedTensor& labels, const RaggedTensor& secondary) {
    if (!secondary.isInitialized())
        throw invalid_argument("RaggedCustomLoss secondary input must be an initialized RaggedTensor.");
    if (secondary.getValues() == predictions.getValues() || secondary.getValues() == labels.getValues())
        throw invalid_argument("RaggedCustomLoss secondary input values must be distinct from predictions and labels values.");
    if (secondary.getOffsets() != predictions.getOffsets())
        throw invalid_argument("RaggedCustomLoss secondary input must use the exact same row partition tensor as predictions.");
    if (secondary.getBatchSize() != predictions.getBatchSize() ||
        secondary.getMaxTotalValues() != predictions.getMaxTotalValues() ||
        secondary.getTrailingDimensions() != predictions.getTrailingDimensions())
        throw invalid_argument("RaggedCustomLoss secondary input must have identical ragged value geometry to predictions.");
}

void requireExampleWeightsCompatible(const RaggedTensor& predictions,
                                     const RaggedTensor& labels,
                                     const vector<RaggedTensor>& secondaries,
                                     const RaggedTensor& exampleWeights) {
    if (!exampleWeights.isInitialized())
        throw invalid_argument("RaggedCustomLoss example_weights must be an initialized RaggedTensor.");
    if (exampleWeights.getValues() == predictions.getValues() || exampleWeights.getValues() == labels.getValues())
        throw invalid_argument("RaggedCustomLoss example_weights values must be distinct from differentiable input values.");
    for (const RaggedTensor& secondary : secondaries)
        if (exampleWeights.getValues() == secondary.getValues())
            throw invalid_argument("RaggedCustomLoss example_weights values must be distinct from differentiable input values.");
    if (exampleWeights.getOffsets() != predictions.getOffsets())
        throw invalid_argument("RaggedCustomLoss example_weights must use the exact same row partition tensor as predictions.");
    if (exampleWeights.getBatchSize() != predictions.getBatchSize() ||
        exampleWeights.getMaxTotalValues() != predictions.getMaxTotalValues() ||
        exampleWeights.getTrailingDimensions() != vector<uint64_t>{1})
        throw invalid_argument("RaggedCustomLoss example_weights must contain one scalar weight per packed token.");
    ThorImplementation::RegressionLossDType::validateExampleWeightDType(
        "RaggedCustomLoss", exampleWeights.getValuesDataType());
}

}  // namespace

RaggedCustomLoss::RaggedCustomLoss(ThorImplementation::DynamicExpression lossExpression,
                                   ThorImplementation::DynamicExpression gradientExpression)
    : lossExpression(std::move(lossExpression)), gradientExpression(std::move(gradientExpression)) {}

RaggedCustomLoss RaggedCustomLoss::Builder::build() {
    if (!_network.has_value() || !_lossExpression.has_value() || !_gradientExpression.has_value() || !_predictions.has_value() ||
        !_labels.has_value()) {
        throw runtime_error("RaggedCustomLoss requires network, expressions, predictions, and labels.");
    }

    requireSamePartitionAndShape(_predictions.value(), _labels.value());
    ThorImplementation::RegressionLossDType::validatePredictionsDType(
        "RaggedCustomLoss", _predictions->getValuesDataType());
    ThorImplementation::RegressionLossDType::validateLabelsDType(
        "RaggedCustomLoss", _labels->getValuesDataType());
    const DataType lossDataType = _lossDataType.value_or(
        ThorImplementation::RegressionLossDType::defaultLossDType(_predictions->getValuesDataType()));
    ThorImplementation::RegressionLossDType::validateLossDType("RaggedCustomLoss", lossDataType);

    RaggedCustomLoss layer(_lossExpression.value(), _gradientExpression.value());
    layer.raggedPredictions = _predictions.value();
    layer.raggedLabels = _labels.value();
    if (_secondaryInputs.size() != _secondaryInputNames.size() || _secondaryInputs.size() != _secondaryGradientNames.size())
        throw runtime_error("RaggedCustomLoss secondary input metadata is inconsistent.");
    for (size_t i = 0; i < _secondaryInputs.size(); ++i) {
        requireSecondaryCompatible(layer.raggedPredictions, layer.raggedLabels, _secondaryInputs[i]);
        for (const RaggedTensor& existing : layer.raggedSecondaryInputs)
            if (existing.getValues() == _secondaryInputs[i].getValues())
                throw invalid_argument("RaggedCustomLoss secondary differentiable inputs must use distinct values tensors.");
        layer.raggedSecondaryInputs.push_back(_secondaryInputs[i]);
        layer.secondaryInputNames.push_back(_secondaryInputNames[i]);
        layer.secondaryGradientNames.push_back(_secondaryGradientNames[i]);
    }
    if (_exampleWeights.has_value()) {
        requireExampleWeightsCompatible(layer.raggedPredictions, layer.raggedLabels, layer.raggedSecondaryInputs, _exampleWeights.value());
        layer.raggedExampleWeights = _exampleWeights.value();
        layer.exampleWeightsTensor = _exampleWeights->getValues();
        layer.exampleWeightsName = _exampleWeightsName;
    }
    layer.predictionsTensor = layer.raggedPredictions.getValues();
    layer.labelsTensor = layer.raggedLabels.getValues();
    layer.lossDataType = lossDataType;
    layer.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
    layer.lossShape = LossShape::RAW;
    layer.predictionsName = _predictionsName;
    layer.labelsName = _labelsName;
    layer.lossName = _lossName;
    layer.gradientName = _gradientName;

    Tensor rawLossValues(lossDataType, layer.raggedPredictions.getValuesDimensions());
    layer.raggedRawLoss = layer.raggedPredictions.withValues(rawLossValues);
    layer.lossShaperInput = rawLossValues;
    layer.lossTensor = rawLossValues;
    layer.network = _network.value();
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

int RaggedCustomLoss::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == raggedPredictions.getValues())
        return static_cast<int>(ThorImplementation::RaggedCustomLoss::InputConnection::PREDICTIONS);
    if (connectingTensor == raggedLabels.getValues())
        return static_cast<int>(ThorImplementation::RaggedCustomLoss::InputConnection::LABELS);
    if (connectingTensor == raggedPredictions.getOffsets())
        return static_cast<int>(ThorImplementation::RaggedCustomLoss::InputConnection::OFFSETS);
    if (raggedExampleWeights.has_value() && connectingTensor == raggedExampleWeights->getValues())
        return static_cast<int>(ThorImplementation::RaggedCustomLoss::InputConnection::EXAMPLE_WEIGHTS);
    for (size_t i = 0; i < raggedSecondaryInputs.size(); ++i)
        if (connectingTensor == raggedSecondaryInputs[i].getValues())
            return static_cast<int>(ThorImplementation::RaggedCustomLoss::InputConnection::SECONDARY_INPUT_BASE) +
                   static_cast<int>(i);
    if (connectingTensor == raggedRawLoss.getValues())
        return 0;
    throw runtime_error("Tensor is not connected to this RaggedCustomLoss layer.");
}

optional<string> RaggedCustomLoss::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedPredictions.getValues()) return predictionsName;
    if (inputTensor == raggedLabels.getValues()) return labelsName;
    if (inputTensor == raggedPredictions.getOffsets()) return "offsets";
    if (raggedExampleWeights.has_value() && inputTensor == raggedExampleWeights->getValues()) return exampleWeightsName;
    for (size_t i = 0; i < raggedSecondaryInputs.size(); ++i)
        if (inputTensor == raggedSecondaryInputs[i].getValues()) return secondaryInputNames[i];
    return nullopt;
}

optional<string> RaggedCustomLoss::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor == raggedRawLoss.getValues()) return lossName;
    return nullopt;
}

bool RaggedCustomLoss::outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const {
    if (outputTensor != raggedRawLoss.getValues())
        throw invalid_argument("Tensor is not an output of this RaggedCustomLoss layer.");
    return true;
}

uint64_t RaggedCustomLoss::getOutputTensorBytes(uint32_t batchSize) const {
    (void)batchSize;
    return raggedRawLoss.getValues().getTotalSizeInBytes();
}

uint64_t RaggedCustomLoss::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)batchSize;
    (void)tensorPlacement;
    // Predictions/labels/offsets are driven by upstream layers. This loss owns
    // the raw loss values plus the training prediction gradient.
    uint64_t bytes = raggedRawLoss.getValues().getTotalSizeInBytes() + raggedPredictions.getValues().getTotalSizeInBytes();
    for (const RaggedTensor& secondary : raggedSecondaryInputs) bytes += secondary.getValues().getTotalSizeInBytes();
    return bytes;
}

shared_ptr<ThorImplementation::Layer> RaggedCustomLoss::stamp(ThorImplementation::TensorPlacement placement,
                                                               shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                               shared_ptr<Thor::Layer> drivingApiLayer,
                                                               Thor::Tensor connectingApiTensor,
                                                               bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);

    auto physical = make_shared<ThorImplementation::RaggedCustomLoss>(lossExpression,
                                                                       gradientExpression,
                                                                       raggedPredictions.getBatchSize(),
                                                                       raggedPredictions.getMaxTotalValues(),
                                                                       predictionsName,
                                                                       labelsName,
                                                                       lossName,
                                                                       gradientName,
                                                                       lossDataType,
                                                                       lossWeight,
                                                                       raggedExampleWeights.has_value() ? optional<string>(exampleWeightsName) : nullopt,
                                                                       secondaryInputNames,
                                                                       secondaryGradientNames);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedCustomLoss::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    auto serializedLossDefinition = lossExpression.getSerializedDefinition();
    auto serializedGradientDefinition = gradientExpression.getSerializedDefinition();
    if (serializedLossDefinition == nullptr || serializedGradientDefinition == nullptr) {
        throw runtime_error("RaggedCustomLoss expressions must be serialization-backed ExpressionDefinitions.");
    }

    json j{{"factory", Layer::Factory::Loss.value()},
           {"version", getLayerVersion()},
           {"layer_type", "ragged_custom_loss"},
           {"loss_shape", LossShape::RAW},
           {"loss_data_type", lossDataType},
           {"predictions_name", predictionsName},
           {"labels_name", labelsName},
           {"loss_name", lossName},
           {"gradient_name", gradientName},
           {"ragged_predictions", raggedPredictions.architectureJson()},
           {"ragged_labels", raggedLabels.architectureJson()},
           {"ragged_raw_loss", raggedRawLoss.architectureJson()},
           {"loss_expression", serializedLossDefinition->architectureJson()},
           {"gradient_expression", serializedGradientDefinition->architectureJson()}};
    if (raggedExampleWeights.has_value()) {
        j["example_weights_name"] = exampleWeightsName;
        j["ragged_example_weights"] = raggedExampleWeights->architectureJson();
    }
    if (!raggedSecondaryInputs.empty()) {
        j["ragged_secondary_inputs"] = json::array();
        for (size_t i = 0; i < raggedSecondaryInputs.size(); ++i) {
            j["ragged_secondary_inputs"].push_back({
                {"input_name", secondaryInputNames[i]},
                {"gradient_name", secondaryGradientNames[i]},
                {"tensor", raggedSecondaryInputs[i].architectureJson()},
            });
        }
        // Preserve the original one-secondary fields for old readers when there is exactly one.
        if (raggedSecondaryInputs.size() == 1) {
            j["secondary_input_name"] = secondaryInputNames.front();
            j["secondary_gradient_name"] = secondaryGradientNames.front();
            j["ragged_secondary_input"] = raggedSecondaryInputs.front().architectureJson();
        }
    }
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void RaggedCustomLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in RaggedCustomLoss::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "ragged_custom_loss")
        throw runtime_error("Layer type mismatch in RaggedCustomLoss::deserialize: " + j.at("layer_type").get<string>());
    if (j.at("loss_shape").get<LossShape>() != LossShape::RAW)
        throw runtime_error("Serialized RaggedCustomLoss must report raw loss.");

    RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "RaggedCustomLoss");
    RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "RaggedCustomLoss");
    requireSamePartitionAndShape(predictions, labels);
    vector<RaggedTensor> secondaryInputs;
    vector<string> secondaryInputNames;
    vector<string> secondaryGradientNames;
    if (j.contains("ragged_secondary_inputs")) {
        for (const json& serializedSecondary : j.at("ragged_secondary_inputs")) {
            RaggedTensor secondary =
                SegmentedPrimitiveDetail::reconstructInput(serializedSecondary.at("tensor"), network, "RaggedCustomLoss");
            requireSecondaryCompatible(predictions, labels, secondary);
            for (const RaggedTensor& existing : secondaryInputs)
                if (existing.getValues() == secondary.getValues())
                    throw runtime_error("Serialized RaggedCustomLoss secondary inputs must use distinct values tensors.");
            secondaryInputs.push_back(secondary);
            secondaryInputNames.push_back(serializedSecondary.at("input_name").get<string>());
            secondaryGradientNames.push_back(serializedSecondary.at("gradient_name").get<string>());
        }
    } else if (j.contains("ragged_secondary_input")) {
        RaggedTensor secondary =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_secondary_input"), network, "RaggedCustomLoss");
        requireSecondaryCompatible(predictions, labels, secondary);
        secondaryInputs.push_back(secondary);
        secondaryInputNames.push_back(j.value("secondary_input_name", string("secondary_input")));
        secondaryGradientNames.push_back(j.value("secondary_gradient_name", secondaryInputNames.front() + "_grad"));
    }
    optional<RaggedTensor> exampleWeights;
    if (j.contains("ragged_example_weights")) {
        exampleWeights = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_example_weights"), network, "RaggedCustomLoss");
        requireExampleWeightsCompatible(predictions, labels, secondaryInputs, exampleWeights.value());
    }
    ThorImplementation::RegressionLossDType::validatePredictionsDType(
        "RaggedCustomLoss", predictions.getValuesDataType());
    ThorImplementation::RegressionLossDType::validateLabelsDType(
        "RaggedCustomLoss", labels.getValuesDataType());
    const DataType serializedLossDType = j.at("loss_data_type").get<DataType>();
    ThorImplementation::RegressionLossDType::validateLossDType("RaggedCustomLoss", serializedLossDType);

    const json& rawJson = j.at("ragged_raw_loss");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        rawJson, j.at("ragged_predictions"), predictions, "RaggedCustomLoss");
    Tensor rawValues = Tensor::deserialize(rawJson.at("values"));
    if (rawValues.getDimensions() != predictions.getValuesDimensions() ||
        rawValues.getDataType() != serializedLossDType) {
        throw runtime_error("RaggedCustomLoss serialized raw loss values do not match prediction geometry/loss dtype.");
    }

    ThorImplementation::ExpressionDefinition lossDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("loss_expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : string{});
    ThorImplementation::ExpressionDefinition gradientDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("gradient_expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : string{});

    RaggedCustomLoss layer(ThorImplementation::DynamicExpression::fromExpressionDefinition(lossDefinition),
                           ThorImplementation::DynamicExpression::fromExpressionDefinition(gradientDefinition));
    layer.raggedPredictions = predictions;
    layer.raggedLabels = labels;
    layer.raggedSecondaryInputs = secondaryInputs;
    layer.secondaryInputNames = secondaryInputNames;
    layer.secondaryGradientNames = secondaryGradientNames;
    layer.raggedExampleWeights = exampleWeights;
    if (exampleWeights.has_value()) {
        layer.exampleWeightsTensor = exampleWeights->getValues();
        layer.exampleWeightsName = j.value("example_weights_name", string("example_weights"));
    }
    layer.raggedRawLoss = predictions.withValues(rawValues);
    layer.predictionsTensor = predictions.getValues();
    layer.labelsTensor = labels.getValues();
    layer.lossShaperInput = rawValues;
    layer.lossTensor = rawValues;
    layer.lossShape = LossShape::RAW;
    layer.lossDataType = serializedLossDType;
    layer.lossWeight = ThorImplementation::lossWeightFromJson(j);
    layer.predictionsName = j.value("predictions_name", string("predictions"));
    layer.labelsName = j.value("labels_name", string("labels"));
    layer.lossName = j.value("loss_name", string("loss"));
    layer.gradientName = j.value("gradient_name", layer.predictionsName + "_grad");
    layer.network = network;
    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Loss::register_layer("ragged_custom_loss", &Thor::RaggedCustomLoss::deserialize);
    return true;
}();
}  // namespace
