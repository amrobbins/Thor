#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"

#include <limits>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;
using namespace std;

namespace Thor {
namespace {

// The semantic M is supplied by the logical neutral input, never recovered
// from a historic physical capacity. Preserve legacy fixed/fixed contracts.
void requireMatchingGeometry(const RaggedTensor& reference, const RaggedTensor& other, const char* description) {
    if (!reference.sharesPartitionWith(other) || reference.isPlacementNeutral() != other.isPlacementNeutral() ||
        reference.getOffsetsDataType() != other.getOffsetsDataType()) {
        throw invalid_argument(string("RaggedCustomLoss ") + description + " must use the same canonical partition and geometry mode.");
    }
    if (reference.isPlacementNeutral()) {
        if (!reference.hasMaxValuesPerRow() || !other.hasMaxValuesPerRow() ||
            reference.getMaxValuesPerRow() != other.getMaxValuesPerRow()) {
            throw invalid_argument(string("RaggedCustomLoss ") + description + " semantic max_values_per_row differs.");
        }
    } else if (reference.getBatchSize() != other.getBatchSize() ||
               reference.getMaxTotalValues() != other.getMaxTotalValues() ||
               (reference.hasMaxValuesPerRow() && other.hasMaxValuesPerRow() &&
                reference.getMaxValuesPerRow() != other.getMaxValuesPerRow())) {
        throw invalid_argument(string("RaggedCustomLoss ") + description + " fixed ragged geometry differs.");
    }
}

uint64_t checkedAdd(uint64_t a, uint64_t b) {
    if (b > numeric_limits<uint64_t>::max() - a)
        throw overflow_error("RaggedCustomLoss memory requirement overflows uint64_t.");
    return a + b;
}

uint64_t placementBytes(const RaggedTensor& tensor, uint32_t batchSize) {
    const uint64_t logicalBytes = tensor.getValues().getTotalSizeInBytes();
    if (!tensor.isPlacementNeutral()) return logicalBytes;
    if (batchSize == 0 || logicalBytes > numeric_limits<uint64_t>::max() / batchSize)
        throw overflow_error("RaggedCustomLoss placement capacity overflows uint64_t.");
    return logicalBytes * batchSize;
}

// The shared reconstruction helper verifies the neutral semantic shape.
// Reject nested JSON field forgery as well as wrong input IDs: the canonical
// graph tensors, not user-supplied copies of their geometry, own identities.
void validateNeutralReference(const json& serialized, const RaggedTensor& reconstructed) {
    if (!reconstructed.isPlacementNeutral()) return;  // Legacy fixed 1.0.0 unchanged.
    const auto checkTensor = [](const json& j, const Tensor& actual) {
        if (j.at("version").get<string>() != "1.0.0" ||
            j.at("id").get<uint64_t>() != actual.getOriginalId() ||
            j.at("dimensions").get<vector<uint64_t>>() != actual.getDimensions() ||
            j.at("data_type").get<DataType>() != actual.getDataType()) {
            throw runtime_error("RaggedCustomLoss serialized neutral input does not match the canonical graph tensor.");
        }
    };
    checkTensor(serialized.at("values"), reconstructed.getValues());
    checkTensor(serialized.at("offsets"), reconstructed.getRowPartitionToken());
}

void requireSamePartitionAndShape(const RaggedTensor& predictions, const RaggedTensor& labels) {
    if (!predictions.isInitialized() || !labels.isInitialized())
        throw invalid_argument("RaggedCustomLoss predictions and labels must be initialized RaggedTensor objects.");
    if (predictions.getValues() == labels.getValues())
        throw invalid_argument("RaggedCustomLoss predictions and labels values must be distinct graph tensors.");
    if (!predictions.sharesPartitionWith(labels))
        throw invalid_argument("RaggedCustomLoss predictions and labels must use the exact same row partition.");
    requireMatchingGeometry(predictions, labels, "labels");
    if (predictions.getTrailingDimensions() != labels.getTrailingDimensions()) {
        throw invalid_argument("RaggedCustomLoss predictions and labels must have identical ragged value geometry.");
    }
}

void requireSecondaryCompatible(const RaggedTensor& predictions, const RaggedTensor& labels, const RaggedTensor& secondary) {
    if (!secondary.isInitialized())
        throw invalid_argument("RaggedCustomLoss secondary input must be an initialized RaggedTensor.");
    if (secondary.getValues() == predictions.getValues() || secondary.getValues() == labels.getValues())
        throw invalid_argument("RaggedCustomLoss secondary input values must be distinct from predictions and labels values.");
    if (!secondary.sharesPartitionWith(predictions))
        throw invalid_argument("RaggedCustomLoss secondary input must use the exact same row partition as predictions.");
    requireMatchingGeometry(predictions, secondary, "secondary input");
    if (secondary.getTrailingDimensions() != predictions.getTrailingDimensions())
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
    if (!exampleWeights.sharesPartitionWith(predictions))
        throw invalid_argument("RaggedCustomLoss example_weights must use the exact same row partition as predictions.");
    requireMatchingGeometry(predictions, exampleWeights, "example_weights");
    if (exampleWeights.getTrailingDimensions() != vector<uint64_t>{1})
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
    // The logical neutral output is [M, ...]. Its physical RAW output is
    // [B*M, ...]; the output-dimensions-include-batch flag stays true because
    // the physical loss already materializes this entire packed output.
    return placementBytes(raggedRawLoss, batchSize);
}

uint64_t RaggedCustomLoss::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    // Predictions, labels and the active-count carrier are upstream-owned;
    // this layer owns RAW output and gradients for differentiable inputs.
    uint64_t bytes = checkedAdd(placementBytes(raggedRawLoss, batchSize), placementBytes(raggedPredictions, batchSize));
    for (const RaggedTensor& secondary : raggedSecondaryInputs)
        bytes = checkedAdd(bytes, placementBytes(secondary, batchSize));
    return bytes;
}

shared_ptr<ThorImplementation::Layer> RaggedCustomLoss::stamp(ThorImplementation::TensorPlacement placement,
                                                               shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                               shared_ptr<Thor::Layer> drivingApiLayer,
                                                               Thor::Tensor connectingApiTensor,
                                                               bool inferenceOnly) const {
    (void)placement;
    (void)drivingApiLayer;
    (void)getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);

    uint64_t runtimeBatch = 0;
    uint64_t runtimeCapacity = 0;
    if (raggedPredictions.isPlacementNeutral()) {
        const RaggedTensor* connected = &raggedPredictions;
        if (connectingApiTensor == raggedLabels.getValues()) {
            connected = &raggedLabels;
        } else if (raggedExampleWeights.has_value() && connectingApiTensor == raggedExampleWeights->getValues()) {
            connected = &raggedExampleWeights.value();
        } else {
            for (const RaggedTensor& secondary : raggedSecondaryInputs) {
                if (connectingApiTensor == secondary.getValues()) {
                    connected = &secondary;
                    break;
                }
            }
        }
        // In particular, the structural-token connection resolves through the
        // managed partition fanout's authoritative batch metadata, not [1]'s extent.
        const ThorImplementation::RaggedTensorDescriptor physical =
            SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
                *connected, drivingLayer, "RaggedCustomLoss");
        runtimeBatch = physical.getBatchSize();
        runtimeCapacity = physical.getMaxTotalValues();
        if (runtimeBatch == 0 || runtimeBatch > numeric_limits<uint32_t>::max() ||
            runtimeCapacity != raggedPredictions.getDescriptorForPlacement(runtimeBatch).getMaxTotalValues()) {
            throw runtime_error("RaggedCustomLoss physical placement disagrees with neutral semantic geometry.");
        }
    } else {
        runtimeBatch = raggedPredictions.getBatchSize();
        runtimeCapacity = raggedPredictions.getMaxTotalValues();
    }

    auto physical = make_shared<ThorImplementation::RaggedCustomLoss>(lossExpression,
                                                                       gradientExpression,
                                                                       runtimeBatch,
                                                                       runtimeCapacity,
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
           {"version", raggedPredictions.isPlacementNeutral() ? "1.1.0" : getLayerVersion()},
           {"layer_type", "ragged_custom_loss"},
           {"loss_shape", LossShape::RAW},
           {"loss_data_type", lossDataType},
           {"predictions_name", predictionsName},
           {"labels_name", labelsName},
           {"loss_name", lossName},
           {"gradient_name", gradientName},
           {"ragged_predictions", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedPredictions)},
           {"ragged_labels", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedLabels)},
           {"ragged_raw_loss", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedRawLoss)},
           {"loss_expression", serializedLossDefinition->architectureJson()},
           {"gradient_expression", serializedGradientDefinition->architectureJson()}};
    if (raggedExampleWeights.has_value()) {
        j["example_weights_name"] = exampleWeightsName;
        j["ragged_example_weights"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedExampleWeights.value());
    }
    if (!raggedSecondaryInputs.empty()) {
        j["ragged_secondary_inputs"] = json::array();
        for (size_t i = 0; i < raggedSecondaryInputs.size(); ++i) {
            j["ragged_secondary_inputs"].push_back({
                {"input_name", secondaryInputNames[i]},
                {"gradient_name", secondaryGradientNames[i]},
                {"tensor", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedSecondaryInputs[i])},
            });
        }
        // Preserve the original one-secondary fields for old readers when there is exactly one.
        if (raggedSecondaryInputs.size() == 1) {
            j["secondary_input_name"] = secondaryInputNames.front();
            j["secondary_gradient_name"] = secondaryGradientNames.front();
            j["ragged_secondary_input"] =
                SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedSecondaryInputs.front());
        }
    }
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void RaggedCustomLoss::deserialize(const json& j, Network* network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in RaggedCustomLoss::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "ragged_custom_loss")
        throw runtime_error("Layer type mismatch in RaggedCustomLoss::deserialize: " + j.at("layer_type").get<string>());
    if (j.at("loss_shape").get<LossShape>() != LossShape::RAW)
        throw runtime_error("Serialized RaggedCustomLoss must report raw loss.");

    RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "RaggedCustomLoss");
    if ((version == "1.1.0") != predictions.isPlacementNeutral())
        throw runtime_error("RaggedCustomLoss layer version disagrees with fixed/neutral geometry.");
    validateNeutralReference(j.at("ragged_predictions"), predictions);
    RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "RaggedCustomLoss");
    validateNeutralReference(j.at("ragged_labels"), labels);
    requireSamePartitionAndShape(predictions, labels);
    vector<RaggedTensor> secondaryInputs;
    vector<string> secondaryInputNames;
    vector<string> secondaryGradientNames;
    if (j.contains("ragged_secondary_inputs")) {
        if (predictions.isPlacementNeutral() && j.at("ragged_secondary_inputs").size() == 1 &&
            j.contains("ragged_secondary_input") &&
            j.at("ragged_secondary_input") != j.at("ragged_secondary_inputs").at(0).at("tensor")) {
            throw runtime_error("RaggedCustomLoss neutral secondary aliases must preserve the same tensor reference.");
        }
        for (const json& serializedSecondary : j.at("ragged_secondary_inputs")) {
            RaggedTensor secondary =
                SegmentedPrimitiveDetail::reconstructInput(serializedSecondary.at("tensor"), network, "RaggedCustomLoss");
            validateNeutralReference(serializedSecondary.at("tensor"), secondary);
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
        validateNeutralReference(j.at("ragged_secondary_input"), secondary);
        requireSecondaryCompatible(predictions, labels, secondary);
        secondaryInputs.push_back(secondary);
        secondaryInputNames.push_back(j.value("secondary_input_name", string("secondary_input")));
        secondaryGradientNames.push_back(j.value("secondary_gradient_name", secondaryInputNames.front() + "_grad"));
    }
    optional<RaggedTensor> exampleWeights;
    if (j.contains("ragged_example_weights")) {
        exampleWeights = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_example_weights"), network, "RaggedCustomLoss");
        validateNeutralReference(j.at("ragged_example_weights"), exampleWeights.value());
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
    if (predictions.isPlacementNeutral()) {
        const uint64_t rawId = rawValues.getOriginalId();
        if (rawId == predictions.getValues().getOriginalId() ||
            rawId == labels.getValues().getOriginalId() ||
            (exampleWeights.has_value() && rawId == exampleWeights->getValues().getOriginalId())) {
            throw runtime_error("RaggedCustomLoss raw output must own a distinct canonical tensor identity.");
        }
        for (const RaggedTensor& secondary : secondaryInputs) {
            if (rawId == secondary.getValues().getOriginalId())
                throw runtime_error("RaggedCustomLoss raw output aliases a differentiable input tensor.");
        }
        // No independent output partition or physical extent may be forged.
        if (rawJson.at("offsets").at("version").get<string>() != "1.0.0" ||
            rawJson.at("offsets").at("dimensions").get<vector<uint64_t>>() != predictions.getOffsetsDimensions() ||
            rawJson.at("offsets").at("data_type").get<DataType>() != predictions.getOffsetsDataType()) {
            throw runtime_error("RaggedCustomLoss neutral raw output must retain the canonical partition token.");
        }
    }
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
