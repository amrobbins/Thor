#include "DeepLearning/Api/Layers/Utility/Slice.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::json;

namespace Thor {

uint64_t Slice::normalizeStart(int64_t requestedStart, uint64_t axisLength) {
    int64_t normalized = requestedStart;
    if (normalized < 0) {
        if (axisLength > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("Slice cannot normalize a negative start for an axis larger than INT64_MAX.");
        }
        normalized += static_cast<int64_t>(axisLength);
    }
    if (normalized < 0 || static_cast<uint64_t>(normalized) > axisLength) {
        throw std::invalid_argument("Slice start is outside the selected logical axis.");
    }
    return static_cast<uint64_t>(normalized);
}

Slice Slice::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_axis.has_value() || !_start.has_value() || !_length.has_value()) {
        throw std::runtime_error("Slice requires network, featureInput, axis, start, and length.");
    }
    if (_length.value() == 0) {
        throw std::invalid_argument("Slice length must be greater than zero.");
    }

    const std::vector<uint64_t> inputDimensions = _raggedFeatureInput.has_value()
                                                      ? _raggedFeatureInput->getTrailingDimensions()
                                                      : _featureInput->getDimensions();
    if (_axis.value() >= inputDimensions.size()) {
        throw std::invalid_argument(_raggedFeatureInput.has_value()
                                        ? "Slice trailing axis is out of range for the RaggedTensor value shape."
                                        : "Slice logical axis is out of range. Batch is excluded from Slice axes.");
    }
    const uint64_t normalizedStart = Slice::normalizeStart(_start.value(), inputDimensions[_axis.value()]);
    if (_length.value() > inputDimensions[_axis.value()] - normalizedStart) {
        throw std::invalid_argument("Slice start + length exceeds the selected logical axis.");
    }

    std::vector<uint64_t> outputDimensions = inputDimensions;
    outputDimensions[_axis.value()] = _length.value();

    Slice slice;
    slice.axis = _axis.value();
    slice.start = _start.value();
    slice.length = _length.value();
    if (_raggedFeatureInput.has_value()) {
        slice.raggedFeatureInput = _raggedFeatureInput.value();
        std::vector<uint64_t> outputValueDimensions;
        outputValueDimensions.reserve(outputDimensions.size() + 1);
        outputValueDimensions.push_back(_raggedFeatureInput->getMaxTotalValues());
        outputValueDimensions.insert(outputValueDimensions.end(), outputDimensions.begin(), outputDimensions.end());
        Tensor outputValues(_featureInput->getDataType(), outputValueDimensions);
        slice.raggedFeatureOutput = _raggedFeatureInput->withValues(outputValues);
        slice.featureInput = _raggedFeatureInput->getValues();
        slice.featureOutput = outputValues;
    } else {
        slice.featureInput = _featureInput.value();
        slice.featureOutput = Tensor(_featureInput->getDataType(), outputDimensions);
    }
    slice.initialized = true;
    slice.addToNetwork(_network.value());
    return slice;
}


std::vector<Tensor> Slice::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (!raggedFeatureInput.has_value()) {
        return {featureOutput.value()};
    }
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != 2) {
        return {};
    }
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutput.value()};
}

void Slice::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInput.has_value()) return;
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void Slice::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int Slice::getConnectionType(Tensor connectingTensor) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (connectingTensor == featureInput.value()) return 0;
    if (raggedFeatureInput.has_value() && connectingTensor == raggedFeatureInput->getOffsets()) return 1;
    if (connectingTensor == featureOutput.value()) return 0;
    throw std::runtime_error("Tensor is not connected to this Slice layer.");
}

uint64_t Slice::getOutputTensorBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    const uint64_t outputBytes = featureOutput->getTotalSizeInBytes();
    if (raggedFeatureInput.has_value()) return outputBytes;
    if (batchSize != 0 && outputBytes > std::numeric_limits<uint64_t>::max() / batchSize) {
        throw std::overflow_error("Slice output size overflows uint64_t.");
    }
    return outputBytes * batchSize;
}

std::shared_ptr<ThorImplementation::Layer> Slice::stamp(ThorImplementation::TensorPlacement placement,
                                                        std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                        std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                        Thor::Tensor connectingApiTensor,
                                                        bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    bool knownInput = connectingApiTensor == featureInput.value();
    if (raggedFeatureInput.has_value() && connectingApiTensor == raggedFeatureInput->getOffsets()) knownInput = true;
    THOR_THROW_IF_FALSE(knownInput);

    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        const RaggedTensor& raggedInput = raggedFeatureInput.value();
        const std::vector<uint64_t> trailingDimensions = raggedInput.getTrailingDimensions();
        THOR_THROW_IF_FALSE(axis < trailingDimensions.size());
        const uint64_t normalizedStart = Slice::normalizeStart(start, trailingDimensions[axis]);

        ThorImplementation::RaggedExpression input = ThorImplementation::RaggedExpression::input(
            "feature_input", "feature_offsets", raggedInput.getDescriptor());
        ThorImplementation::RaggedExpression output = input.sliceTrailingDimension(axis, normalizedStart, length);
        if (output.getDescriptor() != raggedFeatureOutput->getDescriptor()) {
            throw std::runtime_error("Ragged Slice expression output descriptor does not match its API output.");
        }
        ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
            ThorImplementation::Expression::outputs({{"feature_output", output.getValues()}}));

        auto elementsPerValue = [](const std::vector<uint64_t>& dimensions) {
            uint64_t elements = 1;
            for (uint64_t dimension : dimensions) elements *= dimension;
            return elements;
        };
        auto physicalSlice = std::make_shared<ThorImplementation::RaggedCustomLayer>(
            ThorImplementation::DynamicExpression::fromExpressionDefinition(definition),
            std::vector<std::string>{"feature_input", "feature_offsets"},
            std::vector<std::string>{"feature_output"},
            placement,
            inferenceOnly,
            raggedInput.getMaxTotalValues(),
            elementsPerValue(raggedInput.getTrailingDimensions()),
            elementsPerValue(raggedFeatureOutput->getTrailingDimensions()),
            0,
            1,
            getId());
        physicalSlice->setLayerName("Slice");
        return physicalSlice;
    }

    THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());
    const uint64_t logicalAxis = axis;
    const int64_t requestedStart = start;
    const uint64_t requestedLength = length;

    ThorImplementation::DynamicExpression expression(
        std::vector<std::string>{"feature_input"},
        std::vector<std::string>{"feature_output"},
        [logicalAxis, requestedStart, requestedLength](const ThorImplementation::DynamicExpression::TensorMap& inputs,
                                                       const ThorImplementation::DynamicExpression::TensorMap& outputs,
                                                       Stream& stream) {
            auto inputIt = inputs.find("feature_input");
            if (inputIt == inputs.end())
                throw std::runtime_error("Slice runtime expression is missing feature_input.");

            const ThorImplementation::Tensor& input = inputIt->second;
            std::vector<uint64_t> dimensions = input.getDimensions();
            if (dimensions.empty() || logicalAxis + 1 >= dimensions.size()) {
                throw std::runtime_error("Slice runtime input rank does not match its logical axis.");
            }
            const uint64_t physicalAxis = logicalAxis + 1;
            const uint64_t normalizedStart = Slice::normalizeStart(requestedStart, dimensions[physicalAxis]);
            if (requestedLength > dimensions[physicalAxis] - normalizedStart) {
                throw std::runtime_error("Slice runtime start + length exceeds the selected axis.");
            }

            const std::vector<uint64_t> strides = input.getStridesElements();
            if (strides.size() != dimensions.size())
                throw std::runtime_error("Slice runtime input has invalid stride metadata.");
            if (normalizedStart != 0 && strides[physicalAxis] > std::numeric_limits<uint64_t>::max() / normalizedStart)
                throw std::overflow_error("Slice runtime storage offset overflow.");
            const uint64_t elementOffset = normalizedStart * strides[physicalAxis];
            dimensions[physicalAxis] = requestedLength;

            ThorImplementation::Expression featureInput = ThorImplementation::Expression::input("feature_input");
            ThorImplementation::Expression featureOutput = featureInput.stridedView(dimensions, strides, elementOffset);
            ThorImplementation::Outputs expressionOutputs =
                ThorImplementation::Expression::outputs({{"feature_output", featureOutput}});
            auto definition = std::make_shared<ThorImplementation::ExpressionDefinition>(
                ThorImplementation::ExpressionDefinition::fromOutputs(expressionOutputs));

            return ThorImplementation::DynamicExpressionBuild{
                .equation = std::make_shared<ThorImplementation::FusedEquation>(
                    ThorImplementation::FusedEquation::compile(definition->outputs, stream.getGpuNum())),
                .stamp_inputs = inputs,
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = {},
                .serialized_definition = nullptr,
            };
        });

    std::vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor> declaredOutputs{
        ThorImplementation::CustomLayer::DeclaredOutputDescriptor{featureOutput->getDataType(), featureOutput->getDimensions()}};
    auto physicalSlice = std::make_shared<ThorImplementation::CustomLayer>(
        std::move(expression),
        std::vector<std::string>{"feature_input"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        Layer::getId(),
        std::move(declaredOutputs));
    physicalSlice->setLayerName("Slice");
    return physicalSlice;
}

uint64_t Slice::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                      ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    const uint64_t inputBytes = featureInput->getTotalSizeInBytes();
    const uint64_t outputBytes = featureOutput->getTotalSizeInBytes();
    if (inputBytes > std::numeric_limits<uint64_t>::max() - outputBytes)
        throw std::overflow_error("Slice per-batch memory requirement overflow.");
    const uint64_t perBatchBytes = inputBytes + outputBytes;
    if (raggedFeatureInput.has_value()) {
        const uint64_t offsetsBytes = raggedFeatureInput->getOffsets().getTotalSizeInBytes();
        if (perBatchBytes > std::numeric_limits<uint64_t>::max() - offsetsBytes)
            throw std::overflow_error("Ragged Slice memory requirement overflow.");
        return perBatchBytes + offsetsBytes;
    }
    const uint64_t effectiveBatchSize = std::max<uint64_t>(1, batchSize);
    if (perBatchBytes > std::numeric_limits<uint64_t>::max() / effectiveBatchSize)
        throw std::overflow_error("Slice memory requirement overflow.");
    return perBatchBytes * effectiveBatchSize;
}

json Slice::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j{
        {"factory", Layer::Factory::Layer.value()},
        {"version", getLayerVersion()},
        {"layer_type", "slice"},
        {"axis", axis},
        {"start", start},
        {"length", length},
        {"feature_input", featureInput->architectureJson()},
        {"feature_output", featureOutput->architectureJson()},
    };
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["use_ragged"] = true;
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
    return j;
}

void Slice::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in Slice::deserialize: " + j.at("version").get<std::string>());
    if (j.at("layer_type").get<std::string>() != "slice")
        throw std::runtime_error("Layer type mismatch in Slice::deserialize: " + j.at("layer_type").get<std::string>());

    Slice slice;
    const bool useRagged = j.value("use_ragged", false);
    const uint64_t axis = j.at("axis").get<uint64_t>();
    const int64_t start = j.at("start").get<int64_t>();
    const uint64_t length = j.at("length").get<uint64_t>();

    if (useRagged) {
        const json& inputJson = j.at("ragged_feature_input");
        const uint64_t valuesId = inputJson.at("values").at("id").get<uint64_t>();
        const uint64_t offsetsId = inputJson.at("offsets").at("id").get<uint64_t>();
        Tensor values = network->getApiTensorByOriginalId(valuesId);
        Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
        RaggedTensor raggedInput = inputJson.contains("max_values_per_row")
            ? RaggedTensor(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>())
            : RaggedTensor(values, offsets);
        if (raggedInput.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
            raggedInput.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
            throw std::runtime_error("Slice serialized ragged input metadata does not match reconstructed tensors.");
        }
        const std::vector<uint64_t> trailingDimensions = raggedInput.getTrailingDimensions();
        if (axis >= trailingDimensions.size() || length == 0) {
            throw std::runtime_error("Slice serialized trailing axis/length is invalid for the ragged feature input.");
        }
        const uint64_t normalizedStart = Slice::normalizeStart(start, trailingDimensions[axis]);
        if (length > trailingDimensions[axis] - normalizedStart) {
            throw std::runtime_error("Slice serialized start + length exceeds the selected ragged trailing axis.");
        }

        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        RaggedTensor raggedOutput = raggedInput.withValues(outputValues);
        std::vector<uint64_t> expectedTrailing = trailingDimensions;
        expectedTrailing[axis] = length;
        if (raggedOutput.getTrailingDimensions() != expectedTrailing || outputValues.getDataType() != values.getDataType()) {
            throw std::runtime_error("Slice serialized ragged output descriptor does not match axis/start/length.");
        }
        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>() ||
            outputJson.at("offsets").at("id").get<uint64_t>() != offsetsId ||
            outputJson.at("batch_size").get<uint64_t>() != raggedInput.getBatchSize() ||
            outputJson.at("max_total_values").get<uint64_t>() != raggedInput.getMaxTotalValues()) {
            throw std::runtime_error("Slice serialized ragged output must reference feature_output values and preserve the input partition.");
        }

        slice.raggedFeatureInput = raggedInput;
        slice.raggedFeatureOutput = raggedOutput;
        slice.featureInput = values;
        slice.featureOutput = outputValues;
    } else {
        const uint64_t inputOriginalId = j.at("feature_input").at("id").get<uint64_t>();
        Tensor input = network->getApiTensorByOriginalId(inputOriginalId);
        Tensor serializedOutput = Tensor::deserialize(j.at("feature_output"));
        const std::vector<uint64_t>& inputDimensions = input.getDimensions();
        if (axis >= inputDimensions.size() || length == 0) {
            throw std::runtime_error("Slice serialized axis/length is invalid for the feature input.");
        }
        const uint64_t normalizedStart = Slice::normalizeStart(start, inputDimensions[axis]);
        if (length > inputDimensions[axis] - normalizedStart) {
            throw std::runtime_error("Slice serialized start + length exceeds the selected logical axis.");
        }
        std::vector<uint64_t> expectedOutputDimensions = inputDimensions;
        expectedOutputDimensions[axis] = length;
        if (serializedOutput.getDimensions() != expectedOutputDimensions || serializedOutput.getDataType() != input.getDataType()) {
            throw std::runtime_error("Slice serialized output descriptor does not match axis/start/length.");
        }
        slice.featureInput = input;
        slice.featureOutput = serializedOutput;
    }

    slice.axis = axis;
    slice.start = start;
    slice.length = length;
    slice.initialized = true;
    slice.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("slice", &Thor::Slice::deserialize);
    return true;
}();
}  // namespace
