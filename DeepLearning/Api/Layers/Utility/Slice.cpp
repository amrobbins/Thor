#include "DeepLearning/Api/Layers/Utility/Slice.h"

#include "DeepLearning/Implementation/ThorError.h"

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

    const std::vector<uint64_t>& inputDimensions = _featureInput->getDimensions();
    if (_axis.value() >= inputDimensions.size()) {
        throw std::invalid_argument("Slice logical axis is out of range. Batch is excluded from Slice axes.");
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
    slice.featureInput = _featureInput.value();
    slice.featureOutput = Tensor(_featureInput->getDataType(), outputDimensions);
    slice.initialized = true;
    slice.addToNetwork(_network.value());
    return slice;
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
    const uint64_t effectiveBatchSize = std::max<uint64_t>(1, batchSize);
    if (perBatchBytes > std::numeric_limits<uint64_t>::max() / effectiveBatchSize)
        throw std::overflow_error("Slice memory requirement overflow.");
    return perBatchBytes * effectiveBatchSize;
}

json Slice::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    return json{
        {"factory", Layer::Factory::Layer.value()},
        {"version", getLayerVersion()},
        {"layer_type", "slice"},
        {"axis", axis},
        {"start", start},
        {"length", length},
        {"feature_input", featureInput->architectureJson()},
        {"feature_output", featureOutput->architectureJson()},
    };
}

void Slice::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in Slice::deserialize: " + j.at("version").get<std::string>());
    if (j.at("layer_type").get<std::string>() != "slice")
        throw std::runtime_error("Layer type mismatch in Slice::deserialize: " + j.at("layer_type").get<std::string>());

    const uint64_t inputOriginalId = j.at("feature_input").at("id").get<uint64_t>();
    Tensor input = network->getApiTensorByOriginalId(inputOriginalId);
    Tensor serializedOutput = Tensor::deserialize(j.at("feature_output"));

    const uint64_t axis = j.at("axis").get<uint64_t>();
    const int64_t start = j.at("start").get<int64_t>();
    const uint64_t length = j.at("length").get<uint64_t>();
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
    if (serializedOutput.getDimensions() != expectedOutputDimensions ||
        serializedOutput.getDataType() != input.getDataType()) {
        throw std::runtime_error("Slice serialized output descriptor does not match axis/start/length.");
    }

    Slice slice;
    slice.axis = axis;
    slice.start = start;
    slice.length = length;
    slice.featureInput = input;
    slice.featureOutput = serializedOutput;
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
