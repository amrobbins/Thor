#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

uint64_t elementsPerValue(const RaggedTensor& ragged) {
    uint64_t elements = 1;
    for (uint64_t dim : ragged.getTrailingDimensions()) {
        elements *= dim;
    }
    return elements;
}

}  // namespace

TypeConverter::TypeConverter() = default;
TypeConverter::~TypeConverter() = default;

TypeConverter TypeConverter::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_featureInput.has_value());
    THOR_THROW_IF_FALSE(_newDataType.has_value());

    TypeConverter typeConverter;
    Tensor outputValues(_newDataType.value(), _featureInput.value().getDimensions());

    if (_raggedFeatureInput.has_value()) {
        typeConverter.raggedFeatureInput = _raggedFeatureInput.value();
        typeConverter.raggedFeatureOutput = _raggedFeatureInput->withValues(outputValues);
        typeConverter.featureInputs = {_raggedFeatureInput->getValues(), _raggedFeatureInput->getOffsets()};
    } else {
        typeConverter.featureInputs = {_featureInput.value()};
    }
    typeConverter.featureOutputs = {outputValues};
    typeConverter.initialized = true;
    typeConverter.addToNetwork(_network.value());
    return typeConverter;
}

vector<Tensor> TypeConverter::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (!raggedFeatureInput.has_value()) {
        return {featureOutputs.front()};
    }

    if (emittedFeatureOutputAfterAllInputsConnected) {
        return {};
    }
    if (connectedInputPortIndices.size() != featureInputs.size()) {
        return {};
    }

    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void TypeConverter::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInput.has_value()) {
        return;
    }
    const int port = getConnectionType(inputTensor);
    if (port < 0 || static_cast<size_t>(port) >= featureInputs.size()) {
        throw runtime_error("TypeConverter informed of connection for an invalid input port.");
    }
    connectedInputPortIndices.insert(static_cast<uint32_t>(port));
}

void TypeConverter::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int TypeConverter::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i]) {
            return static_cast<int>(i);
        }
    }
    if (!featureOutputs.empty() && connectingTensor == featureOutputs.front()) {
        return 0;
    }
    throw runtime_error("Tensor is not connected to this TypeConverter layer.");
}

uint64_t TypeConverter::getOutputTensorBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(!featureOutputs.empty());
    const uint64_t outputBytes = featureOutputs.front().getTotalSizeInBytes();
    if (raggedFeatureInput.has_value()) THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(outputBytes <= std::numeric_limits<uint64_t>::max() / batchSize);
    return outputBytes * batchSize;
}

uint64_t TypeConverter::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                               ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(!featureOutputs.empty());

    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureInputs.size() == 2);
        THOR_THROW_IF_FALSE(batchSize > 0);
        const uint64_t valuesInputBytes = featureInputs.front().getTotalSizeInBytes();
        const uint64_t offsetsCarrierBytes = featureInputs[1].getTotalSizeInBytes();
        const uint64_t outputBytes = featureOutputs.front().getTotalSizeInBytes();
        THOR_THROW_IF_FALSE(valuesInputBytes <= std::numeric_limits<uint64_t>::max() / batchSize);
        THOR_THROW_IF_FALSE(outputBytes <= std::numeric_limits<uint64_t>::max() / batchSize);
        const uint64_t physicalInputBytes = valuesInputBytes * batchSize;
        const uint64_t physicalOutputBytes = outputBytes * batchSize;
        THOR_THROW_IF_FALSE(physicalInputBytes <= std::numeric_limits<uint64_t>::max() - offsetsCarrierBytes);
        const uint64_t inputAndOffsetsBytes = physicalInputBytes + offsetsCarrierBytes;
        THOR_THROW_IF_FALSE(inputAndOffsetsBytes <= std::numeric_limits<uint64_t>::max() - physicalOutputBytes);
        return inputAndOffsetsBytes + physicalOutputBytes;
    }

    uint64_t bytes = 0;
    for (const Tensor& input : featureInputs) {
        bytes += input.getTotalSizeInBytes();
    }
    bytes += featureOutputs.front().getTotalSizeInBytes();
    return bytes * batchSize;
}

shared_ptr<ThorImplementation::Layer> TypeConverter::stamp(ThorImplementation::TensorPlacement placement,
                                                            shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                            shared_ptr<Thor::Layer> drivingApiLayer,
                                                            Thor::Tensor connectingApiTensor,
                                                            const bool inferenceOnly) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);

    bool knownInput = false;
    for (const Tensor& input : featureInputs) {
        if (connectingApiTensor == input) {
            knownInput = true;
            break;
        }
    }
    THOR_THROW_IF_FALSE(knownInput);

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    const DataType outputDataType = featureOutputs.front().getDataType();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        const RaggedTensor& raggedInput = raggedFeatureInput.value();
        const ThorImplementation::RaggedTensorDescriptor physicalInputDescriptor =
            SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
                raggedInput, drivingLayer, "TypeConverter");
        const ThorImplementation::RaggedTensorDescriptor physicalOutputDescriptor =
            raggedFeatureOutput->getDescriptorForPlacement(physicalInputDescriptor.getBatchSize());
        ThorImplementation::RaggedExpression input =
            ThorImplementation::RaggedExpression::input(
                "feature_input",
                "feature_offsets",
                physicalInputDescriptor,
                ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        ThorImplementation::RaggedExpression output = input.cast(outputDataType);
        if (output.getDescriptor() != physicalOutputDescriptor) {
            throw runtime_error("Ragged TypeConverter expression output descriptor does not match its API output.");
        }

        ExpressionDefinition definition =
            ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output.getValues()}}));
        const uint64_t rowWidth = elementsPerValue(raggedInput);
        auto physicalLayer = make_shared<ThorImplementation::RaggedCustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            vector<string>{"feature_input", "feature_offsets"},
            vector<string>{"feature_output"},
            placement,
            inferenceOnly,
            physicalInputDescriptor.getMaxTotalValues(),
            rowWidth,
            rowWidth,
            0,
            1,
            getId());
        physicalLayer->setLayerName(getLayerType());
        return physicalLayer;
    }

    const DataType inputDataType = featureInputs.front().getDataType();
    Expression input = Expression::input("feature_input", std::nullopt, inputDataType);
    Expression output = input.cast(outputDataType);
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output}}));

    // Batch-shape semantics are an API graph property, not something that can be
    // reconstructed reliably from a physical descriptor. In particular, placement
    // may insert a TensorFanout whose descriptor obscures the semantic producer.
    const bool inputDimensionsIncludeBatch =
        drivingApiLayer != nullptr && drivingApiLayer->outputTensorDimensionsIncludeBatch(connectingApiTensor);
    if (dimensionsIncludeBatch_.has_value() && dimensionsIncludeBatch_.value() != inputDimensionsIncludeBatch) {
        throw runtime_error("TypeConverter input batch-dimension contract changed across stamps.");
    }
    dimensionsIncludeBatch_ = inputDimensionsIncludeBatch;

    auto physicalLayer = make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        vector<string>{"feature_input"},
        vector<string>{"feature_output"},
        placement,
        vector<shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        getId(),
        vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor>{},
        false,
        false,
        vector<bool>{inputDimensionsIncludeBatch},
        std::nullopt);
    physicalLayer->setLayerName(getLayerType());
    return physicalLayer;
}

json TypeConverter::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);

    THOR_THROW_IF_FALSE(raggedFeatureInput.has_value() == raggedFeatureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = raggedFeatureInput.has_value() ? "1.1.0" : getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInputs.front().architectureJson();
    j["feature_output"] = featureOutputs.front().architectureJson();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["use_ragged"] = true;
        j["ragged_feature_input"] =
            SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput.value());
        j["ragged_feature_output"] =
            SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput.value());
    }

    return j;
}

void TypeConverter::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in TypeConverter::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "type_converter")
        throw runtime_error("Layer type mismatch in TypeConverter::deserialize: " + j.at("layer_type").get<string>());

    TypeConverter typeConverter;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        RaggedTensor raggedInput =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "TypeConverter");
        const uint64_t featureInputId = j.at("feature_input").at("id").get<uint64_t>();
        if (featureInputId != j.at("ragged_feature_input").at("values").at("id").get<uint64_t>()) {
            throw runtime_error("TypeConverter serialized feature_input must match its ragged input values.");
        }
        Tensor values = network->getApiTensorByOriginalId(featureInputId);
        if (values != raggedInput.getValues()) {
            throw runtime_error("TypeConverter reconstructed feature_input must match its ragged input values.");
        }

        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>()) {
            throw runtime_error("TypeConverter serialized ragged output values must match feature_output.");
        }
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            outputJson, j.at("ragged_feature_input"), raggedInput, "TypeConverter");
        // The historical feature_output and ragged output may both describe
        // packed [B*M, ...] capacity. Only the owning logical input supplies M.
        Tensor outputValues = SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
            j.at("feature_output"), outputJson, raggedInput, "TypeConverter");
        if (outputValues.getOriginalId() == values.getOriginalId()) {
            throw runtime_error("TypeConverter output values must have a distinct graph identity.");
        }
        if (outputValues.getDimensions() != raggedInput.getValuesDimensions()) {
            throw runtime_error("TypeConverter serialized ragged output logical shape must match its input.");
        }

        RaggedTensor raggedOutput = raggedInput.withValues(outputValues);
        typeConverter.raggedFeatureInput = raggedInput;
        typeConverter.raggedFeatureOutput = raggedOutput;
        typeConverter.featureInputs = {values, raggedInput.getOffsets()};
        typeConverter.featureOutputs = {outputValues};
    } else {
        if (version != "1.0.0")
            throw runtime_error("TypeConverter version 1.1.0 is reserved for placement-neutral ragged geometry.");
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        Tensor input = network->getApiTensorByOriginalId(originalTensorId);
        Tensor output = Tensor::deserialize(j.at("feature_output"));
        typeConverter.featureInputs = {input};
        typeConverter.featureOutputs = {output};
    }

    typeConverter.initialized = true;
    typeConverter.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("type_converter", &Thor::TypeConverter::deserialize);
    return true;
}();
}  // namespace
