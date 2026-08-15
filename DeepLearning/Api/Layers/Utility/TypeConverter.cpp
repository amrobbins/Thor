#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedExpressionLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/RaggedExpression.h"

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
        typeConverter.raggedFeatureOutput = RaggedTensor(outputValues, _raggedFeatureInput->getOffsets());
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
    return raggedFeatureInput.has_value() ? outputBytes : outputBytes * batchSize;
}

uint64_t TypeConverter::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                               ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(!featureOutputs.empty());

    uint64_t bytes = 0;
    for (const Tensor& input : featureInputs) {
        bytes += input.getTotalSizeInBytes();
    }
    bytes += featureOutputs.front().getTotalSizeInBytes();
    return raggedFeatureInput.has_value() ? bytes : bytes * batchSize;
}

shared_ptr<ThorImplementation::Layer> TypeConverter::stamp(ThorImplementation::TensorPlacement placement,
                                                            shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                            shared_ptr<Thor::Layer> drivingApiLayer,
                                                            Thor::Tensor connectingApiTensor,
                                                            const bool inferenceOnly) const {
    (void)drivingLayer;
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
        ThorImplementation::RaggedExpression input =
            ThorImplementation::RaggedExpression::input("feature_input", "feature_offsets", raggedInput.getDescriptor());
        ThorImplementation::RaggedExpression output = input.cast(outputDataType);
        if (output.getDescriptor() != raggedFeatureOutput->getDescriptor()) {
            throw runtime_error("Ragged TypeConverter expression output descriptor does not match its API output.");
        }

        ExpressionDefinition definition =
            ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output.getValues()}}));
        const uint64_t rowWidth = elementsPerValue(raggedInput);
        auto physicalLayer = make_shared<ThorImplementation::RaggedExpressionLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            vector<string>{"feature_input", "feature_offsets"},
            vector<string>{"feature_output"},
            placement,
            inferenceOnly,
            raggedInput.getMaxTotalValues(),
            rowWidth,
            rowWidth,
            0,
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

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInputs.front().architectureJson();
    j["feature_output"] = featureOutputs.front().architectureJson();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["use_ragged"] = true;
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }

    return j;
}

void TypeConverter::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in TypeConverter::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "type_converter")
        throw runtime_error("Layer type mismatch in TypeConverter::deserialize: " + j.at("layer_type").get<string>());

    TypeConverter typeConverter;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        const json& inputJson = j.at("ragged_feature_input");
        const uint64_t valuesId = inputJson.at("values").at("id").get<uint64_t>();
        const uint64_t offsetsId = inputJson.at("offsets").at("id").get<uint64_t>();
        Tensor values = network->getApiTensorByOriginalId(valuesId);
        Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
        RaggedTensor raggedInput(values, offsets);
        if (raggedInput.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
            raggedInput.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
            throw runtime_error("TypeConverter serialized ragged input metadata does not match reconstructed tensors.");
        }

        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        RaggedTensor raggedOutput(outputValues, offsets);
        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>()) {
            throw runtime_error("TypeConverter serialized ragged output values must match feature_output.");
        }
        if (outputJson.at("offsets").at("id").get<uint64_t>() != offsetsId) {
            throw runtime_error("TypeConverter serialized ragged output must preserve the input row partition.");
        }
        if (raggedOutput.getBatchSize() != outputJson.at("batch_size").get<uint64_t>() ||
            raggedOutput.getMaxTotalValues() != outputJson.at("max_total_values").get<uint64_t>()) {
            throw runtime_error("TypeConverter serialized ragged output metadata does not match reconstructed tensors.");
        }

        typeConverter.raggedFeatureInput = raggedInput;
        typeConverter.raggedFeatureOutput = raggedOutput;
        typeConverter.featureInputs = {values, offsets};
        typeConverter.featureOutputs = {outputValues};
    } else {
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
