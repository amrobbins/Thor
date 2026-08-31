#include "DeepLearning/Api/Layers/Utility/Transpose.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

uint64_t elementsPerValue(const RaggedTensor& ragged) {
    uint64_t elements = 1;
    for (uint64_t dim : ragged.getTrailingDimensions()) {
        if (elements > numeric_limits<uint64_t>::max() / dim) {
            throw invalid_argument("Ragged Transpose trailing element count overflows uint64_t.");
        }
        elements *= dim;
    }
    return elements;
}

RaggedTensor reconstructRaggedInput(const json& inputJson, Network* network) {
    Tensor values = network->getApiTensorByOriginalId(inputJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(inputJson.at("offsets").at("id").get<uint64_t>());
    RaggedTensor input = inputJson.contains("max_values_per_row")
                             ? RaggedTensor(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>())
                             : RaggedTensor(values, offsets);
    if (input.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
        input.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
        throw runtime_error("Transpose serialized ragged input metadata does not match reconstructed tensors.");
    }
    return input;
}

}  // namespace

Transpose::Transpose() = default;
Transpose::Transpose(std::optional<ThorImplementation::Expression> epilogue) : epilogue(std::move(epilogue)) {}
Transpose::~Transpose() = default;

Transpose Transpose::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_featureInput.has_value());
    if (_epilogue.has_value()) Transpose::validateEpilogueExpression(_epilogue.value());

    Transpose transpose(_epilogue);
    transpose.outputDataType = _outputDataType.value_or(_featureInput->getDataType());
    transpose.featureInput = _featureInput.value();

    if (_raggedFeatureInput.has_value()) {
        vector<uint64_t> trailing = _raggedFeatureInput->getTrailingDimensions();
        if (trailing.size() < 2) {
            throw invalid_argument("Ragged Transpose requires at least two trailing value dimensions.");
        }
        swap(trailing[trailing.size() - 2], trailing[trailing.size() - 1]);
        vector<uint64_t> outputDimensions;
        outputDimensions.reserve(trailing.size() + 1);
        outputDimensions.push_back(_raggedFeatureInput->getMaxTotalValues());
        outputDimensions.insert(outputDimensions.end(), trailing.begin(), trailing.end());
        transpose.featureOutput = Tensor(transpose.outputDataType, outputDimensions);
        transpose.raggedFeatureInput = _raggedFeatureInput;
        transpose.raggedFeatureOutput = _raggedFeatureInput->withValues(transpose.featureOutput.value());
    } else {
        vector<uint64_t> outputDimensions = _featureInput->getDimensions();
        if (outputDimensions.size() < 2) {
            throw invalid_argument("Transpose requires feature input rank >= 2.");
        }
        swap(outputDimensions[outputDimensions.size() - 2], outputDimensions[outputDimensions.size() - 1]);
        transpose.featureOutput = Tensor(transpose.outputDataType, outputDimensions);
    }

    transpose.initialized = true;
    transpose.addToNetwork(_network.value());
    return transpose;
}

vector<Tensor> Transpose::getAllInputTensors() const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    if (!raggedFeatureInput.has_value()) return {featureInput.value()};
    return {raggedFeatureInput->getValues(), raggedFeatureInput->getOffsets()};
}

vector<Tensor> Transpose::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (!raggedFeatureInput.has_value()) return {featureOutput.value()};
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != 2) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutput.value()};
}

void Transpose::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInput.has_value()) return;
    const int port = getConnectionType(inputTensor);
    if (port < 0 || port > 1) throw runtime_error("Transpose received an invalid ragged input port.");
    connectedInputPortIndices.insert(static_cast<uint32_t>(port));
}

void Transpose::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int Transpose::getConnectionType(Tensor connectingTensor) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (connectingTensor == featureInput.value()) return 0;
    if (raggedFeatureInput.has_value() && connectingTensor == raggedFeatureInput->getOffsets()) return 1;
    if (connectingTensor == featureOutput.value()) return 0;
    throw runtime_error("Tensor is not connected to this Transpose layer.");
}

shared_ptr<ThorImplementation::Layer> Transpose::stamp(ThorImplementation::TensorPlacement placement,
                                                        shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                        shared_ptr<Thor::Layer> drivingApiLayer,
                                                        Thor::Tensor connectingApiTensor,
                                                        const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    (void)getConnectionType(connectingApiTensor);

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        ThorImplementation::RaggedExpression input = ThorImplementation::RaggedExpression::input(
            "feature_input", "feature_offsets", raggedFeatureInput->getDescriptor());
        ThorImplementation::RaggedExpression output = input.transposeTrailingDimensions().cast(outputDataType);
        if (epilogue.has_value()) {
            output = output.mapValues([&](const Expression& values) {
                return Transpose::applyEpilogue(values, epilogue.value()).withOutputDType(outputDataType);
            });
        }
        if (output.getDescriptor() != raggedFeatureOutput->getDescriptor()) {
            throw runtime_error("Ragged Transpose expression output descriptor does not match its API output.");
        }
        ExpressionDefinition definition =
            ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output.getValues()}}));
        const uint64_t rowWidth = elementsPerValue(raggedFeatureInput.value());
        auto physical = make_shared<ThorImplementation::RaggedCustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            vector<string>{"feature_input", "feature_offsets"},
            vector<string>{"feature_output"},
            placement,
            inferenceOnly,
            raggedFeatureInput->getMaxTotalValues(),
            rowWidth,
            rowWidth,
            0,
            1,
            getId());
        physical->setLayerName(getLayerType());
        return physical;
    }

    Expression input = Expression::input("feature_input");
    Expression output = input.transpose().withOutputDType(outputDataType);
    if (epilogue.has_value()) {
        output = Transpose::applyEpilogue(output, epilogue.value()).withOutputDType(outputDataType);
    }
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output}}));

    auto physical = make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        vector<string>{"feature_input"},
        vector<string>{"feature_output"},
        placement,
        vector<shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly);
    physical->setLayerName("Transpose");
    return physical;
}

json Transpose::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["output_data_type"] = outputDataType;
    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value()) serializableEpilogue = makeEpilogueDefinition(epilogue.value());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }
    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    j["use_ragged"] = raggedFeatureInput.has_value();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
    return j;
}

void Transpose::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Transpose::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "transpose")
        throw runtime_error("Layer type mismatch in Transpose::deserialize: " + j.at("layer_type").get<string>());

    optional<ThorImplementation::Expression> epilogue = nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(definition);
    }

    Transpose transpose(epilogue);
    const bool useRagged = j.value("use_ragged", false);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output"));
    transpose.outputDataType = j.contains("output_data_type") ? j.at("output_data_type").get<DataType>() : featureOutput.getDataType();
    if (featureOutput.getDataType() != transpose.outputDataType) {
        throw runtime_error("Transpose::deserialize feature_output dtype must match output_data_type.");
    }

    if (useRagged) {
        RaggedTensor input = reconstructRaggedInput(j.at("ragged_feature_input"), network);
        vector<uint64_t> expectedDimensions = input.getValuesDimensions();
        if (expectedDimensions.size() < 3) {
            throw runtime_error("Ragged Transpose::deserialize requires at least two trailing value dimensions.");
        }
        swap(expectedDimensions[expectedDimensions.size() - 2], expectedDimensions[expectedDimensions.size() - 1]);
        if (featureOutput.getDimensions() != expectedDimensions) {
            throw runtime_error("Ragged Transpose::deserialize feature_output dimensions do not match the trailing-dimension transpose.");
        }
        const json& outputJson = j.at("ragged_feature_output");
        const uint64_t outputValuesId = outputJson.at("values").at("id").get<uint64_t>();
        const uint64_t outputOffsetsId = outputJson.at("offsets").at("id").get<uint64_t>();
        if (outputValuesId != featureOutput.getOriginalId() && outputValuesId != featureOutput.getId()) {
            throw runtime_error("Ragged Transpose serialized ragged output values must match feature_output.");
        }
        if ((outputOffsetsId != input.getOffsets().getOriginalId() && outputOffsetsId != input.getOffsets().getId()) ||
            outputJson.at("batch_size").get<uint64_t>() != input.getBatchSize() ||
            outputJson.at("max_total_values").get<uint64_t>() != input.getMaxTotalValues()) {
            throw runtime_error("Ragged Transpose serialized output must preserve the input row partition.");
        }
        const bool outputHasMaxValuesPerRow = outputJson.contains("max_values_per_row");
        if (outputHasMaxValuesPerRow != input.hasMaxValuesPerRow() ||
            (outputHasMaxValuesPerRow && outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow())) {
            throw runtime_error("Ragged Transpose serialized output must preserve max_values_per_row metadata.");
        }
        transpose.featureInput = input.getValues();
        transpose.featureOutput = featureOutput;
        transpose.raggedFeatureInput = input;
        transpose.raggedFeatureOutput = input.withValues(featureOutput);
    } else {
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        vector<uint64_t> expectedDimensions = featureInput.getDimensions();
        if (expectedDimensions.size() < 2) throw runtime_error("Transpose::deserialize requires feature input rank >= 2.");
        swap(expectedDimensions[expectedDimensions.size() - 2], expectedDimensions[expectedDimensions.size() - 1]);
        if (featureOutput.getDimensions() != expectedDimensions) {
            throw runtime_error("Transpose::deserialize feature_output dimensions do not match feature_input trailing-dimension transpose.");
        }
        transpose.featureInput = featureInput;
        transpose.featureOutput = featureOutput;
    }

    transpose.initialized = true;
    transpose.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("transpose", &Thor::Transpose::deserialize);
    return true;
}();
}  // namespace
