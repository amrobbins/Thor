#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Network/Network.h"

#include <limits>
#include <stdexcept>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

vector<uint64_t> flattenedDimensions(const vector<uint64_t>& inputDimensions, uint32_t numOutputDimensions, const char* what) {
    if (inputDimensions.empty() || numOutputDimensions == 0 || numOutputDimensions >= inputDimensions.size()) {
        throw invalid_argument(string("Flatten ") + what + " requires 0 < numOutputDimensions < input rank.");
    }
    vector<uint64_t> outputDimensions;
    outputDimensions.reserve(numOutputDimensions);
    for (uint32_t i = 0; i < inputDimensions.size(); ++i) {
        if (i < numOutputDimensions) {
            outputDimensions.push_back(inputDimensions[i]);
        } else {
            if (inputDimensions[i] == 0 || outputDimensions.back() > numeric_limits<uint64_t>::max() / inputDimensions[i]) {
                throw invalid_argument(string("Flatten ") + what + " element count overflows uint64_t.");
            }
            outputDimensions.back() *= inputDimensions[i];
        }
    }
    return outputDimensions;
}

RaggedTensor reconstructRaggedInput(const json& inputJson, Network* network) {
    Tensor values = network->getApiTensorByOriginalId(inputJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(inputJson.at("offsets").at("id").get<uint64_t>());
    RaggedTensor input = inputJson.contains("max_values_per_row")
                             ? RaggedTensor(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>())
                             : RaggedTensor(values, offsets);
    if (input.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
        input.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
        throw runtime_error("Flatten serialized ragged input metadata does not match reconstructed tensors.");
    }
    return input;
}

void validateRaggedOutput(const json& outputJson, const RaggedTensor& input, const Tensor& outputValues) {
    const uint64_t serializedValuesId = outputJson.at("values").at("id").get<uint64_t>();
    const uint64_t serializedOffsetsId = outputJson.at("offsets").at("id").get<uint64_t>();
    if (serializedValuesId != outputValues.getOriginalId() && serializedValuesId != outputValues.getId()) {
        throw runtime_error("Flatten serialized ragged output values must match feature_output.");
    }
    if (serializedOffsetsId != input.getOffsets().getOriginalId() && serializedOffsetsId != input.getOffsets().getId()) {
        throw runtime_error("Flatten serialized ragged output must preserve the input row partition.");
    }
    if (outputJson.at("batch_size").get<uint64_t>() != input.getBatchSize() ||
        outputJson.at("max_total_values").get<uint64_t>() != input.getMaxTotalValues()) {
        throw runtime_error("Flatten serialized ragged output capacity metadata does not match the input partition.");
    }
    const bool outputHasMaxValuesPerRow = outputJson.contains("max_values_per_row");
    if (outputHasMaxValuesPerRow != input.hasMaxValuesPerRow() ||
        (outputHasMaxValuesPerRow && outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow())) {
        throw runtime_error("Serialized ragged output must preserve max_values_per_row metadata.");
    }
}

}  // namespace

Flatten Flatten::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_featureInput.has_value());
    THOR_THROW_IF_FALSE(_numOutputDimensions.has_value());

    Flatten flatten;
    flatten.featureInput = _featureInput;
    if (_raggedFeatureInput.has_value()) {
        const vector<uint64_t> inputTrailing = _raggedFeatureInput->getTrailingDimensions();
        const vector<uint64_t> outputTrailing = flattenedDimensions(inputTrailing, _numOutputDimensions.value(), "ragged input");
        vector<uint64_t> outputValueDimensions;
        outputValueDimensions.reserve(outputTrailing.size() + 1);
        outputValueDimensions.push_back(_raggedFeatureInput->getMaxTotalValues());
        outputValueDimensions.insert(outputValueDimensions.end(), outputTrailing.begin(), outputTrailing.end());
        flatten.featureOutput = Tensor(_featureInput->getDataType(), outputValueDimensions);
        flatten.raggedFeatureInput = _raggedFeatureInput;
        flatten.raggedFeatureOutput = _raggedFeatureInput->withValues(flatten.featureOutput.value());
    } else {
        const vector<uint64_t> outputDimensions =
            flattenedDimensions(_featureInput->getDimensions(), _numOutputDimensions.value(), "dense input");
        flatten.featureOutput = Tensor(_featureInput->getDataType(), outputDimensions);
    }
    flatten.initialized = true;
    flatten.addToNetwork(_network.value());
    return flatten;
}

json Flatten::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
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

void Flatten::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Flatten::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "flatten")
        throw runtime_error("Layer type mismatch in Flatten::deserialize: " + j.at("layer_type").get<string>());

    Flatten flatten;
    if (j.value("use_ragged", false)) {
        RaggedTensor input = reconstructRaggedInput(j.at("ragged_feature_input"), network);
        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        const vector<uint64_t> outputDimensions = outputValues.getDimensions();
        if (outputDimensions.size() < 2 || outputDimensions.front() != input.getMaxTotalValues()) {
            throw runtime_error("Ragged Flatten serialized output must preserve the packed capacity dimension.");
        }
        const vector<uint64_t> inputTrailing = input.getTrailingDimensions();
        const vector<uint64_t> outputTrailing(outputDimensions.begin() + 1, outputDimensions.end());
        if (outputTrailing.empty() || outputTrailing.size() >= inputTrailing.size()) {
            throw runtime_error("Ragged Flatten serialized output must reduce the trailing value rank.");
        }
        const vector<uint64_t> expectedTrailing =
            flattenedDimensions(inputTrailing, static_cast<uint32_t>(outputTrailing.size()), "serialized ragged input");
        if (expectedTrailing != outputTrailing) {
            throw runtime_error("Ragged Flatten serialized output dimensions are inconsistent with its input.");
        }
        validateRaggedOutput(j.at("ragged_feature_output"), input, outputValues);
        flatten.featureInput = input.getValues();
        flatten.featureOutput = outputValues;
        flatten.raggedFeatureInput = input;
        flatten.raggedFeatureOutput = input.withValues(outputValues);
    } else {
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output"));
        const vector<uint64_t> expectedDimensions =
            flattenedDimensions(featureInput.getDimensions(), static_cast<uint32_t>(featureOutput.getDimensions().size()), "serialized dense input");
        if (featureOutput.getDimensions() != expectedDimensions) {
            throw runtime_error("Flatten serialized output dimensions are inconsistent with its input.");
        }
        flatten.featureInput = featureInput;
        flatten.featureOutput = featureOutput;
    }
    flatten.initialized = true;
    flatten.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("flatten", &Thor::Flatten::deserialize);
    return true;
}();
}  // namespace
