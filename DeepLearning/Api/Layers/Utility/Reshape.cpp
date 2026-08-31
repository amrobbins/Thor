#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Network/Network.h"

#include <limits>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

uint64_t checkedElementCount(const vector<uint64_t>& dimensions, const char* what) {
    uint64_t count = 1;
    for (uint64_t dim : dimensions) {
        if (dim == 0 || count > numeric_limits<uint64_t>::max() / dim) {
            throw invalid_argument(string("Reshape ") + what + " must contain positive dimensions with a representable element count.");
        }
        count *= dim;
    }
    return count;
}

RaggedTensor reconstructRaggedInput(const json& inputJson, Network* network) {
    const uint64_t valuesId = inputJson.at("values").at("id").get<uint64_t>();
    const uint64_t offsetsId = inputJson.at("offsets").at("id").get<uint64_t>();
    Tensor values = network->getApiTensorByOriginalId(valuesId);
    Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
    RaggedTensor input = inputJson.contains("max_values_per_row")
                             ? RaggedTensor(values, offsets, inputJson.at("max_values_per_row").get<uint64_t>())
                             : RaggedTensor(values, offsets);
    if (input.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
        input.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
        throw runtime_error("Reshape serialized ragged input metadata does not match reconstructed tensors.");
    }
    return input;
}

void validateSerializedRaggedOutput(const json& outputJson, const RaggedTensor& input, const Tensor& outputValues) {
    if (outputJson.at("values").at("id").get<uint64_t>() != outputValues.getOriginalId() &&
        outputJson.at("values").at("id").get<uint64_t>() != outputValues.getId()) {
        throw runtime_error("Reshape serialized ragged output values must match feature_output.");
    }
    if (outputJson.at("offsets").at("id").get<uint64_t>() != input.getOffsets().getOriginalId() &&
        outputJson.at("offsets").at("id").get<uint64_t>() != input.getOffsets().getId()) {
        throw runtime_error("Reshape serialized ragged output must preserve the input row partition.");
    }
    if (outputJson.at("batch_size").get<uint64_t>() != input.getBatchSize() ||
        outputJson.at("max_total_values").get<uint64_t>() != input.getMaxTotalValues()) {
        throw runtime_error("Reshape serialized ragged output capacity metadata does not match the input partition.");
    }
    const bool outputHasMaxValuesPerRow = outputJson.contains("max_values_per_row");
    if (outputHasMaxValuesPerRow != input.hasMaxValuesPerRow() ||
        (outputHasMaxValuesPerRow && outputJson.at("max_values_per_row").get<uint64_t>() != input.getMaxValuesPerRow())) {
        throw runtime_error("Serialized ragged output must preserve max_values_per_row metadata.");
    }
}

}  // namespace

Reshape::Reshape() = default;
Reshape::~Reshape() = default;

Reshape Reshape::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_featureInput.has_value());
    THOR_THROW_IF_FALSE(_newDimensions.has_value());

    Reshape reshape;
    reshape.featureInput = _featureInput;

    if (_raggedFeatureInput.has_value()) {
        const vector<uint64_t> trailingInputDimensions = _raggedFeatureInput->getTrailingDimensions();
        if (checkedElementCount(trailingInputDimensions, "ragged input trailing shape") !=
            checkedElementCount(_newDimensions.value(), "ragged output trailing shape")) {
            throw invalid_argument("Ragged Reshape must preserve the number of elements in each packed value.");
        }
        vector<uint64_t> outputValueDimensions;
        outputValueDimensions.reserve(_newDimensions->size() + 1);
        outputValueDimensions.push_back(_raggedFeatureInput->getMaxTotalValues());
        outputValueDimensions.insert(outputValueDimensions.end(), _newDimensions->begin(), _newDimensions->end());
        reshape.featureOutput = Tensor(_featureInput->getDataType(), outputValueDimensions);
        reshape.raggedFeatureInput = _raggedFeatureInput;
        reshape.raggedFeatureOutput = _raggedFeatureInput->withValues(reshape.featureOutput.value());
        // Ragged packed values already include their full-capacity leading row dimension.
        reshape.newDimensions = outputValueDimensions;
    } else {
        reshape.featureOutput = Tensor(_featureInput->getDataType(), _newDimensions.value());
        if (reshape.featureInput->getTotalNumElements() != reshape.featureOutput->getTotalNumElements()) {
            throw invalid_argument("Reshape input and output must contain the same number of elements.");
        }
        // Dense implementation has one extra physical batch dimension. A leading zero means
        // copy the runtime batch size from featureInput.
        reshape.newDimensions = {0};
        reshape.newDimensions.insert(reshape.newDimensions.end(), _newDimensions->begin(), _newDimensions->end());
    }

    reshape.initialized = true;
    reshape.addToNetwork(_network.value());
    return reshape;
}

json Reshape::architectureJson() const {
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

void Reshape::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Reshape::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "reshape")
        throw runtime_error("Layer type mismatch in Reshape::deserialize: " + j.at("layer_type").get<string>());

    Reshape reshape;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        RaggedTensor input = reconstructRaggedInput(j.at("ragged_feature_input"), network);
        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        const vector<uint64_t> outputDimensions = outputValues.getDimensions();
        if (outputDimensions.size() < 2 || outputDimensions.front() != input.getMaxTotalValues()) {
            throw runtime_error("Ragged Reshape serialized output must preserve the packed capacity dimension.");
        }
        vector<uint64_t> outputTrailing(outputDimensions.begin() + 1, outputDimensions.end());
        if (checkedElementCount(input.getTrailingDimensions(), "serialized ragged input trailing shape") !=
            checkedElementCount(outputTrailing, "serialized ragged output trailing shape")) {
            throw runtime_error("Ragged Reshape serialized output must preserve elements per packed value.");
        }
        validateSerializedRaggedOutput(j.at("ragged_feature_output"), input, outputValues);
        reshape.featureInput = input.getValues();
        reshape.featureOutput = outputValues;
        reshape.raggedFeatureInput = input;
        reshape.raggedFeatureOutput = input.withValues(outputValues);
        reshape.newDimensions = outputDimensions;
    } else {
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
        Tensor featureOutput = Tensor::deserialize(j.at("feature_output"));
        if (featureInput.getTotalNumElements() != featureOutput.getTotalNumElements()) {
            throw runtime_error("In Reshape::deserialize, input and output element counts must match.");
        }
        reshape.featureInput = featureInput;
        reshape.featureOutput = featureOutput;
        reshape.newDimensions = {0U};
        const vector<uint64_t> outputDimensions = featureOutput.getDimensions();
        reshape.newDimensions.insert(reshape.newDimensions.end(), outputDimensions.begin(), outputDimensions.end());
    }
    reshape.initialized = true;
    reshape.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("reshape", &Thor::Reshape::deserialize);
    return true;
}();
}  // namespace
