#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
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
        outputValueDimensions.push_back(_raggedFeatureInput->getMaxValuesPerRow());
        outputValueDimensions.insert(outputValueDimensions.end(), _newDimensions->begin(), _newDimensions->end());
        reshape.featureOutput = Tensor(_featureInput->getDataType(), outputValueDimensions);
        reshape.raggedFeatureInput = _raggedFeatureInput;
        reshape.raggedFeatureOutput = _raggedFeatureInput->withValues(reshape.featureOutput.value());
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
    j["version"] = raggedFeatureInput.has_value() ? "1.1.0" : getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    j["use_ragged"] = raggedFeatureInput.has_value();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_input"] =
            SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput.value());
        j["ragged_feature_output"] =
            SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput.value());
    }
    return j;
}

void Reshape::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in Reshape::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "reshape")
        throw runtime_error("Layer type mismatch in Reshape::deserialize: " + j.at("layer_type").get<string>());

    Reshape reshape;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "Reshape");
        const uint64_t featureInputId = j.at("feature_input").at("id").get<uint64_t>();
        if (featureInputId != j.at("ragged_feature_input").at("values").at("id").get<uint64_t>()) {
            throw runtime_error("Reshape serialized feature_input must match its ragged input values.");
        }
        Tensor featureInput = network->getApiTensorByOriginalId(featureInputId);
        if (featureInput != input.getValues()) {
            throw runtime_error("Reshape reconstructed feature_input must match its ragged input values.");
        }

        Tensor outputValues = SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
            j.at("feature_output"), j.at("ragged_feature_output"), input, "Reshape");
        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>()) {
            throw runtime_error("Reshape serialized ragged output values must match feature_output.");
        }
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            outputJson, j.at("ragged_feature_input"), input, "Reshape");

        const vector<uint64_t> outputDimensions = outputValues.getDimensions();
        const uint64_t expectedLeading = input.getMaxValuesPerRow();
        if (outputDimensions.size() < 2 || outputDimensions.front() != expectedLeading) {
            throw runtime_error("Ragged Reshape serialized output must preserve the logical packed-row dimension.");
        }
        vector<uint64_t> outputTrailing(outputDimensions.begin() + 1, outputDimensions.end());
        if (checkedElementCount(input.getTrailingDimensions(), "serialized ragged input trailing shape") !=
            checkedElementCount(outputTrailing, "serialized ragged output trailing shape")) {
            throw runtime_error("Ragged Reshape serialized output must preserve elements per packed value.");
        }
        if (outputValues.getDataType() != input.getValuesDataType()) {
            throw runtime_error("Ragged Reshape serialized output dtype must match its input.");
        }

        reshape.featureInput = featureInput;
        reshape.featureOutput = outputValues;
        reshape.raggedFeatureInput = input;
        reshape.raggedFeatureOutput = input.withValues(outputValues);
        // stamp() derives the concrete [B*M, ...] shape from logical M.
        reshape.newDimensions = outputDimensions;
    } else {
        if (version != "1.0.0")
            throw runtime_error("Reshape version 1.1.0 is reserved for placement-neutral ragged geometry.");
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
