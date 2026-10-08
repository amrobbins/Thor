#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
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
        outputValueDimensions.push_back(_raggedFeatureInput->isPlacementNeutral()
                                            ? _raggedFeatureInput->getMaxValuesPerRow()
                                            : _raggedFeatureInput->getMaxTotalValues());
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

    const bool placementNeutral = raggedFeatureInput.has_value() && raggedFeatureInput->isPlacementNeutral();
    if (raggedFeatureOutput.has_value()) THOR_THROW_IF_FALSE(placementNeutral == raggedFeatureOutput->isPlacementNeutral());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = placementNeutral ? "1.1.0" : getLayerVersion();
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

void Flatten::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in Flatten::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "flatten")
        throw runtime_error("Layer type mismatch in Flatten::deserialize: " + j.at("layer_type").get<string>());

    Flatten flatten;
    if (j.value("use_ragged", false)) {
        RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "Flatten");
        if ((version == "1.0.0" && input.isPlacementNeutral()) ||
            (version == "1.1.0" && !input.isPlacementNeutral())) {
            throw runtime_error("Flatten serialized layer version does not match its ragged geometry mode.");
        }
        const uint64_t featureInputId = j.at("feature_input").at("id").get<uint64_t>();
        if (featureInputId != j.at("ragged_feature_input").at("values").at("id").get<uint64_t>()) {
            throw runtime_error("Flatten serialized feature_input must match its ragged input values.");
        }
        Tensor featureInput = network->getApiTensorByOriginalId(featureInputId);
        if (featureInput != input.getValues()) {
            throw runtime_error("Flatten reconstructed feature_input must match its ragged input values.");
        }

        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>()) {
            throw runtime_error("Flatten serialized ragged output values must match feature_output.");
        }
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            outputJson, j.at("ragged_feature_input"), input, "Flatten");

        const vector<uint64_t> outputDimensions = outputValues.getDimensions();
        const uint64_t expectedLeading = input.isPlacementNeutral() ? input.getMaxValuesPerRow() : input.getMaxTotalValues();
        if (outputDimensions.size() < 2 || outputDimensions.front() != expectedLeading) {
            throw runtime_error("Ragged Flatten serialized output must preserve the logical packed-row dimension.");
        }
        const vector<uint64_t> inputTrailing = input.getTrailingDimensions();
        const vector<uint64_t> outputTrailing(outputDimensions.begin() + 1, outputDimensions.end());
        if (outputTrailing.empty() || outputTrailing.size() >= inputTrailing.size()) {
            throw runtime_error("Ragged Flatten serialized output must reduce the trailing value rank.");
        }
        const vector<uint64_t> expectedTrailing =
            flattenedDimensions(inputTrailing, static_cast<uint32_t>(outputTrailing.size()), "serialized ragged input");
        if (expectedTrailing != outputTrailing || outputValues.getDataType() != input.getValuesDataType()) {
            throw runtime_error("Ragged Flatten serialized output dimensions are inconsistent with its input.");
        }

        flatten.featureInput = featureInput;
        flatten.featureOutput = outputValues;
        flatten.raggedFeatureInput = input;
        flatten.raggedFeatureOutput = input.withValues(outputValues);
    } else {
        if (version != "1.0.0")
            throw runtime_error("Flatten version 1.1.0 is reserved for placement-neutral ragged geometry.");
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
