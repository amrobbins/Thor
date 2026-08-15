#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json DropOut::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["use_ragged"] = true;
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }

    j["drop_proportion"] = dropProportion;

    return j;
}

void DropOut::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in DropOut::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "drop_out")
        throw runtime_error("Layer type mismatch in DropOut::deserialize: " + j.at("layer_type").get<string>());

    DropOut dropOut;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        const json& inputJson = j.at("ragged_feature_input");
        const uint64_t valuesId = inputJson.at("values").at("id").get<uint64_t>();
        const uint64_t offsetsId = inputJson.at("offsets").at("id").get<uint64_t>();
        if (j.at("feature_input").at("id").get<uint64_t>() != valuesId) {
            throw runtime_error("DropOut serialized ragged feature_input must reference the ragged values tensor.");
        }
        Tensor values = network->getApiTensorByOriginalId(valuesId);
        Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
        RaggedTensor raggedInput(values, offsets);
        if (raggedInput.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
            raggedInput.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
            throw runtime_error("DropOut serialized ragged input metadata does not match reconstructed tensors.");
        }
        if (!ThorImplementation::DropOut::nativeKernelSupportsDataType(values.getDataType())) {
            throw runtime_error("Serialized ragged DropOut values dtype is unsupported.");
        }

        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        if (outputValues.getDimensions() != values.getDimensions() || outputValues.getDataType() != values.getDataType()) {
            throw runtime_error("DropOut serialized ragged output descriptor must match its input values descriptor.");
        }
        RaggedTensor raggedOutput(outputValues, offsets);
        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>() ||
            outputJson.at("offsets").at("id").get<uint64_t>() != offsetsId ||
            outputJson.at("batch_size").get<uint64_t>() != raggedInput.getBatchSize() ||
            outputJson.at("max_total_values").get<uint64_t>() != raggedInput.getMaxTotalValues()) {
            throw runtime_error("DropOut serialized ragged output must preserve the input row partition.");
        }

        dropOut.raggedFeatureInput = raggedInput;
        dropOut.raggedFeatureOutput = raggedOutput;
        dropOut.featureInput = values;
        dropOut.featureOutput = outputValues;
    } else {
        nlohmann::json input = j.at("feature_input").get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        dropOut.featureInput = network->getApiTensorByOriginalId(originalTensorId);
        dropOut.featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());
    }

    dropOut.dropProportion = j.at("drop_proportion").get<float>();
    dropOut.initialized = true;
    dropOut.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("drop_out", &Thor::DropOut::deserialize);
    return true;
}();
}  // namespace
