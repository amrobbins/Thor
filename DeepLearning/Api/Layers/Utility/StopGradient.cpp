#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Network/Network.h"

#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace Thor {

StopGradient::StopGradient() = default;
StopGradient::~StopGradient() = default;

json StopGradient::architectureJson() const {
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

void StopGradient::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in StopGradient::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "stop_gradient")
        throw runtime_error("Layer type mismatch in StopGradient::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    StopGradient stopGradient;
    stopGradient.featureInput = featureInput;
    stopGradient.featureOutput = featureOutput;
    if (j.value("use_ragged", false)) {
        const json& raggedInputJson = j.at("ragged_feature_input");
        const uint64_t offsetsId = raggedInputJson.at("offsets").at("id").get<uint64_t>();
        Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
        RaggedTensor raggedInput = raggedInputJson.contains("max_values_per_row")
            ? RaggedTensor(featureInput, offsets, raggedInputJson.at("max_values_per_row").get<uint64_t>())
            : RaggedTensor(featureInput, offsets);
        if (raggedInput.getBatchSize() != raggedInputJson.at("batch_size").get<uint64_t>() ||
            raggedInput.getMaxTotalValues() != raggedInputJson.at("max_total_values").get<uint64_t>()) {
            throw runtime_error("StopGradient serialized ragged input metadata does not match reconstructed tensors.");
        }
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>() ||
            raggedOutputJson.at("offsets").at("id").get<uint64_t>() != offsetsId ||
            raggedOutputJson.at("batch_size").get<uint64_t>() != raggedInput.getBatchSize() ||
            raggedOutputJson.at("max_total_values").get<uint64_t>() != raggedInput.getMaxTotalValues() ||
            (raggedOutputJson.contains("max_values_per_row") &&
             (!raggedInput.hasMaxValuesPerRow() ||
              raggedOutputJson.at("max_values_per_row").get<uint64_t>() != raggedInput.getMaxValuesPerRow()))) {
            throw runtime_error("StopGradient serialized ragged output must preserve the input row partition and capacity metadata.");
        }
        stopGradient.raggedFeatureInput = raggedInput;
        stopGradient.raggedFeatureOutput = raggedInput.withValues(featureOutput);
    }
    stopGradient.initialized = true;
    stopGradient.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("stop_gradient", &Thor::StopGradient::deserialize);
    return true;
}();
}  // namespace
