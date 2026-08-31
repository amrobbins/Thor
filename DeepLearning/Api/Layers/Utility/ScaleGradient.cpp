#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cmath>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace Thor {

ScaleGradient::ScaleGradient() = default;
ScaleGradient::~ScaleGradient() = default;

json ScaleGradient::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(scale.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    j["scale"] = scale.value();
    j["use_ragged"] = raggedFeatureInput.has_value();
    if (raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
    return j;
}

void ScaleGradient::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in ScaleGradient::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "scale_gradient")
        throw runtime_error("Layer type mismatch in ScaleGradient::deserialize: " + j.at("layer_type").get<string>());

    const float scale = j.at("scale").get<float>();
    if (!std::isfinite(scale))
        throw runtime_error("ScaleGradient::deserialize requires a finite scale.");

    nlohmann::json input = j.at("feature_input").get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    ScaleGradient scaleGradient;
    scaleGradient.featureInput = featureInput;
    scaleGradient.featureOutput = featureOutput;
    scaleGradient.scale = scale;
    if (j.value("use_ragged", false)) {
        const json& raggedInputJson = j.at("ragged_feature_input");
        const uint64_t offsetsId = raggedInputJson.at("offsets").at("id").get<uint64_t>();
        Tensor offsets = network->getApiTensorByOriginalId(offsetsId);
        RaggedTensor raggedInput = raggedInputJson.contains("max_values_per_row")
            ? RaggedTensor(featureInput, offsets, raggedInputJson.at("max_values_per_row").get<uint64_t>())
            : RaggedTensor(featureInput, offsets);
        if (raggedInput.getBatchSize() != raggedInputJson.at("batch_size").get<uint64_t>() ||
            raggedInput.getMaxTotalValues() != raggedInputJson.at("max_total_values").get<uint64_t>()) {
            throw runtime_error("ScaleGradient serialized ragged input metadata does not match reconstructed tensors.");
        }
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>() ||
            raggedOutputJson.at("offsets").at("id").get<uint64_t>() != offsetsId ||
            raggedOutputJson.at("batch_size").get<uint64_t>() != raggedInput.getBatchSize() ||
            raggedOutputJson.at("max_total_values").get<uint64_t>() != raggedInput.getMaxTotalValues() ||
            (raggedOutputJson.contains("max_values_per_row") &&
             (!raggedInput.hasMaxValuesPerRow() ||
              raggedOutputJson.at("max_values_per_row").get<uint64_t>() != raggedInput.getMaxValuesPerRow()))) {
            throw runtime_error("ScaleGradient serialized ragged output must preserve the input row partition and capacity metadata.");
        }
        scaleGradient.raggedFeatureInput = raggedInput;
        scaleGradient.raggedFeatureOutput = raggedInput.withValues(featureOutput);
    }
    scaleGradient.initialized = true;
    scaleGradient.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("scale_gradient", &Thor::ScaleGradient::deserialize);
    return true;
}();
}  // namespace
