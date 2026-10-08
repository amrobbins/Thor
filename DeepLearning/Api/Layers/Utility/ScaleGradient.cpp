#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
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

    const bool placementNeutral = raggedFeatureInput.has_value() && raggedFeatureInput->isPlacementNeutral();
    if (raggedFeatureOutput.has_value()) THOR_THROW_IF_FALSE(placementNeutral == raggedFeatureOutput->isPlacementNeutral());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = placementNeutral ? "1.1.0" : getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    j["scale"] = scale.value();
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

void ScaleGradient::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in ScaleGradient::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "scale_gradient")
        throw runtime_error("Layer type mismatch in ScaleGradient::deserialize: " + j.at("layer_type").get<string>());

    const float serializedScale = j.at("scale").get<float>();
    if (!std::isfinite(serializedScale))
        throw runtime_error("ScaleGradient::deserialize requires a finite scale.");

    ScaleGradient scaleGradient;
    scaleGradient.scale = serializedScale;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        RaggedTensor raggedInput =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "ScaleGradient");
        if ((version == "1.0.0" && raggedInput.isPlacementNeutral()) ||
            (version == "1.1.0" && !raggedInput.isPlacementNeutral())) {
            throw runtime_error("ScaleGradient serialized layer version does not match its ragged geometry mode.");
        }
        const uint64_t featureInputId = j.at("feature_input").at("id").get<uint64_t>();
        if (featureInputId != j.at("ragged_feature_input").at("values").at("id").get<uint64_t>()) {
            throw runtime_error("ScaleGradient serialized feature_input must match its ragged input values.");
        }
        Tensor featureInput = network->getApiTensorByOriginalId(featureInputId);
        if (featureInput != raggedInput.getValues()) {
            throw runtime_error("ScaleGradient reconstructed feature_input must match its ragged input values.");
        }

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output"));
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() !=
            j.at("feature_output").at("id").get<uint64_t>()) {
            throw runtime_error("ScaleGradient serialized ragged output values must match feature_output.");
        }
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            raggedOutputJson, j.at("ragged_feature_input"), raggedInput, "ScaleGradient");
        if (featureOutput.getDataType() != featureInput.getDataType() ||
            featureOutput.getDimensions() != featureInput.getDimensions()) {
            throw runtime_error("ScaleGradient serialized output values descriptor must match its input.");
        }

        scaleGradient.featureInput = featureInput;
        scaleGradient.featureOutput = featureOutput;
        scaleGradient.raggedFeatureInput = raggedInput;
        scaleGradient.raggedFeatureOutput = raggedInput.withValues(featureOutput);
    } else {
        if (version != "1.0.0")
            throw runtime_error("ScaleGradient version 1.1.0 is reserved for placement-neutral ragged geometry.");
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        scaleGradient.featureInput = network->getApiTensorByOriginalId(originalTensorId);
        scaleGradient.featureOutput = Tensor::deserialize(j.at("feature_output"));
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
