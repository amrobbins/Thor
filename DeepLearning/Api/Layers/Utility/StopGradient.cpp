#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
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

void StopGradient::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in StopGradient::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "stop_gradient")
        throw runtime_error("Layer type mismatch in StopGradient::deserialize: " + j.at("layer_type").get<string>());

    StopGradient stopGradient;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        RaggedTensor raggedInput =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "StopGradient");
        if ((version == "1.0.0" && raggedInput.isPlacementNeutral()) ||
            (version == "1.1.0" && !raggedInput.isPlacementNeutral())) {
            throw runtime_error("StopGradient serialized layer version does not match its ragged geometry mode.");
        }
        const uint64_t featureInputId = j.at("feature_input").at("id").get<uint64_t>();
        if (featureInputId != j.at("ragged_feature_input").at("values").at("id").get<uint64_t>()) {
            throw runtime_error("StopGradient serialized feature_input must match its ragged input values.");
        }
        Tensor featureInput = network->getApiTensorByOriginalId(featureInputId);
        if (featureInput != raggedInput.getValues()) {
            throw runtime_error("StopGradient reconstructed feature_input must match its ragged input values.");
        }

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output"));
        const json& raggedOutputJson = j.at("ragged_feature_output");
        if (raggedOutputJson.at("values").at("id").get<uint64_t>() !=
            j.at("feature_output").at("id").get<uint64_t>()) {
            throw runtime_error("StopGradient serialized ragged output values must match feature_output.");
        }
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            raggedOutputJson, j.at("ragged_feature_input"), raggedInput, "StopGradient");
        if (featureOutput.getDataType() != featureInput.getDataType() ||
            featureOutput.getDimensions() != featureInput.getDimensions()) {
            throw runtime_error("StopGradient serialized output values descriptor must match its input.");
        }

        stopGradient.featureInput = featureInput;
        stopGradient.featureOutput = featureOutput;
        stopGradient.raggedFeatureInput = raggedInput;
        stopGradient.raggedFeatureOutput = raggedInput.withValues(featureOutput);
    } else {
        if (version != "1.0.0")
            throw runtime_error("StopGradient version 1.1.0 is reserved for placement-neutral ragged geometry.");
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        stopGradient.featureInput = network->getApiTensorByOriginalId(originalTensorId);
        stopGradient.featureOutput = Tensor::deserialize(j.at("feature_output"));
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
