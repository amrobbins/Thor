#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

std::vector<Tensor> DropOut::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (!raggedFeatureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        return {featureOutput.value()};
    }
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != 2) {
        return {};
    }
    emittedFeatureOutputAfterAllInputsConnected = true;
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    return {featureOutput.value()};
}

void DropOut::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInput.has_value()) return;
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void DropOut::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int DropOut::getConnectionType(Tensor connectingTensor) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (connectingTensor == featureInput.value()) return 0;
    if (raggedFeatureInput.has_value() && connectingTensor == raggedFeatureInput->getOffsets()) return 1;
    if (connectingTensor == featureOutput.value()) return 0;
    throw std::runtime_error("Tensor is not connected to this DropOut layer.");
}

json DropOut::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());


    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = raggedFeatureInput.has_value() ? "1.1.0" : getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    if (raggedFeatureInput.has_value()) {
        j["use_ragged"] = true;
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_input"] =
            SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput.value());
        j["ragged_feature_output"] =
            SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput.value());
    }

    j["drop_proportion"] = dropProportion;
    return j;
}

void DropOut::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in DropOut::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "drop_out")
        throw runtime_error("Layer type mismatch in DropOut::deserialize: " + j.at("layer_type").get<string>());

    DropOut dropOut;
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        RaggedTensor raggedInput =
            SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "DropOut");
        const uint64_t valuesId = j.at("ragged_feature_input").at("values").at("id").get<uint64_t>();
        if (j.at("feature_input").at("id").get<uint64_t>() != valuesId)
            throw runtime_error("DropOut serialized ragged feature_input must reference the ragged values tensor.");
        Tensor values = network->getApiTensorByOriginalId(valuesId);
        if (values != raggedInput.getValues())
            throw runtime_error("DropOut reconstructed feature_input must match its ragged input values.");
        if (!ThorImplementation::DropOut::nativeKernelSupportsDataType(values.getDataType()))
            throw runtime_error("Serialized ragged DropOut values dtype is unsupported.");

        Tensor outputValues = SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
            j.at("feature_output"), j.at("ragged_feature_output"), raggedInput, "DropOut");
        if (outputValues.getDimensions() != values.getDimensions() || outputValues.getDataType() != values.getDataType())
            throw runtime_error("DropOut serialized ragged output descriptor must match its input values descriptor.");
        const json& outputJson = j.at("ragged_feature_output");
        if (outputJson.at("values").at("id").get<uint64_t>() != j.at("feature_output").at("id").get<uint64_t>())
            throw runtime_error("DropOut serialized ragged output values must match feature_output.");
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            outputJson, j.at("ragged_feature_input"), raggedInput, "DropOut");

        dropOut.raggedFeatureInput = raggedInput;
        dropOut.raggedFeatureOutput = raggedInput.withValues(outputValues);
        dropOut.featureInput = values;
        dropOut.featureOutput = outputValues;
    } else {
        if (version != "1.0.0")
            throw runtime_error("DropOut version 1.1.0 is reserved for placement-neutral ragged geometry.");
        const uint64_t originalTensorId = j.at("feature_input").at("id").get<uint64_t>();
        dropOut.featureInput = network->getApiTensorByOriginalId(originalTensorId);
        dropOut.featureOutput = Tensor::deserialize(j.at("feature_output"));
    }

    dropOut.dropProportion = j.at("drop_proportion").get<float>();
    if (dropOut.dropProportion < 0.0f || dropOut.dropProportion > 1.0f)
        throw runtime_error("Serialized DropOut drop_proportion must be in [0, 1].");
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
