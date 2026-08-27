#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Concatenate::Concatenate() = default;
Concatenate::~Concatenate() = default;

json Concatenate::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["concatenation_axis"] = concatenationAxis;

    if (getUseRagged()) {
        j["use_ragged"] = true;
        json raggedInputs = json::array();
        for (const RaggedTensor &input : raggedFeatureInputs) raggedInputs.push_back(input.architectureJson());
        j["ragged_inputs"] = raggedInputs;
        j["ragged_output"] = raggedFeatureOutput.value().architectureJson();
    } else {
        json inputs = json::array();
        for (const Tensor &input : featureInputs) inputs.push_back(input.architectureJson());
        j["inputs"] = inputs;
    }

    json outputs = json::array();
    for (const Tensor &output : featureOutputs) outputs.push_back(output.architectureJson());
    j["outputs"] = outputs;
    return j;
}

void Concatenate::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Concatenate::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "concatenate")
        throw runtime_error("Layer type mismatch in Concatenate::deserialize: " + j.at("layer_type").get<string>());

    Concatenate concatenate;
    concatenate.concatenationAxis = j.at("concatenation_axis").get<uint32_t>();
    concatenate.numInputConnectionsMade = 0;

    if (j.value("use_ragged", false)) {
        auto restoreRagged = [&](const json &r) {
            Tensor values = network->getApiTensorByOriginalId(r.at("values").at("id").get<uint64_t>());
            Tensor offsets = network->getApiTensorByOriginalId(r.at("offsets").at("id").get<uint64_t>());
            return r.contains("max_values_per_row")
                ? RaggedTensor(values, offsets, r.at("max_values_per_row").get<uint64_t>())
                : RaggedTensor(values, offsets);
        };
        for (const json &input : j.at("ragged_inputs"))
            concatenate.raggedFeatureInputs.push_back(restoreRagged(input));
        THOR_THROW_IF_FALSE(concatenate.raggedFeatureInputs.size() >= 2);
        for (const RaggedTensor &input : concatenate.raggedFeatureInputs)
            concatenate.featureInputs.push_back(input.getValues());
        concatenate.featureInputs.push_back(concatenate.raggedFeatureInputs.front().getOffsets());

        Tensor outputValues = Tensor::deserialize(j.at("outputs").at(0));
        concatenate.featureOutputs = {outputValues};
        concatenate.raggedFeatureOutput = concatenate.raggedFeatureInputs.front().withValues(outputValues);
        const json &serializedOutput = j.at("ragged_output");
        if (serializedOutput.at("offsets").at("id").get<uint64_t>() !=
                j.at("ragged_inputs").at(0).at("offsets").at("id").get<uint64_t>() ||
            serializedOutput.at("values").at("id").get<uint64_t>() !=
                j.at("outputs").at(0).at("id").get<uint64_t>()) {
            throw runtime_error("Concatenate serialized ragged output does not preserve the shared row partition.");
        }
    } else {
        for (const json &input : j.at("inputs")) {
            concatenate.featureInputs.push_back(
                network->getApiTensorByOriginalId(input.at("id").get<uint64_t>()));
        }
        THOR_THROW_IF_FALSE(concatenate.featureInputs.size() >= 2);
        concatenate.featureOutputs = {Tensor::deserialize(j.at("outputs").at(0))};
    }

    for (const Tensor &input : concatenate.featureInputs)
        concatenate.outputTensorFromInputTensor[input] = concatenate.featureOutputs[0];
    concatenate.initialized = true;
    concatenate.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("concatenate", &Thor::Concatenate::deserialize);
    return true;
}();
}  // namespace
