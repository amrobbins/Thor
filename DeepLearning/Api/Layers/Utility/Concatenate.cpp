#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

Concatenate::Concatenate() = default;
Concatenate::~Concatenate() = default;

json Concatenate::serialize(const string &storageDir, Stream stream) {
    assert(initialized);
    assert(featureInputs.size() > 0);
    assert(featureOutputs.size() > 0);

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["concatenation_axis"] = concatenationAxis;

    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back(featureInputs[i].serialize());
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back(featureOutputs[i].serialize());
    }
    j["outputs"] = outputs;

    return j;
}

void Concatenate::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Concatenate::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "concatenate")
        throw runtime_error("Layer type mismatch in Concatenate::deserialize: " + j.at("layer_type").get<string>());

    uint32_t concatenationAxis = j.at("concatenation_axis").get<uint32_t>();

    vector<Tensor> featureInputs;
    for (const json &input : j["inputs"]) {
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor tensor = network->getApiTensorByOriginalId(originalTensorId);
        featureInputs.push_back(tensor);
    }
    assert(featureInputs.size() >= 1);

    vector<Tensor> featureOutputs;
    for (const json &output : j["outputs"]) {
        featureOutputs.push_back(Tensor::deserialize(output));
    }
    assert(featureOutputs.size() == 1);

    Concatenate concatenate;
    concatenate.concatenationAxis = concatenationAxis;
    concatenate.numInputConnectionsMade = 0;
    concatenate.featureInputs = featureInputs;
    concatenate.featureOutputs = featureOutputs;

    for (uint32_t i = 0; i < concatenate.featureInputs.size(); ++i) {
        concatenate.outputTensorFromInputTensor[concatenate.featureInputs[i]] = concatenate.featureOutputs[0];
    }

    concatenate.initialized = true;
    concatenate.addToNetwork(network);
}

namespace {
static bool registered = []() {
    Layer::registry["concatenate"] = &Concatenate::deserialize;
    return true;
}();
}
