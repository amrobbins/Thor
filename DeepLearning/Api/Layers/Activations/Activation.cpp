#include "DeepLearning/Api/Layers/Activations/Activation.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Activation::Deserializer>& Activation::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Activation::register_layer(string name, Deserializer fn) { get_registry().emplace(move(name), move(fn)); }

json Activation::serialize(thor_file::TarWriter& archiveWriter, Stream stream) const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Activation.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.get().serialize();
    j["feature_output"] = featureOutput.get().serialize();
    return j;
}

json Activation::serialize(Tensor inputTensor, Tensor outputTensor) const {
    assert(initialized);

    json j;
    j["factory"] = Layer::Factory::Activation.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = inputTensor.serialize();
    j["feature_output"] = outputTensor.serialize();
    return j;
}

void Activation::deserialize(const json& j, Network* network) {
    assert(j.at("factory").get<std::string>() == Layer::Factory::Activation);
    std::string type = j.at("layer_type").get<std::string>();

    unordered_map<string, Activation::Deserializer>& registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}

Tensor Activation::addToNetwork(Tensor inputTensor, Network* network) {
    // The following is admittedly a little funky.
    //
    // I need activations to serve 2 purposes:
    //   1. As a template that can be passed to many layers to use as their activation
    //   2. As a standalone layer
    //
    //  For (1) when the layer uses the activation as a template, the layer will add a clone
    //  of the activation to the network and provide the input tensor. Only the network will remember
    //  the activation's input and output tensors when used as a template.
    //  For (2), the user will provide the input tensor, as with the other layers, and the
    //  activation will have a record of its input and output tensors.
    //
    //  So when a layer calls this version of addToNetwork, what happens if someone used an activation
    //  as both a standalone layer and also as a template for other layers? ... I want it to just work anyway.

    Optional<Tensor> maybeExistingFeatureInput = featureInput;
    Optional<Tensor> maybeExistingFeatureOutput = featureOutput;

    featureInput = inputTensor;
    Tensor activationOutput = featureInput.get().clone();
    featureOutput = activationOutput;
    Layer::addToNetwork(network);

    featureInput = maybeExistingFeatureInput;
    featureOutput = maybeExistingFeatureOutput;

    return activationOutput;
}

}  // namespace Thor
