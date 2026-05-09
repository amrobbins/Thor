#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"

#include "DeepLearning/Api/Layers/Activations/Elu.h"
#include "DeepLearning/Api/Layers/Activations/Exponential.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Selu.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Activations/SoftSign.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"

#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Activation::Deserializer>& Activation::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Activation::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

json Activation::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);

    json j;
    j["factory"] = Layer::Factory::Activation.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    // Template activations embedded in expression-backed layers intentionally do not own API graph tensors.
    // Standalone activation layers still record their graph input/output tensors.
    if (featureInput.isPresent()) {
        j["feature_input"] = featureInput.get().architectureJson();
    }
    if (featureOutput.isPresent()) {
        j["feature_output"] = featureOutput.get().architectureJson();
    }

    return j;
}

void Activation::deserialize(const json& j, Network* network) {
    THOR_THROW_IF_FALSE(j.at("factory").get<std::string>() == Layer::Factory::Activation);
    std::string type = j.at("layer_type").get<std::string>();

    unordered_map<string, Activation::Deserializer>& registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}

std::shared_ptr<Activation> Activation::deserializeTemplate(const json& j) {
    if (j.is_null()) {
        return nullptr;
    }
    if (j.at("factory").get<std::string>() != Layer::Factory::Activation.value()) {
        throw std::runtime_error("Activation template JSON has wrong factory: " + j.at("factory").get<std::string>());
    }
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in Activation::deserializeTemplate: " + j.at("version").get<std::string>());
    }

    const std::string type = j.at("layer_type").get<std::string>();
    std::shared_ptr<Activation> activation;

    if (type == "elu") {
        activation = std::make_shared<Elu>(j.value("alpha", 1.0f));
    } else if (type == "exponential") {
        activation = std::make_shared<Exponential>();
    } else if (type == "gelu") {
        activation = std::make_shared<Gelu>();
    } else if (type == "hard_sigmoid") {
        activation = std::make_shared<HardSigmoid>();
    } else if (type == "relu") {
        activation = std::make_shared<Relu>();
    } else if (type == "selu") {
        activation = std::make_shared<Selu>();
    } else if (type == "sigmoid") {
        activation = Sigmoid::Builder().build();
    } else if (type == "soft_plus") {
        activation = std::make_shared<SoftPlus>();
    } else if (type == "soft_sign") {
        activation = std::make_shared<SoftSign>();
    } else if (type == "softmax") {
        activation = Softmax::Builder().build();
    } else if (type == "swish") {
        activation = std::make_shared<Swish>();
    } else if (type == "tanh") {
        activation = std::make_shared<Tanh>();
    } else {
        throw std::runtime_error("Unknown activation template type: " + type);
    }

    activation->initialized = true;
    return activation;
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
