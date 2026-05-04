#include "DeepLearning/Api/Layers/Activations/Activation.h"

#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Activation::Deserializer>& Activation::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Activation::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

ThorImplementation::Expression Activation::toExpression(const ThorImplementation::Expression& input) const {
    using ThorImplementation::Expression;

    const string layerType = getLayerType();

    if (layerType == "Relu") {
        return input.max(0.0);
    }

    if (layerType == "Sigmoid") {
        Expression one(1.0);
        return one / (one + (-input).exp());
    }

    if (layerType == "Tanh") {
        const Expression exp2x = (input * 2.0).exp();
        return (exp2x - 1.0) / (exp2x + 1.0);
    }

    if (layerType == "HardSigmoid") {
        return ((input * 0.2) + 0.5).min(1.0).max(0.0);
    }

    if (layerType == "SoftPlus") {
        return (input.exp() + 1.0).ln();
    }

    if (layerType == "SoftSign") {
        return input / (input.abs() + 1.0);
    }

    if (layerType == "Exponential") {
        return input.exp();
    }

    if (layerType == "Gelu") {
        // Match the existing CUDA activation approximation:
        // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
        const Expression x3 = input * input * input;
        const Expression inner = (input + (x3 * 0.044715)) * 0.797884561;
        const Expression exp2Inner = (inner * 2.0).exp();
        const Expression tanhInner = (exp2Inner - 1.0) / (exp2Inner + 1.0);
        return input * 0.5 * (tanhInner + 1.0);
    }

    if (layerType == "Selu") {
        constexpr double scale = 1.05070098;
        constexpr double scaleAlpha = 1.758099326;
        return (input.max(0.0) * scale) + ((input.exp() - 1.0).min(0.0) * scaleAlpha);
    }

    if (layerType == "Swish") {
        Expression one(1.0);
        return input * (one / (one + (-input).exp()));
    }

    if (layerType == "Softmax") {
        const Expression shifted = input - input.reduce_max({1}, {});
        const Expression expShifted = shifted.exp();
        return expShifted / expShifted.reduce_sum({1}, {});
    }

    throw std::runtime_error("Activation " + getLayerType() + " does not support expression fusion.");
}

json Activation::architectureJson() const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Activation.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.get().architectureJson();
    j["feature_output"] = featureOutput.get().architectureJson();
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
