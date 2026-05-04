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

    const Expression zero(0.0);
    const Expression one(1.0);
    const Expression two(2.0);
    const string layerType = getLayerType();

    if (layerType == "Relu") {
        return input.max(zero);
    }

    if (layerType == "Sigmoid") {
        return one / (one + (-input).exp());
    }

    if (layerType == "Tanh") {
        const Expression exp2x = (input * two).exp();
        return (exp2x - one) / (exp2x + one);
    }

    if (layerType == "HardSigmoid") {
        return ((input * Expression(0.2)) + Expression(0.5)).min(one).max(zero);
    }

    if (layerType == "SoftPlus") {
        return (input.exp() + one).ln();
    }

    if (layerType == "SoftSign") {
        return input / (input.abs() + one);
    }

    if (layerType == "Exponential") {
        return input.exp();
    }

    if (layerType == "Gelu") {
        // Match the existing CUDA activation approximation:
        // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
        const Expression x3 = input * input * input;
        const Expression inner = (input + (x3 * Expression(0.044715))) * Expression(0.797884561);
        const Expression exp2Inner = (inner * two).exp();
        const Expression tanhInner = (exp2Inner - one) / (exp2Inner + one);
        return input * Expression(0.5) * (tanhInner + one);
    }

    if (layerType == "Selu") {
        const Expression scale(1.05070098);
        const Expression scaleAlpha(1.758099326);
        return (input.max(zero) * scale) + ((input.exp() - one).min(zero) * scaleAlpha);
    }

    if (layerType == "Swish") {
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
