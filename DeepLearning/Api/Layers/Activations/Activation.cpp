#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"

#include "DeepLearning/Api/Layers/Activations/BilinearGlu.h"
#include "DeepLearning/Api/Layers/Activations/Geglu.h"
#include "DeepLearning/Api/Layers/Activations/Glu.h"
#include "DeepLearning/Api/Layers/Activations/Reglu.h"
#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Layers/Activations/Elu.h"
#include "DeepLearning/Api/Layers/Activations/Exponential.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"
#include "DeepLearning/Api/Layers/Activations/HardSwish.h"
#include "DeepLearning/Api/Layers/Activations/HardTanh.h"
#include "DeepLearning/Api/Layers/Activations/Mish.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Relu6.h"
#include "DeepLearning/Api/Layers/Activations/Selu.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Activations/SoftSign.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Activations/Threshold.h"

#include <stdexcept>
#include <optional>

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Activation::Deserializer>& Activation::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Activation::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }


std::shared_ptr<ThorImplementation::Layer> Activation::stampExpressionBackedActivation(ThorImplementation::TensorPlacement placement,
                                                                                        Thor::Tensor connectingApiTensor,
                                                                                        bool inferenceOnly) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    Expression featureInputExpr = Expression::input("feature_input");
    Expression featureOutputExpr = toExpression(featureInputExpr);
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", featureOutputExpr}}));

    std::shared_ptr<ThorImplementation::CustomLayer> physicalActivation = std::make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        std::vector<std::string>{"feature_input"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly);
    physicalActivation->setLayerName(getLayerType());
    return physicalActivation;
}

json Activation::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);

    json j;
    j["factory"] = Layer::Factory::Activation.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    // Template activations embedded in expression-backed layers intentionally do not own API graph tensors.
    // Standalone activation layers still record their graph input/output tensors.
    if (featureInput.has_value()) {
        j["feature_input"] = featureInput.value().architectureJson();
    }
    if (featureOutput.has_value()) {
        j["feature_output"] = featureOutput.value().architectureJson();
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

    if (type == "bilinear_glu") {
        activation = std::make_shared<BilinearGlu>();
    } else if (type == "geglu") {
        activation = std::make_shared<Geglu>();
    } else if (type == "glu") {
        activation = std::make_shared<Glu>();
    } else if (type == "elu") {
        activation = std::make_shared<Elu>(j.value("alpha", 1.0f));
    } else if (type == "exponential") {
        activation = std::make_shared<Exponential>();
    } else if (type == "gelu") {
        activation = std::make_shared<Gelu>();
    } else if (type == "hard_sigmoid") {
        activation = std::make_shared<HardSigmoid>();
    } else if (type == "hard_swish") {
        activation = std::make_shared<HardSwish>();
    } else if (type == "hard_tanh") {
        activation = std::make_shared<HardTanh>(j.value("min_value", -1.0), j.value("max_value", 1.0));
    } else if (type == "mish") {
        activation = std::make_shared<Mish>();
    } else if (type == "reglu") {
        activation = std::make_shared<Reglu>();
    } else if (type == "relu") {
        activation = std::make_shared<Relu>();
    } else if (type == "relu6") {
        activation = std::make_shared<Relu6>();
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
    } else if (type == "swiglu") {
        activation = std::make_shared<Swiglu>();
    } else if (type == "swish") {
        activation = std::make_shared<Swish>();
    } else if (type == "tanh") {
        activation = std::make_shared<Tanh>();
    } else if (type == "threshold") {
        activation = std::make_shared<Threshold>(j.value("threshold", 0.0), j.value("value", 0.0));
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

    std::optional<Tensor> maybeExistingFeatureInput = featureInput;
    std::optional<Tensor> maybeExistingFeatureOutput = featureOutput;

    featureInput = inputTensor;
    Tensor activationOutput = featureInput.value().clone();
    featureOutput = activationOutput;
    Layer::addToNetwork(network);

    featureInput = maybeExistingFeatureInput;
    featureOutput = maybeExistingFeatureOutput;

    return activationOutput;
}

}  // namespace Thor
