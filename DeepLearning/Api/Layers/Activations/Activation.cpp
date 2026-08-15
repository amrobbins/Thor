#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedExpressionLayer.h"
#include "Utilities/Expression/RaggedExpression.h"
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
#include <set>

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Activation::Deserializer>& Activation::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Activation::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

ThorImplementation::RaggedExpression Activation::toRaggedExpression(
    const ThorImplementation::RaggedExpression& input) const {
    return input.mapValues([this](const ThorImplementation::Expression& values) { return toExpression(values); });
}


std::shared_ptr<ThorImplementation::Layer> Activation::stampExpressionBackedActivation(ThorImplementation::TensorPlacement placement,
                                                                                        Thor::Tensor connectingApiTensor,
                                                                                        bool inferenceOnly) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    bool knownInput = connectingApiTensor == featureInput.value();
    if (raggedFeatureInput.has_value() && connectingApiTensor == raggedFeatureInput->getOffsets()) {
        knownInput = true;
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        if (connectingApiTensor == tensor) {
            knownInput = true;
            break;
        }
    }
    THOR_THROW_IF_FALSE(knownInput);

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    std::vector<std::string> inputNames = {"feature_input"};
    if (raggedFeatureInput.has_value()) {
        inputNames.push_back("feature_offsets");
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)tensor;
        inputNames.push_back(name);
    }

    if (raggedFeatureInput.has_value()) {
        const RaggedTensor& ragged = raggedFeatureInput.value();
        ThorImplementation::RaggedExpression raggedInput =
            ThorImplementation::RaggedExpression::input("feature_input", "feature_offsets", ragged.getDescriptor());
        ThorImplementation::RaggedExpression raggedOutput =
            epilogue.has_value()
                ? raggedInput.mapValues([&](const Expression& inputValues) {
                      return applyEpilogue(inputValues, epilogue.value());
                  })
                : toRaggedExpression(raggedInput);

        if (!raggedFeatureOutput.has_value()) {
            throw std::runtime_error("Ragged activation is missing its logical ragged output.");
        }
        if (raggedOutput.getDescriptor() != raggedFeatureOutput->getDescriptor()) {
            throw std::runtime_error("Ragged activation expression output descriptor does not match its API output.");
        }

        ExpressionDefinition definition =
            ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", raggedOutput.getValues()}}));

        uint64_t inputElementsPerValue = 1;
        for (uint64_t dim : ragged.getTrailingDimensions()) {
            inputElementsPerValue *= dim;
        }
        uint64_t outputElementsPerValue = 1;
        for (uint64_t dim : raggedFeatureOutput->getTrailingDimensions()) {
            outputElementsPerValue *= dim;
        }
        auto physicalActivation = std::make_shared<ThorImplementation::RaggedExpressionLayer>(
            DynamicExpression::fromExpressionDefinition(definition), inputNames, std::vector<std::string>{"feature_output"},
            placement, inferenceOnly, ragged.getMaxTotalValues(), inputElementsPerValue, outputElementsPerValue, 0, getId());
        physicalActivation->setLayerName(getLayerType());
        return physicalActivation;
    }

    const DataType featureInputDType = featureInput.value().getDataType();
    Expression featureInputExpr = Expression::input("feature_input", std::nullopt, featureInputDType);
    Expression featureOutputExpr = epilogue.has_value() ? applyEpilogue(featureInputExpr, epilogue.value()) : toExpression(featureInputExpr);
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", featureOutputExpr}}));

    std::shared_ptr<ThorImplementation::CustomLayer> physicalActivation = std::make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition), inputNames, std::vector<std::string>{"feature_output"}, placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{}, inferenceOnly);
    physicalActivation->setLayerName(getLayerType());
    return physicalActivation;
}

void Activation::validateEpilogueAuxInputName(const std::string& inputName) {
    if (inputName.empty()) {
        throw std::invalid_argument("Activation epilogue auxiliary input name cannot be empty.");
    }
    if (inputName.rfind("__", 0) == 0) {
        throw std::invalid_argument("Activation epilogue auxiliary input names cannot start with __: " + inputName + ".");
    }
    static const std::set<std::string> reservedNames = {
        "feature_input",
        "feature_offsets",
        "feature_output",
        epilogueInputName(),
        epilogueOutputName(),
    };
    if (reservedNames.contains(inputName)) {
        throw std::invalid_argument("Activation epilogue auxiliary input name is reserved: " + inputName + ".");
    }
}

std::vector<Tensor> Activation::getFeatureInputs() const {
    std::vector<Tensor> inputs;
    if (featureInput.has_value()) inputs.push_back(featureInput.value());
    if (raggedFeatureInput.has_value()) inputs.push_back(raggedFeatureInput->getOffsets());
    inputs.reserve(inputs.size() + epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        inputs.push_back(tensor);
    }
    return inputs;
}

std::vector<Tensor> Activation::getFeatureOutputs() const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    return {featureOutput.value()};
}

std::vector<Tensor> Activation::getAllOutputTensors() const { return getFeatureOutputs(); }

std::vector<Tensor> Activation::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (!raggedFeatureInput.has_value() && epilogueInputBindings.empty()) {
        return {featureOutput.value()};
    }

    if (emittedFeatureOutputAfterAllInputsConnected) {
        return {};
    }
    const uint32_t requiredInputPorts = static_cast<uint32_t>((raggedFeatureInput.has_value() ? 2 : 1) + epilogueInputBindings.size());
    if (connectedInputPortIndices.size() != requiredInputPorts) {
        return {};
    }

    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutput.value()};
}

void Activation::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInput.has_value() && epilogueInputBindings.empty()) return;
    const uint32_t port = static_cast<uint32_t>(getConnectionType(inputTensor));
    connectedInputPortIndices.insert(port);
}

void Activation::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
}

int Activation::getConnectionType(Tensor connectingTensor) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    if (connectingTensor == featureInput.value()) return 0;
    uint32_t nextPort = 1;
    if (raggedFeatureInput.has_value()) {
        if (connectingTensor == raggedFeatureInput->getOffsets()) return 1;
        nextPort = 2;
    }
    for (uint32_t i = 0; i < epilogueInputBindings.size(); ++i) {
        if (connectingTensor == epilogueInputBindings[i].second) return static_cast<int>(nextPort + i);
    }
    if (connectingTensor == featureOutput.value()) return 0;
    throw std::runtime_error("Tensor is not connected to this activation layer.");
}

uint64_t Activation::getExpressionBackedActivationMemRequirementInBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    uint64_t bytes = featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes();
    if (raggedFeatureInput.has_value()) bytes += raggedFeatureInput->getOffsets().getTotalSizeInBytes();
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        bytes += tensor.getTotalSizeInBytes();
    }
    return raggedFeatureInput.has_value() ? bytes : batchSize * bytes;
}

Tensor Activation::standaloneOutputTensorForInput(const Tensor& inputTensor) const { return inputTensor.clone(); }

void Activation::initializeStandaloneActivation(Tensor inputTensor,
                                                std::optional<ThorImplementation::Expression> epilogueExpression,
                                                std::vector<std::pair<std::string, Tensor>> epilogueInputBindingsValue) {
    THOR_THROW_IF_FALSE(!inputTensor.getDimensions().empty());
    if (!epilogueExpression.has_value() && !epilogueInputBindingsValue.empty()) {
        throw std::invalid_argument("Activation epilogue inputs were provided without an epilogue expression.");
    }

    std::vector<std::string> auxInputNames;
    auxInputNames.reserve(epilogueInputBindingsValue.size());
    std::set<std::string> seenNames;
    for (const auto& [name, tensor] : epilogueInputBindingsValue) {
        validateEpilogueAuxInputName(name);
        if (!seenNames.insert(name).second) {
            throw std::invalid_argument("Activation epilogue input name is duplicated: " + name + ".");
        }
        if (tensor.getDataType() != inputTensor.getDataType()) {
            throw std::invalid_argument("Activation epilogue input '" + name + "' dtype must match feature_input dtype.");
        }
        if (tensor.getDimensions() != inputTensor.getDimensions()) {
            throw std::invalid_argument("Activation epilogue input '" + name + "' shape must match feature_input shape.");
        }
        auxInputNames.push_back(name);
    }
    if (epilogueExpression.has_value()) {
        validateEpilogueExpression(epilogueExpression.value(), auxInputNames);
    }

    raggedFeatureInput.reset();
    raggedFeatureOutput.reset();
    featureInput = inputTensor;
    featureOutput = standaloneOutputTensorForInput(inputTensor);
    epilogue = std::move(epilogueExpression);
    epilogueInputBindings = std::move(epilogueInputBindingsValue);
    serializableEpilogue.reset();
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
}

void Activation::initializeStandaloneActivation(RaggedTensor inputTensor,
                                                std::optional<ThorImplementation::Expression> epilogueExpression,
                                                std::vector<std::pair<std::string, Tensor>> epilogueInputBindingsValue) {
    THOR_THROW_IF_FALSE(inputTensor.isInitialized());
    if (!supportsRaggedStandalone()) {
        throw std::invalid_argument(getLayerType() + " does not yet support standalone RaggedTensor input.");
    }
    const Tensor values = inputTensor.getValues();
    if (!epilogueExpression.has_value() && !epilogueInputBindingsValue.empty())
        throw std::invalid_argument("Activation epilogue inputs were provided without an epilogue expression.");
    std::vector<std::string> auxInputNames;
    std::set<std::string> seenNames;
    for (const auto& [name, tensor] : epilogueInputBindingsValue) {
        validateEpilogueAuxInputName(name);
        if (!seenNames.insert(name).second) throw std::invalid_argument("Activation epilogue input name is duplicated: " + name + ".");
        if (tensor.getDataType() != values.getDataType() || tensor.getDimensions() != values.getDimensions())
            throw std::invalid_argument("Activation ragged epilogue inputs must match packed values dtype and capacity shape.");
        auxInputNames.push_back(name);
    }
    if (epilogueExpression.has_value()) validateEpilogueExpression(epilogueExpression.value(), auxInputNames);
    raggedFeatureInput = inputTensor;
    featureInput = values;
    featureOutput = standaloneOutputTensorForInput(values);
    raggedFeatureOutput = RaggedTensor(featureOutput.value(), inputTensor.getOffsets());
    epilogue = std::move(epilogueExpression);
    epilogueInputBindings = std::move(epilogueInputBindingsValue);
    serializableEpilogue.reset();
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
}

void Activation::deserializeStandaloneFields(const json& j, Network* network) {
    const bool useRagged = j.value("use_ragged", false);
    Tensor inputTensor;
    std::optional<RaggedTensor> raggedInput;
    if (useRagged) {
        const json& raggedJson = j.at("ragged_feature_input");
        const uint64_t valuesId = raggedJson.at("values").at("id").get<uint64_t>();
        const uint64_t offsetsId = raggedJson.at("offsets").at("id").get<uint64_t>();
        inputTensor = network->getApiTensorByOriginalId(valuesId);
        raggedInput = RaggedTensor(inputTensor, network->getApiTensorByOriginalId(offsetsId));
        if (raggedInput->getBatchSize() != raggedJson.at("batch_size").get<uint64_t>() ||
            raggedInput->getMaxTotalValues() != raggedJson.at("max_total_values").get<uint64_t>())
            throw std::runtime_error("Activation serialized ragged input metadata does not match reconstructed tensors.");
    } else {
        inputTensor = network->getApiTensorByOriginalId(j.at("feature_input").at("id").get<uint64_t>());
    }

    std::vector<std::pair<std::string, Tensor>> epilogueBindings;
    if (j.contains("epilogue_inputs")) {
        for (const json& epilogueInputJson : j.at("epilogue_inputs")) {
            std::string inputName = epilogueInputJson.at("name").get<std::string>();
            validateEpilogueAuxInputName(inputName);
            epilogueBindings.emplace_back(inputName,
                network->getApiTensorByOriginalId(epilogueInputJson.at("tensor").at("id").get<uint64_t>()));
        }
    }
    std::vector<std::string> auxInputNames;
    for (const auto& [name, tensor] : epilogueBindings) { (void)tensor; auxInputNames.push_back(name); }

    std::optional<ThorImplementation::Expression> epilogueExpression = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition def = ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogueExpression = epilogueExpressionFromDefinition(def, auxInputNames);
    } else if (!epilogueBindings.empty()) {
        throw std::runtime_error("Activation serialized epilogue_inputs require a non-null epilogue expression.");
    }

    if (raggedInput.has_value()) {
        initializeStandaloneActivation(raggedInput.value(), epilogueExpression, epilogueBindings);
        featureOutput = Tensor::deserialize(j.at("feature_output"));
        raggedFeatureOutput = RaggedTensor(featureOutput.value(), raggedInput->getOffsets());
        if (j.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>() !=
            j.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>())
            throw std::runtime_error("Activation serialized ragged output must preserve the input row partition.");
    } else {
        initializeStandaloneActivation(inputTensor, epilogueExpression, epilogueBindings);
        featureOutput = Tensor::deserialize(j.at("feature_output"));
    }
    initialized = true;
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
    if (featureOutput.has_value()) j["feature_output"] = featureOutput.value().architectureJson();
    if (raggedFeatureInput.has_value()) {
        j["use_ragged"] = true;
        j["ragged_feature_input"] = raggedFeatureInput->architectureJson();
        THOR_THROW_IF_FALSE(raggedFeatureOutput.has_value());
        j["ragged_feature_output"] = raggedFeatureOutput->architectureJson();
    }
    if (epilogue.has_value()) {
        std::vector<std::string> auxInputNames;
        auxInputNames.reserve(epilogueInputBindings.size());
        for (const auto& [name, tensor] : epilogueInputBindings) {
            (void)tensor;
            auxInputNames.push_back(name);
        }
        if (!serializableEpilogue.has_value()) {
            serializableEpilogue = makeEpilogueDefinition(epilogue.value(), auxInputNames);
        }
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }
    json epilogueInputs = json::array();
    for (const auto& [name, tensor] : epilogueInputBindings) {
        epilogueInputs.push_back(json{{"name", name}, {"tensor", tensor.architectureJson()}});
    }
    j["epilogue_inputs"] = epilogueInputs;

    return j;
}

void Activation::deserialize(const json& j, Network* network) {
    THOR_THROW_IF_FALSE(j.at("factory").get<std::string>() == Layer::Factory::Activation);
    std::string type = j.at("layer_type").get<std::string>();

    // Ragged standalone activations share one structural serialization contract.
    // Instantiate the concrete activation from its template fields, then let the
    // common Activation path reconnect canonical values/offset tensors.
    if (j.value("use_ragged", false)) {
        std::shared_ptr<Activation> activation = deserializeTemplate(j);
        activation->deserializeStandaloneFields(j, network);
        activation->addToNetwork(network);
        return;
    }

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

Tensor Activation::addToNetwork(Tensor inputTensor, Network* network) { return addToNetwork(inputTensor, network, std::nullopt, {}); }

RaggedTensor Activation::addToNetwork(RaggedTensor inputTensor, Network* network) { return addToNetwork(inputTensor, network, std::nullopt, {}); }

Tensor Activation::addToNetwork(Tensor inputTensor,
                                Network* network,
                                std::optional<ThorImplementation::Expression> epilogueExpression,
                                std::vector<std::pair<std::string, Tensor>> epilogueInputBindingsValue) {
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
    std::optional<RaggedTensor> maybeExistingRaggedFeatureInput = raggedFeatureInput;
    std::optional<RaggedTensor> maybeExistingRaggedFeatureOutput = raggedFeatureOutput;
    std::optional<ThorImplementation::Expression> maybeExistingEpilogue = epilogue;
    std::vector<std::pair<std::string, Tensor>> maybeExistingEpilogueInputBindings = epilogueInputBindings;
    std::optional<ThorImplementation::ExpressionDefinition> maybeExistingSerializableEpilogue = serializableEpilogue;

    initializeStandaloneActivation(inputTensor, std::move(epilogueExpression), std::move(epilogueInputBindingsValue));
    Tensor activationOutput = featureOutput.value();
    Layer::addToNetwork(network);

    featureInput = maybeExistingFeatureInput;
    featureOutput = maybeExistingFeatureOutput;
    raggedFeatureInput = maybeExistingRaggedFeatureInput;
    raggedFeatureOutput = maybeExistingRaggedFeatureOutput;
    epilogue = maybeExistingEpilogue;
    epilogueInputBindings = maybeExistingEpilogueInputBindings;
    serializableEpilogue = maybeExistingSerializableEpilogue;
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();

    return activationOutput;
}

RaggedTensor Activation::addToNetwork(RaggedTensor inputTensor,
                                      Network* network,
                                      std::optional<ThorImplementation::Expression> epilogueExpression,
                                      std::vector<std::pair<std::string, Tensor>> epilogueInputBindingsValue) {
    std::optional<Tensor> maybeExistingFeatureInput = featureInput;
    std::optional<Tensor> maybeExistingFeatureOutput = featureOutput;
    std::optional<RaggedTensor> maybeExistingRaggedFeatureInput = raggedFeatureInput;
    std::optional<RaggedTensor> maybeExistingRaggedFeatureOutput = raggedFeatureOutput;
    std::optional<ThorImplementation::Expression> maybeExistingEpilogue = epilogue;
    auto maybeExistingEpilogueInputBindings = epilogueInputBindings;
    std::optional<ThorImplementation::ExpressionDefinition> maybeExistingSerializableEpilogue = serializableEpilogue;

    initializeStandaloneActivation(inputTensor, std::move(epilogueExpression), std::move(epilogueInputBindingsValue));
    RaggedTensor activationOutput = raggedFeatureOutput.value();
    Layer::addToNetwork(network);

    featureInput = maybeExistingFeatureInput;
    featureOutput = maybeExistingFeatureOutput;
    raggedFeatureInput = maybeExistingRaggedFeatureInput;
    raggedFeatureOutput = maybeExistingRaggedFeatureOutput;
    epilogue = maybeExistingEpilogue;
    epilogueInputBindings = maybeExistingEpilogueInputBindings;
    serializableEpilogue = maybeExistingSerializableEpilogue;
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
    return activationOutput;
}

}  // namespace Thor
