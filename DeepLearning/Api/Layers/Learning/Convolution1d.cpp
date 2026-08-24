#include "DeepLearning/Api/Layers/Learning/Convolution1d.h"

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

const char *paddingModeName(Convolution1dPaddingMode mode) {
    switch (mode) {
        case Convolution1dPaddingMode::VALID:
            return "valid";
        case Convolution1dPaddingMode::SAME_UPPER:
            return "same_upper";
        case Convolution1dPaddingMode::CAUSAL:
            return "causal";
        case Convolution1dPaddingMode::EXPLICIT:
            return "explicit";
    }
    throw runtime_error("Unknown Convolution1d padding mode.");
}

Convolution1dPaddingMode paddingModeFromName(const string &name) {
    if (name == "valid")
        return Convolution1dPaddingMode::VALID;
    if (name == "same_upper")
        return Convolution1dPaddingMode::SAME_UPPER;
    if (name == "causal")
        return Convolution1dPaddingMode::CAUSAL;
    if (name == "explicit")
        return Convolution1dPaddingMode::EXPLICIT;
    throw runtime_error("Unknown Convolution1d padding_mode: " + name);
}

ThorImplementation::ConvolutionSpatial1d resolveSpatial(Convolution1dPaddingMode mode,
                                                         uint32_t inputWidth,
                                                         uint32_t filterWidth,
                                                         uint32_t stride,
                                                         uint32_t dilation,
                                                         uint32_t explicitLeft,
                                                         uint32_t explicitRight) {
    switch (mode) {
        case Convolution1dPaddingMode::VALID:
            return ThorImplementation::ConvolutionSpatial1d::valid(static_cast<int32_t>(stride), static_cast<int32_t>(dilation));
        case Convolution1dPaddingMode::SAME_UPPER:
            return ThorImplementation::ConvolutionSpatial1d::sameUpper(
                inputWidth, filterWidth, static_cast<int32_t>(stride), static_cast<int32_t>(dilation));
        case Convolution1dPaddingMode::CAUSAL:
            return ThorImplementation::ConvolutionSpatial1d::causal(
                filterWidth, static_cast<int32_t>(stride), static_cast<int32_t>(dilation));
        case Convolution1dPaddingMode::EXPLICIT:
            return ThorImplementation::ConvolutionSpatial1d::explicitPadding(static_cast<int32_t>(explicitLeft),
                                                                               static_cast<int32_t>(explicitRight),
                                                                               static_cast<int32_t>(stride),
                                                                               static_cast<int32_t>(dilation));
    }
    throw runtime_error("Unknown Convolution1d padding mode.");
}

ThorImplementation::DynamicExpression buildConvolution1dExpression(
    bool hasBias,
    uint32_t groups,
    ThorImplementation::ConvolutionSpatial1d spatial,
    ThorImplementation::TensorPlacement placement,
    shared_ptr<Thor::Activation> activation,
    optional<ThorImplementation::Expression> epilogue,
    vector<string> epilogueAuxInputNames) {
    using ImplDataType = ThorImplementation::DataType;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    vector<string> expectedInputNames = {"feature_input"};
    expectedInputNames.insert(expectedInputNames.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());
    expectedInputNames.push_back("weights");
    if (hasBias)
        expectedInputNames.push_back("biases");

    return DynamicExpression(std::move(expectedInputNames),
                             {"feature_output"},
                             [hasBias,
                              groups,
                              spatial,
                              placement,
                              activation = std::move(activation),
                              epilogue,
                              epilogueAuxInputNames = std::move(epilogueAuxInputNames)](
                                 const DynamicExpression::TensorMap &inputs,
                                 const DynamicExpression::TensorMap &outputs,
                                 Stream &stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor &featureInputTensor = inputs.at("feature_input");
        const Tensor &wTensor = inputs.at("weights");
        if (featureInputTensor.getDimensions().size() != 3)
            throw runtime_error("Convolution1d expects feature_input to be 3D NCW.");
        if (wTensor.getDimensions().size() != 3)
            throw runtime_error("Convolution1d expects weights to be 3D KCW.");
        if (groups == 0 || featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1] * groups ||
            wTensor.getDimensions()[0] % groups != 0)
            throw runtime_error("Convolution1d grouped channel geometry is invalid.");
        THOR_THROW_IF_FALSE(featureInputTensor.getPlacement() == placement);
        THOR_THROW_IF_FALSE(wTensor.getPlacement() == placement);

        const uint64_t effectiveFilter =
            static_cast<uint64_t>(spatial.dilation) * (wTensor.getDimensions()[2] - 1ULL) + 1ULL;
        const uint64_t paddedInput = featureInputTensor.getDimensions()[2] + static_cast<uint64_t>(spatial.pre_padding) +
                                     static_cast<uint64_t>(spatial.post_padding);
        if (effectiveFilter > paddedInput)
            throw runtime_error("Convolution1d effective filter is larger than the padded input width.");
        const uint64_t expectedOutputWidth =
            1ULL + (paddedInput - effectiveFilter) / static_cast<uint64_t>(spatial.stride);

        optional<ImplDataType> featureOutputDType = nullopt;
        if (outputs.contains("feature_output")) {
            const Tensor &featureOutputTensor = outputs.at("feature_output");
            if (featureOutputTensor.getDimensions().size() != 3)
                throw runtime_error("Convolution1d expects feature_output to be 3D NCW.");
            if (featureOutputTensor.getDimensions()[0] != featureInputTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[2] != expectedOutputWidth) {
                throw runtime_error("Convolution1d feature_output shape does not match the implied convolution output shape.");
            }
            THOR_THROW_IF_FALSE(featureOutputTensor.getPlacement() == placement);
            featureOutputDType = featureOutputTensor.getDescriptor().getDataType();
        }

        const ImplDataType weightsDType = wTensor.getDescriptor().getDataType();
        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);
        Expression fout = Expression::conv1d(fin, w, spatial, ImplDataType::FP32, featureOutputDType, groups);

        if (hasBias) {
            const Tensor &bTensor = inputs.at("biases");
            if (bTensor.getDimensions().size() != 1)
                throw runtime_error("Convolution1d expects biases to be 1D [K].");
            if (bTensor.getDimensions()[0] != wTensor.getDimensions()[0])
                throw runtime_error("Convolution1d bias size must match number of output channels.");
            const ImplDataType biasDType = bTensor.getDescriptor().getDataType();
            auto b = Expression::input("biases", biasDType, biasDType).unsqueeze({0, 2});
            fout = fout + b;
        }

        if (activation != nullptr)
            fout = activation->toExpression(fout);

        for (const string &auxInputName : epilogueAuxInputNames) {
            const Tensor &auxTensor = inputs.at(auxInputName);
            const vector<uint64_t> expectedAuxShape = {
                featureInputTensor.getDimensions()[0], wTensor.getDimensions()[0], expectedOutputWidth};
            if (auxTensor.getDimensions() != expectedAuxShape) {
                throw runtime_error("Convolution1d epilogue auxiliary input '" + auxInputName +
                                    "' shape must match the convolution feature output shape.");
            }
            if (featureOutputDType.has_value() && auxTensor.getDescriptor().getDataType() != featureOutputDType.value()) {
                throw runtime_error("Convolution1d epilogue auxiliary input '" + auxInputName +
                                    "' dtype must match the convolution feature output dtype.");
            }
            THOR_THROW_IF_FALSE(auxTensor.getPlacement() == placement);
        }
        if (epilogue.has_value())
            fout = Convolution1d::applyEpilogue(fout, epilogue.value());
        if (featureOutputDType.has_value())
            fout = fout.withOutputDType(featureOutputDType.value());

        auto expressionOutputs = Expression::outputs({{"feature_output", fout}});
        return DynamicExpressionBuild{
            make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {},
        };
    });
}

}  // namespace

Convolution1d Convolution1d::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(_featureInputs.size() == 1);
    THOR_THROW_IF_FALSE(_numOutputChannels.has_value());
    THOR_THROW_IF_FALSE(_filterWidth.has_value());

    if (!_stride.has_value())
        _stride = 1;
    if (!_dilation.has_value())
        _dilation = 1;
    if (!_groups.has_value())
        _groups = 1;
    if (!_paddingMode.has_value())
        _paddingMode = Convolution1dPaddingMode::VALID;
    if (!_hasBias.has_value())
        _hasBias = false;
    if (_weightsInitializer == nullptr)
        _weightsInitializer = Glorot::Builder().build();
    if (_biasesInitializer == nullptr)
        _biasesInitializer = Glorot::Builder().build();
    if (_activation == nullptr && !_activationExplicitlyRemoved)
        _activation = Gelu::Builder().build();

    if (!_epilogueInputBindings.empty() && _featureInputs.size() != 1)
        throw invalid_argument("Convolution1d epilogue auxiliary inputs currently require exactly one feature input.");
    if (_epilogue.has_value())
        Convolution1d::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
    else if (!_epilogueInputBindings.empty())
        throw invalid_argument("Convolution1d epilogue_inputs were provided without an epilogue expression.");

    Convolution1d convolution1d(_epilogue, _epilogueInputBindings);
    convolution1d.featureInputs = _featureInputs;
    convolution1d.numOutputChannels = _numOutputChannels.value();
    convolution1d.filterWidth = _filterWidth.value();
    convolution1d.groups = _groups.value();
    convolution1d.paddingMode = _paddingMode.value();
    const uint32_t explicitLeft = _paddingLeft.value_or(0);
    const uint32_t explicitRight = _paddingRight.value_or(0);
    const uint32_t inputWidth = static_cast<uint32_t>(_featureInputs.front().getDimensions()[1]);
    convolution1d.spatial = resolveSpatial(convolution1d.paddingMode,
                                           inputWidth,
                                           convolution1d.filterWidth,
                                           _stride.value(),
                                           _dilation.value(),
                                           explicitLeft,
                                           explicitRight);
    const uint32_t outputWidth = computeOutputWidth(inputWidth, convolution1d.filterWidth, convolution1d.spatial);

    convolution1d.hasBias = _hasBias.value();
    convolution1d.weightsInitializer = _weightsInitializer->clone();
    convolution1d.biasInitializer = _biasesInitializer->clone();
    if (_activation != nullptr)
        convolution1d.activation = _activation;
    convolution1d.weightsOptimizer = _weightsOptimizer;
    convolution1d.biasesOptimizer = _biasesOptimizer;

    const DataType dataType = convolution1d.featureInputs.front().getDataType();
    const uint64_t inputChannels = convolution1d.featureInputs.front().getDimensions()[0];
    if (inputChannels % convolution1d.groups != 0 || convolution1d.numOutputChannels % convolution1d.groups != 0)
        throw invalid_argument("Convolution1d requires input and output channels divisible by groups.");

    ParameterSpecification::Builder weightsParameterBuilder;
    weightsParameterBuilder.name("weights")
        .shape({convolution1d.numOutputChannels, inputChannels / convolution1d.groups, convolution1d.filterWidth})
        .dtype(dataType)
        .initializer(convolution1d.weightsInitializer)
        .trainable(true);
    if (convolution1d.weightsOptimizer != nullptr)
        weightsParameterBuilder.optimizer(convolution1d.weightsOptimizer);
    convolution1d.addParameter(make_shared<ParameterSpecification>(weightsParameterBuilder.build()));

    if (convolution1d.hasBias) {
        ParameterSpecification::Builder biasesParameterBuilder;
        biasesParameterBuilder.name("biases")
            .shape({convolution1d.numOutputChannels})
            .dtype(dataType)
            .initializer(convolution1d.biasInitializer)
            .trainable(true);
        if (convolution1d.biasesOptimizer != nullptr)
            biasesParameterBuilder.optimizer(convolution1d.biasesOptimizer);
        convolution1d.addParameter(make_shared<ParameterSpecification>(biasesParameterBuilder.build()));
    }

    convolution1d.initialized = true;
    for (uint32_t i = 0; i < convolution1d.featureInputs.size(); ++i) {
        convolution1d.featureOutputs.push_back(Tensor(dataType, {convolution1d.numOutputChannels, outputWidth}));
        convolution1d.outputTensorFromInputTensor[convolution1d.featureInputs[i]] = convolution1d.featureOutputs[i];
        convolution1d.inputTensorFromOutputTensor[convolution1d.featureOutputs[i]] = convolution1d.featureInputs[i];
    }
    for (const auto &[name, tensor] : convolution1d.epilogueInputBindings) {
        (void)name;
        THOR_THROW_IF_FALSE(tensor.getDataType() == dataType);
        THOR_THROW_IF_FALSE(tensor.getDimensions() == convolution1d.featureOutputs.front().getDimensions());
        convolution1d.outputTensorFromInputTensor[tensor] = convolution1d.featureOutputs.front();
    }

    convolution1d.standaloneLayerFeatureInputs = convolution1d.featureInputs;
    convolution1d.standaloneLayerFeatureOutputs = convolution1d.getFeatureOutputs();
    convolution1d.addToNetwork(_network.value());
    return convolution1d;
}

void Convolution1d::validateEpilogueAuxInputName(const string &inputName) {
    if (inputName.empty())
        throw invalid_argument("Convolution1d epilogue auxiliary input name cannot be empty.");
    if (inputName.rfind("__", 0) == 0)
        throw invalid_argument("Convolution1d epilogue auxiliary input names cannot start with __: " + inputName + ".");
    static const set<string> reservedNames = {
        "feature_input", "feature_output", "weights", "biases", epilogueInputName(), epilogueOutputName()};
    if (reservedNames.contains(inputName))
        throw invalid_argument("Convolution1d epilogue auxiliary input name is reserved: " + inputName + ".");
}

vector<string> Convolution1d::epilogueAuxInputNames() const {
    vector<string> names;
    names.reserve(epilogueInputBindings.size());
    for (const auto &[name, tensor] : epilogueInputBindings) {
        (void)tensor;
        names.push_back(name);
    }
    return names;
}

vector<Tensor> Convolution1d::getFeatureInputs() const {
    vector<Tensor> inputs = featureInputs;
    inputs.reserve(inputs.size() + epilogueInputBindings.size());
    for (const auto &[name, tensor] : epilogueInputBindings) {
        (void)name;
        inputs.push_back(tensor);
    }
    return inputs;
}

vector<uint32_t> Convolution1d::inputPortIndicesForTensor(Tensor tensor) const {
    vector<uint32_t> ports;
    if (!featureInputs.empty() && tensor.getOriginalId() == featureInputs[0].getOriginalId())
        ports.push_back(0);
    for (uint32_t i = 0; i < epilogueInputBindings.size(); ++i) {
        if (tensor.getOriginalId() == epilogueInputBindings[i].second.getOriginalId())
            ports.push_back(i + 1);
    }
    return ports;
}

Tensor Convolution1d::getFeatureOutput(Tensor inputTensor) const {
    const auto it = outputTensorFromInputTensor.find(inputTensor);
    if (it == outputTensorFromInputTensor.end())
        throw runtime_error("Tensor is not connected to this Convolution1d layer.");
    return it->second;
}

vector<Tensor> Convolution1d::getOutputsFromInput(Tensor inputTensor) {
    if (epilogueInputBindings.empty())
        return {getFeatureOutput(inputTensor)};
    (void)getFeatureOutput(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected)
        return {};
    const uint32_t requiredInputPorts = static_cast<uint32_t>(1 + epilogueInputBindings.size());
    if (connectedInputPortIndices.size() != requiredInputPorts)
        return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs[0]};
}

void Convolution1d::informThatInputConnectionMade(Tensor inputTensor) {
    if (epilogueInputBindings.empty())
        return;
    const vector<uint32_t> ports = inputPortIndicesForTensor(inputTensor);
    if (ports.empty())
        throw runtime_error("Convolution1d informed of connection for unknown input tensor.");
    for (uint32_t port : ports)
        connectedInputPortIndices.insert(port);
}

void Convolution1d::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
}

int Convolution1d::getConnectionType(Tensor connectingTensor) const {
    const vector<uint32_t> inputPorts = inputPortIndicesForTensor(connectingTensor);
    if (!inputPorts.empty()) {
        uint32_t &cursor = nextInputConnectionCursorByTensorOriginalId[connectingTensor.getOriginalId()];
        const uint32_t port = inputPorts[cursor % inputPorts.size()];
        ++cursor;
        return static_cast<int>(port);
    }
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        if (connectingTensor == featureOutputs[i])
            return static_cast<int>(i);
    }
    throw runtime_error("Tensor is not connected to this Convolution1d layer.");
}

shared_ptr<ThorImplementation::Layer> Convolution1d::stamp(ThorImplementation::TensorPlacement placement,
                                                            shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                            shared_ptr<Thor::Layer> drivingApiLayer,
                                                            Thor::Tensor connectingApiTensor,
                                                            const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    vector<shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto &parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    auto physicalConvolution1d = make_shared<ThorImplementation::CustomLayer>(
        buildConvolution1dExpression(hasBias, groups, spatial, placement, activation, epilogue, epilogueAuxInputNames()),
        [&]() {
            vector<string> inputNames = {"feature_input"};
            const vector<string> auxNames = epilogueAuxInputNames();
            inputNames.insert(inputNames.end(), auxNames.begin(), auxNames.end());
            return inputNames;
        }(),
        vector<string>{"feature_output"},
        placement,
        physicalParameters,
        inferenceOnly,
        getId());
    physicalConvolution1d->setLayerName(getLayerType());
    return physicalConvolution1d;
}

json Convolution1d::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "convolution_1d";
    j["layer_name"] = string("layer") + to_string(getId());
    j["data_layout"] = "NCW";
    j["filter_width"] = filterWidth;
    j["stride"] = spatial.stride;
    j["dilation"] = spatial.dilation;
    j["padding_mode"] = paddingModeName(paddingMode);
    j["padding_left"] = spatial.pre_padding;
    j["padding_right"] = spatial.post_padding;
    j["num_output_channels"] = numOutputChannels;
    j["groups"] = groups;
    j["has_bias"] = hasBias;
    j["activation"] = activation != nullptr ? activation->architectureJson() : json(nullptr);

    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value())
            serializableEpilogue = makeEpilogueDefinition(epilogue.value(), epilogueAuxInputNames());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    json inputs = json::array();
    for (const Tensor &input : standaloneLayerFeatureInputs)
        inputs.push_back(input.architectureJson());
    j["inputs"] = inputs;

    json epilogueInputs = json::array();
    for (const auto &[name, tensor] : epilogueInputBindings)
        epilogueInputs.push_back(json{{"name", name}, {"tensor", tensor.architectureJson()}});
    j["epilogue_inputs"] = epilogueInputs;

    json outputs = json::array();
    for (const Tensor &output : standaloneLayerFeatureOutputs)
        outputs.push_back(output.architectureJson());
    j["outputs"] = outputs;
    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json Convolution1d::serialize(thor_file::TarWriter &archiveWriter,
                              Stream stream,
                              bool saveOptimizerState,
                              ThorImplementation::StampedNetwork &stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(
        j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void Convolution1d::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const json &j, Network *network) {
    if (j.at("version").get<string>() != "2.0.0")
        throw runtime_error("Unsupported version in Convolution1d::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "convolution_1d")
        throw runtime_error("Layer type mismatch in Convolution1d::deserialize: " + j.at("layer_type").get<string>());
    if (j.at("data_layout").get<string>() != "NCW")
        throw runtime_error("Convolution1d only supports serialized NCW data_layout, got " + j.at("data_layout").get<string>());

    vector<pair<string, Tensor>> epilogueInputBindings;
    if (j.contains("epilogue_inputs")) {
        for (const json &epilogueInputJson : j.at("epilogue_inputs")) {
            const string inputName = epilogueInputJson.at("name").get<string>();
            validateEpilogueAuxInputName(inputName);
            const uint64_t originalTensorId = epilogueInputJson.at("tensor").at("id").get<uint64_t>();
            epilogueInputBindings.emplace_back(inputName, network->getApiTensorByOriginalId(originalTensorId));
        }
    }
    vector<string> auxInputNames;
    for (const auto &[name, tensor] : epilogueInputBindings) {
        (void)tensor;
        auxInputNames.push_back(name);
    }

    optional<ThorImplementation::Expression> epilogue = nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition definition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(definition, auxInputNames);
    } else if (!epilogueInputBindings.empty()) {
        throw runtime_error("Convolution1d serialized epilogue_inputs require a non-null epilogue expression.");
    }

    Convolution1d convolution1d(epilogue, epilogueInputBindings);
    convolution1d.filterWidth = j.at("filter_width").get<uint32_t>();
    convolution1d.numOutputChannels = j.at("num_output_channels").get<uint32_t>();
    convolution1d.groups = j.at("groups").get<uint32_t>();
    convolution1d.hasBias = j.at("has_bias").get<bool>();
    if (convolution1d.filterWidth == 0 || convolution1d.numOutputChannels == 0 || convolution1d.groups == 0)
        throw runtime_error("Convolution1d serialized filter_width, num_output_channels, and groups must be positive.");
    convolution1d.paddingMode = paddingModeFromName(j.at("padding_mode").get<string>());
    convolution1d.spatial.stride = j.at("stride").get<int32_t>();
    convolution1d.spatial.dilation = j.at("dilation").get<int32_t>();
    convolution1d.spatial.pre_padding = j.at("padding_left").get<int32_t>();
    convolution1d.spatial.post_padding = j.at("padding_right").get<int32_t>();
    if (convolution1d.spatial.stride <= 0 || convolution1d.spatial.dilation <= 0 || convolution1d.spatial.pre_padding < 0 ||
        convolution1d.spatial.post_padding < 0) {
        throw runtime_error("Convolution1d serialized spatial geometry is invalid.");
    }

    if (j.contains("activation") && !j.at("activation").is_null())
        convolution1d.activation = Activation::deserializeTemplate(j.at("activation"));

    for (const json &inputJson : j.at("inputs")) {
        const uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        convolution1d.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
        convolution1d.standaloneLayerFeatureInputs.push_back(convolution1d.featureInputs.back());
    }
    if (convolution1d.featureInputs.size() != 1)
        throw runtime_error("Convolution1d deserialize expected exactly one feature input.");
    if (convolution1d.featureInputs.front().getDimensions().size() != 2 ||
        convolution1d.featureInputs.front().getDimensions()[0] == 0 ||
        convolution1d.featureInputs.front().getDimensions()[1] == 0) {
        throw runtime_error("Convolution1d serialized feature input must be a non-empty CW tensor.");
    }
    if (convolution1d.featureInputs.front().getDimensions()[0] % convolution1d.groups != 0 ||
        convolution1d.numOutputChannels % convolution1d.groups != 0)
        throw runtime_error("Convolution1d serialized grouped channel geometry is invalid.");

    const auto expectedSpatial = resolveSpatial(convolution1d.paddingMode,
                                                static_cast<uint32_t>(convolution1d.featureInputs.front().getDimensions()[1]),
                                                convolution1d.filterWidth,
                                                static_cast<uint32_t>(convolution1d.spatial.stride),
                                                static_cast<uint32_t>(convolution1d.spatial.dilation),
                                                static_cast<uint32_t>(convolution1d.spatial.pre_padding),
                                                static_cast<uint32_t>(convolution1d.spatial.post_padding));
    if (expectedSpatial.pre_padding != convolution1d.spatial.pre_padding ||
        expectedSpatial.post_padding != convolution1d.spatial.post_padding) {
        throw runtime_error("Convolution1d serialized padding does not match its padding mode, input shape, stride, dilation, and filter.");
    }

    for (const json &outputJson : j.at("outputs")) {
        Tensor output = Tensor::deserialize(outputJson, archiveReader.get());
        convolution1d.featureOutputs.push_back(output);
        convolution1d.standaloneLayerFeatureOutputs.push_back(output);
    }
    if (convolution1d.featureInputs.size() != convolution1d.featureOutputs.size())
        throw runtime_error("Convolution1d deserialize expected equal numbers of inputs and outputs.");

    const uint32_t expectedOutputWidth = Builder::computeOutputWidth(
        static_cast<uint32_t>(convolution1d.featureInputs.front().getDimensions()[1]), convolution1d.filterWidth, convolution1d.spatial);
    for (uint32_t i = 0; i < convolution1d.featureInputs.size(); ++i) {
        if (convolution1d.featureInputs[i].getDimensions().size() != 2 || convolution1d.featureOutputs[i].getDimensions().size() != 2 ||
            convolution1d.featureOutputs[i].getDimensions()[0] != convolution1d.numOutputChannels ||
            convolution1d.featureOutputs[i].getDimensions()[1] != expectedOutputWidth) {
            throw runtime_error("Convolution1d serialized input/output shape is inconsistent with its convolution geometry.");
        }
        convolution1d.outputTensorFromInputTensor[convolution1d.featureInputs[i]] = convolution1d.featureOutputs[i];
        convolution1d.inputTensorFromOutputTensor[convolution1d.featureOutputs[i]] = convolution1d.featureInputs[i];
    }

    if (!convolution1d.epilogueInputBindings.empty()) {
        if (convolution1d.featureOutputs.size() != 1)
            throw runtime_error("Convolution1d serialized epilogue_inputs require exactly one primary convolution output.");
        for (const auto &[name, tensor] : convolution1d.epilogueInputBindings) {
            (void)name;
            if (tensor.getDataType() != convolution1d.featureOutputs[0].getDataType() ||
                tensor.getDimensions() != convolution1d.featureOutputs[0].getDimensions()) {
                throw runtime_error("Convolution1d serialized epilogue input does not match the convolution output.");
            }
            convolution1d.outputTensorFromInputTensor[tensor] = convolution1d.featureOutputs[0];
        }
    }

    if (j.contains("parameters")) {
        const json &parametersJson = j.at("parameters");
        if (!parametersJson.is_object())
            throw runtime_error("Convolution1d parameters must be an object keyed by parameter name.");
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            convolution1d.addParameter(make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }
    if (!convolution1d.hasParameter("weights"))
        throw runtime_error("Convolution1d deserialize did not find required weights parameter.");
    if (convolution1d.hasBias && !convolution1d.hasParameter("biases"))
        throw runtime_error("Convolution1d deserialize did not find required biases parameter.");

    convolution1d.initialized = true;
    convolution1d.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("convolution_1d", &Thor::Convolution1d::deserialize);
    return true;
}();
}  // namespace
