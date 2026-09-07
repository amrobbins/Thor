#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"
#include <optional>
#include <map>
#include <set>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

ThorImplementation::DynamicExpression buildConvolution3dExpression(bool hasBias,
                                                                    uint32_t groups,
                                                                    ThorImplementation::ConvolutionSpatial3d spatial,
                                                                    ThorImplementation::DataType computeDataType,
                                                                    ThorImplementation::TensorPlacement placement,
                                                                    std::shared_ptr<Thor::Activation> activation,
                                                                    std::optional<ThorImplementation::Expression> epilogue,
                                                                    std::vector<std::string> epilogueAuxInputNames) {
    using ImplDataType = ThorImplementation::DataType;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    std::vector<std::string> expectedInputNames = {"feature_input"};
    expectedInputNames.insert(expectedInputNames.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());
    expectedInputNames.push_back("weights");
    if (hasBias) {
        expectedInputNames.push_back("biases");
    }

    return DynamicExpression(std::move(expectedInputNames), {"feature_output"},
                             [hasBias,
                              groups,
                              spatial,
                              computeDataType,
                              placement,
                              activation = std::move(activation),
                              epilogue,
                              epilogueAuxInputNames = std::move(epilogueAuxInputNames)](
                                 const DynamicExpression::TensorMap& inputs,
                                 const DynamicExpression::TensorMap& outputs,
                                 Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");
        THOR_THROW_IF_FALSE(wTensor.getPlacement() == placement);

        if (featureInputTensor.getDimensions().size() != 5) {
            throw std::runtime_error("Convolution3d expects feature_input to be 5D NCDHW.");
        }
        if (wTensor.getDimensions().size() != 5) {
            throw std::runtime_error("Convolution3d expects weights to be 5D KCDHW.");
        }
        if (groups == 0 || featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1] * groups ||
            wTensor.getDimensions()[0] % groups != 0) {
            throw std::runtime_error("Convolution3d grouped channel geometry is invalid.");
        }
        THOR_THROW_IF_FALSE(featureInputTensor.getPlacement() == placement);

        auto outputDimension = [](uint64_t inputSize,
                                  uint64_t filterSize,
                                  int32_t stride,
                                  int32_t dilation,
                                  int32_t prePadding,
                                  int32_t postPadding,
                                  const char *axis) -> uint64_t {
            if (filterSize == 0 || stride <= 0 || dilation <= 0 || prePadding < 0 || postPadding < 0) {
                throw std::runtime_error(std::string("Convolution3d has invalid ") + axis + " spatial geometry.");
            }
            const uint64_t effectiveFilter =
                static_cast<uint64_t>(dilation) * (filterSize - 1ULL) + 1ULL;
            const uint64_t paddedInput =
                inputSize + static_cast<uint64_t>(prePadding) + static_cast<uint64_t>(postPadding);
            if (effectiveFilter > paddedInput) {
                throw std::runtime_error(std::string("Convolution3d effective ") + axis +
                                         " filter is larger than the padded input.");
            }
            return 1ULL + (paddedInput - effectiveFilter) / static_cast<uint64_t>(stride);
        };

        const uint64_t expectedOutputDepth = outputDimension(featureInputTensor.getDimensions()[2],
                                                             wTensor.getDimensions()[2],
                                                             spatial.stride_d,
                                                             spatial.dilation_d,
                                                             spatial.pre_padding_d,
                                                             spatial.post_padding_d,
                                                             "depth");
        const uint64_t expectedOutputRows = outputDimension(featureInputTensor.getDimensions()[3],
                                                            wTensor.getDimensions()[3],
                                                            spatial.stride_h,
                                                            spatial.dilation_h,
                                                            spatial.pre_padding_h,
                                                            spatial.post_padding_h,
                                                            "height");
        const uint64_t expectedOutputCols = outputDimension(featureInputTensor.getDimensions()[4],
                                                            wTensor.getDimensions()[4],
                                                            spatial.stride_w,
                                                            spatial.dilation_w,
                                                            spatial.pre_padding_w,
                                                            spatial.post_padding_w,
                                                            "width");
        std::optional<ImplDataType> featureOutputDType = std::nullopt;

        if (outputs.contains("feature_output")) {
            const Tensor& featureOutputTensor = outputs.at("feature_output");
            if (featureOutputTensor.getDimensions().size() != 5) {
                throw std::runtime_error("Convolution3d expects feature_output to be 5D NCDHW.");
            }
            if (featureOutputTensor.getDimensions()[0] != featureInputTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[2] != expectedOutputDepth ||
                featureOutputTensor.getDimensions()[3] != expectedOutputRows ||
                featureOutputTensor.getDimensions()[4] != expectedOutputCols) {
                throw std::runtime_error("Convolution3d feature_output shape does not match the implied convolution output shape.");
            }
            THOR_THROW_IF_FALSE(featureOutputTensor.getPlacement() == placement);
            featureOutputDType = featureOutputTensor.getDescriptor().getDataType();
        }

        const ImplDataType weightsDType = wTensor.getDescriptor().getDataType();

        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);

        Expression fout = Expression::conv3d(fin, w, spatial, computeDataType, featureOutputDType, groups);

        if (hasBias) {
            const Tensor& bTensor = inputs.at("biases");
            if (bTensor.getDimensions().size() != 1) {
                throw std::runtime_error("Convolution3d expects biases to be 1D [K].");
            }
            if (bTensor.getDimensions()[0] != wTensor.getDimensions()[0]) {
                throw std::runtime_error("Convolution3d bias size must match number of output channels.");
            }

            const ImplDataType biasDType = bTensor.getDescriptor().getDataType();
            auto b = Expression::input("biases", biasDType, biasDType).unsqueeze({0, 2, 3, 4});
            fout = fout + b;
        }

        if (activation != nullptr) {
            fout = activation->toExpression(fout);
        }
        for (const std::string& auxInputName : epilogueAuxInputNames) {
            const Tensor& auxTensor = inputs.at(auxInputName);
            const std::vector<uint64_t> expectedAuxShape = {
                featureInputTensor.getDimensions()[0], wTensor.getDimensions()[0], expectedOutputDepth, expectedOutputRows, expectedOutputCols};
            if (auxTensor.getDimensions() != expectedAuxShape) {
                throw std::runtime_error("Convolution3d epilogue auxiliary input '" + auxInputName +
                                         "' shape must match the convolution feature output shape.");
            }
            if (featureOutputDType.has_value() && auxTensor.getDataType() != featureOutputDType.value()) {
                throw std::runtime_error("Convolution3d epilogue auxiliary input '" + auxInputName +
                                         "' dtype must match the convolution feature output dtype.");
            }
            THOR_THROW_IF_FALSE(auxTensor.getPlacement() == placement);
        }
        if (epilogue.has_value()) {
            fout = Convolution3d::applyEpilogue(fout, epilogue.value());
        }
        if (featureOutputDType.has_value()) {
            fout = fout.withOutputDType(featureOutputDType.value());
        }

        auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {},
        };
    });
}

const char *paddingModeName(Convolution3dPaddingMode mode) {
    switch (mode) {
        case Convolution3dPaddingMode::VALID:
            return "valid";
        case Convolution3dPaddingMode::SAME_UPPER:
            return "same_upper";
        case Convolution3dPaddingMode::EXPLICIT:
            return "explicit";
    }
    throw std::runtime_error("Unknown Convolution3d padding mode.");
}

Convolution3dPaddingMode paddingModeFromName(const std::string &name) {
    if (name == "valid")
        return Convolution3dPaddingMode::VALID;
    if (name == "same_upper")
        return Convolution3dPaddingMode::SAME_UPPER;
    if (name == "explicit")
        return Convolution3dPaddingMode::EXPLICIT;
    throw std::runtime_error("Unknown Convolution3d padding_mode: " + name);
}

}  // namespace

void Convolution3d::validateEpilogueAuxInputName(const std::string& inputName) {
    if (inputName.empty()) {
        throw std::invalid_argument("Convolution3d epilogue auxiliary input name cannot be empty.");
    }
    if (inputName.rfind("__", 0) == 0) {
        throw std::invalid_argument("Convolution3d epilogue auxiliary input names cannot start with __: " + inputName + ".");
    }
    static const std::set<std::string> reservedNames = {
        "feature_input",
        "feature_output",
        "weights",
        "biases",
        epilogueInputName(),
        epilogueOutputName(),
    };
    if (reservedNames.contains(inputName)) {
        throw std::invalid_argument("Convolution3d epilogue auxiliary input name is reserved: " + inputName + ".");
    }
}

std::vector<std::string> Convolution3d::epilogueAuxInputNames() const {
    std::vector<std::string> names;
    names.reserve(epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)tensor;
        names.push_back(name);
    }
    return names;
}

std::vector<Tensor> Convolution3d::getFeatureInputs() const {
    std::vector<Tensor> inputs = featureInputs;
    inputs.reserve(inputs.size() + epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        inputs.push_back(tensor);
    }
    return inputs;
}

std::vector<uint32_t> Convolution3d::inputPortIndicesForTensor(Tensor tensor) const {
    std::vector<uint32_t> ports;
    if (!featureInputs.empty() && tensor.getOriginalId() == featureInputs[0].getOriginalId()) {
        ports.push_back(0);
    }
    for (uint32_t i = 0; i < epilogueInputBindings.size(); ++i) {
        if (tensor.getOriginalId() == epilogueInputBindings[i].second.getOriginalId()) {
            ports.push_back(i + 1);
        }
    }
    return ports;
}

Tensor Convolution3d::getFeatureOutput(Tensor inputTensor) const {
    std::map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
    if (it == outputTensorFromInputTensor.end()) {
        throw std::runtime_error("Tensor is not connected to this Convolution3d layer.");
    }
    return it->second;
}

std::vector<Tensor> Convolution3d::getOutputsFromInput(Tensor inputTensor) {
    if (epilogueInputBindings.empty()) {
        return {getFeatureOutput(inputTensor)};
    }

    (void)getFeatureOutput(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected) {
        return {};
    }
    const uint32_t requiredInputPorts = static_cast<uint32_t>(1 + epilogueInputBindings.size());
    if (connectedInputPortIndices.size() != requiredInputPorts) {
        return {};
    }

    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs[0]};
}

void Convolution3d::informThatInputConnectionMade(Tensor inputTensor) {
    if (epilogueInputBindings.empty()) {
        return;
    }
    std::vector<uint32_t> ports = inputPortIndicesForTensor(inputTensor);
    if (ports.empty()) {
        throw std::runtime_error("Convolution3d informed of connection for unknown input tensor.");
    }
    for (uint32_t port : ports) {
        connectedInputPortIndices.insert(port);
    }
}

void Convolution3d::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
}

int Convolution3d::getConnectionType(Tensor connectingTensor) const {
    if (!epilogueInputBindings.empty()) {
        std::vector<uint32_t> inputPorts = inputPortIndicesForTensor(connectingTensor);
        if (!inputPorts.empty()) {
            uint32_t& cursor = nextInputConnectionCursorByTensorOriginalId[connectingTensor.getOriginalId()];
            const uint32_t port = inputPorts[cursor % inputPorts.size()];
            ++cursor;
            return static_cast<int>(port);
        }
    } else {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return static_cast<int>(i);
        }
    }

    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        if (connectingTensor == featureOutputs[i])
            return static_cast<int>(i);
    }
    throw std::runtime_error("Tensor is not connected to this Convolution3d layer.");
}

std::shared_ptr<ThorImplementation::Layer> Convolution3d::stamp(ThorImplementation::TensorPlacement placement,
                                                                 std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                                 std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                                 Thor::Tensor connectingApiTensor,
                                                                 const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;

    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    std::shared_ptr<ThorImplementation::CustomLayer> physicalConvolution3d = std::make_shared<ThorImplementation::CustomLayer>(
        buildConvolution3dExpression(hasBias,
                                     groups,
                                     spatial,
                                     computeDataType,
                                     placement,
                                     activation,
                                     epilogue,
                                     epilogueAuxInputNames()),
        [&]() {
            std::vector<std::string> inputNames = {"feature_input"};
            std::vector<std::string> auxNames = epilogueAuxInputNames();
            inputNames.insert(inputNames.end(), auxNames.begin(), auxNames.end());
            return inputNames;
        }(),
        std::vector<std::string>{"feature_output"},
        placement,
        physicalParameters,
        inferenceOnly,
        getId());
    physicalConvolution3d->setLayerName(getLayerType());

    return physicalConvolution3d;
}

void Convolution3d::buildSupportLayersAndAddToNetwork(Network* network) {
    vector<Tensor> currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    Convolution3d::Builder convolution3dBuilder;
    convolution3dBuilder.network(*network)
        .numOutputChannels(numOutputChannels)
        .filterDepth(filterDepth)
        .filterHeight(filterHeight)
        .filterWidth(filterWidth)
        .depthStride(static_cast<uint32_t>(spatial.stride_d))
        .verticalStride(static_cast<uint32_t>(spatial.stride_h))
        .horizontalStride(static_cast<uint32_t>(spatial.stride_w))
        .depthDilation(static_cast<uint32_t>(spatial.dilation_d))
        .verticalDilation(static_cast<uint32_t>(spatial.dilation_h))
        .horizontalDilation(static_cast<uint32_t>(spatial.dilation_w))
        .groups(groups)
        .hasBias(hasBias)
        .weightsInitializer(weightsInitializer)
        .biasInitializer(biasInitializer)
        .weightsOptimizer(weightsOptimizer)
        .biasesOptimizer(biasesOptimizer);
    switch (paddingMode) {
        case Convolution3dPaddingMode::VALID:
            convolution3dBuilder.validPadding();
            break;
        case Convolution3dPaddingMode::SAME_UPPER:
            convolution3dBuilder.samePadding();
            break;
        case Convolution3dPaddingMode::EXPLICIT:
            convolution3dBuilder.padding(static_cast<uint32_t>(spatial.pre_padding_d),
                                         static_cast<uint32_t>(spatial.post_padding_d),
                                         static_cast<uint32_t>(spatial.pre_padding_h),
                                         static_cast<uint32_t>(spatial.post_padding_h),
                                         static_cast<uint32_t>(spatial.pre_padding_w),
                                         static_cast<uint32_t>(spatial.post_padding_w));
            break;
    }
    if (activation != nullptr) {
        convolution3dBuilder.activation(dynamic_pointer_cast<Activation>(activation->clone()));
    } else {
        convolution3dBuilder.noActivation();
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        convolution3dBuilder.epilogueInput(name, tensor);
    }
    if (epilogue.has_value()) {
        convolution3dBuilder.epilogue(epilogue.value());
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        convolution3dBuilder.featureInput(currentFeatureInputs[i]);
    Convolution3d convolution3d = convolution3dBuilder.build();
    this->id = convolution3d.getId();

    standaloneLayerFeatureInputs = convolution3d.featureInputs;
    standaloneLayerFeatureOutputs = convolution3d.getFeatureOutputs();
    currentFeatureInputs = standaloneLayerFeatureOutputs;

    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    featureOutputs = currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        outputTensorFromInputTensor[tensor] = featureOutputs[0];
    }
}

json Convolution3d::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "convolution_3d";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["data_layout"] = "NCDHW";
    j["filter_width"] = filterWidth;
    j["filter_height"] = filterHeight;
    j["filter_depth"] = filterDepth;
    j["horizontal_stride"] = spatial.stride_w;
    j["vertical_stride"] = spatial.stride_h;
    j["depth_stride"] = spatial.stride_d;
    j["depth_dilation"] = spatial.dilation_d;
    j["vertical_dilation"] = spatial.dilation_h;
    j["horizontal_dilation"] = spatial.dilation_w;
    j["padding_mode"] = paddingModeName(paddingMode);
    j["padding_front"] = spatial.pre_padding_d;
    j["padding_back"] = spatial.post_padding_d;
    j["padding_top"] = spatial.pre_padding_h;
    j["padding_bottom"] = spatial.post_padding_h;
    j["padding_left"] = spatial.pre_padding_w;
    j["padding_right"] = spatial.post_padding_w;
    j["num_output_channels"] = numOutputChannels;
    j["groups"] = groups;
    j["has_bias"] = hasBias;
    j["compute_data_type"] = computeDataType;
    if (activation != nullptr) {
        j["activation"] = activation->architectureJson();
    } else {
        j["activation"] = nullptr;
    }
    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value())
            serializableEpilogue = makeEpilogueDefinition(epilogue.value(), epilogueAuxInputNames());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    json inputs = json::array();
    for (uint32_t i = 0; i < standaloneLayerFeatureInputs.size(); ++i) {
        inputs.push_back(standaloneLayerFeatureInputs[i].architectureJson());
    }
    j["inputs"] = inputs;

    json epilogueInputs = json::array();
    for (const auto& [name, tensor] : epilogueInputBindings) {
        epilogueInputs.push_back(json{{"name", name}, {"tensor", tensor.architectureJson()}});
    }
    j["epilogue_inputs"] = epilogueInputs;

    json outputs = json::array();
    for (uint32_t i = 0; i < standaloneLayerFeatureOutputs.size(); ++i) {
        outputs.push_back(standaloneLayerFeatureOutputs[i].architectureJson());
    }
    j["outputs"] = outputs;

    j["parameters"] = getParametersArchitectureJson()["parameters"];

    return j;
}

json Convolution3d::serialize(thor_file::TarWriter& archiveWriter,
                              Stream stream,
                              bool saveOptimizerState,
                              ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void Convolution3d::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in Convolution3d::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "convolution_3d")
        throw runtime_error("Layer type mismatch in Convolution3d::deserialize: " + j.at("layer_type").get<std::string>());
    if (j.at("data_layout").get<string>() != "NCDHW")
        throw runtime_error("Convolution3d only supports serialized NCDHW data_layout, got " + j.at("data_layout").get<string>());

    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    if (j.contains("epilogue_inputs")) {
        for (const json& epilogueInputJson : j.at("epilogue_inputs")) {
            std::string inputName = epilogueInputJson.at("name").get<std::string>();
            validateEpilogueAuxInputName(inputName);
            uint64_t originalTensorId = epilogueInputJson.at("tensor").at("id").get<uint64_t>();
            epilogueInputBindings.emplace_back(inputName, network->getApiTensorByOriginalId(originalTensorId));
        }
    }
    std::vector<std::string> auxInputNames;
    auxInputNames.reserve(epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)tensor;
        auxInputNames.push_back(name);
    }

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition, auxInputNames);
    } else if (!epilogueInputBindings.empty()) {
        throw runtime_error("Convolution3d serialized epilogue_inputs require a non-null epilogue expression.");
    }

    Convolution3d convolution3d(epilogue, epilogueInputBindings);
    convolution3d.filterWidth = j.at("filter_width").get<uint32_t>();
    convolution3d.filterHeight = j.at("filter_height").get<uint32_t>();
    convolution3d.filterDepth = j.at("filter_depth").get<uint32_t>();
    convolution3d.spatial.stride_w = j.at("horizontal_stride").get<int32_t>();
    convolution3d.spatial.stride_h = j.at("vertical_stride").get<int32_t>();
    convolution3d.spatial.stride_d = j.at("depth_stride").get<int32_t>();
    convolution3d.spatial.dilation_d = j.at("depth_dilation").get<int32_t>();
    convolution3d.spatial.dilation_h = j.at("vertical_dilation").get<int32_t>();
    convolution3d.spatial.dilation_w = j.at("horizontal_dilation").get<int32_t>();
    convolution3d.paddingMode = paddingModeFromName(j.at("padding_mode").get<std::string>());
    convolution3d.spatial.pre_padding_d = j.at("padding_front").get<int32_t>();
    convolution3d.spatial.post_padding_d = j.at("padding_back").get<int32_t>();
    convolution3d.spatial.pre_padding_h = j.at("padding_top").get<int32_t>();
    convolution3d.spatial.post_padding_h = j.at("padding_bottom").get<int32_t>();
    convolution3d.spatial.pre_padding_w = j.at("padding_left").get<int32_t>();
    convolution3d.spatial.post_padding_w = j.at("padding_right").get<int32_t>();
    if (convolution3d.spatial.stride_d <= 0 || convolution3d.spatial.stride_h <= 0 || convolution3d.spatial.stride_w <= 0) {
        throw runtime_error("Convolution3d serialized stride must be positive.");
    }
    if (convolution3d.spatial.dilation_d <= 0 || convolution3d.spatial.dilation_h <= 0 || convolution3d.spatial.dilation_w <= 0) {
        throw runtime_error("Convolution3d serialized dilation must be positive.");
    }
    if (convolution3d.spatial.pre_padding_d < 0 || convolution3d.spatial.post_padding_d < 0 ||
        convolution3d.spatial.pre_padding_h < 0 || convolution3d.spatial.post_padding_h < 0 ||
        convolution3d.spatial.pre_padding_w < 0 || convolution3d.spatial.post_padding_w < 0) {
        throw runtime_error("Convolution3d serialized padding must be non-negative.");
    }
    if (convolution3d.paddingMode == Convolution3dPaddingMode::VALID &&
        (convolution3d.spatial.pre_padding_d != 0 || convolution3d.spatial.post_padding_d != 0 ||
         convolution3d.spatial.pre_padding_h != 0 || convolution3d.spatial.post_padding_h != 0 ||
         convolution3d.spatial.pre_padding_w != 0 || convolution3d.spatial.post_padding_w != 0)) {
        throw runtime_error("Convolution3d serialized VALID padding must resolve to zero padding.");
    }
    convolution3d.numOutputChannels = j.at("num_output_channels").get<uint32_t>();
    convolution3d.groups = j.at("groups").get<uint32_t>();
    if (convolution3d.filterDepth == 0 || convolution3d.filterHeight == 0 || convolution3d.filterWidth == 0 ||
        convolution3d.numOutputChannels == 0 || convolution3d.groups == 0) {
        throw runtime_error("Convolution3d serialized filter dimensions, num_output_channels, and groups must be positive.");
    }
    convolution3d.hasBias = j.at("has_bias").get<bool>();
    convolution3d.computeDataType = j.at("compute_data_type").get<DataType>();
    if (convolution3d.computeDataType != DataType::FP32 && convolution3d.computeDataType != DataType::TF32)
        throw runtime_error("Convolution3d serialized compute_data_type must be fp32 or tf32.");

    if (j.contains("activation") && !j.at("activation").is_null()) {
        convolution3d.activation = Activation::deserializeTemplate(j.at("activation"));
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        convolution3d.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
        convolution3d.standaloneLayerFeatureInputs.push_back(convolution3d.featureInputs.back());
    }
    if (convolution3d.featureInputs.empty()) {
        throw runtime_error("Convolution3d serialized layer requires at least one feature input.");
    }
    const vector<uint64_t>& referenceInputDimensions = convolution3d.featureInputs.front().getDimensions();
    if (referenceInputDimensions.size() != 4 || referenceInputDimensions[0] == 0 || referenceInputDimensions[1] == 0 ||
        referenceInputDimensions[2] == 0 || referenceInputDimensions[3] == 0) {
        throw runtime_error("Convolution3d serialized feature input must be a non-empty CDHW tensor.");
    }
    for (const Tensor& input : convolution3d.featureInputs) {
        if (input.getDimensions() != referenceInputDimensions) {
            throw runtime_error("Convolution3d serialized feature inputs must have identical CDHW dimensions.");
        }
        if (input.getDimensions()[0] % convolution3d.groups != 0 || convolution3d.numOutputChannels % convolution3d.groups != 0) {
            throw runtime_error("Convolution3d serialized groups must divide input and output channels.");
        }
    }
    if (convolution3d.paddingMode == Convolution3dPaddingMode::SAME_UPPER) {
        const auto [expectedFront, expectedBack] = Convolution3d::Builder::computeSamePadding(
            static_cast<uint32_t>(referenceInputDimensions[1]),
            convolution3d.spatial.stride_d,
            convolution3d.filterDepth,
            convolution3d.spatial.dilation_d);
        const auto [expectedTop, expectedBottom] = Convolution3d::Builder::computeSamePadding(
            static_cast<uint32_t>(referenceInputDimensions[2]),
            convolution3d.spatial.stride_h,
            convolution3d.filterHeight,
            convolution3d.spatial.dilation_h);
        const auto [expectedLeft, expectedRight] = Convolution3d::Builder::computeSamePadding(
            static_cast<uint32_t>(referenceInputDimensions[3]),
            convolution3d.spatial.stride_w,
            convolution3d.filterWidth,
            convolution3d.spatial.dilation_w);
        if (convolution3d.spatial.pre_padding_d != static_cast<int32_t>(expectedFront) ||
            convolution3d.spatial.post_padding_d != static_cast<int32_t>(expectedBack) ||
            convolution3d.spatial.pre_padding_h != static_cast<int32_t>(expectedTop) ||
            convolution3d.spatial.post_padding_h != static_cast<int32_t>(expectedBottom) ||
            convolution3d.spatial.pre_padding_w != static_cast<int32_t>(expectedLeft) ||
            convolution3d.spatial.post_padding_w != static_cast<int32_t>(expectedRight)) {
            throw runtime_error(
                "Convolution3d serialized SAME_UPPER padding does not match its input shape, stride, dilation, and filter.");
        }
    }
    const vector<uint64_t> expectedOutputDimensions = {
        convolution3d.numOutputChannels,
        Convolution3d::Builder::computeOutputDimension(static_cast<uint32_t>(referenceInputDimensions[1]),
                                                       convolution3d.spatial.stride_d,
                                                       convolution3d.filterDepth,
                                                       convolution3d.spatial.pre_padding_d,
                                                       convolution3d.spatial.post_padding_d,
                                                       convolution3d.spatial.dilation_d),
        Convolution3d::Builder::computeOutputDimension(static_cast<uint32_t>(referenceInputDimensions[2]),
                                                       convolution3d.spatial.stride_h,
                                                       convolution3d.filterHeight,
                                                       convolution3d.spatial.pre_padding_h,
                                                       convolution3d.spatial.post_padding_h,
                                                       convolution3d.spatial.dilation_h),
        Convolution3d::Builder::computeOutputDimension(static_cast<uint32_t>(referenceInputDimensions[3]),
                                                       convolution3d.spatial.stride_w,
                                                       convolution3d.filterWidth,
                                                       convolution3d.spatial.pre_padding_w,
                                                       convolution3d.spatial.post_padding_w,
                                                       convolution3d.spatial.dilation_w)};
    for (const json& outputJson : j.at("outputs")) {
        Tensor output = Tensor::deserialize(outputJson, archiveReader.get());
        convolution3d.featureOutputs.push_back(output);
        convolution3d.standaloneLayerFeatureOutputs.push_back(output);
    }
    if (convolution3d.featureInputs.size() != convolution3d.featureOutputs.size()) {
        throw runtime_error("Convolution3d deserialize expected equal numbers of inputs and outputs.");
    }
    for (const Tensor& output : convolution3d.featureOutputs) {
        if (output.getDimensions() != expectedOutputDimensions) {
            throw runtime_error("Convolution3d serialized feature output shape does not match its spatial geometry.");
        }
        if (output.getDataType() != convolution3d.featureInputs.front().getDataType()) {
            throw runtime_error("Convolution3d serialized feature output dtype must match its feature input dtype.");
        }
    }
    if (convolution3d.computeDataType == DataType::TF32 && convolution3d.featureInputs.front().getDataType() != DataType::FP32) {
        throw runtime_error("Convolution3d serialized TF32 compute requires FP32 input/weights/output storage.");
    }
    for (uint32_t i = 0; i < convolution3d.featureInputs.size(); ++i) {
        convolution3d.outputTensorFromInputTensor[convolution3d.featureInputs[i]] = convolution3d.featureOutputs[i];
        convolution3d.inputTensorFromOutputTensor[convolution3d.featureOutputs[i]] = convolution3d.featureInputs[i];
    }
    if (!convolution3d.epilogueInputBindings.empty()) {
        if (convolution3d.featureOutputs.size() != 1) {
            throw runtime_error("Convolution3d serialized epilogue_inputs require exactly one primary convolution output.");
        }
        for (const auto& [name, tensor] : convolution3d.epilogueInputBindings) {
            (void)name;
            if (tensor.getDataType() != convolution3d.featureOutputs[0].getDataType()) {
                throw runtime_error("Convolution3d serialized epilogue input dtype does not match the convolution output dtype.");
            }
            if (tensor.getDimensions() != convolution3d.featureOutputs[0].getDimensions()) {
                throw runtime_error("Convolution3d serialized epilogue input shape does not match the convolution output shape.");
            }
            convolution3d.outputTensorFromInputTensor[tensor] = convolution3d.featureOutputs[0];
        }
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("Convolution3d parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            convolution3d.addParameter(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }

    if (!convolution3d.hasParameter("weights")) {
        throw runtime_error("Convolution3d deserialize did not find required weights parameter.");
    }
    if (convolution3d.hasBias && !convolution3d.hasParameter("biases")) {
        throw runtime_error("Convolution3d deserialize did not find required biases parameter.");
    }

    convolution3d.initialized = true;
    convolution3d.addToNetwork(network);
}


}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("convolution_3d", &Thor::Convolution3d::deserialize);
    return true;
}();
}  // namespace
