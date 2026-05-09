#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"
#include <optional>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

ThorImplementation::DynamicExpression buildConvolution3dExpression(bool hasBias,
                                                                    uint32_t strideD,
                                                                    uint32_t strideH,
                                                                    uint32_t strideW,
                                                                    uint32_t padD,
                                                                    uint32_t padH,
                                                                    uint32_t padW,
                                                                    ThorImplementation::TensorPlacement placement,
                                                                    std::shared_ptr<Thor::Activation> activation,
                                                                    std::optional<ThorImplementation::Expression> epilogue) {
    using ImplDataType = ThorImplementation::TensorDescriptor::DataType;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    return DynamicExpression([hasBias, strideD, strideH, strideW, padD, padH, padW, placement, activation = std::move(activation), epilogue](
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
        if (featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
            throw std::runtime_error("Convolution3d input channels must match weight channels.");
        }
        THOR_THROW_IF_FALSE(featureInputTensor.getPlacement() == placement);

        const uint64_t expectedOutputDepth =
            (featureInputTensor.getDimensions()[2] + 2 * padD - wTensor.getDimensions()[2]) / strideD + 1;
        const uint64_t expectedOutputRows =
            (featureInputTensor.getDimensions()[3] + 2 * padH - wTensor.getDimensions()[3]) / strideH + 1;
        const uint64_t expectedOutputCols =
            (featureInputTensor.getDimensions()[4] + 2 * padW - wTensor.getDimensions()[4]) / strideW + 1;
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

        Expression fout = Expression::conv3d(fin, w, strideD, strideH, strideW, padD, padH, padW);

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
            {outputs},
            {},
        };
    });
}

}  // namespace

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
                                     depthStride,
                                     verticalStride,
                                     horizontalStride,
                                     depthPadding,
                                     verticalPadding,
                                     horizontalPadding,
                                     placement,
                                     activation,
                                     epilogue),
        placement,
        physicalParameters,
        inferenceOnly,
        getId(),
        false);
    physicalConvolution3d->setLayerName(getLayerType());

    return physicalConvolution3d;
}

void Convolution3d::buildSupportLayersAndAddToNetwork(Network* network) {
    vector<Tensor> currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    if (featureInputs.front().getDataType() != Tensor::DataType::FP16) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            TypeConverter typeConverter = TypeConverter::Builder()
                                              .network(*network)
                                              .featureInput(currentFeatureInputs[i])
                                              .newDataType(Tensor::DataType::FP16)
                                              .build();
            currentFeatureInputs[i] = typeConverter.getFeatureOutput().value();
        }
    }

    Convolution3d::Builder convolution3dBuilder;
    convolution3dBuilder.network(*network)
        .numOutputChannels(numOutputChannels)
        .filterDepth(filterDepth)
        .filterHeight(filterHeight)
        .filterWidth(filterWidth)
        .depthStride(depthStride)
        .verticalStride(verticalStride)
        .horizontalStride(horizontalStride)
        .depthPadding(depthPadding)
        .verticalPadding(verticalPadding)
        .horizontalPadding(horizontalPadding)
        .hasBias(hasBias)
        .weightsInitializer(weightsInitializer)
        .biasInitializer(biasInitializer)
        .weightsOptimizer(weightsOptimizer)
        .biasesOptimizer(biasesOptimizer);
    if (activation != nullptr) {
        convolution3dBuilder.activation(dynamic_pointer_cast<Activation>(activation->clone()));
    } else {
        convolution3dBuilder.noActivation();
    }
    if (epilogue.has_value()) {
        convolution3dBuilder.epilogue(epilogue.value());
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        convolution3dBuilder.featureInput(currentFeatureInputs[i]);
    Convolution3d convolution3d = convolution3dBuilder.build();
    this->id = convolution3d.getId();

    standaloneLayerFeatureInputs = convolution3d.getFeatureInputs();
    standaloneLayerFeatureOutputs = convolution3d.getFeatureOutputs();
    currentFeatureInputs = standaloneLayerFeatureOutputs;

    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    featureOutputs = currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
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
    j["horizontal_stride"] = horizontalStride;
    j["vertical_stride"] = verticalStride;
    j["depth_stride"] = depthStride;
    j["horizontal_padding"] = horizontalPadding;
    j["vertical_padding"] = verticalPadding;
    j["depth_padding"] = depthPadding;
    j["num_output_channels"] = numOutputChannels;
    j["has_bias"] = hasBias;
    if (activation != nullptr) {
        j["activation"] = activation->architectureJson();
    } else {
        j["activation"] = nullptr;
    }
    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value())
            serializableEpilogue = makeEpilogueDefinition(epilogue.value());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    json inputs = json::array();
    for (uint32_t i = 0; i < standaloneLayerFeatureInputs.size(); ++i) {
        inputs.push_back(standaloneLayerFeatureInputs[i].architectureJson());
    }
    j["inputs"] = inputs;

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

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition);
    }

    Convolution3d convolution3d(epilogue);
    convolution3d.filterWidth = j.at("filter_width").get<uint32_t>();
    convolution3d.filterHeight = j.at("filter_height").get<uint32_t>();
    convolution3d.filterDepth = j.at("filter_depth").get<uint32_t>();
    convolution3d.horizontalStride = j.at("horizontal_stride").get<uint32_t>();
    convolution3d.verticalStride = j.at("vertical_stride").get<uint32_t>();
    convolution3d.depthStride = j.at("depth_stride").get<uint32_t>();
    convolution3d.horizontalPadding = j.at("horizontal_padding").get<uint32_t>();
    convolution3d.verticalPadding = j.at("vertical_padding").get<uint32_t>();
    convolution3d.depthPadding = j.at("depth_padding").get<uint32_t>();
    convolution3d.numOutputChannels = j.at("num_output_channels").get<uint32_t>();
    convolution3d.hasBias = j.at("has_bias").get<bool>();

    if (j.contains("activation") && !j.at("activation").is_null()) {
        convolution3d.activation = Activation::deserializeTemplate(j.at("activation"));
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        convolution3d.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
        convolution3d.standaloneLayerFeatureInputs.push_back(convolution3d.featureInputs.back());
    }
    for (const json& outputJson : j.at("outputs")) {
        Tensor output = Tensor::deserialize(outputJson, archiveReader.get());
        convolution3d.featureOutputs.push_back(output);
        convolution3d.standaloneLayerFeatureOutputs.push_back(output);
    }
    if (convolution3d.featureInputs.size() != convolution3d.featureOutputs.size()) {
        throw runtime_error("Convolution3d deserialize expected equal numbers of inputs and outputs.");
    }
    for (uint32_t i = 0; i < convolution3d.featureInputs.size(); ++i) {
        convolution3d.outputTensorFromInputTensor[convolution3d.featureInputs[i]] = convolution3d.featureOutputs[i];
        convolution3d.inputTensorFromOutputTensor[convolution3d.featureOutputs[i]] = convolution3d.featureInputs[i];
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
