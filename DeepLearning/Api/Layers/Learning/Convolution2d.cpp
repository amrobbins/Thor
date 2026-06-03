#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

ThorImplementation::DynamicExpression buildConvolution2dExpression(bool hasBias,
                                                                   uint32_t strideH,
                                                                   uint32_t strideW,
                                                                   uint32_t padH,
                                                                   uint32_t padW,
                                                                   ThorImplementation::TensorPlacement placement,
                                                                   std::shared_ptr<Thor::Activation> activation,
                                                                   std::optional<ThorImplementation::Expression> epilogue) {
    using ImplDataType = ThorImplementation::DataType;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    return DynamicExpression([hasBias, strideH, strideW, padH, padW, placement, activation = std::move(activation), epilogue](
                                 const DynamicExpression::TensorMap& inputs,
                                 const DynamicExpression::TensorMap& outputs,
                                 Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");
        THOR_THROW_IF_FALSE(wTensor.getPlacement() == placement);

        if (featureInputTensor.getDimensions().size() != 4) {
            throw std::runtime_error("Convolution2d expects feature_input to be 4D NCHW.");
        }
        if (wTensor.getDimensions().size() != 4) {
            throw std::runtime_error("Convolution2d expects weights to be 4D KCRS.");
        }
        if (featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
            throw std::runtime_error("Convolution2d input channels must match weight channels.");
        }
        THOR_THROW_IF_FALSE(featureInputTensor.getPlacement() == placement);

        const uint64_t expectedOutputRows = (featureInputTensor.getDimensions()[2] + 2 * padH - wTensor.getDimensions()[2]) / strideH + 1;
        const uint64_t expectedOutputCols = (featureInputTensor.getDimensions()[3] + 2 * padW - wTensor.getDimensions()[3]) / strideW + 1;
        std::optional<ImplDataType> featureOutputDType = std::nullopt;

        if (outputs.contains("feature_output")) {
            const Tensor& featureOutputTensor = outputs.at("feature_output");
            if (featureOutputTensor.getDimensions().size() != 4) {
                throw std::runtime_error("Convolution2d expects feature_output to be 4D NCHW.");
            }
            if (featureOutputTensor.getDimensions()[0] != featureInputTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[2] != expectedOutputRows ||
                featureOutputTensor.getDimensions()[3] != expectedOutputCols) {
                throw std::runtime_error("Convolution2d feature_output shape does not match the implied convolution output shape.");
            }
            THOR_THROW_IF_FALSE(featureOutputTensor.getPlacement() == placement);
            featureOutputDType = featureOutputTensor.getDescriptor().getDataType();
        }

        const ImplDataType weightsDType = wTensor.getDescriptor().getDataType();

        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);

        Expression fout = Expression::conv2d(fin, w, strideH, strideW, padH, padW, ImplDataType::FP32, featureOutputDType);

        if (hasBias) {
            const Tensor& bTensor = inputs.at("biases");
            if (bTensor.getDimensions().size() != 1) {
                throw std::runtime_error("Convolution2d expects biases to be 1D [K].");
            }
            if (bTensor.getDimensions()[0] != wTensor.getDimensions()[0]) {
                throw std::runtime_error("Convolution2d bias size must match number of output channels.");
            }

            const ImplDataType biasDType = bTensor.getDescriptor().getDataType();
            auto b = Expression::input("biases", biasDType, biasDType).unsqueeze({0, 2, 3});
            fout = fout + b;
        }

        if (activation != nullptr) {
            fout = activation->toExpression(fout);
        }
        if (epilogue.has_value()) {
            fout = Convolution2d::applyEpilogue(fout, epilogue.value());
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

std::shared_ptr<ThorImplementation::Layer> Convolution2d::stamp(ThorImplementation::TensorPlacement placement,
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

    std::shared_ptr<ThorImplementation::CustomLayer> physicalConvolution2d = std::make_shared<ThorImplementation::CustomLayer>(
        buildConvolution2dExpression(
            hasBias, verticalStride, horizontalStride, verticalPadding, horizontalPadding, placement, activation, epilogue),
        placement,
        physicalParameters,
        inferenceOnly,
        getId());
    physicalConvolution2d->setLayerName(getLayerType());

    return physicalConvolution2d;
}

void Convolution2d::buildSupportLayersAndAddToNetwork(Network* network) {
    Convolution2d::Builder convolution2dBuilder;
    convolution2dBuilder.network(*network)
        .numOutputChannels(numOutputChannels)
        .filterHeight(filterHeight)
        .filterWidth(filterWidth)
        .verticalStride(verticalStride)
        .horizontalStride(horizontalStride)
        .verticalPadding(verticalPadding)
        .horizontalPadding(horizontalPadding)
        .hasBias(hasBias)
        .weightsInitializer(weightsInitializer)
        .biasInitializer(biasInitializer)
        .weightsOptimizer(weightsOptimizer)
        .biasesOptimizer(biasesOptimizer);
    if (activation != nullptr) {
        convolution2dBuilder.activation(std::dynamic_pointer_cast<Activation>(activation->clone()));
    } else {
        convolution2dBuilder.noActivation();
    }
    if (epilogue.has_value()) {
        convolution2dBuilder.epilogue(epilogue.value());
    }

    vector<Tensor> currentFeatureInputs;

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    if (useBatchNormalization) {
        BatchNormalization::Builder batchNormBuilder;
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            batchNormBuilder.featureInput(currentFeatureInputs[i]);
        }
        if (batchNormExponentialRunningAverageFactor.has_value())
            batchNormBuilder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor.value());
        if (batchNormEpsilon.has_value())
            batchNormBuilder.epsilon(batchNormEpsilon.value());
        BatchNormalization batchNormalization = batchNormBuilder.network(*network).build();
        currentFeatureInputs = batchNormalization.getFeatureOutputs();
    }

    if (dropProportion > 0.0f) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            DropOut dropOut =
                DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInputs[i]).build();
            currentFeatureInputs[i] = dropOut.getFeatureOutput().value();
        }
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        convolution2dBuilder.featureInput(currentFeatureInputs[i]);
    Convolution2d convolution2d = convolution2dBuilder.build();
    this->id = convolution2d.getId();

    standaloneLayerFeatureInputs = convolution2d.getFeatureInputs();
    standaloneLayerFeatureOutputs = convolution2d.getFeatureOutputs();

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs[i] = standaloneLayerFeatureOutputs[i];

    // Replace the outputs on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual outputs of the compound layer,
    // these are not necessarily the outputs of the stand-alone convolution layer.
    // Network uses single layers, user uses compound layer.
    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    featureOutputs = currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}

json Convolution2d::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "convolution_2d";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["data_layout"] = "NCHW";
    j["filter_width"] = filterWidth;
    j["filter_height"] = filterHeight;
    j["horizontal_stride"] = horizontalStride;
    j["vertical_stride"] = verticalStride;
    j["horizontal_padding"] = horizontalPadding;
    j["vertical_padding"] = verticalPadding;
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

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < standaloneLayerFeatureInputs.size(); ++i) {
        inputs.push_back(standaloneLayerFeatureInputs[i].architectureJson());
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < standaloneLayerFeatureOutputs.size(); ++i) {
        outputs.push_back(standaloneLayerFeatureOutputs[i].architectureJson());
    }
    j["outputs"] = outputs;

    j["parameters"] = getParametersArchitectureJson()["parameters"];

    return j;
}

json Convolution2d::serialize(thor_file::TarWriter& archiveWriter,
                              Stream stream,
                              bool saveOptimizerState,
                              ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void Convolution2d::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in Convolution2d::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "convolution_2d")
        throw runtime_error("Layer type mismatch in Convolution2d::deserialize: " + j.at("layer_type").get<std::string>());
    if (j.at("data_layout").get<string>() != "NCHW")
        throw runtime_error("Convolution2d only supports serialized NCHW data_layout, got " + j.at("data_layout").get<string>());

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition);
    }

    Convolution2d convolution2d(epilogue);
    convolution2d.filterWidth = j.at("filter_width").get<uint32_t>();
    convolution2d.filterHeight = j.at("filter_height").get<uint32_t>();
    convolution2d.horizontalStride = j.at("horizontal_stride").get<uint32_t>();
    convolution2d.verticalStride = j.at("vertical_stride").get<uint32_t>();
    convolution2d.horizontalPadding = j.at("horizontal_padding").get<uint32_t>();
    convolution2d.verticalPadding = j.at("vertical_padding").get<uint32_t>();
    convolution2d.numOutputChannels = j.at("num_output_channels").get<uint32_t>();
    convolution2d.hasBias = j.at("has_bias").get<bool>();

    if (j.contains("activation") && !j.at("activation").is_null()) {
        convolution2d.activation = Activation::deserializeTemplate(j.at("activation"));
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        convolution2d.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
        convolution2d.standaloneLayerFeatureInputs.push_back(convolution2d.featureInputs.back());
    }
    for (const json& outputJson : j.at("outputs")) {
        Tensor output = Tensor::deserialize(outputJson, archiveReader.get());
        convolution2d.featureOutputs.push_back(output);
        convolution2d.standaloneLayerFeatureOutputs.push_back(output);
    }
    if (convolution2d.featureInputs.size() != convolution2d.featureOutputs.size()) {
        throw runtime_error("Convolution2d deserialize expected equal numbers of inputs and outputs.");
    }
    for (uint32_t i = 0; i < convolution2d.featureInputs.size(); ++i) {
        convolution2d.outputTensorFromInputTensor[convolution2d.featureInputs[i]] = convolution2d.featureOutputs[i];
        convolution2d.inputTensorFromOutputTensor[convolution2d.featureOutputs[i]] = convolution2d.featureInputs[i];
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("Convolution2d parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            convolution2d.addParameter(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }

    if (!convolution2d.hasParameter("weights")) {
        throw runtime_error("Convolution2d deserialize did not find required weights parameter.");
    }
    if (convolution2d.hasBias && !convolution2d.hasParameter("biases")) {
        throw runtime_error("Convolution2d deserialize did not find required biases parameter.");
    }

    convolution2d.initialized = true;
    convolution2d.addToNetwork(network);
}

vector<Event> Convolution2d::initialize(shared_ptr<ThorImplementation::TrainableLayer> physicalLayer,
                                        bool isFirstStamp,
                                        shared_ptr<ThorImplementation::TrainableLayer> sisterPhysicalLayer,
                                        std::optional<Event> sisterPhysicalLayerLoadedEvent) {
    vector<Event> initDoneEvents =
        TrainableLayer::initialize(physicalLayer, isFirstStamp, sisterPhysicalLayer, sisterPhysicalLayerLoadedEvent);

    // // Weights are set right now, based on 1 of 3 methods:
    // // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
    // //      * So this is once per GPU since multiple stamps on the same GPU share the weights
    // // 2. Copy from a file - when loading a saved network
    // // 3. Run an initializer to set the weights - on an untrained network
    // if (!isFirstStamp) {
    //     // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
    //     THOR_THROW_IF_FALSE(sisterPhysicalLayer != nullptr);
    //     ThorImplementation::Tensor weights = physicalLayer->getParameter("weights")->getStorage().value();
    //     Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
    //     if (sisterPhysicalLayerLoadedEvent.has_value())
    //         stream.waitEvent(sisterPhysicalLayerLoadedEvent);
    //     weights.copyFromAsync(sisterPhysicalLayer->getParameter("weights")->getStorage(), stream);
    //     if (hasBias) {
    //         ThorImplementation::Tensor biases = physicalLayer->getParameter("biases")->getStorage().value();
    //         std::optional<ThorImplementation::Tensor> sisterLayerBiases = sisterPhysicalLayer->getParameter("biases")->getStorage();
    //         THOR_THROW_IF_FALSE(sisterLayerBiases.has_value());
    //         biases.copyFromAsync(sisterLayerBiases.value(), stream);
    //     }
    //
    //     initDoneEvents.push_back(stream.putEvent(false, true));
    // } else if (weightsFile.has_value()) {
    //     // 2. Copy from a file - when loading a saved network
    //     THOR_THROW_IF_FALSE(archiveReader != nullptr);
    //     THOR_THROW_IF_FALSE(physicalLayer->getParameter("weights")->getStorage().value().getPlacement().getMemDevice() ==
    //            ThorImplementation::TensorPlacement::MemDevices::GPU);
    //     Stream stream =
    //         Stream::getNextUploadStream(physicalLayer->getParameter("weights")->getStorage().value().getPlacement().getDeviceNum());
    //
    //     ThorImplementation::Tensor weights = physicalLayer->getParameter("weights")->getStorage().value();
    //     archiveReader->registerReadRequest(weightsFile.get(), weights);
    //     if (hasBias) {
    //         THOR_THROW_IF_FALSE(biasesFile.has_value());
    //         ThorImplementation::Tensor biases = physicalLayer->getParameter("biases")->getStorage().value();
    //         archiveReader->registerReadRequest(biasesFile.get(), biases);
    //     }
    //
    //     // Can't use the file later, it may not still be there
    //     archiveReader = nullptr;
    //     weightsFile = std::nullopt;
    //     biasesFile = std::nullopt;
    // } else {
    //     // FIXME: This needs to be updated to use Parameter's. It should be moved to API Thor::TrainableLayer
    //     //     // 3. Run an initializer to set the weights - on an untrained network
    //     //     THOR_THROW_IF_FALSE(weightsInitializer != nullptr);
    //     //     if (hasBias)
    //     //         THOR_THROW_IF_FALSE(biasInitializer != nullptr);
    //     //
    //     //     std::optional<Event> initDoneEvent;
    //     //
    //     //     initDoneEvent = weightsInitializer->initialize(physicalLayer->getParameter("weights")->getStorage(), physicalLayer.get());
    //     //     if (initDoneEvent.has_value())
    //     //         initDoneEvents.push_back(initDoneEvent);
    //     //
    //     //     if (physicalLayer->getParameter("biases")->getStorage().has_value()) {
    //     //         initDoneEvent = biasInitializer->initialize(physicalLayer->getParameter("biases")->getStorage().value(),
    //     //         physicalLayer.get()); if (initDoneEvent.has_value())
    //     //             initDoneEvents.push_back(initDoneEvent);
    //     //     }
    // }
    //
    // // if (physicalLayer->hasOptimizer()) {
    // //     // Initialize the optimizer - it will follow the same process as above.
    // //     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalLayer->getOptimizer();
    // //     shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer =
    // //         sisterPhysicalLayer ? sisterPhysicalLayer->getOptimizer() : nullptr;
    // //     THOR_THROW_IF_FALSE(optimizer != nullptr);
    // //
    // //     vector<Event> optimizerInitDoneEvents =
    // //         optimizer->initialize(physicalOptimizer, isFirstStamp, physicalSisterOptimizer, sisterPhysicalLayerLoadedEvent);
    // //     for (uint32_t i = 0; i < optimizerInitDoneEvents.size(); ++i)
    // //         initDoneEvents.push_back(optimizerInitDoneEvents[i]);
    // // }
    // //
    // // if (hasOptimizer()) {
    // //     // Initialize the optimizer - it will follow the same process as above.
    // //     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalLayer->getOptimizer();
    // //     shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer =
    // //         sisterPhysicalLayer ? sisterPhysicalLayer->getOptimizer() : nullptr;
    // //
    // //     vector<Event> optimizerInitDoneEvents =
    // //         optimizer->initialize(physicalOptimizer, isFirstStamp, physicalSisterOptimizer, sisterPhysicalLayerLoadedEvent);
    // //     for (uint32_t i = 0; i < optimizerInitDoneEvents.size(); ++i)
    // //         initDoneEvents.push_back(optimizerInitDoneEvents[i]);
    // // }

    return initDoneEvents;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("convolution_2d", &Thor::Convolution2d::deserialize);
    return true;
}();
}  // namespace
