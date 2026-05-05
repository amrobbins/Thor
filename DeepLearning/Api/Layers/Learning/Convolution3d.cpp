#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"

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
                                                                    std::shared_ptr<Thor::Activation> activation) {
    using ImplDataType = ThorImplementation::TensorDescriptor::DataType;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    return DynamicExpression([hasBias, strideD, strideH, strideW, padD, padH, padW, placement, activation = std::move(activation)](
                                 const DynamicExpression::TensorMap& inputs,
                                 const DynamicExpression::TensorMap& outputs,
                                 Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");
        assert(wTensor.getPlacement() == placement);

        if (featureInputTensor.getDimensions().size() != 5) {
            throw std::runtime_error("Convolution3d expects feature_input to be 5D NCDHW.");
        }
        if (wTensor.getDimensions().size() != 5) {
            throw std::runtime_error("Convolution3d expects weights to be 5D KCDHW.");
        }
        if (featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
            throw std::runtime_error("Convolution3d input channels must match weight channels.");
        }
        assert(featureInputTensor.getPlacement() == placement);

        const uint64_t expectedOutputDepth =
            (featureInputTensor.getDimensions()[2] + 2 * padD - wTensor.getDimensions()[2]) / strideD + 1;
        const uint64_t expectedOutputRows =
            (featureInputTensor.getDimensions()[3] + 2 * padH - wTensor.getDimensions()[3]) / strideH + 1;
        const uint64_t expectedOutputCols =
            (featureInputTensor.getDimensions()[4] + 2 * padW - wTensor.getDimensions()[4]) / strideW + 1;

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
            assert(featureOutputTensor.getPlacement() == placement);
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

    assert(initialized);
    assert(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    Tensor::DataType weightsDataType = Tensor::DataType::FP16;
    std::shared_ptr<ThorImplementation::CustomLayer> physicalConvolution3d = std::make_shared<ThorImplementation::CustomLayer>(
        buildConvolution3dExpression(hasBias,
                                     depthStride,
                                     verticalStride,
                                     horizontalStride,
                                     depthPadding,
                                     verticalPadding,
                                     horizontalPadding,
                                     placement,
                                     activation),
        placement,
        ThorImplementation::Convolution3d::defineParameters(numOutputChannels, hasBias, filterWidth, filterHeight, filterDepth, weightsDataType),
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
            currentFeatureInputs[i] = typeConverter.getFeatureOutput();
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

    if (weightsInitializer != nullptr) {
        j["weights_initializer"] = weightsInitializer->architectureJson();
    }
    if (biasInitializer != nullptr) {
        j["biases_initializer"] = biasInitializer->architectureJson();
    }

    if (hasOptimizer()) {
        j["weights_optimizer"] = weightsOptimizer->architectureJson();
        if (hasBias) {
            j["biases_optimizer"] = biasesOptimizer->architectureJson();
        }
    }

    return j;
}

json Convolution3d::serialize(thor_file::TarWriter& archiveWriter,
                              Stream stream,
                              bool saveOptimizerState,
                              ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    string layerName = string("layer") + to_string(getId());

    shared_ptr<ThorImplementation::TrainableLayer> twbLayer = nullptr;
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
    twbLayer = dynamic_pointer_cast<ThorImplementation::TrainableLayer>(physicalLayer);
    assert(twbLayer != nullptr);

    if (twbLayer != nullptr) {
        if (hasBias) {
            const string biasesFile = layerName + "_biases.gds";
            j["biases_tensor"] = biasesFile;
            ThorImplementation::Tensor biases = twbLayer->getParameter("biases")->getStorage().get();
            archiveWriter.addArchiveFile(biasesFile, biases);
        }

        const string weightsFile = layerName + "_weights.gds";
        j["weights_tensor"] = weightsFile;
        ThorImplementation::Tensor weights = twbLayer->getParameter("weights")->getStorage();
        archiveWriter.addArchiveFile(weightsFile, weights);
    }

    if (hasOptimizer()) {
        j["weights_optimizer"] = weightsOptimizer->serialize(archiveWriter,
                                                             stream,
                                                             twbLayer->getParameter("weights")->getOptimizer(),
                                                             string("layer") + to_string(getId()),
                                                             saveOptimizerState);
        if (hasBias) {
            j["biases_optimizer"] = biasesOptimizer->serialize(archiveWriter,
                                                               stream,
                                                               twbLayer->getParameter("biases")->getOptimizer(),
                                                               string("layer") + to_string(getId()),
                                                               saveOptimizerState);
        }
    }

    return j;
}

void Convolution3d::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)archiveReader;
    (void)j;
    (void)network;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("convolution_3d", &Thor::Convolution3d::deserialize);
    return true;
}();
}  // namespace
