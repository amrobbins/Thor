#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"

using namespace Thor;
using namespace std;

using json = nlohmann::json;

void Convolution2d::buildSupportLayersAndAddToNetwork() {
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
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
        .noActivation();

    vector<Tensor> currentFeatureInputs;

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    // Force the input tensor to this type of layer to be FP16
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

    if (useBatchNormalization) {
        BatchNormalization::Builder batchNormBuilder;
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            batchNormBuilder.featureInput(currentFeatureInputs[i]);
        }
        if (batchNormExponentialRunningAverageFactor.isPresent())
            batchNormBuilder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor);
        if (batchNormEpsilon.isPresent())
            batchNormBuilder.epsilon(batchNormEpsilon);
        BatchNormalization batchNormalization = batchNormBuilder.network(*network).build();
        currentFeatureInputs = batchNormalization.getFeatureOutputs();
    }

    if (dropProportion > 0.0f) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            DropOut dropOut =
                DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInputs[i]).build();
            currentFeatureInputs[i] = dropOut.getFeatureOutput();
        }
    }

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        convolution2dBuilder.featureInput(currentFeatureInputs[i]);
    Convolution2d convolution2d = convolution2dBuilder.build();
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs[i] = convolution2d.getFeatureOutputs()[i];

    if (activationBuilder) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            shared_ptr<Activation::Builder> activationBuilderClone = activationBuilder->clone();
            activationBuilderClone->network(*network);
            activationBuilderClone->featureInput(currentFeatureInputs[i]);
            // Since activation may be one of many classes, the base class is built and its virtual build function is used.
            shared_ptr<Layer> activation = activationBuilderClone->build();
            currentFeatureInputs[i] = activation->getFeatureOutput();
        }
    }

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

json Convolution2d::serialize(Stream stream) {
    json j;
    j["version"] = "1.0.0";
    j["data_layout"] = "NCHW";
    j["filterWidth"] = filterWidth;
    j["filterHeight"] = filterHeight;
    j["horizontalStride"] = horizontalStride;
    j["verticalStride"] = verticalStride;
    j["horizontalPadding"] = horizontalPadding;
    j["verticalPadding"] = verticalPadding;
    j["numOutputChannels"] = numOutputChannels;
    j["has_bias"] = hasBias;

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back({{"id", featureInputs[i].getId()},
                          {"dimensions", featureInputs[i].getDimensions()},
                          {"data_type", json(featureInputs[i].getDataType())}});
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back({{"id", featureOutputs[i].getId()},
                           {"dimensions", featureOutputs[i].getDimensions()},
                           {"data_type", json(featureOutputs[i].getDataType())}});
    }
    j["outputs"] = outputs;

    // Dump the weights to a file and record its name
    vector<ThorImplementation::StampedNetwork> stampedNetworks = network->getStampedNetworks();
    assert(!stampedNetworks.empty());
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetworks[0].getPhysicalLayerFromApiLayer(getId());
    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twbLayer =
        dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(physicalLayer);
    assert(twbLayer != nullptr);

    string layerName = string("layer") + to_string(getId());
    string weightsFilename = string("/tmp/") + layerName + "_weights.gds";
    j["weights_tensor"] = weightsFilename;
    twbLayer->dumpWeightsToFile(weightsFilename, stream);
    if (hasBias) {
        string biasesFilename = string("/tmp/") + layerName + "_biases.gds";
        j["biases_tensor"] = biasesFilename;
        twbLayer->dumpBiasesToFile(biasesFilename, stream);
    }

    return j;
}
