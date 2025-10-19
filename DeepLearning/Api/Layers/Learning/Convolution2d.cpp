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

vector<Event> Convolution2d::initialize(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
                                         bool isFirstStamp,
                                         shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
                                         Optional<Event> sisterLayerLoadedEvent,
                                         vector<shared_ptr<Initializer>> &initializers) {

    // Weights are set right now, based on 1 of 3 methods:
    // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
    // 2. Copy from a file - when loading a saved network
    // 3. Run an initializer to set the weights - on an untrained network
    if (!isFirstStamp) {
        assert (sisterLayer != nullptr);
        ThorImplementation::Tensor weights = layer->getWeights();
        Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
        if (sisterLayerLoadedEvent.isPresent())
            stream.waitEvent(sisterLayerLoadedEvent);
        weights.copyFromAsync(sisterLayer->getWeights(), stream);
        return {stream.putEvent(false, true)};
    } else if (weightsFile.isPresent()) {
        assert (weightsInitializerBuilder.get() == nullptr);
        assert (biasInitializerBuilder.get() == nullptr);
        assert (layer->getWeights().getPlacement() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        Stream stream = Stream::getNextUploadStream(layer->getWeights().getPlacement().getDeviceNum());
        layer->loadWeightsFromFile(weightsFile.get(), stream);
        if (hasBias) {
            assert (biasesFile.isPresent());
            layer->loadWeightsFromFile(biasesFile.get(), stream);
        }

        // Can't use the file later, it may not still be there
        weightsFile = Optional<string>::empty();
        biasesFile = Optional<string>::empty();

        return {stream.putEvent(false, true)};
    } else {
        Optional<Event> initDoneEvent;
        vector<Event> initDoneEvents;

        shared_ptr<Initializer::Builder> weightsInitializerBuilderClone = weightsInitializerBuilder->clone();
        weightsInitializerBuilderClone->tensorToInitialize(layer->getWeights());
        weightsInitializerBuilderClone->layerThatOwnsTensor(layer.get());
        initializers.push_back(weightsInitializerBuilderClone->build());
        initDoneEvent = initializers.back()->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        if (layer->getBiases().isPresent()) {
            shared_ptr<Initializer::Builder> biasInitializerBuilderClone = biasInitializerBuilder->clone();
            biasInitializerBuilderClone->tensorToInitialize(layer->getBiases().get());
            biasInitializerBuilderClone->layerThatOwnsTensor(layer.get());
            initializers.push_back(biasInitializerBuilderClone->build());
            initDoneEvent = initializers.back()->getInitDoneEvent();
            if (initDoneEvent.isPresent())
                initDoneEvents.push_back(initDoneEvent);
        }
        return initDoneEvents;
    }
}
