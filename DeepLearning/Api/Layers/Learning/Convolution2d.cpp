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
    this->id = convolution2d.getId();
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

json Convolution2d::serialize(const string &storageDir, Stream stream) const {
    json j;
    j["version"] = "1.0.0";
    j["layer_type"] = "convolution_2d";
    j["data_layout"] = "NCHW";
    j["filter_width"] = filterWidth;
    j["filter_height"] = filterHeight;
    j["horizontal_stride"] = horizontalStride;
    j["vertical_stride"] = verticalStride;
    j["horizontal_padding"] = horizontalPadding;
    j["vertical_padding"] = verticalPadding;
    j["num_output_channels"] = numOutputChannels;
    j["has_bias"] = hasBias;

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back(featureInputs[i].serialize());
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back(featureOutputs[i].serialize());
    }
    j["outputs"] = outputs;

    // Dump the weights to a file and record its name
    vector<ThorImplementation::StampedNetwork> stampedNetworks = network->getStampedNetworks();
    assert(!stampedNetworks.empty());
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetworks[0].getPhysicalLayerFromApiLayer(getId());
    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twbLayer =
        dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(physicalLayer);
    assert(twbLayer != nullptr);

    filesystem::path dir(storageDir);
    if (!filesystem::exists(dir)) {
        throw runtime_error("Storage directory does not exist: " + dir.string());
    }
    if (!filesystem::is_directory(dir)) {
        throw runtime_error("Storage path is not a directory: " + dir.string());
    }

    string layerName = string("layer") + to_string(getId());
    filesystem::path weightsFile = dir / (layerName + "_weights.gds");
    j["weights_tensor"] = weightsFile.string();
    twbLayer->dumpWeightsToFile(weightsFile.string(), stream);
    if (hasBias) {
        filesystem::path biasesFile = dir / (layerName + "_biases.gds");
        j["biases_tensor"] = biasesFile.string();
        twbLayer->dumpBiasesToFile(biasesFile.string(), stream);
    }

    return j;
}

void Convolution2d::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in Convolution2d::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "convolution_2d")
        throw runtime_error("Layer type mismatch in Convolution2d::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.at("data_layout").get<string>() != "NCHW")
        throw runtime_error("Data layout must be NCHW, but it says it is %s\n." + j.at("data_layout").get<string>());
    uint32_t filterWidth = j.at("filter_width").get<uint32_t>();
    uint32_t filterHeight = j.at("filter_height").get<uint32_t>();
    uint32_t horizontalStride = j.at("horizontal_stride").get<uint32_t>();
    uint32_t verticalStride = j.at("vertical_stride").get<uint32_t>();
    uint32_t horizontalPadding = j.at("horizontal_padding").get<uint32_t>();
    uint32_t verticalPadding = j.at("vertical_padding").get<uint32_t>();
    uint32_t numOutputChannels = j.at("num_output_channels").get<uint32_t>();
    bool hasBias = j.at("has_bias").get<bool>();

    vector<Tensor> featureInputs;
    for (const json &input : j["inputs"]) {
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor tensor = network->getApiTensorByOriginalId(originalTensorId);
        featureInputs.push_back(tensor);
    }

    vector<Tensor> featureOutputs;
    for (const json &output : j["outputs"]) {
        featureOutputs.push_back(Tensor::deserialize(output));
    }

    string weightsFile = j.at("weights_tensor").get<string>();
    string biasesFile;
    if (hasBias)
        biasesFile = j.at("biases_tensor").get<string>();

    Convolution2d convolution2d = Convolution2d();
    convolution2d.filterWidth = filterWidth;
    convolution2d.filterHeight = filterHeight;
    convolution2d.horizontalStride = horizontalStride;
    convolution2d.verticalStride = verticalStride;
    convolution2d.horizontalPadding = horizontalPadding;
    convolution2d.verticalPadding = verticalPadding;
    convolution2d.numOutputChannels = numOutputChannels;
    convolution2d.hasBias = hasBias;
    convolution2d.featureInputs = featureInputs;
    for (uint32_t i = 0; i < convolution2d.featureInputs.size(); ++i) {
        convolution2d.featureOutputs.push_back(featureOutputs[i]);
        convolution2d.outputTensorFromInputTensor[convolution2d.featureInputs[i]] = convolution2d.featureOutputs.back();
        convolution2d.inputTensorFromOutputTensor[convolution2d.featureOutputs.back()] = convolution2d.featureInputs[i];
    }
    convolution2d.weightsFile = weightsFile;
    if (hasBias)
        convolution2d.biasesFile = biasesFile;

    convolution2d.initialized = true;

    convolution2d.addToNetwork(network);
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
        // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
        assert(sisterLayer != nullptr);
        ThorImplementation::Tensor weights = layer->getWeights();
        Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
        if (sisterLayerLoadedEvent.isPresent())
            stream.waitEvent(sisterLayerLoadedEvent);
        weights.copyFromAsync(sisterLayer->getWeights(), stream);
        if (hasBias) {
            ThorImplementation::Tensor biases = layer->getBiases();
            Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterLayer->getBiases();
            assert(sisterLayerBiases.isPresent());
            biases.copyFromAsync(sisterLayerBiases.get(), stream);
        }
        return {stream.putEvent(false, true)};
    } else if (weightsFile.isPresent()) {
        // 2. Copy from a file - when loading a saved network
        assert(weightsInitializerBuilder.get() == nullptr);
        assert(biasInitializerBuilder.get() == nullptr);
        assert(layer->getWeights().getPlacement() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        Stream stream = Stream::getNextUploadStream(layer->getWeights().getPlacement().getDeviceNum());
        layer->loadWeightsFromFile(weightsFile.get(), stream);
        if (hasBias) {
            assert(biasesFile.isPresent());
            layer->loadBiasesFromFile(biasesFile.get(), stream);
        }

        // Can't use the file later, it may not still be there
        weightsFile = Optional<string>::empty();
        biasesFile = Optional<string>::empty();

        return {stream.putEvent(false, true)};
    } else {
        // 3. Run an initializer to set the weights - on an untrained network
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
