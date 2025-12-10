#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void FullyConnected::buildSupportLayersAndAddToNetwork() {
    vector<Tensor> currentFeatureInputs;

    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        currentFeatureInputs.push_back(featureInputs[i]);

    // Flatten to 2 dimensions {batchSize, numInputFeatures} if not already a 2d tensor.
    vector<uint64_t> featureInputDimensions = featureInputs.front().getDimensions();
    assert(!featureInputDimensions.empty());
    if (featureInputDimensions.size() > 1) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            Flatten flatten = Flatten::Builder().network(*network).featureInput(currentFeatureInputs[i]).numOutputDimensions(1).build();
            currentFeatureInputs[i] = flatten.getFeatureOutput();
        }
    }

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
        batchNormalization = batchNormBuilder.network(*network).build();
        currentFeatureInputs = batchNormalization.getFeatureOutputs();
    }

    if (dropProportion > 0.0f) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            dropOut = DropOut::Builder().network(*network).dropProportion(dropProportion).featureInput(currentFeatureInputs[i]).build();
            currentFeatureInputs[i] = dropOut.getFeatureOutput();
        }
    }

    vector<uint64_t> dimensions = currentFeatureInputs[0].getDimensions();

    // I do actually need a second one because the connections of this multi-layer don't match the one that
    // the network will use.
    FullyConnected::Builder fullyConnectedBuilder;
    fullyConnectedBuilder.network(*network)
        .numOutputFeatures(numOutputFeatures)
        .hasBias(hasBias)
        .weightsInitializerBuilder(*weightsInitializerBuilder)
        .biasInitializerBuilder(*biasInitializerBuilder)
        .noActivation();
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        fullyConnectedBuilder.featureInput(currentFeatureInputs[i]);
    FullyConnected standAloneFullyConnected = fullyConnectedBuilder.build();
    this->id = standAloneFullyConnected.getId();
    currentFeatureInputs = standAloneFullyConnected.getFeatureOutputs();

    if (activationBuilder) {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            shared_ptr<Activation::Builder> activationBuilderClone = activationBuilder->clone();
            activationBuilderClone->network(*network);
            activationBuilderClone->featureInput(currentFeatureInputs[i]);
            // Since activation may be one of many classes, the base class is built and its virtual build function is used.
            activation = activationBuilderClone->build();
            currentFeatureInputs[i] = activation->getFeatureOutput();
        }
    }

    // Replace the outputs on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual outputs of the compound layer,
    // these are not necessarily the outputs of the stand-alone fully connected layer.
    // Network uses single layers, user uses compound layer.
    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    featureOutputs = currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}

json FullyConnected::serialize(const string &storageDir, Stream stream) const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "fully_connected";
    j["num_output_features"] = numOutputFeatures;
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
    assert(network->getNumStamps() >= 1);
    ThorImplementation::StampedNetwork &stampedNetwork = network->getStampedNetwork(0);
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
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

    if (hasOptimizer()) {
        j["optimizer"] = optimizer->serialize(storageDir, stream, this, twbLayer);
    }

    return j;
}

void FullyConnected::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in FullyConnected::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "fully_connected")
        throw runtime_error("Layer type mismatch in FullyConnected::deserialize: " + j.at("layer_type").get<std::string>());

    uint32_t numOutputFeatures = j.at("num_output_features").get<uint32_t>();
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

    FullyConnected fullyConnected = FullyConnected();
    fullyConnected.numOutputFeatures = numOutputFeatures;
    fullyConnected.hasBias = hasBias;
    fullyConnected.featureInputs = featureInputs;
    for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
        fullyConnected.featureOutputs.push_back(featureOutputs[i]);
        fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = fullyConnected.featureOutputs.back();
        fullyConnected.inputTensorFromOutputTensor[fullyConnected.featureOutputs.back()] = fullyConnected.featureInputs[i];
    }
    fullyConnected.weightsFile = weightsFile;
    if (hasBias)
        fullyConnected.biasesFile = biasesFile;

    fullyConnected.initialized = true;

    if (j.contains("optimizer")) {
        fullyConnected.optimizer = Optimizer::deserialize(j.at("optimizer"));
    }

    fullyConnected.addToNetwork(network);
}

vector<Event> FullyConnected::initialize(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalLayer,
                                         bool isFirstStamp,
                                         shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterPhysicalLayer,
                                         Optional<Event> sisterPhysicalLayerLoadedEvent,
                                         vector<shared_ptr<Initializer>> &initializers) {
    vector<Event> initDoneEvents;

    // Weights are set right now, based on 1 of 3 methods:
    // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
    //      * So this is once per GPU since multiple stamps on the same GPU share the weights
    // 2. Copy from a file - when loading a saved network
    // 3. Run an initializer to set the weights - on an untrained network
    if (!isFirstStamp) {
        // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
        assert(sisterPhysicalLayer != nullptr);
        ThorImplementation::Tensor weights = physicalLayer->getWeights();
        Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
        if (sisterPhysicalLayerLoadedEvent.isPresent())
            stream.waitEvent(sisterPhysicalLayerLoadedEvent);
        weights.copyFromAsync(sisterPhysicalLayer->getWeights(), stream);
        if (hasBias) {
            ThorImplementation::Tensor biases = physicalLayer->getBiases();
            Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterPhysicalLayer->getBiases();
            assert(sisterLayerBiases.isPresent());
            biases.copyFromAsync(sisterLayerBiases.get(), stream);
        }

        initDoneEvents.push_back(stream.putEvent(false, true));
    } else if (weightsFile.isPresent()) {
        // 2. Copy from a file - when loading a saved network
        assert(weightsInitializerBuilder.get() == nullptr);
        assert(biasInitializerBuilder.get() == nullptr);
        assert(physicalLayer->getWeights().getPlacement() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        Stream stream = Stream::getNextUploadStream(physicalLayer->getWeights().getPlacement().getDeviceNum());
        physicalLayer->loadWeightsFromFile(weightsFile.get(), stream);
        if (hasBias) {
            assert(biasesFile.isPresent());
            physicalLayer->loadBiasesFromFile(biasesFile.get(), stream);
        }

        // Can't use the file later, it may not still be there
        weightsFile = Optional<string>::empty();
        biasesFile = Optional<string>::empty();

        initDoneEvents.push_back(stream.putEvent(false, true));
    } else {
        // 3. Run an initializer to set the weights - on an untrained network
        Optional<Event> initDoneEvent;

        shared_ptr<Initializer::Builder> weightsInitializerBuilderClone = weightsInitializerBuilder->clone();
        weightsInitializerBuilderClone->tensorToInitialize(physicalLayer->getWeights());
        weightsInitializerBuilderClone->layerThatOwnsTensor(physicalLayer.get());
        initializers.push_back(weightsInitializerBuilderClone->build());
        initDoneEvent = initializers.back()->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        if (physicalLayer->getBiases().isPresent()) {
            shared_ptr<Initializer::Builder> biasInitializerBuilderClone = biasInitializerBuilder->clone();
            biasInitializerBuilderClone->tensorToInitialize(physicalLayer->getBiases().get());
            biasInitializerBuilderClone->layerThatOwnsTensor(physicalLayer.get());
            initializers.push_back(biasInitializerBuilderClone->build());
            initDoneEvent = initializers.back()->getInitDoneEvent();
            if (initDoneEvent.isPresent())
                initDoneEvents.push_back(initDoneEvent);
        }
    }

    if (hasOptimizer()) {
        // Initialize the optimizer - it will follow the same process as above.
        shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalLayer->getOptimizer();
        shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer =
            sisterPhysicalLayer ? sisterPhysicalLayer->getOptimizer() : nullptr;

        vector<Event> optimizerInitDoneEvents =
            optimizer->initialize(physicalOptimizer, isFirstStamp, physicalSisterOptimizer, sisterPhysicalLayerLoadedEvent);
        for (uint32_t i = 0; i < optimizerInitDoneEvents.size(); ++i)
            initDoneEvents.push_back(optimizerInitDoneEvents[i]);
    }

    return initDoneEvents;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableWeightsBiasesLayer::register_layer("fully_connected", &Thor::FullyConnected::deserialize);
    return true;
}();
}  // namespace
