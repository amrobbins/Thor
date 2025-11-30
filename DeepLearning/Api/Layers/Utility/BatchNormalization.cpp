#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void BatchNormalization::buildSupportLayersAndAddToNetwork() {
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

    BatchNormalization::Builder batchNormBuilder;
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        batchNormBuilder.featureInput(currentFeatureInputs[i]);
    batchNormBuilder.exponentialRunningAverageFactor(exponentialRunningAverageFactor);
    batchNormBuilder.epsilon(epsilon);
    BatchNormalization standaloneBatchNormalization = batchNormBuilder.network(*network).build();
    this->id = standaloneBatchNormalization.getId();
    currentFeatureInputs = standaloneBatchNormalization.getFeatureOutputs();

    vector<uint64_t> dimensions = currentFeatureInputs[0].getDimensions();

    // Replace the outputs on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual outputs of the compound layer,
    // these are not necessarily the outputs of the stand-alone batch normalization connected layer.
    // Network uses single layers, user uses compound layer.
    outputTensorFromInputTensor.clear();
    inputTensorFromOutputTensor.clear();
    featureOutputs = currentFeatureInputs;
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        outputTensorFromInputTensor[featureInputs[i]] = featureOutputs[i];
        inputTensorFromOutputTensor[featureOutputs[i]] = featureInputs[i];
    }
}

json BatchNormalization::serialize(const string &storageDir, Stream stream) const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "batch_normalization";
    j["exponential_running_average_factor"] = exponentialRunningAverageFactor;
    j["epsilon"] = epsilon;

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
    shared_ptr<ThorImplementation::BatchNormalization> batchNorm =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(physicalLayer);
    assert(batchNorm != nullptr);

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
    batchNorm->dumpWeightsToFile(weightsFile.string(), stream);
    assert(batchNorm->getBiases().isPresent());

    filesystem::path biasesFile = dir / (layerName + "_biases.gds");
    j["biases_tensor"] = biasesFile.string();
    batchNorm->dumpBiasesToFile(biasesFile.string(), stream);

    filesystem::path resultRunningMeanToFile = dir / (layerName + "_means.gds");
    j["means_tensor"] = resultRunningMeanToFile.string();
    batchNorm->dumpResultRunningMeanToFile(resultRunningMeanToFile.string(), stream);

    filesystem::path resultRunningVarianceToFile = dir / (layerName + "_variances.gds");
    j["variances_tensor"] = resultRunningVarianceToFile.string();
    batchNorm->dumpResultRunningVarianceToFile(resultRunningVarianceToFile.string(), stream);

    return j;
}

void BatchNormalization::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BatchNormalization::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "batch_normalization")
        throw runtime_error("Layer type mismatch in BatchNormalization::deserialize: " + j.at("layer_type").get<std::string>());

    float exponentialRunningAverageFactor = j.at("exponential_running_average_factor").get<float>();
    float epsilon = j.at("epsilon").get<float>();

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
    string biasesFile = j.at("biases_tensor").get<string>();
    string meansFile = j.at("means_tensor").get<string>();
    string variancesFile = j.at("variances_tensor").get<string>();

    BatchNormalization batchNormalization = BatchNormalization();
    batchNormalization.exponentialRunningAverageFactor = exponentialRunningAverageFactor;
    batchNormalization.epsilon = epsilon;
    batchNormalization.featureInputs = featureInputs;
    for (uint32_t i = 0; i < batchNormalization.featureInputs.size(); ++i) {
        batchNormalization.featureOutputs.push_back(featureOutputs[i]);
        batchNormalization.outputTensorFromInputTensor[batchNormalization.featureInputs[i]] = batchNormalization.featureOutputs.back();
        batchNormalization.inputTensorFromOutputTensor[batchNormalization.featureOutputs.back()] = batchNormalization.featureInputs[i];
    }
    batchNormalization.weightsFile = weightsFile;
    batchNormalization.biasesFile = biasesFile;
    batchNormalization.runningMeansFile = meansFile;
    batchNormalization.runningVariancesFile = variancesFile;

    batchNormalization.initialized = true;

    batchNormalization.addToNetwork(network);
}

vector<Event> BatchNormalization::initialize(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
                                             bool isFirstStamp,
                                             shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
                                             Optional<Event> sisterLayerLoadedEvent,
                                             vector<shared_ptr<Initializer>> &initializers) {
    shared_ptr<ThorImplementation::BatchNormalization> batchNorm = dynamic_pointer_cast<ThorImplementation::BatchNormalization>(layer);
    shared_ptr<ThorImplementation::BatchNormalization> sisterBatchNorm =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(sisterLayer);

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

        assert(layer->getBiases().isPresent());
        ThorImplementation::Tensor biases = layer->getBiases();
        Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterLayer->getBiases();
        assert(sisterLayerBiases.isPresent());
        biases.copyFromAsync(sisterLayerBiases.get(), stream);

        ThorImplementation::Tensor resultRunningVariance = batchNorm->getResultRunningVariance();
        Optional<ThorImplementation::Tensor> sisterLayerResultRunningVariance = sisterBatchNorm->getResultRunningVariance();
        resultRunningVariance.copyFromAsync(sisterLayerResultRunningVariance, stream);

        ThorImplementation::Tensor resultRunningMean = batchNorm->getResultRunningMean();
        Optional<ThorImplementation::Tensor> sisterLayerResultRunningMean = sisterBatchNorm->getResultRunningMean();
        resultRunningMean.copyFromAsync(sisterLayerResultRunningMean, stream);

        batchNorm->setCurrentExponentialRunningAverageFactor(exponentialRunningAverageFactor);

        return {stream.putEvent(false, true)};
    } else if (weightsFile.isPresent()) {
        // 2. Copy from a file - when loading a saved network
        assert(layer->getWeights().getPlacement() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        Stream stream = Stream::getNextUploadStream(layer->getWeights().getPlacement().getDeviceNum());

        layer->loadWeightsFromFile(weightsFile.get(), stream);
        assert(biasesFile.isPresent());
        layer->loadBiasesFromFile(biasesFile.get(), stream);
        assert(runningVariancesFile.isPresent());
        batchNorm->loadResultRunningVarianceFromFile(runningVariancesFile, stream);
        assert(runningMeansFile.isPresent());
        batchNorm->loadResultRunningMeanFromFile(runningVariancesFile, stream);

        // Can't use the file later, it may not still be there
        weightsFile = Optional<string>::empty();
        biasesFile = Optional<string>::empty();
        runningVariancesFile = Optional<string>::empty();
        runningMeansFile = Optional<string>::empty();

        batchNorm->setCurrentExponentialRunningAverageFactor(exponentialRunningAverageFactor);

        return {stream.putEvent(false, true)};
    } else {
        // 3. Run an initializer to set the weights - on an untrained network
        Optional<Event> initDoneEvent;
        vector<Event> initDoneEvents;

        UniformRandom::Builder onesInitializerBuilder = UniformRandom::Builder().minValue(1.0).maxValue(1.0);

        shared_ptr<Initializer::Builder> weightsInitializerBuilder = onesInitializerBuilder.clone();
        weightsInitializerBuilder->tensorToInitialize(layer->getWeights());
        weightsInitializerBuilder->layerThatOwnsTensor(layer.get());
        initializers.push_back(weightsInitializerBuilder->build());
        initDoneEvent = initializers.back()->getInitDoneEvent();

        shared_ptr<Initializer::Builder> resultRunningVarianceBuilder = onesInitializerBuilder.clone();
        resultRunningVarianceBuilder->tensorToInitialize(batchNorm->getResultRunningVariance());
        resultRunningVarianceBuilder->layerThatOwnsTensor(layer.get());
        initializers.push_back(resultRunningVarianceBuilder->build());
        initDoneEvent = initializers.back()->getInitDoneEvent();

        UniformRandom::Builder zerosInitializerBuilder = UniformRandom::Builder().minValue(0.0).maxValue(0.0);

        assert(layer->getBiases().isPresent());
        shared_ptr<Initializer::Builder> biasInitializerBuilder = zerosInitializerBuilder.clone();
        biasInitializerBuilder->tensorToInitialize(layer->getBiases().get());
        biasInitializerBuilder->layerThatOwnsTensor(layer.get());
        initializers.push_back(biasInitializerBuilder->build());
        initDoneEvent = initializers.back()->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        shared_ptr<Initializer::Builder> resultRunningMeanBuilder = zerosInitializerBuilder.clone();
        resultRunningMeanBuilder->tensorToInitialize(batchNorm->getResultRunningMean());
        resultRunningMeanBuilder->layerThatOwnsTensor(layer.get());
        initializers.push_back(resultRunningMeanBuilder->build());
        initDoneEvent = initializers.back()->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        // Start with the actual average until there are enough elements observed so that the running average
        // is a larger divisor than the actual.
        batchNorm->setCurrentExponentialRunningAverageFactor(1.0);

        return initDoneEvents;
    }
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableWeightsBiasesLayer::register_layer("batch_normalization", &Thor::BatchNormalization::deserialize);
    return true;
}();
}
