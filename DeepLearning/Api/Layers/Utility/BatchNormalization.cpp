#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

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

    filesystem::path weightsFile = (layerName + "_weights.gds");
    j["weights_tensor"] = weightsFile.string();
    batchNorm->dumpWeightsToFile((dir / weightsFile).string(), stream);

    filesystem::path biasesFile = (layerName + "_biases.gds");
    j["biases_tensor"] = biasesFile.string();
    batchNorm->dumpBiasesToFile((dir / biasesFile).string(), stream);

    filesystem::path resultRunningMeanToFile = (layerName + "_means.gds");
    j["means_tensor"] = resultRunningMeanToFile.string();
    batchNorm->dumpResultRunningMeanToFile(dir / resultRunningMeanToFile.string(), stream);

    filesystem::path resultRunningVarianceToFile = (layerName + "_variances.gds");
    j["variances_tensor"] = resultRunningVarianceToFile.string();
    batchNorm->dumpResultRunningVarianceToFile(dir / resultRunningVarianceToFile.string(), stream);

    if (hasOptimizer()) {
        j["optimizer"] = optimizer->serialize(storageDir, stream, this, batchNorm);
    }

    return j;
}

void BatchNormalization::deserialize(const std::string &modelName, const string &storageDir, const json &j, Network *network) {
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
    batchNormalization.storageDir = storageDir;
    batchNormalization.weightsFile = weightsFile;
    batchNormalization.biasesFile = biasesFile;
    batchNormalization.runningMeansFile = meansFile;
    batchNormalization.runningVariancesFile = variancesFile;

    batchNormalization.initialized = true;

    if (j.contains("optimizer")) {
        batchNormalization.optimizer = Optimizer::deserialize(modelName, storageDir, j.at("optimizer"));
    }

    batchNormalization.addToNetwork(network);
}

vector<Event> BatchNormalization::initialize(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalLayer,
                                             bool isFirstStamp,
                                             shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterPhysicalLayer,
                                             Optional<Event> sisterPhysicalLayerLoadedEvent) {
    vector<Event> initDoneEvents =
        TrainableWeightsBiasesLayer::initialize(physicalLayer, isFirstStamp, sisterPhysicalLayer, sisterPhysicalLayerLoadedEvent);
    shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNorm =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(physicalLayer);
    shared_ptr<ThorImplementation::BatchNormalization> sisterPhysicalBatchNorm =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(sisterPhysicalLayer);

    // Weights are set right now, based on 1 of 3 methods:
    // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
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

        assert(physicalLayer->getBiases().isPresent());
        ThorImplementation::Tensor biases = physicalLayer->getBiases();
        Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterPhysicalLayer->getBiases();
        assert(sisterLayerBiases.isPresent());
        biases.copyFromAsync(sisterLayerBiases.get(), stream);

        ThorImplementation::Tensor resultRunningVariance = physicalBatchNorm->getResultRunningVariance();
        Optional<ThorImplementation::Tensor> sisterLayerResultRunningVariance = sisterPhysicalBatchNorm->getResultRunningVariance();
        resultRunningVariance.copyFromAsync(sisterLayerResultRunningVariance, stream);

        ThorImplementation::Tensor resultRunningMean = physicalBatchNorm->getResultRunningMean();
        Optional<ThorImplementation::Tensor> sisterLayerResultRunningMean = sisterPhysicalBatchNorm->getResultRunningMean();
        resultRunningMean.copyFromAsync(sisterLayerResultRunningMean, stream);

        physicalBatchNorm->setCurrentExponentialRunningAverageFactor(exponentialRunningAverageFactor);

        initDoneEvents.emplace_back(false, true);
    } else if (weightsFile.isPresent()) {
        // 2. Copy from a file - when loading a saved network
        assert(physicalLayer->getWeights().getPlacement().getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        Stream stream = Stream::getNextUploadStream(physicalLayer->getWeights().getPlacement().getDeviceNum());

        physicalLayer->loadWeightsFromFile(storageDir.get() + "/" + weightsFile.get(), stream);
        assert(biasesFile.isPresent());
        physicalLayer->loadBiasesFromFile(storageDir.get() + "/" + biasesFile.get(), stream);
        assert(runningVariancesFile.isPresent());
        physicalBatchNorm->loadResultRunningVarianceFromFile(storageDir.get() + "/" + runningVariancesFile.get(), stream);
        assert(runningMeansFile.isPresent());
        physicalBatchNorm->loadResultRunningMeanFromFile(storageDir.get() + "/" + runningMeansFile.get(), stream);

        // Can't use the file later, it may not still be there
        storageDir = Optional<string>::empty();
        weightsFile = Optional<string>::empty();
        biasesFile = Optional<string>::empty();
        runningVariancesFile = Optional<string>::empty();
        runningMeansFile = Optional<string>::empty();

        physicalBatchNorm->setCurrentExponentialRunningAverageFactor(exponentialRunningAverageFactor);

        initDoneEvents.emplace_back(false, true);
    } else {
        // 3. Run an initializer to set the weights - on an untrained network
        Optional<Event> initDoneEvent;

        UniformRandom::Builder onesInitializerBuilder = UniformRandom::Builder().minValue(1.0).maxValue(1.0);

        shared_ptr<Initializer::Builder> weightsInitializerBuilder = onesInitializerBuilder.clone();
        // FIXME: builder chaining
        weightsInitializerBuilder->tensorToInitialize(physicalLayer->getWeights());
        weightsInitializerBuilder->layerThatOwnsTensor(physicalLayer.get());
        shared_ptr<Initializer> weightsInitializer = weightsInitializerBuilder->build();
        weightsInitializer->initialize();
        initDoneEvent = weightsInitializer->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        shared_ptr<Initializer::Builder> resultRunningVarianceBuilder = onesInitializerBuilder.clone();
        resultRunningVarianceBuilder->tensorToInitialize(physicalBatchNorm->getResultRunningVariance());
        resultRunningVarianceBuilder->layerThatOwnsTensor(physicalLayer.get());
        shared_ptr<Initializer> resultRunningVarianceInitializer = resultRunningVarianceBuilder->build();
        resultRunningVarianceInitializer->initialize();
        initDoneEvent = resultRunningVarianceInitializer->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        UniformRandom::Builder zerosInitializerBuilder = UniformRandom::Builder().minValue(0.0).maxValue(0.0);

        assert(physicalLayer->getBiases().isPresent());
        shared_ptr<Initializer::Builder> biasInitializerBuilder = zerosInitializerBuilder.clone();
        biasInitializerBuilder->tensorToInitialize(physicalLayer->getBiases().get());
        biasInitializerBuilder->layerThatOwnsTensor(physicalLayer.get());
        shared_ptr<Initializer> biasInitializer = biasInitializerBuilder->build();
        biasInitializer->initialize();
        initDoneEvent = biasInitializer->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        shared_ptr<Initializer::Builder> resultRunningMeanBuilder = zerosInitializerBuilder.clone();
        resultRunningMeanBuilder->tensorToInitialize(physicalBatchNorm->getResultRunningMean());
        resultRunningMeanBuilder->layerThatOwnsTensor(physicalLayer.get());
        shared_ptr<Initializer> resultRunningMeanInitializer = resultRunningMeanBuilder->build();
        resultRunningMeanInitializer->initialize();
        initDoneEvent = resultRunningMeanInitializer->getInitDoneEvent();
        if (initDoneEvent.isPresent())
            initDoneEvents.push_back(initDoneEvent);

        // Start with the actual average until there are enough elements observed so that the running average
        // is a larger divisor than the actual.
        physicalBatchNorm->setCurrentExponentialRunningAverageFactor(1.0);

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
    }

    return initDoneEvents;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableWeightsBiasesLayer::register_layer("batch_normalization", &Thor::BatchNormalization::deserialize);
    return true;
}();
}  // namespace
