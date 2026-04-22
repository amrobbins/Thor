#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json BatchNormalization::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "batch_normalization";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["exponential_running_average_factor"] = exponentialRunningAverageFactor;
    j["epsilon"] = epsilon;

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back(featureInputs[i].architectureJson());
    }
    j["inputs"] = inputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back(featureOutputs[i].architectureJson());
    }
    j["outputs"] = outputs;

    if (hasOptimizer()) {
        // Not stamped so there is no physical optimizer, so then what do I do? Maybe I expect it can be null?
        j["optimizer"] = optimizer->architectureJson();
    }

    return j;
}

json BatchNormalization::serialize(thor_file::TarWriter &archiveWriter,
                                   Stream stream,
                                   bool saveOptimizerState,
                                   ThorImplementation::StampedNetwork &stampedNetwork) const {
    json j = architectureJson();
    string layerName = string("layer") + to_string(getId());

    // Dump the weights to a file and record its name
    shared_ptr<ThorImplementation::BatchNormalization> batchNorm;
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
    batchNorm = dynamic_pointer_cast<ThorImplementation::BatchNormalization>(physicalLayer);
    assert(batchNorm != nullptr);

    ThorImplementation::Tensor weights;
    ThorImplementation::Tensor biases;
    string weightsFile;
    string biasesFile;

    string resultRunningMeanFile;
    ThorImplementation::Tensor means;

    string resultRunningVarianceFile;
    ThorImplementation::Tensor variance;

    if (batchNorm != nullptr) {
        // FIXME: Simplify this with Parameterizable->serializeParameters(...)
        weightsFile = (layerName + "_weights.gds");
        j["weights_tensor"] = weightsFile;
        weights = batchNorm->getParameter("weights")->getStorage();
        archiveWriter.addArchiveFile(weightsFile, weights);

        biasesFile = (layerName + "_biases.gds");
        j["biases_tensor"] = biasesFile;
        biases = batchNorm->getParameter("biases")->getStorage().get();
        archiveWriter.addArchiveFile(biasesFile, biases);

        resultRunningMeanFile = (layerName + "_means.gds");
        j["means_tensor"] = resultRunningMeanFile;
        means = batchNorm->getParameter("running_mean")->getStorage().get();
        archiveWriter.addArchiveFile(resultRunningMeanFile, means);

        resultRunningVarianceFile = (layerName + "_variances.gds");
        j["variances_tensor"] = resultRunningVarianceFile;
        variance = batchNorm->getParameter("running_variance")->getStorage().get();
        archiveWriter.addArchiveFile(resultRunningVarianceFile, variance);

        j["num_items_observed"] = batchNorm->getNumItemsObserved();
    }

    if (hasOptimizer()) {
        j["weights_optimizer"] = optimizer->serialize(archiveWriter,
                                                      stream,
                                                      batchNorm->getParameter("weights")->getOptimizer(),
                                                      string("layer") + to_string(getId()),
                                                      saveOptimizerState);
        j["biases_optimizer"] = optimizer->serialize(archiveWriter,
                                                     stream,
                                                     batchNorm->getParameter("biases")->getOptimizer(),
                                                     string("layer") + to_string(getId()),
                                                     saveOptimizerState);
    }

    return j;
}

void BatchNormalization::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BatchNormalization::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "batch_normalization")
        throw runtime_error("Layer type mismatch in BatchNormalization::deserialize: " + j.at("layer_type").get<std::string>());

    double numItemsObserved = j.at("num_items_observed").get<uint64_t>();
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

    BatchNormalization batchNormalization = BatchNormalization();
    batchNormalization.exponentialRunningAverageFactor = exponentialRunningAverageFactor;
    batchNormalization.numItemsObserved = numItemsObserved;
    batchNormalization.epsilon = epsilon;
    batchNormalization.featureInputs = featureInputs;
    for (uint32_t i = 0; i < batchNormalization.featureInputs.size(); ++i) {
        batchNormalization.featureOutputs.push_back(featureOutputs[i]);
        batchNormalization.outputTensorFromInputTensor[batchNormalization.featureInputs[i]] = batchNormalization.featureOutputs.back();
        batchNormalization.inputTensorFromOutputTensor[batchNormalization.featureOutputs.back()] = batchNormalization.featureInputs[i];
    }
    batchNormalization.archiveReader = archiveReader;

    if (j.contains("weights_tensor")) {
        batchNormalization.weightsFile = j.at("weights_tensor").get<string>();
        batchNormalization.biasesFile = j.at("biases_tensor").get<string>();
        batchNormalization.runningMeansFile = j.at("means_tensor").get<string>();
        batchNormalization.runningVariancesFile = j.at("variances_tensor").get<string>();
    }

    if (j.contains("optimizer")) {
        batchNormalization.optimizer = Optimizer::deserialize(archiveReader, j.at("optimizer"), network);
    }

    batchNormalization.initialized = true;
    batchNormalization.addToNetwork(network);
}

vector<Event> BatchNormalization::initialize(shared_ptr<ThorImplementation::TrainableLayer> physicalLayer,
                                             bool isFirstStamp,
                                             shared_ptr<ThorImplementation::TrainableLayer> sisterPhysicalLayer,
                                             Optional<Event> sisterPhysicalLayerLoadedEvent) {
    vector<Event> initDoneEvents =
        TrainableLayer::initialize(physicalLayer, isFirstStamp, sisterPhysicalLayer, sisterPhysicalLayerLoadedEvent);
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
        ThorImplementation::Tensor weights = physicalLayer->getParameter("weights")->getStorage();
        Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
        if (sisterPhysicalLayerLoadedEvent.isPresent())
            stream.waitEvent(sisterPhysicalLayerLoadedEvent);
        weights.copyFromAsync(sisterPhysicalLayer->getParameter("weights")->getStorage(), stream);

        assert(physicalLayer->getParameter("biases")->getStorage().isPresent());
        ThorImplementation::Tensor biases = physicalLayer->getParameter("biases")->getStorage();
        Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterPhysicalLayer->getParameter("biases")->getStorage();
        assert(sisterLayerBiases.isPresent());
        biases.copyFromAsync(sisterLayerBiases.get(), stream);

        ThorImplementation::Tensor resultRunningVariance = physicalBatchNorm->getParameter("running_variance")->getStorage().get();
        ;
        Optional<ThorImplementation::Tensor> sisterLayerResultRunningVariance =
            sisterPhysicalBatchNorm->getParameter("running_variance")->getStorage().get();
        ;
        resultRunningVariance.copyFromAsync(sisterLayerResultRunningVariance, stream);

        ThorImplementation::Tensor resultRunningMean = physicalBatchNorm->getParameter("running_mean")->getStorage().get();
        ;
        Optional<ThorImplementation::Tensor> sisterLayerResultRunningMean =
            sisterPhysicalBatchNorm->getParameter("running_mean")->getStorage().get();
        ;
        resultRunningMean.copyFromAsync(sisterLayerResultRunningMean, stream);

        physicalBatchNorm->setExponentialRunningAverageFactor(exponentialRunningAverageFactor);

        initDoneEvents.emplace_back(false, true);
    } else if (weightsFile.isPresent()) {
        // 2. Copy from a file - when loading a saved network
        assert(archiveReader != nullptr);
        assert(physicalLayer->getParameter("weights")->getStorage().get().getPlacement().getMemDevice() ==
               ThorImplementation::TensorPlacement::MemDevices::GPU);
        Stream stream =
            Stream::getNextUploadStream(physicalLayer->getParameter("weights")->getStorage().get().getPlacement().getDeviceNum());

        ThorImplementation::Tensor weights = physicalLayer->getParameter("weights")->getStorage();
        archiveReader->registerReadRequest(weightsFile.get(), weights);
        assert(biasesFile.isPresent());
        ThorImplementation::Tensor biases = physicalLayer->getParameter("biases")->getStorage().get();
        archiveReader->registerReadRequest(biasesFile.get(), biases);

        assert(runningVariancesFile.isPresent());
        ThorImplementation::Tensor variances = physicalBatchNorm->getParameter("running_variance")->getStorage().get();
        archiveReader->registerReadRequest(runningVariancesFile.get(), variances);

        assert(runningMeansFile.isPresent());
        ThorImplementation::Tensor means = physicalBatchNorm->getParameter("running_mean")->getStorage().get();
        archiveReader->registerReadRequest(runningMeansFile.get(), means);

        // Can't use the file later, it may not still be there
        archiveReader = nullptr;
        weightsFile = Optional<string>::empty();
        biasesFile = Optional<string>::empty();
        runningVariancesFile = Optional<string>::empty();
        runningMeansFile = Optional<string>::empty();

        physicalBatchNorm->setExponentialRunningAverageFactor(exponentialRunningAverageFactor);
    } else {
        // FIXME: This needs to be updated to use Parameter's. It needs be moved to API Thor::TrainableLayer
        // // 3. Run an initializer to set the weights - on an untrained network
        // Optional<Event> initDoneEvent;
        //
        // UniformRandom::Builder onesInitializerBuilder = UniformRandom::Builder().minValue(1.0).maxValue(1.0);
        //
        // shared_ptr<Initializer::Builder> weightsInitializerBuilder = onesInitializerBuilder.clone();
        // shared_ptr<Initializer> weightsInitializer = weightsInitializerBuilder->build();
        // initDoneEvent = weightsInitializer->initialize(physicalBatchNorm->getParameter("weights")->getStorage(),
        // physicalBatchNorm.get()); if (initDoneEvent.isPresent())
        //     initDoneEvents.push_back(initDoneEvent);
        //
        // shared_ptr<Initializer::Builder> resultRunningVarianceBuilder = onesInitializerBuilder.clone();
        // shared_ptr<Initializer> resultRunningVarianceInitializer = resultRunningVarianceBuilder->build();
        // initDoneEvent =
        //     resultRunningVarianceInitializer->initialize(physicalBatchNorm->getParameter("running_variance")->getStorage().get();,
        //     physicalBatchNorm.get());
        // if (initDoneEvent.isPresent())
        //     initDoneEvents.push_back(initDoneEvent);
        //
        // UniformRandom::Builder zerosInitializerBuilder = UniformRandom::Builder().minValue(0.0).maxValue(0.0);
        //
        // assert(physicalBatchNorm->getParameter("biases")->getStorage().isPresent());
        // shared_ptr<Initializer::Builder> biasInitializerBuilder = zerosInitializerBuilder.clone();
        // shared_ptr<Initializer> biasInitializer = biasInitializerBuilder->build();
        // initDoneEvent = biasInitializer->initialize(physicalBatchNorm->getParameter("biases")->getStorage(), physicalBatchNorm.get());
        // if (initDoneEvent.isPresent())
        //     initDoneEvents.push_back(initDoneEvent);
        //
        // shared_ptr<Initializer::Builder> resultRunningMeanBuilder = zerosInitializerBuilder.clone();
        // shared_ptr<Initializer> resultRunningMeanInitializer = resultRunningMeanBuilder->build();
        // initDoneEvent = resultRunningMeanInitializer->initialize(physicalBatchNorm->getParameter("running_mean")->getStorage().get();,
        // physicalBatchNorm.get()); if (initDoneEvent.isPresent())
        //     initDoneEvents.push_back(initDoneEvent);
        //
        // // Start with the actual average until there are enough elements observed so that the running average
        // // is a larger divisor than the actual.
        // physicalBatchNorm->setCurrentExponentialRunningAverageFactor(1.0);
    }
    // if (hasOptimizer()) {
    //     // Initialize the optimizer - it will follow the same process as above.
    //     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalBatchNorm->getOptimizer();
    //     shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer =
    //         sisterPhysicalBatchNorm ? sisterPhysicalBatchNorm->getOptimizer() : nullptr;
    //
    //     vector<Event> optimizerInitDoneEvents =
    //         optimizer->initialize(physicalOptimizer, isFirstStamp, physicalSisterOptimizer, sisterPhysicalLayerLoadedEvent);
    //     for (uint32_t i = 0; i < optimizerInitDoneEvents.size(); ++i)
    //         initDoneEvents.push_back(optimizerInitDoneEvents[i]);
    // }

    return initDoneEvents;
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("batch_normalization", &Thor::BatchNormalization::deserialize);
    return true;
}();
}  // namespace
