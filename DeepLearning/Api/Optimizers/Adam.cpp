#include "DeepLearning/Api/Optimizers/Adam.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Adam::~Adam() {}

shared_ptr<ThorImplementation::Optimizer> Adam::stamp(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) {
    Optional<ThorImplementation::Tensor> errorInput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(trainableLayer->getErrorInputs());
    Optional<ThorImplementation::Tensor> errorOutput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(trainableLayer->getErrorOutputs());
    return make_shared<ThorImplementation::Adam>(trainableLayer, alpha, beta1, beta2, epsilon, errorInput, errorOutput);
}

void Adam::setAlpha(float newAlpha) {
    alpha = newAlpha;
    updateParameters();
}

void Adam::setBeta1(float newBeta1) {
    beta1 = newBeta1;
    updateParameters();
}

void Adam::setBeta2(float newBeta2) {
    beta2 = newBeta2;
    updateParameters();
}

void Adam::setEpsilon(float newEpsilon) {
    epsilon = newEpsilon;
    updateParameters();
}

float Adam::getAlpha() { return alpha; }

float Adam::getBeta1() { return beta1; }

float Adam::getBeta2() { return beta2; }

float Adam::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> Adam::clone() const { return make_shared<Adam>(*this); }

void Adam::updateParameters() {
    assert(network != nullptr);
    uint32_t numStamps = network->getNumStamps();
    for (uint32_t i = 0; i < numStamps; ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network->getStampedNetwork(i);
        uint32_t numTrainableLayers = stampedNetwork.getNumTrainableLayers();
        for (uint32_t j = 0; j < numTrainableLayers; ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> &trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybePhysicalOptimizer = trainableLayer->getOptimizer();
            assert(maybePhysicalOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybePhysicalOptimizer.get();
            shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(optimizer);
            if (physicalAdam == nullptr || physicalAdam->getId() != getId())
                continue;
            physicalAdam->setAlpha(alpha);
            physicalAdam->setBeta1(beta1);
            physicalAdam->setBeta2(beta2);
            physicalAdam->setEpsilon(epsilon);
        }
    }
}

//////////////////////////////////

// json Adam::serialize(const string &storageDir, Stream stream) const {
//     // Multi-layers will only serialize the single layer, itself.
//     // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network
//
//     json j;
//     j["factory"] = Layer::Factory::Learning.value();
//     j["version"] = getLayerVersion();
//     j["layer_type"] = "fully_connected";
//     j["num_output_features"] = numOutputFeatures;
//     j["has_bias"] = hasBias;
//
//     // Input connections
//     json inputs = json::array();
//     for (uint32_t i = 0; i < featureInputs.size(); ++i) {
//         inputs.push_back(featureInputs[i].serialize());
//     }
//     j["inputs"] = inputs;
//
//     // Output connections
//     json outputs = json::array();
//     for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
//         outputs.push_back(featureOutputs[i].serialize());
//     }
//     j["outputs"] = outputs;
//
//     // Dump the weights to a file and record its name
//     vector<ThorImplementation::StampedNetwork> stampedNetworks = network->getStampedNetworks();
//     assert(!stampedNetworks.empty());
//     shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetworks[0].getPhysicalLayerFromApiLayer(getId());
//     shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twbLayer =
//         dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(physicalLayer);
//     assert(twbLayer != nullptr);
//
//     filesystem::path dir(storageDir);
//     if (!filesystem::exists(dir)) {
//         throw runtime_error("Storage directory does not exist: " + dir.string());
//     }
//     if (!filesystem::is_directory(dir)) {
//         throw runtime_error("Storage path is not a directory: " + dir.string());
//     }
//
//     string layerName = string("layer") + to_string(getId());
//     filesystem::path weightsFile = dir / (layerName + "_weights.gds");
//     j["weights_tensor"] = weightsFile.string();
//     twbLayer->dumpWeightsToFile(weightsFile.string(), stream);
//     if (hasBias) {
//         filesystem::path biasesFile = dir / (layerName + "_biases.gds");
//         j["biases_tensor"] = biasesFile.string();
//         twbLayer->dumpBiasesToFile(biasesFile.string(), stream);
//     }
//
//     return j;
// }

// void Adam::deserialize(const json &j, Network *network, uint64_t layerId) {
//     // If there is no saved optimizer here, return.
//     Maybe there is always a saved optimizer, the type anyway, weights files are optional
//     if (!j.contains("optimizer") || !j["optimizer"].is_object())
//         throw runtime_error("Malformed optimizer json during deserialization.\n" + j.dump(4).c_str());
//
//     json optimizerJ = j["optimizer"];
//
//     if (optimizerJ.at("version").get<string>() != "1.0.0")
//         throw runtime_error("Unsupported version in Adam::deserialize: " + optimizerJ["version"].get<string>());
//     if (optimizerJ.at("type").get<string>() != "adam")
//         throw runtime_error("Layer type mismatch in Adam::deserialize: " + optimizerJ.at("type").get<string>());
//
//     float t = optimizerJ.at("t").get<float>();
//     float alpha = optimizerJ.at("alpha").get<float>();
//     float beta1 = optimizerJ.at("beta1").get<float>();
//     float beta2 = optimizerJ.at("beta2").get<float>();
//     float epsilon = optimizerJ.at("epsilon").get<float>();
//
//     // So now I want to essentially pick off the file names and save them.
//     string weightsMFile = j.at("weights_m_file").get<string>();
//     string weightsVFile = j.at("weights_v_file").get<string>();
//     string biasesMFile;
//     string biasesVFile;
//     if (layerHasBias) {
//         biasesMFile = j.at("biases_m_file").get<string>();
//         biasesVFile = j.at("biases_v_file").get<string>();
//     }
//
//     Adam adam;
//     adam.t = t;
//     adam.alpha = alpha;
//     adam.beta1 = beta1;
//     adam.beta2 = beta2;
//     adam.epsilon = epsilon;
//     adam.weightsMFile = weightsMFile;
//     adam.weightsVFile = weightsVFile;
//     adam.biasesMFile = biasesMFile;
//     adam.biasesVFile = biasesVFile;
//     // Now, different from the builder, I must add this to just this single layer, rather than as the optimizer for the whole network.
//     // FIXME: I also will need to save the default optimizer to baseline initialize in case any new layers are added to the pre-trained
//     network.
//     // So use layerId and network to assign it.
// }

vector<Event> Adam::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                               Optional<Event> sisterOptimizerLoadedEvent) {
    return physicalOptimizer->initialize(isFirstStamp, physicalSisterOptimizer, sisterOptimizerLoadedEvent);
}

// vector<Event> FullyConnected::initialize(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
//                                          bool isFirstStamp,
//                                          shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
//                                          Optional<Event> sisterLayerLoadedEvent,
//                                          vector<shared_ptr<Initializer>> &initializers) {
//     // Weights are set right now, based on 1 of 3 methods:
//     // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
//     // 2. Copy from a file - when loading a saved network
//     // 3. Run an initializer to set the weights - on an untrained network
//     if (!isFirstStamp) {
//         // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
//         assert(sisterLayer != nullptr);
//         ThorImplementation::Tensor weights = layer->getWeights();
//         Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
//         if (sisterLayerLoadedEvent.isPresent())
//             stream.waitEvent(sisterLayerLoadedEvent);
//         weights.copyFromAsync(sisterLayer->getWeights(), stream);
//         if (hasBias) {
//             ThorImplementation::Tensor biases = layer->getBiases();
//             Optional<ThorImplementation::Tensor> sisterLayerBiases = sisterLayer->getBiases();
//             assert(sisterLayerBiases.isPresent());
//             biases.copyFromAsync(sisterLayerBiases.get(), stream);
//         }
//         return {stream.putEvent(false, true)};
//     } else if (weightsFile.isPresent()) {
//         // 2. Copy from a file - when loading a saved network
//         assert(weightsInitializerBuilder.get() == nullptr);
//         assert(biasInitializerBuilder.get() == nullptr);
//         assert(layer->getWeights().getPlacement() == ThorImplementation::TensorPlacement::MemDevices::GPU);
//         Stream stream = Stream::getNextUploadStream(layer->getWeights().getPlacement().getDeviceNum());
//         layer->loadWeightsFromFile(weightsFile.get(), stream);
//         if (hasBias) {
//             assert(biasesFile.isPresent());
//             layer->loadBiasesFromFile(biasesFile.get(), stream);
//         }
//
//         // Can't use the file later, it may not still be there
//         weightsFile = Optional<string>::empty();
//         biasesFile = Optional<string>::empty();
//
//         return {stream.putEvent(false, true)};
//     } else {
//         // 3. Run an initializer to set the weights - on an untrained network
//         Optional<Event> initDoneEvent;
//         vector<Event> initDoneEvents;
//
//         shared_ptr<Initializer::Builder> weightsInitializerBuilderClone = weightsInitializerBuilder->clone();
//         weightsInitializerBuilderClone->tensorToInitialize(layer->getWeights());
//         weightsInitializerBuilderClone->layerThatOwnsTensor(layer.get());
//         initializers.push_back(weightsInitializerBuilderClone->build());
//         initDoneEvent = initializers.back()->getInitDoneEvent();
//         if (initDoneEvent.isPresent())
//             initDoneEvents.push_back(initDoneEvent);
//
//         if (layer->getBiases().isPresent()) {
//             shared_ptr<Initializer::Builder> biasInitializerBuilderClone = biasInitializerBuilder->clone();
//             biasInitializerBuilderClone->tensorToInitialize(layer->getBiases().get());
//             biasInitializerBuilderClone->layerThatOwnsTensor(layer.get());
//             initializers.push_back(biasInitializerBuilderClone->build());
//             initDoneEvent = initializers.back()->getInitDoneEvent();
//             if (initDoneEvent.isPresent())
//                 initDoneEvents.push_back(initDoneEvent);
//         }
//         return initDoneEvents;
//     }
// }

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adam", &Thor::Adam::deserialize);
    return true;
}();
}
