#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

#include <algorithm>
#include <stdexcept>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace Thor {

TrainableLayer::TrainableLayer(std::vector<std::shared_ptr<ParameterSpecification>> parameters) : Parameterizable(getId()) {
    for (const auto &parameter : parameters)
        addParameter(parameter);
}

unordered_map<string, TrainableLayer::Deserializer> &TrainableLayer::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void TrainableLayer::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

void TrainableLayer::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network) {
    assert(j.at("factory").get<string>() == Layer::Factory::Learning.value());
    string type = j.at("layer_type").get<string>();

    unordered_map<string, TrainableLayer::Deserializer> &registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw runtime_error("Unknown trainable layer type: " + type);

    auto deserializer = it->second;
    deserializer(archiveReader, j, network);
}

void TrainableLayer::attachDefaultOptimizer(std::shared_ptr<Optimizer> optimizer) {
    for (const auto &parameter : getParameters()) {
        if (parameter != nullptr && parameter->isTrainable()) {
            parameter->setOptimizer(optimizer, /*override=*/false);
        }
    }
}

bool TrainableLayer::hasOptimizer() const {
    for (const auto &parameter : parameters) {
        if (parameter != nullptr && parameter->isTrainable() && parameter->hasOptimizer()) {
            return true;
        }
    }
    return false;
}

// void TrainableLayer::stampOptimizer(const std::shared_ptr<ThorImplementation::TrainableLayer> &physicalTrainableLayer) const {
//     if (physicalTrainableLayer->isInferenceOnly()) {
//         return;
//     }
//
//     const std::vector<std::string> physicalParameterNames = physicalTrainableLayer->listParameters();
//     for (const std::string &parameterName : physicalParameterNames) {
//         std::shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter = physicalTrainableLayer->getParameter(parameterName);
//         if (!physicalParameter->isTrainable()) {
//             continue;
//         }
//
//         std::shared_ptr<ParameterSpecification> apiParameter;
//         if (hasParameter(parameterName)) {
//             apiParameter = getParameterSpecification(parameterName);
//             if (apiParameter->getInitializer() != nullptr) {
//                 physicalParameter->setInitializer(apiParameter->getInitializer()->stamp());
//             }
//             if (!apiParameter->isTrainingInitiallyEnabled()) {
//                 physicalParameter->setTrainingEnabled(false);
//             }
//         }
//
//         std::shared_ptr<Optimizer> parameterOptimizer = optimizerForParameter(apiParameter);
//         if (parameterOptimizer == nullptr) {
//             continue;
//         }
//
//         std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = parameterOptimizer->stamp(physicalTrainableLayer);
//         physicalParameter->setOptimizer(physicalOptimizer);
//     }
// }

void TrainableLayer::compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) { MultiConnectionLayer::compile(physicalLayer); }

std::vector<Event> TrainableLayer::initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                              bool isFirstStamp,
                                              std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                              Optional<Event> sisterLayerLoadedEvent) {
    (void)isFirstStamp;
    (void)sisterLayer;
    (void)sisterLayerLoadedEvent;

    assert(layer != nullptr);

    std::vector<Event> initDoneEvents = MultiConnectionLayer::initialize(layer);

    for (const std::string &parameterName : layer->listParameters()) {
        std::shared_ptr<ThorImplementation::PhysicalParameter> parameter = layer->getParameter(parameterName);
        if (parameter == nullptr || !parameter->hasInitializer()) {
            continue;
        }

        Optional<ThorImplementation::Tensor> storage = parameter->getStorage();
        assert(storage.isPresent());

        const ThorImplementation::TensorPlacement placement = storage.get().getPlacement();
        const int gpuNum = placement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU ? placement.getDeviceNum() : 0;
        Stream initStream = Stream::getNextUploadStream(gpuNum);
        parameter->initialize(initStream);
        initDoneEvents.push_back(initStream.putEvent(false, true));
    }

    return initDoneEvents;
}

void TrainableLayer::deserializeParameterArchitectureJson(const json &j, shared_ptr<thor_file::TarReader> &archiveReader) {
    if (j.contains("parameters")) {
        for (const json &parameterJson : j.at("parameters")) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(parameterJson, archiveReader);
            addParameter(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }
}

void TrainableLayer::deserializeParameters(const json &j, shared_ptr<thor_file::TarReader> &archiveReader) {
    deserializeParameterArchitectureJson(j, archiveReader);
}

uint64_t TrainableLayer::getParameterBytes() const {
    uint64_t parameterBytes = 0;
    for (const auto &param : parameters) {
        parameterBytes += param->getTotalSizeInBytes();
    }
    return parameterBytes;
}

json TrainableLayer::architectureJson() const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = getLayerType();
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    // j["activation"] = activation->architectureJson();

    if (!parameters.empty()) {
        json parameters_json = json::object();
        for (const auto &parameter : parameters) {
            parameters_json[parameter->getName()] = parameter->architectureJson();
        }
        j["parameters"] = parameters_json;
    }

    // json inputs = json::array();
    // for (uint32_t i = 0; i < standaloneFCFeatureInputs.size(); ++i) {
    //     inputs.push_back(standaloneFCFeatureInputs[i].architectureJson());
    // }
    // j["inputs"] = inputs;
    //
    // // Output connections
    // json outputs = json::array();
    // for (uint32_t i = 0; i < standaloneFCFeatureOutputs.size(); ++i) {
    //     outputs.push_back(standaloneFCFeatureOutputs[i].architectureJson());
    // }
    // j["outputs"] = outputs;
    //
    // if (weightsInitializer != nullptr) {
    //     j["weights_initializer"] = weightsInitializer->architectureJson();
    // }
    // if (biasesInitializer != nullptr) {
    //     j["biases_initializer"] = biasesInitializer->architectureJson();
    // }
    //
    // if (hasOptimizer()) {
    //     j["weights_optimizer"] = weightsOptimizer->architectureJson();
    //     if (hasBias) {
    //         j["biases_optimizer"] = biasesOptimizer->architectureJson();
    //     }
    // }

    return j;
}

nlohmann::json TrainableLayer::serialize(thor_file::TarWriter &archiveWriter,
                                         Stream stream,
                                         bool saveOptimizerState,
                                         ThorImplementation::StampedNetwork &stampedNetwork) const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network
    json j = architectureJson();

    string layerName = string("layer") + to_string(getId());

    // Dump the weights to a file and record its name
    shared_ptr<ThorImplementation::TrainableLayer> trainableLayer = nullptr;
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
    trainableLayer = dynamic_pointer_cast<ThorImplementation::TrainableLayer>(physicalLayer);
    assert(trainableLayer != nullptr);

    ThorImplementation::Tensor weights;
    ThorImplementation::Tensor biases;
    string weightsFile;
    string biasesFile;
    if (trainableLayer != nullptr) {
        // if (hasBias) {
        //     biasesFile = (layerName + "_biases.gds");
        //     j["biases_tensor"] = biasesFile;
        //     biases = trainableLayer->getParameter("biases")->getStorage().get();
        //     archiveWriter.addArchiveFile(biasesFile, biases);
        // }

        weightsFile = (layerName + "_weights.gds");
        j["weights_tensor"] = weightsFile;
        weights = trainableLayer->getParameter("weights")->getStorage();
        archiveWriter.addArchiveFile(weightsFile, weights);
    }

    // if (hasOptimizer()) {
    //     j["weights_optimizer"] = weightsOptimizer->serialize(archiveWriter,
    //                                                          stream,
    //                                                          trainableLayer->getParameter("weights")->getOptimizer(),
    //                                                          string("layer") + to_string(getId()),
    //                                                          saveOptimizerState);
    //     if (hasBias) {
    //         j["biases_optimizer"] = biasesOptimizer->serialize(archiveWriter,
    //                                                            stream,
    //                                                            trainableLayer->getParameter("biases")->getOptimizer(),
    //                                                            string("layer") + to_string(getId()),
    //                                                            saveOptimizerState);
    //     }
    // }

    return j;
}

}  // namespace Thor
