#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
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
    for (const auto &[name, parameter] : parameters) {
        (void)name;
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
    return MultiConnectionLayer::initialize(layer);
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
    for (const auto &[paramName, param] : parameters) {
        parameterBytes += param->getTotalSizeInBytes();
    }
    return parameterBytes;
}

}  // namespace Thor
