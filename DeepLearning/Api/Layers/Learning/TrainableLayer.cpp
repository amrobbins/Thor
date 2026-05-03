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

void TrainableLayer::attachOptimizer(std::shared_ptr<Optimizer> optimizer) {
    assert(optimizer != nullptr);
    this->optimizer = optimizer;

    for (const auto &parameter : getParameters()) {
        if (parameter != nullptr && parameter->isTrainable()) {
            parameter->setOptimizer(optimizer, /*override=*/false);
        }
    }
}

bool TrainableLayer::hasOptimizer() const {
    if (optimizer != nullptr)
        return true;

    for (const auto &parameter : getParameters()) {
        if (parameter != nullptr && parameter->isTrainable() && parameter->hasOptimizer()) {
            return true;
        }
    }

    return false;
}

std::shared_ptr<Optimizer> TrainableLayer::getOptimizer() const { return optimizer; }

void TrainableLayer::removeOptimizer() {
    optimizer.reset();
}

std::shared_ptr<Optimizer> TrainableLayer::optimizerForParameter(const std::shared_ptr<ParameterSpecification> &parameter) const {
    if (parameter != nullptr && parameter->hasOptimizer()) {
        return parameter->getOptimizer();
    }
    return optimizer;
}

void TrainableLayer::stampOptimizer(const std::shared_ptr<ThorImplementation::TrainableLayer> &physicalTrainableLayer) const {
    if (physicalTrainableLayer->isInferenceOnly()) {
        return;
    }

    const std::vector<std::string> physicalParameterNames = physicalTrainableLayer->listParameters();
    for (const std::string &parameterName : physicalParameterNames) {
        std::shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter = physicalTrainableLayer->getParameter(parameterName);
        if (!physicalParameter->isTrainable()) {
            continue;
        }

        std::shared_ptr<ParameterSpecification> apiParameter;
        if (hasParameter(parameterName)) {
            apiParameter = getParameterSpecification(parameterName);
            if (apiParameter->getInitializer() != nullptr) {
                physicalParameter->setInitializer(apiParameter->getInitializer()->stamp());
            }
            if (!apiParameter->isTrainingInitiallyEnabled()) {
                physicalParameter->setTrainingEnabled(false);
            }
        }

        std::shared_ptr<Optimizer> parameterOptimizer = optimizerForParameter(apiParameter);
        if (parameterOptimizer == nullptr) {
            continue;
        }

        std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = parameterOptimizer->stamp(physicalTrainableLayer);
        physicalParameter->setOptimizer(physicalOptimizer);
    }
}

void TrainableLayer::compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) {
    MultiConnectionLayer::compile(physicalLayer);
}

std::vector<Event> TrainableLayer::initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                              bool isFirstStamp,
                                              std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                              Optional<Event> sisterLayerLoadedEvent) {
    (void)isFirstStamp;
    (void)sisterLayer;
    (void)sisterLayerLoadedEvent;
    return MultiConnectionLayer::initialize(layer);
}

void TrainableLayer::addParameterArchitectureJson(json &j) const {
    if (!getParameters().empty()) {
        j["parameters"] = json::array();
        for (const auto &parameter : getParameters()) {
            if (parameter != nullptr) {
                j["parameters"].push_back(parameter->architectureJson());
            }
        }
    }

    if (optimizer != nullptr) {
        j["optimizer"] = optimizer->architectureJson();
    }
}

void TrainableLayer::deserializeParameterArchitectureJson(const json &j, shared_ptr<thor_file::TarReader> &archiveReader) {
    if (j.contains("parameters")) {
        for (const json &parameterJson : j.at("parameters")) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(parameterJson, archiveReader);
            addParameter(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }

    if (j.contains("optimizer")) {
        optimizer = Optimizer::deserialize(archiveReader, j.at("optimizer"), nullptr);
        for (const auto &parameter : getParameters()) {
            if (parameter != nullptr && parameter->isTrainable()) {
                parameter->setOptimizer(optimizer, /*override=*/false);
            }
        }
    }
}

void TrainableLayer::serializeParameters(json &j,
                                         thor_file::TarWriter &archiveWriter,
                                         Stream stream,
                                         bool saveOptimizerState,
                                         ThorImplementation::StampedNetwork &stampedNetwork) const {
    std::shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(getId());
    std::shared_ptr<ThorImplementation::TrainableLayer> physicalTrainableLayer =
        std::dynamic_pointer_cast<ThorImplementation::TrainableLayer>(physicalLayer);
    assert(physicalTrainableLayer != nullptr);

    const std::string layerName = std::string("layer") + std::to_string(getId());
    j["parameters"] = json::array();

    std::vector<std::string> parameterNames = physicalTrainableLayer->listParameters();
    std::sort(parameterNames.begin(), parameterNames.end());

    for (const std::string &parameterName : parameterNames) {
        std::shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter = physicalTrainableLayer->getParameter(parameterName);
        Optional<ThorImplementation::Tensor> maybeStorage = physicalParameter->getStorage();
        if (maybeStorage.isEmpty()) {
            continue;
        }

        const std::string fileName = layerName + "_" + parameterName + ".gds";
        archiveWriter.addArchiveFile(fileName, maybeStorage.get());

        json parameterJson;
        if (hasParameter(parameterName)) {
            parameterJson = getParameterSpecification(parameterName)->architectureJson();
        } else {
            parameterJson["version"] = ParameterSpecification::getVersion();
            parameterJson["name"] = parameterName;
            parameterJson["storage"] = Tensor(maybeStorage.get().getDataType(), maybeStorage.get().getDimensions()).architectureJson();
            parameterJson["trainable"] = physicalParameter->isTrainable();
            if (physicalParameter->isTrainable()) {
                parameterJson["training_enabled"] = physicalParameter->isTrainingEnabled();
            }
        }
        parameterJson["storage_tensor"] = fileName;

        if (parameterName == "weights") {
            j["weights_tensor"] = fileName;
        } else if (parameterName == "biases") {
            j["biases_tensor"] = fileName;
        }

        if (physicalParameter->hasOptimizer()) {
            std::shared_ptr<Optimizer> apiOptimizer = optimizer;
            if (hasParameter(parameterName) && getParameterSpecification(parameterName)->hasOptimizer()) {
                apiOptimizer = getParameterSpecification(parameterName)->getOptimizer();
            }
            if (apiOptimizer != nullptr) {
                json optimizerJson = apiOptimizer->serialize(archiveWriter,
                                                             stream,
                                                             physicalParameter->getOptimizer(),
                                                             layerName + "_" + parameterName,
                                                             saveOptimizerState);
                parameterJson["optimizer"] = optimizerJson;
                if (parameterName == "weights") {
                    j["weights_optimizer"] = optimizerJson;
                } else if (parameterName == "biases") {
                    j["biases_optimizer"] = optimizerJson;
                }
            }
        }

        j["parameters"].push_back(parameterJson);
    }
}

void TrainableLayer::deserializeParameters(const json &j, shared_ptr<thor_file::TarReader> &archiveReader) {
    deserializeParameterArchitectureJson(j, archiveReader);
}

}  // namespace Thor
