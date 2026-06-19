#include "DeepLearning/Api/Training/TrainingPhase.h"

#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

TrainingPhase::TrainingPhase(std::string name,
                             std::vector<Tensor> lossRoots,
                             std::map<std::string, Tensor> outputs,
                             std::vector<std::string> dependsOn,
                             bool enabled)
    : name(std::move(name)),
      lossRoots(std::move(lossRoots)),
      outputs(std::move(outputs)),
      dependsOn(std::move(dependsOn)),
      enabled(enabled),
      initialized(true) {
    validate();
}

void TrainingPhase::enable() {
    validate();
    enabled = true;
}

void TrainingPhase::disable() {
    validate();
    enabled = false;
}

void TrainingPhase::setEnabled(bool enabled) {
    validate();
    this->enabled = enabled;
}

void TrainingPhase::validate() const {
    if (!initialized) {
        return;
    }
    if (name.empty()) {
        throw std::runtime_error("TrainingPhase requires a non-empty name.");
    }

    for (const Tensor& lossRoot : lossRoots) {
        if (!lossRoot.isInitialized()) {
            throw std::runtime_error("TrainingPhase loss roots must all be initialized tensors.");
        }
    }

    std::set<std::string> seenOutputNames;
    for (const auto& [outputName, outputTensor] : outputs) {
        if (outputName.empty()) {
            throw std::runtime_error("TrainingPhase output names must be non-empty.");
        }
        if (!outputTensor.isInitialized()) {
            throw std::runtime_error("TrainingPhase outputs must all be initialized tensors.");
        }
        if (!seenOutputNames.insert(outputName).second) {
            throw std::runtime_error("TrainingPhase output names must be unique.");
        }
    }

    std::set<std::string> seenDependencies;
    for (const std::string& dependency : dependsOn) {
        if (dependency.empty()) {
            throw std::runtime_error("TrainingPhase dependency names must be non-empty.");
        }
        if (dependency == name) {
            throw std::runtime_error("TrainingPhase '" + name + "' cannot depend on itself.");
        }
        if (!seenDependencies.insert(dependency).second) {
            throw std::runtime_error("TrainingPhase '" + name + "' contains duplicate dependency '" + dependency + "'.");
        }
    }
}

json TrainingPhase::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["name"] = name;
    j["enabled"] = enabled;

    j["loss_roots"] = json::array();
    for (const Tensor& lossRoot : lossRoots) {
        j["loss_roots"].push_back(lossRoot.architectureJson());
    }

    j["outputs"] = json::object();
    for (const auto& [outputName, outputTensor] : outputs) {
        j["outputs"][outputName] = outputTensor.architectureJson();
    }

    j["depends_on"] = json::array();
    for (const std::string& dependency : dependsOn) {
        j["depends_on"].push_back(dependency);
    }

    return j;
}

std::string TrainingPhase::architectureJsonString() const { return architectureJson().dump(); }

TrainingPhase TrainingPhase::deserialize(const json& j, std::shared_ptr<thor_file::TarReader> archiveReader) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0") {
        throw std::runtime_error("Unsupported TrainingPhase version: " + version);
    }

    std::vector<Tensor> lossRoots;
    if (j.contains("loss_roots")) {
        for (const json& lossRootJson : j.at("loss_roots")) {
            lossRoots.push_back(Tensor::deserialize(lossRootJson, archiveReader.get()));
        }
    }

    std::map<std::string, Tensor> outputs;
    if (j.contains("outputs")) {
        for (auto it = j.at("outputs").begin(); it != j.at("outputs").end(); ++it) {
            outputs.emplace(it.key(), Tensor::deserialize(it.value(), archiveReader.get()));
        }
    }

    std::vector<std::string> dependsOn;
    if (j.contains("depends_on")) {
        for (const json& dependencyJson : j.at("depends_on")) {
            dependsOn.push_back(dependencyJson.get<std::string>());
        }
    }

    bool enabled = true;
    if (j.contains("enabled")) {
        enabled = j.at("enabled").get<bool>();
    }

    return TrainingPhase(j.at("name").get<std::string>(), std::move(lossRoots), std::move(outputs), std::move(dependsOn), enabled);
}

}  // namespace Thor
