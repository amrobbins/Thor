#include "DeepLearning/Api/Training/TrainingStep.h"

#include <map>
#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

TrainingStep::TrainingStep(std::string name,
                           std::vector<Tensor> lossRoots,
                           std::shared_ptr<Optimizer> optimizer,
                           std::vector<ParameterReference> updateParameters,
                           uint32_t repeatCount,
                           GradientClearPolicy gradientClearPolicy,
                           std::vector<TrainingInputBinding> inputBindings,
                           bool enabled)
    : name(std::move(name)),
      lossRoots(std::move(lossRoots)),
      phases(legacyLossRootsToPhases(this->name, this->lossRoots)),
      optimizer(std::move(optimizer)),
      updateParameters(std::move(updateParameters)),
      inputBindings(std::move(inputBindings)),
      repeatCount(repeatCount),
      gradientClearPolicy(gradientClearPolicy),
      enabled(enabled),
      initialized(true) {
    validate();
}

TrainingStep::TrainingStep(std::string name,
                           std::vector<std::shared_ptr<TrainingPhase>> phases,
                           std::shared_ptr<Optimizer> optimizer,
                           std::vector<ParameterReference> updateParameters,
                           uint32_t repeatCount,
                           GradientClearPolicy gradientClearPolicy,
                           std::vector<TrainingInputBinding> inputBindings,
                           bool enabled)
    : name(std::move(name)),
      lossRoots(collectAllLossRoots(phases)),
      phases(std::move(phases)),
      optimizer(std::move(optimizer)),
      updateParameters(std::move(updateParameters)),
      inputBindings(std::move(inputBindings)),
      repeatCount(repeatCount),
      gradientClearPolicy(gradientClearPolicy),
      enabled(enabled),
      initialized(true) {
    validate();
}

std::vector<std::shared_ptr<TrainingPhase>> TrainingStep::legacyLossRootsToPhases(const std::string& stepName,
                                                                                  const std::vector<Tensor>& lossRoots) {
    return {std::make_shared<TrainingPhase>(stepName + "_phase", lossRoots)};
}

std::vector<Tensor> TrainingStep::collectAllLossRoots(const std::vector<std::shared_ptr<TrainingPhase>>& phases) {
    std::vector<Tensor> roots;
    for (const std::shared_ptr<TrainingPhase>& phase : phases) {
        if (phase == nullptr) {
            continue;
        }
        const std::vector<Tensor>& phaseRoots = phase->getLossRoots();
        roots.insert(roots.end(), phaseRoots.begin(), phaseRoots.end());
    }
    return roots;
}

void TrainingStep::enable() {
    validate();
    enabled = true;
}

void TrainingStep::disable() {
    validate();
    enabled = false;
}

void TrainingStep::setEnabled(bool enabled) {
    validate();
    this->enabled = enabled;
}

void TrainingStep::validate() const {
    if (!initialized) {
        return;
    }
    if (name.empty()) {
        throw std::runtime_error("TrainingStep requires a non-empty name.");
    }
    if (repeatCount == 0) {
        throw std::runtime_error("TrainingStep repeat_count must be >= 1.");
    }
    if (phases.empty()) {
        throw std::runtime_error("TrainingStep requires at least one TrainingPhase.");
    }
    if (lossRoots.empty()) {
        throw std::runtime_error("TrainingStep requires at least one loss root tensor across its phases.");
    }
    for (const Tensor& lossRoot : lossRoots) {
        if (!lossRoot.isInitialized()) {
            throw std::runtime_error("TrainingStep loss roots must all be initialized tensors.");
        }
    }

    std::set<std::string> seenPhaseNames;
    for (const std::shared_ptr<TrainingPhase>& phase : phases) {
        if (phase == nullptr) {
            throw std::runtime_error("TrainingStep phases must not contain null TrainingPhase references.");
        }
        if (!phase->isInitialized()) {
            throw std::runtime_error("TrainingStep phases must all be initialized TrainingPhase objects.");
        }
        if (!seenPhaseNames.insert(phase->getName()).second) {
            throw std::runtime_error("TrainingStep '" + name + "' contains duplicate phase name '" + phase->getName() + "'.");
        }
    }

    std::set<ParameterReference> seen;
    for (const ParameterReference& parameter : updateParameters) {
        if (!parameter.isInitialized()) {
            throw std::runtime_error("TrainingStep update parameters must all be initialized ParameterReference objects.");
        }
        if (!seen.insert(parameter).second) {
            throw std::runtime_error("TrainingStep update parameter list contains a duplicate reference to layer id " +
                                     std::to_string(parameter.getParameterizableId()) + " parameter '" + parameter.getParameterName() + "'.");
        }
    }
    std::set<std::string> seenNetworkInputNames;
    for (const TrainingInputBinding& inputBinding : inputBindings) {
        if (!inputBinding.isInitialized()) {
            throw std::runtime_error("TrainingStep input bindings must all be initialized TrainingInputBinding objects.");
        }
        if (!seenNetworkInputNames.insert(inputBinding.getNetworkInputName()).second) {
            throw std::runtime_error("TrainingStep input bindings contain duplicate binding for network input '" +
                                     inputBinding.getNetworkInputName() + "'.");
        }
    }
}

void TrainingStep::validateEnabledPhaseDependencies() const {
    validate();
    if (!enabled) {
        return;
    }

    std::map<std::string, uint64_t> phaseIndexByName;
    for (uint64_t i = 0; i < phases.size(); ++i) {
        phaseIndexByName.emplace(phases[i]->getName(), i);
    }

    for (uint64_t phaseIndex = 0; phaseIndex < phases.size(); ++phaseIndex) {
        const TrainingPhase& phase = *phases[phaseIndex];
        if (!phase.isEnabled()) {
            continue;
        }
        for (const std::string& dependencyName : phase.getDependsOn()) {
            auto dependencyIt = phaseIndexByName.find(dependencyName);
            if (dependencyIt == phaseIndexByName.end()) {
                throw std::runtime_error("TrainingStep '" + name + "' enabled phase '" + phase.getName() +
                                         "' depends on missing phase '" + dependencyName + "'.");
            }
            const uint64_t dependencyIndex = dependencyIt->second;
            if (dependencyIndex >= phaseIndex) {
                throw std::runtime_error("TrainingStep '" + name + "' enabled phase '" + phase.getName() +
                                         "' depends on phase '" + dependencyName +
                                         "', but that dependency does not appear earlier in the step.");
            }
            if (!phases[dependencyIndex]->isEnabled()) {
                throw std::runtime_error("TrainingStep '" + name + "' enabled phase '" + phase.getName() +
                                         "' depends on disabled phase '" + dependencyName + "'.");
            }
        }
    }
}

std::vector<Tensor> TrainingStep::getActiveLossRoots() const {
    validate();
    if (!enabled) {
        return {};
    }
    validateEnabledPhaseDependencies();

    std::vector<Tensor> activeRoots;
    for (const std::shared_ptr<TrainingPhase>& phase : phases) {
        if (!phase->isEnabled()) {
            continue;
        }
        const std::vector<Tensor>& phaseRoots = phase->getLossRoots();
        activeRoots.insert(activeRoots.end(), phaseRoots.begin(), phaseRoots.end());
    }
    return activeRoots;
}

std::vector<std::string> TrainingStep::getActivePhaseNames() const {
    validate();
    if (!enabled) {
        return {};
    }
    validateEnabledPhaseDependencies();

    std::vector<std::string> activeNames;
    for (const std::shared_ptr<TrainingPhase>& phase : phases) {
        if (phase->isEnabled()) {
            activeNames.push_back(phase->getName());
        }
    }
    return activeNames;
}

bool TrainingStep::updatesParameter(const ParameterReference& parameter) const {
    for (const ParameterReference& updateParameter : updateParameters) {
        if (updateParameter == parameter) {
            return true;
        }
    }
    return false;
}

json TrainingStep::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["name"] = name;
    j["enabled"] = enabled;
    j["repeat_count"] = repeatCount;
    j["gradient_clear_policy"] = gradientClearPolicy;

    j["loss_roots"] = json::array();
    for (const Tensor& lossRoot : lossRoots) {
        j["loss_roots"].push_back(lossRoot.architectureJson());
    }

    j["phases"] = json::array();
    for (const std::shared_ptr<TrainingPhase>& phase : phases) {
        j["phases"].push_back(phase->architectureJson());
    }

    j["update_parameters"] = json::array();
    for (const ParameterReference& parameter : updateParameters) {
        j["update_parameters"].push_back(parameter.architectureJson());
    }

    j["input_bindings"] = json::array();
    for (const TrainingInputBinding& inputBinding : inputBindings) {
        j["input_bindings"].push_back(inputBinding.architectureJson());
    }

    if (optimizer != nullptr) {
        j["optimizer"] = optimizer->architectureJson();
    }

    return j;
}

std::string TrainingStep::architectureJsonString() const { return architectureJson().dump(); }


TrainingStep TrainingStep::deserialize(const json& j, std::shared_ptr<thor_file::TarReader> archiveReader, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported TrainingStep version: " + version);
    }

    std::vector<ParameterReference> updateParameters;
    for (const json& parameterJson : j.at("update_parameters")) {
        updateParameters.push_back(ParameterReference::deserialize(parameterJson));
    }

    std::vector<TrainingInputBinding> inputBindings;
    if (j.contains("input_bindings")) {
        for (const json& inputBindingJson : j.at("input_bindings")) {
            inputBindings.push_back(TrainingInputBinding::deserialize(inputBindingJson));
        }
    }

    std::shared_ptr<Optimizer> optimizer = nullptr;
    if (j.contains("optimizer")) {
        optimizer = Optimizer::deserialize(archiveReader, j.at("optimizer"), network);
    }

    const uint32_t repeatCount = j.at("repeat_count").get<uint32_t>();
    const GradientClearPolicy gradientClearPolicy = j.at("gradient_clear_policy").get<TrainingStep::GradientClearPolicy>();
    const bool enabled = j.contains("enabled") ? j.at("enabled").get<bool>() : true;

    if (j.contains("phases")) {
        std::vector<std::shared_ptr<TrainingPhase>> phases;
        for (const json& phaseJson : j.at("phases")) {
            phases.push_back(std::make_shared<TrainingPhase>(TrainingPhase::deserialize(phaseJson, archiveReader)));
        }
        return TrainingStep(j.at("name").get<std::string>(),
                            std::move(phases),
                            std::move(optimizer),
                            std::move(updateParameters),
                            repeatCount,
                            gradientClearPolicy,
                            std::move(inputBindings),
                            enabled);
    }

    std::vector<Tensor> lossRoots;
    for (const json& lossRootJson : j.at("loss_roots")) {
        lossRoots.push_back(Tensor::deserialize(lossRootJson, archiveReader.get()));
    }

    return TrainingStep(j.at("name").get<std::string>(),
                        std::move(lossRoots),
                        std::move(optimizer),
                        std::move(updateParameters),
                        repeatCount,
                        gradientClearPolicy,
                        std::move(inputBindings),
                        enabled);
}

}  // namespace Thor
