#include "DeepLearning/Api/Training/StepExecutable.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include <map>
#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

StepExecutable::StepExecutable(const TrainingStep& step, PlacedNetwork& placedNetwork)
    : name(step.getName()),
      lossRoots(step.getLossRoots()),
      resolvedLossRoots(placedNetwork.resolveApiTensors(lossRoots)),
      optimizer(step.getOptimizer()),
      updateParameterReferences(step.getUpdateParameters()),
      resolvedUpdateParameters(placedNetwork.resolveParameterReferences(updateParameterReferences)),
      inputBindings(step.getInputBindings()),
      repeatCount(step.getRepeatCount()),
      gradientClearPolicy(step.getGradientClearPolicy()),
      initialized(true) {
    std::map<std::string, std::string> explicitBindings;
    for (const TrainingInputBinding& inputBinding : inputBindings) {
        if (!placedNetwork.hasNetworkInput(inputBinding.getNetworkInputName())) {
            throw std::runtime_error("StepExecutable input binding references unknown NetworkInput '" +
                                     inputBinding.getNetworkInputName() + "'.");
        }
        explicitBindings[inputBinding.getNetworkInputName()] = inputBinding.getBatchInputName();
    }

    std::set<std::string> requiredBatchInputNameSet;
    for (const std::string& networkInputName : placedNetwork.getNetworkInputNames()) {
        auto explicitIt = explicitBindings.find(networkInputName);
        const std::string& batchInputName = explicitIt == explicitBindings.end() ? networkInputName : explicitIt->second;
        resolvedInputBindings.emplace_back(networkInputName, batchInputName);
        requiredBatchInputNameSet.insert(batchInputName);
    }
    requiredBatchInputNames.assign(requiredBatchInputNameSet.begin(), requiredBatchInputNameSet.end());

    for (const BoundParameter& parameter : resolvedUpdateParameters) {
        if (!parameter.isTrainingEnabled()) {
            throw std::runtime_error("StepExecutable update parameter '" + parameter.getName() +
                                     "' is not training-enabled in the placed network.");
        }
    }

    validate();
}

void StepExecutable::validate() const {
    if (!initialized) {
        return;
    }
    if (name.empty()) {
        throw std::runtime_error("StepExecutable requires a non-empty name.");
    }
    if (repeatCount == 0) {
        throw std::runtime_error("StepExecutable repeat_count must be >= 1.");
    }
    if (lossRoots.empty()) {
        throw std::runtime_error("StepExecutable requires at least one loss root tensor.");
    }
    if (lossRoots.size() != resolvedLossRoots.size()) {
        throw std::runtime_error("StepExecutable resolved loss-root count does not match logical loss-root count.");
    }
    if (updateParameterReferences.size() != resolvedUpdateParameters.size()) {
        throw std::runtime_error("StepExecutable resolved update parameter count does not match logical reference count.");
    }
    if (!updateParameterReferences.empty() && optimizer == nullptr) {
        for (const BoundParameter& parameter : resolvedUpdateParameters) {
            if (!parameter.hasOptimizer()) {
                throw std::runtime_error(
                    "StepExecutable update parameter '" + parameter.getName() +
                    "' does not have an optimizer. Provide a step optimizer, a network default optimizer, or a layer/parameter optimizer override.");
            }
        }
    }

    std::set<std::string> seenNetworkInputNames;
    std::set<std::string> requiredBatchInputNameSet;
    for (const TrainingInputBinding& inputBinding : resolvedInputBindings) {
        if (!inputBinding.isInitialized()) {
            throw std::runtime_error("StepExecutable resolved input bindings must all be initialized.");
        }
        if (!seenNetworkInputNames.insert(inputBinding.getNetworkInputName()).second) {
            throw std::runtime_error("StepExecutable resolved duplicate binding for network input '" +
                                     inputBinding.getNetworkInputName() + "'.");
        }
        requiredBatchInputNameSet.insert(inputBinding.getBatchInputName());
    }
    if (requiredBatchInputNames.size() != requiredBatchInputNameSet.size()) {
        throw std::runtime_error("StepExecutable required batch input names contain duplicates.");
    }
    for (const std::string& batchInputName : requiredBatchInputNames) {
        if (!requiredBatchInputNameSet.contains(batchInputName)) {
            throw std::runtime_error("StepExecutable required batch input name '" + batchInputName +
                                     "' is not present in resolved input bindings.");
        }
    }
}

json StepExecutable::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["name"] = name;
    j["repeat_count"] = repeatCount;
    j["gradient_clear_policy"] = gradientClearPolicy;
    j["planned"] = true;

    j["loss_roots"] = json::array();
    for (const Tensor& lossRoot : lossRoots) {
        j["loss_roots"].push_back(lossRoot.architectureJson());
    }

    j["input_bindings"] = json::array();
    for (const TrainingInputBinding& inputBinding : inputBindings) {
        j["input_bindings"].push_back(inputBinding.architectureJson());
    }

    j["resolved_input_bindings"] = json::array();
    for (const TrainingInputBinding& inputBinding : resolvedInputBindings) {
        j["resolved_input_bindings"].push_back(inputBinding.architectureJson());
    }

    j["resolved_loss_root_count"] = resolvedLossRoots.size();

    j["required_batch_input_names"] = requiredBatchInputNames;

    j["update_parameters"] = json::array();
    for (const ParameterReference& parameter : updateParameterReferences) {
        j["update_parameters"].push_back(parameter.architectureJson());
    }
    j["resolved_update_parameter_count"] = resolvedUpdateParameters.size();

    if (optimizer != nullptr) {
        j["optimizer"] = optimizer->architectureJson();
    }

    return j;
}

std::string StepExecutable::architectureJsonString() const { return architectureJson().dump(); }

}  // namespace Thor
