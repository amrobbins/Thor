#include "DeepLearning/Api/Training/TrainingStep.h"

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
                           std::vector<TrainingInputBinding> inputBindings)
    : name(std::move(name)),
      lossRoots(std::move(lossRoots)),
      optimizer(std::move(optimizer)),
      updateParameters(std::move(updateParameters)),
      inputBindings(std::move(inputBindings)),
      repeatCount(repeatCount),
      gradientClearPolicy(gradientClearPolicy),
      initialized(true) {
    validate();
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
    if (lossRoots.empty()) {
        throw std::runtime_error("TrainingStep requires at least one loss root tensor.");
    }
    for (const Tensor& lossRoot : lossRoots) {
        if (!lossRoot.isInitialized()) {
            throw std::runtime_error("TrainingStep loss roots must all be initialized tensors.");
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
    if (!updateParameters.empty() && optimizer == nullptr) {
        throw std::runtime_error("TrainingStep with update parameters requires an optimizer.");
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
    j["repeat_count"] = repeatCount;
    j["gradient_clear_policy"] = gradientClearPolicy;

    j["loss_roots"] = json::array();
    for (const Tensor& lossRoot : lossRoots) {
        j["loss_roots"].push_back(lossRoot.architectureJson());
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
    if (version != "1.0.0") {
        throw std::runtime_error("Unsupported TrainingStep version: " + version);
    }

    std::vector<Tensor> lossRoots;
    for (const json& lossRootJson : j.at("loss_roots")) {
        lossRoots.push_back(Tensor::deserialize(lossRootJson, archiveReader.get()));
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

    return TrainingStep(j.at("name").get<std::string>(),
                        std::move(lossRoots),
                        std::move(optimizer),
                        std::move(updateParameters),
                        j.at("repeat_count").get<uint32_t>(),
                        j.at("gradient_clear_policy").get<TrainingStep::GradientClearPolicy>(),
                        std::move(inputBindings));
}

}  // namespace Thor
