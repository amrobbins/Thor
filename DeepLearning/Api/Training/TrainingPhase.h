#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace thor_file {
class TarReader;
}

namespace Thor {

class Network;

class TrainingPhase {
   public:
    TrainingPhase() = default;

    // New phase model: a TrainingPhase is an ordinary Network with enable/disable control.
    TrainingPhase(std::string name, std::shared_ptr<Network> network, bool enabled = true);

    // Legacy compatibility constructor. This keeps existing single-network TrainingProgram users and tests
    // working until the runner is migrated to placed phase graphs.
    TrainingPhase(std::string name,
                  std::vector<Tensor> lossRoots = {},
                  std::map<std::string, Tensor> outputs = {},
                  std::vector<std::string> dependsOn = {},
                  bool enabled = true);

    [[nodiscard]] const std::string& getName() const { return name; }
    [[nodiscard]] std::shared_ptr<Network> getNetwork() const { return network; }
    [[nodiscard]] bool hasNetwork() const { return network != nullptr; }
    [[nodiscard]] const std::vector<Tensor>& getLossRoots() const;
    [[nodiscard]] const std::map<std::string, Tensor>& getOutputs() const;
    [[nodiscard]] const std::vector<std::string>& getDependsOn() const { return legacyDependsOn; }
    [[nodiscard]] bool isEnabled() const { return enabled; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    void enable();
    void disable();
    void setEnabled(bool enabled);

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    static TrainingPhase deserialize(const nlohmann::json& j, std::shared_ptr<thor_file::TarReader> archiveReader = nullptr);

    [[nodiscard]] std::string getVersion() const { return "1.1.0"; }

   private:
    void validate() const;
    void refreshNetworkDerivedCaches() const;

    std::string name{};
    std::shared_ptr<Network> network = nullptr;

    // Legacy tensor-root phase fields. New network-backed phases derive roots and outputs from the phase network.
    std::vector<Tensor> legacyLossRoots{};
    std::map<std::string, Tensor> legacyOutputs{};
    std::vector<std::string> legacyDependsOn{};

    mutable bool networkDerivedCachesValid = false;
    mutable std::vector<Tensor> cachedNetworkLossRoots{};
    mutable std::map<std::string, Tensor> cachedNetworkOutputs{};

    bool enabled = true;
    bool initialized = false;
};

}  // namespace Thor
