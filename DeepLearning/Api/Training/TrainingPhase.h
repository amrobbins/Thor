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

    TrainingPhase(std::string name, std::shared_ptr<Network> network, bool enabled = true);

    [[nodiscard]] const std::string& getName() const { return name; }
    [[nodiscard]] std::shared_ptr<Network> getNetwork() const { return network; }
    [[nodiscard]] const std::vector<Tensor>& getLossRoots() const;
    [[nodiscard]] const std::map<std::string, Tensor>& getOutputs() const;
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

    mutable bool networkDerivedCachesValid = false;
    mutable std::vector<Tensor> cachedNetworkLossRoots{};
    mutable std::map<std::string, Tensor> cachedNetworkOutputs{};

    bool enabled = true;
    bool initialized = false;
};

}  // namespace Thor
