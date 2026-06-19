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

class TrainingPhase {
   public:
    TrainingPhase() = default;
    TrainingPhase(std::string name,
                  std::vector<Tensor> lossRoots = {},
                  std::map<std::string, Tensor> outputs = {},
                  std::vector<std::string> dependsOn = {},
                  bool enabled = true);

    [[nodiscard]] const std::string& getName() const { return name; }
    [[nodiscard]] const std::vector<Tensor>& getLossRoots() const { return lossRoots; }
    [[nodiscard]] const std::map<std::string, Tensor>& getOutputs() const { return outputs; }
    [[nodiscard]] const std::vector<std::string>& getDependsOn() const { return dependsOn; }
    [[nodiscard]] bool isEnabled() const { return enabled; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    void enable();
    void disable();
    void setEnabled(bool enabled);

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    static TrainingPhase deserialize(const nlohmann::json& j, std::shared_ptr<thor_file::TarReader> archiveReader = nullptr);

    [[nodiscard]] std::string getVersion() const { return "1.0.0"; }

   private:
    void validate() const;

    std::string name{};
    std::vector<Tensor> lossRoots{};
    std::map<std::string, Tensor> outputs{};
    std::vector<std::string> dependsOn{};
    bool enabled = true;
    bool initialized = false;
};

}  // namespace Thor
