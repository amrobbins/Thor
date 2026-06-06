#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace Thor {

class TrainingInputBinding {
   public:
    TrainingInputBinding() = default;
    TrainingInputBinding(std::string networkInputName, std::string batchInputName);

    [[nodiscard]] const std::string& getNetworkInputName() const { return networkInputName; }
    [[nodiscard]] const std::string& getBatchInputName() const { return batchInputName; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    static TrainingInputBinding deserialize(const nlohmann::json& j);

    [[nodiscard]] std::string getVersion() const { return "1.0.0"; }

    bool operator==(const TrainingInputBinding& other) const;
    bool operator!=(const TrainingInputBinding& other) const { return !(*this == other); }
    bool operator<(const TrainingInputBinding& other) const;

   private:
    void validate() const;

    std::string networkInputName{};
    std::string batchInputName{};
    bool initialized = false;
};

}  // namespace Thor
