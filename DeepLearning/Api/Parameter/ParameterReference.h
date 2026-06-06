#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace Thor {

class ParameterReference {
   public:
    ParameterReference() = default;
    ParameterReference(uint64_t parameterizableId, std::string parameterName);

    [[nodiscard]] uint64_t getParameterizableId() const { return parameterizableId; }
    [[nodiscard]] const std::string& getParameterName() const { return parameterName; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    [[nodiscard]] nlohmann::json architectureJson() const;
    static ParameterReference deserialize(const nlohmann::json& j);

    bool operator==(const ParameterReference& other) const;
    bool operator!=(const ParameterReference& other) const { return !(*this == other); }
    bool operator<(const ParameterReference& other) const;

   private:
    uint64_t parameterizableId = 0;
    std::string parameterName{};
    bool initialized = false;
};

}  // namespace Thor
