#include "DeepLearning/Api/Parameter/ParameterReference.h"

#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

namespace {
constexpr const char* PARAMETER_REFERENCE_VERSION = "1.0.0";
}

ParameterReference::ParameterReference(uint64_t parameterizableId, std::string parameterName)
    : parameterizableId(parameterizableId), parameterName(std::move(parameterName)), initialized(true) {
    if (this->parameterizableId == 0) {
        throw std::runtime_error("ParameterReference requires a non-zero parameterizable id.");
    }
    if (this->parameterName.empty()) {
        throw std::runtime_error("ParameterReference requires a non-empty parameter name.");
    }
}

json ParameterReference::architectureJson() const {
    if (!initialized) {
        throw std::runtime_error("Cannot serialize an uninitialized ParameterReference.");
    }
    return json{{"version", PARAMETER_REFERENCE_VERSION},
                {"parameterizable_id", parameterizableId},
                {"parameter_name", parameterName}};
}

ParameterReference ParameterReference::deserialize(const json& j) {
    const std::string version = j.at("version").get<std::string>();
    if (version != PARAMETER_REFERENCE_VERSION) {
        throw std::runtime_error("Unsupported ParameterReference version: " + version);
    }
    return ParameterReference(j.at("parameterizable_id").get<uint64_t>(), j.at("parameter_name").get<std::string>());
}

bool ParameterReference::operator==(const ParameterReference& other) const {
    return initialized == other.initialized && parameterizableId == other.parameterizableId && parameterName == other.parameterName;
}

bool ParameterReference::operator<(const ParameterReference& other) const {
    if (initialized != other.initialized) {
        return initialized < other.initialized;
    }
    if (parameterizableId != other.parameterizableId) {
        return parameterizableId < other.parameterizableId;
    }
    return parameterName < other.parameterName;
}

}  // namespace Thor
