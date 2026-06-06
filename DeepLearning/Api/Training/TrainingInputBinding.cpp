#include "DeepLearning/Api/Training/TrainingInputBinding.h"

#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

TrainingInputBinding::TrainingInputBinding(std::string networkInputName, std::string batchInputName)
    : networkInputName(std::move(networkInputName)), batchInputName(std::move(batchInputName)), initialized(true) {
    validate();
}

void TrainingInputBinding::validate() const {
    if (!initialized) {
        return;
    }
    if (networkInputName.empty()) {
        throw std::runtime_error("TrainingInputBinding requires a non-empty network input name.");
    }
    if (batchInputName.empty()) {
        throw std::runtime_error("TrainingInputBinding requires a non-empty batch input name.");
    }
}

json TrainingInputBinding::architectureJson() const {
    validate();
    return json{{"version", getVersion()}, {"network_input_name", networkInputName}, {"batch_input_name", batchInputName}};
}

std::string TrainingInputBinding::architectureJsonString() const { return architectureJson().dump(); }

TrainingInputBinding TrainingInputBinding::deserialize(const json& j) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0") {
        throw std::runtime_error("Unsupported TrainingInputBinding version: " + version);
    }
    return TrainingInputBinding(j.at("network_input_name").get<std::string>(), j.at("batch_input_name").get<std::string>());
}

bool TrainingInputBinding::operator==(const TrainingInputBinding& other) const {
    return initialized == other.initialized && networkInputName == other.networkInputName && batchInputName == other.batchInputName;
}

bool TrainingInputBinding::operator<(const TrainingInputBinding& other) const {
    if (initialized != other.initialized) {
        return initialized < other.initialized;
    }
    if (networkInputName != other.networkInputName) {
        return networkInputName < other.networkInputName;
    }
    return batchInputName < other.batchInputName;
}

}  // namespace Thor
