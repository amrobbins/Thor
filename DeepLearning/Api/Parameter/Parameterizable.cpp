#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"

#include <algorithm>
#include <stdexcept>

using namespace std;
using nlohmann::json;

namespace Thor {

void Parameterizable::addParameter(const std::shared_ptr<ParameterSpecification>& parameter) {
    if (parameter == nullptr) {
        throw runtime_error("Trying to add a null parameter specification.");
    }

    const string paramName = parameter->getName();
    if (hasParameter(paramName)) {
        throw runtime_error("Trying to add parameter named " + paramName +
                            " but that parameter is already present. Existing parameters: " + listParametersString());
    }
    parameters[paramName] = parameter;
}

bool Parameterizable::hasParameter(const std::string& name) const { return parameters.contains(name); }

std::vector<std::string> Parameterizable::listParameters() const {
    std::vector<std::string> names;
    names.reserve(parameters.size());

    for (const auto& [key, value] : parameters) {
        (void)value;
        names.push_back(key);
    }

    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::shared_ptr<ParameterSpecification>> Parameterizable::getParameters() const {
    std::vector<std::shared_ptr<ParameterSpecification>> result;
    std::vector<std::string> names = listParameters();
    result.reserve(names.size());

    for (const std::string& name : names) {
        result.push_back(getParameterSpecification(name));
    }

    return result;
}

shared_ptr<ParameterSpecification> Parameterizable::getParameterSpecification(const std::string& name) const {
    auto it = parameters.find(name);
    if (it != parameters.end()) {
        return it->second;
    }
    throw runtime_error("Parameter '" + name + "' is not present on this parameterizable API layer.");
}

BoundParameter Parameterizable::getBoundParameter(PlacedNetwork* placedNetwork, const std::string& name) const {
    return BoundParameter(getParameterSpecification(name), placedNetwork, getParameterizableId());
}

string Parameterizable::listParametersString() const {
    const std::vector<std::string> names = listParameters();

    std::string result;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }

        result += names[i];
    }

    if (result.empty()) {
        result = "<none>";
    }

    return result;
}

json Parameterizable::getParametersArchitectureJson() const {
    json j;
    if (parameters.empty()) {
        j["parameters"] = json::array();
        for (const auto& parameter : getParameters()) {
            if (parameter != nullptr) {
                // Optimizer and initializer serialization is parameter's job.
                j["parameters"].push_back(parameter->architectureJson());
            }
        }
    }
    return j;
}

// Pass j["parameters"] as parameterJson to serialize. j["parameters"] is created by getParametersArchitectureJson().
void Parameterizable::serializeParameters(nlohmann::json& parametersJson,
                                          thor_file::TarWriter& archiveWriter,
                                          Stream stream,
                                          bool saveOptimizerState,
                                          ThorImplementation::StampedNetwork& stampedNetwork,
                                          const string& filenamePrefix) const {
    assert(parametersJson.is_object());

    const uint64_t apiLayerId = getParameterizableId();
    std::shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(apiLayerId);
    std::shared_ptr<ThorImplementation::TrainableLayer> physicalTrainableLayer =
        std::dynamic_pointer_cast<ThorImplementation::TrainableLayer>(physicalLayer);
    assert(physicalTrainableLayer != nullptr);

    for (const auto& [paramName, parameterSpecification] : parameters) {
        json parameterJson = parametersJson.at(paramName);
        assert(parameterJson.is_object());
        parametersJson = BoundParameter::serialize(
            parameterJson, parameterSpecification, archiveWriter, stream, saveOptimizerState, stampedNetwork, filenamePrefix, apiLayerId);
        parametersJson[paramName] = parametersJson;
    }
}

}  // namespace Thor
