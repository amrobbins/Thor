#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"

#include <stdexcept>

using namespace std;

namespace Thor {

void Parameterizable::addParameter(const std::shared_ptr<ParameterSpecification>& parameter) {
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
        names.push_back(key);
    }

    std::sort(names.begin(), names.end());
    return names;
}

shared_ptr<ParameterSpecification> Parameterizable::getParameterSpecification(const std::string& name) const {
    auto it = parameters.find(name);
    if (it != parameters.end()) {
        return it->second;
    }
    throw runtime_error("Parameter '" + name + "' is not present on this parameterizable API layer.");
}

BoundParameter Parameterizable::getBoundParameter(PlacedNetwork& placedNetwork, const std::string& name) const {
    return BoundParameter(getParameterSpecification(name), &placedNetwork, getOwningLayerId());
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

    return result;
}

}  // namespace Thor
