#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"

#include <stdexcept>

using namespace std;

namespace Thor {

shared_ptr<ParameterSpecification> Parameterizable::getParameter(const std::string& name) const {
    for (const auto& parameter : getParameters()) {
        if (parameter != nullptr && parameter->getName() == name)
            return parameter;
    }

    throw runtime_error("Parameter '" + name + "' is not present on this parameterizable API layer.");
}

BoundParameter Parameterizable::getBoundParameter(PlacedNetwork& placedNetwork, const std::string& name) const {
    return BoundParameter(getParameter(name), &placedNetwork, getParameterizableId());
}

vector<BoundParameter> Parameterizable::getBoundParameters(PlacedNetwork& placedNetwork) const {
    vector<BoundParameter> boundParameters;
    boundParameters.reserve(getParameters().size());

    for (const auto& parameter : getParameters()) {
        if (parameter == nullptr)
            throw runtime_error("Parameterizable API layer contained a null Parameter.");
        boundParameters.emplace_back(parameter, &placedNetwork, getParameterizableId());
    }

    return boundParameters;
}

}  // namespace Thor
