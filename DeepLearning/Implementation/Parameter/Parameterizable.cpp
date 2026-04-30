#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

using namespace std;

namespace ThorImplementation {

void Parameterizable::addParameter(const shared_ptr<PhysicalParameter> &parameter) {
    if (parameter == nullptr)
        throw runtime_error("Null parameter encountered.");
    string paramName = parameter->getName();
    if (parameterIndexByName.contains(paramName))
        throw runtime_error("Parameter " + paramName + " listed more than once in parameterizable layer.");
    parameterIndexByName[paramName] = parameters.size();
    parameters.push_back(parameter);
}

bool Parameterizable::hasParameter(const std::string &name) { return parameterIndexByName.contains(name); }

shared_ptr<PhysicalParameter> Parameterizable::getParameter(const string &name) {
    auto it = parameterIndexByName.find(name);
    if (it == parameterIndexByName.end())
        throw runtime_error("Parameter " + name + " is not a parameter of the parameterizable layer.");
    return parameters[it->second];
}

void Parameterizable::dropParameter(const std::string &name) {
    auto it = parameterIndexByName.find(name);
    if (it == parameterIndexByName.end()) {
        return;
    }

    uint32_t oldPos = it->second;
    uint32_t lastPos = static_cast<uint32_t>(parameters.size() - 1);

    if (oldPos != lastPos) {
        parameters[oldPos] = std::move(parameters[lastPos]);
        parameterIndexByName[parameters[oldPos]->getName()] = oldPos;
    }

    parameters.pop_back();
    parameterIndexByName.erase(name);
}

Tensor Parameterizable::getParameterStorage(const string &name) {
    auto it = parameterIndexByName.find(name);
    if (it == parameterIndexByName.end())
        throw runtime_error("Parameter " + name + " is not a parameter of the parameterizable layer.");
    return parameters[it->second]->getStorage();
}

unordered_map<string, shared_ptr<PhysicalParameter>> Parameterizable::getParameters() {
    std::unordered_map<std::string, std::shared_ptr<PhysicalParameter>> allParams;
    for (const auto &parameter : parameters)
        allParams[parameter->getName()] = parameter;
    return allParams;
}

vector<string> Parameterizable::listParameters() {
    vector<string> names;
    for (const auto &param : parameters) {
        names.push_back(param->getName());
    }
    return names;
}

}  // namespace ThorImplementation
