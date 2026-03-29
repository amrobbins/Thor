#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/Parameter.h"

using namespace std;

namespace ThorImplementation {

void Parameterizable::addParam(const shared_ptr<Parameter> &parameter) {
    if (parameter == nullptr)
        throw std::runtime_error("Null parameter encountered.");
    string paramName = parameter->getName();
    if (parameterIndexByName.contains(paramName))
        throw std::runtime_error("Parameter " + paramName + " listed more than once in parameterizable layer.");
    parameterIndexByName[paramName] = parameters.size();
    parameters.push_back(parameter);
}

shared_ptr<Parameter> Parameterizable::getParam(const string &name) {
    auto it = parameterIndexByName.find(name);
    if (it == parameterIndexByName.end())
        throw std::runtime_error("Parameter " + name + " is not a parameter of the parameterizable layer.");
    return parameters[it->second];
}

Tensor Parameterizable::getParamStorage(const string &name) {
    auto it = parameterIndexByName.find(name);
    if (it == parameterIndexByName.end())
        throw std::runtime_error("Parameter " + name + " is not a parameter of the parameterizable layer.");
    return parameters[it->second]->getStorage();
}

vector<string> Parameterizable::listParams() {
    vector<string> names;
    for (const auto &param : parameters) {
        names.push_back(param->getName());
    }
    return names;
}

}  // namespace ThorImplementation
