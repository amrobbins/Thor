#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/Parameter.h"

using namespace std;

namespace ThorImplementation {

void Parameterizable::addParam(const shared_ptr<Parameter> &parameter) {
    string paramName = parameter->getName();
    for (const auto &param : parameters) {
        if (param->getName() == paramName)
            throw runtime_error("Parameter named " + paramName + "added more than once.");
    }
    parameters.push_back(parameter);
}

shared_ptr<Parameter> Parameterizable::getParam(const string &name) {
    for (const auto &param : parameters) {
        if (param->getName() == name)
            return param;
    }
    assert(false);
}

Tensor Parameterizable::getParamStorage(const string &name) {
    for (const auto &param : parameters) {
        if (param->getName() == name)
            return param->getStorage();
    }
    assert(false);
}

vector<string> Parameterizable::listParams(const string &name) {
    vector<string> names;
    for (const auto &param : parameters) {
        names.push_back(param->getName());
    }
    return names;
}

}  // namespace ThorImplementation
