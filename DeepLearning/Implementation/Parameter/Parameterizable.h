#pragma once

#include <memory>
#include <string>
#include <vector>

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

class Parameter;

class Parameterizable {
   public:
    Parameterizable() = default;
    virtual ~Parameterizable() = default;

    void addParameter(const std::shared_ptr<Parameter> &parameter);
    std::shared_ptr<Parameter> getParameter(const std::string &name);
    Tensor getParameterStorage(const std::string &name);
    std::unordered_map<std::string, std::shared_ptr<Parameter>> getParameters();

    std::vector<std::string> listParameters();

   protected:
    std::vector<std::shared_ptr<Parameter>> parameters;
    std::unordered_map<std::string, size_t> parameterIndexByName;
};

}  // namespace ThorImplementation
