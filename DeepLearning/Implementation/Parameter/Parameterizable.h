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

    void addParam(const std::shared_ptr<Parameter> &parameter);
    std::shared_ptr<Parameter> getParam(const std::string &name);
    Tensor getParamStorage(const std::string &name);

    std::vector<std::string> listParams(const std::string &name);

   private:
    std::vector<std::shared_ptr<Parameter>> parameters;
    std::shared_ptr<Optimizer> optimizer;
};

}  // namespace ThorImplementation
