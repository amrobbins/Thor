#pragma once

#include <memory>
#include <string>
#include <vector>

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

class PhysicalParameter;

class Parameterizable {
   public:
    Parameterizable() = default;
    virtual ~Parameterizable() = default;

    void addParameter(const std::shared_ptr<PhysicalParameter> &parameter);
    std::shared_ptr<PhysicalParameter> getParameter(const std::string &name);
    Tensor getParameterStorage(const std::string &name);
    std::unordered_map<std::string, std::shared_ptr<PhysicalParameter>> getParameters();

    std::vector<std::string> listParameters();

   protected:
    std::vector<std::shared_ptr<PhysicalParameter>> parameters{};
    std::unordered_map<std::string, size_t> parameterIndexByName{};
};

}  // namespace ThorImplementation
