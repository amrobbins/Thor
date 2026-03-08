#pragma once

#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/TensorMathFusion/Structs.h"

namespace ThorImplementation {

//  Bind this one to python
class EquationInstance {
   public:
    std::shared_ptr<CompiledEquation> compiled;
    Stream stream;

    Tensor output;
    uint64_t numel = 0;

    EquationInstance(std::shared_ptr<CompiledEquation> compiled, Tensor output, Stream stream, uint64_t numel)
        : compiled(std::move(compiled)), stream(stream), output(output), numel(numel) {}

    void run(const std::vector<Tensor> &inputs);
};

}  // namespace ThorImplementation
