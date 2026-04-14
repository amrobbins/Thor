#pragma once

#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CompiledEquation.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {

class EquationRunner {
   public:
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<RuntimeInputValue>& inputs,
                    const std::vector<Tensor>& outputs,
                    Stream& stream);
};

}  // namespace ThorImplementation
