#pragma once

#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorMathFusion/BroadcastStructs.h"
#include "Utilities/TensorMathFusion/CompiledEquation.h"
#include "Utilities/TensorMathFusion/CudaHelpers.h"

namespace ThorImplementation {

class EquationRunner {
   public:
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<Tensor>& inputs,
                    const std::vector<Tensor>& outputs,
                    Stream& stream);
};

}  // namespace ThorImplementation
