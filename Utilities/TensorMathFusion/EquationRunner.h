#pragma once

#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorMathFusion/CompiledEquation.h"
#include "Utilities/TensorMathFusion/CudaHelpers.h"

namespace ThorImplementation {

class EquationRunner {
   public:
    // FIXME: Eventually: This is hard coded to FP32
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<Tensor>& inputs,
                    Tensor& output,
                    Stream& stream);
};

}  // namespace ThorImplementation
