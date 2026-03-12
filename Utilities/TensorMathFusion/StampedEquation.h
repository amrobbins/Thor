#pragma once

#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/TensorMathFusion/CompiledEquation.h"

namespace ThorImplementation {

class StampedEquation {
   public:
    StampedEquation(std::shared_ptr<CompiledEquation> compiledEquation,
                    const std::vector<Tensor>& inputs,
                    const Tensor& output,
                    const Stream& stream)
        : compiledEquation(std::move(compiledEquation)), inputs(inputs), output(output), stream(stream) {}

    void run();
    Tensor getOutputTensor() const { return output; }

   private:
    std::shared_ptr<CompiledEquation> compiledEquation;
    std::vector<Tensor> inputs;
    Tensor output;
    Stream stream;
};

}  // namespace ThorImplementation
