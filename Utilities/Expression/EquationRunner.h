#pragma once

#include <optional>

#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CompiledEquation.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {

class EquationRunner {
   public:
    // numElements is the element count the flat kernel covers: by default the largest output's. A padded ragged
    // pointwise stage gives its selected width's B*C*W, over outputs that are its reserved storage.
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<RuntimeInputValue>& inputs,
                    const std::vector<Tensor>& outputs,
                    Stream& stream,
                    std::optional<uint64_t> numElements = std::nullopt);
};

}  // namespace ThorImplementation
