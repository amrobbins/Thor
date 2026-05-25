#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

struct CompiledInPlaceRopeTensor {
    uint32_t input_slot = UINT32_MAX;
    std::vector<uint64_t> logical_dims;
    RotaryPositionEmbeddingOptions options;
    DataType dtype = DataType::FP32;
};

struct CompiledInPlaceRope {
    std::vector<CompiledInPlaceRopeTensor> tensors;

    [[nodiscard]] DataType outputDType(size_t output_idx) const {
        if (output_idx >= tensors.size()) {
            throw std::runtime_error("CompiledInPlaceRope output index out of range.");
        }
        return tensors[output_idx].dtype;
    }
};

void runGroupedInPlaceRotaryPositionEmbedding(std::vector<Tensor>& tensors,
                                              const std::vector<RotaryPositionEmbeddingOptions>& options,
                                              const Stream& stream);

}  // namespace ThorImplementation
