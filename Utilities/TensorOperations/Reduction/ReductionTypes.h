#pragma once

// The operations of the public reducers (ReductionOp, ArgReductionOp) and the ARG output options.

#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cstdint>
#include <optional>

namespace ThorImplementation {

enum class ReductionOp : uint8_t {
    Sum = 0,
    Min = 1,
    Max = 2,
    Product = 3,
    Mean = 4,
    L1Norm = 5,
    L2Norm = 6,
    SumSquares = 7,
};

enum class ArgReductionOp : uint8_t {
    ArgMin = 0,
    ArgMax = 1,
};

struct ArgReductionOutputOptions {
    bool produce_value = true;
    bool produce_index = true;
    std::optional<DataType> value_output_dtype = std::nullopt;
    DataType index_output_dtype = DataType::UINT32;
};

}  // namespace ThorImplementation
