#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

void launchVectorSegmentedReduction(ExprOp op,
                                    const Tensor& values,
                                    const Tensor& segment_offsets,
                                    Tensor& output,
                                    uint64_t elements_per_value,
                                    Stream& stream);

void launchVectorSegmentedReduceMinMaxBackward(ExprOp op,
                                                const Tensor& values,
                                                const Tensor& segment_offsets,
                                                const Tensor& grad_output,
                                                Tensor& grad_input,
                                                uint64_t elements_per_value,
                                                Stream& stream);

}  // namespace ThorImplementation
