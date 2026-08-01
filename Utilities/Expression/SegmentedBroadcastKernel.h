#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

namespace ThorImplementation {

void launchSegmentedBroadcast(const Tensor& per_segment_values,
                              const Tensor& segment_offsets,
                              Tensor& output,
                              bool normalize_by_segment_length,
                              Stream& stream);

}  // namespace ThorImplementation
