#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename SOURCE0_TYPE, typename SOURCE1_TYPE, typename DEST_TYPE>
void launchElementwiseSubtract(SOURCE0_TYPE *source0_d, SOURCE1_TYPE *source1_d, DEST_TYPE *dest_d, uint64_t numElements, Stream stream);
