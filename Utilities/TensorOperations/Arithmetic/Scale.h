#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename SOURCE_TYPE, typename SCALAR_TYPE, typename DEST_TYPE>
void launchScale(SOURCE_TYPE *source_d, SCALAR_TYPE scalar_h, DEST_TYPE *dest_d, uint64_t numElements, Stream stream);
