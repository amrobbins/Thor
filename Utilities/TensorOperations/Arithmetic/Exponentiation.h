#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename SOURCE_TYPE, typename DEST_TYPE>
void launchExponentiation(SOURCE_TYPE *source_d, DEST_TYPE *dest_d, uint64_t numElements, Stream stream);
