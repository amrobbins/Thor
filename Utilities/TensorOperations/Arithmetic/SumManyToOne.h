#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename SOURCE_TYPE, typename DEST_TYPE>
void launchSumManyToOne(
    SOURCE_TYPE *source_d, DEST_TYPE *dest_d, uint32_t numElementsPerBatch, uint32_t batchSize, bool invert, Stream stream);
