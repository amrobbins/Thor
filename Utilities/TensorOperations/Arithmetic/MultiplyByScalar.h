#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename SOURCE_TYPE, typename SCALAR_TYPE, typename DEST_TYPE>
void launchMultiplyByScalar(
    SOURCE_TYPE *source_d, SCALAR_TYPE *batchedScalars_d, DEST_TYPE *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
