#pragma once

#include "Utilities/Common/Stream.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

void launchPad(void *dest_d,
               void *source_d,
               unsigned long numDestElements,
               unsigned int numDimensions,
               unsigned long stridePerPaddedDimension_d[],
               unsigned long stridePerUnpaddedDimension_d[],
               unsigned int padBefore_d[],
               unsigned int padAfter_d[],
               ThorImplementation::DataType dataType,
               Stream stream);
