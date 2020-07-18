#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchConcatenate(half *dest,
                       half *source[],
                       long numElements,
                       int numDimensions,
                       int numSourceArrays,
                       int axisDimension,
                       long axisElementsPerSourceArray[],
                       long stridePerDestDimension[],
                       long stridePerSourceDimension[],
                       Stream stream);
