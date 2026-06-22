#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cstddef>
#include <cstdint>

void launchConcatenate(void *dest,
                       void *source[],
                       std::size_t elementSizeBytes,
                       long numElements,
                       int numDimensions,
                       int numSourceArrays,
                       int axisDimension,
                       long axisElementsPerSourceArray[],
                       long stridePerDestDimension[],
                       long stridePerSourceDimension[],
                       Stream stream);
