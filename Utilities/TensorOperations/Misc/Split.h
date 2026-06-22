#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cstddef>
#include <cstdint>

void launchSplit(void *dest[],
                 void *source,
                 std::size_t elementSizeBytes,
                 long numElements,
                 int numDimensions,
                 int numDestArrays,
                 int axisDimension,
                 long axisElementsPerDestArray[],
                 long stridePerSourceDimension[],
                 long stridePerDestDimension[],
                 Stream stream);
