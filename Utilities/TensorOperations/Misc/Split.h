#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchSplit(half *dest[],
                 half *source,
                 long numElements,
                 int numDimensions,
                 int numDestArrays,
                 int axisDimension,
                 long axisElementsPerDestArray[],
                 long stridePerSourceDimension[],
                 long stridePerDestDimension[],
                 Stream stream);
