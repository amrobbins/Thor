#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchRelu(half *dest_d, half *source_d, int numElements, Stream stream);
