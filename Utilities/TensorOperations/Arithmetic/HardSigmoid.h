#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchHardSigmoid(half *featureOut_d, half *featureIn_d, int numElements, Stream stream);

void launchHardSigmoidBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream);
