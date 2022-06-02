#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchElu(half *featureOut_d, half *featureIn_d, int numElements, float alpha, Stream stream);

void launchEluBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, float alpha, Stream stream);
