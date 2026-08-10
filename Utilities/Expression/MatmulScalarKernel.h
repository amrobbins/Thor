#pragma once

#include <cuda_runtime.h>

namespace ThorImplementation {

void launchScaleFp32DeviceScalar(const float* input, float* output, float scale, cudaStream_t stream);
void launchWriteFp32DeviceScalar(float* output, float value, cudaStream_t stream);

}  // namespace ThorImplementation
