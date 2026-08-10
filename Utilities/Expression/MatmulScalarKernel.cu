#include "Utilities/Expression/MatmulScalarKernel.h"

#include <stdexcept>

namespace ThorImplementation {
namespace {

__global__ void scaleFp32DeviceScalarKernel(const float* input, float* output, float scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = input[0] * scale;
    }
}

__global__ void writeFp32DeviceScalarKernel(float* output, float value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = value;
    }
}

}  // namespace

void launchScaleFp32DeviceScalar(const float* input, float* output, float scale, cudaStream_t stream) {
    if (input == nullptr || output == nullptr) {
        throw std::runtime_error("launchScaleFp32DeviceScalar received null pointer.");
    }
    scaleFp32DeviceScalarKernel<<<1, 1, 0, stream>>>(input, output, scale);
}

void launchWriteFp32DeviceScalar(float* output, float value, cudaStream_t stream) {
    if (output == nullptr) {
        throw std::runtime_error("launchWriteFp32DeviceScalar received null output pointer.");
    }
    writeFp32DeviceScalarKernel<<<1, 1, 0, stream>>>(output, value);
}

}  // namespace ThorImplementation
