#include "Utilities/TensorMathFusion/MatmulScalarKernel.h"

#include <stdexcept>

namespace ThorImplementation {
namespace {

__global__ void scaleFp32DeviceScalarKernel(const float* input, float* output, float scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = input[0] * scale;
    }
}

}  // namespace

void launchScaleFp32DeviceScalar(const float* input, float* output, float scale, cudaStream_t stream) {
    if (input == nullptr || output == nullptr) {
        throw std::runtime_error("launchScaleFp32DeviceScalar received null pointer.");
    }
    scaleFp32DeviceScalarKernel<<<1, 1, 0, stream>>>(input, output, scale);
}

}  // namespace ThorImplementation
