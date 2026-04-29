#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

int ScopedGpu::swapActiveDevice(int newDeviceNum) {
    int numGpus;

    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    assert(newDeviceNum < numGpus);

    int previousGpuNum;
    CUDA_CHECK(cudaGetDevice(&previousGpuNum));
    if (newDeviceNum != previousGpuNum) {
        CUDA_CHECK(cudaSetDevice(newDeviceNum));
    }

    return previousGpuNum;
}
