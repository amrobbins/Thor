#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "DeepLearning/Implementation/ThorError.h"

int ScopedGpu::swapActiveDevice(int newDeviceNum) {
    int numGpus;

    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    THOR_THROW_IF_FALSE(newDeviceNum < numGpus);

    int previousGpuNum;
    CUDA_CHECK(cudaGetDevice(&previousGpuNum));
    if (newDeviceNum != previousGpuNum) {
        CUDA_CHECK(cudaSetDevice(newDeviceNum));
    }

    return previousGpuNum;
}
