#pragma once

#include <assert.h>

#include "cuda.h"
#include "cuda_runtime.h"

class ScopedGpu {
   public:
    ScopedGpu(int gpuNum) { setGpuNum(gpuNum); }

    ScopedGpu() = delete;
    ScopedGpu(const ScopedGpu&) = delete;
    ScopedGpu& operator=(const ScopedGpu&) = delete;

    virtual ~ScopedGpu() {
        if (gpuNum != previousGpuNum)
            swapActiveDevice(previousGpuNum);
    }

   private:
    int gpuNum;
    int previousGpuNum;

    void setGpuNum(int gpuNum) {
        this->gpuNum = gpuNum;
        previousGpuNum = swapActiveDevice(gpuNum);
    }

    static int swapActiveDevice(int newDeviceNum) {
        int numGpus;
        cudaError_t cudaStatus;

        cudaStatus = cudaGetDeviceCount(&numGpus);
        assert(cudaStatus == cudaSuccess);
        assert(newDeviceNum < numGpus);

        int previousGpuNum;
        cudaStatus = cudaGetDevice(&previousGpuNum);
        assert(cudaStatus == cudaSuccess);
        if (newDeviceNum != previousGpuNum) {
            cudaStatus = cudaSetDevice(newDeviceNum);
            assert(cudaStatus == cudaSuccess);
        }

        return previousGpuNum;
    }
};
