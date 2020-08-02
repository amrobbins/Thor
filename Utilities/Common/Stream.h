#pragma once

#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Event.h"
#include "ScopedGpu.h"
#include "Utilities/Common/Optional.h"

#include <assert.h>
#include <stdio.h>

#include <cudnn.h>
#include "cuda.h"
#include "cuda_runtime.h"

/**
 * A reference counted container for cudaStream_t.
 *
 * Also carries the gpuNum that the stream exists on.
 */
class Stream {
   public:
    Stream() { uninitialized = true; }

    explicit Stream(int gpuNum) {
        uninitialized = false;

        ScopedGpu scopedGpu(gpuNum);
        cudaError_t cudaStatus;
        this->gpuNum = gpuNum;

        cudaStatus = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
        assert(cudaStatus == cudaSuccess);

        referenceCount = new atomic<int>(1);
    }

    explicit Stream(TensorPlacement placement) {
        uninitialized = false;

        int gpuNum = placement.getMemDevice() == TensorPlacement::MemDevices::GPU ? placement.getDeviceNum() : 0;

        ScopedGpu scopedGpu(gpuNum);
        cudaError_t cudaStatus;
        this->gpuNum = gpuNum;

        cudaStatus = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
        assert(cudaStatus == cudaSuccess);

        referenceCount = new atomic<int>(1);
    }

    Stream(const Stream &stream) {
        *this = stream;  // implemented using operator=
    }

    Stream &operator=(const Stream &stream) {
        uninitialized = stream.uninitialized;
        if (uninitialized)
            return *this;

        referenceCount = stream.referenceCount;
        referenceCount->fetch_add(1);

        gpuNum = stream.gpuNum;
        cudaStream = stream.cudaStream;

        return *this;
    }

    operator cudaStream_t() {
        assert(!uninitialized);
        return cudaStream;
    }

    ~Stream() {
        if (uninitialized)
            return;

        int refCountBeforeDecrement = referenceCount->fetch_sub(1);
        if (refCountBeforeDecrement == 1) {
            delete referenceCount;
            referenceCount = nullptr;

            ScopedGpu scopedGpu(gpuNum);

            if (cudnnHandle.isPresent()) {
                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnDestroy(cudnnHandle);
                assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            }

            cudaError_t cudaStatus;
            cudaStatus = cudaStreamDestroy(cudaStream);
            assert(cudaStatus == cudaSuccess);
        }
    }

    Event putEvent(bool enableTiming = false) {
        assert(!uninitialized);

        ScopedGpu scopedGpu(gpuNum);

        Event event(gpuNum, enableTiming);
        cudaError_t cudaStatus = cudaEventRecord(event.getEvent(), cudaStream);
        assert(cudaStatus == cudaSuccess);

        return event;
    }

    void waitEvent(Event event) {
        assert(!uninitialized);

        ScopedGpu scopedGpu(gpuNum);

        cudaError_t cudaStatus = cudaStreamWaitEvent(cudaStream, event.getEvent(), 0);
        assert(cudaStatus == cudaSuccess);
    }

    void synchronize() {
        assert(!uninitialized);

        cudaError_t cudaStatus = cudaStreamSynchronize(cudaStream);
        if (cudaStatus != cudaSuccess) {
            printf("cuda error on stream synchronize. cudaStatus = %d\n", cudaStatus);
            fflush(stdout);
        }
        assert(cudaStatus == cudaSuccess);
    }

    cudaStream_t getStream() {
        assert(!uninitialized);
        return cudaStream;
    }

    cudnnHandle_t getCudnnHandle() {
        assert(!uninitialized);
        if (cudnnHandle.isEmpty()) {
            ScopedGpu scopedGpu(gpuNum);
            cudnnStatus_t cudnnStatus;
            cudnnHandle_t handle;
            cudnnStatus = cudnnCreate(&handle);
            if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
                printf("cudnnStatus %d : %s   gpu:%d\n", cudnnStatus, cudnnGetErrorString(cudnnStatus), gpuNum);
                fflush(stdout);
            }
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnStatus = cudnnSetStream(handle, cudaStream);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnHandle = handle;
        }
        return cudnnHandle;
    }

    int getGpuNum() const {
        assert(!uninitialized);
        return gpuNum;
    }

    bool isInitialized() { return !uninitialized; }

   private:
    int gpuNum;
    cudaStream_t cudaStream;
    Optional<cudnnHandle_t> cudnnHandle;

    bool uninitialized;
    atomic<int> *referenceCount;
};
