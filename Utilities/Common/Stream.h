#pragma once

#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Event.h"
#include "ScopedGpu.h"
#include "Utilities/Common/Optional.h"
#include "Utilities/Common/ReferenceCounted.h"

#include <cudnn.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include <assert.h>
#include <stdio.h>
#include <mutex>

using std::mutex;

#define DEBUG_REF_COUNTS

/**
 * A reference counted container for cudaStream_t.
 *
 * Also carries the gpuNum that the stream exists on.
 */
class Stream : private ReferenceCounted {
   public:
    Stream() : ReferenceCounted() {}

    enum class Priority { HIGH = 3, REGULAR = 4, LOW = 5 };

    explicit Stream(int gpuNum, Priority priority = Priority::REGULAR) { construct(gpuNum, priority); }

    Stream(const Stream &other) {
        // implemented using operator=
        *this = other;
    }

    explicit Stream(TensorPlacement placement, Priority priority = Priority::REGULAR) {
        int gpuNum = placement.getMemDevice() == TensorPlacement::MemDevices::GPU ? placement.getDeviceNum() : 0;
        construct(gpuNum, priority);
    }

    Stream &operator=(const Stream &other) {
        copyFrom(other);

        return *this;
    }

    operator cudaStream_t() {
        assert(!uninitialized());
        return cudaStream;
    }

    virtual ~Stream() {
        bool shouldDestroy = ReferenceCounted::removeReference();
        if (shouldDestroy)
            destroy();
    }

    Event putEvent(bool enableTiming = false) {
        assert(!uninitialized());

        ScopedGpu scopedGpu(gpuNum);

        Event event(gpuNum, enableTiming);
        cudaError_t cudaStatus = cudaEventRecord(event.getEvent(), cudaStream);
        assert(cudaStatus == cudaSuccess);

        return event;
    }

    void waitEvent(Event event) {
        assert(!uninitialized());

        ScopedGpu scopedGpu(gpuNum);

        cudaError_t cudaStatus = cudaStreamWaitEvent(cudaStream, event.getEvent(), 0);
        assert(cudaStatus == cudaSuccess);
    }

    void synchronize() {
        assert(!uninitialized());

        cudaError_t cudaStatus = cudaStreamSynchronize(cudaStream);
        if (cudaStatus != cudaSuccess) {
            printf("cuda error on stream synchronize. cudaStatus = %d\n", cudaStatus);
            fflush(stdout);
        }
        assert(cudaStatus == cudaSuccess);
    }

    cudaStream_t getStream() {
        assert(!uninitialized());
        return cudaStream;
    }

    cudnnHandle_t getCudnnHandle() {
        assert(!uninitialized());
        mtx->lock();
        if (cudnnHandle->isEmpty()) {
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
            *cudnnHandle = handle;
        }
        mtx->unlock();
        return *cudnnHandle;
    }

    int getGpuNum() const {
        assert(!uninitialized());
        return gpuNum;
    }

    bool isInitialized() { return !uninitialized(); }

    virtual string getObjectName() { return "Stream"; }

   private:
    void construct(int gpuNum, Priority priority) {
        ReferenceCounted::initialize();

        cudnnHandle = new Optional<cudnnHandle_t>;
        mtx = new mutex;

        ScopedGpu scopedGpu(gpuNum);
        cudaError_t cudaStatus;
        this->gpuNum = gpuNum;

        // greatestPriority is given the highest priority in terms of execution, and its numerical value is the minimum of the allowed
        // range.
        int leastPriority, greatestPriority;
        cudaStatus = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        assert(cudaStatus == cudaSuccess);
        int priorityValue;
        if (priority == Priority::HIGH)
            priorityValue = greatestPriority;
        else if (priority == Priority::REGULAR)
            priorityValue = greatestPriority + 1;
        else
            priorityValue = greatestPriority + 2;

        cudaStatus = cudaStreamCreateWithPriority(&cudaStream, cudaStreamNonBlocking, priorityValue);
        assert(cudaStatus == cudaSuccess);
    }

    void copyFrom(const Stream &other) {
        *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

        gpuNum = other.gpuNum;
        cudaStream = other.cudaStream;
        cudnnHandle = other.cudnnHandle;
        mtx = other.mtx;
    }

    void destroy() {
        ScopedGpu scopedGpu(gpuNum);

        if (cudnnHandle->isPresent()) {
            cudnnStatus_t cudnnStatus;
            cudnnStatus = cudnnDestroy(*cudnnHandle);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }

        cudaError_t cudaStatus;
        cudaStatus = cudaStreamDestroy(cudaStream);
        assert(cudaStatus == cudaSuccess);

        delete cudnnHandle;
        cudnnHandle = nullptr;
        delete mtx;
        mtx = nullptr;
    }

   private:
    int gpuNum;
    cudaStream_t cudaStream;
    Optional<cudnnHandle_t> *cudnnHandle;

    mutex *mtx;
};
