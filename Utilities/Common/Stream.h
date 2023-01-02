#pragma once

#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Event.h"
#include "ScopedGpu.h"
#include "Utilities/Common/Optional.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cudnn.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include <assert.h>
#include <stdio.h>
#include <mutex>
#include <unordered_map>

#define DEBUG_REF_COUNTS

/**
 * A reference counted container for cudaStream_t.
 *
 * Also carries the gpuNum that the stream exists on.
 */
class Stream : private ReferenceCounted {
   public:
    Stream() : ReferenceCounted() { isStatic = false; }

    enum class Priority { HIGH = 3, REGULAR = 4, LOW = 5 };

    explicit Stream(int gpuNum, Priority priority = Priority::REGULAR) { construct(gpuNum, priority); }

    Stream(const Stream &other) {
        // implemented using operator=
        *this = other;
    }

    explicit Stream(ThorImplementation::TensorPlacement placement, Priority priority = Priority::REGULAR) {
        int gpuNum = placement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU ? placement.getDeviceNum() : 0;
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

    Event putEvent(bool enableTiming = false, bool expectingHostToWaitOnThisOne = false) {
        assert(!uninitialized());

        ScopedGpu scopedGpu(gpuNum);

        Event event(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
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

    static void deviceSynchronize(int gpuNum) {
        ScopedGpu scopedGpu(gpuNum);
        cudaError_t cudaStatus = cudaDeviceSynchronize();
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
                printf("cudnnStatus %d : %s   gpu:%d   numCudnnHandles %d\n",
                       cudnnStatus,
                       cudnnGetErrorString(cudnnStatus),
                       gpuNum,
                       numCudnnHandles);
                fflush(stdout);
            }
            numCudnnHandles += 1;
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

    virtual std::string getObjectName() { return "Stream"; }

    // It is too late to destroy the cuDNN handle when the destructor of a static string is called,
    // so just don't destroy the cuDNN handle of a static string.
    void informIsStatic() { isStatic = true; }

    uint64_t getId() { return getReferenceCountedId(); }

    // For cases where you just need any stream (for example computing workspace sizes for cudnn) use the static stream to
    // avoid allocating too many streams and crashing Cuda.
    static Stream getStaticStream(uint32_t deviceNum);

   private:
    void construct(int gpuNum, Priority priority) {
        ReferenceCounted::initialize();

        cudnnHandle = new Optional<cudnnHandle_t>;
        mtx = new std::mutex;

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
        // If this is a static stream, it is too late to free cuda resources, or interact with cuda, when it is destroyed.
        // Doesn't much matter as the program is exiting anyway.
        if (isStatic) {
            ScopedGpu scopedGpu(gpuNum);

            // can't destroy the cudnn handle at the point when the static string is destroyed
            if (cudnnHandle->isPresent()) {
                numCudnnHandles -= 1;

                cudnnStatus_t cudnnStatus;
                cudnnStatus = cudnnDestroy(*cudnnHandle);
                assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            }

            cudaError_t cudaStatus;
            cudaStatus = cudaStreamDestroy(cudaStream);
            assert(cudaStatus == cudaSuccess);

            delete cudnnHandle;
            cudnnHandle = nullptr;
        }
        delete mtx;
        mtx = nullptr;
    }

   private:
    int gpuNum;
    cudaStream_t cudaStream;
    Optional<cudnnHandle_t> *cudnnHandle;

    bool isStatic;
    static std::unordered_map<uint32_t, Stream> staticDeviceStreams;

    static int numCudnnHandles;

    std::mutex *mtx;
};
