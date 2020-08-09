#pragma once

#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Event.h"
#include "ScopedGpu.h"
#include "Utilities/Common/Optional.h"

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
class Stream {
   public:
    Stream() {
        uninitialized = true;
        referenceCount = nullptr;
    }

    explicit Stream(int gpuNum) { construct(gpuNum); }

    explicit Stream(TensorPlacement placement) {
        int gpuNum = placement.getMemDevice() == TensorPlacement::MemDevices::GPU ? placement.getDeviceNum() : 0;
        construct(gpuNum);
    }

    void construct(int gpuNum) {
#ifdef DEBUG_REF_COUNTS
        streamsCreated.fetch_add(1);
#endif

        uninitialized = false;

        cudnnHandle = new Optional<cudnnHandle_t>;
        mtx = new mutex;

        ScopedGpu scopedGpu(gpuNum);
        cudaError_t cudaStatus;
        this->gpuNum = gpuNum;

        cudaStatus = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
        assert(cudaStatus == cudaSuccess);

        referenceCount = new atomic<int>(1);
    }

    Stream(const Stream &stream) {
        uninitialized = true;
        referenceCount = nullptr;

        *this = stream;  // implemented using operator=
    }

    Stream &operator=(const Stream &other) {
        // Do not reorder the increment/decrement of refCount here or object may be destroyed prematurely
        if (!other.uninitialized) {
            // other stream is initialized
            other.referenceCount->fetch_add(1);
            if (!uninitialized) {
                // this stream was previously initialized
                removeReference();
            }
            uninitialized = false;
            referenceCount = other.referenceCount;

            gpuNum = other.gpuNum;
            cudaStream = other.cudaStream;
            cudnnHandle = other.cudnnHandle;
            mtx = other.mtx;

            return *this;
        } else {
            // other stream is not initialized
            if (!uninitialized) {
                // this stream was previously initialized
                removeReference();
            }
            uninitialized = true;
            referenceCount = nullptr;
            return *this;
        }
    }

    operator cudaStream_t() {
        assert(!uninitialized);
        return cudaStream;
    }

    virtual ~Stream() { removeReference(); }

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
        mtx->lock();
        if (cudnnHandle->isEmpty()) {
#ifdef DEBUG_REF_COUNTS
            cudnnHandlesCreated.fetch_add(1);
#endif

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
        assert(!uninitialized);
        return gpuNum;
    }

    bool isInitialized() { return !uninitialized; }

   private:
    int gpuNum;
    cudaStream_t cudaStream;
    Optional<cudnnHandle_t> *cudnnHandle;

    bool uninitialized;
    atomic<int> *referenceCount;

    mutex *mtx;

    void removeReference() {
        if (uninitialized) {
            assert(referenceCount == nullptr);
            return;
        }

        int refCountBeforeDecrement = referenceCount->fetch_sub(1);

        if (refCountBeforeDecrement == 1) {
            delete referenceCount;
            referenceCount = nullptr;

#ifdef DEBUG_REF_COUNTS
            streamsDestroyed.fetch_add(1);
#endif

            ScopedGpu scopedGpu(gpuNum);

            if (cudnnHandle->isPresent()) {
#ifdef DEBUG_REF_COUNTS
                cudnnHandlesDestroyed.fetch_add(1);
#endif

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
    }

#ifdef DEBUG_REF_COUNTS
    static atomic<int> streamsCreated;
    static atomic<int> streamsDestroyed;
    static atomic<int> cudnnHandlesCreated;
    static atomic<int> cudnnHandlesDestroyed;

    class RefCountChecker {
       public:
        virtual ~RefCountChecker() {
            printf("streams created %d streams destroyed %d\n", streamsCreated.fetch_add(0), streamsDestroyed.fetch_add(0));
            printf("cudnnHandles created %d cudnnHandles destroyed %d\n",
                   cudnnHandlesCreated.fetch_add(0),
                   cudnnHandlesDestroyed.fetch_add(0));
            fflush(stdout);
        }
    };
    static RefCountChecker refCountChecker;
#endif
};
