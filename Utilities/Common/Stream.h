#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Event.h"
#include "ScopedGpu.h"
#include <optional>
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ThreadJoinQueue.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cudnn.h>
#include <cublasLt.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#define DEBUG_REF_COUNTS

// The following struct is defined to be a base class of the struct that will hold the args to the function that is enqueued.
// The reason for this is that it declares a virtual method so that I can use a dynamic cast to ensure everything is correct.
// Also on the enqueued thread I can't call any cuda function, so I can't delete a tensor for example, so this base class
// allows the parameters to be deleted on another thread.
struct HostFunctionArgsBase {
    virtual ~HostFunctionArgsBase() {}
};

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

    operator cudaStream_t() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return cudaStream;
    }

    virtual ~Stream() {
        bool shouldDestroy = ReferenceCounted::removeReference();
        if (shouldDestroy)
            destroy();
    }

    Event putEvent(bool enableTiming = false, bool expectingHostToWaitOnThisOne = false) const {
        THOR_THROW_IF_FALSE(!uninitialized());

        ScopedGpu scopedGpu(gpuNum);

        Event event(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
        event.record(*this);

        return event;
    }

    void putEvent(Event& event, bool enableTiming = false, bool expectingHostToWaitOnThisOne = false) const {
        THOR_THROW_IF_FALSE(!uninitialized());

        ScopedGpu scopedGpu(gpuNum);

        if (!event.isInitialized()) {
            event = Event(gpuNum, enableTiming, expectingHostToWaitOnThisOne);
        } else {
            THOR_THROW_IF_FALSE(event.getGpuNum() == gpuNum);
        }
        event.record(*this);
    }

    void waitEvent(Event event) const;

    void synchronize() const;

    static void deviceSynchronize(int gpuNum);

    void enqueueHostFunction(cudaHostFn_t function, std::unique_ptr<HostFunctionArgsBase> &&args);

    cudnnHandle_t getCudnnHandle() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        mtx->lock();
        if (!cudnnHandle->has_value()) {
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
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            numCudnnHandles += 1;
            cudnnStatus = cudnnSetStream(handle, cudaStream);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            *cudnnHandle = handle;
        }
        mtx->unlock();
        return cudnnHandle->value();
    }

    cudaStream_t getStream() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return cudaStream;
    }

    cublasHandle_t getCublasHandle() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        mtx->lock();
        if (!cublasHandle->has_value()) {
            ScopedGpu scopedGpu(gpuNum);
            cublasStatus_t cublasStatus;
            cublasHandle_t handle;
            cublasStatus = cublasCreate(&handle);
            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                printf("cublasStatus %d    gpu:%d   numcublasHandles %d\n", cublasStatus, gpuNum, numCublasHandles);
                fflush(stdout);
            }
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            numCublasHandles += 1;
            cublasStatus = cublasSetStream(handle, cudaStream);
            THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
            *cublasHandle = handle;
        }
        mtx->unlock();
        return cublasHandle->value();
    }

    cublasLtHandle_t getCublasLtHandle() const {
        return getCublasLtHandleUnchecked();
    }

    cublasLtHandle_t getCublasLtHandleUnchecked() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        THOR_THROW_IF_FALSE(cublasLtHandle->has_value());
        return cublasLtHandle->value();
    }

    bool operator==(const Stream &other) const { return cudaStream == other.cudaStream && cudaStream != nullptr; }

    int getGpuNum() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return gpuNum;
    }

    bool isInitialized() const { return !uninitialized(); }

    virtual std::string getObjectName() const { return "Stream"; }

    // It is too late to destroy the cuDNN handle when the destructor of a static string is called,
    // so just don't destroy the cuDNN handle of a static string.
    void informIsStatic() { isStatic = true; }

    uint64_t getId() const { return getReferenceCountedId(); }

    static Stream getNextUploadStream(uint32_t deviceNum);
    static void setMaxNumUploadStreams(uint32_t numGradientUpdateStreams);

    static Stream getNextDownloadStream(uint32_t deviceNum);
    static void setMaxNumDownloadStreams(uint32_t numGradientUpdateStreams);

    void launchCleanUpHostFunctionArgs(std::unique_ptr<HostFunctionArgsBase> &&args);

   private:
    void construct(int gpuNum, Priority priority);

    void copyFrom(const Stream &other) {
        *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

        gpuNum = other.gpuNum;
        cudaStream = other.cudaStream;
        cudnnHandle = other.cudnnHandle;
        cublasHandle = other.cublasHandle;
        cublasLtHandle = other.cublasLtHandle;
        isStatic = other.isStatic;
        mtx = other.mtx;
    }

    void destroy();

   private:
    int gpuNum;
    cudaStream_t cudaStream;
    std::optional<cudnnHandle_t> *cudnnHandle;
    std::optional<cublasHandle_t> *cublasHandle;
    std::optional<cublasLtHandle_t> *cublasLtHandle;

    bool isStatic = false;
    static std::unordered_map<uint32_t, Stream> staticDeviceStreams;

    static int numCudnnHandles;
    static int numCublasHandles;
    static int numCublasLtHandles;

    std::mutex *mtx;
};

/**
 * A lazily allocated, owner-scoped pool for gradient-update streams.
 *
 * Each placed model owns one pool per GPU. All physical stamps for that model
 * on the GPU share the pool. The first three requests allocate distinct
 * streams; subsequent requests reuse those streams
 * round-robin. Keeping the pool owner-scoped prevents optimizer work from one
 * model from adding ordering dependencies to another concurrently executing
 * model.
 */
class GradientUpdateStreamPool {
   public:
    static constexpr uint32_t MAX_STREAMS = 3;

    explicit GradientUpdateStreamPool(uint32_t deviceNum) : deviceNum(deviceNum) {}

    Stream getNext();

    uint32_t getDeviceNum() const { return deviceNum; }
    uint32_t getNumAllocatedStreams() const;

   private:
    const uint32_t deviceNum;
    mutable std::mutex mtx;
    std::deque<Stream> streams;
    uint32_t nextStreamIndex = 0;
};
