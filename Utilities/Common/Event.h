#pragma once

#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ScopedGpu.h"

#include <assert.h>

#include <atomic>

#include "cuda.h"
#include "cuda_runtime.h"

using std::atomic;

/**
 * A reference counted container for cudaEvent_t.
 *
 * Also carries the gpuNum that the event exists on.
 */
class Event : private ReferenceCounted {
   public:
    Event() : ReferenceCounted() {}

    explicit Event(int gpuNum, bool enableTiming) { construct(gpuNum, enableTiming); }

    Event(const Event &event) {
        // implemented using operator=
        *this = event;
    }

    Event &operator=(const Event &other) {
        copyFrom(other);
        return *this;
    }

    virtual ~Event() {
        bool shouldDestroy = ReferenceCounted::removeReference();
        if (shouldDestroy)
            destroy();
    }

    operator cudaEvent_t() {
        assert(!uninitialized());
        return cudaEvent;
    }

    cudaEvent_t getEvent() {
        assert(!uninitialized());
        return cudaEvent;
    }

    int getGpuNum() const {
        assert(!uninitialized());
        return gpuNum;
    }

    void synchronize() {
        assert(!uninitialized());

        cudaError_t cudaStatus;
        cudaStatus = cudaEventSynchronize(*this);
        assert(cudaStatus == cudaSuccess);
    }

    float synchronizeAndReportElapsedTimeInMilliseconds(Event startEvent) {
        assert(!uninitialized());

        float milliseconds;

        synchronize();
        cudaError_t cudaStatus;
        cudaStatus = cudaEventElapsedTime(&milliseconds, startEvent, *this);
        assert(cudaStatus == cudaSuccess);
        return milliseconds;
    }

   private:
    int gpuNum;
    cudaEvent_t cudaEvent;

    void construct(int gpuNum, bool enableTiming) {
        ReferenceCounted::initialize();

        ScopedGpu scopedGpu(gpuNum);

        cudaError_t cudaStatus;
        this->gpuNum = gpuNum;

        cudaStatus = cudaEventCreateWithFlags(&cudaEvent, enableTiming ? 0 : cudaEventDisableTiming);
        assert(cudaStatus == cudaSuccess);
    }

    void copyFrom(const Event &other) {
        *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

        gpuNum = other.gpuNum;
        cudaEvent = other.cudaEvent;
    }

    void destroy() {
        ScopedGpu scopedGpu(gpuNum);
        cudaError_t cudaStatus;
        cudaStatus = cudaEventDestroy(cudaEvent);
        assert(cudaStatus == cudaSuccess);
    }
};
