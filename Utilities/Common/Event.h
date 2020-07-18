#pragma once

#include "ScopedGpu.h"

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
class Event {
   public:
    Event() { uninitialized = true; }

    explicit Event(int gpuNum) {
        uninitialized = false;

        ScopedGpu scopedGpu(gpuNum);

        cudaError_t cudaStatus;
        this->gpuNum = gpuNum;

        cudaStatus = cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming);
        assert(cudaStatus == cudaSuccess);

        referenceCount = new atomic<int>(1);
    }

    Event(const Event &event) {
        *this = event;  // implemented using operator=
    }

    Event &operator=(const Event &event) {
        uninitialized = event.uninitialized;
        if (uninitialized)
            return *this;

        referenceCount = event.referenceCount;
        referenceCount->fetch_add(1);

        gpuNum = event.gpuNum;
        cudaEvent = event.cudaEvent;

        return *this;
    }

    ~Event() {
        if (uninitialized)
            return;

        int refCountBeforeDecrement = referenceCount->fetch_sub(1);
        if (refCountBeforeDecrement == 1) {
            delete referenceCount;
            referenceCount = nullptr;

            ScopedGpu scopedGpu(gpuNum);
            cudaError_t cudaStatus;
            cudaStatus = cudaEventDestroy(cudaEvent);
            assert(cudaStatus == cudaSuccess);
        }
    }

    operator cudaEvent_t() {
        assert(!uninitialized);
        return cudaEvent;
    }

    cudaEvent_t getEvent() {
        assert(!uninitialized);
        return cudaEvent;
    }

    int getGpuNum() const {
        assert(!uninitialized);
        return gpuNum;
    }

   private:
    int gpuNum;
    bool uninitialized = false;
    cudaEvent_t cudaEvent;

    atomic<int> *referenceCount;
};
