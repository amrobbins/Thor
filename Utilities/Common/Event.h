#pragma once

#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ScopedGpu.h"

#include <assert.h>

#include <atomic>

#include "cuda.h"
#include "cuda_runtime.h"

class Stream;

/**
 * A reference counted container for cudaEvent_t.
 *
 * Also carries the gpuNum that the event exists on.
 */
class Event : private ReferenceCounted {
   public:
    Event();

    explicit Event(int gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne = false);
    Event(const Event &event);

    Event &operator=(const Event &other);

    virtual ~Event();

    void record(Stream stream);

    operator cudaEvent_t();

    cudaEvent_t getEvent();

    int getGpuNum() const;

    void synchronize();

    float synchronizeAndReportElapsedTimeInMilliseconds(Event startEvent);

    long getReferenceCountedId();

   private:
    int gpuNum;
    cudaEvent_t cudaEvent;

    void construct(int gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne);

    void copyFrom(const Event &other);

    void destroy();
};
