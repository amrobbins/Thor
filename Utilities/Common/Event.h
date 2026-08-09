#pragma once

#include "Utilities/Common/ScopedGpu.h"

#include <cstdint>
#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

class Stream;

/**
 * A shared-ownership container for cudaEvent_t.
 *
 * Distinct Event handle objects may be copied, assigned, reset, and destroyed
 * concurrently while referring to the same CUDA event. Concurrent mutation of
 * the same Event handle object requires external synchronization, matching the
 * std::shared_ptr ownership contract documented in SharedOwnership.h.
 *
 * Also carries the gpuNum that the event exists on.
 */
class Event {
   public:
    Event() = default;

    explicit Event(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne = false);
    Event(const Event &event) = default;
    Event(Event &&event) noexcept = default;

    Event &operator=(const Event &other) = default;
    Event &operator=(Event &&other) noexcept = default;

    virtual ~Event();

    void record(Stream stream);

    operator cudaEvent_t();

    cudaEvent_t getEvent();

    int32_t getGpuNum() const;

    bool isInitialized() const;

    // True when the CUDA event was created with cudaEventBlockingSync for a
    // host thread that will call synchronize().
    [[nodiscard]] bool usesBlockingSync() const;

    void synchronize();

    float synchronizeAndReportElapsedTimeInMilliseconds(Event startEvent);

    uint64_t getId() const;

   private:
    struct State;
    std::shared_ptr<State> state;

    void construct(int32_t gpuNum, bool enableTiming, bool expectingHostToWaitOnThisOne);
};
