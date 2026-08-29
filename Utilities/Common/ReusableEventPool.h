#pragma once

#include "Utilities/Common/Event.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ThorImplementation {

/**
 * Reuses CUDA events for dynamic, submission-local dependency graphs.
 *
 * Fixed recurring dependency edges should normally keep an Event member on the
 * object that owns the edge and use Stream::putEvent(Event&) / Stream::waitFor.
 * This pool is for cases where the number of dependency events is determined
 * dynamically for one host submission.
 *
 * An event may be returned to the pool once every cudaEventRecord and
 * cudaStreamWaitEvent call for its current generation has been issued. CUDA
 * stream waits retain the generation that was current when the wait was
 * enqueued, so a later re-record of the same cudaEvent_t does not retarget an
 * already-submitted wait.
 */
class ReusableEventPool {
   public:
    Event acquire(int32_t gpuNum, bool enableTiming = false, bool expectingHostToWaitOnThisOne = false);

    // Transfers this handle's ownership back to the pool. The caller must not
    // release the same logical lease more than once.
    void release(Event event);

    [[nodiscard]] size_t freeEventCountForTests(int32_t gpuNum,
                                                bool enableTiming = false,
                                                bool expectingHostToWaitOnThisOne = false) const;

   private:
    using EventClasses = std::array<std::array<std::vector<Event>, 2>, 2>;
    std::unordered_map<int32_t, EventClasses> freeEvents;
};

// Submission is already host-thread local in the dynamic execution paths that
// use this facility. Thread-local storage avoids a contended global lease lock.
ReusableEventPool &threadLocalReusableEventPool();

/**
 * RAII owner for a set of events leased from the thread-local reusable pool.
 * All external Event copies should have shorter lexical lifetime than this
 * object so that release happens only after every record/wait API call for the
 * submission has been issued.
 */
class ReusableEventLeases {
   public:
    explicit ReusableEventLeases(size_t expectedEventCount = 0);

    Event acquire(int32_t gpuNum, bool enableTiming = false, bool expectingHostToWaitOnThisOne = false);

    ~ReusableEventLeases();

    ReusableEventLeases(const ReusableEventLeases &) = delete;
    ReusableEventLeases &operator=(const ReusableEventLeases &) = delete;
    ReusableEventLeases(ReusableEventLeases &&) = delete;
    ReusableEventLeases &operator=(ReusableEventLeases &&) = delete;

   private:
    ReusableEventPool &pool;
    std::vector<Event> leasedEvents;
};

}  // namespace ThorImplementation
