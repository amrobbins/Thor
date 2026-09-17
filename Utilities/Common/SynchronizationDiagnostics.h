#pragma once

#include <cstdint>

namespace ThorImplementation {

#ifdef THOR_DEBUG
// Debug/test-only accounting for CUDA synchronization operations emitted
// through Thor's central Event/Stream wrappers. Production Release builds do
// not contain the counters or instrumentation calls.
struct SynchronizationOperationCounts {
    uint64_t eventRecordCount = 0;
    uint64_t streamWaitEventCount = 0;
    uint64_t hostEventSynchronizeCount = 0;
};

void resetSynchronizationOperationCountsForTests();
[[nodiscard]] SynchronizationOperationCounts synchronizationOperationCountsForTests();

namespace detail {
void recordCudaEventRecordForTests() noexcept;
void recordCudaStreamWaitEventForTests() noexcept;
void recordCudaEventSynchronizeForTests() noexcept;
}  // namespace detail
#endif

}  // namespace ThorImplementation
