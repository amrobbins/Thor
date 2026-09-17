#include "Utilities/Common/SynchronizationDiagnostics.h"

#ifdef THOR_DEBUG
#include <atomic>

namespace ThorImplementation {
namespace {

struct AtomicSynchronizationOperationCounts {
    std::atomic<uint64_t> eventRecordCount{0};
    std::atomic<uint64_t> streamWaitEventCount{0};
    std::atomic<uint64_t> hostEventSynchronizeCount{0};
};

AtomicSynchronizationOperationCounts& synchronizationOperationCountsStorage() {
    static AtomicSynchronizationOperationCounts counts;
    return counts;
}

}  // namespace

void resetSynchronizationOperationCountsForTests() {
    AtomicSynchronizationOperationCounts& counts = synchronizationOperationCountsStorage();
    counts.eventRecordCount.store(0, std::memory_order_relaxed);
    counts.streamWaitEventCount.store(0, std::memory_order_relaxed);
    counts.hostEventSynchronizeCount.store(0, std::memory_order_relaxed);
}

SynchronizationOperationCounts synchronizationOperationCountsForTests() {
    const AtomicSynchronizationOperationCounts& counts = synchronizationOperationCountsStorage();
    return SynchronizationOperationCounts{
        .eventRecordCount = counts.eventRecordCount.load(std::memory_order_relaxed),
        .streamWaitEventCount = counts.streamWaitEventCount.load(std::memory_order_relaxed),
        .hostEventSynchronizeCount = counts.hostEventSynchronizeCount.load(std::memory_order_relaxed),
    };
}

void detail::recordCudaEventRecordForTests() noexcept {
    synchronizationOperationCountsStorage().eventRecordCount.fetch_add(1, std::memory_order_relaxed);
}

void detail::recordCudaStreamWaitEventForTests() noexcept {
    synchronizationOperationCountsStorage().streamWaitEventCount.fetch_add(1, std::memory_order_relaxed);
}

void detail::recordCudaEventSynchronizeForTests() noexcept {
    synchronizationOperationCountsStorage().hostEventSynchronizeCount.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace ThorImplementation
#endif
