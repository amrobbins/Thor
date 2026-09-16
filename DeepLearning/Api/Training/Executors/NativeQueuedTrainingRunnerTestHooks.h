#pragma once

#include <cstdint>

namespace Thor::detail {

// Intentional internal observability seam for scheduler-lifetime and scheduling-
// window regression tests. These counters describe run-scoped resource/worker/
// queue reuse, scheduling-window/host-decision boundaries, and exact logical-
// batch submission progress. They are not training telemetry and are inactive
// until reset...ForTests() is called.
struct NativeQueuedSchedulerResourceDiagnosticsForTests {
    uint64_t resourceConstructionCount = 0;
    uint64_t runStateConstructionCount = 0;
    uint64_t schedulingWindowCount = 0;
    uint64_t hostDecisionBarrierCount = 0;
    uint64_t workerThreadStartCount = 0;
    uint64_t submittedBatchCount = 0;
    bool hasSubmittedBatch = false;
    uint64_t maxOptimizerEpochSubmitted = 0;
    uint64_t distinctWorkerThreadsObserved = 0;
    uint64_t distinctResourceInstancesObserved = 0;
    uint64_t distinctRunStateInstancesObserved = 0;
    bool slotStorageStableAcrossSchedulingWindows = true;
    uint64_t firstProcessingFinishedEventId = 0;
    uint64_t firstCompletionFinishedEventId = 0;
    bool processingFinishedEventIdStableAcrossSchedulingWindows = true;
    bool completionFinishedEventIdStableAcrossSchedulingWindows = true;
};

void resetNativeQueuedSchedulerResourceDiagnosticsForTests();

[[nodiscard]] NativeQueuedSchedulerResourceDiagnosticsForTests
peekNativeQueuedSchedulerResourceDiagnosticsForTests();

[[nodiscard]] NativeQueuedSchedulerResourceDiagnosticsForTests
nativeQueuedSchedulerResourceDiagnosticsForTests();

}  // namespace Thor::detail
