#pragma once

#include <cstdint>

namespace Thor::detail {

// Intentional internal observability seam for scheduler-lifetime regression
// tests. These counters describe run-scoped scheduler resource/worker reuse;
// they are not training telemetry and are inactive until reset...ForTests() is
// called.
struct NativeQueuedSchedulerResourceDiagnosticsForTests {
    uint64_t resourceConstructionCount = 0;
    uint64_t executionLaunchCount = 0;
    uint64_t workerThreadStartCount = 0;
    uint64_t distinctWorkerThreadsObserved = 0;
    uint64_t distinctResourceInstancesObserved = 0;
    uint64_t firstProcessingFinishedEventId = 0;
    uint64_t firstCompletionFinishedEventId = 0;
    bool processingFinishedEventIdStableAcrossExecutions = true;
    bool completionFinishedEventIdStableAcrossExecutions = true;
};

void resetNativeQueuedSchedulerResourceDiagnosticsForTests();

[[nodiscard]] NativeQueuedSchedulerResourceDiagnosticsForTests
nativeQueuedSchedulerResourceDiagnosticsForTests();

}  // namespace Thor::detail
