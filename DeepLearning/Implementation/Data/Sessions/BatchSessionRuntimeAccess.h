#pragma once

#include "DeepLearning/Api/Data/BatchSession.h"

namespace ThorImplementation {

/**
 * Internal executor access to BatchSession tail semantics. The public training
 * API intentionally has no exact/wrap configuration; executors select WRAP
 * only when a placed network cannot execute partial tail batches exactly.
 */
class BatchSessionRuntimeAccess {
   public:
    static void setTailMode(Thor::BatchSession& session, BatchTailMode mode) {
        session.setBatchTailModeForRuntime(mode);
    }

    [[nodiscard]] static BatchTailMode getTailMode(const Thor::BatchSession& session) {
        return session.getBatchTailModeForRuntime();
    }

    [[nodiscard]] static uint64_t examplesProcessedPerEpoch(
        Thor::BatchSession& session,
        ExampleType exampleType) {
        return session.getExamplesProcessedPerEpochForRuntime(exampleType);
    }
};

}  // namespace ThorImplementation
