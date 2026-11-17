#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"

#include <atomic>

namespace ThorImplementation {
namespace {

std::atomic<ReductionTuningDepth>& processDefault() {
    static std::atomic<ReductionTuningDepth> depth{ReductionTuningDepth::Regular};
    return depth;
}

std::optional<ReductionTuningDepth>& threadScope() {
    static thread_local std::optional<ReductionTuningDepth> depth;
    return depth;
}

}  // namespace

ReductionTuningDepth defaultReductionTuningDepth() { return processDefault().load(std::memory_order_relaxed); }

void setDefaultReductionTuningDepth(ReductionTuningDepth depth) {
    processDefault().store(depth, std::memory_order_relaxed);
}

ReductionTuningDepth currentReductionTuningDepth() {
    const std::optional<ReductionTuningDepth>& scoped = threadScope();
    return scoped ? *scoped : defaultReductionTuningDepth();
}

ScopedReductionTuningDepth::ScopedReductionTuningDepth(ReductionTuningDepth depth)
    : previous_(threadScope()) {
    threadScope() = depth;
}

ScopedReductionTuningDepth::~ScopedReductionTuningDepth() { threadScope() = previous_; }

}  // namespace ThorImplementation
