#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"

#include <atomic>

namespace ThorImplementation {
namespace {

std::atomic<CubReductionTuningDepth>& processDefault() {
    static std::atomic<CubReductionTuningDepth> depth{CubReductionTuningDepth::Regular};
    return depth;
}

std::optional<CubReductionTuningDepth>& threadScope() {
    static thread_local std::optional<CubReductionTuningDepth> depth;
    return depth;
}

}  // namespace

CubReductionTuningDepth defaultCubReductionTuningDepth() { return processDefault().load(std::memory_order_relaxed); }

void setDefaultCubReductionTuningDepth(CubReductionTuningDepth depth) {
    processDefault().store(depth, std::memory_order_relaxed);
}

CubReductionTuningDepth currentCubReductionTuningDepth() {
    const std::optional<CubReductionTuningDepth>& scoped = threadScope();
    return scoped ? *scoped : defaultCubReductionTuningDepth();
}

ScopedCubReductionTuningDepth::ScopedCubReductionTuningDepth(CubReductionTuningDepth depth)
    : previous_(threadScope()) {
    threadScope() = depth;
}

ScopedCubReductionTuningDepth::~ScopedCubReductionTuningDepth() { threadScope() = previous_; }

}  // namespace ThorImplementation
