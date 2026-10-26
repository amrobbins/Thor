#pragma once

// Shared serialization boundary for GPU reduction autotuning contests.
// Select dependencies and construct candidate programs BEFORE entering this
// scope. All domains (FP8 today, FP32 later) use the same per-device lock.
// Ordinary cached selection, plan composition, and stamped execution never
// acquire it.

#include <memory>
#include <mutex>

namespace ThorImplementation::CubReductionInternal {

class ScopedReductionGpuMeasurement final {
public:
    explicit ScopedReductionGpuMeasurement(int device_ordinal);
    ~ScopedReductionGpuMeasurement() noexcept;

    ScopedReductionGpuMeasurement(const ScopedReductionGpuMeasurement&) = delete;
    ScopedReductionGpuMeasurement& operator=(const ScopedReductionGpuMeasurement&) = delete;
    ScopedReductionGpuMeasurement(ScopedReductionGpuMeasurement&&) = delete;
    ScopedReductionGpuMeasurement& operator=(ScopedReductionGpuMeasurement&&) = delete;

private:
    std::shared_ptr<std::mutex> device_mutex_;
    std::unique_lock<std::mutex> lock_;
};

} // namespace ThorImplementation::CubReductionInternal
