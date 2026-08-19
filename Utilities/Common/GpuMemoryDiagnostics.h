#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ThorImplementation {

struct GpuMemorySnapshot {
    bool available = false;
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
};

[[nodiscard]] bool gpuMemoryDiagnosticsEnabled();
[[nodiscard]] GpuMemorySnapshot queryGpuMemorySnapshot(int gpu_num) noexcept;
[[nodiscard]] std::string formatGpuMemoryBytes(uint64_t bytes);

// Provides allocation-specific context to Tensor's GPU OOM error reporting.
// Nesting is supported; destruction restores the previous context.
class ScopedGpuAllocationContext {
   public:
    explicit ScopedGpuAllocationContext(std::string context);
    ~ScopedGpuAllocationContext();

    ScopedGpuAllocationContext(const ScopedGpuAllocationContext&) = delete;
    ScopedGpuAllocationContext& operator=(const ScopedGpuAllocationContext&) = delete;

   private:
    std::string previous_context;
};

[[nodiscard]] std::string currentGpuAllocationContext();

// Placement-scoped diagnostics are opt-in through THOR_MEMORY_DIAGNOSTICS=1.
// They report net GPU memory change across Network::place() and aggregate the
// workspace allocation requests made while stamping. The allocation logs are
// intentionally emitted before cudaMalloc so an OOM still identifies the
// operation that requested the memory.
class GpuMemoryDiagnosticsPlacementSession {
   public:
    GpuMemoryDiagnosticsPlacementSession(std::string context, std::vector<int32_t> gpu_nums);
    ~GpuMemoryDiagnosticsPlacementSession();

    GpuMemoryDiagnosticsPlacementSession(const GpuMemoryDiagnosticsPlacementSession&) = delete;
    GpuMemoryDiagnosticsPlacementSession& operator=(const GpuMemoryDiagnosticsPlacementSession&) = delete;

   private:
    friend void reportGpuWorkspaceAllocationRequest(std::string_view category,
                                                    int gpu_num,
                                                    uint64_t requested_bytes,
                                                    std::string_view detail);
    struct Impl;
    std::unique_ptr<Impl> impl;
    GpuMemoryDiagnosticsPlacementSession* previous_session = nullptr;
    std::string previous_allocation_context;
};

void reportGpuWorkspaceAllocationRequest(std::string_view category,
                                         int gpu_num,
                                         uint64_t requested_bytes,
                                         std::string_view detail = {});

}  // namespace ThorImplementation
