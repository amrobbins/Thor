#include "Utilities/Common/GpuMemoryDiagnostics.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

namespace ThorImplementation {
namespace {

thread_local std::string gpu_allocation_context;
thread_local GpuMemoryDiagnosticsPlacementSession* active_placement_session = nullptr;

bool environmentEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

struct DeviceRestoreGuard {
    int previous_gpu = -1;
    bool switched = false;

    ~DeviceRestoreGuard() {
        if (switched && previous_gpu >= 0) {
            (void)cudaSetDevice(previous_gpu);
        }
    }
};

std::string escapedDetail(std::string_view detail) {
    std::string out;
    out.reserve(detail.size());
    for (const char c : detail) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

}  // namespace

struct GpuMemoryDiagnosticsPlacementSession::Impl {
    std::string context;
    bool enabled = false;
    std::vector<int32_t> gpu_nums;
    std::map<int32_t, GpuMemorySnapshot> start_by_gpu;
    struct WorkspaceCategory {
        uint64_t count = 0;
        uint64_t total_bytes = 0;
        uint64_t max_bytes = 0;
    };
    std::map<std::string, WorkspaceCategory> workspace_by_category;
    uint64_t workspace_request_count = 0;
    uint64_t workspace_request_bytes_total = 0;
};

bool gpuMemoryDiagnosticsEnabled() { return environmentEnabled("THOR_MEMORY_DIAGNOSTICS"); }

GpuMemorySnapshot queryGpuMemorySnapshot(int gpu_num) noexcept {
    GpuMemorySnapshot snapshot;
    int previous_gpu = -1;
    if (cudaGetDevice(&previous_gpu) != cudaSuccess) {
        (void)cudaGetLastError();
        return snapshot;
    }
    DeviceRestoreGuard restore{.previous_gpu = previous_gpu, .switched = previous_gpu != gpu_num};
    if (restore.switched && cudaSetDevice(gpu_num) != cudaSuccess) {
        (void)cudaGetLastError();
        return snapshot;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return snapshot;
    }
    snapshot.available = true;
    snapshot.free_bytes = static_cast<uint64_t>(free_bytes);
    snapshot.total_bytes = static_cast<uint64_t>(total_bytes);
    return snapshot;
}

std::string formatGpuMemoryBytes(uint64_t bytes) {
    static constexpr double KIB = 1024.0;
    static constexpr double MIB = 1024.0 * KIB;
    static constexpr double GIB = 1024.0 * MIB;
    std::ostringstream out;
    out << bytes << " B (" << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<uint64_t>(GIB)) {
        out << static_cast<double>(bytes) / GIB << " GiB";
    } else if (bytes >= static_cast<uint64_t>(MIB)) {
        out << static_cast<double>(bytes) / MIB << " MiB";
    } else if (bytes >= static_cast<uint64_t>(KIB)) {
        out << static_cast<double>(bytes) / KIB << " KiB";
    } else {
        out << static_cast<double>(bytes) << " B";
    }
    out << ')';
    return out.str();
}

ScopedGpuAllocationContext::ScopedGpuAllocationContext(std::string context)
    : previous_context(std::move(gpu_allocation_context)) {
    gpu_allocation_context = std::move(context);
}

ScopedGpuAllocationContext::~ScopedGpuAllocationContext() { gpu_allocation_context = std::move(previous_context); }

std::string currentGpuAllocationContext() { return gpu_allocation_context; }

GpuMemoryDiagnosticsPlacementSession::GpuMemoryDiagnosticsPlacementSession(std::string context,
                                                                           std::vector<int32_t> gpu_nums)
    : impl(std::make_unique<Impl>()) {
    impl->context = std::move(context);
    impl->enabled = gpuMemoryDiagnosticsEnabled();
    previous_allocation_context = std::move(gpu_allocation_context);
    gpu_allocation_context = impl->context;
    impl->gpu_nums = std::move(gpu_nums);
    std::sort(impl->gpu_nums.begin(), impl->gpu_nums.end());
    impl->gpu_nums.erase(std::unique(impl->gpu_nums.begin(), impl->gpu_nums.end()), impl->gpu_nums.end());
    previous_session = active_placement_session;
    if (!impl->enabled) {
        return;
    }
    active_placement_session = this;
    for (const int32_t gpu_num : impl->gpu_nums) {
        const GpuMemorySnapshot snapshot = queryGpuMemorySnapshot(gpu_num);
        impl->start_by_gpu.emplace(gpu_num, snapshot);
        if (snapshot.available) {
            std::printf("INFO Thor GPU memory placement start: context=\"%s\" gpu=%d free=%s total=%s\n",
                        escapedDetail(impl->context).c_str(),
                        gpu_num,
                        formatGpuMemoryBytes(snapshot.free_bytes).c_str(),
                        formatGpuMemoryBytes(snapshot.total_bytes).c_str());
        }
    }
    std::fflush(stdout);
}

GpuMemoryDiagnosticsPlacementSession::~GpuMemoryDiagnosticsPlacementSession() {
    gpu_allocation_context = std::move(previous_allocation_context);
    if (impl == nullptr || !impl->enabled) {
        return;
    }
    active_placement_session = previous_session;

    for (const int32_t gpu_num : impl->gpu_nums) {
        const GpuMemorySnapshot end = queryGpuMemorySnapshot(gpu_num);
        const auto start_it = impl->start_by_gpu.find(gpu_num);
        if (!end.available || start_it == impl->start_by_gpu.end() || !start_it->second.available) {
            continue;
        }
        const GpuMemorySnapshot& start = start_it->second;
        const int64_t net_allocated = static_cast<int64_t>(start.free_bytes) - static_cast<int64_t>(end.free_bytes);
        std::printf("INFO Thor GPU memory placement summary: context=\"%s\" gpu=%d free_start=%s free_end=%s net_allocated_bytes=%lld workspace_request_count=%llu workspace_requested=%s\n",
                    escapedDetail(impl->context).c_str(),
                    gpu_num,
                    formatGpuMemoryBytes(start.free_bytes).c_str(),
                    formatGpuMemoryBytes(end.free_bytes).c_str(),
                    static_cast<long long>(net_allocated),
                    static_cast<unsigned long long>(impl->workspace_request_count),
                    formatGpuMemoryBytes(impl->workspace_request_bytes_total).c_str());
    }
    for (const auto& [category, stats] : impl->workspace_by_category) {
        std::printf("INFO Thor GPU workspace category summary: context=\"%s\" category=%s count=%llu requested_total=%s max_single=%s\n",
                    escapedDetail(impl->context).c_str(),
                    category.c_str(),
                    static_cast<unsigned long long>(stats.count),
                    formatGpuMemoryBytes(stats.total_bytes).c_str(),
                    formatGpuMemoryBytes(stats.max_bytes).c_str());
    }
    std::fflush(stdout);
}

void reportGpuWorkspaceAllocationRequest(std::string_view category,
                                         int gpu_num,
                                         uint64_t requested_bytes,
                                         std::string_view detail) {
    if (requested_bytes == 0) {
        return;
    }
    GpuMemoryDiagnosticsPlacementSession* session = active_placement_session;
    const bool enabled = gpuMemoryDiagnosticsEnabled();
    if (session != nullptr && session->impl != nullptr) {
        session->impl->workspace_request_count += 1;
        session->impl->workspace_request_bytes_total += requested_bytes;
        auto& stats = session->impl->workspace_by_category[std::string(category)];
        stats.count += 1;
        stats.total_bytes += requested_bytes;
        stats.max_bytes = std::max(stats.max_bytes, requested_bytes);
    }
    if (!enabled) {
        return;
    }
    const GpuMemorySnapshot snapshot = queryGpuMemorySnapshot(gpu_num);
    if (snapshot.available) {
        std::printf("INFO Thor GPU workspace allocation: category=%.*s gpu=%d requested=%s free_before=%s total=%s detail=\"%s\"\n",
                    static_cast<int>(category.size()),
                    category.data(),
                    gpu_num,
                    formatGpuMemoryBytes(requested_bytes).c_str(),
                    formatGpuMemoryBytes(snapshot.free_bytes).c_str(),
                    formatGpuMemoryBytes(snapshot.total_bytes).c_str(),
                    escapedDetail(detail).c_str());
    } else {
        std::printf("INFO Thor GPU workspace allocation: category=%.*s gpu=%d requested=%s detail=\"%s\"\n",
                    static_cast<int>(category.size()),
                    category.data(),
                    gpu_num,
                    formatGpuMemoryBytes(requested_bytes).c_str(),
                    escapedDetail(detail).c_str());
    }
    std::fflush(stdout);
}

}  // namespace ThorImplementation
