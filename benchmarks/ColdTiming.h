#pragma once

// 1164: the reduction sweeps' one timing method, shared by thor_fp8_kr_training_sweep (where it was developed,
// 1152-1153), thor_fp8_retained_cold_l2_sweep --training (RK) and, 1174, thor_dense_production_audit (any dtype).
// 1174: renamed from Fp8ColdTiming.h; nothing in it is FP8-specific.
//
//   - Cold inputs without a flush kernel: every launch reads its own input view, cycling through views that span at
//     least InputReuseBytes (20x a 96 MiB L2), each 256-byte aligned as a fresh cudaMalloc allocation is.
//   - Cold outputs: every launch writes its own view of an output Ring, cycling through views spanning at least
//     OutputReuseBytes, so each launch pays for its output's write-back as a real workload does. (A shared output stays
//     in L2 between launches when it is smaller than L2: at short rows that hid most of the write traffic.)
//   - Per candidate, one untimed launch (it loads the kernel) and one timed probe launch; each sample then times
//     launchesPerSample(probe) back-to-back launches, about interval_us of GPU time, and divides by the count. CUDA
//     events here resolve about 2 us, so a 5 us kernel timed once per sample is on a 40% grid; ~1 ms samples put the
//     event resolution at 0.2%. (KR, 1152-1153: run-to-run p50 0.1-0.3% per (cell, shape); the RK sweep's one launch
//     per sample showed sample spreads of 8-22% below 16 MiB in the 1162/1163 stamp audits.)
//   - Samples come from asynchronous batches (Fp8OracleBatchQueue): pass p visits every cell, so a candidate's samples
//     are spread over the batch and the GPU never idles between them.
//   - The copy reference is timed the same way (ring source and destination views, back to back).

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking::ColdTiming {

inline constexpr uint64_t MiB = 1ULL << 20;
inline constexpr uint64_t GiB = 1ULL << 30;
inline constexpr uint64_t InputReuseBytes = 2 * GiB;   // a cell's input views span at least this
inline constexpr uint64_t OutputReuseBytes = 1 * GiB;  // a cell's output views span at least this
inline constexpr uint64_t MaxViews = 1024;

inline void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

inline uint64_t ceilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
inline uint64_t align256(uint64_t bytes) { return (bytes + 255) & ~uint64_t{255}; }

struct TimingEvent {
    cudaEvent_t handle = nullptr;
    TimingEvent() { checkCuda(cudaEventCreate(&handle), "create timing event"); }
    ~TimingEvent() {
        if (handle) (void)cudaEventDestroy(handle);
    }
    TimingEvent(const TimingEvent&) = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;
};

// One allocation handed out as consecutive 256-byte aligned views (a fresh allocation's alignment), wrapping to the
// start when the next view does not fit. Views are only addresses: launches on one stream never overlap.
class Ring {
   public:
    Ring(const TensorPlacement& placement, uint64_t bytes)
        : storage_(placement, TensorDescriptor(DataType::UINT8, {bytes})), bytes_(bytes) {}
    static uint64_t span(uint64_t bytes) { return align256(bytes) + 256; }
    uint64_t take(uint64_t bytes) {
        const uint64_t need = span(bytes);
        if (need > bytes_) throw std::runtime_error("benchmark view exceeds its ring");
        if (cursor_ + need > bytes_) {
            cursor_ = 0;
            ++wraps_;
        }
        const uint64_t offset = cursor_;
        cursor_ += need;
        return offset;
    }
    Tensor view(DataType dtype, const std::vector<uint64_t>& dims, uint64_t offset) const {
        return storage_.aliasByteStorageAs(dtype, dims, offset);
    }
    Tensor& storage() { return storage_; }
    [[nodiscard]] uint64_t bytes() const { return bytes_; }
    [[nodiscard]] uint64_t wraps() const { return wraps_; }

   private:
    Tensor storage_;
    uint64_t bytes_;
    uint64_t cursor_ = 0;
    uint64_t wraps_ = 0;
};

// Views cycling through at least `reuse` bytes (at least 1, at most MaxViews and half the ring).
inline uint64_t viewCount(uint64_t footprint, uint64_t reuse, uint64_t ring_bytes) {
    const uint64_t span = Ring::span(footprint);
    const uint64_t fit = std::max<uint64_t>(1, ring_bytes / 2 / span);
    return std::clamp<uint64_t>(ceilDiv(reuse, span), 1, std::min(fit, MaxViews));
}

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Launches per sample from a candidate's probe time: about interval_us of GPU time, 1 to max_launches.
inline uint64_t launchesPerSample(double probe_us, double interval_us, uint64_t max_launches) {
    return static_cast<uint64_t>(
        std::clamp(std::ceil(interval_us / std::max(probe_us, 2.0)), 1.0, static_cast<double>(max_launches)));
}

// Copy reference with the same cold method: ring input views to ring output views, back to back.
inline double copyBandwidth(Ring& in, Ring& out, Stream& stream, uint64_t bytes) {
    const int warm = 3, timed = 15;
    std::vector<std::unique_ptr<TimingEvent>> events;
    for (int i = 0; i < 2 * timed; ++i) events.push_back(std::make_unique<TimingEvent>());
    for (int i = 0; i < warm + timed; ++i) {
        const uint64_t src = in.take(bytes), dst = out.take(bytes);
        if (i >= warm) checkCuda(cudaEventRecord(events[2 * (i - warm)]->handle, stream.getStream()), "copy start");
        checkCuda(cudaMemcpyAsync(out.storage().getMemPtr<uint8_t>() + dst, in.storage().getMemPtr<uint8_t>() + src,
                                  bytes, cudaMemcpyDeviceToDevice, stream.getStream()),
                  "reference copy");
        if (i >= warm) checkCuda(cudaEventRecord(events[2 * (i - warm) + 1]->handle, stream.getStream()), "copy stop");
    }
    stream.synchronize();
    std::vector<double> us;
    for (int i = 0; i < timed; ++i) {
        float ms = 0;
        checkCuda(cudaEventElapsedTime(&ms, events[2 * i]->handle, events[2 * i + 1]->handle), "copy elapsed");
        us.push_back(1000.0 * ms);
    }
    return 2.0 * double(bytes) / median(us);  // bytes read + written per us
}

// 1153: copy time against moved bytes (copy n bytes: n read + n written), timed as the cells are: ring source and
// destination views, back-to-back copies per sample (about interval_us), median of 5 interleaved samples.
struct CopyCurve {
    std::vector<std::pair<double, double>> points;  // (moved bytes, us)
    // Log-log interpolation; proportional beyond the end points.
    [[nodiscard]] double us(double moved) const {
        if (points.empty()) return 0;
        if (moved <= points.front().first) return points.front().second * moved / points.front().first;
        for (size_t i = 1; i < points.size(); ++i) {
            if (moved <= points[i].first) {
                const auto [x0, y0] = points[i - 1];
                const auto [x1, y1] = points[i];
                const double f = std::log(moved / x0) / std::log(x1 / x0);
                return std::exp(std::log(y0) + f * std::log(y1 / y0));
            }
        }
        return points.back().second * moved / points.back().first;
    }
};

// 1176: min_moved sets the first point. Below the first point the curve is proportional, which puts a copy of a few KiB
// far under the launch floor, so callers timing small cells pass their smallest moved bytes (rounded down to a power
// of two, at least 4 KiB). The FP8 sweeps keep the 2 MiB start they were trained and audited with.
inline CopyCurve copyCurve(Ring& in, Ring& out, Stream& stream, uint64_t max_moved, double interval_us,
                           uint64_t min_moved = 2 * MiB) {
    uint64_t first = 4096;
    while (first * 2 <= min_moved) first *= 2;
    std::vector<uint64_t> sizes;
    for (uint64_t moved = first;; moved *= 2) {
        sizes.push_back(moved);
        if (moved >= max_moved) break;
    }
    constexpr int samples = 5;
    std::vector<uint64_t> launches;
    for (uint64_t moved : sizes)
        launches.push_back(static_cast<uint64_t>(
            std::clamp(std::ceil(interval_us / (double(moved) / 1.5e6 + 3.0)), 1.0, 256.0)));
    const auto copy = [&](uint64_t n) {
        const uint64_t src = in.take(n), dst = out.take(n);
        checkCuda(cudaMemcpyAsync(out.storage().getMemPtr<uint8_t>() + dst, in.storage().getMemPtr<uint8_t>() + src, n,
                                  cudaMemcpyDeviceToDevice, stream.getStream()),
                  "curve copy");
    };
    for (uint64_t moved : sizes) copy(moved / 2);  // warm
    std::vector<std::unique_ptr<TimingEvent>> events;
    for (size_t i = 0; i < 2 * samples * sizes.size(); ++i) events.push_back(std::make_unique<TimingEvent>());
    for (int p = 0; p < samples; ++p)
        for (size_t i = 0; i < sizes.size(); ++i) {
            const size_t e = 2 * (p * sizes.size() + i);
            checkCuda(cudaEventRecord(events[e]->handle, stream.getStream()), "curve start");
            for (uint64_t l = 0; l < launches[i]; ++l) copy(sizes[i] / 2);
            checkCuda(cudaEventRecord(events[e + 1]->handle, stream.getStream()), "curve stop");
        }
    stream.synchronize();
    CopyCurve curve;
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::vector<double> us;
        for (int p = 0; p < samples; ++p) {
            const size_t e = 2 * (p * sizes.size() + i);
            float ms = 0;
            checkCuda(cudaEventElapsedTime(&ms, events[e]->handle, events[e + 1]->handle), "curve elapsed");
            us.push_back(1000.0 * ms / static_cast<double>(launches[i]));
        }
        curve.points.push_back({static_cast<double>(sizes[i]), median(us)});
    }
    return curve;
}

}  // namespace ThorImplementation::CubReductionBenchmarking::ColdTiming
