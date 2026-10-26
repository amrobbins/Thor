#pragma once

// Reusable single-stream, cache-cold CUDA reduction autotuning utility.
// Deliberately independent of FP8 family selection and production dispatch.
// Callbacks launch the COMPLETE plan, including any FP32 continuation, and
// must enqueue work only: no cudaDevice/Stream/Event synchronization inside.

#include "Utilities/TensorOperations/Cub/ReductionAutotunerCore.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation::ReductionAutotuning {

// The candidate's callback must write every output element it will later read.
// scratch is an empty Tensor if scratch_bytes==0. Output and scratch are
// separate rotating tensors; scratch bytes are for the WHOLE execution plan.
// Both complete and staged plans share the same logical input and output.
struct Candidate {
    std::string name;
    uint64_t scratch_bytes = 0;
    std::function<void(const Tensor& input, Tensor& output, Tensor& scratch, Stream& stream)> launch;
};

struct Options {
    // Zero: eight times the GPU's reported L2 size. No allocation/copies are
    // part of the measured event interval. This is a minimum reuse distance,
    // not a cache-flush memset between individual candidates.
    uint64_t cold_reuse_bytes = 0;
    // Zero: entropy from random_device. Nonzero: reproducible round orders.
    uint64_t random_seed = 0;
    uint32_t warmups_per_candidate = 1;
};

struct RunResult : TuningResult {
    uint64_t random_seed = 0;
    size_t input_slot_count = 0;
    std::string winner_name;
};

namespace detail {

inline void checkCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Reduction autotuning: ") + context + ": " + cudaGetErrorString(status));
    }
}

class TimingEvent {
public:
    TimingEvent() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~TimingEvent() { if (event_) (void)cudaEventDestroy(event_); }
    TimingEvent(const TimingEvent&) = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;
    TimingEvent(TimingEvent&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
    TimingEvent& operator=(TimingEvent&&) = delete;
    [[nodiscard]] cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

inline uint64_t checkedMultiply(uint64_t a, uint64_t b) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
        throw std::overflow_error("Reduction autotuning allocation size overflow");
    }
    return a * b;
}

// A single owning Tensor is split into dense views, each starting on a
// 32-byte boundary with >=128 bytes of padding after its logical data.
// Each slot is physically disjoint and uses the same TensorDescriptor.
class TensorRing {
public:
    TensorRing(TensorPlacement placement, TensorDescriptor descriptor, size_t slots)
        : descriptor_(std::move(descriptor)), stride_(alignedStride(descriptor_.getArraySizeInBytes())),
          count_(slots) {
        if (slots == 0) throw std::invalid_argument("Reduction autotuner ring cannot be empty");
        const uint64_t bytes = checkedMultiply(stride_, static_cast<uint64_t>(slots));
        storage_ = Tensor(placement, TensorDescriptor(DataType::UINT8, {bytes}));
    }
    // View handles are cheap; do not allocate a potentially huge vector of
    // handles when a tiny tensor requires hundreds of thousands of cold slots.
    [[nodiscard]] Tensor at(size_t slot) const {
        if (slot >= count_) throw std::out_of_range("Reduction autotuner slot out of range");
        return storage_.aliasByteStorageAs(descriptor_.getDataType(), descriptor_.getDimensions(),
                                           static_cast<uint64_t>(slot) * stride_);
    }
    [[nodiscard]] uint64_t stride() const { return stride_; }
    [[nodiscard]] size_t count() const { return count_; }
    [[nodiscard]] Tensor& storage() { return storage_; }
    [[nodiscard]] const Tensor& storage() const { return storage_; }
private:
    TensorDescriptor descriptor_;
    uint64_t stride_;
    size_t count_;
    Tensor storage_;
};

} // namespace detail

// Ownership: allocations/events remain live through the single final stream
// synchronization. Caller still owns any additional resources captured by the
// candidate callbacks; those must also survive until this call returns.
// Caller must provide a contiguous GPU input on the same device as stream.
// Device data is cloned into the cold ring before any timed launches.
inline RunResult autotune(const std::vector<Candidate>& candidates,
                          const Tensor& representative_input,
                          const TensorDescriptor& output_descriptor,
                          Stream& stream,
                          Options options = {}) {
    if (candidates.empty()) throw std::invalid_argument("Reduction autotuning needs at least one candidate");
    if (!representative_input.isInitialized() || !representative_input.isDenseContiguous() ||
        representative_input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Reduction autotuning requires contiguous input on stream GPU");
    }
    if (output_descriptor.getArraySizeInBytes() == 0) {
        throw std::invalid_argument("Reduction autotuning requires nonempty output");
    }
    for (const Candidate& candidate : candidates) {
        if (!candidate.launch) throw std::invalid_argument("Reduction autotuning candidate has no launch function");
    }
    const TensorPlacement placement = representative_input.getPlacement();
    ScopedGpu gpu(placement.getDeviceNum());
    int l2_bytes = 0;
    detail::checkCuda(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, placement.getDeviceNum()),
                      "cudaDeviceGetAttribute(L2)");
    if (options.cold_reuse_bytes == 0) {
        if (l2_bytes <= 0) throw std::runtime_error("Reduction autotuning cannot infer L2 size");
        options.cold_reuse_bytes = detail::checkedMultiply(static_cast<uint64_t>(l2_bytes), 8);
    }

    const uint64_t input_bytes = representative_input.getDescriptor().getArraySizeInBytes();
    const uint64_t input_stride = alignedStride(input_bytes);
    const size_t slots = coldSlotCount(input_stride, options.cold_reuse_bytes);
    if (slots > std::numeric_limits<size_t>::max() / candidates.size() && candidates.size() > 1) {
        throw std::overflow_error("Reduction autotuning slot/candidate overflow");
    }
    uint64_t max_scratch_bytes = 0;
    for (const Candidate& candidate : candidates) max_scratch_bytes = std::max(max_scratch_bytes, candidate.scratch_bytes);

    // One backing Tensor for each independently rotating region (input/output/
    // optional scratch). All allocations occur before any timed launch.
    detail::TensorRing inputs(placement, representative_input.getDescriptor(), slots);
    detail::TensorRing outputs(placement, output_descriptor, slots);
    std::unique_ptr<detail::TensorRing> scratch;
    if (max_scratch_bytes != 0) {
        scratch = std::make_unique<detail::TensorRing>(
            placement, TensorDescriptor(DataType::UINT8, {max_scratch_bytes}), slots);
    }

    RunResult result;
    result.random_seed = options.random_seed ? options.random_seed
        : ((static_cast<uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}());
    result.input_slot_count = slots;
    const auto rounds = makeRoundOrders(candidates.size(), result.random_seed);

    // Allocate timing events *before* timed execution; each measurement has
    // independent start/stop events, so no event is rerecorded prematurely.
    if (candidates.size() > std::numeric_limits<size_t>::max() / (MeasurementCount * 2)) {
        throw std::overflow_error("Reduction autotuning event count overflow");
    }
    std::vector<detail::TimingEvent> events;
    events.reserve(candidates.size() * MeasurementCount * 2);
    for (size_t i = 0; i < candidates.size() * MeasurementCount * 2; ++i) events.emplace_back();
    std::vector<Measurements> observations(candidates.size());
    uint64_t cursor = 0;
    auto enqueue = [&](size_t candidate_index, bool measured, size_t round) {
        const size_t slot = static_cast<size_t>(cursor++ % slots);
        const Candidate& candidate = candidates[candidate_index];
        Tensor input_view = inputs.at(slot);
        Tensor output_view = outputs.at(slot);
        // Scratch is an aligned allocation of at least scratch_bytes; a
        // candidate must only access its declared scratch_bytes prefix.
        Tensor scratch_view = candidate.scratch_bytes ? scratch->at(slot) : Tensor();
        const size_t event_index = (round * candidates.size() + candidate_index) * 2;
        if (measured) detail::checkCuda(cudaEventRecord(events[event_index].get(), stream.getStream()),
                                        "cudaEventRecord(start)");
        candidate.launch(input_view, output_view, scratch_view, stream);
        detail::checkCuda(cudaPeekAtLastError(), "candidate launch");
        if (measured) detail::checkCuda(cudaEventRecord(events[event_index + 1].get(), stream.getStream()),
                                        "cudaEventRecord(stop)");
    };

    // All transfers, warmups and measurements are stream-ordered. A callback
    // must never synchronize. One final host sync follows every one of the
    // 5*N measured launches; also fence on error to protect Tensor lifetimes.
    try {
        // Initialize one slot, then double the initialized prefix in-place.
        // Copies never overlap, use only O(log slots) enqueue calls, and
        // reproduce the caller's representative input in EVERY cold slot.
        auto* arena = static_cast<uint8_t*>(inputs.storage().getMemPtr<void>());
        detail::checkCuda(cudaMemsetAsync(arena, 0, inputs.stride(), stream.getStream()),
                          "cudaMemsetAsync(seed padding)");
        detail::checkCuda(cudaMemcpyAsync(arena, representative_input.getMemPtr<void>(), input_bytes,
                                          cudaMemcpyDeviceToDevice, stream.getStream()),
                          "cudaMemcpyAsync(seed slot)");
        for (size_t initialized = 1; initialized < slots;) {
            const size_t copy_slots = std::min(initialized, slots - initialized);
            detail::checkCuda(cudaMemcpyAsync(arena + initialized * inputs.stride(), arena,
                                              copy_slots * inputs.stride(), cudaMemcpyDeviceToDevice,
                                              stream.getStream()), "cudaMemcpyAsync(duplicate seed slots)");
            initialized += copy_slots;
        }
        for (uint32_t warmup = 0; warmup < options.warmups_per_candidate; ++warmup) {
            for (size_t i = 0; i < candidates.size(); ++i) enqueue(i, false, 0);
        }
        for (size_t round = 0; round < MeasurementCount; ++round) {
            for (size_t index : rounds[round]) enqueue(index, true, round);
        }
        detail::checkCuda(cudaStreamSynchronize(stream.getStream()), "final stream synchronization");
    } catch (...) {
        (void)cudaStreamSynchronize(stream.getStream());
        throw;
    }

    for (size_t round = 0; round < MeasurementCount; ++round) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            const size_t event_index = (round * candidates.size() + i) * 2;
            float elapsed_ms = 0.0f;
            detail::checkCuda(cudaEventElapsedTime(&elapsed_ms, events[event_index].get(),
                                                    events[event_index + 1].get()),
                              "cudaEventElapsedTime");
            observations[i][round] = static_cast<double>(elapsed_ms) * 1000.0;
        }
    }
    static_cast<TuningResult&>(result) = chooseWinner(observations, input_bytes);
    result.winner_name = candidates[result.winner_index].name;
    return result;
}

} // namespace ThorImplementation::ReductionAutotuning
