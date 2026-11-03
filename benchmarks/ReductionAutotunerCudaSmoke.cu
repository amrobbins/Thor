#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"

#include <cuda_runtime.h>

#include <cstdint>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionAutotuning;

namespace {
constexpr unsigned BlockThreads = 256;
constexpr unsigned ValuesPerBlock = 4096;
constexpr unsigned Bytes = 4 * 1024 * 1024;
constexpr unsigned Blocks = Bytes / ValuesPerBlock;

// Each CTA reduces 4096 FP8-sized bytes to one FP32 partial. "slow" repeats
// volatile loads (without changing the result) to test winner selection.
__global__ void probeReduce(const uint8_t* source, float* target, unsigned repeats) {
    const unsigned row = blockIdx.x;
    const volatile uint8_t* input = source;
    float value = 0.0f;
    for (unsigned pass = 0; pass < repeats; ++pass) {
        for (unsigned pos = threadIdx.x; pos < ValuesPerBlock; pos += BlockThreads) {
            const float x = static_cast<float>(input[row * ValuesPerBlock + pos]);
            if (pass == 0) value += x;
        }
    }
    for (unsigned delta = 16; delta != 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }
    __shared__ float warp_values[BlockThreads / 32];
    if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < (BlockThreads / 32) ? warp_values[threadIdx.x] : 0.f;
        for (unsigned delta = 16; delta != 0; delta >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, delta);
        }
        if (threadIdx.x == 0) target[row] = value;
    }
}

__global__ void copyPartials(const float* from, float* to, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) to[i] = from[i];
}
} // namespace

int main() {
    try {
        const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
        Stream stream(gpu);
        // Verify the actual CUDA-backed view pointers for all three ring uses.
        // Aligned logical sizes must still have a full 128-byte guard region.
        auto checkRing = [&](const char* label, const TensorDescriptor& descriptor) {
            detail::TensorRing ring(gpu, descriptor, 3);
            const uint64_t logical_bytes = descriptor.getArraySizeInBytes();
            uintptr_t previous = 0;
            for (size_t i = 0; i < 3; ++i) {
                Tensor view = ring.at(i);
                const uintptr_t address = reinterpret_cast<uintptr_t>(view.getMemPtr<void>());
                if (address % TensorViewAlignmentBytes != 0 ||
                    (i != 0 && address - previous < logical_bytes + TensorViewPaddingBytes)) {
                    throw std::runtime_error(std::string(label) + " ring view alignment/padding failed");
                }
                previous = address;
            }
        };
        checkRing("input", TensorDescriptor(DataType::UINT8, {4096}));
        checkRing("output", TensorDescriptor(DataType::FP32, {129}));
        checkRing("scratch", TensorDescriptor(DataType::UINT8, {257}));
        Tensor input(gpu, TensorDescriptor(DataType::UINT8, {Bytes}));
        input.memsetAsync(stream, 1);
        const TensorDescriptor output_type(DataType::FP32, {Blocks});
        std::vector<Candidate> candidates;
        candidates.push_back({"complete_fast", 0, [](const Tensor& x, Tensor& y, Tensor&, Stream& s) {
            probeReduce<<<Blocks, BlockThreads, 0, s.getStream()>>>(x.getMemPtr<uint8_t>(), y.getMemPtr<float>(), 1);
        }});
        candidates.push_back({"staged_fp32_continuation", Blocks * sizeof(float),
                              [](const Tensor& x, Tensor& y, Tensor& scratch, Stream& s) {
            auto* partials = static_cast<float*>(scratch.getMemPtr<void>());
            probeReduce<<<Blocks, BlockThreads, 0, s.getStream()>>>(x.getMemPtr<uint8_t>(), partials, 1);
            copyPartials<<<(Blocks + 255) / 256, 256, 0, s.getStream()>>>(partials, y.getMemPtr<float>(), Blocks);
        }});
        candidates.push_back({"complete_slow", 0, [](const Tensor& x, Tensor& y, Tensor&, Stream& s) {
            probeReduce<<<Blocks, BlockThreads, 0, s.getStream()>>>(x.getMemPtr<uint8_t>(), y.getMemPtr<float>(), 16);
        }});
        Options options;
        options.random_seed = 20261023;
        // 1162: one contest policy (probe, per-candidate launches, abandonment); run it as shipped and with
        // abandonment off, which must measure every candidate and still not pick the slow one.
        auto run = [&](const char* label, const Options& contest) {
            const RunResult result = autotune(candidates, input, output_type, stream, contest);
            std::cout << label << ": " << result.input_slot_count << " rotating slots, winning candidate "
                      << result.winner_index << " (" << result.winner_name << "), " << result.abandoned_count
                      << " abandoned\n";
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto& c = result.candidates[i];
                std::cout << "  " << candidates[i].name << ": probe " << std::fixed << std::setprecision(3)
                          << c.probe_us << " us, " << c.launches_per_measurement << " launches/measurement, ";
                if (c.abandoned) std::cout << "abandoned\n";
                else std::cout << c.trimmed_mean_us << " us, " << c.trimmed_mean_tb_per_s << " TB/s\n";
            }
            if (result.winner_index == 2 || result.candidates[result.winner_index].abandoned)
                throw std::runtime_error(std::string(label) + ": the slow or an abandoned candidate won");
            return result;
        };
        (void)run("Cold reduction autotune", options);
        Options measure_all = options;
        measure_all.abandon_ratio = 0.0;
        if (run("Every candidate measured", measure_all).abandoned_count != 0)
            throw std::runtime_error("abandon_ratio 0 abandoned a candidate");
        std::cout << "Reduction autotuner CUDA smoke passes\n";
    } catch (const std::exception& e) {
        std::cerr << "Reduction autotuner CUDA smoke failed: " << e.what() << '\n';
        return 1;
    }
}
