#include "test/Utilities/TensorOperations/Reduction/ReductionAutotunerDevice.h"

namespace ThorImplementation::ReductionAutotunerDevice {
namespace {

constexpr unsigned BlockThreads = 256;
constexpr unsigned Blocks = static_cast<unsigned>(Partials);

// One CTA reduces BytesPerPartial bytes to one FP32 partial. repeats > 1 re-reads the bytes (volatile) without
// changing the result.
__global__ void reduceBytes(const uint8_t* source, float* target, unsigned repeats) {
    const unsigned row = blockIdx.x;
    const volatile uint8_t* input = source;
    float value = 0.0f;
    for (unsigned pass = 0; pass < repeats; ++pass) {
        for (unsigned pos = threadIdx.x; pos < BytesPerPartial; pos += BlockThreads) {
            const float x = static_cast<float>(input[row * BytesPerPartial + pos]);
            if (pass == 0) value += x;
        }
    }
    for (unsigned delta = 16; delta != 0; delta >>= 1) value += __shfl_down_sync(0xffffffff, value, delta);
    __shared__ float warp_values[BlockThreads / 32];
    if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < (BlockThreads / 32) ? warp_values[threadIdx.x] : 0.0f;
        for (unsigned delta = 16; delta != 0; delta >>= 1) value += __shfl_down_sync(0xffffffff, value, delta);
        if (threadIdx.x == 0) target[row] = value;
    }
}

__global__ void copyPartials(const float* from, float* to, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) to[i] = from[i];
}

}  // namespace

std::vector<ReductionAutotuning::Candidate> candidates() {
    using Span = const StridedSpan&;
    std::vector<ReductionAutotuning::Candidate> result;
    result.push_back({"complete_fast", 0, [](Span x, Span y, Span, Stream& s) {
                          reduceBytes<<<Blocks, BlockThreads, 0, s.getStream()>>>(x.getMemPtr<uint8_t>(),
                                                                                  y.getMemPtr<float>(), 1);
                      }});
    result.push_back({"staged_fp32_continuation", Partials * sizeof(float),
                      [](Span x, Span y, Span scratch, Stream& s) {
                          auto* partials = static_cast<float*>(scratch.getMemPtr<void>());
                          reduceBytes<<<Blocks, BlockThreads, 0, s.getStream()>>>(x.getMemPtr<uint8_t>(), partials,
                                                                                  1);
                          copyPartials<<<(Blocks + 255) / 256, 256, 0, s.getStream()>>>(partials,
                                                                                         y.getMemPtr<float>(), Blocks);
                      }});
    result.push_back({"complete_slow", 0, [](Span x, Span y, Span, Stream& s) {
                          reduceBytes<<<Blocks, BlockThreads, 0, s.getStream()>>>(x.getMemPtr<uint8_t>(),
                                                                                  y.getMemPtr<float>(), 16);
                      }});
    return result;
}

}  // namespace ThorImplementation::ReductionAutotunerDevice
