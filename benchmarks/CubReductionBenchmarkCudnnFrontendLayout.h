#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ThorImplementation::CubReductionBenchmarking {

/**
 * Benchmark-only packed NCHW -> packed NHWC reformat.
 *
 * The cuDNN Frontend runtime reduction engine used by the benchmark requires
 * NHWC-packed input. Thor's production tensor for the R|K|R census case is
 * packed NCHW, so the end-to-end cuDNN candidate must pay this conversion.
 * element_bytes may be 2 (FP16/BF16) or 4 (FP32).
 */
void launchPackedNchwToNhwc(const void* source,
                            void* destination,
                            uint32_t element_bytes,
                            uint64_t n,
                            uint64_t c,
                            uint64_t h,
                            uint64_t w,
                            cudaStream_t stream);

}  // namespace ThorImplementation::CubReductionBenchmarking
