#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kRaggedGatherPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Gather source values with row-local UINT32/UINT64 indices. source_offsets
// defines source partition P and indices_offsets defines destination partition
// Q. output_values has Q's packed capacity and source trailing geometry. Only
// active indices in [0, Q[B]) are read. An index is relative to its source row.
void launchRaggedGather(const Tensor& source_values,
                        const Tensor& source_offsets,
                        const Tensor& indices_values,
                        const Tensor& indices_offsets,
                        Tensor& output_values,
                        uint64_t batch_size,
                        Stream& stream);

// Backward for row-local gather. Each source row is zeroed cooperatively, then
// its active output gradients scatter-add across the trailing value width. A
// single-index row writes directly; rows with multiple indices use atomics
// because row-local indices may repeat. Inactive source-gradient capacity is
// untouched. Backward supports FP16, BF16, and FP32 gradients.
void launchRaggedGatherBackward(const Tensor& source_offsets,
                                const Tensor& indices_values,
                                const Tensor& indices_offsets,
                                const Tensor& output_gradient,
                                Tensor& source_gradient,
                                uint64_t batch_size,
                                Stream& stream);

// Benchmark-only launch metadata and forcing hooks. These call the production
// kernels and preserve their semantics; they exist only so forced-vs-auto
// sweeps can measure the current copy-width/lane choices without duplicating
// kernel code in benchmarks. A forced value of 0 means production auto.
struct RaggedGatherForwardBenchmarkLaunchInfo {
    uint32_t blocks;
    uint32_t copy_width_bytes;
    uint32_t lanes_per_token;
    uint32_t tokens_per_block;
    uint64_t copy_items_per_value;
};

struct RaggedGatherBackwardBenchmarkLaunchInfo {
    uint32_t blocks;
    uint32_t lanes_per_token;
    uint32_t tokens_per_block;
    uint64_t elements_per_value;
};

RaggedGatherForwardBenchmarkLaunchInfo raggedGatherForwardLaunchInfoForBenchmark(
    const Tensor& source_values,
    const Tensor& output_values,
    uint64_t batch_size,
    uint32_t forced_copy_width_bytes = 0,
    uint32_t forced_lanes_per_token = 0);

RaggedGatherBackwardBenchmarkLaunchInfo raggedGatherBackwardLaunchInfoForBenchmark(
    const Tensor& source_gradient,
    uint64_t batch_size,
    uint32_t forced_lanes_per_token = 0);

void launchRaggedGatherForBenchmark(const Tensor& source_values,
                                    const Tensor& source_offsets,
                                    const Tensor& indices_values,
                                    const Tensor& indices_offsets,
                                    Tensor& output_values,
                                    uint64_t batch_size,
                                    uint32_t forced_copy_width_bytes,
                                    uint32_t forced_lanes_per_token,
                                    Stream& stream);

void launchRaggedGatherBackwardForBenchmark(const Tensor& source_offsets,
                                            const Tensor& indices_values,
                                            const Tensor& indices_offsets,
                                            const Tensor& output_gradient,
                                            Tensor& source_gradient,
                                            uint64_t batch_size,
                                            uint32_t forced_lanes_per_token,
                                            Stream& stream);

}  // namespace ThorImplementation
