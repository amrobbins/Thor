#pragma once

// The FP8 RK selector's site descriptor. 1171: entries live in the one dense reduction selection cache
// (DenseReductionSelectionCache.h); toDenseReductionSelectionKey turns this descriptor into that cache's key.
// CUDA-independent.
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

struct Fp8RKReductionRunKey {
    uint64_t extent = 0;
    uint64_t original_arg_domain_stride = 0;
    uint8_t kind = 0;
    bool operator==(const Fp8RKReductionRunKey&) const = default;
};

// Include every parameter that can change the physical shortlist, the
// continuation, the measurement, or the legality of a selected kernel.
// Device ordinal alone is insufficient: device identity and resources matter.
struct Fp8RKSelectionKey {
    std::array<uint8_t, 16> device_uuid{};
    int32_t device_ordinal = -1;
    uint32_t sm_count = 0;
    uint32_t compute_major = 0;
    uint32_t compute_minor = 0;
    uint64_t workspace_budget_bytes = 0;
    uint64_t original_reduction_size = 0;
    uint32_t output_scale_bits = 0;
    uint32_t site_run_index = 0;
    uint8_t input_dtype = 0;
    uint8_t output_dtype = 0;
    uint8_t output_index_dtype = 0;
    uint8_t operation = 0;
    uint8_t aggregate_kind = 0;
    uint8_t input_state = 0;
    uint8_t carried_arg_index_dtype = 0;
    bool produce_value = true;
    bool produce_index = false;
    bool finalizer_pending = true;
    uint8_t tuning_depth = 0;  // 1141: CubReductionTuningDepth the entry was tuned at
    std::vector<Fp8RKReductionRunKey> runs;
    bool operator==(const Fp8RKSelectionKey&) const = default;
};

/** 1171: the dense reduction selection cache key of an RK descriptor (kind Fp8RK; every field above). */
[[nodiscard]] inline DenseReductionSelectionKey toDenseReductionSelectionKey(const Fp8RKSelectionKey& rk) {
    DenseReductionSelectionKey key;
    key.kind = DenseReductionSelectionKind::Fp8RK;
    key.device.uuid = rk.device_uuid;
    key.device.ordinal = rk.device_ordinal;
    key.device.sm_count = rk.sm_count;
    key.device.compute_major = rk.compute_major;
    key.device.compute_minor = rk.compute_minor;
    key.tuning_depth = static_cast<CubReductionTuningDepth>(rk.tuning_depth);
    key.site = {rk.workspace_budget_bytes,
                rk.original_reduction_size,
                rk.output_scale_bits,
                rk.site_run_index,
                rk.input_dtype,
                rk.output_dtype,
                rk.output_index_dtype,
                rk.operation,
                rk.aggregate_kind,
                rk.input_state,
                rk.carried_arg_index_dtype,
                static_cast<uint64_t>(rk.produce_value),
                static_cast<uint64_t>(rk.produce_index),
                static_cast<uint64_t>(rk.finalizer_pending),
                rk.runs.size()};
    for (const Fp8RKReductionRunKey& run : rk.runs) {
        key.site.push_back(run.extent);
        key.site.push_back(run.original_arg_domain_stride);
        key.site.push_back(run.kind);
    }
    return key;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
