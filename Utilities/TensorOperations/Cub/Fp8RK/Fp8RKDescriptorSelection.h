#pragma once

// CUDA- and allocation-independent cache-key construction. The exact same
// request key is used whether the caller has an original Tensor or only a
// logical FP8 dtype, canonical problem, site and stream/device identity.
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionKey.h"

#include <array>
#include <bit>
#include <cstdint>

namespace ThorImplementation { enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal::Fp8RK {

// 1171: the shared device identity of the dense reduction selection cache.
using Fp8RKDeviceSelectionIdentity = DenseReductionDeviceIdentity;

[[nodiscard]] inline Fp8RKSelectionKey makeFp8RKDescriptorSelectionKey(
    const DenseReductionProblem& problem, DenseReductionSite site,
    CubReductionOp operation, uint64_t original_reduction_size, float output_scale,
    DataType input_dtype, const Fp8RKDeviceSelectionIdentity& device,
    uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular) {
    Fp8RKSelectionKey key;
    key.device_uuid = device.uuid;
    key.device_ordinal = device.ordinal;
    key.sm_count = device.sm_count;
    key.compute_major = device.compute_major;
    key.compute_minor = device.compute_minor;
    key.workspace_budget_bytes = workspace_budget_bytes;
    key.original_reduction_size = original_reduction_size;
    key.output_scale_bits = std::bit_cast<uint32_t>(output_scale);
    key.site_run_index = site.run_index;
    key.input_dtype = static_cast<uint8_t>(input_dtype);
    key.output_dtype = static_cast<uint8_t>(problem.final_output.value_dtype);
    key.output_index_dtype = static_cast<uint8_t>(problem.final_output.index_dtype);
    key.operation = static_cast<uint8_t>(operation);
    key.aggregate_kind = static_cast<uint8_t>(problem.aggregate_kind);
    key.input_state = static_cast<uint8_t>(problem.input_state);
    key.carried_arg_index_dtype = static_cast<uint8_t>(problem.carried_arg_index_dtype);
    key.produce_value = problem.final_output.produce_value;
    key.produce_index = problem.final_output.produce_index;
    key.finalizer_pending = problem.finalizer_pending;
    key.tuning_depth = static_cast<uint8_t>(tuning_depth);
    key.runs.reserve(problem.runs.size());
    for (const auto& run : problem.runs) {
        key.runs.push_back({run.extent, run.original_arg_domain_stride,
                            static_cast<uint8_t>(run.kind)});
    }
    return key;
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
