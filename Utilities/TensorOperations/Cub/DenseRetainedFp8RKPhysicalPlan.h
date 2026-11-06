#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * Exactly ONE retained original-input FP8 RK physical pass, not the isolated
 * benchmark program that was used to select it. A staged pass writes its FP32
 * [outer,N,inner] result to an ordinary planner-owned output;
 * that output is not scratch and no FP32 completion is prescribed here.
 *
 * This descriptor intentionally contains neither Tensor/Stream ownership nor
 * selected FP32 continuation passes. The normal pass materializer owns launch resources.
 * 1220: input_dtype FP32 (kernel.element_bytes 4) is a retained FP32 RK VALUE first stage at an FP32 site.
 * 1254: input_dtype FP16 / BF16 (kernel.element_bytes 2) is a retained FP16 / BF16 RK VALUE first stage at an original
 * 2-byte site (the retained FP16 / BF16 routing).
 */
struct DenseRetainedFp8RKPhysicalPlan final : DenseReductionPhysicalPlan {
    Fp8RK::PhysicalKernel kernel{};
    DensePassRole role = DensePassRole::First;
    DataType input_dtype = DataType::FP8_E4M3;
    DataType output_dtype = DataType::FP32;
    std::vector<uint64_t> input_dimensions;
    std::vector<uint64_t> output_dimensions;
    uint32_t reduction_axis = 1;
    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;
};

/**
 * Translate one selected FP8 first-stage physical identity into the canonical
 * one-pass candidate consumed by the existing dense planner. This function
 * performs no selection, allocation, timing, or continuation planning.
 *
 * Ordinary production calls this with the cached first-stage FP8 winner from
 * the official descriptor-only selector; tests may inject a physical choice.
 */
[[nodiscard]] DenseReductionCandidate makeRetainedFp8RKPhysicalCandidate(
    const DenseReductionProblem& problem, DenseReductionSite site,
    const Fp8RK::PhysicalKernel& first_stage,
    const DensePhysicalSelectionContext& context);

/**
 * 1220: the same one-pass candidate for a retained FP32 RK VALUE first stage (element_bytes 4) at an FP32 site: original
 * FP32 input, or an intermediate aggregate of any dtype's earlier passes (the reduced run original or intermediate).
 * The plan's input_dtype is FP32; the role is the pass role (Complete / First on original input, Final / Intermediate
 * on an intermediate).
 */
[[nodiscard]] DenseReductionCandidate makeRetainedFp32RKValueCandidate(const DenseReductionProblem& problem,
                                                                      DenseReductionSite site,
                                                                      const Fp8RK::PhysicalKernel& first_stage);

/**
 * 1254: the same one-pass candidate for a retained FP16 / BF16 RK VALUE first stage (element_bytes 2) at an original
 * FP16 / BF16 RK site. The plan's input_dtype is `input_dtype`.
 */
[[nodiscard]] DenseReductionCandidate makeRetainedF16RKValueCandidate(const DenseReductionProblem& problem,
                                                                     DenseReductionSite site,
                                                                     const Fp8RK::PhysicalKernel& first_stage,
                                                                     DataType input_dtype);

}  // namespace ThorImplementation::CubReductionInternal
