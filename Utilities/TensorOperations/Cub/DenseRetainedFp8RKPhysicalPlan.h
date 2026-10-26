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

}  // namespace ThorImplementation::CubReductionInternal
