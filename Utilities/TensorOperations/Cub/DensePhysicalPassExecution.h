#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * Physical storage for ONE already selected ordinary-dense VALUE pass.
 *
 * intermediate_dimensions is the actual nonterminal output allocation shape, not the canonical planner shape:
 * staged RK emits [outer, shards, inner] FP32, while a subsequent pass may alias/reshape that storage to its own
 * input_dimensions. The caller chooses the final dtype and shape when this is the last mathematical pass.
 * No Tensor is constructed, no kernel is selected, and no workspace is allocated by this descriptor query.
 */
struct DensePhysicalPassLayout {
    std::vector<uint64_t> input_dimensions;
    std::vector<uint64_t> intermediate_dimensions;
    CubReductionGeometry geometry;
};

[[nodiscard]] DensePhysicalPassLayout describeDensePhysicalPass(const DenseReductionPhysicalPlan& physical_plan);

/**
 * Physical scratch only; does not include the pass's FP32 intermediate output allocation.
 * A modern RK kernel has zero separately allocated scratch, including its staged partial output (which is an output,
 * not scratch). For direct passes, query the exact existing production CUB/Thor reducer and stage semantics.
 * input/output may be null for descriptor-only queries, just as in the original production query.
 */
[[nodiscard]] size_t queryDensePhysicalPassScratchBytes(const DenseReductionPhysicalPlan& physical_plan,
                                                       const CubReductionStageSemantics& semantics,
                                                       DataType input_dtype,
                                                       const void* input,
                                                       uint64_t input_elements,
                                                       DataType output_dtype,
                                                       void* output,
                                                       const CubReductionGeometry& geometry,
                                                       float output_scale,
                                                       const Stream& stream);

/**
 * Executes exactly ONE previously selected VALUE physical pass using the existing production implementation.
 * The caller owns/binds all Tensor handles and queried scratch. When the reported scratch size is zero,
 * temp_storage may be an uninitialized Tensor; scratch-free kernels must not dereference it.
 * There is no selection, planning, allocation or synchronization here.
 */
void launchDensePhysicalPass(const DenseReductionPhysicalPlan& physical_plan,
                             const CubReductionStageSemantics& semantics,
                             const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& input,
                             Tensor& output,
                             const CubReductionGeometry& geometry,
                             float output_scale,
                             Stream& stream);

/**
 * Materialize an already selected ordinary VALUE physical-pass sequence into
 * independently owned stamp resources. Does not select kernels or tune.
 * Both retained FP8 first stages and subsequent FP32 stages use this
 * ordinary production pass-list materialization.
 */
[[nodiscard]] DenseExecutablePlan materializeDenseValueExecutablePlan(
    CubReductionOp op, const Tensor& input, const Tensor& output,
    const DenseReductionPlan& plan, uint64_t semantic_total_reduction_size,
    float output_scale, const Stream& stream);

}  // namespace ThorImplementation::CubReductionInternal
