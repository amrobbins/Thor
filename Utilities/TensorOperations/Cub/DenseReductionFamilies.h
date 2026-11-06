#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal {

// 1258 (plans/Thor_FP16_BF16_Dense_Reduction_Plan, phase 10): the legacy ordinary dense families
// (ReducersDenseR / KR / RK, their KParallel / RCooperative physical plans and the direct CUB / tiled plan) are gone.
// Every dense site of every supported dtype is selected by a retained selector (DenseReductionPhysicalSelector.h).
// What remains here are the facts those selectors and the planner's test hook read.

/** VALUE operation of the reduction (the retained selectors and their contexts carry it). */
enum class DenseRKValueOperation : uint8_t {
    Sum = 0,
    Mean,
    L1,
    L2,
    SumSquares,
    Product,
    Minimum,
    Maximum,
};

/**
 * Device and input facts for dense physical selection. original_input_dtype is consulted only while
 * problem.input_state == OriginalInput; every reducer-produced VALUE partial is FP32.
 */
struct DenseRKPlanningContext {
    DataType original_input_dtype = DataType::FP32;
    DenseRKValueOperation value_operation = DenseRKValueOperation::Sum;
    uint32_t multiprocessors = 1;
};

}  // namespace ThorImplementation::CubReductionInternal
