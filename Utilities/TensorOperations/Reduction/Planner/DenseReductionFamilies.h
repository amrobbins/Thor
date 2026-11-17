#pragma once

#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanning.h"

#include <cstdint>

namespace ThorImplementation::ReductionInternal {

// Every dense site of every supported dtype is selected by a per-dtype selector (DenseReductionPhysicalSelector.h).
// What remains here are the facts those selectors and the planner's test hook read.

/** VALUE operation of the reduction (the selectors and their contexts carry it). */
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

}  // namespace ThorImplementation::ReductionInternal
