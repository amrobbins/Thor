#pragma once

#include <optional>
#include <vector>

#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

bool isSupportedFusionFloatingType(DataType dtype);
bool isFp8Type(DataType dtype);

DataType toSupportedComputeDType(ExprOp op, DataType requested_compute_dtype);
DataType toSupportedInputDType(ExprOp op, DataType dtype);

DataType defaultComputeDType(DataType value_dtype);
DataType defaultComputeDType(DataType input_dtype, DataType output_dtype);
DataType promoteTensorValueDTypes(DataType a, DataType b);
DataType promoteTensorValueDTypes(const std::vector<DataType>& dtypes);

// Return the dtype of the tensor storage physically backing a resolved value.
// Passthrough views inherit their source storage; INPUT may expose a promoted
// logical output dtype while still being backed by a lower-precision tensor.
std::optional<DataType> materializedValueStorageDType(const PhysicalExpression& expr, uint32_t node_idx);

void resolveExpressionDTypesInPlace(PhysicalExpression& expr, const std::vector<DataType>& root_input_dtypes);

void resolveOutputsDTypesInPlace(PhysicalOutputs& outputs, const std::vector<DataType>& root_input_dtypes);

}  // namespace ThorImplementation
