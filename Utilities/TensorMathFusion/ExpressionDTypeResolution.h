#pragma once

#include <vector>

#include "Utilities/TensorMathFusion/Expression.h"

namespace ThorImplementation {

using DataType = TensorDescriptor::DataType;

bool isSupportedFusionFloatingType(DataType dtype);
bool isFp8Type(DataType dtype);

DataType toSupportedComputeDType(ExprOp op, DataType requested_compute_dtype);
DataType toSupportedInputDType(ExprOp op, DataType dtype);

DataType defaultComputeDType(DataType value_dtype);
DataType defaultComputeDType(DataType input_dtype, DataType output_dtype);
DataType promoteTensorValueDTypes(DataType a, DataType b);
DataType promoteTensorValueDTypes(const std::vector<DataType>& dtypes);

void resolveExpressionDTypesInPlace(PhysicalExpression& expr, const std::vector<DataType>& root_input_dtypes);

void resolveOutputsDTypesInPlace(PhysicalOutputs& outputs, const std::vector<DataType>& root_input_dtypes);

}  // namespace ThorImplementation
