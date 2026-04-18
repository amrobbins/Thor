#pragma once

#include <vector>

#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

bool isSupportedFusionFloatingType(TensorDescriptor::DataType dtype);
bool isFp8Type(TensorDescriptor::DataType dtype);

TensorDescriptor::DataType toSupportedComputeDType(ExprOp op, TensorDescriptor::DataType requested_compute_dtype);
TensorDescriptor::DataType toSupportedInputDType(ExprOp op, TensorDescriptor::DataType dtype);

TensorDescriptor::DataType defaultComputeDType(TensorDescriptor::DataType value_dtype);
TensorDescriptor::DataType defaultComputeDType(TensorDescriptor::DataType input_dtype, TensorDescriptor::DataType output_dtype);
TensorDescriptor::DataType promoteTensorValueDTypes(TensorDescriptor::DataType a, TensorDescriptor::DataType b);
TensorDescriptor::DataType promoteTensorValueDTypes(const std::vector<TensorDescriptor::DataType>& dtypes);

void resolveExpressionDTypesInPlace(PhysicalExpression& expr, const std::vector<TensorDescriptor::DataType>& root_input_dtypes);

void resolveOutputsDTypesInPlace(PhysicalOutputs& outputs, const std::vector<TensorDescriptor::DataType>& root_input_dtypes);

}  // namespace ThorImplementation
