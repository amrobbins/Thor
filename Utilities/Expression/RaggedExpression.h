#pragma once

#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "Utilities/Expression/Expression.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace ThorImplementation {

struct RaggedExpressionRuntimeExtent {
    Expression activeValueCount = Expression::constantScalar(0.0);
    uint64_t maxActiveValues = 0;
    uint64_t elementsPerValue = 1;

    [[nodiscard]] bool isInitialized() const { return maxActiveValues != 0 && elementsPerValue != 0; }
    [[nodiscard]] uint64_t maxLaunchElements() const;
};

class RaggedExpression {
   public:
    RaggedExpression() = default;
    RaggedExpression(Expression values, Expression offsets, RaggedTensorDescriptor descriptor);

    [[nodiscard]] static RaggedExpression input(const std::string& logical_name, const RaggedTensorDescriptor& descriptor);
    [[nodiscard]] static RaggedExpression input(const std::string& values_name,
                                                const std::string& offsets_name,
                                                const RaggedTensorDescriptor& descriptor);

    [[nodiscard]] bool isInitialized() const { return initialized; }

    [[nodiscard]] const Expression& getValues() const;
    [[nodiscard]] const Expression& getOffsets() const;
    [[nodiscard]] const RaggedTensorDescriptor& getDescriptor() const;
    [[nodiscard]] const RaggedExpressionRuntimeExtent& getRuntimeExtent() const;

    [[nodiscard]] DataType getValuesDataType() const { return getDescriptor().getValuesDataType(); }
    [[nodiscard]] DataType getOffsetsDataType() const { return getDescriptor().getOffsetsDataType(); }
    [[nodiscard]] uint64_t getBatchSize() const { return getDescriptor().getBatchSize(); }
    [[nodiscard]] uint64_t getMaxTotalValues() const { return getDescriptor().getMaxTotalValues(); }
    [[nodiscard]] std::vector<uint64_t> getValuesDimensions() const { return getDescriptor().getValuesDimensions(); }
    [[nodiscard]] std::vector<uint64_t> getTrailingDimensions() const { return getDescriptor().getTrailingDimensions(); }

    [[nodiscard]] std::set<std::string> getInputNames() const;
    [[nodiscard]] std::set<std::string> getMetadataInputNames() const;
    [[nodiscard]] std::set<std::string> getDifferentiableInputNames() const;

    [[nodiscard]] RaggedExpression withValues(Expression new_values, RaggedTensorDescriptor new_descriptor) const;
    [[nodiscard]] RaggedExpression cast(DataType output_dtype) const;

    [[nodiscard]] RaggedExpression operator+(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression operator-(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression operator*(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression operator/(const RaggedExpression& other) const;

    [[nodiscard]] RaggedExpression equal(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression notEqual(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression lessThan(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression lessEqual(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression greaterThan(const RaggedExpression& other) const;
    [[nodiscard]] RaggedExpression greaterEqual(const RaggedExpression& other) const;

    [[nodiscard]] RaggedExpression abs() const;
    [[nodiscard]] RaggedExpression exp() const;
    [[nodiscard]] RaggedExpression ln() const;
    [[nodiscard]] RaggedExpression log() const { return ln(); }
    [[nodiscard]] RaggedExpression relu() const;

    [[nodiscard]] RaggedExpression softmax() const;
    [[nodiscard]] Expression reduce_sum() const;

    [[nodiscard]] Expression segment_sum() const;
    [[nodiscard]] Expression segment_min() const;
    [[nodiscard]] Expression segment_max() const;
    [[nodiscard]] Expression segment_mean() const;
    [[nodiscard]] RaggedExpression segment_softmax() const;
    [[nodiscard]] RaggedExpression segment_log_softmax() const;

   private:
    // `values` is the unwrapped value graph used to compose ragged operations.
    // `executionValues` is an identity marker carrying the device-side logical extent
    // and is the expression exposed for materialization/execution.
    Expression values = Expression::constantScalar(0.0);
    Expression executionValues = Expression::constantScalar(0.0);
    Expression offsets = Expression::constantScalar(0.0);
    RaggedTensorDescriptor descriptor;
    RaggedExpressionRuntimeExtent runtimeExtent;
    bool initialized = false;

    RaggedExpression(Expression values,
                     Expression offsets,
                     RaggedTensorDescriptor descriptor,
                     RaggedExpressionRuntimeExtent runtime_extent);

    [[nodiscard]] RaggedExpression unaryValuewise(ExprOp op, const char* op_name) const;
    [[nodiscard]] RaggedExpression binaryValuewise(const RaggedExpression& other, ExprOp op, const char* op_name) const;
    [[nodiscard]] Expression segmentTotalBroadcast(ScanOp op, const char* op_name) const;

    void validateInitialized(const char* caller) const;
    void validateScalarValues(const char* caller) const;
    static void validateDescriptor(const RaggedTensorDescriptor& descriptor);
    static RaggedExpressionRuntimeExtent makeRuntimeExtent(const Expression& offsets, const RaggedTensorDescriptor& descriptor);
    static Expression markExecutionValues(const Expression& values,
                                          const Expression& offsets,
                                          const RaggedTensorDescriptor& descriptor);
    static uint64_t elementsPerValue(const RaggedTensorDescriptor& descriptor);
    static RaggedTensorDescriptor descriptorWithValuesDataType(const RaggedTensorDescriptor& descriptor, DataType values_dtype);
    static RaggedTensorDescriptor descriptorWithValuesDescriptor(const RaggedTensorDescriptor& descriptor,
                                                                 const TensorDescriptor& values_descriptor);
    static void requireSameOffsetsObject(const RaggedExpression& lhs, const RaggedExpression& rhs, const char* op_name);
    static void requireSameValueShape(const RaggedExpression& lhs, const RaggedExpression& rhs, const char* op_name);
};

[[nodiscard]] RaggedExpression cast(const RaggedExpression& input, DataType output_dtype);
[[nodiscard]] RaggedExpression abs(const RaggedExpression& input);
[[nodiscard]] RaggedExpression exp(const RaggedExpression& input);
[[nodiscard]] RaggedExpression log(const RaggedExpression& input);
[[nodiscard]] RaggedExpression relu(const RaggedExpression& input);
[[nodiscard]] Expression segment_sum(const RaggedExpression& input);
[[nodiscard]] Expression segment_min(const RaggedExpression& input);
[[nodiscard]] Expression segment_max(const RaggedExpression& input);
[[nodiscard]] Expression segment_mean(const RaggedExpression& input);
[[nodiscard]] RaggedExpression segment_softmax(const RaggedExpression& input);
[[nodiscard]] RaggedExpression segment_log_softmax(const RaggedExpression& input);

}  // namespace ThorImplementation
