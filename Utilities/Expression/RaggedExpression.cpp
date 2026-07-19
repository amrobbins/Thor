#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <string>
#include <stdexcept>
#include <utility>

namespace ThorImplementation {
namespace {

uint64_t checkedMul(uint64_t a, uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t.");
    }
    return a * b;
}

std::string raggedOpErrorPrefix(const char* op_name) { return std::string("RaggedExpression::") + op_name + ": "; }

}  // namespace

uint64_t RaggedExpressionRuntimeExtent::maxLaunchElements() const {
    if (!isInitialized()) {
        throw std::runtime_error("RaggedExpressionRuntimeExtent is not initialized.");
    }
    return checkedMul(maxActiveValues, elementsPerValue, "ragged expression runtime extent max launch elements");
}

RaggedExpression::RaggedExpression(Expression values, Expression offsets, RaggedTensorDescriptor descriptor)
    : values(std::move(values)), offsets(std::move(offsets)), descriptor(std::move(descriptor)) {
    validateDescriptor(this->descriptor);
    runtimeExtent = makeRuntimeExtent(this->offsets, this->descriptor);
    executionValues = markExecutionValues(this->values, this->offsets, this->descriptor);
    initialized = true;
}

RaggedExpression::RaggedExpression(Expression values,
                                   Expression offsets,
                                   RaggedTensorDescriptor descriptor,
                                   RaggedExpressionRuntimeExtent runtime_extent)
    : values(std::move(values)), offsets(std::move(offsets)), descriptor(std::move(descriptor)), runtimeExtent(std::move(runtime_extent)) {
    validateDescriptor(this->descriptor);
    if (!runtimeExtent.isInitialized()) {
        throw std::invalid_argument("RaggedExpression requires an initialized runtime extent.");
    }
    if (runtimeExtent.maxActiveValues != this->descriptor.getMaxTotalValues()) {
        throw std::invalid_argument("RaggedExpression runtime extent maxActiveValues must match descriptor maxTotalValues.");
    }
    if (runtimeExtent.elementsPerValue != elementsPerValue(this->descriptor)) {
        throw std::invalid_argument("RaggedExpression runtime extent elementsPerValue must match descriptor trailing dimensions.");
    }
    executionValues = markExecutionValues(this->values, this->offsets, this->descriptor);
    initialized = true;
}

RaggedExpression RaggedExpression::input(const std::string& logical_name, const RaggedTensorDescriptor& descriptor) {
    if (logical_name.empty()) {
        throw std::invalid_argument("RaggedExpression::input requires a non-empty logical input name.");
    }
    return input(logical_name + ".values", logical_name + ".offsets", descriptor);
}

RaggedExpression RaggedExpression::input(const std::string& values_name,
                                         const std::string& offsets_name,
                                         const RaggedTensorDescriptor& descriptor) {
    if (values_name.empty() || offsets_name.empty()) {
        throw std::invalid_argument("RaggedExpression::input requires non-empty values and offsets input names.");
    }
    validateDescriptor(descriptor);
    return RaggedExpression(Expression::input(values_name, std::nullopt, descriptor.getValuesDataType()),
                            Expression::input(offsets_name, std::nullopt, descriptor.getOffsetsDataType()),
                            descriptor);
}

const Expression& RaggedExpression::getValues() const {
    validateInitialized("getValues");
    return executionValues;
}

const Expression& RaggedExpression::getOffsets() const {
    validateInitialized("getOffsets");
    return offsets;
}

const RaggedTensorDescriptor& RaggedExpression::getDescriptor() const {
    validateInitialized("getDescriptor");
    return descriptor;
}

const RaggedExpressionRuntimeExtent& RaggedExpression::getRuntimeExtent() const {
    validateInitialized("getRuntimeExtent");
    return runtimeExtent;
}

std::set<std::string> RaggedExpression::getInputNames() const {
    validateInitialized("getInputNames");
    std::set<std::string> names = executionValues.getInputNames();
    const std::set<std::string> metadata_names = getMetadataInputNames();
    names.insert(metadata_names.begin(), metadata_names.end());
    return names;
}

std::set<std::string> RaggedExpression::getMetadataInputNames() const {
    validateInitialized("getMetadataInputNames");
    return offsets.getInputNames();
}

std::set<std::string> RaggedExpression::getDifferentiableInputNames() const {
    validateInitialized("getDifferentiableInputNames");
    std::set<std::string> names = values.getInputNames();
    for (const std::string& metadata_name : getMetadataInputNames()) {
        names.erase(metadata_name);
    }
    return names;
}

RaggedExpression RaggedExpression::withValues(Expression new_values, RaggedTensorDescriptor new_descriptor) const {
    validateInitialized("withValues");
    validateDescriptor(new_descriptor);
    if (new_descriptor.getRowPartition() != descriptor.getRowPartition()) {
        throw std::invalid_argument("RaggedExpression::withValues requires the row partition to be preserved.");
    }
    return RaggedExpression(std::move(new_values), offsets, std::move(new_descriptor), runtimeExtent);
}

RaggedExpression RaggedExpression::cast(DataType output_dtype) const {
    validateInitialized("cast");
    return withValues(values.cast(output_dtype), descriptorWithValuesDataType(descriptor, output_dtype));
}

RaggedExpression RaggedExpression::operator+(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::ADD, "operator+"); }
RaggedExpression RaggedExpression::operator-(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::SUB, "operator-"); }
RaggedExpression RaggedExpression::operator*(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::MUL, "operator*"); }
RaggedExpression RaggedExpression::operator/(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::DIV, "operator/"); }

RaggedExpression RaggedExpression::equal(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::EQUAL, "equal"); }
RaggedExpression RaggedExpression::notEqual(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::NOT_EQUAL, "notEqual"); }
RaggedExpression RaggedExpression::lessThan(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::LESS, "lessThan"); }
RaggedExpression RaggedExpression::lessEqual(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::LESS_EQUAL, "lessEqual"); }
RaggedExpression RaggedExpression::greaterThan(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::GREATER, "greaterThan"); }
RaggedExpression RaggedExpression::greaterEqual(const RaggedExpression& other) const { return binaryValuewise(other, ExprOp::GREATER_EQUAL, "greaterEqual"); }

RaggedExpression RaggedExpression::abs() const { return unaryValuewise(ExprOp::ABS, "abs"); }
RaggedExpression RaggedExpression::exp() const { return unaryValuewise(ExprOp::EXP, "exp"); }
RaggedExpression RaggedExpression::ln() const { return unaryValuewise(ExprOp::LN, "ln"); }

RaggedExpression RaggedExpression::relu() const {
    validateInitialized("relu");
    return withValues(values.max(Expression::constantScalar(0.0)), descriptor);
}

RaggedExpression RaggedExpression::softmax() const { return segment_softmax(); }

Expression RaggedExpression::reduce_sum() const { return segment_sum(); }

Expression RaggedExpression::segment_sum() const {
    validateInitialized("segment_sum");
    validateScalarValues("segment_sum");
    return values.segmentedReduceSum(offsets);
}

Expression RaggedExpression::segment_min() const {
    validateInitialized("segment_min");
    validateScalarValues("segment_min");
    return values.segmentedReduceMin(offsets);
}

Expression RaggedExpression::segment_max() const {
    validateInitialized("segment_max");
    validateScalarValues("segment_max");
    return values.segmentedReduceMax(offsets);
}

Expression RaggedExpression::segment_mean() const {
    validateInitialized("segment_mean");
    validateScalarValues("segment_mean");

    switch (descriptor.getValuesDataType()) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
        case DataType::FP64:
            break;
        default:
            throw std::invalid_argument("RaggedExpression::segment_mean requires floating-point ragged values.");
    }

    const uint64_t batch_size = descriptor.getBatchSize();
    const Expression offsets_values_dtype = offsets.cast(descriptor.getValuesDataType());
    const Expression start_offsets = offsets_values_dtype.stridedView({batch_size}, {1}, 0);
    const Expression end_offsets = offsets_values_dtype.stridedView({batch_size}, {1}, 1);
    const Expression lengths = end_offsets - start_offsets;
    const Expression sums = segment_sum();
    return Expression::where(lengths.greaterThan(Expression::constantScalar(0.0)), sums / lengths, Expression::constantScalar(0.0));
}

RaggedExpression RaggedExpression::segment_softmax() const {
    validateInitialized("segment_softmax");
    validateScalarValues("segment_softmax");
    const Expression max_values = segmentTotalBroadcast(ScanOp::Max, "segment_softmax");
    const Expression shifted = values - max_values;
    const Expression exp_values = shifted.exp();
    const RaggedExpression exp_ragged = withValues(exp_values, descriptor);
    const Expression denom = exp_ragged.segmentTotalBroadcast(ScanOp::Sum, "segment_softmax");
    return withValues(exp_values / denom, descriptor);
}

RaggedExpression RaggedExpression::segment_log_softmax() const {
    validateInitialized("segment_log_softmax");
    validateScalarValues("segment_log_softmax");
    const Expression max_values = segmentTotalBroadcast(ScanOp::Max, "segment_log_softmax");
    const Expression shifted = values - max_values;
    const Expression exp_values = shifted.exp();
    const RaggedExpression exp_ragged = withValues(exp_values, descriptor);
    const Expression denom = exp_ragged.segmentTotalBroadcast(ScanOp::Sum, "segment_log_softmax");
    return withValues(shifted - denom.ln(), descriptor);
}

RaggedExpression RaggedExpression::unaryValuewise(ExprOp op, const char* op_name) const {
    validateInitialized(op_name);

    switch (op) {
        case ExprOp::ABS:
            return withValues(values.abs(), descriptor);
        case ExprOp::EXP:
            return withValues(values.exp(), descriptor);
        case ExprOp::LN:
            return withValues(values.ln(), descriptor);
        default:
            throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "op is not supported as a ragged valuewise unary op.");
    }
}

RaggedExpression RaggedExpression::binaryValuewise(const RaggedExpression& other, ExprOp op, const char* op_name) const {
    validateInitialized(op_name);
    other.validateInitialized(op_name);
    if (!Expression::isBinaryOp(op)) {
        throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "requested op is not binary.");
    }
    requireSameOffsetsObject(*this, other, op_name);
    requireSameValueShape(*this, other, op_name);

    switch (op) {
        case ExprOp::ADD:
            return withValues(values + other.values, descriptor);
        case ExprOp::SUB:
            return withValues(values - other.values, descriptor);
        case ExprOp::MUL:
            return withValues(values * other.values, descriptor);
        case ExprOp::DIV:
            return withValues(values / other.values, descriptor);
        case ExprOp::EQUAL:
            return withValues(values.equal(other.values), descriptorWithValuesDataType(descriptor, DataType::BOOLEAN));
        case ExprOp::NOT_EQUAL:
            return withValues(values.notEqual(other.values), descriptorWithValuesDataType(descriptor, DataType::BOOLEAN));
        case ExprOp::LESS:
            return withValues(values.lessThan(other.values), descriptorWithValuesDataType(descriptor, DataType::BOOLEAN));
        case ExprOp::LESS_EQUAL:
            return withValues(values.lessEqual(other.values), descriptorWithValuesDataType(descriptor, DataType::BOOLEAN));
        case ExprOp::GREATER:
            return withValues(values.greaterThan(other.values), descriptorWithValuesDataType(descriptor, DataType::BOOLEAN));
        case ExprOp::GREATER_EQUAL:
            return withValues(values.greaterEqual(other.values), descriptorWithValuesDataType(descriptor, DataType::BOOLEAN));
        default:
            throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "op is not supported as a ragged valuewise binary op.");
    }
}

Expression RaggedExpression::segmentTotalBroadcast(ScanOp op, const char* op_name) const {
    validateInitialized(op_name);
    validateScalarValues(op_name);
    switch (op) {
        case ScanOp::Sum: {
            const Expression prefix = values.segmentedScan(offsets, ScanOp::Sum, true, false);
            const Expression suffix = values.segmentedScan(offsets, ScanOp::Sum, true, true);
            return prefix + suffix - values;
        }
        case ScanOp::Min:
        case ScanOp::Max: {
            const Expression prefix = values.segmentedScan(offsets, op, true, false);
            return prefix.segmentedScan(offsets, op, true, true);
        }
        default:
            throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "cannot broadcast this segment scan op.");
    }
}

void RaggedExpression::validateInitialized(const char* caller) const {
    if (!initialized) {
        throw std::runtime_error(raggedOpErrorPrefix(caller) + "ragged expression is not initialized.");
    }
}

void RaggedExpression::validateScalarValues(const char* caller) const {
    validateInitialized(caller);
    if (!descriptor.getTrailingDimensions().empty()) {
        throw std::invalid_argument(raggedOpErrorPrefix(caller) +
                                    "currently supports only scalar ragged values with values shape [max_total_values].");
    }
}

void RaggedExpression::validateDescriptor(const RaggedTensorDescriptor& descriptor) {
    // Reconstructing validates the descriptor invariants and keeps future descriptor changes centralized.
    (void)RaggedTensorDescriptor(descriptor.getValuesDescriptor(), descriptor.getRowPartition(), descriptor.getRaggedRank());
}

RaggedExpressionRuntimeExtent RaggedExpression::makeRuntimeExtent(const Expression& offsets, const RaggedTensorDescriptor& descriptor) {
    validateDescriptor(descriptor);
    RaggedExpressionRuntimeExtent extent;
    extent.activeValueCount = offsets.stridedView({1}, {1}, descriptor.getBatchSize());
    extent.maxActiveValues = descriptor.getMaxTotalValues();
    extent.elementsPerValue = elementsPerValue(descriptor);
    return extent;
}

Expression RaggedExpression::markExecutionValues(const Expression& values,
                                                 const Expression& offsets,
                                                 const RaggedTensorDescriptor& descriptor) {
    return values.withRaggedRuntimeExtent(
        offsets, descriptor.getBatchSize(), descriptor.getMaxTotalValues(), elementsPerValue(descriptor));
}

uint64_t RaggedExpression::elementsPerValue(const RaggedTensorDescriptor& descriptor) {
    uint64_t elements = 1;
    const std::vector<uint64_t> trailing_dimensions = descriptor.getTrailingDimensions();
    for (uint64_t dim : trailing_dimensions) {
        elements = checkedMul(elements, dim, "ragged expression elementsPerValue");
    }
    return elements;
}

RaggedTensorDescriptor RaggedExpression::descriptorWithValuesDataType(const RaggedTensorDescriptor& descriptor, DataType values_dtype) {
    return descriptorWithValuesDescriptor(descriptor, TensorDescriptor(values_dtype, descriptor.getValuesDimensions()));
}

RaggedTensorDescriptor RaggedExpression::descriptorWithValuesDescriptor(const RaggedTensorDescriptor& descriptor,
                                                                        const TensorDescriptor& values_descriptor) {
    return RaggedTensorDescriptor(values_descriptor, descriptor.getRowPartition(), descriptor.getRaggedRank());
}

void RaggedExpression::requireSameOffsetsObject(const RaggedExpression& lhs, const RaggedExpression& rhs, const char* op_name) {
    if (!lhs.offsets.isSameLogicalNode(rhs.offsets)) {
        throw std::invalid_argument(raggedOpErrorPrefix(op_name) +
                                    "binary ragged valuewise ops require the exact same offsets expression object.");
    }
}

void RaggedExpression::requireSameValueShape(const RaggedExpression& lhs, const RaggedExpression& rhs, const char* op_name) {
    if (lhs.descriptor.getRowPartition() != rhs.descriptor.getRowPartition()) {
        throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "ragged row partitions differ.");
    }
    if (lhs.descriptor.getValuesDimensions() != rhs.descriptor.getValuesDimensions()) {
        throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "ragged values shapes differ.");
    }
    if (lhs.descriptor.getValuesDataType() != rhs.descriptor.getValuesDataType()) {
        throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "ragged values dtypes differ.");
    }
}

RaggedExpression cast(const RaggedExpression& input, DataType output_dtype) { return input.cast(output_dtype); }
RaggedExpression abs(const RaggedExpression& input) { return input.abs(); }
RaggedExpression exp(const RaggedExpression& input) { return input.exp(); }
RaggedExpression log(const RaggedExpression& input) { return input.log(); }
RaggedExpression relu(const RaggedExpression& input) { return input.relu(); }
Expression segment_sum(const RaggedExpression& input) { return input.segment_sum(); }
Expression segment_min(const RaggedExpression& input) { return input.segment_min(); }
Expression segment_max(const RaggedExpression& input) { return input.segment_max(); }
Expression segment_mean(const RaggedExpression& input) { return input.segment_mean(); }
RaggedExpression segment_softmax(const RaggedExpression& input) { return input.segment_softmax(); }
RaggedExpression segment_log_softmax(const RaggedExpression& input) { return input.segment_log_softmax(); }

}  // namespace ThorImplementation
