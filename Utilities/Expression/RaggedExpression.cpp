#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

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

    // The number of packed rows is a property of the row partition and therefore remains
    // identical across a ragged-preserving value transform.  The amount of work per row,
    // however, belongs to the values descriptor and may change (for example after slicing
    // [max_values, 2H] down to [max_values, H]).  Preserve the exact active-row-count
    // expression while rebuilding the launch width from the new trailing shape.
    RaggedExpressionRuntimeExtent new_runtime_extent = runtimeExtent;
    new_runtime_extent.elementsPerValue = elementsPerValue(new_descriptor);
    return RaggedExpression(std::move(new_values), offsets, std::move(new_descriptor), std::move(new_runtime_extent));
}

RaggedExpression RaggedExpression::mapValues(const std::function<Expression(const Expression&)>& mapper) const {
    validateInitialized("mapValues");
    if (!mapper) {
        throw std::invalid_argument("RaggedExpression::mapValues requires a valid mapper.");
    }
    return withValues(mapper(values), descriptor);
}

RaggedExpression RaggedExpression::sliceTrailingDimension(uint64_t trailing_axis, uint64_t start, uint64_t length) const {
    validateInitialized("sliceTrailingDimension");

    const std::vector<uint64_t> trailing_dimensions = descriptor.getTrailingDimensions();
    if (trailing_axis >= trailing_dimensions.size()) {
        throw std::invalid_argument("RaggedExpression::sliceTrailingDimension trailing axis is out of range.");
    }
    if (length == 0) {
        throw std::invalid_argument("RaggedExpression::sliceTrailingDimension requires a non-zero slice length.");
    }

    const uint64_t source_axis_size = trailing_dimensions[trailing_axis];
    if (start >= source_axis_size || length > source_axis_size - start) {
        throw std::invalid_argument("RaggedExpression::sliceTrailingDimension slice exceeds the source trailing dimension.");
    }

    std::vector<uint64_t> view_dimensions = descriptor.getValuesDimensions();
    std::vector<uint64_t> view_strides(view_dimensions.size(), 1);
    Expression view_source = values;
    uint64_t base_element_offset = 0;

    // Expression::stridedView strides are storage strides, not strides in the logical
    // index space of an immediately preceding view.  Consequently, consecutive ragged
    // trailing slices must be composed before they reach the runtime alias machinery;
    // otherwise a second view would replace (rather than compose) the first view's row
    // stride.  Collapse a directly preceding STRIDED_VIEW into the new view by reusing
    // its storage strides and accumulated storage offset.  This also gives autodiff one
    // canonical scatter from the final view directly back into the original source.
    if (values.expr && values.nodeIndex < values.expr->nodes.size() &&
        values.expr->nodes.at(values.nodeIndex).op == ExprOp::STRIDED_VIEW) {
        const ExprNode& prior_view = values.expr->nodes.at(values.nodeIndex);
        if (prior_view.view_dims != view_dimensions || prior_view.view_strides.size() != view_dimensions.size() ||
            prior_view.lhs == UINT32_MAX || prior_view.lhs >= values.expr->nodes.size()) {
            throw std::runtime_error("RaggedExpression::sliceTrailingDimension encountered an invalid preceding strided view.");
        }
        view_strides = prior_view.view_strides;
        base_element_offset = prior_view.view_element_offset;
        view_source = Expression::fromPhysicalNode(values.expr, prior_view.lhs);
    } else {
        uint64_t source_stride = 1;
        for (size_t axis = view_dimensions.size(); axis-- > 0;) {
            view_strides[axis] = source_stride;
            source_stride = checkedMul(source_stride, view_dimensions[axis], "ragged trailing slice source stride");
        }
    }

    const size_t values_axis = static_cast<size_t>(trailing_axis) + 1;
    const uint64_t slice_element_offset = checkedMul(start, view_strides[values_axis], "ragged trailing slice element offset");
    if (base_element_offset > std::numeric_limits<uint64_t>::max() - slice_element_offset) {
        throw std::invalid_argument("ragged trailing slice element offset overflows uint64_t.");
    }
    const uint64_t element_offset = base_element_offset + slice_element_offset;
    view_dimensions[values_axis] = length;

    const Expression sliced_values = view_source.stridedView(view_dimensions, view_strides, element_offset);
    const RaggedTensorDescriptor sliced_descriptor(
        TensorDescriptor(descriptor.getValuesDataType(), view_dimensions), descriptor.getRowPartition(), descriptor.getRaggedRank());
    return withValues(sliced_values, sliced_descriptor);
}

RaggedExpression RaggedExpression::sliceLastDimension(uint64_t start, uint64_t length) const {
    validateInitialized("sliceLastDimension");
    const std::vector<uint64_t> trailing_dimensions = descriptor.getTrailingDimensions();
    if (trailing_dimensions.empty()) {
        throw std::invalid_argument("RaggedExpression::sliceLastDimension requires at least one trailing value dimension.");
    }
    return sliceTrailingDimension(trailing_dimensions.size() - 1, start, length);
}

RaggedExpression RaggedExpression::transposeTrailingDimensions() const {
    validateInitialized("transposeTrailingDimensions");
    std::vector<uint64_t> source_dimensions = descriptor.getValuesDimensions();
    if (source_dimensions.size() < 3) {
        throw std::invalid_argument(
            "RaggedExpression::transposeTrailingDimensions requires at least two trailing value dimensions.");
    }

    std::vector<uint64_t> source_strides(source_dimensions.size(), 1);
    Expression view_source = values;
    uint64_t base_element_offset = 0;

    // Compose with an immediately preceding view instead of interpreting its logical
    // tensor as contiguous storage. This is required for Slice -> Transpose: both
    // operations are aliases of the same original packed values allocation, and the
    // transpose must retain the slice's accumulated storage offset and row stride.
    if (values.expr && values.nodeIndex < values.expr->nodes.size() &&
        values.expr->nodes.at(values.nodeIndex).op == ExprOp::STRIDED_VIEW) {
        const ExprNode& prior_view = values.expr->nodes.at(values.nodeIndex);
        if (prior_view.view_dims != source_dimensions || prior_view.view_strides.size() != source_dimensions.size() ||
            prior_view.lhs == UINT32_MAX || prior_view.lhs >= values.expr->nodes.size()) {
            throw std::runtime_error("RaggedExpression::transposeTrailingDimensions encountered an invalid preceding strided view.");
        }
        source_strides = prior_view.view_strides;
        base_element_offset = prior_view.view_element_offset;
        view_source = Expression::fromPhysicalNode(values.expr, prior_view.lhs);
    } else {
        uint64_t stride = 1;
        for (size_t axis = source_dimensions.size(); axis-- > 0;) {
            source_strides[axis] = stride;
            stride = checkedMul(stride, source_dimensions[axis], "ragged trailing transpose source stride");
        }
    }

    std::vector<uint64_t> output_dimensions = source_dimensions;
    std::vector<uint64_t> output_strides = source_strides;
    std::swap(output_dimensions[output_dimensions.size() - 2], output_dimensions[output_dimensions.size() - 1]);
    std::swap(output_strides[output_strides.size() - 2], output_strides[output_strides.size() - 1]);

    Expression transposed_values = view_source.stridedView(output_dimensions, output_strides, base_element_offset);
    RaggedTensorDescriptor transposed_descriptor(
        TensorDescriptor(descriptor.getValuesDataType(), output_dimensions), descriptor.getRowPartition(), descriptor.getRaggedRank());
    return withValues(transposed_values, transposed_descriptor);
}

RaggedExpression RaggedExpression::cast(DataType output_dtype) const {
    validateInitialized("cast");
    return withValues(values.cast(output_dtype), descriptorWithValuesDataType(descriptor, output_dtype));
}

RaggedExpression RaggedExpression::rmsNorm(const Expression& scale,
                                             double epsilon,
                                             std::optional<DataType> compute_dtype,
                                             std::optional<DataType> output_dtype) const {
    validateInitialized("rmsNorm");
    const std::vector<uint64_t> trailing = descriptor.getTrailingDimensions();
    if (trailing.size() != 1 || trailing.front() == 0) {
        throw std::invalid_argument("RaggedExpression::rmsNorm currently requires exactly one channel dimension.");
    }
    const DataType result_dtype = output_dtype.value_or(descriptor.getValuesDataType());
    Expression normalized = Expression::rmsNorm(executionValues,
                                                scale,
                                                trailing.front(),
                                                epsilon,
                                                compute_dtype.value_or(DataType::FP32),
                                                result_dtype,
                                                descriptor.getMaxTotalValues());
    return withValues(normalized, descriptorWithValuesDataType(descriptor, result_dtype));
}

RaggedExpression RaggedExpression::layerNorm(const Expression& scale,
                                             const Expression& bias,
                                             double epsilon,
                                             std::optional<DataType> compute_dtype,
                                             std::optional<DataType> output_dtype) const {
    validateInitialized("layerNorm");
    const std::vector<uint64_t> trailing = descriptor.getTrailingDimensions();
    if (trailing.size() != 1 || trailing.front() == 0) {
        throw std::invalid_argument(
            "RaggedExpression::layerNorm currently requires exactly one non-zero trailing channel dimension.");
    }
    const uint64_t channels = trailing.front();
    const DataType result_dtype = output_dtype.value_or(descriptor.getValuesDataType());
    Expression normalized = Expression::layerNorm(executionValues,
                                                  scale,
                                                  bias,
                                                  channels,
                                                  epsilon,
                                                  compute_dtype.value_or(DataType::FP32),
                                                  result_dtype,
                                                  descriptor.getMaxTotalValues());
    return withValues(normalized, descriptorWithValuesDataType(descriptor, result_dtype));
}

RaggedExpression RaggedExpression::conv1d(const Expression& filter,
                                          uint64_t output_channels,
                                          uint64_t kernel_width,
                                          ConvolutionSpatial1d spatial,
                                          std::optional<DataType> compute_dtype,
                                          std::optional<DataType> output_dtype,
                                          uint64_t groups) const {
    validateInitialized("conv1d");
    const std::vector<uint64_t> trailing = descriptor.getTrailingDimensions();
    if (trailing.size() != 1) {
        throw std::invalid_argument("RaggedExpression::conv1d requires exactly one trailing channel dimension.");
    }
    if (output_channels == 0 || kernel_width == 0 || groups == 0) {
        throw std::invalid_argument("RaggedExpression::conv1d requires positive output_channels, kernel_width, and groups.");
    }
    if (trailing.front() % groups != 0 || output_channels % groups != 0) {
        throw std::invalid_argument("RaggedExpression::conv1d requires input and output channels divisible by groups.");
    }
    if (!descriptor.hasMaxValuesPerRow()) {
        throw std::invalid_argument(
            "RaggedExpression::conv1d requires max_values_per_row in the row-partition descriptor so placement can "
            "prebuild the finite convolution width family.");
    }
    if (spatial.stride != 1) {
        throw std::invalid_argument("RaggedExpression::conv1d T6A supports only stride=1.");
    }
    if (spatial.dilation <= 0) {
        throw std::invalid_argument("RaggedExpression::conv1d dilation must be positive.");
    }
    const ConvolutionSpatial1d causal = ConvolutionSpatial1d::causal(kernel_width, 1, spatial.dilation);
    if (spatial.pre_padding != causal.pre_padding || spatial.post_padding != causal.post_padding) {
        throw std::invalid_argument("RaggedExpression::conv1d T6A supports only causal padding.");
    }

    Expression output_values = Expression::ternaryOp(values, filter, offsets, ExprOp::RAGGED_CONV1D_CAUSAL);
    ExprNode& node = output_values.expr->nodes.at(output_values.nodeIndex);
    node.ragged_conv_spatial_1d = spatial;
    node.ragged_conv1d_input_channels = trailing.front();
    node.ragged_conv1d_output_channels = output_channels;
    node.ragged_conv1d_kernel_width = kernel_width;
    node.ragged_conv1d_groups = groups;
    node.ragged_runtime_batch_size = descriptor.getBatchSize();
    node.ragged_runtime_max_active_values = descriptor.getMaxTotalValues();
    node.ragged_runtime_max_values_per_row = descriptor.getMaxValuesPerRow();
    node.ragged_runtime_elements_per_value = output_channels;
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }
    const DataType values_dtype = output_dtype.value_or(descriptor.getValuesDataType());
    node.output_dtype = values_dtype;
    const RaggedTensorDescriptor output_descriptor(values_dtype,
                                                    {output_channels},
                                                    descriptor.getBatchSize(),
                                                    descriptor.getMaxTotalValues(),
                                                    descriptor.getMaxValuesPerRow(),
                                                    descriptor.getOffsetsDataType(),
                                                    descriptor.getRaggedRank());
    return RaggedExpression(std::move(output_values), offsets, output_descriptor);
}

RaggedExpression RaggedExpression::causalConv1d(const Expression& filter,
                                                uint64_t output_channels,
                                                uint64_t kernel_width,
                                                int32_t dilation,
                                                std::optional<DataType> compute_dtype,
                                                std::optional<DataType> output_dtype,
                                                uint64_t groups) const {
    return conv1d(filter,
                  output_channels,
                  kernel_width,
                  ConvolutionSpatial1d::causal(kernel_width, 1, dilation),
                  compute_dtype,
                  output_dtype,
                  groups);
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
    return withValues(values.relu(), descriptor);
}

RaggedExpression RaggedExpression::softmax() const { return segment_softmax(); }

Expression RaggedExpression::reduce_sum() const { return segment_sum(); }

Expression RaggedExpression::segment_sum() const {
    validateInitialized("segment_sum");
    return Expression::segmentedReduceWithRaggedMetadata(values,
                                                         offsets,
                                                         ExprOp::SEGMENTED_REDUCE_SUM,
                                                         descriptor.getBatchSize(),
                                                         descriptor.getMaxTotalValues(),
                                                         elementsPerValue(descriptor));
}

Expression RaggedExpression::segment_min() const {
    validateInitialized("segment_min");
    return Expression::segmentedReduceWithRaggedMetadata(values,
                                                         offsets,
                                                         ExprOp::SEGMENTED_REDUCE_MIN,
                                                         descriptor.getBatchSize(),
                                                         descriptor.getMaxTotalValues(),
                                                         elementsPerValue(descriptor));
}

Expression RaggedExpression::segment_max() const {
    validateInitialized("segment_max");
    return Expression::segmentedReduceWithRaggedMetadata(values,
                                                         offsets,
                                                         ExprOp::SEGMENTED_REDUCE_MAX,
                                                         descriptor.getBatchSize(),
                                                         descriptor.getMaxTotalValues(),
                                                         elementsPerValue(descriptor));
}

Expression RaggedExpression::segment_mean() const {
    validateInitialized("segment_mean");

    switch (descriptor.getValuesDataType()) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
        case DataType::FP64:
            break;
        default:
            throw std::invalid_argument("RaggedExpression::segment_mean requires floating-point ragged values.");
    }

    return Expression::segmentedReduceWithRaggedMetadata(values,
                                                         offsets,
                                                         ExprOp::SEGMENTED_REDUCE_MEAN,
                                                         descriptor.getBatchSize(),
                                                         descriptor.getMaxTotalValues(),
                                                         elementsPerValue(descriptor));
}

RaggedExpression RaggedExpression::segment_softmax() const {
    validateInitialized("segment_softmax");
    if (descriptor.getTrailingDimensions().empty()) {
        const Expression max_values = segmentTotalBroadcast(ScanOp::Max, "segment_softmax");
        const Expression shifted = values - max_values;
        const Expression exp_values = shifted.exp();
        const RaggedExpression exp_ragged = withValues(exp_values, descriptor);
        const Expression denom = exp_ragged.segmentTotalBroadcast(ScanOp::Sum, "segment_softmax");
        return withValues(exp_values / denom, descriptor);
    }

    const Expression row_max = segment_max();
    const Expression shifted = values - segmentDenseBroadcast(row_max, false);
    const Expression exp_values = shifted.exp();
    const RaggedExpression exp_ragged = withValues(exp_values, descriptor);
    const Expression row_sum = exp_ragged.segment_sum();
    return withValues(exp_values / segmentDenseBroadcast(row_sum, false), descriptor);
}

RaggedExpression RaggedExpression::segment_log_softmax() const {
    validateInitialized("segment_log_softmax");
    if (descriptor.getTrailingDimensions().empty()) {
        const Expression max_values = segmentTotalBroadcast(ScanOp::Max, "segment_log_softmax");
        const Expression shifted = values - max_values;
        const Expression exp_values = shifted.exp();
        const RaggedExpression exp_ragged = withValues(exp_values, descriptor);
        const Expression denom = exp_ragged.segmentTotalBroadcast(ScanOp::Sum, "segment_log_softmax");
        return withValues(shifted - denom.ln(), descriptor);
    }

    const Expression row_max = segment_max();
    const Expression shifted = values - segmentDenseBroadcast(row_max, false);
    const Expression exp_values = shifted.exp();
    const RaggedExpression exp_ragged = withValues(exp_values, descriptor);
    const Expression row_sum = exp_ragged.segment_sum();
    return withValues(shifted - segmentDenseBroadcast(row_sum, false).ln(), descriptor);
}

RaggedExpression RaggedExpression::segment_broadcast(const Expression& per_segment_values,
                                                      const RaggedTensorDescriptor& output_descriptor) const {
    validateInitialized("segment_broadcast");
    validateDescriptor(output_descriptor);
    // withValues() enforces exact RowPartitionDescriptor equality. The explicit
    // width override is necessary because the broadcast value shape is owned by
    // per_segment_values, not by the structural partition carrier's values.
    return withValues(
        segmentDenseBroadcast(per_segment_values, false, elementsPerValue(output_descriptor)), output_descriptor);
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
    if (!descriptor.getTrailingDimensions().empty()) {
        throw std::invalid_argument(raggedOpErrorPrefix(op_name) +
                                    "scalar segmented-scan broadcast cannot be used with trailing value dimensions.");
    }
    const uint64_t batch_size = descriptor.getBatchSize();
    const uint64_t max_active_values = descriptor.getMaxTotalValues();
    switch (op) {
        case ScanOp::Sum: {
            const Expression prefix = Expression::segmentedScanWithRaggedMetadata(
                values, offsets, ScanOp::Sum, true, false, batch_size, max_active_values);
            const Expression suffix = Expression::segmentedScanWithRaggedMetadata(
                values, offsets, ScanOp::Sum, true, true, batch_size, max_active_values);
            return prefix + suffix - values;
        }
        case ScanOp::Min:
        case ScanOp::Max: {
            const Expression prefix = Expression::segmentedScanWithRaggedMetadata(
                values, offsets, op, true, false, batch_size, max_active_values);
            return Expression::segmentedScanWithRaggedMetadata(
                prefix, offsets, op, true, true, batch_size, max_active_values);
        }
        default:
            throw std::invalid_argument(raggedOpErrorPrefix(op_name) + "cannot broadcast this segment scan op.");
    }
}

Expression RaggedExpression::segmentDenseBroadcast(const Expression& per_segment_values,
                                                    bool normalize_by_segment_length,
                                                   std::optional<uint64_t> elements_per_value_override) const {
    validateInitialized("segmentDenseBroadcast");
    Expression out = Expression::binaryOp(per_segment_values, offsets, ExprOp::SEGMENTED_BROADCAST);
    ExprNode& node = out.expr->nodes.at(out.nodeIndex);
    node.ragged_runtime_batch_size = descriptor.getBatchSize();
    node.ragged_runtime_max_active_values = descriptor.getMaxTotalValues();
    node.ragged_runtime_elements_per_value = elements_per_value_override.value_or(elementsPerValue(descriptor));
    if (node.ragged_runtime_elements_per_value == 0) {
        throw std::invalid_argument("RaggedExpression::segmentDenseBroadcast elements-per-value metadata must be non-zero.");
    }
    node.segmented_broadcast_normalize_by_length = normalize_by_segment_length;
    return out;
}

void RaggedExpression::validateInitialized(const char* caller) const {
    if (!initialized) {
        throw std::runtime_error(raggedOpErrorPrefix(caller) + "ragged expression is not initialized.");
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
