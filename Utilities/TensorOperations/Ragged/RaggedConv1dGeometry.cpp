#include "Utilities/TensorOperations/Ragged/RaggedConv1dGeometry.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ThorImplementation {
namespace {

uint64_t checkedAdd(uint64_t lhs, uint64_t rhs, const char* context) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        throw std::overflow_error(context);
    }
    return lhs + rhs;
}

uint64_t checkedMul(uint64_t lhs, uint64_t rhs, const char* context) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::overflow_error(context);
    }
    return lhs * rhs;
}

uint64_t ceilDivPositiveNumerator(uint64_t numerator, uint64_t denominator) {
    if (numerator == 0) {
        return 0;
    }
    return 1 + (numerator - 1) / denominator;
}

uint64_t fixedPaddingOutputLength(uint64_t rowLength,
                                  uint64_t leftPadding,
                                  uint64_t rightPadding,
                                  uint64_t effectiveKernel,
                                  uint64_t stride) {
    uint64_t paddedLength = checkedAdd(rowLength, leftPadding, "Ragged Conv1D padded row length overflows uint64_t.");
    paddedLength = checkedAdd(paddedLength, rightPadding, "Ragged Conv1D padded row length overflows uint64_t.");
    if (paddedLength < effectiveKernel) {
        return 0;
    }
    return 1 + (paddedLength - effectiveKernel) / stride;
}

}  // namespace

RaggedConv1dGeometry RaggedConv1dGeometry::valid(uint64_t kernelWidth, int32_t stride, int32_t dilation) {
    RaggedConv1dGeometry geometry{
        .padding_mode = RaggedConv1dPaddingMode::VALID,
        .kernel_width = kernelWidth,
        .stride = stride,
        .dilation = dilation,
    };
    geometry.validate();
    return geometry;
}

RaggedConv1dGeometry RaggedConv1dGeometry::sameUpper(uint64_t kernelWidth, int32_t stride, int32_t dilation) {
    RaggedConv1dGeometry geometry{
        .padding_mode = RaggedConv1dPaddingMode::SAME_UPPER,
        .kernel_width = kernelWidth,
        .stride = stride,
        .dilation = dilation,
    };
    geometry.validate();
    return geometry;
}

RaggedConv1dGeometry RaggedConv1dGeometry::causal(uint64_t kernelWidth, int32_t stride, int32_t dilation) {
    RaggedConv1dGeometry geometry{
        .padding_mode = RaggedConv1dPaddingMode::CAUSAL,
        .kernel_width = kernelWidth,
        .stride = stride,
        .dilation = dilation,
    };
    geometry.validate();
    return geometry;
}

RaggedConv1dGeometry RaggedConv1dGeometry::explicitPadding(uint64_t kernelWidth,
                                                           uint64_t leftPadding,
                                                           uint64_t rightPadding,
                                                           int32_t stride,
                                                           int32_t dilation) {
    RaggedConv1dGeometry geometry{
        .padding_mode = RaggedConv1dPaddingMode::EXPLICIT,
        .kernel_width = kernelWidth,
        .stride = stride,
        .dilation = dilation,
        .explicit_left_padding = leftPadding,
        .explicit_right_padding = rightPadding,
    };
    geometry.validate();
    return geometry;
}

void RaggedConv1dGeometry::validate() const {
    if (kernel_width == 0) {
        throw std::invalid_argument("Ragged Conv1D kernel_width must be positive.");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Ragged Conv1D stride must be positive.");
    }
    if (dilation <= 0) {
        throw std::invalid_argument("Ragged Conv1D dilation must be positive.");
    }
    switch (padding_mode) {
        case RaggedConv1dPaddingMode::VALID:
        case RaggedConv1dPaddingMode::SAME_UPPER:
        case RaggedConv1dPaddingMode::CAUSAL:
        case RaggedConv1dPaddingMode::EXPLICIT:
            break;
        default:
            throw std::invalid_argument("Ragged Conv1D padding mode is invalid.");
    }
    if (padding_mode != RaggedConv1dPaddingMode::EXPLICIT &&
        (explicit_left_padding != 0 || explicit_right_padding != 0)) {
        throw std::invalid_argument("Ragged Conv1D non-explicit padding modes cannot carry explicit padding values.");
    }
    (void)effectiveKernelWidth();
}

uint64_t RaggedConv1dGeometry::effectiveKernelWidth() const {
    if (kernel_width == 0) {
        throw std::invalid_argument("Ragged Conv1D kernel_width must be positive.");
    }
    if (dilation <= 0) {
        throw std::invalid_argument("Ragged Conv1D dilation must be positive.");
    }
    const uint64_t kernelMinusOne = kernel_width - 1;
    const uint64_t dilatedSpan = checkedMul(
        kernelMinusOne, static_cast<uint64_t>(dilation), "Ragged Conv1D effective kernel width overflows uint64_t.");
    return checkedAdd(dilatedSpan, 1, "Ragged Conv1D effective kernel width overflows uint64_t.");
}

RaggedConv1dResolvedPadding RaggedConv1dGeometry::resolvedPadding(uint64_t rowLength) const {
    validate();
    const uint64_t effectiveKernel = effectiveKernelWidth();
    switch (padding_mode) {
        case RaggedConv1dPaddingMode::VALID:
            return {};
        case RaggedConv1dPaddingMode::CAUSAL:
            return {.left = effectiveKernel - 1, .right = 0};
        case RaggedConv1dPaddingMode::EXPLICIT:
            return {.left = explicit_left_padding, .right = explicit_right_padding};
        case RaggedConv1dPaddingMode::SAME_UPPER: {
            if (rowLength == 0) {
                return {};
            }
            const uint64_t strideU64 = static_cast<uint64_t>(stride);
            const uint64_t outputLength = ceilDivPositiveNumerator(rowLength, strideU64);
            const uint64_t outputSpan = checkedMul(
                outputLength - 1, strideU64, "Ragged Conv1D SAME_UPPER output span overflows uint64_t.");
            const uint64_t required = checkedAdd(
                outputSpan, effectiveKernel, "Ragged Conv1D SAME_UPPER required row extent overflows uint64_t.");
            const uint64_t totalPadding = required > rowLength ? required - rowLength : 0;
            const uint64_t left = totalPadding / 2;
            return {.left = left, .right = totalPadding - left};
        }
    }
    throw std::invalid_argument("Ragged Conv1D padding mode is invalid.");
}

uint64_t RaggedConv1dGeometry::outputLength(uint64_t rowLength) const {
    validate();
    if (padding_mode == RaggedConv1dPaddingMode::SAME_UPPER) {
        return ceilDivPositiveNumerator(rowLength, static_cast<uint64_t>(stride));
    }

    const RaggedConv1dResolvedPadding padding = resolvedPadding(rowLength);
    return fixedPaddingOutputLength(rowLength,
                                    padding.left,
                                    padding.right,
                                    effectiveKernelWidth(),
                                    static_cast<uint64_t>(stride));
}

bool RaggedConv1dGeometry::preservesRowLengths() const {
    validate();
    if (stride != 1) {
        return false;
    }

    const uint64_t effectiveKernelMinusOne = effectiveKernelWidth() - 1;
    switch (padding_mode) {
        case RaggedConv1dPaddingMode::SAME_UPPER:
        case RaggedConv1dPaddingMode::CAUSAL:
            return true;
        case RaggedConv1dPaddingMode::VALID:
            return effectiveKernelMinusOne == 0;
        case RaggedConv1dPaddingMode::EXPLICIT: {
            const uint64_t totalPadding = checkedAdd(explicit_left_padding,
                                                     explicit_right_padding,
                                                     "Ragged Conv1D explicit padding sum overflows uint64_t.");
            return totalPadding == effectiveKernelMinusOne;
        }
    }
    throw std::invalid_argument("Ragged Conv1D padding mode is invalid.");
}

std::vector<uint64_t> deriveRaggedConv1dOutputOffsets(std::span<const uint64_t> inputOffsets,
                                                       const RaggedConv1dGeometry& geometry) {
    geometry.validate();
    if (inputOffsets.empty()) {
        throw std::invalid_argument("Ragged Conv1D input offsets must contain at least the leading zero.");
    }
    if (inputOffsets.front() != 0) {
        throw std::invalid_argument("Ragged Conv1D input offsets must begin at zero.");
    }

    std::vector<uint64_t> outputOffsets(inputOffsets.size(), 0);
    for (size_t row = 0; row + 1 < inputOffsets.size(); ++row) {
        if (inputOffsets[row + 1] < inputOffsets[row]) {
            throw std::invalid_argument("Ragged Conv1D input offsets must be monotonically non-decreasing.");
        }
        const uint64_t inputLength = inputOffsets[row + 1] - inputOffsets[row];
        const uint64_t outputLength = geometry.outputLength(inputLength);
        outputOffsets[row + 1] = checkedAdd(
            outputOffsets[row], outputLength, "Ragged Conv1D derived output offsets overflow uint64_t.");
    }
    return outputOffsets;
}

RaggedConv1dOutputCapacity deriveRaggedConv1dOutputCapacity(uint64_t batchSize,
                                                            uint64_t inputMaxValuesPerRow,
                                                            const RaggedConv1dGeometry& geometry) {
    geometry.validate();
    const uint64_t maxValuesPerRow = geometry.outputLength(inputMaxValuesPerRow);
    const uint64_t maxTotalValues = checkedMul(
        batchSize, maxValuesPerRow, "Ragged Conv1D derived max_total_values overflows uint64_t.");
    return {.max_values_per_row = maxValuesPerRow, .max_total_values = maxTotalValues};
}

}  // namespace ThorImplementation
