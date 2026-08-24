#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

// Canonical resolved spatial metadata for a dense 1D convolution. This is
// intentionally geometry-only: semantic padding modes are resolved before the
// physical expression is built, just as modern Conv2D resolves SAME/VALID to
// concrete pre/post padding before lowering.
struct ConvolutionSpatial1d {
    int32_t stride = 1;
    int32_t dilation = 1;
    int32_t pre_padding = 0;   // left
    int32_t post_padding = 0;  // right

    bool operator==(const ConvolutionSpatial1d& other) const = default;

    static ConvolutionSpatial1d valid(int32_t stride = 1, int32_t dilation = 1) {
        validateBase(stride, dilation);
        return {.stride = stride, .dilation = dilation};
    }

    static ConvolutionSpatial1d explicitPadding(int32_t left, int32_t right, int32_t stride = 1, int32_t dilation = 1) {
        validateBase(stride, dilation);
        if (left < 0 || right < 0) {
            throw std::invalid_argument("ConvolutionSpatial1d explicit padding must be non-negative.");
        }
        return {.stride = stride, .dilation = dilation, .pre_padding = left, .post_padding = right};
    }

    static ConvolutionSpatial1d causal(uint64_t kernel_width, int32_t stride = 1, int32_t dilation = 1) {
        validateBase(stride, dilation);
        validateExtent(kernel_width, "kernel_width");
        const uint64_t left = checkedEffectiveKernelMinusOne(kernel_width, dilation);
        return {.stride = stride, .dilation = dilation, .pre_padding = checkedPadding(left), .post_padding = 0};
    }

    static ConvolutionSpatial1d sameUpper(uint64_t input_width,
                                          uint64_t kernel_width,
                                          int32_t stride = 1,
                                          int32_t dilation = 1) {
        validateBase(stride, dilation);
        validateExtent(input_width, "input_width");
        validateExtent(kernel_width, "kernel_width");

        const uint64_t s = static_cast<uint64_t>(stride);
        const uint64_t output_width = (input_width - 1) / s + 1;
        const uint64_t effective_kernel_minus_one = checkedEffectiveKernelMinusOne(kernel_width, dilation);
        if (effective_kernel_minus_one == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("ConvolutionSpatial1d effective kernel size overflows uint64_t.");
        const uint64_t effective_kernel = effective_kernel_minus_one + 1;
        const uint64_t output_span = output_width - 1;
        if (output_span != 0 && s > (std::numeric_limits<uint64_t>::max() - effective_kernel) / output_span)
            throw std::overflow_error("ConvolutionSpatial1d SAME_UPPER extent calculation overflows uint64_t.");
        const uint64_t required = output_span * s + effective_kernel;
        const uint64_t total_padding = required > input_width ? required - input_width : 0;
        const uint64_t left = total_padding / 2;
        const uint64_t right = total_padding - left;
        return {.stride = stride,
                .dilation = dilation,
                .pre_padding = checkedPadding(left),
                .post_padding = checkedPadding(right)};
    }

   private:
    static void validateBase(int32_t stride, int32_t dilation) {
        if (stride <= 0)
            throw std::invalid_argument("ConvolutionSpatial1d stride must be positive.");
        if (dilation <= 0)
            throw std::invalid_argument("ConvolutionSpatial1d dilation must be positive.");
    }

    static void validateExtent(uint64_t value, const char* name) {
        if (value == 0)
            throw std::invalid_argument(std::string("ConvolutionSpatial1d ") + name + " must be positive.");
    }

    static uint64_t checkedEffectiveKernelMinusOne(uint64_t kernel_width, int32_t dilation) {
        const uint64_t km1 = kernel_width - 1;
        const uint64_t d = static_cast<uint64_t>(dilation);
        if (km1 != 0 && d > std::numeric_limits<uint64_t>::max() / km1)
            throw std::overflow_error("ConvolutionSpatial1d effective kernel size overflows uint64_t.");
        return d * km1;
    }

    static int32_t checkedPadding(uint64_t padding) {
        if (padding > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            throw std::overflow_error("ConvolutionSpatial1d padding exceeds int32_t range.");
        return static_cast<int32_t>(padding);
    }
};

// Canonical spatial metadata for a 2D convolution. Padding is represented
// explicitly as independent pre/post values in each spatial dimension.
struct ConvolutionSpatial2d {
    int32_t stride_h = 1;
    int32_t stride_w = 1;
    int32_t dilation_h = 1;
    int32_t dilation_w = 1;
    int32_t pre_padding_h = 0;
    int32_t post_padding_h = 0;
    int32_t pre_padding_w = 0;
    int32_t post_padding_w = 0;

    bool operator==(const ConvolutionSpatial2d& other) const = default;
};

}  // namespace ThorImplementation
