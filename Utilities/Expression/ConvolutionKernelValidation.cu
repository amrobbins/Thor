#include "Utilities/Expression/ConvolutionKernelValidation.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

constexpr uint32_t kValidationThreads = 256;
constexpr uint32_t kValidationWarps = kValidationThreads / 32;
constexpr uint32_t kCooperativeReductionThreshold = 512;
constexpr uint32_t kDeviceNoBadIndex = std::numeric_limits<uint32_t>::max();
static_assert(kValidationThreads % 32 == 0);
static_assert(kValidationWarps == 8);

struct DeviceConvolutionValidationProblem {
    uint32_t n = 0;
    uint32_t cin = 0;
    uint32_t cout = 0;
    uint32_t in_d = 1;
    uint32_t in_h = 1;
    uint32_t in_w = 1;
    uint32_t out_d = 1;
    uint32_t out_h = 1;
    uint32_t out_w = 1;
    uint32_t kernel_d = 1;
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;
    uint32_t groups = 1;

    int32_t stride_d = 1;
    int32_t stride_h = 1;
    int32_t stride_w = 1;
    int32_t pre_padding_d = 0;
    int32_t pre_padding_h = 0;
    int32_t pre_padding_w = 0;
    int32_t dilation_d = 1;
    int32_t dilation_h = 1;
    int32_t dilation_w = 1;

    int32_t lhs_dtype = static_cast<int32_t>(DataType::FP32);
    int32_t rhs_dtype = static_cast<int32_t>(DataType::FP32);
    int32_t output_dtype = static_cast<int32_t>(DataType::FP32);
    int32_t compute_dtype = static_cast<int32_t>(DataType::FP32);
    uint32_t max_reduction_terms = 0;
};

struct DeviceConvolutionValidationStats {
    uint32_t bad_elements = 0;
    uint32_t first_bad_index = kDeviceNoBadIndex;
    uint32_t first_bad_actual_bits = 0;
    uint32_t first_bad_expected_bits = 0;
    uint32_t first_bad_tolerance_bits = 0;
    uint32_t max_abs_error_bits = 0;
};

__device__ __forceinline__ float loadValidationValue(const void* ptr, int32_t dtype, uint32_t index) {
    switch (static_cast<DataType>(dtype)) {
        case DataType::FP32:
            return static_cast<const float*>(ptr)[index];
        case DataType::FP16:
            return __half2float(static_cast<const half*>(ptr)[index]);
        case DataType::BF16:
            return __bfloat162float(static_cast<const __nv_bfloat16*>(ptr)[index]);
        case DataType::FP8_E4M3:
            return static_cast<float>(static_cast<const __nv_fp8_e4m3*>(ptr)[index]);
        case DataType::FP8_E5M2:
            return static_cast<float>(static_cast<const __nv_fp8_e5m2*>(ptr)[index]);
        default:
            return CUDART_NAN_F;
    }
}

__device__ __forceinline__ void storeValidationValue(void* ptr, int32_t dtype, uint32_t index, float value) {
    switch (static_cast<DataType>(dtype)) {
        case DataType::FP32:
            static_cast<float*>(ptr)[index] = value;
            return;
        case DataType::FP16:
            static_cast<half*>(ptr)[index] = __float2half_rn(value);
            return;
        case DataType::BF16:
            static_cast<__nv_bfloat16*>(ptr)[index] = __float2bfloat16_rn(value);
            return;
        case DataType::FP8_E4M3:
            static_cast<__nv_fp8_e4m3*>(ptr)[index] = ThorLowPrecision::toFp8E4M3Satfinite(value);
            return;
        case DataType::FP8_E5M2:
            static_cast<__nv_fp8_e5m2*>(ptr)[index] = __nv_fp8_e5m2(value);
            return;
        default:
            return;
    }
}

__device__ __forceinline__ float quantizeValidationExpected(float value, int32_t output_dtype) {
    switch (static_cast<DataType>(output_dtype)) {
        case DataType::FP32:
            return value;
        case DataType::FP16:
            return __half2float(__float2half_rn(value));
        case DataType::BF16:
            return __bfloat162float(__float2bfloat16_rn(value));
        case DataType::FP8_E4M3:
            return static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(value));
        case DataType::FP8_E5M2:
            return static_cast<float>(__nv_fp8_e5m2(value));
        default:
            return CUDART_NAN_F;
    }
}

__device__ __forceinline__ float storageAbsoluteTolerance(float expected, int32_t dtype) {
    const float magnitude = fabsf(expected);
    switch (static_cast<DataType>(dtype)) {
        case DataType::FP32:
            return 0.0f;
        case DataType::FP16:
            return 1.1e-3f * magnitude + 6.0e-8f;
        case DataType::BF16:
            return 8.0e-3f * magnitude + 1.0e-40f;
        case DataType::FP8_E4M3:
            return 1.3e-1f * magnitude + 2.0e-3f;
        case DataType::FP8_E5M2:
            return 2.6e-1f * magnitude + 1.6e-5f;
        default:
            return CUDART_INF_F;
    }
}

// The validation operands are +/-1/16, so every non-padding product is exactly
// +/-1/256.  The independent reference accumulates the signed product counts as
// int32_t, making its reduction exact and order-independent.  We deliberately do
// not provide a 64-bit GPU fallback: validation setup rejects any geometry whose
// single-output reduction could overflow int32_t.
__device__ __forceinline__ float convolutionValidationTolerance(float expected,
                                                                 int32_t output_dtype,
                                                                 int32_t compute_dtype,
                                                                 uint32_t max_reduction_terms) {
    const DataType compute_type = static_cast<DataType>(compute_dtype);
    constexpr uint32_t kFp32ExactIntegerTerms = 1U << 24;
    if ((compute_type == DataType::FP32 || compute_type == DataType::TF32) &&
        max_reduction_terms <= kFp32ExactIntegerTerms) {
        // The validator's +/-1/16 operands and +/-1/256 products are exactly
        // representable in both FP32 and TF32, so the integer oracle remains exact.
        return 0.0f;
    }

    const float magnitude_scale = fmaxf(1.0f, fabsf(expected));
    const bool fp16_compute = compute_type == DataType::FP16;
    const float arithmetic_tolerance = (fp16_compute ? 5.0e-3f : 5.0e-5f) * magnitude_scale;
    return arithmetic_tolerance + storageAbsoluteTolerance(expected, output_dtype);
}

__device__ __forceinline__ void recordValidationResult(DeviceConvolutionValidationStats* stats,
                                                        uint32_t index,
                                                        float actual,
                                                        float expected,
                                                        float tolerance) {
    float abs_error = fabsf(actual - expected);
    bool bad = false;
    if (!isfinite(actual) || !isfinite(expected)) {
        bad = !(actual == expected);
        if (bad) {
            abs_error = CUDART_INF_F;
        }
    } else {
        bad = abs_error > tolerance;
    }

    if (isfinite(abs_error) && abs_error >= 0.0f) {
        atomicMax(&stats->max_abs_error_bits, __float_as_uint(abs_error));
    } else if (!isfinite(abs_error)) {
        atomicMax(&stats->max_abs_error_bits, __float_as_uint(CUDART_INF_F));
    }

    if (!bad) {
        return;
    }

    atomicAdd(&stats->bad_elements, 1U);
    const uint32_t won = atomicCAS(&stats->first_bad_index, kDeviceNoBadIndex, index);
    if (won == kDeviceNoBadIndex) {
        stats->first_bad_actual_bits = __float_as_uint(actual);
        stats->first_bad_expected_bits = __float_as_uint(expected);
        stats->first_bad_tolerance_bits = __float_as_uint(tolerance);
    }
}

__device__ __forceinline__ float convolutionValidationPatternValue(uint32_t index, uint32_t seed) {
    // A 32-bit integer hash is sufficient for the deterministic sign pattern and
    // avoids expensive 64-bit integer mixing in every fill/preservation thread.
    uint32_t x = index + 0x9E3779B9U * (seed + 1U);
    x ^= x >> 16;
    x *= 0x7FEB352DU;
    x ^= x >> 15;
    x *= 0x846CA68BU;
    x ^= x >> 16;

    constexpr float magnitude = 1.0f / 16.0f;
    return (x & 1U) != 0 ? magnitude : -magnitude;
}

__global__ void fillConvolutionValidationTensorKernel(void* output, uint32_t num_elements, int32_t dtype, uint32_t seed) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }
    storeValidationValue(output, dtype, index, convolutionValidationPatternValue(index, seed));
}

__global__ void validateConvolutionValidationInputKernel(const void* input,
                                                         uint32_t num_elements,
                                                         int32_t dtype,
                                                         uint32_t seed,
                                                         DeviceConvolutionValidationStats* stats) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }
    const float actual = loadValidationValue(input, dtype, index);
    const float expected = convolutionValidationPatternValue(index, seed);
    recordValidationResult(stats, index, actual, expected, 0.0f);
}

__device__ __forceinline__ uint32_t index5(uint32_t n,
                                           uint32_t c,
                                           uint32_t d,
                                           uint32_t h,
                                           uint32_t w,
                                           uint32_t channels,
                                           uint32_t depth,
                                           uint32_t height,
                                           uint32_t width) {
    return ((((n * channels + c) * depth + d) * height + h) * width + w);
}

__device__ __forceinline__ uint32_t filterIndex5(uint32_t co,
                                                 uint32_t ci_local,
                                                 uint32_t kd,
                                                 uint32_t kh,
                                                 uint32_t kw,
                                                 uint32_t cin_per_group,
                                                 uint32_t kernel_d,
                                                 uint32_t kernel_h,
                                                 uint32_t kernel_w) {
    return ((((co * cin_per_group + ci_local) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
}

__device__ __forceinline__ int32_t validationValueAsScaledInteger(float value) {
    return __float2int_rn(value * 16.0f);
}

__device__ __forceinline__ float validationScaledProductSumAsFloat(int32_t sum) {
    return static_cast<float>(sum) * (1.0f / 256.0f);
}

__device__ __forceinline__ int32_t blockReduceValidationSum(int32_t local_sum, int32_t* warp_sums) {
    constexpr uint32_t kWarpSize = 32;
    const uint32_t lane = threadIdx.x & (kWarpSize - 1);
    const uint32_t warp = threadIdx.x / kWarpSize;

    for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFU, local_sum, offset);
    }

    if (lane == 0) {
        warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        int32_t block_sum = lane < kValidationWarps ? warp_sums[lane] : 0;
        for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xFFFFFFFFU, block_sum, offset);
        }
        return block_sum;
    }
    return 0;
}

__device__ __forceinline__ void recordReferenceSum(DeviceConvolutionValidationStats* stats,
                                                    const void* candidate,
                                                    const DeviceConvolutionValidationProblem& p,
                                                    uint32_t output_index,
                                                    int32_t exact_scaled_sum) {
    const float exact_sum = validationScaledProductSumAsFloat(exact_scaled_sum);
    const float expected = quantizeValidationExpected(exact_sum, p.output_dtype);
    const float actual = loadValidationValue(candidate, p.output_dtype, output_index);
    const float tolerance = convolutionValidationTolerance(expected, p.output_dtype, p.compute_dtype, p.max_reduction_terms);
    recordValidationResult(stats, output_index, actual, expected, tolerance);
}

__global__ void validateConvolutionForwardSerialKernel(const void* x,
                                                       const void* w,
                                                       const void* candidate,
                                                       DeviceConvolutionValidationProblem p,
                                                       DeviceConvolutionValidationStats* stats,
                                                       uint32_t output_elements) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) {
        return;
    }

    uint32_t remaining = index;
    const uint32_t ow = remaining % p.out_w;
    remaining /= p.out_w;
    const uint32_t oh = remaining % p.out_h;
    remaining /= p.out_h;
    const uint32_t od = remaining % p.out_d;
    remaining /= p.out_d;
    const uint32_t co = remaining % p.cout;
    const uint32_t n = remaining / p.cout;

    const uint32_t cin_per_group = p.cin / p.groups;
    const uint32_t cout_per_group = p.cout / p.groups;
    const uint32_t group = co / cout_per_group;
    const uint32_t ci_base = group * cin_per_group;

    int32_t sum = 0;
    for (uint32_t ci_local = 0; ci_local < cin_per_group; ++ci_local) {
        for (uint32_t kd = 0; kd < p.kernel_d; ++kd) {
            const int32_t id = static_cast<int32_t>(od) * p.stride_d - p.pre_padding_d +
                               static_cast<int32_t>(kd) * p.dilation_d;
            if (id < 0 || id >= static_cast<int32_t>(p.in_d)) {
                continue;
            }
            for (uint32_t kh = 0; kh < p.kernel_h; ++kh) {
                const int32_t ih = static_cast<int32_t>(oh) * p.stride_h - p.pre_padding_h +
                                   static_cast<int32_t>(kh) * p.dilation_h;
                if (ih < 0 || ih >= static_cast<int32_t>(p.in_h)) {
                    continue;
                }
                for (uint32_t kw = 0; kw < p.kernel_w; ++kw) {
                    const int32_t iw = static_cast<int32_t>(ow) * p.stride_w - p.pre_padding_w +
                                       static_cast<int32_t>(kw) * p.dilation_w;
                    if (iw < 0 || iw >= static_cast<int32_t>(p.in_w)) {
                        continue;
                    }
                    const float xv = loadValidationValue(
                        x, p.lhs_dtype, index5(n, ci_base + ci_local, id, ih, iw, p.cin, p.in_d, p.in_h, p.in_w));
                    const float wv = loadValidationValue(
                        w, p.rhs_dtype, filterIndex5(co, ci_local, kd, kh, kw, cin_per_group, p.kernel_d, p.kernel_h, p.kernel_w));
                    sum += validationValueAsScaledInteger(xv) * validationValueAsScaledInteger(wv);
                }
            }
        }
    }

    recordReferenceSum(stats, candidate, p, index, sum);
}

__global__ void validateConvolutionForwardCooperativeKernel(const void* x,
                                                            const void* w,
                                                            const void* candidate,
                                                            DeviceConvolutionValidationProblem p,
                                                            DeviceConvolutionValidationStats* stats) {
    const uint32_t index = blockIdx.x;
    uint32_t remaining = index;
    const uint32_t ow = remaining % p.out_w;
    remaining /= p.out_w;
    const uint32_t oh = remaining % p.out_h;
    remaining /= p.out_h;
    const uint32_t od = remaining % p.out_d;
    remaining /= p.out_d;
    const uint32_t co = remaining % p.cout;
    const uint32_t n = remaining / p.cout;

    const uint32_t cin_per_group = p.cin / p.groups;
    const uint32_t cout_per_group = p.cout / p.groups;
    const uint32_t group = co / cout_per_group;
    const uint32_t ci_base = group * cin_per_group;

    int32_t local_sum = 0;
    for (uint32_t reduction_index = threadIdx.x; reduction_index < p.max_reduction_terms; reduction_index += blockDim.x) {
        uint32_t r = reduction_index;
        const uint32_t kw = r % p.kernel_w;
        r /= p.kernel_w;
        const uint32_t kh = r % p.kernel_h;
        r /= p.kernel_h;
        const uint32_t kd = r % p.kernel_d;
        const uint32_t ci_local = r / p.kernel_d;

        const int32_t id = static_cast<int32_t>(od) * p.stride_d - p.pre_padding_d +
                           static_cast<int32_t>(kd) * p.dilation_d;
        const int32_t ih = static_cast<int32_t>(oh) * p.stride_h - p.pre_padding_h +
                           static_cast<int32_t>(kh) * p.dilation_h;
        const int32_t iw = static_cast<int32_t>(ow) * p.stride_w - p.pre_padding_w +
                           static_cast<int32_t>(kw) * p.dilation_w;
        if (id < 0 || id >= static_cast<int32_t>(p.in_d) || ih < 0 || ih >= static_cast<int32_t>(p.in_h) || iw < 0 ||
            iw >= static_cast<int32_t>(p.in_w)) {
            continue;
        }

        const float xv = loadValidationValue(
            x, p.lhs_dtype, index5(n, ci_base + ci_local, id, ih, iw, p.cin, p.in_d, p.in_h, p.in_w));
        const float wv = loadValidationValue(
            w, p.rhs_dtype, filterIndex5(co, ci_local, kd, kh, kw, cin_per_group, p.kernel_d, p.kernel_h, p.kernel_w));
        local_sum += validationValueAsScaledInteger(xv) * validationValueAsScaledInteger(wv);
    }

    __shared__ int32_t warp_sums[kValidationWarps];
    const int32_t sum = blockReduceValidationSum(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        recordReferenceSum(stats, candidate, p, index, sum);
    }
}

__global__ void validateConvolutionBackwardDataSerialKernel(const void* w,
                                                            const void* dy,
                                                            const void* candidate,
                                                            DeviceConvolutionValidationProblem p,
                                                            DeviceConvolutionValidationStats* stats,
                                                            uint32_t output_elements) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) {
        return;
    }

    uint32_t remaining = index;
    const uint32_t iw = remaining % p.in_w;
    remaining /= p.in_w;
    const uint32_t ih = remaining % p.in_h;
    remaining /= p.in_h;
    const uint32_t id = remaining % p.in_d;
    remaining /= p.in_d;
    const uint32_t ci = remaining % p.cin;
    const uint32_t n = remaining / p.cin;

    const uint32_t cin_per_group = p.cin / p.groups;
    const uint32_t cout_per_group = p.cout / p.groups;
    const uint32_t group = ci / cin_per_group;
    const uint32_t ci_local = ci - group * cin_per_group;
    const uint32_t co_begin = group * cout_per_group;
    const uint32_t co_end = co_begin + cout_per_group;

    int32_t sum = 0;
    for (uint32_t co = co_begin; co < co_end; ++co) {
        for (uint32_t kd = 0; kd < p.kernel_d; ++kd) {
            const int32_t od_numerator = static_cast<int32_t>(id) + p.pre_padding_d -
                                         static_cast<int32_t>(kd) * p.dilation_d;
            if (od_numerator < 0 || od_numerator % p.stride_d != 0) {
                continue;
            }
            const int32_t od = od_numerator / p.stride_d;
            if (od >= static_cast<int32_t>(p.out_d)) {
                continue;
            }
            for (uint32_t kh = 0; kh < p.kernel_h; ++kh) {
                const int32_t oh_numerator = static_cast<int32_t>(ih) + p.pre_padding_h -
                                             static_cast<int32_t>(kh) * p.dilation_h;
                if (oh_numerator < 0 || oh_numerator % p.stride_h != 0) {
                    continue;
                }
                const int32_t oh = oh_numerator / p.stride_h;
                if (oh >= static_cast<int32_t>(p.out_h)) {
                    continue;
                }
                for (uint32_t kw = 0; kw < p.kernel_w; ++kw) {
                    const int32_t ow_numerator = static_cast<int32_t>(iw) + p.pre_padding_w -
                                                 static_cast<int32_t>(kw) * p.dilation_w;
                    if (ow_numerator < 0 || ow_numerator % p.stride_w != 0) {
                        continue;
                    }
                    const int32_t ow = ow_numerator / p.stride_w;
                    if (ow >= static_cast<int32_t>(p.out_w)) {
                        continue;
                    }
                    const float wv = loadValidationValue(
                        w, p.lhs_dtype, filterIndex5(co, ci_local, kd, kh, kw, cin_per_group, p.kernel_d, p.kernel_h, p.kernel_w));
                    const float dyv = loadValidationValue(
                        dy, p.rhs_dtype, index5(n, co, od, oh, ow, p.cout, p.out_d, p.out_h, p.out_w));
                    sum += validationValueAsScaledInteger(wv) * validationValueAsScaledInteger(dyv);
                }
            }
        }
    }

    recordReferenceSum(stats, candidate, p, index, sum);
}

__global__ void validateConvolutionBackwardDataCooperativeKernel(const void* w,
                                                                 const void* dy,
                                                                 const void* candidate,
                                                                 DeviceConvolutionValidationProblem p,
                                                                 DeviceConvolutionValidationStats* stats) {
    const uint32_t index = blockIdx.x;
    uint32_t remaining = index;
    const uint32_t iw = remaining % p.in_w;
    remaining /= p.in_w;
    const uint32_t ih = remaining % p.in_h;
    remaining /= p.in_h;
    const uint32_t id = remaining % p.in_d;
    remaining /= p.in_d;
    const uint32_t ci = remaining % p.cin;
    const uint32_t n = remaining / p.cin;

    const uint32_t cin_per_group = p.cin / p.groups;
    const uint32_t cout_per_group = p.cout / p.groups;
    const uint32_t group = ci / cin_per_group;
    const uint32_t ci_local = ci - group * cin_per_group;
    const uint32_t co_begin = group * cout_per_group;

    int32_t local_sum = 0;
    for (uint32_t reduction_index = threadIdx.x; reduction_index < p.max_reduction_terms; reduction_index += blockDim.x) {
        uint32_t r = reduction_index;
        const uint32_t kw = r % p.kernel_w;
        r /= p.kernel_w;
        const uint32_t kh = r % p.kernel_h;
        r /= p.kernel_h;
        const uint32_t kd = r % p.kernel_d;
        const uint32_t co = co_begin + r / p.kernel_d;

        const int32_t od_numerator = static_cast<int32_t>(id) + p.pre_padding_d -
                                     static_cast<int32_t>(kd) * p.dilation_d;
        const int32_t oh_numerator = static_cast<int32_t>(ih) + p.pre_padding_h -
                                     static_cast<int32_t>(kh) * p.dilation_h;
        const int32_t ow_numerator = static_cast<int32_t>(iw) + p.pre_padding_w -
                                     static_cast<int32_t>(kw) * p.dilation_w;
        if (od_numerator < 0 || oh_numerator < 0 || ow_numerator < 0 || od_numerator % p.stride_d != 0 ||
            oh_numerator % p.stride_h != 0 || ow_numerator % p.stride_w != 0) {
            continue;
        }

        const int32_t od = od_numerator / p.stride_d;
        const int32_t oh = oh_numerator / p.stride_h;
        const int32_t ow = ow_numerator / p.stride_w;
        if (od >= static_cast<int32_t>(p.out_d) || oh >= static_cast<int32_t>(p.out_h) || ow >= static_cast<int32_t>(p.out_w)) {
            continue;
        }

        const float wv = loadValidationValue(
            w, p.lhs_dtype, filterIndex5(co, ci_local, kd, kh, kw, cin_per_group, p.kernel_d, p.kernel_h, p.kernel_w));
        const float dyv = loadValidationValue(dy, p.rhs_dtype, index5(n, co, od, oh, ow, p.cout, p.out_d, p.out_h, p.out_w));
        local_sum += validationValueAsScaledInteger(wv) * validationValueAsScaledInteger(dyv);
    }

    __shared__ int32_t warp_sums[kValidationWarps];
    const int32_t sum = blockReduceValidationSum(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        recordReferenceSum(stats, candidate, p, index, sum);
    }
}

__global__ void validateConvolutionBackwardFilterSerialKernel(const void* x,
                                                              const void* dy,
                                                              const void* candidate,
                                                              DeviceConvolutionValidationProblem p,
                                                              DeviceConvolutionValidationStats* stats,
                                                              uint32_t output_elements) {
    const uint32_t weight_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (weight_index >= output_elements) {
        return;
    }

    uint32_t remaining = weight_index;
    const uint32_t kw = remaining % p.kernel_w;
    remaining /= p.kernel_w;
    const uint32_t kh = remaining % p.kernel_h;
    remaining /= p.kernel_h;
    const uint32_t kd = remaining % p.kernel_d;
    remaining /= p.kernel_d;
    const uint32_t cin_per_group = p.cin / p.groups;
    const uint32_t ci_local = remaining % cin_per_group;
    const uint32_t co = remaining / cin_per_group;

    const uint32_t cout_per_group = p.cout / p.groups;
    const uint32_t group = co / cout_per_group;
    const uint32_t ci = group * cin_per_group + ci_local;

    int32_t sum = 0;
    for (uint32_t reduction_index = 0; reduction_index < p.max_reduction_terms; ++reduction_index) {
        uint32_t r = reduction_index;
        const uint32_t ow = r % p.out_w;
        r /= p.out_w;
        const uint32_t oh = r % p.out_h;
        r /= p.out_h;
        const uint32_t od = r % p.out_d;
        const uint32_t n = r / p.out_d;

        const int32_t id = static_cast<int32_t>(od) * p.stride_d - p.pre_padding_d +
                           static_cast<int32_t>(kd) * p.dilation_d;
        const int32_t ih = static_cast<int32_t>(oh) * p.stride_h - p.pre_padding_h +
                           static_cast<int32_t>(kh) * p.dilation_h;
        const int32_t iw = static_cast<int32_t>(ow) * p.stride_w - p.pre_padding_w +
                           static_cast<int32_t>(kw) * p.dilation_w;
        if (id < 0 || id >= static_cast<int32_t>(p.in_d) || ih < 0 || ih >= static_cast<int32_t>(p.in_h) || iw < 0 ||
            iw >= static_cast<int32_t>(p.in_w)) {
            continue;
        }

        const float xv = loadValidationValue(x, p.lhs_dtype, index5(n, ci, id, ih, iw, p.cin, p.in_d, p.in_h, p.in_w));
        const float dyv = loadValidationValue(dy, p.rhs_dtype, index5(n, co, od, oh, ow, p.cout, p.out_d, p.out_h, p.out_w));
        sum += validationValueAsScaledInteger(xv) * validationValueAsScaledInteger(dyv);
    }

    recordReferenceSum(stats, candidate, p, weight_index, sum);
}

__global__ void validateConvolutionBackwardFilterCooperativeKernel(const void* x,
                                                        const void* dy,
                                                        const void* candidate,
                                                        DeviceConvolutionValidationProblem p,
                                                        DeviceConvolutionValidationStats* stats) {
    const uint32_t weight_index = blockIdx.x;
    uint32_t remaining = weight_index;
    const uint32_t kw = remaining % p.kernel_w;
    remaining /= p.kernel_w;
    const uint32_t kh = remaining % p.kernel_h;
    remaining /= p.kernel_h;
    const uint32_t kd = remaining % p.kernel_d;
    remaining /= p.kernel_d;
    const uint32_t cin_per_group = p.cin / p.groups;
    const uint32_t ci_local = remaining % cin_per_group;
    const uint32_t co = remaining / cin_per_group;

    const uint32_t cout_per_group = p.cout / p.groups;
    const uint32_t group = co / cout_per_group;
    const uint32_t ci = group * cin_per_group + ci_local;

    int32_t local_sum = 0;
    for (uint32_t reduction_index = threadIdx.x; reduction_index < p.max_reduction_terms; reduction_index += blockDim.x) {
        uint32_t r = reduction_index;
        const uint32_t ow = r % p.out_w;
        r /= p.out_w;
        const uint32_t oh = r % p.out_h;
        r /= p.out_h;
        const uint32_t od = r % p.out_d;
        const uint32_t n = r / p.out_d;

        const int32_t id = static_cast<int32_t>(od) * p.stride_d - p.pre_padding_d +
                           static_cast<int32_t>(kd) * p.dilation_d;
        const int32_t ih = static_cast<int32_t>(oh) * p.stride_h - p.pre_padding_h +
                           static_cast<int32_t>(kh) * p.dilation_h;
        const int32_t iw = static_cast<int32_t>(ow) * p.stride_w - p.pre_padding_w +
                           static_cast<int32_t>(kw) * p.dilation_w;
        if (id < 0 || id >= static_cast<int32_t>(p.in_d) || ih < 0 || ih >= static_cast<int32_t>(p.in_h) || iw < 0 ||
            iw >= static_cast<int32_t>(p.in_w)) {
            continue;
        }

        const float xv = loadValidationValue(x, p.lhs_dtype, index5(n, ci, id, ih, iw, p.cin, p.in_d, p.in_h, p.in_w));
        const float dyv = loadValidationValue(dy, p.rhs_dtype, index5(n, co, od, oh, ow, p.cout, p.out_d, p.out_h, p.out_w));
        local_sum += validationValueAsScaledInteger(xv) * validationValueAsScaledInteger(dyv);
    }

    __shared__ int32_t warp_sums[kValidationWarps];
    const int32_t sum = blockReduceValidationSum(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        recordReferenceSum(stats, candidate, p, weight_index, sum);
    }
}

[[nodiscard]] bool isSupportedValidationDType(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

constexpr uint64_t kNoBadIndex = std::numeric_limits<uint64_t>::max();

[[nodiscard]] uint64_t checkedProduct(const std::vector<uint64_t>& dims, const char* what) {
    uint64_t product = 1;
    for (uint64_t dim : dims) {
        if (dim == 0 || product > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error(std::string("Invalid/overflowing tensor dimensions for convolution kernel validation: ") + what);
        }
        product *= dim;
    }
    return product;
}

[[nodiscard]] uint32_t checkedUint32Value(uint64_t value, const char* what) {
    if (value == 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Convolution kernel validation requires 32-bit GPU geometry for ") + what + ".");
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t checkedUint32Product(const std::vector<uint64_t>& dims, const char* what) {
    const uint64_t product = checkedProduct(dims, what);
    if (product > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Convolution kernel validation tensor exceeds 32-bit GPU indexing for ") + what + ".");
    }
    return static_cast<uint32_t>(product);
}

[[nodiscard]] uint32_t checkedReductionProduct(const std::vector<uint64_t>& dims, const char* what) {
    const uint64_t product = checkedProduct(dims, what);
    if (product > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string("Convolution kernel validation reduction exceeds the exact int32 accumulator range for ") +
                                 what + ".");
    }
    return static_cast<uint32_t>(product);
}

[[nodiscard]] uint32_t foldValidationSeed(uint64_t seed) {
    return static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(seed >> 32);
}

[[nodiscard]] uint32_t validationElementCount(const Tensor& tensor, const char* what) {
    return checkedUint32Value(tensor.getTotalNumElements(), what);
}

[[nodiscard]] uint32_t validationBlocksForElements(uint32_t num_elements) {
    return (num_elements - 1U) / kValidationThreads + 1U;
}

void requireCoordinateArithmeticFitsInt32(uint32_t input_size,
                                          uint32_t output_size,
                                          uint32_t kernel_size,
                                          int32_t stride,
                                          int32_t pre_padding,
                                          int32_t dilation,
                                          const char* axis) {
    const uint64_t int32_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if (input_size > int32_max || output_size > int32_max || kernel_size > int32_max) {
        throw std::runtime_error(std::string("Convolution kernel validation spatial geometry exceeds int32 coordinate range on axis ") +
                                 axis + ".");
    }

    const uint64_t output_stride_extent = static_cast<uint64_t>(output_size - 1U) * static_cast<uint64_t>(stride);
    const uint64_t kernel_dilation_extent = static_cast<uint64_t>(kernel_size - 1U) * static_cast<uint64_t>(dilation);
    if (output_stride_extent > int32_max || kernel_dilation_extent > int32_max ||
        output_stride_extent + kernel_dilation_extent > int32_max + static_cast<uint64_t>(pre_padding) ||
        static_cast<uint64_t>(input_size - 1U) + static_cast<uint64_t>(pre_padding) > int32_max) {
        throw std::runtime_error(std::string("Convolution kernel validation coordinate arithmetic exceeds int32 range on axis ") + axis +
                                 ".");
    }
}

void requirePackedGpuTensor(const Tensor& tensor, int gpu_num, const char* role) {
    if (!tensor.isInitialized() || tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        tensor.getPlacement().getDeviceNum() != gpu_num) {
        throw std::runtime_error(std::string("Convolution kernel validation requires GPU-resident ") + role + " on the autotune GPU.");
    }
    if (!tensor.isDenseContiguous()) {
        throw std::runtime_error(std::string("Convolution kernel validation requires packed contiguous ") + role + ".");
    }
    if (!isSupportedValidationDType(tensor.getDataType())) {
        throw std::runtime_error(std::string("Convolution kernel validation does not support ") + role + " dtype " +
                                 TensorDescriptor::getElementTypeName(tensor.getDataType()) + ".");
    }
}

DeviceConvolutionValidationProblem makeDeviceProblem(const Tensor& lhs,
                                                     const Tensor& rhs,
                                                     const Tensor& output,
                                                     const ConvolutionKernelValidationSpec& spec) {
    const std::vector<uint64_t> lhs_dims = lhs.getDimensions();
    const std::vector<uint64_t> rhs_dims = rhs.getDimensions();
    const std::vector<uint64_t> out_dims = output.getDimensions();
    const size_t expected_rank = spec.is_3d ? 5 : 4;
    if (lhs_dims.size() != expected_rank || rhs_dims.size() != expected_rank || out_dims.size() != expected_rank) {
        throw std::runtime_error("Convolution kernel validation tensor ranks do not match the convolution geometry.");
    }
    if (spec.groups == 0 || spec.stride_d <= 0 || spec.stride_h <= 0 || spec.stride_w <= 0 || spec.dilation_d <= 0 ||
        spec.dilation_h <= 0 || spec.dilation_w <= 0 || spec.pre_padding_d < 0 || spec.pre_padding_h < 0 ||
        spec.pre_padding_w < 0) {
        throw std::runtime_error("Convolution kernel validation received invalid stride/dilation/padding/groups geometry.");
    }
    if (!isSupportedValidationDType(lhs.getDataType()) || !isSupportedValidationDType(rhs.getDataType()) ||
        !isSupportedValidationDType(output.getDataType())) {
        throw std::runtime_error("Convolution kernel validation received unsupported floating tensor dtype.");
    }
    if (spec.compute_dtype != DataType::FP16 && spec.compute_dtype != DataType::FP32 && spec.compute_dtype != DataType::TF32) {
        throw std::runtime_error("Convolution kernel validation supports FP16, FP32, or TF32 convolution compute.");
    }

    // The CUDA oracle is deliberately a 32-bit implementation.  Validation is a
    // one-time correctness gate for a cuDNN plan, not a second general-purpose
    // convolution engine; giant geometries fail closed instead of slowing every
    // normal validation kernel with 64-bit integer instructions.
    (void)checkedUint32Product(lhs_dims, "lhs");
    (void)checkedUint32Product(rhs_dims, "rhs");
    (void)checkedUint32Product(out_dims, "output");

    DeviceConvolutionValidationProblem p;
    p.groups = checkedUint32Value(spec.groups, "groups");
    p.stride_d = spec.is_3d ? spec.stride_d : 1;
    p.stride_h = spec.stride_h;
    p.stride_w = spec.stride_w;
    p.pre_padding_d = spec.is_3d ? spec.pre_padding_d : 0;
    p.pre_padding_h = spec.pre_padding_h;
    p.pre_padding_w = spec.pre_padding_w;
    p.dilation_d = spec.is_3d ? spec.dilation_d : 1;
    p.dilation_h = spec.dilation_h;
    p.dilation_w = spec.dilation_w;
    p.lhs_dtype = static_cast<int32_t>(lhs.getDataType());
    p.rhs_dtype = static_cast<int32_t>(rhs.getDataType());
    p.output_dtype = static_cast<int32_t>(output.getDataType());
    p.compute_dtype = static_cast<int32_t>(spec.compute_dtype);

    auto dimD = [&](const std::vector<uint64_t>& dims) -> uint32_t {
        return spec.is_3d ? checkedUint32Value(dims[2], "depth") : 1U;
    };
    auto dimH = [&](const std::vector<uint64_t>& dims) -> uint32_t {
        return checkedUint32Value(spec.is_3d ? dims[3] : dims[2], "height");
    };
    auto dimW = [&](const std::vector<uint64_t>& dims) -> uint32_t {
        return checkedUint32Value(spec.is_3d ? dims[4] : dims[3], "width");
    };
    auto filterD = [&](const std::vector<uint64_t>& dims) -> uint32_t {
        return spec.is_3d ? checkedUint32Value(dims[2], "filter depth") : 1U;
    };
    auto filterH = [&](const std::vector<uint64_t>& dims) -> uint32_t {
        return checkedUint32Value(spec.is_3d ? dims[3] : dims[2], "filter height");
    };
    auto filterW = [&](const std::vector<uint64_t>& dims) -> uint32_t {
        return checkedUint32Value(spec.is_3d ? dims[4] : dims[3], "filter width");
    };

    switch (spec.kind) {
        case ConvolutionKernelValidationKind::Forward:
            p.n = checkedUint32Value(lhs_dims[0], "batch");
            p.cin = checkedUint32Value(lhs_dims[1], "input channels");
            p.cout = checkedUint32Value(rhs_dims[0], "output channels");
            p.in_d = dimD(lhs_dims);
            p.in_h = dimH(lhs_dims);
            p.in_w = dimW(lhs_dims);
            p.out_d = dimD(out_dims);
            p.out_h = dimH(out_dims);
            p.out_w = dimW(out_dims);
            p.kernel_d = filterD(rhs_dims);
            p.kernel_h = filterH(rhs_dims);
            p.kernel_w = filterW(rhs_dims);
            if (out_dims[0] != p.n || out_dims[1] != p.cout || p.cin != rhs_dims[1] * spec.groups || p.cout % p.groups != 0) {
                throw std::runtime_error("Convolution forward validation tensors have inconsistent grouped channel geometry.");
            }
            p.max_reduction_terms = checkedReductionProduct(
                {p.cin / p.groups, p.kernel_d, p.kernel_h, p.kernel_w}, "forward reduction");
            break;
        case ConvolutionKernelValidationKind::BackwardData:
            p.n = checkedUint32Value(rhs_dims[0], "batch");
            p.cout = checkedUint32Value(lhs_dims[0], "output channels");
            p.cin = checkedUint32Value(out_dims[1], "input channels");
            p.in_d = dimD(out_dims);
            p.in_h = dimH(out_dims);
            p.in_w = dimW(out_dims);
            p.out_d = dimD(rhs_dims);
            p.out_h = dimH(rhs_dims);
            p.out_w = dimW(rhs_dims);
            p.kernel_d = filterD(lhs_dims);
            p.kernel_h = filterH(lhs_dims);
            p.kernel_w = filterW(lhs_dims);
            if (out_dims[0] != p.n || rhs_dims[1] != p.cout || p.cin != lhs_dims[1] * spec.groups || p.cout % p.groups != 0) {
                throw std::runtime_error("Convolution backward-data validation tensors have inconsistent grouped channel geometry.");
            }
            p.max_reduction_terms = checkedReductionProduct(
                {p.cout / p.groups, p.kernel_d, p.kernel_h, p.kernel_w}, "backward-data reduction");
            break;
        case ConvolutionKernelValidationKind::BackwardFilter:
            p.n = checkedUint32Value(lhs_dims[0], "batch");
            p.cin = checkedUint32Value(lhs_dims[1], "input channels");
            p.cout = checkedUint32Value(rhs_dims[1], "output channels");
            p.in_d = dimD(lhs_dims);
            p.in_h = dimH(lhs_dims);
            p.in_w = dimW(lhs_dims);
            p.out_d = dimD(rhs_dims);
            p.out_h = dimH(rhs_dims);
            p.out_w = dimW(rhs_dims);
            p.kernel_d = filterD(out_dims);
            p.kernel_h = filterH(out_dims);
            p.kernel_w = filterW(out_dims);
            if (rhs_dims[0] != p.n || out_dims[0] != p.cout || p.cin != out_dims[1] * spec.groups || p.cout % p.groups != 0) {
                throw std::runtime_error("Convolution backward-filter validation tensors have inconsistent grouped channel geometry.");
            }
            p.max_reduction_terms = checkedReductionProduct(
                {p.n, p.out_d, p.out_h, p.out_w}, "backward-filter reduction");
            break;
    }

    requireCoordinateArithmeticFitsInt32(
        p.in_d, p.out_d, p.kernel_d, p.stride_d, p.pre_padding_d, p.dilation_d, "depth");
    requireCoordinateArithmeticFitsInt32(
        p.in_h, p.out_h, p.kernel_h, p.stride_h, p.pre_padding_h, p.dilation_h, "height");
    requireCoordinateArithmeticFitsInt32(
        p.in_w, p.out_w, p.kernel_w, p.stride_w, p.pre_padding_w, p.dilation_w, "width");
    return p;
}

}  // namespace

void fillConvolutionKernelValidationTensor(Tensor& tensor, uint64_t seed, Stream& stream) {
    requirePackedGpuTensor(tensor, stream.getGpuNum(), "validation input");
    const uint32_t num_elements = validationElementCount(tensor, "validation input");
    const uint32_t blocks = validationBlocksForElements(num_elements);
    fillConvolutionValidationTensorKernel<<<blocks, kValidationThreads, 0, stream.getStream()>>>(
        tensor.getMemPtr<void>(), num_elements, static_cast<int32_t>(tensor.getDataType()), foldValidationSeed(seed));
    CUDA_CHECK(cudaGetLastError());
}

ConvolutionKernelValidationResult validateConvolutionKernelValidationInputUnchanged(const Tensor& tensor,
                                                                                     uint64_t seed,
                                                                                     Stream& stream) {
    requirePackedGpuTensor(tensor, stream.getGpuNum(), "validation input preservation tensor");
    const uint32_t num_elements = validationElementCount(tensor, "validation input preservation tensor");

    Tensor stats_tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum()),
                        TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(sizeof(DeviceConvolutionValidationStats))}));
    auto* device_stats = reinterpret_cast<DeviceConvolutionValidationStats*>(stats_tensor.getMemPtr<void>());
    DeviceConvolutionValidationStats initial_stats{};
    initial_stats.first_bad_index = kDeviceNoBadIndex;
    CUDA_CHECK(cudaMemcpyAsync(device_stats,
                               &initial_stats,
                               sizeof(DeviceConvolutionValidationStats),
                               cudaMemcpyHostToDevice,
                               stream.getStream()));

    const uint32_t blocks = validationBlocksForElements(num_elements);
    validateConvolutionValidationInputKernel<<<blocks, kValidationThreads, 0, stream.getStream()>>>(
        tensor.getMemPtr<void>(), num_elements, static_cast<int32_t>(tensor.getDataType()), foldValidationSeed(seed), device_stats);
    CUDA_CHECK(cudaGetLastError());

    DeviceConvolutionValidationStats host_stats{};
    CUDA_CHECK(cudaMemcpyAsync(&host_stats,
                               device_stats,
                               sizeof(DeviceConvolutionValidationStats),
                               cudaMemcpyDeviceToHost,
                               stream.getStream()));
    stream.synchronize();

    ConvolutionKernelValidationResult result;
    result.checked_elements = num_elements;
    result.bad_elements = host_stats.bad_elements;
    result.first_bad_index = host_stats.first_bad_index;
    result.first_bad_actual = std::bit_cast<float>(host_stats.first_bad_actual_bits);
    result.first_bad_expected = std::bit_cast<float>(host_stats.first_bad_expected_bits);
    result.first_bad_tolerance = 0.0f;
    result.max_abs_error = std::bit_cast<float>(host_stats.max_abs_error_bits);
    result.passed = result.bad_elements == 0;
    if (result.passed) {
        result.first_bad_index = kNoBadIndex;
    }
    return result;
}

ConvolutionKernelValidationResult validateConvolutionKernelOutput(const Tensor& lhs,
                                                                  const Tensor& rhs,
                                                                  const Tensor& candidate_output,
                                                                  const ConvolutionKernelValidationSpec& spec,
                                                                  Stream& stream) {
    requirePackedGpuTensor(lhs, stream.getGpuNum(), "validation lhs");
    requirePackedGpuTensor(rhs, stream.getGpuNum(), "validation rhs");
    requirePackedGpuTensor(candidate_output, stream.getGpuNum(), "validation output");
    const DeviceConvolutionValidationProblem problem = makeDeviceProblem(lhs, rhs, candidate_output, spec);
    const uint32_t output_elements = validationElementCount(candidate_output, "validation output");

    Tensor stats_tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum()),
                        TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(sizeof(DeviceConvolutionValidationStats))}));
    auto* device_stats = reinterpret_cast<DeviceConvolutionValidationStats*>(stats_tensor.getMemPtr<void>());
    DeviceConvolutionValidationStats initial_stats{};
    initial_stats.first_bad_index = kDeviceNoBadIndex;
    CUDA_CHECK(cudaMemcpyAsync(device_stats,
                               &initial_stats,
                               sizeof(DeviceConvolutionValidationStats),
                               cudaMemcpyHostToDevice,
                               stream.getStream()));

    const bool use_cooperative_reduction = problem.max_reduction_terms > kCooperativeReductionThreshold;
    if (use_cooperative_reduction && output_elements > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "Convolution kernel validation cooperative reference requires at most INT32_MAX output elements.");
    }

    switch (spec.kind) {
        case ConvolutionKernelValidationKind::Forward:
            if (use_cooperative_reduction) {
                validateConvolutionForwardCooperativeKernel<<<output_elements, kValidationThreads, 0, stream.getStream()>>>(
                    lhs.getMemPtr<void>(), rhs.getMemPtr<void>(), candidate_output.getMemPtr<void>(), problem, device_stats);
            } else {
                const uint32_t blocks = validationBlocksForElements(output_elements);
                validateConvolutionForwardSerialKernel<<<blocks, kValidationThreads, 0, stream.getStream()>>>(
                    lhs.getMemPtr<void>(), rhs.getMemPtr<void>(), candidate_output.getMemPtr<void>(), problem, device_stats, output_elements);
            }
            break;
        case ConvolutionKernelValidationKind::BackwardData:
            if (use_cooperative_reduction) {
                validateConvolutionBackwardDataCooperativeKernel<<<output_elements, kValidationThreads, 0, stream.getStream()>>>(
                    lhs.getMemPtr<void>(), rhs.getMemPtr<void>(), candidate_output.getMemPtr<void>(), problem, device_stats);
            } else {
                const uint32_t blocks = validationBlocksForElements(output_elements);
                validateConvolutionBackwardDataSerialKernel<<<blocks, kValidationThreads, 0, stream.getStream()>>>(
                    lhs.getMemPtr<void>(), rhs.getMemPtr<void>(), candidate_output.getMemPtr<void>(), problem, device_stats, output_elements);
            }
            break;
        case ConvolutionKernelValidationKind::BackwardFilter:
            if (use_cooperative_reduction) {
                validateConvolutionBackwardFilterCooperativeKernel<<<output_elements, kValidationThreads, 0, stream.getStream()>>>(
                    lhs.getMemPtr<void>(), rhs.getMemPtr<void>(), candidate_output.getMemPtr<void>(), problem, device_stats);
            } else {
                const uint32_t blocks = validationBlocksForElements(output_elements);
                validateConvolutionBackwardFilterSerialKernel<<<blocks, kValidationThreads, 0, stream.getStream()>>>(
                    lhs.getMemPtr<void>(), rhs.getMemPtr<void>(), candidate_output.getMemPtr<void>(), problem, device_stats, output_elements);
            }
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    DeviceConvolutionValidationStats host_stats{};
    CUDA_CHECK(cudaMemcpyAsync(&host_stats,
                               device_stats,
                               sizeof(DeviceConvolutionValidationStats),
                               cudaMemcpyDeviceToHost,
                               stream.getStream()));
    stream.synchronize();

    ConvolutionKernelValidationResult result;
    result.checked_elements = output_elements;
    result.bad_elements = host_stats.bad_elements;
    result.first_bad_index = host_stats.first_bad_index;
    result.first_bad_actual = std::bit_cast<float>(host_stats.first_bad_actual_bits);
    result.first_bad_expected = std::bit_cast<float>(host_stats.first_bad_expected_bits);
    result.first_bad_tolerance = std::bit_cast<float>(host_stats.first_bad_tolerance_bits);
    result.max_abs_error = std::bit_cast<float>(host_stats.max_abs_error_bits);
    result.passed = result.bad_elements == 0;
    if (result.passed) {
        result.first_bad_index = kNoBadIndex;
    }
    return result;
}

std::string describeConvolutionKernelValidationFailure(const ConvolutionKernelValidationResult& result) {
    if (result.passed) {
        return "passed";
    }
    std::ostringstream out;
    out << "bad_elements=" << result.bad_elements << '/' << result.checked_elements;
    if (result.first_bad_index != kNoBadIndex) {
        out << " first_bad_index=" << result.first_bad_index << " actual=" << result.first_bad_actual
            << " expected=" << result.first_bad_expected << " tolerance=" << result.first_bad_tolerance;
    }
    out << " max_abs_error=" << result.max_abs_error;
    return out.str();
}

}  // namespace ThorImplementation
