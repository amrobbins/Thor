#include "Utilities/Expression/PackedLayerNormBackwardKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

template <typename T>
__device__ inline float asFloat(T v) { return static_cast<float>(v); }
template <>
__device__ inline float asFloat<__half>(__half v) { return __half2float(v); }
template <>
__device__ inline float asFloat<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ inline T fromFloat(float v);
template <>
__device__ inline float fromFloat<float>(float v) { return v; }
template <>
__device__ inline __half fromFloat<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Each CTA handles whole normalized rows, including widths > blockDim.x.
// The two FP32 reductions are over features of the same active row only.
template <typename T>
__global__ void packedLayerNormInputGradient(const T* x,
                                              const float* gamma,
                                              const T* dy,
                                              const float* mean,
                                              const float* inv_std,
                                              T* dx,
                                              uint64_t active_rows,
                                              uint64_t hidden) {
    constexpr uint32_t kThreads = 256;
    __shared__ float sum_q[kThreads];
    __shared__ float sum_qz[kThreads];
    const uint32_t t = threadIdx.x;
    for (uint64_t r = blockIdx.x; r < active_rows; r += gridDim.x) {
        const uint64_t row_start = r * hidden;
        const float mu = mean[r];
        const float inv = inv_std[r];
        float q_total = 0.0f, qz_total = 0.0f;
        for (uint64_t j = t; j < hidden; j += kThreads) {
            const float q = asFloat(dy[row_start + j]) * gamma[j];
            const float z = (asFloat(x[row_start + j]) - mu) * inv;
            q_total += q;
            qz_total += q * z;
        }
        sum_q[t] = q_total;
        sum_qz[t] = qz_total;
        __syncthreads();
        for (uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (t < stride) {
                sum_q[t] += sum_q[t + stride];
                sum_qz[t] += sum_qz[t + stride];
            }
            __syncthreads();
        }
        const float avg_q = sum_q[0] / static_cast<float>(hidden);
        const float avg_qz = sum_qz[0] / static_cast<float>(hidden);
        for (uint64_t j = t; j < hidden; j += kThreads) {
            const float q = asFloat(dy[row_start + j]) * gamma[j];
            const float z = (asFloat(x[row_start + j]) - mu) * inv;
            dx[row_start + j] = fromFloat<T>(inv * (q - avg_q - z * avg_qz));
        }
        __syncthreads();
    }
}

// Deterministic FP32 affine reductions over exactly [0, active_rows).
// No atomics, zero-padding reads, or bucket-capacity assumptions. Parallel
// across features; a future performance pass may tile the active-row axis.
template <typename T>
__global__ void packedLayerNormAffineGradient(const T* x,
                                               const T* dy,
                                               const float* mean,
                                               const float* inv_std,
                                               float* dgamma,
                                               float* dbeta,
                                               uint64_t active_rows,
                                               uint64_t hidden) {
    const uint64_t j = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= hidden) return;
    float dg = 0.0f, db = 0.0f;
    for (uint64_t r = 0; r < active_rows; ++r) {
        const uint64_t i = r * hidden + j;
        const float dy_value = asFloat(dy[i]);
        dg += dy_value * ((asFloat(x[i]) - mean[r]) * inv_std[r]);
        db += dy_value;
    }
    dgamma[j] = dg;
    dbeta[j] = db;
}

void requirePackedDense(const Tensor& t, const std::vector<uint64_t>& dims, const char* name) {
    if (!t.isInitialized() || t.getDimensions() != dims ||
        t.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string("Packed LayerNorm backward incompatible ") + name + " dimensions/placement.");
    }
    std::vector<uint64_t> expected_stride(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        expected_stride[i - 1] = expected_stride[i] * dims[i];
    }
    if (t.getStridesElements() != expected_stride) {
        throw std::invalid_argument(std::string("Packed LayerNorm backward requires contiguous ") + name + ".");
    }
}

template <typename T>
void launchTyped(const Tensor& x, const Tensor& scale, const Tensor& dy, const Tensor& mean,
                 const Tensor& inv_std, Tensor& dx, Tensor& dgamma,
                 Tensor& dbeta, uint64_t active, uint64_t hidden, cudaStream_t stream) {
    if (active == 0) {
        CUDA_CHECK(cudaMemsetAsync(dgamma.getMemPtr(), 0, hidden * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(dbeta.getMemPtr(), 0, hidden * sizeof(float), stream));
        return;
    }
    constexpr uint32_t kThreads = 256;
    const uint32_t row_blocks = static_cast<uint32_t>(std::min<uint64_t>(active, 4096));
    const uint32_t affine_blocks = static_cast<uint32_t>((hidden + kThreads - 1) / kThreads);
    packedLayerNormInputGradient<T><<<row_blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(x.getMemPtr()), static_cast<const float*>(scale.getMemPtr()),
        static_cast<const T*>(dy.getMemPtr()), static_cast<const float*>(mean.getMemPtr()),
        static_cast<const float*>(inv_std.getMemPtr()), static_cast<T*>(dx.getMemPtr()), active, hidden);
    CUDA_CHECK(cudaGetLastError());
    packedLayerNormAffineGradient<T><<<affine_blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(x.getMemPtr()), static_cast<const T*>(dy.getMemPtr()),
        static_cast<const float*>(mean.getMemPtr()), static_cast<const float*>(inv_std.getMemPtr()),
        static_cast<float*>(dgamma.getMemPtr()), static_cast<float*>(dbeta.getMemPtr()), active, hidden);
    CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void launchPackedLayerNormBackwardActivePrefix(const Tensor& x, const Tensor& scale, const Tensor& dY,
                                               const Tensor& mean, const Tensor& inv_variance,
                                               Tensor dX, Tensor dScale, Tensor dBias,
                                               uint64_t active_rows, cudaStream_t stream) {
    const std::vector<uint64_t> dims = x.getDimensions();
    if (dims.size() != 2 || dims[0] == 0 || dims[1] == 0 ||
        active_rows > dims[0] || dims[0] > std::numeric_limits<uint64_t>::max() / dims[1]) {
        throw std::invalid_argument("Packed LayerNorm backward requires valid [capacity,hidden] and active prefix.");
    }
    const std::vector<uint64_t> affine_dims{dims[1]};
    const std::vector<uint64_t> stats_dims{dims[0]};
    requirePackedDense(x, dims, "x");
    requirePackedDense(dY, dims, "dY");
    requirePackedDense(dX, dims, "dX");
    requirePackedDense(scale, affine_dims, "scale");
    requirePackedDense(dScale, affine_dims, "dScale");
    requirePackedDense(dBias, affine_dims, "dBias");
    requirePackedDense(mean, stats_dims, "mean");
    requirePackedDense(inv_variance, stats_dims, "inv_variance");
    const auto place = x.getPlacement();
    const Tensor* placement_tensors[] = {&dY, &dX, &scale, &dScale, &dBias, &mean, &inv_variance};
    for (const Tensor* t : placement_tensors) {
        if (t->getPlacement() != place) {
            throw std::invalid_argument("Packed LayerNorm backward tensors must reside on one GPU.");
        }
    }
    if (scale.getDataType() != DataType::FP32 || dScale.getDataType() != DataType::FP32 ||
        dBias.getDataType() != DataType::FP32 || mean.getDataType() != DataType::FP32 ||
        inv_variance.getDataType() != DataType::FP32 || x.getDataType() != dY.getDataType() ||
        x.getDataType() != dX.getDataType()) {
        throw std::invalid_argument("Packed LayerNorm backward requires matching x/dY/dX and FP32 affine/statistics.");
    }
    switch (x.getDataType()) {
        case DataType::FP32:
            launchTyped<float>(x, scale, dY, mean, inv_variance, dX, dScale, dBias, active_rows, dims[1], stream);
            break;
        case DataType::FP16:
            launchTyped<__half>(x, scale, dY, mean, inv_variance, dX, dScale, dBias, active_rows, dims[1], stream);
            break;
        case DataType::BF16:
            launchTyped<__nv_bfloat16>(x, scale, dY, mean, inv_variance, dX, dScale, dBias, active_rows, dims[1], stream);
            break;
        default:
            throw std::invalid_argument("Packed LayerNorm active-only backward supports FP32, FP16 and BF16 values.");
    }
}

}  // namespace ThorImplementation
