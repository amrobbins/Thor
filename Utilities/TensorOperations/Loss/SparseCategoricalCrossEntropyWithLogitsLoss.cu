#include "SparseCategoricalCrossEntropyWithLogitsLoss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace std;

namespace {

template <typename T>
__device__ inline float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float toFloat<half>(half value) {
    return __half2float(value);
}

template <typename T>
__device__ inline T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline half fromFloat<half>(float value) {
    return __float2half(value);
}

template <typename MASK_TYPE>
__device__ inline bool maskIsValid(const MASK_TYPE *mask, uint32_t row) {
    return toFloat(mask[row]) > 0.5f;
}

template <typename LABEL_TYPE, typename LOGIT_TYPE, typename LOSS_TYPE, typename MASK_TYPE>
__global__ void sparseCategoricalCrossEntropyWithLogitsKernel(uint32_t numClasses,
                                                              uint32_t numRows,
                                                              const LABEL_TYPE *labels,
                                                              const LOGIT_TYPE *logits,
                                                              const MASK_TYPE *mask,
                                                              LOSS_TYPE *loss,
                                                              LOGIT_TYPE *gradient,
                                                              bool computeGradient,
                                                              float gradientScale,
                                                              float lossScale,
                                                              bool hasIgnoreIndex,
                                                              uint32_t ignoreIndex,
                                                              bool hasMask) {
    const uint32_t row = blockIdx.x;
    if (row >= numRows)
        return;

    __shared__ float scratch[256];
    __shared__ float rowMaxShared;
    __shared__ float sumExpShared;
    __shared__ float targetLogitShared;
    __shared__ uint32_t labelShared;
    __shared__ bool validShared;

    const uint32_t label = static_cast<uint32_t>(labels[row]);
    if (threadIdx.x == 0) {
        bool valid = true;
        if (hasIgnoreIndex && label == ignoreIndex)
            valid = false;
        if (valid && hasMask)
            valid = maskIsValid(mask, row);
        if (valid && label >= numClasses)
            valid = false;
        labelShared = label;
        validShared = valid;
    }
    __syncthreads();

    const uint32_t rowOffset = row * numClasses;
    const bool valid = validShared;
    const uint32_t target = labelShared;

    float localMax = -CUDART_INF_F;
    for (uint32_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        const float z = toFloat(logits[rowOffset + c]);
        localMax = fmaxf(localMax, isfinite(z) ? z : -CUDART_INF_F);
    }
    scratch[threadIdx.x] = localMax;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        rowMaxShared = scratch[0];
    __syncthreads();

    const float rowMax = rowMaxShared;
    float localSum = 0.0f;
    for (uint32_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
        const float z = toFloat(logits[rowOffset + c]);
        localSum += expf((isfinite(z) ? z : -CUDART_INF_F) - rowMax);
    }
    scratch[threadIdx.x] = localSum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sumExpShared = scratch[0];
        targetLogitShared = valid ? toFloat(logits[rowOffset + target]) : 0.0f;
    }
    __syncthreads();

    const float sumExp = sumExpShared;
    if (computeGradient) {
        const float invSumExp = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
        for (uint32_t c = threadIdx.x; c < numClasses; c += blockDim.x) {
            float grad = 0.0f;
            if (valid) {
                const float z = toFloat(logits[rowOffset + c]);
                const float softmax = expf((isfinite(z) ? z : -CUDART_INF_F) - rowMax) * invSumExp;
                grad = softmax - (c == target ? 1.0f : 0.0f);
                grad *= gradientScale;
            }
            gradient[rowOffset + c] = fromFloat<LOGIT_TYPE>(grad);
        }
    }

    if (threadIdx.x == 0) {
        float rowLoss = 0.0f;
        if (valid)
            rowLoss = (logf(sumExp) + rowMax - targetLogitShared) * lossScale;
        loss[row] = fromFloat<LOSS_TYPE>(rowLoss);
    }
}

}  // namespace

template <typename LABEL_TYPE, typename LOGIT_TYPE, typename LOSS_TYPE, typename MASK_TYPE>
void launchSparseCategoricalCrossEntropyWithLogits(void *labels_d,
                                                   void *logits_d,
                                                   void *mask_d,
                                                   void *loss_d,
                                                   void *gradient_d,
                                                   uint32_t numClasses,
                                                   uint32_t numRows,
                                                   bool computeGradient,
                                                   uint32_t lossScalingFactor,
                                                   float lossWeight,
                                                   bool hasIgnoreIndex,
                                                   uint32_t ignoreIndex,
                                                   bool hasMask,
                                                   Stream stream) {
    if (std::is_same<LABEL_TYPE, half>::value || std::is_same<LABEL_TYPE, float>::value || std::is_same<LABEL_TYPE, bool>::value) {
        THOR_UNREACHABLE();
        return;
    }
    THOR_THROW_IF_FALSE(numClasses > 1);
    THOR_THROW_IF_FALSE(numRows > 0);
    THOR_THROW_IF_FALSE(labels_d != nullptr);
    THOR_THROW_IF_FALSE(logits_d != nullptr);
    THOR_THROW_IF_FALSE(loss_d != nullptr);
    THOR_THROW_IF_FALSE(!computeGradient || gradient_d != nullptr);
    THOR_THROW_IF_FALSE(!hasMask || mask_d != nullptr);

    ScopedGpu scopedGpu(stream.getGpuNum());

    constexpr uint32_t blockSize = 256;
    const float lossScale = lossWeight;
    const float gradientScale = static_cast<float>(lossScalingFactor) * lossWeight;

    sparseCategoricalCrossEntropyWithLogitsKernel<LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, MASK_TYPE>
        <<<numRows, blockSize, 0, stream.getStream()>>>(numClasses,
                                                        numRows,
                                                        static_cast<const LABEL_TYPE *>(labels_d),
                                                        static_cast<const LOGIT_TYPE *>(logits_d),
                                                        static_cast<const MASK_TYPE *>(mask_d),
                                                        static_cast<LOSS_TYPE *>(loss_d),
                                                        static_cast<LOGIT_TYPE *>(gradient_d),
                                                        computeGradient,
                                                        gradientScale,
                                                        lossScale,
                                                        hasIgnoreIndex,
                                                        ignoreIndex,
                                                        hasMask);
}

#define INSTANTIATE_SPARSE_CE_WITH_LOGITS(LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, MASK_TYPE) \
    template void launchSparseCategoricalCrossEntropyWithLogits<LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, MASK_TYPE>( \
        void *, void *, void *, void *, void *, uint32_t, uint32_t, bool, uint32_t, float, bool, uint32_t, bool, Stream)

#define INSTANTIATE_FOR_MASKS(LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE) \
    INSTANTIATE_SPARSE_CE_WITH_LOGITS(LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, bool); \
    INSTANTIATE_SPARSE_CE_WITH_LOGITS(LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, uint8_t); \
    INSTANTIATE_SPARSE_CE_WITH_LOGITS(LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, half); \
    INSTANTIATE_SPARSE_CE_WITH_LOGITS(LABEL_TYPE, LOGIT_TYPE, LOSS_TYPE, float)

#define INSTANTIATE_FOR_LOSS_TYPES(LABEL_TYPE, LOGIT_TYPE) \
    INSTANTIATE_FOR_MASKS(LABEL_TYPE, LOGIT_TYPE, half); \
    INSTANTIATE_FOR_MASKS(LABEL_TYPE, LOGIT_TYPE, float)

#define INSTANTIATE_FOR_LOGIT_TYPES(LABEL_TYPE) \
    INSTANTIATE_FOR_LOSS_TYPES(LABEL_TYPE, half); \
    INSTANTIATE_FOR_LOSS_TYPES(LABEL_TYPE, float)

INSTANTIATE_FOR_LOGIT_TYPES(uint8_t);
INSTANTIATE_FOR_LOGIT_TYPES(uint16_t);
INSTANTIATE_FOR_LOGIT_TYPES(uint32_t);

#undef INSTANTIATE_FOR_LOGIT_TYPES
#undef INSTANTIATE_FOR_LOSS_TYPES
#undef INSTANTIATE_FOR_MASKS
#undef INSTANTIATE_SPARSE_CE_WITH_LOGITS
