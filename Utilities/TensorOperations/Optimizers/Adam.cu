#include "Adam.h"

using namespace std;

__global__ void adamStep_fp16_moments_fp32(half *__restrict__ weightUpdate,    // FP16 output update (same shape as weights)
                                           const half *__restrict__ gradient,  // FP16 gradients
                                           float *__restrict__ m,              // FP32 first moment
                                           float *__restrict__ v,              // FP32 second moment
                                           const float alphaT,
                                           const float beta1,
                                           const float beta2,
                                           const float epsilon,
                                           const uint32_t length) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;

    float gradBuff = __half2float(gradient[index]);
    float mBuf = m[index];
    float vBuf = v[index];
    mBuf = beta1 * mBuf + (1.0f - beta1) * gradBuff;
    m[index] = mBuf;
    vBuf = beta2 * vBuf + (1.0f - beta2) * (gradBuff * gradBuff);
    v[index] = vBuf;

    // Hard guarantee of no inf
    float upd = -alphaT * mBuf / (sqrtf(fmaxf(vBuf, 0.0f)) + epsilon);
    upd = fminf(fmaxf(upd, -65504.0f), 65504.0f);

    weightUpdate[index] = __float2half_rn(upd);
}

__global__ void adamStep_fp32(float *__restrict__ weightUpdate,
                              const float *__restrict__ gradient,
                              float *__restrict__ m,
                              float *__restrict__ v,
                              const float alphaT,
                              const float beta1,
                              const float beta2,
                              const float epsilon,
                              const uint32_t length) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;

    float gradBuff = gradient[index];
    float mBuf = m[index];
    float vBuf = v[index];
    mBuf = beta1 * mBuf + (1.0f - beta1) * gradBuff;
    m[index] = mBuf;
    vBuf = beta2 * vBuf + (1.0f - beta2) * (gradBuff * gradBuff);
    v[index] = vBuf;
    weightUpdate[index] = -alphaT * mBuf / (sqrtf(vBuf) + epsilon);
}

// Note that the t that is passed to launch adam step is the t that should be used in the computation, it is not meant to
// be incremented by launchAdamStep prior to use.
// m and v must be type float to avoid overflow/underflow -> inf.
template <typename T>
void launchAdamStep(T *weightUpdate_d,
                    T *gradient_d,
                    float *m_d,
                    float *v_d,
                    const float t,
                    float alpha,
                    float beta1,
                    float beta2,
                    float epsilon,
                    uint32_t length,
                    Stream stream) {
    float alphaT = alpha * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

    ScopedGpu scopedGpu(stream.getGpuNum());

    dim3 blockSize(256);
    dim3 gridSize((length + 255) / 256);

    if constexpr (is_same_v<T, half>) {
        adamStep_fp16_moments_fp32<<<gridSize, blockSize, 0, stream>>>(
            weightUpdate_d, gradient_d, m_d, v_d, alphaT, beta1, beta2, epsilon, length);
    } else if constexpr (is_same_v<T, float>) {
        adamStep_fp32<<<gridSize, blockSize, 0, stream>>>(weightUpdate_d, gradient_d, m_d, v_d, alphaT, beta1, beta2, epsilon, length);
    } else {
        static_assert(is_same_v<T, half> || is_same_v<T, float>, "launchAdamStep only supports T=half or T=float");
    }
}

template void launchAdamStep<half>(half *weightUpdate_d,
                                   half *gradient_d,
                                   float *m_d,
                                   float *v_d,
                                   float t,
                                   float alpha,
                                   float beta1,
                                   float beta2,
                                   float epsilon,
                                   uint32_t length,
                                   Stream stream);

template void launchAdamStep<float>(float *weightUpdate_d,
                                    float *gradient_d,
                                    float *m_d,
                                    float *v_d,
                                    float t,
                                    float alpha,
                                    float beta1,
                                    float beta2,
                                    float epsilon,
                                    uint32_t length,
                                    Stream stream);
