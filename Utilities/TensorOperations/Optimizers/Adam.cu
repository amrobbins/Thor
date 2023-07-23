#include "Adam.h"

using namespace std;

__global__ void adamStep(
    half *weightUpdate, half *gradient, half *m, half *v, half alphaT, half beta1, half beta2, half epsilon, uint32_t length) {
    // Each block process 512 elements as half2's
    uint32_t indexHalf2 = blockIdx.x * 256 + threadIdx.x;
    if ((indexHalf2 << 1) >= length)
        return;

    half2 *weightUpdateHalf2 = (half2 *)weightUpdate;
    half2 *gradientHalf2 = (half2 *)gradient;
    half2 *mHalf2 = (half2 *)m;
    half2 *vHalf2 = (half2 *)v;
    half2 alphaTHalf2 = __half2half2(alphaT);
    half2 beta1Half2 = __half2half2(beta1);
    half2 beta2Half2 = __half2half2(beta2);
    half2 epsilonHalf2 = __half2half2(epsilon);
    const half2 ONE_HALF_2 = __half2half2((half)1.0f);
    const half2 NEGATIVE_ONE_HALF_2 = __half2half2((half)-1.0f);

    half2 gradBuffHalf2 = gradientHalf2[indexHalf2];
    mHalf2[indexHalf2] = __hadd2(__hmul2(beta1Half2, mHalf2[indexHalf2]), __hmul2(__hsub2(ONE_HALF_2, beta1Half2), gradBuffHalf2));
    vHalf2[indexHalf2] =
        __hadd2(__hmul2(beta2Half2, vHalf2[indexHalf2]), __hmul2(__hsub2(ONE_HALF_2, beta2Half2), __hmul2(gradBuffHalf2, gradBuffHalf2)));
    weightUpdateHalf2[indexHalf2] =
        __h2div(__hmul2(NEGATIVE_ONE_HALF_2, __hmul2(alphaTHalf2, mHalf2[indexHalf2])), __hadd2(h2sqrt(vHalf2[indexHalf2]), epsilonHalf2));
};

__global__ void adamStep(
    float *weightUpdate, float *gradient, float *m, float *v, float alphaT, float beta1, float beta2, float epsilon, uint32_t length) {
    uint32_t index = blockIdx.x * 256 + threadIdx.x;
    if (index >= length)
        return;

    float gradBuff = gradient[index];
    m[index] = beta1 * m[index] + (1.0f - beta1) * gradBuff;
    v[index] = beta2 * v[index] + (1.0f - beta2) * (gradBuff * gradBuff);
    weightUpdate[index] = -alphaT * m[index] / (sqrtf(v[index]) + epsilon);
};

// Note that the t that is passed to launch adam step is the t that should be used in the computation, it is not meant to
// be incremented by launchAdamStep prior to use.
template <typename T>
void launchAdamStep(
    T *weightUpdate_d, T *gradient_d, T *m_d, T *v_d, float t, T alpha, T beta1, T beta2, T epsilon, uint32_t length, Stream stream) {
    float alphaT = (float)alpha * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

    ScopedGpu scopedGpu(stream.getGpuNum());

    // I declared the two kernels with different names just to be absolutely sure the right one is used.
    if (is_same<decltype(gradient_d), half *>::value) {
        dim3 blockSize(256);
        dim3 gridSize((length + 511) / 512);
        adamStep<<<gridSize, blockSize, 0, stream>>>(weightUpdate_d, gradient_d, m_d, v_d, (half)alphaT, beta1, beta2, epsilon, length);
    } else if (is_same<decltype(gradient_d), float *>::value) {
        dim3 blockSize(256);
        dim3 gridSize((length + 255) / 256);
        adamStep<<<gridSize, blockSize, 0, stream>>>(weightUpdate_d, gradient_d, m_d, v_d, alphaT, beta1, beta2, epsilon, length);
    } else {
        assert(false);
    }
}

template void launchAdamStep<half>(half *weightUpdate_d,
                                   half *gradient_d,
                                   half *m_d,
                                   half *v_d,
                                   float t,
                                   half alpha,
                                   half beta1,
                                   half beta2,
                                   half epsilon,
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