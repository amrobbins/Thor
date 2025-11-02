#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Optimizers/Adam.h"

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ThorImplementation;
using namespace std;

TEST(AdamKernel, Fp32) {
    srand(time(nullptr));
    Stream stream(0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        uint32_t numWeights = (rand() % 3000) + 1;
        Tensor weightsUpdateGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor gradientGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor mGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor vGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));

        Tensor gradient = gradientGpu.clone(cpuPlacement);
        Tensor mInitial = mGpu.clone(cpuPlacement);
        Tensor vInitial = vGpu.clone(cpuPlacement);
        float *gradientMem = gradient.getMemPtr<float>();
        float *mInitialMem = mInitial.getMemPtr<float>();
        float *vInitialMem = vInitial.getMemPtr<float>();
        for (uint32_t i = 0; i < numWeights; ++i) {
            gradientMem[i] = rand() % 20 / ((float)(rand() % 20) + 1);
            if (rand() % 2)
                gradientMem[i] *= -1;
            mInitialMem[i] = rand() % 20 / ((float)(rand() % 20) + 1);
            vInitialMem[i] = rand() % 20 / ((float)(rand() % 20) + 1);
        }
        gradientGpu.copyFromAsync(gradient, stream);
        mInitial.copyFromAsync(mGpu, stream);
        vInitial.copyFromAsync(vGpu, stream);

        float t = 5.0f;
        float alpha = 0.015f;
        float beta1 = 0.6f;
        float beta2 = 0.85f;
        float epsilon = 0.000001f;

        launchAdamStep<float>(weightsUpdateGpu.getMemPtr<float>(),
                              gradientGpu.getMemPtr<float>(),
                              mGpu.getMemPtr<float>(),
                              vGpu.getMemPtr<float>(),
                              t,
                              alpha,
                              beta1,
                              beta2,
                              epsilon,
                              numWeights,
                              stream);

        Tensor mGpu_h = mGpu.clone(cpuPlacement);
        Tensor vGpu_h = vGpu.clone(cpuPlacement);
        Tensor weightsUpdateGpu_h = weightsUpdateGpu.clone(cpuPlacement);
        float *mGpuMem_h = mGpu_h.getMemPtr<float>();
        float *vGpuMem_h = vGpu_h.getMemPtr<float>();
        float *weightsUpdateGpuMem_h = weightsUpdateGpu_h.getMemPtr<float>();
        mGpu_h.copyFromAsync(mGpu, stream);
        vGpu_h.copyFromAsync(vGpu, stream);
        weightsUpdateGpu_h.copyFromAsync(weightsUpdateGpu, stream);
        stream.synchronize();

        for (uint32_t i; i < numWeights; ++i) {
            float expected = beta1 * mInitialMem[i] + (1.0f - beta1) * gradientMem[i];
            ASSERT_LT(abs(mGpuMem_h[i] - expected), 0.00001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            float expected = beta2 * vInitialMem[i] + (1.0f - beta2) * (gradientMem[i] * gradientMem[i]);
            ASSERT_LT(abs(vGpuMem_h[i] - expected), 0.00001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            float alphaT = (float)alpha * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
            float expected = (-alphaT * mGpuMem_h[i]) / (sqrtf(vGpuMem_h[i]) + epsilon);
            ASSERT_LT(abs(weightsUpdateGpuMem_h[i] - expected), 0.00001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            assert(isfinite(weightsUpdateGpuMem_h[i]));
            assert(isfinite(gradientMem[i]));
            assert(isfinite(mGpuMem_h[i]));
            assert(isfinite(vGpuMem_h[i]));
        }
    }
}

TEST(AdamKernel, Fp16) {
    srand(time(nullptr));
    Stream stream(0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        uint32_t numWeights = (rand() % 3000) + 1;
        Tensor weightsUpdateGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor gradientGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor mGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor vGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));

        Tensor gradient = gradientGpu.clone(cpuPlacement);
        Tensor mInitial = mGpu.clone(cpuPlacement);
        Tensor vInitial = vGpu.clone(cpuPlacement);
        half *gradientMem = gradient.getMemPtr<half>();
        half *mInitialMem = mInitial.getMemPtr<half>();
        half *vInitialMem = vInitial.getMemPtr<half>();
        for (uint32_t i = 0; i < numWeights; ++i) {
            gradientMem[i] = rand() % 20 / ((float)(rand() % 20) + 1);
            if (rand() % 2)
                gradientMem[i] *= -1;
            mInitialMem[i] = rand() % 2000 / ((float)(rand() % 1000) + 1);
            vInitialMem[i] = rand() % 2000 / ((float)(rand() % 1000) + 1);
        }
        gradientGpu.copyFromAsync(gradient, stream);
        mInitial.copyFromAsync(mGpu, stream);
        vInitial.copyFromAsync(vGpu, stream);

        float t = 2.0f;
        half alpha = 0.005f;
        half beta1 = 0.8f;
        half beta2 = 0.777f;
        half epsilon = 0.000001f;

        launchAdamStep<half>(weightsUpdateGpu.getMemPtr<half>(),
                             gradientGpu.getMemPtr<half>(),
                             mGpu.getMemPtr<half>(),
                             vGpu.getMemPtr<half>(),
                             t,
                             alpha,
                             beta1,
                             beta2,
                             epsilon,
                             numWeights,
                             stream);

        Tensor mGpu_h = mGpu.clone(cpuPlacement);
        Tensor vGpu_h = vGpu.clone(cpuPlacement);
        Tensor weightsUpdateGpu_h = weightsUpdateGpu.clone(cpuPlacement);
        half *mGpuMem_h = mGpu_h.getMemPtr<half>();
        half *vGpuMem_h = vGpu_h.getMemPtr<half>();
        half *weightsUpdateGpuMem_h = weightsUpdateGpu_h.getMemPtr<half>();
        mGpu_h.copyFromAsync(mGpu, stream);
        vGpu_h.copyFromAsync(vGpu, stream);
        weightsUpdateGpu_h.copyFromAsync(weightsUpdateGpu, stream);
        stream.synchronize();

        for (uint32_t i; i < numWeights; ++i) {
            half expected = beta1 * mInitialMem[i] + ((half)1.0f - beta1) * gradientMem[i];
            ASSERT_LT(abs((float)(mGpuMem_h[i] - expected)), 0.001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            half expected = beta2 * vInitialMem[i] + ((half)1.0f - beta2) * (gradientMem[i] * gradientMem[i]);
            ASSERT_LT(abs((float)(vGpuMem_h[i] - expected)), 0.001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
            half expected = (-alphaT * mGpuMem_h[i]) / ((half)sqrtf(vGpuMem_h[i]) + epsilon);
            half actual = weightsUpdateGpuMem_h[i];
            printf("i=%d expected %f actual %f\n", i, (float)expected, (float)actual);
            EXPECT_LT(abs((float)(actual - expected)), 0.001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            assert(isfinite((float)weightsUpdateGpuMem_h[i]));
            assert(isfinite((float)gradientMem[i]));
            assert(isfinite((float)mGpuMem_h[i]));
            assert(isfinite((float)vGpuMem_h[i]));
        }
    }
}
