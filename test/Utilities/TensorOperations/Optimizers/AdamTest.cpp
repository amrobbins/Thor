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
    Stream stream(0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    for (uint32_t test = 0; test < 100; ++test) {
        srand(0x9E3779B9u * (test + 1u));
        uint32_t numWeights;
        if (test == 0)
            numWeights = 4095;
        else if (test == 1)
            numWeights = 4096;
        else if (test == 2)
            numWeights = 4097;
        else
            numWeights = (rand() % 3000) + 1;
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
        mGpu.copyFromAsync(mInitial, stream);
        vGpu.copyFromAsync(vInitial, stream);

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

        for (uint32_t i = 0; i < numWeights; ++i) {
            float expected = beta1 * mInitialMem[i] + (1.0f - beta1) * gradientMem[i];
            ASSERT_LT(abs(mGpuMem_h[i] - expected), 0.00001);
        }

        for (uint32_t i = 0; i < numWeights; ++i) {
            float expected = beta2 * vInitialMem[i] + (1.0f - beta2) * (gradientMem[i] * gradientMem[i]);
            ASSERT_LT(abs(vGpuMem_h[i] - expected), 0.00001);
        }

        for (uint32_t i = 0; i < numWeights; ++i) {
            float alphaT = (float)alpha * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
            float expected = (-alphaT * mGpuMem_h[i]) / (sqrtf(vGpuMem_h[i]) + epsilon);
            ASSERT_LT(abs(weightsUpdateGpuMem_h[i] - expected), 0.00001);
        }

        for (uint32_t i = 0; i < numWeights; ++i) {
            assert(isfinite(weightsUpdateGpuMem_h[i]));
            assert(isfinite(gradientMem[i]));
            assert(isfinite(mGpuMem_h[i]));
            assert(isfinite(vGpuMem_h[i]));
        }
    }
}

TEST(AdamKernel, Fp16) {
    Stream stream(0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    for (uint32_t test = 0; test < 100; ++test) {
        srand(0x9E3779B9u * (test + 1u));
        uint32_t numWeights;
        if (test == 0)
            numWeights = 4095;
        else if (test == 1)
            numWeights = 4096;
        else if (test == 2)
            numWeights = 4097;
        else
            numWeights = (rand() % 3000) + 1;
        Tensor weightsUpdateGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor gradientGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor mGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor vGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));

        Tensor gradient = gradientGpu.clone(cpuPlacement);
        Tensor mInitial = mGpu.clone(cpuPlacement);
        Tensor vInitial = vGpu.clone(cpuPlacement);
        half *gradientMem = gradient.getMemPtr<half>();
        float *mInitialMem = mInitial.getMemPtr<float>();
        float *vInitialMem = vInitial.getMemPtr<float>();
        for (uint32_t i = 0; i < numWeights; ++i) {
            gradientMem[i] = rand() % 20 / ((float)(rand() % 20) + 1);
            if (rand() % 2)
                gradientMem[i] *= -1;
            mInitialMem[i] = rand() % 2000 / ((float)(rand() % 1000) + 1);
            vInitialMem[i] = rand() % 2000 / ((float)(rand() % 1000) + 1);
        }
        gradientGpu.copyFromAsync(gradient, stream);
        mGpu.copyFromAsync(mInitial, stream);
        vGpu.copyFromAsync(vInitial, stream);

        float t = 2.0f;
        float alpha = 0.005f;
        float beta1 = 0.8f;
        float beta2 = 0.777f;
        float epsilon = 0.0001f;

        launchAdamStep<half>(weightsUpdateGpu.getMemPtr<half>(),
                             gradientGpu.getMemPtr<half>(),
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
        half *weightsUpdateGpuMem_h = weightsUpdateGpu_h.getMemPtr<half>();
        mGpu_h.copyFromAsync(mGpu, stream);
        vGpu_h.copyFromAsync(vGpu, stream);
        weightsUpdateGpu_h.copyFromAsync(weightsUpdateGpu, stream);
        stream.synchronize();

        for (uint32_t i = 0; i < numWeights; ++i) {
            float expected = beta1 * mInitialMem[i] + (1.0f - beta1) * float(gradientMem[i]);
            float actual = mGpuMem_h[i];
            ASSERT_LT(abs(actual - expected), 0.0005);
        }

        for (uint32_t i = 0; i < numWeights; ++i) {
            float expected = beta2 * vInitialMem[i] + (1.0f - beta2) * float(gradientMem[i]) * float(gradientMem[i]);
            float actual = vGpuMem_h[i];
            ASSERT_LT(abs(actual - expected), 0.0005);
        }

        for (uint32_t i = 0; i < numWeights; ++i) {
            float alphaT = alpha * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
            float expectedF = (-alphaT * mGpuMem_h[i]) / (sqrtf(vGpuMem_h[i]) + epsilon);
            expectedF = fminf(fmaxf(expectedF, -65504.0f), 65504.0f);
            half expected = __float2half_rn(expectedF);
            half actual = weightsUpdateGpuMem_h[i];
            EXPECT_LT(abs(__half2float(actual - expected)), 0.0005);
        }

        for (uint32_t i = 0; i < numWeights; ++i) {
            assert(isfinite((float)weightsUpdateGpuMem_h[i]));
            assert(isfinite((float)gradientMem[i]));
            assert(isfinite(mGpuMem_h[i]));
            assert(isfinite(vGpuMem_h[i]));
        }
    }
}
