#include "Thor.h"

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using namespace ThorImplementation;
using namespace std;

TEST(AdamKernel, Fp32) {
    srand(time(nullptr));
    Stream stream(0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        uint32_t numWeights = (rand() % 3000) + 1;
        Tensor weightsUpdateGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor gradientGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor mGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));
        Tensor vGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {numWeights}));

        vector<float> gradient;
        for (uint32_t i = 0; i < numWeights; ++i) {
            gradient.push_back(rand() % 20 / ((float)(rand() % 20) + 1));
            if (rand() % 2)
                gradient.back() *= -1;
        }
        gradientGpu.setValues(gradient, stream);
        vector<float> mInitial;
        for (uint32_t i = 0; i < numWeights; ++i) {
            mInitial.push_back(rand() % 20 / ((float)(rand() % 20) + 1));
        }
        mGpu.setValues(mInitial, stream);
        vector<float> vInitial;
        for (uint32_t i = 0; i < numWeights; ++i) {
            vInitial.push_back(rand() % 20 / ((float)(rand() % 20) + 1));
        }
        vGpu.setValues(vInitial, stream);

        float t = 2.0f;
        float alpha = 0.005f;
        float beta1 = 0.8f;
        float beta2 = 0.777f;
        float epsilon = 0.000001f;

        launchAdamStep<float>((float *)weightsUpdateGpu.getMemPtr(),
                              (float *)gradientGpu.getMemPtr(),
                              (float *)mGpu.getMemPtr(),
                              (float *)vGpu.getMemPtr(),
                              t,
                              alpha,
                              beta1,
                              beta2,
                              epsilon,
                              numWeights,
                              stream);

        vector<float> m;
        mGpu.loadValuesIntoVector(m, stream);
        for (uint32_t i; i < numWeights; ++i) {
            float expected = beta1 * mInitial[i] + (1.0f - beta1) * gradient[i];
            ASSERT_LT(abs(m[i] - expected), 0.00001);
        }

        vector<float> v;
        vGpu.loadValuesIntoVector(v, stream);
        for (uint32_t i; i < numWeights; ++i) {
            float expected = beta2 * vInitial[i] + (1.0f - beta2) * (gradient[i] * gradient[i]);
            ASSERT_LT(abs(v[i] - expected), 0.00001);
        }

        vector<float> weightsUpdate;
        weightsUpdateGpu.loadValuesIntoVector(weightsUpdate, stream);
        for (uint32_t i; i < numWeights; ++i) {
            float alphaT = (float)alpha * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
            float expected = (-alphaT * m[i]) / (sqrtf(v[i]) + epsilon);
            // printf("%f %f %f %f %f %f %f\n", weightsUpdate[i], expected, gradient[i], mInitial[i], vInitial[i], m[i], v[i]);
            ASSERT_LT(abs(weightsUpdate[i] - expected), 0.00001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            assert(isfinite(weightsUpdate[i]));
            assert(isfinite(gradient[i]));
            assert(isfinite(m[i]));
            assert(isfinite(v[i]));
        }
    }
}

TEST(AdamKernel, Fp16) {
    srand(time(nullptr));
    Stream stream(0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        uint32_t numWeights = (rand() % 3000) + 1;
        Tensor weightsUpdateGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor gradientGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor mGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));
        Tensor vGpu(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numWeights}));

        vector<half> gradient;
        for (uint32_t i = 0; i < numWeights; ++i) {
            gradient.push_back(rand() % 20 / ((float)(rand() % 20) + 1));
            if (rand() % 2)
                gradient.back() *= -1;
        }
        gradientGpu.setValues(gradient, stream);
        vector<half> mInitial;
        for (uint32_t i = 0; i < numWeights; ++i) {
            mInitial.push_back(rand() % 2000 / 1000.0f);
        }
        mGpu.setValues(mInitial, stream);
        vector<half> vInitial;
        for (uint32_t i = 0; i < numWeights; ++i) {
            vInitial.push_back(rand() % 2000 / 1000.0f);
        }
        vGpu.setValues(vInitial, stream);

        float t = 2.0f;
        half alpha = 0.005f;
        half beta1 = 0.8f;
        half beta2 = 0.777f;
        half epsilon = 0.000001f;

        launchAdamStep<half>((half *)weightsUpdateGpu.getMemPtr(),
                             (half *)gradientGpu.getMemPtr(),
                             (half *)mGpu.getMemPtr(),
                             (half *)vGpu.getMemPtr(),
                             t,
                             alpha,
                             beta1,
                             beta2,
                             epsilon,
                             numWeights,
                             stream);

        vector<half> m;
        mGpu.loadValuesIntoVector(m, stream);
        for (uint32_t i; i < numWeights; ++i) {
            half expected = beta1 * mInitial[i] + ((half)1.0f - beta1) * gradient[i];
            ASSERT_LT(abs((float)(m[i] - expected)), 0.00001);
        }

        vector<half> v;
        vGpu.loadValuesIntoVector(v, stream);
        for (uint32_t i; i < numWeights; ++i) {
            half expected = beta2 * vInitial[i] + ((half)1.0f - beta2) * (gradient[i] * gradient[i]);
            ASSERT_LT(abs((float)(v[i] - expected)), 0.00001);
        }

        vector<half> weightsUpdate;
        weightsUpdateGpu.loadValuesIntoVector(weightsUpdate, stream);
        for (uint32_t i; i < numWeights; ++i) {
            half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
            half expected = (-alphaT * m[i]) / ((half)sqrtf(v[i]) + epsilon);
            // printf("%f %f %f %f %f %f %f\n", (float)weightsUpdate[i], (float)expected, (float)gradient[i], (float)mInitial[i],
            // (float)vInitial[i], (float)m[i], (float)v[i]);
            ASSERT_LT(abs((float)(weightsUpdate[i] - expected)), 0.00001);
        }

        for (uint32_t i; i < numWeights; ++i) {
            assert(isfinite((float)weightsUpdate[i]));
            assert(isfinite((float)gradient[i]));
            assert(isfinite((float)m[i]));
            assert(isfinite((float)v[i]));
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
