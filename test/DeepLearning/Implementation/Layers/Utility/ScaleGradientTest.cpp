#include "DeepLearning/Implementation/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;

namespace {

template <typename T>
struct ScaleGradientDType;

template <>
struct ScaleGradientDType<float> {
    static constexpr DataType value = DataType::FP32;
    static constexpr float tolerance = 0.0f;
};

template <>
struct ScaleGradientDType<half> {
    static constexpr DataType value = DataType::FP16;
    static constexpr float tolerance = 1.0e-3f;
};

template <>
struct ScaleGradientDType<__nv_bfloat16> {
    static constexpr DataType value = DataType::BF16;
    static constexpr float tolerance = 1.0e-2f;
};

template <typename T>
void expectBackwardScale(float scale) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(ScaleGradientDType<T>::value, {2, 4});

    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    Tensor destCpu(cpuPlacement, descriptor);
    Tensor destGpu(gpuPlacement, descriptor);

    const std::vector<float> values = {1.0f, -2.0f, 3.5f, 4.25f, -5.0f, 6.0f, -7.25f, 8.5f};
    T *source = static_cast<T *>(sourceCpu.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        source[i] = T(values[i]);

    Stream stream(0);
    sourceGpu.copyFromAsync(sourceCpu, stream);

    ScaleGradient layer(scale);
    layer.backProp(std::nullopt, sourceGpu, destGpu, stream);
    destCpu.copyFromAsync(destGpu, stream);
    stream.synchronize();

    const T *dest = static_cast<const T *>(destCpu.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        EXPECT_NEAR(static_cast<float>(dest[i]), values[i] * scale, ScaleGradientDType<T>::tolerance) << "index " << i;
}

}  // namespace

TEST(ScaleGradient, BackwardScalesFp32) { expectBackwardScale<float>(0.125f); }
TEST(ScaleGradient, BackwardScalesFp16) { expectBackwardScale<half>(-0.5f); }
TEST(ScaleGradient, BackwardScalesBf16) { expectBackwardScale<__nv_bfloat16>(0.25f); }
TEST(ScaleGradient, ZeroScaleProducesZeroGradient) { expectBackwardScale<float>(0.0f); }

TEST(ScaleGradient, RejectsNonFiniteScale) {
    EXPECT_THROW(ScaleGradient(std::numeric_limits<float>::infinity()), std::logic_error);
    EXPECT_THROW(ScaleGradient(std::numeric_limits<float>::quiet_NaN()), std::logic_error);
}
