#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

using namespace ThorImplementation;

namespace {

bool hasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::shared_ptr<CompiledLayerNorm> forwardDescriptor() {
    auto c = std::make_shared<CompiledLayerNorm>();
    c->normalized_feature_count = 128;
    c->epsilon = 1.0e-5;
    c->input_dtype = DataType::FP32;
    c->scale_dtype = DataType::FP32;
    c->bias_dtype = DataType::FP32;
    c->output_dtype = DataType::FP32;
    c->compute_dtype = DataType::FP32;
    return c;
}

std::shared_ptr<CompiledLayerNormBackward> backwardDescriptor() {
    auto c = std::make_shared<CompiledLayerNormBackward>();
    c->normalized_feature_count = 128;
    c->epsilon = 1.0e-5;
    c->input_dtype = DataType::FP32;
    c->scale_dtype = DataType::FP32;
    c->dy_dtype = DataType::FP32;
    c->dx_dtype = DataType::FP32;
    c->dscale_dtype = DataType::FP32;
    c->dbias_dtype = DataType::FP32;
    c->compute_dtype = DataType::FP32;
    c->produces_dx = true;
    c->produces_dscale = true;
    c->produces_dbias = true;
    return c;
}

}  // namespace

TEST(LayerNormStampedBackward1C2B2A, MustLinkRealForwardAndRejectsDifferentInput) {
    if (!hasCudaDevice()) GTEST_SKIP() << "CUDA device required";
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor other_x(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor scale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor y(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor dx(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor dscale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor dbias(gpu, TensorDescriptor(DataType::FP32, {128}));
    auto forward = std::make_shared<StampedLayerNorm>(forwardDescriptor(), x, scale, bias, y, stream);
    StampedLayerNormBackward backward(backwardDescriptor(), x, scale, dy, dx, dscale, dbias, stream);
    EXPECT_EQ(backward.backwardExecutablePlanCount(), 1U);
    EXPECT_THROW(backward.runOn(stream), std::runtime_error);
    EXPECT_FALSE(forward->canProvideForwardStateFor(*backwardDescriptor(), other_x, scale, dy, std::nullopt));
    auto incompatible_epsilon = backwardDescriptor();
    incompatible_epsilon->epsilon = 1.0e-4;
    EXPECT_FALSE(forward->canProvideForwardStateFor(*incompatible_epsilon, x, scale, dy, std::nullopt));
    EXPECT_TRUE(backward.tryLinkForwardStateFrom(forward));
    EXPECT_TRUE(forward->trainingForwardState()->retain_for_backward);
    EXPECT_THROW(backward.runOn(stream), std::runtime_error) << "Only the real forward can produce valid statistics";
}

TEST(LayerNormStampedBackward1C2B2A, DenseBackwardExecutesWithRetainedStatisticsAndOwnsPlan) {
    if (!hasCudaDevice()) GTEST_SKIP() << "CUDA device required";
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    Stream stream(0);
    Tensor host_x(cpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor host_dy(cpu, TensorDescriptor(DataType::FP32, {4, 128}));
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 128; ++col) {
            host_x.getMemPtr<float>()[row * 128 + col] = static_cast<float>(col % 17) * 0.05f + static_cast<float>(row);
            host_dy.getMemPtr<float>()[row * 128 + col] = 0.25f + static_cast<float>(col % 7) * 0.125f;
        }
    }
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor scale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor y(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor dx(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor dscale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor dbias(gpu, TensorDescriptor(DataType::FP32, {128}));
    x.copyFromAsync(host_x, stream);
    dy.copyFromAsync(host_dy, stream);
    scale.fill(1.3, stream);
    bias.fill(-0.15, stream);

    auto compiled_forward = forwardDescriptor();
    auto compiled_backward = backwardDescriptor();
    auto forward = std::make_shared<StampedLayerNorm>(compiled_forward, x, scale, bias, y, stream);
    StampedLayerNormBackward backward(compiled_backward, x, scale, dy, dx, dscale, dbias, stream);
    StampedLayerNormBackward other_backward(compiled_backward, x, scale, dy, dx, dscale, dbias, stream);
    ASSERT_TRUE(backward.tryLinkForwardStateFrom(forward));
    ASSERT_TRUE(other_backward.tryLinkForwardStateFrom(forward));
    ASSERT_EQ(backward.backwardExecutablePlanIds().size(), 1U);
    ASSERT_EQ(other_backward.backwardExecutablePlanIds().size(), 1U);
    EXPECT_NE(backward.backwardExecutablePlanIds().front(), other_backward.backwardExecutablePlanIds().front());

    CudnnLayerNorm::instance().clearSelectionCache();
    forward->runOn(stream);
    backward.runOn(stream);
    EXPECT_EQ(CudnnLayerNorm::instance().cachedSelectionCount(), 0U)
        << "Running backward must not compile or select cuDNN plans";

    Tensor host_dbias(cpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor host_dscale(cpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor host_dx(cpu, TensorDescriptor(DataType::FP32, {4, 128}));
    host_dbias.copyFromAsync(dbias, stream);
    host_dscale.copyFromAsync(dscale, stream);
    host_dx.copyFromAsync(dx, stream);
    stream.synchronize();
    for (size_t col = 0; col < 128; ++col) {
        const float expected_bias = 4.0f * (0.25f + static_cast<float>(col % 7) * 0.125f);
        EXPECT_NEAR(host_dbias.getMemPtr<float>()[col], expected_bias, 1.0e-3f);
        EXPECT_TRUE(std::isfinite(host_dscale.getMemPtr<float>()[col]));
    }
    for (size_t idx = 0; idx < 4 * 128; ++idx) {
        EXPECT_TRUE(std::isfinite(host_dx.getMemPtr<float>()[idx]));
    }
}

TEST(LayerNormStampedBackward1C2B2A, PackedCannotSilentlyReduceInactiveBucketTail) {
    if (!hasCudaDevice()) GTEST_SKIP() << "CUDA device required";
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);
    auto c = backwardDescriptor();
    c->packed_row_capacity = 4;
    c->ragged_batch_size = 2;
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor scale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor dx(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor dscale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor dbias(gpu, TensorDescriptor(DataType::FP32, {128}));
    EXPECT_THROW((void)StampedLayerNormBackward(c, x, scale, dy, dx, dscale, dbias, stream), std::runtime_error);
}
