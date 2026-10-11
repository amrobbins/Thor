#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <memory>

using namespace ThorImplementation;
using namespace std;

namespace {

int cudaDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess ? count : 0;
}

shared_ptr<CompiledLayerNorm> makeCompiledLayerNorm() {
    auto compiled = make_shared<CompiledLayerNorm>();
    compiled->normalized_feature_count = 128;
    compiled->epsilon = 1.0e-5;
    compiled->input_dtype = DataType::FP32;
    compiled->scale_dtype = DataType::FP32;
    compiled->bias_dtype = DataType::FP32;
    compiled->output_dtype = DataType::FP32;
    compiled->compute_dtype = DataType::FP32;
    compiled->debug_name = "stamped_layer_norm_execution_ownership";
    return compiled;
}

}  // namespace

TEST(LayerNormExecutionOwnership, EquivalentStampedOperationsShareSelectionButNotExecutable) {
    if (cudaDeviceCount() < 1) {
        GTEST_SKIP() << "CUDA device is required for LayerNorm execution ownership tests.";
    }

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnLayerNorm& layerNorm = CudnnLayerNorm::instance();
    layerNorm.clearSelectionCache();

    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    Tensor inputA(placement, TensorDescriptor(DataType::FP32, {64, 128}));
    Tensor inputB(placement, TensorDescriptor(DataType::FP32, {64, 128}));
    Tensor scale(placement, TensorDescriptor(DataType::FP32, {128}));
    Tensor bias(placement, TensorDescriptor(DataType::FP32, {128}));
    Tensor outputA(placement, TensorDescriptor(DataType::FP32, {64, 128}));
    Tensor outputB(placement, TensorDescriptor(DataType::FP32, {64, 128}));

    inputA.fill(0.5, streamA);
    inputB.fill(0.75, streamB);
    scale.fill(1.0, streamA);
    bias.fill(0.0, streamA);
    streamA.synchronize();
    streamB.synchronize();

    shared_ptr<CompiledLayerNorm> compiled = makeCompiledLayerNorm();
    StampedLayerNorm stampedA(compiled, inputA, scale, bias, outputA, streamA);
    StampedLayerNorm stampedB(compiled, inputB, scale, bias, outputB, streamB);

    ASSERT_NE(stampedA.executablePlanId(), 0U);
    ASSERT_NE(stampedB.executablePlanId(), 0U);
    EXPECT_EQ(stampedA.planSelection(), stampedB.planSelection());
    EXPECT_NE(stampedA.executablePlanId(), stampedB.executablePlanId());
    EXPECT_EQ(layerNorm.cachedSelectionCount(), 1U);

    layerNorm.clearSelectionCache();
    ASSERT_EQ(layerNorm.cachedSelectionCount(), 0U);

    stampedA.runOn(streamA);
    stampedB.runOn(streamB);
    streamA.synchronize();
    streamB.synchronize();

    EXPECT_EQ(layerNorm.cachedSelectionCount(), 0U)
        << "stamped execution must not consult or repopulate the process-global selection cache";
}

TEST(LayerNormForwardTrainingState1C1, RetentionAtPlacementProducesMeanAndInverseVariance) {
    if (cudaDeviceCount() < 1) {
        GTEST_SKIP() << "CUDA device required for cuDNN LayerNorm training statistics.";
    }
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    Stream stream(0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor scale(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor y(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    x.fill(0.5, stream);
    scale.fill(1.0, stream);
    bias.fill(0.0, stream);

    auto state = make_shared<LayerNormForwardState>();
    StampedLayerNorm stamped(makeCompiledLayerNorm(), x, scale, bias, y, stream, std::nullopt, state);
    ASSERT_EQ(stamped.executablePlanCount(), 1U);
    EXPECT_FALSE(state->retain_for_backward);
    EXPECT_FALSE(state->has_valid_state);
    EXPECT_FALSE(state->mean.isInitialized());
    stamped.runOn(stream); // Inference still does not create statistics.
    EXPECT_FALSE(state->has_valid_state);

    stamped.retainForwardStateForBackward();
    ASSERT_TRUE(state->retain_for_backward);
    ASSERT_TRUE(state->mean.isInitialized());
    ASSERT_TRUE(state->inv_variance.isInitialized());
    EXPECT_EQ(state->mean.getDimensions(), (vector<uint64_t>{8}));
    EXPECT_EQ(state->inv_variance.getDimensions(), (vector<uint64_t>{8}));

    CudnnLayerNorm::instance().clearSelectionCache();
    stamped.runOn(stream);
    stream.synchronize();
    EXPECT_EQ(CudnnLayerNorm::instance().cachedSelectionCount(), 0U)
        << "training execution may not prepare cuDNN plans inside runOn";
    ASSERT_TRUE(state->has_valid_state);
    EXPECT_EQ(state->packed_active_rows, 0U);

    Tensor host_mean(cpu, TensorDescriptor(DataType::FP32, {8}));
    Tensor host_inv_variance(cpu, TensorDescriptor(DataType::FP32, {8}));
    host_mean.copyFromAsync(state->mean, stream);
    host_inv_variance.copyFromAsync(state->inv_variance, stream);
    stream.synchronize();
    for (uint64_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(host_mean.getMemPtr<float>()[i], 0.5f, 1.0e-3f);
        EXPECT_TRUE(std::isfinite(host_inv_variance.getMemPtr<float>()[i]));
        EXPECT_NEAR(host_inv_variance.getMemPtr<float>()[i], 1.0f / std::sqrt(1.0e-5f), 1.0f);
    }
}

TEST(LayerNormForwardTrainingState1C1, IndependentlyStampedTrainingStatesDoNotSharePlansOrBuffers) {
    if (cudaDeviceCount() < 1) {
        GTEST_SKIP() << "CUDA device required for cuDNN LayerNorm training ownership.";
    }
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Stream first(0), second(0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor gamma(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor beta(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor y1(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    Tensor y2(gpu, TensorDescriptor(DataType::FP32, {4, 128}));
    x.fill(0.25, first);
    gamma.fill(1.0, first);
    beta.fill(0.0, first);
    first.synchronize();

    auto compiled = makeCompiledLayerNorm();
    auto first_state = make_shared<LayerNormForwardState>();
    auto second_state = make_shared<LayerNormForwardState>();
    StampedLayerNorm one(compiled, x, gamma, beta, y1, first, std::nullopt, first_state);
    StampedLayerNorm two(compiled, x, gamma, beta, y2, second, std::nullopt, second_state);
    one.retainForwardStateForBackward();
    two.retainForwardStateForBackward();
    EXPECT_EQ(one.planSelection(), two.planSelection());
    EXPECT_NE(one.executablePlanId(), two.executablePlanId());
    EXPECT_NE(first_state->mean.getMemPtr<void>(), second_state->mean.getMemPtr<void>());
    EXPECT_NE(first_state->inv_variance.getMemPtr<void>(), second_state->inv_variance.getMemPtr<void>());
    one.runOn(first);
    two.runOn(second);
    first.synchronize();
    second.synchronize();
    EXPECT_TRUE(first_state->has_valid_state);
    EXPECT_TRUE(second_state->has_valid_state);
}

TEST(LayerNormForwardTrainingState1C1, RejectsIncorrectPreallocatedStatisticsShape) {
    if (cudaDeviceCount() < 1) {
        GTEST_SKIP() << "CUDA device required for LayerNorm statistics allocation checks.";
    }
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor gamma(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor beta(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor y(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    auto state = make_shared<LayerNormForwardState>();
    state->mean = Tensor(gpu, TensorDescriptor(DataType::FP32, {7}));
    StampedLayerNorm stamped(makeCompiledLayerNorm(), x, gamma, beta, y, stream, std::nullopt, state);
    EXPECT_THROW(stamped.retainForwardStateForBackward(), std::runtime_error);
    EXPECT_FALSE(state->has_valid_state);
}
