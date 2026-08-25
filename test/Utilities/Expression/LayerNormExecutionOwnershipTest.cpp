#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

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
