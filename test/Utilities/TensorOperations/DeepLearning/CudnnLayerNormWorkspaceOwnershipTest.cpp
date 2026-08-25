#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <optional>

using namespace ThorImplementation;
using namespace std;

namespace {

int cudaDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess)
        return 0;
    return count;
}

optional<Tensor> allocateWorkspace(const TensorPlacement& placement, uint64_t bytes) {
    if (bytes == 0)
        return nullopt;
    return Tensor(placement, TensorDescriptor(DataType::UINT8, {bytes}), 256);
}

CudnnLayerNormDescriptor makeTrainingDescriptor() {
    CudnnLayerNormDescriptor descriptor;
    descriptor.outerSize = 64;
    descriptor.normalizedFeatureCount = 128;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    descriptor.debugName = "layer_norm_workspace_ownership";
    return descriptor;
}

struct LayerNormExecutionTensors {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;
    Tensor mean;
    Tensor invVariance;
    Tensor dy;
    Tensor dx;
    Tensor dscale;
    Tensor dbias;
};

LayerNormExecutionTensors makeExecutionTensors(const TensorPlacement& placement, const CudnnLayerNormDescriptor& descriptor) {
    const uint64_t ioElements = descriptor.outerSize * descriptor.normalizedFeatureCount;
    LayerNormExecutionTensors tensors{
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.normalizedFeatureCount})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.normalizedFeatureCount})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {descriptor.outerSize})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {descriptor.outerSize})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.normalizedFeatureCount})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.normalizedFeatureCount})),
    };
    return tensors;
}

void initializeExecutionTensors(LayerNormExecutionTensors& tensors, Stream stream) {
    tensors.x.fill(0.5, stream);
    tensors.scale.fill(1.0, stream);
    tensors.bias.fill(0.0, stream);
    tensors.dy.fill(1.0, stream);
    stream.synchronize();
}

CudnnLayerNormForwardArgs forwardArgs(LayerNormExecutionTensors& tensors) {
    CudnnLayerNormForwardArgs args;
    args.x = tensors.x;
    args.scale = tensors.scale;
    args.bias = tensors.bias;
    args.y = tensors.y;
    args.mean = tensors.mean;
    args.invVariance = tensors.invVariance;
    return args;
}

CudnnLayerNormBackwardArgs backwardArgs(LayerNormExecutionTensors& tensors) {
    CudnnLayerNormBackwardArgs args;
    args.dy = tensors.dy;
    args.x = tensors.x;
    args.scale = tensors.scale;
    args.mean = tensors.mean;
    args.invVariance = tensors.invVariance;
    args.dx = tensors.dx;
    args.dscale = tensors.dscale;
    args.dbias = tensors.dbias;
    return args;
}

}  // namespace

TEST(LayerNormWorkspaceOwnership, EquivalentOperationsShareOnlySelectionAndSurviveSelectionCacheClear) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for LayerNorm workspace ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnLayerNorm& layerNorm = CudnnLayerNorm::instance();
    layerNorm.clearSelectionCache();

    const uint64_t hitsBefore = layerNorm.selectionCacheHitCount();
    const uint64_t missesBefore = layerNorm.selectionCacheMissCount();
    const CudnnLayerNormDescriptor descriptor = makeTrainingDescriptor();

    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    CudnnLayerNormExecutablePlan forwardPlanA = layerNorm.prepareForward(descriptor, streamA);
    CudnnLayerNormExecutablePlan forwardPlanB = layerNorm.prepareForward(descriptor, streamB);
    CudnnLayerNormExecutablePlan backwardPlanA = layerNorm.prepareBackward(descriptor, streamA);
    CudnnLayerNormExecutablePlan backwardPlanB = layerNorm.prepareBackward(descriptor, streamB);
    const uint64_t preparationsAfterStamping = cudnnFrontendExecutablePreparationCountForTests();

    ASSERT_EQ(layerNorm.cachedSelectionCount(), 2U);
    EXPECT_EQ(layerNorm.selectionCacheMissCount() - missesBefore, 2U);
    EXPECT_EQ(layerNorm.selectionCacheHitCount() - hitsBefore, 2U);

    // Identical independently executable operations reuse the immutable recipe,
    // never the finalized Frontend graph.
    EXPECT_EQ(forwardPlanA.selection(), forwardPlanB.selection());
    EXPECT_EQ(backwardPlanA.selection(), backwardPlanB.selection());
    EXPECT_NE(forwardPlanA.executableId(), forwardPlanB.executableId());
    EXPECT_NE(backwardPlanA.executableId(), backwardPlanB.executableId());

    optional<Tensor> forwardWorkspaceA = allocateWorkspace(placement, forwardPlanA.workspaceBytes());
    optional<Tensor> forwardWorkspaceB = allocateWorkspace(placement, forwardPlanB.workspaceBytes());
    optional<Tensor> backwardWorkspaceA = allocateWorkspace(placement, backwardPlanA.workspaceBytes());
    optional<Tensor> backwardWorkspaceB = allocateWorkspace(placement, backwardPlanB.workspaceBytes());

    if (forwardPlanA.workspaceBytes() > 0) {
        ASSERT_TRUE(forwardWorkspaceA.has_value());
        ASSERT_TRUE(forwardWorkspaceB.has_value());
        EXPECT_NE(forwardWorkspaceA->getMemPtr<void>(), forwardWorkspaceB->getMemPtr<void>());
    }
    if (backwardPlanA.workspaceBytes() > 0) {
        ASSERT_TRUE(backwardWorkspaceA.has_value());
        ASSERT_TRUE(backwardWorkspaceB.has_value());
        EXPECT_NE(backwardWorkspaceA->getMemPtr<void>(), backwardWorkspaceB->getMemPtr<void>());
    }

    LayerNormExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    LayerNormExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    initializeExecutionTensors(tensorsA, streamA);
    initializeExecutionTensors(tensorsB, streamB);

    // Runtime must have no dependency on global selection state. Clearing the
    // process-global recipes after preparation cannot invalidate either owner.
    layerNorm.clearSelectionCache();
    ASSERT_EQ(layerNorm.cachedSelectionCount(), 0U);

    CudnnLayerNormForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnLayerNormForwardArgs forwardB = forwardArgs(tensorsB);
    for (int repetition = 0; repetition < 4; ++repetition) {
        layerNorm.forward(forwardPlanA, forwardA, forwardWorkspaceA, streamA);
        layerNorm.forward(forwardPlanB, forwardB, forwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();
    EXPECT_EQ(layerNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping)
        << "LayerNorm forward hot path must not replay/build/deserialize a cuDNN Frontend plan.";

    CudnnLayerNormBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnLayerNormBackwardArgs backwardB = backwardArgs(tensorsB);
    for (int repetition = 0; repetition < 4; ++repetition) {
        layerNorm.backward(backwardPlanA, backwardA, backwardWorkspaceA, streamA);
        layerNorm.backward(backwardPlanB, backwardB, backwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();
    EXPECT_EQ(layerNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping)
        << "LayerNorm backward hot path must not replay/build/deserialize a cuDNN Frontend plan.";
}
