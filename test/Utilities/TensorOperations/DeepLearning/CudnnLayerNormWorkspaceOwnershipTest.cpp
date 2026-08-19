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

TEST(LayerNormWorkspaceOwnership, SharedCachedGraphsUseCallerOwnedIndependentScratchForConcurrentExecutions) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for LayerNorm workspace ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnLayerNorm& layerNorm = CudnnLayerNorm::instance();
    layerNorm.clearCache();

    const CudnnLayerNormDescriptor descriptor = makeTrainingDescriptor();
    const uint64_t forwardBytes = layerNorm.forwardWorkspaceSizeInBytes(descriptor, gpuNum);
    const uint64_t backwardBytes = layerNorm.backwardWorkspaceSizeInBytes(descriptor, gpuNum);
    ASSERT_EQ(layerNorm.cachedGraphCount(), 2U);

    // Re-querying an identical descriptor must reuse the same cached plans.
    EXPECT_EQ(layerNorm.forwardWorkspaceSizeInBytes(descriptor, gpuNum), forwardBytes);
    EXPECT_EQ(layerNorm.backwardWorkspaceSizeInBytes(descriptor, gpuNum), backwardBytes);
    EXPECT_EQ(layerNorm.cachedGraphCount(), 2U);

    optional<Tensor> forwardWorkspaceA = allocateWorkspace(placement, forwardBytes);
    optional<Tensor> forwardWorkspaceB = allocateWorkspace(placement, forwardBytes);
    optional<Tensor> backwardWorkspaceA = allocateWorkspace(placement, backwardBytes);
    optional<Tensor> backwardWorkspaceB = allocateWorkspace(placement, backwardBytes);

    if (forwardBytes > 0) {
        ASSERT_TRUE(forwardWorkspaceA.has_value());
        ASSERT_TRUE(forwardWorkspaceB.has_value());
        EXPECT_NE(forwardWorkspaceA->getMemPtr<void>(), forwardWorkspaceB->getMemPtr<void>());
    }
    if (backwardBytes > 0) {
        ASSERT_TRUE(backwardWorkspaceA.has_value());
        ASSERT_TRUE(backwardWorkspaceB.has_value());
        EXPECT_NE(backwardWorkspaceA->getMemPtr<void>(), backwardWorkspaceB->getMemPtr<void>());
    }

    LayerNormExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    LayerNormExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    initializeExecutionTensors(tensorsA, streamA);
    initializeExecutionTensors(tensorsB, streamB);

    CudnnLayerNormForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnLayerNormForwardArgs forwardB = forwardArgs(tensorsB);
    layerNorm.forward(descriptor, forwardA, forwardWorkspaceA, streamA);
    layerNorm.forward(descriptor, forwardB, forwardWorkspaceB, streamB);
    streamA.synchronize();
    streamB.synchronize();

    CudnnLayerNormBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnLayerNormBackwardArgs backwardB = backwardArgs(tensorsB);
    layerNorm.backward(descriptor, backwardA, backwardWorkspaceA, streamA);
    layerNorm.backward(descriptor, backwardB, backwardWorkspaceB, streamB);
    streamA.synchronize();
    streamB.synchronize();
}
