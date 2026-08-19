#include "Utilities/TensorOperations/DeepLearning/CudnnAdaptiveLayerNorm.h"

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

CudnnAdaptiveLayerNormDescriptor makeTrainingDescriptor() {
    CudnnAdaptiveLayerNormDescriptor descriptor;
    descriptor.batchSize = 2;
    descriptor.leadingFeatureCount = 4;
    // Keep this ownership regression on a cuDNN-primary-engine-friendly
    // training configuration. Small fp32 rows can be rejected when the engine's
    // vector load width exceeds the normalization width, while fp16/bf16 IO with
    // fp32 adaptive scale/bias is rejected by some primary AdaptiveLayerNorm
    // engines. Neither limitation is related to workspace ownership.
    descriptor.normalizedFeatureCount = 256;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.scaleBiasDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    descriptor.debugName = "adaptive_layer_norm_workspace_ownership";
    return descriptor;
}

struct AdaptiveLayerNormExecutionTensors {
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

AdaptiveLayerNormExecutionTensors makeExecutionTensors(const TensorPlacement& placement,
                                                        const CudnnAdaptiveLayerNormDescriptor& descriptor) {
    const uint64_t ioElements = descriptor.batchSize * descriptor.leadingFeatureCount * descriptor.normalizedFeatureCount;
    const uint64_t scaleBiasElements = descriptor.batchSize * descriptor.normalizedFeatureCount;
    const uint64_t statsElements = descriptor.batchSize * descriptor.leadingFeatureCount;
    return AdaptiveLayerNormExecutionTensors{
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.scaleBiasDataType, {scaleBiasElements})),
        Tensor(placement, TensorDescriptor(descriptor.scaleBiasDataType, {scaleBiasElements})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {statsElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {statsElements})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.scaleBiasDataType, {scaleBiasElements})),
        Tensor(placement, TensorDescriptor(descriptor.scaleBiasDataType, {scaleBiasElements})),
    };
}

void initializeExecutionTensors(AdaptiveLayerNormExecutionTensors& tensors, Stream stream) {
    tensors.x.fill(0.5, stream);
    tensors.scale.fill(1.0, stream);
    tensors.bias.fill(0.0, stream);
    tensors.dy.fill(1.0, stream);
    stream.synchronize();
}

CudnnAdaptiveLayerNormForwardArgs forwardArgs(AdaptiveLayerNormExecutionTensors& tensors) {
    CudnnAdaptiveLayerNormForwardArgs args;
    args.x = tensors.x;
    args.scale = tensors.scale;
    args.bias = tensors.bias;
    args.y = tensors.y;
    args.mean = tensors.mean;
    args.invVariance = tensors.invVariance;
    return args;
}

CudnnAdaptiveLayerNormBackwardArgs backwardArgs(AdaptiveLayerNormExecutionTensors& tensors) {
    CudnnAdaptiveLayerNormBackwardArgs args;
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

TEST(AdaptiveLayerNormWorkspaceOwnership, SharedCachedGraphsUseCallerOwnedIndependentScratchForConcurrentExecutions) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for AdaptiveLayerNorm workspace ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnAdaptiveLayerNorm& adaptiveLayerNorm = CudnnAdaptiveLayerNorm::instance();
    adaptiveLayerNorm.clearCache();

    const CudnnAdaptiveLayerNormDescriptor descriptor = makeTrainingDescriptor();
    const uint64_t forwardBytes = adaptiveLayerNorm.forwardWorkspaceSizeInBytes(descriptor, gpuNum);
    const uint64_t backwardBytes = adaptiveLayerNorm.backwardWorkspaceSizeInBytes(descriptor, gpuNum);
    ASSERT_EQ(adaptiveLayerNorm.cachedGraphCount(), 2U);

    // Identical descriptors share immutable cached plans, not mutable execution scratch.
    EXPECT_EQ(adaptiveLayerNorm.forwardWorkspaceSizeInBytes(descriptor, gpuNum), forwardBytes);
    EXPECT_EQ(adaptiveLayerNorm.backwardWorkspaceSizeInBytes(descriptor, gpuNum), backwardBytes);
    EXPECT_EQ(adaptiveLayerNorm.cachedGraphCount(), 2U);

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

    AdaptiveLayerNormExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    AdaptiveLayerNormExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    initializeExecutionTensors(tensorsA, streamA);
    initializeExecutionTensors(tensorsB, streamB);

    CudnnAdaptiveLayerNormForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnAdaptiveLayerNormForwardArgs forwardB = forwardArgs(tensorsB);
    adaptiveLayerNorm.forward(descriptor, forwardA, forwardWorkspaceA, streamA);
    adaptiveLayerNorm.forward(descriptor, forwardB, forwardWorkspaceB, streamB);
    streamA.synchronize();
    streamB.synchronize();

    CudnnAdaptiveLayerNormBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnAdaptiveLayerNormBackwardArgs backwardB = backwardArgs(tensorsB);
    adaptiveLayerNorm.backward(descriptor, backwardA, backwardWorkspaceA, streamA);
    adaptiveLayerNorm.backward(descriptor, backwardB, backwardWorkspaceB, streamB);
    streamA.synchronize();
    streamB.synchronize();
}
