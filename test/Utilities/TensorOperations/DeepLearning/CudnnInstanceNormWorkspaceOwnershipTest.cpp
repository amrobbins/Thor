#include "Utilities/TensorOperations/DeepLearning/CudnnInstanceNorm.h"

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

CudnnInstanceNormDescriptor makeTrainingDescriptor() {
    CudnnInstanceNormDescriptor descriptor;
    descriptor.batchSize = 8;
    descriptor.channelCount = 16;
    descriptor.spatialElementCount = 32;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    descriptor.debugName = "instance_norm_workspace_ownership";
    return descriptor;
}

struct InstanceNormExecutionTensors {
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

InstanceNormExecutionTensors makeExecutionTensors(const TensorPlacement& placement, const CudnnInstanceNormDescriptor& descriptor) {
    const uint64_t ioElements = descriptor.batchSize * descriptor.channelCount * descriptor.spatialElementCount;
    const uint64_t statsElements = descriptor.batchSize * descriptor.channelCount;
    return InstanceNormExecutionTensors{
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.channelCount})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.channelCount})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {statsElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {statsElements})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.channelCount})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.channelCount})),
    };
}

void initializeExecutionTensors(InstanceNormExecutionTensors& tensors, Stream stream) {
    tensors.x.fill(0.5, stream);
    tensors.scale.fill(1.0, stream);
    tensors.bias.fill(0.0, stream);
    tensors.dy.fill(1.0, stream);
    stream.synchronize();
}

CudnnInstanceNormForwardArgs forwardArgs(InstanceNormExecutionTensors& tensors) {
    CudnnInstanceNormForwardArgs args;
    args.x = tensors.x;
    args.scale = tensors.scale;
    args.bias = tensors.bias;
    args.y = tensors.y;
    args.mean = tensors.mean;
    args.invVariance = tensors.invVariance;
    return args;
}

CudnnInstanceNormBackwardArgs backwardArgs(InstanceNormExecutionTensors& tensors) {
    CudnnInstanceNormBackwardArgs args;
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

TEST(InstanceNormWorkspaceOwnership, SharedCachedGraphsUseCallerOwnedIndependentScratchForConcurrentExecutions) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for InstanceNorm workspace ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnInstanceNorm& instanceNorm = CudnnInstanceNorm::instance();
    instanceNorm.clearCache();

    const CudnnInstanceNormDescriptor descriptor = makeTrainingDescriptor();
    const uint64_t forwardBytes = instanceNorm.forwardWorkspaceSizeInBytes(descriptor, gpuNum);
    const uint64_t backwardBytes = instanceNorm.backwardWorkspaceSizeInBytes(descriptor, gpuNum);
    ASSERT_EQ(instanceNorm.cachedGraphCount(), 2U);

    // Re-querying an identical descriptor must reuse the same cached plans.
    EXPECT_EQ(instanceNorm.forwardWorkspaceSizeInBytes(descriptor, gpuNum), forwardBytes);
    EXPECT_EQ(instanceNorm.backwardWorkspaceSizeInBytes(descriptor, gpuNum), backwardBytes);
    EXPECT_EQ(instanceNorm.cachedGraphCount(), 2U);

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

    InstanceNormExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    InstanceNormExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    initializeExecutionTensors(tensorsA, streamA);
    initializeExecutionTensors(tensorsB, streamB);

    CudnnInstanceNormForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnInstanceNormForwardArgs forwardB = forwardArgs(tensorsB);
    instanceNorm.forward(descriptor, forwardA, forwardWorkspaceA, streamA);
    instanceNorm.forward(descriptor, forwardB, forwardWorkspaceB, streamB);
    streamA.synchronize();
    streamB.synchronize();

    CudnnInstanceNormBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnInstanceNormBackwardArgs backwardB = backwardArgs(tensorsB);
    instanceNorm.backward(descriptor, backwardA, backwardWorkspaceA, streamA);
    instanceNorm.backward(descriptor, backwardB, backwardWorkspaceB, streamB);
    streamA.synchronize();
    streamB.synchronize();
}
