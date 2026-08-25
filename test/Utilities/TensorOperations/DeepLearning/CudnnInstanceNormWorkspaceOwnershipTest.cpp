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

TEST(InstanceNormWorkspaceOwnership, EquivalentOperationsShareOnlySelectionAndSurviveSelectionCacheClear) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for InstanceNorm workspace ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnInstanceNorm& instanceNorm = CudnnInstanceNorm::instance();
    instanceNorm.clearSelectionCache();

    const uint64_t hitsBefore = instanceNorm.selectionCacheHitCount();
    const uint64_t missesBefore = instanceNorm.selectionCacheMissCount();
    const CudnnInstanceNormDescriptor descriptor = makeTrainingDescriptor();

    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    CudnnInstanceNormExecutablePlan forwardPlanA = instanceNorm.prepareForward(descriptor, streamA);
    CudnnInstanceNormExecutablePlan forwardPlanB = instanceNorm.prepareForward(descriptor, streamB);
    CudnnInstanceNormExecutablePlan backwardPlanA = instanceNorm.prepareBackward(descriptor, streamA);
    CudnnInstanceNormExecutablePlan backwardPlanB = instanceNorm.prepareBackward(descriptor, streamB);
    const uint64_t preparationsAfterStamping = cudnnFrontendExecutablePreparationCountForTests();

    ASSERT_EQ(instanceNorm.cachedSelectionCount(), 2U);
    EXPECT_EQ(instanceNorm.selectionCacheMissCount() - missesBefore, 2U);
    EXPECT_EQ(instanceNorm.selectionCacheHitCount() - hitsBefore, 2U);

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

    InstanceNormExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    InstanceNormExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    initializeExecutionTensors(tensorsA, streamA);
    initializeExecutionTensors(tensorsB, streamB);

    instanceNorm.clearSelectionCache();
    ASSERT_EQ(instanceNorm.cachedSelectionCount(), 0U);

    CudnnInstanceNormForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnInstanceNormForwardArgs forwardB = forwardArgs(tensorsB);
    for (int repetition = 0; repetition < 4; ++repetition) {
        instanceNorm.forward(forwardPlanA, forwardA, forwardWorkspaceA, streamA);
        instanceNorm.forward(forwardPlanB, forwardB, forwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();
    EXPECT_EQ(instanceNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping)
        << "InstanceNorm forward hot path must not replay/build/deserialize a cuDNN Frontend plan.";

    CudnnInstanceNormBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnInstanceNormBackwardArgs backwardB = backwardArgs(tensorsB);
    for (int repetition = 0; repetition < 4; ++repetition) {
        instanceNorm.backward(backwardPlanA, backwardA, backwardWorkspaceA, streamA);
        instanceNorm.backward(backwardPlanB, backwardB, backwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();
    EXPECT_EQ(instanceNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping)
        << "InstanceNorm backward hot path must not replay/build/deserialize a cuDNN Frontend plan.";
}
