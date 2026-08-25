#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <optional>

using namespace ThorImplementation;
using namespace std;

namespace {

int cudaDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess ? count : 0;
}

optional<Tensor> allocateWorkspace(const TensorPlacement& placement, uint64_t bytes) {
    if (bytes == 0) return nullopt;
    return Tensor(placement, TensorDescriptor(DataType::UINT8, {bytes}), 256);
}

CudnnRmsNormDescriptor makeTrainingDescriptor() {
    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = 64;
    descriptor.normalizedFeatureCount = 256;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    descriptor.debugName = "rms_norm_workspace_ownership";
    return descriptor;
}

struct ExecutionTensors {
    Tensor x;
    Tensor scale;
    Tensor y;
    Tensor invVariance;
    Tensor dy;
    Tensor dx;
    Tensor dscale;
};

ExecutionTensors makeExecutionTensors(const TensorPlacement& placement, const CudnnRmsNormDescriptor& descriptor) {
    const uint64_t ioElements = descriptor.outerSize * descriptor.normalizedFeatureCount;
    return ExecutionTensors{
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.normalizedFeatureCount})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {descriptor.outerSize})),
        Tensor(placement, TensorDescriptor(descriptor.outputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.inputDataType, {ioElements})),
        Tensor(placement, TensorDescriptor(descriptor.parameterDataType, {descriptor.normalizedFeatureCount})),
    };
}

void initialize(ExecutionTensors& tensors, Stream stream) {
    tensors.x.fill(0.5, stream);
    tensors.scale.fill(1.0, stream);
    tensors.dy.fill(1.0, stream);
    stream.synchronize();
}

CudnnRmsNormForwardArgs forwardArgs(ExecutionTensors& tensors) {
    CudnnRmsNormForwardArgs args;
    args.x = tensors.x;
    args.scale = tensors.scale;
    args.y = tensors.y;
    args.invVariance = tensors.invVariance;
    return args;
}

CudnnRmsNormBackwardArgs backwardArgs(ExecutionTensors& tensors) {
    CudnnRmsNormBackwardArgs args;
    args.dy = tensors.dy;
    args.x = tensors.x;
    args.scale = tensors.scale;
    args.invVariance = tensors.invVariance;
    args.dx = tensors.dx;
    args.dscale = tensors.dscale;
    return args;
}

}  // namespace

TEST(RmsNormWorkspaceOwnership, EquivalentOperationsShareOnlySelectionAndSurviveSelectionCacheClear) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required for RMSNorm ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnRmsNorm& rmsNorm = CudnnRmsNorm::instance();
    rmsNorm.clearSelectionCache();

    const uint64_t hitsBefore = rmsNorm.selectionCacheHitCount();
    const uint64_t missesBefore = rmsNorm.selectionCacheMissCount();
    const CudnnRmsNormDescriptor descriptor = makeTrainingDescriptor();

    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    CudnnRmsNormExecutablePlan forwardPlanA = rmsNorm.prepareForward(descriptor, streamA);
    CudnnRmsNormExecutablePlan forwardPlanB = rmsNorm.prepareForward(descriptor, streamB);
    CudnnRmsNormExecutablePlan backwardPlanA = rmsNorm.prepareBackward(descriptor, streamA);
    CudnnRmsNormExecutablePlan backwardPlanB = rmsNorm.prepareBackward(descriptor, streamB);
    const uint64_t preparationsAfterStamping = cudnnFrontendExecutablePreparationCountForTests();

    ASSERT_EQ(rmsNorm.cachedSelectionCount(), 2U);
    EXPECT_EQ(rmsNorm.selectionCacheMissCount() - missesBefore, 2U);
    EXPECT_EQ(rmsNorm.selectionCacheHitCount() - hitsBefore, 2U);
    EXPECT_EQ(forwardPlanA.selection(), forwardPlanB.selection());
    EXPECT_EQ(backwardPlanA.selection(), backwardPlanB.selection());
    EXPECT_NE(forwardPlanA.executableId(), forwardPlanB.executableId());
    EXPECT_NE(backwardPlanA.executableId(), backwardPlanB.executableId());

    optional<Tensor> forwardWorkspaceA = allocateWorkspace(placement, forwardPlanA.workspaceBytes());
    optional<Tensor> forwardWorkspaceB = allocateWorkspace(placement, forwardPlanB.workspaceBytes());
    optional<Tensor> backwardWorkspaceA = allocateWorkspace(placement, backwardPlanA.workspaceBytes());
    optional<Tensor> backwardWorkspaceB = allocateWorkspace(placement, backwardPlanB.workspaceBytes());
    if (forwardPlanA.workspaceBytes() > 0) {
        EXPECT_NE(forwardWorkspaceA->getMemPtr<void>(), forwardWorkspaceB->getMemPtr<void>());
    }
    if (backwardPlanA.workspaceBytes() > 0) {
        EXPECT_NE(backwardWorkspaceA->getMemPtr<void>(), backwardWorkspaceB->getMemPtr<void>());
    }

    ExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    ExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    initialize(tensorsA, streamA);
    initialize(tensorsB, streamB);

    rmsNorm.clearSelectionCache();
    ASSERT_EQ(rmsNorm.cachedSelectionCount(), 0U);

    CudnnRmsNormForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnRmsNormForwardArgs forwardB = forwardArgs(tensorsB);
    for (int repetition = 0; repetition < 4; ++repetition) {
        rmsNorm.forward(forwardPlanA, forwardA, forwardWorkspaceA, streamA);
        rmsNorm.forward(forwardPlanB, forwardB, forwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();
    EXPECT_EQ(rmsNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping)
        << "RMSNorm forward hot path must not replay/build/deserialize a cuDNN Frontend plan.";

    CudnnRmsNormBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnRmsNormBackwardArgs backwardB = backwardArgs(tensorsB);
    for (int repetition = 0; repetition < 4; ++repetition) {
        rmsNorm.backward(backwardPlanA, backwardA, backwardWorkspaceA, streamA);
        rmsNorm.backward(backwardPlanB, backwardB, backwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();
    EXPECT_EQ(rmsNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping)
        << "RMSNorm backward hot path must not replay/build/deserialize a cuDNN Frontend plan.";
}
