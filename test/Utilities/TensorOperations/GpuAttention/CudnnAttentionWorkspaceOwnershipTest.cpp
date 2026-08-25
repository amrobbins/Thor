#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <optional>
#include <vector>

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

CudnnAttentionDescriptor makeTrainingDescriptor() {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.k = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.v = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.o = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.computeDataType = DataType::FP32;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.generateStats = true;
    descriptor.debugName = "attention_workspace_ownership";
    return descriptor;
}

vector<uint64_t> dimsFor(const AttentionTensorSpec& spec) {
    vector<uint64_t> dims;
    dims.reserve(spec.dimensions.size());
    for (const int64_t dim : spec.dimensions) {
        dims.push_back(static_cast<uint64_t>(dim));
    }
    return dims;
}

struct AttentionExecutionTensors {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor o;
    Tensor stats;
    Tensor dO;
    Tensor dQ;
    Tensor dK;
    Tensor dV;
};

AttentionExecutionTensors makeExecutionTensors(const TensorPlacement& placement, const CudnnAttentionDescriptor& descriptor) {
    const vector<uint64_t> statsDims{static_cast<uint64_t>(descriptor.batchSize()),
                                     static_cast<uint64_t>(descriptor.queryHeads()),
                                     static_cast<uint64_t>(descriptor.queryLength()),
                                     1};
    return AttentionExecutionTensors{
        Tensor(placement, TensorDescriptor(descriptor.q.dataType, dimsFor(descriptor.q))),
        Tensor(placement, TensorDescriptor(descriptor.k.dataType, dimsFor(descriptor.k))),
        Tensor(placement, TensorDescriptor(descriptor.v.dataType, dimsFor(descriptor.v))),
        Tensor(placement, TensorDescriptor(descriptor.o.dataType, dimsFor(descriptor.o))),
        Tensor(placement, TensorDescriptor(DataType::FP32, statsDims)),
        Tensor(placement, TensorDescriptor(descriptor.o.dataType, dimsFor(descriptor.o))),
        Tensor(placement, TensorDescriptor(descriptor.q.dataType, dimsFor(descriptor.q))),
        Tensor(placement, TensorDescriptor(descriptor.k.dataType, dimsFor(descriptor.k))),
        Tensor(placement, TensorDescriptor(descriptor.v.dataType, dimsFor(descriptor.v))),
    };
}

void initializeExecutionTensors(AttentionExecutionTensors& tensors, Stream stream) {
    tensors.q.fill(0.125, stream);
    tensors.k.fill(0.25, stream);
    tensors.v.fill(0.5, stream);
    tensors.dO.fill(1.0, stream);
    stream.synchronize();
}

CudnnAttentionForwardArgs forwardArgs(AttentionExecutionTensors& tensors) {
    CudnnAttentionForwardArgs args{.q = tensors.q, .k = tensors.k, .v = tensors.v, .o = tensors.o};
    args.stats = tensors.stats;
    return args;
}

CudnnAttentionBackwardArgs backwardArgs(AttentionExecutionTensors& tensors) {
    return CudnnAttentionBackwardArgs{.q = tensors.q,
                                      .k = tensors.k,
                                      .v = tensors.v,
                                      .o = tensors.o,
                                      .dO = tensors.dO,
                                      .stats = tensors.stats,
                                      .dQ = tensors.dQ,
                                      .dK = tensors.dK,
                                      .dV = tensors.dV};
}

}  // namespace

TEST(AttentionWorkspaceOwnership, EquivalentOperationsShareOnlySelectionAndSurviveSelectionCacheClear) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for Attention workspace ownership tests.";

    constexpr int gpuNum = 0;
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    CudnnScaledDotProductAttention& attention = CudnnScaledDotProductAttention::instance();
    attention.clearSelectionCache();

    const CudnnAttentionDescriptor descriptor = makeTrainingDescriptor();
    AttentionExecutionTensors tensorsA = makeExecutionTensors(placement, descriptor);
    AttentionExecutionTensors tensorsB = makeExecutionTensors(placement, descriptor);
    Stream streamA(gpuNum);
    Stream streamB(gpuNum);
    initializeExecutionTensors(tensorsA, streamA);
    initializeExecutionTensors(tensorsB, streamB);

    CudnnAttentionForwardArgs forwardA = forwardArgs(tensorsA);
    CudnnAttentionForwardArgs forwardB = forwardArgs(tensorsB);
    CudnnAttentionBackwardArgs backwardA = backwardArgs(tensorsA);
    CudnnAttentionBackwardArgs backwardB = backwardArgs(tensorsB);

    CudnnAttentionExecutablePlan forwardPlanA = attention.prepareForward(descriptor, forwardA, streamA);
    CudnnAttentionExecutablePlan forwardPlanB = attention.prepareForward(descriptor, forwardB, streamB);
    CudnnAttentionExecutablePlan backwardPlanA = attention.prepareBackward(descriptor, backwardA, streamA);
    CudnnAttentionExecutablePlan backwardPlanB = attention.prepareBackward(descriptor, backwardB, streamB);

    ASSERT_EQ(attention.cachedSelectionCount(), 2U);
    EXPECT_EQ(forwardPlanA.selection(), forwardPlanB.selection());
    EXPECT_EQ(backwardPlanA.selection(), backwardPlanB.selection());
    EXPECT_EQ(forwardPlanA.selection().usesSerializedReplay(), true);
    EXPECT_EQ(backwardPlanA.selection().usesSerializedReplay(), true);
    EXPECT_EQ(forwardPlanA.selection().engine_id, -1);
    EXPECT_EQ(backwardPlanA.selection().engine_id, -1);
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

    const uint64_t preparationsAfterStamping = cudnnFrontendExecutablePreparationCountForTests();
    attention.clearSelectionCache();
    ASSERT_EQ(attention.cachedSelectionCount(), 0U);

    for (int repetition = 0; repetition < 4; ++repetition) {
        attention.forward(forwardPlanA, forwardA, forwardWorkspaceA, streamA);
        attention.forward(forwardPlanB, forwardB, forwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();

    for (int repetition = 0; repetition < 4; ++repetition) {
        attention.backward(backwardPlanA, backwardA, backwardWorkspaceA, streamA);
        attention.backward(backwardPlanB, backwardB, backwardWorkspaceB, streamB);
    }
    streamA.synchronize();
    streamB.synchronize();

    EXPECT_EQ(attention.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping);
}
