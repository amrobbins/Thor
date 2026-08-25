#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

constexpr int GPU_NUM = 0;
constexpr size_t BRANCH_COUNT = 4;
constexpr int STRESS_ITERATIONS = 20;

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

uint64_t elementCount(const Tensor& tensor) {
    uint64_t elements = 1;
    for (const uint64_t dim : tensor.getDimensions())
        elements *= dim;
    return elements;
}

vector<float> copyFp32ToHost(const Tensor& gpuTensor, Stream& stream) {
    Tensor cpu = gpuTensor.clone(TensorPlacement(TensorPlacement::MemDevices::CPU));
    cpu.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    const float* values = cpu.getMemPtr<float>();
    return vector<float>(values, values + elementCount(cpu));
}

vector<float> copyFp16ToHost(const Tensor& gpuTensor, Stream& stream) {
    Tensor cpu = gpuTensor.clone(TensorPlacement(TensorPlacement::MemDevices::CPU));
    cpu.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    const half* values = cpu.getMemPtr<half>();
    vector<float> converted(elementCount(cpu));
    for (uint64_t i = 0; i < converted.size(); ++i)
        converted[i] = __half2float(values[i]);
    return converted;
}

CudnnRmsNormDescriptor makeRmsNormDescriptor() {
    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = 64;
    descriptor.normalizedFeatureCount = 256;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    descriptor.debugName = "workspace_concurrency_stress_rmsnorm";
    return descriptor;
}

struct RmsNormBranch {
    Tensor x;
    Tensor scale;
    Tensor y;
    Tensor invVariance;
    Tensor dy;
    Tensor dx;
    Tensor dscale;
    optional<Tensor> forwardWorkspace;
    optional<Tensor> backwardWorkspace;
};

RmsNormBranch makeRmsNormBranch(const TensorPlacement& placement,
                                const CudnnRmsNormDescriptor& descriptor,
                                uint64_t forwardBytes,
                                uint64_t backwardBytes) {
    const uint64_t ioElements = descriptor.outerSize * descriptor.normalizedFeatureCount;
    return RmsNormBranch{
        Tensor(placement, TensorDescriptor(DataType::FP32, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {descriptor.normalizedFeatureCount})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {descriptor.outerSize})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {ioElements})),
        Tensor(placement, TensorDescriptor(DataType::FP32, {descriptor.normalizedFeatureCount})),
        allocateWorkspace(placement, forwardBytes),
        allocateWorkspace(placement, backwardBytes),
    };
}

CudnnRmsNormForwardArgs rmsForwardArgs(RmsNormBranch& branch) {
    CudnnRmsNormForwardArgs args;
    args.x = branch.x;
    args.scale = branch.scale;
    args.y = branch.y;
    args.invVariance = branch.invVariance;
    return args;
}

CudnnRmsNormBackwardArgs rmsBackwardArgs(RmsNormBranch& branch) {
    CudnnRmsNormBackwardArgs args;
    args.dy = branch.dy;
    args.x = branch.x;
    args.scale = branch.scale;
    args.invVariance = branch.invVariance;
    args.dx = branch.dx;
    args.dscale = branch.dscale;
    return args;
}

CudnnAttentionDescriptor makeAttentionDescriptor() {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.k = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.v = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.o = AttentionTensorSpec::bhsd(2, 4, 64, 64, DataType::FP16);
    descriptor.computeDataType = DataType::FP32;
    descriptor.intermediateDataType = DataType::FP32;
    descriptor.generateStats = true;
    descriptor.debugName = "workspace_concurrency_stress_attention";
    return descriptor;
}

vector<uint64_t> dimsFor(const AttentionTensorSpec& spec) {
    vector<uint64_t> dims;
    dims.reserve(spec.dimensions.size());
    for (const int64_t dim : spec.dimensions)
        dims.push_back(static_cast<uint64_t>(dim));
    return dims;
}

struct AttentionBranch {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor o;
    Tensor stats;
    Tensor dO;
    Tensor dQ;
    Tensor dK;
    Tensor dV;
    optional<Tensor> forwardWorkspace;
    optional<Tensor> backwardWorkspace;
};

AttentionBranch makeAttentionBranch(const TensorPlacement& placement,
                                    const CudnnAttentionDescriptor& descriptor,
                                    uint64_t forwardBytes,
                                    uint64_t backwardBytes) {
    const vector<uint64_t> statsDims{static_cast<uint64_t>(descriptor.batchSize()),
                                     static_cast<uint64_t>(descriptor.queryHeads()),
                                     static_cast<uint64_t>(descriptor.queryLength()),
                                     1};
    return AttentionBranch{
        Tensor(placement, TensorDescriptor(descriptor.q.dataType, dimsFor(descriptor.q))),
        Tensor(placement, TensorDescriptor(descriptor.k.dataType, dimsFor(descriptor.k))),
        Tensor(placement, TensorDescriptor(descriptor.v.dataType, dimsFor(descriptor.v))),
        Tensor(placement, TensorDescriptor(descriptor.o.dataType, dimsFor(descriptor.o))),
        Tensor(placement, TensorDescriptor(DataType::FP32, statsDims)),
        Tensor(placement, TensorDescriptor(descriptor.o.dataType, dimsFor(descriptor.o))),
        Tensor(placement, TensorDescriptor(descriptor.q.dataType, dimsFor(descriptor.q))),
        Tensor(placement, TensorDescriptor(descriptor.k.dataType, dimsFor(descriptor.k))),
        Tensor(placement, TensorDescriptor(descriptor.v.dataType, dimsFor(descriptor.v))),
        allocateWorkspace(placement, forwardBytes),
        allocateWorkspace(placement, backwardBytes),
    };
}

CudnnAttentionForwardArgs attentionForwardArgs(AttentionBranch& branch) {
    CudnnAttentionForwardArgs args{.q = branch.q, .k = branch.k, .v = branch.v, .o = branch.o};
    args.stats = branch.stats;
    return args;
}

CudnnAttentionBackwardArgs attentionBackwardArgs(AttentionBranch& branch) {
    return CudnnAttentionBackwardArgs{.q = branch.q,
                                      .k = branch.k,
                                      .v = branch.v,
                                      .o = branch.o,
                                      .dO = branch.dO,
                                      .stats = branch.stats,
                                      .dQ = branch.dQ,
                                      .dK = branch.dK,
                                      .dV = branch.dV};
}

template <typename Branch>
void expectDistinctWorkspacePointers(const array<Branch, BRANCH_COUNT>& branches, bool forward) {
    vector<const void*> pointers;
    for (const Branch& branch : branches) {
        const optional<Tensor>& workspace = forward ? branch.forwardWorkspace : branch.backwardWorkspace;
        if (workspace.has_value())
            pointers.push_back(workspace->getMemPtr<void>());
    }
    for (size_t i = 0; i < pointers.size(); ++i) {
        for (size_t j = i + 1; j < pointers.size(); ++j)
            EXPECT_NE(pointers[i], pointers[j]);
    }
}

}  // namespace

TEST(CudnnWorkspaceConcurrencyStress, LocalRmsNormAndAttentionPlansUseIndependentScratchAcrossBranches) {
    if (cudaDeviceCount() < 1)
        GTEST_SKIP() << "CUDA device is required for cuDNN workspace concurrency stress.";

    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, GPU_NUM);

    CudnnRmsNorm& rmsNorm = CudnnRmsNorm::instance();
    CudnnScaledDotProductAttention& attention = CudnnScaledDotProductAttention::instance();
    rmsNorm.clearSelectionCache();
    attention.clearSelectionCache();

    const CudnnRmsNormDescriptor rmsDescriptor = makeRmsNormDescriptor();
    Stream rmsPreparationStream(GPU_NUM);
    array<optional<CudnnRmsNormExecutablePlan>, BRANCH_COUNT> rmsForwardPlans;
    array<optional<CudnnRmsNormExecutablePlan>, BRANCH_COUNT> rmsBackwardPlans;
    for (size_t i = 0; i < BRANCH_COUNT; ++i) {
        rmsForwardPlans[i].emplace(rmsNorm.prepareForward(rmsDescriptor, rmsPreparationStream));
        rmsBackwardPlans[i].emplace(rmsNorm.prepareBackward(rmsDescriptor, rmsPreparationStream));
    }
    const uint64_t rmsForwardBytes = rmsForwardPlans[0]->workspaceBytes();
    const uint64_t rmsBackwardBytes = rmsBackwardPlans[0]->workspaceBytes();
    ASSERT_EQ(rmsNorm.cachedSelectionCount(), 2U);

    const CudnnAttentionDescriptor attentionDescriptor = makeAttentionDescriptor();
    const uint64_t attentionForwardBytes = attention.forwardWorkspaceSizeInBytes(attentionDescriptor, GPU_NUM);
    const uint64_t attentionBackwardBytes = attention.backwardWorkspaceSizeInBytes(attentionDescriptor, GPU_NUM);
    ASSERT_EQ(attention.cachedSelectionCount(), 2U);

    array<RmsNormBranch, BRANCH_COUNT> rmsBranches{
        makeRmsNormBranch(placement, rmsDescriptor, rmsForwardBytes, rmsBackwardBytes),
        makeRmsNormBranch(placement, rmsDescriptor, rmsForwardBytes, rmsBackwardBytes),
        makeRmsNormBranch(placement, rmsDescriptor, rmsForwardBytes, rmsBackwardBytes),
        makeRmsNormBranch(placement, rmsDescriptor, rmsForwardBytes, rmsBackwardBytes),
    };
    array<AttentionBranch, BRANCH_COUNT> attentionBranches{
        makeAttentionBranch(placement, attentionDescriptor, attentionForwardBytes, attentionBackwardBytes),
        makeAttentionBranch(placement, attentionDescriptor, attentionForwardBytes, attentionBackwardBytes),
        makeAttentionBranch(placement, attentionDescriptor, attentionForwardBytes, attentionBackwardBytes),
        makeAttentionBranch(placement, attentionDescriptor, attentionForwardBytes, attentionBackwardBytes),
    };
    array<optional<CudnnAttentionExecutablePlan>, BRANCH_COUNT> attentionForwardPlans;
    array<optional<CudnnAttentionExecutablePlan>, BRANCH_COUNT> attentionBackwardPlans;
    Stream attentionPreparationStream(GPU_NUM);
    for (size_t i = 0; i < BRANCH_COUNT; ++i) {
        CudnnAttentionForwardArgs forwardArgs = attentionForwardArgs(attentionBranches[i]);
        CudnnAttentionBackwardArgs backwardArgs = attentionBackwardArgs(attentionBranches[i]);
        attentionForwardPlans[i].emplace(attention.prepareForward(attentionDescriptor, forwardArgs, attentionPreparationStream));
        attentionBackwardPlans[i].emplace(attention.prepareBackward(attentionDescriptor, backwardArgs, attentionPreparationStream));
    }
    const uint64_t preparationsAfterStamping = cudnnFrontendExecutablePreparationCountForTests();

    for (size_t i = 0; i < BRANCH_COUNT; ++i) {
        EXPECT_EQ(rmsForwardPlans[i]->selection(), rmsForwardPlans[0]->selection());
        EXPECT_EQ(rmsBackwardPlans[i]->selection(), rmsBackwardPlans[0]->selection());
        EXPECT_EQ(attentionForwardPlans[i]->selection(), attentionForwardPlans[0]->selection());
        EXPECT_EQ(attentionBackwardPlans[i]->selection(), attentionBackwardPlans[0]->selection());
        for (size_t j = i + 1; j < BRANCH_COUNT; ++j) {
            EXPECT_NE(rmsForwardPlans[i]->executableId(), rmsForwardPlans[j]->executableId());
            EXPECT_NE(rmsBackwardPlans[i]->executableId(), rmsBackwardPlans[j]->executableId());
            EXPECT_NE(attentionForwardPlans[i]->executableId(), attentionForwardPlans[j]->executableId());
            EXPECT_NE(attentionBackwardPlans[i]->executableId(), attentionBackwardPlans[j]->executableId());
        }
    }

    expectDistinctWorkspacePointers(rmsBranches, true);
    expectDistinctWorkspacePointers(rmsBranches, false);
    expectDistinctWorkspacePointers(attentionBranches, true);
    expectDistinctWorkspacePointers(attentionBranches, false);

    // C13's runtime gate deliberately removes the global recipes before any
    // concurrent launch. Every branch must remain fully executable using only
    // its independently prepared graph and workspace.
    rmsNorm.clearSelectionCache();
    attention.clearSelectionCache();
    ASSERT_EQ(rmsNorm.cachedSelectionCount(), 0U);
    ASSERT_EQ(attention.cachedSelectionCount(), 0U);

    array<unique_ptr<Stream>, BRANCH_COUNT> streams;
    for (size_t i = 0; i < BRANCH_COUNT; ++i) {
        streams[i] = make_unique<Stream>(GPU_NUM);
        rmsBranches[i].x.fill(0.5, *streams[i]);
        rmsBranches[i].scale.fill(1.0, *streams[i]);
        rmsBranches[i].dy.fill(1.0, *streams[i]);

        attentionBranches[i].q.fill(0.125, *streams[i]);
        attentionBranches[i].k.fill(0.25, *streams[i]);
        attentionBranches[i].v.fill(0.5, *streams[i]);
        attentionBranches[i].dO.fill(1.0, *streams[i]);
    }

    for (size_t i = 0; i < BRANCH_COUNT; ++i)
        streams[i]->synchronize();

    for (int iteration = 0; iteration < STRESS_ITERATIONS; ++iteration) {
        for (size_t i = 0; i < BRANCH_COUNT; ++i) {
            CudnnRmsNormForwardArgs rmsForward = rmsForwardArgs(rmsBranches[i]);
            rmsNorm.forward(rmsForwardPlans[i].value(), rmsForward, rmsBranches[i].forwardWorkspace, *streams[i]);

            CudnnAttentionForwardArgs attentionForward = attentionForwardArgs(attentionBranches[i]);
            attention.forward(attentionForwardPlans[i].value(), attentionForward, attentionBranches[i].forwardWorkspace, *streams[i]);

            CudnnRmsNormBackwardArgs rmsBackward = rmsBackwardArgs(rmsBranches[i]);
            rmsNorm.backward(rmsBackwardPlans[i].value(), rmsBackward, rmsBranches[i].backwardWorkspace, *streams[i]);

            CudnnAttentionBackwardArgs attentionBackward = attentionBackwardArgs(attentionBranches[i]);
            attention.backward(attentionBackwardPlans[i].value(), attentionBackward, attentionBranches[i].backwardWorkspace, *streams[i]);
        }
    }

    for (size_t i = 0; i < BRANCH_COUNT; ++i)
        streams[i]->synchronize();

    // Repeated execution uses only branch-local executable plans; neither
    // preparation nor global selection state changes in the hot path.
    EXPECT_EQ(rmsNorm.cachedSelectionCount(), 0U);
    EXPECT_EQ(attention.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparationsAfterStamping);

    const float expectedRms = 0.5f / sqrtf(0.25f + rmsDescriptor.epsilon);
    for (size_t i = 0; i < BRANCH_COUNT; ++i) {
        const vector<float> rmsValues = copyFp32ToHost(rmsBranches[i].y, *streams[i]);
        for (const float value : rmsValues)
            EXPECT_NEAR(value, expectedRms, 2.0e-4f);

        const vector<float> attentionValues = copyFp16ToHost(attentionBranches[i].o, *streams[i]);
        for (const float value : attentionValues)
            EXPECT_NEAR(value, 0.5f, 2.0e-3f);

        const vector<float> attentionDvValues = copyFp16ToHost(attentionBranches[i].dV, *streams[i]);
        for (const float value : attentionDvValues)
            EXPECT_TRUE(isfinite(value));
    }
}
