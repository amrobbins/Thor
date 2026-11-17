#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;

// Test-only inspection: assert actual stamp-owned scratch allocations, not just
// the public byte count. Intermediates are separate and are not scratch.
namespace ThorImplementation {
struct ReducerWorkspaceProbe {
    static size_t scratchBytes(const StampedReducer& stamp) {
        if (stamp.dense_executable_plan.passes.empty()) {
            return stamp.temp_storage.isInitialized() ? stamp.temp_storage.getArraySizeInBytes() : 0;
        }
        size_t bytes = 0;
        for (const auto& pass : stamp.dense_executable_plan.passes) {
            const size_t allocated = pass.temp_storage.isInitialized()
                                         ? pass.temp_storage.getArraySizeInBytes() : 0;
            EXPECT_EQ(allocated, pass.temp_storage_bytes);
            bytes += allocated;
        }
        return bytes;
    }

    static size_t scratchBytes(const StampedArgReducer& stamp) {
        if (stamp.composed_stages.empty()) {
            return stamp.temp_storage.isInitialized() ? stamp.temp_storage.getArraySizeInBytes() : 0;
        }
        size_t bytes = 0;
        for (const auto& stage : stamp.composed_stages) bytes += scratchBytes(*stage);
        return bytes;
    }
};
}  // namespace ThorImplementation

// The dense BF16 cases check that whatever scratch their passes own is stamped ahead of run and reused.
// The reducer refuses a rotated view; the dense stamp of its bytes, the caller's first part, stamps its scratch ahead
// of run like any dense stamp.
TEST(Reducer, ViewAndDenseScratchIsStampedAheadOfRun) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f,
                                  2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}, {2, 3, 2}, stream);

    // A permuted view with a rotated output: refused; its first part is the dense stamp of its bytes.
    StridedSpan rotated_view = StridedSpan(input, 0, {2, 2, 3}, {1, 6, 2});
    EXPECT_THROW(static_cast<void>(Reducer(ReductionOp::Sum, 2, DataType::FP32).stamp(rotated_view, stream)),
                 NotImplementedException);
    const std::optional<ViewSplit> split = Reducer::splitRefusedView(
        rotated_view.getDimensions(), rotated_view.getStridesElements(), {2});
    ASSERT_TRUE(split.has_value());
    const auto tiled_stamp = Reducer(ReductionOp::Sum, split->dense_axes, DataType::FP32)
                                 .stamp(rotated_view.asDense(split->dense_dimensions), stream);
    EXPECT_EQ(ReducerWorkspaceProbe::scratchBytes(*tiled_stamp), tiled_stamp->getWorkspaceSizeInBytes());

    // Dense BF16 VALUE and ARG (FP16 / BF16 passes): the query, the stamp and the stamped scratch agree.
    Tensor bf16_input = makeGpuTensor({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f,
                                       2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}, {2, 3, 2}, stream, DataType::BF16);
    ArgReducer dense_arg(ArgReductionOp::ArgMax, 1);
    const size_t dense_arg_bytes = dense_arg.queryWorkspaceSizeInBytes(bf16_input.getDescriptor(), stream);
    const auto dense_arg_stamp = dense_arg.stamp(bf16_input, stream);
    EXPECT_EQ(dense_arg_stamp->getWorkspaceSizeInBytes(), dense_arg_bytes);
    ASSERT_FALSE(dense_arg_stamp->getPlannedPassNames().empty());
    EXPECT_EQ(dense_arg_stamp->getPlannedPassNames().front().rfind("f16_arg_", 0), 0U);

    Tensor vector_input = makeGpuTensor({1.0f, 2.0f, 3.0f}, {3}, stream, DataType::BF16);
    Reducer dense_value(ReductionOp::Sum, 0, DataType::FP32);
    const size_t dense_value_bytes = dense_value.queryWorkspaceSizeInBytes(vector_input.getDescriptor(), stream);
    const auto dense_value_stamp = dense_value.stamp(vector_input, stream);
    EXPECT_EQ(dense_value_stamp->getWorkspaceSizeInBytes(), dense_value_bytes);
    EXPECT_EQ(ReducerWorkspaceProbe::scratchBytes(*dense_value_stamp), dense_value_bytes);

    ArgReducer vector_arg(ArgReductionOp::ArgMax, 0);
    const size_t vector_arg_bytes = vector_arg.queryWorkspaceSizeInBytes(vector_input.getDescriptor(), stream);
    const auto vector_arg_stamp = vector_arg.stamp(vector_input, stream);
    EXPECT_EQ(vector_arg_stamp->getWorkspaceSizeInBytes(), vector_arg_bytes);

    tiled_stamp->run();
    dense_arg_stamp->run();
    dense_value_stamp->run();
    vector_arg_stamp->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(tiled_stamp->getOutputTensor(), stream),
                          {15.0f, 21.0f, 18.0f, 24.0f});  // the dense [2, 1, 2], before the rotation
    expectFloatVectorNear(executeRefusedViewInTwoParts(rotated_view, ReductionOp::Sum, {2}, stream),
                          {15.0f, 18.0f, 21.0f, 24.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(dense_value_stamp->getOutputTensor(), stream), {6.0f});
    ASSERT_TRUE(dense_arg_stamp->getIndexOutputTensor().has_value());
    EXPECT_EQ(copyGpuTensorAsUnsigned(dense_arg_stamp->getIndexOutputTensor().value(), stream),
              (std::vector<uint64_t>{2, 2, 2, 2}));
    ASSERT_TRUE(vector_arg_stamp->getIndexOutputTensor().has_value());
    EXPECT_EQ(copyGpuTensorAsUnsigned(vector_arg_stamp->getIndexOutputTensor().value(), stream),
              (std::vector<uint64_t>{2}));

    // run() must reuse the existing scratch, not allocate it on demand.
    dense_value_stamp->run();
    vector_arg_stamp->run();
    stream.synchronize();
    EXPECT_EQ(ReducerWorkspaceProbe::scratchBytes(*dense_value_stamp), dense_value_bytes);
    EXPECT_EQ(dense_value_stamp->getWorkspaceSizeInBytes(), dense_value_bytes);
    EXPECT_EQ(vector_arg_stamp->getWorkspaceSizeInBytes(), vector_arg_bytes);
}


TEST(Reducer, MultiAxisContiguousSuffixUsesFixedSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 2, 3},
                                        stream);

    std::shared_ptr<StampedReducer> stamped =
        Reducer(ReductionOp::Sum, std::vector<uint32_t>{1, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), ReductionLayout::TrailingBlock);
    EXPECT_EQ(stamped->getGeometry().reduction_size, 6U);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 1, 1}));

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {21.0f, 57.0f});
}

TEST(Reducer, DenseRkrSumUsesComposedReductionAndLeadingAxesUseTiledReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);

    std::shared_ptr<StampedReducer> disjoint =
        Reducer(ReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(disjoint->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(disjoint->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1}));
    disjoint->run();

    std::shared_ptr<StampedReducer> leading =
        Reducer(ReductionOp::Sum, std::vector<uint32_t>{0, 1}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(leading->getPath(), ReductionLayout::MiddleBlock);
    EXPECT_EQ(leading->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 1, 2}));
    leading->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(disjoint->getOutputTensor(), stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(leading->getOutputTensor(), stream), {36.0f, 42.0f});
}

TEST(Reducer, DenseRkrSumCompositionPreservesRuntimeScaleAndWorkspaceQuery) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                 {2, 3, 2},
                                 stream);

    Reducer reduction(ReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32, 3.0f);
    const size_t queried_workspace = reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream);
    auto stamped = reduction.stamp(input, stream);
    ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), queried_workspace);
    EXPECT_GE(stamped->getWorkspaceSizeInBytes(), 2U * 3U * sizeof(float));

    // run(scale) overrides the configured scale exactly as it does on single-pass reductions.
    stamped->run(2.0f);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {36.0f, 52.0f, 68.0f});
}

TEST(Reducer, DenseRkrkrSumUsesIntervalPlannedComposedReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(32);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    Tensor input = makeGpuTensor(values, {2, 2, 2, 2, 2}, stream);

    Reducer reduction(ReductionOp::Sum, std::vector<uint32_t>{0, 2, 4}, DataType::FP32, 3.0f);
    const size_t queried_workspace = reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream);
    auto stamped = reduction.stamp(input, stream);
    ASSERT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 2, 1, 2, 1}));
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), queried_workspace);
    EXPECT_GT(stamped->getWorkspaceSizeInBytes(), 0U);

    // run(scale) overrides the configured scale and must apply it only after all FP32 partial reductions complete.
    stamped->run(2.0f);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {184.0f, 216.0f, 312.0f, 344.0f});
}

TEST(Reducer, DenseTrailingRetainedSumUsesIntervalPlannedComposedDenseReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<float> rkrk_values(24);
    for (size_t i = 0; i < rkrk_values.size(); ++i) {
        rkrk_values[i] = static_cast<float>(i + 1);
    }
    Tensor rkrk_input = makeGpuTensor(rkrk_values, {2, 3, 2, 2}, stream);
    Reducer rkrk_reduction(ReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32, 3.0f);
    const size_t rkrk_queried_workspace = rkrk_reduction.queryWorkspaceSizeInBytes(rkrk_input.getDescriptor(), stream);
    auto rkrk = rkrk_reduction.stamp(rkrk_input, stream);
    ASSERT_EQ(rkrk->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(rkrk->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1, 2}));
    EXPECT_EQ(rkrk->getWorkspaceSizeInBytes(), rkrk_queried_workspace);
    EXPECT_GT(rkrk->getWorkspaceSizeInBytes(), 0U);
    rkrk->run(2.0f);

    std::vector<float> krkrk_values(32);
    for (size_t i = 0; i < krkrk_values.size(); ++i) {
        krkrk_values[i] = static_cast<float>(i + 1);
    }
    Tensor krkrk_input = makeGpuTensor(krkrk_values, {2, 2, 2, 2, 2}, stream);
    Reducer krkrk_reduction(ReductionOp::Sum, std::vector<uint32_t>{1, 3}, DataType::FP32, 3.0f);
    const size_t krkrk_queried_workspace =
        krkrk_reduction.queryWorkspaceSizeInBytes(krkrk_input.getDescriptor(), stream);
    auto krkrk = krkrk_reduction.stamp(krkrk_input, stream);
    ASSERT_EQ(krkrk->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(krkrk->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 1, 2, 1, 2}));
    EXPECT_EQ(krkrk->getWorkspaceSizeInBytes(), krkrk_queried_workspace);
    EXPECT_GT(krkrk->getWorkspaceSizeInBytes(), 0U);
    krkrk->run(2.0f);

    stream.synchronize();
    expectFloatVectorNear(
        copyGpuTensorAsFloat(rkrk->getOutputTensor(), stream), {64.0f, 72.0f, 96.0f, 104.0f, 128.0f, 136.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(krkrk->getOutputTensor(), stream),
                          {48.0f, 56.0f, 80.0f, 88.0f, 176.0f, 184.0f, 208.0f, 216.0f});
}

TEST(Reducer, GeneralDenseSumPlannerExecutesManyAlternatingRunsAndSingletonSeparatedRuns) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<float> alternating_values(128);
    for (size_t i = 0; i < alternating_values.size(); ++i) {
        alternating_values[i] = static_cast<float>(i + 1);
    }
    Tensor alternating_input = makeGpuTensor(alternating_values, {2, 2, 2, 2, 2, 2, 2}, stream);
    Reducer alternating_reduction(
        ReductionOp::Sum, std::vector<uint32_t>{0, 2, 4, 6}, DataType::FP32);
    const size_t alternating_workspace =
        alternating_reduction.queryWorkspaceSizeInBytes(alternating_input.getDescriptor(), stream);
    auto alternating = alternating_reduction.stamp(alternating_input, stream);
    ASSERT_EQ(alternating->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(alternating->getWorkspaceSizeInBytes(), alternating_workspace);
    const std::vector<std::vector<uint32_t>> alternating_stage_axes = alternating->getComposedStageAxes();
    ASSERT_EQ(alternating_stage_axes.size(), 4U);
    // Dense VALUE cutover replans from the canonical current-run problem after every pass. The reported axes are
    // therefore current physical run indices, not persistent original-axis identities. Planner selection/order is
    // covered by DenseReductionPlannerTest; this execution test only requires one physical reduction axis per pass
    // and verifies the resulting values below.
    for (const std::vector<uint32_t>& stage_axes : alternating_stage_axes) {
        ASSERT_EQ(stage_axes.size(), 1U);
    }
    alternating->run();

    std::vector<float> singleton_values(12);
    for (size_t i = 0; i < singleton_values.size(); ++i) {
        singleton_values[i] = static_cast<float>(i + 1);
    }
    Tensor singleton_input = makeGpuTensor(singleton_values, {2, 1, 3, 2}, stream);
    auto singleton = Reducer(ReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32)
                         .stamp(singleton_input, stream);
    ASSERT_EQ(singleton->getPath(), ReductionLayout::DisjointRuns);
    singleton->run();

    // Reducing only singleton axes is value-identical. The composed wrapper still needs one direct pass so runtime
    // scaling and output conversion happen exactly once.
    Tensor singleton_only_input = makeGpuTensor(singleton_values, {2, 1, 3, 1, 2}, stream);
    auto singleton_only = Reducer(ReductionOp::Sum, std::vector<uint32_t>{1, 3}, DataType::FP32, 3.0f)
                              .stamp(singleton_only_input, stream);
    ASSERT_EQ(singleton_only->getPath(), ReductionLayout::DisjointRuns);
    singleton_only->run(2.0f);

    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(alternating->getOutputTensor(), stream),
                          {696.0f, 728.0f, 824.0f, 856.0f, 1208.0f, 1240.0f, 1336.0f, 1368.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(singleton->getOutputTensor(), stream), {36.0f, 42.0f});
    std::vector<float> singleton_only_expected(singleton_values.size());
    for (size_t i = 0; i < singleton_values.size(); ++i) {
        singleton_only_expected[i] = 2.0f * singleton_values[i];
    }
    expectFloatVectorNear(copyGpuTensorAsFloat(singleton_only->getOutputTensor(), stream), singleton_only_expected);
}

TEST(Reducer, DenseRkrNonSumOperationsUseComposedDensePath) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                 {2, 3, 2},
                                 stream);

    auto mean = Reducer(ReductionOp::Mean, std::vector<uint32_t>{0, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(mean->getPath(), ReductionLayout::DisjointRuns);
    mean->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(mean->getOutputTensor(), stream), {4.5f, 6.5f, 8.5f});
}

TEST(Reducer, MultiAxisAllAxesUsesDeviceTransformReduce) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);

    std::shared_ptr<StampedReducer> stamped =
        Reducer(ReductionOp::Mean, std::vector<uint32_t>{0, 1, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), ReductionLayout::SingleOutput);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 1, 1}));
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {6.5f});
}

TEST(Reducer, PreallocatedOutputAcceptsAnySingletonEquivalentShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);
    Reducer reduction(ReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32);

    Tensor keep_dimensions(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 3, 1}));
    Tensor squeezed(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    Tensor alternate_singletons(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 1}));
    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 1, 3, 2}));

    std::shared_ptr<StampedReducer> keep_stamped = reduction.stamp(input, keep_dimensions, stream);
    std::shared_ptr<StampedReducer> squeezed_stamped = reduction.stamp(input, squeezed, stream);
    std::shared_ptr<StampedReducer> alternate_stamped = reduction.stamp(input, alternate_singletons, stream);
    EXPECT_EQ(keep_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1}));
    EXPECT_EQ(squeezed_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3}));
    EXPECT_EQ(alternate_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_shape, stream)), std::invalid_argument);

    keep_stamped->run();
    squeezed_stamped->run();
    alternate_stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(keep_dimensions, stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(squeezed, stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(alternate_singletons, stream), {18.0f, 26.0f, 34.0f});
}

TEST(Reducer, SqueezedScalarOutputUsesOneElementShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream);
    Tensor scalar_output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    std::shared_ptr<StampedReducer> stamped =
        Reducer(ReductionOp::Product, std::vector<uint32_t>{0, 1}, DataType::FP32)
            .stamp(input, scalar_output, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(scalar_output, stream), {24.0f});
}

TEST(Reducer, RankNineDenseSumUsesComposedPathRatherThanDynamicStridedMetadata) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(32);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    Tensor input = makeGpuTensor(values, {2, 1, 2, 1, 2, 1, 2, 1, 2}, stream);
    std::shared_ptr<StampedReducer> stamped =
        Reducer(ReductionOp::Sum, std::vector<uint32_t>{0, 2, 4, 6}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), ReductionLayout::DisjointRuns);
    EXPECT_EQ(stamped->getGeometry().rank, 9U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {240.0f, 256.0f});
}

TEST(Reducer, PermutedViewIsRefusedAndRunsAsTheDensePlanOfItsBytesThenTheCopy) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                   {2, 3, 2},
                                   stream);
    // Logical [k,i,j] view of dense physical [i,j,k]. The classifier recognizes [outer=i,reduction=j,inner=k] with the
    // rotated output [k,i]. The reducer refuses it; the caller's dense plan reduces [i,j,k] over j into [i,k] and the
    // dense copy writes [k,i].
    StridedSpan permuted = StridedSpan(storage, 0, {2, 2, 3}, {1, 6, 2});

    expectFloatVectorNear(executeRefusedViewInTwoParts(permuted, ReductionOp::Sum, {2}, stream),
                          {9.0f, 27.0f, 12.0f, 30.0f});
}

TEST(Reducer, DensePhysicalPermutationCanKeepNaturalRetainedOutputOrder) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                   {2, 3, 2},
                                   stream);
    // Logical [j,i,k] over the same dense [i,j,k] storage. Reducing logical j leaves the natural physical [i,k]
    // retained order. The classification stays the natural-output tiled candidate; the view runs as the dense [i, j, k]
    // over j on the dense planner.
    StridedSpan permuted = StridedSpan(storage, 0, {3, 2, 2}, {2, 6, 1});

    auto stamped = Reducer(ReductionOp::Sum, 0, DataType::FP32).stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), ReductionLayout::MiddleBlock);
    ASSERT_TRUE(stamped->getGeometry().permuted_layout.has_value());
    EXPECT_TRUE(stamped->usesDenseExecutablePlan());
    EXPECT_EQ(stamped->getModernRKStrategyChain().rfind("view_physical:", 0), 0U)
        << stamped->getModernRKStrategyChain();
    EXPECT_FALSE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_FALSE(stamped->getGeometry().tiled_output_shared_transpose);
    EXPECT_EQ(stamped->getGeometry().tiled_output_outer_stride, 2U);
    EXPECT_EQ(stamped->getGeometry().tiled_output_inner_stride, 1U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {9.0f, 12.0f, 27.0f, 30.0f});
}

TEST(Reducer, RotatedOutputViewHonorsViewStorageOffset) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f},
                                   {3, 3, 2},
                                   stream);
    // Skip the first physical i-row. getMemPtr() is already offset to the visible dense [2,3,2] region.
    StridedSpan permuted = StridedSpan(storage, 6, {2, 2, 3}, {1, 6, 2});
    ASSERT_EQ(permuted.getStorage().offset, 6U);

    expectFloatVectorNear(executeRefusedViewInTwoParts(permuted, ReductionOp::Sum, {2}, stream),
                          {27.0f, 45.0f, 30.0f, 48.0f});
}

TEST(Reducer, RotatedOutputViewHandlesRetainedHeightAbove65535) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 65537;
    constexpr uint64_t reduction_size = 2;
    constexpr uint64_t inner_size = 3;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    std::vector<float> expected(inner_size * outer_size, 0.0f);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t reduction = 0; reduction < reduction_size; ++reduction) {
            for (uint64_t inner = 0; inner < inner_size; ++inner) {
                const float value = static_cast<float>((outer % 17) + reduction * 3 + inner);
                values[(outer * reduction_size + reduction) * inner_size + inner] = value;
                expected[inner * outer_size + outer] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
    StridedSpan permuted = StridedSpan(storage, 0, {inner_size, outer_size, reduction_size}, {1, reduction_size * inner_size, inner_size});

    expectFloatVectorNear(executeRefusedViewInTwoParts(permuted, ReductionOp::Sum, {2}, stream), expected);
}

TEST(Reducer, PhysicallyContiguousDisjointLogicalAxesRunAsTheDenseTensorOfTheirBytes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_a = 3;
    constexpr uint64_t reduction_b = 5;
    constexpr uint64_t inner_size = 7;
    std::vector<float> values(outer_size * reduction_a * reduction_b * inner_size);
    std::vector<float> expected(outer_size * inner_size, 0.0f);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t a = 0; a < reduction_a; ++a) {
            for (uint64_t b = 0; b < reduction_b; ++b) {
                for (uint64_t inner = 0; inner < inner_size; ++inner) {
                    const float value = static_cast<float>(outer * 100 + a * 10 + b + inner * 0.25);
                    values[((outer * reduction_a + a) * reduction_b + b) * inner_size + inner] = value;
                    expected[outer * inner_size + inner] += value;
                }
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer_size, reduction_a, reduction_b, inner_size}, stream);
    // Physical order [logical 1, logical 0, logical 2, logical 3]. Reduced logical axes {0,2} are
    // disjoint in the view but contiguous in physical storage and can be flattened into one reduction domain.
    // With the natural [outer, inner] output it runs as the dense [outer, a x b, inner] on the dense planner.
    StridedSpan permuted = StridedSpan(storage, 0, {reduction_a, outer_size, reduction_b, inner_size}, {reduction_b * inner_size, reduction_a * reduction_b * inner_size, inner_size, 1});

    auto stamped = Reducer(ReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32)
                       .stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), ReductionLayout::MiddleBlock);
    EXPECT_FALSE(stamped->getGeometry().reduced_axes_are_contiguous);
    ASSERT_TRUE(stamped->getGeometry().permuted_layout.has_value());
    EXPECT_FALSE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_TRUE(stamped->usesDenseExecutablePlan());
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}


TEST(Reducer, Direct2BViewWritesLogicalBAPOrder) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t reduction_size = 3;
    constexpr uint64_t b_size = 5;
    constexpr uint64_t a_size = 2;
    constexpr uint64_t payload_size = 7;
    constexpr uint64_t storage_elements = a_size * reduction_size * b_size * payload_size;

    std::vector<float> values(storage_elements);
    for (uint64_t index = 0; index < storage_elements; ++index) {
        values[index] = static_cast<float>((index % 19) + 1);
    }
    Tensor storage = makeGpuTensor(values, {storage_elements}, stream);

    // Logical [R,B,A,P] aliases compact physical [A,R,B,P]. Reducing R must emit dense logical [B,A,P], so this is the
    // payload-preserving retained-order rotation [A,B,P] -> [B,A,P] (case G). The reducer refuses it; the caller's
    // dense plan reduces [A,R,BxP] over R into [A,B,P] and the dense copy writes [B,A,P].
    StridedSpan permuted = StridedSpan(storage, 0, {reduction_size, b_size, a_size, payload_size}, {b_size * payload_size, payload_size, reduction_size * b_size * payload_size, 1});



    std::vector<float> expected;
    expected.reserve(b_size * a_size * payload_size);
    for (uint64_t b = 0; b < b_size; ++b) {
        for (uint64_t a = 0; a < a_size; ++a) {
            for (uint64_t payload = 0; payload < payload_size; ++payload) {
                float sum = 0.0f;
                for (uint64_t reduction = 0; reduction < reduction_size; ++reduction) {
                    const uint64_t physical_index =
                        (((a * reduction_size + reduction) * b_size + b) * payload_size) + payload;
                    sum += values[physical_index];
                }
                expected.push_back(sum);
            }
        }
    }

    expectFloatVectorNear(executeRefusedViewInTwoParts(permuted, ReductionOp::Sum, {0}, stream), expected);
}

TEST(Reducer, Direct2BViewSupportsPayloadsWiderThanOneSector) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t reduction_size = 2;
    constexpr uint64_t b_size = 2;
    constexpr uint64_t a_size = 3;
    constexpr uint64_t payload_size = 37;
    constexpr uint64_t storage_elements = a_size * reduction_size * b_size * payload_size;
    Tensor storage = makeGpuTensor(std::vector<float>(storage_elements, 1.0f), {storage_elements}, stream);
    StridedSpan permuted = StridedSpan(storage, 0, {reduction_size, b_size, a_size, payload_size}, {b_size * payload_size, payload_size, reduction_size * b_size * payload_size, 1});

    expectFloatVectorNear(executeRefusedViewInTwoParts(permuted, ReductionOp::Sum, {0}, stream),
                          std::vector<float>(b_size * a_size * payload_size, 2.0f));
}

TEST(Reducer, Direct2BViewSupportsEveryValueOperationThroughProductionStamp) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t reduction_size = 4;
    constexpr uint64_t b_size = 3;
    constexpr uint64_t a_size = 2;
    constexpr uint64_t payload_size = 5;
    constexpr uint64_t storage_elements = a_size * reduction_size * b_size * payload_size;
    Tensor storage = makeGpuTensor(std::vector<float>(storage_elements, 1.0f), {storage_elements}, stream);
    StridedSpan permuted = StridedSpan(storage, 0, {reduction_size, b_size, a_size, payload_size}, {b_size * payload_size, payload_size, reduction_size * b_size * payload_size, 1});

    struct OperationCase {
        ReductionOp op;
        float expected;
    };
    const std::vector<OperationCase> cases = {
        {ReductionOp::Sum, 4.0f},
        {ReductionOp::Min, 1.0f},
        {ReductionOp::Max, 1.0f},
        {ReductionOp::Product, 1.0f},
        {ReductionOp::Mean, 1.0f},
        {ReductionOp::L1Norm, 4.0f},
        {ReductionOp::L2Norm, 2.0f},
        {ReductionOp::SumSquares, 4.0f},
    };

    for (const OperationCase& test_case : cases) {

        expectFloatVectorNear(executeRefusedViewInTwoParts(permuted, test_case.op, {0}, stream),
                              std::vector<float>(b_size * a_size * payload_size, test_case.expected));
    }
}

TEST(Reducer, PitchedViewIsCopiedThenReducedByTheDensePlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(24);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    Tensor storage = makeGpuTensor(values, {2, 2, 2, 3}, stream);

    // This is the collapsed repeated-label view produced by the lhs of iirk,kj->ij. The second logical i contributes
    // to the first stride (12 + 6 = 18), leaving two disjoint [reduction=2,inner=3] slabs separated by a physical gap.
    // The reducer refuses the view; the caller copies the slabs into a dense [2,2,3] (copyToDenseAsync) that the dense
    // plan reduces.
    StridedSpan diagonal = StridedSpan(storage, 0, {2, 2, 3}, {18, 3, 1});


    expectFloatVectorNear(executeRefusedViewInTwoParts(diagonal, ReductionOp::Sum, {1}, stream),
                          {5.0f, 7.0f, 9.0f, 41.0f, 43.0f, 45.0f});
}


TEST(Reducer, PitchedViewSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(106, 1.0f);
    Tensor storage = makeGpuTensor(values, {106}, stream);
    StridedSpan pitched_view = StridedSpan(storage, 0, {3, 4, 5}, {40, 7, 1});

    struct OperationCase {
        ReductionOp op;
        float expected;
    };
    const std::vector<OperationCase> cases = {
        {ReductionOp::Sum, 4.0f},
        {ReductionOp::Min, 1.0f},
        {ReductionOp::Max, 1.0f},
        {ReductionOp::Product, 1.0f},
        {ReductionOp::Mean, 1.0f},
        {ReductionOp::L1Norm, 4.0f},
        {ReductionOp::L2Norm, 2.0f},
        {ReductionOp::SumSquares, 4.0f},
    };

    for (const OperationCase& test_case : cases) {

        expectFloatVectorNear(executeRefusedViewInTwoParts(pitched_view, test_case.op, {1}, stream),
                              std::vector<float>(15, test_case.expected));
    }
}

TEST(Reducer, PitchedViewSupportsLeadingReductionWithNoOuterGroup) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(12);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    Tensor storage = makeGpuTensor(values, {2, 2, 3}, stream);

    // Collapsed repeated-label view from iir,j->rj: two contiguous r payloads separated by the diagonal i pitch.
    StridedSpan diagonal = StridedSpan(storage, 0, {2, 3}, {9, 1});


    expectFloatVectorNear(executeRefusedViewInTwoParts(diagonal, ReductionOp::Sum, {0}, stream),
                          {11.0f, 13.0f, 15.0f});
}

TEST(Reducer, UnsupportedViewGateRejectsProductionWhileBenchmarkLegacyHookRemainsExecutable) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<float> values(13);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1);
    }
    Tensor storage = makeGpuTensor(values, {13}, stream);
    StridedSpan unsupported = StridedSpan(storage, 0, {2, 2, 3}, {1, 7, 2});
    Reducer reduction(ReductionOp::Sum, 2, DataType::FP32);

    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     unsupported.getDimensions(), unsupported.getStridesElements(), {2}),
                 NotImplementedException);
    EXPECT_THROW(static_cast<void>(reduction.stamp(unsupported, stream)), NotImplementedException);
}

TEST(Reducer, RankOneAffineViewUsesDeviceTransformReduceWithoutGeneralIndexMetadata) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f},
                                   {3, 3},
                                   stream);
    StridedSpan diagonal = StridedSpan(storage, 0, {3}, {4});

    auto stamped = Reducer(ReductionOp::Sum, 0, DataType::FP32).stamp(diagonal, stream);
    EXPECT_EQ(stamped->getPath(), ReductionLayout::SingleOutput);
    EXPECT_TRUE(stamped->getGeometry().device_transform_uses_affine_stride);
    EXPECT_EQ(stamped->getGeometry().affine_input_stride, 4U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {15.0f});
}

TEST(Reducer, StridedSpanAtAnOffsetIsReducedInPlaceWithoutAView) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                   {13},
                                   stream);
    // The [4, 3] span of the storage from element 1, read transposed (rows {1, 5, 9}, {2, 6, 10}, ...): its bytes are
    // the dense [3, 4] from element 1, reduced over the leading axis in place, with no Tensor view made of it.
    const StridedSpan columns(storage, 1, {4, 3}, {1, 4});
    auto stamped = Reducer(ReductionOp::Sum, 1, DataType::FP32).stamp(columns, stream);
    EXPECT_EQ(stamped->getInput().getMemPtr<void>(), static_cast<void*>(storage.getMemPtr<float>() + 1));
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {15.0f, 18.0f, 21.0f, 24.0f});

    // The [3, 2] span {1, 2}, {5, 6}, {9, 10} (row pitch 4) has gaps between its rows, which the reducer refuses; a
    // caller copies it dense first (Expression's DenseCopy stage).
    const StridedSpan rows(storage, 1, {3, 2}, {4, 1});
    EXPECT_THROW(static_cast<void>(Reducer(ReductionOp::Sum, 1, DataType::FP32).stamp(rows, stream)),
                 NotImplementedException);
    expectFloatVectorNear(executeRefusedViewInTwoParts(rows, ReductionOp::Sum, {1}, stream), {3.0f, 11.0f, 19.0f});
}
