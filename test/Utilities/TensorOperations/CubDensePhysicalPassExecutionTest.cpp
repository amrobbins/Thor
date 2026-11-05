#include "Utilities/TensorOperations/Cub/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

DenseReductionOutputSpec valueOutput() {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    return output;
}

DenseRKFamilyPhysicalPlan stagedKParallel() {
    DenseRKFamilyPhysicalPlan plan;
    plan.implementation = DenseRKProductionImplementation::KParallelPass;
    plan.strategy = DenseRKStrategy::KParallel;
    plan.progress = DenseRKProgress::Staged;
    plan.input_dimensions = {1, 4, 8};
    plan.reduction_axis = 1;
    plan.outer_size = 1;
    plan.reduction_size = 4;
    plan.inner_size = 8;
    plan.shards_per_output = 2;
    plan.packet_bytes = 16;
    plan.block_threads = 32;
    return plan;
}

}  // namespace

TEST(CubDensePhysicalPassExecution, DescribesDirectAndCompleteAndStagedRKWithoutTensorAllocation) {
    DenseDirectReductionPhysicalPlan direct;
    direct.input_dimensions = {3, 7, 5};
    direct.reduction_axis = 1;
    const DensePhysicalPassLayout direct_layout = describeDensePhysicalPass(direct);
    EXPECT_EQ(direct_layout.input_dimensions, (std::vector<uint64_t>{3, 7, 5}));
    // Physical output retains the reduced axis as a singleton; only the logical squeezed view drops it.
    EXPECT_EQ(direct_layout.intermediate_dimensions, (std::vector<uint64_t>{3, 1, 5}));
    EXPECT_EQ(direct_layout.geometry.squeezed_output_dimensions, (std::vector<uint64_t>{3, 5}));
    EXPECT_EQ(direct_layout.geometry.path, CubReductionPath::TiledFixedSegment);

    DenseRKFamilyPhysicalPlan rk = stagedKParallel();
    const DensePhysicalPassLayout staged = describeDensePhysicalPass(rk);
    EXPECT_EQ(staged.input_dimensions, (std::vector<uint64_t>{1, 4, 8}));
    // Geometry describes a completed reduction with a singleton R; a staged kernel instead writes N FP32 shards.
    EXPECT_EQ(staged.geometry.output_dimensions, (std::vector<uint64_t>{1, 1, 8}));
    EXPECT_EQ(staged.geometry.squeezed_output_dimensions, (std::vector<uint64_t>{1, 8}));
    EXPECT_EQ(staged.intermediate_dimensions, (std::vector<uint64_t>{1, 2, 8}));
    EXPECT_EQ(TensorDescriptor(DataType::FP32, staged.intermediate_dimensions).getTotalNumElements(), 16U);

    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        {1, 4, 8}, {1}, DenseReductionAggregateKind::Value, valueOutput());
    const std::vector<DenseReductionSite> sites = enumerateDenseReductionSites(problem);
    ASSERT_EQ(sites.size(), 1U);
    const DenseReductionProblem partial = advanceDenseReductionProblem(problem, sites[0], 2);
    EXPECT_EQ(denseReductionProblemBytes(partial),
              TensorDescriptor(DataType::FP32, staged.intermediate_dimensions).getArraySizeInBytes());
    EXPECT_TRUE(hasDenseReductionSites(partial));

    rk.progress = DenseRKProgress::Complete;
    const DensePhysicalPassLayout complete = describeDensePhysicalPass(rk);
    EXPECT_EQ(complete.intermediate_dimensions, complete.geometry.output_dimensions);
    EXPECT_EQ(complete.intermediate_dimensions, (std::vector<uint64_t>{1, 1, 8}));
    EXPECT_EQ(complete.geometry.squeezed_output_dimensions, (std::vector<uint64_t>{1, 8}));
}

TEST(CubDensePhysicalPassExecution, RejectsUnrecognizedPhysicalGeometry) {
    DenseReductionPhysicalPlan unregistered;
    EXPECT_THROW((void)describeDensePhysicalPass(unregistered), std::logic_error);
}

// 1227: was OfficialDirectFP32PhysicalPassQueriesAndExecutesLikeStampedReduction. FP32 KR sites run the retained FP32
// KR passes now; BF16 still runs the official direct (CUB) pass. The values are small integers, exact in BF16.
TEST(CubDensePhysicalPassExecution, OfficialDirectBF16PhysicalPassQueriesAndExecutesLikeStampedReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<uint64_t> dims{2, 3, 5};
    std::vector<float> values(30);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>((i % 7) + 1);
    Tensor input = makeGpuTensor(values, dims, stream, DataType::BF16);
    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        dims, {2}, DenseReductionAggregateKind::Value, valueOutput());
    const DenseProductionPhysicalSelectionContext selection_context(
        DataType::BF16, DenseRKValueOperation::Sum, stream);
    const DenseReductionCandidate candidate = selectDensePhysicalPass(
        problem, enumerateDenseReductionSites(problem).front(), selection_context.context());
    ASSERT_NE(dynamic_cast<const DenseDirectReductionPhysicalPlan*>(candidate.physical_plan.get()), nullptr);
    const DensePhysicalPassLayout layout = describeDensePhysicalPass(*candidate.physical_plan);
    // The canonical planner merges adjacent retained runs: [2,3,5] becomes the same physical [6,5] storage.
    EXPECT_EQ(layout.input_dimensions, (std::vector<uint64_t>{6, 5}));
    Tensor physical_input = input;
    physical_input.reshape(layout.input_dimensions);
    EXPECT_EQ(input.getDimensions(), dims);
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, DensePassRole::Complete, layout.geometry.reduction_size);
    Tensor physical_output(gpuPlacement, TensorDescriptor(DataType::FP32, layout.geometry.output_dimensions));
    constexpr float output_scale = 2.0f;
    const size_t scratch_bytes = queryDensePhysicalPassScratchBytes(
        *candidate.physical_plan, semantics, physical_input.getDataType(), physical_input.getMemPtr<void>(),
        physical_input.getTotalNumElements(), physical_output.getDataType(), physical_output.getMemPtr<void>(),
        layout.geometry, output_scale, stream);
    // Real CUB paths still own the scratch reported by their query.
    ASSERT_GT(scratch_bytes, 0U);
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
    launchDensePhysicalPass(*candidate.physical_plan, semantics, scratch, scratch_bytes,
                            physical_input, physical_output, layout.geometry, output_scale, stream);

    auto stamped = CubReduction(CubReductionOp::Sum, 2, DataType::FP32, output_scale).stamp(input, stream);
    EXPECT_EQ(scratch_bytes, stamped->getWorkspaceSizeInBytes());
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(physical_output, stream),
                          copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), 1.0e-5f);
}

TEST(CubDensePhysicalPassExecution, AlignedCompleteRKNeedsNoScratchHandleOrAllocation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    DenseRKFamilyPhysicalPlan physical = stagedKParallel();
    physical.implementation = DenseRKProductionImplementation::RCooperativeAlignedComplete;
    physical.strategy = DenseRKStrategy::RCooperative;
    physical.progress = DenseRKProgress::Complete;
    physical.shards_per_output = 1;
    const DensePhysicalPassLayout layout = describeDensePhysicalPass(physical);
    const std::vector<float> values{
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};
    Tensor input = makeGpuTensor(values, physical.input_dimensions, stream, DataType::FP32);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, layout.geometry.output_dimensions));
    const auto semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, DensePassRole::Complete, layout.geometry.reduction_size);
    EXPECT_EQ(layout.geometry.path, CubReductionPath::TiledFixedSegment);
    EXPECT_EQ(queryDensePhysicalPassScratchBytes(
                  physical, semantics, input.getDataType(), input.getMemPtr<void>(),
                  input.getTotalNumElements(), output.getDataType(), output.getMemPtr<void>(),
                  layout.geometry, 1.0f, stream), 0U);

    Tensor no_scratch;
    ASSERT_FALSE(no_scratch.isInitialized());
    EXPECT_NO_THROW(launchDensePhysicalPass(physical, semantics, no_scratch, 0,
                                           input, output, layout.geometry, 1.0f, stream));
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), std::vector<float>(8, 10.0f), 1.0e-5f);
    // Nonzero scratch is not part of this physical kernel's launch contract.
    EXPECT_THROW(launchDensePhysicalPass(physical, semantics, no_scratch, 1,
                                         input, output, layout.geometry, 1.0f, stream), std::logic_error);
}

TEST(CubDensePhysicalPassExecution, SelectedStagedRKProducesPhysicalFP32PartialsAndNoScratch) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const DenseRKFamilyPhysicalPlan physical = stagedKParallel();
    const DensePhysicalPassLayout layout = describeDensePhysicalPass(physical);
    std::vector<float> values;
    for (uint64_t row = 1; row <= 4; ++row)
        for (uint64_t component = 0; component < 8; ++component)
            values.push_back(static_cast<float>(row));
    Tensor input = makeGpuTensor(values, physical.input_dimensions, stream, DataType::FP16);
    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, layout.intermediate_dimensions));
    // The next selected pass may interpret identical physical storage under a canonicalized shape. Reshaping the
    // local handle must leave the output allocation's descriptor unchanged.
    Tensor canonical_alias = partials;
    canonical_alias.reshape({2, 8});
    EXPECT_EQ(partials.getDimensions(), (std::vector<uint64_t>{1, 2, 8}));
    EXPECT_EQ(canonical_alias.getDimensions(), (std::vector<uint64_t>{2, 8}));
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::L2Norm, DensePassRole::First, 4);
    EXPECT_EQ(queryDensePhysicalPassScratchBytes(physical, semantics,
                                                  input.getDataType(), input.getMemPtr<void>(),
                                                  input.getTotalNumElements(), partials.getDataType(),
                                                  partials.getMemPtr<void>(), layout.geometry, 1.0f, stream), 0U);
    Tensor inert_storage(gpuPlacement, TensorDescriptor(DataType::UINT8, {1}));
    launchDensePhysicalPass(physical, semantics, inert_storage, 0, input, partials,
                            layout.geometry, 1.0f, stream);
    stream.synchronize();
    std::vector<float> expected;
    expected.insert(expected.end(), 8, 5.0f);
    expected.insert(expected.end(), 8, 25.0f);
    expectFloatVectorNear(copyGpuTensorAsFloat(partials, stream), expected, 1.0e-5f);
}

TEST(CubDensePhysicalPassExecution, RetainedFp8PassDescribesActualOutputWithoutExtraScratch) {
    DenseReductionOutputSpec spec = valueOutput();
    const DenseReductionProblem initial = makeInitialDenseReductionProblem(
        {2048, 32}, {0}, DenseReductionAggregateKind::Value, spec);
    DensePhysicalSelectionContext context;
    context.rk.original_input_dtype = DataType::FP8_E4M3;
    context.original_value_operation = CubReductionOp::Sum;
    context.original_reduction_size = 2048;
    const Fp8RK::PhysicalKernel physical{Fp8RK::PhysicalFamily::DeepRSmallK,
                                         128, 0, 0, 0, 0, 0, 6, 16, 16 * 32 * 4};
    const DenseReductionCandidate candidate = makeRetainedFp8RKPhysicalCandidate(
        initial, {0}, physical, context);
    const auto layout = describeDensePhysicalPass(*candidate.physical_plan);
    EXPECT_EQ(layout.input_dimensions, (std::vector<uint64_t>{1, 2048, 32}));
    EXPECT_EQ(layout.geometry.output_dimensions, (std::vector<uint64_t>{1, 1, 32}));
    EXPECT_EQ(layout.intermediate_dimensions, (std::vector<uint64_t>{1, 16, 32}));
    EXPECT_EQ(candidate.next_bytes, 16U * 32U * sizeof(float));
    EXPECT_EQ(candidate.scratch_bytes, 0U);
    EXPECT_EQ(denseReductionProblemBytes(candidate.next_problem), candidate.next_bytes);
}

TEST(CubDensePhysicalPassExecution, RetainedFp8StagedScratchQuerySupportsValueSemanticsButRequiresFp32Partials) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    DenseReductionOutputSpec spec = valueOutput();
    const auto initial = makeInitialDenseReductionProblem(
        {2048, 32}, {0}, DenseReductionAggregateKind::Value, spec);
    DensePhysicalSelectionContext context;
    context.rk.original_input_dtype = DataType::FP8_E4M3;
    context.original_value_operation = CubReductionOp::Sum;
    context.original_reduction_size = 2048;
    const Fp8RK::PhysicalKernel physical{Fp8RK::PhysicalFamily::DeepRSmallK,
                                         128, 0, 0, 0, 0, 0, 6, 16, 16 * 32 * 4};
    const auto selected = makeRetainedFp8RKPhysicalCandidate(initial, {0}, physical, context);
    const auto layout = describeDensePhysicalPass(*selected.physical_plan);
    auto semantics = makeValueReductionStageSemantics(CubReductionOp::Sum, DensePassRole::First, 2048);
    auto query = [&](float scale, DataType dtype) {
        return queryDensePhysicalPassScratchBytes(*selected.physical_plan, semantics,
            DataType::FP8_E4M3, nullptr, 2048 * 32, dtype, nullptr,
            layout.geometry, scale, stream);
    };
    // Scratch query must not reintroduce the removed SUM/FP32-only or
    // unit-scale production gates. The staged pass still owns no scratch;
    // its FP32 partial output is a separate intermediate tensor.
    EXPECT_EQ(query(1.0f, DataType::FP32), 0U);
    EXPECT_EQ(query(2.0f, DataType::FP32), 0U);
    semantics.combine = CubReductionStageCombine::Product;
    EXPECT_EQ(query(1.0f, DataType::FP32), 0U);
    semantics.combine = CubReductionStageCombine::Sum;
    semantics.input_transform = CubReductionStageInputTransform::Square;
    EXPECT_EQ(query(1.0f, DataType::FP32), 0U);
    // A staged pass cannot perform final output conversion or finalization.
    EXPECT_THROW((void)query(1.0f, DataType::FP16), std::logic_error);
    semantics.finalize = CubReductionStageFinalize::SquareRoot;
    EXPECT_THROW((void)query(1.0f, DataType::FP32), std::logic_error);
}
