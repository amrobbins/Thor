#include "Utilities/TensorOperations/Cub/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
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

}  // namespace

TEST(CubDensePhysicalPassExecution, RejectsUnrecognizedPhysicalGeometry) {
    DenseReductionPhysicalPlan unregistered;
    EXPECT_THROW((void)describeDensePhysicalPass(unregistered), std::logic_error);
}

// 1227: was OfficialDirectFP32PhysicalPassQueriesAndExecutesLikeStampedReduction. 1257: was
// OfficialDirectBF16PhysicalPassQueriesAndExecutesLikeStampedReduction; BF16 KR sites run the retained FP16 / BF16 KR
// passes now (no dense dtype runs the direct CUB pass). The selected pass queries, launches and computes exactly what
// the stamped reduction does. The values are small integers, exact in BF16.
TEST(CubDensePhysicalPassExecution, RetainedBF16KRPhysicalPassQueriesAndExecutesLikeStampedReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<uint64_t> dims{2, 3, 5};
    std::vector<float> values(30);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>((i % 7) + 1);
    Tensor input = makeGpuTensor(values, dims, stream, DataType::BF16);
    const DenseReductionProblem problem = makeInitialDenseReductionProblem(
        dims, {2}, DenseReductionAggregateKind::Value, valueOutput());
    constexpr float output_scale = 2.0f;
    const DenseProductionPhysicalSelectionContext selection_context(
        DataType::BF16, DenseRKValueOperation::Sum, stream, 5, output_scale, CubReductionOp::Sum);
    const DenseReductionCandidate candidate = selectDensePhysicalPass(
        problem, enumerateDenseReductionSites(problem).front(), selection_context.context());
    const auto* kr = dynamic_cast<const DenseRetainedFp8KRPhysicalPlan*>(candidate.physical_plan.get());
    ASSERT_NE(kr, nullptr);
    EXPECT_EQ(kr->input_dtype, DataType::BF16);
    const DensePhysicalPassLayout layout = describeDensePhysicalPass(*candidate.physical_plan);
    // The canonical planner merges adjacent retained runs: [2,3,5] becomes the same physical [6,5] storage.
    EXPECT_EQ(layout.input_dimensions, (std::vector<uint64_t>{6, 5}));
    Tensor physical_input = input;
    physical_input.reshape(layout.input_dimensions);
    EXPECT_EQ(input.getDimensions(), dims);
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, DensePassRole::Complete, layout.geometry.reduction_size);
    Tensor physical_output(gpuPlacement, TensorDescriptor(DataType::FP32, layout.geometry.output_dimensions));
    const size_t scratch_bytes = queryDensePhysicalPassScratchBytes(
        *candidate.physical_plan, semantics, physical_input.getDataType(), physical_input.getMemPtr<void>(),
        physical_input.getTotalNumElements(), physical_output.getDataType(), physical_output.getMemPtr<void>(),
        layout.geometry, output_scale, stream);
    EXPECT_EQ(scratch_bytes, candidate.scratch_bytes);
    Tensor scratch;
    if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
    launchDensePhysicalPass(*candidate.physical_plan, semantics, scratch, scratch_bytes,
                            physical_input, physical_output, layout.geometry, output_scale, stream);

    auto stamped = CubReduction(CubReductionOp::Sum, 2, DataType::FP32, output_scale).stamp(input, stream);
    EXPECT_EQ(scratch_bytes, stamped->getWorkspaceSizeInBytes());
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(physical_output, stream),
                          copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), 1.0e-5f);
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
