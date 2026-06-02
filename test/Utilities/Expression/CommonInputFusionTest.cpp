#include "Utilities/Expression/EquationCompiler.h"

#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "gtest/gtest.h"

using namespace ThorImplementation;

TEST(EquationCompiler, SharedInputsBecomeOneFusedStage) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    auto outs = Expression::outputs({
        {"sum", x + y},
        {"prod", x * y},
    });

    auto stages = EquationCompiler::splitAtReductionBoundaries(outs.physicalOutputs());

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[0].outputs.size(), 2);
}

TEST(EquationCompiler, DisjointInputsStaySeparateStages) {
    auto a = Expression::input("a");
    auto b = Expression::input("b");
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    auto outs = Expression::outputs({
        {"left", a + b},
        {"right", x * y},
    });

    auto stages = EquationCompiler::splitAtReductionBoundaries(outs.physicalOutputs());

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(EquationCompiler, TransitiveSharedInputsBecomeOneFusedStage) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");
    auto z = Expression::input("z");

    auto outs = Expression::outputs({
        {"xy", x + y},
        {"xz", x * z},
        {"y_shift", y - 1.0},
    });

    auto stages = EquationCompiler::splitAtReductionBoundaries(outs.physicalOutputs());

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[0].outputs.size(), 3);
}

TEST(EquationCompiler, ReductionBoundaryStillSplitsStages) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    auto trunk = x + y;
    auto outs = Expression::outputs({
        {"trunk", trunk},
        {"sum_last", trunk.reduce_sum({1}, {})},
    });

    auto physical = outs.physicalOutputs();
    resolveOutputsDTypesInPlace(physical,
                                {
                                    DataType::FP32,
                                    DataType::FP32,
                                });

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::Reduction);
}

TEST(EquationCompiler, ReductionBoundaryCommonSubexpressionDoesNotCreateExtraKernels) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    auto trunk = x + y;
    auto outs = Expression::outputs({
        {"trunk", trunk},
        {"sum_last", (x + y).reduce_sum({1}, {})},
    });

    auto physical = outs.physicalOutputs();
    resolveOutputsDTypesInPlace(physical,
                                {
                                    DataType::FP32,
                                    DataType::FP32,
                                });

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::Reduction);
}

TEST(EquationCompiler, RmsNormIsOwnBoundaryStageAndCompilesDescriptor) {

    auto x = Expression::input("x", DataType::FP16, DataType::FP16);
    auto scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    auto y = Expression::rmsNorm(x, scale, 32, 1.0e-5, DataType::FP32, DataType::FP16);
    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP16, DataType::FP32});

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::RmsNorm);
    ASSERT_EQ(stages[0].outputs.size(), 1);

    const ExprNode& node = stages[0].expr.nodes.at(stages[0].outputs[0].local_node_idx);
    EXPECT_EQ(node.op, ExprOp::RMSNORM);
    EXPECT_EQ(node.rms_norm_fused_activation, CudnnRmsNormFusedActivation::NONE);

    auto compiled = EquationCompiler::compileRmsNorm(stages[0].expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->normalized_feature_count, 32);
    EXPECT_EQ(compiled->input_dtype, DataType::FP16);
    EXPECT_EQ(compiled->scale_dtype, DataType::FP32);
    EXPECT_EQ(compiled->output_dtype, DataType::FP16);
    EXPECT_EQ(compiled->compute_dtype, DataType::FP32);
    EXPECT_EQ(compiled->fused_activation, CudnnRmsNormFusedActivation::NONE);
}

TEST(EquationCompiler, SwishHelperDoesNotImplicitlyTurnRmsNormIntoCudnnFusion) {

    auto x = Expression::input("x", DataType::BF16, DataType::BF16);
    auto scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    auto y = Expression::rmsNorm(x, scale, 32, 1.0e-5, DataType::FP32, DataType::BF16).swish();
    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16, DataType::FP32});

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::RmsNorm);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::FusedKernel);
    const ExprNode& node = stages[0].expr.nodes.at(stages[0].outputs[0].local_node_idx);
    EXPECT_EQ(node.op, ExprOp::RMSNORM);
    EXPECT_EQ(node.rms_norm_fused_activation, CudnnRmsNormFusedActivation::NONE);
}

TEST(EquationCompiler, RmsNormConsumesPrecedingPointwiseStageWithoutAbsorbingIt) {

    auto x = Expression::input("x", DataType::FP16, DataType::FP16);
    auto scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    auto y = (x + 1.0).rmsNorm(scale, 32, 1.0e-5, DataType::FP32, DataType::FP16);
    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP16, DataType::FP32});

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 2);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    EXPECT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::RmsNorm);
    EXPECT_EQ(stages[1].input_value_ids.size(), 2);
    auto compiled = EquationCompiler::compileRmsNorm(stages[1].expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->fused_activation, CudnnRmsNormFusedActivation::NONE);
}
