#include "Utilities/TensorMathFusion/EquationCompiler.h"

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

    auto stages = EquationCompiler::splitAtReductionBoundaries(outs.physicalOutputs());

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

    auto stages = EquationCompiler::splitAtReductionBoundaries(outs.physicalOutputs());

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::Reduction);
}
