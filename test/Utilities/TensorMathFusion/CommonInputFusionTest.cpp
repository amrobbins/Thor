#include "Utilities/TensorMathFusion/EquationCompiler.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

TEST(EquationCompiler, SharedInputsBecomeOneFusedStage) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    Outputs outputs = Expression::outputs({
        {"sum", x + y},
        {"prod", x * y},
    });
    PhysicalOutputs physicaloutputs;
    physicaloutputs.expr = outputs.expression();
    physicaloutputs.outputs = outputs.namedOutputs();

    auto stages = EquationCompiler::splitAtReductionBoundaries(physicaloutputs);

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[0].outputs.size(), 2);
}

TEST(EquationCompiler, DisjointInputsStaySeparateStages) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");
    auto a = Expression::input("a");
    auto b = Expression::input("b");

    Outputs outputs = Expression::outputs({
        {"left", x + y},
        {"right", a * b},
    });
    PhysicalOutputs physicaloutputs;
    physicaloutputs.expr = outputs.expression();
    physicaloutputs.outputs = outputs.namedOutputs();

    auto stages = EquationCompiler::splitAtReductionBoundaries(physicaloutputs);

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::FusedKernel);
}
