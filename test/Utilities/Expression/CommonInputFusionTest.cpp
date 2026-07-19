#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/EquationCompiler.h"

#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/RaggedExpression.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace ThorImplementation;

TEST(EquationCompiler, MatmulExplicitTf32ComputeSurvivesStageSplitAndCompile) {
    auto x = Expression::input("x", DataType::FP32, DataType::FP32);
    auto w = Expression::input("w", DataType::FP32, DataType::FP32);
    auto y = Expression::matmul(x, w, false, false, DataType::TF32, DataType::FP32);

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32});

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::Matmul);

    auto compiled = EquationCompiler::compileMatmul(stages[0].expr, stages[0].outputs);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->lhs_dtype, DataType::FP32);
    EXPECT_EQ(compiled->rhs_dtype, DataType::FP32);
    EXPECT_EQ(compiled->aux_dtype, DataType::FP32);
    EXPECT_EQ(compiled->output_dtype, DataType::FP32);
    EXPECT_EQ(compiled->compute_dtype, DataType::TF32);
}

TEST(EquationCompiler, MatmulRejectsImplicitMixedOperandDtypeFallback) {
    auto x = Expression::input("x", DataType::FP32, DataType::FP32);
    auto w = Expression::input("w", DataType::BF16, DataType::BF16);
    auto y = Expression::matmul(x, w, false, false, DataType::BF16, DataType::FP32);

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::BF16});

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::Matmul);

    EXPECT_THROW((void)EquationCompiler::compileMatmul(stages[0].expr, stages[0].outputs), std::runtime_error);
}

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

TEST(ExpressionDTypeResolution, CudnnValueReductionAlwaysMaterializesFp32) {
    auto x = Expression::input("x", DataType::FP16, DataType::FP16);
    auto sum = x.reduce_sum({1}, {}).withOutputDType(DataType::FP16);

    auto physical = Expression::outputs({{"sum", sum}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP16});

    const ExprNode& reduction = physical.expr->nodes.at(physical.outputs.at(0).node_idx);
    ASSERT_EQ(reduction.op, ExprOp::REDUCE_SUM);
    ASSERT_TRUE(reduction.output_dtype.has_value());
    EXPECT_EQ(reduction.output_dtype.value(), DataType::FP32);
    ASSERT_TRUE(reduction.compute_dtype.has_value());
    EXPECT_EQ(reduction.compute_dtype.value(), DataType::FP32);

    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);
    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::Reduction);

    auto compiled = EquationCompiler::compileReduction(stages[0].expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->input_dtype, DataType::FP16);
    EXPECT_EQ(compiled->output_dtype, DataType::FP32);
    EXPECT_EQ(compiled->compute_dtype, DataType::FP32);
}

TEST(EquationCompiler, ExplicitCastAfterReductionControlsLowPrecisionStorage) {
    auto x = Expression::input("x", DataType::FP16, DataType::FP16);
    auto y = x.reduce_sum({1}, {}).cast(DataType::FP16);

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP16});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::Reduction);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::FusedKernel);

    auto compiled_reduction = EquationCompiler::compileReduction(stages[0].expr);
    ASSERT_NE(compiled_reduction, nullptr);
    EXPECT_EQ(compiled_reduction->output_dtype, DataType::FP32);

    ASSERT_EQ(stages[1].outputs.size(), 1);
    const ExprNode& cast_output = stages[1].expr.nodes.at(stages[1].outputs[0].local_node_idx);
    EXPECT_EQ(cast_output.op, ExprOp::CAST);
    ASSERT_TRUE(cast_output.output_dtype.has_value());
    EXPECT_EQ(cast_output.output_dtype.value(), DataType::FP16);
}

TEST(ExpressionDTypeResolution, CudnnReductionPromotesBf16ToFp32WithoutNarrowingThroughFp16) {
    EXPECT_EQ(toSupportedInputDType(ExprOp::REDUCE_SUM, DataType::BF16), DataType::FP32);
    EXPECT_EQ(toSupportedInputDType(ExprOp::REDUCE_MAX, DataType::BF16), DataType::FP32);
    EXPECT_EQ(toSupportedInputDType(ExprOp::REDUCE_ARGMAX, DataType::BF16), DataType::FP32);

    EXPECT_EQ(toSupportedInputDType(ExprOp::REDUCE_SUM, DataType::FP8_E4M3), DataType::FP16);
    EXPECT_EQ(toSupportedInputDType(ExprOp::REDUCE_SUM, DataType::FP8_E5M2), DataType::FP16);
}

TEST(ExpressionDTypeResolution, CudnnSoftmaxPreservesBf16AndOnlyPromotesFp8) {
    EXPECT_EQ(toSupportedInputDType(ExprOp::SOFTMAX, DataType::BF16), DataType::BF16);
    EXPECT_EQ(toSupportedInputDType(ExprOp::SOFTMAX, DataType::FP16), DataType::FP16);
    EXPECT_EQ(toSupportedInputDType(ExprOp::SOFTMAX, DataType::FP32), DataType::FP32);
    EXPECT_EQ(toSupportedInputDType(ExprOp::SOFTMAX, DataType::FP8_E4M3), DataType::FP16);
    EXPECT_EQ(toSupportedInputDType(ExprOp::SOFTMAX, DataType::FP8_E5M2), DataType::FP16);
}

TEST(EquationCompiler, Bf16ReductionCompatibilityDoesNotRewriteProducerStorageDtype) {
    auto x = Expression::input("x", DataType::BF16, DataType::BF16);
    auto trunk = x + 1.0;
    auto outputs = Expression::outputs({
        {"trunk", trunk},
        {"sum", trunk.reduce_sum({1}, {})},
    });

    auto physical = outputs.physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 2);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::Reduction);

    bool found_bf16_trunk = false;
    for (const CompiledStageOutput& output : stages[0].outputs) {
        if (output.local_node_idx == UINT32_MAX) {
            continue;
        }
        const ExprNode& output_node = stages[0].expr.nodes.at(output.local_node_idx);
        ASSERT_TRUE(output_node.output_dtype.has_value());
        if (output.name == "trunk") {
            EXPECT_EQ(output_node.output_dtype.value(), DataType::BF16);
            ASSERT_TRUE(output_node.compute_dtype.has_value());
            EXPECT_EQ(output_node.compute_dtype.value(), DataType::BF16);
            found_bf16_trunk = true;
        }
    }
    EXPECT_TRUE(found_bf16_trunk);

    auto compiled_reduction = EquationCompiler::compileReduction(stages[1].expr);
    ASSERT_NE(compiled_reduction, nullptr);
    EXPECT_EQ(compiled_reduction->input_dtype, DataType::FP32);
    EXPECT_EQ(compiled_reduction->output_dtype, DataType::FP32);
}

TEST(EquationCompiler, ReductionBoundaryPropagatesProducerComputePolicyWithoutLeakingReductionPolicy) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    auto trunk = (x + 1.0) * (y - 0.5);
    auto outputs = Expression::outputs({
        {"sum", trunk.reduce_sum({2}, {})},
        {"pointwise", x + 2.0},
    });

    auto physical = outputs.physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP16, DataType::FP32});

    bool found_widened_fp16_branch = false;
    bool found_fp16_pointwise_output = false;
    for (const ExprNode& node : physical.expr->nodes) {
        if (node.op != ExprOp::ADD || !node.output_dtype.has_value() || !node.compute_dtype.has_value()) {
            continue;
        }

        if (node.output_dtype.value() == DataType::FP16 && node.compute_dtype.value() == DataType::FP32) {
            found_widened_fp16_branch = true;
        }
        if (node.output_dtype.value() == DataType::FP16 && node.compute_dtype.value() == DataType::FP16) {
            found_fp16_pointwise_output = true;
        }
    }

    EXPECT_TRUE(found_widened_fp16_branch);
    EXPECT_TRUE(found_fp16_pointwise_output);
}

TEST(EquationCompiler, Bf16SoftmaxCompilesWithBf16InputAndOutput) {
    auto x = Expression::input("x", DataType::BF16, DataType::BF16);
    auto y = x.softmax();

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::Softmax);

    auto compiled_softmax = EquationCompiler::compileSoftmax(stages[0].expr);
    ASSERT_NE(compiled_softmax, nullptr);
    EXPECT_EQ(compiled_softmax->input_dtype, DataType::BF16);
    EXPECT_EQ(compiled_softmax->output_dtype, DataType::BF16);
}

TEST(CudaSourceEmitter, Bf16SpecialFunctionsNeverNarrowThroughFp16) {
    auto x = Expression::input("x", DataType::BF16, DataType::BF16);
    auto y = x.expm1() + x.log1p() + x.tanh() + x.normcdf();

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::BF16});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "bf16_special_functions");

    EXPECT_NE(source.find("expm1f(float("), std::string::npos);
    EXPECT_NE(source.find("log1pf(float("), std::string::npos);
    EXPECT_NE(source.find("tanhf(float("), std::string::npos);
    EXPECT_NE(source.find("normcdff(float("), std::string::npos);
    EXPECT_EQ(source.find("float(half(float("), std::string::npos);
    EXPECT_EQ(source.find("__float22half2_rn(__bfloat1622float2"), std::string::npos);
}

TEST(CudaSourceEmitter, Fp8E4M3CastsUseExplicitSatfiniteIntrinsics) {
    auto x = Expression::input("x", DataType::FP32, DataType::FP32);
    auto y = x.cast(DataType::FP8_E4M3);

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "fp8_e4m3_satfinite_cast");

    EXPECT_NE(source.find("__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)"), std::string::npos);
    EXPECT_NE(source.find("thor_to_fp8_e4m3_satfinite("), std::string::npos);
}


TEST(CudaSourceEmitter, DenseValuewiseKernelDoesNotUseRaggedRuntimeExtentPath) {
    auto x = Expression::input("x", DataType::FP32, DataType::FP32);
    auto y = x.abs() + Expression::constantScalar(1.0);

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    const std::optional<DataType> vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stages[0]);
    EXPECT_FALSE(vectorized_dtype.has_value());
    EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stages[0]), 4U);

    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "dense_valuewise_regression");
    EXPECT_EQ(source.find("active_values_raw"), std::string::npos);
    EXPECT_EQ(source.find("runtime_numel_u64"), std::string::npos);
    EXPECT_EQ(source.find("grid_stride"), std::string::npos);
}

TEST(CudaSourceEmitter, WideScalarFlatPreservesAlignedFastPathAndHandlesUnalignedAliasesAndTailsScalarly) {
    auto x = Expression::input("x", DataType::FP32, DataType::FP32);
    auto y = x.abs() + Expression::constantScalar(1.0);

    auto physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(CudaSourceEmitter::flatElementsPerThread(stages[0]), 4U);

    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "wide_scalar_alignment_guard");
    EXPECT_NE(source.find("const float* in0"), std::string::npos);
    EXPECT_NE(source.find("float* out0"), std::string::npos);
    EXPECT_NE(source.find("if (full_chunk && chunk_aligned)"), std::string::npos);
    EXPECT_NE(source.find("*reinterpret_cast<const float4*>(in0 + base)"), std::string::npos);
    EXPECT_NE(source.find("*reinterpret_cast<float4*>(out0 + base)"), std::string::npos);
    EXPECT_NE(source.find("in0[lane_idx_0]"), std::string::npos);
    EXPECT_NE(source.find("out0[lane_idx_0]"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedValuewiseKernelReadsOffsetsBatchElementOnDevice) {
    const RaggedTensorDescriptor descriptor(DataType::FP32, {}, 4, 12, DataType::UINT32);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    auto physical = Expression::outputs({{"y", ragged.relu().getValues()}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::UINT32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    EXPECT_FALSE(CudaSourceEmitter::getVectorizedStageStorageDType(stages[0]).has_value());
    EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stages[0]), 1U);

    uint32_t offsets_input_slot = UINT32_MAX;
    for (uint32_t slot = 0; slot < stages[0].expr.inputs.size(); ++slot) {
        if (stages[0].expr.inputs[slot].name == "x.offsets") {
            offsets_input_slot = slot;
            break;
        }
    }
    ASSERT_NE(offsets_input_slot, UINT32_MAX);

    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "ragged_valuewise_extent");
    const std::string active_count_load =
        "active_values_raw = static_cast<unsigned long long>(in" + std::to_string(offsets_input_slot) + "[4ULL])";
    EXPECT_NE(source.find(active_count_load), std::string::npos);
    EXPECT_NE(source.find("runtime_numel_u64 = active_values * 1ULL"), std::string::npos);
    EXPECT_NE(source.find("for (; idx < runtime_numel; idx += grid_stride)"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedExtentRejectsMixedDenseOutputInOneFusedKernel) {
    const RaggedTensorDescriptor descriptor(DataType::FP32, {}, 4, 12, DataType::UINT32);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    auto physical = Expression::outputs({
        {"ragged", ragged.relu().getValues()},
        {"dense", Expression::input("x.values", DataType::FP32, DataType::FP32) + Expression::constantScalar(1.0)},
    }).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::UINT32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    EXPECT_THROW((void)CudaSourceEmitter::emitFlat(stages[0], "mixed_ragged_dense_extent"), std::runtime_error);
}

TEST(EquationCompiler, RaggedExtentRejectsImplicitDenseReductionFallback) {
    const RaggedTensorDescriptor descriptor(DataType::FP32, {}, 4, 12, DataType::UINT32);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    const PhysicalOutputs outputs =
        Expression::outputs({{"invalid", ragged.getValues().reduce_sum({0}, {}, DataType::FP32)}}).physicalOutputs();

    EXPECT_THROW((void)EquationCompiler::splitAtReductionBoundaries(outputs), std::runtime_error);
}

TEST(CudaSourceEmitter, MultipleRaggedOutputsSharingOffsetsCanFuseTogether) {
    const RaggedTensorDescriptor descriptor(DataType::FP32, {}, 4, 12, DataType::UINT32);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    auto physical = Expression::outputs({
        {"relu", ragged.relu().getValues()},
        {"abs", ragged.abs().getValues()},
    }).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::UINT32});
    auto stages = EquationCompiler::splitAtReductionBoundaries(physical);

    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    EXPECT_NO_THROW({
        const std::string source = CudaSourceEmitter::emitFlat(stages[0], "shared_ragged_extent_outputs");
        EXPECT_NE(source.find("active_values_raw"), std::string::npos);
    });
}
