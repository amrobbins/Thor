#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/EquationCompiler.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

const NamedOutput& onlyOutput(const PhysicalOutputs& outputs) {
    if (outputs.outputs.size() != 1u) {
        throw std::runtime_error("CudaSourceEmitterTest expected exactly one output.");
    }
    return outputs.outputs.front();
}

PhysicalOutputs makeExplicitBf16Fp8CancellationOutputs(DataType fp8_dtype) {
    const Expression input = Expression::input("input", fp8_dtype, fp8_dtype);
    const Expression square = (input * input).withComputeDType(DataType::BF16);
    const Expression cancellation = (square - square).withComputeDType(DataType::BF16);

    PhysicalOutputs outputs = Expression::outputs({{"output", cancellation}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {fp8_dtype});
    return outputs;
}

PhysicalOutputs makeExplicitBf16Fp8ScalarOutputs(DataType fp8_dtype) {
    const Expression input = Expression::input("input", fp8_dtype, fp8_dtype);
    const Expression scale = Expression::runtimeScalar("scale", DataType::FP32, DataType::FP32);
    const Expression shifted = (input + Expression::constantScalar(2.0)).withComputeDType(DataType::BF16);
    const Expression output = (shifted * scale).withComputeDType(DataType::BF16);

    PhysicalOutputs outputs = Expression::outputs({{"output", output}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {fp8_dtype, DataType::FP32});
    return outputs;
}

PhysicalOutputs makeExplicitBf16MixedFp8Outputs(DataType input_dtype, DataType output_dtype) {
    const Expression input = Expression::input("input", input_dtype, input_dtype);
    const Expression output = (input * input).withDTypes(DataType::BF16, output_dtype);

    PhysicalOutputs outputs = Expression::outputs({{"output", output}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {input_dtype});
    return outputs;
}

PhysicalExecutionStage makePhysicalSingleOutputStage(const PhysicalOutputs& outputs, MaterializedTensorLayout layout) {
    const NamedOutput& output = onlyOutput(outputs);
    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = *outputs.expr,
        .input_value_ids = {0u},
        .outputs = {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 1u,
            .materialized_layout = layout,
        }},
    };
}

CompiledExecutionStage makeCompiledSingleOutputStage(
    const PhysicalOutputs& outputs, MaterializedTensorLayout layout = MaterializedTensorLayout::RowMajor) {
    const NamedOutput& output = onlyOutput(outputs);
    return CompiledExecutionStage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u},
        {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 1u,
            .materialized_layout = layout,
        }});
}

SpecializedBroadcastGroup makeOneDimensionalBroadcastGroup(SpecializedInputLoadKind load_kind) {
    SpecializedBroadcastGroup group;
    group.numel = 8u;
    group.output_dims = {8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u};
    group.used_input_broadcast_offset_required = {true};
    group.used_input_visible_dims = {{8u}};
    group.used_input_visible_strides = {{1u}};
    group.used_input_load_kinds = {load_kind};
    group.active_axes = {SpecializedBroadcastAxis{
        .dim = 8u,
        .output_stride = 1u,
        .input_strides = {1u},
    }};
    return group;
}

void expectFp8Bf16VectorSource(const std::string& source, DataType fp8_dtype) {
    EXPECT_NE(source.find("#include <cuda_bf16.h>"), std::string::npos);
    EXPECT_NE(source.find("__nv_bfloat162 t"), std::string::npos);
    EXPECT_NE(source.find("__float22bfloat162_rn(static_cast<float2>("), std::string::npos);
    EXPECT_EQ(source.find("half2 t"), std::string::npos);
    EXPECT_EQ(source.find("static_cast<__half2>("), std::string::npos);

    if (fp8_dtype == DataType::FP8_E4M3) {
        EXPECT_NE(source.find("thor_to_fp8x2_e4m3_satfinite(__bfloat1622float2(t"), std::string::npos);
    } else {
        ASSERT_EQ(fp8_dtype, DataType::FP8_E5M2);
        EXPECT_NE(source.find("thor_to_fp8x2_e5m2_nosat(__bfloat1622float2(t"), std::string::npos);
    }
}

void expectExplicitBf16Fp8FlatVectorization(DataType fp8_dtype) {
    PhysicalOutputs outputs = makeExplicitBf16Fp8CancellationOutputs(fp8_dtype);
    auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    ASSERT_EQ(stages.size(), 1u);
    ASSERT_EQ(stages.front().kind, PhysicalExecutionStage::Kind::FusedKernel);
    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stages.front()), fp8_dtype);

    const std::string source = CudaSourceEmitter::emitFlat(stages.front(), "fp8_bf16_flat");
    expectFp8Bf16VectorSource(source, fp8_dtype);
}

void expectExplicitBf16Fp8ScalarsUseBf16Vectors(DataType fp8_dtype) {
    PhysicalOutputs outputs = makeExplicitBf16Fp8ScalarOutputs(fp8_dtype);
    auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    ASSERT_EQ(stages.size(), 1u);
    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stages.front()), fp8_dtype);

    const std::string source = CudaSourceEmitter::emitFlat(stages.front(), "fp8_bf16_scalars");
    EXPECT_NE(source.find("__floats2bfloat162_rn(2.0f, 2.0f)"), std::string::npos);
    EXPECT_NE(source.find("__halves2bfloat162(__nv_bfloat16(in1), __nv_bfloat16(in1))"), std::string::npos);
    EXPECT_EQ(source.find("half2 t"), std::string::npos);
}

void expectExplicitBf16Fp8TransposedVectorization(DataType fp8_dtype) {
    PhysicalOutputs outputs = makeExplicitBf16Fp8CancellationOutputs(fp8_dtype);
    const PhysicalExecutionStage stage = makePhysicalSingleOutputStage(outputs, MaterializedTensorLayout::Transposed);

    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stage), fp8_dtype);

    const std::string source = CudaSourceEmitter::emitFlat(stage, "fp8_bf16_transpose");
    expectFp8Bf16VectorSource(source, fp8_dtype);
}

void expectExplicitBf16MixedFp8TransposedVectorization(DataType input_dtype, DataType output_dtype) {
    PhysicalOutputs outputs = makeExplicitBf16MixedFp8Outputs(input_dtype, output_dtype);
    const PhysicalExecutionStage stage = makePhysicalSingleOutputStage(outputs, MaterializedTensorLayout::Transposed);

    const std::string source = CudaSourceEmitter::emitFlat(stage, "mixed_fp8_bf16_transpose");
    EXPECT_NE(source.find("#include <cuda_bf16.h>"), std::string::npos);
    EXPECT_NE(source.find("__nv_bfloat162 t"), std::string::npos);
    EXPECT_NE(source.find("__float22bfloat162_rn(static_cast<float2>("), std::string::npos);
    EXPECT_EQ(source.find("half2 t"), std::string::npos);
    EXPECT_EQ(source.find("static_cast<__half2>("), std::string::npos);

    if (output_dtype == DataType::FP8_E4M3) {
        EXPECT_NE(source.find("thor_to_fp8x2_e4m3_satfinite(__bfloat1622float2(t"), std::string::npos);
    } else {
        ASSERT_EQ(output_dtype, DataType::FP8_E5M2);
        EXPECT_NE(source.find("thor_to_fp8x2_e5m2_nosat(__bfloat1622float2(t"), std::string::npos);
    }
}

void expectExplicitBf16Fp8TransposedSpecializedBroadcastVectorization(DataType fp8_dtype) {
    PhysicalOutputs outputs = makeExplicitBf16Fp8CancellationOutputs(fp8_dtype);
    CompiledExecutionStage stage = makeCompiledSingleOutputStage(outputs, MaterializedTensorLayout::Transposed);

    SpecializedBroadcastGroup group;
    group.numel = 16u;
    group.output_dims = {4u, 4u};
    group.output_indices = {0u};
    group.used_input_slots = {0u};
    group.used_input_broadcast_offset_required = {true};
    group.used_input_visible_dims = {{4u, 4u}};
    group.used_input_visible_strides = {{4u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 4u, .output_stride = 4u, .input_strides = {4u}},
        SpecializedBroadcastAxis{.dim = 4u, .output_stride = 1u, .input_strides = {1u}},
    };

    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fp8_bf16_transposed_broadcast");
    expectFp8Bf16VectorSource(source, fp8_dtype);
}

void expectExplicitBf16Fp8SpecializedBroadcastVectorization(DataType fp8_dtype, SpecializedInputLoadKind load_kind) {
    PhysicalOutputs outputs = makeExplicitBf16Fp8CancellationOutputs(fp8_dtype);
    CompiledExecutionStage stage = makeCompiledSingleOutputStage(outputs);
    const SpecializedBroadcastGroup group = makeOneDimensionalBroadcastGroup(load_kind);

    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stage), fp8_dtype);

    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fp8_bf16_broadcast"));
    EXPECT_NE(source.find("#include <cuda_bf16.h>"), std::string::npos);
    EXPECT_NE(source.find("__nv_bfloat162 t"), std::string::npos);
    EXPECT_EQ(source.find("half2 t"), std::string::npos);
    EXPECT_EQ(source.find("static_cast<__half2>("), std::string::npos);

    if (load_kind == SpecializedInputLoadKind::NativeVector) {
        EXPECT_NE(source.find("__float22bfloat162_rn(static_cast<float2>("), std::string::npos);
    } else {
        EXPECT_NE(source.find("__floats2bfloat162_rn(float(in0["), std::string::npos);
    }

    if (fp8_dtype == DataType::FP8_E4M3) {
        EXPECT_NE(source.find("thor_to_fp8x2_e4m3_satfinite(__bfloat1622float2(t"), std::string::npos);
    } else {
        ASSERT_EQ(fp8_dtype, DataType::FP8_E5M2);
        EXPECT_NE(source.find("thor_to_fp8x2_e5m2_nosat(__bfloat1622float2(t"), std::string::npos);
    }
}

void expectDefaultFp8FlatVectorizationUsesBf16(DataType fp8_dtype) {
    const Expression input = Expression::input("input", fp8_dtype, fp8_dtype);
    PhysicalOutputs outputs = Expression::outputs({{"output", input * input}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {fp8_dtype});
    auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    ASSERT_EQ(stages.size(), 1u);
    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stages.front()), fp8_dtype);

    const std::string source = CudaSourceEmitter::emitFlat(stages.front(), "fp8_default_bf16_flat");
    expectFp8Bf16VectorSource(source, fp8_dtype);
}

void expectExplicitFp16Fp8FlatVectorizationRemainsAvailable(DataType fp8_dtype) {
    const Expression input = Expression::input("input", fp8_dtype, fp8_dtype);
    const Expression output = (input * input).withComputeDType(DataType::FP16);
    PhysicalOutputs outputs = Expression::outputs({{"output", output}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {fp8_dtype});
    auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    ASSERT_EQ(stages.size(), 1u);
    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stages.front()), fp8_dtype);

    const std::string source = CudaSourceEmitter::emitFlat(stages.front(), "fp8_explicit_fp16_flat");
    EXPECT_NE(source.find("half2 t"), std::string::npos);
    EXPECT_NE(source.find("static_cast<__half2>("), std::string::npos);
    EXPECT_EQ(source.find("__nv_bfloat162 t"), std::string::npos);
}

void expectSameDTypeCastSupportedByVectorizedSpecializedBroadcast(DataType dtype) {
    const Expression input = Expression::input("input", dtype, dtype);
    PhysicalOutputs outputs = Expression::outputs({{"output", input.cast(dtype)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {dtype});

    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u},
        {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 1u,
        }});

    SpecializedBroadcastGroup group;
    group.numel = 8u;
    group.output_dims = {8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u};
    group.used_input_broadcast_offset_required = {true};
    group.used_input_visible_dims = {{8u}};
    group.used_input_visible_strides = {{1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector};
    group.active_axes = {SpecializedBroadcastAxis{
        .dim = 8u,
        .output_stride = 1u,
        .input_strides = {1u},
    }};

    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stage), dtype);

    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
    EXPECT_NE(source.find("t1 = t0;"), std::string::npos);
}

}  // namespace

TEST(CudaSourceEmitter, VectorizedSpecializedBroadcastAcceptsSameDTypeBf16Cast) {
    expectSameDTypeCastSupportedByVectorizedSpecializedBroadcast(DataType::BF16);
}

TEST(CudaSourceEmitter, VectorizedSpecializedBroadcastAcceptsSameDTypeFp16Cast) {
    expectSameDTypeCastSupportedByVectorizedSpecializedBroadcast(DataType::FP16);
}

TEST(CudaSourceEmitter, RaggedSpecializedBroadcastUsesDeviceActiveExtentAndSkipsOffsetsAsValueData) {
    const Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression marked = values.withRaggedRuntimeExtent(offsets, 2, 6, 2);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::UINT32});

    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u},
        {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 2u,
        }});

    SpecializedBroadcastGroup group;
    group.numel = 12u;
    group.output_dims = {6u, 2u};
    group.output_indices = {0u};
    group.used_input_slots = {0u};
    group.used_input_broadcast_offset_required = {true};
    group.used_input_visible_dims = {{6u, 2u}};
    group.used_input_visible_strides = {{2u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 2u, .input_strides = {2u}},
        SpecializedBroadcastAxis{.dim = 2u, .output_stride = 1u, .input_strides = {1u}},
    };

    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
    EXPECT_NE(source.find("active_values_raw = static_cast<unsigned long long>(in1[2ULL])"), std::string::npos);
    EXPECT_NE(source.find("runtime_numel_g0_u64 = active_values * 2ULL"), std::string::npos);
    EXPECT_NE(source.find("idx += grid_stride"), std::string::npos);
    EXPECT_EQ(source.find("in1_offset"), std::string::npos);
}

TEST(CudaSourceEmitter, VectorizedFlatSupportsFp8StorageWithExplicitBf16Compute) {
    expectExplicitBf16Fp8FlatVectorization(DataType::FP8_E4M3);
    expectExplicitBf16Fp8FlatVectorization(DataType::FP8_E5M2);
}

TEST(CudaSourceEmitter, VectorizedFp8Bf16ComputeKeepsLiteralsAndRuntimeScalarsInBf16) {
    expectExplicitBf16Fp8ScalarsUseBf16Vectors(DataType::FP8_E4M3);
    expectExplicitBf16Fp8ScalarsUseBf16Vectors(DataType::FP8_E5M2);
}

TEST(CudaSourceEmitter, VectorizedTransposedMaterializationSupportsFp8StorageWithExplicitBf16Compute) {
    expectExplicitBf16Fp8TransposedVectorization(DataType::FP8_E4M3);
    expectExplicitBf16Fp8TransposedVectorization(DataType::FP8_E5M2);
}

TEST(CudaSourceEmitter, MixedFp8TransposedMaterializationSupportsExplicitBf16Compute) {
    expectExplicitBf16MixedFp8TransposedVectorization(DataType::FP8_E4M3, DataType::FP8_E5M2);
    expectExplicitBf16MixedFp8TransposedVectorization(DataType::FP8_E5M2, DataType::FP8_E4M3);
}

TEST(CudaSourceEmitter, TransposedSpecializedBroadcastSupportsFp8StorageWithExplicitBf16Compute) {
    expectExplicitBf16Fp8TransposedSpecializedBroadcastVectorization(DataType::FP8_E4M3);
    expectExplicitBf16Fp8TransposedSpecializedBroadcastVectorization(DataType::FP8_E5M2);
}

TEST(CudaSourceEmitter, VectorizedSpecializedBroadcastSupportsFp8StorageWithExplicitBf16Compute) {
    for (const DataType fp8_dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        expectExplicitBf16Fp8SpecializedBroadcastVectorization(fp8_dtype, SpecializedInputLoadKind::NativeVector);
        expectExplicitBf16Fp8SpecializedBroadcastVectorization(fp8_dtype, SpecializedInputLoadKind::ScalarPack);
    }
}

TEST(CudaSourceEmitter, VectorizedFp8DefaultComputeUsesBf16) {
    expectDefaultFp8FlatVectorizationUsesBf16(DataType::FP8_E4M3);
    expectDefaultFp8FlatVectorizationUsesBf16(DataType::FP8_E5M2);
}

TEST(CudaSourceEmitter, VectorizedFp8ExplicitFp16ComputeRemainsAvailable) {
    expectExplicitFp16Fp8FlatVectorizationRemainsAvailable(DataType::FP8_E4M3);
    expectExplicitFp16Fp8FlatVectorizationRemainsAvailable(DataType::FP8_E5M2);
}
