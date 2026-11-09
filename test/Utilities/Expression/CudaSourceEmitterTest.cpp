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
        EXPECT_NE(source.find("thor_to_fp8x2_e5m2_satfinite(__bfloat1622float2(t"), std::string::npos);
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
        EXPECT_NE(source.find("thor_to_fp8x2_e5m2_satfinite(__bfloat1622float2(t"), std::string::npos);
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
    EXPECT_EQ(source.find("half2 t"), std::string::npos);
    EXPECT_EQ(source.find("static_cast<__half2>("), std::string::npos);
    const std::string satfinite = fp8_dtype == DataType::FP8_E4M3 ? "thor_to_fp8_e4m3_satfinite" : "thor_to_fp8_e5m2_satfinite";

    if (load_kind == SpecializedInputLoadKind::NativeVector) {
        // A contiguous operand lets the wide packet family own all 8 FP8 lanes in
        // one 8-byte packet; the explicit BF16 arithmetic is kept per lane.
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 8u);
        EXPECT_NE(source.find("unsigned long long in0_packet{}"), std::string::npos);
        EXPECT_NE(source.find("const __nv_bfloat16 t"), std::string::npos);
        // FP8 lanes are decoded and encoded two at a time around the per-lane BF16 arithmetic.
        EXPECT_NE(source.find("static_cast<float2>(reinterpret_cast<const __nv_fp8x2_"), std::string::npos) << source;
        EXPECT_NE(source.find(std::string(fp8_dtype == DataType::FP8_E4M3 ? "thor_to_fp8x2_e4m3_satfinite" : "thor_to_fp8x2_e5m2_satfinite") +
                              "(make_float2(float("),
                  std::string::npos)
            << source;
        EXPECT_EQ(source.find("= " + satfinite + "("), std::string::npos) << source;
    } else {
        // Strided scalar-pack operands stay on the paired BF16 vector2 path.
        EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 2u);
        EXPECT_NE(source.find("__nv_bfloat162 t"), std::string::npos);
        EXPECT_NE(source.find("__floats2bfloat162_rn(float(in0["), std::string::npos);
        EXPECT_NE(source.find(std::string(fp8_dtype == DataType::FP8_E4M3 ? "thor_to_fp8x2_e4m3_satfinite" : "thor_to_fp8x2_e5m2_satfinite") +
                              "(__bfloat1622float2(t"),
                  std::string::npos);
    }
}

void expectDefaultFp8FlatComputesInFp32WithSixteenScalarPackets(DataType fp8_dtype) {
    const Expression input = Expression::input("input", fp8_dtype, fp8_dtype);
    PhysicalOutputs outputs = Expression::outputs({{"output", input * input}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {fp8_dtype});
    auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    ASSERT_EQ(stages.size(), 1u);
    // FP32 arithmetic (the default for low-precision storage) uses the per-lane
    // packet emitter: one 16-byte packet = 16 FP8 lanes per tensor per thread.
    EXPECT_FALSE(CudaSourceEmitter::getVectorizedStageStorageDType(stages.front()).has_value());
    EXPECT_EQ(CudaSourceEmitter::flatElementsPerThread(stages.front()), 16u);

    const std::string source = CudaSourceEmitter::emitFlat(stages.front(), "fp8_default_fp32_flat");
    EXPECT_NE(source.find("in0_chunk = *reinterpret_cast<const float4*>(in0 + scalar_base)"), std::string::npos);
    EXPECT_NE(source.find("const float t"), std::string::npos);
    EXPECT_EQ(source.find("__nv_bfloat162 t"), std::string::npos);
    EXPECT_EQ(source.find("half2 t"), std::string::npos);
    // Two lanes per conversion: fp8x2 decode after the packet load, fp8x2 encode before the store.
    const std::string fmt = fp8_dtype == DataType::FP8_E4M3 ? "e4m3" : "e5m2";
    EXPECT_NE(source.find("const float2 in0_f0 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_" + fmt +
                          "*>(&in0_chunk)[0]);"),
              std::string::npos)
        << source;
    EXPECT_NE(source.find("const float2 in0_f7 = "), std::string::npos);
    EXPECT_NE(source.find("reinterpret_cast<__nv_fp8x2_" + fmt + "*>(&out0_chunk)[7] = thor_to_fp8x2_" + fmt +
                          "_satfinite(make_float2("),
              std::string::npos);
    EXPECT_EQ(source.find("= thor_to_fp8_" + fmt + "_satfinite("), std::string::npos);
    EXPECT_EQ(source.find("in0_chunk_data["), std::string::npos);
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
    // FP16 -> FP8 encodes straight from half2 (cvt.rn.satfinite.*x2.f16x2), not through float2.
    const std::string fmt = fp8_dtype == DataType::FP8_E4M3 ? "e4m3" : "e5m2";
    EXPECT_NE(source.find("thor_half2_to_fp8x2_" + fmt + "_satfinite(t"), std::string::npos) << source;
    EXPECT_EQ(source.find("satfinite(__half22float2("), std::string::npos) << source;
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
    ASSERT_EQ(CudaSourceEmitter::getVectorizedStageStorageDType(stage), dtype);

    // Width 8 is a packet multiple: dense broadcast uses the 16-byte packet family.
    // Width 6 is not: it keeps the vector2 pair path.
    for (uint64_t width : {uint64_t{8}, uint64_t{6}}) {
        SpecializedBroadcastGroup group;
        group.numel = width;
        group.output_dims = {width};
        group.output_indices = {0u};
        group.used_input_slots = {0u};
        group.used_input_broadcast_offset_required = {true};
        group.used_input_visible_dims = {{width}};
        group.used_input_visible_strides = {{1u}};
        group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector};
        group.active_axes = {SpecializedBroadcastAxis{
            .dim = width,
            .output_stride = 1u,
            .input_strides = {1u},
        }};

        std::string source;
        ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
        if (width == 8) {
            EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 8u);
            EXPECT_NE(source.find("t1_rbp0 = t0_rbp0;"), std::string::npos);
            EXPECT_NE(source.find("packet_count_g0 = 1U"), std::string::npos);
            EXPECT_NE(source.find("in0_packet = *reinterpret_cast<const float4*>(in0 + in0_offset_base)"), std::string::npos);
            EXPECT_NE(source.find("*reinterpret_cast<float4*>(out0 + scalar_base) = out0_packet"), std::string::npos);
            EXPECT_EQ(source.find("active_values"), std::string::npos);
        } else {
            EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 2u);
            EXPECT_NE(source.find("t1 = t0;"), std::string::npos);
            EXPECT_NE(source.find("in0_native_aligned"), std::string::npos);
            EXPECT_NE(source.find("out0_native_aligned"), std::string::npos);
            EXPECT_NE(source.find("reinterpret_cast<unsigned long long>(in0 + in0_offset0)"), std::string::npos);
            EXPECT_NE(source.find("out0[idx0] = out0_packed_scalars[0]"), std::string::npos);
        }
    }
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

TEST(CudaSourceEmitter, RaggedSpecializedBroadcastUsesEightScalarBf16PacketsWhenRowsArePacketAligned) {
    const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withRaggedRuntimeExtent(
        active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::BF16, DataType::UINT32});

    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 3u,
        }});

    SpecializedBroadcastGroup group;
    group.numel = 48u;
    group.output_dims = {6u, 8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 8u}, {1u, 8u}};
    group.used_input_visible_strides = {{8u, 1u}, {8u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::NativeVector};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 8u, .input_strides = {8u, 0u}},
        SpecializedBroadcastAxis{.dim = 8u, .output_stride = 1u, .input_strides = {1u, 1u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 8u);

    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
    EXPECT_NE(source.find("active_values_raw = static_cast<unsigned long long>(in2[0ULL])"), std::string::npos);
    EXPECT_NE(source.find("packet_count_g0"), std::string::npos);
    EXPECT_NE(source.find("float4 in0_packet{}"), std::string::npos);
    EXPECT_NE(source.find("float4 in1_packet{}"), std::string::npos);
    EXPECT_NE(source.find("reinterpret_cast<float4*>(out0 + scalar_base)"), std::string::npos);
    // Vector access is predicated on this packet's actual addresses: operands and
    // outputs may be storage aliases that are element aligned only.
    EXPECT_NE(source.find("const bool vector_packet = ((reinterpret_cast<unsigned long long>(in0 + in0_offset_base) & 15ULL) == 0ULL)"),
              std::string::npos);
    EXPECT_NE(source.find("out0[scalar_base + 7] = reinterpret_cast<const __nv_bfloat16*>(&out0_packet)[7]"), std::string::npos);
    EXPECT_EQ(source.find("in2_offset"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedSpecializedBroadcastReusesScalarOperandAcrossEightScalarPacket) {
    const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withRaggedRuntimeExtent(
        active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::BF16, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 48u;
    group.output_dims = {6u, 8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 8u}, {1u, 1u}};
    group.used_input_visible_strides = {{8u, 1u}, {1u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 8u, .input_strides = {8u, 0u}},
        SpecializedBroadcastAxis{.dim = 8u, .output_stride = 1u, .input_strides = {1u, 0u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 8u);
    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
    EXPECT_NE(source.find("const __nv_bfloat16 in1_packet_scalar = in1[in1_offset_base]"), std::string::npos);
    EXPECT_EQ(source.find("in1_offset_base + 1"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedSpecializedBroadcastFallsBackForNonPacketAlignedRowWidth) {
    const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withRaggedRuntimeExtent(
        active_count, 2, 6, 7, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::BF16, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 42u;
    group.output_dims = {6u, 7u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 7u}, {1u, 7u}};
    group.used_input_visible_strides = {{7u, 1u}, {7u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::ScalarPack, SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 7u, .input_strides = {7u, 0u}},
        SpecializedBroadcastAxis{.dim = 7u, .output_stride = 1u, .input_strides = {1u, 1u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 1u);
    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
    EXPECT_EQ(source.find("float4 in0_packet{}"), std::string::npos);
    EXPECT_NE(source.find("idx += grid_stride"), std::string::npos);
}


TEST(CudaSourceEmitter, RaggedMixedSpecializedBroadcastUsesFourScalarPacketsWithDenseMixedChunkWidths) {
    const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withOutputDType(DataType::FP32).withRaggedRuntimeExtent(
        active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::FP32, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 48u;
    group.output_dims = {6u, 8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 8u}, {1u, 8u}};
    group.used_input_visible_strides = {{8u, 1u}, {8u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::NativeVector};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 8u, .input_strides = {8u, 0u}},
        SpecializedBroadcastAxis{.dim = 8u, .output_stride = 1u, .input_strides = {1u, 1u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 4u);
    std::string source;
    ASSERT_NO_THROW(source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel"));
    EXPECT_NE(source.find("unsigned long long in0_packet{}"), std::string::npos);
    EXPECT_NE(source.find("float4 in1_packet{}"), std::string::npos);
    EXPECT_NE(source.find("float4 out0_packet"), std::string::npos);
    EXPECT_NE(source.find("packet_count_g0"), std::string::npos);
    EXPECT_EQ(source.find("float4 in0_packet{}"), std::string::npos);
    EXPECT_EQ(source.find("in2_offset"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedMixedSpecializedBroadcastReusesFp32ScalarAcrossFourScalarPacket) {
    const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + scale).withOutputDType(DataType::FP32).withRaggedRuntimeExtent(
        active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::FP32, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 48u;
    group.output_dims = {6u, 8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 8u}, {1u, 1u}};
    group.used_input_visible_strides = {{8u, 1u}, {1u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 8u, .input_strides = {8u, 0u}},
        SpecializedBroadcastAxis{.dim = 8u, .output_stride = 1u, .input_strides = {1u, 0u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 4u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel");
    EXPECT_NE(source.find("const float in1_packet_scalar = in1[in1_offset_base]"), std::string::npos);
    EXPECT_EQ(source.find("in1_offset_base + 1"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedMixedSpecializedBroadcastFallsBackWhenRowIsNotFourScalarAligned) {
    const Expression values = Expression::input("values", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withOutputDType(DataType::FP32).withRaggedRuntimeExtent(
        active_count, 2, 6, 7, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::FP32, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 42u;
    group.output_dims = {6u, 7u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 7u}, {1u, 7u}};
    group.used_input_visible_strides = {{7u, 1u}, {7u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::ScalarPack, SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 7u, .input_strides = {7u, 0u}},
        SpecializedBroadcastAxis{.dim = 7u, .output_stride = 1u, .input_strides = {1u, 1u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 1u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel");
    EXPECT_EQ(source.find("in0_packet"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedFp32SpecializedBroadcastUsesFourScalarFloat4PacketsWhenRowsAreAligned) {
    const Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withRaggedRuntimeExtent(
        active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 48u;
    group.output_dims = {6u, 8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 8u}, {1u, 8u}};
    group.used_input_visible_strides = {{8u, 1u}, {8u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::NativeVector};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 8u, .input_strides = {8u, 0u}},
        SpecializedBroadcastAxis{.dim = 8u, .output_stride = 1u, .input_strides = {1u, 1u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 4u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel");
    EXPECT_NE(source.find("float4 in0_packet{}"), std::string::npos);
    EXPECT_NE(source.find("float4 in1_packet{}"), std::string::npos);
    EXPECT_NE(source.find("float4 out0_packet"), std::string::npos);
    EXPECT_NE(source.find("reinterpret_cast<float4*>(out0 + scalar_base)"), std::string::npos);
    EXPECT_EQ(source.find("in2_offset"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedFp32SpecializedBroadcastReusesScalarAcrossFourScalarPacket) {
    const Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + scale).withRaggedRuntimeExtent(
        active_count, 2, 6, 8, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 48u;
    group.output_dims = {6u, 8u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 8u}, {1u, 1u}};
    group.used_input_visible_strides = {{8u, 1u}, {1u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 8u, .input_strides = {8u, 0u}},
        SpecializedBroadcastAxis{.dim = 8u, .output_stride = 1u, .input_strides = {1u, 0u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 4u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel");
    EXPECT_NE(source.find("const float in1_packet_scalar = in1[in1_offset_base]"), std::string::npos);
    EXPECT_EQ(source.find("in1_offset_base + 1"), std::string::npos);
}

TEST(CudaSourceEmitter, RaggedFp32SpecializedBroadcastFallsBackWhenRowIsNotFourScalarAligned) {
    const Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
    const Expression marked = (values + bias).withRaggedRuntimeExtent(
        active_count, 2, 6, 7, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    PhysicalOutputs outputs = Expression::outputs({{"output", marked}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32, DataType::UINT32});
    const NamedOutput& output = onlyOutput(outputs);
    CompiledExecutionStage stage(
        *outputs.expr,
        std::shared_ptr<CompiledEquation>{},
        {0u, 1u, 2u},
        {CompiledStageOutput{.name = output.name, .local_node_idx = output.node_idx, .value_id = 3u}});

    SpecializedBroadcastGroup group;
    group.numel = 42u;
    group.output_dims = {6u, 7u};
    group.output_indices = {0u};
    group.used_input_slots = {0u, 1u};
    group.used_input_broadcast_offset_required = {true, true};
    group.used_input_visible_dims = {{6u, 7u}, {1u, 7u}};
    group.used_input_visible_strides = {{7u, 1u}, {7u, 1u}};
    group.used_input_load_kinds = {SpecializedInputLoadKind::ScalarPack, SpecializedInputLoadKind::ScalarPack};
    group.active_axes = {
        SpecializedBroadcastAxis{.dim = 6u, .output_stride = 7u, .input_strides = {7u, 0u}},
        SpecializedBroadcastAxis{.dim = 7u, .output_stride = 1u, .input_strides = {1u, 1u}},
    };

    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 1u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fused_kernel");
    EXPECT_EQ(source.find("float4 in0_packet"), std::string::npos);
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

TEST(CudaSourceEmitter, Fp8DefaultComputeUsesFp32WithSixteenScalarPackets) {
    expectDefaultFp8FlatComputesInFp32WithSixteenScalarPackets(DataType::FP8_E4M3);
    expectDefaultFp8FlatComputesInFp32WithSixteenScalarPackets(DataType::FP8_E5M2);
}

TEST(CudaSourceEmitter, Fp8PassthroughOutputKeepsByteCopyWhileOtherInputsDecodeInPairs) {
    const Expression x = Expression::input("x", DataType::FP8_E4M3, DataType::FP8_E4M3);
    const Expression y = Expression::input("y", DataType::FP8_E4M3, DataType::FP8_E4M3);
    PhysicalOutputs outputs = Expression::outputs({{"product", x * y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP8_E4M3, DataType::FP8_E4M3});
    const NamedOutput& product = onlyOutput(outputs);
    const PhysicalExpression& expr = *outputs.expr;
    uint32_t x_node = UINT32_MAX;
    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        if (expr.nodes[i].op == ExprOp::INPUT && expr.inputs.at(expr.nodes[i].input_slot).name == "x") {
            x_node = i;
        }
    }
    ASSERT_NE(x_node, UINT32_MAX);
    const uint32_t x_slot = expr.nodes[x_node].input_slot;
    // The compiler normally routes a pure passthrough elsewhere; the emitter must still
    // handle a fused stage whose output is an FP8 input without decoding and re-encoding it.
    PhysicalExecutionStage stage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = expr,
        .input_value_ids = {0u, 1u},
        .outputs = {CompiledStageOutput{.name = product.name, .local_node_idx = product.node_idx, .value_id = 2u},
                    CompiledStageOutput{.name = "copy", .local_node_idx = x_node, .value_id = 3u}},
    };
    ASSERT_EQ(CudaSourceEmitter::flatElementsPerThread(stage), 16u);
    const std::string source = CudaSourceEmitter::emitFlat(stage, "fp8_passthrough");

    const std::string copied = "in" + std::to_string(x_slot);
    const std::string decoded = "in" + std::to_string(1u - x_slot);
    EXPECT_NE(source.find("const float2 " + decoded + "_f0 = "), std::string::npos) << source;
    EXPECT_EQ(source.find("const float2 " + copied + "_f0 = "), std::string::npos) << source;
    EXPECT_NE(source.find("out1_chunk_data[0] = t" + std::to_string(x_node) + "_l0;"), std::string::npos) << source;
    EXPECT_NE(source.find("reinterpret_cast<__nv_fp8x2_e4m3*>(&out0_chunk)[0] = thor_to_fp8x2_e4m3_satfinite(make_float2("),
              std::string::npos)
        << source;
}

TEST(CudaSourceEmitter, VectorizedFp8ExplicitFp16ComputeRemainsAvailable) {
    expectExplicitFp16Fp8FlatVectorizationRemainsAvailable(DataType::FP8_E4M3);
    expectExplicitFp16Fp8FlatVectorizationRemainsAvailable(DataType::FP8_E5M2);
}
TEST(CudaSourceEmitter, TransposedMaterializationPackedAccessesCheckActualAddresses) {
    const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
    const Expression y = Expression::input("y", DataType::BF16, DataType::BF16);
    PhysicalOutputs outputs = Expression::outputs({{"output", x + y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::BF16, DataType::BF16});
    const NamedOutput& output = onlyOutput(outputs);
    PhysicalExecutionStage stage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = *outputs.expr,
        .input_value_ids = {0u, 1u},
        .outputs = {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 2u,
            .materialized_layout = MaterializedTensorLayout::Transposed,
        }},
    };
    const std::string source = CudaSourceEmitter::emitFlat(stage, "fused_kernel");
    // Runtime tensors can be element-aligned storage aliases: an index that is a
    // pack multiple does not make the address pack aligned.
    EXPECT_NE(source.find("(reinterpret_cast<unsigned long long>(in0 + idx_base) % sizeof("), std::string::npos) << source;
    EXPECT_NE(source.find("(reinterpret_cast<unsigned long long>(in1 + idx_base) % sizeof("), std::string::npos);
    EXPECT_NE(source.find("(reinterpret_cast<unsigned long long>(out0 + out_base_idx) % sizeof("), std::string::npos);
}

namespace {

std::string emitTransposedBinaryFlat(DataType input_dtype, DataType output_dtype) {
    const Expression x = Expression::input("x", input_dtype, input_dtype);
    const Expression y = Expression::input("y", input_dtype, input_dtype);
    PhysicalOutputs outputs = Expression::outputs({{"output", (x + y).withOutputDType(output_dtype)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {input_dtype, input_dtype});
    const NamedOutput& output = onlyOutput(outputs);
    PhysicalExecutionStage stage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = *outputs.expr,
        .input_value_ids = {0u, 1u},
        .outputs = {CompiledStageOutput{
            .name = output.name,
            .local_node_idx = output.node_idx,
            .value_id = 2u,
            .materialized_layout = MaterializedTensorLayout::Transposed,
        }},
    };
    return CudaSourceEmitter::emitFlat(stage, "fused_kernel");
}

}  // namespace

TEST(CudaSourceEmitter, TransposedMaterializationSelectsTileLanesWithoutRuntimeIndexedRegisters) {
    // Indexing a register word or chunk at a thread-dependent lane forces nvcc to spill it to local memory
    // (census local_bytes_per_thread = 8 on the BF16 transposes). Lanes are selected with shifts or selects.
    const std::string packed = emitTransposedBinaryFlat(DataType::BF16, DataType::BF16);
    EXPECT_NE(packed.find("const unsigned int inputLaneShift = inputColLane * SCALAR_BITS;"), std::string::npos) << packed;
    EXPECT_EQ(packed.find("[inputColLane]"), std::string::npos) << packed;

    const std::string cross_width = emitTransposedBinaryFlat(DataType::BF16, DataType::FP32);
    EXPECT_NE(cross_width.find("(LANE == 0U ? in0_chunk_data[0] : in0_chunk_data[1])"), std::string::npos) << cross_width;
    EXPECT_EQ(cross_width.find("_chunk_data[LANE]"), std::string::npos) << cross_width;
}

TEST(CudaSourceEmitter, CrossWidthFp8TransposeDecodesLoadedChunkInPairs) {
    // FP8 -> FP32/BF16/FP16 transposes decode the loaded 4-lane chunk as two fp8x2 pairs, with constant
    // pair and lane indices only (a runtime-indexed register array would be placed in local memory).
    for (const DataType output_dtype : {DataType::FP32, DataType::BF16, DataType::FP16}) {
        const std::string source = emitTransposedBinaryFlat(DataType::FP8_E4M3, output_dtype);
        EXPECT_NE(source.find("const float2 in0_f0 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3*>(&in0_chunk)[0]);"),
                  std::string::npos)
            << source;
        EXPECT_NE(source.find("const float2 in1_f1 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3*>(&in1_chunk)[1]);"),
                  std::string::npos)
            << source;
        EXPECT_EQ(source.find("(in0_chunk).x"), std::string::npos) << source;
        EXPECT_EQ(source.find("[LANE]"), std::string::npos) << source;
    }
}

TEST(CudaSourceEmitter, Fp8ToFp16ConversionsDoNotRoundTripThroughFp32) {
    // FP8 -> FP16 is exact: with explicit FP16 compute the loaded chunk decodes straight to half2 pairs and
    // the per-lane edge path converts each FP8 value directly (no FP8 -> FP16 -> FP32 -> FP16).
    const Expression x = Expression::input("x", DataType::FP8_E5M2, DataType::FP8_E5M2);
    const Expression y = Expression::input("y", DataType::FP8_E5M2, DataType::FP8_E5M2);
    PhysicalOutputs outputs =
        Expression::outputs({{"output", (x + y).withDTypes(DataType::FP16, DataType::FP16)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP8_E5M2, DataType::FP8_E5M2});
    const PhysicalExecutionStage stage = makePhysicalSingleOutputStage(outputs, MaterializedTensorLayout::Transposed);
    const std::string source = CudaSourceEmitter::emitFlat(stage, "fused_kernel");
    EXPECT_NE(source.find("const __half2 in0_h0 = static_cast<__half2>(reinterpret_cast<const __nv_fp8x2_e5m2*>(&in0_chunk)[0]);"),
              std::string::npos)
        << source;
    EXPECT_EQ(source.find("const float2 in0_f0"), std::string::npos) << source;
    EXPECT_NE(source.find("__half(__nv_cvt_fp8_to_halfraw(("), std::string::npos) << source;
    EXPECT_EQ(source.find("half(float("), std::string::npos) << source;
}

TEST(CudaSourceEmitter, Fp8PackedTransposeLoadsEachInputChunkOnce) {
    // A thread's four FP8 input lanes are one 4-byte load (both pairs) on the vectorized path, guarded by
    // the actual address: runtime tensors may be element-offset views.
    const std::string source = emitTransposedBinaryFlat(DataType::FP8_E4M3, DataType::FP8_E4M3);
    EXPECT_NE(source.find("const InputPack in0_chunk = *reinterpret_cast<const InputPack*>(in0 + idx_base);"), std::string::npos)
        << source;
    EXPECT_NE(source.find("(reinterpret_cast<unsigned long long>(in1 + idx_base) % sizeof(InputPack)) == 0ULL"), std::string::npos)
        << source;
    EXPECT_NE(source.find("reinterpret_cast<const InputVec2*>(&in0_chunk)[PAIR]"), std::string::npos) << source;
    EXPECT_EQ(source.find("reinterpret_cast<const InputVec2*>(in0 + idx_base)"), std::string::npos) << source;
}

TEST(CudaSourceEmitter, TransposedPairLanesResolveIdentityOpsAsTheScalarLanesDo) {
    // x * 1 is an alias: the scalar edge lanes copy the stored FP8 bytes, so the pair lanes must too
    // (decode + satfinite re-encode turns +-Inf into +-max finite and canonicalises NaN).
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const std::string fmt = dtype == DataType::FP8_E4M3 ? "e4m3" : "e5m2";
        const Expression x = Expression::input("x", dtype, dtype);
        PhysicalOutputs outputs = Expression::outputs({{"output", x * Expression::constantScalar(1.0)}}).physicalOutputs();
        resolveOutputsDTypesInPlace(outputs, {dtype});
        const PhysicalExecutionStage stage = makePhysicalSingleOutputStage(outputs, MaterializedTensorLayout::Transposed);
        const std::string source = CudaSourceEmitter::emitFlat(stage, "fused_kernel");
        EXPECT_NE(source.find("output_vec2[PAIR] = reinterpret_cast<const OutputVec2*>(&in0_chunk)[PAIR];"), std::string::npos)
            << source;
        EXPECT_EQ(source.find("thor_to_fp8x2_" + fmt + "_satfinite(t"), std::string::npos) << source;
    }
    // Inside an expression the alias is a copy of its source, not a multiply by a literal one.
    const Expression x = Expression::input("x", DataType::FP8_E5M2, DataType::FP8_E5M2);
    const Expression y = Expression::input("y", DataType::FP8_E5M2, DataType::FP8_E5M2);
    PhysicalOutputs outputs = Expression::outputs({{"output", x * Expression::constantScalar(1.0) + y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP8_E5M2, DataType::FP8_E5M2});
    const PhysicalExecutionStage stage = makePhysicalSingleOutputStage(outputs, MaterializedTensorLayout::Transposed);
    const std::string source = CudaSourceEmitter::emitFlat(stage, "fused_kernel");
    EXPECT_NE(source.find("thor_to_fp8x2_e5m2_satfinite(t"), std::string::npos) << source;
    EXPECT_EQ(source.find("_tp0.x * t"), std::string::npos) << source;
}

namespace {

std::string emitSingleOutputFlat(const Expression& output, const std::vector<DataType>& input_dtypes) {
    PhysicalOutputs outputs = Expression::outputs({{"output", output}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, input_dtypes);
    auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    if (stages.size() != 1u || stages[0].kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("expected one fused stage");
    }
    return CudaSourceEmitter::emitFlat(stages[0], "fused_kernel");
}

size_t countOccurrences(const std::string& haystack, const std::string& needle) {
    size_t count = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + 1)) ++count;
    return count;
}

}  // namespace

TEST(CudaSourceEmitter, PowByConstantTwoIsEmittedAsMultiplyInEveryFlatPath) {
    for (DataType dtype : {DataType::FP32, DataType::BF16, DataType::FP16, DataType::FP8_E4M3}) {
        const Expression x = Expression::input("x", dtype, dtype);
        const std::string source = emitSingleOutputFlat(x.pow(Expression::constantScalar(2.0)), {dtype});
        EXPECT_EQ(source.find("powf("), std::string::npos) << TensorDescriptor::getElementTypeName(dtype) << "\n" << source;
    }
    // Non-reducible constant exponents keep the exact powf semantics (sqrtf differs
    // from pow(x, 0.5) at -0 and -inf).
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const std::string half_power = emitSingleOutputFlat(x.pow(Expression::constantScalar(0.5)), {DataType::FP32});
    EXPECT_NE(half_power.find("powf("), std::string::npos);
    // A runtime exponent is never rewritten.
    const Expression e = Expression::input("e", DataType::FP32, DataType::FP32);
    const std::string runtime_power = emitSingleOutputFlat(x.pow(e), {DataType::FP32, DataType::FP32});
    EXPECT_EQ(countOccurrences(runtime_power, "powf("), 4u);
}

TEST(CudaSourceEmitter, VectorizedAbsUsesNativeHalf2Abs) {
    for (DataType dtype : {DataType::BF16, DataType::FP16}) {
        // Packed half2/bfloat162 arithmetic is used only on explicit request.
        const Expression x = Expression::input("x", dtype, dtype);
        const std::string source = emitSingleOutputFlat(x.abs().withComputeDType(dtype), {dtype});
        EXPECT_NE(source.find("__habs2("), std::string::npos) << TensorDescriptor::getElementTypeName(dtype);
        EXPECT_EQ(source.find("__hmax2("), std::string::npos);
    }
}

TEST(CudaSourceEmitter, IntegerAndBooleanCastsToLowPrecisionFloatsEmit) {
    // Dtype resolution accepts these casts; the emitter used to throw "Unsupported scalar cast".
    const Expression mask = Expression::input("mask", DataType::BOOLEAN, DataType::BOOLEAN);
    const Expression x = Expression::input("x", DataType::FP8_E4M3, DataType::FP8_E4M3);
    std::string fp8_source;
    ASSERT_NO_THROW(fp8_source = emitSingleOutputFlat(x * mask.cast(DataType::FP8_E4M3), {DataType::BOOLEAN, DataType::FP8_E4M3}));
    EXPECT_NE(fp8_source.find("thor_to_fp8_e4m3_satfinite(("), std::string::npos) << fp8_source;

    const Expression count = Expression::input("count", DataType::INT32, DataType::INT32);
    std::string bf16_source;
    ASSERT_NO_THROW(bf16_source = emitSingleOutputFlat(count.cast(DataType::BF16), {DataType::INT32}));
    // Rounded once from the integer: an FP32 hop would double-round above 2^24.
    EXPECT_NE(bf16_source.find("__int2bfloat16_rn("), std::string::npos) << bf16_source;

    std::string fp16_source;
    ASSERT_NO_THROW(fp16_source = emitSingleOutputFlat(count.cast(DataType::FP16), {DataType::INT32}));
    EXPECT_NE(fp16_source.find("half(float("), std::string::npos) << fp16_source;
}

TEST(CudaSourceEmitter, ExplicitCastToFp8EncodesLanePairs) {
    // Quantize (cast to FP8) is the common FP8 producer: encode the cast input two lanes at a time.
    const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
    const std::string source = emitSingleOutputFlat((x * x).cast(DataType::FP8_E4M3), {DataType::BF16});
    EXPECT_NE(source.find("reinterpret_cast<__nv_fp8x2_e4m3*>(&out0_chunk)[0] = thor_to_fp8x2_e4m3_satfinite(make_float2("),
              std::string::npos)
        << source;
    EXPECT_EQ(source.find("out0_chunk_data[0] = "), std::string::npos) << source;
}
