#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

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
