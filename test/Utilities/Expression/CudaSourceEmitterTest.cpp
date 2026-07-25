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
