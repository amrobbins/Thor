// Emitter-level regression tests for index mapping in index-aware fused kernels: RoPE partner
// loads, strided views broadcast by their parent, shape aliases under the tiled logical-transpose
// consumer, 32-bit index eligibility for source-side spans, the batched FP8 transposed broadcast
// pair path, gather addressing from a single coordinate decode, and out-of-range gather indices. Stages and broadcast groups
// are built directly, as the runtime planner would resolve them for the shapes in each comment.
#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/EquationCompiler.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

struct StageBuilder {
    PhysicalExpression expr;

    uint32_t push(const ExprNode& node) {
        expr.nodes.push_back(node);
        return static_cast<uint32_t>(expr.nodes.size() - 1);
    }
    uint32_t input(const std::string& name, DataType dtype) {
        const uint32_t slot = static_cast<uint32_t>(expr.inputs.size());
        expr.inputs.push_back(NamedInput{.name = name, .slot = slot, .kind = NamedInput::Kind::Tensor});
        ExprNode n{};
        n.op = ExprOp::INPUT;
        n.input_slot = slot;
        n.input_tensor_dtype = dtype;
        n.output_dtype = dtype;
        n.compute_dtype = dtype;
        return push(n);
    }
    uint32_t binary(ExprOp op, uint32_t lhs, uint32_t rhs, DataType compute, DataType output) {
        ExprNode n{};
        n.op = op;
        n.lhs = lhs;
        n.rhs = rhs;
        n.compute_dtype = compute;
        n.output_dtype = output;
        return push(n);
    }
    uint32_t unary(ExprOp op, uint32_t lhs, DataType compute, DataType output) {
        ExprNode n{};
        n.op = op;
        n.lhs = lhs;
        n.compute_dtype = compute;
        n.output_dtype = output;
        return push(n);
    }
    CompiledExecutionStage compiled(uint32_t output_node,
                                    MaterializedTensorLayout layout = MaterializedTensorLayout::RowMajor) const {
        PhysicalExpression e = expr;
        e.output_node = output_node;
        std::vector<uint32_t> ids;
        for (uint32_t i = 0; i < expr.inputs.size(); ++i) ids.push_back(i);
        return CompiledExecutionStage(
            e,
            std::shared_ptr<CompiledEquation>{},
            ids,
            {CompiledStageOutput{.name = "out0", .local_node_idx = output_node, .value_id = 100u, .materialized_layout = layout}});
    }
};

struct GroupInput {
    uint32_t slot;
    std::vector<uint64_t> broadcast_strides;  // per output axis, 0 = broadcast
    std::vector<uint64_t> visible_dims;
    std::vector<uint64_t> visible_strides;
};

SpecializedBroadcastGroup makeGroup(const std::vector<uint64_t>& output_dims,
                                    const std::vector<GroupInput>& inputs,
                                    const std::vector<std::vector<uint64_t>>& node_dims) {
    SpecializedBroadcastGroup g;
    g.numel = 1;
    for (uint64_t d : output_dims) g.numel *= d;
    g.output_dims = output_dims;
    g.output_indices = {0u};
    std::vector<uint64_t> output_strides(output_dims.size(), 1);
    for (size_t i = output_dims.size(); i-- > 1;) output_strides[i - 1] = output_strides[i] * output_dims[i];
    for (size_t axis = 0; axis < output_dims.size(); ++axis) {
        g.active_axes.push_back(SpecializedBroadcastAxis{.dim = output_dims[axis], .output_stride = output_strides[axis], .input_strides = {}});
    }
    for (const GroupInput& in : inputs) {
        g.used_input_slots.push_back(in.slot);
        g.used_input_broadcast_offset_required.push_back(true);
        g.used_input_visible_dims.push_back(in.visible_dims);
        g.used_input_visible_strides.push_back(in.visible_strides);
        g.used_input_load_kinds.push_back(SpecializedInputLoadKind::ScalarPack);
        for (size_t axis = 0; axis < output_dims.size(); ++axis) g.active_axes[axis].input_strides.push_back(in.broadcast_strides[axis]);
    }
    g.node_dims = node_dims;
    return g;
}

uint32_t ropeOf(StageBuilder& b, uint32_t lhs, bool interleaved) {
    ExprNode n{};
    n.op = ExprOp::ROPE;
    n.lhs = lhs;
    n.compute_dtype = DataType::FP32;
    n.output_dtype = DataType::BF16;
    n.rope_interleaved = interleaved;
    return b.push(n);
}

}  // namespace

TEST(FusedIndexMappingEmitter, RopePartnerReadsEachInputAtItsOwnOffset) {
    // rope(x + bias): x [2,4,16,64], bias [64] broadcast along every axis but the head dim.
    StageBuilder b;
    const uint32_t x = b.input("x", DataType::BF16);
    const uint32_t bias = b.input("bias", DataType::BF16);
    const uint32_t sum = b.binary(ExprOp::ADD, x, bias, DataType::FP32, DataType::FP32);
    const uint32_t rope = ropeOf(b, sum, false);
    const CompiledExecutionStage stage = b.compiled(rope);
    const std::vector<uint64_t> dims{2, 4, 16, 64};
    const SpecializedBroadcastGroup group =
        makeGroup(dims,
                  {{0, {4096, 1024, 64, 1}, dims, {4096, 1024, 64, 1}}, {1, {0, 0, 0, 1}, {64}, {1}}},
                  {dims, {64}, dims, dims});
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "rope_kernel");

    // The partner differs only in the head-dim coordinate: each input moves from its own offset
    // (bias moves along its only axis). Reading at a flat output index runs past the 64-element bias.
    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 2u);
    EXPECT_NE(source.find("const unsigned int in1_offset_e1 = in1_offset_e0 + (rope_d1 - rope_d0);"), std::string::npos) << source;
    EXPECT_NE(source.find("const unsigned int in0_offset_e1 = in0_offset_e0 + (rope_d1 - rope_d0);"), std::string::npos) << source;
    EXPECT_EQ(source.find("[rope_peer_idx]"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, RopePairPerThreadComputesTheAngleOncePerPair) {
    StageBuilder b;
    const uint32_t x = b.input("x", DataType::BF16);
    const uint32_t rope = ropeOf(b, x, false);
    const CompiledExecutionStage stage = b.compiled(rope);
    const std::vector<uint64_t> dims{2, 4, 16, 64};
    const SpecializedBroadcastGroup group = makeGroup(dims, {{0, {4096, 1024, 64, 1}, dims, {4096, 1024, 64, 1}}}, {dims, dims});
    ASSERT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 2u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "rope_kernel");
    size_t count = 0;
    for (size_t pos = source.find("sincosf("); pos != std::string::npos; pos = source.find("sincosf(", pos + 1)) ++count;
    EXPECT_EQ(count, 1u) << source;
    // Both lanes use the per-element formulas: first lane cur*c - peer*s, second lane peer*s + cur*c.
    EXPECT_NE(source.find("rope_out0 = rope_v0 * rope_c - rope_v1 * rope_s;"), std::string::npos) << source;
    EXPECT_NE(source.find("rope_out1 = rope_v0 * rope_s + rope_v1 * rope_c;"), std::string::npos) << source;
    EXPECT_NE(source.find("out0[e0] = "), std::string::npos);
    EXPECT_NE(source.find("out0[e1] = "), std::string::npos);
    // Unscaled frequencies come from an emit-time table (double, rounded once), not a powf per pair.
    EXPECT_NE(source.find("static const float rope_inv_freq[] = {1.0f, "), std::string::npos) << source;
    EXPECT_EQ(source.find("powf("), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, BroadcastRopeResultKeepsPerElementPartnerAtInputOffset) {
    // rope(x[1,4,16,64]) + y[2,4,16,64]: the RoPE result is broadcast over the batch, so the stage
    // keeps the per-element RoPE path; its partner read must still use x's own offset.
    StageBuilder b;
    const uint32_t x = b.input("x", DataType::BF16);
    const uint32_t y = b.input("y", DataType::BF16);
    const uint32_t rope = ropeOf(b, x, false);
    const uint32_t sum = b.binary(ExprOp::ADD, rope, y, DataType::FP32, DataType::BF16);
    const CompiledExecutionStage stage = b.compiled(sum);
    const std::vector<uint64_t> dims{2, 4, 16, 64};
    const std::vector<uint64_t> x_dims{1, 4, 16, 64};
    const SpecializedBroadcastGroup group = makeGroup(
        dims, {{0, {0, 1024, 64, 1}, x_dims, {4096, 1024, 64, 1}}, {1, {4096, 1024, 64, 1}, dims, {4096, 1024, 64, 1}}}, {x_dims, dims, x_dims, dims});
    EXPECT_EQ(CudaSourceEmitter::specializedBroadcastElementsPerThread(stage, {group}), 1u);
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "rope_kernel");
    EXPECT_NE(source.find("in0[(rope_first_lane ? (static_cast<unsigned long long>(in0_offset) + rope_peer_abs_delta)"), std::string::npos)
        << source;
    EXPECT_EQ(source.find("[rope_peer_idx]"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, RopePartnerOnStridedInputScalesDeltaByInnerStride) {
    // Interleaved RoPE on a [2,4,16,64] input whose head dim has stride 2 (every other column).
    StageBuilder b;
    const uint32_t x = b.input("x", DataType::BF16);
    const uint32_t rope = ropeOf(b, x, true);
    const CompiledExecutionStage stage = b.compiled(rope);
    const std::vector<uint64_t> dims{2, 4, 16, 64};
    const SpecializedBroadcastGroup group = makeGroup(dims, {{0, {8192, 2048, 128, 2}, dims, {8192, 2048, 128, 2}}}, {dims, dims});
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "rope_kernel");
    EXPECT_NE(source.find("const unsigned int in0_offset_e1 = in0_offset_e0 + (rope_d1 - rope_d0) * 2U;"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, StridedViewBroadcastByParentCollapsesBeforeDecodingCoordinates) {
    // view(x, dims [3,1], strides [4,1], offset 1) + y[3,4]: the view broadcasts along the last axis.
    StageBuilder b;
    const uint32_t x = b.input("x", DataType::FP32);
    const uint32_t y = b.input("y", DataType::FP32);
    ExprNode view{};
    view.op = ExprOp::STRIDED_VIEW;
    view.lhs = x;
    view.compute_dtype = DataType::FP32;
    view.output_dtype = DataType::FP32;
    view.view_dims = {3, 1};
    view.view_strides = {4, 1};
    view.view_element_offset = 1;
    const uint32_t sv = b.push(view);
    const uint32_t sum = b.binary(ExprOp::ADD, sv, y, DataType::FP32, DataType::FP32);
    const CompiledExecutionStage stage = b.compiled(sum);
    SpecializedBroadcastGroup group = makeGroup({3, 4}, {{0, {0, 0}, {3, 4}, {4, 1}}, {1, {4, 1}, {3, 4}, {4, 1}}}, {{3, 4}, {3, 4}, {3, 1}, {3, 4}});
    group.used_input_broadcast_offset_required = {false, true};
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "sv_kernel");

    // The view's own index is the row: idx / 4. Decoding view coordinates straight from idx
    // would take row = idx % 3.
    EXPECT_NE(source.find("view_idx"), std::string::npos) << source;
    EXPECT_NE(source.find("((idx >> 2) % 3U)"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, LogicalTransposeConsumerCollapsesUnsqueezedOperand) {
    // x[4,6]^T + b[6].unsqueeze(1): b is [6,1], broadcast across the 4 output columns.
    StageBuilder b;
    const uint32_t x = b.input("x", DataType::FP32);
    const uint32_t bias = b.input("b", DataType::FP32);
    const uint32_t t = b.unary(ExprOp::TRANSPOSE, x, DataType::FP32, DataType::FP32);
    ExprNode unsq{};
    unsq.op = ExprOp::UNSQUEEZE;
    unsq.lhs = bias;
    unsq.compute_dtype = DataType::FP32;
    unsq.output_dtype = DataType::FP32;
    unsq.unsqueeze_axes = {1};
    const uint32_t u = b.push(unsq);
    const uint32_t sum = b.binary(ExprOp::ADD, t, u, DataType::FP32, DataType::FP32);
    const CompiledExecutionStage stage = b.compiled(sum);
    const SpecializedBroadcastGroup group =
        makeGroup({6, 4}, {{0, {0, 0}, {4, 6}, {6, 1}}, {1, {1, 0}, {6}, {1}}}, {{4, 6}, {6}, {6, 4}, {6, 1}, {6, 4}});
    ASSERT_TRUE(CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, {group}));
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "tu_kernel");

    // b is read at the output row (out_idx / 4), never at the [6,4] output index itself.
    EXPECT_NE(source.find("alias_idx"), std::string::npos) << source;
    EXPECT_NE(source.find("((out_idx >> 2) % 6U)"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, SourceSideSpansBeyond32BitsDisableUInt32IndexMath) {
    // Small output, large source: a [1024, 64] slice whose rows are 5e6 elements apart.
    SpecializedBroadcastGroup group = makeGroup({1024, 64}, {{0, {5000000, 1}, {1024, 64}, {5000000, 1}}}, {{1024, 64}, {1024, 64}});
    EXPECT_FALSE(CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath({group}));

    // A gathered tensor larger than 2^32 elements, e.g. [32768, 152064] logits, in node_dims.
    SpecializedBroadcastGroup gather = makeGroup({32768, 1}, {{0, {1, 0}, {32768, 1}, {1, 1}}}, {{32768, 1}, {32768, 152064}});
    EXPECT_FALSE(CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath({gather}));

    // The ordinary case stays on 32-bit index math.
    SpecializedBroadcastGroup dense = makeGroup({1024, 64}, {{0, {64, 1}, {1024, 64}, {64, 1}}}, {{1024, 64}, {1024, 64}});
    EXPECT_TRUE(CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath({dense}));
}

TEST(FusedIndexMappingEmitter, BatchedFp8TransposedBroadcastPairLanesIncludeBatchOffset) {
    // [2,8,8] FP8 x + bias[8], transposed output: every batch after the first must read its own matrix.
    for (const DataType compute : {DataType::FP32, DataType::BF16}) {
        StageBuilder b;
        const uint32_t x = b.input("x", DataType::FP8_E4M3);
        const uint32_t bias = b.input("b", DataType::FP8_E4M3);
        const uint32_t sum = b.binary(ExprOp::ADD, x, bias, compute, DataType::FP8_E4M3);
        const CompiledExecutionStage stage = b.compiled(sum, MaterializedTensorLayout::Transposed);
        SpecializedBroadcastGroup group;
        group.numel = 128;
        group.output_dims = {2, 8, 8};
        group.output_indices = {0u};
        group.used_input_slots = {0u, 1u};
        group.used_input_broadcast_offset_required = {true, true};
        group.used_input_visible_dims = {{2, 8, 8}, {8}};
        group.used_input_visible_strides = {{64, 8, 1}, {1}};
        group.used_input_load_kinds = {SpecializedInputLoadKind::NativeVector, SpecializedInputLoadKind::NativeVector};
        group.active_axes = {SpecializedBroadcastAxis{.dim = 2, .output_stride = 64, .input_strides = {64, 0}},
                             SpecializedBroadcastAxis{.dim = 8, .output_stride = 8, .input_strides = {8, 0}},
                             SpecializedBroadcastAxis{.dim = 8, .output_stride = 1, .input_strides = {1, 1}}};
        const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "fp8_transposed_broadcast");
        EXPECT_NE(source.find("logical_idx_btp0_0 = matrixOffset + "), std::string::npos) << source;
    }
}

namespace {
size_t countOccurrences(const std::string& haystack, const std::string& needle) {
    size_t count = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + needle.size())) ++count;
    return count;
}
}  // namespace

TEST(FusedIndexMappingEmitter, RankThreeGatherDecodesOutputCoordinatesOnce) {
    // take_along_axis(values, idx, axis=1): values [6,10,7] with storage strides {160,16,2}, idx
    // [6,3,7] with permuted strides {1,6,18}. Both addresses are built from one decode of the
    // output index: two divisions (by 7, then by 3), no modulo. Composing a logical source index
    // and decoding it again through the visible strides of each tensor took 8 modulos and 9
    // divisions, all 64-bit.
    StageBuilder b;
    const uint32_t values = b.input("values", DataType::FP32);
    const uint32_t idx = b.input("idx", DataType::INT32);
    ExprNode take{};
    take.op = ExprOp::TAKE_ALONG_AXIS;
    take.lhs = values;
    take.rhs = idx;
    take.compute_dtype = DataType::FP32;
    take.output_dtype = DataType::FP32;
    take.reduction_axes = {1};
    const uint32_t t = b.push(take);
    const CompiledExecutionStage stage = b.compiled(t);
    SpecializedBroadcastGroup group = makeGroup({6, 3, 7},
                                                {{0, {0, 0, 0}, {6, 10, 7}, {160, 16, 2}}, {1, {0, 0, 0}, {6, 3, 7}, {1, 6, 18}}},
                                                {{6, 10, 7}, {6, 3, 7}, {6, 3, 7}});
    group.used_input_broadcast_offset_required = {false, false};
    ASSERT_TRUE(CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath({group}));
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(stage, {group}, "take_kernel");

    EXPECT_EQ(countOccurrences(source, " % "), 0u) << source;
    EXPECT_EQ(countOccurrences(source, " / "), 2u) << source;
    EXPECT_EQ(source.find("unsigned long long"), std::string::npos) << source;
    EXPECT_EQ(source.find("input_residual"), std::string::npos) << source;
    EXPECT_NE(source.find("in1[take_g0_o0_im2_0_q1 + take_g0_o0_im2_0_c1 * 6U + take_g0_o0_im2_0_c2 * 18U]"), std::string::npos) << source;
    EXPECT_NE(source.find("take_g0_o0_im2_0_q1 * 160U + (take_gather_idx_g0_o0_im2_0 << 4) + (take_g0_o0_im2_0_c2 << 1)"),
              std::string::npos)
        << source;
}

namespace {
// take_along_axis(values[6,10], idx[6,4], axis=1) with the given index dtype; FP32 values.
CompiledExecutionStage gatherStage(DataType index_dtype) {
    StageBuilder b;
    const uint32_t values = b.input("values", DataType::FP32);
    const uint32_t idx = b.input("idx", index_dtype);
    ExprNode take{};
    take.op = ExprOp::TAKE_ALONG_AXIS;
    take.lhs = values;
    take.rhs = idx;
    take.compute_dtype = DataType::FP32;
    take.output_dtype = DataType::FP32;
    take.reduction_axes = {1};
    return b.compiled(b.push(take));
}

SpecializedBroadcastGroup gatherGroup() {
    SpecializedBroadcastGroup group =
        makeGroup({6, 4}, {{0, {0, 0}, {6, 10}, {16, 1}}, {1, {0, 0}, {6, 4}, {4, 1}}}, {{6, 10}, {6, 4}, {6, 4}});
    group.used_input_broadcast_offset_required = {false, false};
    return group;
}
}  // namespace

TEST(FusedIndexMappingEmitter, OutOfRangeGatherIndexReadsZeroAndStaysInsideTheSource) {
    // An index at or past the gather-axis length (10) yields 0, as an out-of-vocabulary
    // embedding row does. Its gather coordinate is 0, so the load stays inside the source. With
    // UINT32 indices and 32-bit index math the bound check is one 32-bit compare.
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(gatherStage(DataType::UINT32), {gatherGroup()}, "take_kernel");
    EXPECT_NE(source.find("const bool take_in_range_g0_o0_im2_0 = take_gather_idx_g0_o0_im2_0_raw < 10U;"), std::string::npos)
        << source;
    EXPECT_NE(source.find("take_gather_idx_g0_o0_im2_0 = take_in_range_g0_o0_im2_0 ? take_gather_idx_g0_o0_im2_0_raw : 0U;"),
              std::string::npos)
        << source;
    EXPECT_NE(source.find("= take_in_range_g0_o0_im2_0 ? t0_"), std::string::npos) << source;
    EXPECT_NE(source.find(" : 0.0f;"), std::string::npos) << source;
    EXPECT_EQ(source.find("unsigned long long"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, Uint64GatherIndexIsBoundsCheckedBeforeNarrowing) {
    // 32-bit index math narrows the gathered index to unsigned int. A UINT64 index of 2^32 + 3
    // would wrap to 3, an in-range element, so the bound is checked on the full 64-bit value.
    const std::string source = CudaSourceEmitter::emitSpecializedBroadcast(gatherStage(DataType::UINT64), {gatherGroup()}, "take_kernel");
    ASSERT_TRUE(CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath({gatherGroup()}));
    EXPECT_NE(source.find("const unsigned long long take_gather_idx_g0_o0_im2_0_raw = "), std::string::npos) << source;
    EXPECT_NE(source.find("take_gather_idx_g0_o0_im2_0_raw < 10ULL;"), std::string::npos) << source;
    EXPECT_NE(source.find("? static_cast<unsigned int>(take_gather_idx_g0_o0_im2_0_raw) : 0U;"), std::string::npos) << source;
}

TEST(FusedIndexMappingEmitter, LogicalTransposeConsumerVectorizesLaneMathOnlyForHalfPrecisionCompute) {
    // x[35,67]^T + y[35,67]^T. The consumer's "vectorized" path evaluates two output lanes (rows
    // out_idx and out_idx + numRows, stored separately) with half2/bfloat162 arithmetic. FP32
    // compute (the default) has no packed arithmetic to gain and stays per lane.
    struct Case {
        DataType storage;
        DataType compute;
        uint32_t expected_vectorized;
    };
    const std::vector<Case> cases = {{DataType::FP16, DataType::FP32, 0u},
                                     {DataType::FP16, DataType::FP16, 1u},
                                     {DataType::BF16, DataType::FP32, 0u},
                                     {DataType::BF16, DataType::BF16, 1u},
                                     {DataType::FP8_E4M3, DataType::FP32, 0u},
                                     {DataType::FP8_E4M3, DataType::BF16, 1u}};
    for (const Case& c : cases) {
        StageBuilder b;
        const uint32_t x = b.input("x", c.storage);
        const uint32_t y = b.input("y", c.storage);
        const uint32_t xt = b.unary(ExprOp::TRANSPOSE, x, c.storage, c.storage);
        const uint32_t yt = b.unary(ExprOp::TRANSPOSE, y, c.storage, c.storage);
        const uint32_t sum = b.binary(ExprOp::ADD, xt, yt, c.compute, c.storage);
        const CompiledExecutionStage stage = b.compiled(sum);
        const SpecializedBroadcastGroup group = makeGroup(
            {67, 35}, {{0, {0, 0}, {35, 67}, {67, 1}}, {1, {0, 0}, {35, 67}, {67, 1}}}, {{35, 67}, {35, 67}, {67, 35}, {67, 35}, {67, 35}});
        ASSERT_TRUE(CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, {group}));
        EXPECT_EQ(CudaSourceEmitter::tiledLogicalTransposeConsumerVectorizedOutputCount(stage, {group}), c.expected_vectorized)
            << "storage " << static_cast<int>(c.storage) << " compute " << static_cast<int>(c.compute);
    }
}

TEST(FusedIndexMappingEmitter, LogicalTransposeConsumerMixedAndComputedFrontiersVectorizeWithFp16Compute) {
    const SpecializedBroadcastGroup dense_group = makeGroup(
        {65, 37}, {{0, {0, 0}, {37, 65}, {65, 1}}, {1, {0, 0}, {37, 65}, {65, 1}}}, {{37, 65}, {37, 65}, {65, 37}, {65, 37}, {65, 37}});
    for (const DataType compute : {DataType::FP32, DataType::FP16}) {
        // FP8 x and FP16 y frontiers, FP16 output.
        StageBuilder b;
        const uint32_t x = b.input("x", DataType::FP8_E4M3);
        const uint32_t y = b.input("y", DataType::FP16);
        const uint32_t xt = b.unary(ExprOp::TRANSPOSE, x, DataType::FP8_E4M3, DataType::FP8_E4M3);
        const uint32_t yt = b.unary(ExprOp::TRANSPOSE, y, DataType::FP16, DataType::FP16);
        const uint32_t sum = b.binary(ExprOp::ADD, xt, yt, compute, DataType::FP16);
        const CompiledExecutionStage stage = b.compiled(sum);
        ASSERT_TRUE(CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, {dense_group}));
        EXPECT_EQ(CudaSourceEmitter::tiledLogicalTransposeConsumerVectorizedOutputCount(stage, {dense_group}),
                  compute == DataType::FP16 ? 1u : 0u);
    }
    for (const DataType compute : {DataType::FP32, DataType::FP16}) {
        // Computed frontier: f = (x + pre_bias[67])^T, out = f + f.
        StageBuilder b;
        const uint32_t x = b.input("x", DataType::FP16);
        const uint32_t bias = b.input("pre_bias", DataType::FP16);
        const uint32_t pre = b.binary(ExprOp::ADD, x, bias, DataType::FP32, DataType::FP16);
        const uint32_t f = b.unary(ExprOp::TRANSPOSE, pre, DataType::FP16, DataType::FP16);
        const uint32_t out = b.binary(ExprOp::ADD, f, f, compute, DataType::FP16);
        const CompiledExecutionStage stage = b.compiled(out);
        const SpecializedBroadcastGroup group = makeGroup(
            {67, 35}, {{0, {0, 0}, {35, 67}, {67, 1}}, {1, {0, 0}, {67}, {1}}}, {{35, 67}, {67}, {35, 67}, {67, 35}, {67, 35}});
        ASSERT_TRUE(CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, {group}));
        EXPECT_EQ(CudaSourceEmitter::tiledLogicalTransposeConsumerVectorizedOutputCount(stage, {group}), compute == DataType::FP16 ? 1u : 0u);
    }
}
