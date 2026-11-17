// The reducer refuses the views it has no kernels for (Reducer::analyzeGeometry throws NotImplementedException), and
// Reducer::splitRefusedView says how a caller runs them in two parts. A rotated view (F: [O, R, I] read as [I, O]; G:
// [A, R, B, P] read as [B, A, P]) is the dense reduction of its bytes, then the dense copy of that output read rotated;
// a pitched view (H) is the dense copy of its rows, then the dense reduction of the copy, or the dense reduction of its
// rows read at their pitch, then the copy of the output's columns. All are checked bit for bit against the dense stamp
// of the same problem. A pitched view with enough outputs is read in place (Reducer::pitchedCompleteLaunch: RuntimeK
// PacketComplete at the view's row and slab pitches); every pitched shape is also run that way at each W and checked
// the same way.

#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "Utilities/Exceptions.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;

namespace {

// Values exact in every storage dtype (FP8 E5M2 has two mantissa bits), so the dense and view inputs hold the same
// numbers whatever the dtype.
const std::vector<float> kValues = {-2.0f, -1.5f, -1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f};

std::vector<float> pattern(uint64_t count, uint64_t seed) {
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) values[i] = kValues[(i * 7 + seed * 3 + i / 5) % kValues.size()];
    return values;
}

const std::vector<DataType> kDtypes = {DataType::FP32, DataType::FP16, DataType::BF16, DataType::FP8_E4M3,
                                       DataType::FP8_E5M2};

Reducer reduction(ReductionOp op, uint32_t axis) {
    return Reducer(op, axis, std::nullopt, 1.0f, ReductionTuningDepth::Quick);
}

std::vector<float> runAndCopy(const std::shared_ptr<StampedReducer>& stamped, Stream& stream) {
    stamped->run();
    stream.synchronize();
    return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
}

void expectSameBits(const std::vector<float>& got, const std::vector<float>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            ADD_FAILURE() << "element " << i << ": " << got[i] << " against " << want[i];
            return;
        }
    }
}

struct RotatedShape {
    uint64_t a, r, b, p;
};

// The view [r, b, a] (F, p == 1) or [r, b, a, p] (G) of a dense [a, r, b, p]: reducing axis 0 leaves [1, b, a, p].
StridedSpan rotatedView(const Tensor& storage, uint64_t offset, const RotatedShape& s) {
    if (s.p == 1) return StridedSpan(storage, offset, {s.r, s.b, s.a}, {s.b, 1, s.r * s.b});
    return StridedSpan(storage, offset, {s.r, s.b, s.a, s.p}, {s.b * s.p, s.p, s.r * s.b * s.p, 1});
}

// out[b, a, p] from the dense reduction's [a, b, p].
std::vector<float> rotate(const std::vector<float>& dense, const RotatedShape& s) {
    std::vector<float> out(dense.size());
    for (uint64_t a = 0; a < s.a; ++a)
        for (uint64_t b = 0; b < s.b; ++b)
            for (uint64_t e = 0; e < s.p; ++e) out[(b * s.a + a) * s.p + e] = dense[(a * s.b + b) * s.p + e];
    return out;
}

void expectRefused(ReductionOp op, uint32_t axis, const StridedSpan& view, Stream& stream) {
    EXPECT_THROW(static_cast<void>(reduction(op, axis).stamp(view, stream)), NotImplementedException);
}

void checkRotated(const RotatedShape& s, DataType dtype, ReductionOp op, Stream& stream) {
    SCOPED_TRACE("a " + std::to_string(s.a) + " r " + std::to_string(s.r) + " b " + std::to_string(s.b) + " p " +
                 std::to_string(s.p) + " dtype " + std::to_string(static_cast<int>(dtype)) + " op " +
                 std::to_string(static_cast<int>(op)));
    Tensor storage = makeGpuTensor(pattern(s.a * s.r * s.b * s.p, s.a + s.b), {s.a, s.r, s.b * s.p}, stream, dtype);
    const StridedSpan view = rotatedView(storage, 0, s);
    expectRefused(op, 0, view, stream);
    const std::optional<ViewSplit> split =
        Reducer::splitRefusedView(view.getDimensions(), view.getStridesElements(), {0});
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->kind, ViewSplit::Kind::RotatedOutput);
    EXPECT_EQ(split->dense_dimensions, (std::vector<uint64_t>{s.a, s.r, s.b * s.p}));
    EXPECT_EQ(split->dense_axes, std::vector<uint32_t>{1});
    EXPECT_EQ(split->a, s.a);
    EXPECT_EQ(split->b, s.b);
    EXPECT_EQ(split->p, s.p);

    const Tensor two_parts = runRefusedViewInTwoParts(view, op, {0}, stream);
    std::vector<uint64_t> keepdims = view.getDimensions();
    keepdims[0] = 1;
    EXPECT_EQ(two_parts.getDimensions(), keepdims);
    const auto dense = reduction(op, 1).stamp(StridedSpan(storage), stream);
    EXPECT_EQ(two_parts.getDataType(), dense->getOutputTensor().getDataType());
    expectSameBits(copyGpuTensorAsFloat(two_parts, stream), rotate(runAndCopy(dense, stream), s));
}

struct PitchedShape {
    uint64_t slabs, rows, inner, slab_pitch, row_pitch;
    [[nodiscard]] uint64_t span() const { return (slabs - 1) * slab_pitch + (rows - 1) * row_pitch + inner; }
};

// The view [slabs, rows, inner] (singleton extents left out) over storage element `offset`, and its reduced axis.
StridedSpan pitchedView(const Tensor& storage, uint64_t offset, const PitchedShape& s, uint32_t* axis) {
    std::vector<uint64_t> dims, strides;
    if (s.slabs > 1) {
        dims.push_back(s.slabs);
        strides.push_back(s.slab_pitch);
    }
    *axis = static_cast<uint32_t>(dims.size());
    dims.push_back(s.rows);
    strides.push_back(s.row_pitch);
    if (s.inner > 1) {
        dims.push_back(s.inner);
        strides.push_back(1);
    }
    return StridedSpan(storage, offset, dims, strides);
}

std::vector<float> packed(const std::vector<float>& storage, uint64_t offset, const PitchedShape& s) {
    std::vector<float> out;
    out.reserve(s.slabs * s.rows * s.inner);
    for (uint64_t o = 0; o < s.slabs; ++o)
        for (uint64_t r = 0; r < s.rows; ++r)
            for (uint64_t i = 0; i < s.inner; ++i)
                out.push_back(storage[offset + o * s.slab_pitch + r * s.row_pitch + i]);
    return out;
}

uint64_t elementBytes(DataType dtype) {
    return static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
}

// Whether splitRefusedView of the span reads the rows at their pitch (SlicedOutput): no slab gaps, the gap after the
// last row inside the storage (`padding` elements after the view), and fewer bytes than the packed copy
// (Layout/ViewSplit.cpp, slicedRowSplit).
bool expectSliced(const PitchedShape& s, uint64_t padding, uint64_t input_bytes, uint64_t output_bytes) {
    if (s.rows < 2 || s.row_pitch <= s.inner || (s.slabs > 1 && s.slab_pitch != s.rows * s.row_pitch)) return false;
    if (padding < s.row_pitch - s.inner) return false;
    const uint64_t rows = s.slabs * s.rows;
    const uint64_t sliced = rows * s.row_pitch * input_bytes + s.slabs * (s.row_pitch + 2 * s.inner) * output_bytes;
    const uint64_t packed = 3 * rows * s.inner * input_bytes + s.slabs * s.inner * output_bytes;
    return sliced < packed;
}

// `padding` storage elements after the view's last (a sliced read needs row_pitch - inner of them).
void checkPitched(const PitchedShape& s,
                  DataType dtype,
                  ReductionOp op,
                  uint64_t offset,
                  uint64_t padding,
                  Stream& stream) {
    SCOPED_TRACE("slabs " + std::to_string(s.slabs) + " rows " + std::to_string(s.rows) + " inner " +
                 std::to_string(s.inner) + " pitches " + std::to_string(s.slab_pitch) + " " +
                 std::to_string(s.row_pitch) + " offset " + std::to_string(offset) + " padding " +
                 std::to_string(padding) + " dtype " + std::to_string(static_cast<int>(dtype)) + " op " +
                 std::to_string(static_cast<int>(op)));
    const std::vector<float> values = pattern(offset + s.span() + padding, s.rows);
    Tensor storage = makeGpuTensor(values, {offset + s.span() + padding}, stream, dtype);
    uint32_t axis = 0;
    const StridedSpan view = pitchedView(storage, offset, s, &axis);
    const DataType output_dtype = reduction(op, axis).resolveOutputDataType(dtype);
    const bool in_place = Reducer::pitchedCompleteLaunch(view, {axis}, output_dtype).has_value();
    if (!in_place) expectRefused(op, axis, view, stream);
    const std::optional<ViewSplit> split =
        Reducer::splitRefusedView(view.getDimensions(), view.getStridesElements(), {axis});
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->kind, ViewSplit::Kind::PackedInput);  // the storage is not known: packed
    EXPECT_EQ(split->dense_dimensions, view.getDimensions());
    EXPECT_EQ(split->dense_axes, std::vector<uint32_t>{axis});

    const bool sliced = expectSliced(s, padding, elementBytes(dtype), elementBytes(output_dtype));
    EXPECT_EQ(Reducer::splitRefusedView(view, {axis}, output_dtype).has_value(), !in_place);
    const std::optional<ViewSplit> span_split =
        Reducer::splitRefusedView(view, {axis}, output_dtype, false);
    ASSERT_TRUE(span_split.has_value());
    ASSERT_EQ(span_split->kind,
              sliced ? ViewSplit::Kind::SlicedOutput : ViewSplit::Kind::PackedInput);
    if (sliced) {
        EXPECT_EQ(span_split->dense_dimensions, (std::vector<uint64_t>{s.slabs, s.rows, s.row_pitch}));
        EXPECT_EQ(span_split->dense_axes, std::vector<uint32_t>{1});
        EXPECT_EQ(span_split->buffer_dimensions, (std::vector<uint64_t>{s.slabs, s.row_pitch}));
        EXPECT_EQ(span_split->copy_dimensions, (std::vector<uint64_t>{s.slabs, s.inner}));
        EXPECT_EQ(span_split->copy_strides, (std::vector<uint64_t>{s.row_pitch, 1}));
        // One element fewer of storage and the last row's gap is not readable: packed.
        const uint64_t short_elements = offset + s.span() + s.row_pitch - s.inner - 1;
        Tensor short_storage = makeGpuTensor(
            std::vector<float>(values.begin(), values.begin() + short_elements), {short_elements}, stream, dtype);
        uint32_t short_axis = 0;
        const StridedSpan short_view = pitchedView(short_storage, offset, s, &short_axis);
        const std::optional<ViewSplit> short_split =
            Reducer::splitRefusedView(short_view, {short_axis}, output_dtype, false);
        ASSERT_TRUE(short_split.has_value());
        EXPECT_EQ(short_split->kind, ViewSplit::Kind::PackedInput);
    }

    // The sums, minima and L1 norms of these values are exact in FP32 in any order, so either split gives the bits of
    // the dense reduction of the packed rows.
    Tensor dense_input = makeGpuTensor(packed(values, offset, s), view.getDimensions(), stream, dtype);
    const auto dense = reduction(op, axis).stamp(StridedSpan(dense_input), stream);
    std::optional<ViewSplit::Kind> ran;
    const Tensor two_parts = runRefusedViewInTwoParts(view, op, {axis}, stream, std::nullopt, &ran);
    ASSERT_TRUE(ran.has_value());
    EXPECT_EQ(*ran, span_split->kind);
    EXPECT_EQ(two_parts.getDimensions(), dense->getOutputTensor().getDimensions());
    const std::vector<float> want = runAndCopy(dense, stream);
    expectSameBits(copyGpuTensorAsFloat(two_parts, stream), want);

    // The rows read in place at their pitches, at each W, and as stamp() runs it when it does.
    for (const uint32_t warps : {1U, 2U, 4U, 8U}) {
        SCOPED_TRACE("W " + std::to_string(warps));
        const auto pitched = reduction(op, axis).stampPitched(view, warps, stream);
        ASSERT_TRUE(pitched->getPitchedCompleteLaunch().has_value());
        EXPECT_EQ(pitched->getPitchedCompleteLaunch()->warps_per_output, warps);
        EXPECT_EQ(pitched->getOutputTensor().getDimensions(), dense->getOutputTensor().getDimensions());
        expectSameBits(runAndCopy(pitched, stream), want);
    }
    if (in_place) {
        const auto stamped = reduction(op, axis).stamp(view, stream);
        ASSERT_TRUE(stamped->getPitchedCompleteLaunch().has_value());
        expectSameBits(runAndCopy(stamped, stream), want);
    }
}

}  // namespace

TEST(ReducerViewSplit, RotatedViewsAreTheDenseReductionOfTheirBytesThenTheCopy) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<RotatedShape> shapes = {
        {3, 5, 7, 1},     // F
        {2, 4097, 3, 1},  // F, few outputs and a long R
        {300, 2, 33, 1},  // F, many outputs and R = 2
        {40, 9, 600, 1},  // F, wide rows
        {2, 3, 5, 7},     // G
        {3, 2, 2, 37},    // G, a payload wider than a sector in every dtype
        {33, 3, 35, 3},   // G, partial tiles on both sides
    };
    for (const RotatedShape& shape : shapes)
        for (const DataType dtype : kDtypes)
            for (const ReductionOp op : {ReductionOp::Sum, ReductionOp::Max, ReductionOp::L2Norm})
                checkRotated(shape, dtype, op, stream);
    for (const DataType dtype : {DataType::FP32, DataType::FP8_E4M3})
        for (const ReductionOp op : {ReductionOp::Sum, ReductionOp::Mean, ReductionOp::Product,
                                     ReductionOp::Min, ReductionOp::Max, ReductionOp::L1Norm,
                                     ReductionOp::L2Norm, ReductionOp::SumSquares})
            checkRotated({5, 17, 4, 9}, dtype, op, stream);  // G: FP32 rows of 36 bytes, FP8 rows of 9
}

// Packed: the dense copy of the rows, then the dense reduction. Sliced when the slabs have no gaps and the storage
// holds the last row's gap: the dense reduction of the rows read at their pitch, then the copy of the output's columns.
// Each shape runs without padding (the last row's gap outside the storage: packed) and with it.
TEST(ReducerViewSplit, PitchedViewsArePackedOrReadAtTheirRowPitch) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<PitchedShape> shapes = {
        {2, 2, 3, 18, 3},           // an einsum diagonal: slab gaps
        {1, 513, 5, 0, 7},          // a slice of the last axis: row gaps, no outer group
        {3, 4, 5, 40, 7},           // both
        {4, 1000, 6, 6100, 6},      // slab gaps, long rows
        {1, 100000, 2, 0, 3},       // few outputs, a long R
        {2, 7, 1, 30, 4},           // no inner extent
        {6, 3, 33, 200, 40},        // rows wider than a sector
        {5, 9, 6, 72, 8},           // row gaps only, slabs at R x row pitch
        {7, 2, 300, 620, 310},      // R = 2: sliced for 4-byte input, packed for narrower (more output than input)
        {200, 50, 33, 1700, 34},    // many outputs
        {1, 6, 40000, 0, 41000},    // few long rows, gaps of a few KB
    };
    for (const PitchedShape& shape : shapes)
        for (const uint64_t padding : {uint64_t{0}, shape.row_pitch})
            for (const DataType dtype : kDtypes)
                for (const ReductionOp op : {ReductionOp::Sum, ReductionOp::Min, ReductionOp::L1Norm})
                    checkPitched(shape, dtype, op, 0, padding, stream);
}

TEST(ReducerViewSplit, ViewsAtMisalignedBases) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // An F view one FP16 element into its storage.
    const RotatedShape s = {6, 70, 9, 1};
    const std::vector<float> values = pattern(1 + s.a * s.r * s.b, 4);
    Tensor storage = makeGpuTensor(values, {1 + s.a * s.r * s.b}, stream, DataType::FP16);
    const StridedSpan view = rotatedView(storage, 1, s);
    expectRefused(ReductionOp::Sum, 0, view, stream);
    const auto dense =
        reduction(ReductionOp::Sum, 1).stamp(StridedSpan(storage, 1, {s.a, s.r, s.b}, {s.r * s.b, s.b, 1}), stream);
    expectSameBits(copyGpuTensorAsFloat(runRefusedViewInTwoParts(view, ReductionOp::Sum, {0}, stream), stream),
                   rotate(runAndCopy(dense, stream), s));

    // H views 1 and 3 bytes into FP8 storage, and 2 bytes into FP16.
    checkPitched({3, 4, 5, 40, 7}, DataType::FP8_E4M3, ReductionOp::Sum, 1, 0, stream);
    checkPitched({2, 300, 6, 2000, 6}, DataType::FP8_E5M2, ReductionOp::Max, 3, 0, stream);
    checkPitched({1, 513, 5, 0, 7}, DataType::FP16, ReductionOp::L2Norm, 1, 0, stream);
    // Sliced reads at misaligned bases (the dense [O, R, Sr] from an odd element).
    checkPitched({1, 513, 5, 0, 7}, DataType::FP16, ReductionOp::L1Norm, 1, 7, stream);
    checkPitched({5, 9, 6, 72, 8}, DataType::FP8_E4M3, ReductionOp::Sum, 3, 8, stream);
    checkPitched({5, 9, 6, 72, 8}, DataType::FP32, ReductionOp::Min, 1, 8, stream);
}

// A pitched view with K tiles of 256 bytes or more whose launch keeps enough bytes in flight (4 KiB per SM against a
// sliced split, 2 KiB against a packed one) is read in place: one PacketComplete launch at its pitches, W the most
// warps per item that fit one wave of resident warps, with at least four rows per warp. It has no split. One with
// narrow rows or few outputs is refused and split. The K tiles are balanced under the dtype's cap.
TEST(ReducerViewSplit, PitchedViewsThatKeepEnoughInFlightAreReadInPlace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    int sm_count = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
    const uint64_t resident_warps = 32 * static_cast<uint64_t>(sm_count);

    // [O, R, I] = [8192, 4, 64] FP32 at row pitch 80, slab pitch 400 (row and slab gaps: packed otherwise): 8192
    // tiles of 256 bytes, W 1 (four rows), min(8192, resident warps) x 256 bytes in flight.
    const PitchedShape many = {8192, 4, 64, 400, 80};
    const std::vector<float> values = pattern(many.span(), 2);
    Tensor storage = makeGpuTensor(values, {many.span()}, stream, DataType::FP32);
    uint32_t axis = 0;
    const StridedSpan view = pitchedView(storage, 0, many, &axis);
    const std::optional<PitchedCompleteLaunch> launch =
        Reducer::pitchedCompleteLaunch(view, {axis}, DataType::FP32);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->tile_components, 64U);
    EXPECT_EQ(launch->warps_per_output, 1U);
    EXPECT_TRUE(Reducer::readsSpanInPlace(view, {axis}, DataType::FP32));
    EXPECT_FALSE(Reducer::splitRefusedView(view, {axis}, DataType::FP32).has_value());
    EXPECT_TRUE(Reducer::splitRefusedView(view, {axis}, DataType::FP32, false).has_value());
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry(view.getDimensions(), view.getStridesElements(),
                                                            {axis})),
                 NotImplementedException);  // the layout alone is still refused
    const auto stamped = reduction(ReductionOp::Sum, axis).stamp(view, stream);
    EXPECT_EQ(stamped->getModernRKStrategyChain(), "pitched_complete:w1_t64");
    EXPECT_EQ(stamped->getInput().getMemPtr<void>(), storage.getMemPtr<void>());  // no copy
    Tensor dense_input = makeGpuTensor(packed(values, 0, many), view.getDimensions(), stream);
    expectSameBits(runAndCopy(stamped, stream),
                   runAndCopy(reduction(ReductionOp::Sum, axis).stamp(StridedSpan(dense_input), stream), stream));

    // W: [512, 64, 124] FP32 (one tile of 496 bytes per slab) takes the most warps per item that fit one wave.
    const PitchedShape deep = {512, 64, 124, 64 * 130, 130};
    Tensor deep_storage = makeGpuTensor(pattern(deep.span(), 6), {deep.span()}, stream, DataType::FP32);
    const StridedSpan deep_view = pitchedView(deep_storage, 0, deep, &axis);
    uint32_t want_warps = 1;
    while (want_warps < 8 && 512ULL * want_warps * 2 <= resident_warps) want_warps *= 2;
    if (const auto deep_launch = Reducer::pitchedCompleteLaunch(deep_view, {axis}, DataType::FP32);
        deep_launch.has_value()) {
        EXPECT_EQ(deep_launch->warps_per_output, want_warps);
    } else {
        EXPECT_LT(std::min<uint64_t>(512ULL * want_warps, resident_warps) * 496, 4096ULL * sm_count);
    }

    // At least four rows per warp: R = 7 is W 1 however many items fit.
    const PitchedShape seven = {2048, 7, 124, 7 * 130, 130};
    Tensor seven_storage = makeGpuTensor(pattern(seven.span(), 7), {seven.span()}, stream, DataType::FP32);
    const StridedSpan seven_view = pitchedView(seven_storage, 0, seven, &axis);
    if (const auto seven_launch = Reducer::pitchedCompleteLaunch(seven_view, {axis}, DataType::FP32);
        seven_launch.has_value()) {
        EXPECT_EQ(seven_launch->warps_per_output, 1U);
    }

    // Rows of 20 bytes: as many outputs, but tiles under 256 bytes. Refused and split.
    const PitchedShape narrow = {65536, 4, 5, 40, 8};
    Tensor narrow_storage = makeGpuTensor(pattern(narrow.span(), 3), {narrow.span()}, stream, DataType::FP32);
    const StridedSpan narrow_view = pitchedView(narrow_storage, 0, narrow, &axis);
    EXPECT_FALSE(Reducer::pitchedCompleteLaunch(narrow_view, {axis}, DataType::FP32).has_value());
    EXPECT_TRUE(Reducer::pitchedCompleteLaunch(narrow_view, {axis}, DataType::FP32, 2U).has_value());  // forced
    EXPECT_TRUE(Reducer::splitRefusedView(narrow_view, {axis}, DataType::FP32).has_value());
    expectRefused(ReductionOp::Sum, axis, narrow_view, stream);

    // Balanced tiles: 300 FP16 components (cap 248) are two tiles of 150.
    const PitchedShape wide = {1024, 3, 300, 2000, 600};
    Tensor wide_storage = makeGpuTensor(pattern(wide.span(), 3), {wide.span()}, stream, DataType::FP16);
    const StridedSpan wide_view = pitchedView(wide_storage, 0, wide, &axis);
    const std::optional<PitchedCompleteLaunch> wide_launch =
        Reducer::pitchedCompleteLaunch(wide_view, {axis}, DataType::FP32, 4U);
    ASSERT_TRUE(wide_launch.has_value());
    EXPECT_EQ(wide_launch->tile_components, 150U);
    EXPECT_EQ(wide_launch->warps_per_output, 4U);

    // One output row: refused and split, though PacketComplete reads it when forced.
    const PitchedShape one = {1, 513, 64, 0, 70};
    Tensor one_storage = makeGpuTensor(pattern(one.span(), 5), {one.span()}, stream, DataType::FP32);
    const StridedSpan one_view = pitchedView(one_storage, 0, one, &axis);
    EXPECT_FALSE(Reducer::pitchedCompleteLaunch(one_view, {axis}, DataType::FP32).has_value());
    EXPECT_FALSE(Reducer::readsSpanInPlace(one_view, {axis}, DataType::FP32));
    EXPECT_TRUE(Reducer::splitRefusedView(one_view, {axis}, DataType::FP32).has_value());
    expectRefused(ReductionOp::Sum, axis, one_view, stream);
    EXPECT_NO_THROW(static_cast<void>(reduction(ReductionOp::Sum, axis).stampPitched(one_view, 2, stream)));

    // Not pitched views: a rotated one, and a dense tensor. W is 1, 2, 4 or 8.
    const StridedSpan rotated = rotatedView(storage, 0, {3, 5, 7, 1});
    EXPECT_FALSE(Reducer::pitchedCompleteLaunch(rotated, {0}, DataType::FP32).has_value());
    EXPECT_THROW(static_cast<void>(reduction(ReductionOp::Sum, 0).stampPitched(rotated, 1, stream)),
                 NotImplementedException);
    EXPECT_FALSE(Reducer::pitchedCompleteLaunch(StridedSpan(dense_input), {1}, DataType::FP32).has_value());
    EXPECT_THROW(static_cast<void>(Reducer::pitchedCompleteLaunch(view, {axis}, DataType::FP32, 3U)),
                 std::invalid_argument);
}

TEST(ReducerViewSplit, ViewsTheReducerRunsAreNotSplit) {
    const auto expectNotSplit = [](const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides,
                                   const std::vector<uint32_t>& axes) {
        EXPECT_NO_THROW(static_cast<void>(Reducer::analyzeGeometry(dims, strides, axes)));
        EXPECT_FALSE(Reducer::splitRefusedView(dims, strides, axes).has_value());
    };
    expectNotSplit({4, 5, 6}, {30, 6, 1}, {1});  // dense
    expectNotSplit({5, 4, 6}, {6, 30, 1}, {0});  // D: [a, r, b] read as [r, a, b], the output [a, b] in order
    expectNotSplit({1000}, {3}, {0});            // a rank-1 strided view (StridedR)
}
