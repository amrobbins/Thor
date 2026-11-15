// 1365: plans/Thor_View_Reduction_Plan_2026-11-08.md phase 1. VIEW-DIRECT-1 (every axis reduced over a compact physical
// permutation) and VIEW-DIRECT-2A (reduced axes physically trailing, retained axes in output order) run the retained
// dense planner on the view's bytes (StridedSpan::asDense). Each view stamp must give the bits of the dense tensor of
// the same bytes reduced directly, select the same passes, and keep the view's classification as its path.
// 1374: B, dense rows and dense ARG past CUB's old `int` fixed-segment size (deferred from phase 1).
// 1376 (phase 5): D (natural-output permutation-aware tiled views) and E (split physical reductions) the same way; a
// permuted output kept the tiled kernels (1390: the reducer refuses it).
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

const std::vector<CubReductionOp> kOps = {CubReductionOp::Sum,    CubReductionOp::Min,    CubReductionOp::Max,
                                          CubReductionOp::Product, CubReductionOp::Mean,   CubReductionOp::L1Norm,
                                          CubReductionOp::L2Norm,  CubReductionOp::SumSquares};

std::vector<DataType> valueDtypes() {
    return {DataType::FP32, DataType::FP16, DataType::BF16
#if THOR_CUB_ENABLE_FP8_TYPES
            ,
            DataType::FP8_E4M3, DataType::FP8_E5M2
#endif
    };
}

// Values in [0.75, 1.25] (with signs), so a product over a few thousand elements stays finite.
std::vector<float> values(size_t count, uint32_t seed) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t h = static_cast<uint32_t>(i) * 2654435761u + seed * 40503u;
        const float magnitude = 0.75f + static_cast<float>((h >> 8) % 9) / 16.0f;
        out[i] = ((h >> 20) % 7 == 0) ? -magnitude : magnitude;
    }
    return out;
}

std::vector<uint64_t> denseStridesOf(const std::vector<uint64_t>& dims) { return StridedSpan::denseStrides(dims); }

// A view of `physical` (a dense tensor's dims, starting `offset` elements into its storage) under the axis order
// `order`: view axis v is physical axis order[v]. `singleton` inserts an extent-one axis with a huge stride at that
// view position (nothing when it is past the rank).
struct ViewCase {
    const char* name;
    std::vector<uint64_t> physical;
    std::vector<uint32_t> order;
    uint64_t offset;
    size_t singleton;  // view position of an inserted extent-one axis; >= rank for none
    // The reduced physical axes: all of them (A), a suffix (B), a block with an inner block (D), several runs (E).
    std::vector<uint32_t> physical_axes;
    CubReductionPath expected_path;
};

struct BuiltView {
    StridedSpan view;
    StridedSpan dense;
    std::vector<uint32_t> view_axes;
};

BuiltView build(const ViewCase& c, const Tensor& storage) {
    const std::vector<uint64_t> dense_strides = denseStridesOf(c.physical);
    std::vector<uint64_t> dims, strides;
    std::vector<uint32_t> view_axes;
    for (size_t v = 0; v <= c.order.size(); ++v) {
        if (v == c.singleton) {
            // Reduced with the rest in a full reduction (A); retained (an extent-one output axis) otherwise (B).
            if (c.physical_axes.size() == c.physical.size()) view_axes.push_back(static_cast<uint32_t>(dims.size()));
            dims.push_back(1);
            strides.push_back(uint64_t{1} << 40);
        }
        if (v == c.order.size()) break;
        const uint32_t p = c.order[v];
        if (std::find(c.physical_axes.begin(), c.physical_axes.end(), p) != c.physical_axes.end()) {
            view_axes.push_back(static_cast<uint32_t>(dims.size()));
        }
        dims.push_back(c.physical[p]);
        strides.push_back(dense_strides[p]);
    }
    return {StridedSpan(storage, c.offset, dims, strides), StridedSpan(storage, c.offset, c.physical, dense_strides),
            view_axes};
}

const std::vector<ViewCase>& cases() {
    static const std::vector<ViewCase> all = {
        // A: a rank-3 permutation, every axis reduced.
        {"A rank 3", {6, 40, 7}, {2, 0, 1}, 0, 99, {0, 1, 2}, CubReductionPath::DeviceTransformReduce},
        // A: a rank-4 permutation with an extent-one axis of huge stride, 3 elements into its storage (not 16-byte
        // aligned for any dtype here).
        {"A rank 4 singleton offset", {3, 5, 11, 4}, {3, 1, 0, 2}, 3, 2, {0, 1, 2, 3},
         CubReductionPath::DeviceTransformReduce},
        // B: [O1, O2, R1, R2] read as [O1, O2, R2, R1].
        {"B rank 4", {5, 3, 17, 9}, {0, 1, 3, 2}, 0, 99, {2, 3}, CubReductionPath::ContiguousFixedSegment},
        // B: a rank-3 trailing permutation with an extent-one axis, 5 elements in.
        {"B rank 3 singleton offset", {12, 8, 13}, {0, 2, 1}, 5, 1, {1, 2}, CubReductionPath::ContiguousFixedSegment},
        // 1376, D: physical [O, R, I] read as [R, O, I] (the classification stays the natural-output tiled candidate).
        {"D rank 3", {6, 50, 9}, {1, 0, 2}, 0, 99, {1}, CubReductionPath::TiledFixedSegment},
        // D without an outer block: [R1, R2, I] read as [I, 1, R2, R1], 3 elements in.
        {"D rank 3 no outer singleton offset", {12, 30, 5}, {2, 1, 0}, 3, 1, {0, 1},
         CubReductionPath::TiledFixedSegment},
        // D: [O, R1, R2, I] read as [R2, O, I, R1].
        {"D rank 4", {4, 9, 13, 6}, {2, 0, 3, 1}, 0, 99, {1, 2}, CubReductionPath::TiledFixedSegment},
        // 1376, E: [R1, O1, R2, O2] read as [O1, R1, O2, R2] (threw before 1376).
        {"E rank 4", {7, 6, 9, 5}, {1, 0, 3, 2}, 0, 99, {0, 2}, CubReductionPath::ComposedDense},
        // E: [O1, R1, O2, R2, O3] read as [R2, O1, 1, O2, R1, O3], 5 elements in.
        {"E rank 5 singleton offset", {4, 11, 3, 8, 5}, {3, 0, 2, 1, 4}, 5, 2, {1, 3}, CubReductionPath::ComposedDense},
    };
    return all;
}

uint64_t storageElements(const ViewCase& c) {
    uint64_t elements = c.offset;
    uint64_t product = 1;
    for (uint64_t d : c.physical) product *= d;
    return elements + product;
}

}  // namespace

TEST(CubReductionViewPhysical, ViewsGiveTheBitsAndPassesOfTheDenseTensorOfTheirBytes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const ViewCase& c : cases()) {
        for (DataType dtype : valueDtypes()) {
            const uint64_t count = storageElements(c);
            const Tensor storage = makeGpuTensor(values(count, static_cast<uint32_t>(count)), {count}, stream, dtype);
            const BuiltView built = build(c, storage);
            ASSERT_FALSE(built.view.isDenseContiguous());
            ASSERT_TRUE(built.dense.isDenseContiguous());
            for (CubReductionOp op : kOps) {
                SCOPED_TRACE(std::string(c.name) + " dtype " + std::to_string(static_cast<int>(dtype)) + " op " +
                             std::to_string(static_cast<int>(op)));
                auto view_stamp = CubReduction(op, built.view_axes, DataType::FP32).stamp(built.view, stream);
                auto dense_stamp = CubReduction(op, c.physical_axes, DataType::FP32).stamp(built.dense, stream);
                EXPECT_EQ(view_stamp->getPath(), c.expected_path);
                ASSERT_TRUE(view_stamp->usesDenseExecutablePlan());
                ASSERT_TRUE(dense_stamp->usesDenseExecutablePlan());
                EXPECT_EQ(view_stamp->getModernRKStrategyChain(),
                          "view_physical:" + dense_stamp->getModernRKStrategyChain());
                EXPECT_EQ(view_stamp->getDenseExecutablePassCount(), dense_stamp->getDenseExecutablePassCount());
                EXPECT_EQ(view_stamp->getOutputTensor().getTotalNumElements(),
                          dense_stamp->getOutputTensor().getTotalNumElements());
                view_stamp->run();
                dense_stamp->run();
                const std::vector<float> view_out = copyGpuTensorAsFloat(view_stamp->getOutputTensor(), stream);
                const std::vector<float> dense_out = copyGpuTensorAsFloat(dense_stamp->getOutputTensor(), stream);
                ASSERT_EQ(view_out.size(), dense_out.size());
                for (size_t i = 0; i < view_out.size(); ++i) {
                    ASSERT_EQ(memcmp(&view_out[i], &dense_out[i], sizeof(float)), 0)
                        << "output " << i << ": view " << view_out[i] << " dense " << dense_out[i];
                }
            }
        }
    }
}

// 1376: a permuted output order is not the dense tensor's output order: [O, R, I] read as [I, R, O] over R gives the
// [I, O] rotation of the dense [O, I]. 1387 ran it as the dense plan of its bytes then a rotation pass inside the
// reducer; 1390: the reducer refuses it, and the caller's two parts give the rotated bits (CubReductionViewSplitTest).
TEST(CubReductionViewPhysical, PermutedOutputViewsAreRefusedAndRunInTwoParts) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t o = 4, r = 5, i = 6;
    const Tensor storage = makeGpuTensor(values(o * r * i, 3), {o * r * i}, stream);
    const StridedSpan view(storage, 0, {i, r, o}, {1, i, r * i});
    const StridedSpan dense(storage, 0, {o, r, i}, {r * i, i, 1});
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(view, stream)),
                 NotImplementedException);
    auto dense_stamp = CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(dense, stream);
    dense_stamp->run();
    const std::vector<float> dense_out = copyGpuTensorAsFloat(dense_stamp->getOutputTensor(), stream);  // [O, I]
    std::vector<float> rotated(o * i);
    for (uint64_t a = 0; a < o; ++a) {
        for (uint64_t c = 0; c < i; ++c) rotated[c * o + a] = dense_out[a * i + c];
    }
    expectFloatVectorNear(executeRefusedViewInTwoParts(view, CubReductionOp::Sum, {1}, stream), rotated);
}

TEST(CubReductionViewPhysical, PreallocatedOutputsKeepTheirLogicalAndKeepdimsShapes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const ViewCase& b = cases()[2];  // B rank 4: logical output [5, 3]
    const Tensor storage = makeGpuTensor(values(storageElements(b), 7), {storageElements(b)}, stream);
    const BuiltView built = build(b, storage);
    const CubReduction reduction(CubReductionOp::Sum, built.view_axes, DataType::FP32);
    auto dense_stamp = CubReduction(CubReductionOp::Sum, b.physical_axes, DataType::FP32).stamp(built.dense, stream);
    dense_stamp->run();
    const std::vector<float> expected = copyGpuTensorAsFloat(dense_stamp->getOutputTensor(), stream);
    for (const std::vector<uint64_t>& shape : {std::vector<uint64_t>{5, 3}, std::vector<uint64_t>{5, 3, 1, 1},
                                                std::vector<uint64_t>{1, 5, 1, 3}}) {
        SCOPED_TRACE(shape.size());
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, shape));
        auto stamped = reduction.stamp(built.view, output, stream);
        ASSERT_TRUE(stamped->usesDenseExecutablePlan());
        stamped->run();
        EXPECT_EQ(stamped->getOutputTensor().getDimensions(), shape);
        EXPECT_EQ(copyGpuTensorAsFloat(output, stream), expected);
    }

    // A: a keepdims [1, 1, 1] output of a full reduction.
    const ViewCase& a = cases()[0];
    const Tensor a_storage = makeGpuTensor(values(storageElements(a), 9), {storageElements(a)}, stream);
    const BuiltView a_built = build(a, a_storage);
    Tensor keepdims(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 1, 1}));
    auto a_stamp =
        CubReduction(CubReductionOp::Sum, a_built.view_axes, DataType::FP32).stamp(a_built.view, keepdims, stream);
    auto a_dense = CubReduction(CubReductionOp::Sum, a.physical_axes, DataType::FP32).stamp(a_built.dense, stream);
    a_stamp->run();
    a_dense->run();
    EXPECT_EQ(copyGpuTensorAsFloat(keepdims, stream), copyGpuTensorAsFloat(a_dense->getOutputTensor(), stream));
}

TEST(CubReductionViewPhysical, FewLongAndManyTinySegmentsSumExactly) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Shape {
        const char* name;
        std::vector<uint64_t> physical;  // [outer, R1, R2], read as [outer, R2, R1]
    };
    // Few long segments (one block per segment under the old CUB path) and many tiny ones.
    for (const Shape& shape : {Shape{"4 x 2^20", {4, 1024, 1024}}, Shape{"65536 x 6", {65536, 2, 3}},
                               Shape{"3 x 3 x 5", {3, 3, 5}}}) {
        SCOPED_TRACE(shape.name);
        const uint64_t outer = shape.physical[0];
        const uint64_t r = shape.physical[1] * shape.physical[2];
        // Small integers, so every partial sum is exact in FP32 (at most 7 x 2^20 < 2^24).
        std::vector<float> host(outer * r);
        for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>((i * 5 + i / 7) % 8);
        const Tensor storage = makeGpuTensor(host, {outer * r}, stream);
        const StridedSpan view(storage, 0, {outer, shape.physical[2], shape.physical[1]},
                               {r, 1, shape.physical[2]});
        auto stamped = CubReduction(CubReductionOp::Sum, {1, 2}, DataType::FP32).stamp(view, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::ContiguousFixedSegment);
        ASSERT_TRUE(stamped->usesDenseExecutablePlan());
        stamped->run();
        std::vector<float> expected(outer, 0.0f);
        for (uint64_t o = 0; o < outer; ++o) {
            for (uint64_t k = 0; k < r; ++k) expected[o] += host[o * r + k];
        }
        EXPECT_EQ(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReductionViewPhysical, BenchmarkRunsReadTheReplacementViewAsDense) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const ViewCase& b = cases()[2];
    const uint64_t count = storageElements(b);
    const Tensor storage = makeGpuTensor(values(count, 1), {count}, stream);
    const Tensor other = makeGpuTensor(values(count, 2), {count}, stream);
    const BuiltView bound = build(b, storage);
    const BuiltView replacement = build(b, other);
    auto stamped = CubReduction(CubReductionOp::Sum, bound.view_axes, DataType::FP32).stamp(bound.view, stream);
    auto other_dense =
        CubReduction(CubReductionOp::Sum, b.physical_axes, DataType::FP32).stamp(replacement.dense, stream);
    other_dense->run();
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, stamped->getOutputTensor().getDimensions()));
    stamped->runOnTensorsForBenchmark(replacement.view, output, stream);
    EXPECT_EQ(copyGpuTensorAsFloat(output, stream), copyGpuTensorAsFloat(other_dense->getOutputTensor(), stream));
    // A dense replacement, or a view of other strides, is refused.
    EXPECT_THROW(stamped->runOnTensorsForBenchmark(replacement.dense, output, stream), std::invalid_argument);
}

// ---- 1374: rows past CUB's old `int` fixed-segment size. Two FP8 E4M3 rows of R = 65537 x 32771 = 2^31 + 229379
// elements (4.3 GB), zero but for a few planted values, three of them at or past row-local index 2^31; R is odd, so row
// 1 starts unaligned. Every SUM, MIN, MAX, PRODUCT, L1 and sum-of-squares result is exact; MEAN and L2 are one
// rounding from the exact value.

namespace {

constexpr uint64_t kLongA = 65537;
constexpr uint64_t kLongB = 32771;
constexpr uint64_t kLongR = kLongA * kLongB;
constexpr uint64_t kLongRows = 2;
constexpr uint64_t kLongBytes = kLongRows * kLongR;  // FP8: one byte per element
// The input, the stamp autotuner's representative input of the same size (a selection-cache miss allocates one), and
// 1 GiB for partials and the rest.
constexpr uint64_t kLongPeakBytes = 2 * kLongBytes + (uint64_t{1} << 30);
static_assert(kLongR > static_cast<uint64_t>(std::numeric_limits<int>::max()));
static_assert(kLongR < static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));  // a UINT32 index holds it

struct Planted {
    uint64_t row;
    uint64_t index;  // row-local
    uint8_t bits;    // FP8 E4M3
    float value;
};

const std::vector<Planted>& planted() {
    static const std::vector<Planted> all = {
        {0, 0, 0x38, 1.0f},  // +1
        {0, (uint64_t{1} << 31) + 5, 0xB8, -1.0f},  // -1, row 0's minimum
        {0, kLongR - 1, 0x40, 2.0f},  // +2 in the last element, row 0's maximum
        {1, 7, 0xC4, -3.0f},  // -3, row 1's minimum
        {1, uint64_t{1} << 31, 0x48, 4.0f},  // +4 at exactly 2^31, row 1's maximum
    };
    return all;
}

bool haveFreeDeviceBytes(uint64_t bytes) {
    size_t free_bytes = 0, total_bytes = 0;
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes >= bytes;
}

void fillLongRows(Tensor& storage, Stream& stream) {
    uint8_t* const base = static_cast<uint8_t*>(storage.getMemPtr<void>());
    ASSERT_EQ(cudaMemsetAsync(base, 0, kLongBytes, stream.getStream()), cudaSuccess);
    for (const Planted& p : planted()) {
        ASSERT_EQ(cudaMemcpyAsync(base + p.row * kLongR + p.index, &p.bits, 1, cudaMemcpyHostToDevice,
                                  stream.getStream()),
                  cudaSuccess);
    }
    stream.synchronize();
}

// The exact result in FP64 (the other elements are +0).
double longRowExpected(CubReductionOp op, uint64_t row) {
    double sum = 0, l1 = 0, squares = 0, lo = 0, hi = 0;
    for (const Planted& p : planted()) {
        if (p.row != row) continue;
        sum += p.value;
        l1 += std::fabs(p.value);
        squares += static_cast<double>(p.value) * p.value;
        lo = std::min(lo, static_cast<double>(p.value));
        hi = std::max(hi, static_cast<double>(p.value));
    }
    switch (op) {
        case CubReductionOp::Sum: return sum;
        case CubReductionOp::Min: return lo;
        case CubReductionOp::Max: return hi;
        case CubReductionOp::Product: return 0.0;
        case CubReductionOp::Mean: return sum / static_cast<double>(kLongR);
        case CubReductionOp::L1Norm: return l1;
        case CubReductionOp::L2Norm: return std::sqrt(squares);
        case CubReductionOp::SumSquares: return squares;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

TEST(CubReductionViewPhysical, RowsPastTheCubIntLimitReduceExactly) {
    REQUIRE_CUDA_DEVICE();
#if !THOR_CUB_ENABLE_FP8_TYPES
    GTEST_SKIP() << "FP8 reductions are not compiled in.";
#else
    if (!haveFreeDeviceBytes(kLongPeakBytes)) {
        GTEST_SKIP() << "needs " << kLongPeakBytes << " bytes of free device memory";
    }
    Stream stream(0);
    Tensor storage(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {kLongBytes}));
    ASSERT_NO_FATAL_FAILURE(fillLongRows(storage, stream));
    // The dense [2, R] over axis 1, and VIEW-DIRECT-2A: the same bytes as [2, A, B] read as [2, B, A] over {1, 2}.
    const StridedSpan dense(storage, 0, {kLongRows, kLongR}, {kLongR, 1});
    const StridedSpan view(storage, 0, {kLongRows, kLongB, kLongA}, {kLongR, 1, kLongB});
    ASSERT_TRUE(dense.isDenseContiguous());
    ASSERT_FALSE(view.isDenseContiguous());
    for (CubReductionOp op : kOps) {
        SCOPED_TRACE("op " + std::to_string(static_cast<int>(op)));
        auto dense_stamp = CubReduction(op, std::vector<uint32_t>{1}, DataType::FP32).stamp(dense, stream);
        auto view_stamp = CubReduction(op, std::vector<uint32_t>{1, 2}, DataType::FP32).stamp(view, stream);
        EXPECT_EQ(dense_stamp->getPath(), CubReductionPath::ContiguousFixedSegment);
        EXPECT_EQ(view_stamp->getPath(), CubReductionPath::ContiguousFixedSegment);
        ASSERT_TRUE(dense_stamp->usesDenseExecutablePlan());
        ASSERT_TRUE(view_stamp->usesDenseExecutablePlan());
        EXPECT_EQ(view_stamp->getModernRKStrategyChain(), "view_physical:" + dense_stamp->getModernRKStrategyChain());
        dense_stamp->run();
        view_stamp->run();
        const std::vector<float> dense_out = copyGpuTensorAsFloat(dense_stamp->getOutputTensor(), stream);
        const std::vector<float> view_out = copyGpuTensorAsFloat(view_stamp->getOutputTensor(), stream);
        ASSERT_EQ(dense_out.size(), kLongRows);
        ASSERT_EQ(view_out.size(), kLongRows);
        for (uint64_t row = 0; row < kLongRows; ++row) {
            ASSERT_EQ(memcmp(&view_out[row], &dense_out[row], sizeof(float)), 0)
                << "row " << row << ": view " << view_out[row] << " dense " << dense_out[row];
            const double expected = longRowExpected(op, row);
            if (op == CubReductionOp::Mean || op == CubReductionOp::L2Norm) {
                EXPECT_NEAR(dense_out[row], expected, std::fabs(expected) * 1e-6) << "row " << row;
            } else {
                EXPECT_EQ(dense_out[row], static_cast<float>(expected)) << "row " << row;
            }
        }
    }
#endif
}

TEST(CubReductionViewPhysical, ArgRowsPastTheCubIntLimitFindTheirIndices) {
    REQUIRE_CUDA_DEVICE();
#if !THOR_CUB_ENABLE_FP8_TYPES
    GTEST_SKIP() << "FP8 reductions are not compiled in.";
#else
    if (!haveFreeDeviceBytes(kLongPeakBytes)) {
        GTEST_SKIP() << "needs " << kLongPeakBytes << " bytes of free device memory";
    }
    Stream stream(0);
    Tensor storage(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {kLongBytes}));
    ASSERT_NO_FATAL_FAILURE(fillLongRows(storage, stream));
    const StridedSpan dense(storage, 0, {kLongRows, kLongR}, {kLongR, 1});
    struct Expected {
        CubArgReductionOp op;
        uint64_t index[kLongRows];
        float value[kLongRows];
    };
    for (const Expected& e : {Expected{CubArgReductionOp::ArgMax, {kLongR - 1, uint64_t{1} << 31}, {2.0f, 4.0f}},
                              Expected{CubArgReductionOp::ArgMin, {(uint64_t{1} << 31) + 5, 7}, {-1.0f, -3.0f}}}) {
        SCOPED_TRACE("op " + std::to_string(static_cast<int>(e.op)));
        CubArgReductionOutputOptions outputs;
        outputs.value_output_dtype = DataType::FP32;
        auto stamped = CubArgReduction(e.op, 1, outputs).stamp(dense, stream);
        ASSERT_FALSE(stamped->getPlannedPassNames().empty());
        stamped->run();
        const std::vector<float> values = copyGpuTensorAsFloat(stamped->getValueOutputTensor().value(), stream);
        const std::vector<uint64_t> indices = copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream);
        ASSERT_EQ(values.size(), kLongRows);
        ASSERT_EQ(indices.size(), kLongRows);
        for (uint64_t row = 0; row < kLongRows; ++row) {
            EXPECT_EQ(indices[row], e.index[row]) << "row " << row;
            EXPECT_EQ(values[row], e.value[row]) << "row " << row;
        }
    }
#endif
}
