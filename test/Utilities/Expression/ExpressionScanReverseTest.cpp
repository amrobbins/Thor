// Reverse scans through the public Expression API: dense scan(..., reverse), scanWithIndices and
// segmentedScanWithIndices with reverse, and the backward of a dense reverse sum. A reverse scan runs each row from
// its last element to its first; of equal elements an arg scan takes the first it meets, the last in the row. Values
// are small integers in FP32, so every sum is exact and outputs are compared exactly.

#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for reverse scan tests.";                    \
        }                                                                                         \
    } while (false)

template <typename T>
Tensor makeTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(dtype, dims));
    EXPECT_EQ(cpu.getTotalNumElements(), values.size());
    std::memcpy(cpu.getMemPtr(), values.data(), values.size() * sizeof(T));
    Tensor gpu(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(dtype, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
std::vector<T> readTensor(const Tensor& gpu, Stream& stream) {
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<T> values(gpu.getTotalNumElements());
    std::memcpy(values.data(), cpu.getMemPtr(), values.size() * sizeof(T));
    return values;
}

constexpr uint32_t kNoIndex = 0xFFFFFFFFU;

// A reverse scan of rows [bounds[r], bounds[r + 1]) by its definition: values (sum, min, max) and the winner's flat
// index (min, max; the first met wins, so a tie goes to the later element).
struct Reference {
    std::vector<float> values;
    std::vector<uint32_t> indices;
};

Reference reverseScan(const std::vector<float>& x, const std::vector<uint64_t>& bounds, ScanOp op, bool inclusive) {
    Reference ref{std::vector<float>(x.size()), std::vector<uint32_t>(x.size(), kNoIndex)};
    const float identity = op == ScanOp::Sum ? 0.0f : op == ScanOp::Max ? -INFINITY : INFINITY;
    for (uint64_t r = 0; r + 1 < bounds.size(); ++r) {
        float acc = identity;
        uint32_t winner = kNoIndex;
        for (uint64_t i = bounds[r + 1]; i-- > bounds[r];) {
            const float before = acc;
            const uint32_t before_winner = winner;
            if (op == ScanOp::Sum) {
                acc += x[i];
            } else if (winner == kNoIndex || (op == ScanOp::Max ? x[i] > acc : x[i] < acc)) {
                acc = x[i];
                winner = static_cast<uint32_t>(i);
            }
            ref.values[i] = inclusive ? acc : before;
            ref.indices[i] = inclusive ? winner : before_winner;
        }
    }
    return ref;
}

// Rows of 7 with ties, so the tie rule shows: the later of two equal elements wins in a reverse scan.
const std::vector<float> kX{3, 1, 4, 1, 5, 4, 2,    //
                            2, 2, 7, 7, 1, 0, 0,    //
                            -1, 6, -3, 6, -3, 2, 2};
constexpr uint64_t kRows = 3;
constexpr uint64_t kCols = 7;

std::vector<uint64_t> denseBounds() { return {0, kCols, 2 * kCols, 3 * kCols}; }

}  // namespace

TEST(ExpressionScanReverse, DenseReverseScansRunFromEachRowsEnd) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeTensor(DataType::FP32, {kRows, kCols}, kX, stream);
    for (const bool inclusive : {true, false}) {
        SCOPED_TRACE(inclusive ? "inclusive" : "exclusive");
        const Expression x = Expression::input("x");
        FusedEquation equation = FusedEquation::compile(Expression::outputs({{"sum", x.scan(ScanOp::Sum, -1, inclusive, true)},
                                                                             {"max", x.scan(ScanOp::Max, -1, inclusive, true)},
                                                                             {"min", x.scan(ScanOp::Min, -1, inclusive, true)},
                                                                             {"argmax", x.scan(ScanOp::ArgMax, -1, inclusive, true)},
                                                                             {"argmin", x.scan(ScanOp::ArgMin, -1, inclusive, true)}})
                                                            .physicalOutputs(),
                                                        0);
        StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
        plan.run();
        stream.synchronize();

        EXPECT_EQ(readTensor<float>(plan.output("sum"), stream), reverseScan(kX, denseBounds(), ScanOp::Sum, inclusive).values);
        EXPECT_EQ(readTensor<float>(plan.output("max"), stream), reverseScan(kX, denseBounds(), ScanOp::Max, inclusive).values);
        EXPECT_EQ(readTensor<float>(plan.output("min"), stream), reverseScan(kX, denseBounds(), ScanOp::Min, inclusive).values);
        EXPECT_EQ(readTensor<uint32_t>(plan.output("argmax"), stream), reverseScan(kX, denseBounds(), ScanOp::Max, inclusive).indices);
        EXPECT_EQ(readTensor<uint32_t>(plan.output("argmin"), stream), reverseScan(kX, denseBounds(), ScanOp::Min, inclusive).indices);
    }
}

// The value and index halves of a reverse pair share one scan stage, as forward pairs do, dense and ragged.
TEST(ExpressionScanReverse, ReverseScansWithIndicesShareOneStage) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<uint32_t> offsets{0, 5, 5, 13, 21};
    const std::vector<uint64_t> ragged_bounds(offsets.begin(), offsets.end());
    Tensor dense_input = makeTensor(DataType::FP32, {kRows, kCols}, kX, stream);
    Tensor flat_input = makeTensor(DataType::FP32, {kRows * kCols}, kX, stream);
    Tensor offsets_tensor = makeTensor(DataType::UINT32, {offsets.size()}, offsets, stream);

    for (const ScanOp op : {ScanOp::Max, ScanOp::Min}) {
        for (const bool inclusive : {true, false}) {
            for (const bool ragged : {false, true}) {
                SCOPED_TRACE(std::string(op == ScanOp::Max ? "max" : "min") + (inclusive ? " inclusive" : " exclusive") +
                             (ragged ? " ragged" : " dense"));
                const Expression x = Expression::input("x");
                const auto [values, indices] =
                    ragged ? x.segmentedScanWithIndices(Expression::input("offsets", DataType::UINT32, DataType::UINT32), op, inclusive, true)
                           : x.scanWithIndices(op, -1, inclusive, true);
                FusedEquation equation =
                    FusedEquation::compile(Expression::outputs({{"values", values}, {"indices", indices}}).physicalOutputs(), 0);
                StampedExecutionPlan plan = ragged ? equation.stamp({{"x", flat_input}, {"offsets", offsets_tensor}}, stream)
                                                   : equation.stamp({{"x", dense_input}}, stream);
                EXPECT_EQ(plan.stageKindNames().size(), 1U);
                plan.run();
                stream.synchronize();

                const Reference ref = reverseScan(kX, ragged ? ragged_bounds : denseBounds(), op, inclusive);
                EXPECT_EQ(readTensor<float>(plan.output("values"), stream), ref.values);
                EXPECT_EQ(readTensor<uint32_t>(plan.output("indices"), stream), ref.indices);
            }
        }
    }
}

// y[i] sums x from the row's end down to i, so dx[i] sums dy from the row's start up to i: a forward scan of dy
// (exclusive: up to the element before i).
TEST(ExpressionScanReverse, DenseReverseSumBackwardIsAForwardScan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> dy{1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, 2, 0, 2, 0, 2, 0, 2};
    Tensor input = makeTensor(DataType::FP32, {kRows, kCols}, kX, stream);
    Tensor grad = makeTensor(DataType::FP32, {kRows, kCols}, dy, stream);
    for (const bool inclusive : {true, false}) {
        SCOPED_TRACE(inclusive ? "inclusive" : "exclusive");
        const Expression y = Expression::input("x").scan(ScanOp::Sum, -1, inclusive, true);
        FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
        FusedEquation backward = forward.compileBackward({"x"}, "dy");
        StampedExecutionPlan plan = backward.stamp({{"x", input}, {"dy", grad}}, stream);
        plan.run();
        stream.synchronize();

        std::vector<float> want(dy.size());
        for (uint64_t r = 0; r < kRows; ++r) {
            float acc = 0.0f;
            for (uint64_t c = 0; c < kCols; ++c) {
                const uint64_t i = r * kCols + c;
                if (inclusive) acc += dy[i];
                want[i] = acc;
                if (!inclusive) acc += dy[i];
            }
        }
        EXPECT_EQ(readTensor<float>(plan.output("x_grad"), stream), want);
    }
}
