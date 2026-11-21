// The backward of a min or max scan: each element's gradient goes to the winner of its prefix (the element its output
// was taken from), so dx[w] is the sum of dy over the elements whose winner is w, and 0 for an element that never wins.
// The winner is the arg scan's: the first of equal elements in scan order, NaN below every number. Gradients are small
// multiples of 1/4, so every sum is exact in FP32 whatever its order and the outputs are compared exactly. x and dy may
// be any float dtypes; dx is x's gradient dtype (for an input, x's own dtype), and is compared with the FP32 sum rounded
// to it by the host.

#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for scan min/max backward tests.";           \
        }                                                                                         \
    } while (false)

template <typename T>
void writeTensor(Tensor& gpu, const std::vector<T>& values, Stream& stream) {
    ASSERT_EQ(gpu.getTotalNumElements(), values.size());
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    std::memcpy(cpu.getMemPtr(), values.data(), values.size() * sizeof(T));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
}

template <typename T>
Tensor makeTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<T>& values, Stream& stream) {
    Tensor gpu(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(dtype, dims));
    writeTensor(gpu, values, stream);
    return gpu;
}

std::vector<float> readFloats(const Tensor& gpu, Stream& stream) {
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(gpu.getTotalNumElements());
    std::memcpy(values.data(), cpu.getMemPtr(), values.size() * sizeof(float));
    return values;
}

// FP32 values as a tensor of FP32, FP16 or BF16 (the values here are exact in each), and back.
Tensor makeFloatTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    if (dtype == DataType::FP16) {
        std::vector<__half> stored(values.size());
        for (uint64_t i = 0; i < values.size(); ++i) stored[i] = __float2half_rn(values[i]);
        return makeTensor(dtype, dims, stored, stream);
    }
    if (dtype == DataType::BF16) {
        std::vector<__nv_bfloat16> stored(values.size());
        for (uint64_t i = 0; i < values.size(); ++i) stored[i] = __float2bfloat16_rn(values[i]);
        return makeTensor(dtype, dims, stored, stream);
    }
    return makeTensor(dtype, dims, values, stream);
}

std::vector<float> readAsFloats(const Tensor& gpu, Stream& stream) {
    const DataType dtype = gpu.getDataType();
    if (dtype == DataType::FP32) return readFloats(gpu, stream);
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(gpu.getTotalNumElements());
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = dtype == DataType::FP16 ? __half2float(cpu.getMemPtr<__half>()[i]) : __bfloat162float(cpu.getMemPtr<__nv_bfloat16>()[i]);
    }
    return values;
}

// FP32 sums as dx in dtype stores them: rounded to nearest once.
std::vector<float> roundedTo(DataType dtype, std::vector<float> values) {
    for (float& value : values) {
        if (dtype == DataType::FP16) value = __half2float(__float2half_rn(value));
        if (dtype == DataType::BF16) value = __bfloat162float(__float2bfloat16_rn(value));
    }
    return values;
}

struct ScanCase {
    ScanOp op;
    bool inclusive;
    bool reverse;

    std::string name() const {
        std::string text = op == ScanOp::Max ? "max" : "min";
        text += inclusive ? " inclusive" : " exclusive";
        text += reverse ? " reverse" : " forward";
        return text;
    }
};

// Whether `candidate` replaces `winner`: only when strictly better, so the first of equal elements keeps winning, and
// a NaN never replaces anything while any number replaces a NaN.
bool replaces(float candidate, float winner, ScanOp op) {
    if (std::isnan(candidate)) return false;
    if (std::isnan(winner)) return true;
    return op == ScanOp::Max ? candidate > winner : candidate < winner;
}

// The backward by its definition, over rows [bounds[r], bounds[r + 1]). Elements outside the rows stay 0.
std::vector<float> referenceBackward(const std::vector<float>& x,
                                     const std::vector<float>& dy,
                                     const std::vector<uint64_t>& bounds,
                                     const ScanCase& scan) {
    std::vector<float> dx(x.size(), 0.0f);
    for (uint64_t r = 0; r + 1 < bounds.size(); ++r) {
        const uint64_t begin = bounds[r];
        const uint64_t end = bounds[r + 1];
        bool has_winner = false;
        uint64_t winner = 0;
        for (uint64_t step = 0; step < end - begin; ++step) {
            const uint64_t i = scan.reverse ? end - 1 - step : begin + step;
            if (!scan.inclusive && has_winner) dx[winner] += dy[i];
            if (!has_winner || replaces(x[i], x[winner], scan.op)) {
                winner = i;
                has_winner = true;
            }
            if (scan.inclusive) dx[winner] += dy[i];
        }
    }
    return dx;
}

// Values from a small set, so equal elements are common, with runs of winners from one element to thousands long.
std::vector<float> patternValues(uint64_t count, uint64_t seed) {
    std::vector<float> values(count);
    uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1;
    for (uint64_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint64_t draw = state >> 33;
        // Mostly small values; one in 512 is a new extreme candidate, so winners change rarely in long rows.
        const float base = static_cast<float>(draw % 9) - 4.0f;
        values[i] = (draw >> 8) % 512 == 0 ? base * static_cast<float>(1 + (draw >> 20) % 64) : base;
    }
    return values;
}

std::vector<float> patternGradients(uint64_t count, uint64_t seed) {
    std::vector<float> values(count);
    uint64_t state = seed * 0xD1B54A32D192ED03ULL + 7;
    for (uint64_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        values[i] = (static_cast<float>((state >> 40) % 33) - 16.0f) * 0.25f;
    }
    return values;
}

void expectEqualOverRows(const std::vector<float>& got, const std::vector<float>& want, uint64_t active, const std::string& label) {
    ASSERT_GE(got.size(), active);
    ASSERT_GE(want.size(), active);
    uint32_t reported = 0;
    for (uint64_t i = 0; i < active; ++i) {
        if (got[i] != want[i] && reported++ < 8) {
            ADD_FAILURE() << label << ": element " << i << " is " << got[i] << ", expected " << want[i];
        }
    }
    EXPECT_EQ(reported, 0U) << label << ": " << reported << " elements differ";
}

std::vector<uint64_t> denseBounds(uint64_t rows, uint64_t cols) {
    std::vector<uint64_t> bounds(rows + 1);
    for (uint64_t r = 0; r <= rows; ++r) bounds[r] = r * cols;
    return bounds;
}

}  // namespace

TEST(ExpressionScanMinMaxBackward, DenseGradientsGoToEachPrefixWinner) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    struct Shape {
        uint64_t rows;
        uint64_t cols;
    };
    // A few short rows; rows of 5 over many tiles; long rows, where one winner's run spans tiles; one row. Max and min,
    // inclusive and exclusive, forward and reverse.
    for (const Shape& shape : {Shape{3, 8}, Shape{4000, 5}, Shape{4, 5000}, Shape{1, 70000}}) {
        const uint64_t count = shape.rows * shape.cols;
        std::vector<float> x = patternValues(count, shape.cols);
        // Equal elements, -0 against +0, and NaN: at a row's start, and after a number.
        x[0] = std::numeric_limits<float>::quiet_NaN();
        x[1] = -0.0f;
        x[2] = 0.0f;
        x[count - 2] = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> dy = patternGradients(count, shape.rows);
        Tensor input = makeTensor(DataType::FP32, {shape.rows, shape.cols}, x, stream);
        Tensor grad = makeTensor(DataType::FP32, {shape.rows, shape.cols}, dy, stream);

        for (const ScanCase& scan : {ScanCase{ScanOp::Max, true, false},
                                     ScanCase{ScanOp::Max, false, false},
                                     ScanCase{ScanOp::Min, true, false},
                                     ScanCase{ScanOp::Min, false, false},
                                     ScanCase{ScanOp::Max, true, true},
                                     ScanCase{ScanOp::Max, false, true},
                                     ScanCase{ScanOp::Min, true, true},
                                     ScanCase{ScanOp::Min, false, true}}) {
            const std::string label = std::to_string(shape.rows) + " x " + std::to_string(shape.cols) + ", " + scan.name();
            SCOPED_TRACE(label);
            const Expression y = Expression::input("x").scan(scan.op, -1, scan.inclusive, scan.reverse);
            FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
            FusedEquation backward = forward.compileBackward({"x"}, "dy");
            StampedExecutionPlan plan = backward.stamp({{"x", input}, {"dy", grad}}, stream);
            plan.run();
            stream.synchronize();

            const Tensor dx = plan.output("x_grad");
            ASSERT_EQ(dx.getDataType(), DataType::FP32);
            expectEqualOverRows(readFloats(dx, stream), referenceBackward(x, dy, denseBounds(shape.rows, shape.cols), scan), count, label);
        }
    }
}

TEST(ExpressionScanMinMaxBackward, RaggedGradientsGoToEachPrefixWinner) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    struct Rows {
        std::vector<uint32_t> offsets;
        uint64_t capacity;
    };
    // Empty rows at the start, in the middle and at the end; then long rows over many tiles, the rows ending at the
    // capacity.
    std::vector<Rows> row_sets{Rows{{0, 0, 3, 3, 7, 12, 12}, 12}};
    {
        Rows long_rows{{0}, 0};
        for (const uint32_t length : {1U, 9000U, 0U, 3U, 30000U, 17U, 4096U, 2U}) {
            long_rows.offsets.push_back(long_rows.offsets.back() + length);
        }
        long_rows.capacity = long_rows.offsets.back();
        row_sets.push_back(long_rows);
    }

    for (const Rows& rows : row_sets) {
        std::vector<float> x = patternValues(rows.capacity, rows.offsets.size());
        x[rows.capacity - 1] = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> dy = patternGradients(rows.capacity, rows.capacity);
        const std::vector<uint64_t> bounds(rows.offsets.begin(), rows.offsets.end());
        Tensor input = makeTensor(DataType::FP32, {rows.capacity}, x, stream);
        Tensor grad = makeTensor(DataType::FP32, {rows.capacity}, dy, stream);
        Tensor offsets = makeTensor(DataType::UINT32, {rows.offsets.size()}, rows.offsets, stream);

        for (const ScanOp op : {ScanOp::Max, ScanOp::Min}) {
            for (const bool inclusive : {true, false}) {
                for (const bool reverse : {false, true}) {
                    const ScanCase scan{op, inclusive, reverse};
                    const std::string label = std::to_string(rows.capacity) + " elements, " + scan.name();
                    SCOPED_TRACE(label);
                    const Expression y = Expression::input("x").segmentedScan(Expression::input("offsets", DataType::UINT32, DataType::UINT32), op, inclusive, reverse);
                    FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
                    FusedEquation backward = forward.compileBackward({"x"}, "dy");
                    StampedExecutionPlan plan = backward.stamp({{"x", input}, {"offsets", offsets}, {"dy", grad}}, stream);
                    plan.run();
                    stream.synchronize();

                    const Tensor dx = plan.output("x_grad");
                    ASSERT_EQ(dx.getDataType(), DataType::FP32);
                    expectEqualOverRows(readFloats(dx, stream), referenceBackward(x, dy, bounds, scan), rows.capacity, label);
                }
            }
        }
    }
}

// The rows can end before the capacity, and what is past them is undefined: the input, the gradient, and the indices
// the arg scan left there on an earlier run. None of it may reach an element inside the rows.
//
// The first run's rows cover the whole capacity, and element 7 wins from position 7 to the end. The second run's rows
// end at 9, with large gradients past them: an index of 7 left at positions 9 to 15 would add them to element 7.
TEST(ExpressionScanMinMaxBackward, RaggedGradientsPastTheRowsReachNothingInsideThem) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t capacity = 16;
    constexpr uint64_t second_end = 9;

    for (const bool inclusive : {true, false}) {
        const ScanCase scan{ScanOp::Max, inclusive, false};
        SCOPED_TRACE(scan.name());
        const Expression y = Expression::input("x").segmentedScan(Expression::input("offsets", DataType::UINT32, DataType::UINT32), ScanOp::Max, inclusive, false);
        FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
        FusedEquation backward = forward.compileBackward({"x"}, "dy");

        std::vector<float> x(capacity, 1.0f);
        x[2] = 3.0f;
        x[7] = 5.0f;
        std::vector<float> dy(capacity, 0.25f);
        Tensor input = makeTensor(DataType::FP32, {capacity}, x, stream);
        Tensor grad = makeTensor(DataType::FP32, {capacity}, dy, stream);
        Tensor offsets = makeTensor(DataType::UINT32, {3}, std::vector<uint32_t>{0, 6, capacity}, stream);
        StampedExecutionPlan plan = backward.stamp({{"x", input}, {"offsets", offsets}, {"dy", grad}}, stream);

        plan.run();
        stream.synchronize();
        expectEqualOverRows(readFloats(plan.output("x_grad"), stream), referenceBackward(x, dy, {0, 6, capacity}, scan), capacity, "rows to the capacity");

        // The same plan, the rows now ending at 9. Past them the gradient is large and the input would win.
        for (uint64_t i = second_end; i < capacity; ++i) {
            dy[i] = 64.0f;
            x[i] = 100.0f;
        }
        writeTensor(input, x, stream);
        writeTensor(grad, dy, stream);
        writeTensor(offsets, std::vector<uint32_t>{0, 6, second_end}, stream);
        plan.run();
        stream.synchronize();
        expectEqualOverRows(readFloats(plan.output("x_grad"), stream), referenceBackward(x, dy, {0, 6, second_end}, scan), second_end, "rows ending at 9");
    }
}

// x and dy in their own dtypes, dense and ragged, with no cast: dx comes out in x's gradient dtype, x's own dtype for an
// input (as a sum scan's backward), and holds each FP32 sum rounded once.
TEST(ExpressionScanMinMaxBackward, MixedDtypesNeedNoCast) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Dtypes {
        DataType x;
        DataType dy;
    };
    constexpr uint64_t rows = 4;
    constexpr uint64_t cols = 5000;
    constexpr uint64_t count = rows * cols;
    std::vector<float> x = patternValues(count, 11);
    x[1] = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> dy = patternGradients(count, 12);
    const std::vector<uint32_t> ragged_offsets{0, 3, 3, 9000, count};
    const std::vector<uint64_t> ragged_bounds(ragged_offsets.begin(), ragged_offsets.end());

    for (const Dtypes dtypes : {Dtypes{DataType::FP16, DataType::FP16},
                                Dtypes{DataType::BF16, DataType::FP32},
                                Dtypes{DataType::FP32, DataType::BF16},
                                Dtypes{DataType::BF16, DataType::FP16}}) {
        Tensor input = makeFloatTensor(dtypes.x, {rows, cols}, x, stream);
        Tensor grad = makeFloatTensor(dtypes.dy, {rows, cols}, dy, stream);
        Tensor flat_input = makeFloatTensor(dtypes.x, {count}, x, stream);
        Tensor flat_grad = makeFloatTensor(dtypes.dy, {count}, dy, stream);
        Tensor offsets = makeTensor(DataType::UINT32, {ragged_offsets.size()}, ragged_offsets, stream);
        for (const ScanCase& scan : {ScanCase{ScanOp::Max, true, false}, ScanCase{ScanOp::Min, false, true}}) {
            for (const bool ragged : {false, true}) {
                const std::string label = "x " + TensorDescriptor::getElementTypeName(dtypes.x) + ", dy " +
                                          TensorDescriptor::getElementTypeName(dtypes.dy) + ", " + scan.name() +
                                          (ragged ? ", ragged" : ", dense");
                SCOPED_TRACE(label);
                const Expression y =
                    ragged ? Expression::input("x").segmentedScan(
                                 Expression::input("offsets", DataType::UINT32, DataType::UINT32), scan.op, scan.inclusive, scan.reverse)
                           : Expression::input("x").scan(scan.op, -1, scan.inclusive, scan.reverse);
                FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
                FusedEquation backward = forward.compileBackward({"x"}, "dy");
                StampedExecutionPlan plan =
                    ragged ? backward.stamp({{"x", flat_input}, {"offsets", offsets}, {"dy", flat_grad}}, stream)
                           : backward.stamp({{"x", input}, {"dy", grad}}, stream);
                plan.run();
                stream.synchronize();

                const Tensor dx = plan.output("x_grad");
                ASSERT_EQ(dx.getDataType(), dtypes.x);
                const std::vector<float> want =
                    roundedTo(dtypes.x, referenceBackward(roundedTo(dtypes.x, x), dy, ragged ? ragged_bounds : denseBounds(rows, cols), scan));
                expectEqualOverRows(readAsFloats(dx, stream), want, count, label);
            }
        }
    }
}
