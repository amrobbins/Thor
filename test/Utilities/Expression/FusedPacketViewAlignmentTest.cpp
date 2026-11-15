// Fused packet kernels must stay correct when an input is a dense span (a strided view taken inside the expression)
// whose first element is element aligned but not packet aligned, and when the span's final packet would extend past it
// into live neighbouring storage. A dense span reaches the flat / specialized-broadcast fast paths.
//
// Every case reads its inputs as spans at random offsets into larger allocations whose other elements hold a
// sentinel, writes a dense output tensor pre-filled with the sentinel, and checks the output values (elements a
// ragged kernel must not write still hold the sentinel). Shapes and offsets are randomized; the seed is printed so
// failures reproduce (THOR_TEST_SEED=<seed> to replay).

#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for fused packet view tests.";              \
        }                                                                                         \
    } while (false)

TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);

// Exactly representable in FP32, FP16, BF16, FP8 E4M3 and FP8 E5M2, and never
// produced by the test arithmetic below.
constexpr float kSentinel = 64.0f;

uint64_t testSeed() {
    if (const char* env = std::getenv("THOR_TEST_SEED")) {
        return std::strtoull(env, nullptr, 10);
    }
    return std::random_device{}();
}

Tensor makeGpuStorage(const std::vector<float>& values, DataType dtype, Stream& stream) {
    const std::vector<uint64_t> dims{static_cast<uint64_t>(values.size())};
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, dims));
    float* cpu_values = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) cpu_values[i] = values[i];
    Tensor fp32_gpu(gpu_placement, TensorDescriptor(DataType::FP32, dims));
    fp32_gpu.copyFromAsync(cpu, stream);
    if (dtype == DataType::FP32) {
        stream.synchronize();
        return fp32_gpu;
    }
    Tensor converted(gpu_placement, TensorDescriptor(dtype, dims));
    TypeConverter::convertType(fp32_gpu.getMemPtr<void>(),
                               converted.getMemPtr<void>(),
                               DataType::FP32,
                               dtype,
                               static_cast<long>(values.size()),
                               stream,
                               gpu_placement.getDeviceNum());
    stream.synchronize();
    return converted;
}

Tensor makeGpuUint32(uint32_t value, Stream& stream) {
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::UINT32, {1}));
    cpu.getMemPtr<uint32_t>()[0] = value;
    Tensor gpu(gpu_placement, TensorDescriptor(DataType::UINT32, {1}));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyStorageAsFloat(Tensor storage, Stream& stream) {
    Tensor fp32_gpu(gpu_placement, TensorDescriptor(DataType::FP32, storage.getDimensions()));
    if (storage.getDataType() == DataType::FP32) {
        fp32_gpu.copyFromAsync(storage, stream);
    } else {
        TypeConverter::convertType(storage.getMemPtr<void>(),
                                   fp32_gpu.getMemPtr<void>(),
                                   storage.getDataType(),
                                   DataType::FP32,
                                   static_cast<long>(storage.getTotalNumElements()),
                                   stream,
                                   gpu_placement.getDeviceNum());
    }
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, storage.getDimensions()));
    cpu.copyFromAsync(fp32_gpu, stream);
    stream.synchronize();
    const float* values = cpu.getMemPtr<float>();
    return std::vector<float>(values, values + storage.getTotalNumElements());
}

// `numel` elements at element `offset` inside an allocation with `guard` sentinel elements after them: an expression
// reads them as a dense span (spanOf).
struct ViewStorage {
    Tensor storage;
    std::vector<uint64_t> dims;
    uint64_t offset = 0;
    uint64_t numel = 0;
};

ViewStorage makeView(const std::vector<uint64_t>& dims,
                     uint64_t offset,
                     const std::vector<float>& view_values,
                     DataType dtype,
                     Stream& stream) {
    constexpr uint64_t guard = 40;  // > largest packet (16 FP8 lanes) + slack
    uint64_t numel = 1;
    for (uint64_t d : dims) numel *= d;
    std::vector<float> host(offset + numel + guard, kSentinel);
    for (uint64_t i = 0; i < numel; ++i) host[offset + i] = view_values.empty() ? kSentinel : view_values[i];
    ViewStorage result;
    result.storage = makeGpuStorage(host, dtype, stream);
    result.dims = dims;
    result.offset = offset;
    result.numel = numel;
    return result;
}

// The expression input `name` read as the dense span `storage` describes.
Expression spanOf(const std::string& name, DataType dtype, const ViewStorage& storage) {
    std::vector<uint64_t> strides(storage.dims.size(), 1);
    for (size_t axis = storage.dims.size(); axis-- > 1;) strides[axis - 1] = strides[axis] * storage.dims[axis];
    return Expression::input(name, dtype, dtype).stridedView(storage.dims, strides, storage.offset);
}

// A dense output tensor of its own, every element the sentinel.
Tensor makeSentinelOutput(const std::vector<uint64_t>& dims, DataType dtype, Stream& stream) {
    uint64_t numel = 1;
    for (uint64_t d : dims) numel *= d;
    Tensor out = makeGpuStorage(std::vector<float>(numel, kSentinel), dtype, stream);
    out.reshape(dims);
    return out;
}

// Values on a 0.5 grid in [-2, 1.5]: sums of two stay exactly representable in
// every storage dtype exercised here, including FP8 E5M2.
std::vector<float> gridValues(uint64_t count, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(-4, 3);
    std::vector<float> values(count);
    for (float& v : values) v = 0.5f * static_cast<float>(dist(rng));
    return values;
}

// Checks the output against `expected` (std::nullopt = must still hold the sentinel).
void expectOutput(const Tensor& out, const std::vector<std::optional<float>>& expected, Stream& stream, const std::string& label) {
    const std::vector<float> all = copyStorageAsFloat(out, stream);
    ASSERT_EQ(all.size(), expected.size()) << label;
    for (uint64_t i = 0; i < all.size(); ++i) {
        ASSERT_EQ(all[i], expected[i].value_or(kSentinel)) << label << ": output element " << i;
    }
}

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

struct FlatDTypes {
    DataType x;
    DataType y;
    DataType out;
};

}  // namespace

// Flat fused kernels: packed BF16/FP16/FP8 (vector2 path) and FP32 / mixed
// (wide-scalar path), with misaligned input/output view bases and lengths that
// leave a partial final packet.
TEST(FusedPacketViewAlignment, FlatKernelsHandleOffsetViewsAndPartialPackets) {
    REQUIRE_CUDA_DEVICE();
    const uint64_t seed = testSeed();
    SCOPED_TRACE("THOR_TEST_SEED=" + std::to_string(seed));
    std::mt19937_64 rng(seed);
    Stream stream(0);

    const std::vector<FlatDTypes> families = {
        {DataType::BF16, DataType::BF16, DataType::BF16},
        {DataType::FP16, DataType::FP16, DataType::FP16},
        {DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3},
        {DataType::FP8_E5M2, DataType::FP8_E5M2, DataType::FP8_E5M2},
        {DataType::FP32, DataType::FP32, DataType::FP32},
        {DataType::BF16, DataType::FP32, DataType::FP32},
    };
    std::uniform_int_distribution<uint64_t> numel_dist(1, 300);
    std::uniform_int_distribution<uint64_t> offset_dist(0, 17);

    for (const FlatDTypes& family : families) {
        for (int trial = 0; trial < 4; ++trial) {
            const uint64_t numel = numel_dist(rng);
            // Trial 0 keeps every base packet aligned so the vector path's partial
            // final packet is exercised on its own.
            const uint64_t x_offset = trial == 0 ? 0 : offset_dist(rng);
            const uint64_t y_offset = trial == 0 ? 0 : offset_dist(rng);
            const std::string label = dtypeName(family.x) + "+" + dtypeName(family.y) + "->" + dtypeName(family.out) +
                                      " numel=" + std::to_string(numel) + " offsets=" + std::to_string(x_offset) + "," +
                                      std::to_string(y_offset);

            const std::vector<float> xv = gridValues(numel, rng);
            const std::vector<float> yv = gridValues(numel, rng);
            ViewStorage xs = makeView({numel}, x_offset, xv, family.x, stream);
            ViewStorage ys = makeView({numel}, y_offset, yv, family.y, stream);
            Tensor z = makeSentinelOutput({numel}, family.out, stream);

            const Expression x = spanOf("x", family.x, xs);
            const Expression y = spanOf("y", family.y, ys);
            FusedEquation eq =
                FusedEquation::compile(Expression::outputs({{"z", (x + y).withOutputDType(family.out)}}).physicalOutputs(), 0);
            eq.run({{"x", xs.storage}, {"y", ys.storage}}, z, stream);

            std::vector<std::optional<float>> expected(numel);
            for (uint64_t i = 0; i < numel; ++i) expected[i] = xv[i] + yv[i];
            expectOutput(z, expected, stream, label);
        }
    }
}

// Dense specialized broadcast (vector2 BF16 path and scalar FP32 path): odd
// element counts and misaligned bases must not write past the output view.
TEST(FusedPacketViewAlignment, DenseBroadcastHandlesOffsetViewsAndOddTails) {
    REQUIRE_CUDA_DEVICE();
    const uint64_t seed = testSeed();
    SCOPED_TRACE("THOR_TEST_SEED=" + std::to_string(seed));
    std::mt19937_64 rng(seed);
    Stream stream(0);

    std::uniform_int_distribution<uint64_t> rows_dist(1, 9);
    std::uniform_int_distribution<uint64_t> offset_dist(0, 9);
    for (DataType dtype : {DataType::BF16, DataType::FP32}) {
        for (bool row_bias : {false, true}) {
            // One even-width and one odd-width shape per configuration; widths that
            // are multiples of 8 reach the widest packet paths.
            for (uint64_t cols : {uint64_t{16}, uint64_t{2} * rows_dist(rng) + 1}) {
                const uint64_t rows = rows_dist(rng);
                const uint64_t numel = rows * cols;
                const uint64_t x_offset = offset_dist(rng);
                const uint64_t b_offset = offset_dist(rng);
                const std::string label = dtypeName(dtype) + (row_bias ? " row-bias " : " scalar ") + std::to_string(rows) + "x" +
                                          std::to_string(cols) + " offsets=" + std::to_string(x_offset) + "," +
                                          std::to_string(b_offset);

                const std::vector<uint64_t> b_dims = row_bias ? std::vector<uint64_t>{1, cols} : std::vector<uint64_t>{1, 1};
                const uint64_t b_numel = row_bias ? cols : 1;
                const std::vector<float> xv = gridValues(numel, rng);
                const std::vector<float> bv = gridValues(b_numel, rng);
                ViewStorage xs = makeView({rows, cols}, x_offset, xv, dtype, stream);
                ViewStorage bs = makeView(b_dims, b_offset, bv, dtype, stream);
                Tensor z = makeSentinelOutput({rows, cols}, dtype, stream);

                const Expression x = spanOf("x", dtype, xs);
                const Expression b = spanOf("b", dtype, bs);
                FusedEquation eq =
                    FusedEquation::compile(Expression::outputs({{"z", (x + b).withOutputDType(dtype)}}).physicalOutputs(), 0);
                eq.run({{"x", xs.storage}, {"b", bs.storage}}, z, stream);

                std::vector<std::optional<float>> expected(numel);
                for (uint64_t r = 0; r < rows; ++r) {
                    for (uint64_t c = 0; c < cols; ++c) {
                        expected[r * cols + c] = xv[r * cols + c] + bv[row_bias ? c : 0];
                    }
                }
                expectOutput(z, expected, stream, label);
            }
        }
    }
}

// Ragged runtime-extent kernels (flat and specialized broadcast): only the
// active prefix may be written, and misaligned view bases must still work.
TEST(FusedPacketViewAlignment, RaggedKernelsHandleOffsetViews) {
    REQUIRE_CUDA_DEVICE();
    const uint64_t seed = testSeed();
    SCOPED_TRACE("THOR_TEST_SEED=" + std::to_string(seed));
    std::mt19937_64 rng(seed);
    Stream stream(0);

    constexpr uint64_t batch = 2;
    constexpr uint64_t max_active = 12;
    constexpr uint64_t width = 16;  // packet aligned so the widest ragged paths are selected
    std::uniform_int_distribution<uint64_t> active_dist(0, max_active);
    std::uniform_int_distribution<uint64_t> offset_dist(0, 9);

    for (DataType dtype : {DataType::BF16, DataType::FP32}) {
        for (bool broadcast : {false, true}) {
            const uint64_t active = active_dist(rng);
            const uint64_t x_offset = offset_dist(rng);
            const uint64_t y_offset = offset_dist(rng);
            const std::string label = dtypeName(dtype) + (broadcast ? " ragged row-bias" : " ragged flat") +
                                      " active=" + std::to_string(active) + " offsets=" + std::to_string(x_offset) + "," +
                                      std::to_string(y_offset);

            const std::vector<uint64_t> y_dims = broadcast ? std::vector<uint64_t>{1, width} : std::vector<uint64_t>{max_active, width};
            const uint64_t y_numel = broadcast ? width : max_active * width;
            const std::vector<float> xv = gridValues(max_active * width, rng);
            const std::vector<float> yv = gridValues(y_numel, rng);
            ViewStorage xs = makeView({max_active, width}, x_offset, xv, dtype, stream);
            ViewStorage ys = makeView(y_dims, y_offset, yv, dtype, stream);
            Tensor out = makeSentinelOutput({max_active, width}, dtype, stream);
            Tensor count = makeGpuUint32(static_cast<uint32_t>(active), stream);

            const Expression x = spanOf("x", dtype, xs);
            const Expression y = spanOf("y", dtype, ys);
            const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
            const Expression z = (x + y).withOutputDType(dtype).withRaggedRuntimeExtent(
                active_count, batch, max_active, width, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
            FusedEquation eq = FusedEquation::compile(Expression::outputs({{"z", z}}).physicalOutputs(), 0);
            eq.run({{"x", xs.storage}, {"y", ys.storage}, {"active_count", count}}, out, stream);

            std::vector<std::optional<float>> expected(max_active * width);
            for (uint64_t i = 0; i < active * width; ++i) {
                expected[i] = xv[i] + yv[broadcast ? (i % width) : i];
            }
            expectOutput(out, expected, stream, label);
        }
    }
}

// Tiled-transpose materialization (flat and specialized broadcast): packed
// input loads and packed output stores must check the actual address, not only
// that the element index is a pack multiple.
TEST(FusedPacketViewAlignment, TransposedMaterializationHandlesOffsetViews) {
    REQUIRE_CUDA_DEVICE();
    const uint64_t seed = testSeed();
    SCOPED_TRACE("THOR_TEST_SEED=" + std::to_string(seed));
    std::mt19937_64 rng(seed);
    Stream stream(0);

    std::uniform_int_distribution<uint64_t> dim_dist(33, 80);
    std::uniform_int_distribution<uint64_t> offset_dist(0, 9);
    for (DataType dtype : {DataType::BF16, DataType::FP8_E4M3, DataType::FP32}) {
        for (bool row_bias : {false, true}) {
            const uint64_t rows = dim_dist(rng);
            const uint64_t cols = 2 * (dim_dist(rng) / 2);  // even, so pair/pack paths are eligible
            const uint64_t x_offset = offset_dist(rng);
            const uint64_t y_offset = offset_dist(rng);
            const std::string label = dtypeName(dtype) + (row_bias ? " transposed row-bias " : " transposed ") + std::to_string(rows) +
                                      "x" + std::to_string(cols) + " offsets=" + std::to_string(x_offset) + "," +
                                      std::to_string(y_offset);

            const std::vector<uint64_t> y_dims = row_bias ? std::vector<uint64_t>{1, cols} : std::vector<uint64_t>{rows, cols};
            const uint64_t y_numel = row_bias ? cols : rows * cols;
            const std::vector<float> xv = gridValues(rows * cols, rng);
            const std::vector<float> yv = gridValues(y_numel, rng);
            ViewStorage xs = makeView({rows, cols}, x_offset, xv, dtype, stream);
            ViewStorage ys = makeView(y_dims, y_offset, yv, dtype, stream);
            Tensor z = makeSentinelOutput({cols, rows}, dtype, stream);

            const Expression x = spanOf("x", dtype, xs);
            const Expression y = spanOf("y", dtype, ys);
            FusedEquation eq =
                FusedEquation::compile(Expression::outputs({{"z", (x + y).withOutputDType(dtype).transpose()}}).physicalOutputs(), 0);
            eq.run({{"x", xs.storage}, {"y", ys.storage}}, z, stream);

            std::vector<std::optional<float>> expected(rows * cols);
            for (uint64_t r = 0; r < rows; ++r) {
                for (uint64_t c = 0; c < cols; ++c) {
                    expected[c * rows + r] = xv[r * cols + c] + yv[row_bias ? c : r * cols + c];
                }
            }
            expectOutput(z, expected, stream, label);
        }
    }
}
