// Fused packet kernels must stay correct when a runtime tensor is a dense storage
// alias whose base is element aligned but not packet aligned, and when the final
// packet of a view would extend past the view into live neighbouring storage.
// Tensor::isDenseContiguous() does not look at the element offset, so such views
// reach the flat / specialized-broadcast fast paths.
//
// Every case writes into an output view that sits inside a larger allocation
// pre-filled with a sentinel, then checks the view values and that the guard
// regions before and after the view are untouched. Shapes and offsets are
// randomized; the seed is printed so failures reproduce
// (THOR_TEST_SEED=<seed> to replay).

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

// A dense view of `numel` elements at element `offset` inside an allocation with
// `guard` sentinel elements after it.
struct ViewStorage {
    Tensor storage;
    Tensor view;
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
    std::vector<uint64_t> strides(dims.size(), 1);
    for (size_t axis = dims.size(); axis-- > 1;) strides[axis - 1] = strides[axis] * dims[axis];
    result.view = result.storage.aliasView(dims, strides, offset);
    result.offset = offset;
    result.numel = numel;
    return result;
}

// Values on a 0.5 grid in [-2, 1.5]: sums of two stay exactly representable in
// every storage dtype exercised here, including FP8 E5M2.
std::vector<float> gridValues(uint64_t count, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(-4, 3);
    std::vector<float> values(count);
    for (float& v : values) v = 0.5f * static_cast<float>(dist(rng));
    return values;
}

// Checks the view region against `expected` (std::nullopt = must still hold the
// sentinel) and that the guard regions on both sides were not written.
void expectOutputView(const ViewStorage& out,
                      const std::vector<std::optional<float>>& expected,
                      Stream& stream,
                      const std::string& label) {
    const std::vector<float> all = copyStorageAsFloat(out.storage, stream);
    for (uint64_t i = 0; i < out.offset; ++i) {
        ASSERT_EQ(all[i], kSentinel) << label << ": wrote before the view at storage element " << i;
    }
    for (uint64_t i = 0; i < out.numel; ++i) {
        const float want = expected[i].value_or(kSentinel);
        ASSERT_EQ(all[out.offset + i], want) << label << ": view element " << i;
    }
    for (uint64_t i = out.offset + out.numel; i < all.size(); ++i) {
        ASSERT_EQ(all[i], kSentinel) << label << ": wrote past the view at storage element " << i;
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
        const Expression x = Expression::input("x", family.x, family.x);
        const Expression y = Expression::input("y", family.y, family.y);
        FusedEquation eq = FusedEquation::compile(Expression::outputs({{"z", (x + y).withOutputDType(family.out)}}).physicalOutputs(), 0);

        for (int trial = 0; trial < 4; ++trial) {
            const uint64_t numel = numel_dist(rng);
            // Trial 0 keeps every base packet aligned so the vector path's partial
            // final packet is exercised on its own.
            const uint64_t x_offset = trial == 0 ? 0 : offset_dist(rng);
            const uint64_t y_offset = trial == 0 ? 0 : offset_dist(rng);
            const uint64_t z_offset = trial == 0 ? 0 : offset_dist(rng);
            const std::string label = dtypeName(family.x) + "+" + dtypeName(family.y) + "->" + dtypeName(family.out) +
                                      " numel=" + std::to_string(numel) + " offsets=" + std::to_string(x_offset) + "," +
                                      std::to_string(y_offset) + "," + std::to_string(z_offset);

            const std::vector<float> xv = gridValues(numel, rng);
            const std::vector<float> yv = gridValues(numel, rng);
            ViewStorage xs = makeView({numel}, x_offset, xv, family.x, stream);
            ViewStorage ys = makeView({numel}, y_offset, yv, family.y, stream);
            ViewStorage zs = makeView({numel}, z_offset, {}, family.out, stream);
            ASSERT_TRUE(zs.view.isDenseContiguous());

            eq.run({{"x", xs.view}, {"y", ys.view}}, zs.view, stream);

            std::vector<std::optional<float>> expected(numel);
            for (uint64_t i = 0; i < numel; ++i) expected[i] = xv[i] + yv[i];
            expectOutputView(zs, expected, stream, label);
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
                const uint64_t z_offset = offset_dist(rng);
                const std::string label = dtypeName(dtype) + (row_bias ? " row-bias " : " scalar ") + std::to_string(rows) + "x" +
                                          std::to_string(cols) + " offsets=" + std::to_string(x_offset) + "," +
                                          std::to_string(b_offset) + "," + std::to_string(z_offset);

                const Expression x = Expression::input("x", dtype, dtype);
                const Expression b = Expression::input("b", dtype, dtype);
                FusedEquation eq = FusedEquation::compile(Expression::outputs({{"z", (x + b).withOutputDType(dtype)}}).physicalOutputs(), 0);

                const std::vector<uint64_t> b_dims = row_bias ? std::vector<uint64_t>{1, cols} : std::vector<uint64_t>{1, 1};
                const uint64_t b_numel = row_bias ? cols : 1;
                const std::vector<float> xv = gridValues(numel, rng);
                const std::vector<float> bv = gridValues(b_numel, rng);
                ViewStorage xs = makeView({rows, cols}, x_offset, xv, dtype, stream);
                ViewStorage bs = makeView(b_dims, b_offset, bv, dtype, stream);
                ViewStorage zs = makeView({rows, cols}, z_offset, {}, dtype, stream);

                eq.run({{"x", xs.view}, {"b", bs.view}}, zs.view, stream);

                std::vector<std::optional<float>> expected(numel);
                for (uint64_t r = 0; r < rows; ++r) {
                    for (uint64_t c = 0; c < cols; ++c) {
                        expected[r * cols + c] = xv[r * cols + c] + bv[row_bias ? c : 0];
                    }
                }
                expectOutputView(zs, expected, stream, label);
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
            const uint64_t z_offset = offset_dist(rng);
            const std::string label = dtypeName(dtype) + (broadcast ? " ragged row-bias" : " ragged flat") +
                                      " active=" + std::to_string(active) + " offsets=" + std::to_string(x_offset) + "," +
                                      std::to_string(y_offset) + "," + std::to_string(z_offset);

            const Expression x = Expression::input("x", dtype, dtype);
            const Expression y = Expression::input("y", dtype, dtype);
            const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
            const Expression z = (x + y).withOutputDType(dtype).withRaggedRuntimeExtent(
                active_count, batch, max_active, width, RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
            FusedEquation eq = FusedEquation::compile(Expression::outputs({{"z", z}}).physicalOutputs(), 0);

            const std::vector<uint64_t> y_dims = broadcast ? std::vector<uint64_t>{1, width} : std::vector<uint64_t>{max_active, width};
            const uint64_t y_numel = broadcast ? width : max_active * width;
            const std::vector<float> xv = gridValues(max_active * width, rng);
            const std::vector<float> yv = gridValues(y_numel, rng);
            ViewStorage xs = makeView({max_active, width}, x_offset, xv, dtype, stream);
            ViewStorage ys = makeView(y_dims, y_offset, yv, dtype, stream);
            ViewStorage zs = makeView({max_active, width}, z_offset, {}, dtype, stream);
            Tensor count = makeGpuUint32(static_cast<uint32_t>(active), stream);

            eq.run({{"x", xs.view}, {"y", ys.view}, {"active_count", count}}, zs.view, stream);

            std::vector<std::optional<float>> expected(max_active * width);
            for (uint64_t i = 0; i < active * width; ++i) {
                expected[i] = xv[i] + yv[broadcast ? (i % width) : i];
            }
            expectOutputView(zs, expected, stream, label);
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
            const uint64_t z_offset = offset_dist(rng);
            const std::string label = dtypeName(dtype) + (row_bias ? " transposed row-bias " : " transposed ") + std::to_string(rows) +
                                      "x" + std::to_string(cols) + " offsets=" + std::to_string(x_offset) + "," +
                                      std::to_string(y_offset) + "," + std::to_string(z_offset);

            const Expression x = Expression::input("x", dtype, dtype);
            const Expression y = Expression::input("y", dtype, dtype);
            FusedEquation eq =
                FusedEquation::compile(Expression::outputs({{"z", (x + y).withOutputDType(dtype).transpose()}}).physicalOutputs(), 0);

            const std::vector<uint64_t> y_dims = row_bias ? std::vector<uint64_t>{1, cols} : std::vector<uint64_t>{rows, cols};
            const uint64_t y_numel = row_bias ? cols : rows * cols;
            const std::vector<float> xv = gridValues(rows * cols, rng);
            const std::vector<float> yv = gridValues(y_numel, rng);
            ViewStorage xs = makeView({rows, cols}, x_offset, xv, dtype, stream);
            ViewStorage ys = makeView(y_dims, y_offset, yv, dtype, stream);
            ViewStorage zs = makeView({cols, rows}, z_offset, {}, dtype, stream);

            eq.run({{"x", xs.view}, {"y", ys.view}}, zs.view, stream);

            std::vector<std::optional<float>> expected(rows * cols);
            for (uint64_t r = 0; r < rows; ++r) {
                for (uint64_t c = 0; c < cols; ++c) {
                    expected[c * rows + r] = xv[r * cols + c] + yv[row_bias ? c : r * cols + c];
                }
            }
            expectOutputView(zs, expected, stream, label);
        }
    }
}
