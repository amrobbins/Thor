// Original FP8 RK selection honours the input's alignment, as FP32 and FP16 / BF16 do. An FP8 RK pass names the base
// alignment it reads (RK::rkPassInputAlignment): the RuntimeK passes any byte address, AlignedK its 2- or 8-byte
// packet, DeepRSmallK 16 bytes. RuntimeK PacketComplete takes its row prefixes from the base address, so it reads any
// address at any R and K; below 16 bytes its tile may be narrower than a packet. The VALUE OneWindow kernel takes its
// windows from the base address too, so it reads any address when K fits a window that also holds a prefix
// (RK::oneWindowUnalignedCapacity), and 16 bytes otherwise. FP8 ARG selection filters its passes the same way
// (retainedFp8ArgPassInputAlignment). The ARG OneWindow twins (FP8, FP32, FP16 / BF16) take their windows from the base
// address too, under the same K limit, and the ARG PacketComplete twins their row prefixes, at any K.
//
//   - PassesNameTheAlignmentTheyRead / OneWindowHoldsAPrefixBelowSixteenBytes /
//     PacketCompleteTilesMayBeNarrowerThanAPacket: host-only identity rules.
//   - ArgOneWindowReadsEveryByteOffset: every FP8 ARG OneWindow unit and ownership at bases 1, 3, 8 and 15
//     bytes off a 16-byte boundary gives the values and indices of the same launch at an aligned base.
//   - ArgPacketCompleteReadsEveryByteOffset: the FP8 ARG PacketComplete twin, every warps-per-output and the
//     tiles VALUE proposes, at bases 1, 3, 8 and 15 bytes off a 16-byte boundary gives the aligned launch's outputs.
//   - MisalignedInputsUseTolerantPasses: FP8 E4M3 / E5M2 inputs 1, 2, 3 and 8 bytes off a 16-byte boundary reduce
//     exactly on tolerant passes (SUM of small integers and MAX are exact in any order).
//   - ShortLastSitesRunAtEveryAlignment: [40, 12, 64] and [40, 12, 5] over axis 1 (a last site with R <= 32, which no
//     rotated pass reads) at 1, 2 and 3 bytes in.
//   - DISABLED_TolerantPassTiming (run with --gtest_also_run_disabled_tests): SUM of FP8 E4M3, FP16 and FP32 inputs
//     one element off a 16-byte boundary against the same inputs aligned, and a copy of the same bytes, over a ring of
//     inputs past L2. Prints CSV (alignment_timing lines). ARGMAX of the same inputs too (alignment_timing_arg
//     lines, the same columns; the chains are the planned ARG pass names). Measurement only.

#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "test/Utilities/TensorOperations/OutputSpan.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"
#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;
namespace Internal = ThorImplementation::ReductionInternal;
namespace RK = ThorImplementation::ReductionInternal::RK;
using RK::PhysicalFamily;
using RK::PhysicalKernel;
using RK::RuntimeLayout;

namespace {

uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }

struct Shape {
    std::vector<uint64_t> dims;
    std::vector<uint32_t> axes;
    const char* name;
};

uint64_t product(const std::vector<uint64_t>& dims) {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n;
}

// The host SUM or MAX of `values` (row-major `dims`) over `axes`, in the order of the retained axes.
std::vector<float> reference(const std::vector<float>& values, const Shape& shape, bool maximum) {
    std::vector<bool> reduced(shape.dims.size(), false);
    for (uint32_t axis : shape.axes) reduced[axis] = true;
    uint64_t outputs = 1;
    for (size_t a = 0; a < shape.dims.size(); ++a)
        if (!reduced[a]) outputs *= shape.dims[a];
    std::vector<double> out(outputs, maximum ? -1e30 : 0.0);
    std::vector<uint64_t> index(shape.dims.size(), 0);
    for (uint64_t e = 0; e < values.size(); ++e) {
        uint64_t o = 0;
        for (size_t a = 0; a < shape.dims.size(); ++a)
            if (!reduced[a]) o = o * shape.dims[a] + index[a];
        out[o] = maximum ? std::max<double>(out[o], values[e]) : out[o] + values[e];
        for (size_t a = shape.dims.size(); a-- > 0;) {
            if (++index[a] < shape.dims[a]) break;
            index[a] = 0;
        }
    }
    return std::vector<float>(out.begin(), out.end());
}

// `values` as `dtype` elements `offset_bytes` into a UINT8 storage kept alive in `storages`.
StridedSpan misaligned(const std::vector<float>& values, const std::vector<uint64_t>& dims, DataType dtype,
                       uint64_t offset_bytes, std::vector<Tensor>& storages, Stream& stream) {
    const uint64_t elements = values.size();
    const Tensor packed = makeGpuTensor(values, {elements}, stream, dtype);
    const uint64_t size = offset_bytes + elements + 16;
    storages.push_back(makeGpuUnsignedTensor(std::vector<uint64_t>(size, 0), {size}, stream, DataType::UINT8));
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(static_cast<uint8_t*>(storages.back().getMemPtr<void>()) + offset_bytes,
                                           packed.getMemPtr<void>(), elements, cudaMemcpyDeviceToDevice,
                                           stream.getStream()));
    stream.synchronize();
    return StridedSpan(storages.back()).bytesAs(offset_bytes, dtype, dims);
}

// `packed`'s bytes `offset_bytes` into a new UINT8 storage kept alive in `storages`, read as `dims`.
StridedSpan misalignedCopy(const Tensor& packed, const std::vector<uint64_t>& dims, DataType dtype,
                           uint64_t offset_bytes, std::vector<Tensor>& storages, Stream& stream) {
    const uint64_t bytes = packed.getArraySizeInBytes();
    storages.emplace_back(gpuPlacement, TensorDescriptor(DataType::UINT8, {offset_bytes + bytes + 16}));
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(static_cast<uint8_t*>(storages.back().getMemPtr<void>()) + offset_bytes,
                                           packed.getMemPtr<void>(), bytes, cudaMemcpyDeviceToDevice,
                                           stream.getStream()));
    return StridedSpan(storages.back()).bytesAs(offset_bytes, dtype, dims);
}

std::vector<float> smallIntegers(std::mt19937_64& rng, uint64_t n) {
    std::vector<float> values(n);
    for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 7) - 3);
    return values;
}

}  // namespace

TEST(ReducerFp8Alignment, PassesNameTheAlignmentTheyRead) {
    PhysicalKernel kernel{};
    // OneWindow reads any address while K fits its window with a prefix, and 16 bytes past that.
    kernel.family = PhysicalFamily::OneWindow;
    kernel.c = 16;
    for (const uint64_t outputs : {uint64_t{1}, uint64_t{2}, uint64_t{4}, uint64_t{8}}) {
        kernel.a = outputs;
        const uint64_t capacity = RK::oneWindowUnalignedCapacity(outputs, 16, 1);
        EXPECT_EQ(RK::rkPassInputAlignment(kernel, 1), 1u);
        EXPECT_EQ(RK::rkPassInputAlignment(kernel, capacity), 1u);
        EXPECT_EQ(RK::rkPassInputAlignment(kernel, capacity + 1), 16u);
    }
    kernel = PhysicalKernel{};
    kernel.family = PhysicalFamily::DeepRSmallK;
    EXPECT_EQ(RK::rkPassInputAlignment(kernel, 5), 16u);
    kernel.family = PhysicalFamily::AlignedK;
    for (const uint64_t packet : {uint64_t{2}, uint64_t{8}}) {
        kernel.a = packet;
        EXPECT_EQ(RK::rkPassInputAlignment(kernel, 64), packet);
    }
    kernel.family = PhysicalFamily::RuntimeK;
    for (const RuntimeLayout layout :
         {RuntimeLayout::PacketComplete, RuntimeLayout::RotatedPrefix, RuntimeLayout::RotatedExact}) {
        kernel.a = static_cast<uint64_t>(layout);
        EXPECT_EQ(RK::rkPassInputAlignment(kernel, 101), 1u);
    }
}

// Off a 16-byte boundary every row group starts with a prefix of up to one packet less an element, so a window of
// 32 / outputs packets holds that many elements fewer.
TEST(ReducerFp8Alignment, OneWindowHoldsAPrefixBelowSixteenBytes) {
    const uint64_t outputs[4] = {1, 2, 4, 8};
    const uint64_t fp8[4] = {497, 241, 113, 49};
    const uint64_t f16[4] = {249, 121, 57, 25};
    const uint64_t fp32[4] = {125, 61, 29, 13};
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(RK::oneWindowUnalignedCapacity(outputs[i], 16, 1), fp8[i]);
        EXPECT_EQ(RK::oneWindowUnalignedCapacity(outputs[i], 16, 2), f16[i]);
        EXPECT_EQ(RK::oneWindowUnalignedCapacity(outputs[i], 16, 4), fp32[i]);
    }
}

TEST(ReducerFp8Alignment, ArgPassesNameTheAlignmentTheyRead) {
    Internal::Fp8ArgPhysicalPlan plan;
    // The OneWindow twin reads any address while K fits its window with a prefix (O8 P16: 49) and the input 2^32 - 16
    // bytes; 16 bytes past either, and for a plan without its unit or geometry.
    plan.family = PhysicalFamily::OneWindow;
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 16u);
    plan.unit = {8, 16, 2, false};
    plan.outer_size = 1000;
    plan.reduction_size = 8;
    plan.inner_size = 49;
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 1u);
    plan.inner_size = 50;
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 16u);
    plan.inner_size = 5;
    plan.outer_size = 107374182;  // x 8 x 5 = 2^32 - 16 bytes, the most an unaligned base holds
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 1u);
    plan.outer_size += 1;
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 16u);
    plan = Internal::Fp8ArgPhysicalPlan{};
    plan.family = PhysicalFamily::DeepRSmallK;
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 16u);
    plan.family = PhysicalFamily::AlignedK;
    for (const uint32_t packet : {2u, 8u}) {
        plan.packet_bytes = packet;
        EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), packet);
    }
    plan.family = PhysicalFamily::RuntimeK;
    plan.packet_bytes = Internal::FP8_RUNTIME_K_ARG_PACKET_BYTES;
    plan.tile_components = 0;  // rotated
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 1u);
    plan.tile_components = 64;  // the PacketComplete twin (any address)
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 1u);
    plan.kr.group_lanes = 4;  // R / KR
    EXPECT_EQ(Internal::retainedFp8ArgPassInputAlignment(plan), 1u);
}

// A PacketComplete tile may be narrower than a packet (the proposal's split of a K the selector reads below 16 bytes);
// it still needs ceil(K / tile) tiles.
TEST(ReducerFp8Alignment, PacketCompleteTilesMayBeNarrowerThanAPacket) {
    const auto packet = [](uint64_t outer, uint64_t r, uint64_t k, uint64_t tile) {
        PhysicalKernel kernel{};
        kernel.family = PhysicalFamily::RuntimeK;
        kernel.a = static_cast<uint64_t>(RuntimeLayout::PacketComplete);
        kernel.b = 1;
        kernel.c = tile;
        kernel.d = ceilDiv(k, tile);
        kernel.e = r;
        kernel.f = 1;
        kernel.resident_ctas_per_sm = 4;
        kernel.first_stage_ctas = ceilDiv(outer * kernel.d, 8);
        return kernel;
    };
    EXPECT_TRUE(RK::structurallyValid(packet(4, 8, 5, 5), 4, 8, 5));
    EXPECT_TRUE(RK::structurallyValid(packet(4, 8, 1, 1), 4, 8, 1));
    EXPECT_TRUE(RK::structurallyValid(packet(4, 8, 5, 4), 4, 8, 5));
    EXPECT_TRUE(RK::structurallyValid(packet(3, 2, 17, 9), 3, 2, 17));     // K 17 in tiles of 9
    EXPECT_TRUE(RK::structurallyValid(packet(3, 2, 130, 15), 3, 2, 130));
    EXPECT_TRUE(RK::structurallyValid(packet(4, 8, 20, 16), 4, 8, 20));
    PhysicalKernel wrong_tiles = packet(4, 8, 20, 10);
    wrong_tiles.d = 3;  // 20 / 10 is two tiles
    EXPECT_FALSE(RK::structurallyValid(wrong_tiles, 4, 8, 20));
    EXPECT_FALSE(RK::structurallyValid(packet(4, 8, 600, 497), 4, 8, 600));  // past the 496 cap
}

// The FP8 ARG OneWindow kernel takes its windows from the base address. Each unit that holds K with a prefix, at every
// warps-per-output it supports, gives at a misaligned base exactly what it gives at an aligned one (ties included: the
// values are small integers, so every component has many equal rows and the lowest must win).
TEST(ReducerFp8Alignment, ArgOneWindowReadsEveryByteOffset) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937_64 rng(1383);
    constexpr uint64_t Outer = 37;
    uint64_t launches = 0;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const uint64_t r : {uint64_t{1}, uint64_t{13}, uint64_t{70}}) {
            for (const uint64_t k : {uint64_t{1}, uint64_t{5}, uint64_t{16}, uint64_t{33}, uint64_t{49}, uint64_t{113},
                                     uint64_t{240}, uint64_t{497}}) {
                const std::vector<uint64_t> dims{Outer, r, k};
                const std::vector<float> values = smallIntegers(rng, product(dims));
                const ReductionGeometry geometry = ArgReducer::analyzeDenseGeometry(dims, {1});
                std::vector<Tensor> storages;
                const StridedSpan aligned = misaligned(values, dims, dtype, 0, storages, stream);
                for (const Internal::OneWindowUnit& unit : Internal::fp8OneWindowUnits(k)) {
                    if (k > RK::oneWindowUnalignedCapacity(unit.outputs_per_warp, unit.packet_bytes, 1)) continue;
                    for (const uint32_t w : {1u, 2u, 4u, 8u}) {
                        if (!Internal::fp8OneWindowArgSupports(Outer, r, k, unit, w)) continue;
                        Tensor want_value(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, k}));
                        Tensor want_index(gpuPlacement, TensorDescriptor(DataType::UINT32, {Outer, k}));
                        Internal::launchFp8OneWindowArg(aligned, outputSpan(want_value),
                                                        outputSpan(want_index), geometry, unit, w, true, 1,
                                                        stream);
                        stream.synchronize();
                        const std::vector<float> value_reference = copyGpuTensorAsFloat(want_value, stream);
                        const std::vector<uint64_t> index_reference = copyGpuTensorAsUnsigned(want_index, stream);
                        for (const uint64_t offset_bytes : {uint64_t{1}, uint64_t{3}, uint64_t{8}, uint64_t{15}}) {
                            SCOPED_TRACE(std::string(dtype == DataType::FP8_E4M3 ? "e4m3" : "e5m2") +
                                         " R=" + std::to_string(r) + " K=" + std::to_string(k) + " o" +
                                         std::to_string(unit.outputs_per_warp) + "_w" + std::to_string(w) + "_c" +
                                         std::to_string(unit.max_components_per_lane) +
                                         (unit.load_ahead ? "_la" : "") + " offset " + std::to_string(offset_bytes));
                            const StridedSpan input = misaligned(values, dims, dtype, offset_bytes, storages, stream);
                            Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, k}));
                            Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {Outer, k}));
                            Internal::launchFp8OneWindowArg(input, outputSpan(value), outputSpan(index),
                                                            geometry, unit, w, true, 1, stream);
                            stream.synchronize();
                            ASSERT_EQ(copyGpuTensorAsUnsigned(index, stream), index_reference);
                            ASSERT_EQ(copyGpuTensorAsFloat(value, stream), value_reference);
                            ++launches;
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(launches, 0u);
}

// The FP8 ARG PacketComplete twin takes its row prefixes from the base address. At every warps-per-output and the ARG
// tile of each VALUE tile, a misaligned base gives exactly the aligned launch's values and indices (small-integer
// values, so ties are everywhere and the lowest row must win). K crosses tile and packet boundaries; R includes 1.
TEST(ReducerFp8Alignment, ArgPacketCompleteReadsEveryByteOffset) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937_64 rng(1384);
    constexpr uint64_t Outer = 5;
    uint64_t launches = 0;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const uint64_t r : {uint64_t{1}, uint64_t{8}, uint64_t{11}, uint64_t{40}}) {
            for (const uint64_t k : {uint64_t{3}, uint64_t{17}, uint64_t{130}, uint64_t{256}, uint64_t{600},
                                     uint64_t{16384}}) {
                const std::vector<uint64_t> dims{Outer, r, k};
                const std::vector<float> values = smallIntegers(rng, product(dims));
                const ReductionGeometry geometry = ArgReducer::analyzeDenseGeometry(dims, {1});
                std::vector<Tensor> storages;
                const StridedSpan aligned = misaligned(values, dims, dtype, 0, storages, stream);
                std::vector<uint32_t> tiles;
                for (const uint32_t value_tile : {9u, 16u, 100u, 248u, 496u}) {
                    const uint32_t tile = Internal::fp8PacketCompleteArgTileComponents(k, value_tile);
                    if (std::find(tiles.begin(), tiles.end(), tile) == tiles.end()) tiles.push_back(tile);
                }
                for (const uint32_t tile : tiles) {
                    for (const uint32_t w : {1u, 2u, 4u, 8u}) {
                        if (!Internal::fp8PacketCompleteArgSupports(Outer, r, k, w, tile)) continue;
                        Tensor want_value(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, k}));
                        Tensor want_index(gpuPlacement, TensorDescriptor(DataType::UINT32, {Outer, k}));
                        Internal::launchFp8PacketCompleteArg(aligned, outputSpan(want_value), outputSpan(want_index),
                                                             geometry, w, tile, true, 1, stream);
                        stream.synchronize();
                        const std::vector<float> value_reference = copyGpuTensorAsFloat(want_value, stream);
                        const std::vector<uint64_t> index_reference = copyGpuTensorAsUnsigned(want_index, stream);
                        for (const uint64_t offset_bytes : {uint64_t{1}, uint64_t{3}, uint64_t{8}, uint64_t{15}}) {
                            SCOPED_TRACE(std::string(dtype == DataType::FP8_E4M3 ? "e4m3" : "e5m2") +
                                         " R=" + std::to_string(r) + " K=" + std::to_string(k) + " tile " +
                                         std::to_string(tile) + " W" + std::to_string(w) + " offset " +
                                         std::to_string(offset_bytes));
                            const StridedSpan input = misaligned(values, dims, dtype, offset_bytes, storages, stream);
                            Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {Outer, k}));
                            Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {Outer, k}));
                            Internal::launchFp8PacketCompleteArg(input, outputSpan(value), outputSpan(index), geometry,
                                                                 w, tile, true, 1, stream);
                            stream.synchronize();
                            ASSERT_EQ(copyGpuTensorAsUnsigned(index, stream), index_reference);
                            ASSERT_EQ(copyGpuTensorAsFloat(value, stream), value_reference);
                            ++launches;
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(launches, 0u);
}

TEST(ReducerFp8Alignment, MisalignedInputsUseTolerantPasses) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    std::mt19937_64 rng(1378);
    const std::vector<Shape> shapes{{{16, 3000, 64}, {1}, "rk"},
                                    {{8, 1000, 101}, {1}, "odd_k_rk"},
                                    {{2, 5000, 6}, {1}, "deep_rk"},
                                    {{5000, 300}, {1}, "kr"},
                                    {{6, 11, 30, 8, 5}, {1, 3}, "short_sites"},
                                    {{8, 600, 30, 50}, {1, 3}, "two_axes"}};
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const uint64_t offset_bytes : {uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{8}, uint64_t{0}}) {
            const uint64_t alignment = offset_bytes == 0 ? 16 : offset_bytes & (~offset_bytes + 1);
            for (const Shape& shape : shapes) {
                SCOPED_TRACE(std::string(dtype == DataType::FP8_E4M3 ? "e4m3 " : "e5m2 ") + shape.name + " offset " +
                             std::to_string(offset_bytes));
                std::vector<Tensor> storages;
                const std::vector<float> values = smallIntegers(rng, product(shape.dims));
                const StridedSpan input = misaligned(values, shape.dims, dtype, offset_bytes, storages, stream);
                ASSERT_EQ(alignment, input.alignmentBytes(16));
                for (const bool maximum : {false, true}) {
                    SCOPED_TRACE(maximum ? "max" : "sum");
                    auto stamped =
                        Reducer(maximum ? ReductionOp::Max : ReductionOp::Sum, shape.axes, DataType::FP32)
                            .stamp(input, stream);
                    const std::vector<DensePhysicalPassDiagnostic> passes = stamped->getDensePhysicalPassDiagnostics();
                    ASSERT_FALSE(passes.empty());
                    const DensePhysicalPassDiagnostic& first = passes.front();
                    if (first.family.rfind("retained_fp8_rk:", 0) == 0) {
                        // OneWindow is offered below 16 bytes only where K fits its window with a prefix.
                        const bool tolerant = alignment == 16 || first.family == "retained_fp8_rk:runtime_k" ||
                                              first.family == "retained_fp8_rk:one_window" ||
                                              (first.family == "retained_fp8_rk:aligned_k" && first.packet_bytes &&
                                               *first.packet_bytes <= alignment);
                        EXPECT_TRUE(tolerant) << first.family;
                    }
                    stamped->run();
                    stream.synchronize();
                    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                                          reference(values, shape, maximum));
                }
            }
        }
    }
}

TEST(ReducerFp8Alignment, ShortLastSitesRunAtEveryAlignment) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937_64 rng(13781);
    for (const Shape& shape : {Shape{{40, 12, 64}, {1}, "even_k"}, Shape{{40, 12, 5}, {1}, "odd_k"}}) {
        for (const uint64_t offset_bytes : {uint64_t{1}, uint64_t{3}, uint64_t{2}}) {
            SCOPED_TRACE(std::string(shape.name) + " offset " + std::to_string(offset_bytes));
            std::vector<Tensor> storages;
            const std::vector<float> values = smallIntegers(rng, product(shape.dims));
            const StridedSpan input =
                misaligned(values, shape.dims, DataType::FP8_E4M3, offset_bytes, storages, stream);
            for (const ReductionOp op : {ReductionOp::Sum, ReductionOp::Max}) {
                auto stamped = Reducer(op, shape.axes, DataType::FP32).stamp(input, stream);
                const std::vector<DensePhysicalPassDiagnostic> passes = stamped->getDensePhysicalPassDiagnostics();
                ASSERT_EQ(passes.size(), 1u);
                // R = 12: no rotated pass; OneWindow, PacketComplete, or AlignedK's 2-byte packet at an even base and
                // K.
                const bool tolerant = passes.front().family == "retained_fp8_rk:runtime_k" ||
                                      passes.front().family == "retained_fp8_rk:one_window" ||
                                      (offset_bytes % 2 == 0 && shape.dims[2] % 2 == 0 &&
                                       passes.front().family == "retained_fp8_rk:aligned_k" &&
                                       passes.front().packet_bytes == std::optional<size_t>(2));
                EXPECT_TRUE(tolerant) << passes.front().family;
                stamped->run();
                stream.synchronize();
                expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                                      reference(values, shape, op == ReductionOp::Max));
            }
        }
    }
}

TEST(ReducerFp8Alignment, DISABLED_TolerantPassTiming) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937_64 rng(13782);
    // Sites with R <= 32 (no rotated pass), then R > 32 sites. About 1-64 MiB of FP8 (x2 FP16, x4 FP32).
    const std::vector<Shape> shapes{{{64, 8, 64, 16, 32}, {1, 3}, "short_short_8m"},
                                    {{512, 8, 256, 4, 16}, {1, 3}, "short_short_64m"},
                                    {{64, 30, 1024, 8, 4}, {1, 3}, "r30_r8_64m"},
                                    {{2048, 16, 32, 64}, {1, 3}, "r16_kr64_64m"},
                                    {{8000, 8, 5, 40, 4}, {1, 3}, "r8_k800_49m"},
                                    {{4, 11, 30000, 8, 5}, {1, 3}, "r11_r8_k5_50m"},
                                    {{8, 600, 2, 4, 512}, {1, 3}, "r600_r4_9m"},
                                    {{16, 3000, 64}, {1}, "rk_3m"},
                                    {{256, 1000, 256}, {1}, "rk_63m"},
                                    {{8, 1000, 101}, {1}, "odd_k_rk_1m"},
                                    {{262144, 16, 16}, {1}, "r16_k16_64m"},
                                    {{1048576, 8, 5}, {1}, "r8_k5_40m"}};
    constexpr uint64_t RingBytes = uint64_t{512} << 20;
    constexpr int Iterations = 60;
    cudaEvent_t begin, end;
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&begin));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&end));
    const auto time_us = [&](auto&& launch) {
        for (int i = 0; i < 4; ++i) launch(i);
        EXPECT_EQ(cudaSuccess, cudaEventRecord(begin, stream.getStream()));
        for (int i = 0; i < Iterations; ++i) launch(i);
        EXPECT_EQ(cudaSuccess, cudaEventRecord(end, stream.getStream()));
        EXPECT_EQ(cudaSuccess, cudaEventSynchronize(end));
        float ms = 0;
        EXPECT_EQ(cudaSuccess, cudaEventElapsedTime(&ms, begin, end));
        return 1000.0 * ms / Iterations;
    };
    // FP8 E4M3, FP16 and FP32, each one element off a 16-byte boundary (1, 2 and 4 bytes) against aligned.
    // SUM, then ARGMAX (default outputs: value in the input's dtype, UINT32 index) of the same inputs.
    std::cout << "# alignment_timing,dtype,shape,bytes,copies,offset,aligned_us,odd_us,odd_over_aligned,odd_tbps,"
                 "copy_us,odd_of_copy,aligned_chain,odd_chain\n";
    const auto join = [](const std::vector<std::string>& names) {
        std::string chain;
        for (const std::string& name : names) chain += (chain.empty() ? "" : ">") + name;
        return chain.empty() ? std::string("-") : chain;
    };
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP16, DataType::FP32}) {
        const uint64_t element_bytes = dtype == DataType::FP32 ? 4 : dtype == DataType::FP16 ? 2 : 1;
        const char* dtype_name = dtype == DataType::FP32 ? "fp32" : dtype == DataType::FP16 ? "fp16" : "e4m3";
        for (const Shape& shape : shapes) {
            const uint64_t elements = product(shape.dims);
            const uint64_t bytes = elements * element_bytes;
            const uint64_t copies = std::max<uint64_t>(2, ceilDiv(RingBytes, bytes));
            const Tensor packed = makeGpuTensor(smallIntegers(rng, elements), {elements}, stream, dtype);
            std::vector<Tensor> storages;
            std::vector<StridedSpan> inputs[2];
            for (const int odd : {0, 1})
                for (uint64_t c = 0; c < copies; ++c)
                    inputs[odd].push_back(
                        misalignedCopy(packed, shape.dims, dtype, odd ? element_bytes : 0, storages, stream));
            stream.synchronize();
            Tensor destination(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
            const double copy_us = time_us([&](int i) {
                EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(destination.getMemPtr<void>(),
                                                       inputs[1][static_cast<size_t>(i) % copies].getMemPtr<void>(),
                                                       bytes, cudaMemcpyDeviceToDevice, stream.getStream()));
            });
            for (const bool arg : {false, true}) {
                double us[2] = {0, 0};
                std::string chain[2];
                for (const int odd : {0, 1}) {
                    // One stamp per ring copy; each run is the stamp of the next copy.
                    std::vector<std::function<void()>> runs;
                    if (arg) {
                        std::vector<std::shared_ptr<StampedArgReducer>> stamps;
                        for (const StridedSpan& input : inputs[odd])
                            stamps.push_back(
                                ArgReducer(ArgReductionOp::ArgMax, shape.axes).stamp(input, stream));
                        chain[odd] = join(stamps.front()->getPlannedPassNames());
                        for (const auto& stamped : stamps) runs.emplace_back([stamped] { stamped->run(); });
                    } else {
                        std::vector<std::shared_ptr<StampedReducer>> stamps;
                        for (const StridedSpan& input : inputs[odd])
                            stamps.push_back(
                                Reducer(ReductionOp::Sum, shape.axes, DataType::FP32).stamp(input, stream));
                        chain[odd] = stamps.front()->getModernRKStrategyChain();
                        for (const auto& stamped : stamps) runs.emplace_back([stamped] { stamped->run(); });
                    }
                    stream.synchronize();
                    us[odd] = time_us([&](int i) { runs[static_cast<size_t>(i) % runs.size()](); });
                }
                std::cout << (arg ? "alignment_timing_arg," : "alignment_timing,") << dtype_name << ','
                          << shape.name << ',' << bytes << ',' << copies << ',' << element_bytes << ',' << us[0]
                          << ',' << us[1] << ',' << us[1] / us[0] << ',' << static_cast<double>(bytes) / us[1] / 1e6
                          << ',' << copy_us << ',' << copy_us / us[1] << ',' << chain[0] << ',' << chain[1]
                          << std::endl;
            }
        }
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
}
