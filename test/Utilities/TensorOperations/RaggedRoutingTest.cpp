// 1325 (phase 7, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the routed CubSegmentedReduction /
// CubSegmentedArgReduction stamps. 1328 (phase 8): the switch is removed, so every stamp of a site a ragged family runs
// is routed, and these tests check them against host references (the legacy stamps they were compared with in 1325 are
// no longer reachable for these sites). A routed stamp names the ragged passes, owns the selection's workspace,
// computes the exact reference (integer inputs: SUM, MIN, MAX and ARG bit for bit, MEAN to a rounding) over every
// dtype, KR and RK D, rank-3 inputs, both offset widths and index widths, inactive capacity and empty segments, reads
// its runtime offsets at run time, and runs on replacement tensors. 1330 (phase 9): the legacy kernels are deleted; a
// site no ragged family runs throws at stamp.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

std::vector<uint64_t> randomOffsets(uint64_t active, uint64_t segments, std::mt19937_64& rng) {
    std::vector<uint64_t> offsets(segments + 1, 0);
    for (uint64_t i = 1; i < segments; ++i) offsets[i] = rng() % (active + 1);
    offsets[segments] = active;
    std::sort(offsets.begin(), offsets.end());
    for (uint64_t i = 6; i < segments; i += 6) offsets[i] = offsets[i - 1];
    return offsets;
}

// The reference: SUM / MEAN / MIN / MAX per segment and component (empty: 0, 0, +inf, -inf), in FP32.
std::vector<float> valueReference(CubReductionOp op, const std::vector<float>& values, const std::vector<uint64_t>& o,
                                  uint64_t d) {
    const uint64_t b = o.size() - 1;
    std::vector<float> out(b * d);
    for (uint64_t s = 0; s < b; ++s) {
        for (uint64_t c = 0; c < d; ++c) {
            float acc = op == CubReductionOp::Min   ? std::numeric_limits<float>::infinity()
                        : op == CubReductionOp::Max ? -std::numeric_limits<float>::infinity()
                                                    : 0.0f;
            for (uint64_t r = o[s]; r < o[s + 1]; ++r) {
                const float v = values[r * d + c];
                if (op == CubReductionOp::Min) acc = std::min(acc, v);
                else if (op == CubReductionOp::Max) acc = std::max(acc, v);
                else acc += v;
            }
            if (op == CubReductionOp::Mean) acc = o[s + 1] > o[s] ? acc / static_cast<float>(o[s + 1] - o[s]) : 0.0f;
            out[s * d + c] = acc;
        }
    }
    return out;
}

// The reference winner: the global packed index row x D + component of the first extremum, or the sentinel.
std::vector<uint64_t> argReference(CubArgReductionOp op,
                                   const std::vector<float>& values,
                                   const std::vector<uint64_t>& o,
                                   uint64_t d,
                                   DataType index) {
    const uint64_t b = o.size() - 1;
    const uint64_t sentinel = index == DataType::UINT64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    std::vector<uint64_t> out(b * d, sentinel);
    for (uint64_t s = 0; s < b; ++s) {
        for (uint64_t c = 0; c < d; ++c) {
            for (uint64_t r = o[s]; r < o[s + 1]; ++r) {
                const uint64_t i = r * d + c;
                const uint64_t& best = out[s * d + c];
                if (best == sentinel || (op == CubArgReductionOp::ArgMin ? values[i] < values[best]
                                                                          : values[i] > values[best]))
                    out[s * d + c] = i;
            }
        }
    }
    return out;
}

void expectValues(const std::vector<float>& got, const std::vector<float>& want, bool mean) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (mean) {
            ASSERT_NEAR(got[i], want[i], 1e-5f * std::max(1.0f, std::fabs(want[i]))) << i;
        } else {
            uint32_t a, b;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &want[i], 4);
            ASSERT_EQ(a, b) << i << ": " << got[i] << " vs " << want[i];
        }
    }
}

}  // namespace

TEST(RaggedRouting, RoutedStampsComputeTheReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937_64 rng(1328);
    std::vector<DataType> dtypes{DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    dtypes.push_back(DataType::FP8_E4M3);
#endif
    size_t k = 0;
    for (const DataType dtype : dtypes) {
        for (const std::vector<uint64_t>& trailing :
             std::vector<std::vector<uint64_t>>{{}, {3}, {2, 4}, {17}, {64}, {300}}) {
            const DataType offsets_dtype = k++ % 2 ? DataType::UINT64 : DataType::UINT32;
            uint64_t d = 1;
            for (uint64_t t : trailing) d *= t;
            SCOPED_TRACE(std::string(TensorDescriptor::getElementTypeName(dtype)) + " D " + std::to_string(d) +
                         (offsets_dtype == DataType::UINT64 ? " uint64 offsets" : " uint32 offsets"));
            const uint64_t capacity = d > 100 ? 400 : 3000, segments = d > 100 ? 37 : 211;
            const uint64_t active = capacity - capacity / 6;  // inactive capacity
            std::vector<uint64_t> dims{capacity};
            dims.insert(dims.end(), trailing.begin(), trailing.end());
            std::vector<float> values(capacity * d);
            for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 13) - 6);  // exact, many ties
            const std::vector<uint64_t> offsets_host = randomOffsets(active, segments, rng);
            const Tensor input = makeGpuTensor(values, dims, stream, dtype);
            const Tensor offsets = makeGpuUnsignedTensor(offsets_host, {segments + 1}, stream, offsets_dtype);
            const bool validated = d == 3;  // stamp() reads the offsets at stamp time; the rest stampRuntimeOffsets
            std::vector<uint64_t> out_dims = dims;
            out_dims[0] = segments;

            for (const CubReductionOp op :
                 {CubReductionOp::Sum, CubReductionOp::Mean, CubReductionOp::Min, CubReductionOp::Max}) {
                for (const DataType out_dtype : {DataType::FP32, DataType::BF16}) {
                    if (out_dtype == DataType::BF16 && op != CubReductionOp::Max) continue;  // a narrower output once
                    SCOPED_TRACE("op " + std::to_string(static_cast<int>(op)));
                    Tensor output(gpuPlacement, TensorDescriptor(out_dtype, out_dims));
                    const CubSegmentedReduction reduction(op, out_dtype);
                    auto stamped = validated ? reduction.stamp(input, output, offsets, stream)
                                             : reduction.stampRuntimeOffsets(input, output, offsets, stream);
                    EXPECT_TRUE(stamped->usesRetainedRagged());
                    EXPECT_EQ(stamped->getPath(), CubReductionPath::OffsetSegmented);
                    ASSERT_EQ(stamped->getPassNames().size(), 2U);
                    EXPECT_EQ(stamped->getPassNames().back(), "ragged_finish");
                    EXPECT_GT(stamped->getWorkspaceSizeInBytes(), 1U);
                    stamped->run();
                    stream.synchronize();
                    // Integer results up to a few hundred are exact in BF16 at MAX (|v| <= 6).
                    expectValues(copyGpuTensorAsFloat(output, stream), valueReference(op, values, offsets_host, d),
                                 op == CubReductionOp::Mean);
                }
            }
            for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
                for (const DataType index : {DataType::UINT32, DataType::UINT64}) {
                    SCOPED_TRACE("arg op " + std::to_string(static_cast<int>(op)) + " index " +
                                 std::to_string(static_cast<int>(index)));
                    Tensor winners(gpuPlacement, TensorDescriptor(index, out_dims));
                    const CubSegmentedArgReduction reduction(op, index);
                    auto stamped = validated ? reduction.stamp(input, winners, offsets, stream)
                                             : reduction.stampRuntimeOffsets(input, winners, offsets, stream);
                    EXPECT_TRUE(stamped->usesRetainedRagged());
                    ASSERT_EQ(stamped->getPassNames().size(), 2U);
                    EXPECT_EQ(stamped->getPassNames().back(), "ragged_finish_arg");
                    stamped->run();
                    stream.synchronize();
                    EXPECT_EQ(copyGpuTensorAsUnsigned(winners, stream),
                              argReference(op, values, offsets_host, d, index));
                }
            }
        }
    }
}

TEST(RaggedRouting, RoutedStampsReadRuntimeOffsetsAndRunOnReplacementTensors) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> a(600 * 8), b(600 * 8);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
        b[i] = static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
    }
    for (const std::vector<uint64_t>& dims : std::vector<std::vector<uint64_t>>{{600}, {600, 8}}) {
        SCOPED_TRACE(dims.size() == 1 ? "D 1 (KR)" : "D 8 (RK)");
        const uint64_t d = dims.size() == 1 ? 1 : 8;
        const std::vector<float> a_values(a.begin(), a.begin() + 600 * d), b_values(b.begin(), b.begin() + 600 * d);
        const Tensor input_a = makeGpuTensor(a_values, dims, stream);
        const Tensor input_b = makeGpuTensor(b_values, dims, stream);
        Tensor offsets = makeGpuUnsignedTensor({0, 100, 100, 400}, {4}, stream);
        std::vector<uint64_t> out_dims = dims;
        out_dims[0] = 3;
        Tensor out(gpuPlacement, TensorDescriptor(DataType::FP32, out_dims));
        auto stamped = CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32)
                           .stampRuntimeOffsets(input_a, out, offsets, stream);
        ASSERT_TRUE(stamped->usesRetainedRagged());
        stamped->run();
        stream.synchronize();
        expectValues(copyGpuTensorAsFloat(out, stream),
                     valueReference(CubReductionOp::Sum, a_values, {0, 100, 100, 400}, d), false);
        // New runtime offsets, read when run.
        overwriteGpuUnsignedTensor(offsets, {0, 0, 599, 600}, stream);
        stamped->run();
        stream.synchronize();
        expectValues(copyGpuTensorAsFloat(out, stream),
                     valueReference(CubReductionOp::Sum, a_values, {0, 0, 599, 600}, d), false);
        // Replacement tensors (the benchmark hook); the bound output is left alone.
        const std::vector<float> bound = copyGpuTensorAsFloat(out, stream);
        Tensor out_b(gpuPlacement, TensorDescriptor(DataType::FP32, out_dims));
        stamped->runOnTensorsForBenchmark(input_b, offsets, out_b, stream);
        stream.synchronize();
        expectValues(copyGpuTensorAsFloat(out_b, stream),
                     valueReference(CubReductionOp::Sum, b_values, {0, 0, 599, 600}, d), false);
        expectValues(copyGpuTensorAsFloat(out, stream), bound, false);
    }
}

// 1329: rows past the old 64 MiB limit (FP32 D 2^24 + 4: 64 MiB + 16 bytes) route to ragged RK through the segmented
// stamps and compute the reference: SUM and MIN, ARGMIN with UINT32 indices, over 4 capacity rows (the last inactive
// and poisoned) in segments of 2, 0 and 1 rows. About 1 GB of GPU memory, with the contest's copies.
TEST(RaggedRouting, RowsPastTheOld64MiBLimitRouteAndComputeTheReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t d = (uint64_t{1} << 24) + 4, capacity = 4;
    const auto value = [](uint64_t row, uint64_t column) {
        return row == 3 ? -100.0f : static_cast<float>(static_cast<int>((column % 9 + 2 * row) % 9) - 4);
    };
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, {capacity, d}));
    float* host = cpu.getMemPtr<float>();
    for (uint64_t r = 0; r < capacity; ++r)
        for (uint64_t c = 0; c < d; ++c) host[r * d + c] = value(r, c);
    Tensor input(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, d}));
    input.copyFromAsync(cpu, stream);
    const Tensor offsets = makeGpuUnsignedTensor({0, 2, 2, 3}, {4}, stream);
    stream.synchronize();
    const auto download = [&](const Tensor& gpu, DataType dtype) {
        Tensor out(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(dtype, gpu.getDimensions()));
        out.copyFromAsync(gpu, stream);
        stream.synchronize();
        return out;
    };
    for (const CubReductionOp op : {CubReductionOp::Sum, CubReductionOp::Min}) {
        SCOPED_TRACE(op == CubReductionOp::Sum ? "SUM" : "MIN");
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {3, d}));
        auto stamped = CubSegmentedReduction(op, DataType::FP32).stampRuntimeOffsets(input, output, offsets, stream);
        ASSERT_TRUE(stamped->usesRetainedRagged());
        EXPECT_EQ(stamped->getPassNames()[0].rfind("ragged_rk:", 0), 0U);
        stamped->run();
        const Tensor got = download(output, DataType::FP32);
        const float* g = got.getMemPtr<float>();
        for (uint64_t c = 0; c < d; ++c) {
            const float a = value(0, c), b = value(1, c), e = value(2, c);
            const float s0 = op == CubReductionOp::Sum ? a + b : std::min(a, b);
            const float s1 = op == CubReductionOp::Sum ? 0.0f : std::numeric_limits<float>::infinity();
            ASSERT_EQ(g[c], s0) << c;
            ASSERT_EQ(g[d + c], s1) << c;
            ASSERT_EQ(g[2 * d + c], e) << c;
        }
    }
    Tensor winners(gpuPlacement, TensorDescriptor(DataType::UINT32, {3, d}));
    auto argmin = CubSegmentedArgReduction(CubArgReductionOp::ArgMin, DataType::UINT32)
                      .stampRuntimeOffsets(input, winners, offsets, stream);
    ASSERT_TRUE(argmin->usesRetainedRagged());
    argmin->run();
    const Tensor got = download(winners, DataType::UINT32);
    const uint32_t* w = got.getMemPtr<uint32_t>();
    for (uint64_t c = 0; c < d; ++c) {
        const uint64_t first = value(1, c) < value(0, c) ? d + c : c;
        ASSERT_EQ(w[c], first) << c;
        ASSERT_EQ(w[d + c], 0xFFFFFFFFU) << c;
        ASSERT_EQ(w[2 * d + c], 2 * d + c) << c;
    }
}

// 1330 (phase 9): the legacy offset-segmented kernels are deleted, so a site no ragged family runs is an error at stamp
// time, with the reason: FP32 rows of 2^29 + 4 components (2 GiB + 16 bytes), VALUE and ARG (UINT32 indices). The
// stamp throws before it reads or tunes anything; the tensors are never filled. About 6 GB of GPU memory.
TEST(RaggedRouting, RowsOver2GiBAreAnErrorAtStamp) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t d = (uint64_t{1} << 29) + 4;
    const Tensor input(gpuPlacement, TensorDescriptor(DataType::FP32, {1, d}));
    const Tensor offsets = makeGpuUnsignedTensor({0, 1}, {2}, stream);
    stream.synchronize();
    const auto expectRefused = [](const auto& stamp_it) {
        try {
            stamp_it();
            FAIL() << "the stamp must throw";
        } catch (const std::invalid_argument& error) {
            const std::string what = error.what();
            EXPECT_NE(what.find("no ragged kernel runs this site"), std::string::npos) << what;
            EXPECT_NE(what.find("2 GiB"), std::string::npos) << what;
        }
    };
    {
        const Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {1, d}));
        expectRefused([&] {
            static_cast<void>(
                CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stampRuntimeOffsets(input, output, offsets,
                                                                                                stream));
        });
    }
    const Tensor winners(gpuPlacement, TensorDescriptor(DataType::UINT32, {1, d}));
    expectRefused([&] {
        static_cast<void>(CubSegmentedArgReduction(CubArgReductionOp::ArgMax, DataType::UINT32)
                              .stampRuntimeOffsets(input, winners, offsets, stream));
    });
}
