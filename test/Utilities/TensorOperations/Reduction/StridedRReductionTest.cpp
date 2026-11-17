// Case C, the full reduction of a rank-1 strided view, runs StridedR
// (Utilities/TensorOperations/Reduction/Kernels/StridedReduction.h): the stride-0 closed form in FP64, else the Split
// kernel, its grid from n.
//
// The kernel cases use values every summation order adds exactly (multiples of 0.5 up to 1.5, sums below 2^23;
// products of +-1 with at most 40 factors of 2 or 0.5), so each result has one exact reference whatever the order. The
// gaps between a view's elements hold NaN, so a load off the view poisons the result.
#include "Utilities/TensorOperations/Einsum/Einsum.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StridedReduction.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
using namespace ThorImplementation::ReductionInternal;
using namespace ThorImplementation::ReductionTestSupport;

namespace {

const std::vector<ReductionOp> kOps = {ReductionOp::Sum,     ReductionOp::Min,    ReductionOp::Max,
                                       ReductionOp::Product, ReductionOp::Mean,   ReductionOp::L1Norm,
                                       ReductionOp::L2Norm,  ReductionOp::SumSquares};

std::vector<DataType> valueDtypes() {
    return {DataType::FP32, DataType::FP16, DataType::BF16
#if THOR_CUB_ENABLE_FP8_TYPES
            ,
            DataType::FP8_E4M3, DataType::FP8_E5M2
#endif
    };
}

uint64_t elementBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16:
        case DataType::BF16: return 2;
        default: return 1;
    }
}

const char* opName(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum: return "sum";
        case ReductionOp::Min: return "min";
        case ReductionOp::Max: return "max";
        case ReductionOp::Product: return "product";
        case ReductionOp::Mean: return "mean";
        case ReductionOp::L1Norm: return "l1";
        case ReductionOp::L2Norm: return "l2";
        case ReductionOp::SumSquares: return "sum_squares";
    }
    return "?";
}

uint32_t bitsOf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t hash(uint64_t i, uint32_t seed) {
    return static_cast<uint32_t>(i * 2654435761ULL) ^ (seed * 40503U) ^ static_cast<uint32_t>(i >> 32);
}

// Every op but PRODUCT: +-0.5, +-1, +-1.5 (exact in every input dtype).
float additiveValue(uint64_t i, uint32_t seed) {
    const uint32_t h = hash(i, seed);
    const float magnitude = 0.5f * static_cast<float>(1 + (h >> 7) % 3);
    return ((h >> 13) % 5 == 0) ? -magnitude : magnitude;
}

// PRODUCT: +-1, with 2 and 0.5 at 20 positions each, so every partial product is a power of two within 2^+-40.
std::vector<float> productValues(uint64_t n, uint32_t seed) {
    std::vector<float> out(n);
    for (uint64_t i = 0; i < n; ++i) out[i] = (hash(i, seed) >> 11) % 3 == 0 ? -1.0f : 1.0f;
    for (uint64_t k = 0; k < 40 && n > 1; ++k) {
        const uint64_t at = k * (n - 1) / 39;
        out[at] = std::copysign(k % 2 == 0 ? 2.0f : 0.5f, out[at]);
    }
    return out;
}

std::vector<float> viewValues(ReductionOp op, uint64_t n, uint32_t seed) {
    if (op == ReductionOp::Product) return productValues(n, seed);
    std::vector<float> out(n);
    for (uint64_t i = 0; i < n; ++i) out[i] = additiveValue(i, seed);
    return out;
}

// The exact result (every partial is exact for these values), finalized as the kernels finalize in FP32.
float reference(ReductionOp op, const std::vector<float>& v) {
    double sum = 0, abs_sum = 0, squares = 0, product = 1;
    float minimum = std::numeric_limits<float>::infinity(), maximum = -minimum;
    for (float x : v) {
        sum += x;
        abs_sum += std::fabs(x);
        squares += static_cast<double>(x) * x;
        product *= x;
        minimum = std::min(minimum, x);
        maximum = std::max(maximum, x);
    }
    switch (op) {
        case ReductionOp::Sum: return static_cast<float>(sum);
        case ReductionOp::Mean: return static_cast<float>(sum) / static_cast<float>(v.size());
        case ReductionOp::Min: return minimum;
        case ReductionOp::Max: return maximum;
        case ReductionOp::Product: return static_cast<float>(product);
        case ReductionOp::L1Norm: return static_cast<float>(abs_sum);
        case ReductionOp::SumSquares: return static_cast<float>(squares);
        case ReductionOp::L2Norm: return std::sqrt(static_cast<float>(squares));
    }
    return 0;
}

// A rank-1 view of `values` at stride s from element `offset` of a storage tensor whose other elements are NaN.
struct StridedInput {
    Tensor storage;
    StridedSpan view;
};

StridedInput makeStrided(const std::vector<float>& values, uint64_t stride, uint64_t offset, DataType dtype,
                         Stream& stream) {
    const uint64_t n = values.size();
    const uint64_t count = offset + (n - 1) * stride + 1;
    std::vector<float> host(count, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t i = 0; i < n; ++i) host[offset + i * stride] = values[i];
    StridedInput out;
    out.storage = makeGpuTensor(host, {count}, stream, dtype);
    out.view = StridedSpan(out.storage, offset, {n}, {stride});
    return out;
}

float runLaunch(ReductionOp op, const StridedRLaunch& launch, const StridedSpan& view, Stream& stream,
                float scale = 1.0f) {
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor workspace;
    if (launch.workspace_bytes != 0)
        workspace = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
    const StridedSpan scratch = workspace.isInitialized() ? StridedSpan(workspace) : StridedSpan();
    stridedRInitWorkspace(scratch, stream);  // Split's ticket starts at 0
    launchStridedR(op, launch, view, StridedSpan(output), scratch, scale, stream);
    return copyGpuTensorAsFloat(output, stream).front();
}

void expectSameFloat(float got, float expected, const std::string& what) {
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(got)) << what << ": got " << got;
    } else {
        EXPECT_EQ(bitsOf(got), bitsOf(expected)) << what << ": got " << got << ", expected " << expected;
    }
}

// The launches every kernel case runs: Split at forced grids (one CTA, a few, one wave, the persistent grid, the
// production grid, and more CTAs than elements).
std::vector<StridedRLaunch> forcedLaunches(uint64_t n) {
    std::vector<StridedRLaunch> out;
    for (uint32_t grid : {1U, 2U, 7U, 170U, stridedRPersistentGrid(0), stridedRSplitGrid(n, 0)})
        out.push_back(stridedRSplitLaunch(grid));
    if (n < 10000) out.push_back(stridedRSplitLaunch(static_cast<uint32_t>(n + 5)));  // CTAs past the elements
    return out;
}

}  // namespace

// ---- The stride-0 closed form.

TEST(StridedRReduction, ClosedFormEvaluatesEveryOperationInFp64WithOneRounding) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Case {
        float x;
        uint64_t n;
        std::vector<float> expected;  // in kOps order: SUM MIN MAX PRODUCT MEAN L1 L2 SumSquares
    };
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<Case> cases = {
        {1.5f, 7, {10.5f, 1.5f, 1.5f, 2187.0f / 128.0f, 1.5f, 10.5f, static_cast<float>(1.5 * std::sqrt(7.0)), 15.75f}},
        // PRODUCT's sign follows the parity of n past 2^24 (an FP32 count could not tell 2^24 + 1 from 2^24).
        {-1.0f, (uint64_t{1} << 24) + 1,
         {-16777216.0f, -1.0f, -1.0f, -1.0f, -1.0f, 16777216.0f, 4096.0f, 16777216.0f}},
        {-1.0f, (uint64_t{1} << 24) + 2,
         {-16777218.0f, -1.0f, -1.0f, 1.0f, -1.0f, 16777218.0f, static_cast<float>(std::sqrt(16777218.0)),
          16777218.0f}},
        {-1.0f, (uint64_t{1} << 33) + 1,
         {static_cast<float>(-8589934593.0), -1.0f, -1.0f, -1.0f, -1.0f, static_cast<float>(8589934593.0),
          static_cast<float>(std::sqrt(8589934593.0)), static_cast<float>(8589934593.0)}},
        // MEAN is x where x n is past FP32's range (a summation would overflow to inf before dividing).
        {std::ldexp(1.0f, 100), uint64_t{1} << 30,
         {inf, std::ldexp(1.0f, 100), std::ldexp(1.0f, 100), inf, std::ldexp(1.0f, 100), inf, std::ldexp(1.0f, 115),
          inf}},
        {nan, 5, {nan, nan, nan, nan, nan, nan, nan, nan}},
        // -0: the additive results are the summation's +0; MIN / MAX keep -0; PRODUCT's sign follows the parity.
        {-0.0f, 3, {0.0f, -0.0f, -0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
        {-0.0f, 4, {0.0f, -0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
        {-inf, 4, {-inf, -inf, -inf, inf, -inf, inf, inf, inf}},
    };
    for (const Case& c : cases) {
        const Tensor storage = makeGpuTensor({c.x}, {1}, stream);
        const StridedSpan view(storage, 0, {c.n}, {0});
        for (size_t o = 0; o < kOps.size(); ++o) {
            const std::string what = std::string(opName(kOps[o])) + " x=" + std::to_string(c.x) +
                                     " n=" + std::to_string(c.n);
            auto stamped = Reducer(kOps[o], 0, DataType::FP32).stamp(view, stream);
            EXPECT_EQ(stamped->getPath(), ReductionLayout::SingleOutput);
            ASSERT_TRUE(stamped->getStridedRLaunch().has_value());
            EXPECT_EQ(stamped->getStridedRLaunch()->family, StridedRFamily::ClosedForm);
            EXPECT_EQ(stamped->getModernRKStrategyChain(), "strided_r:closed_form");
            EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 0U);
            stamped->run();
            expectSameFloat(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front(), c.expected[o], what);
        }
    }
}

TEST(StridedRReduction, ClosedFormReadsEveryInputDtypeAndAppliesTheRuntimeScale) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (DataType dtype : valueDtypes()) {
        for (const float x : {1.5f, -0.5f}) {
            const Tensor storage = makeGpuTensor({x}, {1}, stream, dtype);
            const StridedSpan view(storage, 0, {1000}, {0});
            for (ReductionOp op : kOps) {
                const std::string what = std::string(opName(op)) + " dtype " + std::to_string(static_cast<int>(dtype));
                auto stamped = Reducer(op, 0, DataType::FP32).stamp(view, stream);
                stamped->run(-2.0f);
                // The exact value in FP64 (PRODUCT: 1.5^1000 is finite in FP64, inf in FP32; 0.5^1000 is +0 in FP32).
                const double n = 1000.0;
                double exact = 0;
                switch (op) {
                    case ReductionOp::Sum: exact = x * n; break;
                    case ReductionOp::Mean:
                    case ReductionOp::Min:
                    case ReductionOp::Max: exact = x; break;
                    case ReductionOp::Product: exact = std::pow(static_cast<double>(x), 1000.0); break;
                    case ReductionOp::L1Norm: exact = std::fabs(x) * n; break;
                    case ReductionOp::SumSquares: exact = static_cast<double>(x) * x * n; break;
                    case ReductionOp::L2Norm: exact = std::fabs(x) * std::sqrt(n); break;
                }
                expectSameFloat(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front(),
                                static_cast<float>(exact) * -2.0f, what);
            }
        }
    }
}

// ---- The StridedR kernel.

TEST(StridedRReduction, SplitMatchesExactReferencesAcrossStridesSizesDtypesAndOps) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Shape {
        uint64_t n, stride, offset;
    };
    for (DataType dtype : valueDtypes()) {
        const uint64_t eb = elementBytes(dtype);
        std::vector<Shape> shapes = {{2, 2, 0},       {5, 3, 1},       {33, 7, 0},      {4095, 2, 3},
                                     {4097, 3, 0},    {4097, 4096, 0}, {70001, 5, 2},   {300001, 2, 0},
                                     {1000003, 3, 1}};
        // Pitches around 16, 32 and 64 bytes.
        for (uint64_t pitch : {16U, 32U, 64U}) {
            for (int64_t delta : {-1, 0, 1}) {
                const uint64_t s = static_cast<uint64_t>(static_cast<int64_t>(pitch / eb) + delta);
                if (s >= 2) shapes.push_back({9001, s, 1});
            }
        }
        for (const Shape& shape : shapes) {
            const uint32_t seed = static_cast<uint32_t>(shape.stride);
            const std::vector<float> additive = viewValues(ReductionOp::Sum, shape.n, seed);
            const std::vector<float> product = viewValues(ReductionOp::Product, shape.n, seed);
            const StridedInput additive_input = makeStrided(additive, shape.stride, shape.offset, dtype, stream);
            const StridedInput product_input = makeStrided(product, shape.stride, shape.offset, dtype, stream);
            for (ReductionOp op : kOps) {
                const bool is_product = op == ReductionOp::Product;
                const StridedInput& input = is_product ? product_input : additive_input;
                const float expected = reference(op, is_product ? product : additive);
                for (const StridedRLaunch& launch : forcedLaunches(shape.n)) {
                    const std::string what = std::string(opName(op)) + " dtype " +
                                             std::to_string(static_cast<int>(dtype)) + " n " + std::to_string(shape.n) +
                                             " s " + std::to_string(shape.stride) + " " + stridedRLaunchName(launch);
                    expectSameFloat(runLaunch(op, launch, input.view, stream), expected, what);
                }
            }
        }
    }
}

TEST(StridedRReduction, NanPropagatesAndMinMaxOrderSignedZeros) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t n = 5000;
    for (uint64_t at : {uint64_t{0}, n / 2, n - 1}) {
        std::vector<float> values = viewValues(ReductionOp::Sum, n, 3);
        values[at] = std::numeric_limits<float>::quiet_NaN();
        const StridedInput input = makeStrided(values, 3, 0, DataType::FP32, stream);
        for (ReductionOp op : kOps) {
            for (const StridedRLaunch& launch : forcedLaunches(n)) {
                EXPECT_TRUE(std::isnan(runLaunch(op, launch, input.view, stream)))
                    << opName(op) << " NaN at " << at << " " << stridedRLaunchName(launch);
            }
        }
    }
    // min.NaN orders -0 below +0: MIN of a mix is -0, MAX of a mix is +0; a view of one sign keeps it.
    struct Zeros {
        std::vector<float> values;
        float minimum, maximum;
    };
    std::vector<float> mixed(3001, 0.0f);
    for (size_t i = 0; i < mixed.size(); i += 7) mixed[i] = -0.0f;
    const std::vector<Zeros> zeros = {{mixed, -0.0f, 0.0f},
                                      {std::vector<float>(3001, -0.0f), -0.0f, -0.0f},
                                      {std::vector<float>(3001, 0.0f), 0.0f, 0.0f}};
    for (const Zeros& z : zeros) {
        const StridedInput input = makeStrided(z.values, 5, 0, DataType::FP32, stream);
        for (const StridedRLaunch& launch : forcedLaunches(z.values.size())) {
            expectSameFloat(runLaunch(ReductionOp::Min, launch, input.view, stream), z.minimum,
                            "min " + stridedRLaunchName(launch));
            expectSameFloat(runLaunch(ReductionOp::Max, launch, input.view, stream), z.maximum,
                            "max " + stridedRLaunchName(launch));
        }
    }
}

TEST(StridedRReduction, SplitIsBitIdenticalFromRunToRun) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // Values whose sum rounds: the bits depend on the order, which is fixed for one launch.
    const uint64_t n = (uint64_t{1} << 20) + 7;
    std::vector<float> values(n);
    for (uint64_t i = 0; i < n; ++i) values[i] = static_cast<float>(hash(i, 9) % 100003) / 77777.0f - 0.6f;
    const StridedInput input = makeStrided(values, 3, 0, DataType::FP32, stream);
    for (ReductionOp op : {ReductionOp::Sum, ReductionOp::Mean, ReductionOp::L2Norm}) {
        const StridedRLaunch split = stridedRSplitLaunch(stridedRPersistentGrid(0));
        const uint32_t first = bitsOf(runLaunch(op, split, input.view, stream));
        for (int repeat = 0; repeat < 4; ++repeat)
            EXPECT_EQ(bitsOf(runLaunch(op, split, input.view, stream)), first) << opName(op);

        auto stamped = Reducer(op, 0, DataType::FP32).stamp(input.view, stream);
        ASSERT_TRUE(stamped->getStridedRLaunch().has_value());
        EXPECT_EQ(stamped->getStridedRLaunch()->family, StridedRFamily::Split);
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), stamped->getStridedRLaunch()->workspace_bytes);
        stamped->run();
        const uint32_t stamped_first = bitsOf(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front());
        for (int repeat = 0; repeat < 4; ++repeat) {
            stamped->run();
            EXPECT_EQ(bitsOf(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front()), stamped_first);
        }
    }
}

// Split is one launch. One workspace, initialized once, serves launch after launch (each leaves the ticket at 0), at
// grids of one CTA, fewer CTAs than threads, and more CTAs than the finishing CTA has threads. The operations alternate
// so a launch that did not finish would leave the previous operation's value in the output.
TEST(StridedRReduction, SplitFinishesInItsOwnLaunchOnAReusedWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t n = 70001;
    const std::vector<float> values = viewValues(ReductionOp::Sum, n, 5);
    const StridedInput input = makeStrided(values, 5, 2, DataType::FP32, stream);
    for (uint32_t grid : {1U, 3U, 137U, 512U, 513U, 2000U}) {
        const StridedRLaunch launch = stridedRSplitLaunch(grid);
        EXPECT_EQ(launch.workspace_bytes, 4U * (uint64_t{grid} + 1));
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
        stridedRInitWorkspace(StridedSpan(workspace), stream);
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (ReductionOp op : {ReductionOp::Sum, ReductionOp::Max, ReductionOp::SumSquares,
                                   ReductionOp::Min, ReductionOp::Mean}) {
                launchStridedR(op, launch, input.view, StridedSpan(output), StridedSpan(workspace), 1.0f, stream);
                expectSameFloat(copyGpuTensorAsFloat(output, stream).front(), reference(op, values),
                                std::string(opName(op)) + " " + stridedRLaunchName(launch) + " repeat " +
                                    std::to_string(repeat));
            }
        }
    }
}

// Element addresses are 64-bit: an FP16 view whose span passes 4 GiB, its unique maximum in the last element.
TEST(StridedRReduction, SpansBeyondFourGiBUse64BitAddresses) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t n = (uint64_t{1} << 20) + 3;
    const uint64_t stride = 2053;
    const uint64_t count = (n - 1) * stride + 1;
    const uint64_t bytes = 2 * count;
    ASSERT_GT(bytes, uint64_t{1} << 32);
    size_t free_bytes = 0, total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    if (free_bytes < bytes + (uint64_t{1} << 29)) GTEST_SKIP() << "needs " << bytes << " bytes of free device memory";

    std::vector<float> values = viewValues(ReductionOp::Sum, n, 5);
    values.back() = 3.0f;
    std::vector<__half> host(n);
    for (uint64_t i = 0; i < n; ++i) host[i] = __float2half_rn(values[i]);
    Tensor storage(gpuPlacement, TensorDescriptor(DataType::FP16, {count}));
    ASSERT_EQ(cudaMemsetAsync(storage.getMemPtr<void>(), 0xff, bytes, stream.getStream()), cudaSuccess);  // FP16 NaN
    ASSERT_EQ(cudaMemcpy2DAsync(storage.getMemPtr<void>(), 2 * stride, host.data(), 2, 2, n, cudaMemcpyHostToDevice,
                                stream.getStream()),
              cudaSuccess);
    stream.synchronize();
    const StridedSpan view(storage, 0, {n}, {stride});

    for (ReductionOp op : {ReductionOp::Sum, ReductionOp::Max, ReductionOp::L1Norm}) {
        auto stamped = Reducer(op, 0, DataType::FP32).stamp(view, stream);
        ASSERT_TRUE(stamped->getStridedRLaunch().has_value());
        stamped->run();
        expectSameFloat(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front(), reference(op, values),
                        opName(op));
    }
}

// ---- Routing, the stamp and the launch choice.

TEST(StridedRReduction, StampsRouteEveryOpAndDtypeToStridedR) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (DataType dtype : valueDtypes()) {
        for (ReductionOp op : kOps) {
            const std::vector<float> values = viewValues(op, 9001, 11);
            const StridedInput input = makeStrided(values, 3, 1, dtype, stream);  // offset 1: not 16-byte aligned
            const std::string what = std::string(opName(op)) + " dtype " + std::to_string(static_cast<int>(dtype));
            auto stamped = Reducer(op, 0, DataType::FP32).stamp(input.view, stream);
            EXPECT_EQ(stamped->getPath(), ReductionLayout::SingleOutput) << what;
            EXPECT_TRUE(stamped->getGeometry().device_transform_uses_affine_stride) << what;
            EXPECT_FALSE(stamped->usesDenseExecutablePlan()) << what;
            ASSERT_TRUE(stamped->getStridedRLaunch().has_value()) << what;
            const StridedRLaunch launch = *stamped->getStridedRLaunch();
            EXPECT_NE(launch.family, StridedRFamily::ClosedForm) << what;
            EXPECT_EQ(stamped->getModernRKStrategyChain(), stridedRLaunchName(launch)) << what;
            EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), launch.workspace_bytes) << what;
            stamped->run();
            const float expected = reference(op, values);
            expectSameFloat(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front(), expected, what);
            stamped->run(0.5f);  // a runtime scale (MAX's folded sign included)
            expectSameFloat(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front(), expected * 0.5f, what);
        }
    }
    // A BF16 output stores the FP32 result rounded once.
    const std::vector<float> values = viewValues(ReductionOp::Sum, 9001, 11);
    const StridedInput input = makeStrided(values, 3, 0, DataType::FP16, stream);
    auto stamped = Reducer(ReductionOp::Sum, 0, DataType::BF16).stamp(input.view, stream);
    stamped->run();
    EXPECT_EQ(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream).front(),
              __bfloat162float(__float2bfloat16_rn(reference(ReductionOp::Sum, values))));
}

TEST(StridedRReduction, EinsumTraceRunsStridedR) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t n = 3001;
    std::vector<float> matrix(n * n, 0.25f);
    std::vector<float> diagonal(n);
    for (uint64_t i = 0; i < n; ++i) matrix[i * n + i] = diagonal[i] = additiveValue(i, 17);
    const Tensor input = makeGpuTensor(matrix, {n, n}, stream);
    auto einsum = Einsum("ii->").stamp({input}, stream);
    ASSERT_TRUE(einsum->getStandaloneReductionPath().has_value());
    EXPECT_EQ(einsum->getStandaloneReductionPath().value(), ReductionLayout::SingleOutput);
    einsum->run();
    EXPECT_EQ(copyGpuTensorAsFloat(einsum->getOutputTensor(), stream).front(),
              reference(ReductionOp::Sum, diagonal));
}

TEST(StridedRReduction, TheLaunchFollowsFromTheViewWithNothingTimed) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // s > 0: Split at one CTA per 512 elements, at most the persistent grid; its workspace is G partials and the ticket.
    const uint32_t persistent = stridedRPersistentGrid(0);
    for (uint64_t n : {uint64_t{1}, uint64_t{512}, uint64_t{513}, uint64_t{12347}, uint64_t{1} << 22}) {
        const std::vector<float> values = viewValues(ReductionOp::Sum, n, 2);
        const StridedInput input = makeStrided(values, 3, 1, DataType::BF16, stream);
        const StridedRLaunch launch = stridedRLaunchFor(input.view, DataType::FP32, 0);
        const uint32_t expected_grid = static_cast<uint32_t>(std::min<uint64_t>(persistent, (n + 511) / 512));
        EXPECT_EQ(launch.family, StridedRFamily::Split) << n;
        EXPECT_EQ(launch.grid, expected_grid) << n;
        EXPECT_EQ(launch.workspace_bytes, 4U * (uint64_t{expected_grid} + 1)) << n;
        EXPECT_EQ(launch, stridedRSplitLaunch(stridedRSplitGrid(n, 0))) << n;
        auto stamped = Reducer(ReductionOp::Max, 0, DataType::FP32).stamp(input.view, stream);
        if (n == 1) {
            // One element at any stride is a compact permutation (an extent-one axis may carry any stride), so the
            // dense planner takes it as a dense one-element tensor and StridedR never sees it.
            EXPECT_FALSE(stamped->getStridedRLaunch().has_value());
            EXPECT_EQ(stamped->getModernRKStrategyChain().rfind("view_physical:", 0), 0U)
                << stamped->getModernRKStrategyChain();
            continue;
        }
        ASSERT_TRUE(stamped->getStridedRLaunch().has_value()) << n;
        EXPECT_EQ(*stamped->getStridedRLaunch(), launch) << n;
        EXPECT_EQ(stamped->getModernRKStrategyChain(), "strided_r:split_g" + std::to_string(expected_grid)) << n;
    }
    EXPECT_EQ(stridedRSplitGrid(uint64_t{1} << 30, 0), persistent);

    // s == 0: the closed form, no workspace.
    const Tensor scalar = makeGpuTensor({2.0f}, {1}, stream);
    const StridedSpan broadcast_view(scalar, 0, {99}, {0});
    const StridedRLaunch closed = stridedRLaunchFor(broadcast_view, DataType::FP32, 0);
    EXPECT_EQ(closed.family, StridedRFamily::ClosedForm);
    EXPECT_EQ(closed.workspace_bytes, 0U);
    auto broadcast = Reducer(ReductionOp::Sum, 0, DataType::FP32).stamp(broadcast_view, stream);
    ASSERT_TRUE(broadcast->getStridedRLaunch().has_value());
    EXPECT_EQ(*broadcast->getStridedRLaunch(), closed);
    EXPECT_EQ(broadcast->getWorkspaceSizeInBytes(), 0U);

    // Spans StridedR does not take.
    const Tensor ints = makeGpuTensor(std::vector<float>(8, 1.0f), {8}, stream, DataType::FP32);
    EXPECT_THROW(static_cast<void>(stridedRLaunchFor(StridedSpan(ints, 0, {4}, {2}), DataType::INT32, 0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(stridedRLaunchFor(StridedSpan(ints, 0, {2, 2}, {4, 2}), DataType::FP32, 0)),
                 std::invalid_argument);
}
