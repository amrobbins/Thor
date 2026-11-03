// 1146-1148: retained FP8 R / KR reductions. The candidate shape generator (host), every candidate shape of the Pack,
// Group and Split VALUE and ARG kernels against host references (misaligned inputs, NaN / -0 / ties for ARG, repeated
// launches on one uninitialized Split scratch), and the production stamps that select among them.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;
namespace KR = ThorImplementation::CubReductionInternal::Fp8KR;

TEST(CubReductionFp8KR, CandidateShapesAreLegalDistinctAndOrdered) {
    const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},       {1, 7},       {1, 4096},    {1, 1ULL << 30}, {3, 2},      {1000000, 5}, {64, 31},
        {17, 129},    {8, 4100},    {2, 70001},   {4, 262147},     {170, 1024}, {100000, 64}, {1, 1023},
        {4096, 4096}, {2720, 8192}, {5000, 1000}, {1, 3ULL << 33}, {1, 1ULL << 25}, {100, 1ULL << 20},
        {11298, 46814}, {3000, 16384}, {1, 18509157}};
    for (const auto& [rows, row_bytes] : sites) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << row_bytes);
        const auto shapes = KR::fp8KRCandidateShapes(rows, row_bytes, 170);
        ASSERT_FALSE(shapes.empty());
        std::set<std::string> names;
        bool any_unsplit = false;
        for (const auto& shape : shapes) {
            EXPECT_TRUE(KR::fp8KRSegmentShapeLegal(shape));
            EXPECT_EQ(rows, shape.rows);
            EXPECT_EQ(row_bytes, shape.row_bytes);
            EXPECT_TRUE(names.insert(KR::fp8KRShapeName(shape)).second) << KR::fp8KRShapeName(shape);
            switch (shape.family()) {
                case KR::Fp8KRFamily::Split:
                    EXPECT_EQ(KR::FP8_KR_THREADS, shape.group_lanes);
                    EXPECT_LE(shape.chunks(), KR::FP8_KR_MAX_CHUNKS);
                    EXPECT_EQ((shape.segments() * 4 + 15) / 16 * 16, KR::fp8KRScratchBytes(shape, false));
                    EXPECT_EQ((shape.segments() * 8 + 15) / 16 * 16, KR::fp8KRScratchBytes(shape, true));
                    break;
                case KR::Fp8KRFamily::Pack:
                    EXPECT_LE(row_bytes, 16u);
                    EXPECT_LE(shape.rows_per_thread * row_bytes, KR::FP8_KR_MAX_PACK_SPAN_BYTES);
                    EXPECT_EQ(0u, KR::fp8KRScratchBytes(shape, false));
                    break;
                case KR::Fp8KRFamily::Group:
                    any_unsplit = true;
                    EXPECT_EQ(0u, KR::fp8KRScratchBytes(shape, false));
                    EXPECT_LE(shape.group_lanes, std::max<uint64_t>(1, 4 * ((row_bytes + 3) / 4)));
                    break;
                case KR::Fp8KRFamily::Block:
                    any_unsplit = true;
                    EXPECT_EQ(0u, KR::fp8KRScratchBytes(shape, false));
                    EXPECT_EQ(KR::FP8_KR_THREADS, shape.group_lanes);
                    EXPECT_GE(row_bytes, 2 * KR::FP8_KR_MIN_BLOCK_ROW_BYTES);
                    EXPECT_LT(row_bytes, 64 * KR::FP8_KR_MIN_BLOCK_ROW_BYTES);
                    EXPECT_GE(rows, 170u);
                    break;
            }
        }
        // Rows enough to fill the GPU: never split first; short rows lead with Pack.
        if (rows >= 170 * 16) {
            EXPECT_TRUE(any_unsplit);
            EXPECT_NE(KR::Fp8KRFamily::Split, shapes.front().family());
        }
        // 1157: rows of 2-64 KiB with at least one row per SM lead with Block.
        if (rows >= 170 && row_bytes >= 2048 && row_bytes < 65536) {
            EXPECT_EQ(KR::Fp8KRFamily::Block, shapes.front().family());
        }
        if (row_bytes <= 16) EXPECT_EQ(KR::Fp8KRFamily::Pack, shapes.front().family());
        // Few long rows: the first guess splits them (below 64 KiB, 1157, Block first when rows fill the SMs).
        if (rows < 170 * 16 && row_bytes >= 64 * 1024) EXPECT_EQ(KR::Fp8KRFamily::Split, shapes.front().family());
        // 1154: many rows of 64 KiB or more: Group first, Split second.
        if (rows >= 170 * 16 && row_bytes >= 64 * 1024) {
            ASSERT_GE(shapes.size(), 2u);
            EXPECT_EQ(KR::Fp8KRFamily::Group, shapes[0].family());
            EXPECT_EQ(KR::Fp8KRFamily::Split, shapes[1].family());
        }
    }
    // Illegal shapes are rejected.
    EXPECT_FALSE(KR::fp8KRSegmentShapeLegal({1, 100, 100, 3, 1}));
    EXPECT_FALSE(KR::fp8KRSegmentShapeLegal({1, 100000, 8200, 256, 1}));      // chunk not a multiple of 16
    EXPECT_FALSE(KR::fp8KRSegmentShapeLegal({1, 100000, 8192, 32, 1}));       // Split is one CTA per chunk
    EXPECT_FALSE(KR::fp8KRSegmentShapeLegal({1, 1ULL << 30, 4096, 256, 1}));  // too many chunks
    EXPECT_FALSE(KR::fp8KRSegmentShapeLegal({8, 40, 40, 1, 4}));               // Pack rows <= 32 bytes
    EXPECT_FALSE(KR::fp8KRSegmentShapeLegal({8, 16, 16, 1, 16}));              // Pack span <= 128 bytes
    EXPECT_TRUE(KR::fp8KRSegmentShapeLegal({1, 100000, 8192, 256, 1}));
    EXPECT_TRUE(KR::fp8KRSegmentShapeLegal({8, 16, 16, 1, 8}));
}

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

struct ByteInput {
    Tensor storage;  // UINT8, offset + rows * row_bytes (+ 3) bytes
    Tensor input;    // the FP8 view at `offset`
    std::vector<uint8_t> bytes;
};

bool finiteFp8(uint8_t byte, bool e5m2) { return std::isfinite(Reference::decode(byte, e5m2)); }

// Finite VALUE bytes of magnitude <= 16, or (exact products) only +-0.5, +-1, +-2.
std::vector<uint8_t> valueBytes(uint64_t count, bool e5m2, bool exact_products, std::mt19937& rng) {
    std::vector<uint8_t> bytes(count);
    for (uint8_t& byte : bytes) {
        for (;;) {
            const uint8_t candidate = static_cast<uint8_t>(rng());
            const float value = Reference::decode(candidate, e5m2);
            if (!finiteFp8(candidate, e5m2) || std::fabs(value) > 16.0f) continue;
            if (exact_products) {
                const float magnitude = std::fabs(value);
                if (magnitude != 0.5f && magnitude != 1.0f && magnitude != 2.0f) continue;
            }
            byte = candidate;
            break;
        }
    }
    return bytes;
}

// ARG bytes: random, with NaNs, zeros of both signs and the extremes mixed in; every third seed is tie-heavy.
std::vector<uint8_t> argBytes(uint64_t count, uint32_t seed) {
    static constexpr std::array<uint8_t, 10> SPECIALS{{0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7B, 0xFB, 0x01, 0x81}};
    static constexpr std::array<uint8_t, 4> TIES{{0x38, 0xB8, 0x00, 0x80}};
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes(count);
    const bool tie_heavy = (seed % 3u) == 0;
    for (uint8_t& byte : bytes) {
        byte = tie_heavy               ? TIES[rng() % TIES.size()]
               : (rng() % 512u) == 0 ? SPECIALS[rng() % SPECIALS.size()]
                                       : static_cast<uint8_t>(rng());
    }
    return bytes;
}

ByteInput upload(std::vector<uint8_t> bytes, uint64_t rows, uint64_t row_bytes, uint64_t offset, DataType dtype,
                 Stream& stream) {
    std::vector<uint64_t> padded(offset + bytes.size() + 3, 0x11);
    for (size_t i = 0; i < bytes.size(); ++i) padded[offset + i] = bytes[i];
    ByteInput result;
    result.storage = makeGpuUnsignedTensor(padded, {padded.size()}, stream, DataType::UINT8);
    result.input = result.storage.aliasByteStorageAs(dtype, {rows, row_bytes}, offset);
    result.bytes = std::move(bytes);
    return result;
}

// 1148: Split scratch needs no initialization; fill it with garbage to prove it.
std::optional<Tensor> scratchFor(const KR::Fp8KRSegmentShape& shape, bool arg, Stream& stream) {
    const uint64_t bytes = KR::fp8KRScratchBytes(shape, arg);
    if (bytes == 0) return std::nullopt;
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    scratch.fillRandom(0.0, 255.0, stream);
    stream.synchronize();
    return scratch;
}

const std::vector<std::pair<uint64_t, uint64_t>>& testSites() {
    static const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},   {1, 7},    {1, 4096}, {1, 8192},  {1, 100003}, {3, 2},      {5, 3},    {1000, 5}, {77, 16},
        {1001, 6}, {37, 5},   {100, 3},  {4099, 2},
        {64, 31}, {17, 129}, {8, 9000}, {2, 70001}, {4, 262147}, {300, 1000}, {9, 4},   {1, 1 << 20},
        {2800, 4101}, {3000, 1500}, {600, 8166}, {170, 2048}};  // 1156-1157: Block (listed for 2-64 KiB rows;
                                                                   // shapesToTest adds it wherever legal)
    return sites;
}

// The production list plus (1156) the Block shape wherever it is legal, as the training sweep explores it.
std::vector<KR::Fp8KRSegmentShape> shapesToTest(uint64_t rows, uint64_t row_bytes) {
    std::vector<KR::Fp8KRSegmentShape> shapes = KR::fp8KRCandidateShapes(rows, row_bytes, 170);
    const KR::Fp8KRSegmentShape block{rows, row_bytes, row_bytes, KR::FP8_KR_THREADS, 1, false};
    if (KR::fp8KRSegmentShapeLegal(block) && std::find(shapes.begin(), shapes.end(), block) == shapes.end()) {
        shapes.push_back(block);
    }
    return shapes;
}

struct ValueCase {
    CubReductionOp op;
    double (*reference)(const std::vector<double>&);
};

double sumOf(const std::vector<double>& x) {
    double s = 0;
    for (double v : x) s += v;
    return s;
}

}  // namespace

TEST(CubReductionFp8KR, ValueEveryShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(1146);
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const auto& [rows, row_bytes] : testSites()) {
            // E4M3 runs 16-byte aligned (the vector Pack kernels), E5M2 at misaligned views (the generic kernels).
            const uint64_t offset = e5m2 ? (rows + row_bytes) % 4 : 0;
            for (const CubReductionOp op : {CubReductionOp::Sum, CubReductionOp::Max, CubReductionOp::Min,
                                            CubReductionOp::Mean, CubReductionOp::L1Norm, CubReductionOp::L2Norm,
                                            CubReductionOp::Product}) {
                if (op == CubReductionOp::Product && row_bytes > 16) continue;
                const ByteInput in = upload(valueBytes(rows * row_bytes, e5m2, op == CubReductionOp::Product, rng),
                                            rows, row_bytes, offset, dtype, stream);
                std::vector<double> expected(rows), magnitude(rows);
                for (uint64_t row = 0; row < rows; ++row) {
                    double acc = op == CubReductionOp::Product ? 1.0
                                 : op == CubReductionOp::Max  ? -INFINITY
                                 : op == CubReductionOp::Min  ? INFINITY
                                                              : 0.0;
                    double abs_sum = 0;
                    for (uint64_t i = 0; i < row_bytes; ++i) {
                        const double x = Reference::decode(in.bytes[row * row_bytes + i], e5m2);
                        abs_sum += std::fabs(x) * (op == CubReductionOp::L2Norm ? std::fabs(x) : 1.0);
                        switch (op) {
                            case CubReductionOp::Sum:
                            case CubReductionOp::Mean: acc += x; break;
                            case CubReductionOp::Max: acc = std::max(acc, x); break;
                            case CubReductionOp::Min: acc = std::min(acc, x); break;
                            case CubReductionOp::L1Norm: acc += std::fabs(x); break;
                            case CubReductionOp::L2Norm: acc += x * x; break;
                            case CubReductionOp::Product: acc *= x; break;
                            default: break;
                        }
                    }
                    if (op == CubReductionOp::Mean) {
                        acc /= static_cast<double>(row_bytes);
                        abs_sum /= static_cast<double>(row_bytes);
                    }
                    if (op == CubReductionOp::L2Norm) {
                        acc = std::sqrt(acc);
                        abs_sum = std::sqrt(abs_sum);
                    }
                    expected[row] = acc;
                    magnitude[row] = abs_sum;
                }
                const CubReductionStageSemantics semantics =
                    makeValueReductionStageSemantics(op, DensePassRole::Complete, row_bytes);
                for (const auto& shape : shapesToTest(rows, row_bytes)) {
                    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype) << " rows=" << rows
                                                      << " R=" << row_bytes << " offset=" << offset
                                                      << " op=" << static_cast<int>(op) << " shape="
                                                      << KR::fp8KRShapeName(shape));
                    std::optional<Tensor> scratch = scratchFor(shape, false, stream);
                    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                    for (int launch = 0; launch < 2; ++launch) {  // a repeated launch on the same scratch gives the same result
                        launchFp8KRValueSegments(semantics, in.input, output, shape,
                                                 scratch ? scratch->getMemPtr<void>() : nullptr,
                                                 scratch ? scratch->getArraySizeInBytes() : 0, 1.0f, stream);
                    }
                    stream.synchronize();
                    const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
                    for (uint64_t row = 0; row < rows; ++row) {
                        const double tolerance =
                            (op == CubReductionOp::Max || op == CubReductionOp::Min || op == CubReductionOp::Product)
                                ? 0.0
                                : 2e-4 * magnitude[row] + 1e-6;
                        ASSERT_NEAR(expected[row], static_cast<double>(got[row]), tolerance) << "row " << row;
                    }
                }
            }
        }
    }
}

// A runtime output scale and a narrow output dtype are applied once, after finalization (staged shapes in the finisher).
TEST(CubReductionFp8KR, ValueScaleAndOutputDtype) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(11461);
    const uint64_t rows = 3, row_bytes = 70001;
    const ByteInput in = upload(valueBytes(rows * row_bytes, false, false, rng), rows, row_bytes, 1,
                                DataType::FP8_E4M3, stream);
    const auto semantics = makeValueReductionStageSemantics(CubReductionOp::Mean, DensePassRole::Complete, row_bytes);
    std::vector<double> expected(rows);
    for (uint64_t row = 0; row < rows; ++row) {
        double s = 0;
        for (uint64_t i = 0; i < row_bytes; ++i) s += Reference::decode(in.bytes[row * row_bytes + i], false);
        expected[row] = -2.5 * s / static_cast<double>(row_bytes);
    }
    for (const auto& shape : shapesToTest(rows, row_bytes)) {
        SCOPED_TRACE(KR::fp8KRShapeName(shape));
        std::optional<Tensor> scratch = scratchFor(shape, false, stream);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, {rows}));
        launchFp8KRValueSegments(semantics, in.input, output, shape, scratch ? scratch->getMemPtr<void>() : nullptr,
                                 scratch ? scratch->getArraySizeInBytes() : 0, -2.5f, stream);
        stream.synchronize();
        const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
        for (uint64_t row = 0; row < rows; ++row)
            ASSERT_NEAR(expected[row], got[row], 1e-2 * std::fabs(expected[row]) + 1e-3) << "row " << row;
    }
}

TEST(CubReductionFp8KR, ArgEveryShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    uint32_t seed = 1146;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const bool e5m2 = dtype == DataType::FP8_E5M2;
        for (const auto& [rows, row_bytes] : testSites()) {
            const uint64_t offset = e5m2 ? (rows * 3 + row_bytes) % 4 : 0;
            const ByteInput in = upload(argBytes(rows * row_bytes, ++seed), rows, row_bytes, offset, dtype, stream);
            for (const bool maximum : {false, true}) {
                std::vector<Reference::Candidate> expected(rows);
                for (uint64_t row = 0; row < rows; ++row) {
                    Reference::Candidate best{0, Reference::decode(in.bytes[row * row_bytes], e5m2)};
                    for (uint64_t i = 1; i < row_bytes; ++i)
                        best = Reference::combine(best, {i, Reference::decode(in.bytes[row * row_bytes + i], e5m2)},
                                                  maximum);
                    expected[row] = best;
                }
                for (const auto& shape : shapesToTest(rows, row_bytes)) {
                    std::optional<Tensor> scratch = scratchFor(shape, true, stream);
                    for (const bool with_value : {true, false}) {
                        const DataType index_dtype = with_value ? DataType::UINT64 : DataType::UINT32;
                        SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype) << " rows=" << rows
                                                          << " R=" << row_bytes << " offset=" << offset
                                                          << " max=" << maximum << " value=" << with_value
                                                          << " shape=" << KR::fp8KRShapeName(shape));
                        std::optional<Tensor> value;
                        if (with_value) value = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                        Tensor index(gpuPlacement, TensorDescriptor(index_dtype, {rows}));
                        const uint64_t stride = with_value ? 3 : 1;
                        for (int launch = 0; launch < 2; ++launch) {
                            launchFp8KRArgSegments(in.input, value ? &*value : nullptr, &index, shape,
                                                   scratch ? scratch->getMemPtr<void>() : nullptr,
                                                   scratch ? scratch->getArraySizeInBytes() : 0, maximum, stride,
                                                   stream);
                        }
                        stream.synchronize();
                        const std::vector<uint64_t> got_indices = copyGpuTensorAsUnsigned(index, stream);
                        const std::vector<float> got_values =
                            value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                        for (uint64_t row = 0; row < rows; ++row) {
                            ASSERT_EQ(expected[row].index * stride, got_indices[row]) << "row " << row;
                            if (value)
                                ASSERT_TRUE(Reference::sameValue(expected[row].value, got_values[row])) << "row " << row;
                        }
                    }
                }
            }
        }
    }
}

// The production stamps (selection among the retained shapes only: 1148 removed the direct CUB pass, 1173 made any
// direct pass for an original FP8 site an error) at R and KR sites, alone and after an RK pass.
TEST(CubReductionFp8KR, ProductionStampsMatchReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937 rng(11462);
    struct Case {
        std::vector<uint64_t> dims;
        std::vector<uint32_t> axes;
    };
    const std::vector<Case> cases{{{100003}, {0}},        {{4, 262147}, {1}}, {{4096, 24}, {1}},
                                  {{6, 5, 1000}, {1, 2}}, {{9, 7, 33}, {2}},  {{2, 70001}, {0, 1}}};
    for (const Case& c : cases) {
        uint64_t elements = 1;
        for (uint64_t d : c.dims) elements *= d;
        std::vector<uint8_t> bytes = valueBytes(elements, false, false, rng);
        std::vector<uint64_t> wide(bytes.begin(), bytes.end());
        Tensor storage = makeGpuUnsignedTensor(wide, {elements}, stream, DataType::UINT8);
        Tensor input = storage.aliasByteStorageAs(DataType::FP8_E4M3, c.dims, 0);
        std::vector<float> decoded(elements);
        for (uint64_t i = 0; i < elements; ++i) decoded[i] = Reference::decode(bytes[i], false);
        Tensor fp32_input = makeGpuTensor(decoded, c.dims, stream, DataType::FP32);
        SCOPED_TRACE(::testing::Message() << "elements=" << elements << " axes=" << c.axes.size());

        for (const CubReductionOp op : {CubReductionOp::Sum, CubReductionOp::Max}) {
            const std::vector<float> expected = executeFp32Output(fp32_input, op, c.axes, stream);
            const std::vector<float> got = executeFp32Output(input, op, c.axes, stream);
            ASSERT_EQ(expected.size(), got.size());
            for (size_t i = 0; i < got.size(); ++i)
                ASSERT_NEAR(expected[i], got[i], op == CubReductionOp::Max ? 0.0f : 1e-3f * (1.0f + std::fabs(expected[i])))
                    << "op " << static_cast<int>(op) << " output " << i;
        }

        CubArgReductionOutputOptions outputs;
        outputs.produce_value = true;
        outputs.produce_index = true;
        outputs.value_output_dtype = DataType::FP32;
        outputs.index_output_dtype = DataType::UINT64;
        for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
            auto fp8 = CubArgReduction(op, c.axes, outputs).stamp(input, stream);
            auto fp32 = CubArgReduction(op, c.axes, outputs).stamp(fp32_input, stream);
            fp8->run();
            fp32->run();
            stream.synchronize();
            const std::vector<uint64_t> got = copyGpuTensorAsUnsigned(*fp8->getIndexOutputTensor(), stream);
            const std::vector<uint64_t> expected = copyGpuTensorAsUnsigned(*fp32->getIndexOutputTensor(), stream);
            ASSERT_EQ(expected, got) << "arg op " << static_cast<int>(op);
        }
    }
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
