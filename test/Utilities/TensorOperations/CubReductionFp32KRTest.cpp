// 1214: and the carried-index FP32 ARG pass (phase 8) at every FP32 ARG pool shape.
// 1181: and the FP32 ARG kernels (two-word candidates) at every FP32 ARG pool shape: NaN, -0 / +0, infinities and
// ties, value and index outputs, an original-domain index stride, and uninitialized Split scratch.
// 1179: retained R / KR VALUE kernels on FP32 input (the FP8 Group / Pack / Block / Split kernels at one element per
// word). The element-aware shape vocabulary (host), and every shape of the FP32 candidate pool against a host
// reference: every VALUE semantics, MIN / MAX with |x| and x^2 folds, 4-byte aligned but not 16-byte aligned inputs,
// repeated launches on one uninitialized Split scratch, a runtime output scale and a narrow output dtype.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace KR = ThorImplementation::CubReductionInternal::Fp8KR;

namespace {

// Sites in elements: (rows, R). Bytes are 4 R.
const std::vector<std::pair<uint64_t, uint64_t>>& fp32Sites() {
    static const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},     {1, 7},      {1, 1024},   {1, 25001}, {3, 2},     {1000, 5},  {77, 4},    {1001, 3},
        {4099, 1},  {5, 8},      {64, 31},    {17, 129},  {8, 2250},  {2, 17501}, {4, 65537}, {300, 250},
        {9, 1},     {1, 1 << 18}, {2800, 1025}, {600, 2042}, {170, 512}, {5000, 2}};
    return sites;
}

}  // namespace

TEST(CubReductionFp32KR, ElementAwareShapesKeepFp8AndAreFp32Legal) {
    const std::vector<std::pair<uint64_t, uint64_t>> byte_sites{
        {1, 1}, {1, 7}, {3, 2}, {1000, 5}, {64, 31}, {17, 129}, {8, 4100}, {2, 70001}, {170, 1024}, {100000, 64},
        {4096, 4096}, {2720, 8192}, {1, 1ULL << 25}, {11298, 46814}, {1, 18509157}, {1000, 6}, {4099, 3}};
    // Element size 1 is FP8 unchanged: the same list and pool, positions included.
    for (const auto& [rows, row_bytes] : byte_sites) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << row_bytes);
        EXPECT_EQ(KR::fp8KRCandidateShapes(rows, row_bytes, 170),
                  KR::retainedKRCandidateShapes(rows, row_bytes, 170, 1));
        const auto fp8_pool = KR::fp8KRCandidatePool(rows, row_bytes, 170);
        const auto pool = KR::retainedKRCandidatePool(rows, row_bytes, 170, 1);
        ASSERT_EQ(fp8_pool.size(), pool.size());
        for (size_t i = 0; i < pool.size(); ++i) {
            EXPECT_EQ(fp8_pool[i].shape, pool[i].shape);
            EXPECT_EQ(fp8_pool[i].list_position, pool[i].list_position);
        }
    }
    // Element size 4: FP32 rows at 4 R bytes; every shape legal for FP32, no vector Pack, the list a prefix of the
    // pool, and (no vector row length is a 4-byte multiple) the shapes FP8 lists at those bytes, with (1180) Block
    // second for rows of at least 64 KiB and at least one row per SM.
    std::vector<std::pair<uint64_t, uint64_t>> sites = fp32Sites();
    sites.insert(sites.end(), {{512, 16384}, {512, 65536}, {170, 16384}, {169, 16384}, {2048, 16384}, {512, 16383}});
    for (const auto& [rows, elements] : sites) {
        const uint64_t row_bytes = 4 * elements;
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const auto shapes = KR::retainedKRCandidateShapes(rows, row_bytes, 170, 4);
        ASSERT_FALSE(shapes.empty());
        const KR::Fp8KRSegmentShape block{rows, row_bytes, row_bytes, KR::FP8_KR_THREADS, 1, false};
        std::vector<KR::Fp8KRSegmentShape> fp8 = KR::fp8KRCandidateShapes(rows, row_bytes, 170);
        if (rows >= 170 && row_bytes >= KR::FP32_KR_BLOCK_SECOND_ROW_BYTES) {
            ASSERT_GE(shapes.size(), 2u);
            EXPECT_EQ(block, shapes[1]);
            if (std::find(fp8.begin(), fp8.end(), block) == fp8.end()) fp8.insert(fp8.begin() + 1, block);
        }
        EXPECT_EQ(fp8, shapes);
        for (const auto& shape : shapes) {
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(shape, 4)) << KR::fp8KRShapeName(shape);
            EXPECT_FALSE(shape.vector);
        }
        const auto pool = KR::retainedKRCandidatePool(rows, row_bytes, 170, 4);
        for (size_t i = 0; i < pool.size(); ++i) {
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(pool[i].shape, 4));
            if (i < shapes.size()) {
                EXPECT_EQ(shapes[i], pool[i].shape);
                EXPECT_EQ(i + 1, pool[i].list_position);
            } else {
                EXPECT_EQ(0u, pool[i].list_position);
            }
        }
    }
    // Rows of partial elements and vector Pack are not FP32 shapes.
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 6, 6, 1, 8, true}, 4));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 6, 6, 1, 8, true}, 1));
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 6, 6, 4, 1, false}, 4));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 8, 8, 4, 1, false}, 4));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 8, 8, 4, 1, false}, 2));  // 1233: FP16 / BF16
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 8, 8, 4, 1, false}, 3));
    EXPECT_THROW(static_cast<void>(KR::retainedKRCandidateShapes(8, 6, 170, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(KR::retainedKRCandidatePool(8, 6, 170, 4)), std::invalid_argument);
}

namespace {

struct Fp32Input {
    Tensor storage;  // UINT8: offset + 4 * count + 4 bytes
    Tensor input;    // the FP32 [rows, R] view at `offset`
    std::vector<float> values;
};

// FP32 values on a byte storage at a byte offset (a multiple of 4; 4 or 8 leave the view off 16-byte alignment).
Fp32Input uploadFp32(std::vector<float> values, uint64_t rows, uint64_t elements, uint64_t offset, Stream& stream) {
    std::vector<uint64_t> padded(offset + 4 * values.size() + 4, 0x11);
    for (size_t i = 0; i < values.size(); ++i) {
        uint8_t bytes[4];
        std::memcpy(bytes, &values[i], 4);
        for (int b = 0; b < 4; ++b) padded[offset + 4 * i + b] = bytes[b];
    }
    Fp32Input result;
    result.storage = makeGpuUnsignedTensor(padded, {padded.size()}, stream, DataType::UINT8);
    result.input = result.storage.aliasByteStorageAs(DataType::FP32, {rows, elements}, offset);
    result.values = std::move(values);
    return result;
}

// Dyadic values k / 4, |k| <= 32: sums, |x| sums, MIN / MAX are exact in FP32 at every tested length, so a dropped or
// repeated element shows. Exact products: only +-0.5, +-1, +-2.
std::vector<float> fp32Values(uint64_t count, bool exact_products, std::mt19937& rng) {
    static constexpr float PRODUCT_VALUES[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f};
    std::vector<float> values(count);
    for (float& value : values) {
        value = exact_products ? PRODUCT_VALUES[rng() % 6]
                               : static_cast<float>(static_cast<int>(rng() % 65) - 32) / 4.0f;
    }
    return values;
}

std::optional<Tensor> fp32ScratchFor(const KR::Fp8KRSegmentShape& shape, Stream& stream) {
    const uint64_t bytes = KR::fp8KRScratchBytes(shape, false);
    if (bytes == 0) return std::nullopt;
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    scratch.fillRandom(0.0, 255.0, stream);  // the Split scratch needs no initialization
    stream.synchronize();
    return scratch;
}

std::vector<KR::Fp8KRSegmentShape> fp32ShapesToTest(uint64_t rows, uint64_t elements) {
    std::vector<KR::Fp8KRSegmentShape> shapes;
    for (const auto& entry : KR::retainedKRCandidatePool(rows, 4 * elements, 170, 4)) shapes.push_back(entry.shape);
    return shapes;
}

enum class Fp32Case { Sum, Mean, L1, L2, Product, Max, Min, MaxAbs, MinAbs, MaxSquare, MinSquare };

struct Fp32CaseSpec {
    Fp32Case kind;
    const char* name;
};

constexpr Fp32CaseSpec FP32_CASES[] = {
    {Fp32Case::Sum, "sum"},     {Fp32Case::Mean, "mean"},       {Fp32Case::L1, "l1"},
    {Fp32Case::L2, "l2"},       {Fp32Case::Product, "product"}, {Fp32Case::Max, "max"},
    {Fp32Case::Min, "min"},     {Fp32Case::MaxAbs, "max|x|"},   {Fp32Case::MinAbs, "min|x|"},
    {Fp32Case::MaxSquare, "max x^2"}, {Fp32Case::MinSquare, "min x^2"}};

CubReductionInternal::CubReductionStageSemantics fp32Semantics(Fp32Case kind, uint64_t elements) {
    using namespace CubReductionInternal;
    const auto op_semantics = [&](CubReductionOp op) {
        return makeValueReductionStageSemantics(op, DensePassRole::Complete, elements);
    };
    const auto extremum = [](CubReductionStageCombine combine, CubReductionStageInputTransform transform) {
        CubReductionStageSemantics semantics;
        semantics.combine = combine;
        semantics.input_transform = transform;
        return semantics;
    };
    switch (kind) {
        case Fp32Case::Sum: return op_semantics(CubReductionOp::Sum);
        case Fp32Case::Mean: return op_semantics(CubReductionOp::Mean);
        case Fp32Case::L1: return op_semantics(CubReductionOp::L1Norm);
        case Fp32Case::L2: return op_semantics(CubReductionOp::L2Norm);
        case Fp32Case::Product: return op_semantics(CubReductionOp::Product);
        case Fp32Case::Max: return op_semantics(CubReductionOp::Max);
        case Fp32Case::Min: return op_semantics(CubReductionOp::Min);
        case Fp32Case::MaxAbs:
            return extremum(CubReductionStageCombine::Maximum, CubReductionStageInputTransform::AbsoluteValue);
        case Fp32Case::MinAbs:
            return extremum(CubReductionStageCombine::Minimum, CubReductionStageInputTransform::AbsoluteValue);
        case Fp32Case::MaxSquare:
            return extremum(CubReductionStageCombine::Maximum, CubReductionStageInputTransform::Square);
        case Fp32Case::MinSquare:
            return extremum(CubReductionStageCombine::Minimum, CubReductionStageInputTransform::Square);
    }
    throw std::logic_error("unknown FP32 KR test case");
}

double fp32Reference(Fp32Case kind, const float* row, uint64_t elements) {
    double acc = kind == Fp32Case::Product ? 1.0
                 : (kind == Fp32Case::Max || kind == Fp32Case::MaxAbs || kind == Fp32Case::MaxSquare) ? -INFINITY
                 : (kind == Fp32Case::Min || kind == Fp32Case::MinAbs || kind == Fp32Case::MinSquare) ? INFINITY
                                                                                                       : 0.0;
    for (uint64_t i = 0; i < elements; ++i) {
        const double x = row[i];
        switch (kind) {
            case Fp32Case::Sum:
            case Fp32Case::Mean: acc += x; break;
            case Fp32Case::L1: acc += std::fabs(x); break;
            case Fp32Case::L2: acc += x * x; break;
            case Fp32Case::Product: acc *= x; break;
            case Fp32Case::Max: acc = std::max(acc, x); break;
            case Fp32Case::Min: acc = std::min(acc, x); break;
            case Fp32Case::MaxAbs: acc = std::max(acc, std::fabs(x)); break;
            case Fp32Case::MinAbs: acc = std::min(acc, std::fabs(x)); break;
            case Fp32Case::MaxSquare: acc = std::max(acc, x * x); break;
            case Fp32Case::MinSquare: acc = std::min(acc, x * x); break;
        }
    }
    if (kind == Fp32Case::Mean) acc /= static_cast<double>(elements);
    if (kind == Fp32Case::L2) acc = std::sqrt(acc);
    return acc;
}

// Exact where the dyadic inputs make the FP32 result exact; Mean is one division (and the epilogue's rounding); L2's
// sum of squares can round past 2^24 / 16 for the longest rows, and its square root halves that relative error.
double fp32Tolerance(Fp32Case kind, double expected) {
    switch (kind) {
        case Fp32Case::Mean: return 4e-7 * std::fabs(expected) + 1e-30;
        case Fp32Case::L2: return 1e-4 * std::fabs(expected) + 1e-30;
        default: return 0.0;
    }
}

}  // namespace

TEST(CubReductionFp32KR, ValueEveryPoolShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(1179);
    for (const auto& [rows, elements] : fp32Sites()) {
        // 4-byte aligned views, alternating between 16-byte aligned and 4 / 8 bytes past it.
        const uint64_t offset = 4 * ((rows + elements) % 3);
        for (const Fp32CaseSpec& spec : FP32_CASES) {
            if (spec.kind == Fp32Case::Product && elements > 8) continue;
            const Fp32Input in = uploadFp32(fp32Values(rows * elements, spec.kind == Fp32Case::Product, rng), rows,
                                            elements, offset, stream);
            std::vector<double> expected(rows);
            for (uint64_t row = 0; row < rows; ++row)
                expected[row] = fp32Reference(spec.kind, in.values.data() + row * elements, elements);
            const CubReductionStageSemantics semantics = fp32Semantics(spec.kind, elements);
            for (const auto& shape : fp32ShapesToTest(rows, elements)) {
                SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements << " offset=" << offset
                                                  << " case=" << spec.name << " shape=" << KR::fp8KRShapeName(shape));
                std::optional<Tensor> scratch = fp32ScratchFor(shape, stream);
                Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                for (int launch = 0; launch < 2; ++launch) {  // a repeated launch on the same scratch: same result
                    launchFp32KRValueSegments(semantics, in.input, output, shape,
                                              scratch ? scratch->getMemPtr<void>() : nullptr,
                                              scratch ? scratch->getArraySizeInBytes() : 0, 1.0f, stream);
                }
                stream.synchronize();
                const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
                for (uint64_t row = 0; row < rows; ++row) {
                    ASSERT_NEAR(expected[row], static_cast<double>(got[row]), fp32Tolerance(spec.kind, expected[row]))
                        << "row " << row;
                }
            }
        }
    }
}

// A runtime output scale (negative: MAX's folded negation composes with it) and a narrow output dtype apply once.
TEST(CubReductionFp32KR, ValueScaleAndOutputDtype) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(11791);
    const uint64_t rows = 3, elements = 17501;
    const Fp32Input in = uploadFp32(fp32Values(rows * elements, false, rng), rows, elements, 4, stream);
    for (const Fp32Case kind : {Fp32Case::Mean, Fp32Case::Max}) {
        const auto semantics = fp32Semantics(kind, elements);
        for (const auto& shape : fp32ShapesToTest(rows, elements)) {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(kind) << " " << KR::fp8KRShapeName(shape));
            std::optional<Tensor> scratch = fp32ScratchFor(shape, stream);
            Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, {rows}));
            launchFp32KRValueSegments(semantics, in.input, output, shape,
                                      scratch ? scratch->getMemPtr<void>() : nullptr,
                                      scratch ? scratch->getArraySizeInBytes() : 0, -2.5f, stream);
            stream.synchronize();
            const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
            for (uint64_t row = 0; row < rows; ++row) {
                const double expected = -2.5 * fp32Reference(kind, in.values.data() + row * elements, elements);
                ASSERT_NEAR(expected, got[row], 1e-2 * std::fabs(expected) + 1e-3) << "row " << row;
            }
        }
    }
}

// The launcher rejects what the FP32 kernels do not cover.
TEST(CubReductionFp32KR, ValueRejectsUnsupportedInputs) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(11792);
    const uint64_t rows = 8, elements = 4;
    const auto semantics = fp32Semantics(Fp32Case::Sum, elements);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
    const Fp32Input in = uploadFp32(fp32Values(rows * elements, false, rng), rows, elements, 0, stream);
    const KR::Fp8KRSegmentShape group{rows, 4 * elements, 4 * elements, 4, 1, false};
    // A shape in elements instead of bytes.
    EXPECT_THROW(launchFp32KRValueSegments(semantics, in.input, output, {rows, elements, elements, 4, 1, false},
                                           nullptr, 0, 1.0f, stream),
                 std::logic_error);
    // Vector Pack is FP8-only.
    EXPECT_THROW(launchFp32KRValueSegments(semantics, in.input, output, {rows, 16, 16, 1, 1, true}, nullptr, 0, 1.0f,
                                           stream),
                 std::logic_error);
    // FP8 input to the FP32 launcher.
    Tensor bytes = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {rows, 4 * elements}, 0);
    EXPECT_THROW(launchFp32KRValueSegments(semantics, bytes, output, group, nullptr, 0, 1.0f, stream),
                 std::logic_error);
    // A Split shape without scratch.
    const uint64_t long_elements = 4096;
    const Fp32Input long_in = uploadFp32(fp32Values(long_elements, false, rng), 1, long_elements, 0, stream);
    Tensor one(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    EXPECT_THROW(launchFp32KRValueSegments(fp32Semantics(Fp32Case::Sum, long_elements), long_in.input, one,
                                           {1, 4 * long_elements, 4096, KR::FP8_KR_THREADS, 1, false}, nullptr, 0,
                                           1.0f, stream),
                 std::logic_error);
    // The legal shape runs.
    EXPECT_NO_THROW(launchFp32KRValueSegments(semantics, in.input, output, group, nullptr, 0, 1.0f, stream));
    stream.synchronize();
}

namespace {

// ARG values: random bit patterns (NaNs, infinities, subnormals and both zeros among them) with the specials mixed in;
// every third seed is tie-heavy (+-1 and both zeros only).
std::vector<float> fp32ArgValues(uint64_t count, uint32_t seed) {
    static const float SPECIALS[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 3.0e38f, -3.0e38f};
    static const float TIES[] = {1.0f, -1.0f, 0.0f, -0.0f};
    std::mt19937 rng(seed);
    std::vector<float> values(count);
    const bool tie_heavy = (seed % 3u) == 0;
    for (float& value : values) {
        if (tie_heavy) {
            value = TIES[rng() % 4];
        } else if ((rng() % 256u) == 0) {
            value = SPECIALS[rng() % 9];
        } else {
            const uint32_t bits = rng();
            std::memcpy(&value, &bits, 4);
        }
    }
    return values;
}

std::optional<Tensor> fp32ArgScratchFor(const KR::Fp8KRSegmentShape& shape, Stream& stream) {
    const uint64_t bytes = KR::fp8KRScratchBytes(shape, true);
    if (bytes == 0) return std::nullopt;
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    scratch.fillRandom(0.0, 255.0, stream);
    stream.synchronize();
    return scratch;
}

}  // namespace

TEST(CubReductionFp32KR, ArgEveryPoolShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    namespace Reference = ThorImplementation::Fp8ArgReference;
    Stream stream(0);
    uint32_t seed = 1181;
    for (const auto& [rows, elements] : fp32Sites()) {
        const uint64_t offset = 4 * ((rows * 3 + elements) % 3);
        const Fp32Input in = uploadFp32(fp32ArgValues(rows * elements, ++seed), rows, elements, offset, stream);
        for (const bool maximum : {false, true}) {
            std::vector<Reference::Candidate> expected(rows);
            for (uint64_t row = 0; row < rows; ++row) {
                Reference::Candidate best{0, in.values[row * elements]};
                for (uint64_t i = 1; i < elements; ++i)
                    best = Reference::combine(best, {i, in.values[row * elements + i]}, maximum);
                expected[row] = best;
            }
            for (const auto& entry : KR::retainedKRCandidatePool(rows, 4 * elements, 170, 4)) {
                const KR::Fp8KRSegmentShape& shape = entry.shape;
                if (!KR::fp32KRArgShapeLegal(shape)) continue;
                std::optional<Tensor> scratch = fp32ArgScratchFor(shape, stream);
                for (const bool with_value : {true, false}) {
                    const DataType index_dtype = with_value ? DataType::UINT64 : DataType::UINT32;
                    SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements << " offset=" << offset
                                                      << " max=" << maximum << " value=" << with_value
                                                      << " shape=" << KR::fp8KRShapeName(shape));
                    std::optional<Tensor> value;
                    if (with_value) value = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                    Tensor index(gpuPlacement, TensorDescriptor(index_dtype, {rows}));
                    const uint64_t stride = with_value ? 3 : 1;
                    for (int launch = 0; launch < 2; ++launch) {
                        launchFp32KRArgSegments(in.input, value ? &*value : nullptr, &index, shape,
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

// The ARG launcher rejects what the FP32 ARG kernels do not cover.
TEST(CubReductionFp32KR, ArgRejectsUnsupportedInputs) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t rows = 8, elements = 4;
    const Fp32Input in = uploadFp32(fp32ArgValues(rows * elements, 7), rows, elements, 0, stream);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {rows}));
    const KR::Fp8KRSegmentShape group{rows, 4 * elements, 4 * elements, 4, 1, false};
    // A shape in elements instead of bytes; no outputs; a wrong output size; FP8 input.
    EXPECT_THROW(launchFp32KRArgSegments(in.input, nullptr, &index, {rows, elements, elements, 4, 1, false}, nullptr,
                                         0, false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32KRArgSegments(in.input, nullptr, nullptr, group, nullptr, 0, false, 1, stream),
                 std::logic_error);
    Tensor short_index(gpuPlacement, TensorDescriptor(DataType::UINT32, {rows - 1}));
    EXPECT_THROW(launchFp32KRArgSegments(in.input, nullptr, &short_index, group, nullptr, 0, false, 1, stream),
                 std::logic_error);
    Tensor bytes = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {rows, 4 * elements}, 0);
    EXPECT_THROW(launchFp32KRArgSegments(bytes, nullptr, &index, group, nullptr, 0, false, 1, stream),
                 std::logic_error);
    // A UINT32 index that cannot hold row-local index * stride.
    EXPECT_THROW(launchFp32KRArgSegments(in.input, nullptr, &index, group, nullptr, 0, false, 1ULL << 31, stream),
                 std::logic_error);
    // Split chunks of more than 2^31 elements are not FP32 ARG shapes.
    EXPECT_FALSE(KR::fp32KRArgShapeLegal({1, uint64_t{1} << 36, uint64_t{1} << 34, KR::FP8_KR_THREADS, 1, false}));
    EXPECT_TRUE(KR::fp32KRArgShapeLegal({1, uint64_t{1} << 36, uint64_t{1} << 33, KR::FP8_KR_THREADS, 1, false}));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({1, uint64_t{1} << 36, uint64_t{1} << 34, KR::FP8_KR_THREADS, 1,
                                                    false}, 4));
    EXPECT_NO_THROW(launchFp32KRArgSegments(in.input, nullptr, &index, group, nullptr, 0, true, 1, stream));
    stream.synchronize();
}

namespace {

// 1214: carried indices that are not monotone in the position and give distinct composed indices within a row, as an
// earlier pass's partials do: at row stride 0 (an intermediate run) distinct values per row (above 2^32 for a UINT64
// index), otherwise values below the stride.
std::vector<uint64_t> fp32KRCarriedIndices(uint64_t rows, uint64_t elements, uint64_t stride, DataType index_dtype,
                                           uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> carried(rows * elements);
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t row_seed = rng();
        for (uint64_t p = 0; p < elements; ++p) {
            uint64_t c = stride == 0 ? ((p * 0x9E3779B1ULL + row_seed) & 0xFFFFFFFFULL) : rng() % stride;
            if (stride == 0 && index_dtype == DataType::UINT64) c += uint64_t{1} << 40;
            carried[row * elements + p] = c;
        }
    }
    return carried;
}

}  // namespace

// 1214: the carried-index FP32 KR ARG pass (phase 8) at every FP32 ARG pool shape: UINT32 and UINT64 carried indices,
// row strides 0 (an intermediate run) and 7 (an original run), argmin and argmax over NaN, -0 / +0, infinities and
// ties, inputs off 16-byte alignment, value and index outputs, repeated launches on one uninitialized wide Split
// scratch: value bits and the composed index exactly. Then the rejections.
TEST(CubReductionFp32KR, CarriedArgEveryPoolShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    namespace Reference = ThorImplementation::Fp8ArgReference;
    Stream stream(0);
    uint32_t seed = 1214;
    for (const auto& [rows, elements] : fp32Sites()) {
        const uint64_t offset = 4 * ((rows * 3 + elements) % 3);
        const Fp32Input in = uploadFp32(fp32ArgValues(rows * elements, ++seed), rows, elements, offset, stream);
        for (const DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
            for (const uint64_t stride : {uint64_t{0}, uint64_t{7}}) {
                const std::vector<uint64_t> carried = fp32KRCarriedIndices(rows, elements, stride, index_dtype, ++seed);
                const Tensor carried_index = makeGpuUnsignedTensor(carried, {rows, elements}, stream, index_dtype);
                for (const bool maximum : {false, true}) {
                    std::vector<Reference::Candidate> expected(rows);
                    for (uint64_t row = 0; row < rows; ++row) {
                        Reference::Candidate best{carried[row * elements], in.values[row * elements]};
                        for (uint64_t i = 1; i < elements; ++i) {
                            best = Reference::combine(
                                best, {carried[row * elements + i] + i * stride, in.values[row * elements + i]},
                                maximum);
                        }
                        expected[row] = best;
                    }
                    for (const auto& entry : KR::retainedKRCandidatePool(rows, 4 * elements, 170, 4)) {
                        const KR::Fp8KRSegmentShape& shape = entry.shape;
                        if (!KR::fp32KRArgShapeLegal(shape)) continue;
                        const uint64_t scratch_bytes = fp32KRCarriedArgScratchBytes(shape);
                        ASSERT_EQ(scratch_bytes != 0, shape.family() == KR::Fp8KRFamily::Split);
                        std::optional<Tensor> scratch;
                        if (scratch_bytes != 0) {
                            scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
                            scratch->fillRandom(0.0, 255.0, stream);
                        }
                        const bool with_value = (rows + elements + static_cast<uint64_t>(maximum)) % 2 == 0;
                        SCOPED_TRACE(::testing::Message()
                                     << "rows=" << rows << " R=" << elements << " offset=" << offset << " index "
                                     << static_cast<int>(index_dtype) << " stride " << stride << " max=" << maximum
                                     << " value=" << with_value << " shape=" << KR::fp8KRShapeName(shape));
                        std::optional<Tensor> value;
                        if (with_value) value = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {rows}));
                        for (int launch = 0; launch < 2; ++launch) {
                            launchFp32KRCarriedArgSegments(in.input, carried_index, value ? &*value : nullptr, &index,
                                                           shape, scratch ? scratch->getMemPtr<void>() : nullptr,
                                                           scratch_bytes, maximum, stride, stream);
                        }
                        stream.synchronize();
                        const std::vector<uint64_t> got_indices = copyGpuTensorAsUnsigned(index, stream);
                        const std::vector<float> got_values =
                            value ? copyGpuTensorAsFloat(*value, stream) : std::vector<float>{};
                        for (uint64_t row = 0; row < rows; ++row) {
                            ASSERT_EQ(expected[row].index, got_indices[row]) << "row " << row;
                            if (value) {
                                ASSERT_TRUE(Reference::sameValue(expected[row].value, got_values[row]))
                                    << "row " << row;
                            }
                        }
                    }
                }
            }
        }
    }
}

// The carried launcher rejects what the carried FP32 KR ARG kernels do not cover.
TEST(CubReductionFp32KR, CarriedArgRejectsUnsupportedInputs) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t rows = 4, elements = 20000;
    const Fp32Input in = uploadFp32(fp32ArgValues(rows * elements, 9), rows, elements, 0, stream);
    const Tensor carried = makeGpuUnsignedTensor(fp32KRCarriedIndices(rows, elements, 0, DataType::UINT32, 9),
                                                 {rows, elements}, stream, DataType::UINT32);
    const Tensor short_carried =
        makeGpuUnsignedTensor(fp32KRCarriedIndices(rows, elements - 1, 0, DataType::UINT32, 9), {rows, elements - 1},
                              stream, DataType::UINT32);
    const Tensor byte_carried =
        makeGpuUnsignedTensor(std::vector<uint64_t>(rows * elements, 1), {rows, elements}, stream, DataType::UINT8);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {rows}));
    const KR::Fp8KRSegmentShape group{rows, 4 * elements, 4 * elements, 32, 1, false};
    const KR::Fp8KRSegmentShape split{rows, 4 * elements, 4096, KR::FP8_KR_THREADS, 1, false};
    ASSERT_TRUE(KR::fp32KRArgShapeLegal(group));
    ASSERT_TRUE(KR::fp32KRArgShapeLegal(split));
    ASSERT_EQ(0u, fp32KRCarriedArgScratchBytes(group));
    // 20 chunks per row: 80 keys (320 bytes) round to 512 bytes, then 80 tails.
    ASSERT_EQ(512u + rows * 20u * 8u, fp32KRCarriedArgScratchBytes(split));
    const uint64_t scratch_bytes = fp32KRCarriedArgScratchBytes(split);
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes + 256}));
    auto* scratch_base = scratch.getMemPtr<uint8_t>();
    // A carried index of another size or dtype; no outputs; a Split without, with too little or misaligned scratch.
    EXPECT_THROW(launchFp32KRCarriedArgSegments(in.input, short_carried, nullptr, &index, group, nullptr, 0, false, 0,
                                                stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32KRCarriedArgSegments(in.input, byte_carried, nullptr, &index, group, nullptr, 0, false, 0,
                                                stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32KRCarriedArgSegments(in.input, carried, nullptr, nullptr, group, nullptr, 0, false, 0,
                                                stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32KRCarriedArgSegments(in.input, carried, nullptr, &index, split, nullptr, 0, false, 0,
                                                stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32KRCarriedArgSegments(in.input, carried, nullptr, &index, split, scratch_base,
                                                scratch_bytes - 1, false, 0, stream),
                 std::logic_error);
    EXPECT_THROW(launchFp32KRCarriedArgSegments(in.input, carried, nullptr, &index, split, scratch_base + 16,
                                                scratch_bytes, false, 0, stream),
                 std::logic_error);
    EXPECT_NO_THROW(launchFp32KRCarriedArgSegments(in.input, carried, nullptr, &index, group, nullptr, 0, true, 0,
                                                   stream));
    EXPECT_NO_THROW(launchFp32KRCarriedArgSegments(in.input, carried, nullptr, &index, split, scratch_base,
                                                   scratch_bytes, true, 0, stream));
    stream.synchronize();
}
