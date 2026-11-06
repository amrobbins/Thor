// 1233: retained R / KR VALUE kernels on FP16 / BF16 input (the FP8 Group / Pack / Block / Split kernels at two
// elements per word, plans/Thor_FP16_BF16_Dense_Reduction_Plan phase 2). The element-size-2 shape vocabulary (host),
// and every shape of the 2-byte candidate pool against a host reference, for both formats: every VALUE semantics, MIN
// / MAX with |x| and x^2 folds, rows that start mid-word (2-byte aligned views and odd row lengths), repeated launches
// on one uninitialized Split scratch, a runtime output scale and a narrow output dtype, and NaN / infinity
// propagation with a NaN-propagating reference. 1234: and the 2-byte vector Pack kernel (rows of 1 to 8 elements) on
// 16-byte aligned inputs, its generic Pack fallback otherwise, and the partial last owner. 1237: the 2-byte ARG pass
// (FP8's ARG kernels at two elements per word) at every 2-byte ARG pool shape: random bit patterns (NaNs, infinities,
// subnormals, both zeros), tie-heavy rows, value and index outputs, an original-domain index stride, rows that start
// mid-word, and uninitialized Split scratch.

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

// Sites in elements: (rows, R). Bytes are 2 R. Odd R puts every other row's start mid-word.
const std::vector<std::pair<uint64_t, uint64_t>>& f16Sites() {
    static const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},     {1, 7},     {1, 1024},  {1, 25001}, {3, 2},      {1000, 5},   {77, 4},     {1001, 3},
        {4099, 1},  {5, 8},     {64, 31},   {17, 129},  {8, 2250},   {2, 17501},  {4, 65537},  {300, 250},
        {9, 1},     {3, 3},     {7, 9},     {2, 2049},  {1, 1 << 18}, {2800, 1025}, {600, 2042}, {170, 512},
        {5000, 2},  {333, 16}};
    return sites;
}

const char* formatName(DataType dtype) { return dtype == DataType::FP16 ? "fp16" : "bf16"; }

}  // namespace

TEST(CubReductionF16KR, ElementSizeTwoShapesAreFp8sWithTheirOwnVectorPack) {
    // Element size 2: rows at 2 R bytes; the list is FP8's at those bytes without FP8's vector Pack shapes, with (1234)
    // the 2-byte vector shape second (1235) for rows of 1 to 8 elements filling at least one thread; every shape legal
    // for element size 2, the list a prefix of the pool, and the vector shape in the pool at any row count.
    std::vector<std::pair<uint64_t, uint64_t>> sites = f16Sites();
    sites.insert(sites.end(), {{512, 32768}, {170, 8192}, {100000, 32}, {1, 1ULL << 24}, {11298, 23407}});
    for (const auto& [rows, elements] : sites) {
        const uint64_t row_bytes = 2 * elements;
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const auto shapes = KR::retainedKRCandidateShapes(rows, row_bytes, 170, 2);
        ASSERT_FALSE(shapes.empty());
        std::vector<KR::Fp8KRSegmentShape> fp8 = KR::fp8KRCandidateShapes(rows, row_bytes, 170);
        std::erase_if(fp8, [](const KR::Fp8KRSegmentShape& shape) { return shape.vector; });
        const bool vector_length = elements <= 8;
        const KR::Fp8KRSegmentShape vector{rows, row_bytes, row_bytes, 1, KR::fp8KRVectorRows(row_bytes), true};
        if (vector_length && rows >= KR::fp8KRVectorRows(row_bytes)) fp8.insert(fp8.begin() + 1, vector);
        EXPECT_EQ(fp8, shapes);
        for (const auto& shape : shapes) {
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(shape, 2)) << KR::fp8KRShapeName(shape);
        }
        const auto pool = KR::retainedKRCandidatePool(rows, row_bytes, 170, 2);
        EXPECT_EQ(vector_length ? 1 : 0, std::count_if(pool.begin(), pool.end(), [](const KR::Fp8KRPoolEntry& entry) {
                      return entry.shape.vector;
                  }));
        for (size_t i = 0; i < pool.size(); ++i) {
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(pool[i].shape, 2));
            if (i < shapes.size()) {
                EXPECT_EQ(shapes[i], pool[i].shape);
                EXPECT_EQ(i + 1, pool[i].list_position);
            } else {
                EXPECT_EQ(0u, pool[i].list_position);
            }
        }
    }
    // Rows of partial elements are not 2-byte shapes; vector Pack only at the 2-byte lengths with fp8KRVectorRows rows
    // per thread; the 1- and 4-byte answers are unchanged.
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 3, 3, 4, 1, false}, 2));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 3, 3, 4, 1, false}, 1));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 6, 6, 1, 8, true}, 2));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 6, 6, 1, 8, true}, 1));
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 6, 6, 1, 8, true}, 4));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 4, 4, 1, 4, true}, 2));
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 4, 4, 1, 4, true}, 1));
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 4, 4, 1, 2, true}, 2));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 16, 16, 1, 2, true}, 2));
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 16, 16, 1, 1, true}, 2));  // one row per thread: a Group
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 18, 18, 1, 8, true}, 2));
    EXPECT_TRUE(KR::retainedKRShapeLegalForAnyElement({8, 4, 4, 1, 4, true}));
    EXPECT_EQ(0u, KR::fp8KRScratchBytes({8, 4, 4, 1, 4, true}, false));
    EXPECT_TRUE(KR::retainedKRShapeLegalForElement({8, 6, 6, 4, 1, false}, 2));
    EXPECT_FALSE(KR::retainedKRShapeLegalForElement({8, 6, 6, 4, 1, false}, 4));
    EXPECT_THROW(static_cast<void>(KR::retainedKRCandidateShapes(8, 7, 170, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(KR::retainedKRCandidatePool(8, 7, 170, 2)), std::invalid_argument);
}

namespace {

// The 16-bit pattern of a value exactly representable in the format (finite values are checked to be exact), with
// infinities and a quiet NaN for the specials.
uint16_t f16Bits(float value, DataType dtype) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    if (dtype == DataType::BF16) {
        if (std::isnan(value)) return 0x7fc0;
        if ((bits & 0xffffU) != 0) throw std::logic_error("value is not exact in BF16");
        return static_cast<uint16_t>(bits >> 16);
    }
    if (std::isnan(value)) return 0x7e00;
    if (std::isinf(value)) return static_cast<uint16_t>(sign | 0x7c00U);
    if ((bits & 0x7fffffffU) == 0) return sign;
    const int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffU;
    if (exponent < 1 || exponent > 30 || (mantissa & 0x1fffU) != 0)
        throw std::logic_error("value is not exact in FP16");
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

struct F16Input {
    Tensor storage;  // UINT8: offset + 2 * count + 4 bytes
    Tensor input;    // the FP16 / BF16 [rows, R] view at `offset`
    std::vector<float> values;
};

// Values on a byte storage at a byte offset (even; 2 or 6 start the view mid-word).
F16Input uploadF16(std::vector<float> values, DataType dtype, uint64_t rows, uint64_t elements, uint64_t offset,
                   Stream& stream) {
    std::vector<uint64_t> padded(offset + 2 * values.size() + 4, 0x11);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = f16Bits(values[i], dtype);
        padded[offset + 2 * i] = bits & 0xffU;
        padded[offset + 2 * i + 1] = bits >> 8;
    }
    F16Input result;
    result.storage = makeGpuUnsignedTensor(padded, {padded.size()}, stream, DataType::UINT8);
    result.input = result.storage.aliasByteStorageAs(dtype, {rows, elements}, offset);
    result.values = std::move(values);
    return result;
}

// Dyadic values k / 4, |k| <= 32 (exact in FP16 and BF16): sums, |x| sums, MIN / MAX are exact in FP32 at every
// tested length, so a dropped or repeated element shows. Exact products: only +-0.5, +-1, +-2.
std::vector<float> f16Values(uint64_t count, bool exact_products, std::mt19937& rng) {
    static constexpr float PRODUCT_VALUES[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f};
    std::vector<float> values(count);
    for (float& value : values) {
        value = exact_products ? PRODUCT_VALUES[rng() % 6]
                               : static_cast<float>(static_cast<int>(rng() % 65) - 32) / 4.0f;
    }
    return values;
}

std::optional<Tensor> f16ScratchFor(const KR::Fp8KRSegmentShape& shape, Stream& stream) {
    const uint64_t bytes = KR::fp8KRScratchBytes(shape, false);
    if (bytes == 0) return std::nullopt;
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    scratch.fillRandom(0.0, 255.0, stream);  // the Split scratch needs no initialization
    stream.synchronize();
    return scratch;
}

std::vector<KR::Fp8KRSegmentShape> f16ShapesToTest(uint64_t rows, uint64_t elements) {
    std::vector<KR::Fp8KRSegmentShape> shapes;
    for (const auto& entry : KR::retainedKRCandidatePool(rows, 2 * elements, 170, 2)) shapes.push_back(entry.shape);
    return shapes;
}

enum class F16Case { Sum, Mean, L1, L2, Product, Max, Min, MaxAbs, MinAbs, MaxSquare, MinSquare };

struct F16CaseSpec {
    F16Case kind;
    const char* name;
};

constexpr F16CaseSpec F16_CASES[] = {
    {F16Case::Sum, "sum"},     {F16Case::Mean, "mean"},       {F16Case::L1, "l1"},
    {F16Case::L2, "l2"},       {F16Case::Product, "product"}, {F16Case::Max, "max"},
    {F16Case::Min, "min"},     {F16Case::MaxAbs, "max|x|"},   {F16Case::MinAbs, "min|x|"},
    {F16Case::MaxSquare, "max x^2"}, {F16Case::MinSquare, "min x^2"}};

CubReductionInternal::CubReductionStageSemantics f16Semantics(F16Case kind, uint64_t elements) {
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
        case F16Case::Sum: return op_semantics(CubReductionOp::Sum);
        case F16Case::Mean: return op_semantics(CubReductionOp::Mean);
        case F16Case::L1: return op_semantics(CubReductionOp::L1Norm);
        case F16Case::L2: return op_semantics(CubReductionOp::L2Norm);
        case F16Case::Product: return op_semantics(CubReductionOp::Product);
        case F16Case::Max: return op_semantics(CubReductionOp::Max);
        case F16Case::Min: return op_semantics(CubReductionOp::Min);
        case F16Case::MaxAbs:
            return extremum(CubReductionStageCombine::Maximum, CubReductionStageInputTransform::AbsoluteValue);
        case F16Case::MinAbs:
            return extremum(CubReductionStageCombine::Minimum, CubReductionStageInputTransform::AbsoluteValue);
        case F16Case::MaxSquare:
            return extremum(CubReductionStageCombine::Maximum, CubReductionStageInputTransform::Square);
        case F16Case::MinSquare:
            return extremum(CubReductionStageCombine::Minimum, CubReductionStageInputTransform::Square);
    }
    throw std::logic_error("unknown FP16 / BF16 KR test case");
}

// NaN-propagating (the device MIN / MAX is min.NaN; the host std::max is not, the 1223 lesson).
double nanMax(double a, double b) { return std::isnan(a) || std::isnan(b) ? NAN : std::max(a, b); }
double nanMin(double a, double b) { return std::isnan(a) || std::isnan(b) ? NAN : std::min(a, b); }

double f16Reference(F16Case kind, const float* row, uint64_t elements) {
    double acc = kind == F16Case::Product ? 1.0
                 : (kind == F16Case::Max || kind == F16Case::MaxAbs || kind == F16Case::MaxSquare) ? -INFINITY
                 : (kind == F16Case::Min || kind == F16Case::MinAbs || kind == F16Case::MinSquare) ? INFINITY
                                                                                                    : 0.0;
    for (uint64_t i = 0; i < elements; ++i) {
        const double x = row[i];
        switch (kind) {
            case F16Case::Sum:
            case F16Case::Mean: acc += x; break;
            case F16Case::L1: acc += std::fabs(x); break;
            case F16Case::L2: acc += x * x; break;
            case F16Case::Product: acc *= x; break;
            case F16Case::Max: acc = nanMax(acc, x); break;
            case F16Case::Min: acc = nanMin(acc, x); break;
            case F16Case::MaxAbs: acc = nanMax(acc, std::fabs(x)); break;
            case F16Case::MinAbs: acc = nanMin(acc, std::fabs(x)); break;
            case F16Case::MaxSquare: acc = nanMax(acc, x * x); break;
            case F16Case::MinSquare: acc = nanMin(acc, x * x); break;
        }
    }
    if (kind == F16Case::Mean) acc /= static_cast<double>(elements);
    if (kind == F16Case::L2) acc = std::sqrt(acc);
    return acc;
}

// Exact where the dyadic inputs make the FP32 result exact; Mean is one division (and the epilogue's rounding); L2's
// sum of squares can round past 2^24 / 16 for the longest rows, and its square root halves that relative error.
double f16Tolerance(F16Case kind, double expected) {
    switch (kind) {
        case F16Case::Mean: return 4e-7 * std::fabs(expected) + 1e-30;
        case F16Case::L2: return 1e-4 * std::fabs(expected) + 1e-30;
        default: return 0.0;
    }
}

void expectMatches(double expected, float got, double tolerance, uint64_t row) {
    if (std::isnan(expected)) {
        ASSERT_TRUE(std::isnan(got)) << "row " << row << " got " << got;
    } else if (std::isinf(expected)) {
        ASSERT_EQ(expected, static_cast<double>(got)) << "row " << row;
    } else {
        ASSERT_NEAR(expected, static_cast<double>(got), tolerance) << "row " << row;
    }
}

}  // namespace

TEST(CubReductionF16KR, ValueEveryPoolShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(1233);
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [rows, elements] : f16Sites()) {
            // 2-byte aligned views: 16-byte aligned, and 2, 4 and 6 bytes past it (2 and 6 start mid-word).
            const uint64_t offset = 2 * ((rows + elements) % 4);
            for (const F16CaseSpec& spec : F16_CASES) {
                if (spec.kind == F16Case::Product && elements > 8) continue;
                const F16Input in = uploadF16(f16Values(rows * elements, spec.kind == F16Case::Product, rng), dtype,
                                              rows, elements, offset, stream);
                std::vector<double> expected(rows);
                for (uint64_t row = 0; row < rows; ++row)
                    expected[row] = f16Reference(spec.kind, in.values.data() + row * elements, elements);
                const CubReductionStageSemantics semantics = f16Semantics(spec.kind, elements);
                for (const auto& shape : f16ShapesToTest(rows, elements)) {
                    SCOPED_TRACE(::testing::Message() << formatName(dtype) << " rows=" << rows << " R=" << elements
                                                      << " offset=" << offset << " case=" << spec.name
                                                      << " shape=" << KR::fp8KRShapeName(shape));
                    std::optional<Tensor> scratch = f16ScratchFor(shape, stream);
                    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                    for (int launch = 0; launch < 2; ++launch) {  // a repeated launch on the same scratch
                        launchF16KRValueSegments(semantics, in.input, output, shape,
                                                 scratch ? scratch->getMemPtr<void>() : nullptr,
                                                 scratch ? scratch->getArraySizeInBytes() : 0, 1.0f, stream);
                    }
                    stream.synchronize();
                    const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
                    for (uint64_t row = 0; row < rows; ++row)
                        expectMatches(expected[row], got[row], f16Tolerance(spec.kind, expected[row]), row);
                }
            }
        }
    }
}

// NaN, infinities and signed zeros: NaN propagates through SUM and the folded MIN / MAX, and MAX's sign fold keeps
// infinities in order. Rows of at most 4096 elements keep the sums of +-1024 exact.
TEST(CubReductionF16KR, ValueSpecialsPropagate) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(12331);
    static constexpr float SPECIALS[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.5f, -2.5f, 1024.0f, -1024.0f, INFINITY, -INFINITY};
    const std::vector<std::pair<uint64_t, uint64_t>> sites{{1, 7},    {1000, 5}, {1001, 3}, {64, 31}, {17, 129},
                                                           {8, 2250}, {300, 250}, {7, 9},   {2, 4095}, {333, 16}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [rows, elements] : sites) {
            const uint64_t offset = 2 * ((rows + elements) % 4);
            std::vector<float> values(rows * elements);
            for (uint64_t row = 0; row < rows; ++row) {
                // Every third row is finite; the others take infinities, and every third of those one NaN.
                for (uint64_t i = 0; i < elements; ++i)
                    values[row * elements + i] = SPECIALS[rng() % (row % 3 == 0 ? 8 : 10)];
                if (row % 3 == 2) values[row * elements + rng() % elements] = NAN;
            }
            const F16Input in = uploadF16(values, dtype, rows, elements, offset, stream);
            for (const F16Case kind :
                 {F16Case::Sum, F16Case::Max, F16Case::Min, F16Case::MaxAbs, F16Case::MinAbs, F16Case::MinSquare}) {
                const CubReductionStageSemantics semantics = f16Semantics(kind, elements);
                for (const auto& shape : f16ShapesToTest(rows, elements)) {
                    SCOPED_TRACE(::testing::Message() << formatName(dtype) << " rows=" << rows << " R=" << elements
                                                      << " case=" << static_cast<int>(kind)
                                                      << " shape=" << KR::fp8KRShapeName(shape));
                    std::optional<Tensor> scratch = f16ScratchFor(shape, stream);
                    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                    launchF16KRValueSegments(semantics, in.input, output, shape,
                                             scratch ? scratch->getMemPtr<void>() : nullptr,
                                             scratch ? scratch->getArraySizeInBytes() : 0, 1.0f, stream);
                    stream.synchronize();
                    const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
                    for (uint64_t row = 0; row < rows; ++row)
                        expectMatches(f16Reference(kind, in.values.data() + row * elements, elements), got[row], 0.0,
                                      row);
                }
            }
        }
    }
}

// A runtime output scale (negative: MAX's folded negation composes with it) and a narrow output dtype apply once.
TEST(CubReductionF16KR, ValueScaleAndOutputDtype) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(12332);
    const uint64_t rows = 3, elements = 17501;
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        const F16Input in = uploadF16(f16Values(rows * elements, false, rng), dtype, rows, elements, 2, stream);
        for (const F16Case kind : {F16Case::Mean, F16Case::Max}) {
            const auto semantics = f16Semantics(kind, elements);
            for (const auto& shape : f16ShapesToTest(rows, elements)) {
                SCOPED_TRACE(::testing::Message() << formatName(dtype) << " " << static_cast<int>(kind) << " "
                                                  << KR::fp8KRShapeName(shape));
                std::optional<Tensor> scratch = f16ScratchFor(shape, stream);
                Tensor output(gpuPlacement, TensorDescriptor(DataType::BF16, {rows}));
                launchF16KRValueSegments(semantics, in.input, output, shape,
                                         scratch ? scratch->getMemPtr<void>() : nullptr,
                                         scratch ? scratch->getArraySizeInBytes() : 0, -2.5f, stream);
                stream.synchronize();
                const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
                for (uint64_t row = 0; row < rows; ++row) {
                    const double expected = -2.5 * f16Reference(kind, in.values.data() + row * elements, elements);
                    ASSERT_NEAR(expected, got[row], 1e-2 * std::fabs(expected) + 1e-3) << "row " << row;
                }
            }
        }
    }
}

// 1234: vector Pack at every 2-byte length, both formats, every semantics: row counts below one thread's rows, whole
// threads, a partial last owner, and several blocks; 16-byte aligned (the vector kernel) and 2 bytes off (the same
// shape through the generic Pack kernel).
TEST(CubReductionF16KR, ValueVectorPackMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(1234);
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (uint64_t elements = 1; elements <= 8; ++elements) {
            const uint64_t row_bytes = 2 * elements;
            const uint64_t per_thread = KR::fp8KRVectorRows(row_bytes);
            const KR::Fp8KRSegmentShape probe{1, row_bytes, row_bytes, 1, static_cast<uint32_t>(per_thread), true};
            ASSERT_TRUE(KR::retainedKRShapeLegalForElement(probe, 2));
            for (const uint64_t rows : {uint64_t{1}, per_thread, 5 * per_thread + per_thread / 2 + 1,
                                        KR::FP8_KR_THREADS * per_thread * 3 + 7}) {
                for (const uint64_t offset : {uint64_t{0}, uint64_t{2}}) {
                    for (const F16CaseSpec& spec : F16_CASES) {
                        const F16Input in = uploadF16(f16Values(rows * elements, spec.kind == F16Case::Product, rng),
                                                      dtype, rows, elements, offset, stream);
                        const KR::Fp8KRSegmentShape shape{rows, row_bytes, row_bytes, 1,
                                                          static_cast<uint32_t>(per_thread), true};
                        SCOPED_TRACE(::testing::Message() << formatName(dtype) << " rows=" << rows << " R=" << elements
                                                          << " offset=" << offset << " case=" << spec.name);
                        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                        launchF16KRValueSegments(f16Semantics(spec.kind, elements), in.input, output, shape, nullptr, 0,
                                                 1.0f, stream);
                        stream.synchronize();
                        const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
                        for (uint64_t row = 0; row < rows; ++row) {
                            const double expected =
                                f16Reference(spec.kind, in.values.data() + row * elements, elements);
                            expectMatches(expected, got[row], f16Tolerance(spec.kind, expected), row);
                        }
                    }
                }
            }
        }
    }
}

// The launcher rejects what the 2-byte kernels do not cover.
TEST(CubReductionF16KR, ValueRejectsUnsupportedInputs) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    std::mt19937 rng(12333);
    const uint64_t rows = 8, elements = 4;
    const auto semantics = f16Semantics(F16Case::Sum, elements);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
    const F16Input in = uploadF16(f16Values(rows * elements, false, rng), DataType::FP16, rows, elements, 0, stream);
    const KR::Fp8KRSegmentShape group{rows, 2 * elements, 2 * elements, 4, 1, false};
    // A shape in elements instead of bytes.
    EXPECT_THROW(launchF16KRValueSegments(semantics, in.input, output, {rows, elements, elements, 4, 1, false},
                                          nullptr, 0, 1.0f, stream),
                 std::logic_error);
    // A vector shape at an FP8-only vector length (3 bytes: not whole elements) or the wrong rows per thread.
    EXPECT_THROW(launchF16KRValueSegments(semantics, in.input, output, {rows, 8, 8, 1, 4, true}, nullptr, 0, 1.0f,
                                          stream),
                 std::logic_error);
    const F16Input nine = uploadF16(f16Values(rows * 9, false, rng), DataType::BF16, rows, 9, 0, stream);
    EXPECT_THROW(launchF16KRValueSegments(f16Semantics(F16Case::Sum, 9), nine.input, output,
                                          {rows, 18, 18, 1, KR::fp8KRVectorRows(18), true}, nullptr, 0, 1.0f, stream),
                 std::logic_error);
    // FP8 and FP32 input to the 2-byte launcher.
    Tensor bytes = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {rows, 2 * elements}, 0);
    EXPECT_THROW(launchF16KRValueSegments(semantics, bytes, output, group, nullptr, 0, 1.0f, stream),
                 std::logic_error);
    Tensor words = in.storage.aliasByteStorageAs(DataType::FP32, {rows, elements / 2}, 0);
    EXPECT_THROW(launchF16KRValueSegments(semantics, words, output, group, nullptr, 0, 1.0f, stream),
                 std::logic_error);
    // A Split shape without scratch.
    const uint64_t long_elements = 8192;
    const F16Input long_in = uploadF16(f16Values(long_elements, false, rng), DataType::FP16, 1, long_elements, 0,
                                       stream);
    Tensor one(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    EXPECT_THROW(launchF16KRValueSegments(f16Semantics(F16Case::Sum, long_elements), long_in.input, one,
                                          {1, 2 * long_elements, 4096, KR::FP8_KR_THREADS, 1, false}, nullptr, 0,
                                          1.0f, stream),
                 std::logic_error);
    // The legal shape runs.
    EXPECT_NO_THROW(launchF16KRValueSegments(semantics, in.input, output, group, nullptr, 0, 1.0f, stream));
    stream.synchronize();
}

namespace {

// The 16-bit patterns as stored (any pattern: NaN, infinity, subnormal, either zero).
float decodeF16Bits(uint16_t bits, DataType dtype) {
    if (dtype == DataType::BF16) {
        const uint32_t u = uint32_t{bits} << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    const uint32_t sign = (bits & 0x8000U) << 16, exponent = (bits >> 10) & 31U, mantissa = bits & 0x3FFU;
    if (exponent == 0) {
        const float v = std::ldexp(static_cast<float>(mantissa), -24);
        return sign != 0 ? -v : v;
    }
    const uint32_t u = exponent == 31 ? sign | 0x7F800000U | (mantissa << 13)
                                      : sign | ((exponent + 112U) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct F16BitsInput {
    Tensor storage;
    Tensor input;
    std::vector<float> values;  // decoded
};

F16BitsInput uploadF16Bits(const std::vector<uint16_t>& bits, DataType dtype, uint64_t rows, uint64_t elements,
                           uint64_t offset, Stream& stream) {
    std::vector<uint64_t> padded(offset + 2 * bits.size() + 4, 0x7E);
    F16BitsInput result;
    result.values.resize(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        padded[offset + 2 * i] = bits[i] & 0xFFU;
        padded[offset + 2 * i + 1] = bits[i] >> 8;
        result.values[i] = decodeF16Bits(bits[i], dtype);
    }
    result.storage = makeGpuUnsignedTensor(padded, {padded.size()}, stream, DataType::UINT8);
    result.input = result.storage.aliasByteStorageAs(dtype, {rows, elements}, offset);
    return result;
}

// Random bit patterns with the specials mixed in; every third seed is tie-heavy (+-1 and both zeros only).
std::vector<uint16_t> f16ArgBits(uint64_t count, DataType dtype, uint32_t seed) {
    const bool bf16 = dtype == DataType::BF16;
    const std::vector<uint16_t> specials = bf16 ? std::vector<uint16_t>{0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x3F80,
                                                                       0xBF80, 0x7F7F, 0xFF7F, 0x0001}
                                                : std::vector<uint16_t>{0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x3C00,
                                                                       0xBC00, 0x7BFF, 0xFBFF, 0x0001};
    const std::vector<uint16_t> ties = bf16 ? std::vector<uint16_t>{0x3F80, 0xBF80, 0x0000, 0x8000}
                                            : std::vector<uint16_t>{0x3C00, 0xBC00, 0x0000, 0x8000};
    std::mt19937 rng(seed);
    std::vector<uint16_t> bits(count);
    const bool tie_heavy = (seed % 3U) == 0;
    for (uint16_t& b : bits) {
        if (tie_heavy) b = ties[rng() % ties.size()];
        else if ((rng() % 64U) == 0) b = specials[rng() % specials.size()];
        else b = static_cast<uint16_t>(rng());
    }
    return bits;
}

std::optional<Tensor> f16ArgScratchFor(const KR::Fp8KRSegmentShape& shape, Stream& stream) {
    const uint64_t bytes = KR::fp8KRScratchBytes(shape, true);
    if (bytes == 0) return std::nullopt;
    Tensor scratch(gpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    scratch.fillRandom(0.0, 255.0, stream);
    stream.synchronize();
    return scratch;
}

}  // namespace

TEST(CubReductionF16KR, ArgShapesAreTheTwoByteShapesWithoutVectorPack) {
    for (const auto& [rows, elements] : f16Sites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        for (const auto& entry : KR::retainedKRCandidatePool(rows, 2 * elements, 170, 2))
            EXPECT_EQ(!entry.shape.vector, KR::f16KRArgShapeLegal(entry.shape)) << KR::fp8KRShapeName(entry.shape);
    }
    EXPECT_FALSE(KR::f16KRArgShapeLegal({8, 6, 6, 1, 8, true}));
    EXPECT_FALSE(KR::f16KRArgShapeLegal({8, 7, 7, 4, 1, false}));
    EXPECT_TRUE(KR::f16KRArgShapeLegal({8, 6, 6, 4, 1, false}));
    EXPECT_TRUE(KR::f16KRArgShapeLegal({1, uint64_t{1} << 48, uint64_t{1} << 40, KR::FP8_KR_THREADS, 1, false}));
}

TEST(CubReductionF16KR, ArgEveryPoolShapeMatchesReference) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    namespace Reference = ThorImplementation::Fp8ArgReference;
    Stream stream(0);
    uint32_t seed = 1237;
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [rows, elements] : f16Sites()) {
            const uint64_t offset = 2 * ((rows * 3 + elements) % 4);
            const F16BitsInput in =
                uploadF16Bits(f16ArgBits(rows * elements, dtype, ++seed), dtype, rows, elements, offset, stream);
            for (const bool maximum : {false, true}) {
                std::vector<Reference::Candidate> expected(rows);
                for (uint64_t row = 0; row < rows; ++row) {
                    Reference::Candidate best{0, in.values[row * elements]};
                    for (uint64_t i = 1; i < elements; ++i)
                        best = Reference::combine(best, {i, in.values[row * elements + i]}, maximum);
                    expected[row] = best;
                }
                for (const auto& entry : KR::retainedKRCandidatePool(rows, 2 * elements, 170, 2)) {
                    const KR::Fp8KRSegmentShape& shape = entry.shape;
                    if (!KR::f16KRArgShapeLegal(shape)) continue;
                    std::optional<Tensor> scratch = f16ArgScratchFor(shape, stream);
                    for (const bool with_value : {true, false}) {
                        const DataType index_dtype = with_value ? DataType::UINT64 : DataType::UINT32;
                        SCOPED_TRACE(::testing::Message() << formatName(dtype) << " rows=" << rows << " R=" << elements
                                                          << " offset=" << offset << " max=" << maximum
                                                          << " value=" << with_value
                                                          << " shape=" << KR::fp8KRShapeName(shape));
                        std::optional<Tensor> value;
                        if (with_value) value = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {rows}));
                        Tensor index(gpuPlacement, TensorDescriptor(index_dtype, {rows}));
                        const uint64_t stride = with_value ? 3 : 1;
                        for (int launch = 0; launch < 2; ++launch) {
                            launchF16KRArgSegments(in.input, value ? &*value : nullptr, &index, shape,
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
                                ASSERT_TRUE(Reference::sameValue(expected[row].value, got_values[row]))
                                    << "row " << row << " expected " << expected[row].value << " got "
                                    << got_values[row];
                        }
                    }
                }
            }
        }
    }
}

// The ARG launcher rejects what the 2-byte ARG kernels do not cover.
TEST(CubReductionF16KR, ArgRejectsUnsupportedInputs) {
    REQUIRE_CUDA_DEVICE();
    using namespace CubReductionInternal;
    Stream stream(0);
    const uint64_t rows = 8, elements = 4;
    const F16BitsInput in = uploadF16Bits(f16ArgBits(rows * elements, DataType::FP16, 7), DataType::FP16, rows,
                                          elements, 0, stream);
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {rows}));
    const KR::Fp8KRSegmentShape group{rows, 2 * elements, 2 * elements, 4, 1, false};
    // A shape in elements instead of bytes; vector Pack; no outputs; a wrong output size; FP8 input.
    EXPECT_THROW(launchF16KRArgSegments(in.input, nullptr, &index, {rows, elements, elements, 4, 1, false}, nullptr, 0,
                                        false, 1, stream),
                 std::logic_error);
    EXPECT_THROW(launchF16KRArgSegments(in.input, nullptr, &index, {rows, 8, 8, 1, 2, true}, nullptr, 0, false, 1,
                                        stream),
                 std::logic_error);
    EXPECT_THROW(launchF16KRArgSegments(in.input, nullptr, nullptr, group, nullptr, 0, false, 1, stream),
                 std::logic_error);
    Tensor short_index(gpuPlacement, TensorDescriptor(DataType::UINT32, {rows - 1}));
    EXPECT_THROW(launchF16KRArgSegments(in.input, nullptr, &short_index, group, nullptr, 0, false, 1, stream),
                 std::logic_error);
    Tensor bytes = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {rows, 2 * elements}, 0);
    EXPECT_THROW(launchF16KRArgSegments(bytes, nullptr, &index, group, nullptr, 0, false, 1, stream),
                 std::logic_error);
    // A UINT32 index that cannot hold row-local index * stride.
    EXPECT_THROW(launchF16KRArgSegments(in.input, nullptr, &index, group, nullptr, 0, false, 1ULL << 31, stream),
                 std::logic_error);
    // A Split shape without scratch.
    const uint64_t long_elements = 8192;
    const F16BitsInput long_in = uploadF16Bits(f16ArgBits(long_elements, DataType::BF16, 8), DataType::BF16, 1,
                                               long_elements, 0, stream);
    Tensor one(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    EXPECT_THROW(launchF16KRArgSegments(long_in.input, nullptr, &one,
                                        {1, 2 * long_elements, 4096, KR::FP8_KR_THREADS, 1, false}, nullptr, 0, false,
                                        1, stream),
                 std::logic_error);
    EXPECT_NO_THROW(launchF16KRArgSegments(in.input, nullptr, &index, group, nullptr, 0, true, 1, stream));
    stream.synchronize();
}
