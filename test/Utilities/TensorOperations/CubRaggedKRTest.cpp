// 1265: retained ragged KR VALUE for FP8 input at D = 1 (launchRaggedKRFp8Value, CubRaggedKRFp8.cu, with the finisher
// in CubRaggedFinish.cu; not yet routed). Exact against a host reference on dyadic inputs (multiples of 0.5, so every
// FP32 partial sum is exact in any order) for both formats, SUM / MEAN / MIN / MAX, every audit distribution, both
// shapes (1268: weighted and unweighted splits), grids from 1 CTA to the persistent grid, UINT32 and UINT64 offsets. Also: inactive capacity poisoned with
// NaN, misaligned input bases, garbage workspace, NaN propagation, bit-identical repeated runs, narrow output dtypes,
// empty segments and the launcher's contract. Seeds are in every failure message.
// 1280: D = 2..15 (the components kernel): every D against the reference per component, misaligned bases, NaN, both
// offset widths and the finisher's one CTA per component; the 32-rows-per-lane shapes are gone.
// 1284 (phase 3b): one launcher for every input dtype (launchRaggedKRValue, CubRaggedKR.cu; the kernels in
// RaggedKRKernels.cuh): FP16, BF16 (D 1..7) and FP32 (D 1..3) through the same tests, element-aligned bases, and the
// contract's per-dtype bounds.
// 1285 (phase 4): ragged RK VALUE (launchRaggedRKValue, CubRaggedRK.cu) for FP8 rows of 16 bytes and more through the
// same reference: D 16 to 5000, every K tile cap, CTA budgets from 1, misaligned bases, NaN, empty and dense segments,
// narrow outputs, the launch parts apart, and the contract.
// 1286: RK's vector end stores: every output dtype against the FP32 output converted on the host by the same
// conversions, on short segments at D aligned and not (every FP32 misalignment, FP16 / BF16 pairs and singles, FP8
// aligned and scalar, partial chunks), every op with NaN.
// 1295: ragged RK for FP16, BF16 (D >= 8) and FP32 (D >= 4) input: narrow, two-row and K-tiled rows against the
// reference over caps and CTA budgets, element-aligned misaligned bases, NaN, empty and dense segments, and each
// output dtype against the FP32 output converted on the host.
// 1298 (phase 5): ragged KR ARG for FP8 at D = 1 (launchRaggedKRArg): random bit patterns, specials and tie-heavy pools
// against the legacy semantics over grids, shapes, both offset and index widths; ties split across CTAs (equal values,
// NaNs, -0 / +0, a planted extremum twice); inactive capacity poisoned with NaN, misaligned bases, garbage workspace,
// repeats; empty segments, tiny problems, the launch parts apart, and the contract. 1300: D = 2..15 against the reference
// per component, and ties split across CTAs at D > 1. 1311: every ARG test also runs FP16 and BF16 (D 1 to 7).
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"
#include "benchmarks/RaggedReductionAuditCells.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
using namespace ThorImplementation::CubReductionInternal;
namespace A = ThorImplementation::RaggedReductionAudit;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

struct Format {
    DataType dtype;
    A::StorageFormat format;
    const char* name;
    uint32_t bytes;  // 1284: the element's bytes
    uint32_t nan;    // a NaN encoding (little-endian, `bytes` wide)
};
const Format FORMATS[] = {{DataType::FP8_E4M3, A::StorageFormat::FP8_E4M3, "e4m3", 1, 0x7F},
                          {DataType::FP8_E5M2, A::StorageFormat::FP8_E5M2, "e5m2", 1, 0x7F}};
// 1284: the 16- and 32-bit inputs.
const Format WIDE_FORMATS[] = {{DataType::FP16, A::StorageFormat::FP16, "fp16", 2, 0x7E00},
                               {DataType::BF16, A::StorageFormat::BF16, "bf16", 2, 0x7FC0},
                               {DataType::FP32, A::StorageFormat::FP32, "fp32", 4, 0x7FC00000}};
const float VALUES[] = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};

constexpr size_t VALUE_COUNT = sizeof(VALUES) / sizeof(VALUES[0]);

uint32_t maxComponents(const Format& f) { return RaggedMergePath::raggedKRMaxComponents(f.bytes); }

void putElement(std::vector<uint64_t>& bytes, uint64_t at, uint32_t bits, uint32_t width) {
    for (uint32_t i = 0; i < width; ++i) bytes[at + i] = (bits >> (8 * i)) & 0xFFU;
}

// VALUES' encodings in `f` (+0 for zero), checked against the audit's decoder (1284: little-endian, `f.bytes` wide;
// FP8 and the 16-bit formats by search, FP32 by its bits).
std::vector<uint32_t> encodings(const Format& f) {
    std::vector<uint32_t> out(VALUE_COUNT, 0);
    const uint32_t sign_bit = 1U << (8 * f.bytes - 1);
    for (size_t v = 0; v < VALUE_COUNT; ++v) {
        bool found = false;
        if (f.bytes == 4) {
            std::memcpy(&out[v], &VALUES[v], 4);
            found = true;
        }
        for (uint32_t b = 0; f.bytes < 4 && b < (1U << (8 * f.bytes)) && !found; ++b) {
            const uint8_t le[2] = {static_cast<uint8_t>(b & 0xFFU), static_cast<uint8_t>(b >> 8)};
            if (A::decodeElement(f.format, le) == VALUES[v] && !(VALUES[v] == 0.0f && (b & sign_bit))) {
                out[v] = b;
                found = true;
            }
        }
        uint8_t le[4];
        for (uint32_t i = 0; i < 4; ++i) le[i] = static_cast<uint8_t>(out[v] >> (8 * i));
        if (!found || A::decodeElement(f.format, le) != VALUES[v])
            ADD_FAILURE() << VALUES[v] << " has no exact " << f.name << " encoding";
    }
    return out;
}

struct RaggedInput {
    std::vector<float> values;  // active rows x D
    std::vector<uint64_t> offsets;
    uint64_t d = 1;
    Tensor storage, input, offsets_tensor;
};

// Active rows of dyadic values; rows past `rows` up to the capacity hold NaN (never decoded); `misalign` bytes before.
// 1280: D components per row ([capacity, D] when D > 1); `nan_elements` index rows x D. 1284: `misalign` is in bytes
// (a multiple of the element's bytes, which the launcher requires).
RaggedInput makeInput(const Format& f, uint64_t rows, uint64_t capacity, uint64_t segments, A::Distribution d,
                      uint32_t offset_width, uint64_t misalign, uint64_t seed, Stream& stream,
                      const std::vector<uint64_t>& nan_elements = {}, uint64_t components = 1) {
    RaggedInput in;
    in.d = components;
    in.values.resize(rows * components);
    const std::vector<uint32_t> code = encodings(f);
    const uint32_t width = f.bytes;
    EXPECT_EQ(misalign % width, 0U) << f.name;
    std::vector<uint64_t> bytes(misalign + capacity * components * width + 16, 0x11);
    for (uint64_t e = 0; e < capacity * components; ++e) {
        if (e < rows * components) {
            const uint64_t v = A::mix(seed ^ (e * 0x9E37)) % VALUE_COUNT;
            in.values[e] = VALUES[v];
            putElement(bytes, misalign + e * width, code[v], width);
        } else {
            putElement(bytes, misalign + e * width, f.nan, width);
        }
    }
    for (uint64_t e : nan_elements) {
        in.values[e] = std::numeric_limits<float>::quiet_NaN();
        putElement(bytes, misalign + e * width, f.nan, width);
    }
    in.storage = makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8);
    in.input = components == 1 ? in.storage.aliasByteStorageAs(f.dtype, {capacity}, misalign)
                               : in.storage.aliasByteStorageAs(f.dtype, {capacity, components}, misalign);
    in.offsets = A::makeOffsets(rows, segments, d, seed);
    in.offsets_tensor =
        makeGpuUnsignedTensor(in.offsets, {segments + 1}, stream, offset_width == 8 ? DataType::UINT64 : DataType::UINT32);
    return in;
}

std::vector<float> reference(const RaggedInput& in, CubReductionOp op) {
    const uint64_t segments = in.offsets.size() - 1, D = in.d;
    std::vector<float> out(segments * D);
    for (uint64_t o = 0; o < segments * D; ++o) {
        const uint64_t s = o / D, c = o % D;
        const uint64_t begin = in.offsets[s], end = in.offsets[s + 1];
        double sum = 0;
        float lo = std::numeric_limits<float>::infinity(), hi = -std::numeric_limits<float>::infinity();
        bool nan = false;
        for (uint64_t r = begin; r < end; ++r) {
            const float v = in.values[r * D + c];
            if (std::isnan(v)) nan = true;
            sum += v;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        switch (op) {
            case CubReductionOp::Sum: out[o] = nan ? NAN : static_cast<float>(sum); break;
            case CubReductionOp::Mean:
                out[o] = end == begin ? 0.0f : (nan ? NAN : static_cast<float>(sum) / static_cast<float>(end - begin));
                break;
            case CubReductionOp::Min: out[o] = nan ? NAN : lo; break;
            default: out[o] = nan ? NAN : hi; break;
        }
    }
    return out;
}

std::vector<float> run(const RaggedInput& in, CubReductionOp op, const RaggedKRShape& shape, uint32_t grid,
                       Stream& stream, DataType output_dtype = DataType::FP32, uint64_t garbage_seed = 7) {
    const uint64_t segments = in.offsets.size() - 1;
    Tensor output(gpuPlacement, in.d == 1 ? TensorDescriptor(output_dtype, {segments})
                                          : TensorDescriptor(output_dtype, {segments, in.d}));
    std::vector<uint64_t> garbage(raggedValueWorkspaceBytes(grid, static_cast<uint32_t>(in.d)));
    for (size_t i = 0; i < garbage.size(); ++i) garbage[i] = A::mix(garbage_seed + i) & 0xFF;
    Tensor workspace = makeGpuUnsignedTensor(garbage, {garbage.size()}, stream, DataType::UINT8);
    launchRaggedKRValue(op, shape, grid, in.input, in.offsets_tensor, output, workspace, stream);
    stream.synchronize();
    return copyGpuTensorAsFloat(output, stream);
}

// 1285: the same through ragged RK on a CTA budget.
std::vector<float> runRK(const RaggedInput& in, CubReductionOp op, const RaggedRKShape& shape, uint32_t ctas,
                         Stream& stream, DataType output_dtype = DataType::FP32, uint64_t garbage_seed = 7) {
    const uint64_t segments = in.offsets.size() - 1;
    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, {segments, in.d}));
    const RaggedRKLaunch launch = raggedRKLaunch(shape, ctas, in.input.getDataType(), in.d, in.input.getDimensions()[0]);
    std::vector<uint64_t> garbage(launch.workspace_bytes);
    for (size_t i = 0; i < garbage.size(); ++i) garbage[i] = A::mix(garbage_seed + i) & 0xFF;
    Tensor workspace = makeGpuUnsignedTensor(garbage, {garbage.size()}, stream, DataType::UINT8);
    launchRaggedRKValue(op, shape, ctas, in.input, in.offsets_tensor, output, workspace, stream);
    stream.synchronize();
    return copyGpuTensorAsFloat(output, stream);
}

void expectExact(const std::vector<float>& got, const std::vector<float>& expected, const std::string& what) {
    ASSERT_EQ(got.size(), expected.size()) << what;
    for (size_t s = 0; s < got.size(); ++s) {
        if (std::isnan(expected[s])) {
            ASSERT_TRUE(std::isnan(got[s])) << what << " segment " << s << " got " << got[s];
        } else {
            ASSERT_EQ(got[s], expected[s]) << what << " segment " << s;
        }
    }
}

const CubReductionOp OPS[] = {CubReductionOp::Sum, CubReductionOp::Mean, CubReductionOp::Min, CubReductionOp::Max};
const char* opName(CubReductionOp op) {
    return op == CubReductionOp::Sum ? "sum" : op == CubReductionOp::Mean ? "mean" : op == CubReductionOp::Min ? "min" : "max";
}

}  // namespace

TEST(CubRaggedKR, Fp8MatchesTheExactReferenceAcrossDistributionsShapesAndGrids) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t seed = 1265;
    for (const Format& f : FORMATS) {
        for (uint64_t trial = 0; trial < 5; ++trial) {
            const A::Distribution d = A::DISTRIBUTIONS[trial];
            const uint64_t h = A::mix(seed + trial);
            const uint64_t rows = 1 + h % 60000;
            const uint64_t segments = 1 + A::mix(h) % (trial % 2 ? rows : 2000);
            const RaggedInput in = makeInput(f, rows, rows, segments, d, trial % 2 ? 8 : 4, 0, h, stream);
            for (const RaggedKRShape shape : {RaggedKRShape{16, 0}, RaggedKRShape{16, 0, false},
                                              RaggedKRShape{16, 0, true, 40}}) {
                const uint32_t persistent = raggedKRGrid(shape, f.dtype, 0);
                for (uint32_t grid : {1U, 2U, 7U, 64U, persistent}) {
                    for (CubReductionOp op : OPS) {
                        const std::string what = std::string(f.name) + " " + A::distributionName(d) + " seed " +
                                                 std::to_string(h) + " " + raggedKRShapeName(shape) + " grid " +
                                                 std::to_string(grid) + " " + opName(op);
                        expectExact(run(in, op, shape, grid, stream), reference(in, op), what);
                    }
                }
            }
        }
    }
}

TEST(CubRaggedKR, Fp8IgnoresInactiveCapacityAndGarbageWorkspaceAtAnyAlignment) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : FORMATS) {
        for (uint64_t misalign = 0; misalign < 16; ++misalign) {  // 1268: rows stage in 16-byte chunks
            const uint64_t h = A::mix(0xCAFE + misalign);
            const uint64_t rows = 5000 + h % 5000;
            const RaggedInput in =
                makeInput(f, rows, rows + rows / 2, 1 + h % 300, A::Distribution::Pareto, 4, misalign, h, stream);
            for (CubReductionOp op : OPS) {
                const std::string what = std::string(f.name) + " misalign " + std::to_string(misalign) + " " + opName(op);
                expectExact(run(in, op, RaggedKRShape{16, 0}, 37, stream, DataType::FP32, h), reference(in, op), what);
            }
        }
    }
}

TEST(CubRaggedKR, Fp8PropagatesNaNAndRepeatsBitForBit) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : FORMATS) {
        const uint64_t rows = 40000;
        const std::vector<uint64_t> nan_rows{5, 777, 20000, 39999};
        const RaggedInput in = makeInput(f, rows, rows, 600, A::Distribution::Random, 4, 0, 0xBEEF, stream, nan_rows);
        for (CubReductionOp op : OPS) {
            const std::vector<float> first = run(in, op, RaggedKRShape{16, 0}, 23, stream);
            expectExact(first, reference(in, op), std::string(f.name) + " NaN " + opName(op));
            const std::vector<float> again = run(in, op, RaggedKRShape{16, 0}, 23, stream, DataType::FP32, 99);
            for (size_t s = 0; s < first.size(); ++s) {
                uint32_t a, b;
                std::memcpy(&a, &first[s], 4);
                std::memcpy(&b, &again[s], 4);
                if (!std::isnan(first[s])) ASSERT_EQ(a, b) << f.name << " " << opName(op) << " segment " << s;
            }
        }
    }
}

// 1275: the launch parts the audit times alone: the main kernel then the finisher, as two launches on one workspace,
// computes what one launch does.
TEST(CubRaggedKR, MainThenFinishLaunchedApartMatchOneLaunch) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const Format& f = FORMATS[0];
    const RaggedInput in = makeInput(f, 50000, 50000, 3000, A::Distribution::Pareto, 4, 3, 0x1275, stream);
    const RaggedKRShape shape{16, 0};
    const uint32_t grid = raggedKRGrid(shape, f.dtype, 0);
    for (CubReductionOp op : OPS) {
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {in.offsets.size() - 1}));
        Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(grid)}));
        launchRaggedKRValue(op, shape, grid, in.input, in.offsets_tensor, output, workspace, stream,
                            RaggedLaunchParts::MainOnly);
        launchRaggedKRValue(op, shape, grid, in.input, in.offsets_tensor, output, workspace, stream,
                            RaggedLaunchParts::FinishOnly);
        stream.synchronize();
        const std::vector<float> apart = copyGpuTensorAsFloat(output, stream);
        const std::vector<float> together = run(in, op, shape, grid, stream);
        ASSERT_EQ(apart.size(), together.size());
        for (size_t s = 0; s < apart.size(); ++s) {
            uint32_t a, b;
            std::memcpy(&a, &apart[s], 4);
            std::memcpy(&b, &together[s], 4);
            ASSERT_EQ(a, b) << opName(op) << " segment " << s;
        }
    }
}

TEST(CubRaggedKR, Fp8StoresNarrowOutputDtypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // MIN / MAX outputs are input values, exact in every output dtype; SUM over three rows (at most 12, in halves) is
    // exact in FP16 and BF16 but not in FP8.
    // 1284: and from 16- and 32-bit inputs.
    for (const Format& f : {FORMATS[0], WIDE_FORMATS[0], WIDE_FORMATS[2]}) {
        const RaggedInput in = makeInput(f, 3000, 3000, 1000, A::Distribution::Uniform, 4, 0, 42, stream);
        for (DataType out : {DataType::FP16, DataType::BF16, DataType::FP8_E4M3, DataType::FP32}) {
            for (CubReductionOp op : {CubReductionOp::Min, CubReductionOp::Max, CubReductionOp::Sum}) {
                if (out == DataType::FP8_E4M3 && op == CubReductionOp::Sum) continue;
                expectExact(run(in, op, RaggedKRShape{16, 0}, 11, stream, out), reference(in, op),
                            std::string(f.name) + " output dtype " + std::to_string(static_cast<int>(out)) + " " +
                                opName(op));
            }
        }
    }
}

TEST(CubRaggedKR, EmptySegmentsTinyProblemsAndTheContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // Every segment empty: SUM / MEAN 0, MIN +inf, MAX -inf.
    const RaggedInput empty = makeInput(FORMATS[1], 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, stream);
    for (CubReductionOp op : OPS) expectExact(run(empty, op, RaggedKRShape{16, 0}, 9, stream), reference(empty, op), opName(op));

    // More CTAs than path events: the empty CTAs are last and write identity records.
    const RaggedInput tiny = makeInput(FORMATS[0], 3, 3, 2, A::Distribution::Random, 4, 2, 6, stream);
    for (CubReductionOp op : OPS) expectExact(run(tiny, op, RaggedKRShape{16, 0}, 64, stream), reference(tiny, op), opName(op));

    // The contract.
    Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4)}));
    const RaggedInput in = makeInput(FORMATS[0], 100, 100, 10, A::Distribution::Uniform, 4, 0, 5, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {10}));
    const auto launch = [&](CubReductionOp op, RaggedKRShape shape, uint32_t grid, const Tensor& input, Tensor& ws) {
        launchRaggedKRValue(op, shape, grid, input, in.offsets_tensor, output, ws, stream);
    };
    EXPECT_THROW(launch(CubReductionOp::Product, {16, 0}, 4, in.input, workspace), std::invalid_argument);
    EXPECT_THROW(launch(CubReductionOp::Sum, {4, 0}, 4, in.input, workspace), std::invalid_argument);
    EXPECT_THROW(launch(CubReductionOp::Sum, {32, 0}, 4, in.input, workspace), std::invalid_argument);  // 1280
    EXPECT_THROW(launch(CubReductionOp::Sum, {16, 0}, 0, in.input, workspace), std::invalid_argument);
    EXPECT_THROW(launch(CubReductionOp::Sum, {16, 0}, RAGGED_FINISH_MAX_GRID + 1, in.input, workspace), std::invalid_argument);
    Tensor small(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4) - 1}));
    EXPECT_THROW(launch(CubReductionOp::Sum, {16, 0}, 4, in.input, small), std::invalid_argument);
    // 1284: FP16 / BF16 / FP32 input is legal; integers are not.
    Tensor fp16 = makeGpuTensor(std::vector<float>(100, 1.0f), {100}, stream, DataType::FP16);
    EXPECT_NO_THROW(launch(CubReductionOp::Sum, {16, 0}, 4, fp16, workspace));
    Tensor int32 = in.storage.aliasByteStorageAs(DataType::INT32, {25}, 0);
    EXPECT_THROW(launch(CubReductionOp::Sum, {16, 0}, 4, int32, workspace), std::invalid_argument);
    // 1280: D = 2 is legal (the components kernel), with an output of segments x D; rows of 16 bytes are RK's.
    Tensor wide = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {50, 2}, 0);
    EXPECT_THROW(launch(CubReductionOp::Sum, {16, 0}, 4, wide, workspace), std::invalid_argument);  // output 10, not 20
    Tensor row16 = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {6, 16}, 0);
    Tensor output16(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 16}));
    EXPECT_THROW(launchRaggedKRValue(CubReductionOp::Sum, {16, 0}, 4, row16, in.offsets_tensor, output16, workspace,
                                     stream),
                 std::invalid_argument);
    Tensor output2(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 2}));
    EXPECT_THROW(launchRaggedKRValue(CubReductionOp::Sum, {16, 0}, 4, wide, in.offsets_tensor, output2, workspace,
                                     stream),
                 std::invalid_argument);  // the workspace holds D = 1's records, not D = 2's
    Tensor workspace2(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4, 2)}));
    EXPECT_NO_THROW(launchRaggedKRValue(CubReductionOp::Sum, {16, 0}, 4, wide, in.offsets_tensor, output2,
                                        workspace2, stream));
    // 1284: rows under 16 bytes in every dtype (FP16 / BF16 D <= 7, FP32 D <= 3).
    Tensor workspace8(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4, 8)}));
    Tensor output8(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 8}));
    Tensor output4(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 4}));
    Tensor output7(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 7}));
    Tensor output3(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 3}));
    Tensor workspace7(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4, 7)}));
    Tensor workspace3(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4, 3)}));
    Tensor workspace4(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedValueWorkspaceBytes(4, 4)}));
    const auto launch_into = [&](const Tensor& input, Tensor& out, Tensor& ws) {
        launchRaggedKRValue(CubReductionOp::Sum, {16, 0}, 4, input, in.offsets_tensor, out, ws, stream);
    };
    EXPECT_THROW(launch_into(in.storage.aliasByteStorageAs(DataType::FP16, {6, 8}, 0), output8, workspace8),
                 std::invalid_argument);
    EXPECT_THROW(launch_into(in.storage.aliasByteStorageAs(DataType::BF16, {6, 8}, 0), output8, workspace8),
                 std::invalid_argument);
    EXPECT_THROW(launch_into(in.storage.aliasByteStorageAs(DataType::FP32, {6, 4}, 0), output4, workspace4),
                 std::invalid_argument);
    EXPECT_NO_THROW(launch_into(in.storage.aliasByteStorageAs(DataType::BF16, {7, 7}, 2), output7, workspace7));
    EXPECT_NO_THROW(launch_into(in.storage.aliasByteStorageAs(DataType::FP32, {8, 3}, 4), output3, workspace3));
    // (A tensor view is always aligned to its element: aliasByteStorageAs refuses a misaligned base, so the
    // launcher's own alignment check is defensive.)
    EXPECT_THROW(static_cast<void>(in.storage.aliasByteStorageAs(DataType::FP16, {7, 7}, 1)), std::runtime_error);
    stream.synchronize();
}

// 1280: D = 2..15 (the components kernel and the finisher's CTA per component) against the reference per component:
// every D, every distribution, both offset widths, misaligned bases, grids from 1 CTA to the persistent grid.
TEST(CubRaggedKR, Fp8ComponentsMatchTheExactReferenceForEveryD) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (uint64_t D = 2; D <= RaggedMergePath::RAGGED_KR_MAX_COMPONENTS; ++D) {
        const Format& f = FORMATS[D % 2];
        const A::Distribution d = A::DISTRIBUTIONS[D % 5];
        const uint64_t h = A::mix(0x1280 + D);
        const uint64_t rows = 1 + h % 20000;
        const uint64_t segments = 1 + A::mix(h) % (D % 3 == 0 ? rows : 1500);
        const RaggedInput in = makeInput(f, rows, rows + D, segments, d, D % 2 ? 8 : 4, D % 16, h, stream, {}, D);
        const RaggedKRShape shape{16, 0};
        const uint32_t persistent = raggedKRGrid(shape, f.dtype, 0, static_cast<uint32_t>(D));
        for (uint32_t grid : {1U, 7U, 64U, persistent}) {
            for (CubReductionOp op : OPS) {
                const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                         A::distributionName(d) + " seed " + std::to_string(h) + " grid " +
                                         std::to_string(grid) + " " + opName(op);
                expectExact(run(in, op, shape, grid, stream), reference(in, op), what);
            }
        }
    }
}

// 1280: NaN in some components only, bit-identical repeats, every segment empty and dense ends (each row its own
// segment, at the tile's cap) at D > 1.
TEST(CubRaggedKR, Fp8ComponentsPropagateNaNRepeatAndHandleEmptyAndDenseSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (uint64_t D : {3ULL, 8ULL, 15ULL}) {
        const RaggedInput in =
            makeInput(FORMATS[0], 30000, 30000, 500, A::Distribution::Random, 4, 5, 0xBEEF + D, stream,
                      {5 * D + 1, 777 * D + D - 1, 29999 * D}, D);
        for (CubReductionOp op : OPS) {
            const std::vector<float> first = run(in, op, RaggedKRShape{16, 0}, 23, stream);
            expectExact(first, reference(in, op), "D " + std::to_string(D) + " NaN " + opName(op));
            const std::vector<float> again = run(in, op, RaggedKRShape{16, 0}, 23, stream, DataType::FP32, 99);
            for (size_t o = 0; o < first.size(); ++o) {
                uint32_t a, b;
                std::memcpy(&a, &first[o], 4);
                std::memcpy(&b, &again[o], 4);
                if (!std::isnan(first[o])) ASSERT_EQ(a, b) << "D " << D << " " << opName(op) << " output " << o;
            }
        }
        const RaggedInput empty = makeInput(FORMATS[1], 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, stream, {}, D);
        for (CubReductionOp op : OPS)
            expectExact(run(empty, op, RaggedKRShape{16, 0}, 9, stream), reference(empty, op), opName(op));
        const uint64_t rows = 3000;
        const RaggedInput dense =
            makeInput(FORMATS[1], rows, rows, rows, A::Distribution::Uniform, 4, 1, 0xD5 + D, stream, {}, D);
        for (CubReductionOp op : OPS)
            expectExact(run(dense, op, RaggedKRShape{16, 0}, 13, stream, DataType::BF16), reference(dense, op),
                        "D " + std::to_string(D) + " dense " + opName(op));
    }
}

TEST(CubRaggedKR, ShapesAndTheirGrids) {
    REQUIRE_CUDA_DEVICE();
    EXPECT_TRUE(raggedKRShapeLegal({16, 0}));
    EXPECT_TRUE(raggedKRShapeLegal({16, 4}));
    EXPECT_FALSE(raggedKRShapeLegal({12, 0}));
    EXPECT_FALSE(raggedKRShapeLegal({8, 0}));
    EXPECT_FALSE(raggedKRShapeLegal({32, 0}));  // 1280: 16 rows per lane only
    EXPECT_FALSE(raggedKRShapeLegal({16, 0, true, 256}));
    EXPECT_FALSE(raggedKRShapeLegal({16, 0, true, -2}));
    // 1272: weighted names carry the end's extra weight in force.
    EXPECT_EQ(raggedKRShapeName({16, 2}), "ragged_kr:i16_c2_e" + std::to_string(raggedKREndExtra({16, 2})));
    EXPECT_EQ(raggedKRShapeName({32, 0, true, 6}), "ragged_kr:i32_c0_e6");
    EXPECT_EQ(raggedKRShapeName({32, 0, false}), "ragged_kr:i32_c0_u");
    EXPECT_EQ(raggedKREndExtra({32, 0, false, 6}), 0U);
    // 1268: rows weigh their byte, ends their offset plus output bytes (1272: plus the extra).
    const DataType E4M3 = DataType::FP8_E4M3;
    const auto w = raggedKRWeights({32, 0, true, 0}, E4M3, DataType::UINT64, DataType::FP16);
    EXPECT_EQ(w.row, 1U);
    EXPECT_EQ(w.end, 10U);
    EXPECT_EQ(raggedKRWeights({32, 0, true, 9}, E4M3, DataType::UINT32, DataType::FP32).end, 17U);
    EXPECT_EQ(raggedKRWeights({32, 0}, E4M3, DataType::UINT32, DataType::FP32).end, 8U + raggedKREndExtra({32, 0}));
    EXPECT_EQ(raggedKRWeights({32, 0, false}, E4M3, DataType::UINT32, DataType::FP32).end, 1U);
    // 1280: at D, a row weighs D bytes and an end its offset, D outputs and the extra.
    const auto w7 = raggedKRWeights({16, 0, true, 4}, E4M3, DataType::UINT32, DataType::FP16, 7);
    EXPECT_EQ(w7.row, 7U);
    EXPECT_EQ(w7.end, 4U + 7U * 2U + 4U);
    EXPECT_THROW(static_cast<void>(raggedKRWeights({16, 0}, E4M3, DataType::UINT32, DataType::FP32, 16)),
                 std::invalid_argument);
    // 1284: a row weighs D x the element's bytes; D is bounded by rows under 16 bytes.
    EXPECT_EQ(raggedKRWeights({16, 0}, DataType::BF16, DataType::UINT32, DataType::FP32, 7).row, 14U);
    EXPECT_EQ(raggedKRWeights({16, 0}, DataType::FP32, DataType::UINT32, DataType::FP32, 3).row, 12U);
    EXPECT_EQ(raggedKRWeights({16, 0, true, 4}, DataType::FP32, DataType::UINT64, DataType::FP32, 3).end, 8U + 12U + 4U);
    EXPECT_THROW(static_cast<void>(raggedKRWeights({16, 0}, DataType::FP16, DataType::UINT32, DataType::FP32, 8)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRWeights({16, 0}, DataType::FP32, DataType::UINT32, DataType::FP32, 4)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRWeights({16, 0}, DataType::INT32, DataType::UINT32, DataType::FP32)),
                 std::invalid_argument);
    EXPECT_EQ(raggedKRInputBytes(DataType::FP8_E5M2), 1U);
    EXPECT_EQ(raggedKRInputBytes(DataType::BF16), 2U);
    EXPECT_EQ(raggedKRInputBytes(DataType::FP32), 4U);
    EXPECT_EQ(raggedKRInputBytes(DataType::UINT8), 0U);
    int sms = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
    for (const RaggedKRShape shape : {RaggedKRShape{16, 0}, RaggedKRShape{16, 1}}) {
        for (const auto& [dtype, components] : std::vector<std::pair<DataType, uint32_t>>{
                 {DataType::FP8_E5M2, 1}, {DataType::FP8_E5M2, 2}, {DataType::FP8_E5M2, 15}, {DataType::FP16, 1},
                 {DataType::BF16, 7}, {DataType::FP32, 1}, {DataType::FP32, 3}}) {
            const uint32_t grid = raggedKRGrid(shape, dtype, 0, components);
            EXPECT_GE(grid, static_cast<uint32_t>(sms) * (shape.ctas_per_sm == 1 ? 1U : 2U)) << raggedKRShapeName(shape);
            EXPECT_LE(grid, RAGGED_FINISH_MAX_GRID);
            if (shape.ctas_per_sm == 1) EXPECT_EQ(grid, static_cast<uint32_t>(sms));
        }
    }
    EXPECT_THROW(static_cast<void>(raggedKRGrid({16, 0}, DataType::UINT8, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRGrid({16, 0}, DataType::FP8_E4M3, 0, 16)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRGrid({16, 0}, DataType::FP16, 0, 8)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRGrid({16, 0}, DataType::FP32, 0, 4)), std::invalid_argument);
    EXPECT_EQ(raggedValueWorkspaceBytes(4080), 16ULL * 4080);  // 1273's size at D = 1
}

// 1284: FP16, BF16 and FP32 at D = 1 (8 and 4 rows per lane): every distribution, both splits, grids from 1 CTA to the
// persistent grid, both offset widths.
TEST(CubRaggedKR, WideMatchesTheExactReferenceAcrossDistributionsShapesAndGrids) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t seed = 1284;
    for (const Format& f : WIDE_FORMATS) {
        for (uint64_t trial = 0; trial < 5; ++trial) {
            const A::Distribution d = A::DISTRIBUTIONS[trial];
            const uint64_t h = A::mix(seed + trial + 7 * f.bytes);
            const uint64_t rows = 1 + h % 60000;
            const uint64_t segments = 1 + A::mix(h) % (trial % 2 ? rows : 2000);
            const RaggedInput in = makeInput(f, rows, rows, segments, d, trial % 2 ? 8 : 4, 0, h, stream);
            for (const RaggedKRShape shape : {RaggedKRShape{16, 0}, RaggedKRShape{16, 0, false}}) {
                const uint32_t persistent = raggedKRGrid(shape, f.dtype, 0);
                for (uint32_t grid : {1U, 7U, 64U, persistent}) {
                    for (CubReductionOp op : OPS) {
                        const std::string what = std::string(f.name) + " " + A::distributionName(d) + " seed " +
                                                 std::to_string(h) + " " + raggedKRShapeName(shape) + " grid " +
                                                 std::to_string(grid) + " " + opName(op);
                        expectExact(run(in, op, shape, grid, stream), reference(in, op), what);
                    }
                }
            }
        }
    }
}

// 1284: inactive capacity poisoned with NaN, garbage workspaces, every element-aligned base within 16 bytes, NaN
// propagation and bit-identical repeats, at D = 1 and at every D >= 2 of each width.
TEST(CubRaggedKR, WideIgnoresInactiveCapacityAtAnyAlignmentAndPropagatesNaN) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : WIDE_FORMATS) {
        for (uint64_t D = 1; D <= maxComponents(f); ++D) {
            for (uint64_t misalign = 0; misalign < 16; misalign += f.bytes) {
                const uint64_t h = A::mix(0x1284CAFE + 31 * misalign + D + 100 * f.bytes);
                const uint64_t rows = 3000 + h % 5000;
                const RaggedInput in = makeInput(f, rows, rows + rows / 2, 1 + h % 300, A::Distribution::Pareto, 4,
                                                 misalign, h, stream, {}, D);
                for (CubReductionOp op : OPS) {
                    const std::string what = std::string(f.name) + " D " + std::to_string(D) + " misalign " +
                                             std::to_string(misalign) + " " + opName(op);
                    expectExact(run(in, op, RaggedKRShape{16, 0}, 37, stream, DataType::FP32, h), reference(in, op),
                                what);
                }
            }
            const uint64_t rows = 30000;
            const RaggedInput in = makeInput(f, rows, rows, 600, A::Distribution::Random, 4, 0, 0xBEEF + D, stream,
                                             {5 * D, 777 * D + D - 1, 20000 * D + D / 2, rows * D - 1}, D);
            for (CubReductionOp op : OPS) {
                const std::string what = std::string(f.name) + " D " + std::to_string(D) + " NaN " + opName(op);
                const std::vector<float> first = run(in, op, RaggedKRShape{16, 0}, 23, stream);
                expectExact(first, reference(in, op), what);
                const std::vector<float> again = run(in, op, RaggedKRShape{16, 0}, 23, stream, DataType::FP32, 99);
                for (size_t o = 0; o < first.size(); ++o) {
                    uint32_t a, b;
                    std::memcpy(&a, &first[o], 4);
                    std::memcpy(&b, &again[o], 4);
                    if (!std::isnan(first[o])) ASSERT_EQ(a, b) << what << " output " << o;
                }
            }
        }
    }
}

// 1284: FP16 / BF16 at D = 2..7 and FP32 at D = 2..3 (the components kernel at 8 and 4 rows per worker) against the
// reference per component: every distribution, both offset widths, element-aligned bases, grids from 1 CTA to the
// persistent grid; every segment empty and dense ends (each row its own segment) at each width's largest D.
TEST(CubRaggedKR, WideComponentsMatchTheExactReferenceForEveryD) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : WIDE_FORMATS) {
        for (uint64_t D = 2; D <= maxComponents(f); ++D) {
            const A::Distribution d = A::DISTRIBUTIONS[(D + f.bytes) % 5];
            const uint64_t h = A::mix(0x1284 + D + 16 * f.bytes);
            const uint64_t rows = 1 + h % 20000;
            const uint64_t segments = 1 + A::mix(h) % (D % 3 == 0 ? rows : 1500);
            const uint64_t misalign = (D * f.bytes) % 16;
            const RaggedInput in =
                makeInput(f, rows, rows + D, segments, d, D % 2 ? 8 : 4, misalign, h, stream, {}, D);
            const RaggedKRShape shape{16, 0};
            const uint32_t persistent = raggedKRGrid(shape, f.dtype, 0, static_cast<uint32_t>(D));
            for (uint32_t grid : {1U, 7U, 64U, persistent}) {
                for (CubReductionOp op : OPS) {
                    const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                             A::distributionName(d) + " seed " + std::to_string(h) + " grid " +
                                             std::to_string(grid) + " " + opName(op);
                    expectExact(run(in, op, shape, grid, stream), reference(in, op), what);
                }
            }
        }
        const uint64_t D = maxComponents(f);
        const RaggedInput empty = makeInput(f, 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, stream, {}, D);
        for (CubReductionOp op : OPS)
            expectExact(run(empty, op, RaggedKRShape{16, 0}, 9, stream), reference(empty, op),
                        std::string(f.name) + " empty " + opName(op));
        const uint64_t rows = 3000;
        const RaggedInput dense =
            makeInput(f, rows, rows, rows, A::Distribution::Uniform, 4, f.bytes, 0xD5 + D, stream, {}, D);
        for (CubReductionOp op : OPS)
            expectExact(run(dense, op, RaggedKRShape{16, 0}, 13, stream, DataType::BF16), reference(dense, op),
                        std::string(f.name) + " dense " + opName(op));
    }
}

// 1285: ragged RK against the exact reference per column: D from 16 to 5000 (one K tile at every worker count, and
// K-tiled rows), every K tile cap and the auto cap, CTA budgets from 1 to the GPU's, both formats and offset widths,
// every distribution, misaligned bases.
TEST(CubRaggedKR, RkMatchesTheExactReferenceForEveryDAndTileCap) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t ds[] = {16, 17, 31, 64, 255, 256, 513, 1024, 5000};
    const uint32_t caps[] = {0, 32, 128, 512, 64, 256};
    const uint32_t full = raggedRKCtas(RaggedRKShape{}, DataType::FP8_E4M3, 0);
    for (size_t t = 0; t < sizeof(ds) / sizeof(ds[0]); ++t) {
        const uint64_t D = ds[t];
        const Format& f = FORMATS[t % 2];
        const A::Distribution d = A::DISTRIBUTIONS[t % 5];
        const uint64_t h = A::mix(0x1285 + D);
        const uint64_t rows = 1 + h % std::max<uint64_t>(64, 3000000 / D);
        const uint64_t segments = 1 + A::mix(h) % (t % 3 == 0 ? rows : 700);
        const RaggedInput in = makeInput(f, rows, rows + 3, segments, d, t % 2 ? 8 : 4, D % 16, h, stream, {}, D);
        for (size_t c = 0; c < 3; ++c) {
            RaggedRKShape shape;
            shape.tile_bytes = caps[(t + c) % 6];
            for (uint32_t ctas : {1U, 7U, 300U, full}) {
                for (CubReductionOp op : OPS) {
                    const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                             A::distributionName(d) + " seed " + std::to_string(h) + " " +
                                             raggedRKShapeName(shape) + " ctas " + std::to_string(ctas) + " " + opName(op);
                    expectExact(runRK(in, op, shape, ctas, stream), reference(in, op), what);
                }
            }
        }
    }
}

// 1285: inactive capacity poisoned with NaN and garbage workspaces at every base misalignment (aligned D = 32 takes the
// one-load path at misalignment 0); NaN in some columns and bit-identical repeats; every segment empty; dense ends
// (each row its own segment) into a narrow output; the main kernel and the finisher launched apart.
TEST(CubRaggedKR, RkIgnoresInactiveCapacityPropagatesNaNAndHandlesEmptyAndDenseSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t full = raggedRKCtas(RaggedRKShape{}, DataType::FP8_E5M2, 0);
    for (uint64_t D : {32ULL, 19ULL, 700ULL}) {
        for (uint64_t misalign = 0; misalign < 16; misalign += (D == 32 ? 1 : 5)) {
            const uint64_t h = A::mix(0x1285CAFE + 31 * misalign + D);
            const uint64_t rows = 500 + h % (D > 100 ? 2000 : 20000);
            const RaggedInput in =
                makeInput(FORMATS[1], rows, rows + rows / 2, 1 + h % 300, A::Distribution::Pareto, 4, misalign, h, stream,
                          {}, D);
            for (CubReductionOp op : OPS) {
                const std::string what = "D " + std::to_string(D) + " misalign " + std::to_string(misalign) + " " + opName(op);
                expectExact(runRK(in, op, RaggedRKShape{}, 37, stream, DataType::FP32, h), reference(in, op), what);
            }
        }
        const uint64_t rows = D > 100 ? 3000 : 30000;
        const RaggedInput nan = makeInput(FORMATS[0], rows, rows, 400, A::Distribution::Random, 4, 3, 0xBEEF + D, stream,
                                          {5 * D + 1, 777 * D + D - 1, rows * D - 1}, D);
        for (CubReductionOp op : OPS) {
            const std::string what = "D " + std::to_string(D) + " NaN " + opName(op);
            const std::vector<float> first = runRK(nan, op, RaggedRKShape{}, full, stream);
            expectExact(first, reference(nan, op), what);
            const std::vector<float> again = runRK(nan, op, RaggedRKShape{}, full, stream, DataType::FP32, 99);
            for (size_t o = 0; o < first.size(); ++o) {
                uint32_t a, b;
                std::memcpy(&a, &first[o], 4);
                std::memcpy(&b, &again[o], 4);
                if (!std::isnan(first[o])) ASSERT_EQ(a, b) << what << " output " << o;
            }
            // The main kernel, then the finisher, as two launches on one workspace and one output. 1312: the two
            // launches share the output (the main kernel stores the segments it finishes, the finisher the rest); the
            // output starts poisoned (0x5A5A5A5A, finite) so a segment neither launch stores fails.
            const RaggedRKLaunch launch = raggedRKLaunch(RaggedRKShape{}, full, nan.input.getDataType(), D, rows);
            Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
            const uint64_t segments = nan.offsets.size() - 1;
            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {segments, D}));
            ASSERT_EQ(cudaMemsetAsync(output.getMemPtr<void>(), 0x5A, segments * D * sizeof(float), stream.getStream()),
                      cudaSuccess);
            launchRaggedRKValue(op, RaggedRKShape{}, full, nan.input, nan.offsets_tensor, output, workspace, stream,
                                RaggedLaunchParts::MainOnly);
            launchRaggedRKValue(op, RaggedRKShape{}, full, nan.input, nan.offsets_tensor, output, workspace, stream,
                                RaggedLaunchParts::FinishOnly);
            stream.synchronize();
            expectExact(copyGpuTensorAsFloat(output, stream), first, what + " apart");
        }
        const RaggedInput empty = makeInput(FORMATS[1], 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, stream, {}, D);
        for (CubReductionOp op : OPS)
            expectExact(runRK(empty, op, RaggedRKShape{}, 9, stream), reference(empty, op),
                        "D " + std::to_string(D) + " empty " + opName(op));
        const uint64_t dense_rows = D > 100 ? 600 : 3000;
        const RaggedInput dense = makeInput(FORMATS[0], dense_rows, dense_rows, dense_rows, A::Distribution::Uniform, 4,
                                            1, 0xD5 + D, stream, {}, D);
        for (CubReductionOp op : {CubReductionOp::Min, CubReductionOp::Max, CubReductionOp::Sum}) {
            for (DataType out : {DataType::BF16, DataType::FP16}) {
                for (uint32_t cap : {0U, 32U}) {
                    RaggedRKShape shape;
                    shape.tile_bytes = cap;
                    expectExact(runRK(dense, op, shape, 13, stream, out), reference(dense, op),
                                "D " + std::to_string(D) + " dense " + raggedRKShapeName(shape) + " " + opName(op));
                }
            }
        }
    }
}

// 1286: the vector end stores against the scalar conversions (storeFp32AsRuntimeDType's): the FP32 run, converted on
// the host to each output dtype, bit for bit (NaN as NaN).
TEST(CubRaggedKR, RkVectorEndStoresMatchTheScalarConversionsForEveryOutputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const auto narrow = [](DataType dtype, float v) -> float {
        switch (dtype) {
            case DataType::FP16: return __half2float(__float2half_rn(v));
            case DataType::BF16: return __bfloat162float(__float2bfloat16_rn(v));
            case DataType::FP8_E4M3: return static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(v));
            case DataType::FP8_E5M2: return static_cast<float>(ThorLowPrecision::toFp8E5M2Satfinite(v));
            default: return v;
        }
    };
    const uint32_t full = raggedRKCtas(RaggedRKShape{}, DataType::FP8_E4M3, 0);
    for (uint64_t D : {16ULL, 17ULL, 64ULL, 255ULL, 256ULL}) {
        const uint64_t h = A::mix(0x1286 + D);
        const uint64_t rows = 2000 + h % 3000;
        const Format& f = FORMATS[D % 2];
        const RaggedInput in = makeInput(f, rows, rows + 5, rows / (1 + D % 3), A::Distribution::Random, D % 2 ? 8 : 4,
                                         D % 16, h, stream, {3 * D + 2, (rows / 2) * D + D - 1}, D);
        for (CubReductionOp op : OPS) {
            for (uint32_t ctas : {7U, full}) {
                const std::vector<float> wide = runRK(in, op, RaggedRKShape{}, ctas, stream);
                expectExact(wide, reference(in, op), "D " + std::to_string(D) + " fp32 " + opName(op));
                for (DataType out : {DataType::FP16, DataType::BF16, DataType::FP8_E4M3, DataType::FP8_E5M2}) {
                    std::vector<float> expected(wide.size());
                    for (size_t o = 0; o < wide.size(); ++o) expected[o] = narrow(out, wide[o]);
                    expectExact(runRK(in, op, RaggedRKShape{}, ctas, stream, out), expected,
                                "D " + std::to_string(D) + " output " + std::to_string(static_cast<int>(out)) + " ctas " +
                                    std::to_string(ctas) + " " + opName(op));
                }
            }
        }
    }
}

// 1295: ragged RK for the 16- and 32-bit inputs. Per format: narrow rows (16 to 32 bytes, 4 rows per worker), two-row
// contiguous rows and K-tiled rows, against the exact reference per column over caps (and the auto cap) and CTA
// budgets from 1 to the GPU's, both offset widths, every distribution, bases misaligned by whole elements; NaN in
// some columns, every segment empty, dense ends; and every narrower output dtype against the FP32 output converted on
// the host (the vector stores at 8 and 4 values per lane).
TEST(CubRaggedKR, RkWideInputsMatchTheExactReferenceForEveryDTileCapAndOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const auto narrow = [](DataType dtype, float v) -> float {
        switch (dtype) {
            case DataType::FP16: return __half2float(__float2half_rn(v));
            case DataType::BF16: return __bfloat162float(__float2bfloat16_rn(v));
            case DataType::FP8_E4M3: return static_cast<float>(ThorLowPrecision::toFp8E4M3Satfinite(v));
            case DataType::FP8_E5M2: return static_cast<float>(ThorLowPrecision::toFp8E5M2Satfinite(v));
            default: return v;
        }
    };
    const uint32_t caps[] = {0, 32, 128, 512, 64, 256};
    for (size_t fi = 0; fi < sizeof(WIDE_FORMATS) / sizeof(WIDE_FORMATS[0]); ++fi) {
        const Format& f = WIDE_FORMATS[fi];
        const uint64_t e = f.bytes;  // D in elements for 16, 17-18, 32, 34, 256, 510, 1026 and 5000-byte rows
        const uint64_t ds[] = {16 / e, 16 / e + 1, 32 / e, 32 / e + 1, 256 / e, 512 / e - 1, 1024 / e + 1, 5000 / e};
        const uint32_t full = raggedRKCtas(RaggedRKShape{}, f.dtype, 0);
        for (size_t t = 0; t < sizeof(ds) / sizeof(ds[0]); ++t) {
            const uint64_t D = ds[t];
            const A::Distribution d = A::DISTRIBUTIONS[(t + fi) % 5];
            const uint64_t h = A::mix(0x1295 + D + 977 * fi);
            const uint64_t rows = 1 + h % std::max<uint64_t>(64, 3000000 / (D * e));
            const uint64_t segments = 1 + A::mix(h) % ((t + fi) % 3 == 0 ? rows : 700);
            const uint64_t misalign = (e * (D + t)) % 16 / e * e;
            const RaggedInput in = makeInput(f, rows, rows + 3, segments, d, (t + fi) % 2 ? 8 : 4, misalign, h, stream,
                                             {}, D);
            const std::vector<float> expected = reference(in, CubReductionOp::Sum);
            for (size_t c = 0; c < 2; ++c) {
                RaggedRKShape shape;
                shape.tile_bytes = caps[(t + c + fi) % 6];
                for (uint32_t ctas : {1U, 7U, 300U, full}) {
                    for (CubReductionOp op : OPS) {
                        const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                                 A::distributionName(d) + " seed " + std::to_string(h) + " " +
                                                 raggedRKShapeName(shape) + " ctas " + std::to_string(ctas) + " " +
                                                 opName(op);
                        expectExact(runRK(in, op, shape, ctas, stream),
                                    op == CubReductionOp::Sum ? expected : reference(in, op), what);
                    }
                }
            }
        }
        // NaN, repeats, every narrower output, empty and dense segments at a narrow, a two-row and a K-tiled D.
        for (uint64_t D : {16 / e + 1, 256 / e + 1, 2048 / e + 3}) {
            const uint64_t rows = D * e > 1000 ? 2000 : 20000;
            const RaggedInput nan = makeInput(f, rows, rows + 7, rows / 3, A::Distribution::Random, 4, 2 * e,
                                              0xF00D + D + fi, stream, {3 * D + 1, 777 * D + D - 1, rows * D - 1}, D);
            for (CubReductionOp op : OPS) {
                const std::string what = std::string(f.name) + " D " + std::to_string(D) + " NaN " + opName(op);
                for (uint32_t ctas : {7U, full}) {
                    const std::vector<float> wide = runRK(nan, op, RaggedRKShape{}, ctas, stream);
                    expectExact(wide, reference(nan, op), what + " ctas " + std::to_string(ctas));
                    const std::vector<float> again = runRK(nan, op, RaggedRKShape{}, ctas, stream, DataType::FP32, 99);
                    for (size_t o = 0; o < wide.size(); ++o) {
                        uint32_t a, b;
                        std::memcpy(&a, &wide[o], 4);
                        std::memcpy(&b, &again[o], 4);
                        if (!std::isnan(wide[o])) ASSERT_EQ(a, b) << what << " repeat output " << o;
                    }
                    for (DataType out : {DataType::FP16, DataType::BF16, DataType::FP8_E4M3, DataType::FP8_E5M2}) {
                        std::vector<float> converted(wide.size());
                        for (size_t o = 0; o < wide.size(); ++o) converted[o] = narrow(out, wide[o]);
                        expectExact(runRK(nan, op, RaggedRKShape{}, ctas, stream, out), converted,
                                    what + " output " + std::to_string(static_cast<int>(out)) + " ctas " +
                                        std::to_string(ctas));
                    }
                }
            }
            const RaggedInput empty = makeInput(f, 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, stream, {}, D);
            for (CubReductionOp op : OPS)
                expectExact(runRK(empty, op, RaggedRKShape{}, 9, stream), reference(empty, op),
                            std::string(f.name) + " D " + std::to_string(D) + " empty " + opName(op));
            const uint64_t dense_rows = D * e > 1000 ? 600 : 3000;
            const RaggedInput dense = makeInput(f, dense_rows, dense_rows, dense_rows, A::Distribution::Uniform, 8, e,
                                                0xD5 + D, stream, {}, D);
            for (CubReductionOp op : OPS) {
                for (uint32_t cap : {0U, 32U}) {
                    RaggedRKShape shape;
                    shape.tile_bytes = cap;
                    expectExact(runRK(dense, op, shape, 13, stream), reference(dense, op),
                                std::string(f.name) + " D " + std::to_string(D) + " dense " + raggedRKShapeName(shape) +
                                    " " + opName(op));
                }
            }
        }
    }
}

#if THOR_RAGGED_RK_PROBE
// 1290 (measurement only, run with --gtest_also_run_disabled_tests): where the RK main kernel's fixed time goes. Per
// cell: the main kernel's event time (median of 15) against the moved bytes at copy rate, then per CTA (the probe,
// one more launch) the entry time from the first CTA's, the search, the first tile's wait, the tiles, the exit and
// the exit time from the first entry; and the event time at 8 CTAs per SM.
TEST(CubRaggedKR, DISABLED_RkStartupTailProbe) {
    REQUIRE_CUDA_DEVICE();
    using RaggedMergePath::RAGGED_RK_PROBE_WORDS;
    Stream stream(0);
    struct Case {
        const char* name;
        uint64_t D, rows, segments;
        A::Distribution d;
    };
    const Case cases[] = {
        {"D256 mean1000 44MiB", 256, 172032, 172, A::Distribution::Uniform},
        {"D256 mean1000 11MiB", 256, 43008, 43, A::Distribution::Uniform},
        {"D256 mean1000 pareto", 256, 172032, 172, A::Distribution::Pareto},
        {"D64 mean100 44MiB", 64, 688128, 6881, A::Distribution::Uniform},
        {"D64 mean100 11MiB", 64, 172032, 1720, A::Distribution::Uniform},
        {"D4096 mean100 44MiB", 4096, 10752, 107, A::Distribution::Uniform},
        {"D4096 mean100 11MiB", 4096, 2688, 27, A::Distribution::Uniform},
        {"D17 mean100 44MiB", 17, 2650000, 26500, A::Distribution::Uniform},
        {"D256 mean3.4 44MiB", 256, 172032, 50000, A::Distribution::Uniform},
    };
    const auto pct = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[std::min<size_t>(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
    };
    cudaEvent_t e0, e1;
    ASSERT_EQ(cudaEventCreate(&e0), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&e1), cudaSuccess);
    std::printf("cell | event us, at copy us, overhead us | CTAs, tiles p50/max | entry p50/max | search p50/max | "
                "first tile p50/max | loop p50/max | exit p50/p90/max (us from the first entry) | 8 per SM us\n");
    for (const Case& c : cases) {
        Tensor input(gpuPlacement, TensorDescriptor(DataType::FP8_E4M3, {c.rows, c.D}));
        ASSERT_EQ(cudaMemsetAsync(input.getMemPtr<void>(), 0x38, c.rows * c.D, stream.getStream()), cudaSuccess);
        const std::vector<uint64_t> offsets = A::makeOffsets(c.rows, c.segments, c.d, 1290);
        Tensor offsets_tensor = makeGpuUnsignedTensor(offsets, {c.segments + 1}, stream, DataType::UINT32);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {c.segments, c.D}));
        double ms_at[2] = {0, 0};
        uint32_t grid = 0;
        std::vector<unsigned long long> probe;
        for (int variant = 0; variant < 2; ++variant) {
            RaggedRKShape shape;
            shape.ctas_per_sm = variant == 0 ? 0 : 8;
            const uint32_t ctas = raggedRKCtas(shape, DataType::FP8_E4M3, 0);
            const RaggedRKLaunch launch = raggedRKLaunch(shape, ctas, DataType::FP8_E4M3, c.D, c.rows);
            Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
            const auto run = [&] {
                launchRaggedRKValue(CubReductionOp::Sum, shape, ctas, input, offsets_tensor, output, workspace, stream,
                                    RaggedLaunchParts::MainOnly);
            };
            run();
            std::vector<double> times;
            for (int r = 0; r < 15; ++r) {
                cudaEventRecord(e0, stream.getStream());
                run();
                cudaEventRecord(e1, stream.getStream());
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                times.push_back(ms);
            }
            ms_at[variant] = pct(times, 0.5);
            if (variant == 0) {
                grid = launch.grid;
                unsigned long long* device_probe = nullptr;
                ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_probe), uint64_t{grid} * RAGGED_RK_PROBE_WORDS * 8), cudaSuccess);
                raggedRKSetProbe(device_probe);
                run();
                raggedRKSetProbe(nullptr);
                stream.synchronize();
                probe.resize(uint64_t{grid} * RAGGED_RK_PROBE_WORDS);
                ASSERT_EQ(cudaMemcpyAsync(probe.data(), device_probe, probe.size() * 8, cudaMemcpyDeviceToHost, stream.getStream()),
                          cudaSuccess);
                stream.synchronize();
                cudaFree(device_probe);
            }
        }
        const double moved = static_cast<double>(c.rows * c.D + (c.segments + 1) * 4 + c.segments * c.D * 4);
        const double event_us = ms_at[0] * 1000.0, copy_us = moved / 1.54e6;
        unsigned long long first_entry = ~0ULL;
        for (uint32_t b = 0; b < grid; ++b) first_entry = std::min(first_entry, probe[b * RAGGED_RK_PROBE_WORDS]);
        std::vector<double> entry, search, first, loop, exit_at, tiles;
        for (uint32_t b = 0; b < grid; ++b) {
            const unsigned long long* r = &probe[b * RAGGED_RK_PROBE_WORDS];
            const double span_ns = static_cast<double>(r[6] - r[0]);
            const double cycles = static_cast<double>(r[5] - r[1]);
            const double ns_per_cycle = cycles > 0 && span_ns > 0 ? span_ns / cycles : 1.0 / 2.8;
            entry.push_back((r[0] - first_entry) / 1000.0);
            search.push_back((r[2] - r[1]) * ns_per_cycle / 1000.0);
            first.push_back((r[3] - r[2]) * ns_per_cycle / 1000.0);
            loop.push_back((r[4] - r[3]) * ns_per_cycle / 1000.0);
            exit_at.push_back((r[6] - first_entry) / 1000.0);
            tiles.push_back(static_cast<double>(r[7] & 0xffffffffULL));
        }
        std::printf("%s | %.2f, %.2f, %.2f | %u, %.0f/%.0f | %.2f/%.2f | %.2f/%.2f | %.2f/%.2f | %.2f/%.2f | "
                    "%.2f/%.2f/%.2f | %.2f\n",
                    c.name, event_us, copy_us, event_us - copy_us, grid, pct(tiles, 0.5), pct(tiles, 1.0),
                    pct(entry, 0.5), pct(entry, 1.0), pct(search, 0.5), pct(search, 1.0), pct(first, 0.5),
                    pct(first, 1.0), pct(loop, 0.5), pct(loop, 1.0), pct(exit_at, 0.5), pct(exit_at, 0.9),
                    pct(exit_at, 1.0), ms_at[1] * 1000.0);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
#endif

TEST(CubRaggedKR, RkShapesLaunchesAndTheContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    EXPECT_TRUE(raggedRKShapeLegal(RaggedRKShape{}));
    EXPECT_TRUE(raggedRKShapeLegal(RaggedRKShape{128, 8}));
    EXPECT_FALSE(raggedRKShapeLegal(RaggedRKShape{16}));
    EXPECT_FALSE(raggedRKShapeLegal(RaggedRKShape{96}));
    EXPECT_FALSE(raggedRKShapeLegal(RaggedRKShape{1024}));
    EXPECT_FALSE(raggedRKShapeLegal(RaggedRKShape{0, 33}));
    EXPECT_FALSE(raggedRKShapeLegal(RaggedRKShape{0, 0, true, 256}));
    EXPECT_EQ(raggedRKShapeName(RaggedRKShape{}), "ragged_rk:ta_c0_e4");
    EXPECT_EQ(raggedRKShapeName(RaggedRKShape{128, 8, true, 9}), "ragged_rk:t128_c8_e9");
    EXPECT_EQ(raggedRKShapeName(RaggedRKShape{512, 0, false}), "ragged_rk:t512_c0_u");
    int sms = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
    const uint32_t full = raggedRKCtas(RaggedRKShape{}, DataType::FP8_E4M3, 0);
    EXPECT_GE(full, static_cast<uint32_t>(sms) * 8U);
    EXPECT_EQ(raggedRKCtas(RaggedRKShape{0, 1}, DataType::FP8_E4M3, 0), static_cast<uint32_t>(sms));
    // 1295: every input KR takes (1285-1294: FP8 only).
    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32})
        EXPECT_GE(raggedRKCtas(RaggedRKShape{}, dtype, 0), static_cast<uint32_t>(sms) * 8U);
    EXPECT_THROW(static_cast<void>(raggedRKCtas(RaggedRKShape{}, DataType::UINT8, 0)), std::invalid_argument);
    // Launches: K tiles, ranges and the workspace.
    const RaggedRKLaunch l16 = raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP8_E4M3, 16, 1 << 20);
    EXPECT_EQ(l16.geometry.ktiles, 1U);
    EXPECT_EQ(l16.geometry.workers, 32U);
    EXPECT_EQ(l16.ranges, 2720U);
    EXPECT_EQ(l16.workspace_bytes, RaggedMergePath::raggedRecordLayout(2720, 16).bytes);
    const RaggedRKLaunch l5000 = raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP8_E4M3, 5000, 3355);  // 16 MiB
    EXPECT_EQ(l5000.tile_cap, 32U);
    EXPECT_EQ(l5000.geometry.ktiles, 157U);
    EXPECT_EQ(l5000.ranges, 17U);
    EXPECT_EQ(l5000.grid, 17U * 157U);
    EXPECT_TRUE(raggedFinishUsesWide(5000));
    EXPECT_TRUE(raggedFinishUsesWide(16));
    EXPECT_FALSE(raggedFinishUsesWide(15));
    // 1291: the wide finisher's launch: 256 records per warp, up to 8 warps' components per CTA.
    for (uint32_t grid : {1U, 17U, 256U, 257U, 1024U, 1360U, 2720U, 4096U}) {
        const RaggedFinishWideLaunch w = raggedFinishWideLaunch(grid);
        EXPECT_GE(w.warps_per_component * RAGGED_FINISH_WIDE_WARP_RECORDS, grid) << grid;
        EXPECT_LT((w.warps_per_component - 1) * RAGGED_FINISH_WIDE_WARP_RECORDS, grid) << grid;
        EXPECT_EQ(w.threads, 32 * w.warps_per_component * w.components_per_cta) << grid;
        EXPECT_LE(w.threads, RAGGED_FINISH_WIDE_MAX_THREADS) << grid;
    }
    EXPECT_EQ(raggedFinishWideLaunch(17).components_per_cta, 8U);
    EXPECT_EQ(raggedFinishWideLaunch(2720).warps_per_component, 11U);
    EXPECT_EQ(raggedFinishWideLaunch(4096).threads, 512U);
    // 1294: the direct form up to 128 records (4 per lane), transposed above.
    EXPECT_FALSE(raggedFinishWideTransposes(1));
    EXPECT_FALSE(raggedFinishWideTransposes(RAGGED_FINISH_WIDE_DIRECT_MAX_GRID));
    EXPECT_TRUE(raggedFinishWideTransposes(RAGGED_FINISH_WIDE_DIRECT_MAX_GRID + 1));
    EXPECT_TRUE(raggedFinishWideTransposes(2720));
    EXPECT_EQ(RAGGED_FINISH_WIDE_DIRECT_MAX_GRID % 32, 0U);
    EXPECT_LE(RAGGED_FINISH_WIDE_DIRECT_MAX_GRID / 32, RAGGED_FINISH_WIDE_ROUNDS);
    const RaggedRKLaunch l512 = raggedRKLaunch(RaggedRKShape{512}, 2720, DataType::FP8_E4M3, 1024, 100);
    EXPECT_EQ(l512.geometry.ktiles, 2U);
    EXPECT_EQ(l512.ranges, 1360U);
    EXPECT_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP8_E4M3, 15, 100)),
                 std::invalid_argument);  // KR's
    EXPECT_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 0, DataType::FP8_E4M3, 16, 100)), std::invalid_argument);
    // 1295: rows of 16 bytes and more for every input (FP32 D 4, FP16 D 8; FP32 D 3 and BF16 D 7 are KR's).
    EXPECT_NO_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 10, DataType::FP32, 4, 100)));
    EXPECT_NO_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 10, DataType::FP16, 8, 100)));
    EXPECT_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 10, DataType::FP32, 3, 100)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 10, DataType::BF16, 7, 100)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 10, DataType::INT32, 16, 100)), std::invalid_argument);
    const RaggedRKLaunch l32 = raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP32, 4, 1 << 20);
    EXPECT_EQ(l32.geometry.workers, 32U);  // a 16-byte row: one chunk, 32 workers (narrow rows)
    EXPECT_EQ(RaggedMergePath::raggedRKRowsPerWorker(l32.geometry), RaggedMergePath::RAGGED_RK_ROWS_NARROW);
    // The launcher's contract.
    const RaggedInput in = makeInput(FORMATS[0], 100, 100, 10, A::Distribution::Uniform, 4, 0, 5, stream, {}, 16);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 16}));
    const RaggedRKLaunch launch = raggedRKLaunch(RaggedRKShape{}, 4, DataType::FP8_E4M3, 16, 100);
    Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
    Tensor small(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes - 1}));
    const auto go = [&](CubReductionOp op, const Tensor& input, Tensor& out, Tensor& ws) {
        launchRaggedRKValue(op, RaggedRKShape{}, 4, input, in.offsets_tensor, out, ws, stream);
    };
    EXPECT_NO_THROW(go(CubReductionOp::Sum, in.input, output, workspace));
    EXPECT_THROW(go(CubReductionOp::Product, in.input, output, workspace), std::invalid_argument);
    EXPECT_THROW(go(CubReductionOp::Sum, in.input, output, small), std::invalid_argument);
    Tensor output_short(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 15}));
    EXPECT_THROW(go(CubReductionOp::Sum, in.input, output_short, workspace), std::invalid_argument);
    Tensor narrow = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {100, 15}, 0);
    EXPECT_THROW(go(CubReductionOp::Sum, narrow, output_short, workspace), std::invalid_argument);  // KR's rows
    Tensor flat = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {1600}, 0);
    EXPECT_THROW(go(CubReductionOp::Sum, flat, output, workspace), std::invalid_argument);
    // 1295: 16-bit rows under 16 bytes are KR's (16-bit rows of 16 bytes run: D 8); a non-float input is refused.
    Tensor fp16 = in.storage.aliasByteStorageAs(DataType::FP16, {100, 7}, 0);
    Tensor output7(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 7}));
    EXPECT_THROW(go(CubReductionOp::Sum, fp16, output7, workspace), std::invalid_argument);
    Tensor fp16_rows = in.storage.aliasByteStorageAs(DataType::FP16, {100, 8}, 0);
    Tensor output8(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 8}));
    EXPECT_NO_THROW(go(CubReductionOp::Sum, fp16_rows, output8, workspace));
    Tensor bytes = in.storage.aliasByteStorageAs(DataType::UINT8, {100, 16}, 0);
    EXPECT_THROW(go(CubReductionOp::Sum, bytes, output, workspace), std::invalid_argument);
    stream.synchronize();
}

// 1298 (phase 5): ragged KR ARG, FP8 at D = 1 (launchRaggedKRArg, CubRaggedKRArg.cu; the kernel RaggedKRArgKernels.cuh,
// the finisher CubRaggedFinishArg.cu; 1311 FP16 / BF16; 1313 FP32 with its pair finisher CubRaggedFinishArgFp32.cu),
// against a host reference with the legacy semantics (values decoded to FP32,
// NaNs propagate, the lowest index wins ties, -0 ties +0, the sentinel for a segment without rows).
namespace {

// 1311: the ARG formats (FP8 both, FP16, BF16; 1313: FP32) and their special encodings.
const Format ARG_FORMATS[] = {FORMATS[0], FORMATS[1], WIDE_FORMATS[0], WIDE_FORMATS[1], WIDE_FORMATS[2]};
struct ArgSpecials {
    uint32_t one, nan_a, nan_b, nan_c, negative_zero, largest, lowest;  // largest: the top non-NaN (inf where it exists)
};
ArgSpecials argSpecials(const Format& f) {
    switch (f.dtype) {
        case DataType::FP8_E4M3: return {0x38, 0x7F, 0xFF, 0x7F, 0x80, 0x7E, 0xFE};
        case DataType::FP8_E5M2: return {0x3C, 0x7F, 0xFF, 0x7D, 0x80, 0x7C, 0xFC};
        case DataType::FP16: return {0x3C00, 0x7E00, 0xFE00, 0x7C01, 0x8000, 0x7C00, 0xFC00};
        case DataType::FP32:  // 1313
            return {0x3F800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x80000000, 0x7F800000, 0xFF800000};
        default: return {0x3F80, 0x7FC0, 0xFFC0, 0x7F81, 0x8000, 0x7F80, 0xFF80};  // BF16
    }
}
bool argIsNaN(const Format& f, uint32_t x) {
    switch (f.dtype) {
        case DataType::FP8_E4M3: return (x & 0x7FU) == 0x7FU;
        case DataType::FP8_E5M2: return (x & 0x7FU) > 0x7CU;
        case DataType::FP16: return (x & 0x7FFFU) > 0x7C00U;
        case DataType::FP32: return (x & 0x7FFFFFFFU) > 0x7F800000U;  // 1313
        default: return (x & 0x7FFFU) > 0x7F80U;
    }
}
uint32_t argMaxComponents(const Format& f) { return 15U / f.bytes; }  // rows under 16 bytes (1313: FP32 D 3)

// The element patterns a pool draws from: 0 any pattern (NaN, +-0, infinities included), 1 a few values (ties
// everywhere, specials included), 2 any non-NaN pattern. 1311: any ARG format (1298-1310: argByte, FP8).
uint32_t argElement(const Format& f, uint32_t pool, uint64_t h) {
    const uint32_t mask = f.bytes == 1 ? 0xFFU : (f.bytes == 2 ? 0xFFFFU : 0xFFFFFFFFU);
    if (pool == 0) return static_cast<uint32_t>(h) & mask;
    if (pool == 1) {
        const ArgSpecials x = argSpecials(f);
        const uint32_t few[] = {0, x.negative_zero, x.one, x.one | x.negative_zero, x.largest, x.lowest,
                                x.largest - 1, x.lowest - 1, x.one, x.one, x.nan_c};
        return few[h % (sizeof(few) / sizeof(few[0]))];
    }
    uint32_t b = static_cast<uint32_t>(h) & mask;
    while (argIsNaN(f, b)) b = (b * 37U + 11U) & mask;
    return b;
}

struct ArgInput {
    std::vector<float> values;  // active rows x D, decoded
    std::vector<uint64_t> offsets;
    uint64_t d = 1;  // 1300
    Tensor storage, input, offsets_tensor;
};

// [capacity] (1300: [capacity, D] when D > 1) of format f with `rows` active rows from `pool`, the inactive capacity
// all-ones bytes (a NaN in every ARG format: it would win if read), `misalign` elements before row 0. `set` overrides
// elements (row x D + component) with patterns. 1311: any ARG format.
ArgInput makeArgInput(const Format& f, uint64_t rows, uint64_t capacity, uint64_t segments, A::Distribution d,
                      uint32_t offset_width, uint64_t misalign, uint64_t seed, uint32_t pool, Stream& stream,
                      const std::vector<std::pair<uint64_t, uint32_t>>& set = {}, uint64_t components = 1) {
    ArgInput in;
    in.d = components;
    const uint64_t elements = rows * components, width = f.bytes, base = misalign * width;
    std::vector<uint64_t> bytes(base + capacity * components * width + 16, 0xFF);
    std::vector<uint32_t> patterns(elements);
    for (uint64_t e = 0; e < elements; ++e) patterns[e] = argElement(f, pool, A::mix(seed ^ (e * 0x9E3779B97F4A7C15ULL)));
    for (const auto& [e, x] : set) patterns[e] = x;
    in.values.resize(elements);
    for (uint64_t e = 0; e < elements; ++e) {
        uint8_t b[4];
        for (uint64_t i = 0; i < width; ++i) {
            b[i] = static_cast<uint8_t>(patterns[e] >> (8 * i));
            bytes[base + e * width + i] = b[i];
        }
        in.values[e] = A::decodeElement(f.format, b);
    }
    in.storage = makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8);
    in.input = components == 1 ? in.storage.aliasByteStorageAs(f.dtype, {capacity}, base)
                               : in.storage.aliasByteStorageAs(f.dtype, {capacity, components}, base);
    in.offsets = A::makeOffsets(rows, segments, d, seed);
    in.offsets_tensor =
        makeGpuUnsignedTensor(in.offsets, {segments + 1}, stream, offset_width == 8 ? DataType::UINT64 : DataType::UINT32);
    return in;
}

std::vector<uint64_t> argReference(const ArgInput& in, CubArgReductionOp op, DataType index_dtype) {
    const uint64_t segments = in.offsets.size() - 1, D = in.d;
    const uint64_t sentinel = index_dtype == DataType::UINT64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    std::vector<uint64_t> out(segments * D, sentinel);
    for (uint64_t s = 0; s < segments; ++s) {
        const uint64_t begin = in.offsets[s], end = in.offsets[s + 1];
        if (begin == end) continue;
        for (uint64_t c = 0; c < D; ++c) {
            uint64_t best = begin;
            for (uint64_t r = begin + 1; r < end; ++r) {
                const float v = in.values[r * D + c], b = in.values[best * D + c];
                if (std::isnan(b)) break;  // the first NaN wins
                if (std::isnan(v) || (op == CubArgReductionOp::ArgMin ? v < b : v > b)) best = r;
            }
            out[s * D + c] = best * D + c;  // the packed index
        }
    }
    return out;
}

std::vector<uint64_t> runArg(const ArgInput& in, CubArgReductionOp op, const RaggedKRShape& shape, uint32_t grid,
                             Stream& stream, DataType index_dtype = DataType::UINT32, uint64_t garbage_seed = 7) {
    const uint64_t segments = in.offsets.size() - 1;
    Tensor output = makeGpuUnsignedTensor(std::vector<uint64_t>(segments * in.d, 0x5A5A5A5AU),
                                          in.d == 1 ? std::vector<uint64_t>{segments}
                                                    : std::vector<uint64_t>{segments, in.d},
                                          stream, index_dtype);
    std::vector<uint64_t> garbage(raggedArgWorkspaceBytes(in.input.getDataType(), grid, static_cast<uint32_t>(in.d)));
    for (size_t i = 0; i < garbage.size(); ++i) garbage[i] = A::mix(garbage_seed + i) & 0xFF;
    Tensor workspace = makeGpuUnsignedTensor(garbage, {garbage.size()}, stream, DataType::UINT8);
    launchRaggedKRArg(op, shape, grid, in.input, in.offsets_tensor, output, workspace, stream);
    stream.synchronize();
    return copyGpuTensorAsUnsigned(output, stream);
}

void expectIndices(const std::vector<uint64_t>& got, const std::vector<uint64_t>& expected, const std::string& what) {
    ASSERT_EQ(got.size(), expected.size()) << what;
    for (size_t s = 0; s < got.size(); ++s) ASSERT_EQ(got[s], expected[s]) << what << " segment " << s;
}

const CubArgReductionOp ARG_OPS[] = {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax};
const char* argOpName(CubArgReductionOp op) { return op == CubArgReductionOp::ArgMin ? "argmin" : "argmax"; }

}  // namespace

TEST(CubRaggedKR, ArgMatchesTheReferenceForRandomBitsSpecialsAndTies) {  // 1311: every ARG format (1298: FP8)
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t seed = 1298;
    for (const Format& f : ARG_FORMATS) {
        for (uint64_t trial = 0; trial < 10; ++trial) {
            const A::Distribution d = A::DISTRIBUTIONS[trial % 5];
            const uint32_t pool = static_cast<uint32_t>(trial % 3);
            const uint64_t h = A::mix(seed + trial + 100 * static_cast<uint64_t>(f.dtype));
            const uint64_t rows = 1 + h % 70000;
            const uint64_t segments = 1 + A::mix(h) % (trial % 2 ? rows : 3000);
            const ArgInput in = makeArgInput(f, rows, rows, segments, d, trial % 2 ? 8 : 4, 0, h, pool, stream);
            for (const RaggedKRShape shape : {RaggedKRShape{16, 0}, RaggedKRShape{16, 0, false}}) {
                const uint32_t persistent = raggedKRArgGrid(shape, f.dtype, 0);
                for (uint32_t grid : {1U, 2U, 7U, 64U, persistent}) {
                    for (CubArgReductionOp op : ARG_OPS) {
                        for (DataType index : {DataType::UINT32, DataType::UINT64}) {
                            const std::string what = std::string(f.name) + " " + A::distributionName(d) + " pool " +
                                                     std::to_string(pool) + " seed " + std::to_string(h) + " " +
                                                     raggedKRShapeName(shape) + " grid " + std::to_string(grid) + " " +
                                                     argOpName(op) + (index == DataType::UINT64 ? " u64" : " u32");
                            expectIndices(runArg(in, op, shape, grid, stream, index), argReference(in, op, index), what);
                        }
                    }
                }
            }
        }
    }
}

// Ties split across CTAs: long segments of one value (every CTA holds the same key), of NaNs (several encodings), and
// of zeros with -0 and +0 mixed; the extremum planted twice far apart. The lowest index must win in every grid.
TEST(CubRaggedKR, ArgTiesSplitAcrossCtasGoToTheLowestIndex) {  // 1311: every ARG format (1298: FP8)
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : ARG_FORMATS) {
        const ArgSpecials x = argSpecials(f);
        const uint64_t rows = 200000;
        const uint32_t one = x.one;
        struct Case {
            const char* name;
            std::vector<uint32_t> bytes;  // patterns cycled over the rows
        };
        const Case cases[] = {{"equal", {one}},
                              {"nan", {x.nan_a, x.nan_b, x.nan_c}},
                              {"zeros", {x.negative_zero, 0, 0, x.negative_zero, x.negative_zero}},
                              {"plus_zeros", {0}}};
        for (const Case& c : cases) {
            std::vector<std::pair<uint64_t, uint32_t>> set;
            for (uint64_t r = 0; r < rows; ++r) set.emplace_back(r, c.bytes[r % c.bytes.size()]);
            for (uint64_t segments : {1ULL, 3ULL, 40ULL}) {
                const ArgInput in = makeArgInput(f, rows, rows, segments, A::Distribution::Uniform, 4, 5, 9, 1, stream, set);
                for (uint32_t grid : {2U, 13U, 300U, raggedKRArgGrid(RaggedKRShape{16, 0}, f.dtype, 0)}) {
                    for (CubArgReductionOp op : ARG_OPS) {
                        const std::vector<uint64_t> got = runArg(in, op, RaggedKRShape{16, 0}, grid, stream);
                        const std::string what = std::string(f.name) + " " + c.name + " segments " +
                                                 std::to_string(segments) + " grid " + std::to_string(grid) + " " +
                                                 argOpName(op);
                        expectIndices(got, argReference(in, op, DataType::UINT32), what);
                        for (uint64_t s = 0; s < segments; ++s) {  // every row ties: each segment's first row
                            if (in.offsets[s] < in.offsets[s + 1]) {
                                ASSERT_EQ(got[s], in.offsets[s]) << what;
                            }
                        }
                    }
                }
            }
        }
        // The extremum twice, 150000 rows apart (different CTAs at every grid but 1), everything else larger / smaller.
        for (CubArgReductionOp op : ARG_OPS) {
            const uint32_t extreme = op == CubArgReductionOp::ArgMin ? x.lowest : x.largest;
            std::vector<std::pair<uint64_t, uint32_t>> set;
            for (uint64_t r = 0; r < rows; ++r) set.emplace_back(r, one);
            set.emplace_back(170000, extreme);
            set.emplace_back(20000, extreme);
            const ArgInput in = makeArgInput(f, rows, rows, 1, A::Distribution::Uniform, 8, 0, 3, 1, stream, set);
            for (uint32_t grid : {1U, 2U, 5U, 4096U}) {
                const std::vector<uint64_t> got = runArg(in, op, RaggedKRShape{16, 0}, grid, stream, DataType::UINT64);
                ASSERT_EQ(got.size(), 1U);
                EXPECT_EQ(got[0], 20000U) << f.name << " " << argOpName(op) << " grid " << grid;
            }
        }
    }
}

TEST(CubRaggedKR, ArgIgnoresInactiveCapacityGarbageWorkspaceAndRepeats) {  // 1311: every ARG format
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : ARG_FORMATS) {
        for (uint64_t misalign = 0; misalign < 16; ++misalign) {
            const uint64_t h = A::mix(0xA1298 + misalign);
            const uint64_t rows = 5000 + h % 5000;
            // The inactive capacity is NaN, which would win every segment that read it.
            const ArgInput in =
                makeArgInput(f, rows, rows + rows / 2, 1 + h % 300, A::Distribution::Pareto, 4, misalign, h, 2, stream);
            for (CubArgReductionOp op : ARG_OPS) {
                const std::string what = std::string(f.name) + " misalign " + std::to_string(misalign) + " " + argOpName(op);
                const std::vector<uint64_t> first = runArg(in, op, RaggedKRShape{16, 0}, 37, stream, DataType::UINT32, h);
                expectIndices(first, argReference(in, op, DataType::UINT32), what);
                expectIndices(runArg(in, op, RaggedKRShape{16, 0}, 37, stream, DataType::UINT32, h + 1), first,
                              what + " repeat");
            }
        }
    }
}

TEST(CubRaggedKR, ArgEmptySegmentsTinyProblemsPartsAndTheContract) {  // 1311: FP16 / BF16 too; 1313: FP32
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : ARG_FORMATS) {
        // Every segment empty: the sentinel of each index width.
        const ArgInput empty = makeArgInput(f, 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, 0, stream);
        for (CubArgReductionOp op : ARG_OPS) {
            for (DataType index : {DataType::UINT32, DataType::UINT64})
                expectIndices(runArg(empty, op, RaggedKRShape{16, 0}, 9, stream, index), argReference(empty, op, index),
                              std::string(f.name) + " empty " + argOpName(op));
        }
        // Half the segments empty, and more CTAs than path events (the empty CTAs are last).
        const ArgInput half = makeArgInput(f, 20000, 20000, 9000, A::Distribution::HalfEmpty, 4, 0, 4, 1, stream);
        const ArgInput tiny = makeArgInput(f, 3, 3, 2, A::Distribution::Random, 4, 2, 6, 0, stream);
        for (CubArgReductionOp op : ARG_OPS) {
            expectIndices(runArg(half, op, RaggedKRShape{16, 0}, 170, stream), argReference(half, op, DataType::UINT32),
                          std::string(f.name) + " half empty " + argOpName(op));
            expectIndices(runArg(tiny, op, RaggedKRShape{16, 0}, 64, stream), argReference(tiny, op, DataType::UINT32),
                          std::string(f.name) + " tiny " + argOpName(op));
        }
    }

    // The main kernel then the finisher, launched apart on one workspace, compute what one launch does (1313: FP32's
    // pair records and finisher too).
    for (const Format& f : {FORMATS[0], WIDE_FORMATS[2]}) {
        const ArgInput apart = makeArgInput(f, 50000, 50000, 3000, A::Distribution::Pareto, 4, 3, 0x1298, 1, stream);
        const uint32_t grid = raggedKRArgGrid(RaggedKRShape{16, 0}, f.dtype, 0);
        for (CubArgReductionOp op : ARG_OPS) {
            Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {apart.offsets.size() - 1}));
            Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(f.dtype, grid)}));
            launchRaggedKRArg(op, RaggedKRShape{16, 0}, grid, apart.input, apart.offsets_tensor, output, workspace,
                              stream, RaggedLaunchParts::MainOnly);
            launchRaggedKRArg(op, RaggedKRShape{16, 0}, grid, apart.input, apart.offsets_tensor, output, workspace,
                              stream, RaggedLaunchParts::FinishOnly);
            stream.synchronize();
            expectIndices(copyGpuTensorAsUnsigned(output, stream), argReference(apart, op, DataType::UINT32),
                          std::string(f.name) + " apart " + argOpName(op));
        }
    }
    const ArgInput in = makeArgInput(FORMATS[0], 50000, 50000, 3000, A::Distribution::Pareto, 4, 3, 0x1298, 1, stream);

    // The contract.
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP8_E4M3, 1));
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP8_E5M2, 1));
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP8_E4M3, 2));  // 1300: D up to 15
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP8_E5M2, 15));
    EXPECT_FALSE(raggedKRArgSupported(DataType::FP8_E4M3, 16));
    EXPECT_FALSE(raggedKRArgSupported(DataType::FP8_E4M3, 0));
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP16, 1));  // 1311: FP16 / BF16 D 1 to 7
    EXPECT_TRUE(raggedKRArgSupported(DataType::BF16, 7));
    EXPECT_FALSE(raggedKRArgSupported(DataType::FP16, 8));
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP32, 1));  // 1313: FP32 D 1 to 3
    EXPECT_TRUE(raggedKRArgSupported(DataType::FP32, 3));
    EXPECT_FALSE(raggedKRArgSupported(DataType::FP32, 4));
    EXPECT_FALSE(raggedKRArgSupported(DataType::INT32, 1));
    EXPECT_EQ(raggedArgWorkspaceBytes(DataType::FP8_E4M3, 4), 32U + 32U + 32U);  // keys, heads, tails, padded to 16 B
    EXPECT_EQ(raggedArgWorkspaceBytes(DataType::BF16, 3, 2), 32U + 48U + 48U);
    // 1313: FP32's pair records, the grid rounded up to 4: keys 8 x 4, then 4- and 8-byte arrays of 4 x D entries.
    EXPECT_EQ(raggedArgWorkspaceBytes(DataType::FP32, 4), 32U + 16U + 32U + 16U + 32U);
    EXPECT_EQ(raggedArgWorkspaceBytes(DataType::FP32, 3, 2), 32U + 2U * (32U + 64U));
    const RaggedMergePath::PathWeights w =
        raggedKRArgWeights(RaggedKRShape{16, 0}, DataType::FP8_E4M3, DataType::UINT64, DataType::UINT32);
    EXPECT_EQ(w.row, 1U);
    EXPECT_EQ(w.end, 8U + 4U + 4U);
    const RaggedMergePath::PathWeights w16 =
        raggedKRArgWeights(RaggedKRShape{16, 0}, DataType::BF16, DataType::UINT32, DataType::UINT64, 3);
    EXPECT_EQ(w16.row, 6U);
    EXPECT_EQ(w16.end, 4U + 3U * 8U + 4U);
    EXPECT_THROW(static_cast<void>(raggedKRArgWeights(RaggedKRShape{16, 0}, DataType::FP8_E4M3, DataType::UINT32,
                                                      DataType::FP32)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRArgGrid(RaggedKRShape{16, 0}, DataType::FP32, 0, 4)), std::invalid_argument);
    const uint64_t segments = in.offsets.size() - 1;
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments}));
    Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP8_E4M3, 4)}));
    Tensor small(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP8_E4M3, 4) - 1}));
    const auto go = [&](CubArgReductionOp op, uint32_t g, const Tensor& input, Tensor& out, Tensor& ws) {
        launchRaggedKRArg(op, RaggedKRShape{16, 0}, g, input, in.offsets_tensor, out, ws, stream);
    };
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMax, 4, in.input, output, workspace));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 0, in.input, output, workspace), std::invalid_argument);
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, RAGGED_FINISH_MAX_GRID + 1, in.input, output, workspace),
                 std::invalid_argument);
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, in.input, output, small), std::invalid_argument);
    EXPECT_THROW(launchRaggedKRArg(CubArgReductionOp::ArgMin, RaggedKRShape{4, 0}, 4, in.input, in.offsets_tensor, output,
                                   workspace, stream),
                 std::invalid_argument);
    Tensor float_output(gpuPlacement, TensorDescriptor(DataType::FP32, {segments}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, in.input, float_output, workspace), std::invalid_argument);
    Tensor short_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments - 1}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, in.input, short_output, workspace), std::invalid_argument);
    // 1300: D = 2 with its own records (D = 1's workspace is too small); rows of 16 bytes are not KR's.
    Tensor wide = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {25000, 2}, 0);
    Tensor output2(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments, 2}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, wide, output2, workspace), std::invalid_argument);
    Tensor workspace2(gpuPlacement,
                      TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP8_E4M3, 4, 2)}));
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMax, 4, wide, output2, workspace2));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, wide, output, workspace2), std::invalid_argument);  // segments, not x 2
    Tensor row16 = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {3000, 16}, 0);
    Tensor output16(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments, 16}));
    Tensor workspace16(gpuPlacement,
                       TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP8_E4M3, 4, 16)}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, row16, output16, workspace16), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(raggedKRArgGrid(RaggedKRShape{16, 0}, DataType::FP8_E4M3, 0, 16)),
                 std::invalid_argument);
    // 1311: FP16 at D = 1 and BF16 at D = 2 take their own kernels; FP16 rows of 16 bytes are refused.
    Tensor fp16 = in.storage.aliasByteStorageAs(DataType::FP16, {25000}, 0);
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMax, 4, fp16, output, workspace));
    Tensor bf16 = in.storage.aliasByteStorageAs(DataType::BF16, {12500, 2}, 0);
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMin, 4, bf16, output2, workspace2));
    Tensor fp16_row16 = in.storage.aliasByteStorageAs(DataType::FP16, {3000, 8}, 0);
    Tensor output8(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments, 8}));
    Tensor workspace8(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP16, 4, 7)}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, fp16_row16, output8, workspace8), std::invalid_argument);
    // 1313: FP32 at D = 1 and D = 3 with their pair records (the 64-bit records of the same grid are too small); FP32
    // rows of 16 bytes are refused.
    Tensor fp32 = in.storage.aliasByteStorageAs(DataType::FP32, {12500}, 0);
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, fp32, output, workspace), std::invalid_argument);
    Tensor pairs(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP32, 4)}));
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMax, 4, fp32, output, pairs));
    Tensor fp32_d3 = in.storage.aliasByteStorageAs(DataType::FP32, {4000, 3}, 0);
    Tensor output3(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments, 3}));
    Tensor pairs3(gpuPlacement, TensorDescriptor(DataType::UINT8, {raggedArgWorkspaceBytes(DataType::FP32, 4, 3)}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMin, 4, fp32_d3, output3, pairs), std::invalid_argument);
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMin, 4, fp32_d3, output3, pairs3));
    Tensor fp32_row16 = in.storage.aliasByteStorageAs(DataType::FP32, {3000, 4}, 0);
    Tensor output4(gpuPlacement, TensorDescriptor(DataType::UINT32, {segments, 4}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, 4, fp32_row16, output4, workspace16), std::invalid_argument);
    stream.synchronize();
}

// 1300: D = 2..15 (raggedKRArgComponentsKernel and the finisher's CTA per component) against the reference per
// component (1311: FP16 / BF16 at D 2..7 too): every D, every format, the three pools, every distribution, both offset and index widths,
// misaligned bases, inactive capacity of NaN, grids from 1 CTA to the persistent grid.
TEST(CubRaggedKR, ArgComponentsMatchTheReferenceForEveryD) {  // 1311: every ARG format, D up to its rows under 16 B
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (uint64_t D = 2; D <= RaggedMergePath::RAGGED_KR_MAX_COMPONENTS; ++D) {
        for (const Format& f : ARG_FORMATS) {
            if (D > argMaxComponents(f)) continue;
            const uint64_t fi = static_cast<uint64_t>(&f - ARG_FORMATS);
            const uint64_t h = A::mix(0x1300 + D * 7 + 3 * fi);
            const A::Distribution d = A::DISTRIBUTIONS[(D + 2 * fi) % 5];
            const uint32_t pool = static_cast<uint32_t>((D + h) % 3);
            const uint64_t rows = 1 + h % 20000;
            const uint64_t segments = 1 + A::mix(h) % (D % 2 ? rows : 1500);
            const ArgInput in = makeArgInput(f, rows, rows + rows / 3, segments, d, D % 2 ? 8 : 4, h % 16, h, pool,
                                             stream, {}, D);
            const RaggedKRShape shape{16, 0};
            for (uint32_t grid : {1U, 3U, 64U, raggedKRArgGrid(shape, f.dtype, 0, static_cast<uint32_t>(D))}) {
                for (CubArgReductionOp op : ARG_OPS) {
                    const DataType index = (grid + D) % 2 ? DataType::UINT64 : DataType::UINT32;
                    const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                             A::distributionName(d) + " pool " + std::to_string(pool) + " seed " +
                                             std::to_string(h) + " grid " + std::to_string(grid) + " " + argOpName(op);
                    expectIndices(runArg(in, op, shape, grid, stream, index), argReference(in, op, index), what);
                }
            }
        }
    }
}

// 1300: ties split across CTAs at D > 1: every element equal (each component's winner is its segment's first row),
// and NaNs in different components.
TEST(CubRaggedKR, ArgComponentsTiesSplitAcrossCtasGoToTheLowestIndex) {  // 1311: every ARG format
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : ARG_FORMATS) {
        const ArgSpecials x = argSpecials(f);
        for (uint64_t D : {2ULL, 3ULL, 7ULL, 15ULL}) {
            if (D > argMaxComponents(f)) continue;
            const uint64_t rows = 60000;
            std::vector<std::pair<uint64_t, uint32_t>> set;
            for (uint64_t e = 0; e < rows * D; ++e) {
                const uint64_t c = e % D;
                set.emplace_back(e, c % 3 == 0 ? x.one : (c % 3 == 1 ? x.nan_a : (e % 2 ? x.negative_zero : 0U)));
            }
            const ArgInput in = makeArgInput(f, rows, rows, 3, A::Distribution::Uniform, 4, 1, 5, 1, stream, set, D);
            for (uint32_t grid : {2U, 13U, 300U}) {
                for (CubArgReductionOp op : ARG_OPS) {
                    const std::vector<uint64_t> got = runArg(in, op, RaggedKRShape{16, 0}, grid, stream);
                    const std::string what = std::string(f.name) + " D " + std::to_string(D) + " grid " +
                                             std::to_string(grid) + " " + argOpName(op);
                    expectIndices(got, argReference(in, op, DataType::UINT32), what);
                    for (uint64_t s = 0; s < 3; ++s) {
                        for (uint64_t c = 0; c < D; ++c) {
                            ASSERT_EQ(got[s * D + c], in.offsets[s] * D + c) << what;
                        }
                    }
                }
            }
        }
    }
}

// 1314 (phase 5): ragged RK ARG, FP8 at D >= 16 (launchRaggedRKArg, CubRaggedRKArg.cu; the kernel
// RaggedRKArgKernels.cuh; the finishers launchRaggedFinishArg: the wide one above D 15, direct up to 128 ranges and
// transposed above), against the ARG reference per column (argReference). 1320: FP16 / BF16 at D >= 8 too. 1321: FP32
// at D >= 4, its pair records and pair finishers (launchRaggedFinishArgPair: the block one up to D 15, the wide one
// above, one to 16 warps per component).
namespace {

// 1320: the RK ARG formats (FP8 both, FP16, BF16; 1321: FP32) and each one's smallest RK row (16 bytes).
const Format RK_ARG_FORMATS[] = {FORMATS[0], FORMATS[1], WIDE_FORMATS[0], WIDE_FORMATS[1], WIDE_FORMATS[2]};
uint64_t rkArgMinComponents(const Format& f) { return 16U / f.bytes; }

std::vector<uint64_t> runRKArg(const ArgInput& in, CubArgReductionOp op, const RaggedRKShape& shape, uint32_t ctas,
                               Stream& stream, DataType index_dtype = DataType::UINT32, uint64_t garbage_seed = 7) {
    const uint64_t segments = in.offsets.size() - 1;
    Tensor output = makeGpuUnsignedTensor(std::vector<uint64_t>(segments * in.d, 0x5A5A5A5AU), {segments, in.d},
                                          stream, index_dtype);
    const RaggedRKLaunch launch =
        raggedRKArgLaunch(shape, ctas, in.input.getDataType(), in.d, in.input.getDimensions()[0]);
    std::vector<uint64_t> garbage(launch.workspace_bytes);
    for (size_t i = 0; i < garbage.size(); ++i) garbage[i] = A::mix(garbage_seed + i) & 0xFF;
    Tensor workspace = makeGpuUnsignedTensor(garbage, {garbage.size()}, stream, DataType::UINT8);
    launchRaggedRKArg(op, shape, ctas, in.input, in.offsets_tensor, output, workspace, stream);
    stream.synchronize();
    return copyGpuTensorAsUnsigned(output, stream);
}

}  // namespace

// Every D from 16 to 5000 (one K tile at every worker count, narrow rows at 4 rows per worker, K-tiled rows), K tile
// caps and the auto cap, CTA budgets from 1 to the GPU's, both formats, the three element pools, every distribution,
// both offset and index widths, misaligned bases. 1320: each D in an FP8 format and an F16 one (FP16 / BF16), and the
// F16 rows of 16 to 30 bytes (D 8, 9, 12, 15). 1321: and in FP32 (rows of 16 bytes and more: D 4 to 7 too).
TEST(CubRaggedKR, RkArgMatchesTheReferenceForEveryDTileCapAndIndex) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t ds[] = {16, 17, 24, 31, 32, 64, 255, 256, 513, 1024, 5000, 8, 9, 12, 15, 4, 5, 6, 7};
    const uint32_t caps[] = {0, 32, 128, 512, 64, 256};
    for (size_t r = 0; r < 3 * sizeof(ds) / sizeof(ds[0]); ++r) {
        const size_t t = r / 3, kind = r % 3;  // FP8, F16, FP32 (1321)
        const uint64_t D = ds[t];
        const Format& f = kind == 2 ? WIDE_FORMATS[2] : (kind == 1 ? WIDE_FORMATS[t % 2] : FORMATS[t % 2]);
        if (D < rkArgMinComponents(f)) continue;
        const uint32_t full = raggedRKArgCtas(RaggedRKShape{}, f.dtype, 0);
        const A::Distribution d = A::DISTRIBUTIONS[t % 5];
        const uint32_t pool = static_cast<uint32_t>(t % 3);
        const uint64_t h = A::mix(0x1314 + D + 1000003 * kind);
        const uint64_t rows = 1 + h % std::max<uint64_t>(64, 2000000 / (D * f.bytes));
        const uint64_t segments = 1 + A::mix(h) % (t % 3 == 0 ? rows : 700);
        const ArgInput in = makeArgInput(f, rows, rows + 3, segments, d, t % 2 ? 8 : 4, D % 16, h, pool, stream, {}, D);
        for (size_t c = 0; c < 2; ++c) {
            RaggedRKShape shape;
            shape.tile_bytes = caps[(t + c) % 6];
            for (uint32_t ctas : {1U, 7U, 300U, full}) {
                for (CubArgReductionOp op : ARG_OPS) {
                    const DataType index = (ctas + t) % 2 ? DataType::UINT64 : DataType::UINT32;
                    const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                             A::distributionName(d) + " pool " + std::to_string(pool) + " seed " +
                                             std::to_string(h) + " " + raggedRKShapeName(shape) + " ctas " +
                                             std::to_string(ctas) + " " + argOpName(op);
                    expectIndices(runRKArg(in, op, shape, ctas, stream, index), argReference(in, op, index), what);
                }
            }
        }
    }
}

// Ties split across ranges and K tiles: every element of a column equal (one value, NaNs in several encodings, -0 and
// +0 mixed), so each column's winner is its segment's first row; the extremum planted twice in one column far apart.
TEST(CubRaggedKR, RkArgTiesSplitAcrossCtasGoToTheLowestIndex) {  // 1320: FP16 / BF16 too (and D 8, 9); 1321: FP32
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Format& f : RK_ARG_FORMATS) {
        const ArgSpecials x = argSpecials(f);
        for (uint64_t D : {4ULL, 5ULL, 8ULL, 9ULL, 16ULL, 17ULL, 300ULL}) {
            if (D < rkArgMinComponents(f)) continue;
            const uint64_t rows = D > 100 ? 4000 : 40000;
            std::vector<std::pair<uint64_t, uint32_t>> set;
            for (uint64_t e = 0; e < rows * D; ++e) {
                const uint64_t c = e % D, r = e / D;
                const uint32_t nans[] = {x.nan_a, x.nan_b, x.nan_c};
                set.emplace_back(e, c % 3 == 0 ? x.one
                                               : (c % 3 == 1 ? nans[r % 3] : (r % 2 ? x.negative_zero : 0U)));
            }
            const ArgInput in = makeArgInput(f, rows, rows, 3, A::Distribution::Uniform, 4, 1, 5, 1, stream, set, D);
            for (const RaggedRKShape shape : {RaggedRKShape{}, RaggedRKShape{32}}) {
                for (uint32_t ctas : {2U, 13U, 300U}) {
                    for (CubArgReductionOp op : ARG_OPS) {
                        const std::vector<uint64_t> got = runRKArg(in, op, shape, ctas, stream);
                        const std::string what = std::string(f.name) + " D " + std::to_string(D) + " " +
                                                 raggedRKShapeName(shape) + " ctas " + std::to_string(ctas) + " " +
                                                 argOpName(op);
                        expectIndices(got, argReference(in, op, DataType::UINT32), what);
                        for (uint64_t s = 0; s < 3; ++s)
                            for (uint64_t c = 0; c < D; ++c) ASSERT_EQ(got[s * D + c], in.offsets[s] * D + c) << what;
                    }
                }
            }
        }
        for (CubArgReductionOp op : ARG_OPS) {  // column 5 of D 40: the extremum at rows 900 and 20000 of 30000
            const uint64_t D = 40, rows = 30000;
            const uint32_t extreme = op == CubArgReductionOp::ArgMin ? x.lowest : x.largest;
            std::vector<std::pair<uint64_t, uint32_t>> set;
            for (uint64_t e = 0; e < rows * D; ++e) set.emplace_back(e, x.one);
            set.emplace_back(20000 * D + 5, extreme);
            set.emplace_back(900 * D + 5, extreme);
            const ArgInput in = makeArgInput(f, rows, rows, 1, A::Distribution::Uniform, 8, 0, 3, 1, stream, set, D);
            for (uint32_t ctas : {1U, 2U, 5U, 4096U}) {
                const std::vector<uint64_t> got = runRKArg(in, op, RaggedRKShape{}, ctas, stream, DataType::UINT64);
                ASSERT_EQ(got.size(), D);
                EXPECT_EQ(got[5], 900U * D + 5U) << f.name << " " << argOpName(op) << " ctas " << ctas;
                EXPECT_EQ(got[6], 6U) << f.name << " " << argOpName(op) << " ctas " << ctas;  // all ties: row 0
            }
        }
    }
}

// Inactive capacity of NaN (it would win if read), garbage workspaces and repeats at every base misalignment (D 32
// takes the one-load path at misalignment 0); every segment empty; half empty; tiny problems; the main kernel and the
// finisher launched apart (block and wide finishers); the contract.
// 1320: FP16 / BF16 too: misalignments counted in elements (FP16 D 16 takes the one-load path at misalignments 0
// and 8), the empty, half-empty, tiny and apart runs in each RK format.
// 1321: FP32 too (its misalignments in elements as well; D 8 takes the one-load path at misalignments 0, 4, 8 and 12):
// the apart runs at D 19 and 700 take the wide pair finisher, D 8 the block one.
TEST(CubRaggedKR, RkArgIgnoresInactiveCapacityEmptySegmentsPartsAndTheContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t full = raggedRKArgCtas(RaggedRKShape{}, DataType::FP8_E5M2, 0);
    for (uint64_t D : {32ULL, 19ULL, 700ULL, 16ULL, 8ULL}) {
        const uint64_t step = D == 32 || D == 16 || D == 8 ? 1 : 5;
        for (uint64_t misalign = 0; misalign < 16; misalign += step) {
            const Format& f = D == 16  ? WIDE_FORMATS[misalign % 2]
                              : D == 8 ? WIDE_FORMATS[2]
                                       : RK_ARG_FORMATS[(misalign / step + D) % 5];
            const uint64_t h = A::mix(0x1314CAFE + 31 * misalign + D);
            const uint64_t rows = 500 + h % (D > 100 ? 2000 : 20000);
            const ArgInput in = makeArgInput(f, rows, rows + rows / 2, 1 + h % 300, A::Distribution::Pareto, 4,
                                             misalign, h, 2, stream, {}, D);
            for (CubArgReductionOp op : ARG_OPS) {
                const std::string what = std::string(f.name) + " D " + std::to_string(D) + " misalign " +
                                         std::to_string(misalign) + " " + argOpName(op);
                const std::vector<uint64_t> first = runRKArg(in, op, RaggedRKShape{}, 37, stream, DataType::UINT32, h);
                expectIndices(first, argReference(in, op, DataType::UINT32), what);
                expectIndices(runRKArg(in, op, RaggedRKShape{}, 37, stream, DataType::UINT32, h + 1), first,
                              what + " repeat");
            }
        }
        for (const Format& f : RK_ARG_FORMATS) {
            if (D < rkArgMinComponents(f) || (f.bytes == 1 && D == 16)) continue;  // D 16 runs FP8 elsewhere
            const std::string name = std::string(f.name) + " D " + std::to_string(D);
            const ArgInput empty = makeArgInput(f, 0, 64, 50, A::Distribution::Uniform, 8, 0, 3, 0, stream, {}, D);
            const ArgInput half =
                makeArgInput(f, 3000, 3000, 1500, A::Distribution::HalfEmpty, 4, 0, 4, 1, stream, {}, D);
            const ArgInput tiny = makeArgInput(f, 3, 3, 2, A::Distribution::Random, 4, 2, 6, 0, stream, {}, D);
            const uint32_t f_full = raggedRKArgCtas(RaggedRKShape{}, f.dtype, 0);
            for (CubArgReductionOp op : ARG_OPS) {
                for (DataType index : {DataType::UINT32, DataType::UINT64})
                    expectIndices(runRKArg(empty, op, RaggedRKShape{}, 9, stream, index),
                                  argReference(empty, op, index), name + " empty " + argOpName(op));
                expectIndices(runRKArg(half, op, RaggedRKShape{}, f_full, stream),
                              argReference(half, op, DataType::UINT32), name + " half empty " + argOpName(op));
                expectIndices(runRKArg(tiny, op, RaggedRKShape{}, 64, stream),
                              argReference(tiny, op, DataType::UINT32), name + " tiny " + argOpName(op));
            }
            // The main kernel then the finisher, two launches on one workspace and one output.
            const ArgInput apart =
                makeArgInput(f, 4000, 4000, 300, A::Distribution::Random, 4, 0, 0x1314, 0, stream, {}, D);
            for (uint32_t ctas : {7U, f_full}) {
                const RaggedRKLaunch launch = raggedRKArgLaunch(RaggedRKShape{}, ctas, f.dtype, D, 4000);
                for (CubArgReductionOp op : ARG_OPS) {
                    Tensor output = makeGpuUnsignedTensor(std::vector<uint64_t>(300 * D, 0x5A5A5A5AU), {300, D},
                                                          stream, DataType::UINT32);
                    Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
                    launchRaggedRKArg(op, RaggedRKShape{}, ctas, apart.input, apart.offsets_tensor, output, workspace,
                                      stream, RaggedLaunchParts::MainOnly);
                    launchRaggedRKArg(op, RaggedRKShape{}, ctas, apart.input, apart.offsets_tensor, output, workspace,
                                      stream, RaggedLaunchParts::FinishOnly);
                    stream.synchronize();
                    expectIndices(copyGpuTensorAsUnsigned(output, stream), argReference(apart, op, DataType::UINT32),
                                  name + " apart ctas " + std::to_string(ctas) + " " + argOpName(op));
                }
            }
        }
    }

    // The contract.
    EXPECT_TRUE(raggedRKArgSupported(DataType::FP8_E4M3, 16));
    EXPECT_TRUE(raggedRKArgSupported(DataType::FP8_E5M2, 5000));
    EXPECT_FALSE(raggedRKArgSupported(DataType::FP8_E4M3, 15));  // KR's
    EXPECT_TRUE(raggedRKArgSupported(DataType::FP16, 8));        // 1320: FP16 / BF16 from 16-byte rows
    EXPECT_TRUE(raggedRKArgSupported(DataType::BF16, 5000));
    EXPECT_FALSE(raggedRKArgSupported(DataType::FP16, 7));       // KR's
    EXPECT_FALSE(raggedRKArgSupported(DataType::BF16, 7));
    EXPECT_TRUE(raggedRKArgSupported(DataType::FP32, 4));        // 1321: FP32 from 16-byte rows
    EXPECT_TRUE(raggedRKArgSupported(DataType::FP32, 5000));
    EXPECT_FALSE(raggedRKArgSupported(DataType::FP32, 3));       // KR's
    EXPECT_FALSE(raggedRKArgSupported(DataType::INT32, 4));      // not yet
    EXPECT_FALSE(raggedRKArgSupported(DataType::FP16, (RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES / 2) + 1));
    EXPECT_FALSE(raggedRKArgSupported(DataType::FP32, (RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES / 4) + 1));
    int sms = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
    EXPECT_GE(full, static_cast<uint32_t>(sms));
    EXPECT_EQ(raggedRKArgCtas(RaggedRKShape{0, 1}, DataType::FP8_E4M3, 0), static_cast<uint32_t>(sms));
    EXPECT_GE(raggedRKArgCtas(RaggedRKShape{}, DataType::FP16, 0), static_cast<uint32_t>(sms));  // 1320
    EXPECT_GE(raggedRKArgCtas(RaggedRKShape{}, DataType::FP32, 0), static_cast<uint32_t>(sms));  // 1321
    EXPECT_THROW(static_cast<void>(raggedRKArgCtas(RaggedRKShape{}, DataType::INT32, 0)), std::invalid_argument);
    const RaggedRKLaunch l16 = raggedRKArgLaunch(RaggedRKShape{}, 2720, DataType::FP8_E4M3, 16, 1 << 20);
    EXPECT_EQ(l16.geometry.ktiles, 1U);
    EXPECT_EQ(l16.ranges, 2720U);
    EXPECT_EQ(l16.workspace_bytes, RaggedMergePath::raggedArgRecordLayout(2720, 16).bytes);
    // The auto cap holds the records (16 bytes per column and range) within 1/16 of the input: at 64 MiB and 2720
    // CTAs VALUE takes 128 bytes, ARG 64.
    EXPECT_EQ(raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP8_E4M3, 4096, 16384).tile_cap, 128U);
    EXPECT_EQ(raggedRKArgLaunch(RaggedRKShape{}, 2720, DataType::FP8_E4M3, 4096, 16384).tile_cap, 64U);
    const RaggedMergePath::PathWeights w =
        raggedRKArgWeights(RaggedRKShape{}, DataType::FP8_E4M3, DataType::UINT64, DataType::UINT32, 20);
    EXPECT_EQ(w.row, 20U);
    EXPECT_EQ(w.end, 8U + 20U * 4U + 4U);
    EXPECT_THROW(static_cast<void>(raggedRKArgWeights(RaggedRKShape{}, DataType::FP8_E4M3, DataType::UINT32,
                                                      DataType::FP32, 20)),
                 std::invalid_argument);
    const RaggedMergePath::PathWeights wb =  // 1320: a BF16 row is 2 D bytes
        raggedRKArgWeights(RaggedRKShape{}, DataType::BF16, DataType::UINT32, DataType::UINT64, 20);
    EXPECT_EQ(wb.row, 40U);
    EXPECT_EQ(wb.end, 4U + 20U * 8U + 4U);
    EXPECT_EQ(raggedRKArgLaunch(RaggedRKShape{}, 2720, DataType::FP16, 8, 1 << 20).workspace_bytes,
              RaggedMergePath::raggedArgRecordLayout(2720, 8).bytes);
    // 1321: FP32's pair records (24 bytes per column and range): the pair layout, and the auto cap at 48 MiB and 2720
    // CTAs 128 bytes (16-byte records would take 256, VALUE 512).
    EXPECT_EQ(raggedRKArgLaunch(RaggedRKShape{}, 2720, DataType::FP32, 4, 1 << 20).workspace_bytes,
              RaggedMergePath::raggedArgPairRecordLayout(2720, 4).bytes);
    EXPECT_EQ(raggedRKArgLaunch(RaggedRKShape{}, 2720, DataType::FP32, 768, 16384).tile_cap, 128U);
    EXPECT_EQ(raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP32, 768, 16384).tile_cap, 512U);
    const ArgInput in = makeArgInput(FORMATS[0], 1000, 1000, 10, A::Distribution::Uniform, 4, 0, 9, 1, stream, {}, 20);
    const RaggedRKLaunch launch = raggedRKArgLaunch(RaggedRKShape{}, 7, DataType::FP8_E4M3, 20, 1000);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::UINT32, {10, 20}));
    Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
    Tensor small(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes - 1}));
    const auto go = [&](CubArgReductionOp op, const RaggedRKShape& shape, const Tensor& input, Tensor& out,
                        Tensor& ws) { launchRaggedRKArg(op, shape, 7, input, in.offsets_tensor, out, ws, stream); };
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMax, RaggedRKShape{}, in.input, output, workspace));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, RaggedRKShape{}, in.input, output, small), std::invalid_argument);
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, RaggedRKShape{96}, in.input, output, workspace), std::invalid_argument);
    Tensor float_output(gpuPlacement, TensorDescriptor(DataType::FP32, {10, 20}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, RaggedRKShape{}, in.input, float_output, workspace),
                 std::invalid_argument);
    Tensor short_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {10, 19}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMax, RaggedRKShape{}, in.input, short_output, workspace),
                 std::invalid_argument);
    Tensor narrow = in.storage.aliasByteStorageAs(DataType::FP8_E4M3, {1000, 15}, 0);  // KR's
    Tensor output15(gpuPlacement, TensorDescriptor(DataType::UINT32, {10, 15}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMin, RaggedRKShape{}, narrow, output15, workspace), std::invalid_argument);
    Tensor fp16 = in.storage.aliasByteStorageAs(DataType::FP16, {1000, 10}, 0);  // 1320: FP16 RK ARG at D 10
    Tensor output10(gpuPlacement, TensorDescriptor(DataType::UINT32, {10, 10}));
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMin, RaggedRKShape{}, fp16, output10, workspace));
    Tensor fp16_narrow = in.storage.aliasByteStorageAs(DataType::FP16, {1000, 7}, 0);  // KR's
    Tensor output7(gpuPlacement, TensorDescriptor(DataType::UINT32, {10, 7}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMin, RaggedRKShape{}, fp16_narrow, output7, workspace),
                 std::invalid_argument);
    Tensor fp32 = in.storage.aliasByteStorageAs(DataType::FP32, {500, 10}, 0);  // 1321: FP32 RK ARG at D 10
    Tensor pairs(gpuPlacement, TensorDescriptor(DataType::UINT8,
                                                {raggedRKArgLaunch(RaggedRKShape{}, 7, DataType::FP32, 10, 500)
                                                     .workspace_bytes}));
    EXPECT_NO_THROW(go(CubArgReductionOp::ArgMin, RaggedRKShape{}, fp32, output10, pairs));
    Tensor fp32_narrow = in.storage.aliasByteStorageAs(DataType::FP32, {1000, 3}, 0);  // KR's
    Tensor output3(gpuPlacement, TensorDescriptor(DataType::UINT32, {10, 3}));
    EXPECT_THROW(go(CubArgReductionOp::ArgMin, RaggedRKShape{}, fp32_narrow, output3, pairs), std::invalid_argument);
    Tensor int32 = in.storage.aliasByteStorageAs(DataType::INT32, {500, 10}, 0);  // not yet
    EXPECT_THROW(go(CubArgReductionOp::ArgMin, RaggedRKShape{}, int32, output10, pairs), std::invalid_argument);
    stream.synchronize();
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
