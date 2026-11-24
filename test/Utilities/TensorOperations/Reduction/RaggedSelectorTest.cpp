// The ragged selector (selectCachedRagged, RaggedStampAutotuner.h), called directly (RaggedRoutingTest runs it through
// the stamps). Host: the family of each dtype and D (KR below 16-byte rows, RK from 16), the candidate lists, the
// site's refusals. GPU: every legal candidate is timed, the selection is cached per site under kind RaggedKR / RaggedRK
// (every field of the site its own entry, the tuning depth ignored), a site without rows or segments times nothing, and
// the winner computes what the first candidate and a host reference compute (VALUE SUM / MAX / MEAN and ARG, misaligned
// inputs, inactive capacity, empty segments).

#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedStampAutotuner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;
using namespace ThorImplementation::ReductionInternal;

namespace {

RaggedSite valueSite(DataType input, uint64_t d, ReductionOp op, uint64_t capacity, uint64_t segments) {
    RaggedSite site;
    site.value_op = op;
    site.input_dtype = input;
    site.output_dtype = DataType::FP32;
    site.components = d;
    site.capacity_rows = capacity;
    site.segments = segments;
    return site;
}

RaggedSite argSite(DataType input, uint64_t d, ArgReductionOp op, uint64_t capacity, uint64_t segments,
                   DataType index = DataType::UINT32) {
    RaggedSite site = valueSite(input, d, ReductionOp::Sum, capacity, segments);
    site.arg = true;
    site.arg_op = op;
    site.output_dtype = index;
    return site;
}

std::vector<std::string> candidateNames(RaggedFamily family) {
    std::vector<std::string> names;
    for (const RaggedSelection& s : raggedCandidateSelections(family)) names.push_back(s.winner_name);
    return names;
}

// Random offsets over the first active rows of the capacity, about every seventh segment forced empty.
std::vector<uint64_t> randomOffsets(uint64_t active, uint64_t segments, std::mt19937_64& rng) {
    std::vector<uint64_t> offsets(segments + 1, 0);
    for (uint64_t i = 1; i < segments; ++i) offsets[i] = rng() % (active + 1);
    offsets[segments] = active;
    std::sort(offsets.begin(), offsets.end());
    for (uint64_t i = 7; i < segments; i += 7) offsets[i] = offsets[i - 1];
    return offsets;
}

struct Problem {
    RaggedSite site;
    std::vector<float> values;  // capacity x D, row major
    std::vector<uint64_t> offsets;
    Tensor input, offsets_tensor;
};

Problem makeProblem(const RaggedSite& site, uint64_t seed, Stream& stream) {
    Problem p;
    p.site = site;
    std::mt19937_64 rng(seed);
    const uint64_t n = site.capacity_rows, d = site.components;
    p.values.resize(n * d);
    for (uint64_t i = 0; i < n * d; ++i)
        p.values[i] = static_cast<float>(static_cast<int>(rng() % 17) - 8);  // exact in every input dtype
    p.offsets = randomOffsets(n - n / 7, site.segments, rng);  // the last seventh of the capacity inactive
    p.input = makeGpuTensor(p.values, d == 1 ? std::vector<uint64_t>{n} : std::vector<uint64_t>{n, d}, stream,
                            site.input_dtype);
    p.offsets_tensor = makeGpuUnsignedTensor(p.offsets, {site.segments + 1}, stream, site.offsets_dtype);
    return p;
}

// One launch of `selection` on the problem; the outputs as unsigned (ARG) or float bits (VALUE).
std::vector<uint64_t> runSelection(const RaggedSelection& selection, Problem& p, Stream& stream) {
    const RaggedSelectedLaunch launch = raggedSelectedLaunch(selection, p.site, 0);
    const uint64_t b = p.site.segments, d = p.site.components;
    // The output in its own dtype (the index dtype, or FP32 for VALUE), filled with garbage first; read back as bits.
    const DataType output_dtype = p.site.arg ? p.site.output_dtype : DataType::FP32;
    const uint64_t width = output_dtype == DataType::UINT64 ? 8 : 4;
    Tensor output(gpuPlacement, TensorDescriptor(output_dtype, {b, d}));
    output.memsetAsync(stream, 0x5A);
    Tensor workspace(gpuPlacement, TensorDescriptor(DataType::UINT8, {launch.workspace_bytes}));
    launchRaggedSelection(selection, p.site, launch, p.input, p.offsets_tensor, output, workspace, stream);
    std::vector<uint8_t> bytes(b * d * width);
    EXPECT_EQ(cudaMemcpyAsync(bytes.data(), output.getMemPtr<void>(), bytes.size(), cudaMemcpyDeviceToHost,
                              stream.getStream()),
              cudaSuccess);
    stream.synchronize();
    std::vector<uint64_t> bits(b * d, 0);
    for (uint64_t i = 0; i < bits.size(); ++i) std::memcpy(&bits[i], bytes.data() + i * width, width);
    return bits;
}

// The host reference: ARG the first winner (lowest row) per column or the sentinel; VALUE SUM / MAX exact.
std::vector<uint64_t> reference(const Problem& p) {
    const uint64_t b = p.site.segments, d = p.site.components;
    std::vector<uint64_t> out(b * d);
    const uint64_t sentinel = p.site.output_dtype == DataType::UINT64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    for (uint64_t s = 0; s < b; ++s) {
        for (uint64_t c = 0; c < d; ++c) {
            if (p.site.arg) {
                uint64_t best = sentinel;
                for (uint64_t r = p.offsets[s]; r < p.offsets[s + 1]; ++r) {
                    const float v = p.values[r * d + c];
                    if (best == sentinel) {
                        best = r * d + c;
                        continue;
                    }
                    const float w = p.values[best];
                    if (p.site.arg_op == ArgReductionOp::ArgMin ? v < w : v > w) best = r * d + c;
                }
                out[s * d + c] = best;
            } else {
                float acc = p.site.value_op == ReductionOp::Max ? -std::numeric_limits<float>::infinity() : 0.0f;
                for (uint64_t r = p.offsets[s]; r < p.offsets[s + 1]; ++r) {
                    const float v = p.values[r * d + c];
                    acc = p.site.value_op == ReductionOp::Max ? std::max(acc, v) : acc + v;
                }
                uint32_t bits = 0;
                std::memcpy(&bits, &acc, sizeof(bits));
                out[s * d + c] = bits;
            }
        }
    }
    return out;
}

}  // namespace

TEST(RaggedSelector, FamiliesCandidatesAndRefusals) {
    // KR below 16-byte rows, RK from 16, VALUE and ARG alike.
    for (const bool arg : {false, true}) {
        SCOPED_TRACE(arg ? "ARG" : "VALUE");
        for (const DataType fp8 : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            EXPECT_EQ(raggedFamilyFor(arg, fp8, 1), RaggedFamily::KR);
            EXPECT_EQ(raggedFamilyFor(arg, fp8, 15), RaggedFamily::KR);
            EXPECT_EQ(raggedFamilyFor(arg, fp8, 16), RaggedFamily::RK);
        }
        for (const DataType f16 : {DataType::FP16, DataType::BF16}) {
            EXPECT_EQ(raggedFamilyFor(arg, f16, 7), RaggedFamily::KR);
            EXPECT_EQ(raggedFamilyFor(arg, f16, 8), RaggedFamily::RK);
            EXPECT_EQ(raggedFamilyFor(arg, f16, 5000), RaggedFamily::RK);
            EXPECT_FALSE(raggedFamilyFor(arg, f16, RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES / 2 + 1));
        }
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP32, 1), RaggedFamily::KR);
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP32, 3), RaggedFamily::KR);
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP32, 4), RaggedFamily::RK);
        EXPECT_FALSE(raggedFamilyFor(arg, DataType::FP32, 0));
        EXPECT_FALSE(raggedFamilyFor(arg, DataType::INT32, 4));
        EXPECT_FALSE(raggedFamilyFor(arg, DataType::UINT8, 16));
    }

    // Three candidates a family (time them all while at most RAGGED_TIME_ALL_CANDIDATE_LIMIT), each legal.
    EXPECT_EQ(candidateNames(RaggedFamily::KR),
              (std::vector<std::string>{"ragged_kr:i16_c0_e12", "ragged_kr:i16_c0_e4", "ragged_kr:i16_c8_e4"}));
    EXPECT_EQ(candidateNames(RaggedFamily::RK),
              (std::vector<std::string>{"ragged_rk:ta_c0_e4", "ragged_rk:t64_c0_e4", "ragged_rk:t512_c0_e4"}));
    EXPECT_LE(raggedKRCandidateShapes().size(), RAGGED_TIME_ALL_CANDIDATE_LIMIT);
    EXPECT_LE(raggedRKCandidateShapes().size(), RAGGED_TIME_ALL_CANDIDATE_LIMIT);
    for (const RaggedKRShape& shape : raggedKRCandidateShapes()) EXPECT_TRUE(raggedKRShapeLegal(shape));
    for (const RaggedRKShape& shape : raggedRKCandidateShapes()) EXPECT_TRUE(raggedRKShapeLegal(shape));
    for (const RaggedSelection& s : raggedCandidateSelections(RaggedFamily::RK)) {
        EXPECT_EQ(s.family, RaggedFamily::RK);
        EXPECT_EQ(s.winner_name, raggedSelectionShapeName(s));
        EXPECT_EQ(s.candidate_count, 0U);
    }

    // The synthetic offsets, built incrementally, against floor(i N / B) (UINT32 clamped).
    for (const auto& [n, b] : std::vector<std::pair<uint64_t, uint64_t>>{
             {0, 1}, {1, 1}, {7, 3}, {1000, 7}, {1000, 1000}, {3, 10}, {123456789, 9973}, {uint64_t{1} << 40, 999983},
             {(uint64_t{1} << 33) + 5, 17}}) {
        for (const DataType dtype : {DataType::UINT32, DataType::UINT64}) {
            const std::vector<uint64_t> got = raggedUniformOffsets(n, b, dtype);
            ASSERT_EQ(got.size(), b + 1);
            const uint64_t cap = dtype == DataType::UINT32 ? uint64_t{0xFFFFFFFFU} : ~uint64_t{0};
            for (uint64_t i = 0; i <= b; i += 1 + b / 1000) {
                const unsigned __int128 row = static_cast<unsigned __int128>(i) * n / b;
                ASSERT_EQ(got[i], std::min<uint64_t>(static_cast<uint64_t>(row), cap))
                    << "N " << n << " B " << b << " i " << i;
            }
            EXPECT_EQ(got[b], std::min(n, cap));
        }
    }
    EXPECT_THROW(static_cast<void>(raggedUniformOffsets(10, 0, DataType::UINT32)), std::invalid_argument);

    // Refused before anything runs: the stream is never used (no GPU needed).
    const Stream no_stream;
    const auto refuses = [&](RaggedSite site) {
        EXPECT_THROW(static_cast<void>(selectCachedRagged(site, no_stream)), std::invalid_argument);
    };
    refuses(valueSite(DataType::INT32, 4, ReductionOp::Sum, 100, 10));
    refuses(valueSite(DataType::FP32, 0, ReductionOp::Sum, 100, 10));
    refuses(valueSite(DataType::FP32, 4, ReductionOp::L1Norm, 100, 10));
    refuses(valueSite(DataType::FP32, 4, ReductionOp::Product, 100, 10));
    RaggedSite site = valueSite(DataType::FP32, 4, ReductionOp::Sum, 100, 10);
    site.offsets_dtype = DataType::INT32;
    refuses(site);
    site = valueSite(DataType::FP32, 4, ReductionOp::Sum, 100, 10);
    site.output_dtype = DataType::UINT32;
    refuses(site);
    site = argSite(DataType::FP16, 8, ArgReductionOp::ArgMax, 100, 10, DataType::FP32);
    refuses(site);
    refuses(argSite(DataType::FP32, 3, ArgReductionOp::ArgMin, (uint64_t{1} << 32) / 3 + 1, 10));  // UINT32 index
}

TEST(RaggedSelector, TimesEveryLegalCandidateAndCachesPerSite) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    // The attempt counts are process-wide and clear() keeps them, so earlier tests that tuned ragged reductions
    // leave them non-zero: count from here.
    const uint64_t kr_before = cache.tuningAttempts(DenseReductionSelectionKind::RaggedKR);
    const uint64_t rk_before = cache.tuningAttempts(DenseReductionSelectionKind::RaggedRK);
    const auto kr_attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::RaggedKR) - kr_before; };
    const auto rk_attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::RaggedRK) - rk_before; };
    const auto attempts = [&] { return kr_attempts() + rk_attempts(); };

    const RaggedSite base = valueSite(DataType::FP32, 1, ReductionOp::Sum, 50000, 300);
    const RaggedSelection first = selectCachedRagged(base, stream);
    EXPECT_EQ(kr_attempts(), 1U);
    EXPECT_EQ(rk_attempts(), 0U);
    EXPECT_EQ(first.family, RaggedFamily::KR);
    EXPECT_EQ(first.candidate_count, 3U);
    EXPECT_EQ(first.winner_name, raggedSelectionShapeName(first));
    const std::vector<std::string> kr = candidateNames(RaggedFamily::KR);
    EXPECT_NE(std::find(kr.begin(), kr.end(), first.winner_name), kr.end());
    // The contest as measured, in list order. The winner is the first candidate in list order within RAGGED_TIE_MARGIN
    // of the least measured time.
    EXPECT_EQ(first.contest_names, kr);
    ASSERT_EQ(first.contest_us.size(), 3U);
    double best = 0.0;
    for (double us : first.contest_us)
        if (us > 0.0 && (best == 0.0 || us < best)) best = us;
    size_t expected_winner = 0;
    while (!(first.contest_us[expected_winner] > 0.0 &&
             first.contest_us[expected_winner] <= best * (1.0 + RAGGED_TIE_MARGIN)))
        ++expected_winner;
    EXPECT_EQ(first.winner_name, kr[expected_winner]);

    // Cached: again, and at every ambient depth (the depth is not part of the key).
    EXPECT_EQ(selectCachedRagged(base, stream).winner_name, first.winner_name);
    for (const ReductionTuningDepth depth :
         {ReductionTuningDepth::Quick, ReductionTuningDepth::Regular, ReductionTuningDepth::Ultra}) {
        ScopedReductionTuningDepth scope(depth);
        EXPECT_EQ(selectCachedRagged(base, stream).winner_name, first.winner_name);
    }
    EXPECT_EQ(attempts(), 1U);
    EXPECT_EQ(raggedSelectionKey(base, stream).tuning_depth, ReductionTuningDepth::Regular);

    // Every field of the site is its own entry.
    std::vector<RaggedSite> variants;
    for (int field = 0; field < 8; ++field) {
        RaggedSite v = base;
        switch (field) {
            case 0: v.value_op = ReductionOp::Max; break;
            case 1: v.input_dtype = DataType::FP16; break;
            case 2: v.offsets_dtype = DataType::UINT64; break;
            case 3: v.output_dtype = DataType::BF16; break;
            case 4: v.components = 2; break;
            case 5: v.capacity_rows = 50001; break;
            case 6: v.segments = 301; break;
            default: v = argSite(DataType::FP32, 1, ArgReductionOp::ArgMin, 50000, 300); break;
        }
        variants.push_back(v);
    }
    uint64_t expected = attempts();
    for (size_t i = 0; i < variants.size(); ++i) {
        SCOPED_TRACE("variant " + std::to_string(i));
        const RaggedSelection s = selectCachedRagged(variants[i], stream);
        EXPECT_EQ(attempts(), ++expected);
        EXPECT_EQ(s.candidate_count, 3U);
        EXPECT_FALSE(raggedSelectionKey(variants[i], stream) == raggedSelectionKey(base, stream));
        (void)selectCachedRagged(variants[i], stream);
        EXPECT_EQ(attempts(), expected);
    }

    // An RK site: kind RaggedRK, its three candidates.
    const RaggedSite rk = valueSite(DataType::FP16, 64, ReductionOp::Min, 20000, 500);
    const RaggedSelection r = selectCachedRagged(rk, stream);
    EXPECT_EQ(rk_attempts(), 1U);
    EXPECT_EQ(r.family, RaggedFamily::RK);
    EXPECT_EQ(r.candidate_count, 3U);
    const std::vector<std::string> rk_names = candidateNames(RaggedFamily::RK);
    EXPECT_NE(std::find(rk_names.begin(), rk_names.end(), r.winner_name), rk_names.end());

    // Nothing to time without rows or segments: the first candidate, not cached.
    const uint64_t before = attempts();
    const RaggedSelection empty =
        selectCachedRagged(valueSite(DataType::FP32, 4, ReductionOp::Sum, 1000, 0), stream);
    EXPECT_EQ(empty.family, RaggedFamily::RK);
    EXPECT_EQ(empty.candidate_count, 0U);
    EXPECT_EQ(empty.winner_name, "ragged_rk:ta_c0_e4");
    (void)selectCachedRagged(argSite(DataType::FP16, 3, ArgReductionOp::ArgMax, 0, 10), stream);
    EXPECT_EQ(attempts(), before);
    cache.clear();
}

TEST(RaggedSelector, WinnerComputesWhatTheFirstCandidateAndTheReferenceCompute) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    denseReductionSelectionCache().clear();
    std::vector<RaggedSite> sites;
    sites.push_back(valueSite(DataType::FP16, 3, ReductionOp::Sum, 30000, 700));  // KR
    {
        RaggedSite s = valueSite(DataType::FP32, 64, ReductionOp::Max, 6000, 300);  // RK
        s.offsets_dtype = DataType::UINT64;
        sites.push_back(s);
    }
    sites.push_back(valueSite(DataType::BF16, 9, ReductionOp::Mean, 20000, 900));  // RK, odd D
    sites.push_back(argSite(DataType::FP32, 1, ArgReductionOp::ArgMin, 40000, 1200));  // KR D 1
    sites.push_back(argSite(DataType::FP8_E4M3, 5, ArgReductionOp::ArgMax, 30000, 400));  // KR components
    {
        RaggedSite s = argSite(DataType::BF16, 17, ArgReductionOp::ArgMin, 9000, 250, DataType::UINT64);  // RK
        sites.push_back(s);
    }
    sites.push_back(argSite(DataType::FP32, 300, ArgReductionOp::ArgMax, 1500, 60));  // RK, wide pair finisher

    for (size_t i = 0; i < sites.size(); ++i) {
        const RaggedSite& site = sites[i];
        SCOPED_TRACE("site " + std::to_string(i) + " D " + std::to_string(site.components));
        const RaggedSelection selection = selectCachedRagged(site, stream);
        EXPECT_EQ(selection.candidate_count, 3U);
        Problem p = makeProblem(site, 0x1322 + i, stream);
        const std::vector<uint64_t> got = runSelection(selection, p, stream);
        const std::vector<uint64_t> first =
            runSelection(raggedCandidateSelections(selection.family).front(), p, stream);
        EXPECT_EQ(got, first);
        if (site.arg || site.value_op != ReductionOp::Mean) {
            const std::vector<uint64_t> expected = reference(p);
            ASSERT_EQ(got.size(), expected.size());
            for (size_t k = 0; k < got.size(); ++k) ASSERT_EQ(got[k], expected[k]) << "output " << k;
        }
    }
    denseReductionSelectionCache().clear();
}

// Ragged RK takes rows up to 2 GiB (RAGGED_RK_MAX_ROW_BYTES), FP8 / FP16 within the finisher's 2^30 components. Past
// the CTA budget's K tiles the launch is one path range of one CTA per K tile, and the workspace is that range's
// records (no GPU needed).
TEST(RaggedSelector, RkTakesRowsUpTo2GiB) {
    EXPECT_EQ(RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES, uint64_t{1} << 31);
    for (const bool arg : {false, true}) {
        SCOPED_TRACE(arg ? "ARG" : "VALUE");
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP16, (uint64_t{1} << 25) + 8), RaggedFamily::RK);  // 64 MiB + 16
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP16, uint64_t{1} << 30), RaggedFamily::RK);        // 2 GiB
        EXPECT_FALSE(raggedFamilyFor(arg, DataType::FP16, (uint64_t{1} << 30) + 1));                 // the finisher
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP32, uint64_t{1} << 29), RaggedFamily::RK);        // 2 GiB
        EXPECT_FALSE(raggedFamilyFor(arg, DataType::FP32, (uint64_t{1} << 29) + 1));                 // past 2 GiB
        EXPECT_EQ(raggedFamilyFor(arg, DataType::FP8_E4M3, uint64_t{1} << 30), RaggedFamily::RK);    // 1 GiB
        EXPECT_FALSE(raggedFamilyFor(arg, DataType::FP8_E4M3, (uint64_t{1} << 30) + 1));
    }
    struct Row {
        DataType dtype;
        uint64_t d;
    };
    for (const Row& row : {Row{DataType::FP32, (uint64_t{1} << 24) + 4}, Row{DataType::FP16, uint64_t{1} << 29},
                           Row{DataType::FP32, uint64_t{1} << 29}}) {
        const uint64_t bytes = row.dtype == DataType::FP32 ? 4 : 2;
        SCOPED_TRACE("row bytes " + std::to_string(row.d * bytes));
        for (const RaggedRKShape& shape : raggedRKCandidateShapes()) {
            const RaggedRKLaunch v = raggedRKLaunch(shape, 2720, row.dtype, row.d, 3);
            EXPECT_EQ(v.ranges, 1U);
            EXPECT_EQ(v.grid, v.geometry.ktiles);
            EXPECT_GT(v.geometry.ktiles, 2720U);
            EXPECT_EQ(v.workspace_bytes, RaggedMergePath::raggedRecordLayout(1, static_cast<uint32_t>(row.d)).bytes);
            const RaggedRKLaunch a = raggedRKArgLaunch(shape, 2720, row.dtype, row.d, 3);
            EXPECT_EQ(a.ranges, 1U);
            EXPECT_EQ(a.grid, a.geometry.ktiles);
        }
    }
    EXPECT_THROW(static_cast<void>(raggedRKLaunch(RaggedRKShape{}, 2720, DataType::FP32, (uint64_t{1} << 29) + 4, 3)),
                 std::invalid_argument);
}
