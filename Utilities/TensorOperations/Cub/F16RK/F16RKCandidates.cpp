#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal::F16RK {
namespace {
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;
using Fp8RK::RuntimeLayout;

constexpr uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }
bool checkedMul(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}
// FP32 partial bytes of `shards` per output, or false on overflow.
bool partialBytes(const F16RKSite& site, uint64_t shards, uint64_t& bytes) {
    return checkedMul(site.outer, shards, bytes) && checkedMul(bytes, site.k, bytes) && checkedMul(bytes, 4, bytes);
}

PhysicalKernel f16Kernel(PhysicalFamily family) {
    PhysicalKernel kernel{};
    kernel.family = family;
    kernel.element_bytes = 2;
    return kernel;
}

std::vector<PhysicalKernel> oneWindowKernels(const F16RKSite& site) {
    std::vector<PhysicalKernel> kernels;
    // The kernel's offsets are 32-bit byte offsets (f16OneWindowFitsIndexing).
    uint64_t elements = 0;
    if (!checkedMul(site.outer, site.r, elements) || !checkedMul(elements, site.k, elements) ||
        elements > std::numeric_limits<uint32_t>::max() / 2)
        return kernels;
    for (uint32_t outputs : {1u, 2u, 4u, 8u}) {
        for (uint32_t warps : {1u, 2u, 4u, 8u}) {
            if ((outputs > 1 && warps > 1) || warps > site.r) continue;
            // The window holds (32 / O) * 8 components less a 0..7 prefix (any aligned K fills it). The residency
            // query excludes an ownership without a 2-byte unit at K (O4 below K 9, O8 above K 16).
            const uint64_t capacity = (32 / outputs) * 8 - (site.k % 8 == 0 ? 0 : 7);
            if (site.k > capacity) continue;
            PhysicalKernel kernel = f16Kernel(PhysicalFamily::OneWindow);
            kernel.a = outputs;
            kernel.b = warps;
            kernel.c = 16;
            kernel.first_stage_ctas = ceilDiv(site.outer, (8 / warps) * outputs);
            kernels.push_back(kernel);
        }
    }
    return kernels;
}

std::vector<PhysicalKernel> deepRSmallKKernels(const F16RKSite& site) {
    std::vector<PhysicalKernel> kernels;
    for (uint64_t rows : {64u, 128u, 256u, 512u, 1024u}) {
        if (!f16DeepRSmallKSupports(site.outer, site.r, site.k, rows)) continue;
        const uint64_t shards = ceilDiv(site.r, rows);
        PhysicalKernel kernel = f16Kernel(PhysicalFamily::DeepRSmallK);
        kernel.a = rows;
        if (!checkedMul(site.outer, shards, kernel.first_stage_ctas) ||
            !partialBytes(site, shards, kernel.first_stage_scratch_bytes))
            continue;
        kernels.push_back(kernel);
    }
    return kernels;
}

// AlignedK: 16-, 8- then 4-byte packets, CTA 32 / 64 / 128, Complete then FP8's two staged shard targets (32 and 128
// rows); and the 2-byte packet at odd K, where no wider packet is legal (FP32's 4-byte rule, 1228).
std::vector<PhysicalKernel> alignedKKernels(const F16RKSite& site) {
    std::vector<PhysicalKernel> kernels;
    std::vector<uint64_t> shard_counts{1};
    for (uint64_t target : {32u, 128u}) {
        if (site.r > target) {
            const uint64_t shards = ceilDiv(site.r, target);
            if (std::find(shard_counts.begin(), shard_counts.end(), shards) == shard_counts.end())
                shard_counts.push_back(shards);
        }
    }
    for (uint32_t packet : {16u, 8u, 4u, 2u}) {
        if (packet == 2 && site.k % 2 == 0) continue;
        for (uint32_t threads : {32u, 64u, 128u}) {
            for (uint64_t shards : shard_counts) {
                if (!f16AlignedKSupports(site.outer, site.r, site.k, packet, threads, shards)) continue;
                PhysicalKernel kernel = f16Kernel(PhysicalFamily::AlignedK);
                kernel.a = packet;
                kernel.b = threads;
                kernel.c = shards;
                const uint64_t tiles = ceilDiv(site.k, uint64_t{packet / 2} * threads);
                if (!checkedMul(site.outer, shards, kernel.first_stage_ctas) ||
                    !checkedMul(kernel.first_stage_ctas, tiles, kernel.first_stage_ctas))
                    continue;
                if (shards > 1 && !partialBytes(site, shards, kernel.first_stage_scratch_bytes)) continue;
                kernels.push_back(kernel);
            }
        }
    }
    return kernels;
}

// RuntimeK: the shared proposal generator at element size 2 (PacketComplete at the one- and two-wave tiles, rotated
// Prefix at every retained depth), within the 2-byte launchers' limits.
std::vector<PhysicalKernel> runtimeKKernels(const F16RKSite& site, const F16RKResidency& residency) {
    std::vector<PhysicalKernel> kernels;
    for (const PhysicalKernel& kernel : Fp8RK::enumerateRetainedRuntimeK(
             {site.outer, site.r, site.k, site.sm_count, 2}, [&](const PhysicalKernel& p) { return residency(p); })) {
        const bool packet = static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::PacketComplete;
        const bool supported =
            packet ? f16PacketCompleteSupports(site.outer, site.r, site.k, static_cast<uint32_t>(kernel.b),
                                               static_cast<uint32_t>(kernel.c))
                   : f16RuntimeKRotatedSupports(site.outer, site.r, site.k, kernel.e);
        if (supported) kernels.push_back(kernel);
    }
    return kernels;
}
}  // namespace

uint32_t f16RKCompiledResidency(const PhysicalKernel& kernel, DataType input_dtype, uint64_t k,
                                const CubReductionStageSemantics& semantics) {
    if (kernel.element_bytes != 2 || (input_dtype != DataType::FP16 && input_dtype != DataType::BF16)) return 0;
    try {
        switch (kernel.family) {
            case PhysicalFamily::OneWindow:
                return Fp8RK::oneWindowResidency(input_dtype, k, semantics.combine, static_cast<uint32_t>(kernel.a),
                                                 static_cast<uint32_t>(kernel.b), static_cast<uint32_t>(kernel.c));
            case PhysicalFamily::DeepRSmallK:
                return Fp8RK::deepRSmallKResidency(input_dtype, k, semantics, kernel.a);
            case PhysicalFamily::AlignedK:
                // 2-byte AlignedK's occupancy does not depend on the output dtype.
                return Fp8RK::alignedKResidency(input_dtype, DataType::FP32, semantics,
                                                static_cast<uint32_t>(kernel.a), static_cast<uint32_t>(kernel.b),
                                                kernel.c);
            case PhysicalFamily::RuntimeK:
                return Fp8RK::runtimeKResidency(input_dtype, semantics, kernel);
        }
    } catch (const std::logic_error&) {
        return 0;
    }
    return 0;
}

std::vector<F16RKPoolEntry> f16RKCandidatePool(const F16RKSite& site, const F16RKResidency& residency) {
    std::vector<F16RKPoolEntry> pool;
    if (!site.outer || !site.r || !site.k || !site.sm_count) return pool;
    if (!residency) throw std::invalid_argument("FP16 / BF16 RK candidate pool requires a residency callback");
    if (site.input_dtype != DataType::FP16 && site.input_dtype != DataType::BF16)
        throw std::invalid_argument("FP16 / BF16 RK candidate pool requires FP16 or BF16 input");
    uint64_t elements = 0;
    if (!checkedMul(site.outer, site.r, elements) || !checkedMul(elements, site.k, elements)) return pool;

    // Each family's legal, resident, in-budget kernels in enumeration order.
    std::array<std::vector<PhysicalKernel>, 4> families{oneWindowKernels(site), deepRSmallKKernels(site),
                                                        alignedKKernels(site), {}};
    std::array<std::vector<PhysicalKernel>, 4> kept;
    for (size_t f = 0; f < 3; ++f) {
        for (PhysicalKernel kernel : families[f]) {
            if (kernel.first_stage_scratch_bytes > site.scratch_budget_bytes) continue;
            kernel.resident_ctas_per_sm = residency(kernel);
            if (!kernel.resident_ctas_per_sm) continue;
            if (!Fp8RK::structurallyValid(kernel, site.outer, site.r, site.k))
                throw std::logic_error("FP16 / BF16 RK candidate generator built an invalid physical identity");
            kept[f].push_back(kernel);
        }
    }
    for (const PhysicalKernel& kernel : runtimeKKernels(site, residency)) {
        if (kernel.first_stage_scratch_bytes <= site.scratch_budget_bytes) kept[3].push_back(kernel);
    }
    // Interleave by local rank.
    size_t longest = 0;
    for (const auto& family : kept) longest = std::max(longest, family.size());
    for (size_t rank = 0; rank < longest; ++rank) {
        for (const auto& family : kept) {
            if (rank < family.size()) pool.push_back({family[rank], static_cast<uint32_t>(rank + 1)});
        }
    }
    return pool;
}

std::array<float, F16RKRankerModel::FEATURE_COUNT> f16RKRankerFeatures(const F16RKPoolEntry& entry,
                                                                       const F16RKSite& site, uint32_t aggregate) {
    static_assert(F16RKRankerModel::FEATURE_COUNT == 26);
    const PhysicalKernel& p = entry.kernel;
    double packet = 0, threads = 0, shards = 0, warps = 0, tile = 0, out = 0, rows = 0, mode = 0;
    double packet_components = 0;
    switch (p.family) {
        case PhysicalFamily::OneWindow:
            out = static_cast<double>(p.a);
            warps = static_cast<double>(p.b);
            packet = static_cast<double>(p.c);
            packet_components = packet / 2;
            threads = warps * 32;
            mode = 1;
            break;
        case PhysicalFamily::DeepRSmallK:
            packet = 16;
            packet_components = 8;
            rows = static_cast<double>(p.a);
            shards = static_cast<double>(ceilDiv(site.r, p.a));
            mode = 2;
            break;
        case PhysicalFamily::AlignedK:
            packet = static_cast<double>(p.a);
            packet_components = packet / 2;
            threads = static_cast<double>(p.b);
            shards = static_cast<double>(p.c);
            rows = static_cast<double>(ceilDiv(site.r, p.c));
            tile = static_cast<double>(ceilDiv(site.k, (p.a / 2) * p.b));
            mode = 3;
            break;
        case PhysicalFamily::RuntimeK:
            warps = static_cast<double>(p.b);
            threads = warps * 32;
            tile = static_cast<double>(p.d);
            rows = static_cast<double>(p.e);
            shards = static_cast<double>(ceilDiv(site.r, p.e));
            mode = static_cast<RuntimeLayout>(p.a) == RuntimeLayout::PacketComplete ? 4 : 7;
            break;
    }
    const double o = static_cast<double>(site.outer), rr = static_cast<double>(site.r);
    const double kk = static_cast<double>(site.k);
    const auto lg = [](double v) -> float { return static_cast<float>(std::log2(v)); };
    return {lg(o), lg(rr), lg(kk), lg(o * rr * kk), lg(o * kk), lg(rr / kk),
            site.input_dtype == DataType::BF16 ? 1.0f : 0.0f,  // FP8's E4M3 slot: the 2-byte format
            float(static_cast<unsigned>(site.semantics.combine)),
            float(static_cast<unsigned>(site.semantics.input_transform)),
            float(static_cast<unsigned>(p.family)), float(entry.local_rank),
            lg(packet + 1), lg(threads + 1), lg(shards + 1), lg(warps + 1),
            lg(tile + 1), lg(out + 1), lg(rows + 1), static_cast<float>(mode),
            lg(o * shards + 1), lg(o * shards * std::max(tile, 1.0) + 1),
            lg(o * shards * kk * 4 + 1), lg(o * rr * kk * 2 / 1048576.0 + 1),
            lg(rr / std::max(rows, 1.0) + 1),
            lg(kk / std::max(packet_components * std::max(threads, 1.0), 1.0) + 1),
            float(aggregate)};
}

size_t f16RKShortlistLength(CubReductionTuningDepth depth) {
    if (!f16RKRankerTrained()) return Fp8RK::fp8RKTuningPlan(depth).candidates;
    switch (depth) {
        case CubReductionTuningDepth::Quick: return F16_RK_SHORTLIST_DEPTHS[0];
        case CubReductionTuningDepth::Regular: return F16_RK_SHORTLIST_DEPTHS[1];
        case CubReductionTuningDepth::Ultra: return F16_RK_SHORTLIST_DEPTHS[2];
    }
    return F16_RK_SHORTLIST_DEPTHS[1];
}

std::vector<PhysicalKernel> f16RKRankedKernels(const std::vector<F16RKPoolEntry>& pool, const F16RKSite& site,
                                               size_t count, uint32_t aggregate) {
    std::vector<std::pair<float, const F16RKPoolEntry*>> scored;
    scored.reserve(pool.size());
    for (const F16RKPoolEntry& entry : pool) {
        const float margin = f16RKRankerTrained()
                                 ? F16RKRankerModel::margin(f16RKRankerFeatures(entry, site, aggregate))
                                 : 0.0f;
        scored.emplace_back(margin, &entry);
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<PhysicalKernel> kernels;
    for (const auto& [margin, entry] : scored) {
        static_cast<void>(margin);
        if (kernels.size() == count) break;
        kernels.push_back(entry->kernel);
    }
    return kernels;
}

uint32_t f16RKPassInputAlignment(const PhysicalKernel& kernel) {
    switch (kernel.family) {
        case PhysicalFamily::AlignedK: return static_cast<uint32_t>(kernel.a);
        case PhysicalFamily::RuntimeK:
            return static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::RotatedPrefix ? 2U : 16U;
        default: return 16U;
    }
}

}  // namespace ThorImplementation::CubReductionInternal::F16RK
