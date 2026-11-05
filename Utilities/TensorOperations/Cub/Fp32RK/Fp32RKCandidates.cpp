#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal::Fp32RK {
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
bool partialBytes(const Fp32RKSite& site, uint64_t shards, uint64_t& bytes) {
    return checkedMul(site.outer, shards, bytes) && checkedMul(bytes, site.k, bytes) && checkedMul(bytes, 4, bytes);
}

PhysicalKernel fp32Kernel(PhysicalFamily family) {
    PhysicalKernel kernel{};
    kernel.family = family;
    kernel.element_bytes = 4;
    return kernel;
}

std::vector<PhysicalKernel> oneWindowKernels(const Fp32RKSite& site) {
    std::vector<PhysicalKernel> kernels;
    // The kernel's offsets are 32-bit byte offsets (fp32OneWindowFitsIndexing).
    uint64_t elements = 0;
    if (!checkedMul(site.outer, site.r, elements) || !checkedMul(elements, site.k, elements) ||
        elements > std::numeric_limits<uint32_t>::max() / 4)
        return kernels;
    for (uint32_t outputs : {1u, 2u, 4u, 8u}) {
        for (uint32_t warps : {1u, 2u, 4u, 8u}) {
            if ((outputs > 1 && warps > 1) || warps > site.r) continue;
            // The window holds (32 / O) * 4 components less a 0..3 prefix (any aligned K fills it).
            const uint64_t capacity = (32 / outputs) * 4 - (site.k % 4 == 0 ? 0 : 3);
            if (site.k > capacity) continue;
            PhysicalKernel kernel = fp32Kernel(PhysicalFamily::OneWindow);
            kernel.a = outputs;
            kernel.b = warps;
            kernel.c = 16;
            kernel.first_stage_ctas = ceilDiv(site.outer, (8 / warps) * outputs);
            kernels.push_back(kernel);
        }
    }
    return kernels;
}

std::vector<PhysicalKernel> deepRSmallKKernels(const Fp32RKSite& site) {
    std::vector<PhysicalKernel> kernels;
    for (uint64_t rows : {64u, 128u, 256u, 512u, 1024u}) {
        if (!fp32DeepRSmallKSupports(site.outer, site.r, site.k, rows)) continue;
        const uint64_t shards = ceilDiv(site.r, rows);
        PhysicalKernel kernel = fp32Kernel(PhysicalFamily::DeepRSmallK);
        kernel.a = rows;
        if (!checkedMul(site.outer, shards, kernel.first_stage_ctas) ||
            !partialBytes(site, shards, kernel.first_stage_scratch_bytes))
            continue;
        kernels.push_back(kernel);
    }
    return kernels;
}

// AlignedK: 16- then 8-byte packets, CTA 32 / 64 / 128, Complete then FP8's two staged shard targets (32 and 128 rows).
// 1228: and the 4-byte packet at odd K, where no wider packet is legal (the retired KParallel's scalar-packet regime;
// at even K the 8-byte packet does the same work in half the loads).
std::vector<PhysicalKernel> alignedKKernels(const Fp32RKSite& site) {
    std::vector<PhysicalKernel> kernels;
    std::vector<uint64_t> shard_counts{1};
    for (uint64_t target : {32u, 128u}) {
        if (site.r > target) {
            const uint64_t shards = ceilDiv(site.r, target);
            if (std::find(shard_counts.begin(), shard_counts.end(), shards) == shard_counts.end())
                shard_counts.push_back(shards);
        }
    }
    for (uint32_t packet : {16u, 8u, 4u}) {
        if (packet == 4 && site.k % 2 == 0) continue;
        for (uint32_t threads : {32u, 64u, 128u}) {
            for (uint64_t shards : shard_counts) {
                if (!fp32AlignedKSupports(site.outer, site.r, site.k, packet, threads, shards)) continue;
                PhysicalKernel kernel = fp32Kernel(PhysicalFamily::AlignedK);
                kernel.a = packet;
                kernel.b = threads;
                kernel.c = shards;
                const uint64_t tiles = ceilDiv(site.k, uint64_t{packet / 4} * threads);
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

// RuntimeK: the shared proposal generator at element size 4 (PacketComplete at the one- and two-wave tiles, rotated
// Prefix at every retained depth), within the FP32 launchers' limits.
std::vector<PhysicalKernel> runtimeKKernels(const Fp32RKSite& site, const Fp32RKResidency& residency) {
    std::vector<PhysicalKernel> kernels;
    for (const PhysicalKernel& kernel : Fp8RK::enumerateRetainedRuntimeK(
             {site.outer, site.r, site.k, site.sm_count, 4}, [&](const PhysicalKernel& p) { return residency(p); })) {
        const bool packet = static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::PacketComplete;
        const bool supported =
            packet ? fp32PacketCompleteSupports(site.outer, site.r, site.k, static_cast<uint32_t>(kernel.b),
                                                static_cast<uint32_t>(kernel.c))
                   : fp32RuntimeKRotatedSupports(site.outer, site.r, site.k, kernel.e);
        if (supported) kernels.push_back(kernel);
    }
    return kernels;
}
}  // namespace

uint32_t fp32RKCompiledResidency(const PhysicalKernel& kernel, uint64_t k, const CubReductionStageSemantics& semantics) {
    if (kernel.element_bytes != 4) return 0;
    try {
        switch (kernel.family) {
            case PhysicalFamily::OneWindow:
                return Fp8RK::oneWindowResidency(DataType::FP32, k, semantics.combine, static_cast<uint32_t>(kernel.a),
                                                 static_cast<uint32_t>(kernel.b), static_cast<uint32_t>(kernel.c));
            case PhysicalFamily::DeepRSmallK:
                return Fp8RK::deepRSmallKResidency(DataType::FP32, k, semantics, kernel.a);
            case PhysicalFamily::AlignedK:
                // FP32 AlignedK's occupancy does not depend on the output dtype.
                return Fp8RK::alignedKResidency(DataType::FP32, DataType::FP32, semantics,
                                                static_cast<uint32_t>(kernel.a), static_cast<uint32_t>(kernel.b),
                                                kernel.c);
            case PhysicalFamily::RuntimeK:
                return Fp8RK::runtimeKResidency(DataType::FP32, semantics, kernel);
        }
    } catch (const std::logic_error&) {
        return 0;
    }
    return 0;
}

std::vector<Fp32RKPoolEntry> fp32RKCandidatePool(const Fp32RKSite& site, const Fp32RKResidency& residency) {
    std::vector<Fp32RKPoolEntry> pool;
    if (!site.outer || !site.r || !site.k || !site.sm_count) return pool;
    if (!residency) throw std::invalid_argument("FP32 RK candidate pool requires a residency callback");
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
                throw std::logic_error("FP32 RK candidate generator built an invalid physical identity");
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

std::array<float, Fp32RKRankerModel::FEATURE_COUNT> fp32RKRankerFeatures(const Fp32RKPoolEntry& entry,
                                                                         const Fp32RKSite& site, uint32_t aggregate) {
    static_assert(Fp32RKRankerModel::FEATURE_COUNT == 26);
    const PhysicalKernel& p = entry.kernel;
    double packet = 0, threads = 0, shards = 0, warps = 0, tile = 0, out = 0, rows = 0, mode = 0;
    double packet_components = 0;
    switch (p.family) {
        case PhysicalFamily::OneWindow:
            out = static_cast<double>(p.a);
            warps = static_cast<double>(p.b);
            packet = static_cast<double>(p.c);
            packet_components = packet / 4;
            threads = warps * 32;
            mode = 1;
            break;
        case PhysicalFamily::DeepRSmallK:
            packet = 16;
            packet_components = 4;
            rows = static_cast<double>(p.a);
            shards = static_cast<double>(ceilDiv(site.r, p.a));
            mode = 2;
            break;
        case PhysicalFamily::AlignedK:
            packet = static_cast<double>(p.a);
            packet_components = packet / 4;
            threads = static_cast<double>(p.b);
            shards = static_cast<double>(p.c);
            rows = static_cast<double>(ceilDiv(site.r, p.c));
            tile = static_cast<double>(ceilDiv(site.k, (p.a / 4) * p.b));
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
            0.0f,  // FP8's E4M3 slot
            float(static_cast<unsigned>(site.semantics.combine)),
            float(static_cast<unsigned>(site.semantics.input_transform)),
            float(static_cast<unsigned>(p.family)), float(entry.local_rank),
            lg(packet + 1), lg(threads + 1), lg(shards + 1), lg(warps + 1),
            lg(tile + 1), lg(out + 1), lg(rows + 1), static_cast<float>(mode),
            lg(o * shards + 1), lg(o * shards * std::max(tile, 1.0) + 1),
            lg(o * shards * kk * 4 + 1), lg(o * rr * kk * 4 / 1048576.0 + 1),
            lg(rr / std::max(rows, 1.0) + 1),
            lg(kk / std::max(packet_components * std::max(threads, 1.0), 1.0) + 1),
            float(aggregate)};
}

size_t fp32RKShortlistLength(CubReductionTuningDepth depth) {
    if (!fp32RKRankerTrained()) return Fp8RK::fp8RKTuningPlan(depth).candidates;
    switch (depth) {
        case CubReductionTuningDepth::Quick: return FP32_RK_SHORTLIST_DEPTHS[0];
        case CubReductionTuningDepth::Regular: return FP32_RK_SHORTLIST_DEPTHS[1];
        case CubReductionTuningDepth::Ultra: return FP32_RK_SHORTLIST_DEPTHS[2];
    }
    return FP32_RK_SHORTLIST_DEPTHS[1];
}

std::vector<PhysicalKernel> fp32RKRankedKernels(const std::vector<Fp32RKPoolEntry>& pool, const Fp32RKSite& site,
                                                size_t count, uint32_t aggregate) {
    std::vector<std::pair<float, const Fp32RKPoolEntry*>> scored;
    scored.reserve(pool.size());
    for (const Fp32RKPoolEntry& entry : pool) {
        const float margin = fp32RKRankerTrained()
                                 ? Fp32RKRankerModel::margin(fp32RKRankerFeatures(entry, site, aggregate))
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

uint32_t fp32RKPassInputAlignment(const Fp8RK::PhysicalKernel& kernel) {
    switch (kernel.family) {
        case PhysicalFamily::AlignedK: return static_cast<uint32_t>(kernel.a);
        case PhysicalFamily::RuntimeK:
            return static_cast<RuntimeLayout>(kernel.a) == RuntimeLayout::RotatedPrefix ? 4U : 16U;
        default: return 16U;
    }
}

}  // namespace ThorImplementation::CubReductionInternal::Fp32RK
