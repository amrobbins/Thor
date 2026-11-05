#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCarriedArgCandidates.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp32RK {
namespace {
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;
using Fp8RK::RuntimeLayout;

constexpr uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }

// The FP32 OneWindow unit of ownership O at K's bucket (exactly one per (O, P16, bucket)).
std::optional<Fp8OneWindowRuntimeKUnit> oneWindowUnit(uint64_t k, uint64_t outputs_per_warp, uint64_t packet_bytes) {
    if (packet_bytes != 16) return std::nullopt;
    for (const Fp8OneWindowRuntimeKUnit& unit : fp32OneWindowRuntimeKUnits(k)) {
        if (unit.outputs_per_warp == outputs_per_warp) return unit;
    }
    return std::nullopt;
}

uint64_t passShards(const PhysicalKernel& identity, uint64_t r) {
    switch (identity.family) {
        case PhysicalFamily::DeepRSmallK: return ceilDiv(r, identity.a);
        case PhysicalFamily::AlignedK: return identity.c;
        case PhysicalFamily::RuntimeK: return ceilDiv(r, identity.e);
        default: return 1;
    }
}

PhysicalKernel rotatedIdentity(uint64_t r, uint64_t k, uint64_t depth) {
    PhysicalKernel identity{};
    identity.family = PhysicalFamily::RuntimeK;
    identity.element_bytes = 4;
    identity.a = static_cast<uint64_t>(RuntimeLayout::RotatedPrefix);
    identity.c = FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS;
    identity.d = ceilDiv(k, identity.c);
    identity.e = depth;
    identity.f = ceilDiv(r, depth);
    return identity;
}
}  // namespace

bool fp32RKCarriedArgPassLegal(const PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k) {
    if (identity.element_bytes != 4 || outer == 0 || r == 0 || k == 0) return false;
    switch (identity.family) {
        case PhysicalFamily::OneWindow: {
            const auto unit = oneWindowUnit(k, identity.a, identity.c);
            return unit && fp32OneWindowCarriedArgSupports(outer, r, k, *unit, static_cast<uint32_t>(identity.b));
        }
        case PhysicalFamily::DeepRSmallK:
            return fp32DeepRSmallKSupports(outer, r, k, identity.a);
        case PhysicalFamily::AlignedK:  // 1228: the 4-byte packet is VALUE only
            return identity.a != 4 && fp32AlignedKSupports(outer, r, k, static_cast<uint32_t>(identity.a),
                                                           static_cast<uint32_t>(identity.b), identity.c);
        case PhysicalFamily::RuntimeK:
            return static_cast<RuntimeLayout>(identity.a) == RuntimeLayout::RotatedPrefix && identity.e != 0 &&
                   fp32RuntimeKCarriedArgSupports(outer, r, k, identity.e);
        default:
            return false;
    }
}

uint64_t fp32RKCarriedArgScratchBytes(const PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k) {
    return fp32ArgWideScratchBytes(outer, passShards(identity, r), k);
}

std::vector<Fp32RKArgCandidate> fp32RKCarriedArgCandidates(uint64_t outer, uint64_t r, uint64_t k,
                                                           uint64_t scratch_budget_bytes) {
    std::vector<Fp32RKArgCandidate> result;
    std::array<uint32_t, 4> family_rank{};
    const auto add = [&](PhysicalKernel identity) {
        identity.element_bytes = 4;
        if (!fp32RKCarriedArgPassLegal(identity, outer, r, k)) return;
        const uint64_t scratch = fp32RKCarriedArgScratchBytes(identity, outer, r, k);
        if (scratch > scratch_budget_bytes) return;
        const auto family = static_cast<size_t>(identity.family);
        identity.first_stage_scratch_bytes = scratch;
        result.push_back({identity, ++family_rank[family], scratch});
    };
    if (outer == 0 || r == 0 || k == 0) return result;
    // OneWindow: O1 at W1 / W2 / W4 / W8, then O2 / O4 / O8 at W1 (the FP32 VALUE pool's order).
    for (uint32_t outputs : {1u, 2u, 4u, 8u}) {
        for (uint32_t warps : {1u, 2u, 4u, 8u}) {
            if ((outputs > 1 && warps > 1) || warps > r) continue;
            PhysicalKernel identity{};
            identity.family = PhysicalFamily::OneWindow;
            identity.a = outputs;
            identity.b = warps;
            identity.c = 16;
            add(identity);
        }
    }
    // DeepRSmallK: the retained depths below R.
    for (uint64_t depth : {64u, 128u, 256u, 512u, 1024u}) {
        PhysicalKernel identity{};
        identity.family = PhysicalFamily::DeepRSmallK;
        identity.a = depth;
        add(identity);
    }
    // AlignedK: 16- then 8-byte packets, CTA 32 / 64 / 128, one shard then the 32- and 128-row shard targets.
    std::vector<uint64_t> shard_counts{1};
    for (uint64_t target : {32u, 128u}) {
        if (r > target) {
            const uint64_t shards = ceilDiv(r, target);
            if (std::find(shard_counts.begin(), shard_counts.end(), shards) == shard_counts.end())
                shard_counts.push_back(shards);
        }
    }
    for (uint32_t packet : {16u, 8u}) {
        for (uint32_t threads : {32u, 64u, 128u}) {
            for (uint64_t shards : shard_counts) {
                PhysicalKernel identity{};
                identity.family = PhysicalFamily::AlignedK;
                identity.a = packet;
                identity.b = threads;
                identity.c = shards;
                add(identity);
            }
        }
    }
    // Rotated: the Complete pass at depth R, then the retained depths below R.
    add(rotatedIdentity(r, k, r));
    for (uint64_t depth : Fp8RK::Fp8RuntimeKPolicy::rotated_row_depths) {
        if (depth < r) add(rotatedIdentity(r, k, depth));
    }
    return result;
}

void launchFp32RKCarriedArgPass(const PhysicalKernel& identity, const Tensor& input, const Tensor& carried_index,
                                Tensor* value_output, Tensor* index_output, const CubReductionGeometry& geometry,
                                void* wide_scratch, bool maximum, uint64_t row_stride, Stream& stream) {
    const uint64_t outer = geometry.outer_size, r = geometry.reduction_size, k = geometry.inner_size;
    if (!fp32RKCarriedArgPassLegal(identity, outer, r, k))
        throw std::logic_error("FP32 RK carried ARG pass identity is not legal at this geometry.");
    if (fp32RKCarriedArgScratchBytes(identity, outer, r, k) != 0 && wide_scratch == nullptr)
        throw std::logic_error("FP32 RK carried ARG staged pass requires its wide scratch.");
    switch (identity.family) {
        case PhysicalFamily::OneWindow: {
            const Fp8OneWindowRuntimeKUnit unit = *oneWindowUnit(k, identity.a, identity.c);
            launchFp32OneWindowCarriedArg(input, carried_index, value_output, index_output, geometry, unit,
                                          static_cast<uint32_t>(identity.b), maximum, row_stride, stream);
            return;
        }
        case PhysicalFamily::DeepRSmallK:
            launchFp32DeepRSmallKCarriedArg(input, carried_index, value_output, index_output, geometry, identity.a,
                                            wide_scratch, maximum, row_stride, stream);
            return;
        case PhysicalFamily::AlignedK:
            launchFp32AlignedKCarriedArg(input, carried_index, value_output, index_output, geometry,
                                         static_cast<uint32_t>(identity.a), static_cast<uint32_t>(identity.b),
                                         identity.c, wide_scratch, maximum, row_stride, stream);
            return;
        case PhysicalFamily::RuntimeK:
            launchFp32RuntimeKCarriedArg(input, carried_index, value_output, index_output, geometry, identity.e,
                                         wide_scratch, maximum, row_stride, stream);
            return;
        default:
            throw std::logic_error("FP32 RK carried ARG has no pass for this family.");
    }
}

}  // namespace ThorImplementation::CubReductionInternal::Fp32RK
