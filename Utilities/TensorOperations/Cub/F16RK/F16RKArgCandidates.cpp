#include "Utilities/TensorOperations/Cub/F16RK/F16RKArgCandidates.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::F16RK {
namespace {
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;
using Fp8RK::RuntimeLayout;

constexpr uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }

// The rotated ARG pass's ARG-only depth limit, FP8 ARG's: single-shard passes up to 1024 rows, and 1024 as the
// fallback depth.
constexpr uint64_t ROTATED_ARG_ONLY_MAX_ROWS = 1024;

// The 2-byte OneWindow unit of ownership O at K's bucket (exactly one per (O, P16, bucket)).
std::optional<Fp8OneWindowRuntimeKUnit> oneWindowUnit(uint64_t k, uint64_t outputs_per_warp, uint64_t packet_bytes) {
    if (packet_bytes != 16) return std::nullopt;
    for (const Fp8OneWindowRuntimeKUnit& unit : f16OneWindowRuntimeKUnits(k)) {
        if (unit.outputs_per_warp == outputs_per_warp) return unit;
    }
    return std::nullopt;
}

bool isPacketComplete(const PhysicalKernel& identity) {
    return static_cast<RuntimeLayout>(identity.a) == RuntimeLayout::PacketComplete;
}

// Partials per output of the pass (1 for a Complete pass).
uint64_t passShards(const PhysicalKernel& identity, uint64_t r) {
    switch (identity.family) {
        case PhysicalFamily::OneWindow: return 1;
        case PhysicalFamily::DeepRSmallK: return ceilDiv(r, identity.a);
        case PhysicalFamily::AlignedK: return identity.c;
        case PhysicalFamily::RuntimeK: return isPacketComplete(identity) ? 1 : ceilDiv(r, identity.e);
    }
    return 1;
}

PhysicalKernel rotatedIdentity(uint64_t r, uint64_t k, uint64_t depth) {
    PhysicalKernel identity{};
    identity.family = PhysicalFamily::RuntimeK;
    identity.element_bytes = 2;
    identity.a = static_cast<uint64_t>(RuntimeLayout::RotatedPrefix);
    identity.c = F16_RUNTIME_K_ROTATED_TILE_COMPONENTS;
    identity.d = ceilDiv(k, identity.c);
    identity.e = depth;
    identity.f = ceilDiv(r, depth);
    return identity;
}

bool samePass(const PhysicalKernel& a, const PhysicalKernel& b) {
    return a.family == b.family && a.a == b.a && a.b == b.b && a.c == b.c && a.d == b.d && a.e == b.e && a.f == b.f;
}
}  // namespace

bool f16RKArgPassLegal(const PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k) {
    if (identity.element_bytes != 2 || outer == 0 || r == 0 || k == 0) return false;
    switch (identity.family) {
        case PhysicalFamily::OneWindow: {
            const auto unit = oneWindowUnit(k, identity.a, identity.c);
            return unit && f16OneWindowArgSupports(outer, r, k, *unit, static_cast<uint32_t>(identity.b));
        }
        case PhysicalFamily::DeepRSmallK:
            return f16DeepRSmallKArgSupports(outer, r, k, identity.a);
        case PhysicalFamily::AlignedK:
            return f16AlignedKArgSupports(outer, r, k, static_cast<uint32_t>(identity.a),
                                           static_cast<uint32_t>(identity.b), identity.c);
        case PhysicalFamily::RuntimeK:
            if (isPacketComplete(identity)) {
                return f16PacketCompleteArgSupports(outer, r, k, static_cast<uint32_t>(identity.b),
                                                     static_cast<uint32_t>(identity.c));
            }
            return static_cast<RuntimeLayout>(identity.a) == RuntimeLayout::RotatedPrefix && identity.e != 0 &&
                   f16RuntimeKArgSupports(outer, r, k, identity.e);
    }
    return false;
}

uint64_t f16RKArgScratchBytes(const PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k) {
    return fp32ArgPackedScratchBytes(outer, passShards(identity, r), k);
}

std::vector<F16RKArgCandidate> f16RKArgCandidates(const std::vector<F16RKPoolEntry>& value_pool, uint64_t outer,
                                                    uint64_t r, uint64_t k, uint64_t scratch_budget_bytes) {
    std::vector<F16RKArgCandidate> result;
    std::array<uint32_t, 4> family_rank{};
    const auto add = [&](PhysicalKernel identity, uint32_t source_local_rank) {
        identity.element_bytes = 2;
        if (!f16RKArgPassLegal(identity, outer, r, k)) return;
        for (const F16RKArgCandidate& existing : result) {
            if (samePass(existing.identity, identity)) return;
        }
        const uint64_t scratch = f16RKArgScratchBytes(identity, outer, r, k);
        if (scratch > scratch_budget_bytes) return;
        const auto family = static_cast<size_t>(identity.family);
        const uint32_t local_rank = source_local_rank != 0 ? source_local_rank : family_rank[family] + 1;
        family_rank[family] = std::max(family_rank[family], local_rank);
        // The ARG pass's own staging: the scratch is packed partials, not the VALUE stage's FP32 partials.
        identity.first_stage_scratch_bytes = scratch;
        result.push_back({identity, local_rank, scratch});
    };
    for (const F16RKPoolEntry& entry : value_pool) add(entry.kernel, entry.local_rank);
    if (r <= ROTATED_ARG_ONLY_MAX_ROWS) add(rotatedIdentity(r, k, r), 0);
    add(rotatedIdentity(r, k, std::min(r, ROTATED_ARG_ONLY_MAX_ROWS)), 0);
    return result;
}

std::vector<F16RKPoolEntry> f16RKArgRankerPool(const std::vector<F16RKArgCandidate>& candidates) {
    std::vector<F16RKPoolEntry> pool;
    pool.reserve(candidates.size());
    for (const F16RKArgCandidate& candidate : candidates) pool.push_back({candidate.identity, candidate.local_rank});
    return pool;
}

void launchF16RKArgPass(const PhysicalKernel& identity, const Tensor& input, Tensor* value_output,
                         Tensor* index_output, const CubReductionGeometry& geometry, uint64_t* packed_scratch,
                         bool maximum, uint64_t domain_stride, Stream& stream) {
    const uint64_t outer = geometry.outer_size, r = geometry.reduction_size, k = geometry.inner_size;
    if (!f16RKArgPassLegal(identity, outer, r, k))
        throw std::logic_error("FP16 / BF16 RK ARG pass identity is not legal at this geometry.");
    if (f16RKArgScratchBytes(identity, outer, r, k) != 0 && packed_scratch == nullptr)
        throw std::logic_error("FP16 / BF16 RK ARG staged pass requires its packed scratch.");
    switch (identity.family) {
        case PhysicalFamily::OneWindow: {
            const Fp8OneWindowRuntimeKUnit unit = *oneWindowUnit(k, identity.a, identity.c);
            launchF16OneWindowArg(input, value_output, index_output, geometry, unit,
                                   static_cast<uint32_t>(identity.b), maximum, domain_stride, stream);
            return;
        }
        case PhysicalFamily::DeepRSmallK:
            launchF16DeepRSmallKArg(input, value_output, index_output, geometry, identity.a, packed_scratch, maximum,
                                     domain_stride, stream);
            return;
        case PhysicalFamily::AlignedK:
            launchF16AlignedKArg(input, value_output, index_output, geometry, static_cast<uint32_t>(identity.a),
                                  static_cast<uint32_t>(identity.b), identity.c, packed_scratch, maximum,
                                  domain_stride, stream);
            return;
        case PhysicalFamily::RuntimeK:
            if (isPacketComplete(identity)) {
                launchF16PacketCompleteArg(input, value_output, index_output, geometry,
                                            static_cast<uint32_t>(identity.b), static_cast<uint32_t>(identity.c),
                                            maximum, domain_stride, stream);
            } else {
                launchF16RuntimeKArg(input, value_output, index_output, geometry, identity.e, packed_scratch, maximum,
                                      domain_stride, stream);
            }
            return;
    }
    throw std::logic_error("Unknown FP16 / BF16 RK ARG family.");
}

}  // namespace ThorImplementation::CubReductionInternal::F16RK
