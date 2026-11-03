#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal {
namespace {

[[nodiscard]] uint64_t multiplyChecked(uint64_t a, uint64_t b) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("Retained FP8 ARG geometry overflows uint64_t");
    return a * b;
}

}  // namespace

// 1142: the plain ARG twin of a VALUE identity. The 1128/1130 interim ARG-only rules (DeepR depth floor, rotated /
// OneWindow / AlignedK instead of PacketComplete) are gone: ARG now ranks and autotunes its own candidates.
static std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgTwinImpl(const Fp8RK::PhysicalKernel& value_kernel,
                                                                             DataType input_dtype,
                                                                             uint64_t outer,
                                                                             uint64_t r,
                                                                             uint64_t inner) {
    if (input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) return std::nullopt;
    if (value_kernel.family == Fp8RK::PhysicalFamily::DeepRSmallK) {
        // DeepRSmallK identity: a = rows per shard. The twin keeps the VALUE ownership by K and the depth.
        const uint64_t depth = value_kernel.a;
        if (!fp8DeepRSmallKArgSupports(outer, r, inner, depth)) return std::nullopt;
        DenseRetainedFp8ArgPhysicalPlan plan;
        plan.family = Fp8RK::PhysicalFamily::DeepRSmallK;
        plan.rows_per_shard = depth;
        plan.shards = (r + depth - 1) / depth;
        plan.input_dtype = input_dtype;
        plan.input_dimensions = {outer, r, inner};
        plan.outer_size = outer;
        plan.reduction_size = r;
        plan.inner_size = inner;
        return plan;
    }
    if (value_kernel.family == Fp8RK::PhysicalFamily::RuntimeK) {
        // RuntimeK identity: a = layout, b = warps per output (PacketComplete), c = tile components, e = rows per
        // shard. PacketComplete keeps W and takes fp8PacketCompleteArgTileComponents (1127: at most 256 components, the
        // fewest key words per row); rotated layouts keep the depth
        // (the ARG kernel has its own K tile).
        if (value_kernel.a == static_cast<uint64_t>(Fp8RK::RuntimeLayout::PacketComplete)) {
            const auto warps = static_cast<uint32_t>(value_kernel.b);
            const uint32_t tile = fp8PacketCompleteArgTileComponents(inner, static_cast<uint32_t>(value_kernel.c));
            if (!fp8PacketCompleteArgSupports(outer, r, inner, warps, tile)) return std::nullopt;
            DenseRetainedFp8ArgPhysicalPlan plan;
            plan.family = Fp8RK::PhysicalFamily::RuntimeK;
            plan.warps_per_output = warps;
            plan.tile_components = tile;
            plan.rows_per_shard = r;
            plan.shards = 1;
            plan.packet_bytes = 16;
            plan.input_dtype = input_dtype;
            plan.input_dimensions = {outer, r, inner};
            plan.outer_size = outer;
            plan.reduction_size = r;
            plan.inner_size = inner;
            return plan;
        }
        const uint64_t rows = value_kernel.e;
        if (!fp8RuntimeKArgSupports(outer, r, inner, rows, FP8_RUNTIME_K_ARG_PACKET_BYTES)) return std::nullopt;
        DenseRetainedFp8ArgPhysicalPlan plan;
        plan.family = Fp8RK::PhysicalFamily::RuntimeK;
        plan.rows_per_shard = rows;
        plan.shards = (r + rows - 1) / rows;
        plan.packet_bytes = FP8_RUNTIME_K_ARG_PACKET_BYTES;
        plan.input_dtype = input_dtype;
        plan.input_dimensions = {outer, r, inner};
        plan.outer_size = outer;
        plan.reduction_size = r;
        plan.inner_size = inner;
        return plan;
    }
    if (value_kernel.family == Fp8RK::PhysicalFamily::AlignedK) {
        // AlignedK identity: a = packet bytes, b = CTA threads, c = shards per output (1: Complete). The twin keeps all
        // three and VALUE's balanced shard rows.
        const auto packet = static_cast<uint32_t>(value_kernel.a);
        const auto threads = static_cast<uint32_t>(value_kernel.b);
        if (!fp8AlignedKArgSupports(outer, r, inner, packet, threads, value_kernel.c)) return std::nullopt;
        DenseRetainedFp8ArgPhysicalPlan plan;
        plan.family = Fp8RK::PhysicalFamily::AlignedK;
        plan.rows_per_shard = (r + value_kernel.c - 1) / value_kernel.c;
        plan.shards = value_kernel.c;
        plan.packet_bytes = packet;
        plan.cta_threads = threads;
        plan.input_dtype = input_dtype;
        plan.input_dimensions = {outer, r, inner};
        plan.outer_size = outer;
        plan.reduction_size = r;
        plan.inner_size = inner;
        return plan;
    }
    if (value_kernel.family != Fp8RK::PhysicalFamily::OneWindow) return std::nullopt;
    // OneWindow identity: a = outputs per warp, b = warps per output, c = packet bytes. Exactly one compiled unit has
    // that ownership and packet at K's bucket (oneWindowUnitForVariant); its ARG twin has the same shape.
    const auto o = static_cast<uint32_t>(value_kernel.a);
    const auto w = static_cast<uint32_t>(value_kernel.b);
    const auto p = static_cast<uint32_t>(value_kernel.c);
    for (const Fp8OneWindowRuntimeKUnit& unit : fp8OneWindowRuntimeKUnits(inner)) {
        if (unit.outputs_per_warp != o || unit.packet_bytes != p) continue;
        if (!fp8OneWindowArgSupports(outer, r, inner, unit, w)) return std::nullopt;
        DenseRetainedFp8ArgPhysicalPlan plan;
        plan.family = Fp8RK::PhysicalFamily::OneWindow;
        plan.unit = unit;
        plan.warps_per_output = w;
        plan.input_dtype = input_dtype;
        plan.input_dimensions = {outer, r, inner};
        plan.outer_size = outer;
        plan.reduction_size = r;
        plan.inner_size = inner;
        return plan;
    }
    return std::nullopt;
}

std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgRawTwin(const Fp8RK::PhysicalKernel& value_kernel,
                                                                     DataType input_dtype,
                                                                     uint64_t outer,
                                                                     uint64_t r,
                                                                     uint64_t inner) {
    return retainedFp8ArgTwinImpl(value_kernel, input_dtype, outer, r, inner);
}

std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgRotatedDeepPass(DataType input_dtype,
                                                                          uint64_t outer,
                                                                          uint64_t r,
                                                                          uint64_t inner) {
    if (input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) return std::nullopt;
    constexpr uint64_t FallbackRows = 1024;  // the deepest retained rotated depth
    const uint64_t rows = r < FallbackRows ? r : FallbackRows;
    if (!fp8RuntimeKArgSupports(outer, r, inner, rows, FP8_RUNTIME_K_ARG_PACKET_BYTES)) return std::nullopt;
    DenseRetainedFp8ArgPhysicalPlan plan;
    plan.family = Fp8RK::PhysicalFamily::RuntimeK;
    plan.rows_per_shard = rows;
    plan.shards = (r + rows - 1) / rows;
    plan.packet_bytes = FP8_RUNTIME_K_ARG_PACKET_BYTES;
    plan.input_dtype = input_dtype;
    plan.input_dimensions = {outer, r, inner};
    plan.outer_size = outer;
    plan.reduction_size = r;
    plan.inner_size = inner;
    return plan;
}

DenseReductionCandidate makeRetainedFp8ArgCandidateForPlan(const DenseReductionProblem& problem,
                                                           DenseReductionSite site,
                                                           DenseRetainedFp8ArgPhysicalPlan plan) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::OriginalInput ||
        classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK ||
        problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction)
        throw std::invalid_argument("Retained FP8 ARG physical pass requires an original FP8 ARG RK site");
    uint64_t outer = 1, inner = 1;
    const uint64_t r = problem.runs[site.run_index].extent;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) outer = multiplyChecked(outer, problem.runs[i].extent);
        if (i > site.run_index) inner = multiplyChecked(inner, problem.runs[i].extent);
    }
    if (plan.outer_size != outer || plan.reduction_size != r || plan.inner_size != inner)
        throw std::invalid_argument("Retained FP8 ARG plan does not match its site geometry");
    // Staged plans carry packed partials in scratch and finish in the same pass (Complete to the planner).
    plan.packed_partials = plan.shards > 1;

    DenseReductionCandidate candidate;
    candidate.family = DenseReducerFamily::RK;
    candidate.site = site;
    candidate.next_problem = advanceDenseReductionProblem(problem, site, 1);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.scratch_bytes = retainedFp8ArgScratchBytes(plan);
    candidate.physical_plan = std::make_shared<DenseRetainedFp8ArgPhysicalPlan>(std::move(plan));
    return candidate;
}

std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgPlanForIdentity(const Fp8RK::PhysicalKernel& identity,
                                                                            DataType input_dtype,
                                                                            uint64_t outer,
                                                                            uint64_t r,
                                                                            uint64_t inner) {
    if (input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2) return std::nullopt;
    DenseRetainedFp8ArgPhysicalPlan plan;
    plan.family = identity.family;
    plan.input_dtype = input_dtype;
    plan.input_dimensions = {outer, r, inner};
    plan.outer_size = outer;
    plan.reduction_size = r;
    plan.inner_size = inner;
    switch (identity.family) {
        case Fp8RK::PhysicalFamily::OneWindow: {
            // (O, W, P): exactly one compiled unit of that ownership and packet holds K (as the twin).
            const auto o = static_cast<uint32_t>(identity.a);
            const auto w = static_cast<uint32_t>(identity.b);
            const auto p = static_cast<uint32_t>(identity.c);
            for (const Fp8OneWindowRuntimeKUnit& unit : fp8OneWindowRuntimeKUnits(inner)) {
                if (unit.outputs_per_warp != o || unit.packet_bytes != p) continue;
                if (!fp8OneWindowArgSupports(outer, r, inner, unit, w)) return std::nullopt;
                plan.unit = unit;
                plan.warps_per_output = w;
                return plan;
            }
            return std::nullopt;
        }
        case Fp8RK::PhysicalFamily::DeepRSmallK:
            if (!fp8DeepRSmallKArgSupports(outer, r, inner, identity.a)) return std::nullopt;
            plan.rows_per_shard = identity.a;
            plan.shards = (r + identity.a - 1) / identity.a;
            break;
        case Fp8RK::PhysicalFamily::AlignedK:
            if (!fp8AlignedKArgSupports(outer, r, inner, static_cast<uint32_t>(identity.a),
                                        static_cast<uint32_t>(identity.b), identity.c))
                return std::nullopt;
            plan.packet_bytes = static_cast<uint32_t>(identity.a);
            plan.cta_threads = static_cast<uint32_t>(identity.b);
            plan.shards = identity.c;
            plan.rows_per_shard = (r + identity.c - 1) / identity.c;
            break;
        case Fp8RK::PhysicalFamily::RuntimeK:
            if (identity.a == static_cast<uint64_t>(Fp8RK::RuntimeLayout::PacketComplete)) {
                // Identity c is the ARG tile itself (retainedFp8ArgPhysicalIdentity).
                const auto warps = static_cast<uint32_t>(identity.b);
                const auto tile = static_cast<uint32_t>(identity.c);
                if (tile == 0 || !fp8PacketCompleteArgSupports(outer, r, inner, warps, tile)) return std::nullopt;
                plan.warps_per_output = warps;
                plan.tile_components = tile;
                plan.rows_per_shard = r;
                plan.packet_bytes = 16;
            } else {
                if (identity.e == 0 ||
                    !fp8RuntimeKArgSupports(outer, r, inner, identity.e, FP8_RUNTIME_K_ARG_PACKET_BYTES))
                    return std::nullopt;
                plan.rows_per_shard = identity.e;
                plan.shards = (r + identity.e - 1) / identity.e;
                plan.packet_bytes = FP8_RUNTIME_K_ARG_PACKET_BYTES;
            }
            break;
    }
    plan.packed_partials = plan.shards > 1;
    return plan;
}

uint64_t retainedFp8ArgScratchBytes(const DenseRetainedFp8ArgPhysicalPlan& plan) {
    if (plan.kr.group_lanes != 0) return Fp8KR::fp8KRScratchBytes(plan.kr, true);
    if (!plan.packed_partials) return 0;
    return multiplyChecked(multiplyChecked(multiplyChecked(plan.outer_size, plan.shards), plan.inner_size),
                           sizeof(uint32_t));
}

Fp8RK::PhysicalKernel retainedFp8ArgPhysicalIdentity(const DenseRetainedFp8ArgPhysicalPlan& plan) {
    Fp8RK::PhysicalKernel identity{};
    identity.family = plan.family;
    switch (plan.family) {
        case Fp8RK::PhysicalFamily::OneWindow:
            identity.a = plan.unit.outputs_per_warp;
            identity.b = plan.warps_per_output;
            identity.c = plan.unit.packet_bytes;
            break;
        case Fp8RK::PhysicalFamily::DeepRSmallK:
            identity.a = plan.rows_per_shard;
            break;
        case Fp8RK::PhysicalFamily::AlignedK:
            identity.a = plan.packet_bytes;
            identity.b = plan.cta_threads;
            identity.c = plan.shards;
            break;
        case Fp8RK::PhysicalFamily::RuntimeK:
            if (plan.tile_components != 0) {
                identity.a = static_cast<uint64_t>(Fp8RK::RuntimeLayout::PacketComplete);
                identity.b = plan.warps_per_output;
                identity.c = plan.tile_components;
            } else {
                // The rotated ARG kernel tiles K by its own packet width (249 components at 8-byte packets).
                identity.a = static_cast<uint64_t>(Fp8RK::RuntimeLayout::RotatedPrefix);
                identity.b = 0;
                identity.c = fp8RuntimeKArgTileComponents(plan.packet_bytes);
            }
            identity.d = (plan.inner_size + identity.c - 1) / identity.c;
            identity.e = plan.rows_per_shard;
            identity.f = plan.shards;
            break;
    }
    return identity;
}

namespace {
[[nodiscard]] bool sameArgPlan(const DenseRetainedFp8ArgPhysicalPlan& a, const DenseRetainedFp8ArgPhysicalPlan& b) {
    return a.family == b.family && a.unit == b.unit && a.warps_per_output == b.warps_per_output &&
           a.rows_per_shard == b.rows_per_shard && a.shards == b.shards && a.packet_bytes == b.packet_bytes &&
           a.cta_threads == b.cta_threads && a.tile_components == b.tile_components;
}
}  // namespace

std::vector<RetainedFp8ArgCandidate> retainedFp8ArgCandidates(std::span<const RetainedFp8ValueSource> value_candidates,
                                                              DataType input_dtype,
                                                              uint64_t outer,
                                                              uint64_t r,
                                                              uint64_t inner) {
    std::vector<RetainedFp8ArgCandidate> result;
    std::array<uint32_t, 4> family_count{};
    // 1142: twins are generated in a canonical source order (family, then local rank), so the local rank a
    // duplicated twin keeps (its first source's) does not depend on how the caller ordered the VALUE candidates.
    std::vector<RetainedFp8ValueSource> sources(value_candidates.begin(), value_candidates.end());
    std::stable_sort(sources.begin(), sources.end(), [](const RetainedFp8ValueSource& a, const RetainedFp8ValueSource& b) {
        return std::make_pair(static_cast<int>(a.kernel.family), a.local_rank) <
               std::make_pair(static_cast<int>(b.kernel.family), b.local_rank);
    });
    const auto add = [&](DenseRetainedFp8ArgPhysicalPlan plan, uint32_t source_local_rank) {
        plan.packed_partials = plan.shards > 1;
        for (const auto& existing : result)
            if (sameArgPlan(existing.plan, plan)) return;
        const auto family = static_cast<size_t>(plan.family);
        const uint32_t local_rank = source_local_rank != 0 ? source_local_rank : family_count[family] + 1;
        family_count[family] = std::max(family_count[family], local_rank);
        RetainedFp8ArgCandidate candidate;
        candidate.identity = retainedFp8ArgPhysicalIdentity(plan);
        candidate.local_rank = local_rank;
        candidate.plan = std::move(plan);
        result.push_back(std::move(candidate));
    };
    // Twins of every VALUE candidate, in the VALUE order, carrying the VALUE candidate's local rank.
    for (const RetainedFp8ValueSource& source : sources) {
        if (auto twin = retainedFp8ArgRawTwin(source.kernel, input_dtype, outer, r, inner)) add(std::move(*twin), source.local_rank);
    }
    // ARG-only passes no VALUE identity maps to: the single-shard rotated pass (R <= 1024) and the A5 fallback depth.
    if (r <= 1024 && fp8RuntimeKArgSupports(outer, r, inner, r, FP8_RUNTIME_K_ARG_PACKET_BYTES)) {
        DenseRetainedFp8ArgPhysicalPlan plan;
        plan.family = Fp8RK::PhysicalFamily::RuntimeK;
        plan.rows_per_shard = r;
        plan.shards = 1;
        plan.packet_bytes = FP8_RUNTIME_K_ARG_PACKET_BYTES;
        plan.input_dtype = input_dtype;
        plan.input_dimensions = {outer, r, inner};
        plan.outer_size = outer;
        plan.reduction_size = r;
        plan.inner_size = inner;
        add(std::move(plan), 0);
    }
    if (auto rotated = retainedFp8ArgRotatedDeepPass(input_dtype, outer, r, inner)) add(std::move(*rotated), 0);
    return result;
}

void launchRetainedFp8ArgPlan(const DenseRetainedFp8ArgPhysicalPlan& plan,
                              const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint32_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream) {
    if (plan.kr.group_lanes != 0) {
        // 1146: retained FP8 R / KR. The pass's geometry is the planner's direct R / KR geometry; the shape is authoritative.
        if (geometry.input_elements != plan.kr.rows * plan.kr.row_bytes)
            throw std::invalid_argument("Retained FP8 KR ARG pass geometry does not match its shape.");
        launchFp8KRArgSegments(input, value_output, index_output, plan.kr, packed_scratch,
                               retainedFp8ArgScratchBytes(plan), maximum, domain_stride, stream);
        return;
    }
    if (plan.packed_partials) {
        // ARG-A3g: packed staged pass = first stage into scratch + packed continuation into the requested outputs.
        if (packed_scratch == nullptr) throw std::invalid_argument("Packed FP8 ARG pass requires its scratch.");
        uint64_t uniform_rows = plan.rows_per_shard;
        if (plan.family == Fp8RK::PhysicalFamily::DeepRSmallK) {
            launchFp8DeepRSmallKArgFirstStagePacked(input, packed_scratch, geometry, plan.rows_per_shard, plan.shards,
                                                    maximum, stream);
        } else if (plan.family == Fp8RK::PhysicalFamily::RuntimeK) {
            launchFp8RuntimeKArgPass(input, nullptr, nullptr, geometry, plan.rows_per_shard, plan.packet_bytes, maximum,
                                     1, stream, packed_scratch);
        } else if (plan.family == Fp8RK::PhysicalFamily::AlignedK) {
            launchFp8AlignedKArgPass(input, nullptr, nullptr, geometry, plan.packet_bytes, plan.cta_threads, plan.shards,
                                     maximum, 1, stream, packed_scratch);
            uniform_rows = 0;  // balanced split
        } else {
            throw std::logic_error("Packed FP8 ARG partials are not defined for this family.");
        }
        launchFp8ArgPackedContinuation(packed_scratch, value_output, index_output, plan.input_dtype, geometry.outer_size,
                                       geometry.reduction_size, geometry.inner_size, plan.shards, uniform_rows,
                                       maximum, domain_stride, stream);
        return;
    }
    switch (plan.family) {
        case Fp8RK::PhysicalFamily::RuntimeK:
            if (plan.tile_components != 0) {
                launchFp8PacketCompleteArg(input, value_output, index_output, geometry, plan.warps_per_output,
                                           plan.tile_components, maximum, domain_stride, stream);
            } else {
                launchFp8RuntimeKArgPass(input, value_output, index_output, geometry, plan.rows_per_shard,
                                         plan.packet_bytes, maximum, domain_stride, stream);
            }
            return;
        case Fp8RK::PhysicalFamily::AlignedK:
            launchFp8AlignedKArgPass(input, value_output, index_output, geometry, plan.packet_bytes, plan.cta_threads,
                                     plan.shards, maximum, domain_stride, stream);
            return;
        case Fp8RK::PhysicalFamily::DeepRSmallK:
            if (value_output == nullptr || index_output == nullptr)
                throw std::invalid_argument("Unpacked FP8 DeepRSmallK ARG partials need value and index outputs.");
            launchFp8DeepRSmallKArgFirstStage(input, *value_output, *index_output, geometry, plan.rows_per_shard,
                                              plan.shards, maximum, domain_stride, stream);
            return;
        case Fp8RK::PhysicalFamily::OneWindow:
            launchFp8OneWindowArgComplete(input, value_output, index_output, geometry, plan.unit,
                                          plan.warps_per_output, maximum, domain_stride, stream);
            return;
    }
    throw std::logic_error("Unknown retained FP8 ARG family.");
}

}  // namespace ThorImplementation::CubReductionInternal
