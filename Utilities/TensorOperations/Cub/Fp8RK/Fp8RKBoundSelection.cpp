#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKBoundSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <variant>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
using namespace Fp8GeneralRParallelPolicy;

RankerSemantics typedSemantics(const BoundSelectionRequest& r) {
    RankerSemantics semantics{};
    switch (r.input_dtype) {
        case DataType::FP8_E4M3: semantics.format = Fp8Format::E4M3; break;
        case DataType::FP8_E5M2: semantics.format = Fp8Format::E5M2; break;
        default: throw std::invalid_argument("FP8 RK selector requires FP8 source dtype");
    }
    switch (r.semantics.combine) {
        case CubReductionStageCombine::Sum: semantics.combine = Combine::Sum; break;
        case CubReductionStageCombine::Product: semantics.combine = Combine::Product; break;
        case CubReductionStageCombine::Minimum: semantics.combine = Combine::Min; break;
        case CubReductionStageCombine::Maximum: semantics.combine = Combine::Max; break;
    }
    switch (r.semantics.input_transform) {
        case CubReductionStageInputTransform::Identity: semantics.transform = InputTransform::Identity; break;
        case CubReductionStageInputTransform::AbsoluteValue: semantics.transform = InputTransform::Abs; break;
        case CubReductionStageInputTransform::Square: semantics.transform = InputTransform::Square; break;
    }
    return semantics;
}

// Only unsupported geometry is an expected absence. CUDA runtime errors are
// NOT masked; they must fail stamping instead of silently changing the list.
template <typename Callback>
uint32_t supportedResidency(Callback&& callback) {
    try { return callback(); }
    catch (const std::logic_error&) { return 0; }
}

PhysicalKernel bind(const CandidatePlan& p) {
    PhysicalKernel kernel{};
    kernel.family = static_cast<PhysicalFamily>(p.selection.mode);
    kernel.resident_ctas_per_sm = p.resident_ctas_per_sm;
    kernel.first_stage_ctas = p.first_stage_work_groups;
    kernel.first_stage_scratch_bytes = p.scratch_bytes;
    switch (p.selection.mode) {
        case Mode::OneWindow: {
            const auto v = std::get<OneWindowConfiguration>(p.selection.configuration).variant;
            kernel.a = v.outputs_per_warp;
            kernel.b = v.warps_per_output;
            kernel.c = v.packet_bytes;
            break;
        }
        case Mode::DeepRSmallK:
            kernel.a = std::get<DeepRSmallKConfiguration>(p.selection.configuration).variant.rows_per_shard;
            break;
        case Mode::AlignedK: {
            const auto v = std::get<AlignedKConfiguration>(p.selection.configuration).variant;
            kernel.a = v.packet;
            kernel.b = v.threads;
            kernel.c = v.shards_per_output;
            break;
        }
        case Mode::RuntimeK: {
            const auto v = std::get<RuntimeKConfiguration>(p.selection.configuration).variant;
            switch (v.layout) {
                case Fp8RuntimeKPolicy::Layout::PacketComplete:
                    kernel.a = static_cast<uint64_t>(RuntimeLayout::PacketComplete); break;
                case Fp8RuntimeKPolicy::Layout::RotatedPrefix:
                    kernel.a = static_cast<uint64_t>(RuntimeLayout::RotatedPrefix); break;
                case Fp8RuntimeKPolicy::Layout::RotatedExact:
                    kernel.a = static_cast<uint64_t>(RuntimeLayout::RotatedExact); break;
                case Fp8RuntimeKPolicy::Layout::PacketStaged:
                    throw std::logic_error("Rejected RuntimeK PacketStaged appeared in legal shortlist");
            }
            kernel.b = v.warps_per_output;
            kernel.c = v.tile_components;
            kernel.d = v.tile_count;
            kernel.e = v.rows_per_shard;
            kernel.f = v.shards_per_output;
            break;
        }
    }
    return kernel;
}

Fp8RuntimeKPolicy::Layout policyLayout(RuntimeLayout layout) {
    switch (layout) {
        case RuntimeLayout::PacketComplete: return Fp8RuntimeKPolicy::Layout::PacketComplete;
        case RuntimeLayout::RotatedPrefix: return Fp8RuntimeKPolicy::Layout::RotatedPrefix;
        case RuntimeLayout::RotatedExact: return Fp8RuntimeKPolicy::Layout::RotatedExact;
    }
    throw std::invalid_argument("Rejected or invalid RuntimeK layout");
}
}

std::vector<PhysicalKernel> selectBoundFirstStages(
    const BoundSelectionRequest& request,
    std::span<const PhysicalKernel> registered_runtime_k) {
    if (!request.outer || !request.r || !request.k || !request.sm_count) return {};
    const RankerSemantics semantics = typedSemantics(request);
    const ProductionSelectionContext context{
        request.outer, request.r, request.k, request.sm_count, semantics,
        request.first_stage_scratch_budget_bytes};

    std::vector<Fp8RuntimeKPolicy::Candidate> runtime;
    runtime.reserve(registered_runtime_k.size());
    for (auto registered : registered_runtime_k) {
        if (registered.family != PhysicalFamily::RuntimeK ||
            !structurallyValid(registered, request.outer, request.r, request.k) ||
            registered.first_stage_scratch_bytes > request.first_stage_scratch_budget_bytes)
            continue;
        const uint32_t actual_resident = supportedResidency([&] {
            return runtimeKResidency(request.input_dtype, request.semantics, registered);
        });
        if (!actual_resident) continue;
        registered.resident_ctas_per_sm = actual_resident;
        const Fp8RuntimeKPolicy::Variant variant{
            policyLayout(static_cast<RuntimeLayout>(registered.a)),
            static_cast<uint32_t>(registered.b), static_cast<uint32_t>(registered.c),
            static_cast<uint32_t>(registered.d), registered.e, registered.f};
        // Complete metadata is recomputed from the validated compiled proposal,
        // not fabricated by the selector; legal() rejects inconsistent counts.
        Fp8RuntimeKPolicy::Candidate candidate{
            variant, registered.first_stage_ctas, registered.first_stage_scratch_bytes, actual_resident};
        if (Fp8RuntimeKPolicy::legal(candidate, request.outer, request.r,
                                    request.k, request.sm_count)) runtime.push_back(candidate);
    }
    const auto ranked = selectProductionCandidates(
        context,
        [&](Fp8OneWindowPolicy::Variant v) {
            return supportedResidency([&] {
                return oneWindowResidency(request.input_dtype, request.k, request.semantics.combine,
                                          v.outputs_per_warp, v.warps_per_output, v.packet_bytes);
            });
        },
        [&](Fp8DeepRSmallKPolicy::Variant v) {
            return supportedResidency([&] {
                return deepRSmallKResidency(request.input_dtype, request.k, request.semantics,
                                            v.rows_per_shard);
            });
        },
        [&](Fp8AlignedKPolicy::Variant v) {
            return supportedResidency([&] {
                return alignedKResidency(request.input_dtype,
                    v.shards_per_output == 1 ? request.complete_output_dtype : DataType::FP32,
                    request.semantics, v.packet, v.threads, v.shards_per_output);
            });
        },
        runtime);
    std::vector<PhysicalKernel> physical;
    physical.reserve(ranked.size());
    for (const auto& candidate : ranked) {
        auto kernel = bind(candidate);
        if (!structurallyValid(kernel, request.outer, request.r, request.k))
            throw std::logic_error("Ranked retained FP8 RK kernel has invalid physical binding");
        physical.push_back(kernel);
    }
    return physical;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
