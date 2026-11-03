#pragma once

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * ARG-A2/A3: exactly ONE original-input FP8 RK ARG physical pass. 1142: chosen by ARG's own ranked, autotuned
 * shortlist (Fp8RK::selectCachedArgRKSite) from retainedFp8ArgCandidates; each candidate is the ARG twin of a retained
 * VALUE MIN/MAX kernel or an ARG-only rotated pass. OneWindow twins are Complete: the nominated R is
 * eliminated and the pass writes the public outputs (final pass) or an FP32 value + carried index intermediate.
 * DeepRSmallK twins are staged: they write FP32 value + carried index partials [outer, shards, inner], leaving an
 * IntermediateReduction run of `shards` for the carried-index ARG reducer. RuntimeK and AlignedK twins are Complete
 * with one shard and staged otherwise (AlignedK keeps VALUE's balanced shard rows, packet and CTA size). The index written is row * the nominated
 * run's original ARG-domain stride.
 */
struct DenseRetainedFp8ArgPhysicalPlan final : DenseReductionPhysicalPlan {
    Fp8RK::PhysicalFamily family = Fp8RK::PhysicalFamily::OneWindow;
    Fp8OneWindowRuntimeKUnit unit{};  // OneWindow
    uint32_t warps_per_output = 1;    // OneWindow
    uint64_t rows_per_shard = 0;      // DeepRSmallK, RuntimeK (RuntimeK: R for the single-shard Complete twin)
    uint64_t shards = 1;              // partials per output (1 for Complete twins)
    uint32_t packet_bytes = 16;       // RuntimeK: the ARG rotated kernel's packet (8 or 16); AlignedK: VALUE's (2 or 8)
    uint32_t cta_threads = 0;         // AlignedK: VALUE's CTA size (32, 64 or 128)
    uint32_t tile_components = 0;     // RuntimeK PacketComplete twin (nonzero): ARG K tile; warps_per_output = VALUE's W
    // ARG-A3g (1116): a staged twin (shards > 1) writes 4-byte packed partials to its own scratch and finishes with
    // launchFp8ArgPackedContinuation in the same pass, so the planner sees one Complete pass with scratch.
    bool packed_partials = false;
    // 1146: a retained FP8 R / KR pass (launchFp8KRArgSegments) when kr.group_lanes != 0; family and the RK fields
    // above are then unused. input_dimensions are the site's current runs (the nominated run last), outer_size the
    // rows, reduction_size the row length and inner_size 1. A staged shape sets packed_partials: its 64-bit chunk
    // candidates live in the pass's scratch (retainedFp8ArgScratchBytes).
    Fp8KR::Fp8KRSegmentShape kr{};
    DataType input_dtype = DataType::FP8_E4M3;
    std::vector<uint64_t> input_dimensions;  // {outer, r, inner}
    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;
};

/** Scratch of a packed staged twin: one uint32 lane candidate per [outer, shards, inner] partial (0 if not packed). */
[[nodiscard]] uint64_t retainedFp8ArgScratchBytes(const DenseRetainedFp8ArgPhysicalPlan& plan);

/** RuntimeK rotated ARG twins run at this packet width (lanes own at most 8 components, the 1105 rule; 1110 measured
 *  the 16-byte packet 1.56x slower, and 1112 stopped compiling it). */
inline constexpr uint32_t FP8_RUNTIME_K_ARG_PACKET_BYTES = 8;

/** The ARG twin of `value_kernel` at [outer, r, inner], when one exists and is legal (the candidate generator for
 *  ranked ARG selection). */
[[nodiscard]] std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgRawTwin(const Fp8RK::PhysicalKernel& value_kernel,
                                                                                  DataType input_dtype,
                                                                                  uint64_t outer,
                                                                                  uint64_t r,
                                                                                  uint64_t inner);

/** A5: the rotated RuntimeK ARG pass (8-byte packets, depth min(R, 1024)), an ARG-only candidate in every candidate
 *  set, so a site whose VALUE identities have no legal twin is still covered; nullopt only beyond the rotated kernel's
 *  own limits (grid extents, 2^23-row shards). 1172: was retainedFp8ArgFallbackTwin. */
[[nodiscard]] std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgRotatedDeepPass(DataType input_dtype,
                                                                                       uint64_t outer,
                                                                                       uint64_t r,
                                                                                       uint64_t inner);

/** 1133: the physical identity (PhysicalKernel fields) of an ARG pass, for ranker features: OneWindow (O, W, P),
 *  DeepRSmallK depth, AlignedK (packet, CTA, shards), RuntimeK PacketComplete (W, ARG tile) or rotated
 *  (RotatedPrefix, the ARG kernel's tile, depth, shards). */
[[nodiscard]] Fp8RK::PhysicalKernel retainedFp8ArgPhysicalIdentity(const DenseRetainedFp8ArgPhysicalPlan& plan);

/** A VALUE candidate and its 1-based local rank in the mode-local VALUE order. */
struct RetainedFp8ValueSource {
    Fp8RK::PhysicalKernel kernel;
    uint32_t local_rank = 0;
};

/** A legal ARG pass with its ranker identity and local rank (packed_partials set). */
struct RetainedFp8ArgCandidate {
    DenseRetainedFp8ArgPhysicalPlan plan;
    Fp8RK::PhysicalKernel identity;
    uint32_t local_rank = 0;
};

/**
 * 1133: every distinct legal ARG pass at [outer, r, inner]: the raw twins of `value_candidates` (VALUE MIN/MAX
 * candidates, in their order and with their local ranks), then the single-shard rotated pass (R <= 1024) and the A5
 * fallback depth, which no VALUE identity maps to. Duplicates keep the first occurrence.
 */
[[nodiscard]] std::vector<RetainedFp8ArgCandidate> retainedFp8ArgCandidates(
    std::span<const RetainedFp8ValueSource> value_candidates, DataType input_dtype, uint64_t outer, uint64_t r,
    uint64_t inner);

/** Launches one retained ARG pass (packed staged: first stage into `packed_scratch`, then the packed continuation). */
void launchRetainedFp8ArgPlan(const DenseRetainedFp8ArgPhysicalPlan& plan,
                              const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint32_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream);

/** 1142: the ARG pass an identity names (the inverse of retainedFp8ArgPhysicalIdentity; packed_partials set), when
 *  it is legal at [outer, r, inner]. Selections cache only the identity. */
[[nodiscard]] std::optional<DenseRetainedFp8ArgPhysicalPlan> retainedFp8ArgPlanForIdentity(
    const Fp8RK::PhysicalKernel& identity, DataType input_dtype, uint64_t outer, uint64_t r, uint64_t inner);

/** 1142: the one-pass planner candidate for an original FP8 ARG RK site running `plan`. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp8ArgCandidateForPlan(const DenseReductionProblem& problem,
                                                                        DenseReductionSite site,
                                                                        DenseRetainedFp8ArgPhysicalPlan plan);

}  // namespace ThorImplementation::CubReductionInternal
