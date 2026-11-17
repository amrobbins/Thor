#pragma once

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ThorImplementation::ReductionInternal {

/**
 * Exactly ONE original-input FP8 RK ARG physical pass. Chosen by ARG's own ranked, autotuned shortlist
 * (RK::Fp8::selectCachedArgRKSite) from retainedFp8ArgCandidates; each candidate is the ARG twin of a VALUE MIN/MAX
 * kernel or an ARG-only rotated pass. OneWindow twins are Complete: the nominated R is eliminated and the pass writes
 * the public outputs (final pass) or an FP32 value + carried index intermediate.
 * DeepRSmallK twins are staged: they write FP32 value + carried index partials [outer, shards, inner], leaving an
 * IntermediateReduction run of `shards` for the carried-index ARG reducer. RuntimeK and AlignedK twins are Complete
 * with one shard and staged otherwise (AlignedK keeps VALUE's balanced shard rows, packet and CTA size). The index written is row * the nominated
 * run's original ARG-domain stride.
 */
struct Fp8ArgPhysicalPlan final : DenseReductionPhysicalPlan {
    RK::PhysicalFamily family = RK::PhysicalFamily::OneWindow;
    OneWindowUnit unit{};  // OneWindow
    uint32_t warps_per_output = 1;    // OneWindow
    uint64_t rows_per_shard = 0;      // DeepRSmallK, RuntimeK (RuntimeK: R for the single-shard Complete twin)
    uint64_t shards = 1;              // partials per output (1 for Complete twins)
    uint32_t packet_bytes = 16;       // RuntimeK: the ARG rotated kernel's packet (8 or 16); AlignedK: VALUE's (2 or 8)
    uint32_t cta_threads = 0;         // AlignedK: VALUE's CTA size (32, 64 or 128)
    uint32_t tile_components = 0;     // RuntimeK PacketComplete twin (nonzero): ARG K tile; warps_per_output = VALUE's W
    // A staged twin (shards > 1) writes 4-byte packed partials to its own scratch and finishes with
    // launchFp8ArgPackedContinuation in the same pass, so the planner sees one Complete pass with scratch.
    bool packed_partials = false;
    // An FP8 R / KR pass (launchFp8KRArgSegments) when kr.group_lanes != 0; family and the RK fields above are then
    // unused. input_dimensions are the site's current runs (the nominated run last), outer_size the rows,
    // reduction_size the row length and inner_size 1. A staged shape sets packed_partials: its 64-bit chunk candidates
    // live in the pass's scratch (retainedFp8ArgScratchBytes).
    KR::KRSegmentShape kr{};
    DataType input_dtype = DataType::FP8_E4M3;
    std::vector<uint64_t> input_dimensions;  // {outer, r, inner}
    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;
};

/** The input base alignment an FP8 ARG pass needs: the rotated RuntimeK passes and the R / KR passes any byte address
 *  (each lane aligns its window down and rotates the prefix away; KR reads words on the tensor's word grid), AlignedK
 *  its packet (2 or 8 bytes), DeepRSmallK 16 bytes. OneWindow any byte address when the plan's K fits
 *  RK::oneWindowUnalignedCapacity and its input 2^32 - 16 bytes, else 16 bytes. The PacketComplete twin any byte
 *  address. */
[[nodiscard]] uint32_t retainedFp8ArgPassInputAlignment(const Fp8ArgPhysicalPlan& plan);

/** Scratch of a packed staged twin: one uint32 lane candidate per [outer, shards, inner] partial (0 if not packed). */
[[nodiscard]] uint64_t retainedFp8ArgScratchBytes(const Fp8ArgPhysicalPlan& plan);

/** RuntimeK rotated ARG twins run at this packet width (lanes own at most 8 components; the 16-byte packet measured
 *  1.56x slower and is not compiled). */
inline constexpr uint32_t FP8_RUNTIME_K_ARG_PACKET_BYTES = 8;

/** The ARG twin of `value_kernel` at [outer, r, inner], when one exists and is legal (the candidate generator for
 *  ranked ARG selection). */
[[nodiscard]] std::optional<Fp8ArgPhysicalPlan> retainedFp8ArgRawTwin(const RK::PhysicalKernel& value_kernel,
                                                                                  DataType input_dtype,
                                                                                  uint64_t outer,
                                                                                  uint64_t r,
                                                                                  uint64_t inner);

/** The rotated RuntimeK ARG pass (8-byte packets, depth min(R, 1024)), an ARG-only candidate in every candidate set, so
 *  a site whose VALUE identities have no legal twin is still covered; nullopt only beyond the rotated kernel's own
 *  limits (grid extents, 2^23-row shards). */
[[nodiscard]] std::optional<Fp8ArgPhysicalPlan> retainedFp8ArgRotatedDeepPass(DataType input_dtype,
                                                                                       uint64_t outer,
                                                                                       uint64_t r,
                                                                                       uint64_t inner);

/** The physical identity (PhysicalKernel fields) of an ARG pass, for ranker features: OneWindow (O, W, P), DeepRSmallK
 *  depth, AlignedK (packet, CTA, shards), RuntimeK PacketComplete (W, ARG tile) or rotated (RotatedPrefix, the ARG
 *  kernel's tile, depth, shards). */
[[nodiscard]] RK::PhysicalKernel retainedFp8ArgPhysicalIdentity(const Fp8ArgPhysicalPlan& plan);

/** A VALUE candidate and its 1-based local rank in the mode-local VALUE order. */
struct RetainedFp8ValueSource {
    RK::PhysicalKernel kernel;
    uint32_t local_rank = 0;
};

/** A legal ARG pass with its ranker identity and local rank (packed_partials set). */
struct RetainedFp8ArgCandidate {
    Fp8ArgPhysicalPlan plan;
    RK::PhysicalKernel identity;
    uint32_t local_rank = 0;
};

/**
 * Every distinct legal ARG pass at [outer, r, inner]: the raw twins of `value_candidates` (VALUE MIN/MAX candidates, in
 * their order and with their local ranks), then the single-shard rotated pass (R <= 1024) and the fallback depth, which
 * no VALUE identity maps to. Duplicates keep the first occurrence.
 */
[[nodiscard]] std::vector<RetainedFp8ArgCandidate> retainedFp8ArgCandidates(
    std::span<const RetainedFp8ValueSource> value_candidates, DataType input_dtype, uint64_t outer, uint64_t r,
    uint64_t inner);

/** Launches one ARG pass (packed staged: first stage into `packed_scratch`, then the packed continuation). */
void launchRetainedFp8ArgPlan(const Fp8ArgPhysicalPlan& plan,
                              const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint32_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream);

/** The ARG pass an identity names (the inverse of retainedFp8ArgPhysicalIdentity; packed_partials set), when it is
 *  legal at [outer, r, inner]. Selections cache only the identity. */
[[nodiscard]] std::optional<Fp8ArgPhysicalPlan> retainedFp8ArgPlanForIdentity(
    const RK::PhysicalKernel& identity, DataType input_dtype, uint64_t outer, uint64_t r, uint64_t inner);

/** The one-pass planner candidate for an original FP8 ARG RK site running `plan`. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp8ArgCandidateForPlan(const DenseReductionProblem& problem,
                                                                        DenseReductionSite site,
                                                                        Fp8ArgPhysicalPlan plan);

}  // namespace ThorImplementation::ReductionInternal
