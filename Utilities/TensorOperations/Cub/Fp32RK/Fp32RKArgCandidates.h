#pragma once

// 1204: the FP32 RK ARG candidate generator and pass launcher (plans/Thor_FP32_Dense_Reduction_Plan, phase 6).
// Every FP32 RK ARG pass is the twin of an FP32 RK VALUE first stage with the same physical identity (FP8 RK's
// PhysicalKernel fields at element_bytes 4), or one of two ARG-only rotated passes:
//   OneWindow   (a O, b W, c 16)          Complete; the FP32 unit of that ownership at K's bucket.
//   DeepRSmallK (a depth)                 staged: packed partials, finished by the FP32 ARG continuation.
//   AlignedK    (a packet, b CTA, c shards) Complete at one shard, else staged with VALUE's balanced rows.
//   RuntimeK PacketComplete (b W, c tile) Complete.
//   RuntimeK rotated (c 125, d tiles, e depth, f shards) Complete when depth >= R, else staged; the ARG-only passes are
//                                         the single-shard pass (R <= 1024) and the depth min(R, 1024) fallback,
//                                         which no VALUE identity maps to.
// Every pass writes the requested outputs itself (staged passes run their continuation in the same call), so a
// candidate's only extra memory is its packed partial scratch (fp32ArgPackedScratchBytes). The VALUE pool supplies
// the twins in its order with its local ranks; the ARG-only passes take the next local rank of their family, as FP8
// ARG's candidates do.

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp32RK {

struct Fp32RKArgCandidate {
    Fp8RK::PhysicalKernel identity;  // element_bytes 4
    uint32_t local_rank = 0;         // 1-based within its family
    uint64_t scratch_bytes = 0;      // packed partials of a staged pass, else 0
};

/** True when the FP32 ARG pass `identity` names is legal at [outer, r, k]. */
[[nodiscard]] bool fp32RKArgPassLegal(const Fp8RK::PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k);

/** Packed partial scratch of the pass (0 for a Complete pass). Assumes fp32RKArgPassLegal. */
[[nodiscard]] uint64_t fp32RKArgScratchBytes(const Fp8RK::PhysicalKernel& identity, uint64_t outer, uint64_t r,
                                             uint64_t k);

/** Every distinct legal FP32 ARG pass at [outer, r, k] within `scratch_budget_bytes`: the twins of `value_pool` (an
 *  fp32RKCandidatePool at MIN / MAX semantics) in its order with its local ranks, then the ARG-only rotated passes. */
[[nodiscard]] std::vector<Fp32RKArgCandidate> fp32RKArgCandidates(const std::vector<Fp32RKPoolEntry>& value_pool,
                                                                  uint64_t outer, uint64_t r, uint64_t k,
                                                                  uint64_t scratch_budget_bytes);

/** The candidates as ranker pool entries (identity and local rank), for fp32RKRankedKernels with the ARG aggregate. */
[[nodiscard]] std::vector<Fp32RKPoolEntry> fp32RKArgRankerPool(const std::vector<Fp32RKArgCandidate>& candidates);

/** Runs the pass: FP32 input [outer, r, k], value (any float storage) and / or index (UINT32 / UINT64) outputs of
 *  outer x k, index = row * domain_stride. `packed_scratch` holds fp32RKArgScratchBytes (null for a Complete pass).
 *  Throws std::logic_error for an illegal identity. */
void launchFp32RKArgPass(const Fp8RK::PhysicalKernel& identity,
                         const Tensor& input,
                         Tensor* value_output,
                         Tensor* index_output,
                         const CubReductionGeometry& geometry,
                         uint64_t* packed_scratch,
                         bool maximum,
                         uint64_t domain_stride,
                         Stream& stream);

}  // namespace ThorImplementation::CubReductionInternal::Fp32RK
