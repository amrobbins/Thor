#pragma once

// The carried-index (intermediate) FP32 RK ARG candidates and pass launcher. A carried site is an intermediate ARG RK
// site: FP32 values with a carried index per element (the values and indices an earlier pass left), which arises when
// an ARG reduction spans two non-adjacent axes. The passes are the carried twins of the FP32 RK ARG families, all but
// PacketComplete:
//   OneWindow (a O, b W, c 16)                            Complete; the FP32 unit of that ownership at K's bucket.
//   DeepRSmallK (a depth)                                 staged at a compiled depth below R (K <= 125).
//   AlignedK (a packet, b CTA, c shards)                 Complete at one shard, else staged with VALUE's balanced rows.
//   RuntimeK rotated (c 125, d tiles, e depth, f shards)  Complete at depth R; staged at FP8's compiled depths below R.
// Staged passes write wide partials (fp32ArgWideScratchBytes) and finish them in the same call. The identities are
// FP32 RK ARG's (Fp32RKArgCandidates.h); carried-ness is the site's input state, not the identity's.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/TensorOperations/Reduction/Layout/ReductionGeometry.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp32/Fp32RKArgCandidates.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::ReductionInternal::RK::Fp32 {

/** True when the carried pass `identity` names is legal at [outer, r, k]. */
[[nodiscard]] bool fp32RKCarriedArgPassLegal(const RK::PhysicalKernel& identity, uint64_t outer, uint64_t r,
                                             uint64_t k);

/** Wide partial scratch of the pass (0 for a Complete pass). Assumes fp32RKCarriedArgPassLegal. */
[[nodiscard]] uint64_t fp32RKCarriedArgScratchBytes(const RK::PhysicalKernel& identity, uint64_t outer, uint64_t r,
                                                    uint64_t k);

/** Every legal carried pass at [outer, r, k] within `scratch_budget_bytes`, in family order: OneWindow (O1 W1/W2/W4/W8,
 *  then O2/O4/O8 W1), DeepRSmallK (depths 64 to 1024), AlignedK (16- then 8-byte packets, CTA 32 / 64 / 128, one shard
 *  then FP8's 32- and 128-row shard targets), then rotated (depth R, then the compiled depths below R); local ranks
 *  within each family in that order. */
[[nodiscard]] std::vector<Fp32RKArgCandidate> fp32RKCarriedArgCandidates(uint64_t outer, uint64_t r, uint64_t k,
                                                                         uint64_t scratch_budget_bytes);

/** Runs the pass: FP32 values and a UINT32 / UINT64 carried index of [outer, r, k], value (any float storage) and / or
 *  index (UINT32 / UINT64) outputs of outer x k holding the composed index (carried + row * row_stride). `wide_scratch`
 *  holds fp32RKCarriedArgScratchBytes, 256-byte aligned (null for a Complete pass). Throws std::logic_error for an
 *  illegal identity. */
void launchFp32RKCarriedArgPass(const RK::PhysicalKernel& identity,
                                const StridedSpan& input,
                                const StridedSpan& carried_index,
                                const StridedSpan* value_output,
                                const StridedSpan* index_output,
                                const ReductionGeometry& geometry,
                                void* wide_scratch,
                                bool maximum,
                                uint64_t row_stride,
                                Stream& stream);

}  // namespace ThorImplementation::ReductionInternal::RK::Fp32
