#pragma once

// 1251: the FP16 / BF16 RK ARG candidate generator and pass launcher (plans/Thor_FP16_BF16_Dense_Reduction_Plan,
// phase 6). FP32's (Fp32RK/Fp32RKArgCandidates.h) at element size 2: every 2-byte RK ARG pass (1248-1250, each element
// widened to FP32 and given the FP32 key) is the twin of a 2-byte RK VALUE first stage with the same physical identity
// (element_bytes 2), or one of the two ARG-only rotated passes:
//   OneWindow   (a O, b W, c 16)            Complete; the 2-byte unit of that ownership at K's bucket.
//   DeepRSmallK (a depth)                   staged: packed partials, finished by the FP32 ARG continuation.
//   AlignedK    (a packet, b CTA, c shards) Complete at one shard, else staged; 4-, 8- or 16-byte packets (the VALUE
//                                           pool's 2-byte packet at odd K has no ARG twin).
//   RuntimeK PacketComplete (b W, c tile)   Complete.
//   RuntimeK rotated (c 249, d tiles, e depth, f shards) Complete when depth >= R, else staged; the ARG-only passes are
//                                           the single-shard pass (R <= 1024) and the depth min(R, 1024) fallback.
// Every pass writes the requested outputs itself, so a candidate's only extra memory is its packed partial scratch
// (fp32ArgPackedScratchBytes). The VALUE pool supplies the twins in its order with its local ranks; the ARG-only
// passes take the next local rank of their family.

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal::F16RK {

struct F16RKArgCandidate {
    Fp8RK::PhysicalKernel identity;  // element_bytes 2
    uint32_t local_rank = 0;         // 1-based within its family
    uint64_t scratch_bytes = 0;      // packed partials of a staged pass, else 0
};

/** True when the 2-byte ARG pass `identity` names is legal at [outer, r, k]. */
[[nodiscard]] bool f16RKArgPassLegal(const Fp8RK::PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k);

/** Packed partial scratch of the pass (0 for a Complete pass). Assumes f16RKArgPassLegal. */
[[nodiscard]] uint64_t f16RKArgScratchBytes(const Fp8RK::PhysicalKernel& identity, uint64_t outer, uint64_t r,
                                            uint64_t k);

/** Every distinct legal 2-byte ARG pass at [outer, r, k] within `scratch_budget_bytes`: the twins of `value_pool` (an
 *  f16RKCandidatePool at MIN / MAX semantics) in its order with its local ranks, then the ARG-only rotated passes. */
[[nodiscard]] std::vector<F16RKArgCandidate> f16RKArgCandidates(const std::vector<F16RKPoolEntry>& value_pool,
                                                                uint64_t outer, uint64_t r, uint64_t k,
                                                                uint64_t scratch_budget_bytes);

/** The candidates as ranker pool entries (identity and local rank), for f16RKRankedKernels with the ARG aggregate. */
[[nodiscard]] std::vector<F16RKPoolEntry> f16RKArgRankerPool(const std::vector<F16RKArgCandidate>& candidates);

/** Runs the pass: FP16 / BF16 input [outer, r, k], value (any float storage) and / or index (UINT32 / UINT64) outputs
 *  of outer x k, index = row * domain_stride. `packed_scratch` holds f16RKArgScratchBytes (null for a Complete pass).
 *  Throws std::logic_error for an illegal identity. */
void launchF16RKArgPass(const Fp8RK::PhysicalKernel& identity,
                        const Tensor& input,
                        Tensor* value_output,
                        Tensor* index_output,
                        const CubReductionGeometry& geometry,
                        uint64_t* packed_scratch,
                        bool maximum,
                        uint64_t domain_stride,
                        Stream& stream);

}  // namespace ThorImplementation::CubReductionInternal::F16RK
