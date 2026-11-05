#pragma once

// 1192: the FP32 RK VALUE candidate generator (plans/Thor_FP32_Dense_Reduction_Plan, phase 5). Host-only.
// The pool is every retained FP32 first stage legal at a site: OneWindow, DeepRSmallK, AlignedK and RuntimeK
// (PacketComplete, rotated Prefix), as Fp8RK::PhysicalKernel identities with element_bytes 4, each with its compiled
// residency (zero excludes it) and, for staged stages, FP32 partials within the scratch budget. FP8's frozen per-family
// scores are calibrated on FP8 timings and are not reused: within a family the pool follows a fixed enumeration order
// (the entry's local rank, a ranker feature), and the families are interleaved by local rank, as FP8's candidates are
// before its model ranks them. With a trained Fp32RKRankerModel the shortlist is the pool ordered by the model's margin;
// without one it is the pool's prefix. One ranker per dtype: the 26 features are FP8's, in elements where FP8's are
// in bytes (see fp32RKRankerFeatures).

#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKRankerModel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp32RK {

struct Fp32RKSite {
    uint64_t outer = 0, r = 0, k = 0;
    uint32_t sm_count = 0;
    CubReductionStageSemantics semantics{};
    uint64_t scratch_budget_bytes = std::numeric_limits<uint64_t>::max();
};

struct Fp32RKPoolEntry {
    Fp8RK::PhysicalKernel kernel;
    uint32_t local_rank = 0;  // 1-based within its family, in enumeration order
};

/** The compiled residency (CTAs per SM) of an FP32 first stage under the site's semantics; 0 when no retained kernel
 *  runs it. Production uses fp32RKCompiledResidency; tests may substitute their own. */
using Fp32RKResidency = std::function<uint32_t(const Fp8RK::PhysicalKernel&)>;

/** The production residency: the retained FP32 kernels' occupancy queries (Fp8RKPhysicalKernels.h adapters). A kernel
 *  with no compiled specialization (std::logic_error from the query) is 0; other errors propagate. */
[[nodiscard]] uint32_t fp32RKCompiledResidency(const Fp8RK::PhysicalKernel& kernel, uint64_t k,
                                               const CubReductionStageSemantics& semantics);

/** 1222: the input base alignment (bytes) the FP32 RK pass `kernel` requires, VALUE stage or ARG twin alike: 4 (any
 *  FP32 address) for the rotated RuntimeK pass, the packet for AlignedK, 16 for OneWindow, DeepRSmallK and RuntimeK
 *  PacketComplete. The carried-index twins read fresh, aligned intermediates. */
[[nodiscard]] uint32_t fp32RKPassInputAlignment(const Fp8RK::PhysicalKernel& kernel);

/** The pool at `site`, families interleaved by local rank (OneWindow, DeepRSmallK, AlignedK, RuntimeK). Every entry
 *  satisfies Fp8RK::structurallyValid and the family's FP32 launch limits. */
[[nodiscard]] std::vector<Fp32RKPoolEntry> fp32RKCandidatePool(const Fp32RKSite& site, const Fp32RKResidency& residency);

/** FP8's 26 ranker features (Fp8GeneralRParallelPolicy::Detail::rankerFeatures) for an FP32 entry: the E4M3 slot is
 *  0, packet sizes are in bytes, tiles and the K / lane-span ratio count components, and the input-size feature is
 *  in MiB of input bytes. `aggregate` is 0 for VALUE (1 ARG index, 2 ARG value and index). */
[[nodiscard]] std::array<float, Fp32RKRankerModel::FEATURE_COUNT> fp32RKRankerFeatures(const Fp32RKPoolEntry& entry,
                                                                                       const Fp32RKSite& site,
                                                                                       uint32_t aggregate = 0);

/** True once a trained Fp32RKRankerModel.h is installed. */
[[nodiscard]] constexpr bool fp32RKRankerTrained() { return Fp32RKRankerModel::TRAINED; }

/** Shortlist lengths (Quick, Regular, Ultra) of the trained FP32 RK ranker, for VALUE and ARG.
 *  1230: model a37fea04, retrained for 1228's 4-byte AlignedK packet on 21255 FP32 cells (1228's sweep: 14682 VALUE and
 *  ARG cells at 1, 4, 96 and 192 MiB; and 1205's 6573 ARG cells at 96 and 192 MiB; sm_count 170). train.py's rules
 *  chose Quick 1, Regular 4, Ultra 13. Out-of-fold max regret of the ranked top N, ARG / VALUE: N=3 1.309 / 1.172 (p99
 *  1.017 / 1.023); N=8 1.031 / 1.015; N=13 1.003 / 1.008 (held out: N=3 1.094 / 1.116, N=8 1.009 / 1.055, N=13 1.000 /
 *  1.000). User decision 2026-11-04: {3, 8, 13}; a contest round costs about 3x, 8x and 13x one run of the fastest
 *  kernel.
 *  1206: model aa090137 (19719 cells at 96 and 192 MiB) ran {2, 6, 14} (user decision 2026-11-03). */
inline constexpr std::array<size_t, 3> FP32_RK_SHORTLIST_DEPTHS{3, 8, 13};

/** How many candidates `depth` times: FP32_RK_SHORTLIST_DEPTHS when trained, else the FP8 RK depths (6 / 12 / 24). */
[[nodiscard]] size_t fp32RKShortlistLength(CubReductionTuningDepth depth);

/** The first `count` kernels to time: the pool ordered by the trained ranker's margin (ties in pool order), or the
 *  pool's prefix without a trained model. */
[[nodiscard]] std::vector<Fp8RK::PhysicalKernel> fp32RKRankedKernels(const std::vector<Fp32RKPoolEntry>& pool,
                                                                     const Fp32RKSite& site, size_t count,
                                                                     uint32_t aggregate = 0);

}  // namespace ThorImplementation::CubReductionInternal::Fp32RK
