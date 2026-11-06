#pragma once

// 1246: the FP16 / BF16 RK VALUE candidate generator (plans/Thor_FP16_BF16_Dense_Reduction_Plan, phase 5). Host-only.
// FP32's generator (Fp32RK/Fp32RKCandidates.h) at element size 2: the pool is every retained 2-byte first stage legal
// at a site (OneWindow, DeepRSmallK, AlignedK and RuntimeK PacketComplete / rotated Prefix, the 1240-1245 kernels), as
// Fp8RK::PhysicalKernel identities with element_bytes 2, each with its compiled residency for the site's format (zero
// excludes it) and, for staged stages, FP32 partials within the scratch budget. Within a family the pool follows a
// fixed enumeration order (the entry's local rank, a ranker feature); the families are interleaved by local rank. With
// a trained F16RKRankerModel the shortlist is the pool ordered by the model's margin; without one, the pool's prefix.
// One ranker for both formats: FP8's 26 features in 2-byte elements, the E4M3 slot holding BF16 (the format
// separates FP16 from BF16 as it separates E4M3 from E5M2).

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace ThorImplementation::CubReductionInternal::F16RK {

struct F16RKSite {
    uint64_t outer = 0, r = 0, k = 0;
    uint32_t sm_count = 0;
    DataType input_dtype = DataType::FP16;  // FP16 or BF16
    CubReductionStageSemantics semantics{};
    uint64_t scratch_budget_bytes = std::numeric_limits<uint64_t>::max();
};

struct F16RKPoolEntry {
    Fp8RK::PhysicalKernel kernel;
    uint32_t local_rank = 0;  // 1-based within its family, in enumeration order
};

/** The compiled residency (CTAs per SM) of a 2-byte first stage under the site's semantics; 0 when no retained kernel
 *  runs it. Production uses f16RKCompiledResidency; tests may substitute their own. */
using F16RKResidency = std::function<uint32_t(const Fp8RK::PhysicalKernel&)>;

/** The production residency: the retained FP16 / BF16 kernels' occupancy queries (Fp8RKPhysicalKernels.h adapters)
 *  for `input_dtype`. A kernel with no compiled specialization (std::logic_error from the query) is 0; other errors
 *  propagate. */
[[nodiscard]] uint32_t f16RKCompiledResidency(const Fp8RK::PhysicalKernel& kernel, DataType input_dtype, uint64_t k,
                                              const CubReductionStageSemantics& semantics);

/** The input base alignment (bytes) the 2-byte RK pass `kernel` requires: 2 (any FP16 / BF16 address) for the rotated
 *  RuntimeK pass, the packet for AlignedK, 16 for OneWindow, DeepRSmallK and RuntimeK PacketComplete. */
[[nodiscard]] uint32_t f16RKPassInputAlignment(const Fp8RK::PhysicalKernel& kernel);

/** The pool at `site`, families interleaved by local rank (OneWindow, DeepRSmallK, AlignedK, RuntimeK). Every entry
 *  satisfies Fp8RK::structurallyValid and the family's 2-byte launch limits. */
[[nodiscard]] std::vector<F16RKPoolEntry> f16RKCandidatePool(const F16RKSite& site, const F16RKResidency& residency);

/** FP8's 26 ranker features (fp32RKRankerFeatures' layout) for a 2-byte entry: the E4M3 slot is 1 for BF16 and 0 for
 *  FP16, packet sizes are in bytes, tiles and the K / lane-span ratio count components, and the input-size feature is
 *  in MiB of input bytes (2 per element). `aggregate` is 0 for VALUE (1 ARG index, 2 ARG value and index). */
[[nodiscard]] std::array<float, F16RKRankerModel::FEATURE_COUNT> f16RKRankerFeatures(const F16RKPoolEntry& entry,
                                                                                     const F16RKSite& site,
                                                                                     uint32_t aggregate = 0);

/** True once a trained F16RKRankerModel.h is installed (phase 7). */
[[nodiscard]] constexpr bool f16RKRankerTrained() { return F16RKRankerModel::TRAINED; }

/** 1253: the trained ranker's shortlist depths, Quick / Regular / Ultra. Model 3ad15d57 (15,538 FP16 / BF16 cells of
 *  the 1251 sweep, VALUE and ARG at 1-256 MiB); train.py printed Quick 2, Regular 4, Ultra 14. Out-of-fold max regret
 *  of the ranked top N, ARG / VALUE: N=4 1.144 / 1.106 (p99 1.016 / 1.020); N=8 1.029 / 1.044; N=12 1.016 / 1.020
 *  (held out: N=4 1.067 / 1.027, N=8 1.031 / 1.023, N=12 1.002 / 1.000). User decision 2026-11-05: {4, 8, 12}. */
inline constexpr std::array<size_t, 3> F16_RK_SHORTLIST_DEPTHS{4, 8, 12};

/** How many candidates `depth` times: F16_RK_SHORTLIST_DEPTHS when trained, else the FP8 RK depths (6 / 12 / 24). */
[[nodiscard]] size_t f16RKShortlistLength(CubReductionTuningDepth depth);

/** The first `count` kernels to time: the pool ordered by the trained ranker's margin (ties in pool order), or the
 *  pool's prefix without a trained model. */
[[nodiscard]] std::vector<Fp8RK::PhysicalKernel> f16RKRankedKernels(const std::vector<F16RKPoolEntry>& pool,
                                                                    const F16RKSite& site, size_t count,
                                                                    uint32_t aggregate = 0);

}  // namespace ThorImplementation::CubReductionInternal::F16RK
