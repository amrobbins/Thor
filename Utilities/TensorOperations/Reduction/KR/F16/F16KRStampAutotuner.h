#pragma once

// Stamp-time selection of FP16 / BF16 R / KR VALUE and ARG passes.
// The FP32 KR selector's pattern at element size 2: the candidates are the 2-byte ranker's shortlist (without a trained
// model, the 2-byte heuristic list's prefix at the RK depths, Quick 6, Regular 12, Ultra 24), and the shared contest
// (ReductionAutotuning::autotune) times them end to end, a Split shape's finishing kernel included. Winners are cached
// per process and device in the one dense reduction selection cache, kind F16KR, per stage; a deeper selection answers
// a shallower request. The key holds the input dtype, so FP16 and BF16 never share an entry (both have element size 2).
//
// Domain: original FP16 / BF16 input at an R or KR site. Intermediate passes read FP32 partials and belong to the FP32
// selectors. selectDensePhysicalPass calls it for every original FP16 / BF16 R / KR site. Shapes are in bytes
// (row_bytes = 2 * elements per row), as everywhere in the KR vocabulary.

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"
#include "Utilities/TensorOperations/Reduction/ReductionTypes.h"

#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanning.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation::ReductionInternal::KR::F16 {

struct F16KRSelection final : AcceleratorBackendSelectionRecipeTag {
    std::optional<KR::KRSegmentShape> shape;  // nullopt: semantics the kernels do not compile
    size_t candidate_count = 0;
    std::string winner_name;
};

/** The [rows, row_bytes] of a 2-byte R or KR site: rows 1 for R; row_bytes = 2 * the nominated run's extent. */
struct F16KRSiteGeometry {
    uint64_t rows = 1;
    uint64_t row_bytes = 0;
};
[[nodiscard]] F16KRSiteGeometry f16KRSiteGeometry(const DenseReductionProblem& problem, DenseReductionSite site);

/** True for FP16 and BF16. */
[[nodiscard]] bool isF16KRInputDtype(DataType dtype);

/** True for an original-input FP16 / BF16 R or KR site (the selector's domain; semantics are checked by the
 *  selector). */
[[nodiscard]] bool isOriginalF16KRSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                       DataType original_input_dtype);

/** Shortlist lengths (Quick, Regular, Ultra) of the trained FP16 / BF16 KR ranker.
 *  Model e425a0b4 (18514 FP16 / BF16 cells, 64 KiB-512 MiB, sm_count 170: 12256 and 6258 from two sweeps, VALUE and
 *  ARG). train.py's rules chose {1, 2, 4}. Out of fold (VALUE / ARG): N=1 p99 1.028 / 1.039; N=2 p99 1.010 / 1.010, max
 *  1.068 / 1.146; N=4 max 1.054 / 1.024; N=8 max 1.013 / 1.000. Held out: N=2 max 1.019 / 1.013, N=4 max 1.012 / 1.005.
 *  FP8's model on the same cells, unchanged: N=1 p99 1.033 / 1.066. The depths are {4, 6, 8}. */
inline constexpr std::array<size_t, 3> F16_KR_SHORTLIST_DEPTHS{4, 6, 8};

/** True once a trained F16KRRankerModel.h is installed. */
[[nodiscard]] bool f16KRRankerTrained();

/** The 2-byte pool (KR::retainedKRCandidatePool at element size 2); for an ARG site (site.kind != 0) without the
 *  shapes the 2-byte ARG kernels cannot run (KR::f16KRArgShapeLegal: vector Pack). Entries keep their pool list
 *  positions. */
[[nodiscard]] std::vector<KR::KRPoolEntry> f16KRCandidatePool(uint64_t rows, uint64_t row_bytes,
                                                              uint32_t multiprocessors,
                                                              const KR::KRRankerSite& site);

/** The first `count` shapes to time: the pool ordered by the trained ranker's margin (ties in pool order), or,
 *  without a trained model, the listed entries of the pool (the 2-byte heuristic list) in list order. */
[[nodiscard]] std::vector<KR::KRSegmentShape> f16KRRankedShapes(uint64_t rows, uint64_t row_bytes,
                                                                     uint32_t multiprocessors,
                                                                     const KR::KRRankerSite& site,
                                                                     size_t count);

/** How many ranked shapes `depth` times: F16_KR_SHORTLIST_DEPTHS when trained, else the RK depths. */
[[nodiscard]] size_t f16KRShortlistLength(ReductionTuningDepth depth);

/** The shapes the selector times at `depth`: f16KRRankedShapes with f16KRShortlistLength(depth). */
[[nodiscard]] std::vector<KR::KRSegmentShape> f16KRShortlist(uint64_t rows, uint64_t row_bytes,
                                                                  uint32_t multiprocessors,
                                                                  const KR::KRRankerSite& site,
                                                                  ReductionTuningDepth depth);

/** The ranker site of a 2-byte ARG pass: kind 2 (value and index) or 1 (index only), combine 3 (argmax) or 2, the
 *  alternate-format slot the BF16 flag. */
[[nodiscard]] KR::KRRankerSite f16KRArgRankerSite(bool maximum, bool produce_value, bool bf16);

/** FP16 / BF16 VALUE R / KR site (isOriginalF16KRSite for `input_dtype`). `operation` is the public operation; the
 *  pass role sets the semantics; output_scale applies on the final pass. The ranker site's alternate-format slot is
 *  the BF16 flag (krValueRankerSite). */
[[nodiscard]] F16KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                   DenseReductionSite site,
                                                   DataType input_dtype,
                                                   ReductionOp operation,
                                                   uint64_t original_reduction_size,
                                                   float output_scale,
                                                   uint32_t multiprocessors,
                                                   const Stream& stream,
                                                   ReductionTuningDepth tuning_depth);

/** Original FP16 / BF16 ARG R / KR site (argmax when `maximum`). Always returns a shape. The outputs are timed as the
 *  stamped pass writes them: the requested outputs on the final site, else the FP32 value and the carried index (the
 *  intermediate the FP32 carried ARG selector continues). */
[[nodiscard]] F16KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                 DenseReductionSite site,
                                                 DataType input_dtype,
                                                 bool maximum,
                                                 uint32_t multiprocessors,
                                                 const Stream& stream,
                                                 ReductionTuningDepth tuning_depth);

}  // namespace ThorImplementation::ReductionInternal::KR::F16
