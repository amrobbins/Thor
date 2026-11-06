#pragma once

// 1236: stamp-time selection of retained FP16 / BF16 R / KR VALUE passes (plans/Thor_FP16_BF16_Dense_Reduction_Plan,
// phase 2), and 1237 ARG passes (phase 3).
// The FP32 KR selector's pattern at element size 2: the candidates are the 2-byte ranker's shortlist (without a
// trained model, the 2-byte heuristic list's prefix at the RK depths, Quick 6, Regular 12, Ultra 24), and the shared
// contest (ReductionAutotuning::autotune) times them end to end, a Split shape's finishing kernel included. Winners
// are cached per process and device in the one dense reduction selection cache, kind F16KR, per stage; a deeper
// selection answers a shallower request. The key holds the input dtype, so FP16 and BF16 never share an entry (the
// 1215 / 1216 lesson: both have element size 2).
//
// Domain: original FP16 / BF16 input at an R or KR site. Intermediate passes read FP32 partials and belong to the
// retained FP32 selectors. Not routed yet (phase 8). Shapes are in bytes (row_bytes = 2 * elements per row), as
// everywhere in the retained KR vocabulary.

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal::F16KR {

struct F16KRSelection final : AcceleratorBackendSelectionRecipeTag {
    std::optional<Fp8KR::Fp8KRSegmentShape> shape;  // nullopt: semantics the retained kernels do not compile
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
 *  1239: model e425a0b4 (18514 FP16 / BF16 cells, 64 KiB-512 MiB, sm_count 170: the 1238 sweep's 12256 and 1237's
 *  6258, VALUE and ARG). train.py's rules chose {1, 2, 4}. Out of fold (VALUE / ARG): N=1 p99 1.028 / 1.039; N=2 p99
 *  1.010 / 1.010, max 1.068 / 1.146; N=4 max 1.054 / 1.024; N=8 max 1.013 / 1.000. Held out: N=2 max 1.019 / 1.013,
 *  N=4 max 1.012 / 1.005. FP8's model on the same cells, unchanged: N=1 p99 1.033 / 1.066. User decision 2026-11-05:
 *  {4, 6, 8}. */
inline constexpr std::array<size_t, 3> F16_KR_SHORTLIST_DEPTHS{4, 6, 8};

/** True once a trained F16KRRankerModel.h is installed (phase 4). */
[[nodiscard]] bool f16KRRankerTrained();

/** The 2-byte pool (Fp8KR::retainedKRCandidatePool at element size 2); for an ARG site (site.kind != 0) without the
 *  shapes the 2-byte ARG kernels cannot run (Fp8KR::f16KRArgShapeLegal: vector Pack). Entries keep their pool list
 *  positions. */
[[nodiscard]] std::vector<Fp8KR::Fp8KRPoolEntry> f16KRCandidatePool(uint64_t rows, uint64_t row_bytes,
                                                                    uint32_t multiprocessors,
                                                                    const Fp8KR::Fp8KRRankerSite& site);

/** The first `count` shapes to time: the pool ordered by the trained ranker's margin (ties in pool order), or,
 *  without a trained model, the listed entries of the pool (the 2-byte heuristic list) in list order. */
[[nodiscard]] std::vector<Fp8KR::Fp8KRSegmentShape> f16KRRankedShapes(uint64_t rows, uint64_t row_bytes,
                                                                     uint32_t multiprocessors,
                                                                     const Fp8KR::Fp8KRRankerSite& site,
                                                                     size_t count);

/** How many ranked shapes `depth` times: F16_KR_SHORTLIST_DEPTHS when trained, else the RK depths. */
[[nodiscard]] size_t f16KRShortlistLength(CubReductionTuningDepth depth);

/** The shapes the selector times at `depth`: f16KRRankedShapes with f16KRShortlistLength(depth). */
[[nodiscard]] std::vector<Fp8KR::Fp8KRSegmentShape> f16KRShortlist(uint64_t rows, uint64_t row_bytes,
                                                                  uint32_t multiprocessors,
                                                                  const Fp8KR::Fp8KRRankerSite& site,
                                                                  CubReductionTuningDepth depth);

/** The ranker site of a 2-byte ARG pass: kind 2 (value and index) or 1 (index only), combine 3 (argmax) or 2, the
 *  alternate-format slot the BF16 flag. */
[[nodiscard]] Fp8KR::Fp8KRRankerSite f16KRArgRankerSite(bool maximum, bool produce_value, bool bf16);

/** FP16 / BF16 VALUE R / KR site (isOriginalF16KRSite for `input_dtype`). `operation` is the public operation; the
 *  pass role sets the semantics; output_scale applies on the final pass. The ranker site's alternate-format slot is
 *  the BF16 flag (fp8KRValueRankerSite). */
[[nodiscard]] F16KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                   DenseReductionSite site,
                                                   DataType input_dtype,
                                                   CubReductionOp operation,
                                                   uint64_t original_reduction_size,
                                                   float output_scale,
                                                   uint32_t multiprocessors,
                                                   const Stream& stream,
                                                   CubReductionTuningDepth tuning_depth);

/** 1237: original FP16 / BF16 ARG R / KR site (argmax when `maximum`). Always returns a retained shape. The outputs
 *  are timed as the stamped pass writes them: the requested outputs on the final site, else the FP32 value and the
 *  carried index (the intermediate the FP32 carried ARG selector continues). */
[[nodiscard]] F16KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                 DenseReductionSite site,
                                                 DataType input_dtype,
                                                 bool maximum,
                                                 uint32_t multiprocessors,
                                                 const Stream& stream,
                                                 CubReductionTuningDepth tuning_depth);

}  // namespace ThorImplementation::CubReductionInternal::F16KR
