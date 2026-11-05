#pragma once

// 1180-1181: stamp-time selection of retained FP32 R / KR VALUE and ARG passes (plans/Thor_FP32_Dense_Reduction_Plan,
// phases 2 and 3).
// The FP8 KR selector's pattern at element size 4: the candidates are the FP32 ranker's shortlist (1182; without a
// trained model, the FP32 heuristic list's prefix at the RK depths, Quick 6, Regular 12, Ultra 24), and the shared
// contest (ReductionAutotuning::autotune) times them end to end, a Split shape's finishing kernel included. Winners
// are cached per process and device in the one dense reduction selection cache, kind Fp32KR; a deeper selection
// answers a shallower request.
//
// Routed (1220 VALUE, 1221 ARG, unconditional since 1225 / 1227): selectDensePhysicalPass calls it for every FP32
// R / KR site. Shapes are in bytes (row_bytes = 4 * elements per row), as everywhere
// in the retained KR vocabulary.

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

namespace ThorImplementation::CubReductionInternal::Fp32KR {

struct Fp32KRSelection final : AcceleratorBackendSelectionRecipeTag {
    std::optional<Fp8KR::Fp8KRSegmentShape> shape;  // nullopt: semantics the retained kernels do not compile
    size_t candidate_count = 0;
    std::string winner_name;
};

/** The [rows, row_bytes] of an FP32 R or KR site: rows 1 for R; row_bytes = 4 * the nominated run's extent. */
struct Fp32KRSiteGeometry {
    uint64_t rows = 1;
    uint64_t row_bytes = 0;
};
[[nodiscard]] Fp32KRSiteGeometry fp32KRSiteGeometry(const DenseReductionProblem& problem, DenseReductionSite site);

/** True for an original-input FP32 R or KR site (the selector's domain; semantics are checked by the selector). */
[[nodiscard]] bool isOriginalFp32KRSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                        DataType original_input_dtype);

/** 1215: a carried-index FP32 ARG R or KR site: an ARG intermediate aggregate (the FP32 values and carried index,
 *  problem.carried_arg_index_dtype, of any dtype's earlier passes). The carried ARG selector's domain. */
[[nodiscard]] bool isCarriedFp32KRArgSite(const DenseReductionProblem& problem, DenseReductionSite site);

/** 1207: an FP32 VALUE R or KR site: original FP32 input, or an intermediate aggregate (FP32 partials of any dtype's
 *  earlier passes; `original_input_dtype` is then not consulted). The VALUE selector's domain. */
[[nodiscard]] bool isFp32KRValueSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                     DataType original_input_dtype);

// ---- 1182: the FP32 KR ranker (plans/Thor_FP32_Dense_Reduction_Plan, phase 4). One ranker per dtype (user decision
// 2026-11-02): Fp32KRRankerModel.h, trained by `benchmarks/fp8_kr_ranker/train.py --dtype=fp32` on the FP32 cells of
// `thor_fp8_kr_training_sweep --dtypes=fp32`, with FP8's 24 features (Fp8KR::fp8KRRankerFeatures: the shapes are in
// bytes for both dtypes; the E5M2 feature is 0) over the FP32 pool. 1183: the trained model is installed. Without one
// (a placeholder with TRAINED = false) the selectors time the FP32 heuristic list's prefix at the RK depths.

/** Shortlist lengths (Quick, Regular, Ultra) of the trained FP32 ranker.
 *  1183: model 35c2616f (12256 FP32 cells, 64 KiB-512 MiB, sm_count 170). train.py's rules chose {1, 2, 4}. Regret of
 *  the ranked top N, out of fold (VALUE / ARG): N=1 p99 1.041 / 1.037; N=2 p99 1.006 / 1.007, max 1.22 / 1.18; N=4 max
 *  1.013 / 1.013; N=6 max 1.000 / 1.003; N=8 the fastest shape in every out-of-fold and held-out cell (the heuristic
 *  list needed 12). 1184 (user decision 2026-11-02): {2, 4, 6}, one step above each printed minimum (Quick at N=2 has
 *  p99 1.006 / 1.007; Regular at N=4 max 1.013; Ultra at N=6 max 1.003); a contest round at N=2 and N=4 costs about 2x
 *  and 5x one run of the fastest shape.
 *  1218: model 787279aa (24512 FP32 cells: 1183's 12256 VALUE and ARG plus 1217's 6128 UINT32 and 6128 UINT64 carried
 *  ARG). train.py's rules chose {1, 2, 4} again. Out of fold (VALUE / ARG / carried ARG):
 *  N=2 p99 1.008 / 1.008 / 1.009, max 1.22 / 1.18 / 1.16; N=4 max 1.013 / 1.013 / 1.031; N=6 max 1.000 / 1.012 /
 *  1.005; N=8 the fastest shape everywhere. The 1183 model on the (unseen) carried cells: N=2 p99 1.020, N=4 p99 1.003
 *  max 1.044. A confirmation sweep (seed 1182, VALUE / ARG and UINT64 carried ARG) matched these.
 *  1219 (user decision 2026-11-04): {4, 4, 6}: Quick times as many shapes as Regular (four is cheap for autotuning;
 *  Quick at N=2 had carried p99 1.007-1.009, N=4 max 1.004 on the confirmation cells). */
inline constexpr std::array<size_t, 3> FP32_KR_SHORTLIST_DEPTHS{4, 4, 6};

/** True once a trained Fp32KRRankerModel.h is installed. */
[[nodiscard]] bool fp32KRRankerTrained();

/** The FP32 pool (Fp8KR::retainedKRCandidatePool at element size 4); for an ARG site (site.kind != 0) without the
 *  shapes the FP32 ARG kernels cannot run (Fp8KR::fp32KRArgShapeLegal). Entries keep their pool list positions. */
[[nodiscard]] std::vector<Fp8KR::Fp8KRPoolEntry> fp32KRCandidatePool(uint64_t rows, uint64_t row_bytes,
                                                                     uint32_t multiprocessors,
                                                                     const Fp8KR::Fp8KRRankerSite& site);

/** The first `count` shapes to time: the pool ordered by the trained ranker's margin (ties in pool order), or, without
 *  a trained model, the listed entries of the pool (the FP32 heuristic list) in list order. */
[[nodiscard]] std::vector<Fp8KR::Fp8KRSegmentShape> fp32KRRankedShapes(uint64_t rows, uint64_t row_bytes,
                                                                      uint32_t multiprocessors,
                                                                      const Fp8KR::Fp8KRRankerSite& site,
                                                                      size_t count);

/** How many ranked shapes `depth` times: FP32_KR_SHORTLIST_DEPTHS when trained, else the RK depths. */
[[nodiscard]] size_t fp32KRShortlistLength(CubReductionTuningDepth depth);

/** The shapes the selectors time at `depth`: fp32KRRankedShapes with fp32KRShortlistLength(depth). */
[[nodiscard]] std::vector<Fp8KR::Fp8KRSegmentShape> fp32KRShortlist(uint64_t rows, uint64_t row_bytes,
                                                                   uint32_t multiprocessors,
                                                                   const Fp8KR::Fp8KRRankerSite& site,
                                                                   CubReductionTuningDepth depth);

/** The ranker site of an FP32 ARG pass: kind 2 (value and index) or 1 (index only), combine 3 (argmax) or 2; 1215: at
 *  a carried-index site (`carried`) kind 4 or 3. */
[[nodiscard]] Fp8KR::Fp8KRRankerSite fp32KRArgRankerSite(bool maximum, bool produce_value, bool carried = false);

/** FP32 VALUE R / KR site (isFp32KRValueSite: original or, 1207, intermediate). `operation` is the public operation;
 *  the pass role sets the semantics (an intermediate applies no input transform); output_scale applies on the final
 *  pass. */
[[nodiscard]] Fp32KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                    DenseReductionSite site,
                                                    CubReductionOp operation,
                                                    uint64_t original_reduction_size,
                                                    float output_scale,
                                                    uint32_t multiprocessors,
                                                    const Stream& stream,
                                                    CubReductionTuningDepth tuning_depth);

/** 1181: original FP32 ARG R / KR site (argmax when `maximum`). Always returns a retained shape. */
[[nodiscard]] Fp32KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                  DenseReductionSite site,
                                                  bool maximum,
                                                  uint32_t multiprocessors,
                                                  const Stream& stream,
                                                  CubReductionTuningDepth tuning_depth);

/** 1215: the selection at a carried-index FP32 ARG R / KR site (isCarriedFp32KRArgSite; phase 8). The
 *  FP32 ARG pool (the ARG-legal shapes) ranked with the carried ranker site, the shortlist timed as
 *  launchFp32KRCarriedArgSegments runs it (a Split shape's finishing kernel included) with the reduced run's
 *  original-domain stride (0 for an intermediate run), writing selectCachedArgSite's outputs, timing on the
 *  representative FP32 values and carried index (contiguous, the site's elements; the contest input holds both,
 *  Fp32CarriedArgContestInput.h, so both stream cold). Always returns a retained shape. */
[[nodiscard]] Fp32KRSelection autotuneCarriedArgSite(const DenseReductionProblem& problem,
                                                     DenseReductionSite site,
                                                     bool maximum,
                                                     uint32_t multiprocessors,
                                                     const Tensor& representative_values,
                                                     const Tensor& representative_carried_index,
                                                     const Stream& stream,
                                                     CubReductionTuningDepth tuning_depth);

/** The cached carried ARG winner (kind Fp32KR, its own site kind with the carried index dtype and run kind), tuned on
 *  a miss with a representative contest input (values 1.0, carried index 0). */
[[nodiscard]] Fp32KRSelection selectCachedCarriedArgSite(const DenseReductionProblem& problem,
                                                         DenseReductionSite site,
                                                         bool maximum,
                                                         uint32_t multiprocessors,
                                                         const Stream& stream,
                                                         CubReductionTuningDepth tuning_depth);

}  // namespace ThorImplementation::CubReductionInternal::Fp32KR
