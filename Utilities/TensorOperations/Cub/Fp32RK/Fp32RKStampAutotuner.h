#pragma once

// 1192: stamp-time selection of retained FP32 RK VALUE first stages (plans/Thor_FP32_Dense_Reduction_Plan, phase 5).
// The FP8 RK selector's pattern at element size 4: the candidates are the FP32 pool's shortlist (Fp32RKCandidates.h:
// the trained FP32 ranker's top N, or without one the pool's family round-robin prefix at the FP8 RK depths, Quick 6,
// Regular 12, Ultra 24), each timed end to end with its hypothetical FP32 continuation (a staged stage's partials are
// finished by the production FP32 passes, as for FP8), in the shared contest with FP8 RK's rounds and trim. The
// winner's first stage is cached per process and device in the dense reduction selection cache, kind Fp32RK; a
// deeper selection answers a shallower request.
//
// Routed (1220 VALUE, 1221 ARG, unconditional since 1225 / 1227): selectDensePhysicalPass calls it for every FP32 RK
// site.

#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <cstdint>
#include <limits>

class Stream;
namespace ThorImplementation { class Tensor; enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal::Fp32RK {

/** The FP32 RK recipe: FP8's (the winner's first-stage PhysicalKernel, element_bytes 4). */
using Fp32RKSelection = Fp8RK::Fp8RKSelection;

/** Cache-bypassing measurement at an FP32 VALUE RK site, timing on `representative_input` (contiguous, the site's
 *  [outer, R, K] elements, on the stream's GPU). The site is original FP32 input or (1207, phase 8) an intermediate
 *  aggregate: FP32 partials of any dtype's earlier passes, reduced with the pass role's semantics (no input transform;
 *  the finalizer and output scale on the last pass). `original_reduction_size` is the whole original reduction domain
 *  (MEAN's divisor). */
[[nodiscard]] Fp32RKSelection autotuneValueSite(const DenseReductionProblem& problem,
                                                DenseReductionSite site,
                                                CubReductionOp operation,
                                                uint64_t original_reduction_size,
                                                float output_scale,
                                                const Tensor& representative_input,
                                                const Stream& stream,
                                                uint64_t workspace_budget_bytes = std::numeric_limits<uint64_t>::max(),
                                                CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
                                                uint64_t input_alignment_bytes = 16);

/** Production selection: the cached winner (kind Fp32RK), tuned on a cache miss with a temporary input. */
[[nodiscard]] Fp32RKSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                    DenseReductionSite site,
                                                    CubReductionOp operation,
                                                    uint64_t original_reduction_size,
                                                    float output_scale,
                                                    const Stream& stream,
                                                    uint64_t workspace_budget_bytes =
                                                        std::numeric_limits<uint64_t>::max(),
                                                    CubReductionTuningDepth tuning_depth =
                                                        CubReductionTuningDepth::Regular,
                                                    uint64_t input_alignment_bytes = 16);

/** 1222: `input_alignment_bytes` (the original input's Tensor::alignmentBytes(16)) keeps, at an original-input site,
 *  only the passes that tolerate it (fp32RKPassInputAlignment; the rotated RuntimeK pass runs at any FP32 address), and
 *  separates the cache entries; it is ignored at intermediate sites (fresh, aligned allocations). */

/** 1204: FP32 RK ARG's own ranked, autotuned selection at an original FP32 ARG RK site. The candidates
 *  are fp32RKArgCandidates (the ARG twins of the FP32 VALUE MIN / MAX pool plus the ARG-only rotated passes) within
 *  the workspace budget, ordered by the FP32 RK ranker with the ARG aggregate feature, the first
 *  fp32RKShortlistLength(depth) timed as the pass runs (staged passes include their packed continuation), writing the
 *  site's outputs (the requested ones on the final site, else FP32 value + carried index). The recipe's `physical` is
 *  the winner's ARG identity (launchFp32RKArgPass runs it); candidate_count 0 means no FP32 ARG pass is legal. */
[[nodiscard]] Fp32RKSelection autotuneArgSite(const DenseReductionProblem& problem,
                                              DenseReductionSite site,
                                              bool maximum,
                                              uint64_t original_reduction_size,
                                              const Tensor& representative_input,
                                              const Stream& stream,
                                              uint64_t workspace_budget_bytes = std::numeric_limits<uint64_t>::max(),
                                              CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
                                              uint64_t input_alignment_bytes = 16);

/** The cached ARG winner (kind Fp32RK; the key's aggregate kind and MIN / MAX operation separate it from VALUE),
 *  tuned on a miss with a temporary input. */
[[nodiscard]] Fp32RKSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                  DenseReductionSite site,
                                                  bool maximum,
                                                  uint64_t original_reduction_size,
                                                  const Stream& stream,
                                                  uint64_t workspace_budget_bytes =
                                                      std::numeric_limits<uint64_t>::max(),
                                                  CubReductionTuningDepth tuning_depth =
                                                      CubReductionTuningDepth::Regular,
                                                  uint64_t input_alignment_bytes = 16);

/** 1215: the selection at a carried-index (intermediate) FP32 ARG RK site (phase 8): the FP32 values
 *  and the carried index (problem.carried_arg_index_dtype, UINT32 / UINT64) an earlier pass of any dtype left; the
 *  reduced run is an original reduction (index stride its original-domain stride) or an intermediate one (stride 0:
 *  the carried index is already composed). The candidates are fp32RKCarriedArgCandidates within the workspace budget,
 *  ordered by the FP32 RK ranker with the carried aggregate (3 index, 4 value and index, as the carried sweep cells),
 *  the first fp32RKShortlistLength(depth) timed as launchFp32RKCarriedArgPass runs them (staged passes include the
 *  wide continuation), writing the site's outputs (autotuneArgSite's). The contest input holds the values and the
 *  carried index together (Fp32CarriedArgContestInput.h), so both stream cold. `physical` is the winner's identity;
 *  candidate_count 0 means no carried pass is legal. */
[[nodiscard]] Fp32RKSelection autotuneCarriedArgSite(const DenseReductionProblem& problem,
                                                     DenseReductionSite site,
                                                     bool maximum,
                                                     uint64_t original_reduction_size,
                                                     const Tensor& representative_values,
                                                     const Tensor& representative_carried_index,
                                                     const Stream& stream,
                                                     uint64_t workspace_budget_bytes =
                                                         std::numeric_limits<uint64_t>::max(),
                                                     CubReductionTuningDepth tuning_depth =
                                                         CubReductionTuningDepth::Regular);

/** The cached carried ARG winner (kind Fp32RK; the key's input state, carried index dtype and runs separate it from the
 *  original ARG site), tuned on a miss with a representative input (values 1.0, carried index 0). */
[[nodiscard]] Fp32RKSelection selectCachedCarriedArgSite(const DenseReductionProblem& problem,
                                                         DenseReductionSite site,
                                                         bool maximum,
                                                         uint64_t original_reduction_size,
                                                         const Stream& stream,
                                                         uint64_t workspace_budget_bytes =
                                                             std::numeric_limits<uint64_t>::max(),
                                                         CubReductionTuningDepth tuning_depth =
                                                             CubReductionTuningDepth::Regular);

}  // namespace ThorImplementation::CubReductionInternal::Fp32RK
