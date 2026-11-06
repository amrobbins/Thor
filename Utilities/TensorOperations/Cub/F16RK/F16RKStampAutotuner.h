#pragma once

// 1247: stamp-time selection of retained FP16 / BF16 RK VALUE first stages (plans/Thor_FP16_BF16_Dense_Reduction_Plan,
// phase 5). The FP32 RK selector's pattern (Fp32RK/Fp32RKStampAutotuner.h) at element size 2: the candidates are the
// 2-byte pool's shortlist (F16RKCandidates.h: since 1253 the trained 2-byte ranker's top N at F16_RK_SHORTLIST_DEPTHS,
// Quick 4, Regular 8, Ultra 12; without one the pool's family round-robin prefix at 6 / 12 / 24), each timed end to end with its
// hypothetical FP32 continuation (a staged stage's FP32 partials are finished by the production FP32 passes, as for
// FP8 and FP32), in the shared contest with FP8 RK's rounds and trim. The winner's first stage is cached per process
// and device in the dense reduction selection cache, kind F16RK; a deeper selection answers a shallower request. The
// key holds the input dtype, so FP16 and BF16 never share an entry, and the input alignment.
//
// 1251: the ARG selector (FP32's autotuneArgSite at element size 2): the 2-byte ARG candidates (F16RKArgCandidates.h,
// the twins of the MIN / MAX VALUE pool and the ARG-only rotated passes), ranked with the ARG aggregate, each timed as
// it runs (a staged pass includes its packed FP32 ARG continuation), cached under kind F16RK beside VALUE (the key's
// aggregate kind separates them).
//
// Domain: original FP16 / BF16 input at an RK site. Intermediate passes read FP32 partials and belong to the retained
// FP32 selectors. Not routed yet (phase 8).

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <cstdint>
#include <limits>

class Stream;
namespace ThorImplementation { class Tensor; enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal::F16RK {

/** The FP16 / BF16 RK recipe: FP8's (the winner's first-stage PhysicalKernel, element_bytes 2). */
using F16RKSelection = Fp8RK::Fp8RKSelection;

/** True for an original-input FP16 / BF16 VALUE or ARG RK site (the selector's domain; semantics are checked by the
 *  selector). */
[[nodiscard]] bool isOriginalF16RKSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                       DataType original_input_dtype);

/** Cache-bypassing measurement at an original FP16 / BF16 VALUE RK site, timing on `representative_input`
 *  (contiguous, the site's [outer, R, K] elements in FP16 or BF16, on the stream's GPU). `input_alignment_bytes` (the
 *  original input's Tensor::alignmentBytes(16), at least 2) keeps only the passes that run at it
 *  (f16RKPassInputAlignment: the rotated RuntimeK pass runs at any 2-byte address). */
[[nodiscard]] F16RKSelection autotuneValueSite(const DenseReductionProblem& problem,
                                               DenseReductionSite site,
                                               CubReductionOp operation,
                                               uint64_t original_reduction_size,
                                               float output_scale,
                                               const Tensor& representative_input,
                                               const Stream& stream,
                                               uint64_t workspace_budget_bytes = std::numeric_limits<uint64_t>::max(),
                                               CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
                                               uint64_t input_alignment_bytes = 16);

/** Production selection: the cached winner (kind F16RK), tuned on a cache miss with a temporary `input_dtype` input
 *  (values 1.0). */
[[nodiscard]] F16RKSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                   DenseReductionSite site,
                                                   DataType input_dtype,
                                                   CubReductionOp operation,
                                                   uint64_t original_reduction_size,
                                                   float output_scale,
                                                   const Stream& stream,
                                                   uint64_t workspace_budget_bytes =
                                                       std::numeric_limits<uint64_t>::max(),
                                                   CubReductionTuningDepth tuning_depth =
                                                       CubReductionTuningDepth::Regular,
                                                   uint64_t input_alignment_bytes = 16);

/** 1251: cache-bypassing measurement at an original FP16 / BF16 ARG RK site (outputs as the problem requests on the
 *  final site, else the FP32 value and carried index), timing on `representative_input` as autotuneValueSite. A
 *  selection with candidate_count 0 when no 2-byte ARG pass fits the budget. */
[[nodiscard]] F16RKSelection autotuneArgSite(const DenseReductionProblem& problem,
                                             DenseReductionSite site,
                                             bool maximum,
                                             uint64_t original_reduction_size,
                                             const Tensor& representative_input,
                                             const Stream& stream,
                                             uint64_t workspace_budget_bytes = std::numeric_limits<uint64_t>::max(),
                                             CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
                                             uint64_t input_alignment_bytes = 16);

/** 1251: production ARG selection: the cached winner (kind F16RK), tuned on a cache miss with a temporary
 *  `input_dtype` input (values 1.0). */
[[nodiscard]] F16RKSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                 DenseReductionSite site,
                                                 DataType input_dtype,
                                                 bool maximum,
                                                 uint64_t original_reduction_size,
                                                 const Stream& stream,
                                                 uint64_t workspace_budget_bytes =
                                                     std::numeric_limits<uint64_t>::max(),
                                                 CubReductionTuningDepth tuning_depth =
                                                     CubReductionTuningDepth::Regular,
                                                 uint64_t input_alignment_bytes = 16);

}  // namespace ThorImplementation::CubReductionInternal::F16RK
