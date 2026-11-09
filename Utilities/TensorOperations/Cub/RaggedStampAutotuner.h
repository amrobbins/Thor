#pragma once

// 1322 (phase 6, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): stamp-time selection of a ragged reduction's launch
// shape, VALUE (SUM, MEAN, MIN, MAX) and ARG (ARGMIN, ARGMAX). Not routed yet (phase 7): nothing in a stamp calls it.
//
// Exactly one family runs a site: ragged KR for rows under 16 bytes, ragged RK from 16 bytes (raggedFamilyFor). Each
// family has a short candidate list, the shapes the ragged audit runs, so the contest times every legal candidate of
// the family; there is no ranker and no tuning depth (user decision 2026-11-08: time them all, unless the candidates
// more than triple; RAGGED_TIME_ALL_CANDIDATE_LIMIT). The ambient CubReductionTuningDepth is ignored, so networks of
// every depth share one entry per site.
//
// 1324: the winner is the first candidate in list order within RAGGED_TIE_MARGIN (1%) of the contest's best.
//
// The contest is the shared one (ReductionAutotuning::autotune: L2-cold inputs, a probe, interleaved rounds) with fixed
// rounds, on synthetic offsets (decision 4: B segments of uniform length over all N capacity rows) and an input of
// 1.0 in every element at the site's 16-byte misalignment. Each candidate is timed as the stamp runs it: the main
// kernel and the finisher (launchRaggedSelection). A candidate whose launch the launcher refuses for this site (a
// CTA's range past 2^31 events on a small grid, say) is dropped before the contest.
//
// Winners are cached per process and device in the one dense reduction selection cache, kinds RaggedKR / RaggedRK.
// The key holds the operation, the input / offsets / output dtypes, D, the capacity N, the segments B and the input's
// address mod 16.

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

enum class RaggedFamily : uint8_t { KR = 1, RK = 2 };

/** What a ragged stamp reduces: everything known at stamp time (not the offsets' contents). */
struct RaggedSite {
    bool arg = false;
    CubReductionOp value_op = CubReductionOp::Sum;           // VALUE: Sum, Mean, Min or Max
    CubArgReductionOp arg_op = CubArgReductionOp::ArgMin;    // ARG
    DataType input_dtype = DataType::FP32;
    DataType offsets_dtype = DataType::UINT32;
    DataType output_dtype = DataType::FP32;  // VALUE: the value dtype; ARG: the index dtype, UINT32 or UINT64
    uint64_t components = 1;                  // D
    uint64_t capacity_rows = 0;               // N
    uint64_t segments = 0;                    // B
    uint32_t input_head = 0;                  // the input's address mod 16, a multiple of the element's bytes
};

/** A selected ragged launch shape: the KR shape for family KR, the RK shape for family RK. */
struct RaggedSelection final : AcceleratorBackendSelectionRecipeTag {
    RaggedFamily family = RaggedFamily::KR;
    RaggedKRShape kr_shape{};
    RaggedRKShape rk_shape{};
    size_t candidate_count = 0;  // the legal candidates timed (0: nothing to time, an empty site)
    std::string winner_name;     // "ragged_kr:i16_c0_e12", "ragged_rk:ta_c0_e4"
    // 1323: the contest as measured (diagnostic, for the audit): each legal candidate's name and trimmed mean time on
    // the synthetic offsets, in list order; 0 for a candidate its probe ruled out.
    std::vector<std::string> contest_names;
    std::vector<double> contest_us;
};

/** Time every legal candidate while a family has at most this many (three times today's three). */
inline constexpr size_t RAGGED_TIME_ALL_CANDIDATE_LIMIT = 9;
/** The contest's rounds and the rounds trimmed at each end. */
inline constexpr size_t RAGGED_CONTEST_ROUNDS = 9;
inline constexpr size_t RAGGED_CONTEST_TRIMMED = 1;
/** 1324: the winner is the first candidate in list order whose contest time is within this fraction of the best (user
 *  decision 2026-11-08, from the 1323 audit). */
inline constexpr double RAGGED_TIE_MARGIN = 0.01;

/** The family that runs `site`'s input dtype and D, or nullopt when neither does: KR for rows under 16 bytes
 *  (VALUE: D 1 to raggedKRMaxComponents; ARG: raggedKRArgSupported), RK from 16 bytes (VALUE: rows up to
 *  RAGGED_RK_MAX_ROW_BYTES; ARG: raggedRKArgSupported). */
[[nodiscard]] std::optional<RaggedFamily> raggedFamilyFor(bool arg, DataType input_dtype, uint64_t components);

/** 1324: the contest's synthetic offsets (decision 4): offsets[i] = floor(i N / B) for i = 0 to B, clamped to the
 *  range of `offsets_dtype` (UINT32 or UINT64). */
[[nodiscard]] std::vector<uint64_t> raggedUniformOffsets(uint64_t capacity_rows, uint64_t segments,
                                                         DataType offsets_dtype);

/** The candidates, the audit's shapes, the most frequent winner first: KR i16_c0_e12, i16_c0_e4, i16_c8_e4; RK
 *  ta_c0_e4, t64_c0_e4, t512_c0_e4. The same lists serve VALUE and ARG. */
[[nodiscard]] std::vector<RaggedKRShape> raggedKRCandidateShapes();
[[nodiscard]] std::vector<RaggedRKShape> raggedRKCandidateShapes();

/** The selection of each candidate of `family`, in list order (candidate_count 0, winner_name the shape's name). */
[[nodiscard]] std::vector<RaggedSelection> raggedCandidateSelections(RaggedFamily family);

/** The shape name of a selection: raggedKRShapeName or raggedRKShapeName. */
[[nodiscard]] std::string raggedSelectionShapeName(const RaggedSelection& selection);

/** What a selection launches for a site: the KR grid or the RK CTA budget, and the workspace it needs. */
struct RaggedSelectedLaunch {
    uint32_t grid_or_ctas = 0;
    uint64_t workspace_bytes = 0;
};
/** Throws std::invalid_argument when the selection's family does not run the site. */
[[nodiscard]] RaggedSelectedLaunch raggedSelectedLaunch(const RaggedSelection& selection, const RaggedSite& site,
                                                        int gpu);

/** Runs `selection` for `site` (the main kernel and the finisher): launchRaggedKRValue / launchRaggedRKValue /
 *  launchRaggedKRArg / launchRaggedRKArg with `launch`'s grid or CTA budget. The tensors are the launchers'; the site's
 *  operation is the one run. Allocation-free. */
void launchRaggedSelection(const RaggedSelection& selection,
                           const RaggedSite& site,
                           const RaggedSelectedLaunch& launch,
                           const Tensor& input,
                           const Tensor& offsets,
                           Tensor& output,
                           Tensor& workspace,
                           Stream& stream);

/** The cache key of `site` on `stream`'s GPU (kind RaggedKR or RaggedRK; the depth always Regular). Throws
 *  std::invalid_argument for a site no family runs. */
[[nodiscard]] DenseReductionSelectionKey raggedSelectionKey(const RaggedSite& site, const Stream& stream);

/**
 * The cached selection for `site`, else the contest's winner (cached). Throws std::invalid_argument for a site no
 * family runs, an operation, offsets or output dtype the launchers do not take, an input head that is not a multiple of
 * the element's bytes below 16, or a site every candidate refuses. A site without rows or segments has nothing to time:
 * it returns the family's first candidate (candidate_count 0) and caches nothing.
 */
[[nodiscard]] RaggedSelection selectCachedRagged(const RaggedSite& site, const Stream& stream);

/** 1325: true when selectCachedRagged takes `site` (a family runs it and its operation and dtypes are the launchers'),
 *  without throwing. */
[[nodiscard]] bool raggedSiteRoutable(const RaggedSite& site);

/** 1325 (phase 7): a routed offset-segmented stamp's ragged plan (CubReduction.cpp): its site, the
 *  selection and the launch. Immutable once stamped. */
struct RaggedRoutedStamp {
    RaggedSite site;
    RaggedSelection selection;
    RaggedSelectedLaunch launch;
};

/** 1325: the ragged plan of an offset-segmented stamp: input [N, D...], output [B, D...] (the value dtype, or the index
 *  dtype when `arg`), offsets [B + 1]. Runs the selector (a contest on a miss). 1330: throws std::invalid_argument,
 *  with the reason, for a site no ragged family runs (1325-1329: nullptr, and the legacy kernels ran it): rows over
 *  2 GiB, more than 2^30 components, or an input address not a multiple of its element's bytes.
 *  The input's address mod 16 is part of the site, so the stamp is valid for the tensors it was stamped on and others
 *  of the same alignment. */
[[nodiscard]] std::shared_ptr<const RaggedRoutedStamp> raggedRouteSegmented(bool arg,
                                                                            CubReductionOp value_op,
                                                                            CubArgReductionOp arg_op,
                                                                            const Tensor& input,
                                                                            const Tensor& output,
                                                                            const Tensor& offsets,
                                                                            const Stream& stream);

/** 1325: runs a routed stamp on `input` (viewed as [N] or [N, D]), `offsets`, `output` and the stamp's `workspace`
 *  (at least routed.launch.workspace_bytes). Allocation-free; two launches on `stream`. */
void runRaggedRouted(const RaggedRoutedStamp& routed,
                     const Tensor& input,
                     const Tensor& offsets,
                     Tensor& output,
                     Tensor& workspace,
                     Stream& stream);

/** 1325: a routed stamp's passes (diagnostic): the winner's name, then the finisher's. */
[[nodiscard]] std::vector<std::string> raggedRoutedPassNames(const RaggedRoutedStamp& routed);

}  // namespace ThorImplementation::CubReductionInternal
