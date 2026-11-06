#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * 1146: exactly ONE retained original-input FP8 R / KR VALUE pass (launchFp8KRValueSegments). The site's rows are the
 * current runs before it and its row is the nominated (last) run. The pass is Complete for the planner: a staged shape
 * keeps its chunk partials in the pass's own scratch and finishes them in the same pass, so its output is the ordinary
 * [rows] result (final storage on the last pass, FP32 otherwise).
 * 1220: input_dtype FP32 is the retained FP32 R / KR VALUE pass (launchFp32KRValueSegments; the shape in bytes) at an
 * FP32 site, original or intermediate. 1254: input_dtype FP16 / BF16 is the retained 2-byte pass
 * (launchF16KRValueSegments; the shape in bytes, 2 per element) at an original FP16 / BF16 site.
 */
struct DenseRetainedFp8KRPhysicalPlan final : DenseReductionPhysicalPlan {
    Fp8KR::Fp8KRSegmentShape shape{};
    DensePassRole role = DensePassRole::Complete;
    DataType input_dtype = DataType::FP8_E4M3;
    std::vector<uint64_t> input_dimensions;  // the current run extents
    uint32_t reduction_axis = 0;             // the last run
};

/** The [rows, row_bytes] of an R or KR site (rows 1 for R). */
struct Fp8KRSiteGeometry {
    uint64_t rows = 1;
    uint64_t row_bytes = 0;
};
[[nodiscard]] Fp8KRSiteGeometry fp8KRSiteGeometry(const DenseReductionProblem& problem, DenseReductionSite site);

/** True for an original-input FP8 R or KR site (the retained R / KR selectors' domain; semantics are checked there). */
[[nodiscard]] bool isOriginalFp8KRSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                       DataType original_input_dtype);

/** The one-pass VALUE planner candidate running `shape` at the site. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp8KRValueCandidate(const DenseReductionProblem& problem,
                                                                     DenseReductionSite site,
                                                                     const Fp8KR::Fp8KRSegmentShape& shape,
                                                                     DataType input_dtype);

/** The one-pass ARG planner candidate running `shape` at the site (a DenseRetainedFp8ArgPhysicalPlan with `kr`). */
[[nodiscard]] DenseReductionCandidate makeRetainedFp8KRArgCandidate(const DenseReductionProblem& problem,
                                                                   DenseReductionSite site,
                                                                   const Fp8KR::Fp8KRSegmentShape& shape,
                                                                   DataType input_dtype);

/** 1220: the one-pass VALUE candidate running the retained FP32 R / KR `shape` (in bytes: row_bytes = 4 x the row's
 *  elements) at an FP32 site: original FP32 input, or an intermediate aggregate of any dtype's earlier passes (the
 *  caller, the retained FP32 routing, has established that). The plan's input_dtype is FP32. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp32KRValueCandidate(const DenseReductionProblem& problem,
                                                                      DenseReductionSite site,
                                                                      const Fp8KR::Fp8KRSegmentShape& shape);

/** 1254: the one-pass VALUE candidate running the retained FP16 / BF16 R / KR `shape` (in bytes: row_bytes = 2 x the
 *  row's elements) at an original `input_dtype` (FP16 or BF16) site. */
[[nodiscard]] DenseReductionCandidate makeRetainedF16KRValueCandidate(const DenseReductionProblem& problem,
                                                                     DenseReductionSite site,
                                                                     const Fp8KR::Fp8KRSegmentShape& shape,
                                                                     DataType input_dtype);

}  // namespace ThorImplementation::CubReductionInternal
