#pragma once

// 1366 (phase 2, plans/Thor_View_Reduction_Plan_2026-11-08.md): case C, the full VALUE reduction of a rank-1 strided
// view (n elements at a stride of s elements: einsum diagonals, matrix columns, broadcasts), on Thor kernels instead of
// CUB's DeviceReduce::Reduce over an affine iterator.
//
// - Stride 0: the closed form (StridedRFamily::ClosedForm). One thread reads x once and evaluates the operation in FP64
//   (SUM x n, MEAN x, MIN / MAX x, PRODUCT x^n by squaring, L1 |x| n, SumSquares x^2 n, L2 |x| sqrt(n)), rounds once to
//   FP32, applies output_scale and stores the output dtype (decision 2). Additive results are 0 + the product, so a -0
//   input gives the +0 the summation gives.
// - Stride s >= 1: the StridedR kernel (CubStridedR.cu), one template over the input type and the combine (add,
//   product, min; MAX is the min of the negated values with a negated output_scale, as the dense KR kernels fold it).
//   Lane l of a warp reads logical element i + l, and each thread keeps STRIDED_R_UNROLL independent loads in flight,
//   one accumulator each; the input transform (identity, |x|, x^2) is a uniform runtime branch.
//   - Split: G CTAs, CTA c owning the contiguous logical range [c q + min(c, r), (c + 1) q + min(c + 1, r)) with
//     n = q G + r; each writes one FP32 partial (transformed, combined, not finalized, negated for MAX) to the
//     workspace. 1370: one launch. After its partial is stored and fenced, each CTA adds 1 to a ticket in the workspace
//     (one atomicAdd); the CTA that brings it to G is the last to arrive, so it combines the G partials in a fixed
//     order, finalizes, stores, and sets the ticket back to 0. Only the choice of the finishing CTA varies between
//     runs, so the result is still bit-identical from run to run on one stamp. The price is state: the ticket must be
//     0 when a launch starts, so a workspace is zeroed once (stridedRInitWorkspace) before its first launch.
//     (1366-1369: a second launch, the retained KR Split finisher, combined the partials.)
//
// Selection (1371): none. The launch follows from the view: ClosedForm at s == 0, else Split at G = stridedRSplitGrid
// (n) (the persistent grid, at most one CTA per 512 elements, so G = 1 for n <= 512). 1366-1370 also had Block (one
// CTA, no workspace) and timed it against Split at stamp (decision 3, kind StridedR in the dense selection cache).
// Once Split finished in one launch, Block had no advantage left: it ran at one SM's request rate, never faster than
// Split in the 1370 audit, and the contest, which could not resolve times of 2-4 us, picked it where it lost up to
// 1 us. One way to make the call, and no stamp-time timing.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/StridedRLaunch.h"

#include <cstdint>
#include <string>

namespace ThorImplementation::CubReductionInternal {

/** The StridedR CTA and the independent loads per thread: a CTA iteration covers STRIDED_R_TILE logical elements. */
inline constexpr uint32_t STRIDED_R_THREADS = 512;
inline constexpr uint32_t STRIDED_R_UNROLL = 8;
inline constexpr uint64_t STRIDED_R_TILE = uint64_t{STRIDED_R_THREADS} * STRIDED_R_UNROLL;

/** "strided_r:closed_form", "strided_r:split_g<G>". */
[[nodiscard]] std::string stridedRLaunchName(const StridedRLaunch& launch);

/** The input dtypes StridedR reads (FP8 E4M3 / E5M2, FP16, BF16, FP32) and the outputs it stores (the same five). */
[[nodiscard]] bool stridedRSupportsInput(DataType dtype);
[[nodiscard]] bool stridedRSupportsOutput(DataType dtype);

/** Split's persistent grid on `gpu`: SMs x the kernel's resident CTAs per SM (a device query). */
[[nodiscard]] uint32_t stridedRPersistentGrid(int gpu);
/** Split's grid for n elements: the persistent grid, at most one CTA per STRIDED_R_THREADS elements, at least 1. */
[[nodiscard]] uint32_t stridedRSplitGrid(uint64_t elements, int gpu);
/** A Split launch of `grid` CTAs (1 to 2^31 - 1; a forced grid larger than n leaves the extra CTAs' partials at the
 *  identity). Its workspace is grid FP32 partials and the 4-byte ticket. */
[[nodiscard]] StridedRLaunch stridedRSplitLaunch(uint32_t grid);

/** 1370: zeroes `workspace` on `stream` and waits for it. Whoever allocates a Split workspace calls this once, before
 *  the first launch that uses it (an empty span is a no-op). Launches leave the ticket at 0, so never again. */
void stridedRInitWorkspace(const StridedSpan& workspace, const Stream& stream);

/**
 * Reduces the rank-1 `input` (dims {n}, strides {s}, n >= 1) with `op` into element 0 of `output` (one element of any
 * supported output dtype), at `output_scale`. ClosedForm requires s == 0. `workspace` holds at least
 * launch.workspace_bytes, 4-byte aligned (an empty span for ClosedForm); for Split it must have been zeroed
 * by stridedRInitWorkspace, and no two launches may use one workspace at the same time. One launch, stream-ordered; no
 * allocation and no synchronization. Element addresses are 64-bit (base + i s elements).
 */
void launchStridedR(CubReductionOp op,
                    const StridedRLaunch& launch,
                    const StridedSpan& input,
                    const StridedSpan& output,
                    const StridedSpan& workspace,
                    float output_scale,
                    Stream& stream);

/**
 * 1371: the launch for a stamp reducing `input` (a nonempty rank-1 GPU span of a supported input dtype) into
 * `output_dtype` on `gpu`: ClosedForm when its stride is 0, else stridedRSplitLaunch(stridedRSplitGrid(n, gpu)).
 * Nothing is timed or cached. Throws std::invalid_argument for an input or output StridedR does not take.
 */
[[nodiscard]] StridedRLaunch stridedRLaunchFor(const StridedSpan& input, DataType output_dtype, int gpu);

}  // namespace ThorImplementation::CubReductionInternal
