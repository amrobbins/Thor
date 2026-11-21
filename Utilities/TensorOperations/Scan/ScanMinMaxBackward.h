#pragma once

// The backward of a min or max scan: for y = scan(x) (inclusive or exclusive, forward or reverse, dense or ragged rows,
// as SegmentedScan computes it), dx from x and dy in one pass, one read of x and dy and one write of dx per element.
// Each element's dy goes to its prefix's winner (the element its y was taken from: the first of equal elements met,
// -0 equal to +0, NaN below every number, as SegmentedScan's arg scans pick it), so dx[w] is the sum of dy over the
// elements whose winner is w, and 0 for an element that never wins. An exclusive scan's first element in a row has no
// winner, and its dy goes nowhere. The sums are taken in FP32 in a fixed order, so dx is bitwise reproducible.
//
// Dtypes: x, dy and dx are each FP32, FP16, BF16, FP8 E4M3 or FP8 E5M2, in any combination. x and dy are converted to
// FP32 as they are read (exact), and dx rounded once from its FP32 sum as it is written (to nearest; FP8 saturating to
// its finite range). The three tensors must be on the same GPU, have the same element count (at most 2^31, as the arg
// scans), and start on an address aligned to 8 elements (32 bytes in FP32). Ragged offsets are UINT32 or UINT64, [rows + 1] of them,
// nondecreasing, and read at every launch; the rows end at offsets[rows], and elements of dx from there on are not
// written. Elements before offsets[0] are a row of their own.

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Scan/ScanMinMaxBackwardLaunch.h"
#include "Utilities/TensorOperations/Scan/SegmentedScanLaunch.h"

#include <cstdint>

namespace ThorImplementation {

class ScanMinMaxBackward {
   public:
    /**
     * min: the backward of a min scan (a max scan's otherwise). Rows of `row_length` elements; row_length must divide
     * the element count.
     */
    ScanMinMaxBackward(bool min,
                       SegmentedScanMode mode,
                       SegmentedScanDirection direction,
                       const Tensor& x,
                       const Tensor& dy,
                       const Tensor& dx,
                       uint64_t row_length);
    /** Ragged rows: segment_offsets is a rank-1 UINT32 or UINT64 tensor of rows + 1 offsets. */
    ScanMinMaxBackward(bool min,
                       SegmentedScanMode mode,
                       SegmentedScanDirection direction,
                       const Tensor& x,
                       const Tensor& dy,
                       const Tensor& dx,
                       const Tensor& segment_offsets);

    // A copy would share the workspace, and two launches on one workspace may not overlap: there is none.
    ScanMinMaxBackward(const ScanMinMaxBackward&) = delete;
    ScanMinMaxBackward& operator=(const ScanMinMaxBackward&) = delete;

    /** Whether the kernel takes these dtypes of x, dy and dx: each a float. */
    [[nodiscard]] static bool isSupported(DataType x_dtype, DataType dy_dtype, DataType dx_dtype);

    /**
     * Launches the backward on `stream`. Launches of one ScanMinMaxBackward must not overlap: run them on one stream,
     * or order them. Changes no host state, so it may be captured in a CUDA graph and replayed.
     */
    void runOn(Stream& stream) const;

    [[nodiscard]] uint64_t getTileElements() const { return ScanMinMaxBackwardKernels::kGradientTile; }
    [[nodiscard]] uint64_t getNumTiles() const { return launch_args.keys.num_tiles; }
    /** 0 when no tile waits on another (dense rows whose length divides the tile). */
    [[nodiscard]] uint64_t getWorkspaceSizeInBytes() const;

   private:
    ScanMinMaxBackward(bool min,
                       SegmentedScanMode mode,
                       SegmentedScanDirection direction,
                       const Tensor& x,
                       const Tensor& dy,
                       const Tensor& dx);
    void makeWorkspace();

    SegmentedScanDirection direction;
    Tensor x;
    Tensor dy;
    mutable Tensor dx;       // written by runOn
    Tensor segment_offsets;  // uninitialized for dense rows
    Tensor workspace;        // UINT64 published runs and sums; uninitialized when no tile waits
    ScanMinMaxBackwardKernels::BackwardLaunch launch_args;
};

}  // namespace ThorImplementation
