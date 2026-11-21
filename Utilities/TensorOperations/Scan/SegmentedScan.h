#pragma once

// Thor's segmented scan: the inclusive or exclusive, forward or reverse sum, product, min, max, argmax or argmin of
// every row of a tensor, rows being dense (every `row_length` elements of the flat tensor) or ragged (segment offsets),
// in one pass over the tensor (one read and one write per element). Floats accumulate in FP32 and round once into the
// output; UINT32 accumulates in UINT32 (sums and products wrap). Results are bitwise reproducible run to run. Min and
// max skip NaN: an output is NaN only while every element so far in its row is NaN. Argmax and argmin write the flat
// index (into the whole tensor) of the largest or smallest element so far in its row: the first of equal elements, -0
// equal to +0, and NaN below every number (the row's first element while every element so far is NaN). Given a values
// tensor, they also write that element's value in the same pass, bit for bit (a NaN as the canonical NaN).
//
// A reverse scan runs each row from its last element to its first: "so far" counts from the row's end, and of equal
// elements an arg scan takes the first it meets, the last in the row (as CUB's reverse arg scans do).
//
// An exclusive scan writes for each element the inclusive result of the element before it in its row (reverse: after
// it), and for a row's first element (reverse: its last) the op's identity, as CUB's scans do: 0 for a sum, 1 for a
// product; for a min (max) +inf (-inf), or for E4M3, which has no infinity, its largest (lowest) value, +-448; for
// UINT32 its largest value (0). An exclusive arg scan writes the index 0xFFFFFFFF there, and its value is the min's
// identity (argmin) or the max's (argmax).
//
// Dtypes: a scan writes its own dtype (FP32, UINT32, FP16, BF16, FP8 E4M3, FP8 E5M2), or a float sum or product writes
// any other float dtype (its FP32 accumulator rounded once into it; FP8 saturating); an arg scan writes UINT32 indices
// (and values in the input's dtype), and takes at most 2^31 elements, as CUB's scans do. Input and output must be on the same GPU, have
// the same element count, and start on 16-byte aligned addresses. Ragged offsets are UINT32 or UINT64, [rows + 1] of
// them, nondecreasing, and read at every launch. The rows end at offsets[rows]: elements from there on are neither read
// nor written, and the work follows the elements before it. Elements before offsets[0] are a row of their own.

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Scan/SegmentedScanLaunch.h"

#include <cstdint>
#include <optional>

namespace ThorImplementation {

class SegmentedScan {
   public:
    /**
     * Rows of `row_length` elements; row_length must divide the element count (row_length = the count: one row).
     * arg_values: an arg scan's winning values, written in the same pass (isValueOutputSupported); none by default.
     */
    SegmentedScan(SegmentedScanOp op,
                  SegmentedScanMode mode,
                  SegmentedScanDirection direction,
                  const Tensor& input,
                  const Tensor& output,
                  uint64_t row_length,
                  const std::optional<Tensor>& arg_values = std::nullopt);
    /** Ragged rows: segment_offsets is a rank-1 UINT32 or UINT64 tensor of rows + 1 offsets. */
    SegmentedScan(SegmentedScanOp op,
                  SegmentedScanMode mode,
                  SegmentedScanDirection direction,
                  const Tensor& input,
                  const Tensor& output,
                  const Tensor& segment_offsets,
                  const std::optional<Tensor>& arg_values = std::nullopt);

    // A copy would share the workspace, and two scans' launches on one workspace may not overlap: there is none.
    SegmentedScan(const SegmentedScan&) = delete;
    SegmentedScan& operator=(const SegmentedScan&) = delete;

    /** Whether the kernel takes this op from this input dtype to this output dtype (in either mode and direction). */
    [[nodiscard]] static bool isSupported(SegmentedScanOp op, DataType input_dtype, DataType output_dtype);
    /**
     * Whether the kernel takes these tensors: supported dtypes, the same GPU and element count, 16-byte aligned, and
     * for an arg scan at most 2^31 elements.
     */
    [[nodiscard]] static bool isSupported(SegmentedScanOp op, const Tensor& input, const Tensor& output);
    /** Whether an arg scan can write its values to arg_values: the input's dtype, GPU and count, 16-byte aligned. */
    [[nodiscard]] static bool isValueOutputSupported(SegmentedScanOp op,
                                                     const Tensor& input,
                                                     const Tensor& arg_values);
    [[nodiscard]] static bool isOffsetDataTypeSupported(DataType dtype);

    /**
     * Launches the scan on `stream`. Launches of one SegmentedScan must not overlap: run them on one stream, or order
     * them, as with any scan workspace. Changes no host state, so it may be captured in a CUDA graph and replayed, and
     * replays and other launches mix in any order.
     */
    void runOn(Stream& stream) const;

    [[nodiscard]] uint64_t getTileElements() const { return tile_elements; }
    [[nodiscard]] uint64_t getNumTiles() const { return num_tiles; }
    /** 0 when no tile waits on another (dense rows whose length divides the tile). */
    [[nodiscard]] uint64_t getWorkspaceSizeInBytes() const;

   private:
    SegmentedScan(SegmentedScanOp op,
                  SegmentedScanMode mode,
                  SegmentedScanDirection direction,
                  const Tensor& input,
                  const Tensor& output,
                  const std::optional<Tensor>& arg_values);
    void makeWorkspace();

    SegmentedScanOp op;
    SegmentedScanMode mode;
    SegmentedScanDirection direction;
    Tensor input;
    mutable Tensor output;      // written by runOn
    mutable Tensor arg_values;  // written by runOn; uninitialized when not asked for
    Tensor segment_offsets;  // uninitialized for dense rows
    uint64_t num_items = 0;
    uint64_t row_length = 0;  // dense rows
    uint32_t rows = 0;        // ragged rows
    uint64_t tile_elements = 0;
    uint64_t num_tiles = 0;
    Tensor workspace;  // UINT64 published runs; uninitialized when no tile waits
    SegmentedScanKernels::Workspace workspace_args;  // its words; null when no tile waits
};

}  // namespace ThorImplementation
