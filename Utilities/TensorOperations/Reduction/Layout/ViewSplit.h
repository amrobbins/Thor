#pragma once

// The two parts by which a caller runs a value reduction of a view the reducer refuses, and the in-place launch
// for a pitched view.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {

/**
 * The two parts by which a caller runs a value reduction of a view the reducer refuses. Reductions read their input and
 * write their output in physical order; a permutation is the dense copy's (copyToDenseAsync,
 * Utilities/TensorOperations/Copy/StridedCopy.h). Reducer::analyzeGeometry throws NotImplementedException for these
 * views, and Reducer::splitRefusedView says how to run them.
 */
struct ViewSplit {
    enum class Kind : uint8_t {
        // F (a compact permutation [O, R, I] whose output is [I, O]) and G ([A, R, B, P] whose output is [B, A, P]):
        // reduce the view's bytes, the dense dense_dimensions = [a, r, b x p] from its first element (input.asDense),
        // over dense_axes into a dense buffer_dimensions = [a, b x p] of the output dtype. The output, dense [b, a, p]
        // in its logical order (keepdims extents aside), is that buffer read with copy_dimensions {b, a, p} and
        // copy_strides {p, b x p, 1}: its dense copy.
        RotatedOutput,
        // H whose rows read at their pitch cost less than the packed copy: [O, R, I] rows of I contiguous elements at
        // row pitch Sr > I, with no gap between slabs (slab pitch R x Sr, or O = 1). Reduce the dense
        // dense_dimensions = [O, R, Sr] from the view's first element (input.withSourceLayout: it reaches Sr - I
        // elements past the view's last, inside its source tensor) over dense_axes into a dense buffer_dimensions = [O,
        // Sr]; the output [O, I] is that buffer read with copy_dimensions {O, I} and copy_strides {Sr, 1}. The gap
        // columns are reduced and dropped. Only Reducer::splitRefusedView of a span returns it.
        SlicedOutput,
        // H (rows of contiguous elements with gaps between them): reduce the view's dense copy (dense_dimensions, the
        // view's own, in its dtype) over dense_axes (the view's own).
        PackedInput,
    };
    Kind kind = Kind::RotatedOutput;
    std::vector<uint64_t> dense_dimensions;
    std::vector<uint32_t> dense_axes;
    // RotatedOutput / SlicedOutput: the dense reduction's output and how its copy reads it as the output.
    std::vector<uint64_t> buffer_dimensions;
    std::vector<uint64_t> copy_dimensions;
    std::vector<uint64_t> copy_strides;
    uint64_t a = 1, b = 1, p = 1;  // RotatedOutput
    // SlicedOutput: O, R, I and Sr above.
    uint64_t rows_outer = 1, rows = 1, row_width = 1, row_pitch = 1;

    // RotatedOutput and SlicedOutput reduce the view's bytes (or its source's) and copy the output; PackedInput copies
    // the input.
    [[nodiscard]] bool copiesOutput() const { return kind != Kind::PackedInput; }
    // RotatedOutput / SlicedOutput: what the first part reduces, the dense dense_dimensions from `view`'s first
    // element, and what the second part copies, the dense reduction's output `buffer` (buffer_dimensions) read as the
    // output's dense copy_dimensions.
    [[nodiscard]] StridedSpan reducedInput(const StridedSpan& view) const {
        if (kind == Kind::RotatedOutput) return view.asDense(dense_dimensions);
        if (kind == Kind::SlicedOutput)
            return view.withSourceLayout(dense_dimensions, StridedSpan::denseStrides(dense_dimensions));
        throw std::logic_error("ViewSplit::reducedInput: a packed split reduces the view's dense copy.");
    }
    [[nodiscard]] StridedSpan copySource(const Tensor& buffer) const {
        return StridedSpan(buffer).withLayout(copy_dimensions, copy_strides);
    }
};

/**
 * How the reducer reads a pitched view ([O, R, I] rows of I contiguous elements at row pitch Sr, slabs at pitch So) in
 * place: one RuntimeK PacketComplete launch at the view's pitches, warps_per_output (1 / 2 / 4 / 8) warps per [outer, K
 * tile] work item over R and K tiles of at most tile_components (FP32 124, FP16 / BF16 248, FP8 496). No workspace; the
 * output is the dense keepdims [O, I]. Reducer::pitchedCompleteLaunch says when.
 */
struct PitchedCompleteLaunch {
    uint32_t warps_per_output = 1;
    uint32_t tile_components = 1;
    bool operator==(const PitchedCompleteLaunch&) const = default;
};

}  // namespace ThorImplementation
