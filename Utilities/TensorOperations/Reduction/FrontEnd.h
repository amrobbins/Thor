#pragma once

// The front end's shared helpers: argument and output checks, the view problems the dense planner runs, and
// the pitched Complete launch. Internal to Reduction/; the definitions are in FrontEnd.cpp, Layout/ViewSplit.cpp and
// Layout/ReductionGeometry.cpp.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StageSemantics.h"
#include "Utilities/TensorOperations/Reduction/Layout/ReductionGeometry.h"
#include "Utilities/TensorOperations/Reduction/Layout/ViewSplit.h"
#include "Utilities/TensorOperations/Reduction/ReductionTypes.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::ReductionFrontEnd {

// The dense problem a view reduction is in storage order. A compact physical permutation whose retained axes keep
// their physical order is the bytes of a dense tensor reduced over its physical axes, and its outputs are already in
// the logical output's order. That covers
// - (A) every axis reduced, a dense [input_elements] over axis 0;
// - (B) the reduced axes the physical suffix, a dense [output_elements, reduction_size] over axis 1;
// - (D) natural-output permutation-aware views: one reduced physical block with an inner block, a dense
//   [outer, R, inner] over the R axis (the Thor tiled kernels were 1.04-15,000x slower in every audit cell);
// - (E) split physical reductions: the reduced axes several physical runs.
// The physical non-singleton axes are collapsed into alternating retained / reduced runs. The problem runs the dense
// planner on StridedSpan::asDense of the input: the dense tensor's kernels, selection cache entries and bits. A
// permuted output order (a rotated output) returns nothing here: the reducer refuses it, and the caller runs it in two
// parts (refusedViewSplit).
struct ViewPhysicalDenseProblem {
    std::vector<uint64_t> dimensions;
    std::vector<uint32_t> axes;
};

// The runs of a compact physical permutation (geometry.physical_layout_is_dense_permutation), if its retained axes keep
// their physical order and it reduces at least one non-singleton axis; else nothing.
[[nodiscard]] std::optional<ViewPhysicalDenseProblem> densePermutationRuns(const std::vector<uint64_t>& dimensions,
                                                                          const ReductionGeometry& geometry);

[[nodiscard]] std::optional<ViewPhysicalDenseProblem> viewPhysicalDenseProblem(const std::vector<uint64_t>& dimensions,
                                                                              bool dense_contiguous,
                                                                              const ReductionGeometry& geometry);

[[nodiscard]] std::optional<ViewPhysicalDenseProblem> viewPhysicalDenseProblem(const StridedSpan& input,
                                                                              const ReductionGeometry& geometry);

// The views the reducer refuses and how a caller runs them in two parts (ViewSplit, Layout/ViewSplit.h): the caller
// does the layout part with its dense copy (copyToDenseAsync), then a dense stamp. (Tiled kernels for these views ran
// 5-22,000x slower than the dense stamp of the same bytes.)
// - rotated (F: a compact permutation [O, R, I] with output [I, O]; G: [A, R, B, P] with output
//   [B, A, P]): the bytes are the dense [a, r, b x p] over axis 1, which writes [a, b, p]; the output is that read as
//   [b, a, p].
// - packed (H: rows of contiguous elements with gaps between them): the caller reduces the
//   view's dense copy (its own dimensions and dtype) over the same axes. When the view's storage is
//   known, slicedRowSplit may read the rows at their pitch instead (ViewSplit::Kind::SlicedOutput).
// A compact permutation whose retained axes keep their physical order is not refused (viewPhysicalDenseProblem).
[[nodiscard]] std::optional<ViewSplit> refusedViewSplit(const std::vector<uint64_t>& dimensions,
                                                        bool dense_contiguous,
                                                        const ReductionGeometry& geometry);

[[nodiscard]] bool isSupportedFloatingStorageDType(DataType dtype);

void requireSupportedFloatingStorageDType(DataType dtype, const char* role);

[[nodiscard]] bool isSupportedArgIndexDType(DataType dtype);

void requireSupportedArgIndexDType(DataType dtype);

// Tensor or StridedSpan (both read their placement the same way).
template <typename TensorLike>
void requireCompatibleStream(const TensorLike& input, const Stream& stream) {
    if (!stream.isInitialized()) {
        throw std::invalid_argument("Tensor reduction requires an initialized stream.");
    }
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Tensor reduction stream must belong to the input tensor's GPU.");
    }
}

template <typename TensorLike>
void requireGpuTensorView(const TensorLike& tensor, const char* name) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string(name) + " must be initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(name) + " must be a GPU tensor.");
    }
}

[[nodiscard]] std::vector<uint64_t> stripSingletonDimensions(const std::vector<uint64_t>& dimensions);

[[nodiscard]] bool isDirectDenseReductionPath(ReductionLayout path);

[[nodiscard]] ReductionInternal::ReductionStageRole physicalStageRole(bool apply_input_transform,
                                                                             bool apply_output_finalize);

struct TensorReachableAddressSpan {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

template <typename TensorLike>
[[nodiscard]] TensorReachableAddressSpan tensorReachableAddressSpan(const TensorLike& tensor) {
    const std::vector<uint64_t> dimensions = tensor.getDimensions();
    const std::vector<uint64_t> strides = tensor.getStridesElements();
    if (dimensions.size() != strides.size() || dimensions.empty()) {
        throw std::logic_error("Tensor reduction received invalid tensor view metadata while checking storage overlap.");
    }

    uint64_t max_element_offset = 0;
    for (size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
        const uint64_t extent_minus_one = dimensions[dimension] - 1;
        if (extent_minus_one != 0
            && strides[dimension] > (std::numeric_limits<uint64_t>::max() - max_element_offset) / extent_minus_one) {
            throw std::invalid_argument("Tensor reduction tensor view address span overflows uint64_t.");
        }
        max_element_offset += extent_minus_one * strides[dimension];
    }

    if (max_element_offset == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Tensor reduction tensor view address span overflows uint64_t.");
    }
    const uint64_t span_elements = max_element_offset + 1;
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(tensor.getDataType()));
    if (span_elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
        throw std::invalid_argument("Tensor reduction tensor view byte span overflows uint64_t.");
    }
    const uint64_t span_bytes = span_elements * element_bytes;

    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.template getMemPtr<void>());
    if (span_bytes > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - begin)) {
        throw std::invalid_argument("Tensor reduction tensor view address span exceeds the process address space.");
    }
    return TensorReachableAddressSpan{begin, begin + static_cast<uintptr_t>(span_bytes)};
}

template <typename Lhs, typename Rhs>
[[nodiscard]] bool tensorStorageOverlaps(const Lhs& lhs, const Rhs& rhs) {
    // Tensor::getArraySizeInBytes() is the logical payload size. For alias views that may be much larger than the
    // physical address span (broadcast/zero-stride) or smaller than it (gapped/strided). Safety checks therefore use
    // the conservative interval containing every address the view can actually reach. Strides are non-negative in
    // Tensor, so the interval begins at getMemPtr() and ends at the maximum strided coordinate plus one element.
    const TensorReachableAddressSpan lhs_span = tensorReachableAddressSpan(lhs);
    const TensorReachableAddressSpan rhs_span = tensorReachableAddressSpan(rhs);
    return lhs_span.begin < rhs_span.end && rhs_span.begin < lhs_span.end;
}

void requireDenseContiguousReductionOutput(const Tensor& output, const char* role);

void requireSupportedArgOperation(ArgReductionOp op);

void requireSupportedOperation(ReductionOp op);

[[nodiscard]] size_t checkedWorkspaceAdd(size_t lhs, size_t rhs, const char* message);

// The view's rows as the 3-D span [O, R, I] at strides [So, Sr, 1] from its first element (the view's own elements; the
// pitched classification orders its logical axes physically, so element [0, ..., 0] is the first).
[[nodiscard]] StridedSpan pitchedRowsSpan(const StridedSpan& view, const PitchedLayout& pitched);

// One in-place PacketComplete launch of a pitched view's rows (`rows`, from pitchedRowsSpan) into the dense keepdims
// output.
void launchPitchedComplete(const PitchedCompleteLaunch& launch,
                           const ReductionInternal::ReductionStageSemantics& semantics,
                           const StridedSpan& rows,
                           const StridedSpan& output,
                           const ReductionGeometry& geometry,
                           float output_scale,
                           Stream& stream);

}  // namespace ThorImplementation::ReductionFrontEnd
