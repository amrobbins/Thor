// Offset-segmented (ragged) VALUE and ARG reduction: the four segmented classes.

#include "Utilities/TensorOperations/Reduction/SegmentedReducer.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/TensorRequirements.h"
#include "Utilities/TensorOperations/Reduction/FrontEnd.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedStampAutotuner.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

namespace {

using namespace TensorRequirements;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

[[nodiscard]] uint64_t segmentedElementsPerValue(const Tensor& input) {
    const std::vector<uint64_t>& dimensions = input.getDimensions();
    if (dimensions.empty()) {
        throw std::invalid_argument("Segmented reduction requires values with rank >= 1.");
    }
    uint64_t elements_per_value = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] != 0
            && elements_per_value > std::numeric_limits<uint64_t>::max() / dimensions[axis]) {
            throw std::invalid_argument("Segmented-reduction trailing value size overflows uint64_t.");
        }
        elements_per_value *= dimensions[axis];
    }
    if (elements_per_value == 0) {
        throw std::invalid_argument("Segmented reduction requires a non-zero trailing value size.");
    }
    return elements_per_value;
}

void requireSupportedSegmentedOperation(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
        case ReductionOp::Mean:
        case ReductionOp::Min:
        case ReductionOp::Max:
            return;
        case ReductionOp::Product:
        case ReductionOp::L1Norm:
        case ReductionOp::L2Norm:
        case ReductionOp::SumSquares:
            break;
    }
    throw std::invalid_argument("Offset-segmented reduction supports sum, mean, min, and max.");
}

void requireSegmentOffsets(const Tensor& input, const Tensor& segment_offsets, const Stream& stream) {
    requireDenseContiguousGpuTensor(segment_offsets, "segment offsets");
    requireSameGpuPlacement(input, segment_offsets, "input", "segment offsets");
    requireCompatibleStream(input, stream);
    if (!SegmentedReducer::isOffsetDataTypeSupported(segment_offsets.getDataType())) {
        throw std::invalid_argument("Segmented-reduction offsets must use UINT32 or an enabled UINT64 dtype.");
    }
    const std::vector<uint64_t>& dimensions = segment_offsets.getDimensions();
    if (dimensions.size() != 1 || dimensions[0] < 2) {
        throw std::invalid_argument(
            "Segmented-reduction offsets must be rank 1 with shape [num_segments + 1] and at least one segment.");
    }
    if (tensorStorageOverlaps(input, segment_offsets)) {
        throw std::invalid_argument("Segmented-reduction input and offsets storage must not overlap.");
    }
}

void validateSegmentOffsetContentsAtStamp(const Tensor& input,
                                          const Tensor& segment_offsets,
                                          const Stream& stream) {
    Tensor cpu_offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), segment_offsets.getDescriptor());
    cpu_offsets.copyFromAsync(segment_offsets, stream);
    stream.synchronize();

    const uint64_t offset_count = segment_offsets.getTotalNumElements();
    auto validate = [&](auto* offsets) {
        if (static_cast<uint64_t>(offsets[0]) != 0) {
            throw std::invalid_argument("Segmented-reduction offsets must begin at zero.");
        }
        uint64_t previous = 0;
        for (uint64_t i = 1; i < offset_count; ++i) {
            const uint64_t current = static_cast<uint64_t>(offsets[i]);
            if (current < previous) {
                throw std::invalid_argument("Segmented-reduction offsets must be nondecreasing.");
            }
            if (current > input.getDimensions()[0]) {
                throw std::invalid_argument("Segmented-reduction offset exceeds the values tensor row capacity.");
            }
            previous = current;
        }
    };

    switch (segment_offsets.getDataType()) {
        case DataType::UINT32:
            validate(cpu_offsets.getMemPtr<uint32_t>());
            return;
        case DataType::UINT64:
            validate(cpu_offsets.getMemPtr<uint64_t>());
            return;
        default:
            throw std::invalid_argument("Segmented-reduction offsets must use UINT32 or UINT64.");
    }
}

void requireExpectedSegmentedOutput(const Tensor& input,
                                    const Tensor& output,
                                    const Tensor& segment_offsets,
                                    DataType output_dtype,
                                    uint64_t num_segments) {
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument(
            "Segmented-reduction preallocated output dtype does not match the configured output dtype.");
    }

    if (input.getDimensions().size() == 1) {
        if (output.getTotalNumElements() != num_segments
            || stripSingletonDimensions(output.getDimensions())
                   != stripSingletonDimensions(std::vector<uint64_t>{num_segments})) {
            throw std::invalid_argument(
                "Segmented-reduction scalar output must preserve the [num_segments] non-singleton layout.");
        }
    } else {
        std::vector<uint64_t> expected_dimensions = input.getDimensions();
        expected_dimensions[0] = num_segments;
        if (output.getDimensions() != expected_dimensions) {
            throw std::invalid_argument("Segmented-reduction vector output must have shape [num_segments,D...].");
        }
    }

    if (tensorStorageOverlaps(input, output) || tensorStorageOverlaps(segment_offsets, output)) {
        throw std::invalid_argument(
            "Segmented-reduction output storage must not overlap input or offsets storage.");
    }
}

void requireExpectedSegmentedArgOutput(const Tensor& input,
                                       const Tensor& output,
                                       const Tensor& segment_offsets,
                                       DataType output_dtype,
                                       uint64_t num_segments) {
    requireDenseContiguousGpuTensor(output, "index output");
    requireSameGpuPlacement(input, output, "input", "index output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument("Segmented arg-reduction output dtype does not match the configured index dtype.");
    }
    std::vector<uint64_t> expected_dimensions = input.getDimensions();
    expected_dimensions[0] = num_segments;
    if (output.getDimensions() != expected_dimensions) {
        throw std::invalid_argument(
            "Segmented arg-reduction output must have shape [num_segments,D...].");
    }
    if (tensorStorageOverlaps(input, output) || tensorStorageOverlaps(segment_offsets, output)) {
        throw std::invalid_argument("Segmented arg-reduction output storage must not overlap input or offsets storage.");
    }
}

}  // namespace

SegmentedReducer::SegmentedReducer(ReductionOp op,
                                                       std::optional<DataType> output_dtype)
    : op(op), output_dtype(output_dtype) {
    requireSupportedSegmentedOperation(op);
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "segmented output");
    }
}

bool SegmentedReducer::isInputDataTypeSupported(DataType dtype) {
    return isSupportedFloatingStorageDType(dtype);
}

bool SegmentedReducer::isOffsetDataTypeSupported(DataType dtype) {
    if (dtype == DataType::UINT32) {
        return true;
    }
    if (dtype == DataType::UINT64) {
        return true;
    }
    return false;
}

DataType SegmentedReducer::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "segmented input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "segmented output");
    return resolved;
}

std::shared_ptr<StampedSegmentedReducer> SegmentedReducer::stamp(
    const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    std::vector<uint64_t> output_dimensions = input.getDimensions();
    output_dimensions[0] = num_segments;
    Tensor output(input.getPlacement(),
                  TensorDescriptor(resolveOutputDataType(input.getDataType()), output_dimensions));
    return stampValidated(input, output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedSegmentedReducer> SegmentedReducer::stamp(
    const Tensor& input,
    const Tensor& preallocated_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedOutput(input,
                                   preallocated_output,
                                   segment_offsets,
                                   resolveOutputDataType(input.getDataType()),
                                   num_segments);
    return stampValidated(input, preallocated_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedSegmentedReducer> SegmentedReducer::stampRuntimeOffsets(
    const Tensor& input,
    const Tensor& preallocated_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented input");
    requireSegmentOffsets(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedOutput(input,
                                   preallocated_output,
                                   segment_offsets,
                                   resolveOutputDataType(input.getDataType()),
                                   num_segments);
    return stampValidated(input, preallocated_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedSegmentedReducer> SegmentedReducer::stampValidated(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_segments,
    const Stream& stream) const {
    const uint64_t num_items = input.getTotalNumElements();
    if (input.getDimensions().size() == 1
        && (num_items > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            || num_segments > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
        throw std::invalid_argument("Scalar segmented-reduction item or segment count exceeds int64 limits.");
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());
    // The stamp runs the selector's ragged winner; its boundary workspace is the stamp's temporary storage. Every site;
    // raggedRouteSegmented throws, with the reason, for a site no ragged family runs (rows over 2 GiB).
    std::shared_ptr<const ReductionInternal::RaggedRoutedStamp> routed = ReductionInternal::raggedRouteSegmented(
        false, op, ArgReductionOp::ArgMin, input, output, segment_offsets, stream);
    const size_t bytes = static_cast<size_t>(routed->launch.workspace_bytes);
    Tensor workspace(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(bytes)}));
    std::shared_ptr<StampedSegmentedReducer> stamped(new StampedSegmentedReducer(
        op, input, output, segment_offsets, num_items, num_segments, bytes, workspace, stream));
    stamped->ragged = std::move(routed);
    return stamped;
}

StampedSegmentedReducer::StampedSegmentedReducer(ReductionOp op,
                                                 const Tensor& input,
                                                 const Tensor& output,
                                                 const Tensor& segment_offsets,
                                                 uint64_t num_items,
                                                 uint64_t num_segments,
                                                 size_t temp_storage_bytes,
                                                 const Tensor& temp_storage,
                                                 const Stream& stream)
    : op(op),
      input(input),
      output(output),
      segment_offsets(segment_offsets),
      num_items(num_items),
      num_segments(num_segments),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedSegmentedReducer::run() { runOn(stream); }

void StampedSegmentedReducer::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    ReductionInternal::runRaggedRouted(*ragged, input_span, offsets_span, output_span, workspace_span, run_stream);
}

SegmentedArgReducer::SegmentedArgReducer(ArgReductionOp op, DataType index_output_dtype)
    : op(op), index_output_dtype(index_output_dtype) {
    requireSupportedArgOperation(op);
    requireSupportedArgIndexDType(index_output_dtype);
}

bool SegmentedArgReducer::isInputDataTypeSupported(DataType dtype) {
    return isSupportedFloatingStorageDType(dtype);
}

bool SegmentedArgReducer::isOffsetDataTypeSupported(DataType dtype) {
    if (dtype == DataType::UINT32) {
        return true;
    }
    if (dtype == DataType::UINT64) {
        return true;
    }
    return false;
}

std::shared_ptr<StampedSegmentedArgReducer> SegmentedArgReducer::stamp(
    const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented arg input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    std::vector<uint64_t> output_dimensions = input.getDimensions();
    output_dimensions[0] = num_segments;
    Tensor output(input.getPlacement(), TensorDescriptor(index_output_dtype, output_dimensions));
    return stampValidated(input, output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedSegmentedArgReducer> SegmentedArgReducer::stamp(
    const Tensor& input,
    const Tensor& preallocated_index_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented arg input");
    requireSegmentOffsets(input, segment_offsets, stream);
    validateSegmentOffsetContentsAtStamp(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedArgOutput(input, preallocated_index_output, segment_offsets, index_output_dtype, num_segments);
    return stampValidated(input, preallocated_index_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedSegmentedArgReducer> SegmentedArgReducer::stampRuntimeOffsets(
    const Tensor& input,
    const Tensor& preallocated_index_output,
    const Tensor& segment_offsets,
    const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    static_cast<void>(segmentedElementsPerValue(input));
    requireSupportedFloatingStorageDType(input.getDataType(), "segmented arg input");
    requireSegmentOffsets(input, segment_offsets, stream);
    const uint64_t num_segments = segment_offsets.getDimensions()[0] - 1;
    requireExpectedSegmentedArgOutput(input, preallocated_index_output, segment_offsets, index_output_dtype, num_segments);
    return stampValidated(input, preallocated_index_output, segment_offsets, num_segments, stream);
}

std::shared_ptr<StampedSegmentedArgReducer> SegmentedArgReducer::stampValidated(
    const Tensor& input,
    const Tensor& index_output,
    const Tensor& segment_offsets,
    uint64_t num_segments,
    const Stream& stream) const {
    const uint64_t num_items = input.getTotalNumElements();
    if (index_output_dtype == DataType::UINT32 && num_items > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Segmented arg-reduction packed index domain does not fit in UINT32.");
    }
    if (num_segments > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("Segmented arg-reduction segment count exceeds int64 limits.");
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());
    // As SegmentedReducer::stampValidated, with the ragged ARG kernels (every site).
    std::shared_ptr<const ReductionInternal::RaggedRoutedStamp> routed = ReductionInternal::raggedRouteSegmented(
        true, ReductionOp::Min, op, input, index_output, segment_offsets, stream);
    const size_t bytes = static_cast<size_t>(routed->launch.workspace_bytes);
    Tensor workspace(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(bytes)}));
    std::shared_ptr<StampedSegmentedArgReducer> stamped(new StampedSegmentedArgReducer(
        op, input, index_output, segment_offsets, num_segments, bytes, workspace, stream));
    stamped->ragged = std::move(routed);
    return stamped;
}

StampedSegmentedArgReducer::StampedSegmentedArgReducer(ArgReductionOp op,
                                                       const Tensor& input,
                                                       const Tensor& index_output,
                                                       const Tensor& segment_offsets,
                                                       uint64_t num_segments,
                                                       size_t temp_storage_bytes,
                                                       const Tensor& temp_storage,
                                                       const Stream& stream)
    : op(op),
      input(input),
      index_output(index_output),
      segment_offsets(segment_offsets),
      num_segments(num_segments),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedSegmentedArgReducer::run() { runOn(stream); }

void StampedSegmentedArgReducer::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    ReductionInternal::runRaggedRouted(*ragged, input_span, offsets_span, index_output_span, workspace_span,
                                       run_stream);
}

std::vector<std::string> StampedSegmentedReducer::getPassNames() const {
    return ReductionInternal::raggedRoutedPassNames(*ragged);
}

std::vector<std::string> StampedSegmentedArgReducer::getPassNames() const {
    return ReductionInternal::raggedRoutedPassNames(*ragged);
}

}  // namespace ThorImplementation
