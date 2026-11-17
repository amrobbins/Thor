#pragma once

// The public offset-segmented (ragged) reductions, VALUE and ARG: SegmentedReducer, SegmentedArgReducer and
// their stamps.

#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

class StampedSegmentedArgReducer;
class StampedSegmentedReducer;
namespace ReductionInternal {
struct RaggedRoutedStamp;  // RaggedStampAutotuner.h
}

/**
 * Describes an offset-segmented reduction over dense values [N,D...] -> [B,D...].
 *
 * Segment i is the half-open row range [offsets[i], offsets[i + 1]). Rank-1 input retains the scalar CUB fast path;
 * vector-valued input uses a Thor-owned coalesced CUDA backend under the same centralized API. Inputs are converted to
 * FP32 and all reduction arithmetic is performed in FP32. Empty segments use Reducer::getFp32EmptyReductionValue().
 * Mean divides by the segment row count and returns zero for an empty segment. The ordinary stamp() API validates
 * offset contents immediately. stampRuntimeOffsets() is for graph inputs whose contents are populated only after
 * stamping; callers of that path must preserve the same zero-based, nondecreasing, in-bounds row-partition contract at
 * execution.
 */
class SegmentedReducer {
   public:
    explicit SegmentedReducer(ReductionOp op,
                              std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] ReductionOp getOperation() const { return op; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
    [[nodiscard]] DataType resolveOutputDataType(DataType input_dtype) const;
    [[nodiscard]] static bool isInputDataTypeSupported(DataType dtype);
    [[nodiscard]] static bool isOffsetDataTypeSupported(DataType dtype);

    [[nodiscard]] std::shared_ptr<StampedSegmentedReducer> stamp(
        const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedSegmentedReducer> stamp(
        const Tensor& input,
        const Tensor& preallocated_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

    // Expression/network offsets are runtime inputs and therefore do not contain
    // meaningful row-partition data while the physical graph is being stamped.
    // This path validates tensor shape/type/storage but deliberately defers content
    // validity to the runtime row-partition producer.
    [[nodiscard]] std::shared_ptr<StampedSegmentedReducer> stampRuntimeOffsets(
        const Tensor& input,
        const Tensor& preallocated_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

   private:
    [[nodiscard]] std::shared_ptr<StampedSegmentedReducer> stampValidated(
        const Tensor& input,
        const Tensor& output,
        const Tensor& segment_offsets,
        uint64_t num_segments,
        const Stream& stream) const;

    ReductionOp op;
    std::optional<DataType> output_dtype;
};

/** Concrete, allocation-free-at-run-time offset-segmented reduction. */
class StampedSegmentedReducer {
   public:
    void run();
    void runOn(Stream& run_stream) const;
    /**
     * Benchmarks only: runs this stamp on another input, offsets and output of the stamp's dtypes and element counts,
     * with the stamp's own workspace, so the ragged audit can time a stamp on rotating cold views without re-stamping.
     * The replacements are spans (the audits' cold views are spans of one arena). Production binds its tensors.
     */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                  const StridedSpan& replacement_segment_offsets,
                                  const StridedSpan& replacement_output,
                                  Stream& run_stream) const;
    /** The physical passes this stamp runs, in order (diagnostic only). */
    [[nodiscard]] std::vector<std::string> getPassNames() const;

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] ReductionOp getOperation() const { return op; }
    [[nodiscard]] ReductionLayout getPath() const { return ReductionLayout::Segmented; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getOutputDataType() const { return output.getDataType(); }
    [[nodiscard]] DataType getAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] DataType getOffsetDataType() const { return segment_offsets.getDataType(); }
    [[nodiscard]] uint64_t getNumItems() const { return num_items; }
    [[nodiscard]] uint64_t getNumSegments() const { return num_segments; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    /** True when the stamp runs the ragged kernels (always). */
    [[nodiscard]] bool usesRetainedRagged() const { return ragged != nullptr; }

   private:
    friend class SegmentedReducer;

    StampedSegmentedReducer(ReductionOp op,
                            const Tensor& input,
                            const Tensor& output,
                            const Tensor& segment_offsets,
                            uint64_t num_items,
                            uint64_t num_segments,
                            size_t temp_storage_bytes,
                            const Tensor& temp_storage,
                            const Stream& stream);

    ReductionOp op;
    const Tensor input;
    mutable Tensor output;
    const Tensor segment_offsets;
    const uint64_t num_items;
    const uint64_t num_segments;
    // For a direct reduction this is the backend temporary-storage size. For a DisjointRuns parent it is the
    // aggregate stamped workspace footprint: all child temp storage plus all non-final value/index intermediates.
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    // The bound tensors as spans, made once at stamp so that a run converts nothing.
    const StridedSpan input_span{input};
    const StridedSpan offsets_span{segment_offsets};
    const StridedSpan output_span{output};
    const StridedSpan workspace_span{temp_storage};
    Stream stream;
    // The ragged plan (its workspace is temp_storage); never null.
    std::shared_ptr<const ReductionInternal::RaggedRoutedStamp> ragged;
};

/**
 * Describes an offset-segmented argmin or argmax over packed values [N,D...].
 *
 * Row offsets [B+1] produce global packed-value winner indices [B,D...]. Empty segments produce the configured
 * UINT64_MAX/UINT32_MAX sentinel independently for every trailing component. Candidate values are compared in FP32,
 * NaNs propagate, and the lowest packed index wins ties, matching ArgReducer.
 */
class SegmentedArgReducer {
   public:
    explicit SegmentedArgReducer(ArgReductionOp op, DataType index_output_dtype = DataType::UINT64);

    [[nodiscard]] ArgReductionOp getOperation() const { return op; }
    [[nodiscard]] DataType getIndexOutputDataType() const { return index_output_dtype; }
    [[nodiscard]] static bool isInputDataTypeSupported(DataType dtype);
    [[nodiscard]] static bool isOffsetDataTypeSupported(DataType dtype);

    [[nodiscard]] std::shared_ptr<StampedSegmentedArgReducer> stamp(
        const Tensor& input, const Tensor& segment_offsets, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedSegmentedArgReducer> stamp(
        const Tensor& input,
        const Tensor& preallocated_index_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

    // Variant for graph/runtime-populated offsets; see SegmentedReducer::stampRuntimeOffsets.
    [[nodiscard]] std::shared_ptr<StampedSegmentedArgReducer> stampRuntimeOffsets(
        const Tensor& input,
        const Tensor& preallocated_index_output,
        const Tensor& segment_offsets,
        const Stream& stream) const;

   private:
    [[nodiscard]] std::shared_ptr<StampedSegmentedArgReducer> stampValidated(
        const Tensor& input,
        const Tensor& index_output,
        const Tensor& segment_offsets,
        uint64_t num_segments,
        const Stream& stream) const;

    ArgReductionOp op;
    DataType index_output_dtype;
};

class StampedSegmentedArgReducer {
   public:
    void run();
    void runOn(Stream& run_stream) const;
    /** Benchmarks only: as StampedSegmentedReducer::runOnTensorsForBenchmark, with the index output. */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                  const StridedSpan& replacement_segment_offsets,
                                  const StridedSpan& replacement_index_output,
                                  Stream& run_stream) const;
    /** The physical passes this stamp runs, in order (diagnostic only). */
    [[nodiscard]] std::vector<std::string> getPassNames() const;

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] Tensor getIndexOutputTensor() const { return index_output; }
    [[nodiscard]] DataType getOffsetDataType() const { return segment_offsets.getDataType(); }
    [[nodiscard]] uint64_t getNumSegments() const { return num_segments; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    /** True when the stamp runs the ragged ARG kernels (always). */
    [[nodiscard]] bool usesRetainedRagged() const { return ragged != nullptr; }

   private:
    friend class SegmentedArgReducer;

    StampedSegmentedArgReducer(ArgReductionOp op,
                               const Tensor& input,
                               const Tensor& index_output,
                               const Tensor& segment_offsets,
                               uint64_t num_segments,
                               size_t temp_storage_bytes,
                               const Tensor& temp_storage,
                               const Stream& stream);

    ArgReductionOp op;
    const Tensor input;
    mutable Tensor index_output;
    const Tensor segment_offsets;
    uint64_t num_segments;
    size_t temp_storage_bytes;
    Tensor temp_storage;
    // The bound tensors as spans, made once at stamp so that a run converts nothing.
    const StridedSpan input_span{input};
    const StridedSpan offsets_span{segment_offsets};
    const StridedSpan index_output_span{index_output};
    const StridedSpan workspace_span{temp_storage};
    Stream stream;
    // The ragged plan (its workspace is temp_storage); never null.
    std::shared_ptr<const ReductionInternal::RaggedRoutedStamp> ragged;
};

}  // namespace ThorImplementation
