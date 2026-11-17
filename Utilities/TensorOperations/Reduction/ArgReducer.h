#pragma once

// The public ARG reduction (argmin / argmax with the value): ArgReducer and StampedArgReducer.

#include "Utilities/TensorOperations/Reduction/Reducer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

class StampedArgReducer;

/**
 * Describes an index-producing CUB argmin or argmax reduction.
 *
 * The candidate value is converted to FP32 before comparison. The winning index is the local flattened index within
 * the logical reduction domain, with reduced coordinates ordered row-major by the sorted reduction-axis list. NaNs
 * propagate and the lowest local index wins ties, including ties between multiple NaNs.
 */
class ArgReducer {
   public:
    ArgReducer(ArgReductionOp op,
               uint32_t axis,
               ArgReductionOutputOptions outputs = {},
               std::optional<ReductionTuningDepth> tuning_depth = std::nullopt);
    ArgReducer(ArgReductionOp op,
               std::vector<uint32_t> axes,
               ArgReductionOutputOptions outputs = {},
               std::optional<ReductionTuningDepth> tuning_depth = std::nullopt);

    [[nodiscard]] ArgReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] const ArgReductionOutputOptions& getOutputOptions() const { return outputs; }
    [[nodiscard]] ReductionTuningDepth getTuningDepth() const { return tuning_depth; }
    [[nodiscard]] DataType resolveValueOutputDataType(DataType input_dtype) const;
    [[nodiscard]] static float getFp32EmptyReductionValue(ArgReductionOp op);
    [[nodiscard]] static uint64_t getEmptyReductionIndex() { return std::numeric_limits<uint64_t>::max(); }

    /**
     * Returns the executable path for an ordinary dense ARG reduction. analyzeGeometry() already selects an ordained
     * dense family, including DisjointRuns for disjoint reduced runs; this wrapper reasserts that invariant without
     * changing the path. Index-width limits are the stamp's (a UINT32 index output needs a local domain that fits);
     * there is no segment-size limit.
     */
    [[nodiscard]] static ReductionGeometry analyzeDenseGeometry(
        const std::vector<uint64_t>& input_dimensions, const std::vector<uint32_t>& axes);

    /** Adds one reduced run's local winner coordinate to an already-composed original flattened arg index. */
    [[nodiscard]] static uint64_t composeOriginalArgIndex(uint64_t previous_index,
                                                          uint64_t local_run_index,
                                                          uint64_t domain_stride);

    /**
     * Queries the complete stamped workspace footprint without allocating input, output, intermediate, or temporary
     * tensors: every non-final FP32 value / carried-index intermediate of the planned pass list plus each pass's
     * scratch, matching StampedArgReducer::getWorkspaceSizeInBytes().
     */
    [[nodiscard]] size_t queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                   const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedArgReducer> stamp(const StridedSpan& input, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedArgReducer> stamp(
        const StridedSpan& input,
        const std::optional<Tensor>& preallocated_value_output,
        const std::optional<Tensor>& preallocated_index_output,
        const Stream& stream) const;

   private:
    // planDenseReduction()'s pass list, each pass a planned ARG stage.
    [[nodiscard]] std::shared_ptr<StampedArgReducer> stampPlannedDenseValidated(
        const StridedSpan& input,
        std::optional<Tensor> value_output,
        std::optional<Tensor> index_output,
        const ReductionGeometry& geometry,
        const Stream& stream) const;
    ArgReductionOp op;
    std::vector<uint32_t> axes;
    ArgReductionOutputOptions outputs;
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular;
};

/**
 * Concrete argmin/argmax reduction with no run-time allocation. Every planned pass's scratch and all value / index
 * intermediates are allocated while stamping and retained by this object for subsequent runs.
 */
class StampedArgReducer {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    /**
     * Benchmarks only: runs a planned ARG stamp on another input of the same dtype and element count, so cold-input
     * sweeps can time the stamped production plan without re-stamping. Production binds its input.
     */
    void runOnInputForBenchmark(const StridedSpan& replacement_input, Stream& run_stream) const;
    /**
     * Benchmarks only: runs this planned stamp on another input and other outputs of the stamp's dtypes and element
     * counts. Intermediates and scratch stay the stamp's own. Cold-L2 audits time a production stamp on rotating input
     * and output views without re-stamping. A replacement output must be given exactly when the stamp has that output.
     * Production binds its tensors.
     */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input,
                                  const std::optional<StridedSpan>& replacement_value_output,
                                  const std::optional<StridedSpan>& replacement_index_output,
                                  Stream& run_stream) const;
    /** Executed passes in order (e.g. "fp8_arg_one_window:o8_w1_p16_c1", "f16_arg_rk:..."). Diagnostic only. */
    [[nodiscard]] std::vector<std::string> getPlannedPassNames() const;

    [[nodiscard]] uint32_t gpuNum() const { return input.getPlacement().getDeviceNum(); }
    [[nodiscard]] ArgReductionOp getOperation() const { return op; }
    [[nodiscard]] ReductionLayout getPath() const { return geometry.path; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getValueAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] const ReductionGeometry& getGeometry() const { return geometry; }
    [[nodiscard]] std::vector<std::vector<uint32_t>> getComposedStageAxes() const;
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    [[nodiscard]] const std::optional<Tensor>& getValueOutputTensor() const { return value_output; }
    [[nodiscard]] const std::optional<Tensor>& getIndexOutputTensor() const { return index_output; }

   private:
    friend class ArgReducer;
    // Workspace-allocation regression tests; never used by production dispatch.
    friend struct ReducerWorkspaceProbe;

    StampedArgReducer(ArgReductionOp op,
                      ReductionGeometry geometry,
                      const StridedSpan& value_input,
                      std::optional<Tensor> carried_index_input,
                      std::optional<Tensor> value_output,
                      std::optional<Tensor> index_output,
                      size_t temp_storage_bytes,
                      const Tensor& temp_storage,
                      uint64_t domain_stride,
                      DataType carried_index_dtype,
                      const Stream& stream);

    StampedArgReducer(ArgReductionOp op,
                      ReductionGeometry geometry,
                      const StridedSpan& input,
                      std::optional<Tensor> value_output,
                      std::optional<Tensor> index_output,
                      size_t workspace_size_bytes,
                      std::vector<std::shared_ptr<StampedArgReducer>> composed_stages,
                      const Stream& stream);

    ArgReductionOp op;
    ReductionGeometry geometry;
    const StridedSpan input;
    mutable std::optional<Tensor> value_output;
    mutable std::optional<Tensor> index_output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;

    // Present only on the per-pass stages of a planned ARG stamp. The first stage has no carried-index input;
    // later stages consume the previous stage's SoA index tensor. domain_stride maps this stage's local row coordinate
    // directly into the original flattened reduction domain.
    std::optional<Tensor> carried_index_input;
    // The outputs and the carried index as the spans the launches take, made once at stamp (bindLaunchSpans).
    std::optional<StridedSpan> value_output_span;
    std::optional<StridedSpan> index_output_span;
    std::optional<StridedSpan> carried_index_span;
    std::optional<DataType> composed_carried_index_dtype;
    uint64_t composed_domain_stride = 1;
    std::vector<std::shared_ptr<StampedArgReducer>> composed_stages;
    // A planned stage that launches an FP8 ARG physical pass.
    std::shared_ptr<const ReductionInternal::Fp8ArgPhysicalPlan> fp8_arg_plan;
    // A planned stage that launches an FP32 ARG pass (the FP32 routing), original or carried.
    std::shared_ptr<const ReductionInternal::Fp32ArgPhysicalPlan> fp32_arg_plan;

    void launchBoundStage(const StridedSpan& stage_input, Stream& run_stream) const;
    void bindLaunchSpans();
    void launchBoundStage(const StridedSpan& stage_input, const StridedSpan* value_output_ptr,
                          const StridedSpan* index_output_ptr,
                          Stream& run_stream) const;

    Stream stream;
};

}  // namespace ThorImplementation
