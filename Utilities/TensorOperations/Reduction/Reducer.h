#pragma once

// The public value reduction: Reducer describes it, stamp() selects and allocates, StampedReducer runs it.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StridedRLaunch.h"
#include "Utilities/TensorOperations/Reduction/Layout/ReductionGeometry.h"
#include "Utilities/TensorOperations/Reduction/Layout/ViewSplit.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseExecutablePlan.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"
#include "Utilities/TensorOperations/Reduction/ReductionTypes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

class StampedReducer;

/**
 * Describes a one-or-more-axis CUB reduction.
 *
 * Reducer is intentionally not executable. stamp() validates the concrete tensors, selects the centralized
 * reduction backend, queries any required temporary storage, and allocates both the output and workspace.
 * StampedReducer::run()
 * subsequently performs no allocation or CUB planning.
 *
 * Reduction axes must be non-empty, unique, and strictly increasing. Every path converts input values to FP32 before
 * applying the operation-specific input transform and reduction operator. Accumulation therefore always occurs in
 * FP32. Operation-specific output finalization, such as mean division or the final square root for L2 norm, also
 * occurs in FP32 before the configured storage conversion. The output storage dtype defaults to the input storage
 * dtype and may be overridden explicitly. A finite runtime output scale, defaulting to one, is applied in FP32 after
 * operation-specific finalization and before the single storage conversion.
 */
class Reducer {
   public:
    // tuning_depth: stamp-time autotuning depth of the tuned sites. Nullopt takes the ambient depth
    // (resolveReductionTuningDepth: this thread's ScopedReductionTuningDepth, else the process default).
    Reducer(ReductionOp op,
            uint32_t axis,
            std::optional<DataType> output_dtype = std::nullopt,
            float output_scale = 1.0f,
            std::optional<ReductionTuningDepth> tuning_depth = std::nullopt);
    Reducer(ReductionOp op,
            std::vector<uint32_t> axes,
            std::optional<DataType> output_dtype = std::nullopt,
            float output_scale = 1.0f,
            std::optional<ReductionTuningDepth> tuning_depth = std::nullopt);

    [[nodiscard]] ReductionOp getOperation() const { return op; }
    [[nodiscard]] uint32_t getAxis() const { return axes.front(); }
    [[nodiscard]] const std::vector<uint32_t>& getAxes() const { return axes; }
    [[nodiscard]] std::optional<DataType> getConfiguredOutputDataType() const { return output_dtype; }
    [[nodiscard]] float getOutputScale() const { return output_scale; }
    [[nodiscard]] ReductionTuningDepth getTuningDepth() const { return tuning_depth; }
    [[nodiscard]] DataType resolveOutputDataType(DataType input_dtype) const;

    /**
     * Returns the FP32 result defined for an empty reduction domain.
     *
     * Dense Thor tensors currently reject zero-sized dimensions, so this value is primarily an explicit semantic
     * contract for reuse by future offset-segmented reductions. Mean follows Thor's empty-segment convention and
     * produces zero.
     */
    [[nodiscard]] static float getFp32EmptyReductionValue(ReductionOp op);

    /**
     * Analyzes reduction geometry and selects an ordained execution family. Dense ownership is a hard invariant, and
     * there is no logical-index fallback. Unsupported rank>1 views throw NotImplementedException instead of being
     * assigned a catch-all execution path.
     */
    [[nodiscard]] static ReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               uint32_t axis);
    [[nodiscard]] static ReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint32_t>& axes);
    [[nodiscard]] static ReductionGeometry analyzeGeometry(const std::vector<uint64_t>& input_dimensions,
                                                               const std::vector<uint64_t>& input_strides,
                                                               const std::vector<uint32_t>& axes);
    /** How a caller runs a view analyzeGeometry refuses (ViewSplit), else nothing: the reducer owns the layout, or
     *  nothing does. */
    [[nodiscard]] static std::optional<ViewSplit> splitRefusedView(
        const std::vector<uint64_t>& input_dimensions,
        const std::vector<uint64_t>& input_strides,
        const std::vector<uint32_t>& axes);
    /** The same for a view whose storage is known. A pitched view (H) is SlicedOutput when its rows can be read at
     *  their pitch inside the view's source tensor (StridedSpan::withSourceLayout) and that moves fewer bytes than the
     *  packed copy, with `output_dtype` the reduction's output; otherwise as above. Nothing for a pitched view the
     *  reducer reads in place (pitchedCompleteLaunch). */
    [[nodiscard]] static std::optional<ViewSplit> splitRefusedView(const StridedSpan& input,
                                                                   const std::vector<uint32_t>& axes,
                                                                   DataType output_dtype);
    /** As above with `reducer_reads_pitched` false: the split a pitched view the reducer reads in place
     *  (pitchedCompleteLaunch) would have otherwise (benchmarks compare the two). */
    [[nodiscard]] static std::optional<ViewSplit> splitRefusedView(const StridedSpan& input,
                                                                   const std::vector<uint32_t>& axes,
                                                                   DataType output_dtype,
                                                                   bool reducer_reads_pitched);

    /**
     * The in-place launch for a pitched view (H) the reducer reads, or nothing: not a pitched view, a dtype or extent
     * PacketComplete does not take (an input base not element-aligned, more than 2^32 - 1 outer items or K, more than
     * 2^32 - 9 rows), K tiles under 256 bytes, or too little read in flight to beat the split it replaces (the reducer
     * then refuses the view and splitRefusedView says how to run it). The K tiles are balanced under the dtype's cap.
     * warps_per_output is the most of 1 / 2 / 4 / 8 whose launch fits one wave of resident warps (32 per SM), with at
     * least four rows per warp; in place when that launch keeps 4 KiB per SM in flight against a sliced split, 2 KiB
     * per SM against a packed one (`output_dtype`, the reduction's output, decides which: splitRefusedView).
     * `warps_per_output` forces W and skips the tile-width and in-flight rules (benchmarks).
     */
    [[nodiscard]] static std::optional<PitchedCompleteLaunch> pitchedCompleteLaunch(
        const StridedSpan& input,
        const std::vector<uint32_t>& axes,
        DataType output_dtype,
        std::optional<uint32_t> warps_per_output = std::nullopt);
    /** True when a value reduction stamps `input` as it is: analyzeGeometry accepts its layout, or it is a pitched view
     *  the reducer reads in place (pitchedCompleteLaunch, with the reduction's `output_dtype`). */
    [[nodiscard]] static bool readsSpanInPlace(const StridedSpan& input,
                                               const std::vector<uint32_t>& axes,
                                               DataType output_dtype);

    /**
     * Returns the executable geometry for a value reduction.
     *
     * analyzeGeometry() classifies structural access/shape, including the transitional DisjointRuns geometry tag for
     * ordinary dense disjoint reductions. VALUE stamping now feeds every ordinary dense R/KR/RK problem through the
     * flat dense planner; special views remain on their ordained view executors. This wrapper validates the value
     * operation while preserving that structural geometry metadata (no segment-size limit; every executor carries
     * 64-bit geometry).
     */
    [[nodiscard]] static ReductionGeometry analyzeValueGeometry(ReductionOp op,
                                                                    const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint32_t>& axes);
    [[nodiscard]] static ReductionGeometry analyzeValueGeometry(ReductionOp op,
                                                                    const std::vector<uint64_t>& input_dimensions,
                                                                    const std::vector<uint64_t>& input_strides,
                                                                    const std::vector<uint32_t>& axes);
    /** The same for a span: a pitched view the reducer reads in place (pitchedCompleteLaunch, with the reduction's
     *  `output_dtype`) is accepted. */
    [[nodiscard]] static ReductionGeometry analyzeValueSpanGeometry(ReductionOp op,
                                                                        const StridedSpan& input,
                                                                        const std::vector<uint32_t>& axes,
                                                                        DataType output_dtype);

    /** Queries backend temporary storage without allocating input, output, or workspace tensors. */
    [[nodiscard]] size_t queryWorkspaceSizeInBytes(const TensorDescriptor& input_descriptor,
                                                   const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedReducer> stamp(const StridedSpan& input, const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedReducer> stamp(const StridedSpan& input,
                                                        const Tensor& preallocated_output,
                                                        const Stream& stream) const;
    /** Benchmarks: a pitched view's in-place stamp at `warps_per_output`, whether or not the fill rule picks it
     *  (pitchedCompleteLaunch with W forced). Throws NotImplementedException when PacketComplete cannot read it. */
    [[nodiscard]] std::shared_ptr<StampedReducer> stampPitched(const StridedSpan& input,
                                                               uint32_t warps_per_output,
                                                               const Stream& stream) const;

   private:
    ReductionOp op;
    std::vector<uint32_t> axes;
    // The classification analyzeGeometry refuses F / G / H from: every view an ordained layout names.
    [[nodiscard]] static ReductionGeometry classifyGeometry(const std::vector<uint64_t>& input_dimensions,
                                                                const std::vector<uint64_t>& input_strides,
                                                                const std::vector<uint32_t>& axes);
    [[nodiscard]] std::shared_ptr<StampedReducer> stampValidated(const StridedSpan& input,
                                                                 const Tensor& output,
                                                                 const ReductionGeometry& geometry,
                                                                 const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedReducer> stampPitchedComplete(
        const StridedSpan& input,
        const Tensor& output,
        const ReductionGeometry& geometry,
        const PitchedCompleteLaunch& launch,
        const Stream& stream) const;

    std::optional<DataType> output_dtype;
    float output_scale;
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular;
};

/**
 * Concrete, allocation-free-at-run-time CUB reduction operation.
 *
 * A stamped operation is bound to its input/output tensors and geometry. runOn() may use another stream on the same
 * GPU, but the tensor bindings, reduction operation, axes, output dtype, selected CUB path, and workspace size are
 * fixed at stamp time.
 */
class StampedReducer {
   public:
    void run();
    void run(float output_scale);
    void runOn(Stream& run_stream) const;
    void runOn(Stream& run_stream, float output_scale) const;
    /**
     * Benchmarks only: runs the stamped plan on another input and output with the stamp's dtypes and element counts, at
     * the stamp's output scale. Intermediates and scratch stay the stamp's own. Cold-L2 audits time a production stamp
     * on rotating input and output views without re-stamping. Production binds its tensors.
     * A stamp of a non-dense view input takes a replacement view of the same dims and strides (the view audit); a dense
     * stamp takes a dense replacement. A physical-dense view stamp's first pass reads the replacement view as dense
     * (StridedSpan::asDense), as it reads its own; its strategy chain starts "view_physical:".
     */
    void runOnTensorsForBenchmark(const StridedSpan& replacement_input, const StridedSpan& replacement_output,
                                  Stream& run_stream) const;

    [[nodiscard]] uint32_t gpuNum() const { return output.getPlacement().getDeviceNum(); }
    [[nodiscard]] const StridedSpan& getInput() const { return input; }
    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] ReductionOp getOperation() const { return op; }
    [[nodiscard]] ReductionLayout getPath() const { return geometry.path; }
    [[nodiscard]] DataType getInputDataType() const { return input.getDataType(); }
    [[nodiscard]] DataType getOutputDataType() const { return output.getDataType(); }
    [[nodiscard]] DataType getAccumulatorDataType() const { return DataType::FP32; }
    [[nodiscard]] const ReductionGeometry& getGeometry() const { return geometry; }
    /**
     * Returns the current-run reduction axis executed by each physical VALUE pass when the structural geometry is
     * DisjointRuns. Ordinary dense VALUE always sources this from DenseExecutablePlan.
     * This is diagnostic planning metadata only; run() never consults it.
     */
    [[nodiscard]] std::vector<std::vector<uint32_t>> getComposedStageAxes() const;
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const { return temp_storage_bytes; }
    [[nodiscard]] float getOutputScale() const { return output_scale; }

    /** True when ordinary dense execution is represented as a flat ordered physical-pass plan. */
    [[nodiscard]] bool usesDenseExecutablePlan() const { return !dense_executable_plan.passes.empty(); }
    [[nodiscard]] size_t getDenseExecutablePassCount() const { return dense_executable_plan.passes.size(); }
    /** Physical sequence as stamped; never selects, allocates, or retunes. */
    [[nodiscard]] std::vector<DensePhysicalPassDiagnostic> getDensePhysicalPassDiagnostics() const;

    // Backward-compatible census metadata; physical parameters belong to the
    // per-pass diagnostics above, not a misleading first-RK special case.
    [[nodiscard]] bool usesModernRKFamilyPlan() const;
    [[nodiscard]] const std::string& getModernRKStrategyChain() const { return modern_rk_strategy_chain; }
    /** A rank-1 strided view's StridedR launch (closed form or Split; StridedReduction.h), else nothing.
     *  Its name ("strided_r:...") is the strategy chain. */
    [[nodiscard]] const std::optional<ReductionInternal::StridedRLaunch>& getStridedRLaunch() const {
        return strided_r_launch;
    }
    /** A pitched view's in-place PacketComplete launch, else nothing. Its name ("pitched_complete:w<W>_t<tile>") is the
     *  strategy chain. */
    [[nodiscard]] const std::optional<PitchedCompleteLaunch>& getPitchedCompleteLaunch() const {
        return pitched_complete_launch;
    }

   private:
    friend class Reducer;
    // Test-only (Fp8RKStampTest): reads a stamp's passes and builds a stamp from an injected plan.
    friend struct Fp8RKStampProbe;
    // Workspace-allocation regression tests; never used by production dispatch.
    friend struct ReducerWorkspaceProbe;

    StampedReducer(ReductionOp op,
                   ReductionGeometry geometry,
                   const StridedSpan& input,
                   const Tensor& output,
                   size_t temp_storage_bytes,
                   const Tensor& temp_storage,
                   bool apply_input_transform,
                   bool apply_output_finalize,
                   uint64_t semantic_total_reduction_size,
                   float output_scale,
                   const Stream& stream);

    StampedReducer(ReductionOp op,
                   ReductionGeometry geometry,
                   const StridedSpan& input,
                   const Tensor& output,
                   ReductionInternal::DenseExecutablePlan dense_executable_plan,
                   std::string modern_rk_strategy_chain,
                   bool apply_input_transform,
                   bool apply_output_finalize,
                   uint64_t semantic_total_reduction_size,
                   float output_scale,
                   const Stream& stream);

    ReductionOp op;
    ReductionGeometry geometry;
    const StridedSpan input;
    mutable Tensor output;
    const size_t temp_storage_bytes;
    Tensor temp_storage;
    // output and temp_storage as the spans the direct launch takes, made once at stamp (initialized from them).
    const StridedSpan output_span{output};
    const StridedSpan temp_storage_span{temp_storage};
    ReductionInternal::DenseExecutablePlan dense_executable_plan;
    std::string modern_rk_strategy_chain;
    // Physical passes need the logical operation's transform/finalizer placement, not merely the public operation.
    // This prevents L1/L2/SumSquares transforms from being re-applied to FP32 shard partials.
    bool apply_input_transform = true;
    bool apply_output_finalize = true;
    uint64_t semantic_total_reduction_size = 1;
    // Case C (a rank-1 strided view) runs StridedR; its Split partials are temp_storage.
    std::optional<ReductionInternal::StridedRLaunch> strided_r_launch;
    // A pitched view read in place (one PacketComplete launch; no workspace).
    std::optional<PitchedCompleteLaunch> pitched_complete_launch;

    const float output_scale;
    Stream stream;
};

}  // namespace ThorImplementation
