#include "benchmarks/CubReductionBenchmarkCandidate.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

struct DenseRunTrailingReducedPlan {
    std::vector<uint32_t> leading_reduction_axes;
    std::vector<uint32_t> trailing_reduction_axes;
    std::vector<uint64_t> output_dimensions;
};

[[nodiscard]] std::optional<DenseRunTrailingReducedPlan> makeDenseRunTrailingReducedPlan(
    const TensorDescriptor& descriptor,
    const std::vector<uint32_t>& axes) {
    const std::vector<uint64_t> dimensions = descriptor.getDimensions();
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }
    const std::vector<CubReductionDenseRun>& runs = geometry.dense_run_geometry->runs;
    if (runs.size() != 3 || runs[0].kind != CubReductionDenseRunKind::Reduced
        || runs[1].kind != CubReductionDenseRunKind::Retained
        || runs[2].kind != CubReductionDenseRunKind::Reduced || runs[2].physical_stride != 1) {
        return std::nullopt;
    }

    // RKR-2A deliberately stays in the large-streaming regime established by RUN-1A/RKR-1.  The experiment asks a
    // narrow architectural question: can a dense R|K|R reduction be decomposed into Thor's already-good contiguous
    // suffix and tiled-middle reducers instead of requiring a bespoke fused kernel?
    if (runs[2].extent < 256 || runs[1].extent == 0) {
        return std::nullopt;
    }

    std::vector<bool> is_reduced(dimensions.size(), false);
    for (uint32_t axis : axes) {
        is_reduced[axis] = true;
    }

    size_t first_retained_axis = dimensions.size();
    size_t last_retained_axis = 0;
    for (size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
        if (!is_reduced[dimension]) {
            if (first_retained_axis == dimensions.size()) {
                first_retained_axis = dimension;
            }
            last_retained_axis = dimension;
        }
    }
    if (first_retained_axis == dimensions.size()) {
        return std::nullopt;
    }

    DenseRunTrailingReducedPlan plan;
    plan.output_dimensions = geometry.output_dimensions;

    for (uint32_t axis : axes) {
        if (axis < first_retained_axis) {
            plan.leading_reduction_axes.push_back(axis);
        } else if (axis > last_retained_axis) {
            plan.trailing_reduction_axes.push_back(axis);
        } else {
            // Keep RKR-2A intentionally narrow: logical reduced axes must be a prefix and suffix around one retained
            // logical block.  More elaborate singleton/interleaved cases can be addressed only if the decomposition
            // proves worthwhile on the census geometry.
            return std::nullopt;
        }
    }

    if (plan.leading_reduction_axes.empty() || plan.trailing_reduction_axes.empty()) {
        return std::nullopt;
    }
    return plan;
}

class StampedDenseRunTrailingReducedCandidate final : public StampedReductionCandidate {
   public:
    StampedDenseRunTrailingReducedCandidate(Tensor input,
                                            DenseRunTrailingReducedPlan plan,
                                            const Stream& stream)
        : input_(std::move(input)), plan_(std::move(plan)) {
        // Pass 1 peels the physically contiguous trailing reduced run.  Request an FP32 output explicitly so the
        // intermediate preserves Thor's FP32 accumulation contract and stage 2 never re-quantizes partial sums.
        CubReduction trailing_reduction(
            CubReductionOp::Sum, plan_.trailing_reduction_axes, DataType::FP32);
        trailing_stage_ = trailing_reduction.stamp(input_, stream);
        if (trailing_stage_->getPath() != CubReductionPath::ContiguousFixedSegment) {
            throw std::logic_error(
                "RKR-2A expected the trailing reduction pass to select ContiguousFixedSegment.");
        }
        intermediate_ = trailing_stage_->getOutputTensor();
        if (intermediate_.getDataType() != DataType::FP32) {
            throw std::logic_error("RKR-2A trailing reduction did not produce the required FP32 intermediate.");
        }

        // Pass 2 reduces the remaining leading R run across the FP32 [R,K] intermediate.  For R|K|R this is a dense
        // [outer=1,reduction=R,inner=K] problem and must select the existing TiledFixedSegment family.  Configure the
        // original storage dtype only at this final pass so conversion happens once, after the complete FP32 sum.
        CubReduction leading_reduction(
            CubReductionOp::Sum, plan_.leading_reduction_axes, input_.getDataType());
        leading_stage_ = leading_reduction.stamp(intermediate_, stream);
        if (leading_stage_->getPath() != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("RKR-2A expected the leading reduction pass to select TiledFixedSegment.");
        }
        output_ = leading_stage_->getOutputTensor();
        if (output_.getDimensions() != plan_.output_dimensions || output_.getDataType() != input_.getDataType()) {
            throw std::logic_error("RKR-2A composed reduction produced an unexpected output descriptor.");
        }

        workspace_bytes_ = intermediate_.getArraySizeInBytes() + trailing_stage_->getWorkspaceSizeInBytes()
                           + leading_stage_->getWorkspaceSizeInBytes();
    }

    void runOn(Stream& stream) const override {
        trailing_stage_->runOn(stream);
        leading_stage_->runOn(stream);
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return workspace_bytes_; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "dense_run_trailing_reduced";
        metadata.strategy = "contiguous_suffix_fp32_then_tiled_fp32";
        // Both passes deliberately reuse production-managed launch policies.  Leaving the launch-policy optionals
        // unset makes the census print backend_managed instead of inventing metadata the composed candidate does not own.
        return metadata;
    }

   private:
    Tensor input_;
    DenseRunTrailingReducedPlan plan_;
    std::shared_ptr<StampedCubReduction> trailing_stage_;
    Tensor intermediate_;
    std::shared_ptr<StampedCubReduction> leading_stage_;
    Tensor output_;
    size_t workspace_bytes_ = 0;
};

class DenseRunTrailingReducedCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override { return "dense_run_trailing_reduced"; }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        if (op != CubReductionOp::Sum) {
            return false;
        }
        const DataType dtype = input_descriptor.getDataType();
        if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
            return false;
        }
        return makeDenseRunTrailingReducedPlan(input_descriptor, axes).has_value();
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "dense_run_trailing_reduced candidate requires a supported dense contiguous R|K|R SUM geometry.");
        }
        DenseRunTrailingReducedPlan plan =
            makeDenseRunTrailingReducedPlan(input.getDescriptor(), axes).value();
        return std::make_unique<StampedDenseRunTrailingReducedCandidate>(input, std::move(plan), stream);
    }
};

const ReductionCandidateRegistrar<DenseRunTrailingReducedCandidate> dense_run_trailing_reduced_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
