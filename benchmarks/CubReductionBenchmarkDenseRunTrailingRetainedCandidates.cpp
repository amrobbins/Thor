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

enum class TwoReducedRunOrder {
    EarlierFirst,
    LaterFirst,
};

struct DenseRunTrailingRetainedPlan {
    std::vector<uint32_t> earlier_reduction_axes;
    std::vector<uint32_t> later_reduction_axes;
    std::vector<uint64_t> output_dimensions;
};

[[nodiscard]] std::optional<DenseRunTrailingRetainedPlan> makeDenseRunTrailingRetainedPlan(
    const TensorDescriptor& descriptor,
    const std::vector<uint32_t>& axes) {
    const std::vector<uint64_t> dimensions = descriptor.getDimensions();
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }

    // TRAILING-RETAINED-1 benchmarks exactly the two unresolved dense SUM census topologies:
    //
    //     R K R K
    //     K R K R K
    //
    // Both contain two reduced runs and end in retained data. Every pass is expressible through the production
    // TiledFixedSegment family. Production now uses the generic ComposedDense planner; keep this historical candidate
    // narrow so it continues to reproduce the pass-order experiment rather than becoming a second planner.
    const std::vector<CubReductionDenseRun>& runs = geometry.dense_run_geometry->runs;
    const bool is_rkrk = runs.size() == 4 && runs[0].kind == CubReductionDenseRunKind::Reduced
                         && runs[1].kind == CubReductionDenseRunKind::Retained
                         && runs[2].kind == CubReductionDenseRunKind::Reduced
                         && runs[3].kind == CubReductionDenseRunKind::Retained;
    const bool is_krkrk = runs.size() == 5 && runs[0].kind == CubReductionDenseRunKind::Retained
                          && runs[1].kind == CubReductionDenseRunKind::Reduced
                          && runs[2].kind == CubReductionDenseRunKind::Retained
                          && runs[3].kind == CubReductionDenseRunKind::Reduced
                          && runs[4].kind == CubReductionDenseRunKind::Retained;
    if (!is_rkrk && !is_krkrk) {
        return std::nullopt;
    }

    // Singleton-separated variants are handled by production ComposedDense. Reject them here so this historical
    // candidate continues to change only pass order for the exact non-singleton census geometries.
    for (uint64_t dimension : dimensions) {
        if (dimension == 1) {
            return std::nullopt;
        }
    }

    std::vector<bool> is_reduced(dimensions.size(), false);
    for (uint32_t axis : axes) {
        is_reduced[axis] = true;
    }

    std::vector<std::vector<uint32_t>> reduced_runs;
    bool previous_was_reduced = false;
    for (uint32_t dimension = 0; dimension < dimensions.size(); ++dimension) {
        const bool reduced = is_reduced[dimension];
        if (reduced && !previous_was_reduced) {
            reduced_runs.emplace_back();
        }
        if (reduced) {
            reduced_runs.back().push_back(dimension);
        }
        previous_was_reduced = reduced;
    }
    if (reduced_runs.size() != 2 || reduced_runs[0].empty() || reduced_runs[1].empty()) {
        return std::nullopt;
    }

    DenseRunTrailingRetainedPlan plan;
    plan.earlier_reduction_axes = std::move(reduced_runs[0]);
    plan.later_reduction_axes = std::move(reduced_runs[1]);
    plan.output_dimensions = geometry.output_dimensions;
    return plan;
}

class StampedDenseRunTrailingRetainedCandidate final : public StampedReductionCandidate {
   public:
    StampedDenseRunTrailingRetainedCandidate(Tensor input,
                                             DenseRunTrailingRetainedPlan plan,
                                             TwoReducedRunOrder order,
                                             const Stream& stream)
        : input_(std::move(input)), plan_(std::move(plan)), order_(order) {
        const std::vector<uint32_t>& first_axes = order_ == TwoReducedRunOrder::LaterFirst
                                                      ? plan_.later_reduction_axes
                                                      : plan_.earlier_reduction_axes;
        const std::vector<uint32_t>& second_axes = order_ == TwoReducedRunOrder::LaterFirst
                                                       ? plan_.earlier_reduction_axes
                                                       : plan_.later_reduction_axes;

        // Pass 1 always emits FP32.  This is both the accumulation type and the storage type for partial aggregates;
        // no FP16/BF16 quantization is allowed between composed stages.
        CubReduction first_reduction(CubReductionOp::Sum, first_axes, DataType::FP32);
        first_stage_ = first_reduction.stamp(input_, stream);
        if (first_stage_->getPath() != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error(
                "TRAILING-RETAINED-1 expected the first composed pass to select TiledFixedSegment.");
        }
        intermediate_ = first_stage_->getOutputTensor();
        if (intermediate_.getDataType() != DataType::FP32) {
            throw std::logic_error(
                "TRAILING-RETAINED-1 first composed pass did not produce the required FP32 intermediate.");
        }

        // Pass 2 is final: reduce the other run and convert only once to the original storage dtype.  Both logical
        // axis sets remain valid because CubReduction preserves reduced dimensions as singleton dimensions.
        CubReduction second_reduction(CubReductionOp::Sum, second_axes, input_.getDataType());
        second_stage_ = second_reduction.stamp(intermediate_, stream);
        if (second_stage_->getPath() != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error(
                "TRAILING-RETAINED-1 expected the final composed pass to select TiledFixedSegment.");
        }
        output_ = second_stage_->getOutputTensor();
        if (output_.getDimensions() != plan_.output_dimensions || output_.getDataType() != input_.getDataType()) {
            throw std::logic_error(
                "TRAILING-RETAINED-1 composed reduction produced an unexpected output descriptor.");
        }

        workspace_bytes_ = intermediate_.getArraySizeInBytes() + first_stage_->getWorkspaceSizeInBytes()
                           + second_stage_->getWorkspaceSizeInBytes();
    }

    void runOn(Stream& stream) const override {
        first_stage_->runOn(stream, 1.0f);
        second_stage_->runOn(stream, 1.0f);
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return workspace_bytes_; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "dense_runs_trailing_retained";
        metadata.strategy = order_ == TwoReducedRunOrder::LaterFirst
                                ? "later_reduced_fp32_then_earlier_reduced"
                                : "earlier_reduced_fp32_then_later_reduced";
        // Both passes use production-managed TiledFixedSegment policies.  Leave optional launch-policy fields unset.
        return metadata;
    }

   private:
    Tensor input_;
    DenseRunTrailingRetainedPlan plan_;
    TwoReducedRunOrder order_;
    std::shared_ptr<StampedCubReduction> first_stage_;
    Tensor intermediate_;
    std::shared_ptr<StampedCubReduction> second_stage_;
    Tensor output_;
    size_t workspace_bytes_ = 0;
};

template <TwoReducedRunOrder Order>
class DenseRunTrailingRetainedCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override {
        if constexpr (Order == TwoReducedRunOrder::LaterFirst) {
            return "dense_runs_trailing_retained_later_first";
        }
        return "dense_runs_trailing_retained_earlier_first";
    }

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
        return makeDenseRunTrailingRetainedPlan(input_descriptor, axes).has_value();
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "TRAILING-RETAINED-1 candidates require a supported dense contiguous R|K|R|K or K|R|K|R|K SUM "
                "geometry.");
        }
        DenseRunTrailingRetainedPlan plan =
            makeDenseRunTrailingRetainedPlan(input.getDescriptor(), axes).value();
        return std::make_unique<StampedDenseRunTrailingRetainedCandidate>(input, std::move(plan), Order, stream);
    }
};

const ReductionCandidateRegistrar<DenseRunTrailingRetainedCandidate<TwoReducedRunOrder::LaterFirst>>
    dense_run_trailing_retained_later_first_registrar;
const ReductionCandidateRegistrar<DenseRunTrailingRetainedCandidate<TwoReducedRunOrder::EarlierFirst>>
    dense_run_trailing_retained_earlier_first_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
