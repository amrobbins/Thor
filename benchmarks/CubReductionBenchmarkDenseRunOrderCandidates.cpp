#include "benchmarks/CubReductionBenchmarkCandidate.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

enum class DenseRunOrder {
    LeftToRight,
    RightToLeft,
};

struct DenseRunOrderPlan {
    std::vector<std::vector<uint32_t>> reduced_run_axes;
    std::vector<uint64_t> output_dimensions;
};

[[nodiscard]] bool isDirectDensePath(CubReductionPath path) {
    return path == CubReductionPath::DeviceTransformReduce || path == CubReductionPath::ContiguousFixedSegment
           || path == CubReductionPath::TiledFixedSegment;
}

[[nodiscard]] std::optional<DenseRunOrderPlan> makeDenseRunOrderPlan(const TensorDescriptor& descriptor,
                                                                     const std::vector<uint32_t>& axes) {
    const std::vector<uint64_t> dimensions = descriptor.getDimensions();
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);
    if (!geometry.dense_run_geometry.has_value()) {
        return std::nullopt;
    }

    std::vector<bool> is_reduced(dimensions.size(), false);
    for (uint32_t axis : axes) {
        is_reduced[axis] = true;
    }

    // Match production's physical-run interpretation: singleton dimensions do not create a physical run boundary.
    // A retained singleton between reduced non-singletons may therefore be absorbed into a contiguous logical stage.
    struct ReducedRunSpan {
        uint32_t first_axis = 0;
        uint32_t last_axis = 0;
    };
    std::vector<ReducedRunSpan> spans;
    bool previous_non_singleton_was_reduced = false;
    for (uint32_t axis = 0; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] == 1) {
            continue;
        }
        if (is_reduced[axis]) {
            if (!previous_non_singleton_was_reduced) {
                spans.push_back(ReducedRunSpan{axis, axis});
            } else {
                spans.back().last_axis = axis;
            }
        }
        previous_non_singleton_was_reduced = is_reduced[axis];
    }

    // This candidate exists only to compare ordering of genuinely distinct reduced runs.
    if (spans.size() < 2) {
        return std::nullopt;
    }

    DenseRunOrderPlan plan;
    plan.reduced_run_axes.reserve(spans.size());
    for (const ReducedRunSpan& span : spans) {
        std::vector<uint32_t> stage_axes;
        stage_axes.reserve(static_cast<size_t>(span.last_axis - span.first_axis) + 1U);
        for (uint32_t axis = span.first_axis; axis <= span.last_axis; ++axis) {
            stage_axes.push_back(axis);
        }
        plan.reduced_run_axes.push_back(std::move(stage_axes));
    }
    plan.output_dimensions = geometry.output_dimensions;
    return plan;
}

class StampedDenseRunOrderCandidate final : public StampedReductionCandidate {
   public:
    StampedDenseRunOrderCandidate(Tensor input,
                                  DenseRunOrderPlan plan,
                                  std::vector<size_t> stage_order,
                                  std::string strategy,
                                  const Stream& stream)
        : input_(std::move(input)), plan_(std::move(plan)), strategy_(std::move(strategy)) {
        if (stage_order.size() != plan_.reduced_run_axes.size()) {
            throw std::invalid_argument(
                "Dense-run order benchmark stage order must cover every reduced run exactly once.");
        }
        std::vector<bool> seen(stage_order.size(), false);
        for (size_t run_index : stage_order) {
            if (run_index >= stage_order.size() || seen[run_index]) {
                throw std::invalid_argument(
                    "Dense-run order benchmark stage order must be a permutation of the reduced runs.");
            }
            seen[run_index] = true;
        }

        stages_.reserve(stage_order.size());
        Tensor current_input = input_;
        for (size_t stage_index = 0; stage_index < stage_order.size(); ++stage_index) {
            const bool is_final_stage = stage_index + 1 == stage_order.size();
            const DataType output_dtype = is_final_stage ? input_.getDataType() : DataType::FP32;
            CubReduction reduction(CubReductionOp::Sum, plan_.reduced_run_axes[stage_order[stage_index]], output_dtype);
            std::shared_ptr<StampedCubReduction> stage = reduction.stamp(current_input, stream);
            if (!isDirectDensePath(stage->getPath())) {
                throw std::logic_error(
                    "Dense-run order benchmark expected every composed pass to resolve to a direct dense reducer.");
            }

            workspace_bytes_ += stage->getWorkspaceSizeInBytes();
            if (!is_final_stage) {
                const Tensor intermediate = stage->getOutputTensor();
                if (intermediate.getDataType() != DataType::FP32) {
                    throw std::logic_error("Dense-run order benchmark requires FP32 intermediate partial aggregates.");
                }
                workspace_bytes_ += intermediate.getArraySizeInBytes();
                current_input = intermediate;
            } else {
                output_ = stage->getOutputTensor();
            }
            stages_.push_back(std::move(stage));
        }

        if (output_.getDimensions() != plan_.output_dimensions || output_.getDataType() != input_.getDataType()) {
            throw std::logic_error("Dense-run order benchmark produced an unexpected output descriptor.");
        }
    }

    void runOn(Stream& stream) const override {
        for (size_t stage_index = 0; stage_index + 1 < stages_.size(); ++stage_index) {
            stages_[stage_index]->runOn(stream, 1.0f);
        }
        stages_.back()->runOn(stream, 1.0f);
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return workspace_bytes_; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "dense_run_order_probe";
        metadata.strategy = strategy_;
        return metadata;
    }

   private:
    Tensor input_;
    DenseRunOrderPlan plan_;
    std::string strategy_;
    std::vector<std::shared_ptr<StampedCubReduction>> stages_;
    Tensor output_;
    size_t workspace_bytes_ = 0;
};

template <DenseRunOrder Order>
class DenseRunOrderCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override {
        if constexpr (Order == DenseRunOrder::LeftToRight) {
            return "dense_runs_left_to_right";
        }
        return "dense_runs_right_to_left";
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
        return makeDenseRunOrderPlan(input_descriptor, axes).has_value();
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "Dense-run order candidates require a dense contiguous SUM with at least two reduced runs.");
        }
        DenseRunOrderPlan plan = makeDenseRunOrderPlan(input.getDescriptor(), axes).value();
        std::vector<size_t> stage_order;
        stage_order.reserve(plan.reduced_run_axes.size());
        if constexpr (Order == DenseRunOrder::LeftToRight) {
            for (size_t i = 0; i < plan.reduced_run_axes.size(); ++i) {
                stage_order.push_back(i);
            }
        } else {
            for (size_t i = plan.reduced_run_axes.size(); i-- > 0;) {
                stage_order.push_back(i);
            }
        }
        const char* strategy =
            Order == DenseRunOrder::LeftToRight ? "left_to_right_fp32" : "right_to_left_fp32";
        return std::make_unique<StampedDenseRunOrderCandidate>(
            input, std::move(plan), std::move(stage_order), strategy, stream);
    }
};


class DenseRunEndOrderCandidate final : public ReductionCandidate {
   public:
    DenseRunEndOrderCandidate(std::string name, size_t reduced_run_count, std::string end_choices)
        : name_(std::move(name)), reduced_run_count_(reduced_run_count), end_choices_(std::move(end_choices)) {
        if (reduced_run_count_ < 3 || end_choices_.size() + 1 != reduced_run_count_) {
            throw std::invalid_argument("Dense-run mixed end-order candidate has an invalid decision count.");
        }
        for (char choice : end_choices_) {
            if (choice != 'L' && choice != 'R') {
                throw std::invalid_argument("Dense-run mixed end-order choices must contain only L or R.");
            }
        }
        strategy_ = "ends_";
        for (char choice : end_choices_) {
            strategy_.push_back(choice == 'L' ? 'l' : 'r');
        }
        strategy_ += "_fp32";
    }

    [[nodiscard]] std::string_view getName() const override { return name_; }

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
        const std::optional<DenseRunOrderPlan> plan = makeDenseRunOrderPlan(input_descriptor, axes);
        return plan.has_value() && plan->reduced_run_axes.size() == reduced_run_count_;
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "Dense-run mixed end-order candidate requires a matching dense contiguous SUM geometry.");
        }

        DenseRunOrderPlan plan = makeDenseRunOrderPlan(input.getDescriptor(), axes).value();
        std::vector<size_t> stage_order;
        stage_order.reserve(reduced_run_count_);
        size_t left = 0;
        size_t right = reduced_run_count_ - 1;
        for (char choice : end_choices_) {
            if (choice == 'L') {
                stage_order.push_back(left++);
            } else {
                stage_order.push_back(right--);
            }
        }
        if (left != right) {
            throw std::logic_error("Dense-run mixed end-order decisions did not leave exactly one final run.");
        }
        stage_order.push_back(left);

        return std::make_unique<StampedDenseRunOrderCandidate>(
            input, std::move(plan), std::move(stage_order), strategy_, stream);
    }

   private:
    std::string name_;
    size_t reduced_run_count_;
    std::string end_choices_;
    std::string strategy_;
};

struct DenseRunMixedEndOrderRegistrars {
    DenseRunMixedEndOrderRegistrars() {
        // Three reduced runs have four possible end-elimination sequences. Pure LL and RR are already covered by the
        // existing left-to-right/right-to-left candidates; register the two mixed choices.
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_lr", 3, "LR"));
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_rl", 3, "RL"));

        // Four reduced runs have eight end-elimination sequences. Pure LLL/RRR are already covered, so benchmark the
        // six mixed orders. This is enough to determine whether the eventual production planner must make a stage-wise
        // choice rather than selecting one global direction for the whole composition.
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_llr", 4, "LLR"));
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_lrl", 4, "LRL"));
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_lrr", 4, "LRR"));
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_rll", 4, "RLL"));
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_rlr", 4, "RLR"));
        registerReductionCandidate(
            std::make_unique<DenseRunEndOrderCandidate>("dense_runs_ends_rrl", 4, "RRL"));
    }
};

const DenseRunMixedEndOrderRegistrars dense_run_mixed_end_order_registrars;

const ReductionCandidateRegistrar<DenseRunOrderCandidate<DenseRunOrder::LeftToRight>>
    dense_run_left_to_right_registrar;
const ReductionCandidateRegistrar<DenseRunOrderCandidate<DenseRunOrder::RightToLeft>>
    dense_run_right_to_left_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
