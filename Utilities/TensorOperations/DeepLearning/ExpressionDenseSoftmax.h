#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ThorImplementation {

enum class ExpressionDenseSoftmaxKind { Softmax, LogSoftmax };

const char* toString(ExpressionDenseSoftmaxKind kind);

/**
 * Isolated dense Softmax/LogSoftmax implementation expressed entirely through
 * Thor Expression.
 *
 * D5.1 intentionally does not wire this into Thor's existing Softmax layer or
 * Expression's SOFTMAX lowering.  It exists as a separately qualified backend
 * so its numerical and dtype contract can be verified before migration.
 *
 * The operation is over the final dimension of a packed [outerSize,
 * channelCount] logical view.  FP16/BF16/FP32 inputs are accepted; FP8 input
 * storage is deliberately rejected so callers must make any logits widening
 * explicit before Softmax.  Output storage is independent and may still be
 * selected explicitly (including FP8); an omitted output dtype resolves to the
 * input dtype.  All internal pointwise arithmetic and reductions are explicitly
 * FP32.  Reduction equations lower through Thor's normal reduction machinery
 * (currently CUB), so there is no dtype-compatibility adapter at this boundary.
 *
 * The implementation intentionally uses explicit Expression-equation
 * materialization boundaries around reductions.  Expression composition is
 * immutable and can otherwise clone an ancestor reduction when that value is
 * recombined with one of its descendants.  The staged form guarantees one max
 * reduction and one sum reduction in forward without duplicated reduction work.
 */
struct ExpressionDenseSoftmaxDescriptor {
    uint64_t outerSize = 0;
    uint64_t channelCount = 0;

    DataType inputDataType = DataType::FP16;
    std::optional<DataType> outputDataType = std::nullopt;
    DataType computeDataType = DataType::FP32;
    ExpressionDenseSoftmaxKind kind = ExpressionDenseSoftmaxKind::Softmax;
    std::string debugName = "thor_expression_dense_softmax";

    [[nodiscard]] DataType resolvedOutputDataType() const noexcept {
        return outputDataType.value_or(inputDataType);
    }

    void validateForward() const;
    void validateBackward() const;
};

struct ExpressionDenseSoftmaxForwardArgs {
    Tensor x;
    Tensor y;
};

struct ExpressionDenseSoftmaxBackwardArgs {
    // Backward consumes the materialized forward output.  Therefore selecting a
    // wider output dtype also preserves that wider stage boundary for backward.
    Tensor y;
    Tensor dy;
    Tensor dx;
};

class ExpressionDenseSoftmaxPlan {
   public:
    ExpressionDenseSoftmaxPlan(const ExpressionDenseSoftmaxPlan&) = delete;
    ExpressionDenseSoftmaxPlan& operator=(const ExpressionDenseSoftmaxPlan&) = delete;
    ExpressionDenseSoftmaxPlan(ExpressionDenseSoftmaxPlan&&) noexcept = default;
    ExpressionDenseSoftmaxPlan& operator=(ExpressionDenseSoftmaxPlan&&) noexcept = default;
    ~ExpressionDenseSoftmaxPlan() = default;

    [[nodiscard]] const ExpressionDenseSoftmaxDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }
    [[nodiscard]] bool isForward() const noexcept { return pass_ == Pass::Forward; }
    [[nodiscard]] bool isBackward() const noexcept { return pass_ == Pass::Backward; }
    [[nodiscard]] size_t equationCount() const noexcept { return equations_.size(); }
    [[nodiscard]] const FusedEquation& equation(size_t index) const { return *equations_.at(index); }

   private:
    enum class Pass { Forward, Backward };

    ExpressionDenseSoftmaxPlan(ExpressionDenseSoftmaxDescriptor descriptor,
                               Pass pass,
                               int gpuNum,
                               std::vector<std::shared_ptr<FusedEquation>> equations)
        : descriptor_(std::move(descriptor)), pass_(pass), gpu_num_(gpuNum), equations_(std::move(equations)) {}

    ExpressionDenseSoftmaxDescriptor descriptor_;
    Pass pass_;
    int gpu_num_ = -1;
    std::vector<std::shared_ptr<FusedEquation>> equations_;

    friend class ExpressionDenseSoftmax;
};

class ExpressionDenseSoftmax {
   public:
    static ExpressionDenseSoftmax& instance();

    [[nodiscard]] ExpressionDenseSoftmaxPlan prepareForward(const ExpressionDenseSoftmaxDescriptor& descriptor,
                                                            int gpuNum);
    [[nodiscard]] ExpressionDenseSoftmaxPlan prepareBackward(const ExpressionDenseSoftmaxDescriptor& descriptor,
                                                             int gpuNum);

    void forward(const ExpressionDenseSoftmaxPlan& plan,
                 const ExpressionDenseSoftmaxForwardArgs& args,
                 Stream& stream) const;
    void backward(const ExpressionDenseSoftmaxPlan& plan,
                  const ExpressionDenseSoftmaxBackwardArgs& args,
                  Stream& stream) const;

   private:
    ExpressionDenseSoftmax() = default;
};

static_assert(!std::is_copy_constructible_v<ExpressionDenseSoftmaxPlan>);
static_assert(!std::is_copy_assignable_v<ExpressionDenseSoftmaxPlan>);
static_assert(std::is_move_constructible_v<ExpressionDenseSoftmaxPlan>);
static_assert(std::is_move_assignable_v<ExpressionDenseSoftmaxPlan>);

}  // namespace ThorImplementation
