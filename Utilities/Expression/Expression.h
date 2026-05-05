#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {
struct PhysicalExecutionStage;
class Outputs;

enum class ExprOp : uint16_t {
    INPUT = 3,
    RUNTIME_SCALAR,
    TENSOR_RUNTIME_SCALAR,
    SCALAR_FP,
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    NEG,
    ABS,
    EXP,
    EXPM1,
    EXP2,
    EXP10,
    LN,
    LOG1P,
    LOG2,
    LOG10,
    SQRT,
    TANH,
    NORMCDF,
    SOFTMAX,
    FILL,
    UNSQUEEZE,
    SQUEEZE,
    TRANSPOSE,
    MIN,
    MAX,
    MIN_GRAD_LEFT,
    MIN_GRAD_RIGHT,
    MAX_GRAD_LEFT,
    MAX_GRAD_RIGHT,
    MATMUL,
    GEMM,
    CONV2D,
    CONV2D_BACKWARD_DATA,
    CONV2D_BACKWARD_FILTER,
    REDUCE_SUM,
    REDUCE_PROD,
    REDUCE_MIN,
    REDUCE_MAX,
    REDUCE_ARGMIN,
    REDUCE_ARGMAX,
    REDUCE_MIN_BACKWARD,
    REDUCE_MAX_BACKWARD,
    REDUCE_AVG,
    REDUCE_NORM1,
    REDUCE_NORM2,
};

inline bool isCudnnReduceOp(ExprOp op) {
    return op == ExprOp::REDUCE_SUM || op == ExprOp::REDUCE_PROD || op == ExprOp::REDUCE_MIN || op == ExprOp::REDUCE_MAX ||
           op == ExprOp::REDUCE_ARGMIN || op == ExprOp::REDUCE_ARGMAX || op == ExprOp::REDUCE_AVG || op == ExprOp::REDUCE_NORM1 ||
           op == ExprOp::REDUCE_NORM2;
}

inline bool isCudnnSoftmaxOp(ExprOp op) { return op == ExprOp::SOFTMAX; }

inline cudnnReduceTensorOp_t toCudnnReduceOp(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_SUM:
            return CUDNN_REDUCE_TENSOR_ADD;
        case ExprOp::REDUCE_PROD:
            return CUDNN_REDUCE_TENSOR_MUL;
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_ARGMIN:
            return CUDNN_REDUCE_TENSOR_MIN;
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMAX:
            return CUDNN_REDUCE_TENSOR_MAX;
        case ExprOp::REDUCE_AVG:
            return CUDNN_REDUCE_TENSOR_AVG;
        case ExprOp::REDUCE_NORM1:
            return CUDNN_REDUCE_TENSOR_NORM1;
        case ExprOp::REDUCE_NORM2:
            return CUDNN_REDUCE_TENSOR_NORM2;
        default:
            throw;
    }
}

struct ExprNode {
    ExprOp op;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;  // unused for unary/scalar ops
    uint32_t aux = UINT32_MAX;  // third input for ternary ops like GEMM
    uint32_t input_slot = UINT32_MAX;
    double scalar_fp = 0.0;
    double alpha_fp = 1.0;
    double beta_fp = 0.0;
    uint32_t alpha_node = UINT32_MAX;
    uint32_t beta_node = UINT32_MAX;
    bool transpose_lhs = false;
    bool transpose_rhs = false;
    bool transpose_aux = false;
    int32_t conv_stride_h = 1;
    int32_t conv_stride_w = 1;
    int32_t conv_pad_h = 0;
    int32_t conv_pad_w = 0;
    cudnnSoftmaxAlgorithm_t softmax_algorithm = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;

    // For INPUT / RUNTIME_SCALAR nodes only: actual dtype of the bound runtime value.
    Optional<TensorDescriptor::DataType> input_tensor_dtype = Optional<TensorDescriptor::DataType>::empty();

    // Value semantics for this node.
    Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty();
    Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty();
    Optional<TensorDescriptor::DataType> backward_output_dtype = Optional<TensorDescriptor::DataType>::empty();
    Optional<TensorDescriptor::DataType> backward_compute_dtype = Optional<TensorDescriptor::DataType>::empty();

    // for shape/reduction nodes only
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    std::vector<uint64_t> unsqueeze_axes;
    std::vector<uint64_t> fill_dims;
};

struct NamedInput {
    enum class Kind : uint8_t {
        Tensor,
        RuntimeScalarFp32,
        TensorRuntimeScalar,
    };

    std::string name;
    uint32_t slot;
    Kind kind = Kind::Tensor;
};

struct NamedOutput {
    std::string name;
    uint32_t node_idx;
};

struct PhysicalExpression {
    std::vector<ExprNode> nodes;
    std::vector<NamedInput> inputs;
    uint32_t output_node = UINT32_MAX;

    uint32_t numInputs() const { return static_cast<uint32_t>(inputs.size()); }

    uint32_t getOrCreateInputSlot(const std::string& name, NamedInput::Kind kind = NamedInput::Kind::Tensor);

    std::set<std::string> getInputNames() const {
        std::set<std::string> names;
        for (const auto& in : inputs) {
            names.insert(in.name);
        }
        return names;
    }
};

struct PhysicalOutputs {
    std::shared_ptr<PhysicalExpression> expr;
    std::vector<NamedOutput> outputs;
};

struct ExpressionDefinition {
    PhysicalOutputs outputs;
    std::vector<std::string> expected_input_names;
    std::vector<std::string> expected_output_names;
    std::string canonical_hash;

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] static ExpressionDefinition fromOutputs(const Outputs& outputs);
    [[nodiscard]] static ExpressionDefinition deserialize(const nlohmann::json& j);
    void validate() const;
};

class Outputs {
   public:
    [[nodiscard]] const std::shared_ptr<PhysicalExpression>& expression() const { return expr; }
    [[nodiscard]] const std::vector<NamedOutput>& namedOutputs() const { return outputs; }

    [[nodiscard]] PhysicalOutputs physicalOutputs() const {
        if (!expr) {
            throw std::runtime_error("Outputs has no backing expression graph.");
        }
        return PhysicalOutputs{
            .expr = expr,
            .outputs = outputs,
        };
    }

    [[nodiscard]] static Outputs fromPhysicalOutputs(PhysicalOutputs physicalOutputs) {
        if (!physicalOutputs.expr) {
            throw std::runtime_error("Outputs::fromPhysicalOutputs requires a non-null PhysicalExpression.");
        }

        return Outputs(std::move(physicalOutputs.expr), std::move(physicalOutputs.outputs));
    }

   private:
    std::shared_ptr<PhysicalExpression> expr;
    std::vector<NamedOutput> outputs;

    Outputs(std::shared_ptr<PhysicalExpression> expr, std::vector<NamedOutput> outputs)
        : expr(std::move(expr)), outputs(std::move(outputs)) {}

    friend class Expression;
};

class Expression {
   public:
    [[nodiscard]] static Stream& getNextHelperStream(uint32_t gpu_num);

    [[nodiscard]] Expression(double value);

    [[nodiscard]] std::set<std::string> getInputNames() const;

    [[nodiscard]] static Outputs outputs(const std::vector<std::pair<std::string, Expression>>& named_exprs);
    [[nodiscard]] static Outputs outputs(std::initializer_list<std::pair<std::string, Expression>> named_exprs);

    [[nodiscard]] static Expression input(
        const std::string& name,
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
        Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    [[nodiscard]] static Expression runtimeScalar(
        const std::string& name,
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
        Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    [[nodiscard]] static Expression tensorRuntimeScalar(
        const std::string& name,
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
        Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    [[nodiscard]] static Expression constantScalar(double value);

    [[nodiscard]] static Expression fromPhysicalNode(std::shared_ptr<PhysicalExpression> expr, uint32_t nodeIndex) {
        if (!expr) {
            throw std::invalid_argument("Expression::fromPhysicalNode requires a non-null PhysicalExpression.");
        }
        if (nodeIndex >= expr->nodes.size()) {
            throw std::out_of_range("Expression::fromPhysicalNode node index is out of range.");
        }
        return Expression(std::move(expr), nodeIndex);
    }

    [[nodiscard]] PhysicalExpression expression() const;

    [[nodiscard]] Expression operator+(const Expression& other) const;
    [[nodiscard]] Expression operator-(const Expression& other) const;
    [[nodiscard]] Expression operator*(const Expression& other) const;
    [[nodiscard]] Expression operator/(const Expression& other) const;

    [[nodiscard]] Expression operator-() const;

    [[nodiscard]] Expression abs() const;
    [[nodiscard]] Expression ln() const;
    [[nodiscard]] Expression log1p() const;
    [[nodiscard]] Expression log2() const;
    [[nodiscard]] Expression log10() const;
    [[nodiscard]] Expression log(double base) const;
    [[nodiscard]] Expression exp() const;
    [[nodiscard]] Expression expm1() const;
    [[nodiscard]] Expression exp2() const;
    [[nodiscard]] Expression exp10() const;
    [[nodiscard]] Expression sqrt() const;
    [[nodiscard]] static Expression sqrt(const Expression& expr);
    [[nodiscard]] Expression normcdf() const;
    [[nodiscard]] Expression unsqueeze(const std::vector<uint64_t>& unsqueeze_axes) const;
    [[nodiscard]] Expression squeeze(const std::vector<uint64_t>& squeeze_axes) const;
    [[nodiscard]] Expression transpose() const;
    [[nodiscard]] Expression pow(const Expression& exponent) const;

    // Numerically stable activation-shaped expression helpers. These prefer dedicated CUDA special-function
    // expression ops where available (tanhf, log1pf, expm1f, normcdff).
    [[nodiscard]] Expression sigmoid() const;
    [[nodiscard]] Expression tanh() const;
    [[nodiscard]] Expression softplus() const;
    [[nodiscard]] Expression elu(double alpha = 1.0) const;
    [[nodiscard]] Expression selu() const;
    [[nodiscard]] Expression gelu() const;
    [[nodiscard]] Expression swish() const;
    [[nodiscard]] Expression softmax(cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE,
                                     cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL) const;
    [[nodiscard]] Expression logSoftmax(cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL) const;

    [[nodiscard]] static Expression matmul(
        const Expression& lhs,
        const Expression& rhs,
        bool transpose_lhs = false,
        bool transpose_rhs = false,
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
        Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    [[nodiscard]] static Expression gemm(const Expression& lhs,
                                         const Expression& rhs,
                                         const Expression& addend,
                                         double alpha = 1.0,
                                         double beta = 1.0,
                                         bool transpose_lhs = false,
                                         bool transpose_rhs = false,
                                         bool transpose_addend = false,
                                         Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
                                         Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    [[nodiscard]] static Expression gemm(const Expression& lhs,
                                         const Expression& rhs,
                                         const Expression& addend,
                                         const Expression& alpha,
                                         const Expression& beta,
                                         bool transpose_lhs = false,
                                         bool transpose_rhs = false,
                                         bool transpose_addend = false,
                                         Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
                                         Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    [[nodiscard]] static Expression conv2d(
        const Expression& input,
        const Expression& filter,
        int32_t stride_h = 1,
        int32_t stride_w = 1,
        int32_t pad_h = 0,
        int32_t pad_w = 0,
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
        Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());

    [[nodiscard]] Expression reduction(ExprOp op,
                                       const std::vector<uint64_t>& reduction_axes,
                                       const std::vector<uint64_t>& squeeze_axes,
                                       Optional<TensorDescriptor::DataType> compute_dtype) const;
    [[nodiscard]] Expression reduce_sum(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression reduce_prod(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression reduce_min(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression reduce_max(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression argmin(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression argmax(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression reduce_mean(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression reduce_norm1(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression reduce_norm2(
        const std::vector<uint64_t>& reduction_axes = {},
        const std::vector<uint64_t>& squeeze_axes = {},
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty()) const;

    [[nodiscard]] Expression min(const Expression& other) const;
    [[nodiscard]] Expression max(const Expression& other) const;

    [[nodiscard]] Expression withDTypes(
        Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
        Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty()) const;
    [[nodiscard]] Expression withComputeDType(TensorDescriptor::DataType compute_dtype) const;
    [[nodiscard]] Expression withOutputDType(TensorDescriptor::DataType output_dtype) const;

    [[nodiscard]] static bool isLeafOp(ExprOp op);
    [[nodiscard]] static bool isUnaryOp(ExprOp op);
    [[nodiscard]] static bool isBinaryOp(ExprOp op);
    [[nodiscard]] static bool isTernaryOp(ExprOp op);

   private:
    std::shared_ptr<PhysicalExpression> expr;
    uint32_t nodeIndex = UINT32_MAX;

    [[nodiscard]] Expression(std::shared_ptr<PhysicalExpression> expr, uint32_t nodeIndex) : expr(std::move(expr)), nodeIndex(nodeIndex) {}

    [[nodiscard]] static Expression binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op);
    [[nodiscard]] static Expression ternaryOp(const Expression& lhsExpr, const Expression& rhsExpr, const Expression& auxExpr, ExprOp op);
    [[nodiscard]] static Expression unaryOp(const Expression& inputExpr, ExprOp op);

    static uint32_t encodeLowerableGemmScaleExpression(const Expression& scale_expr,
                                                       PhysicalExpression& dst,
                                                       std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name,
                                                       double& scale_fp);
};

inline Expression Expression::sigmoid() const {
    const Expression zero(0.0);
    const Expression one(1.0);

    // Stable for both signs:
    //   x >= 0: 1 / (1 + exp(-x))
    //   x <  0: exp(x) / (1 + exp(x))
    const Expression neg = -*this;
    const Expression numerator = (-neg.max(zero)).exp();
    const Expression denominator = one + (-this->abs()).exp();
    return numerator / denominator;
}

inline Expression Expression::softplus() const {
    const Expression zero(0.0);

    // log(1 + exp(x)) = max(x, 0) + log1p(exp(-abs(x)))
    return this->max(zero) + (-this->abs()).exp().log1p();
}

inline Expression Expression::elu(double alpha) const {
    const Expression zero(0.0);

    // Use expm1(min(x, 0)) so positive x does not evaluate exp(x), and negative values near zero keep precision.
    return this->max(zero) + (this->min(zero).expm1() * Expression(alpha));
}

inline Expression Expression::selu() const {
    const Expression zero(0.0);
    const Expression scale(1.05070098);
    const Expression scaleAlpha(1.758099326);

    return (this->max(zero) * scale) + (this->min(zero).expm1() * scaleAlpha);
}

inline Expression Expression::gelu() const {
    // Exact GELU: x * Phi(x), where Phi is the standard normal CDF.
    return *this * this->normcdf();
}

inline Expression Expression::swish() const { return *this * this->sigmoid(); }

std::string formatFloatCanonical(double x);
bool isCommutative(ExprOp op);
std::string opName(ExprOp op);
std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo);
std::string canonicalize(const PhysicalExpression& expr);
std::string canonicalize(const PhysicalOutputs& outputs);
std::string expressionHash(const PhysicalOutputs& outputs);
std::string canonicalize(const PhysicalExecutionStage& stage);

}  // namespace ThorImplementation
