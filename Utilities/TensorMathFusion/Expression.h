#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {
struct PhysicalExecutionStage;

enum class ExprOp : uint16_t {
    INPUT = 3,
    RUNTIME_SCALAR,
    SCALAR_FP,
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    NEG,
    ABS,
    EXP,
    EXP2,
    EXP10,
    LN,
    LOG2,
    LOG10,
    SQRT,
    FILL,
    UNSQUEEZE,
    SQUEEZE,
    MIN,
    MAX,
    MIN_GRAD_LEFT,
    MIN_GRAD_RIGHT,
    MAX_GRAD_LEFT,
    MAX_GRAD_RIGHT,
    REDUCE_SUM,
    REDUCE_PROD,
    REDUCE_MIN,
    REDUCE_MAX,
    REDUCE_ARGMIN,
    REDUCE_ARGMAX,
    REDUCE_MIN_BACKWARD,
    REDUCE_MAX_BACKWARD,
    // REDUCE_AMAX, <-- requires indices, not doing for now.
    REDUCE_AVG,
    REDUCE_NORM1,
    REDUCE_NORM2,
};

inline bool isCudnnReduceOp(ExprOp op) {
    return op == ExprOp::REDUCE_SUM || op == ExprOp::REDUCE_PROD || op == ExprOp::REDUCE_MIN || op == ExprOp::REDUCE_MAX ||
           op == ExprOp::REDUCE_ARGMIN || op == ExprOp::REDUCE_ARGMAX || op == ExprOp::REDUCE_AVG || op == ExprOp::REDUCE_NORM1 ||
           op == ExprOp::REDUCE_NORM2;
}

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
    uint32_t input_slot = UINT32_MAX;
    double scalar_fp = 0.0;

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

   private:
    std::shared_ptr<PhysicalExpression> expr;
    std::vector<NamedOutput> outputs;

    Outputs(std::shared_ptr<PhysicalExpression> expr, std::vector<NamedOutput> outputs)
        : expr(std::move(expr)), outputs(std::move(outputs)) {}

    friend class Expression;
};

class Expression {
   public:
    static Stream& getNextHelperStream(uint32_t gpu_num);

    Expression(double value);

    std::set<std::string> getInputNames() const;

    static Outputs outputs(const std::vector<std::pair<std::string, Expression>>& named_exprs);
    static Outputs outputs(std::initializer_list<std::pair<std::string, Expression>> named_exprs);

    static Expression input(const std::string& name,
                            Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
                            Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    static Expression runtimeScalar(const std::string& name,
                                    Optional<TensorDescriptor::DataType> compute_dtype = Optional<TensorDescriptor::DataType>::empty(),
                                    Optional<TensorDescriptor::DataType> output_dtype = Optional<TensorDescriptor::DataType>::empty());
    static Expression constant_scalar(double value);

    [[nodiscard]] PhysicalExpression expression() const;

    Expression operator+(const Expression& other) const;
    Expression operator-(const Expression& other) const;
    Expression operator*(const Expression& other) const;
    Expression operator/(const Expression& other) const;

    Expression operator-() const;

    [[nodiscard]] Expression abs() const;
    [[nodiscard]] Expression ln() const;
    [[nodiscard]] Expression log2() const;
    [[nodiscard]] Expression log10() const;
    [[nodiscard]] Expression log(double base) const;
    [[nodiscard]] Expression exp() const;
    [[nodiscard]] Expression exp2() const;
    [[nodiscard]] Expression exp10() const;
    [[nodiscard]] Expression sqrt() const;
    [[nodiscard]] Expression unsqueeze(const std::vector<uint64_t>& unsqueeze_axes) const;
    [[nodiscard]] Expression squeeze(const std::vector<uint64_t>& squeeze_axes) const;
    [[nodiscard]] Expression pow(const Expression& exponent) const;

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

    static bool isLeafOp(ExprOp op);
    static bool isUnaryOp(ExprOp op);
    static bool isBinaryOp(ExprOp op);

   private:
    std::shared_ptr<PhysicalExpression> expr;
    uint32_t nodeIndex = UINT32_MAX;

    Expression(std::shared_ptr<PhysicalExpression> expr, uint32_t nodeIndex) : expr(std::move(expr)), nodeIndex(nodeIndex) {}

    static Expression binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op);
    static Expression unaryOp(const Expression& inputExpr, ExprOp op);
};

std::string formatFloatCanonical(double x);
bool isCommutative(ExprOp op);
std::string opName(ExprOp op);
std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo);
std::string canonicalize(const PhysicalExpression& expr);
std::string canonicalize(const PhysicalExecutionStage& stage);

}  // namespace ThorImplementation
