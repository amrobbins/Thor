#pragma once

#include <optional>

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
#include <utility>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"
#include "Utilities/Expression/CudaKernelSecurity.h"

namespace ThorImplementation {
struct PhysicalExecutionStage;
class CudaKernelExpression;
class Expression;
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
    EQUAL,
    NOT_EQUAL,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_NOT,
    CAST,
    WHERE,
    NEG,
    ABS,
    CEIL,
    FLOOR,
    ROUND,
    TRUNC,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    SINH,
    COSH,
    ASINH,
    ACOSH,
    ATANH,
    ERF,
    ERFC,
    ERFCX,
    ERFINV,
    ERFCINV,
    TGAMMA,
    LGAMMA,
    DIGAMMA,
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
    ROPE,
    SOFTMAX,
    FILL,
    RESHAPE,
    STRIDED_VIEW,
    STRIDED_VIEW_BACKWARD,
    UNSQUEEZE,
    SQUEEZE,
    TRANSPOSE,
    TAKE_ALONG_AXIS,
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
    CONV3D,
    CONV3D_BACKWARD_DATA,
    CONV3D_BACKWARD_FILTER,
    REDUCE_SUM,
    REDUCE_PROD,
    REDUCE_MIN,
    REDUCE_MAX,
    REDUCE_ARGMIN,
    REDUCE_ARGMAX,
    REDUCE_MIN_BACKWARD,
    REDUCE_MAX_BACKWARD,
    SCAN_MIN_BACKWARD,
    SCAN_MAX_BACKWARD,
    SEGMENTED_SCAN_MIN_BACKWARD,
    SEGMENTED_SCAN_MAX_BACKWARD,
    REDUCE_AVG,
    REDUCE_NORM1,
    REDUCE_NORM2,
    SCAN,
    RMSNORM,
    ATTENTION,
    ATTENTION_BACKWARD_Q,
    ATTENTION_BACKWARD_K,
    ATTENTION_BACKWARD_V,
    ATTENTION_BACKWARD_BIAS,
    EMBEDDING_LOOKUP,
    CUDA_KERNEL_OUTPUT,
    SEGMENTED_SCAN,
    SEGMENTED_REDUCE_SUM,
    SEGMENTED_REDUCE_MIN,
    SEGMENTED_REDUCE_MAX,
    RAGGED_VALUEWISE_EXTENT,
};

enum class RotaryScalingKind : uint8_t {
    None = 0,
    Linear = 1,
    DynamicNTK = 2,
    Yarn = 3,
    LongRope = 4,
    Llama3 = 5,
};

enum class MatmulEpilogue : uint8_t {
    Default = 0,
    Relu = 1,
    Gelu = 2,
};

enum class MatmulBackwardEpilogue : uint8_t {
    Default = 0,
    DRelu = 1,
    DGelu = 2,
};

enum class ScanOp : uint8_t {
    Sum = 0,
    Min = 1,
    Max = 2,
    Product = 3,
    ArgMin = 4,
    ArgMax = 5,
};

enum class ScanMode : uint8_t {
    Exclusive = 0,
    Inclusive = 1,
};

struct RotaryPositionEmbeddingOptions {
    uint32_t sequence_axis = 2;
    uint32_t head_dim_axis = 3;
    uint64_t rotary_dim = 0;  // 0 means use the full head-dimension extent at stamp/codegen time.
    double base = 10000.0;
    int64_t position_offset = 0;
    bool interleaved = false;
    bool inverse = false;
    RotaryScalingKind scaling_kind = RotaryScalingKind::None;
    double scaling_factor = 1.0;
    uint64_t original_max_position_embeddings = 0;
    std::optional<double> attention_factor = std::nullopt;
    double yarn_beta_fast = 32.0;
    double yarn_beta_slow = 1.0;
    double llama3_low_freq_factor = 1.0;
    double llama3_high_freq_factor = 4.0;
    std::vector<double> long_rope_short_factors;
    std::vector<double> long_rope_long_factors;
    std::optional<DataType> output_dtype = std::nullopt;
    std::optional<DataType> compute_dtype = std::nullopt;
    // Allows planner/runtime to rotate a private projection output in-place instead of materializing a separate RoPE output.
    // Defaults to false because out-of-place RoPE has benchmarked faster; set true only when memory pressure matters.
    bool allow_in_place_materialization = false;
};

inline bool isValueReductionOp(ExprOp op) {
    return op == ExprOp::REDUCE_SUM || op == ExprOp::REDUCE_PROD || op == ExprOp::REDUCE_MIN || op == ExprOp::REDUCE_MAX ||
           op == ExprOp::REDUCE_AVG || op == ExprOp::REDUCE_NORM1 || op == ExprOp::REDUCE_NORM2;
}

inline bool isReductionOp(ExprOp op) {
    return isValueReductionOp(op) || op == ExprOp::REDUCE_ARGMIN || op == ExprOp::REDUCE_ARGMAX;
}

inline bool isSoftmaxOp(ExprOp op) { return op == ExprOp::SOFTMAX; }

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
    MatmulEpilogue matmul_epilogue = MatmulEpilogue::Default;
    MatmulBackwardEpilogue matmul_backward_epilogue = MatmulBackwardEpilogue::Default;
    uint32_t matmul_epilogue_aux = UINT32_MAX;
    int32_t conv_stride_d = 1;
    int32_t conv_stride_h = 1;
    int32_t conv_stride_w = 1;
    int32_t conv_pad_d = 0;
    int32_t conv_pad_h = 0;
    int32_t conv_pad_w = 0;
    cudnnSoftmaxAlgorithm_t softmax_algorithm = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;

    AttentionTensorLayout attention_q_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout attention_k_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout attention_v_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout attention_o_layout = AttentionTensorLayout::BHSD;
    AttentionMaskKind attention_mask_kind = AttentionMaskKind::None;
    int64_t attention_diagonal_left_bound = 0;
    int64_t attention_diagonal_right_bound = 0;
    bool attention_has_scale = false;
    float attention_scale = 0.0f;
    bool attention_use_alibi_mask = false;
    bool attention_use_bias = false;
    bool attention_use_padding_mask = false;
    bool attention_use_ragged_offsets = false;
    bool attention_use_paged_kv_cache = false;
    int64_t attention_paged_kv_max_sequence_length = 0;
    float attention_dropout_probability = 0.0f;
    bool attention_use_fp8_forward_scaling = false;
    uint32_t attention_seq_len_q_node = UINT32_MAX;
    uint32_t attention_seq_len_kv_node = UINT32_MAX;
    uint32_t attention_ragged_offset_q_node = UINT32_MAX;
    uint32_t attention_ragged_offset_kv_node = UINT32_MAX;
    uint32_t attention_page_table_k_node = UINT32_MAX;
    uint32_t attention_page_table_v_node = UINT32_MAX;
    uint32_t attention_dropout_seed_node = UINT32_MAX;
    uint32_t attention_dropout_offset_node = UINT32_MAX;
    uint32_t attention_descale_q_node = UINT32_MAX;
    uint32_t attention_descale_k_node = UINT32_MAX;
    uint32_t attention_descale_v_node = UINT32_MAX;
    uint32_t attention_descale_s_node = UINT32_MAX;
    uint32_t attention_scale_s_node = UINT32_MAX;
    uint32_t attention_scale_o_node = UINT32_MAX;
    uint32_t attention_amax_s_node = UINT32_MAX;
    uint32_t attention_amax_o_node = UINT32_MAX;

    uint32_t rope_sequence_axis = 2;
    uint32_t rope_head_dim_axis = 3;
    uint64_t rope_rotary_dim = 0;
    double rope_base = 10000.0;
    int64_t rope_position_offset = 0;
    bool rope_interleaved = false;
    bool rope_inverse = false;
    RotaryScalingKind rope_scaling_kind = RotaryScalingKind::None;
    double rope_scaling_factor = 1.0;
    uint64_t rope_original_max_position_embeddings = 0;
    double rope_attention_factor = 1.0;
    double rope_yarn_beta_fast = 32.0;
    double rope_yarn_beta_slow = 1.0;
    double rope_llama3_low_freq_factor = 1.0;
    double rope_llama3_high_freq_factor = 4.0;
    std::vector<double> rope_long_rope_short_factors;
    std::vector<double> rope_long_rope_long_factors;
    bool rope_allow_in_place_materialization = false;

    uint64_t rms_norm_normalized_feature_count = 0;
    double rms_norm_epsilon = 1.0e-5;
    CudnnRmsNormFusedActivation rms_norm_fused_activation = CudnnRmsNormFusedActivation::NONE;

    bool embedding_has_padding_index = false;
    uint64_t embedding_padding_index = 0;

    ScanOp scan_op = ScanOp::Sum;
    ScanMode scan_mode = ScanMode::Exclusive;
    uint64_t scan_axis = UINT64_MAX;  // UINT64_MAX means final axis.
    bool scan_reverse = false;

    // RAGGED_VALUEWISE_EXTENT metadata. rhs is the canonical offsets tensor.
    uint64_t ragged_runtime_batch_size = 0;
    uint64_t ragged_runtime_max_active_values = 0;
    uint64_t ragged_runtime_elements_per_value = 1;

    // For INPUT / RUNTIME_SCALAR nodes only: actual dtype of the bound runtime value.
    std::optional<DataType> input_tensor_dtype = std::nullopt;

    // Value semantics for this node.
    std::optional<DataType> output_dtype = std::nullopt;
    std::optional<DataType> compute_dtype = std::nullopt;
    std::optional<DataType> backward_output_dtype = std::nullopt;
    std::optional<DataType> backward_compute_dtype = std::nullopt;

    // for shape/reduction nodes only
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> reshape_dims;
    std::vector<uint64_t> view_dims;
    std::vector<uint64_t> view_strides;
    uint64_t view_element_offset = 0;
    std::vector<uint64_t> squeeze_axes;
    std::vector<uint64_t> unsqueeze_axes;
    std::vector<uint64_t> fill_dims;

    // For CUDA_KERNEL_OUTPUT nodes only. All output nodes for one custom CUDA
    // kernel application share cuda_kernel_spec_index and differ by
    // cuda_kernel_output_index. cuda_kernel_input_nodes contains the graph
    // inputs to the kernel in the user-declared ABI order.
    uint32_t cuda_kernel_spec_index = UINT32_MAX;
    uint32_t cuda_kernel_output_index = UINT32_MAX;
    std::vector<uint32_t> cuda_kernel_input_nodes;
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
    std::vector<std::shared_ptr<const CudaKernelExpression>> cuda_kernel_expressions;
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

struct PhysicalConditionalOutputs;

struct PhysicalOutputs {
    std::shared_ptr<PhysicalExpression> expr;
    std::vector<NamedOutput> outputs;
    std::shared_ptr<PhysicalConditionalOutputs> conditional;

    [[nodiscard]] bool isConditional() const { return static_cast<bool>(conditional); }
};

struct PhysicalConditionalOutputs {
    PhysicalOutputs predicate;
    PhysicalOutputs then_branch;
    PhysicalOutputs else_branch;
};

struct ExpressionDefinition {
    PhysicalOutputs outputs;
    std::vector<std::string> expected_input_names;
    std::vector<std::string> expected_output_names;
    std::string canonical_hash;
    mutable nlohmann::json cuda_kernel_manifest_signature;

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] nlohmann::json architectureJsonWithCudaKernelManifestSignature() const;
    [[nodiscard]] bool hasCudaKernelExpressions() const;
    [[nodiscard]] std::vector<std::string> cudaKernelSigningPublicKeys() const;
    [[nodiscard]] std::vector<CudaKernelOutOfBandKeys> cudaKernelOutOfBandKeys() const;
    [[nodiscard]] std::vector<CudaKernelSourceInspection> cudaKernelSourceInfo() const;
    [[nodiscard]] std::vector<std::string> cudaKernelSources() const;
    [[nodiscard]] nlohmann::json cudaKernelSourceInfoJson() const;
    void allowUnsafeLoadedCudaKernelSourceCompilation(const std::string& trusted_ed25519_public_key,
                                                       const std::string& trusted_source_decryption_key = "");
    [[nodiscard]] static ExpressionDefinition fromOutputs(const Outputs& outputs);
    [[nodiscard]] static ExpressionDefinition deserialize(const nlohmann::json& j,
                                                        bool allow_unsafe_loaded_cuda_source = false,
                                                        const std::string& trusted_ed25519_public_key = "",
                                                        const std::string& trusted_source_decryption_key = "");
    void validate() const;
};

class Outputs {
   public:
    [[nodiscard]] const std::shared_ptr<PhysicalExpression>& expression() const { return expr; }
    [[nodiscard]] const std::vector<NamedOutput>& namedOutputs() const { return outputs; }
    [[nodiscard]] bool isConditional() const { return static_cast<bool>(conditional_outputs); }

    [[nodiscard]] PhysicalOutputs physicalOutputs() const {
        if (!expr && !conditional_outputs) {
            throw std::runtime_error("Outputs has no backing expression graph or conditional outputs.");
        }
        return PhysicalOutputs{
            .expr = expr,
            .outputs = outputs,
            .conditional = conditional_outputs,
        };
    }

    [[nodiscard]] static Outputs fromPhysicalOutputs(PhysicalOutputs physicalOutputs) {
        if (!physicalOutputs.expr && !physicalOutputs.conditional) {
            throw std::runtime_error("Outputs::fromPhysicalOutputs requires a non-null PhysicalExpression or conditional outputs.");
        }

        return Outputs(std::move(physicalOutputs.expr), std::move(physicalOutputs.outputs), std::move(physicalOutputs.conditional));
    }

    [[nodiscard]] static Outputs conditional(const Expression& predicate, const Outputs& then_outputs, const Outputs& else_outputs);
    [[nodiscard]] static Outputs ifElse(const Expression& predicate, const Outputs& then_outputs, const Outputs& else_outputs) {
        return conditional(predicate, then_outputs, else_outputs);
    }

   private:
    std::shared_ptr<PhysicalExpression> expr;
    std::vector<NamedOutput> outputs;
    std::shared_ptr<PhysicalConditionalOutputs> conditional_outputs;

    Outputs(std::shared_ptr<PhysicalExpression> expr,
            std::vector<NamedOutput> outputs,
            std::shared_ptr<PhysicalConditionalOutputs> conditional_outputs = nullptr)
        : expr(std::move(expr)), outputs(std::move(outputs)), conditional_outputs(std::move(conditional_outputs)) {}

    friend class Expression;
};

struct AttentionOptions {
    AttentionTensorLayout q_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout k_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout v_layout = AttentionTensorLayout::BHSD;
    AttentionTensorLayout o_layout = AttentionTensorLayout::BHSD;
    AttentionMaskKind mask_kind = AttentionMaskKind::None;
    int64_t diagonal_left_bound = 0;
    int64_t diagonal_right_bound = 0;
    std::optional<float> attention_scale = std::nullopt;
    bool use_alibi_mask = false;
    bool use_padding_mask = false;
    float dropout_probability = 0.0f;
    bool use_fp8_forward_scaling = false;
    bool use_paged_kv_cache = false;
    int64_t paged_kv_max_sequence_length = 0;
    std::optional<DataType> compute_dtype = std::nullopt;
    std::optional<DataType> output_dtype = std::nullopt;
};

class Expression {
   public:
    [[nodiscard]] static Stream& getNextHelperStream(uint32_t gpu_num);

    [[nodiscard]] Expression(double value);

    [[nodiscard]] std::set<std::string> getInputNames() const;

    [[nodiscard]] static Outputs outputs(const std::vector<std::pair<std::string, Expression>>& named_exprs);
    [[nodiscard]] static Outputs outputs(std::initializer_list<std::pair<std::string, Expression>> named_exprs);

    [[nodiscard]] static Expression input(const std::string& name,
                                          std::optional<DataType> compute_dtype = std::nullopt,
                                          std::optional<DataType> output_dtype = std::nullopt);
    [[nodiscard]] static Expression runtimeScalar(const std::string& name,
                                                  std::optional<DataType> compute_dtype = std::nullopt,
                                                  std::optional<DataType> output_dtype = std::nullopt);
    [[nodiscard]] static Expression tensorRuntimeScalar(const std::string& name,
                                                        std::optional<DataType> compute_dtype = std::nullopt,
                                                        std::optional<DataType> output_dtype = std::nullopt);
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
    [[nodiscard]] bool isSameLogicalNode(const Expression& other) const {
        return expr && other.expr && expr.get() == other.expr.get() && nodeIndex == other.nodeIndex;
    }

    [[nodiscard]] Expression operator+(const Expression& other) const;
    [[nodiscard]] Expression operator-(const Expression& other) const;
    [[nodiscard]] Expression operator*(const Expression& other) const;
    [[nodiscard]] Expression operator/(const Expression& other) const;

    [[nodiscard]] Expression operator==(const Expression& other) const;
    [[nodiscard]] Expression operator!=(const Expression& other) const;
    [[nodiscard]] Expression operator<(const Expression& other) const;
    [[nodiscard]] Expression operator<=(const Expression& other) const;
    [[nodiscard]] Expression operator>(const Expression& other) const;
    [[nodiscard]] Expression operator>=(const Expression& other) const;

    [[nodiscard]] Expression operator-() const;
    [[nodiscard]] Expression operator!() const;

    [[nodiscard]] Expression equal(const Expression& other) const;
    [[nodiscard]] Expression notEqual(const Expression& other) const;
    [[nodiscard]] Expression lessThan(const Expression& other) const;
    [[nodiscard]] Expression lessEqual(const Expression& other) const;
    [[nodiscard]] Expression greaterThan(const Expression& other) const;
    [[nodiscard]] Expression greaterEqual(const Expression& other) const;
    [[nodiscard]] Expression logicalAnd(const Expression& other) const;
    [[nodiscard]] Expression logicalOr(const Expression& other) const;
    [[nodiscard]] Expression logicalNot() const;
    [[nodiscard]] Expression cast(DataType output_dtype) const;
    [[nodiscard]] Expression select(const Expression& true_value, const Expression& false_value) const;

    [[nodiscard]] static Expression equal(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression notEqual(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression lessThan(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression lessEqual(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression greaterThan(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression greaterEqual(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression logicalAnd(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression logicalOr(const Expression& lhs, const Expression& rhs);
    [[nodiscard]] static Expression logicalNot(const Expression& input);
    [[nodiscard]] static Expression cast(const Expression& input, DataType output_dtype) { return input.cast(output_dtype); }
    [[nodiscard]] static Expression where(const Expression& condition, const Expression& true_value, const Expression& false_value);
    [[nodiscard]] static Expression select(const Expression& condition, const Expression& true_value, const Expression& false_value);

    [[nodiscard]] Expression abs() const;
    [[nodiscard]] Expression ceil() const;
    [[nodiscard]] Expression floor() const;
    [[nodiscard]] Expression round() const;
    [[nodiscard]] Expression trunc() const;
    [[nodiscard]] Expression sin() const;
    [[nodiscard]] Expression cos() const;
    [[nodiscard]] Expression tan() const;
    [[nodiscard]] Expression csc() const;
    [[nodiscard]] Expression sec() const;
    [[nodiscard]] Expression cot() const;
    [[nodiscard]] Expression asin() const;
    [[nodiscard]] Expression acos() const;
    [[nodiscard]] Expression atan() const;
    [[nodiscard]] Expression acsc() const;
    [[nodiscard]] Expression asec() const;
    [[nodiscard]] Expression acot() const;
    [[nodiscard]] Expression sinh() const;
    [[nodiscard]] Expression cosh() const;
    [[nodiscard]] Expression csch() const;
    [[nodiscard]] Expression sech() const;
    [[nodiscard]] Expression coth() const;
    [[nodiscard]] Expression asinh() const;
    [[nodiscard]] Expression acosh() const;
    [[nodiscard]] Expression atanh() const;
    [[nodiscard]] Expression acsch() const;
    [[nodiscard]] Expression asech() const;
    [[nodiscard]] Expression acoth() const;
    [[nodiscard]] Expression erf() const;
    [[nodiscard]] Expression erfc() const;
    [[nodiscard]] Expression erfcx() const;
    [[nodiscard]] Expression erfinv() const;
    [[nodiscard]] Expression erfcinv() const;
    [[nodiscard]] Expression tgamma() const;
    [[nodiscard]] Expression lgamma() const;
    [[nodiscard]] Expression digamma() const;
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
    [[nodiscard]] Expression rotaryPositionEmbedding(RotaryPositionEmbeddingOptions options = {}) const;
    [[nodiscard]] Expression rope(RotaryPositionEmbeddingOptions options = {}) const { return rotaryPositionEmbedding(std::move(options)); }
    [[nodiscard]] static Expression rotaryPositionEmbedding(const Expression& input, RotaryPositionEmbeddingOptions options = {}) {
        return input.rotaryPositionEmbedding(std::move(options));
    }
    [[nodiscard]] static Expression rope(const Expression& input, RotaryPositionEmbeddingOptions options = {}) {
        return input.rotaryPositionEmbedding(std::move(options));
    }
    [[nodiscard]] Expression reshape(const std::vector<uint64_t>& new_dims) const;
    [[nodiscard]] Expression stridedView(const std::vector<uint64_t>& dims,
                                         const std::vector<uint64_t>& strides_elements,
                                         uint64_t element_offset = 0) const;
    [[nodiscard]] Expression aliasView(const std::vector<uint64_t>& dims,
                                       const std::vector<uint64_t>& strides_elements,
                                       uint64_t element_offset = 0) const {
        return stridedView(dims, strides_elements, element_offset);
    }
    [[nodiscard]] Expression unsqueeze(const std::vector<uint64_t>& unsqueeze_axes) const;
    [[nodiscard]] Expression squeeze(const std::vector<uint64_t>& squeeze_axes) const;
    [[nodiscard]] Expression transpose() const;
    [[nodiscard]] Expression takeAlongAxis(const Expression& indices, int64_t axis = -1) const;
    [[nodiscard]] static Expression takeAlongAxis(const Expression& input, const Expression& indices, int64_t axis = -1);
    [[nodiscard]] Expression scan(ScanOp op = ScanOp::Sum, int64_t axis = -1, bool inclusive = true) const;
    [[nodiscard]] static Expression scan(const Expression& input,
                                         ScanOp op = ScanOp::Sum,
                                         int64_t axis = -1,
                                         bool inclusive = true);
    [[nodiscard]] Expression exclusiveScanSum(int64_t axis = -1) const { return scan(ScanOp::Sum, axis, false); }
    [[nodiscard]] Expression inclusiveScanSum(int64_t axis = -1) const { return scan(ScanOp::Sum, axis, true); }
    [[nodiscard]] Expression prefixCount(bool inclusive = true, int64_t axis = -1) const;
    [[nodiscard]] Expression segmentedScan(const Expression& offsets,
                                           ScanOp op = ScanOp::Sum,
                                           bool inclusive = true,
                                           bool reverse = false) const;
    [[nodiscard]] static Expression segmentedScan(const Expression& input,
                                                  const Expression& offsets,
                                                  ScanOp op = ScanOp::Sum,
                                                  bool inclusive = true,
                                                  bool reverse = false);
    [[nodiscard]] Expression segmentedReduceSum(const Expression& offsets) const;
    [[nodiscard]] Expression segmentedReduceMin(const Expression& offsets) const;
    [[nodiscard]] Expression segmentedReduceMax(const Expression& offsets) const;
    [[nodiscard]] static Expression segmentedReduceSum(const Expression& input, const Expression& offsets);
    [[nodiscard]] static Expression segmentedReduceMin(const Expression& input, const Expression& offsets);
    [[nodiscard]] static Expression segmentedReduceMax(const Expression& input, const Expression& offsets);
    [[nodiscard]] Expression withRaggedRuntimeExtent(const Expression& offsets,
                                                     uint64_t batch_size,
                                                     uint64_t max_active_values,
                                                     uint64_t elements_per_value) const;
    [[nodiscard]] std::pair<Expression, Expression> scanWithIndices(ScanOp op, int64_t axis = -1, bool inclusive = true) const;
    [[nodiscard]] std::pair<Expression, Expression> segmentedScanWithIndices(const Expression& offsets,
                                                                             ScanOp op,
                                                                             bool inclusive = true) const;
    [[nodiscard]] static Expression exclusiveScanSum(const Expression& input, int64_t axis = -1) {
        return scan(input, ScanOp::Sum, axis, false);
    }
    [[nodiscard]] static Expression inclusiveScanSum(const Expression& input, int64_t axis = -1) {
        return scan(input, ScanOp::Sum, axis, true);
    }
    [[nodiscard]] static Expression prefixCount(const Expression& mask, bool inclusive = true, int64_t axis = -1) {
        return mask.prefixCount(inclusive, axis);
    }
    [[nodiscard]] Expression pow(const Expression& exponent) const;

    // Numerically stable activation-shaped expression helpers. These prefer dedicated CUDA special-function
    // expression ops where available (tanhf, log1pf, expm1f, normcdff).
    [[nodiscard]] Expression sigmoid() const;
    [[nodiscard]] Expression tanh() const;
    [[nodiscard]] Expression softplus() const;
    [[nodiscard]] Expression elu(double alpha = 1.0) const;
    [[nodiscard]] Expression selu() const;
    [[nodiscard]] Expression gelu() const;
    [[nodiscard]] Expression mish() const;
    [[nodiscard]] Expression relu6() const;
    [[nodiscard]] Expression hardTanh(double min_value = -1.0, double max_value = 1.0) const;
    [[nodiscard]] Expression hardSwish() const;
    [[nodiscard]] Expression threshold(double threshold = 0.0, double value = 0.0) const;
    [[nodiscard]] Expression swish() const;
    [[nodiscard]] Expression softmax(cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE,
                                     cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL) const;
    [[nodiscard]] Expression logSoftmax(cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL) const;

    [[nodiscard]] static Expression matmul(const Expression& lhs,
                                           const Expression& rhs,
                                           bool transpose_lhs = false,
                                           bool transpose_rhs = false,
                                           std::optional<DataType> compute_dtype = std::nullopt,
                                           std::optional<DataType> output_dtype = std::nullopt);
    [[nodiscard]] static Expression gemm(const Expression& lhs,
                                         const Expression& rhs,
                                         const Expression& addend,
                                         double alpha = 1.0,
                                         double beta = 1.0,
                                         bool transpose_lhs = false,
                                         bool transpose_rhs = false,
                                         bool transpose_addend = false,
                                         std::optional<DataType> compute_dtype = std::nullopt,
                                         std::optional<DataType> output_dtype = std::nullopt);
    [[nodiscard]] static Expression gemm(const Expression& lhs,
                                         const Expression& rhs,
                                         const Expression& addend,
                                         const Expression& alpha,
                                         const Expression& beta,
                                         bool transpose_lhs = false,
                                         bool transpose_rhs = false,
                                         bool transpose_addend = false,
                                         std::optional<DataType> compute_dtype = std::nullopt,
                                         std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] static Expression rmsNorm(const Expression& input,
                                            const Expression& scale,
                                            uint64_t normalized_feature_count,
                                            double epsilon = 1.0e-5,
                                            std::optional<DataType> compute_dtype = std::nullopt,
                                            std::optional<DataType> output_dtype = std::nullopt);
    [[nodiscard]] Expression rmsNorm(const Expression& scale,
                                      uint64_t normalized_feature_count,
                                      double epsilon = 1.0e-5,
                                      std::optional<DataType> compute_dtype = std::nullopt,
                                      std::optional<DataType> output_dtype = std::nullopt) const {
        return rmsNorm(*this, scale, normalized_feature_count, epsilon, compute_dtype, output_dtype);
    }

    [[nodiscard]] static Expression embeddingLookup(const Expression& indices,
                                                    const Expression& weights,
                                                    std::optional<uint64_t> padding_index = std::nullopt,
                                                    std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] static Expression scaledDotProductAttention(const Expression& q,
                                                              const Expression& k,
                                                              const Expression& v,
                                                              AttentionOptions options = {});
    [[nodiscard]] static Expression scaledDotProductAttentionWithDropout(const Expression& q,
                                                                         const Expression& k,
                                                                         const Expression& v,
                                                                         const Expression& dropout_seed,
                                                                         const Expression& dropout_offset,
                                                                         AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttention(
        const Expression& q, const Expression& k, const Expression& v, const Expression& bias, AttentionOptions options = {});
    [[nodiscard]] static Expression scaledDotProductAttentionWithDropout(const Expression& q,
                                                                         const Expression& k,
                                                                         const Expression& v,
                                                                         const Expression& bias,
                                                                         const Expression& dropout_seed,
                                                                         const Expression& dropout_offset,
                                                                         AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttention(const Expression& q,
                                                              const Expression& k,
                                                              const Expression& v,
                                                              const Expression& q_seq_len,
                                                              const Expression& kv_seq_len,
                                                              AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionFp8Forward(const Expression& q,
                                                                        const Expression& k,
                                                                        const Expression& v,
                                                                        const Expression& descale_q,
                                                                        const Expression& descale_k,
                                                                        const Expression& descale_v,
                                                                        const Expression& descale_s,
                                                                        const Expression& scale_s,
                                                                        const Expression& scale_o,
                                                                        const Expression& amax_s,
                                                                        const Expression& amax_o,
                                                                        AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionFp8Forward(const Expression& q,
                                                                        const Expression& k,
                                                                        const Expression& v,
                                                                        const Expression& q_seq_len,
                                                                        const Expression& kv_seq_len,
                                                                        const Expression& descale_q,
                                                                        const Expression& descale_k,
                                                                        const Expression& descale_v,
                                                                        const Expression& descale_s,
                                                                        const Expression& scale_s,
                                                                        const Expression& scale_o,
                                                                        const Expression& amax_s,
                                                                        const Expression& amax_o,
                                                                        AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionPagedKv(const Expression& q,
                                                                     const Expression& k,
                                                                     const Expression& v,
                                                                     const Expression& q_seq_len,
                                                                     const Expression& kv_seq_len,
                                                                     const Expression& page_table_k,
                                                                     const Expression& page_table_v,
                                                                     AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionRagged(const Expression& q,
                                                                    const Expression& k,
                                                                    const Expression& v,
                                                                    const Expression& q_offsets,
                                                                    const Expression& kv_offsets,
                                                                    AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionRagged(const Expression& q,
                                                                    const Expression& k,
                                                                    const Expression& v,
                                                                    const Expression& q_seq_len,
                                                                    const Expression& kv_seq_len,
                                                                    const Expression& q_offsets,
                                                                    const Expression& kv_offsets,
                                                                    AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionRagged(const Expression& q,
                                                                    const Expression& k,
                                                                    const Expression& v,
                                                                    const Expression& bias,
                                                                    const Expression& q_seq_len,
                                                                    const Expression& kv_seq_len,
                                                                    const Expression& q_offsets,
                                                                    const Expression& kv_offsets,
                                                                    AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionRagged(const Expression& q,
                                                                    const Expression& k,
                                                                    const Expression& v,
                                                                    const Expression& q_seq_len,
                                                                    const Expression& kv_seq_len,
                                                                    const Expression& q_offsets,
                                                                    const Expression& kv_offsets,
                                                                    const Expression& dropout_seed,
                                                                    const Expression& dropout_offset,
                                                                    AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttentionRagged(const Expression& q,
                                                                    const Expression& k,
                                                                    const Expression& v,
                                                                    const Expression& bias,
                                                                    const Expression& q_seq_len,
                                                                    const Expression& kv_seq_len,
                                                                    const Expression& q_offsets,
                                                                    const Expression& kv_offsets,
                                                                    const Expression& dropout_seed,
                                                                    const Expression& dropout_offset,
                                                                    AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttention(const Expression& q,
                                                              const Expression& k,
                                                              const Expression& v,
                                                              const Expression& q_seq_len,
                                                              const Expression& kv_seq_len,
                                                              const Expression& dropout_seed,
                                                              const Expression& dropout_offset,
                                                              AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttention(const Expression& q,
                                                              const Expression& k,
                                                              const Expression& v,
                                                              const Expression& bias,
                                                              const Expression& q_seq_len,
                                                              const Expression& kv_seq_len,
                                                              AttentionOptions options);
    [[nodiscard]] static Expression scaledDotProductAttention(const Expression& q,
                                                              const Expression& k,
                                                              const Expression& v,
                                                              const Expression& bias,
                                                              const Expression& q_seq_len,
                                                              const Expression& kv_seq_len,
                                                              const Expression& dropout_seed,
                                                              const Expression& dropout_offset,
                                                              AttentionOptions options);
    [[nodiscard]] static Expression attention(const Expression& q,
                                              const Expression& k,
                                              const Expression& v,
                                              AttentionOptions options = {}) {
        return scaledDotProductAttention(q, k, v, std::move(options));
    }
    [[nodiscard]] static Expression attention(
        const Expression& q, const Expression& k, const Expression& v, const Expression& bias, AttentionOptions options = {}) {
        return scaledDotProductAttention(q, k, v, bias, std::move(options));
    }
    [[nodiscard]] static Expression attention(const Expression& q,
                                              const Expression& k,
                                              const Expression& v,
                                              const Expression& q_seq_len,
                                              const Expression& kv_seq_len,
                                              AttentionOptions options) {
        return scaledDotProductAttention(q, k, v, q_seq_len, kv_seq_len, std::move(options));
    }
    [[nodiscard]] static Expression attention(const Expression& q,
                                              const Expression& k,
                                              const Expression& v,
                                              const Expression& bias,
                                              const Expression& q_seq_len,
                                              const Expression& kv_seq_len,
                                              AttentionOptions options) {
        return scaledDotProductAttention(q, k, v, bias, q_seq_len, kv_seq_len, std::move(options));
    }

    [[nodiscard]] static Expression conv2d(const Expression& input,
                                           const Expression& filter,
                                           int32_t stride_h = 1,
                                           int32_t stride_w = 1,
                                           int32_t pad_h = 0,
                                           int32_t pad_w = 0,
                                           std::optional<DataType> compute_dtype = std::nullopt,
                                           std::optional<DataType> output_dtype = std::nullopt);
    [[nodiscard]] static Expression conv3d(const Expression& input,
                                           const Expression& filter,
                                           int32_t stride_d = 1,
                                           int32_t stride_h = 1,
                                           int32_t stride_w = 1,
                                           int32_t pad_d = 0,
                                           int32_t pad_h = 0,
                                           int32_t pad_w = 0,
                                           std::optional<DataType> compute_dtype = std::nullopt,
                                           std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] Expression reduction(ExprOp op,
                                       const std::vector<uint64_t>& reduction_axes,
                                       const std::vector<uint64_t>& squeeze_axes,
                                       std::optional<DataType> compute_dtype) const;
    [[nodiscard]] Expression reduce_sum(const std::vector<uint64_t>& reduction_axes = {},
                                        const std::vector<uint64_t>& squeeze_axes = {},
                                        std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression reduce_prod(const std::vector<uint64_t>& reduction_axes = {},
                                         const std::vector<uint64_t>& squeeze_axes = {},
                                         std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression reduce_min(const std::vector<uint64_t>& reduction_axes = {},
                                        const std::vector<uint64_t>& squeeze_axes = {},
                                        std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression reduce_max(const std::vector<uint64_t>& reduction_axes = {},
                                        const std::vector<uint64_t>& squeeze_axes = {},
                                        std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression argmin(const std::vector<uint64_t>& reduction_axes = {},
                                    const std::vector<uint64_t>& squeeze_axes = {},
                                    std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression argmax(const std::vector<uint64_t>& reduction_axes = {},
                                    const std::vector<uint64_t>& squeeze_axes = {},
                                    std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression reduce_mean(const std::vector<uint64_t>& reduction_axes = {},
                                         const std::vector<uint64_t>& squeeze_axes = {},
                                         std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression reduce_norm1(const std::vector<uint64_t>& reduction_axes = {},
                                          const std::vector<uint64_t>& squeeze_axes = {},
                                          std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] Expression reduce_norm2(const std::vector<uint64_t>& reduction_axes = {},
                                          const std::vector<uint64_t>& squeeze_axes = {},
                                          std::optional<DataType> compute_dtype = std::nullopt) const;

    [[nodiscard]] Expression min(const Expression& other) const;
    [[nodiscard]] Expression max(const Expression& other) const;

    [[nodiscard]] Expression clamp(const Expression& lower_bound, const Expression& upper_bound) const;
    [[nodiscard]] Expression clamp(double lower_bound, double upper_bound) const;
    [[nodiscard]] static Expression clamp(const Expression& input, const Expression& lower_bound, const Expression& upper_bound);
    [[nodiscard]] static Expression clamp(const Expression& input, double lower_bound, double upper_bound);

    [[nodiscard]] Expression dotProduct(const Expression& other, std::optional<DataType> compute_dtype = std::nullopt) const;
    [[nodiscard]] static Expression dotProduct(const Expression& lhs,
                                               const Expression& rhs,
                                               std::optional<DataType> compute_dtype = std::nullopt);

    [[nodiscard]] Expression outerProduct(const Expression& other,
                                          std::optional<DataType> compute_dtype = std::nullopt,
                                          std::optional<DataType> output_dtype = std::nullopt) const;
    [[nodiscard]] static Expression outerProduct(const Expression& lhs,
                                                 const Expression& rhs,
                                                 std::optional<DataType> compute_dtype = std::nullopt,
                                                 std::optional<DataType> output_dtype = std::nullopt);

    [[nodiscard]] Expression withDTypes(std::optional<DataType> compute_dtype = std::nullopt,
                                        std::optional<DataType> output_dtype = std::nullopt) const;
    [[nodiscard]] Expression withComputeDType(DataType compute_dtype) const;
    [[nodiscard]] Expression withOutputDType(DataType output_dtype) const;

    [[nodiscard]] Expression substituteInput(const std::string& input_name, const Expression& replacement) const;

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
    [[nodiscard]] static Expression quaternaryOp(
        const Expression& lhsExpr, const Expression& rhsExpr, const Expression& auxExpr, const Expression& fourthExpr, ExprOp op);
    [[nodiscard]] static Expression unaryOp(const Expression& inputExpr, ExprOp op);
    [[nodiscard]] static uint32_t cloneInto(const PhysicalExpression& src,
                                            PhysicalExpression& dst,
                                            std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name);

    [[nodiscard]] static Expression attentionWithOptionalMetadata(const Expression& q,
                                                                  const Expression& k,
                                                                  const Expression& v,
                                                                  const Expression* bias,
                                                                  const Expression* q_seq_len,
                                                                  const Expression* kv_seq_len,
                                                                  const Expression* q_ragged_offsets,
                                                                  const Expression* kv_ragged_offsets,
                                                                  const Expression* page_table_k,
                                                                  const Expression* page_table_v,
                                                                  const Expression* dropout_seed,
                                                                  const Expression* dropout_offset,
                                                                  AttentionOptions options,
                                                                  const Expression* descale_q = nullptr,
                                                                  const Expression* descale_k = nullptr,
                                                                  const Expression* descale_v = nullptr,
                                                                  const Expression* descale_s = nullptr,
                                                                  const Expression* scale_s = nullptr,
                                                                  const Expression* scale_o = nullptr,
                                                                  const Expression* amax_s = nullptr,
                                                                  const Expression* amax_o = nullptr);

    friend class CudaKernelExpression;

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

std::string formatFloatCanonical(double x);
bool isCommutative(ExprOp op);
std::string opName(ExprOp op);
std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo);
std::string canonicalize(const PhysicalExpression& expr);
std::string canonicalize(const PhysicalOutputs& outputs);
std::string expressionHash(const PhysicalOutputs& outputs);
std::string canonicalize(const PhysicalExecutionStage& stage);

}  // namespace ThorImplementation
