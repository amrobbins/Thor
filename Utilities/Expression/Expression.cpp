#include "Utilities/Expression/Expression.h"
#include <optional>
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/CudaKernelSecurity.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string_view>
#include "DeepLearning/Implementation/ThorError.h"

using DataType = ThorImplementation::DataType;
using json = nlohmann::json;

namespace ThorImplementation {

static void validateUserInputName(const std::string& name) {
    if (name.rfind("__arg", 0) == 0) {
        throw std::runtime_error("Input names may not start with reserved prefix '__arg'.");
    }
}

namespace {

class HelperStreamPool {
   public:
    static constexpr uint32_t MAX_HELPER_STREAMS_PER_GPU = 8;
    static_assert((MAX_HELPER_STREAMS_PER_GPU & (MAX_HELPER_STREAMS_PER_GPU - 1)) == 0,
                  "MAX_HELPER_STREAMS_PER_GPU must be a power of two");
    static constexpr uint32_t HELPER_STREAM_MASK = MAX_HELPER_STREAMS_PER_GPU - 1;

    struct PerGpuHelperStreams {
        std::array<Stream, MAX_HELPER_STREAMS_PER_GPU> streams;
        std::atomic<uint32_t> next_index{0};

        explicit PerGpuHelperStreams(int32_t gpu)
            : streams{Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu)} {
            for (Stream& stream : streams) {
                stream.informIsStatic();
            }
        }
    };

    Stream& getNextHelperStream(uint32_t gpu_num) {
        ensureInitialized();
        THOR_THROW_IF_FALSE(gpu_num < per_gpu_.size());

        PerGpuHelperStreams& state = *per_gpu_[gpu_num];
        const uint32_t idx = state.next_index.fetch_add(1, std::memory_order_relaxed) & HELPER_STREAM_MASK;
        return state.streams[idx];
    }

   private:
    void ensureInitialized() {
        std::call_once(init_once_, [this]() {
            const uint32_t num_gpus = MachineEvaluator::instance().getNumGpus();
            per_gpu_.reserve(num_gpus);
            for (uint32_t g = 0; g < num_gpus; ++g) {
                per_gpu_.push_back(std::make_unique<PerGpuHelperStreams>(g));
            }
        });
    }

    std::once_flag init_once_;
    std::vector<std::unique_ptr<PerGpuHelperStreams>> per_gpu_;
};

HelperStreamPool helperStreamPool;

std::string namedInputKindToString(NamedInput::Kind kind) {
    switch (kind) {
        case NamedInput::Kind::Tensor:
            return "tensor";
        case NamedInput::Kind::RuntimeScalarFp32:
            return "runtime_scalar_fp32";
        case NamedInput::Kind::TensorRuntimeScalar:
            return "tensor_runtime_scalar";
        default:
            throw std::runtime_error("Unknown NamedInput::Kind.");
    }
}

NamedInput::Kind namedInputKindFromString(const std::string& kind) {
    if (kind == "tensor")
        return NamedInput::Kind::Tensor;
    if (kind == "runtime_scalar_fp32")
        return NamedInput::Kind::RuntimeScalarFp32;
    if (kind == "tensor_runtime_scalar")
        return NamedInput::Kind::TensorRuntimeScalar;
    throw std::runtime_error("Unknown expression input kind '" + kind + "'.");
}

std::string exprOpExternalName(ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
            return "input";
        case ExprOp::RUNTIME_SCALAR:
            return "runtime_scalar";
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            return "tensor_runtime_scalar";
        case ExprOp::SCALAR_FP:
            return "scalar_fp";
        case ExprOp::ADD:
            return "add";
        case ExprOp::SUB:
            return "sub";
        case ExprOp::MUL:
            return "mul";
        case ExprOp::DIV:
            return "div";
        case ExprOp::POW:
            return "pow";
        case ExprOp::EQUAL:
            return "equal";
        case ExprOp::NOT_EQUAL:
            return "not_equal";
        case ExprOp::LESS:
            return "less";
        case ExprOp::LESS_EQUAL:
            return "less_equal";
        case ExprOp::GREATER:
            return "greater";
        case ExprOp::GREATER_EQUAL:
            return "greater_equal";
        case ExprOp::LOGICAL_AND:
            return "logical_and";
        case ExprOp::LOGICAL_OR:
            return "logical_or";
        case ExprOp::LOGICAL_NOT:
            return "logical_not";
        case ExprOp::WHERE:
            return "where";
        case ExprOp::NEG:
            return "neg";
        case ExprOp::ABS:
            return "abs";
        case ExprOp::CEIL:
            return "ceil";
        case ExprOp::FLOOR:
            return "floor";
        case ExprOp::ROUND:
            return "round";
        case ExprOp::TRUNC:
            return "trunc";
        case ExprOp::SIN:
            return "sin";
        case ExprOp::COS:
            return "cos";
        case ExprOp::TAN:
            return "tan";
        case ExprOp::ASIN:
            return "asin";
        case ExprOp::ACOS:
            return "acos";
        case ExprOp::ATAN:
            return "atan";
        case ExprOp::SINH:
            return "sinh";
        case ExprOp::COSH:
            return "cosh";
        case ExprOp::ASINH:
            return "asinh";
        case ExprOp::ACOSH:
            return "acosh";
        case ExprOp::ATANH:
            return "atanh";
        case ExprOp::ERF:
            return "erf";
        case ExprOp::ERFC:
            return "erfc";
        case ExprOp::ERFCX:
            return "erfcx";
        case ExprOp::ERFINV:
            return "erfinv";
        case ExprOp::ERFCINV:
            return "erfcinv";
        case ExprOp::TGAMMA:
            return "tgamma";
        case ExprOp::LGAMMA:
            return "lgamma";
        case ExprOp::DIGAMMA:
            return "digamma";
        case ExprOp::EXP:
            return "exp";
        case ExprOp::EXPM1:
            return "expm1";
        case ExprOp::EXP2:
            return "exp2";
        case ExprOp::EXP10:
            return "exp10";
        case ExprOp::LN:
            return "ln";
        case ExprOp::LOG1P:
            return "log1p";
        case ExprOp::LOG2:
            return "log2";
        case ExprOp::LOG10:
            return "log10";
        case ExprOp::SQRT:
            return "sqrt";
        case ExprOp::TANH:
            return "tanh";
        case ExprOp::NORMCDF:
            return "normcdf";
        case ExprOp::ROPE:
            return "rope";
        case ExprOp::SOFTMAX:
            return "softmax";
        case ExprOp::FILL:
            return "fill";
        case ExprOp::RESHAPE:
            return "reshape";
        case ExprOp::STRIDED_VIEW:
            return "strided_view";
        case ExprOp::STRIDED_VIEW_BACKWARD:
            return "strided_view_backward";
        case ExprOp::UNSQUEEZE:
            return "unsqueeze";
        case ExprOp::SQUEEZE:
            return "squeeze";
        case ExprOp::TRANSPOSE:
            return "transpose";
        case ExprOp::MIN:
            return "min";
        case ExprOp::MAX:
            return "max";
        case ExprOp::MIN_GRAD_LEFT:
            return "min_grad_left";
        case ExprOp::MIN_GRAD_RIGHT:
            return "min_grad_right";
        case ExprOp::MAX_GRAD_LEFT:
            return "max_grad_left";
        case ExprOp::MAX_GRAD_RIGHT:
            return "max_grad_right";
        case ExprOp::MATMUL:
            return "matmul";
        case ExprOp::GEMM:
            return "gemm";
        case ExprOp::CONV2D:
            return "conv2d";
        case ExprOp::CONV2D_BACKWARD_DATA:
            return "conv2d_backward_data";
        case ExprOp::CONV2D_BACKWARD_FILTER:
            return "conv2d_backward_filter";
        case ExprOp::CONV3D:
            return "conv3d";
        case ExprOp::CONV3D_BACKWARD_DATA:
            return "conv3d_backward_data";
        case ExprOp::CONV3D_BACKWARD_FILTER:
            return "conv3d_backward_filter";
        case ExprOp::REDUCE_SUM:
            return "reduce_sum";
        case ExprOp::REDUCE_PROD:
            return "reduce_prod";
        case ExprOp::REDUCE_MIN:
            return "reduce_min";
        case ExprOp::REDUCE_MAX:
            return "reduce_max";
        case ExprOp::REDUCE_ARGMIN:
            return "reduce_argmin";
        case ExprOp::REDUCE_ARGMAX:
            return "reduce_argmax";
        case ExprOp::REDUCE_MIN_BACKWARD:
            return "reduce_min_backward";
        case ExprOp::REDUCE_MAX_BACKWARD:
            return "reduce_max_backward";
        case ExprOp::REDUCE_AVG:
            return "reduce_avg";
        case ExprOp::REDUCE_NORM1:
            return "reduce_norm1";
        case ExprOp::REDUCE_NORM2:
            return "reduce_norm2";
        case ExprOp::RMSNORM:
            return "rmsnorm";
        case ExprOp::ATTENTION:
            return "attention";
        case ExprOp::ATTENTION_BACKWARD_Q:
            return "attention_backward_q";
        case ExprOp::ATTENTION_BACKWARD_K:
            return "attention_backward_k";
        case ExprOp::ATTENTION_BACKWARD_V:
            return "attention_backward_v";
        case ExprOp::ATTENTION_BACKWARD_BIAS:
            return "attention_backward_bias";
        case ExprOp::EMBEDDING_LOOKUP:
            return "embedding_lookup";
        case ExprOp::CUDA_KERNEL_OUTPUT:
            return "cuda_kernel_output";
        default:
            throw std::runtime_error("Unknown ExprOp.");
    }
}

ExprOp exprOpFromExternalName(const std::string& op) {
    static const std::unordered_map<std::string, ExprOp> lookup = {
        {"input", ExprOp::INPUT},
        {"runtime_scalar", ExprOp::RUNTIME_SCALAR},
        {"tensor_runtime_scalar", ExprOp::TENSOR_RUNTIME_SCALAR},
        {"scalar_fp", ExprOp::SCALAR_FP},
        {"add", ExprOp::ADD},
        {"sub", ExprOp::SUB},
        {"mul", ExprOp::MUL},
        {"div", ExprOp::DIV},
        {"pow", ExprOp::POW},
        {"equal", ExprOp::EQUAL},
        {"eq", ExprOp::EQUAL},
        {"not_equal", ExprOp::NOT_EQUAL},
        {"ne", ExprOp::NOT_EQUAL},
        {"less", ExprOp::LESS},
        {"less_than", ExprOp::LESS},
        {"lt", ExprOp::LESS},
        {"less_equal", ExprOp::LESS_EQUAL},
        {"less_than_or_equal", ExprOp::LESS_EQUAL},
        {"le", ExprOp::LESS_EQUAL},
        {"greater", ExprOp::GREATER},
        {"greater_than", ExprOp::GREATER},
        {"gt", ExprOp::GREATER},
        {"greater_equal", ExprOp::GREATER_EQUAL},
        {"greater_than_or_equal", ExprOp::GREATER_EQUAL},
        {"ge", ExprOp::GREATER_EQUAL},
        {"logical_and", ExprOp::LOGICAL_AND},
        {"and", ExprOp::LOGICAL_AND},
        {"logical_or", ExprOp::LOGICAL_OR},
        {"or", ExprOp::LOGICAL_OR},
        {"logical_not", ExprOp::LOGICAL_NOT},
        {"not", ExprOp::LOGICAL_NOT},
        {"where", ExprOp::WHERE},
        {"select", ExprOp::WHERE},
        {"neg", ExprOp::NEG},
        {"abs", ExprOp::ABS},
        {"ceil", ExprOp::CEIL},
        {"floor", ExprOp::FLOOR},
        {"round", ExprOp::ROUND},
        {"trunc", ExprOp::TRUNC},
        {"sin", ExprOp::SIN},
        {"cos", ExprOp::COS},
        {"tan", ExprOp::TAN},
        {"asin", ExprOp::ASIN},
        {"acos", ExprOp::ACOS},
        {"atan", ExprOp::ATAN},
        {"sinh", ExprOp::SINH},
        {"cosh", ExprOp::COSH},
        {"asinh", ExprOp::ASINH},
        {"acosh", ExprOp::ACOSH},
        {"atanh", ExprOp::ATANH},
        {"erf", ExprOp::ERF},
        {"erfc", ExprOp::ERFC},
        {"erfcx", ExprOp::ERFCX},
        {"erfinv", ExprOp::ERFINV},
        {"erfcinv", ExprOp::ERFCINV},
        {"tgamma", ExprOp::TGAMMA},
        {"lgamma", ExprOp::LGAMMA},
        {"digamma", ExprOp::DIGAMMA},
        {"exp", ExprOp::EXP},
        {"expm1", ExprOp::EXPM1},
        {"exp2", ExprOp::EXP2},
        {"exp10", ExprOp::EXP10},
        {"ln", ExprOp::LN},
        {"log1p", ExprOp::LOG1P},
        {"log2", ExprOp::LOG2},
        {"log10", ExprOp::LOG10},
        {"sqrt", ExprOp::SQRT},
        {"tanh", ExprOp::TANH},
        {"normcdf", ExprOp::NORMCDF},
        {"rope", ExprOp::ROPE},
        {"rotary_position_embedding", ExprOp::ROPE},
        {"softmax", ExprOp::SOFTMAX},
        {"fill", ExprOp::FILL},
        {"reshape", ExprOp::RESHAPE},
        {"strided_view", ExprOp::STRIDED_VIEW},
        {"alias_view", ExprOp::STRIDED_VIEW},
        {"strided_view_backward", ExprOp::STRIDED_VIEW_BACKWARD},
        {"alias_view_backward", ExprOp::STRIDED_VIEW_BACKWARD},
        {"unsqueeze", ExprOp::UNSQUEEZE},
        {"squeeze", ExprOp::SQUEEZE},
        {"transpose", ExprOp::TRANSPOSE},
        {"min", ExprOp::MIN},
        {"max", ExprOp::MAX},
        {"min_grad_left", ExprOp::MIN_GRAD_LEFT},
        {"min_grad_right", ExprOp::MIN_GRAD_RIGHT},
        {"max_grad_left", ExprOp::MAX_GRAD_LEFT},
        {"max_grad_right", ExprOp::MAX_GRAD_RIGHT},
        {"matmul", ExprOp::MATMUL},
        {"gemm", ExprOp::GEMM},
        {"conv2d", ExprOp::CONV2D},
        {"conv2d_backward_data", ExprOp::CONV2D_BACKWARD_DATA},
        {"conv2d_backward_filter", ExprOp::CONV2D_BACKWARD_FILTER},
        {"conv3d", ExprOp::CONV3D},
        {"conv3d_backward_data", ExprOp::CONV3D_BACKWARD_DATA},
        {"conv3d_backward_filter", ExprOp::CONV3D_BACKWARD_FILTER},
        {"reduce_sum", ExprOp::REDUCE_SUM},
        {"reduce_prod", ExprOp::REDUCE_PROD},
        {"reduce_min", ExprOp::REDUCE_MIN},
        {"reduce_max", ExprOp::REDUCE_MAX},
        {"reduce_argmin", ExprOp::REDUCE_ARGMIN},
        {"reduce_argmax", ExprOp::REDUCE_ARGMAX},
        {"reduce_min_backward", ExprOp::REDUCE_MIN_BACKWARD},
        {"reduce_max_backward", ExprOp::REDUCE_MAX_BACKWARD},
        {"reduce_avg", ExprOp::REDUCE_AVG},
        {"reduce_norm1", ExprOp::REDUCE_NORM1},
        {"reduce_norm2", ExprOp::REDUCE_NORM2},
        {"rmsnorm", ExprOp::RMSNORM},
        {"attention", ExprOp::ATTENTION},
        {"attention_backward_q", ExprOp::ATTENTION_BACKWARD_Q},
        {"attention_backward_k", ExprOp::ATTENTION_BACKWARD_K},
        {"attention_backward_v", ExprOp::ATTENTION_BACKWARD_V},
        {"attention_backward_bias", ExprOp::ATTENTION_BACKWARD_BIAS},
        {"embedding_lookup", ExprOp::EMBEDDING_LOOKUP},
        {"cuda_kernel_output", ExprOp::CUDA_KERNEL_OUTPUT},
    };

    auto it = lookup.find(op);
    if (it == lookup.end()) {
        throw std::runtime_error("Unknown expression op '" + op + "'.");
    }
    return it->second;
}

json optionalDTypeJson(const std::optional<DataType>& dtype) {
    if (!dtype.has_value()) {
        return nullptr;
    }
    return json(dtype.value());
}

std::optional<DataType> optionalDTypeFromJson(const json& value) {
    if (value.is_null()) {
        return std::nullopt;
    }
    return value.get<DataType>();
}

void setOptionalDTypeJson(json& dst, const char* key, const std::optional<DataType>& dtype) { dst[key] = optionalDTypeJson(dtype); }

void parseOptionalDTypeField(const json& src, const char* key, std::optional<DataType>& dst) {
    if (!src.contains(key)) {
        dst = std::nullopt;
        return;
    }
    dst = optionalDTypeFromJson(src.at(key));
}

const char* matmulEpilogueName(MatmulEpilogue epilogue) {
    switch (epilogue) {
        case MatmulEpilogue::Default:
            return "default";
        case MatmulEpilogue::Relu:
            return "relu";
        case MatmulEpilogue::Gelu:
            return "gelu";
    }
    throw std::runtime_error("Unknown MatmulEpilogue value.");
}

MatmulEpilogue matmulEpilogueFromName(const std::string& name) {
    if (name == "default")
        return MatmulEpilogue::Default;
    if (name == "relu")
        return MatmulEpilogue::Relu;
    if (name == "gelu")
        return MatmulEpilogue::Gelu;
    throw std::runtime_error("Unknown matmul epilogue name in serialized expression: " + name);
}

const char* matmulBackwardEpilogueName(MatmulBackwardEpilogue epilogue) {
    switch (epilogue) {
        case MatmulBackwardEpilogue::Default:
            return "default";
        case MatmulBackwardEpilogue::DRelu:
            return "drelu";
        case MatmulBackwardEpilogue::DGelu:
            return "dgelu";
    }
    throw std::runtime_error("Unknown MatmulBackwardEpilogue value.");
}

MatmulBackwardEpilogue matmulBackwardEpilogueFromName(const std::string& name) {
    if (name == "default")
        return MatmulBackwardEpilogue::Default;
    if (name == "drelu")
        return MatmulBackwardEpilogue::DRelu;
    if (name == "dgelu")
        return MatmulBackwardEpilogue::DGelu;
    throw std::runtime_error("Unknown matmul backward epilogue name in serialized expression: " + name);
}

uint64_t fnv1a64(const std::string& text) {
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;

    uint64_t hash = kOffset;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= kPrime;
    }
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << value;
    return ss.str();
}

json exprNodeToJson(const ExprNode& node) {
    json j;
    j["op"] = exprOpExternalName(node.op);
    j["lhs"] = node.lhs;
    j["rhs"] = node.rhs;
    j["aux"] = node.aux;
    j["input_slot"] = node.input_slot;
    j["scalar_fp"] = node.scalar_fp;
    j["alpha_fp"] = node.alpha_fp;
    j["beta_fp"] = node.beta_fp;
    j["alpha_node"] = node.alpha_node;
    j["beta_node"] = node.beta_node;
    j["transpose_lhs"] = node.transpose_lhs;
    j["transpose_rhs"] = node.transpose_rhs;
    j["transpose_aux"] = node.transpose_aux;
    j["matmul_epilogue"] = matmulEpilogueName(node.matmul_epilogue);
    j["matmul_backward_epilogue"] = matmulBackwardEpilogueName(node.matmul_backward_epilogue);
    j["matmul_epilogue_aux"] = node.matmul_epilogue_aux;
    j["conv_stride_d"] = node.conv_stride_d;
    j["conv_stride_h"] = node.conv_stride_h;
    j["conv_stride_w"] = node.conv_stride_w;
    j["conv_pad_d"] = node.conv_pad_d;
    j["conv_pad_h"] = node.conv_pad_h;
    j["conv_pad_w"] = node.conv_pad_w;
    j["softmax_algorithm"] = static_cast<int>(node.softmax_algorithm);
    j["softmax_mode"] = static_cast<int>(node.softmax_mode);
    j["attention_q_layout"] = static_cast<int>(node.attention_q_layout);
    j["attention_k_layout"] = static_cast<int>(node.attention_k_layout);
    j["attention_v_layout"] = static_cast<int>(node.attention_v_layout);
    j["attention_o_layout"] = static_cast<int>(node.attention_o_layout);
    j["attention_mask_kind"] = static_cast<int>(node.attention_mask_kind);
    j["attention_diagonal_left_bound"] = node.attention_diagonal_left_bound;
    j["attention_diagonal_right_bound"] = node.attention_diagonal_right_bound;
    j["attention_has_scale"] = node.attention_has_scale;
    j["attention_scale"] = node.attention_scale;
    j["attention_use_alibi_mask"] = node.attention_use_alibi_mask;
    j["attention_use_bias"] = node.attention_use_bias;
    j["attention_use_padding_mask"] = node.attention_use_padding_mask;
    j["attention_use_ragged_offsets"] = node.attention_use_ragged_offsets;
    j["attention_use_paged_kv_cache"] = node.attention_use_paged_kv_cache;
    j["attention_paged_kv_max_sequence_length"] = node.attention_paged_kv_max_sequence_length;
    j["attention_dropout_probability"] = node.attention_dropout_probability;
    j["attention_use_fp8_forward_scaling"] = node.attention_use_fp8_forward_scaling;
    j["attention_seq_len_q_node"] = node.attention_seq_len_q_node;
    j["attention_seq_len_kv_node"] = node.attention_seq_len_kv_node;
    j["attention_ragged_offset_q_node"] = node.attention_ragged_offset_q_node;
    j["attention_ragged_offset_kv_node"] = node.attention_ragged_offset_kv_node;
    j["attention_page_table_k_node"] = node.attention_page_table_k_node;
    j["attention_page_table_v_node"] = node.attention_page_table_v_node;
    j["attention_dropout_seed_node"] = node.attention_dropout_seed_node;
    j["attention_dropout_offset_node"] = node.attention_dropout_offset_node;
    j["attention_descale_q_node"] = node.attention_descale_q_node;
    j["attention_descale_k_node"] = node.attention_descale_k_node;
    j["attention_descale_v_node"] = node.attention_descale_v_node;
    j["attention_descale_s_node"] = node.attention_descale_s_node;
    j["attention_scale_s_node"] = node.attention_scale_s_node;
    j["attention_scale_o_node"] = node.attention_scale_o_node;
    j["attention_amax_s_node"] = node.attention_amax_s_node;
    j["attention_amax_o_node"] = node.attention_amax_o_node;
    j["rope_sequence_axis"] = node.rope_sequence_axis;
    j["rope_head_dim_axis"] = node.rope_head_dim_axis;
    j["rope_rotary_dim"] = node.rope_rotary_dim;
    j["rope_base"] = node.rope_base;
    j["rope_position_offset"] = node.rope_position_offset;
    j["rope_interleaved"] = node.rope_interleaved;
    j["rope_inverse"] = node.rope_inverse;
    j["rope_scaling_kind"] = static_cast<int>(node.rope_scaling_kind);
    j["rope_scaling_factor"] = node.rope_scaling_factor;
    j["rope_original_max_position_embeddings"] = node.rope_original_max_position_embeddings;
    j["rope_attention_factor"] = node.rope_attention_factor;
    j["rope_yarn_beta_fast"] = node.rope_yarn_beta_fast;
    j["rope_yarn_beta_slow"] = node.rope_yarn_beta_slow;
    j["rope_llama3_low_freq_factor"] = node.rope_llama3_low_freq_factor;
    j["rope_llama3_high_freq_factor"] = node.rope_llama3_high_freq_factor;
    j["rope_long_rope_short_factors"] = node.rope_long_rope_short_factors;
    j["rope_long_rope_long_factors"] = node.rope_long_rope_long_factors;
    j["rope_allow_in_place_materialization"] = node.rope_allow_in_place_materialization;
    j["rms_norm_normalized_feature_count"] = node.rms_norm_normalized_feature_count;
    j["rms_norm_epsilon"] = node.rms_norm_epsilon;
    j["rms_norm_fused_activation"] = toString(node.rms_norm_fused_activation);
    j["embedding_has_padding_index"] = node.embedding_has_padding_index;
    j["embedding_padding_index"] = node.embedding_padding_index;
    setOptionalDTypeJson(j, "input_tensor_dtype", node.input_tensor_dtype);
    setOptionalDTypeJson(j, "output_dtype", node.output_dtype);
    setOptionalDTypeJson(j, "compute_dtype", node.compute_dtype);
    setOptionalDTypeJson(j, "backward_output_dtype", node.backward_output_dtype);
    setOptionalDTypeJson(j, "backward_compute_dtype", node.backward_compute_dtype);
    j["reduction_axes"] = node.reduction_axes;
    j["reshape_dims"] = node.reshape_dims;
    j["view_dims"] = node.view_dims;
    j["view_strides"] = node.view_strides;
    j["view_element_offset"] = node.view_element_offset;
    j["squeeze_axes"] = node.squeeze_axes;
    j["unsqueeze_axes"] = node.unsqueeze_axes;
    j["fill_dims"] = node.fill_dims;
    j["cuda_kernel_spec_index"] = node.cuda_kernel_spec_index;
    j["cuda_kernel_output_index"] = node.cuda_kernel_output_index;
    j["cuda_kernel_input_nodes"] = node.cuda_kernel_input_nodes;
    return j;
}

ExprNode exprNodeFromJson(const json& j) {
    ExprNode node;
    node.op = exprOpFromExternalName(j.at("op").get<std::string>());
    node.lhs = j.value("lhs", UINT32_MAX);
    node.rhs = j.value("rhs", UINT32_MAX);
    node.aux = j.value("aux", UINT32_MAX);
    node.input_slot = j.value("input_slot", UINT32_MAX);
    node.scalar_fp = j.value("scalar_fp", 0.0);
    node.alpha_fp = j.value("alpha_fp", 1.0);
    node.beta_fp = j.value("beta_fp", 0.0);
    node.alpha_node = j.value("alpha_node", UINT32_MAX);
    node.beta_node = j.value("beta_node", UINT32_MAX);
    node.transpose_lhs = j.value("transpose_lhs", false);
    node.transpose_rhs = j.value("transpose_rhs", false);
    node.transpose_aux = j.value("transpose_aux", false);
    node.matmul_epilogue = matmulEpilogueFromName(j.value("matmul_epilogue", std::string("default")));
    node.matmul_backward_epilogue = matmulBackwardEpilogueFromName(j.value("matmul_backward_epilogue", std::string("default")));
    node.matmul_epilogue_aux = j.value("matmul_epilogue_aux", UINT32_MAX);
    node.conv_stride_d = j.value("conv_stride_d", 1);
    node.conv_stride_h = j.value("conv_stride_h", 1);
    node.conv_stride_w = j.value("conv_stride_w", 1);
    node.conv_pad_d = j.value("conv_pad_d", 0);
    node.conv_pad_h = j.value("conv_pad_h", 0);
    node.conv_pad_w = j.value("conv_pad_w", 0);
    node.softmax_algorithm = static_cast<cudnnSoftmaxAlgorithm_t>(j.value("softmax_algorithm", static_cast<int>(CUDNN_SOFTMAX_ACCURATE)));
    node.softmax_mode = static_cast<cudnnSoftmaxMode_t>(j.value("softmax_mode", static_cast<int>(CUDNN_SOFTMAX_MODE_CHANNEL)));
    node.attention_q_layout =
        static_cast<AttentionTensorLayout>(j.value("attention_q_layout", static_cast<int>(AttentionTensorLayout::BHSD)));
    node.attention_k_layout =
        static_cast<AttentionTensorLayout>(j.value("attention_k_layout", static_cast<int>(AttentionTensorLayout::BHSD)));
    node.attention_v_layout =
        static_cast<AttentionTensorLayout>(j.value("attention_v_layout", static_cast<int>(AttentionTensorLayout::BHSD)));
    node.attention_o_layout =
        static_cast<AttentionTensorLayout>(j.value("attention_o_layout", static_cast<int>(AttentionTensorLayout::BHSD)));
    node.attention_mask_kind = static_cast<AttentionMaskKind>(j.value("attention_mask_kind", static_cast<int>(AttentionMaskKind::None)));
    node.attention_diagonal_left_bound = j.value("attention_diagonal_left_bound", int64_t{0});
    node.attention_diagonal_right_bound = j.value("attention_diagonal_right_bound", int64_t{0});
    node.attention_has_scale = j.value("attention_has_scale", false);
    node.attention_scale = j.value("attention_scale", 0.0f);
    node.attention_use_alibi_mask = j.value("attention_use_alibi_mask", false);
    node.attention_use_bias = j.value("attention_use_bias", false);
    node.attention_use_padding_mask = j.value("attention_use_padding_mask", false);
    node.attention_use_ragged_offsets = j.value("attention_use_ragged_offsets", false);
    node.attention_use_paged_kv_cache = j.value("attention_use_paged_kv_cache", false);
    node.attention_paged_kv_max_sequence_length = j.value("attention_paged_kv_max_sequence_length", int64_t{0});
    node.attention_dropout_probability = j.value("attention_dropout_probability", 0.0f);
    node.attention_use_fp8_forward_scaling = j.value("attention_use_fp8_forward_scaling", false);
    node.attention_seq_len_q_node = j.value("attention_seq_len_q_node", UINT32_MAX);
    node.attention_seq_len_kv_node = j.value("attention_seq_len_kv_node", UINT32_MAX);
    node.attention_ragged_offset_q_node = j.value("attention_ragged_offset_q_node", UINT32_MAX);
    node.attention_ragged_offset_kv_node = j.value("attention_ragged_offset_kv_node", UINT32_MAX);
    node.attention_page_table_k_node = j.value("attention_page_table_k_node", UINT32_MAX);
    node.attention_page_table_v_node = j.value("attention_page_table_v_node", UINT32_MAX);
    node.attention_dropout_seed_node = j.value("attention_dropout_seed_node", UINT32_MAX);
    node.attention_dropout_offset_node = j.value("attention_dropout_offset_node", UINT32_MAX);
    node.attention_descale_q_node = j.value("attention_descale_q_node", UINT32_MAX);
    node.attention_descale_k_node = j.value("attention_descale_k_node", UINT32_MAX);
    node.attention_descale_v_node = j.value("attention_descale_v_node", UINT32_MAX);
    node.attention_descale_s_node = j.value("attention_descale_s_node", UINT32_MAX);
    node.attention_scale_s_node = j.value("attention_scale_s_node", UINT32_MAX);
    node.attention_scale_o_node = j.value("attention_scale_o_node", UINT32_MAX);
    node.attention_amax_s_node = j.value("attention_amax_s_node", UINT32_MAX);
    node.attention_amax_o_node = j.value("attention_amax_o_node", UINT32_MAX);
    node.rope_sequence_axis = j.value("rope_sequence_axis", uint32_t{2});
    node.rope_head_dim_axis = j.value("rope_head_dim_axis", uint32_t{3});
    node.rope_rotary_dim = j.value("rope_rotary_dim", uint64_t{0});
    node.rope_base = j.value("rope_base", 10000.0);
    node.rope_position_offset = j.value("rope_position_offset", int64_t{0});
    node.rope_interleaved = j.value("rope_interleaved", false);
    node.rope_inverse = j.value("rope_inverse", false);
    node.rope_scaling_kind = static_cast<RotaryScalingKind>(j.value("rope_scaling_kind", static_cast<int>(RotaryScalingKind::None)));
    node.rope_scaling_factor = j.value("rope_scaling_factor", 1.0);
    node.rope_original_max_position_embeddings = j.value("rope_original_max_position_embeddings", uint64_t{0});
    node.rope_attention_factor = j.value("rope_attention_factor", 1.0);
    node.rope_yarn_beta_fast = j.value("rope_yarn_beta_fast", 32.0);
    node.rope_yarn_beta_slow = j.value("rope_yarn_beta_slow", 1.0);
    node.rope_llama3_low_freq_factor = j.value("rope_llama3_low_freq_factor", 1.0);
    node.rope_llama3_high_freq_factor = j.value("rope_llama3_high_freq_factor", 4.0);
    node.rope_long_rope_short_factors = j.value("rope_long_rope_short_factors", std::vector<double>{});
    node.rope_long_rope_long_factors = j.value("rope_long_rope_long_factors", std::vector<double>{});
    node.rope_allow_in_place_materialization = j.value("rope_allow_in_place_materialization", false);
    node.rms_norm_normalized_feature_count = j.value("rms_norm_normalized_feature_count", uint64_t{0});
    node.rms_norm_epsilon = j.value("rms_norm_epsilon", 1.0e-5);
    node.rms_norm_fused_activation = j.contains("rms_norm_fused_activation")
                                         ? cudnnRmsNormFusedActivationFromString(j.at("rms_norm_fused_activation").get<std::string>())
                                         : CudnnRmsNormFusedActivation::NONE;
    node.embedding_has_padding_index = j.value("embedding_has_padding_index", false);
    node.embedding_padding_index = j.value("embedding_padding_index", uint64_t{0});
    parseOptionalDTypeField(j, "input_tensor_dtype", node.input_tensor_dtype);
    parseOptionalDTypeField(j, "output_dtype", node.output_dtype);
    parseOptionalDTypeField(j, "compute_dtype", node.compute_dtype);
    parseOptionalDTypeField(j, "backward_output_dtype", node.backward_output_dtype);
    parseOptionalDTypeField(j, "backward_compute_dtype", node.backward_compute_dtype);
    node.reduction_axes = j.value("reduction_axes", std::vector<uint64_t>{});
    node.reshape_dims = j.value("reshape_dims", std::vector<uint64_t>{});
    node.view_dims = j.value("view_dims", std::vector<uint64_t>{});
    node.view_strides = j.value("view_strides", std::vector<uint64_t>{});
    node.view_element_offset = j.value("view_element_offset", uint64_t{0});
    node.squeeze_axes = j.value("squeeze_axes", std::vector<uint64_t>{});
    node.unsqueeze_axes = j.value("unsqueeze_axes", std::vector<uint64_t>{});
    node.fill_dims = j.value("fill_dims", std::vector<uint64_t>{});
    node.cuda_kernel_spec_index = j.value("cuda_kernel_spec_index", UINT32_MAX);
    node.cuda_kernel_output_index = j.value("cuda_kernel_output_index", UINT32_MAX);
    node.cuda_kernel_input_nodes = j.value("cuda_kernel_input_nodes", std::vector<uint32_t>{});
    return node;
}

}  // namespace

Stream& Expression::getNextHelperStream(uint32_t gpu_num) { return helperStreamPool.getNextHelperStream(gpu_num); }

std::set<std::string> Expression::getInputNames() const {
    if (expr == nullptr)
        return {};
    return expr->getInputNames();
}

std::string formatFloatCanonical(double x) {
    std::ostringstream ss;
    ss << std::setprecision(9) << x;
    return ss.str();
}

bool isCommutative(ExprOp op) {
    return op == ExprOp::ADD || op == ExprOp::MUL || op == ExprOp::MIN || op == ExprOp::MAX || op == ExprOp::EQUAL ||
           op == ExprOp::NOT_EQUAL || op == ExprOp::LOGICAL_AND || op == ExprOp::LOGICAL_OR;
}

std::string opName(ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
            return "IN";
        case ExprOp::RUNTIME_SCALAR:
            return "RIN";
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            return "TRIN";
        case ExprOp::SCALAR_FP:
            return "F32";
        case ExprOp::ADD:
            return "ADD";
        case ExprOp::SUB:
            return "SUB";
        case ExprOp::MUL:
            return "MUL";
        case ExprOp::DIV:
            return "DIV";
        case ExprOp::POW:
            return "POW";
        case ExprOp::EQUAL:
            return "EQ";
        case ExprOp::NOT_EQUAL:
            return "NE";
        case ExprOp::LESS:
            return "LT";
        case ExprOp::LESS_EQUAL:
            return "LE";
        case ExprOp::GREATER:
            return "GT";
        case ExprOp::GREATER_EQUAL:
            return "GE";
        case ExprOp::LOGICAL_AND:
            return "LAND";
        case ExprOp::LOGICAL_OR:
            return "LOR";
        case ExprOp::LOGICAL_NOT:
            return "LNOT";
        case ExprOp::WHERE:
            return "WHERE";
        case ExprOp::NEG:
            return "NEG";
        case ExprOp::ABS:
            return "ABS";
        case ExprOp::CEIL:
            return "CEIL";
        case ExprOp::FLOOR:
            return "FLOOR";
        case ExprOp::ROUND:
            return "ROUND";
        case ExprOp::TRUNC:
            return "TRUNC";
        case ExprOp::SIN:
            return "SIN";
        case ExprOp::COS:
            return "COS";
        case ExprOp::TAN:
            return "TAN";
        case ExprOp::ASIN:
            return "ASIN";
        case ExprOp::ACOS:
            return "ACOS";
        case ExprOp::ATAN:
            return "ATAN";
        case ExprOp::SINH:
            return "SINH";
        case ExprOp::COSH:
            return "COSH";
        case ExprOp::ASINH:
            return "ASINH";
        case ExprOp::ACOSH:
            return "ACOSH";
        case ExprOp::ATANH:
            return "ATANH";
        case ExprOp::ERF:
            return "ERF";
        case ExprOp::ERFC:
            return "ERFC";
        case ExprOp::ERFCX:
            return "ERFCX";
        case ExprOp::ERFINV:
            return "ERFINV";
        case ExprOp::ERFCINV:
            return "ERFCINV";
        case ExprOp::TGAMMA:
            return "TGAMMA";
        case ExprOp::LGAMMA:
            return "LGAMMA";
        case ExprOp::DIGAMMA:
            return "DIGAMMA";
        case ExprOp::EXP:
            return "EXP";
        case ExprOp::EXPM1:
            return "EXPM1";
        case ExprOp::LN:
            return "LOG";
        case ExprOp::LOG1P:
            return "LOG1P";
        case ExprOp::SQRT:
            return "SQRT";
        case ExprOp::TANH:
            return "TANH";
        case ExprOp::NORMCDF:
            return "NORMCDF";
        case ExprOp::ROPE:
            return "ROPE";
        case ExprOp::SOFTMAX:
            return "SOFTMAX";
        case ExprOp::FILL:
            return "FILL";
        case ExprOp::RESHAPE:
            return "RESHAPE";
        case ExprOp::STRIDED_VIEW:
            return "VIEW";
        case ExprOp::STRIDED_VIEW_BACKWARD:
            return "VIEW_BWD";
        case ExprOp::UNSQUEEZE:
            return "UNSQ";
        case ExprOp::SQUEEZE:
            return "SQZ";
        case ExprOp::TRANSPOSE:
            return "TRANSPOSE";
        case ExprOp::EXP2:
            return "EXP2";
        case ExprOp::EXP10:
            return "EXP10";
        case ExprOp::LOG2:
            return "LOG2";
        case ExprOp::LOG10:
            return "LOG10";
        case ExprOp::MIN:
            return "MIN";
        case ExprOp::MAX:
            return "MAX";
        case ExprOp::MIN_GRAD_LEFT:
            return "MIN_GL";
        case ExprOp::MIN_GRAD_RIGHT:
            return "MIN_GR";
        case ExprOp::MAX_GRAD_LEFT:
            return "MAX_GL";
        case ExprOp::MAX_GRAD_RIGHT:
            return "MAX_GR";
        case ExprOp::MATMUL:
            return "MATMUL";
        case ExprOp::GEMM:
            return "GEMM";
        case ExprOp::CONV2D:
            return "CONV2D";
        case ExprOp::CONV2D_BACKWARD_DATA:
            return "CONV2D_BWD_DATA";
        case ExprOp::CONV2D_BACKWARD_FILTER:
            return "CONV2D_BWD_FILTER";
        case ExprOp::CONV3D:
            return "CONV3D";
        case ExprOp::CONV3D_BACKWARD_DATA:
            return "CONV3D_BWD_DATA";
        case ExprOp::CONV3D_BACKWARD_FILTER:
            return "CONV3D_BWD_FILTER";
        case ExprOp::REDUCE_SUM:
            return "RSUM";
        case ExprOp::REDUCE_PROD:
            return "RPROD";
        case ExprOp::REDUCE_MIN:
            return "RMIN";
        case ExprOp::REDUCE_MAX:
            return "RMAX";
        case ExprOp::REDUCE_ARGMIN:
            return "RARGMIN";
        case ExprOp::REDUCE_ARGMAX:
            return "RARGMAX";
        case ExprOp::REDUCE_MIN_BACKWARD:
            return "RMIN_BW";
        case ExprOp::REDUCE_MAX_BACKWARD:
            return "RMAX_BW";
        case ExprOp::REDUCE_AVG:
            return "RAVG";
        case ExprOp::REDUCE_NORM1:
            return "RNORM1";
        case ExprOp::REDUCE_NORM2:
            return "RNORM2";
        case ExprOp::RMSNORM:
            return "RMSNORM";
        case ExprOp::ATTENTION:
            return "ATTENTION";
        case ExprOp::ATTENTION_BACKWARD_Q:
            return "ATTN_BW_Q";
        case ExprOp::ATTENTION_BACKWARD_K:
            return "ATTN_BW_K";
        case ExprOp::ATTENTION_BACKWARD_V:
            return "ATTN_BW_V";
        case ExprOp::ATTENTION_BACKWARD_BIAS:
            return "ATTN_BW_BIAS";
        case ExprOp::EMBEDDING_LOOKUP:
            return "EMBEDDING_LOOKUP";
        case ExprOp::CUDA_KERNEL_OUTPUT:
            return "CUDA_KERNEL";
        default:
            throw std::runtime_error("Unknown ExprOp");
    }
}

static std::string formatUIntVectorCanonical(const std::vector<uint64_t>& values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            ss << ",";
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static std::string formatDoubleVectorCanonical(const std::vector<double>& values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            ss << ",";
        ss << formatFloatCanonical(values[i]);
    }
    ss << "]";
    return ss.str();
}

static std::string formatOptionalDTypeCanonical(const std::optional<DataType>& dtype) {
    if (!dtype.has_value()) {
        return "none";
    }
    return TensorDescriptor::getElementTypeName(dtype.value());
}

static void appendNodeDTypeMetadata(std::string& out, const ExprNode& n) {
    out += ";input=" + formatOptionalDTypeCanonical(n.input_tensor_dtype);
    out += ";out=" + formatOptionalDTypeCanonical(n.output_dtype);
    out += ";compute=" + formatOptionalDTypeCanonical(n.compute_dtype);
    out += ";bwd_out=" + formatOptionalDTypeCanonical(n.backward_output_dtype);
    out += ";bwd_compute=" + formatOptionalDTypeCanonical(n.backward_compute_dtype);
}

static std::string canonicalizeNode(const PhysicalExpression& expr,
                                    uint32_t nodeIndex,
                                    std::vector<std::string>& memo,
                                    std::vector<uint8_t>& memoReady) {
    if (nodeIndex >= expr.nodes.size()) {
        throw std::runtime_error("canonicalizeNode nodeIndex out of range.");
    }

    if (memoReady[nodeIndex]) {
        return memo[nodeIndex];
    }

    const ExprNode& n = expr.nodes[nodeIndex];
    std::string out;

    switch (n.op) {
        case ExprOp::INPUT:
            out = "IN" + std::to_string(n.input_slot);
            break;

        case ExprOp::RUNTIME_SCALAR:
            out = "RIN" + std::to_string(n.input_slot);
            break;

        case ExprOp::TENSOR_RUNTIME_SCALAR:
            out = "TRIN" + std::to_string(n.input_slot);
            break;

        case ExprOp::SCALAR_FP:
            out = "F32(" + formatFloatCanonical(n.scalar_fp) + ")";
            break;

        case ExprOp::FILL:
            out = "FILL(" + formatFloatCanonical(n.scalar_fp) + ";dims=" + formatUIntVectorCanonical(n.fill_dims) + ")";
            break;

        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::CEIL:
        case ExprOp::FLOOR:
        case ExprOp::ROUND:
        case ExprOp::TRUNC:
        case ExprOp::SIN:
        case ExprOp::COS:
        case ExprOp::TAN:
        case ExprOp::ASIN:
        case ExprOp::ACOS:
        case ExprOp::ATAN:
        case ExprOp::SINH:
        case ExprOp::COSH:
        case ExprOp::ASINH:
        case ExprOp::ACOSH:
        case ExprOp::ATANH:
        case ExprOp::ERF:
        case ExprOp::ERFC:
        case ExprOp::ERFCX:
        case ExprOp::ERFINV:
        case ExprOp::ERFCINV:
        case ExprOp::TGAMMA:
        case ExprOp::LGAMMA:
        case ExprOp::DIGAMMA:
        case ExprOp::EXP:
        case ExprOp::EXPM1:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG1P:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::TANH:
        case ExprOp::NORMCDF:
        case ExprOp::LOGICAL_NOT:
        case ExprOp::TRANSPOSE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) + ")";
            break;
        case ExprOp::ROPE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) + ";seqAxis=" + std::to_string(n.rope_sequence_axis) +
                  ";dimAxis=" + std::to_string(n.rope_head_dim_axis) + ";rotaryDim=" + std::to_string(n.rope_rotary_dim) +
                  ";base=" + formatFloatCanonical(n.rope_base) + ";offset=" + std::to_string(n.rope_position_offset) +
                  ";interleaved=" + std::to_string(n.rope_interleaved ? 1 : 0) + ";inverse=" + std::to_string(n.rope_inverse ? 1 : 0) +
                  ";scaling=" + std::to_string(static_cast<int>(n.rope_scaling_kind)) +
                  ";factor=" + formatFloatCanonical(n.rope_scaling_factor) +
                  ";originalMax=" + std::to_string(n.rope_original_max_position_embeddings) +
                  ";attentionFactor=" + formatFloatCanonical(n.rope_attention_factor) +
                  ";yarnBetaFast=" + formatFloatCanonical(n.rope_yarn_beta_fast) +
                  ";yarnBetaSlow=" + formatFloatCanonical(n.rope_yarn_beta_slow) +
                  ";llama3LowFreqFactor=" + formatFloatCanonical(n.rope_llama3_low_freq_factor) +
                  ";llama3HighFreqFactor=" + formatFloatCanonical(n.rope_llama3_high_freq_factor) +
                  ";longRopeShortFactors=" + formatDoubleVectorCanonical(n.rope_long_rope_short_factors) +
                  ";longRopeLongFactors=" + formatDoubleVectorCanonical(n.rope_long_rope_long_factors) + ")";
            break;
        case ExprOp::SOFTMAX:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";algorithm=" + std::to_string(static_cast<int>(n.softmax_algorithm)) +
                  ";mode=" + std::to_string(static_cast<int>(n.softmax_mode)) + ")";
            break;
        case ExprOp::RESHAPE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";dims=" + formatUIntVectorCanonical(n.reshape_dims) + ")";
            break;
        case ExprOp::STRIDED_VIEW:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) + ";dims=" + formatUIntVectorCanonical(n.view_dims) +
                  ";strides=" + formatUIntVectorCanonical(n.view_strides) + ";offset=" + std::to_string(n.view_element_offset) + ")";
            break;
        case ExprOp::STRIDED_VIEW_BACKWARD:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";sourceDims=" + formatUIntVectorCanonical(n.fill_dims) + ";viewDims=" + formatUIntVectorCanonical(n.view_dims) +
                  ";viewStrides=" + formatUIntVectorCanonical(n.view_strides) + ";offset=" + std::to_string(n.view_element_offset) + ")";
            break;
        case ExprOp::UNSQUEEZE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";axes=" + formatUIntVectorCanonical(n.unsqueeze_axes) + ")";
            break;
        case ExprOp::SQUEEZE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";axes=" + formatUIntVectorCanonical(n.squeeze_axes) + ")";
            break;

        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2: {
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";axes=" + formatUIntVectorCanonical(n.reduction_axes) + ";squeeze=" + formatUIntVectorCanonical(n.squeeze_axes) + ")";
            break;
        }

        case ExprOp::REDUCE_MIN_BACKWARD:
        case ExprOp::REDUCE_MAX_BACKWARD: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);
            out = opName(n.op) + "(" + a + "," + b + ";axes=" + formatUIntVectorCanonical(n.reduction_axes) +
                  ";squeeze=" + formatUIntVectorCanonical(n.squeeze_axes) + ")";
            break;
        }

        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::EQUAL:
        case ExprOp::NOT_EQUAL:
        case ExprOp::LESS:
        case ExprOp::LESS_EQUAL:
        case ExprOp::GREATER:
        case ExprOp::GREATER_EQUAL:
        case ExprOp::LOGICAL_AND:
        case ExprOp::LOGICAL_OR:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::MATMUL:
        case ExprOp::RMSNORM:
        case ExprOp::EMBEDDING_LOOKUP:
        case ExprOp::CONV2D:
        case ExprOp::CONV2D_BACKWARD_DATA:
        case ExprOp::CONV2D_BACKWARD_FILTER:
        case ExprOp::CONV3D:
        case ExprOp::CONV3D_BACKWARD_DATA:
        case ExprOp::CONV3D_BACKWARD_FILTER: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);

            if (isCommutative(n.op) && a > b)
                std::swap(a, b);

            out = opName(n.op) + "(" + a + "," + b;
            if (n.op == ExprOp::MATMUL) {
                out += ";tA=" + std::to_string(n.transpose_lhs ? 1 : 0);
                out += ";tB=" + std::to_string(n.transpose_rhs ? 1 : 0);
                out += ";epilogue=" + std::string(matmulEpilogueName(n.matmul_epilogue));
                out += ";backwardEpilogue=" + std::string(matmulBackwardEpilogueName(n.matmul_backward_epilogue));
                if (n.matmul_epilogue_aux != UINT32_MAX) {
                    out += ";epilogueAux=" + canonicalizeNode(expr, n.matmul_epilogue_aux, memo, memoReady);
                }
            } else if (n.op == ExprOp::RMSNORM) {
                out += ";hidden=" + std::to_string(n.rms_norm_normalized_feature_count);
                out += ";epsilon=" + formatFloatCanonical(n.rms_norm_epsilon);
                out += ";fusedActivation=" + std::string(toString(n.rms_norm_fused_activation));
            } else if (n.op == ExprOp::EMBEDDING_LOOKUP) {
                out += ";padding=";
                out += n.embedding_has_padding_index ? std::to_string(n.embedding_padding_index) : std::string("none");
            } else if (n.op == ExprOp::CONV2D || n.op == ExprOp::CONV2D_BACKWARD_DATA || n.op == ExprOp::CONV2D_BACKWARD_FILTER ||
                       n.op == ExprOp::CONV3D || n.op == ExprOp::CONV3D_BACKWARD_DATA || n.op == ExprOp::CONV3D_BACKWARD_FILTER) {
                if (n.op == ExprOp::CONV3D || n.op == ExprOp::CONV3D_BACKWARD_DATA || n.op == ExprOp::CONV3D_BACKWARD_FILTER) {
                    out += ";sD=" + std::to_string(n.conv_stride_d);
                    out += ";pD=" + std::to_string(n.conv_pad_d);
                }
                out += ";sH=" + std::to_string(n.conv_stride_h);
                out += ";sW=" + std::to_string(n.conv_stride_w);
                out += ";pH=" + std::to_string(n.conv_pad_h);
                out += ";pW=" + std::to_string(n.conv_pad_w);
                if (!n.fill_dims.empty()) {
                    out += ";shape=";
                    for (size_t i = 0; i < n.fill_dims.size(); ++i) {
                        if (i != 0)
                            out += "x";
                        out += std::to_string(n.fill_dims[i]);
                    }
                }
            }
            out += ")";
            break;
        }

        case ExprOp::WHERE: {
            std::string cond = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string true_value = canonicalizeNode(expr, n.rhs, memo, memoReady);
            std::string false_value = canonicalizeNode(expr, n.aux, memo, memoReady);
            out = opName(n.op) + "(" + cond + "," + true_value + "," + false_value + ")";
            break;
        }

        case ExprOp::ATTENTION:
        case ExprOp::ATTENTION_BACKWARD_Q:
        case ExprOp::ATTENTION_BACKWARD_K:
        case ExprOp::ATTENTION_BACKWARD_V:
        case ExprOp::ATTENTION_BACKWARD_BIAS: {
            std::string q = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string k = canonicalizeNode(expr, n.rhs, memo, memoReady);
            std::string v = canonicalizeNode(expr, n.aux, memo, memoReady);
            out = opName(n.op) + "(" + q + "," + k + "," + v;
            if (n.op == ExprOp::ATTENTION && n.alpha_node != UINT32_MAX) {
                out += ",bias=" + canonicalizeNode(expr, n.alpha_node, memo, memoReady);
            }
            if (n.op != ExprOp::ATTENTION && n.alpha_node != UINT32_MAX) {
                out += ",dO=" + canonicalizeNode(expr, n.alpha_node, memo, memoReady);
            }
            if (n.op != ExprOp::ATTENTION && n.beta_node != UINT32_MAX) {
                out += ",bias=" + canonicalizeNode(expr, n.beta_node, memo, memoReady);
            }
            out +=
                ";qLayout=" + std::to_string(static_cast<int>(n.attention_q_layout)) +
                ";kLayout=" + std::to_string(static_cast<int>(n.attention_k_layout)) +
                ";vLayout=" + std::to_string(static_cast<int>(n.attention_v_layout)) +
                ";oLayout=" + std::to_string(static_cast<int>(n.attention_o_layout)) +
                ";mask=" + std::to_string(static_cast<int>(n.attention_mask_kind)) +
                ";left=" + std::to_string(n.attention_diagonal_left_bound) + ";right=" + std::to_string(n.attention_diagonal_right_bound) +
                ";hasScale=" + std::to_string(n.attention_has_scale ? 1 : 0) + ";scale=" + formatFloatCanonical(n.attention_scale) +
                ";alibi=" + std::to_string(n.attention_use_alibi_mask ? 1 : 0) + ";bias=" + std::to_string(n.attention_use_bias ? 1 : 0) +
                ";padding=" + std::to_string(n.attention_use_padding_mask ? 1 : 0) +
                ";ragged=" + std::to_string(n.attention_use_ragged_offsets ? 1 : 0) +
                ";dropout=" + formatFloatCanonical(n.attention_dropout_probability);
            if (n.attention_use_padding_mask) {
                if (n.attention_seq_len_q_node != UINT32_MAX) {
                    out += ",seqQ=" + canonicalizeNode(expr, n.attention_seq_len_q_node, memo, memoReady);
                }
                if (n.attention_seq_len_kv_node != UINT32_MAX) {
                    out += ",seqKV=" + canonicalizeNode(expr, n.attention_seq_len_kv_node, memo, memoReady);
                }
            }
            if (n.attention_use_ragged_offsets) {
                if (n.attention_ragged_offset_q_node != UINT32_MAX) {
                    out += ",raggedQ=" + canonicalizeNode(expr, n.attention_ragged_offset_q_node, memo, memoReady);
                }
                if (n.attention_ragged_offset_kv_node != UINT32_MAX) {
                    out += ",raggedKV=" + canonicalizeNode(expr, n.attention_ragged_offset_kv_node, memo, memoReady);
                }
            }
            if (n.attention_dropout_probability > 0.0f) {
                if (n.attention_dropout_seed_node != UINT32_MAX) {
                    out += ",dropoutSeed=" + canonicalizeNode(expr, n.attention_dropout_seed_node, memo, memoReady);
                }
                if (n.attention_dropout_offset_node != UINT32_MAX) {
                    out += ",dropoutOffset=" + canonicalizeNode(expr, n.attention_dropout_offset_node, memo, memoReady);
                }
            }
            if (n.attention_use_fp8_forward_scaling) {
                out += ",descaleQ=" + canonicalizeNode(expr, n.attention_descale_q_node, memo, memoReady);
                out += ",descaleK=" + canonicalizeNode(expr, n.attention_descale_k_node, memo, memoReady);
                out += ",descaleV=" + canonicalizeNode(expr, n.attention_descale_v_node, memo, memoReady);
                out += ",descaleS=" + canonicalizeNode(expr, n.attention_descale_s_node, memo, memoReady);
                out += ",scaleS=" + canonicalizeNode(expr, n.attention_scale_s_node, memo, memoReady);
                out += ",scaleO=" + canonicalizeNode(expr, n.attention_scale_o_node, memo, memoReady);
                out += ",amaxS=" + canonicalizeNode(expr, n.attention_amax_s_node, memo, memoReady);
                out += ",amaxO=" + canonicalizeNode(expr, n.attention_amax_o_node, memo, memoReady);
            }
            out += ")";
            break;
        }

        case ExprOp::CUDA_KERNEL_OUTPUT: {
            if (n.cuda_kernel_spec_index >= expr.cuda_kernel_expressions.size() ||
                !expr.cuda_kernel_expressions[n.cuda_kernel_spec_index]) {
                throw std::runtime_error("CudaKernelExpression node references missing kernel spec while canonicalizing.");
            }
            out = opName(n.op) + "(" + expr.cuda_kernel_expressions[n.cuda_kernel_spec_index]->cacheSignature() +
                  ";spec=" + std::to_string(n.cuda_kernel_spec_index) + ";out=" + std::to_string(n.cuda_kernel_output_index) + ";inputs=[";
            for (size_t i = 0; i < n.cuda_kernel_input_nodes.size(); ++i) {
                if (i > 0) {
                    out += ",";
                }
                out += canonicalizeNode(expr, n.cuda_kernel_input_nodes[i], memo, memoReady);
            }
            out += "])";
            break;
        }

        case ExprOp::GEMM: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);
            std::string c = canonicalizeNode(expr, n.aux, memo, memoReady);
            auto gemmScaleString = [&](const char* label, uint32_t scale_node, double scale_fp) {
                if (scale_node != UINT32_MAX) {
                    return std::string(";") + label + "Node=" + canonicalizeNode(expr, scale_node, memo, memoReady) + ";" + label +
                           "Scale=" + formatFloatCanonical(scale_fp);
                }
                return std::string(";") + label + "=" + formatFloatCanonical(scale_fp);
            };
            out = opName(n.op) + "(" + a + "," + b + "," + c + gemmScaleString("alpha", n.alpha_node, n.alpha_fp) +
                  gemmScaleString("beta", n.beta_node, n.beta_fp) + ";tA=" + std::to_string(n.transpose_lhs ? 1 : 0) +
                  ";tB=" + std::to_string(n.transpose_rhs ? 1 : 0) + ";tC=" + std::to_string(n.transpose_aux ? 1 : 0) +
                  ";epilogue=" + std::string(matmulEpilogueName(n.matmul_epilogue)) +
                  ";backwardEpilogue=" + std::string(matmulBackwardEpilogueName(n.matmul_backward_epilogue));
            if (n.matmul_epilogue_aux != UINT32_MAX) {
                out += ";epilogueAux=" + canonicalizeNode(expr, n.matmul_epilogue_aux, memo, memoReady);
            }
            out += ")";
            break;
        }

        default:
            throw std::runtime_error("Unsupported ExprOp in canonicalizeNode: " + std::to_string(static_cast<int>(n.op)));
    }

    appendNodeDTypeMetadata(out, n);
    memo[nodeIndex] = out;
    memoReady[nodeIndex] = 1;
    return memo[nodeIndex];
}

std::string canonicalize(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("canonicalize(PhysicalExpression): output_node out of range.");
    }

    std::vector<std::string> memo(expr.nodes.size());
    std::vector<uint8_t> memoReady(expr.nodes.size(), 0);
    return canonicalizeNode(expr, expr.output_node, memo, memoReady);
}

std::string canonicalize(const PhysicalExecutionStage& stage) {
    std::ostringstream ss;

    ss << "kind=";
    switch (stage.kind) {
        case PhysicalExecutionStage::Kind::FusedKernel:
            ss << "fused";
            break;
        case PhysicalExecutionStage::Kind::Reduction:
            ss << "reduction";
            break;
        case PhysicalExecutionStage::Kind::ArgMinMax:
            ss << "argminmax";
            break;
        case PhysicalExecutionStage::Kind::Softmax:
            ss << "softmax";
            break;
        case PhysicalExecutionStage::Kind::RmsNorm:
            ss << "rmsnorm";
            break;
        case PhysicalExecutionStage::Kind::Matmul:
            ss << "matmul";
            break;
        case PhysicalExecutionStage::Kind::Convolution:
            ss << "convolution";
            break;
        case PhysicalExecutionStage::Kind::ConvolutionBackward:
            ss << "convolution_backward";
            break;
        case PhysicalExecutionStage::Kind::ReduceMinMaxBackward:
            ss << "reduce_minmax_backward";
            break;
        default:
            throw std::runtime_error("canonicalize(PhysicalExecutionStage): unknown stage kind.");
    }

    ss << ";inputs=[";
    for (size_t i = 0; i < stage.input_value_ids.size(); ++i) {
        if (i > 0)
            ss << ",";
        ss << stage.input_value_ids[i];
    }
    ss << "]";

    std::vector<std::string> memo(stage.expr.nodes.size());
    std::vector<uint8_t> memoReady(stage.expr.nodes.size(), 0);

    ss << ";outputs=[";
    for (size_t i = 0; i < stage.outputs.size(); ++i) {
        if (i > 0)
            ss << ",";

        const CompiledStageOutput& out = stage.outputs[i];
        ss << "{local_node_idx=" << out.local_node_idx << ";expr=";

        if (out.local_node_idx == UINT32_MAX) {
            ss << "NONE";
        } else {
            ss << canonicalizeNode(stage.expr, out.local_node_idx, memo, memoReady);
        }

        ss << ";layout=";
        switch (out.materialized_layout) {
            case MaterializedTensorLayout::RowMajor:
                ss << "row_major";
                break;
            case MaterializedTensorLayout::Transposed:
                ss << "transposed";
                break;
            default:
                throw std::runtime_error("canonicalize(PhysicalExecutionStage): unknown output materialization layout.");
        }

        ss << "}";
    }
    ss << "]";

    return ss.str();
}

std::string canonicalize(const PhysicalOutputs& outputs) {
    if (!outputs.expr) {
        throw std::runtime_error("canonicalize(PhysicalOutputs): expr is null.");
    }

    std::ostringstream ss;
    ss << "inputs=[";
    for (size_t i = 0; i < outputs.expr->inputs.size(); ++i) {
        if (i > 0)
            ss << ",";
        const NamedInput& input = outputs.expr->inputs[i];
        ss << "{" << input.slot << ":" << input.name << ":" << namedInputKindToString(input.kind) << "}";
    }
    ss << "]";

    std::vector<std::string> memo(outputs.expr->nodes.size());
    std::vector<uint8_t> memoReady(outputs.expr->nodes.size(), 0);
    ss << ";outputs=[";
    for (size_t i = 0; i < outputs.outputs.size(); ++i) {
        if (i > 0)
            ss << ",";
        const NamedOutput& output = outputs.outputs[i];
        if (output.node_idx >= outputs.expr->nodes.size()) {
            throw std::runtime_error("canonicalize(PhysicalOutputs): output node index out of range.");
        }
        ss << "{" << output.name << ":" << canonicalizeNode(*outputs.expr, output.node_idx, memo, memoReady) << "}";
    }
    ss << "]";

    return ss.str();
}

std::string expressionHash(const PhysicalOutputs& outputs) { return "fnv1a64:" + hex64(fnv1a64(canonicalize(outputs))); }

void ExpressionDefinition::validate() const {
    if (!outputs.expr) {
        throw std::runtime_error("ExpressionDefinition requires a non-null PhysicalExpression.");
    }
    if (outputs.outputs.empty()) {
        throw std::runtime_error("ExpressionDefinition requires at least one named output.");
    }

    std::unordered_set<uint32_t> seen_slots;
    std::unordered_set<std::string> seen_input_names;
    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.name.empty()) {
            throw std::runtime_error("ExpressionDefinition input name cannot be empty.");
        }
        if (!seen_slots.insert(input.slot).second) {
            throw std::runtime_error("ExpressionDefinition has duplicate input slot " + std::to_string(input.slot) + ".");
        }
        if (!seen_input_names.insert(input.name).second) {
            throw std::runtime_error("ExpressionDefinition has duplicate input name '" + input.name + "'.");
        }
    }

    if (seen_slots.size() != outputs.expr->inputs.size()) {
        throw std::runtime_error("ExpressionDefinition input slot validation failed.");
    }
    for (uint32_t i = 0; i < outputs.expr->inputs.size(); ++i) {
        if (!seen_slots.contains(i)) {
            throw std::runtime_error("ExpressionDefinition input slots must be contiguous starting at 0.");
        }
    }

    auto validateNodeIndex = [&](uint32_t node_idx, const std::string& field_name) {
        if (node_idx == UINT32_MAX) {
            throw std::runtime_error("ExpressionDefinition node field '" + field_name + "' is missing.");
        }
        if (node_idx >= outputs.expr->nodes.size()) {
            throw std::runtime_error("ExpressionDefinition node field '" + field_name + "' is out of range.");
        }
    };

    for (size_t node_idx = 0; node_idx < outputs.expr->nodes.size(); ++node_idx) {
        const ExprNode& node = outputs.expr->nodes[node_idx];
        const auto node_index_u32 = static_cast<uint32_t>(node_idx);

        if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            if (node.input_slot == UINT32_MAX || node.input_slot >= outputs.expr->inputs.size()) {
                throw std::runtime_error("ExpressionDefinition input node has invalid input slot.");
            }
        }

        if (Expression::isUnaryOp(node.op)) {
            validateNodeIndex(node.lhs, "lhs");
            if (node.lhs >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition unary node must reference an earlier node.");
            }
        }
        if (Expression::isBinaryOp(node.op)) {
            validateNodeIndex(node.lhs, "lhs");
            validateNodeIndex(node.rhs, "rhs");
            if (node.lhs >= node_index_u32 || node.rhs >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition binary node must reference earlier nodes.");
            }
        }
        if (Expression::isTernaryOp(node.op)) {
            validateNodeIndex(node.lhs, "lhs");
            validateNodeIndex(node.rhs, "rhs");
            validateNodeIndex(node.aux, "aux");
            if (node.lhs >= node_index_u32 || node.rhs >= node_index_u32 || node.aux >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ternary node must reference earlier nodes.");
            }
        }
        if ((node.op == ExprOp::GEMM) && node.alpha_node != UINT32_MAX && node.alpha_node >= node_index_u32) {
            throw std::runtime_error("ExpressionDefinition GEMM alpha_node must reference an earlier node.");
        }
        if ((node.op == ExprOp::GEMM) && node.beta_node != UINT32_MAX && node.beta_node >= node_index_u32) {
            throw std::runtime_error("ExpressionDefinition GEMM beta_node must reference an earlier node.");
        }
        if ((node.op == ExprOp::MATMUL || node.op == ExprOp::GEMM) && node.matmul_backward_epilogue != MatmulBackwardEpilogue::Default) {
            validateNodeIndex(node.matmul_epilogue_aux, "matmul backward epilogue aux");
            if (node.matmul_epilogue_aux >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition matmul backward epilogue aux must reference an earlier node.");
            }
        }
        if ((node.op == ExprOp::ATTENTION) && node.attention_use_bias) {
            validateNodeIndex(node.alpha_node, "attention bias");
            if (node.alpha_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION bias node must reference an earlier node.");
            }
        }
        if ((node.op == ExprOp::ATTENTION || node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
             node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) &&
            node.attention_use_padding_mask) {
            validateNodeIndex(node.attention_seq_len_q_node, "attention q_seq_len");
            validateNodeIndex(node.attention_seq_len_kv_node, "attention kv_seq_len");
            if (node.attention_seq_len_q_node >= node_index_u32 || node.attention_seq_len_kv_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION padding-mask sequence length nodes must reference earlier nodes.");
            }
        }
        if ((node.op == ExprOp::ATTENTION || node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
             node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) &&
            node.attention_use_ragged_offsets) {
            validateNodeIndex(node.attention_ragged_offset_q_node, "attention q ragged offset");
            validateNodeIndex(node.attention_ragged_offset_kv_node, "attention kv ragged offset");
            if (node.attention_ragged_offset_q_node >= node_index_u32 || node.attention_ragged_offset_kv_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION ragged offset nodes must reference earlier nodes.");
            }
        }
        if ((node.op == ExprOp::ATTENTION || node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
             node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) &&
            node.attention_use_paged_kv_cache) {
            validateNodeIndex(node.attention_page_table_k_node, "attention page_table_k");
            validateNodeIndex(node.attention_page_table_v_node, "attention page_table_v");
            if (node.attention_page_table_k_node >= node_index_u32 || node.attention_page_table_v_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION paged KV page-table nodes must reference earlier nodes.");
            }
        }
        if ((node.op == ExprOp::ATTENTION || node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
             node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) &&
            node.attention_dropout_probability > 0.0f) {
            validateNodeIndex(node.attention_dropout_seed_node, "attention dropout seed");
            validateNodeIndex(node.attention_dropout_offset_node, "attention dropout offset");
            if (node.attention_dropout_seed_node >= node_index_u32 || node.attention_dropout_offset_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION dropout seed/offset nodes must reference earlier nodes.");
            }
        }
        if (node.op == ExprOp::ATTENTION && node.attention_use_fp8_forward_scaling) {
            validateNodeIndex(node.attention_descale_q_node, "attention descale_q");
            validateNodeIndex(node.attention_descale_k_node, "attention descale_k");
            validateNodeIndex(node.attention_descale_v_node, "attention descale_v");
            validateNodeIndex(node.attention_descale_s_node, "attention descale_s");
            validateNodeIndex(node.attention_scale_s_node, "attention scale_s");
            validateNodeIndex(node.attention_scale_o_node, "attention scale_o");
            validateNodeIndex(node.attention_amax_s_node, "attention amax_s");
            validateNodeIndex(node.attention_amax_o_node, "attention amax_o");
            if (node.attention_descale_q_node >= node_index_u32 || node.attention_descale_k_node >= node_index_u32 ||
                node.attention_descale_v_node >= node_index_u32 || node.attention_descale_s_node >= node_index_u32 ||
                node.attention_scale_s_node >= node_index_u32 || node.attention_scale_o_node >= node_index_u32 ||
                node.attention_amax_s_node >= node_index_u32 || node.attention_amax_o_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION FP8 scale/descale/amax nodes must reference earlier nodes.");
            }
        }
        if ((node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
             node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) &&
            node.attention_use_bias) {
            validateNodeIndex(node.beta_node, "attention backward bias");
            if (node.beta_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION_BACKWARD bias node must reference an earlier node.");
            }
        }
        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            if (node.cuda_kernel_spec_index >= outputs.expr->cuda_kernel_expressions.size() ||
                !outputs.expr->cuda_kernel_expressions[node.cuda_kernel_spec_index]) {
                throw std::runtime_error("ExpressionDefinition CudaKernelExpression node references an invalid kernel spec.");
            }
            const auto& kernel = outputs.expr->cuda_kernel_expressions[node.cuda_kernel_spec_index];
            if (node.cuda_kernel_output_index >= kernel->outputs().size()) {
                throw std::runtime_error("ExpressionDefinition CudaKernelExpression node references an invalid output spec.");
            }
            const auto& output_spec = kernel->outputs()[node.cuda_kernel_output_index];
            if (!node.output_dtype.has_value()) {
                throw std::runtime_error("ExpressionDefinition CudaKernelExpression output node is missing output dtype.");
            }
            if (node.output_dtype.value() != output_spec.dtype) {
                throw std::runtime_error("ExpressionDefinition CudaKernelExpression output node dtype does not match the kernel output spec.");
            }
            if (node.cuda_kernel_input_nodes.size() != kernel->inputs().size()) {
                throw std::runtime_error("ExpressionDefinition CudaKernelExpression input node count does not match the kernel input spec.");
            }
            for (size_t input_idx = 0; input_idx < node.cuda_kernel_input_nodes.size(); ++input_idx) {
                const uint32_t input_node = node.cuda_kernel_input_nodes[input_idx];
                validateNodeIndex(input_node, "cuda kernel input");
                if (input_node >= node_index_u32) {
                    throw std::runtime_error("ExpressionDefinition CudaKernelExpression input nodes must reference earlier nodes.");
                }

                const ExprNode& input_expr_node = outputs.expr->nodes[input_node];
                const auto& input_spec = kernel->inputs()[input_idx];
                switch (input_spec.kind) {
                    case CudaKernelExpression::TensorParamSpec::Kind::Tensor:
                        if (input_expr_node.op == ExprOp::RUNTIME_SCALAR || input_expr_node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
                            throw std::runtime_error(
                                "ExpressionDefinition CudaKernelExpression tensor input is wired to a runtime scalar node.");
                        }
                        break;
                    case CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar:
                        if (input_expr_node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
                            throw std::runtime_error(
                                "ExpressionDefinition CudaKernelExpression tensor runtime scalar input is wired to the wrong node kind.");
                        }
                        break;
                    case CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar:
                        if (input_expr_node.op != ExprOp::RUNTIME_SCALAR) {
                            throw std::runtime_error(
                                "ExpressionDefinition CudaKernelExpression host runtime scalar input is wired to the wrong node kind.");
                        }
                        break;
                }
                if (input_expr_node.output_dtype.has_value() && input_expr_node.output_dtype.value() != input_spec.dtype) {
                    throw std::runtime_error("ExpressionDefinition CudaKernelExpression input node dtype does not match the kernel input spec.");
                }
            }
        }
    }

    std::unordered_set<std::string> seen_output_names;
    for (const NamedOutput& output : outputs.outputs) {
        if (output.name.empty()) {
            throw std::runtime_error("ExpressionDefinition output name cannot be empty.");
        }
        if (!seen_output_names.insert(output.name).second) {
            throw std::runtime_error("ExpressionDefinition has duplicate output name '" + output.name + "'.");
        }
        if (output.node_idx >= outputs.expr->nodes.size()) {
            throw std::runtime_error("ExpressionDefinition output node index out of range.");
        }
    }

    if (!expected_input_names.empty()) {
        std::unordered_set<std::string> expected(expected_input_names.begin(), expected_input_names.end());
        if (expected.size() != expected_input_names.size()) {
            throw std::runtime_error("ExpressionDefinition expected_input_names contains duplicates.");
        }
        if (expected != seen_input_names) {
            throw std::runtime_error("ExpressionDefinition expected_input_names do not match the serialized inputs.");
        }
    }

    if (!expected_output_names.empty()) {
        std::unordered_set<std::string> expected(expected_output_names.begin(), expected_output_names.end());
        if (expected.size() != expected_output_names.size()) {
            throw std::runtime_error("ExpressionDefinition expected_output_names contains duplicates.");
        }
        if (expected != seen_output_names) {
            throw std::runtime_error("ExpressionDefinition expected_output_names do not match the serialized outputs.");
        }
    }
}

json ExpressionDefinition::architectureJson() const {
    validate();

    json j;
    j["type"] = "thor.expression";
    j["schema_version"] = 1;
    j["inputs"] = json::array();
    j["nodes"] = json::array();
    j["outputs"] = json::array();
    j["expected_input_names"] = expected_input_names;
    j["expected_output_names"] = expected_output_names;
    if (!outputs.expr->cuda_kernel_expressions.empty()) {
        j["cuda_kernels"] = json::array();
        for (const auto& kernel : outputs.expr->cuda_kernel_expressions) {
            if (!kernel) {
                throw std::runtime_error("ExpressionDefinition::architectureJson encountered a null CudaKernelExpression spec.");
            }
            j["cuda_kernels"].push_back(kernel->architectureJson());
        }
    }

    for (const NamedInput& input : outputs.expr->inputs) {
        j["inputs"].push_back(json{{"slot", input.slot}, {"name", input.name}, {"kind", namedInputKindToString(input.kind)}});
    }
    for (const ExprNode& node : outputs.expr->nodes) {
        j["nodes"].push_back(exprNodeToJson(node));
    }
    for (const NamedOutput& output : outputs.outputs) {
        j["outputs"].push_back(json{{"name", output.name}, {"node", output.node_idx}});
    }

    const std::string computed_hash = expressionHash(outputs);
    j["canonical_hash"] = computed_hash;
    return j;
}

json ExpressionDefinition::architectureJsonWithCudaKernelManifestSignature() const {
    if (!hasCudaKernelExpressions()) {
        return architectureJson();
    }

    json signed_payload = architectureJson();
    const std::vector<CudaKernelOutOfBandKeys> out_of_band_keys = cudaKernelGenerateAndAttachManifestSignatures(signed_payload);
    if (out_of_band_keys.empty()) {
        throw std::runtime_error("ExpressionDefinition has CUDA kernels but CUDA source protection produced no out-of-band keys.");
    }

    auto sig_it = signed_payload.find("cuda_kernel_manifest_signature");
    if (sig_it == signed_payload.end() || !sig_it->is_object()) {
        throw std::runtime_error("ExpressionDefinition CUDA manifest signing did not attach a manifest signature.");
    }
    cuda_kernel_manifest_signature = *sig_it;
    return signed_payload;
}

bool ExpressionDefinition::hasCudaKernelExpressions() const { return outputs.expr && !outputs.expr->cuda_kernel_expressions.empty(); }

std::vector<std::string> ExpressionDefinition::cudaKernelSigningPublicKeys() const {
    if (!hasCudaKernelExpressions()) {
        return {};
    }
    const json signed_payload = architectureJsonWithCudaKernelManifestSignature();
    return collectCudaKernelSigningPublicKeys(signed_payload);
}

std::vector<CudaKernelOutOfBandKeys> ExpressionDefinition::cudaKernelOutOfBandKeys() const {
    if (!hasCudaKernelExpressions()) {
        return {};
    }
    const json signed_payload = architectureJsonWithCudaKernelManifestSignature();
    return collectCudaKernelOutOfBandKeys(signed_payload);
}

std::vector<CudaKernelSourceInspection> ExpressionDefinition::cudaKernelSourceInfo() const {
    validate();
    std::vector<CudaKernelSourceInspection> result;
    if (!outputs.expr) {
        return result;
    }
    result.reserve(outputs.expr->cuda_kernel_expressions.size());
    for (const auto& kernel : outputs.expr->cuda_kernel_expressions) {
        if (!kernel) {
            throw std::runtime_error("ExpressionDefinition has a null CudaKernelExpression spec.");
        }
        const auto info = kernel->sourceInfo();
        CudaKernelSourceInspection entry;
        entry.name = info.name;
        entry.entrypoint = info.entrypoint;
        entry.source = info.source;
        entry.compiled_source = info.compiled_source;
        entry.compiled_source_hash = info.source_hash;
        entry.loaded_source_compilation_allowed = info.loaded_source_compilation_allowed;
        if (!cuda_kernel_manifest_signature.is_null()) {
            entry.signature_algorithm = cuda_kernel_manifest_signature.value("algorithm", std::string{});
            entry.signing_public_key_fingerprint = cuda_kernel_manifest_signature.value("public_key_fingerprint", std::string{});
            entry.signature = cuda_kernel_manifest_signature.value("signature", std::string{});
        }
        result.push_back(std::move(entry));
    }
    return result;
}

std::vector<std::string> ExpressionDefinition::cudaKernelSources() const {
    std::vector<std::string> sources;
    for (const CudaKernelSourceInspection& info : cudaKernelSourceInfo()) {
        sources.push_back(info.source);
    }
    return sources;
}

json ExpressionDefinition::cudaKernelSourceInfoJson() const { return cudaKernelSourceInspectionListToJson(cudaKernelSourceInfo()); }

void ExpressionDefinition::allowUnsafeLoadedCudaKernelSourceCompilation(const std::string& trusted_ed25519_public_key,
                                                                           const std::string& trusted_source_decryption_key) {
    if (!outputs.expr || outputs.expr->cuda_kernel_expressions.empty()) {
        return;
    }
    json j = architectureJsonWithCudaKernelManifestSignature();
    *this = deserialize(j, true, trusted_ed25519_public_key, trusted_source_decryption_key);
}

ExpressionDefinition ExpressionDefinition::fromOutputs(const Outputs& outputs) {
    ExpressionDefinition definition;
    definition.outputs = outputs.physicalOutputs();

    for (const NamedInput& input : definition.outputs.expr->inputs) {
        definition.expected_input_names.push_back(input.name);
    }
    for (const NamedOutput& output : definition.outputs.outputs) {
        definition.expected_output_names.push_back(output.name);
    }
    definition.canonical_hash = expressionHash(definition.outputs);
    definition.validate();
    return definition;
}

ExpressionDefinition ExpressionDefinition::deserialize(const json& j,
                                                       bool allow_unsafe_loaded_cuda_source,
                                                       const std::string& trusted_ed25519_public_key,
                                                       const std::string& trusted_source_decryption_key) {
    if (j.at("type").get<std::string>() != "thor.expression") {
        throw std::runtime_error("ExpressionDefinition::deserialize type mismatch: " + j.at("type").get<std::string>());
    }
    const int schema_version = j.at("schema_version").get<int>();
    if (schema_version != 1) {
        throw std::runtime_error("Unsupported thor.expression schema_version: " + std::to_string(schema_version));
    }

    json expression_json = j;
    const bool has_cuda_kernels = j.contains("cuda_kernels") && !j.at("cuda_kernels").empty();
    const bool has_plaintext_cuda_sources = has_cuda_kernels && cudaKernelExpressionJsonContainsPlaintextSources(j);
    const bool has_encrypted_cuda_sources = has_cuda_kernels && cudaKernelExpressionJsonContainsEncryptedSources(j);
    if (has_plaintext_cuda_sources) {
        throw std::runtime_error(
            "Serialized CudaKernelExpression CUDA source must be encrypted. Refusing to load a saved model that contains plaintext CUDA "
            "source, even if a signature or encrypted_source field is also present.");
    }
    if (has_cuda_kernels && !has_encrypted_cuda_sources) {
        throw std::runtime_error(
            "Serialized CudaKernelExpression kernels must contain encrypted_source/source_encryption metadata. Refusing plaintext or "
            "unprotected CUDA source from a saved model.");
    }
    if (has_encrypted_cuda_sources) {
        CudaKernelSignatureVerificationResult verification = cudaKernelVerifyManifestSignature(j, trusted_ed25519_public_key);
        if (!verification.verified) {
            throw std::runtime_error(verification.message);
        }
        expression_json = cudaKernelDecryptSerializedCudaSources(j, trusted_source_decryption_key);
    }

    ExpressionDefinition definition;
    definition.outputs.expr = std::make_shared<PhysicalExpression>();
    definition.expected_input_names = expression_json.value("expected_input_names", std::vector<std::string>{});
    definition.expected_output_names = expression_json.value("expected_output_names", std::vector<std::string>{});
    definition.canonical_hash = expression_json.value("canonical_hash", std::string{});
    if (j.contains("cuda_kernel_manifest_signature")) {
        definition.cuda_kernel_manifest_signature = j.at("cuda_kernel_manifest_signature");
    }

    for (const json& input_json : expression_json.at("inputs")) {
        definition.outputs.expr->inputs.push_back(NamedInput{
            .name = input_json.at("name").get<std::string>(),
            .slot = input_json.at("slot").get<uint32_t>(),
            .kind = namedInputKindFromString(input_json.at("kind").get<std::string>()),
        });
    }

    bool cuda_kernel_source_compilation_allowed = false;
    if (has_cuda_kernels) {
        if (allow_unsafe_loaded_cuda_source) {
            cuda_kernel_source_compilation_allowed = true;
        }

        for (const json& kernel_json : expression_json.at("cuda_kernels")) {
            definition.outputs.expr->cuda_kernel_expressions.push_back(std::make_shared<CudaKernelExpression>(
                CudaKernelExpression::deserialize(kernel_json, cuda_kernel_source_compilation_allowed)));
        }
    }

    for (const json& node_json : expression_json.at("nodes")) {
        definition.outputs.expr->nodes.push_back(exprNodeFromJson(node_json));
    }

    for (const json& output_json : expression_json.at("outputs")) {
        definition.outputs.outputs.push_back(NamedOutput{
            .name = output_json.at("name").get<std::string>(),
            .node_idx = output_json.at("node").get<uint32_t>(),
        });
    }

    definition.validate();

    const std::string computed_hash = expressionHash(definition.outputs);
    if (!definition.canonical_hash.empty() && definition.canonical_hash != computed_hash) {
        throw std::runtime_error("ExpressionDefinition canonical_hash mismatch. Expected '" + definition.canonical_hash +
                                 "' but computed '" + computed_hash + "'.");
    }
    definition.canonical_hash = computed_hash;
    return definition;
}

bool Expression::isLeafOp(const ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL:
            return true;
        default:
            return false;
    }
}

bool Expression::isUnaryOp(const ExprOp op) {
    switch (op) {
        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::CEIL:
        case ExprOp::FLOOR:
        case ExprOp::ROUND:
        case ExprOp::TRUNC:
        case ExprOp::SIN:
        case ExprOp::COS:
        case ExprOp::TAN:
        case ExprOp::ASIN:
        case ExprOp::ACOS:
        case ExprOp::ATAN:
        case ExprOp::SINH:
        case ExprOp::COSH:
        case ExprOp::ASINH:
        case ExprOp::ACOSH:
        case ExprOp::ATANH:
        case ExprOp::ERF:
        case ExprOp::ERFC:
        case ExprOp::ERFCX:
        case ExprOp::ERFINV:
        case ExprOp::ERFCINV:
        case ExprOp::TGAMMA:
        case ExprOp::LGAMMA:
        case ExprOp::DIGAMMA:
        case ExprOp::EXP:
        case ExprOp::EXPM1:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG1P:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::TANH:
        case ExprOp::NORMCDF:
        case ExprOp::LOGICAL_NOT:
        case ExprOp::ROPE:
        case ExprOp::SOFTMAX:
        case ExprOp::TRANSPOSE:
        case ExprOp::RESHAPE:
        case ExprOp::STRIDED_VIEW:
        case ExprOp::STRIDED_VIEW_BACKWARD:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2:
            return true;
        default:
            return false;
    }
}

bool Expression::isBinaryOp(const ExprOp op) {
    switch (op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::EQUAL:
        case ExprOp::NOT_EQUAL:
        case ExprOp::LESS:
        case ExprOp::LESS_EQUAL:
        case ExprOp::GREATER:
        case ExprOp::GREATER_EQUAL:
        case ExprOp::LOGICAL_AND:
        case ExprOp::LOGICAL_OR:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::MATMUL:
        case ExprOp::RMSNORM:
        case ExprOp::EMBEDDING_LOOKUP:
        case ExprOp::CONV2D:
        case ExprOp::CONV2D_BACKWARD_DATA:
        case ExprOp::CONV2D_BACKWARD_FILTER:
        case ExprOp::CONV3D:
        case ExprOp::CONV3D_BACKWARD_DATA:
        case ExprOp::CONV3D_BACKWARD_FILTER:
        case ExprOp::REDUCE_MIN_BACKWARD:
        case ExprOp::REDUCE_MAX_BACKWARD:
            return true;
        default:
            return false;
    }
}

bool Expression::isTernaryOp(const ExprOp op) {
    switch (op) {
        case ExprOp::GEMM:
        case ExprOp::ATTENTION:
        case ExprOp::ATTENTION_BACKWARD_Q:
        case ExprOp::ATTENTION_BACKWARD_K:
        case ExprOp::ATTENTION_BACKWARD_V:
        case ExprOp::ATTENTION_BACKWARD_BIAS:
        case ExprOp::WHERE:
            return true;
        default:
            return false;
    }
}

namespace {

uint32_t remapCudaKernelSpecForClone(const PhysicalExpression& src,
                                     uint32_t src_spec_idx,
                                     PhysicalExpression& dst,
                                     std::unordered_map<uint32_t, uint32_t>& cuda_spec_remap) {
    if (src_spec_idx >= src.cuda_kernel_expressions.size() || !src.cuda_kernel_expressions[src_spec_idx]) {
        throw std::runtime_error("CudaKernelExpression clone references missing kernel spec.");
    }
    auto it = cuda_spec_remap.find(src_spec_idx);
    if (it != cuda_spec_remap.end()) {
        return it->second;
    }
    const uint32_t dst_spec_idx = static_cast<uint32_t>(dst.cuda_kernel_expressions.size());
    dst.cuda_kernel_expressions.push_back(src.cuda_kernel_expressions[src_spec_idx]);
    cuda_spec_remap.emplace(src_spec_idx, dst_spec_idx);
    return dst_spec_idx;
}

uint32_t cloneSubtreeImpl(const PhysicalExpression& src,
                          uint32_t srcNodeIndex,
                          PhysicalExpression& dst,
                          std::unordered_map<uint32_t, uint32_t>& oldToNew,
                          std::unordered_map<uint32_t, uint32_t>& cudaSpecRemap) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end())
        return it->second;

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    ExprNode newNode = srcNode;

    if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op");
        newNode.lhs = cloneSubtreeImpl(src, srcNode.lhs, dst, oldToNew, cudaSpecRemap);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for binary op");
        if (srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing rhs for binary op");
        newNode.lhs = cloneSubtreeImpl(src, srcNode.lhs, dst, oldToNew, cudaSpecRemap);
        newNode.rhs = cloneSubtreeImpl(src, srcNode.rhs, dst, oldToNew, cudaSpecRemap);
        newNode.aux = UINT32_MAX;
        if (srcNode.matmul_epilogue_aux != UINT32_MAX) {
            newNode.matmul_epilogue_aux = cloneSubtreeImpl(src, srcNode.matmul_epilogue_aux, dst, oldToNew, cudaSpecRemap);
        }
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for ternary op");
        newNode.lhs = cloneSubtreeImpl(src, srcNode.lhs, dst, oldToNew, cudaSpecRemap);
        newNode.rhs = cloneSubtreeImpl(src, srcNode.rhs, dst, oldToNew, cudaSpecRemap);
        newNode.aux = cloneSubtreeImpl(src, srcNode.aux, dst, oldToNew, cudaSpecRemap);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node = cloneSubtreeImpl(src, srcNode.alpha_node, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneSubtreeImpl(src, srcNode.beta_node, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.matmul_epilogue_aux != UINT32_MAX) {
            newNode.matmul_epilogue_aux = cloneSubtreeImpl(src, srcNode.matmul_epilogue_aux, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.attention_use_padding_mask) {
            if (srcNode.attention_seq_len_q_node == UINT32_MAX || srcNode.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing padding-mask sequence length node while cloning.");
            }
            newNode.attention_seq_len_q_node = cloneSubtreeImpl(src, srcNode.attention_seq_len_q_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_seq_len_kv_node = cloneSubtreeImpl(src, srcNode.attention_seq_len_kv_node, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.attention_use_ragged_offsets) {
            if (srcNode.attention_ragged_offset_q_node == UINT32_MAX || srcNode.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing ragged offset node while cloning.");
            }
            newNode.attention_ragged_offset_q_node =
                cloneSubtreeImpl(src, srcNode.attention_ragged_offset_q_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_ragged_offset_kv_node =
                cloneSubtreeImpl(src, srcNode.attention_ragged_offset_kv_node, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.attention_use_paged_kv_cache) {
            if (srcNode.attention_page_table_k_node == UINT32_MAX || srcNode.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing paged KV page-table nodes while cloning.");
            }
            newNode.attention_page_table_k_node = cloneSubtreeImpl(src, srcNode.attention_page_table_k_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_page_table_v_node = cloneSubtreeImpl(src, srcNode.attention_page_table_v_node, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.attention_dropout_probability > 0.0f) {
            if (srcNode.attention_dropout_seed_node == UINT32_MAX || srcNode.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing dropout seed/offset node while cloning.");
            }
            newNode.attention_dropout_seed_node = cloneSubtreeImpl(src, srcNode.attention_dropout_seed_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_dropout_offset_node =
                cloneSubtreeImpl(src, srcNode.attention_dropout_offset_node, dst, oldToNew, cudaSpecRemap);
        }
        if (srcNode.attention_use_fp8_forward_scaling) {
            if (srcNode.attention_descale_q_node == UINT32_MAX || srcNode.attention_descale_k_node == UINT32_MAX ||
                srcNode.attention_descale_v_node == UINT32_MAX || srcNode.attention_descale_s_node == UINT32_MAX ||
                srcNode.attention_scale_s_node == UINT32_MAX || srcNode.attention_scale_o_node == UINT32_MAX ||
                srcNode.attention_amax_s_node == UINT32_MAX || srcNode.attention_amax_o_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing FP8 scale/descale/amax node while cloning.");
            }
            newNode.attention_descale_q_node = cloneSubtreeImpl(src, srcNode.attention_descale_q_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_descale_k_node = cloneSubtreeImpl(src, srcNode.attention_descale_k_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_descale_v_node = cloneSubtreeImpl(src, srcNode.attention_descale_v_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_descale_s_node = cloneSubtreeImpl(src, srcNode.attention_descale_s_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_scale_s_node = cloneSubtreeImpl(src, srcNode.attention_scale_s_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_scale_o_node = cloneSubtreeImpl(src, srcNode.attention_scale_o_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_amax_s_node = cloneSubtreeImpl(src, srcNode.attention_amax_s_node, dst, oldToNew, cudaSpecRemap);
            newNode.attention_amax_o_node = cloneSubtreeImpl(src, srcNode.attention_amax_o_node, dst, oldToNew, cudaSpecRemap);
        }
    } else if (srcNode.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        newNode.cuda_kernel_spec_index = remapCudaKernelSpecForClone(src, srcNode.cuda_kernel_spec_index, dst, cudaSpecRemap);
        newNode.cuda_kernel_input_nodes.clear();
        newNode.cuda_kernel_input_nodes.reserve(srcNode.cuda_kernel_input_nodes.size());
        for (uint32_t input_node : srcNode.cuda_kernel_input_nodes) {
            newNode.cuda_kernel_input_nodes.push_back(cloneSubtreeImpl(src, input_node, dst, oldToNew, cudaSpecRemap));
        }
        newNode.lhs = UINT32_MAX;
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isLeafOp(srcNode.op)) {
        // nothing to recurse into
    } else {
        std::string error_message = "Malformed expression: unsupported op in cloneSubtree: " + std::to_string(static_cast<int>(srcNode.op));
        throw std::runtime_error(error_message.c_str());
    }

    uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(newNode);
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}

uint32_t cloneSubtree(const PhysicalExpression& src,
                      uint32_t srcNodeIndex,
                      PhysicalExpression& dst,
                      std::unordered_map<uint32_t, uint32_t>& oldToNew) {
    std::unordered_map<uint32_t, uint32_t> cudaSpecRemap;
    return cloneSubtreeImpl(src, srcNodeIndex, dst, oldToNew, cudaSpecRemap);
}

uint32_t cloneSubtreeWithMergedInputsImpl(const PhysicalExpression& src,
                                          uint32_t srcNodeIndex,
                                          PhysicalExpression& dst,
                                          std::unordered_map<uint32_t, uint32_t>& oldToNew,
                                          std::unordered_map<std::string, uint32_t>& dstInputSlotsByName,
                                          std::unordered_map<uint32_t, uint32_t>& cudaSpecRemap) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end())
        return it->second;

    const ExprNode& srcNode = src.nodes.at(srcNodeIndex);
    ExprNode newNode = srcNode;

    if (srcNode.op == ExprOp::INPUT || srcNode.op == ExprOp::RUNTIME_SCALAR || srcNode.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        if (srcNode.input_slot >= src.inputs.size()) {
            throw std::runtime_error("Input slot out of range while merging expression outputs.");
        }

        const std::string& inputName = src.inputs[srcNode.input_slot].name;
        const NamedInput::Kind inputKind = src.inputs[srcNode.input_slot].kind;
        auto slotIt = dstInputSlotsByName.find(inputName);
        uint32_t mergedSlot;

        if (slotIt != dstInputSlotsByName.end()) {
            mergedSlot = slotIt->second;
            if (mergedSlot >= dst.inputs.size()) {
                throw std::runtime_error("Merged input slot out of range while merging expression outputs.");
            }
            if (dst.inputs[mergedSlot].kind != inputKind) {
                throw std::runtime_error("Input kind mismatch while merging expression outputs for input: " + inputName);
            }
        } else {
            mergedSlot = static_cast<uint32_t>(dst.inputs.size());
            dst.inputs.push_back(NamedInput{inputName, mergedSlot, inputKind});
            dstInputSlotsByName.emplace(inputName, mergedSlot);
        }

        newNode.input_slot = mergedSlot;
        newNode.lhs = UINT32_MAX;
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputsImpl(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for binary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputsImpl(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        newNode.rhs = cloneSubtreeWithMergedInputsImpl(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        newNode.aux = UINT32_MAX;
        if (srcNode.matmul_epilogue_aux != UINT32_MAX) {
            newNode.matmul_epilogue_aux =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.matmul_epilogue_aux, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for ternary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputsImpl(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        newNode.rhs = cloneSubtreeWithMergedInputsImpl(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        newNode.aux = cloneSubtreeWithMergedInputsImpl(src, srcNode.aux, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.alpha_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneSubtreeWithMergedInputsImpl(src, srcNode.beta_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.matmul_epilogue_aux != UINT32_MAX) {
            newNode.matmul_epilogue_aux =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.matmul_epilogue_aux, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.attention_use_padding_mask) {
            if (srcNode.attention_seq_len_q_node == UINT32_MAX || srcNode.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing padding-mask sequence length node while merging outputs.");
            }
            newNode.attention_seq_len_q_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_seq_len_q_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_seq_len_kv_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_seq_len_kv_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.attention_use_ragged_offsets) {
            if (srcNode.attention_ragged_offset_q_node == UINT32_MAX || srcNode.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing ragged offset node while merging outputs.");
            }
            newNode.attention_ragged_offset_q_node = cloneSubtreeWithMergedInputsImpl(
                src, srcNode.attention_ragged_offset_q_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_ragged_offset_kv_node = cloneSubtreeWithMergedInputsImpl(
                src, srcNode.attention_ragged_offset_kv_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.attention_use_paged_kv_cache) {
            if (srcNode.attention_page_table_k_node == UINT32_MAX || srcNode.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing paged KV page-table nodes while cloning merged inputs.");
            }
            newNode.attention_page_table_k_node = cloneSubtreeWithMergedInputsImpl(
                src, srcNode.attention_page_table_k_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_page_table_v_node = cloneSubtreeWithMergedInputsImpl(
                src, srcNode.attention_page_table_v_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.attention_dropout_probability > 0.0f) {
            if (srcNode.attention_dropout_seed_node == UINT32_MAX || srcNode.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing dropout seed/offset node while merging outputs.");
            }
            newNode.attention_dropout_seed_node = cloneSubtreeWithMergedInputsImpl(
                src, srcNode.attention_dropout_seed_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_dropout_offset_node = cloneSubtreeWithMergedInputsImpl(
                src, srcNode.attention_dropout_offset_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
        if (srcNode.attention_use_fp8_forward_scaling) {
            if (srcNode.attention_descale_q_node == UINT32_MAX || srcNode.attention_descale_k_node == UINT32_MAX ||
                srcNode.attention_descale_v_node == UINT32_MAX || srcNode.attention_descale_s_node == UINT32_MAX ||
                srcNode.attention_scale_s_node == UINT32_MAX || srcNode.attention_scale_o_node == UINT32_MAX ||
                srcNode.attention_amax_s_node == UINT32_MAX || srcNode.attention_amax_o_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing FP8 scale/descale/amax node while merging outputs.");
            }
            newNode.attention_descale_q_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_descale_q_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_descale_k_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_descale_k_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_descale_v_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_descale_v_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_descale_s_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_descale_s_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_scale_s_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_scale_s_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_scale_o_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_scale_o_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_amax_s_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_amax_s_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
            newNode.attention_amax_o_node =
                cloneSubtreeWithMergedInputsImpl(src, srcNode.attention_amax_o_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
        }
    } else if (srcNode.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        newNode.cuda_kernel_spec_index = remapCudaKernelSpecForClone(src, srcNode.cuda_kernel_spec_index, dst, cudaSpecRemap);
        newNode.cuda_kernel_input_nodes.clear();
        newNode.cuda_kernel_input_nodes.reserve(srcNode.cuda_kernel_input_nodes.size());
        for (uint32_t input_node : srcNode.cuda_kernel_input_nodes) {
            newNode.cuda_kernel_input_nodes.push_back(
                cloneSubtreeWithMergedInputsImpl(src, input_node, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap));
        }
        newNode.lhs = UINT32_MAX;
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (srcNode.op == ExprOp::SCALAR_FP || srcNode.op == ExprOp::FILL) {
        // nothing to recurse into
    } else {
        throw std::runtime_error("Unsupported op while merging expression outputs: " + std::to_string(static_cast<int>(srcNode.op)));
    }

    uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(newNode));
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}

uint32_t cloneSubtreeWithMergedInputs(const PhysicalExpression& src,
                                      uint32_t srcNodeIndex,
                                      PhysicalExpression& dst,
                                      std::unordered_map<uint32_t, uint32_t>& oldToNew,
                                      std::unordered_map<std::string, uint32_t>& dstInputSlotsByName) {
    std::unordered_map<uint32_t, uint32_t> cudaSpecRemap;
    return cloneSubtreeWithMergedInputsImpl(src, srcNodeIndex, dst, oldToNew, dstInputSlotsByName, cudaSpecRemap);
}

uint32_t cloneSubtreeWithInputSubstitution(const PhysicalExpression& src,
                                           uint32_t srcNodeIndex,
                                           const std::string& substituteInputName,
                                           uint32_t substituteNodeIndex,
                                           PhysicalExpression& dst,
                                           std::unordered_map<uint32_t, uint32_t>& oldToNew) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end()) {
        return it->second;
    }
    if (srcNodeIndex >= src.nodes.size()) {
        throw std::runtime_error("Expression substitution source node index is out of range.");
    }

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    ExprNode newNode = srcNode;

    if (srcNode.op == ExprOp::INPUT || srcNode.op == ExprOp::RUNTIME_SCALAR || srcNode.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        if (srcNode.input_slot >= src.inputs.size()) {
            throw std::runtime_error("Expression substitution input slot is out of range.");
        }

        const NamedInput& input = src.inputs[srcNode.input_slot];
        if (input.name == substituteInputName) {
            oldToNew[srcNodeIndex] = substituteNodeIndex;
            return substituteNodeIndex;
        }

        const uint32_t dstSlot = dst.getOrCreateInputSlot(input.name, input.kind);
        newNode.input_slot = dstSlot;
        newNode.lhs = UINT32_MAX;
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX) {
            throw std::runtime_error("Malformed expression: missing lhs for unary op while substituting input.");
        }
        newNode.lhs = cloneSubtreeWithInputSubstitution(src, srcNode.lhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX) {
            throw std::runtime_error("Malformed expression: missing child for binary op while substituting input.");
        }
        newNode.lhs = cloneSubtreeWithInputSubstitution(src, srcNode.lhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.rhs = cloneSubtreeWithInputSubstitution(src, srcNode.rhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.aux = UINT32_MAX;
        if (srcNode.matmul_epilogue_aux != UINT32_MAX) {
            newNode.matmul_epilogue_aux = cloneSubtreeWithInputSubstitution(
                src, srcNode.matmul_epilogue_aux, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX) {
            throw std::runtime_error("Malformed expression: missing child for ternary op while substituting input.");
        }
        newNode.lhs = cloneSubtreeWithInputSubstitution(src, srcNode.lhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.rhs = cloneSubtreeWithInputSubstitution(src, srcNode.rhs, substituteInputName, substituteNodeIndex, dst, oldToNew);
        newNode.aux = cloneSubtreeWithInputSubstitution(src, srcNode.aux, substituteInputName, substituteNodeIndex, dst, oldToNew);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node =
                cloneSubtreeWithInputSubstitution(src, srcNode.alpha_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node =
                cloneSubtreeWithInputSubstitution(src, srcNode.beta_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.matmul_epilogue_aux != UINT32_MAX) {
            newNode.matmul_epilogue_aux = cloneSubtreeWithInputSubstitution(
                src, srcNode.matmul_epilogue_aux, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.attention_use_padding_mask) {
            if (srcNode.attention_seq_len_q_node == UINT32_MAX || srcNode.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing padding-mask sequence length node while substituting input.");
            }
            newNode.attention_seq_len_q_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_seq_len_q_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_seq_len_kv_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_seq_len_kv_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.attention_dropout_probability > 0.0f) {
            if (srcNode.attention_dropout_seed_node == UINT32_MAX || srcNode.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing dropout seed/offset node while substituting input.");
            }
            newNode.attention_dropout_seed_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_dropout_seed_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_dropout_offset_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_dropout_offset_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.attention_use_ragged_offsets) {
            if (srcNode.attention_ragged_offset_q_node == UINT32_MAX || srcNode.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing ragged offset node while substituting input.");
            }
            newNode.attention_ragged_offset_q_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_ragged_offset_q_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_ragged_offset_kv_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_ragged_offset_kv_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.attention_use_paged_kv_cache) {
            if (srcNode.attention_page_table_k_node == UINT32_MAX || srcNode.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing paged KV page-table nodes while substituting input.");
            }
            newNode.attention_page_table_k_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_page_table_k_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_page_table_v_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_page_table_v_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
        if (srcNode.attention_use_fp8_forward_scaling) {
            if (srcNode.attention_descale_q_node == UINT32_MAX || srcNode.attention_descale_k_node == UINT32_MAX ||
                srcNode.attention_descale_v_node == UINT32_MAX || srcNode.attention_descale_s_node == UINT32_MAX ||
                srcNode.attention_scale_s_node == UINT32_MAX || srcNode.attention_scale_o_node == UINT32_MAX ||
                srcNode.attention_amax_s_node == UINT32_MAX || srcNode.attention_amax_o_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing FP8 scale/descale/amax node while substituting input.");
            }
            newNode.attention_descale_q_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_descale_q_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_descale_k_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_descale_k_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_descale_v_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_descale_v_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_descale_s_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_descale_s_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_scale_s_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_scale_s_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_scale_o_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_scale_o_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_amax_s_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_amax_s_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
            newNode.attention_amax_o_node = cloneSubtreeWithInputSubstitution(
                src, srcNode.attention_amax_o_node, substituteInputName, substituteNodeIndex, dst, oldToNew);
        }
    } else if (Expression::isLeafOp(srcNode.op)) {
        // constants/fill nodes have no children and can be copied directly.
    } else {
        throw std::runtime_error("Unsupported op while substituting expression input: " + std::to_string(static_cast<int>(srcNode.op)));
    }

    const uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(newNode));
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}

}  // namespace

Expression Expression::input(const std::string& name, std::optional<DataType> compute_dtype, std::optional<DataType> output_dtype) {
    validateUserInputName(name);
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::INPUT;
    node.input_slot = out->getOrCreateInputSlot(name, NamedInput::Kind::Tensor);

    // output_dtype means the graph value produced by this input defaults to that dtype,
    // even though the actual bound runtime tensor may have a different dtype.
    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }

    out->nodes.push_back(node);
    out->output_node = 0;

    return Expression(out, 0);
}

Expression Expression::runtimeScalar(const std::string& name, std::optional<DataType> compute_dtype, std::optional<DataType> output_dtype) {
    validateUserInputName(name);
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::RUNTIME_SCALAR;
    node.input_slot = out->getOrCreateInputSlot(name, NamedInput::Kind::RuntimeScalarFp32);

    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }

    out->nodes.push_back(node);
    out->output_node = 0;

    return Expression(out, 0);
}

Expression Expression::tensorRuntimeScalar(const std::string& name,
                                           std::optional<DataType> compute_dtype,
                                           std::optional<DataType> output_dtype) {
    validateUserInputName(name);
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::TENSOR_RUNTIME_SCALAR;
    node.input_slot = out->getOrCreateInputSlot(name, NamedInput::Kind::TensorRuntimeScalar);

    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }

    out->nodes.push_back(node);
    out->output_node = 0;

    return Expression(out, 0);
}

Expression::Expression(double value) {
    expr = std::make_shared<PhysicalExpression>();

    ExprNode node{};
    node.op = ExprOp::SCALAR_FP;
    node.scalar_fp = value;

    nodeIndex = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(node);
    expr->output_node = nodeIndex;
}

Expression Expression::constantScalar(double value) { return Expression(value); }
// Expression Expression::scalar(int64_t value) { return Expression(value); }

PhysicalExpression Expression::expression() const {
    if (!expr)
        throw std::runtime_error("Expr has no underlying expression");

    PhysicalExpression out = *expr;
    out.output_node = nodeIndex;
    return out;
}

Expression Expression::substituteInput(const std::string& input_name, const Expression& replacement) const {
    if (input_name.empty()) {
        throw std::invalid_argument("Expression::substituteInput requires a non-empty input name.");
    }
    if (!expr) {
        throw std::runtime_error("Expression::substituteInput called on an expression with no underlying graph.");
    }
    if (!replacement.expr) {
        throw std::runtime_error("Expression::substituteInput replacement has no underlying graph.");
    }
    if (replacement.nodeIndex >= replacement.expr->nodes.size()) {
        throw std::runtime_error("Expression::substituteInput replacement node index is out of range.");
    }
    if (nodeIndex >= expr->nodes.size()) {
        throw std::runtime_error("Expression::substituteInput source node index is out of range.");
    }

    auto composed = std::make_shared<PhysicalExpression>(*replacement.expr);
    composed->output_node = replacement.nodeIndex;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    const uint32_t composedRoot =
        cloneSubtreeWithInputSubstitution(*expr, nodeIndex, input_name, replacement.nodeIndex, *composed, oldToNew);
    composed->output_node = composedRoot;
    return Expression(std::move(composed), composedRoot);
}

struct MergeInputsResult {
    std::vector<NamedInput> mergedInputs;
    std::vector<uint32_t> lhsSlotRemap;
    std::vector<uint32_t> rhsSlotRemap;
};

static MergeInputsResult mergeInputsByName(const PhysicalExpression& lhs, const PhysicalExpression& rhs) {
    MergeInputsResult result;

    std::unordered_map<std::string, uint32_t> mergedByName;
    mergedByName.reserve(lhs.inputs.size() + rhs.inputs.size());

    auto getOrCreateMergedSlot = [&](const std::string& name, NamedInput::Kind kind) -> uint32_t {
        auto it = mergedByName.find(name);
        if (it != mergedByName.end()) {
            if (it->second >= result.mergedInputs.size()) {
                throw std::runtime_error("Merged input slot out of range while combining expressions.");
            }
            if (result.mergedInputs[it->second].kind != kind) {
                throw std::runtime_error("Input kind mismatch while combining expressions for input: " + name);
            }
            return it->second;
        }

        const uint32_t slot = static_cast<uint32_t>(result.mergedInputs.size());
        mergedByName.emplace(name, slot);
        result.mergedInputs.push_back(NamedInput{name, slot, kind});
        return slot;
    };

    result.lhsSlotRemap.resize(lhs.inputs.size());
    for (size_t i = 0; i < lhs.inputs.size(); ++i) {
        result.lhsSlotRemap[i] = getOrCreateMergedSlot(lhs.inputs[i].name, lhs.inputs[i].kind);
    }

    result.rhsSlotRemap.resize(rhs.inputs.size());
    for (size_t i = 0; i < rhs.inputs.size(); ++i) {
        result.rhsSlotRemap[i] = getOrCreateMergedSlot(rhs.inputs[i].name, rhs.inputs[i].kind);
    }

    return result;
}

static void remapClonedInputSlots(const PhysicalExpression& sourceExpr,
                                  const std::unordered_map<uint32_t, uint32_t>& oldToNewNodeMap,
                                  const std::vector<uint32_t>& slotRemap,
                                  PhysicalExpression& outExpr) {
    for (const auto& [oldNodeIndex, newNodeIndex] : oldToNewNodeMap) {
        const ExprNode& oldNode = sourceExpr.nodes.at(oldNodeIndex);
        if (oldNode.op != ExprOp::INPUT && oldNode.op != ExprOp::RUNTIME_SCALAR && oldNode.op != ExprOp::TENSOR_RUNTIME_SCALAR)
            continue;

        if (oldNode.input_slot >= slotRemap.size()) {
            throw std::runtime_error("Input slot out of range while remapping cloned expression.");
        }

        ExprNode& newNode = outExpr.nodes.at(newNodeIndex);
        newNode.input_slot = slotRemap[oldNode.input_slot];
    }
}

Expression Expression::binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op) {
    if (!lhsExpr.expr || !rhsExpr.expr)
        throw std::runtime_error("Cannot combine empty expressions");

    auto out = std::make_shared<PhysicalExpression>();

    const MergeInputsResult mergedInputs = mergeInputsByName(*lhsExpr.expr, *rhsExpr.expr);
    out->inputs = mergedInputs.mergedInputs;

    struct CloneState {
        std::unordered_map<uint32_t, uint32_t> node_map;
        std::unordered_map<uint32_t, uint32_t> cuda_spec_remap;
    };
    std::unordered_map<const PhysicalExpression*, CloneState> clone_states;
    auto clone_expr = [&](const Expression& expression) -> uint32_t {
        CloneState& state = clone_states[expression.expr.get()];
        return cloneSubtreeImpl(*expression.expr, expression.nodeIndex, *out, state.node_map, state.cuda_spec_remap);
    };

    uint32_t newLhsIndex = clone_expr(lhsExpr);
    uint32_t newRhsIndex = clone_expr(rhsExpr);

    remapClonedInputSlots(*lhsExpr.expr, clone_states.at(lhsExpr.expr.get()).node_map, mergedInputs.lhsSlotRemap, *out);
    remapClonedInputSlots(*rhsExpr.expr, clone_states.at(rhsExpr.expr.get()).node_map, mergedInputs.rhsSlotRemap, *out);

    ExprNode node{};
    node.op = op;
    node.lhs = newLhsIndex;
    node.rhs = newRhsIndex;

    uint32_t newIndex = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = newIndex;

    return Expression(out, newIndex);
}

Expression Expression::ternaryOp(const Expression& lhsExpr, const Expression& rhsExpr, const Expression& auxExpr, ExprOp op) {
    if (!lhsExpr.expr || !rhsExpr.expr || !auxExpr.expr)
        throw std::runtime_error("Cannot combine empty expressions");

    auto out = std::make_shared<PhysicalExpression>();

    const MergeInputsResult lhs_rhs_inputs = mergeInputsByName(*lhsExpr.expr, *rhsExpr.expr);

    std::unordered_map<std::string, uint32_t> mergedByName;
    mergedByName.reserve(lhs_rhs_inputs.mergedInputs.size());
    for (const NamedInput& input : lhs_rhs_inputs.mergedInputs) {
        mergedByName.emplace(input.name, input.slot);
    }

    // Reuse the final out expression directly so slot indices stay stable.
    out->inputs = lhs_rhs_inputs.mergedInputs;
    for (const NamedInput& input : auxExpr.expr->inputs) {
        auto it = mergedByName.find(input.name);
        if (it != mergedByName.end()) {
            if (out->inputs[it->second].kind != input.kind) {
                throw std::runtime_error("Input kind mismatch while combining expressions for input: " + input.name);
            }
        } else {
            const uint32_t slot = static_cast<uint32_t>(out->inputs.size());
            out->inputs.push_back(NamedInput{input.name, slot, input.kind});
            mergedByName.emplace(input.name, slot);
        }
    }

    struct CloneState {
        std::unordered_map<uint32_t, uint32_t> node_map;
        std::unordered_map<uint32_t, uint32_t> cuda_spec_remap;
    };
    std::unordered_map<const PhysicalExpression*, CloneState> clone_states;
    auto clone_expr = [&](const Expression& expression) -> uint32_t {
        CloneState& state = clone_states[expression.expr.get()];
        return cloneSubtreeImpl(*expression.expr, expression.nodeIndex, *out, state.node_map, state.cuda_spec_remap);
    };

    uint32_t newLhsIndex = clone_expr(lhsExpr);
    uint32_t newRhsIndex = clone_expr(rhsExpr);
    uint32_t newAuxIndex = clone_expr(auxExpr);

    std::vector<uint32_t> lhsSlotRemap(lhsExpr.expr->inputs.size());
    for (size_t i = 0; i < lhsExpr.expr->inputs.size(); ++i)
        lhsSlotRemap[i] = mergedByName.at(lhsExpr.expr->inputs[i].name);
    std::vector<uint32_t> rhsSlotRemap(rhsExpr.expr->inputs.size());
    for (size_t i = 0; i < rhsExpr.expr->inputs.size(); ++i)
        rhsSlotRemap[i] = mergedByName.at(rhsExpr.expr->inputs[i].name);
    std::vector<uint32_t> auxSlotRemap(auxExpr.expr->inputs.size());
    for (size_t i = 0; i < auxExpr.expr->inputs.size(); ++i)
        auxSlotRemap[i] = mergedByName.at(auxExpr.expr->inputs[i].name);

    remapClonedInputSlots(*lhsExpr.expr, clone_states.at(lhsExpr.expr.get()).node_map, lhsSlotRemap, *out);
    remapClonedInputSlots(*rhsExpr.expr, clone_states.at(rhsExpr.expr.get()).node_map, rhsSlotRemap, *out);
    remapClonedInputSlots(*auxExpr.expr, clone_states.at(auxExpr.expr.get()).node_map, auxSlotRemap, *out);

    ExprNode node{};
    node.op = op;
    node.lhs = newLhsIndex;
    node.rhs = newRhsIndex;
    node.aux = newAuxIndex;

    uint32_t newIndex = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = newIndex;

    return Expression(out, newIndex);
}

Expression Expression::quaternaryOp(
    const Expression& lhsExpr, const Expression& rhsExpr, const Expression& auxExpr, const Expression& fourthExpr, ExprOp op) {
    if (!lhsExpr.expr || !rhsExpr.expr || !auxExpr.expr || !fourthExpr.expr)
        throw std::runtime_error("Cannot combine empty expressions");

    auto out = std::make_shared<PhysicalExpression>();

    const MergeInputsResult lhs_rhs_inputs = mergeInputsByName(*lhsExpr.expr, *rhsExpr.expr);
    std::unordered_map<std::string, uint32_t> mergedByName;
    mergedByName.reserve(lhs_rhs_inputs.mergedInputs.size());
    for (const NamedInput& input : lhs_rhs_inputs.mergedInputs) {
        mergedByName.emplace(input.name, input.slot);
    }

    out->inputs = lhs_rhs_inputs.mergedInputs;
    auto merge_inputs_from = [&](const PhysicalExpression& src) {
        for (const NamedInput& input : src.inputs) {
            auto it = mergedByName.find(input.name);
            if (it != mergedByName.end()) {
                if (out->inputs[it->second].kind != input.kind) {
                    throw std::runtime_error("Input kind mismatch while combining expressions for input: " + input.name);
                }
            } else {
                const uint32_t slot = static_cast<uint32_t>(out->inputs.size());
                out->inputs.push_back(NamedInput{input.name, slot, input.kind});
                mergedByName.emplace(input.name, slot);
            }
        }
    };
    merge_inputs_from(*auxExpr.expr);
    merge_inputs_from(*fourthExpr.expr);

    struct CloneState {
        std::unordered_map<uint32_t, uint32_t> node_map;
        std::unordered_map<uint32_t, uint32_t> cuda_spec_remap;
    };
    std::unordered_map<const PhysicalExpression*, CloneState> clone_states;
    auto clone_expr = [&](const Expression& expression) -> uint32_t {
        CloneState& state = clone_states[expression.expr.get()];
        return cloneSubtreeImpl(*expression.expr, expression.nodeIndex, *out, state.node_map, state.cuda_spec_remap);
    };

    uint32_t newLhsIndex = clone_expr(lhsExpr);
    uint32_t newRhsIndex = clone_expr(rhsExpr);
    uint32_t newAuxIndex = clone_expr(auxExpr);
    uint32_t newFourthIndex = clone_expr(fourthExpr);

    auto remap_for = [&](const Expression& expression) {
        const PhysicalExpression& src = *expression.expr;
        std::vector<uint32_t> slotRemap(src.inputs.size());
        for (size_t i = 0; i < src.inputs.size(); ++i)
            slotRemap[i] = mergedByName.at(src.inputs[i].name);
        remapClonedInputSlots(src, clone_states.at(expression.expr.get()).node_map, slotRemap, *out);
    };
    remap_for(lhsExpr);
    remap_for(rhsExpr);
    remap_for(auxExpr);
    remap_for(fourthExpr);

    ExprNode node{};
    node.op = op;
    node.lhs = newLhsIndex;
    node.rhs = newRhsIndex;
    node.aux = newAuxIndex;
    node.alpha_node = newFourthIndex;

    uint32_t newIndex = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = newIndex;

    return Expression(out, newIndex);
}

namespace {

static bool isTransposePushThroughUnaryOp(ExprOp op) {
    switch (op) {
        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::CEIL:
        case ExprOp::FLOOR:
        case ExprOp::ROUND:
        case ExprOp::TRUNC:
        case ExprOp::SIN:
        case ExprOp::COS:
        case ExprOp::TAN:
        case ExprOp::ASIN:
        case ExprOp::ACOS:
        case ExprOp::ATAN:
        case ExprOp::SINH:
        case ExprOp::COSH:
        case ExprOp::ASINH:
        case ExprOp::ACOSH:
        case ExprOp::ATANH:
        case ExprOp::ERF:
        case ExprOp::ERFC:
        case ExprOp::ERFCX:
        case ExprOp::ERFINV:
        case ExprOp::ERFCINV:
        case ExprOp::TGAMMA:
        case ExprOp::LGAMMA:
        case ExprOp::DIGAMMA:
        case ExprOp::EXP:
        case ExprOp::EXPM1:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG1P:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::TANH:
        case ExprOp::NORMCDF:
            return true;
        default:
            return false;
    }
}

static bool isTransposePushThroughBinaryOp(ExprOp op) {
    switch (op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::EQUAL:
        case ExprOp::NOT_EQUAL:
        case ExprOp::LESS:
        case ExprOp::LESS_EQUAL:
        case ExprOp::GREATER:
        case ExprOp::GREATER_EQUAL:
        case ExprOp::LOGICAL_AND:
        case ExprOp::LOGICAL_OR:
        case ExprOp::MIN:
        case ExprOp::MAX:
            return true;
        default:
            return false;
    }
}

static bool isScalarLikeTransposeOperand(const ExprNode& node) {
    return node.op == ExprOp::SCALAR_FP || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR;
}

static std::optional<uint32_t> tryNormalizeTransposeAtRoot(PhysicalExpression& expr, uint32_t root_idx) {
    if (root_idx >= expr.nodes.size()) {
        throw std::runtime_error("tryNormalizeTransposeAtRoot root index out of range.");
    }

    const ExprNode& root = expr.nodes[root_idx];
    if (root.op == ExprOp::TRANSPOSE) {
        if (root.lhs == UINT32_MAX || root.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed transpose expression: missing lhs while normalizing transpose chain.");
        }
        return root.lhs;
    }

    if (isTransposePushThroughUnaryOp(root.op)) {
        if (root.lhs == UINT32_MAX || root.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed unary expression while normalizing transpose chain.");
        }
        const ExprNode& child = expr.nodes[root.lhs];
        if (child.op != ExprOp::TRANSPOSE) {
            return std::nullopt;
        }
        if (child.lhs == UINT32_MAX || child.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed inner transpose expression while normalizing unary transpose chain.");
        }

        ExprNode normalized = root;
        normalized.lhs = child.lhs;
        normalized.rhs = UINT32_MAX;
        normalized.aux = UINT32_MAX;
        const uint32_t normalized_idx = static_cast<uint32_t>(expr.nodes.size());
        expr.nodes.push_back(std::move(normalized));
        return normalized_idx;
    }

    if (isTransposePushThroughBinaryOp(root.op)) {
        if (root.lhs == UINT32_MAX || root.rhs == UINT32_MAX || root.lhs >= expr.nodes.size() || root.rhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed binary expression while normalizing transpose chain.");
        }

        auto unwrap_transpose_operand = [&](uint32_t operand_idx, bool& consumed_transpose) -> uint32_t {
            const ExprNode& operand = expr.nodes[operand_idx];
            if (operand.op == ExprOp::TRANSPOSE) {
                if (operand.lhs == UINT32_MAX || operand.lhs >= expr.nodes.size()) {
                    throw std::runtime_error("Malformed inner transpose expression while normalizing binary transpose chain.");
                }
                consumed_transpose = true;
                return operand.lhs;
            }
            return operand_idx;
        };

        bool consumed_lhs_transpose = false;
        bool consumed_rhs_transpose = false;
        uint32_t normalized_lhs = unwrap_transpose_operand(root.lhs, consumed_lhs_transpose);
        uint32_t normalized_rhs = unwrap_transpose_operand(root.rhs, consumed_rhs_transpose);
        if (!consumed_lhs_transpose && !consumed_rhs_transpose) {
            return std::nullopt;
        }

        auto transpose_if_needed = [&](uint32_t operand_idx, bool already_consumed_transpose) -> uint32_t {
            if (already_consumed_transpose || isScalarLikeTransposeOperand(expr.nodes[operand_idx])) {
                return operand_idx;
            }

            ExprNode transpose_node;
            transpose_node.op = ExprOp::TRANSPOSE;
            transpose_node.lhs = operand_idx;
            const uint32_t transpose_idx = static_cast<uint32_t>(expr.nodes.size());
            expr.nodes.push_back(std::move(transpose_node));
            return transpose_idx;
        };

        normalized_lhs = transpose_if_needed(normalized_lhs, consumed_lhs_transpose);
        normalized_rhs = transpose_if_needed(normalized_rhs, consumed_rhs_transpose);

        ExprNode normalized = root;
        normalized.lhs = normalized_lhs;
        normalized.rhs = normalized_rhs;
        normalized.aux = UINT32_MAX;
        const uint32_t normalized_idx = static_cast<uint32_t>(expr.nodes.size());
        expr.nodes.push_back(std::move(normalized));
        return normalized_idx;
    }

    return std::nullopt;
}

}  // namespace

Expression Expression::unaryOp(const Expression& inputExpr, ExprOp op) {
    if (!inputExpr.expr)
        throw std::runtime_error("Cannot apply unary op to empty expression");

    auto out = std::make_shared<PhysicalExpression>();
    out->inputs = inputExpr.expr->inputs;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    uint32_t newLhsIndex = cloneSubtree(*inputExpr.expr, inputExpr.nodeIndex, *out, oldToNew);

    if (op == ExprOp::TRANSPOSE) {
        std::optional<uint32_t> normalized_idx = tryNormalizeTransposeAtRoot(*out, newLhsIndex);
        if (normalized_idx.has_value()) {
            out->output_node = normalized_idx.value();
            return Expression(out, normalized_idx.value());
        }
    }

    ExprNode node{};
    node.op = op;
    node.lhs = newLhsIndex;

    uint32_t newIndex = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = newIndex;

    return Expression(out, newIndex);
}

Expression Expression::operator+(const Expression& other) const { return binaryOp(*this, other, ExprOp::ADD); }
Expression Expression::operator-(const Expression& other) const { return binaryOp(*this, other, ExprOp::SUB); }
Expression Expression::operator*(const Expression& other) const { return binaryOp(*this, other, ExprOp::MUL); }
Expression Expression::operator/(const Expression& other) const { return binaryOp(*this, other, ExprOp::DIV); }
Expression Expression::operator==(const Expression& other) const { return equal(other); }
Expression Expression::operator!=(const Expression& other) const { return notEqual(other); }
Expression Expression::operator<(const Expression& other) const { return lessThan(other); }
Expression Expression::operator<=(const Expression& other) const { return lessEqual(other); }
Expression Expression::operator>(const Expression& other) const { return greaterThan(other); }
Expression Expression::operator>=(const Expression& other) const { return greaterEqual(other); }
Expression Expression::operator-() const { return unaryOp(*this, ExprOp::NEG); }
Expression Expression::operator!() const { return logicalNot(); }
Expression Expression::equal(const Expression& other) const { return binaryOp(*this, other, ExprOp::EQUAL); }
Expression Expression::notEqual(const Expression& other) const { return binaryOp(*this, other, ExprOp::NOT_EQUAL); }
Expression Expression::lessThan(const Expression& other) const { return binaryOp(*this, other, ExprOp::LESS); }
Expression Expression::lessEqual(const Expression& other) const { return binaryOp(*this, other, ExprOp::LESS_EQUAL); }
Expression Expression::greaterThan(const Expression& other) const { return binaryOp(*this, other, ExprOp::GREATER); }
Expression Expression::greaterEqual(const Expression& other) const { return binaryOp(*this, other, ExprOp::GREATER_EQUAL); }
Expression Expression::logicalAnd(const Expression& other) const { return binaryOp(*this, other, ExprOp::LOGICAL_AND); }
Expression Expression::logicalOr(const Expression& other) const { return binaryOp(*this, other, ExprOp::LOGICAL_OR); }
Expression Expression::logicalNot() const { return unaryOp(*this, ExprOp::LOGICAL_NOT); }
Expression Expression::equal(const Expression& lhs, const Expression& rhs) { return lhs.equal(rhs); }
Expression Expression::notEqual(const Expression& lhs, const Expression& rhs) { return lhs.notEqual(rhs); }
Expression Expression::lessThan(const Expression& lhs, const Expression& rhs) { return lhs.lessThan(rhs); }
Expression Expression::lessEqual(const Expression& lhs, const Expression& rhs) { return lhs.lessEqual(rhs); }
Expression Expression::greaterThan(const Expression& lhs, const Expression& rhs) { return lhs.greaterThan(rhs); }
Expression Expression::greaterEqual(const Expression& lhs, const Expression& rhs) { return lhs.greaterEqual(rhs); }
Expression Expression::logicalAnd(const Expression& lhs, const Expression& rhs) { return lhs.logicalAnd(rhs); }
Expression Expression::logicalOr(const Expression& lhs, const Expression& rhs) { return lhs.logicalOr(rhs); }
Expression Expression::logicalNot(const Expression& input) { return input.logicalNot(); }
Expression Expression::select(const Expression& true_value, const Expression& false_value) const { return where(*this, true_value, false_value); }
Expression Expression::where(const Expression& condition, const Expression& true_value, const Expression& false_value) {
    return ternaryOp(condition, true_value, false_value, ExprOp::WHERE);
}
Expression Expression::select(const Expression& condition, const Expression& true_value, const Expression& false_value) {
    return where(condition, true_value, false_value);
}
Expression Expression::abs() const { return unaryOp(*this, ExprOp::ABS); }
Expression Expression::ceil() const { return unaryOp(*this, ExprOp::CEIL); }
Expression Expression::floor() const { return unaryOp(*this, ExprOp::FLOOR); }
Expression Expression::round() const { return unaryOp(*this, ExprOp::ROUND); }
Expression Expression::trunc() const { return unaryOp(*this, ExprOp::TRUNC); }
Expression Expression::sin() const { return unaryOp(*this, ExprOp::SIN); }
Expression Expression::cos() const { return unaryOp(*this, ExprOp::COS); }
Expression Expression::tan() const { return unaryOp(*this, ExprOp::TAN); }
Expression Expression::csc() const { return Expression::constantScalar(1.0) / this->sin(); }
Expression Expression::sec() const { return Expression::constantScalar(1.0) / this->cos(); }
Expression Expression::cot() const { return Expression::constantScalar(1.0) / this->tan(); }
Expression Expression::asin() const { return unaryOp(*this, ExprOp::ASIN); }
Expression Expression::acos() const { return unaryOp(*this, ExprOp::ACOS); }
Expression Expression::atan() const { return unaryOp(*this, ExprOp::ATAN); }
Expression Expression::acsc() const { return (Expression::constantScalar(1.0) / *this).asin(); }
Expression Expression::asec() const { return (Expression::constantScalar(1.0) / *this).acos(); }
Expression Expression::acot() const { return (Expression::constantScalar(1.0) / *this).atan(); }
Expression Expression::sinh() const { return unaryOp(*this, ExprOp::SINH); }
Expression Expression::cosh() const { return unaryOp(*this, ExprOp::COSH); }
Expression Expression::csch() const { return Expression::constantScalar(1.0) / this->sinh(); }
Expression Expression::sech() const { return Expression::constantScalar(1.0) / this->cosh(); }
Expression Expression::coth() const { return Expression::constantScalar(1.0) / this->tanh(); }
Expression Expression::asinh() const { return unaryOp(*this, ExprOp::ASINH); }
Expression Expression::acosh() const { return unaryOp(*this, ExprOp::ACOSH); }
Expression Expression::atanh() const { return unaryOp(*this, ExprOp::ATANH); }
Expression Expression::acsch() const { return (Expression::constantScalar(1.0) / *this).asinh(); }
Expression Expression::asech() const { return (Expression::constantScalar(1.0) / *this).acosh(); }
Expression Expression::acoth() const { return (Expression::constantScalar(1.0) / *this).atanh(); }
Expression Expression::erf() const { return unaryOp(*this, ExprOp::ERF); }
Expression Expression::erfc() const { return unaryOp(*this, ExprOp::ERFC); }
Expression Expression::erfcx() const { return unaryOp(*this, ExprOp::ERFCX); }
Expression Expression::erfinv() const { return unaryOp(*this, ExprOp::ERFINV); }
Expression Expression::erfcinv() const { return unaryOp(*this, ExprOp::ERFCINV); }
Expression Expression::tgamma() const { return unaryOp(*this, ExprOp::TGAMMA); }
Expression Expression::lgamma() const { return unaryOp(*this, ExprOp::LGAMMA); }
Expression Expression::digamma() const { return unaryOp(*this, ExprOp::DIGAMMA); }
Expression Expression::expm1() const { return unaryOp(*this, ExprOp::EXPM1); }
Expression Expression::log1p() const { return unaryOp(*this, ExprOp::LOG1P); }
Expression Expression::sqrt() const { return unaryOp(*this, ExprOp::SQRT); }
Expression Expression::sqrt(const Expression& expr) { return unaryOp(expr, ExprOp::SQRT); }
Expression Expression::tanh() const { return unaryOp(*this, ExprOp::TANH); }
Expression Expression::normcdf() const { return unaryOp(*this, ExprOp::NORMCDF); }

Expression Expression::softmax(cudnnSoftmaxAlgorithm_t algorithm, cudnnSoftmaxMode_t mode) const {
    if (algorithm == CUDNN_SOFTMAX_LOG) {
        throw std::invalid_argument("Expression::softmax computes ordinary softmax; use Expression::logSoftmax for CUDNN_SOFTMAX_LOG.");
    }
    if (algorithm != CUDNN_SOFTMAX_FAST && algorithm != CUDNN_SOFTMAX_ACCURATE) {
        throw std::invalid_argument("Expression::softmax received unsupported cudnnSoftmaxAlgorithm_t.");
    }
    if (mode != CUDNN_SOFTMAX_MODE_CHANNEL && mode != CUDNN_SOFTMAX_MODE_INSTANCE) {
        throw std::invalid_argument("Expression::softmax received unsupported cudnnSoftmaxMode_t.");
    }
    Expression out = unaryOp(*this, ExprOp::SOFTMAX);
    out.expr->nodes[out.nodeIndex].softmax_algorithm = algorithm;
    out.expr->nodes[out.nodeIndex].softmax_mode = mode;
    return out;
}

Expression Expression::logSoftmax(cudnnSoftmaxMode_t mode) const {
    if (mode != CUDNN_SOFTMAX_MODE_CHANNEL && mode != CUDNN_SOFTMAX_MODE_INSTANCE) {
        throw std::invalid_argument("Expression::logSoftmax received unsupported cudnnSoftmaxMode_t.");
    }
    Expression out = unaryOp(*this, ExprOp::SOFTMAX);
    out.expr->nodes[out.nodeIndex].softmax_algorithm = CUDNN_SOFTMAX_LOG;
    out.expr->nodes[out.nodeIndex].softmax_mode = mode;
    return out;
}

Expression Expression::reshape(const std::vector<uint64_t>& new_dims) const {
    if (!expr)
        throw std::runtime_error("Cannot reshape an empty expression");
    if (new_dims.empty())
        throw std::invalid_argument("Expression::reshape requires at least one dimension.");
    for (uint64_t d : new_dims) {
        if (d == 0)
            throw std::invalid_argument("Expression::reshape dimensions must be non-zero.");
    }
    Expression out = unaryOp(*this, ExprOp::RESHAPE);
    out.expr->nodes[out.nodeIndex].reshape_dims = new_dims;
    return out;
}

Expression Expression::stridedView(const std::vector<uint64_t>& dims,
                                   const std::vector<uint64_t>& strides_elements,
                                   uint64_t element_offset) const {
    if (!expr)
        throw std::runtime_error("Cannot create a strided view from an empty expression");
    if (dims.empty() || dims.size() != strides_elements.size()) {
        throw std::invalid_argument("Expression::stridedView requires dimensions and strides with the same non-zero rank.");
    }
    for (uint64_t d : dims) {
        if (d == 0)
            throw std::invalid_argument("Expression::stridedView dimensions must be non-zero.");
    }
    Expression out = unaryOp(*this, ExprOp::STRIDED_VIEW);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.view_dims = dims;
    node.view_strides = strides_elements;
    node.view_element_offset = element_offset;
    return out;
}

Expression Expression::unsqueeze(const std::vector<uint64_t>& unsqueeze_axes) const {
    if (!expr)
        throw std::runtime_error("Cannot unsqueeze an empty expression");

    Expression out = unaryOp(*this, ExprOp::UNSQUEEZE);
    std::vector<uint64_t> normalized = unsqueeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    out.expr->nodes[out.nodeIndex].unsqueeze_axes = std::move(normalized);
    return out;
}

Expression Expression::squeeze(const std::vector<uint64_t>& squeeze_axes) const {
    if (!expr)
        throw std::runtime_error("Cannot squeeze an empty expression");

    Expression out = unaryOp(*this, ExprOp::SQUEEZE);
    std::vector<uint64_t> normalized = squeeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    out.expr->nodes[out.nodeIndex].squeeze_axes = std::move(normalized);
    return out;
}

Expression Expression::transpose() const { return unaryOp(*this, ExprOp::TRANSPOSE); }

Expression Expression::pow(const Expression& exponent) const { return binaryOp(*this, exponent, ExprOp::POW); }

Expression Expression::matmul(const Expression& lhs,
                              const Expression& rhs,
                              bool transpose_lhs,
                              bool transpose_rhs,
                              std::optional<DataType> compute_dtype,
                              std::optional<DataType> output_dtype) {
    Expression out = binaryOp(lhs, rhs, ExprOp::MATMUL);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.transpose_lhs = transpose_lhs;
    node.transpose_rhs = transpose_rhs;
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }
    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    return out;
}

Expression Expression::rmsNorm(const Expression& input,
                               const Expression& scale,
                               uint64_t normalized_feature_count,
                               double epsilon,
                               std::optional<DataType> compute_dtype,
                               std::optional<DataType> output_dtype) {
    if (normalized_feature_count == 0) {
        throw std::invalid_argument("Expression::rmsNorm normalized_feature_count must be non-zero.");
    }
    if (!(epsilon > 0.0)) {
        throw std::invalid_argument("Expression::rmsNorm epsilon must be > 0.");
    }

    Expression out = binaryOp(input, scale, ExprOp::RMSNORM);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.rms_norm_normalized_feature_count = normalized_feature_count;
    node.rms_norm_epsilon = epsilon;
    node.rms_norm_fused_activation = CudnnRmsNormFusedActivation::NONE;
    node.compute_dtype = compute_dtype.value_or(DataType::FP32);
    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    return out;
}

Expression Expression::swish() const { return *this * this->sigmoid(); }

static uint32_t cloneSubtreeIntoMergedExpression(const Expression& src_expr,
                                                 PhysicalExpression& dst,
                                                 std::unordered_map<uint32_t, uint32_t>& old_to_new,
                                                 std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name) {
    PhysicalExpression src = src_expr.expression();
    return cloneSubtreeWithMergedInputs(src, src.output_node, dst, old_to_new, dst_input_slots_by_name);
}

uint32_t Expression::cloneInto(const PhysicalExpression& src,
                               PhysicalExpression& dst,
                               std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name) {
    if (src.output_node >= src.nodes.size()) {
        throw std::runtime_error("Expression::cloneInto source output node is out of range.");
    }
    std::unordered_map<uint32_t, uint32_t> old_to_new;
    return cloneSubtreeWithMergedInputs(src, src.output_node, dst, old_to_new, dst_input_slots_by_name);
}

uint32_t Expression::encodeLowerableGemmScaleExpression(const Expression& scale_expr,
                                                        PhysicalExpression& dst,
                                                        std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name,
                                                        double& scale_fp) {
    PhysicalExpression scale = scale_expr.expression();

    struct SimpleScaleEncodingResult {
        bool success = false;
        uint32_t dynamic_node = UINT32_MAX;
        double constant_scale = 1.0;
    };

    std::function<SimpleScaleEncodingResult(uint32_t)> try_encode_simple = [&](uint32_t node_idx) -> SimpleScaleEncodingResult {
        const ExprNode& src_node = scale.nodes.at(node_idx);
        if (src_node.op == ExprOp::SCALAR_FP) {
            return SimpleScaleEncodingResult{true, UINT32_MAX, src_node.scalar_fp};
        }
        if (src_node.op == ExprOp::INPUT || src_node.op == ExprOp::RUNTIME_SCALAR || src_node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            std::unordered_map<uint32_t, uint32_t> old_to_new;
            uint32_t cloned =
                cloneSubtreeIntoMergedExpression(Expression(scale_expr.expr, node_idx), dst, old_to_new, dst_input_slots_by_name);
            return SimpleScaleEncodingResult{true, cloned, 1.0};
        }
        if (src_node.op == ExprOp::MUL) {
            const SimpleScaleEncodingResult lhs = try_encode_simple(src_node.lhs);
            const SimpleScaleEncodingResult rhs = try_encode_simple(src_node.rhs);
            if (!lhs.success || !rhs.success) {
                return {};
            }
            if (lhs.dynamic_node != UINT32_MAX && rhs.dynamic_node != UINT32_MAX) {
                return {};
            }
            return SimpleScaleEncodingResult{
                true, lhs.dynamic_node != UINT32_MAX ? lhs.dynamic_node : rhs.dynamic_node, lhs.constant_scale * rhs.constant_scale};
        }
        return {};
    };

    const SimpleScaleEncodingResult simple = try_encode_simple(scale.output_node);
    if (simple.success) {
        scale_fp *= simple.constant_scale;
        return simple.dynamic_node;
    }

    std::unordered_map<uint32_t, uint32_t> old_to_new;
    return cloneSubtreeIntoMergedExpression(scale_expr, dst, old_to_new, dst_input_slots_by_name);
}

Expression Expression::gemm(const Expression& lhs,
                            const Expression& rhs,
                            const Expression& addend,
                            double alpha,
                            double beta,
                            bool transpose_lhs,
                            bool transpose_rhs,
                            bool transpose_addend,
                            std::optional<DataType> compute_dtype,
                            std::optional<DataType> output_dtype) {
    return gemm(lhs,
                rhs,
                addend,
                Expression::constantScalar(alpha),
                Expression::constantScalar(beta),
                transpose_lhs,
                transpose_rhs,
                transpose_addend,
                compute_dtype,
                output_dtype);
}

Expression Expression::gemm(const Expression& lhs,
                            const Expression& rhs,
                            const Expression& addend,
                            const Expression& alpha,
                            const Expression& beta,
                            bool transpose_lhs,
                            bool transpose_rhs,
                            bool transpose_addend,
                            std::optional<DataType> compute_dtype,
                            std::optional<DataType> output_dtype) {
    if (!lhs.expr || !rhs.expr || !addend.expr || !alpha.expr || !beta.expr) {
        throw std::runtime_error("Cannot build GEMM from empty expressions.");
    }

    auto out = std::make_shared<PhysicalExpression>();
    std::unordered_map<std::string, uint32_t> merged_by_name;

    auto clone_root = [&](const Expression& src_expr) {
        std::unordered_map<uint32_t, uint32_t> old_to_new;
        return cloneSubtreeIntoMergedExpression(src_expr, *out, old_to_new, merged_by_name);
    };

    const uint32_t new_lhs_index = clone_root(lhs);
    const uint32_t new_rhs_index = clone_root(rhs);
    const uint32_t new_aux_index = clone_root(addend);

    ExprNode node{};
    node.op = ExprOp::GEMM;
    node.lhs = new_lhs_index;
    node.rhs = new_rhs_index;
    node.aux = new_aux_index;
    node.alpha_fp = 1.0;
    node.beta_fp = 1.0;
    node.alpha_node = encodeLowerableGemmScaleExpression(alpha, *out, merged_by_name, node.alpha_fp);
    node.beta_node = encodeLowerableGemmScaleExpression(beta, *out, merged_by_name, node.beta_fp);
    node.transpose_lhs = transpose_lhs;
    node.transpose_rhs = transpose_rhs;
    node.transpose_aux = transpose_addend;
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }
    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }

    const uint32_t new_index = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = new_index;
    return Expression(out, new_index);
}

Expression Expression::rotaryPositionEmbedding(RotaryPositionEmbeddingOptions options) const {
    if (!expr) {
        throw std::runtime_error("Cannot build RoPE from an empty expression.");
    }
    if (options.base <= 0.0) {
        throw std::runtime_error("RoPE base must be positive.");
    }
    if (options.scaling_factor <= 0.0) {
        throw std::runtime_error("RoPE scaling factor must be positive.");
    }
    if (options.attention_factor.has_value() && options.attention_factor.value() <= 0.0) {
        throw std::runtime_error("RoPE attention_factor must be positive.");
    }
    if (options.rotary_dim != 0 && ((options.rotary_dim & 1ULL) != 0ULL)) {
        throw std::runtime_error("RoPE rotary_dim must be even.");
    }
    if (options.scaling_kind == RotaryScalingKind::DynamicNTK && options.original_max_position_embeddings == 0) {
        throw std::runtime_error("Dynamic-NTK RoPE scaling requires original_max_position_embeddings.");
    }
    if ((options.scaling_kind == RotaryScalingKind::Yarn || options.scaling_kind == RotaryScalingKind::LongRope ||
         options.scaling_kind == RotaryScalingKind::Llama3) &&
        options.original_max_position_embeddings == 0) {
        throw std::runtime_error("This RoPE scaling kind requires original_max_position_embeddings.");
    }
    if (options.scaling_kind == RotaryScalingKind::Yarn && (options.yarn_beta_fast <= 0.0 || options.yarn_beta_slow <= 0.0)) {
        throw std::runtime_error("YaRN RoPE scaling requires positive beta parameters.");
    }
    if (options.scaling_kind == RotaryScalingKind::Llama3 &&
        (options.llama3_low_freq_factor <= 0.0 || options.llama3_high_freq_factor <= 0.0 ||
         options.llama3_high_freq_factor <= options.llama3_low_freq_factor)) {
        throw std::runtime_error("Llama3 RoPE scaling requires 0 < low_freq_factor < high_freq_factor.");
    }
    if (options.scaling_kind == RotaryScalingKind::LongRope) {
        if (options.rotary_dim == 0) {
            throw std::runtime_error("LongRoPE scaling requires an explicit rotary_dim.");
        }
        const uint64_t expected = options.rotary_dim / 2;
        if (options.long_rope_short_factors.size() != expected || options.long_rope_long_factors.size() != expected) {
            throw std::runtime_error("LongRoPE scaling requires short/long factor lists of length rotary_dim / 2.");
        }
        for (double factor : options.long_rope_short_factors) {
            if (factor <= 0.0) {
                throw std::runtime_error("LongRoPE short factors must be positive.");
            }
        }
        for (double factor : options.long_rope_long_factors) {
            if (factor <= 0.0) {
                throw std::runtime_error("LongRoPE long factors must be positive.");
            }
        }
    }

    Expression out = unaryOp(*this, ExprOp::ROPE);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.rope_sequence_axis = options.sequence_axis;
    node.rope_head_dim_axis = options.head_dim_axis;
    node.rope_rotary_dim = options.rotary_dim;
    node.rope_base = options.base;
    node.rope_position_offset = options.position_offset;
    node.rope_interleaved = options.interleaved;
    node.rope_inverse = options.inverse;
    node.rope_scaling_kind = options.scaling_kind;
    node.rope_scaling_factor = options.scaling_factor;
    node.rope_original_max_position_embeddings = options.original_max_position_embeddings;
    node.rope_attention_factor = [&]() -> double {
        if (options.attention_factor.has_value()) {
            return options.attention_factor.value();
        }
        if (options.scaling_kind == RotaryScalingKind::Yarn) {
            return options.scaling_factor <= 1.0 ? 1.0 : 0.1 * std::log(options.scaling_factor) + 1.0;
        }
        if (options.scaling_kind == RotaryScalingKind::LongRope) {
            return options.scaling_factor <= 1.0
                       ? 1.0
                       : std::sqrt(1.0 + std::log(options.scaling_factor) /
                                             std::log(static_cast<double>(options.original_max_position_embeddings)));
        }
        return 1.0;
    }();
    node.rope_yarn_beta_fast = options.yarn_beta_fast;
    node.rope_yarn_beta_slow = options.yarn_beta_slow;
    node.rope_llama3_low_freq_factor = options.llama3_low_freq_factor;
    node.rope_llama3_high_freq_factor = options.llama3_high_freq_factor;
    node.rope_long_rope_short_factors = options.long_rope_short_factors;
    node.rope_long_rope_long_factors = options.long_rope_long_factors;
    node.rope_allow_in_place_materialization = options.allow_in_place_materialization;
    if (options.output_dtype.has_value()) {
        node.output_dtype = options.output_dtype.value();
    }
    if (options.compute_dtype.has_value()) {
        node.compute_dtype = options.compute_dtype.value();
    }
    return out;
}

namespace {
bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
}

bool experimentalCudnnAttentionSupportSurfaceProbeEnabled() {
    return envFlagEnabled("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE") ||
           envFlagEnabled("THOR_RUN_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE") ||
           envFlagEnabled("THOR_RUN_EXPERIMENTAL_CUDNN_FP8_ATTENTION_SUPPORT_SURFACE");
}

void applyAttentionOptions(ExprNode& node, const AttentionOptions& options, bool use_bias) {
    node.attention_q_layout = options.q_layout;
    node.attention_k_layout = options.k_layout;
    node.attention_v_layout = options.v_layout;
    node.attention_o_layout = options.o_layout;
    node.attention_mask_kind = options.mask_kind;
    node.attention_diagonal_left_bound = options.diagonal_left_bound;
    node.attention_diagonal_right_bound = options.diagonal_right_bound;
    node.attention_has_scale = options.attention_scale.has_value();
    node.attention_scale = options.attention_scale.value_or(0.0f);
    node.attention_use_alibi_mask = options.use_alibi_mask;
    node.attention_use_bias = use_bias;
    node.attention_use_padding_mask = options.use_padding_mask;
    node.attention_use_paged_kv_cache = options.use_paged_kv_cache;
    node.attention_paged_kv_max_sequence_length = options.paged_kv_max_sequence_length;
    node.attention_dropout_probability = options.dropout_probability;
    node.attention_use_fp8_forward_scaling = options.use_fp8_forward_scaling;
    if (options.compute_dtype.has_value()) {
        node.compute_dtype = options.compute_dtype.value();
    }
    if (options.output_dtype.has_value()) {
        node.output_dtype = options.output_dtype.value();
    }
}

void validateAttentionOptions(const AttentionOptions& options, bool use_bias, bool use_ragged_offsets = false) {
    if (use_ragged_offsets && (options.q_layout != AttentionTensorLayout::BSHD || options.k_layout != AttentionTensorLayout::BSHD ||
                               options.v_layout != AttentionTensorLayout::BSHD || options.o_layout != AttentionTensorLayout::BSHD)) {
        throw std::runtime_error(
            "Ragged attention requires BSHD physical layouts for q/k/v/o because cuDNN ragged offsets index packed token-contiguous THD "
            "storage.");
    }
    if (options.diagonal_left_bound < 0 || options.diagonal_right_bound < 0) {
        throw std::runtime_error("Attention diagonal/sliding-window bounds must be non-negative.");
    }
    if (options.use_alibi_mask) {
        const bool uses_causal_diagonal = options.mask_kind == AttentionMaskKind::CausalTopLeft ||
                                          options.mask_kind == AttentionMaskKind::CausalBottomRight ||
                                          options.mask_kind == AttentionMaskKind::SlidingWindowTopLeft ||
                                          options.mask_kind == AttentionMaskKind::SlidingWindowBottomRight;
        if ((!uses_causal_diagonal || options.diagonal_right_bound != 0) && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
            throw std::runtime_error(
                "AttentionOptions::use_alibi_mask requires causal diagonal masking with diagonal_right_bound == 0 because cuDNN rejects "
                "ALiBi with positive right bounds.");
        }
    }
    if (options.dropout_probability < 0.0f || options.dropout_probability >= 1.0f) {
        throw std::runtime_error("AttentionOptions::dropout_probability must be in [0, 1).");
    }
    if (use_ragged_offsets && options.use_paged_kv_cache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throw std::runtime_error("Ragged attention and paged KV cache are separate variable-length modes and cannot be combined.");
    }
    if (options.use_paged_kv_cache && options.paged_kv_max_sequence_length <= 0) {
        throw std::runtime_error("Paged KV attention requires paged_kv_max_sequence_length > 0.");
    }
    if (options.use_paged_kv_cache && !options.use_padding_mask) {
        throw std::runtime_error("Paged KV attention requires q_seq_len and kv_seq_len padding-mask tensors.");
    }
    if (options.use_paged_kv_cache && use_bias && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throw std::runtime_error("Paged KV attention with additive bias is disabled until the paged-bias layout is defined.");
    }
    if (options.use_paged_kv_cache && options.dropout_probability > 0.0f && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throw std::runtime_error("Paged KV attention is inference-only and cannot currently be combined with dropout.");
    }
    if ((options.mask_kind == AttentionMaskKind::CausalBottomRight || options.mask_kind == AttentionMaskKind::SlidingWindowBottomRight) &&
        (use_bias || options.use_alibi_mask || options.dropout_probability > 0.0f) &&
        !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throw std::runtime_error(
            "AttentionMaskKind::CausalBottomRight/SlidingWindowBottomRight is currently supported only without additive bias, ALiBi, or "
            "dropout in the cuDNN primary SDPA path.");
    }
}
}  // namespace

Expression Expression::attentionWithOptionalMetadata(const Expression& q,
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
                                                     const Expression* descale_q,
                                                     const Expression* descale_k,
                                                     const Expression* descale_v,
                                                     const Expression* descale_s,
                                                     const Expression* scale_s,
                                                     const Expression* scale_o,
                                                     const Expression* amax_s,
                                                     const Expression* amax_o) {
    if (!q.expr || !k.expr || !v.expr || (bias != nullptr && !bias->expr) || (q_seq_len != nullptr && !q_seq_len->expr) ||
        (kv_seq_len != nullptr && !kv_seq_len->expr) || (q_ragged_offsets != nullptr && !q_ragged_offsets->expr) ||
        (kv_ragged_offsets != nullptr && !kv_ragged_offsets->expr) || (page_table_k != nullptr && !page_table_k->expr) ||
        (page_table_v != nullptr && !page_table_v->expr) || (dropout_seed != nullptr && !dropout_seed->expr) ||
        (dropout_offset != nullptr && !dropout_offset->expr) || (descale_q != nullptr && !descale_q->expr) ||
        (descale_k != nullptr && !descale_k->expr) || (descale_v != nullptr && !descale_v->expr) ||
        (descale_s != nullptr && !descale_s->expr) || (scale_s != nullptr && !scale_s->expr) || (scale_o != nullptr && !scale_o->expr) ||
        (amax_s != nullptr && !amax_s->expr) || (amax_o != nullptr && !amax_o->expr)) {
        throw std::runtime_error("Cannot build attention from empty expressions.");
    }
    const bool use_ragged_offsets = q_ragged_offsets != nullptr || kv_ragged_offsets != nullptr;
    const bool use_paged_kv_cache = page_table_k != nullptr || page_table_v != nullptr;
    if ((q_ragged_offsets == nullptr) != (kv_ragged_offsets == nullptr)) {
        throw std::runtime_error("q_ragged_offsets and kv_ragged_offsets must be provided together for ragged attention.");
    }
    if ((page_table_k == nullptr) != (page_table_v == nullptr)) {
        throw std::runtime_error("page_table_k and page_table_v must be provided together for paged KV attention.");
    }
    if (use_paged_kv_cache) {
        options.use_paged_kv_cache = true;
    }
    validateAttentionOptions(options, bias != nullptr, use_ragged_offsets);
    if (options.use_padding_mask && (q_seq_len == nullptr || kv_seq_len == nullptr)) {
        throw std::runtime_error("AttentionOptions::use_padding_mask requires q_seq_len and kv_seq_len expressions.");
    }
    if (use_ragged_offsets && (q_seq_len == nullptr || kv_seq_len == nullptr)) {
        throw std::runtime_error(
            "Ragged attention requires q_seq_len and kv_seq_len expressions in addition to q/kv ragged offsets because cuDNN THD uses "
            "sequence lengths for padding-mask semantics.");
    }
    if (!options.use_padding_mask && (q_seq_len != nullptr || kv_seq_len != nullptr)) {
        throw std::runtime_error("q_seq_len/kv_seq_len were provided but AttentionOptions::use_padding_mask is false.");
    }
    if (options.use_paged_kv_cache && !use_paged_kv_cache) {
        throw std::runtime_error("AttentionOptions::use_paged_kv_cache requires page_table_k and page_table_v expressions.");
    }
    if (options.dropout_probability > 0.0f && (dropout_seed == nullptr || dropout_offset == nullptr)) {
        throw std::runtime_error("AttentionOptions::dropout_probability requires dropout_seed and dropout_offset expressions.");
    }
    if (options.dropout_probability == 0.0f && (dropout_seed != nullptr || dropout_offset != nullptr)) {
        throw std::runtime_error("dropout_seed/dropout_offset were provided but AttentionOptions::dropout_probability is zero.");
    }
    if ((dropout_seed == nullptr) != (dropout_offset == nullptr)) {
        throw std::runtime_error("dropout_seed and dropout_offset must be provided together for attention dropout.");
    }
    const bool use_fp8_forward_scaling = descale_q != nullptr || descale_k != nullptr || descale_v != nullptr || descale_s != nullptr ||
                                         scale_s != nullptr || scale_o != nullptr || amax_s != nullptr || amax_o != nullptr ||
                                         options.use_fp8_forward_scaling;
    if (use_fp8_forward_scaling) {
        if (descale_q == nullptr || descale_k == nullptr || descale_v == nullptr || descale_s == nullptr || scale_s == nullptr ||
            scale_o == nullptr || amax_s == nullptr || amax_o == nullptr) {
            throw std::runtime_error(
                "FP8 attention forward requires all descale_q/descale_k/descale_v/descale_s/scale_s/scale_o/amax_s/amax_o expressions.");
        }
        if (bias != nullptr) {
            throw std::runtime_error("FP8 attention forward does not support additive bias on the validated cuDNN support surface.");
        }
        if (dropout_seed != nullptr || dropout_offset != nullptr || options.dropout_probability > 0.0f) {
            throw std::runtime_error("FP8 attention forward does not support dropout on the validated cuDNN support surface.");
        }
        if (q_ragged_offsets != nullptr || kv_ragged_offsets != nullptr) {
            throw std::runtime_error("FP8 attention forward is not enabled for ragged offsets on the validated cuDNN support surface.");
        }
        if (page_table_k != nullptr || page_table_v != nullptr || options.use_paged_kv_cache) {
            throw std::runtime_error("FP8 attention forward is not enabled for paged KV on the validated cuDNN support surface.");
        }
        if (options.use_alibi_mask) {
            throw std::runtime_error("FP8 attention forward is not enabled for ALiBi on the validated cuDNN support surface.");
        }
        if (options.mask_kind == AttentionMaskKind::CausalBottomRight || options.mask_kind == AttentionMaskKind::SlidingWindowTopLeft ||
            options.mask_kind == AttentionMaskKind::SlidingWindowBottomRight) {
            throw std::runtime_error("FP8 attention forward is enabled only for no-mask or causal-top-left mask semantics.");
        }
        options.use_fp8_forward_scaling = true;
    }

    auto out = std::make_shared<PhysicalExpression>();
    std::unordered_map<std::string, uint32_t> merged_by_name;

    auto clone_root = [&](const Expression& src_expr) {
        std::unordered_map<uint32_t, uint32_t> old_to_new;
        return cloneSubtreeIntoMergedExpression(src_expr, *out, old_to_new, merged_by_name);
    };

    const uint32_t q_node = clone_root(q);
    const uint32_t k_node = clone_root(k);
    const uint32_t v_node = clone_root(v);
    const uint32_t bias_node = bias != nullptr ? clone_root(*bias) : UINT32_MAX;
    const uint32_t q_len_node = q_seq_len != nullptr ? clone_root(*q_seq_len) : UINT32_MAX;
    const uint32_t kv_len_node = kv_seq_len != nullptr ? clone_root(*kv_seq_len) : UINT32_MAX;
    const uint32_t q_ragged_node = q_ragged_offsets != nullptr ? clone_root(*q_ragged_offsets) : UINT32_MAX;
    const uint32_t kv_ragged_node = kv_ragged_offsets != nullptr ? clone_root(*kv_ragged_offsets) : UINT32_MAX;
    const uint32_t page_table_k_node = page_table_k != nullptr ? clone_root(*page_table_k) : UINT32_MAX;
    const uint32_t page_table_v_node = page_table_v != nullptr ? clone_root(*page_table_v) : UINT32_MAX;
    const uint32_t dropout_seed_node = dropout_seed != nullptr ? clone_root(*dropout_seed) : UINT32_MAX;
    const uint32_t dropout_offset_node = dropout_offset != nullptr ? clone_root(*dropout_offset) : UINT32_MAX;
    const uint32_t descale_q_node = descale_q != nullptr ? clone_root(*descale_q) : UINT32_MAX;
    const uint32_t descale_k_node = descale_k != nullptr ? clone_root(*descale_k) : UINT32_MAX;
    const uint32_t descale_v_node = descale_v != nullptr ? clone_root(*descale_v) : UINT32_MAX;
    const uint32_t descale_s_node = descale_s != nullptr ? clone_root(*descale_s) : UINT32_MAX;
    const uint32_t scale_s_node = scale_s != nullptr ? clone_root(*scale_s) : UINT32_MAX;
    const uint32_t scale_o_node = scale_o != nullptr ? clone_root(*scale_o) : UINT32_MAX;
    const uint32_t amax_s_node = amax_s != nullptr ? clone_root(*amax_s) : UINT32_MAX;
    const uint32_t amax_o_node = amax_o != nullptr ? clone_root(*amax_o) : UINT32_MAX;

    ExprNode node{};
    node.op = ExprOp::ATTENTION;
    node.lhs = q_node;
    node.rhs = k_node;
    node.aux = v_node;
    node.alpha_node = bias_node;
    node.attention_seq_len_q_node = q_len_node;
    node.attention_seq_len_kv_node = kv_len_node;
    node.attention_use_ragged_offsets = use_ragged_offsets;
    node.attention_ragged_offset_q_node = q_ragged_node;
    node.attention_ragged_offset_kv_node = kv_ragged_node;
    node.attention_use_paged_kv_cache = options.use_paged_kv_cache;
    node.attention_paged_kv_max_sequence_length = options.paged_kv_max_sequence_length;
    node.attention_page_table_k_node = page_table_k_node;
    node.attention_page_table_v_node = page_table_v_node;
    node.attention_dropout_seed_node = dropout_seed_node;
    node.attention_dropout_offset_node = dropout_offset_node;
    node.attention_descale_q_node = descale_q_node;
    node.attention_descale_k_node = descale_k_node;
    node.attention_descale_v_node = descale_v_node;
    node.attention_descale_s_node = descale_s_node;
    node.attention_scale_s_node = scale_s_node;
    node.attention_scale_o_node = scale_o_node;
    node.attention_amax_s_node = amax_s_node;
    node.attention_amax_o_node = amax_o_node;
    applyAttentionOptions(node, options, bias != nullptr);

    const uint32_t new_index = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(std::move(node));
    out->output_node = new_index;
    return Expression(out, new_index);
}


Expression Expression::embeddingLookup(const Expression& indices,
                                       const Expression& weights,
                                       std::optional<uint64_t> padding_index,
                                       std::optional<DataType> output_dtype) {
    Expression out = binaryOp(indices, weights, ExprOp::EMBEDDING_LOOKUP);
    ExprNode& node = out.expr->nodes.at(out.nodeIndex);
    node.embedding_has_padding_index = padding_index.has_value();
    node.embedding_padding_index = padding_index.value_or(0);
    node.output_dtype = output_dtype;
    return out;
}

Expression Expression::scaledDotProductAttention(const Expression& q, const Expression& k, const Expression& v, AttentionOptions options) {
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionWithDropout(const Expression& q,
                                                            const Expression& k,
                                                            const Expression& v,
                                                            const Expression& dropout_seed,
                                                            const Expression& dropout_offset,
                                                            AttentionOptions options) {
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::scaledDotProductAttention(
    const Expression& q, const Expression& k, const Expression& v, const Expression& bias, AttentionOptions options) {
    return attentionWithOptionalMetadata(
        q, k, v, &bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionWithDropout(const Expression& q,
                                                            const Expression& k,
                                                            const Expression& v,
                                                            const Expression& bias,
                                                            const Expression& dropout_seed,
                                                            const Expression& dropout_offset,
                                                            AttentionOptions options) {
    return attentionWithOptionalMetadata(
        q, k, v, &bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionFp8Forward(const Expression& q,
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
                                                           AttentionOptions options) {
    options.use_fp8_forward_scaling = true;
    return attentionWithOptionalMetadata(q,
                                         k,
                                         v,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         std::move(options),
                                         &descale_q,
                                         &descale_k,
                                         &descale_v,
                                         &descale_s,
                                         &scale_s,
                                         &scale_o,
                                         &amax_s,
                                         &amax_o);
}

Expression Expression::scaledDotProductAttentionFp8Forward(const Expression& q,
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
                                                           AttentionOptions options) {
    options.use_padding_mask = true;
    options.use_fp8_forward_scaling = true;
    return attentionWithOptionalMetadata(q,
                                         k,
                                         v,
                                         nullptr,
                                         &q_seq_len,
                                         &kv_seq_len,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         std::move(options),
                                         &descale_q,
                                         &descale_k,
                                         &descale_v,
                                         &descale_s,
                                         &scale_s,
                                         &scale_o,
                                         &amax_s,
                                         &amax_o);
}

Expression Expression::scaledDotProductAttentionPagedKv(const Expression& q,
                                                        const Expression& k,
                                                        const Expression& v,
                                                        const Expression& q_seq_len,
                                                        const Expression& kv_seq_len,
                                                        const Expression& page_table_k,
                                                        const Expression& page_table_v,
                                                        AttentionOptions options) {
    options.use_paged_kv_cache = true;
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, &q_seq_len, &kv_seq_len, nullptr, nullptr, &page_table_k, &page_table_v, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionRagged(const Expression& q,
                                                       const Expression& k,
                                                       const Expression& v,
                                                       const Expression& q_offsets,
                                                       const Expression& kv_offsets,
                                                       AttentionOptions options) {
    options.use_padding_mask = false;
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, nullptr, nullptr, &q_offsets, &kv_offsets, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionRagged(const Expression& q,
                                                       const Expression& k,
                                                       const Expression& v,
                                                       const Expression& q_seq_len,
                                                       const Expression& kv_seq_len,
                                                       const Expression& q_offsets,
                                                       const Expression& kv_offsets,
                                                       AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, &q_seq_len, &kv_seq_len, &q_offsets, &kv_offsets, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionRagged(const Expression& q,
                                                       const Expression& k,
                                                       const Expression& v,
                                                       const Expression& bias,
                                                       const Expression& q_seq_len,
                                                       const Expression& kv_seq_len,
                                                       const Expression& q_offsets,
                                                       const Expression& kv_offsets,
                                                       AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, &bias, &q_seq_len, &kv_seq_len, &q_offsets, &kv_offsets, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionRagged(const Expression& q,
                                                       const Expression& k,
                                                       const Expression& v,
                                                       const Expression& q_seq_len,
                                                       const Expression& kv_seq_len,
                                                       const Expression& q_offsets,
                                                       const Expression& kv_offsets,
                                                       const Expression& dropout_seed,
                                                       const Expression& dropout_offset,
                                                       AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(q,
                                         k,
                                         v,
                                         nullptr,
                                         &q_seq_len,
                                         &kv_seq_len,
                                         &q_offsets,
                                         &kv_offsets,
                                         nullptr,
                                         nullptr,
                                         &dropout_seed,
                                         &dropout_offset,
                                         std::move(options));
}

Expression Expression::scaledDotProductAttentionRagged(const Expression& q,
                                                       const Expression& k,
                                                       const Expression& v,
                                                       const Expression& bias,
                                                       const Expression& q_seq_len,
                                                       const Expression& kv_seq_len,
                                                       const Expression& q_offsets,
                                                       const Expression& kv_offsets,
                                                       const Expression& dropout_seed,
                                                       const Expression& dropout_offset,
                                                       AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(q,
                                         k,
                                         v,
                                         &bias,
                                         &q_seq_len,
                                         &kv_seq_len,
                                         &q_offsets,
                                         &kv_offsets,
                                         nullptr,
                                         nullptr,
                                         &dropout_seed,
                                         &dropout_offset,
                                         std::move(options));
}

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 const Expression& dropout_seed,
                                                 const Expression& dropout_offset,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, nullptr, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& bias,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, &bias, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& bias,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 const Expression& dropout_seed,
                                                 const Expression& dropout_offset,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(
        q, k, v, &bias, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::conv2d(const Expression& input,
                              const Expression& filter,
                              int32_t stride_h,
                              int32_t stride_w,
                              int32_t pad_h,
                              int32_t pad_w,
                              std::optional<DataType> compute_dtype,
                              std::optional<DataType> output_dtype) {
    if (stride_h <= 0 || stride_w <= 0) {
        throw std::runtime_error("conv2d stride must be positive.");
    }
    if (pad_h < 0 || pad_w < 0) {
        throw std::runtime_error("conv2d padding must be non-negative.");
    }

    Expression out = binaryOp(input, filter, ExprOp::CONV2D);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.conv_stride_h = stride_h;
    node.conv_stride_w = stride_w;
    node.conv_pad_h = pad_h;
    node.conv_pad_w = pad_w;
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }
    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    return out;
}

Expression Expression::conv3d(const Expression& input,
                              const Expression& filter,
                              int32_t stride_d,
                              int32_t stride_h,
                              int32_t stride_w,
                              int32_t pad_d,
                              int32_t pad_h,
                              int32_t pad_w,
                              std::optional<DataType> compute_dtype,
                              std::optional<DataType> output_dtype) {
    if (stride_d <= 0 || stride_h <= 0 || stride_w <= 0) {
        throw std::runtime_error("conv3d stride must be positive.");
    }
    if (pad_d < 0 || pad_h < 0 || pad_w < 0) {
        throw std::runtime_error("conv3d padding must be non-negative.");
    }

    Expression out = binaryOp(input, filter, ExprOp::CONV3D);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.conv_stride_d = stride_d;
    node.conv_stride_h = stride_h;
    node.conv_stride_w = stride_w;
    node.conv_pad_d = pad_d;
    node.conv_pad_h = pad_h;
    node.conv_pad_w = pad_w;
    if (compute_dtype.has_value()) {
        node.compute_dtype = compute_dtype.value();
    }
    if (output_dtype.has_value()) {
        node.output_dtype = output_dtype.value();
    }
    return out;
}

// Reductions
DataType validate_reduction_compute_type(std::optional<DataType> compute_dtype) {
    const DataType requested_compute_dtype = compute_dtype.has_value() ? compute_dtype.value() : DataType::FP32;
    return toSupportedComputeDType(ExprOp::REDUCE_SUM, requested_compute_dtype);
}

static bool isArgReductionOp(ExprOp op) { return op == ExprOp::REDUCE_ARGMIN || op == ExprOp::REDUCE_ARGMAX; }

Expression Expression::reduction(ExprOp op,
                                 const std::vector<uint64_t>& reduction_axes,
                                 const std::vector<uint64_t>& squeeze_axes,
                                 std::optional<DataType> compute_dtype) const {
    Expression out = unaryOp(*this, op);

    out.expr->nodes[out.nodeIndex].reduction_axes = reduction_axes;
    out.expr->nodes[out.nodeIndex].squeeze_axes = squeeze_axes;
    out.expr->nodes[out.nodeIndex].compute_dtype = validate_reduction_compute_type(compute_dtype);
    if (isArgReductionOp(op)) {
        out.expr->nodes[out.nodeIndex].output_dtype = DataType::UINT32;
        out.expr->nodes[out.nodeIndex].backward_output_dtype = DataType::UINT32;
        out.expr->nodes[out.nodeIndex].backward_compute_dtype = DataType::FP32;
    }
    return out;
}

Expression Expression::reduce_sum(const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_SUM, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_prod(const std::vector<uint64_t>& reduction_axes,
                                   const std::vector<uint64_t>& squeeze_axes,
                                   std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_PROD, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_min(const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_MIN, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_max(const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_MAX, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::argmin(const std::vector<uint64_t>& reduction_axes,
                              const std::vector<uint64_t>& squeeze_axes,
                              std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_ARGMIN, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::argmax(const std::vector<uint64_t>& reduction_axes,
                              const std::vector<uint64_t>& squeeze_axes,
                              std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_ARGMAX, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_mean(const std::vector<uint64_t>& reduction_axes,
                                   const std::vector<uint64_t>& squeeze_axes,
                                   std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_AVG, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_norm1(const std::vector<uint64_t>& reduction_axes,
                                    const std::vector<uint64_t>& squeeze_axes,
                                    std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_NORM1, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_norm2(const std::vector<uint64_t>& reduction_axes,
                                    const std::vector<uint64_t>& squeeze_axes,
                                    std::optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_NORM2, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::withDTypes(std::optional<DataType> compute_dtype, std::optional<DataType> output_dtype) const {
    if (!expr)
        throw std::runtime_error("Cannot override dtypes on an empty expression");
    if (nodeIndex >= expr->nodes.size())
        throw std::runtime_error("Cannot override dtypes on an invalid expression node");

    auto out = std::make_shared<PhysicalExpression>();
    out->inputs = expr->inputs;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    uint32_t newRootIndex = cloneSubtree(*expr, nodeIndex, *out, oldToNew);
    out->output_node = newRootIndex;

    ExprNode& root = out->nodes[newRootIndex];
    if (compute_dtype.has_value()) {
        root.compute_dtype = compute_dtype.value();
    }
    if (output_dtype.has_value()) {
        root.output_dtype = output_dtype.value();
    }

    return Expression(out, newRootIndex);
}

Expression Expression::withComputeDType(DataType compute_dtype) const { return withDTypes(compute_dtype, std::nullopt); }

Expression Expression::withOutputDType(DataType output_dtype) const { return withDTypes(std::nullopt, output_dtype); }

Expression Expression::ln() const { return unaryOp(*this, ExprOp::LN); }
Expression Expression::log2() const { return unaryOp(*this, ExprOp::LOG2); }
Expression Expression::log10() const { return unaryOp(*this, ExprOp::LOG10); }
Expression Expression::log(double base) const {
    if (base <= 0.0f || base == 1.0f) {
        throw std::runtime_error("log base must be positive and not equal to 1, received base = " + std::to_string(base));
    }
    return this->ln() / Expression::constantScalar(std::log(base));
}

Expression Expression::min(const Expression& other) const { return binaryOp(*this, other, ExprOp::MIN); }
Expression Expression::max(const Expression& other) const { return binaryOp(*this, other, ExprOp::MAX); }

Expression Expression::clamp(const Expression& lower_bound, const Expression& upper_bound) const {
    return this->max(lower_bound).min(upper_bound);
}

Expression Expression::clamp(double lower_bound, double upper_bound) const {
    if (lower_bound > upper_bound) {
        throw std::invalid_argument("Expression::clamp requires lower_bound <= upper_bound.");
    }
    return clamp(Expression::constantScalar(lower_bound), Expression::constantScalar(upper_bound));
}

Expression Expression::clamp(const Expression& input, const Expression& lower_bound, const Expression& upper_bound) {
    return input.clamp(lower_bound, upper_bound);
}

Expression Expression::clamp(const Expression& input, double lower_bound, double upper_bound) {
    return input.clamp(lower_bound, upper_bound);
}

Expression Expression::dotProduct(const Expression& other, std::optional<DataType> compute_dtype) const {
    return dotProduct(*this, other, compute_dtype);
}

Expression Expression::dotProduct(const Expression& lhs, const Expression& rhs, std::optional<DataType> compute_dtype) {
    return (lhs * rhs).reduce_sum(/*reduction_axes=*/{}, /*squeeze_axes=*/{UINT64_MAX}, compute_dtype);
}

Expression Expression::outerProduct(const Expression& other,
                                    std::optional<DataType> compute_dtype,
                                    std::optional<DataType> output_dtype) const {
    return outerProduct(*this, other, compute_dtype, output_dtype);
}

Expression Expression::outerProduct(const Expression& lhs,
                                    const Expression& rhs,
                                    std::optional<DataType> compute_dtype,
                                    std::optional<DataType> output_dtype) {
    return matmul(lhs.unsqueeze({1}), rhs.unsqueeze({0}), false, false, compute_dtype, output_dtype);
}

// e^x_i
Expression Expression::exp() const { return unaryOp(*this, ExprOp::EXP); }
// 2^x_i
Expression Expression::exp2() const { return unaryOp(*this, ExprOp::EXP2); }
Expression Expression::exp10() const { return unaryOp(*this, ExprOp::EXP10); }
// Can also use Expression::scalar(s).pow(x) for s^x_i

uint32_t PhysicalExpression::getOrCreateInputSlot(const std::string& name, NamedInput::Kind kind) {
    for (const NamedInput& input : inputs) {
        if (input.name == name) {
            if (input.kind != kind) {
                throw std::runtime_error("Input kind mismatch for input: " + name);
            }
            return input.slot;
        }
    }

    const uint32_t slot = static_cast<uint32_t>(inputs.size());
    inputs.push_back(NamedInput{name, slot, kind});
    return slot;
}

Outputs Expression::outputs(const std::vector<std::pair<std::string, Expression>>& named_exprs) {
    if (named_exprs.empty()) {
        throw std::runtime_error("Expression::outputs requires at least one named output.");
    }

    auto merged = std::make_shared<PhysicalExpression>();
    std::vector<NamedOutput> outputs;
    outputs.reserve(named_exprs.size());

    std::unordered_set<std::string> seen_names;
    std::unordered_map<std::string, uint32_t> mergedInputSlotsByName;

    for (const auto& [name, expr] : named_exprs) {
        if (name.empty()) {
            throw std::runtime_error("Output name cannot be empty.");
        }

        if (!seen_names.insert(name).second) {
            throw std::runtime_error("Duplicate output name: " + name);
        }

        if (!expr.expr) {
            throw std::runtime_error("Output expression has no backing PhysicalExpression.");
        }

        if (expr.nodeIndex == UINT32_MAX) {
            throw std::runtime_error("Output expression has invalid node index.");
        }

        if (expr.nodeIndex >= expr.expr->nodes.size()) {
            throw std::runtime_error("Output expression node index is out of range.");
        }

        std::unordered_map<uint32_t, uint32_t> oldToNew;
        uint32_t mergedRoot = cloneSubtreeWithMergedInputs(*expr.expr, expr.nodeIndex, *merged, oldToNew, mergedInputSlotsByName);

        outputs.push_back(NamedOutput{
            .name = name,
            .node_idx = mergedRoot,
        });
    }

    return Outputs(std::move(merged), std::move(outputs));
}

Outputs Expression::outputs(std::initializer_list<std::pair<std::string, Expression>> named_exprs) {
    return outputs(std::vector<std::pair<std::string, Expression>>(named_exprs));
}

}  // namespace ThorImplementation
