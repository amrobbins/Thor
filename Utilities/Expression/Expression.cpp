#include "Utilities/Expression/Expression.h"
#include <optional>
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <functional>
#include <sstream>
#include "DeepLearning/Implementation/ThorError.h"

using DataType = ThorImplementation::TensorDescriptor::DataType;
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
        case ExprOp::NEG:
            return "neg";
        case ExprOp::ABS:
            return "abs";
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
        {"neg", ExprOp::NEG},
        {"abs", ExprOp::ABS},
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
        {"attention", ExprOp::ATTENTION},
        {"attention_backward_q", ExprOp::ATTENTION_BACKWARD_Q},
        {"attention_backward_k", ExprOp::ATTENTION_BACKWARD_K},
        {"attention_backward_v", ExprOp::ATTENTION_BACKWARD_V},
        {"attention_backward_bias", ExprOp::ATTENTION_BACKWARD_BIAS},
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
    j["attention_seq_len_q_node"] = node.attention_seq_len_q_node;
    j["attention_seq_len_kv_node"] = node.attention_seq_len_kv_node;
    j["attention_ragged_offset_q_node"] = node.attention_ragged_offset_q_node;
    j["attention_ragged_offset_kv_node"] = node.attention_ragged_offset_kv_node;
    j["attention_page_table_k_node"] = node.attention_page_table_k_node;
    j["attention_page_table_v_node"] = node.attention_page_table_v_node;
    j["attention_dropout_seed_node"] = node.attention_dropout_seed_node;
    j["attention_dropout_offset_node"] = node.attention_dropout_offset_node;
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
    setOptionalDTypeJson(j, "input_tensor_dtype", node.input_tensor_dtype);
    setOptionalDTypeJson(j, "output_dtype", node.output_dtype);
    setOptionalDTypeJson(j, "compute_dtype", node.compute_dtype);
    setOptionalDTypeJson(j, "backward_output_dtype", node.backward_output_dtype);
    setOptionalDTypeJson(j, "backward_compute_dtype", node.backward_compute_dtype);
    j["reduction_axes"] = node.reduction_axes;
    j["reshape_dims"] = node.reshape_dims;
    j["squeeze_axes"] = node.squeeze_axes;
    j["unsqueeze_axes"] = node.unsqueeze_axes;
    j["fill_dims"] = node.fill_dims;
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
    node.attention_seq_len_q_node = j.value("attention_seq_len_q_node", UINT32_MAX);
    node.attention_seq_len_kv_node = j.value("attention_seq_len_kv_node", UINT32_MAX);
    node.attention_ragged_offset_q_node = j.value("attention_ragged_offset_q_node", UINT32_MAX);
    node.attention_ragged_offset_kv_node = j.value("attention_ragged_offset_kv_node", UINT32_MAX);
    node.attention_page_table_k_node = j.value("attention_page_table_k_node", UINT32_MAX);
    node.attention_page_table_v_node = j.value("attention_page_table_v_node", UINT32_MAX);
    node.attention_dropout_seed_node = j.value("attention_dropout_seed_node", UINT32_MAX);
    node.attention_dropout_offset_node = j.value("attention_dropout_offset_node", UINT32_MAX);
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
    parseOptionalDTypeField(j, "input_tensor_dtype", node.input_tensor_dtype);
    parseOptionalDTypeField(j, "output_dtype", node.output_dtype);
    parseOptionalDTypeField(j, "compute_dtype", node.compute_dtype);
    parseOptionalDTypeField(j, "backward_output_dtype", node.backward_output_dtype);
    parseOptionalDTypeField(j, "backward_compute_dtype", node.backward_compute_dtype);
    node.reduction_axes = j.value("reduction_axes", std::vector<uint64_t>{});
    node.reshape_dims = j.value("reshape_dims", std::vector<uint64_t>{});
    node.squeeze_axes = j.value("squeeze_axes", std::vector<uint64_t>{});
    node.unsqueeze_axes = j.value("unsqueeze_axes", std::vector<uint64_t>{});
    node.fill_dims = j.value("fill_dims", std::vector<uint64_t>{});
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

bool isCommutative(ExprOp op) { return op == ExprOp::ADD || op == ExprOp::MUL || op == ExprOp::MIN || op == ExprOp::MAX; }

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
        case ExprOp::NEG:
            return "NEG";
        case ExprOp::ABS:
            return "ABS";
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
                  ";originalMax=" + std::to_string(n.rope_original_max_position_embeddings) + ")";
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
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::MATMUL:
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
            out += ")";
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
                  ";tB=" + std::to_string(n.transpose_rhs ? 1 : 0) + ";tC=" + std::to_string(n.transpose_aux ? 1 : 0) + ")";
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
        if ((node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
             node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) &&
            node.attention_use_bias) {
            validateNodeIndex(node.beta_node, "attention backward bias");
            if (node.beta_node >= node_index_u32) {
                throw std::runtime_error("ExpressionDefinition ATTENTION_BACKWARD bias node must reference an earlier node.");
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

ExpressionDefinition ExpressionDefinition::deserialize(const json& j) {
    if (j.at("type").get<std::string>() != "thor.expression") {
        throw std::runtime_error("ExpressionDefinition::deserialize type mismatch: " + j.at("type").get<std::string>());
    }
    const int schema_version = j.at("schema_version").get<int>();
    if (schema_version != 1) {
        throw std::runtime_error("Unsupported thor.expression schema_version: " + std::to_string(schema_version));
    }

    ExpressionDefinition definition;
    definition.outputs.expr = std::make_shared<PhysicalExpression>();
    definition.expected_input_names = j.value("expected_input_names", std::vector<std::string>{});
    definition.expected_output_names = j.value("expected_output_names", std::vector<std::string>{});
    definition.canonical_hash = j.value("canonical_hash", std::string{});

    for (const json& input_json : j.at("inputs")) {
        definition.outputs.expr->inputs.push_back(NamedInput{
            .name = input_json.at("name").get<std::string>(),
            .slot = input_json.at("slot").get<uint32_t>(),
            .kind = namedInputKindFromString(input_json.at("kind").get<std::string>()),
        });
    }

    for (const json& node_json : j.at("nodes")) {
        definition.outputs.expr->nodes.push_back(exprNodeFromJson(node_json));
    }

    for (const json& output_json : j.at("outputs")) {
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
        case ExprOp::ROPE:
        case ExprOp::SOFTMAX:
        case ExprOp::TRANSPOSE:
        case ExprOp::RESHAPE:
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
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::MATMUL:
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
            return true;
        default:
            return false;
    }
}

namespace {

uint32_t cloneSubtree(const PhysicalExpression& src,
                      uint32_t srcNodeIndex,
                      PhysicalExpression& dst,
                      std::unordered_map<uint32_t, uint32_t>& oldToNew) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end())
        return it->second;

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    ExprNode newNode = srcNode;

    if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for binary op");
        if (srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing rhs for binary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
        newNode.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for ternary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
        newNode.aux = cloneSubtree(src, srcNode.aux, dst, oldToNew);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node = cloneSubtree(src, srcNode.alpha_node, dst, oldToNew);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneSubtree(src, srcNode.beta_node, dst, oldToNew);
        }
        if (srcNode.attention_use_padding_mask) {
            if (srcNode.attention_seq_len_q_node == UINT32_MAX || srcNode.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing padding-mask sequence length node while cloning.");
            }
            newNode.attention_seq_len_q_node = cloneSubtree(src, srcNode.attention_seq_len_q_node, dst, oldToNew);
            newNode.attention_seq_len_kv_node = cloneSubtree(src, srcNode.attention_seq_len_kv_node, dst, oldToNew);
        }
        if (srcNode.attention_use_ragged_offsets) {
            if (srcNode.attention_ragged_offset_q_node == UINT32_MAX || srcNode.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing ragged offset node while cloning.");
            }
            newNode.attention_ragged_offset_q_node = cloneSubtree(src, srcNode.attention_ragged_offset_q_node, dst, oldToNew);
            newNode.attention_ragged_offset_kv_node = cloneSubtree(src, srcNode.attention_ragged_offset_kv_node, dst, oldToNew);
        }
        if (srcNode.attention_use_paged_kv_cache) {
            if (srcNode.attention_page_table_k_node == UINT32_MAX || srcNode.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing paged KV page-table nodes while cloning.");
            }
            newNode.attention_page_table_k_node = cloneSubtree(src, srcNode.attention_page_table_k_node, dst, oldToNew);
            newNode.attention_page_table_v_node = cloneSubtree(src, srcNode.attention_page_table_v_node, dst, oldToNew);
        }
        if (srcNode.attention_dropout_probability > 0.0f) {
            if (srcNode.attention_dropout_seed_node == UINT32_MAX || srcNode.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing dropout seed/offset node while cloning.");
            }
            newNode.attention_dropout_seed_node = cloneSubtree(src, srcNode.attention_dropout_seed_node, dst, oldToNew);
            newNode.attention_dropout_offset_node = cloneSubtree(src, srcNode.attention_dropout_offset_node, dst, oldToNew);
        }
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

uint32_t cloneSubtreeWithMergedInputs(const PhysicalExpression& src,
                                      uint32_t srcNodeIndex,
                                      PhysicalExpression& dst,
                                      std::unordered_map<uint32_t, uint32_t>& oldToNew,
                                      std::unordered_map<std::string, uint32_t>& dstInputSlotsByName) {
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
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for binary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = cloneSubtreeWithMergedInputs(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName);
        newNode.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for ternary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = cloneSubtreeWithMergedInputs(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName);
        newNode.aux = cloneSubtreeWithMergedInputs(src, srcNode.aux, dst, oldToNew, dstInputSlotsByName);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node = cloneSubtreeWithMergedInputs(src, srcNode.alpha_node, dst, oldToNew, dstInputSlotsByName);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneSubtreeWithMergedInputs(src, srcNode.beta_node, dst, oldToNew, dstInputSlotsByName);
        }
        if (srcNode.attention_use_padding_mask) {
            if (srcNode.attention_seq_len_q_node == UINT32_MAX || srcNode.attention_seq_len_kv_node == UINT32_MAX) {
                throw std::runtime_error(
                    "Malformed attention expression: missing padding-mask sequence length node while merging outputs.");
            }
            newNode.attention_seq_len_q_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_seq_len_q_node, dst, oldToNew, dstInputSlotsByName);
            newNode.attention_seq_len_kv_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_seq_len_kv_node, dst, oldToNew, dstInputSlotsByName);
        }
        if (srcNode.attention_use_ragged_offsets) {
            if (srcNode.attention_ragged_offset_q_node == UINT32_MAX || srcNode.attention_ragged_offset_kv_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing ragged offset node while merging outputs.");
            }
            newNode.attention_ragged_offset_q_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_ragged_offset_q_node, dst, oldToNew, dstInputSlotsByName);
            newNode.attention_ragged_offset_kv_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_ragged_offset_kv_node, dst, oldToNew, dstInputSlotsByName);
        }
        if (srcNode.attention_use_paged_kv_cache) {
            if (srcNode.attention_page_table_k_node == UINT32_MAX || srcNode.attention_page_table_v_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing paged KV page-table nodes while cloning merged inputs.");
            }
            newNode.attention_page_table_k_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_page_table_k_node, dst, oldToNew, dstInputSlotsByName);
            newNode.attention_page_table_v_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_page_table_v_node, dst, oldToNew, dstInputSlotsByName);
        }
        if (srcNode.attention_dropout_probability > 0.0f) {
            if (srcNode.attention_dropout_seed_node == UINT32_MAX || srcNode.attention_dropout_offset_node == UINT32_MAX) {
                throw std::runtime_error("Malformed attention expression: missing dropout seed/offset node while merging outputs.");
            }
            newNode.attention_dropout_seed_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_dropout_seed_node, dst, oldToNew, dstInputSlotsByName);
            newNode.attention_dropout_offset_node =
                cloneSubtreeWithMergedInputs(src, srcNode.attention_dropout_offset_node, dst, oldToNew, dstInputSlotsByName);
        }
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

    std::unordered_map<uint32_t, uint32_t> lhsMap;
    std::unordered_map<uint32_t, uint32_t> rhsMap;

    uint32_t newLhsIndex = cloneSubtree(*lhsExpr.expr, lhsExpr.nodeIndex, *out, lhsMap);
    uint32_t newRhsIndex = cloneSubtree(*rhsExpr.expr, rhsExpr.nodeIndex, *out, rhsMap);

    remapClonedInputSlots(*lhsExpr.expr, lhsMap, mergedInputs.lhsSlotRemap, *out);
    remapClonedInputSlots(*rhsExpr.expr, rhsMap, mergedInputs.rhsSlotRemap, *out);

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

    std::unordered_map<uint32_t, uint32_t> lhsMap;
    std::unordered_map<uint32_t, uint32_t> rhsMap;
    std::unordered_map<uint32_t, uint32_t> auxMap;

    uint32_t newLhsIndex = cloneSubtree(*lhsExpr.expr, lhsExpr.nodeIndex, *out, lhsMap);
    uint32_t newRhsIndex = cloneSubtree(*rhsExpr.expr, rhsExpr.nodeIndex, *out, rhsMap);
    uint32_t newAuxIndex = cloneSubtree(*auxExpr.expr, auxExpr.nodeIndex, *out, auxMap);

    std::vector<uint32_t> lhsSlotRemap(lhsExpr.expr->inputs.size());
    for (size_t i = 0; i < lhsExpr.expr->inputs.size(); ++i)
        lhsSlotRemap[i] = mergedByName.at(lhsExpr.expr->inputs[i].name);
    std::vector<uint32_t> rhsSlotRemap(rhsExpr.expr->inputs.size());
    for (size_t i = 0; i < rhsExpr.expr->inputs.size(); ++i)
        rhsSlotRemap[i] = mergedByName.at(rhsExpr.expr->inputs[i].name);
    std::vector<uint32_t> auxSlotRemap(auxExpr.expr->inputs.size());
    for (size_t i = 0; i < auxExpr.expr->inputs.size(); ++i)
        auxSlotRemap[i] = mergedByName.at(auxExpr.expr->inputs[i].name);

    remapClonedInputSlots(*lhsExpr.expr, lhsMap, lhsSlotRemap, *out);
    remapClonedInputSlots(*rhsExpr.expr, rhsMap, rhsSlotRemap, *out);
    remapClonedInputSlots(*auxExpr.expr, auxMap, auxSlotRemap, *out);

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

    std::unordered_map<uint32_t, uint32_t> lhsMap;
    std::unordered_map<uint32_t, uint32_t> rhsMap;
    std::unordered_map<uint32_t, uint32_t> auxMap;
    std::unordered_map<uint32_t, uint32_t> fourthMap;

    uint32_t newLhsIndex = cloneSubtree(*lhsExpr.expr, lhsExpr.nodeIndex, *out, lhsMap);
    uint32_t newRhsIndex = cloneSubtree(*rhsExpr.expr, rhsExpr.nodeIndex, *out, rhsMap);
    uint32_t newAuxIndex = cloneSubtree(*auxExpr.expr, auxExpr.nodeIndex, *out, auxMap);
    uint32_t newFourthIndex = cloneSubtree(*fourthExpr.expr, fourthExpr.nodeIndex, *out, fourthMap);

    auto remap_for = [&](const PhysicalExpression& src, std::unordered_map<uint32_t, uint32_t>& map) {
        std::vector<uint32_t> slotRemap(src.inputs.size());
        for (size_t i = 0; i < src.inputs.size(); ++i)
            slotRemap[i] = mergedByName.at(src.inputs[i].name);
        remapClonedInputSlots(src, map, slotRemap, *out);
    };
    remap_for(*lhsExpr.expr, lhsMap);
    remap_for(*rhsExpr.expr, rhsMap);
    remap_for(*auxExpr.expr, auxMap);
    remap_for(*fourthExpr.expr, fourthMap);

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

Expression Expression::unaryOp(const Expression& inputExpr, ExprOp op) {
    if (!inputExpr.expr)
        throw std::runtime_error("Cannot apply unary op to empty expression");

    auto out = std::make_shared<PhysicalExpression>();
    out->inputs = inputExpr.expr->inputs;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    uint32_t newLhsIndex = cloneSubtree(*inputExpr.expr, inputExpr.nodeIndex, *out, oldToNew);

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
Expression Expression::operator-() const { return unaryOp(*this, ExprOp::NEG); }
Expression Expression::abs() const { return unaryOp(*this, ExprOp::ABS); }
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

static uint32_t cloneSubtreeIntoMergedExpression(const Expression& src_expr,
                                                 PhysicalExpression& dst,
                                                 std::unordered_map<uint32_t, uint32_t>& old_to_new,
                                                 std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name) {
    PhysicalExpression src = src_expr.expression();
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
    if (options.rotary_dim != 0 && ((options.rotary_dim & 1ULL) != 0ULL)) {
        throw std::runtime_error("RoPE rotary_dim must be even.");
    }
    if (options.scaling_kind == RotaryScalingKind::DynamicNTK && options.original_max_position_embeddings == 0) {
        throw std::runtime_error("Dynamic-NTK RoPE scaling requires original_max_position_embeddings.");
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
    if (options.output_dtype.has_value()) {
        node.output_dtype = options.output_dtype.value();
    }
    if (options.compute_dtype.has_value()) {
        node.compute_dtype = options.compute_dtype.value();
    }
    return out;
}

namespace {
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
    if (options.compute_dtype.has_value()) {
        node.compute_dtype = options.compute_dtype.value();
    }
    if (options.output_dtype.has_value()) {
        node.output_dtype = options.output_dtype.value();
    }
}

void validateAttentionOptions(const AttentionOptions& options, bool use_bias, bool use_ragged_offsets = false) {
    if (use_ragged_offsets &&
        (options.q_layout != AttentionTensorLayout::BSHD || options.k_layout != AttentionTensorLayout::BSHD ||
         options.v_layout != AttentionTensorLayout::BSHD || options.o_layout != AttentionTensorLayout::BSHD)) {
        throw std::runtime_error(
            "Ragged attention requires BSHD physical layouts for q/k/v/o because cuDNN ragged offsets index packed token-contiguous THD storage.");
    }
    if (options.diagonal_left_bound < 0 || options.diagonal_right_bound < 0) {
        throw std::runtime_error("Attention diagonal/sliding-window bounds must be non-negative.");
    }
    if (options.use_alibi_mask) {
        const bool uses_causal_diagonal = options.mask_kind == AttentionMaskKind::CausalTopLeft ||
                                          options.mask_kind == AttentionMaskKind::CausalBottomRight ||
                                          options.mask_kind == AttentionMaskKind::SlidingWindowTopLeft ||
                                          options.mask_kind == AttentionMaskKind::SlidingWindowBottomRight;
        if (!uses_causal_diagonal || options.diagonal_right_bound != 0) {
            throw std::runtime_error(
                "AttentionOptions::use_alibi_mask requires causal diagonal masking with diagonal_right_bound == 0.");
        }
    }
    if (options.dropout_probability < 0.0f || options.dropout_probability >= 1.0f) {
        throw std::runtime_error("AttentionOptions::dropout_probability must be in [0, 1).");
    }
    if (use_ragged_offsets && options.use_paged_kv_cache) {
        throw std::runtime_error("Ragged attention and paged KV cache are separate variable-length modes and cannot be combined.");
    }
    if (use_ragged_offsets && (use_bias || options.dropout_probability > 0.0f)) {
        throw std::runtime_error(
            "Ragged attention currently supports q/k/v plus q/kv sequence lengths and ragged offsets only; additive bias and dropout are disabled until their packed layouts are defined.");
    }
    if (options.use_paged_kv_cache && options.paged_kv_max_sequence_length <= 0) {
        throw std::runtime_error("Paged KV attention requires paged_kv_max_sequence_length > 0.");
    }
    if (options.use_paged_kv_cache && !options.use_padding_mask) {
        throw std::runtime_error("Paged KV attention requires q_seq_len and kv_seq_len padding-mask tensors.");
    }
    if (options.use_paged_kv_cache && use_bias) {
        throw std::runtime_error("Paged KV attention with additive bias is disabled until the paged-bias layout is defined.");
    }
    if (options.use_paged_kv_cache && options.dropout_probability > 0.0f) {
        throw std::runtime_error("Paged KV attention is inference-only and cannot currently be combined with dropout.");
    }
    if (options.mask_kind == AttentionMaskKind::CausalBottomRight &&
        (use_bias || options.use_alibi_mask || options.dropout_probability > 0.0f)) {
        throw std::runtime_error(
            "AttentionMaskKind::CausalBottomRight is currently supported only without additive bias, ALiBi, or dropout in the cuDNN primary SDPA path.");
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
                                                     AttentionOptions options) {
    if (!q.expr || !k.expr || !v.expr || (bias != nullptr && !bias->expr) || (q_seq_len != nullptr && !q_seq_len->expr) ||
        (kv_seq_len != nullptr && !kv_seq_len->expr) || (q_ragged_offsets != nullptr && !q_ragged_offsets->expr) ||
        (kv_ragged_offsets != nullptr && !kv_ragged_offsets->expr) || (page_table_k != nullptr && !page_table_k->expr) ||
        (page_table_v != nullptr && !page_table_v->expr) || (dropout_seed != nullptr && !dropout_seed->expr) ||
        (dropout_offset != nullptr && !dropout_offset->expr)) {
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
            "Ragged attention requires q_seq_len and kv_seq_len expressions in addition to q/kv ragged offsets because cuDNN THD uses sequence lengths for padding-mask semantics.");
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
    applyAttentionOptions(node, options, bias != nullptr);

    const uint32_t new_index = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(std::move(node));
    out->output_node = new_index;
    return Expression(out, new_index);
}

Expression Expression::scaledDotProductAttention(const Expression& q, const Expression& k, const Expression& v, AttentionOptions options) {
    return attentionWithOptionalMetadata(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionWithDropout(const Expression& q,
                                                            const Expression& k,
                                                            const Expression& v,
                                                            const Expression& dropout_seed,
                                                            const Expression& dropout_offset,
                                                            AttentionOptions options) {
    return attentionWithOptionalMetadata(q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::scaledDotProductAttention(
    const Expression& q, const Expression& k, const Expression& v, const Expression& bias, AttentionOptions options) {
    return attentionWithOptionalMetadata(q, k, v, &bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
}

Expression Expression::scaledDotProductAttentionWithDropout(const Expression& q,
                                                            const Expression& k,
                                                            const Expression& v,
                                                            const Expression& bias,
                                                            const Expression& dropout_seed,
                                                            const Expression& dropout_offset,
                                                            AttentionOptions options) {
    return attentionWithOptionalMetadata(q, k, v, &bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(q, k, v, nullptr, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
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

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 const Expression& dropout_seed,
                                                 const Expression& dropout_offset,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(q, k, v, nullptr, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
}

Expression Expression::scaledDotProductAttention(const Expression& q,
                                                 const Expression& k,
                                                 const Expression& v,
                                                 const Expression& bias,
                                                 const Expression& q_seq_len,
                                                 const Expression& kv_seq_len,
                                                 AttentionOptions options) {
    options.use_padding_mask = true;
    return attentionWithOptionalMetadata(q, k, v, &bias, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, std::move(options));
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
    return attentionWithOptionalMetadata(q, k, v, &bias, &q_seq_len, &kv_seq_len, nullptr, nullptr, nullptr, nullptr, &dropout_seed, &dropout_offset, std::move(options));
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
