#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include "CudaSourceEmitter.h"

#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

namespace ThorImplementation {

namespace {

bool experimentalCudnnAttentionSupportSurfaceProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE");
    return value != nullptr && std::string_view(value) == "1";
}

bool experimentalCudnnRaggedBiasBackwardProbeEnabled() {
    const char* value = std::getenv("THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD");
    return (value != nullptr && std::string_view(value) == "1") || experimentalCudnnAttentionSupportSurfaceProbeEnabled();
}

}  // namespace


static bool isThorCudnnConvolutionFloatingDType(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

static const char* thorCudnnConvolutionFloatingDTypesMessage() {
    return "supported Thor/cuDNN convolution floating dtypes are FP8_E4M3, FP8_E5M2, FP16, BF16, and FP32";
}

static bool convolutionComputeDTypeIsCompatibleWithTensorDTypes(const std::vector<DataType>& tensor_dtypes, DataType compute_dtype) {
    bool requires_fp32_compute = false;
    for (DataType dtype : tensor_dtypes) {
        if (dtype == DataType::FP32 || dtype == DataType::BF16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
            requires_fp32_compute = true;
            break;
        }
    }

    if (requires_fp32_compute) {
        return compute_dtype == DataType::FP32;
    }
    return compute_dtype == DataType::FP16 || compute_dtype == DataType::FP32;
}

#define NVRTC_CHECK(call)                                                                                                      \
    do {                                                                                                                       \
        nvrtcResult err__ = (call);                                                                                            \
        if (err__ != NVRTC_SUCCESS) {                                                                                          \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #call + " failed with " + \
                                     nvrtcGetErrorString(err__) + " (" + std::to_string(static_cast<int>(err__)) + ")");       \
        }                                                                                                                      \
    } while (0)

static std::string getNvrtcProgramLog(nvrtcProgram prog) {
    size_t log_size = 0;
    nvrtcResult size_res = nvrtcGetProgramLogSize(prog, &log_size);
    if (size_res != NVRTC_SUCCESS) {
        return std::string("<failed to query NVRTC log size: ") + nvrtcGetErrorString(size_res) + ">";
    }

    if (log_size <= 1) {
        return "";
    }

    std::string log(log_size, '\0');
    nvrtcResult log_res = nvrtcGetProgramLog(prog, log.data());
    if (log_res != NVRTC_SUCCESS) {
        return std::string("<failed to fetch NVRTC log: ") + nvrtcGetErrorString(log_res) + ">";
    }

    return log;
}

static nvrtcResult nvrtcCompileProgramChecked(
    nvrtcProgram prog, int num_options, const char* const* options, const char* call_text, const char* file, int line) {
    nvrtcResult res = nvrtcCompileProgram(prog, num_options, options);
    if (res == NVRTC_SUCCESS) {
        return res;
    }

    const std::string log = getNvrtcProgramLog(prog);

    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call_text + " failed with " +
                             nvrtcGetErrorString(res) + " (" + std::to_string(static_cast<int>(res)) + ")" +
                             (log.empty() ? "" : "\n" + log));
}

#define NVRTC_COMPILE_CHECK(prog, num_options, options) \
    nvrtcCompileProgramChecked(                         \
        (prog), (num_options), (options), "nvrtcCompileProgram(" #prog ", " #num_options ", " #options ")", __FILE__, __LINE__)

static LruCacheThreadSafe<EquationCacheKey, shared_ptr<CompiledEquation>> compiledEquationCache(10'000);

static shared_ptr<CompiledEquation> cacheLookup(const EquationCacheKey& key) {
    optional<shared_ptr<CompiledEquation>> hit = compiledEquationCache.get(key);
    if (hit.has_value()) {
        return hit.value();
    }
    return nullptr;
}

static void cacheInsert(const EquationCacheKey& key, shared_ptr<CompiledEquation>& compiledEquation) {
    compiledEquationCache.put(key, compiledEquation);
}

// static unordered_map<std::string, shared_ptr<CompiledEquation>> specializedBroadcastCache;
static LruCacheThreadSafe<std::string, shared_ptr<CompiledEquation>> specializedBroadcastCache(10'000);

static std::string makeSpecializedBroadcastCacheKey(const std::string& cuda_src, const EquationSignature& sig) {
    std::string key;
    key.reserve(cuda_src.size() + 128);
    key += "|sm=" + std::to_string(sig.sm_major) + std::to_string(sig.sm_minor);
    key += "|dev=" + std::to_string(sig.device_num);
    key += "|fast=" + std::to_string(sig.use_fast_math ? 1 : 0);
    key += "|src=";
    key += cuda_src;
    return key;
}

#ifndef THOR_CUDA_CCCL_INCLUDE_DIR
#define THOR_CUDA_CCCL_INCLUDE_DIR ""
#endif

#ifndef THOR_NVRTC_BUNDLED_HEADERS_DIR
#define THOR_NVRTC_BUNDLED_HEADERS_DIR ""
#endif

string EquationCompiler::getCudaIncludeDir() {
    if (const char* p = std::getenv("THOR_CUDA_INCLUDE_DIR")) {
        if (*p)
            return std::string(p);
    }

    if (std::string(THOR_CUDA_INCLUDE_DIR).empty() == false) {
        return THOR_CUDA_INCLUDE_DIR;  // compile-time fallback from CMake
    }

    if (const char* p = std::getenv("CUDA_HOME")) {
        return std::string(p) + "/include";
    }

    if (const char* p = std::getenv("CUDA_PATH")) {
        return std::string(p) + "/include";
    }

    return "";
}

vector<string> EquationCompiler::getCudaIncludeDirs() {
    vector<string> include_dirs;

    auto add_unique = [&include_dirs](const string& dir) {
        if (dir.empty())
            return;
        if (std::find(include_dirs.begin(), include_dirs.end(), dir) == include_dirs.end())
            include_dirs.push_back(dir);
    };

    add_unique(getCudaIncludeDir());

    if (const char* p = std::getenv("THOR_CUDA_CCCL_INCLUDE_DIR")) {
        if (*p)
            add_unique(std::string(p));
    }

    add_unique(THOR_CUDA_CCCL_INCLUDE_DIR);  // compile-time fallback from CMake

    return include_dirs;
}

string EquationCompiler::getNvrtcBundledHeadersDir() {
    if (const char* p = std::getenv("THOR_NVRTC_BUNDLED_HEADERS_DIR")) {
        if (*p)
            return std::string(p);
    }

    if (std::string(THOR_NVRTC_BUNDLED_HEADERS_DIR).empty() == false) {
        return THOR_NVRTC_BUNDLED_HEADERS_DIR;
    }

    return "";
}

static void ensureCudaContextCurrent(int device_num) {
    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, device_num));

    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&ctx));

    if (ctx == nullptr) {
        CUcontext primary;
        CU_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        CU_CHECK(cuCtxSetCurrent(primary));
        return;
    }

    CUdevice currentDevice;
    CU_CHECK(cuCtxGetDevice(&currentDevice));
    if ((int)currentDevice != device_num) {
        CUcontext primary;
        CU_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
        CU_CHECK(cuCtxSetCurrent(primary));
    }
}

struct StageNodeKey {
    ExprOp op = ExprOp::INPUT;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;
    uint32_t aux = UINT32_MAX;
    uint32_t input_slot = UINT32_MAX;
    uint64_t scalar_bits = 0;
    uint64_t alpha_bits = 0;
    uint64_t beta_bits = 0;
    uint32_t alpha_node = UINT32_MAX;
    uint32_t beta_node = UINT32_MAX;
    bool transpose_lhs = false;
    bool transpose_rhs = false;
    bool transpose_aux = false;
    int32_t conv_stride_d = 1;
    int32_t conv_stride_h = 1;
    int32_t conv_stride_w = 1;
    int32_t conv_pad_d = 0;
    int32_t conv_pad_h = 0;
    int32_t conv_pad_w = 0;
    uint32_t rope_sequence_axis = 2;
    uint32_t rope_head_dim_axis = 3;
    uint64_t rope_rotary_dim = 0;
    uint64_t rope_base_bits = 0;
    int64_t rope_position_offset = 0;
    bool rope_interleaved = false;
    bool rope_inverse = false;
    int32_t rope_scaling_kind = 0;
    uint64_t rope_scaling_factor_bits = 0;
    uint64_t rope_original_max_position_embeddings = 0;
    uint64_t rope_attention_factor_bits = 0;
    uint64_t rope_yarn_beta_fast_bits = 0;
    uint64_t rope_yarn_beta_slow_bits = 0;
    uint64_t rope_llama3_low_freq_factor_bits = 0;
    uint64_t rope_llama3_high_freq_factor_bits = 0;
    std::vector<uint64_t> rope_long_rope_short_factor_bits;
    std::vector<uint64_t> rope_long_rope_long_factor_bits;
    bool rope_allow_in_place_materialization = false;
    bool attention_use_bias = false;
    uint64_t attention_dropout_probability_bits = 0;
    int32_t input_tensor_dtype = -1;
    int32_t output_dtype = -1;
    int32_t compute_dtype = -1;
    int32_t backward_output_dtype = -1;
    int32_t backward_compute_dtype = -1;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    std::vector<uint64_t> unsqueeze_axes;
    std::vector<uint64_t> fill_dims;
    std::vector<uint64_t> view_dims;
    std::vector<uint64_t> view_strides;
    uint64_t view_element_offset = 0;
    uint64_t ragged_runtime_batch_size = 0;
    uint64_t ragged_runtime_max_active_values = 0;
    uint64_t ragged_runtime_elements_per_value = 1;

    bool operator==(const StageNodeKey& other) const = default;
};

struct StageNodeKeyHash {
    size_t operator()(const StageNodeKey& k) const noexcept {
        size_t h = std::hash<int>{}(static_cast<int>(k.op));
        hashCombine(h, std::hash<uint32_t>{}(k.lhs));
        hashCombine(h, std::hash<uint32_t>{}(k.rhs));
        hashCombine(h, std::hash<uint32_t>{}(k.aux));
        hashCombine(h, std::hash<uint32_t>{}(k.input_slot));
        hashCombine(h, std::hash<uint64_t>{}(k.scalar_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.alpha_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.beta_bits));
        hashCombine(h, std::hash<uint32_t>{}(k.alpha_node));
        hashCombine(h, std::hash<uint32_t>{}(k.beta_node));
        hashCombine(h, std::hash<bool>{}(k.transpose_lhs));
        hashCombine(h, std::hash<bool>{}(k.transpose_rhs));
        hashCombine(h, std::hash<bool>{}(k.transpose_aux));
        hashCombine(h, std::hash<int32_t>{}(k.conv_stride_d));
        hashCombine(h, std::hash<int32_t>{}(k.conv_stride_h));
        hashCombine(h, std::hash<int32_t>{}(k.conv_stride_w));
        hashCombine(h, std::hash<int32_t>{}(k.conv_pad_d));
        hashCombine(h, std::hash<int32_t>{}(k.conv_pad_h));
        hashCombine(h, std::hash<int32_t>{}(k.conv_pad_w));
        hashCombine(h, std::hash<uint32_t>{}(k.rope_sequence_axis));
        hashCombine(h, std::hash<uint32_t>{}(k.rope_head_dim_axis));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_rotary_dim));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_base_bits));
        hashCombine(h, std::hash<int64_t>{}(k.rope_position_offset));
        hashCombine(h, std::hash<bool>{}(k.rope_interleaved));
        hashCombine(h, std::hash<bool>{}(k.rope_inverse));
        hashCombine(h, std::hash<int32_t>{}(k.rope_scaling_kind));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_scaling_factor_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_original_max_position_embeddings));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_attention_factor_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_yarn_beta_fast_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_yarn_beta_slow_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_llama3_low_freq_factor_bits));
        hashCombine(h, std::hash<uint64_t>{}(k.rope_llama3_high_freq_factor_bits));
        hashCombine(h, std::hash<size_t>{}(k.rope_long_rope_short_factor_bits.size()));
        for (uint64_t factor_bits : k.rope_long_rope_short_factor_bits)
            hashCombine(h, std::hash<uint64_t>{}(factor_bits));
        hashCombine(h, std::hash<size_t>{}(k.rope_long_rope_long_factor_bits.size()));
        for (uint64_t factor_bits : k.rope_long_rope_long_factor_bits)
            hashCombine(h, std::hash<uint64_t>{}(factor_bits));
        hashCombine(h, std::hash<bool>{}(k.rope_allow_in_place_materialization));
        hashCombine(h, std::hash<bool>{}(k.attention_use_bias));
        hashCombine(h, std::hash<uint64_t>{}(k.attention_dropout_probability_bits));
        hashCombine(h, std::hash<int32_t>{}(k.input_tensor_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.output_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.compute_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.backward_output_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.backward_compute_dtype));
        hashCombine(h, std::hash<size_t>{}(k.reduction_axes.size()));
        for (uint64_t axis : k.reduction_axes)
            hashCombine(h, std::hash<uint64_t>{}(axis));
        hashCombine(h, std::hash<size_t>{}(k.squeeze_axes.size()));
        for (uint64_t axis : k.squeeze_axes)
            hashCombine(h, std::hash<uint64_t>{}(axis));
        hashCombine(h, std::hash<size_t>{}(k.unsqueeze_axes.size()));
        for (uint64_t axis : k.unsqueeze_axes)
            hashCombine(h, std::hash<uint64_t>{}(axis));
        hashCombine(h, std::hash<size_t>{}(k.fill_dims.size()));
        for (uint64_t dim : k.fill_dims)
            hashCombine(h, std::hash<uint64_t>{}(dim));
        hashCombine(h, std::hash<size_t>{}(k.view_dims.size()));
        for (uint64_t dim : k.view_dims)
            hashCombine(h, std::hash<uint64_t>{}(dim));
        hashCombine(h, std::hash<size_t>{}(k.view_strides.size()));
        for (uint64_t stride : k.view_strides)
            hashCombine(h, std::hash<uint64_t>{}(stride));
        hashCombine(h, std::hash<uint64_t>{}(k.view_element_offset));
        if (k.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
            hashCombine(h, std::hash<uint64_t>{}(k.ragged_runtime_batch_size));
            hashCombine(h, std::hash<uint64_t>{}(k.ragged_runtime_max_active_values));
            hashCombine(h, std::hash<uint64_t>{}(k.ragged_runtime_elements_per_value));
        }
        return h;
    }
};

static bool isCommutativeStageOp(ExprOp op) {
    return op == ExprOp::ADD || op == ExprOp::MUL || op == ExprOp::MIN || op == ExprOp::MAX || op == ExprOp::EQUAL ||
           op == ExprOp::NOT_EQUAL || op == ExprOp::LOGICAL_AND || op == ExprOp::LOGICAL_OR;
}

static uint64_t scalarBits(double x) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(x));
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int32_t optionalDTypeTag(const std::optional<DataType>& dtype) {
    return dtype.has_value() ? static_cast<int32_t>(dtype.value()) : -1;
}

static StageNodeKey makeStageNodeKey(const ExprNode& n) {
    StageNodeKey key;
    key.op = n.op;
    key.input_tensor_dtype = optionalDTypeTag(n.input_tensor_dtype);
    key.output_dtype = optionalDTypeTag(n.output_dtype);
    key.compute_dtype = optionalDTypeTag(n.compute_dtype);
    key.backward_output_dtype = optionalDTypeTag(n.backward_output_dtype);
    key.backward_compute_dtype = optionalDTypeTag(n.backward_compute_dtype);
    key.transpose_lhs = n.transpose_lhs;
    key.transpose_rhs = n.transpose_rhs;
    key.transpose_aux = n.transpose_aux;
    key.conv_stride_d = n.conv_stride_d;
    key.conv_stride_h = n.conv_stride_h;
    key.conv_stride_w = n.conv_stride_w;
    key.conv_pad_d = n.conv_pad_d;
    key.conv_pad_h = n.conv_pad_h;
    key.conv_pad_w = n.conv_pad_w;
    key.alpha_bits = scalarBits(n.alpha_fp);
    key.beta_bits = scalarBits(n.beta_fp);
    key.alpha_node = n.alpha_node;
    key.beta_node = n.beta_node;
    key.rope_sequence_axis = n.rope_sequence_axis;
    key.rope_head_dim_axis = n.rope_head_dim_axis;
    key.rope_rotary_dim = n.rope_rotary_dim;
    key.rope_base_bits = scalarBits(n.rope_base);
    key.rope_position_offset = n.rope_position_offset;
    key.rope_interleaved = n.rope_interleaved;
    key.rope_inverse = n.rope_inverse;
    key.rope_scaling_kind = static_cast<int32_t>(n.rope_scaling_kind);
    key.rope_scaling_factor_bits = scalarBits(n.rope_scaling_factor);
    key.rope_original_max_position_embeddings = n.rope_original_max_position_embeddings;
    key.rope_attention_factor_bits = scalarBits(n.rope_attention_factor);
    key.rope_yarn_beta_fast_bits = scalarBits(n.rope_yarn_beta_fast);
    key.rope_yarn_beta_slow_bits = scalarBits(n.rope_yarn_beta_slow);
    key.rope_llama3_low_freq_factor_bits = scalarBits(n.rope_llama3_low_freq_factor);
    key.rope_llama3_high_freq_factor_bits = scalarBits(n.rope_llama3_high_freq_factor);
    key.rope_long_rope_short_factor_bits.reserve(n.rope_long_rope_short_factors.size());
    for (double factor : n.rope_long_rope_short_factors)
        key.rope_long_rope_short_factor_bits.push_back(scalarBits(factor));
    key.rope_long_rope_long_factor_bits.reserve(n.rope_long_rope_long_factors.size());
    for (double factor : n.rope_long_rope_long_factors)
        key.rope_long_rope_long_factor_bits.push_back(scalarBits(factor));
    key.rope_allow_in_place_materialization = n.rope_allow_in_place_materialization;
    key.attention_use_bias = n.attention_use_bias;
    key.attention_dropout_probability_bits = scalarBits(n.attention_dropout_probability);
    if (n.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
        key.ragged_runtime_batch_size = n.ragged_runtime_batch_size;
        key.ragged_runtime_max_active_values = n.ragged_runtime_max_active_values;
        key.ragged_runtime_elements_per_value = n.ragged_runtime_elements_per_value;
    }

    switch (n.op) {
        case ExprOp::INPUT:
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            key.input_slot = n.input_slot;
            break;

        case ExprOp::SCALAR_FP:
            key.scalar_bits = scalarBits(n.scalar_fp);
            break;

        case ExprOp::FILL:
            key.scalar_bits = scalarBits(n.scalar_fp);
            key.fill_dims = n.fill_dims;
            break;

        default:
            if (!Expression::isLeafOp(n.op)) {
                key.lhs = n.lhs;
            }
            if (Expression::isBinaryOp(n.op)) {
                key.rhs = n.rhs;
                if (isCommutativeStageOp(n.op) && key.lhs > key.rhs) {
                    std::swap(key.lhs, key.rhs);
                }
            }
            if (Expression::isTernaryOp(n.op)) {
                key.rhs = n.rhs;
                key.aux = n.aux;
            }
            key.reduction_axes = n.reduction_axes;
            key.view_dims = n.view_dims;
            key.view_strides = n.view_strides;
            key.view_element_offset = n.view_element_offset;
            key.squeeze_axes = n.squeeze_axes;
            key.unsqueeze_axes = n.unsqueeze_axes;
            break;
    }

    return key;
}

static void deduplicateFusedStageExpr(PhysicalExpression& stage_expr, std::vector<CompiledStageOutput>& stage_outputs) {
    if (stage_outputs.empty()) {
        throw std::runtime_error("deduplicateFusedStageExpr requires at least one stage output.");
    }

    std::vector<ExprNode> dedup_nodes;
    dedup_nodes.reserve(stage_expr.nodes.size());

    std::unordered_map<StageNodeKey, uint32_t, StageNodeKeyHash> key_to_new_idx;
    std::vector<uint32_t> old_to_new(stage_expr.nodes.size(), UINT32_MAX);

    std::function<uint32_t(uint32_t)> remapNode = [&](uint32_t old_idx) -> uint32_t {
        if (old_idx >= stage_expr.nodes.size()) {
            throw std::runtime_error("deduplicateFusedStageExpr saw node index out of range.");
        }

        if (old_to_new[old_idx] != UINT32_MAX) {
            return old_to_new[old_idx];
        }

        ExprNode n = stage_expr.nodes[old_idx];

        if (!Expression::isLeafOp(n.op)) {
            n.lhs = remapNode(n.lhs);
        }

        if (Expression::isBinaryOp(n.op)) {
            n.rhs = remapNode(n.rhs);
        }
        if (Expression::isTernaryOp(n.op)) {
            n.rhs = remapNode(n.rhs);
            n.aux = remapNode(n.aux);
            if (n.alpha_node != UINT32_MAX) {
                n.alpha_node = remapNode(n.alpha_node);
            }
            if (n.beta_node != UINT32_MAX) {
                n.beta_node = remapNode(n.beta_node);
            }
        }

        if (isCommutativeStageOp(n.op) && Expression::isBinaryOp(n.op) && n.lhs > n.rhs) {
            std::swap(n.lhs, n.rhs);
        }

        StageNodeKey key = makeStageNodeKey(n);
        auto it = key_to_new_idx.find(key);
        if (it != key_to_new_idx.end()) {
            old_to_new[old_idx] = it->second;
            return it->second;
        }

        uint32_t new_idx = static_cast<uint32_t>(dedup_nodes.size());
        dedup_nodes.push_back(std::move(n));
        key_to_new_idx.emplace(key, new_idx);
        old_to_new[old_idx] = new_idx;
        return new_idx;
    };

    for (CompiledStageOutput& output : stage_outputs) {
        output.local_node_idx = remapNode(output.local_node_idx);
    }

    stage_expr.nodes = std::move(dedup_nodes);
    stage_expr.output_node = stage_outputs.front().local_node_idx;
}

static void compactFusedStageInputs(PhysicalExpression& stage_expr, std::vector<uint32_t>& stage_input_value_ids) {
    if (stage_expr.inputs.size() != stage_input_value_ids.size()) {
        throw std::runtime_error("Fused stage input metadata mismatch while compacting deduplicated stage inputs.");
    }

    std::vector<uint8_t> slot_used(stage_expr.inputs.size(), 0);
    for (const ExprNode& node : stage_expr.nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        if (node.input_slot >= slot_used.size()) {
            throw std::runtime_error("Leaf input slot out of range while compacting deduplicated fused stage inputs.");
        }

        slot_used[node.input_slot] = 1;
    }

    bool needs_compaction = false;
    for (uint32_t slot = 0; slot < slot_used.size(); ++slot) {
        if (!slot_used[slot]) {
            needs_compaction = true;
            break;
        }
    }

    if (!needs_compaction) {
        return;
    }

    std::vector<uint32_t> old_to_new_slot(stage_expr.inputs.size(), UINT32_MAX);

    std::vector<NamedInput> compacted_inputs;
    compacted_inputs.reserve(stage_expr.inputs.size());

    std::vector<uint32_t> compacted_input_value_ids;
    compacted_input_value_ids.reserve(stage_input_value_ids.size());

    for (uint32_t old_slot = 0; old_slot < stage_expr.inputs.size(); ++old_slot) {
        if (!slot_used[old_slot]) {
            continue;
        }

        uint32_t new_slot = static_cast<uint32_t>(compacted_inputs.size());
        old_to_new_slot[old_slot] = new_slot;

        NamedInput input = stage_expr.inputs[old_slot];
        input.slot = new_slot;
        compacted_inputs.push_back(std::move(input));
        compacted_input_value_ids.push_back(stage_input_value_ids[old_slot]);
    }

    for (ExprNode& node : stage_expr.nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        if (node.input_slot >= old_to_new_slot.size() || old_to_new_slot[node.input_slot] == UINT32_MAX) {
            throw std::runtime_error("Encountered unresolved leaf input slot while compacting deduplicated fused stage inputs.");
        }

        node.input_slot = old_to_new_slot[node.input_slot];
    }

    stage_expr.inputs = std::move(compacted_inputs);
    stage_input_value_ids = std::move(compacted_input_value_ids);
}

static bool isArgMinMaxOp(ExprOp op) { return op == ExprOp::REDUCE_ARGMIN || op == ExprOp::REDUCE_ARGMAX; }
static bool isMatmulOp(ExprOp op) { return op == ExprOp::MATMUL || op == ExprOp::GEMM; }
static bool isConvolutionForwardOp(ExprOp op) { return op == ExprOp::CONV2D || op == ExprOp::CONV3D; }
static bool isConvolutionBackwardOp(ExprOp op) {
    return op == ExprOp::CONV2D_BACKWARD_DATA || op == ExprOp::CONV2D_BACKWARD_FILTER || op == ExprOp::CONV3D_BACKWARD_DATA ||
           op == ExprOp::CONV3D_BACKWARD_FILTER;
}
static bool isConvolutionOp(ExprOp op) { return isConvolutionForwardOp(op) || isConvolutionBackwardOp(op); }
static bool isReduceMinMaxBackwardOp(ExprOp op) { return op == ExprOp::REDUCE_MIN_BACKWARD || op == ExprOp::REDUCE_MAX_BACKWARD; }
static bool isScanMinMaxBackwardOp(ExprOp op) {
    return op == ExprOp::SCAN_MIN_BACKWARD || op == ExprOp::SCAN_MAX_BACKWARD ||
           op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD;
}
static bool isScanOp(ExprOp op) { return op == ExprOp::SCAN || op == ExprOp::SEGMENTED_SCAN; }
static bool isSegmentedReduceOp(ExprOp op) {
    return op == ExprOp::SEGMENTED_REDUCE_SUM || op == ExprOp::SEGMENTED_REDUCE_MIN || op == ExprOp::SEGMENTED_REDUCE_MAX;
}
static bool isRmsNormOp(ExprOp op) { return op == ExprOp::RMSNORM; }
static bool isEmbeddingLookupOp(ExprOp op) { return op == ExprOp::EMBEDDING_LOOKUP; }
static bool isAttentionOp(ExprOp op) { return op == ExprOp::ATTENTION; }
static bool isAttentionBackwardOp(ExprOp op) {
    return op == ExprOp::ATTENTION_BACKWARD_Q || op == ExprOp::ATTENTION_BACKWARD_K || op == ExprOp::ATTENTION_BACKWARD_V ||
           op == ExprOp::ATTENTION_BACKWARD_BIAS;
}

static bool expressionHasIndexAwareOps(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::ROPE || node.op == ExprOp::TRANSPOSE || node.op == ExprOp::TAKE_ALONG_AXIS;
    });
}
static bool isTransposeOp(ExprOp op) { return op == ExprOp::TRANSPOSE; }

static bool isStageBoundaryOp(ExprOp op) {
    return isReductionOp(op) || isSoftmaxOp(op) || isScanOp(op) || isSegmentedReduceOp(op) || isRmsNormOp(op) || isMatmulOp(op) || isAttentionOp(op) ||
           isAttentionBackwardOp(op) || isConvolutionOp(op) || isReduceMinMaxBackwardOp(op) || isScanMinMaxBackwardOp(op) || isEmbeddingLookupOp(op) ||
           op == ExprOp::STRIDED_VIEW || op == ExprOp::CUDA_KERNEL_OUTPUT;
}

static void validateRaggedRuntimeExtentConsumers(const PhysicalExpression& expr) {
    std::vector<bool> depends_on_ragged_extent(expr.nodes.size(), false);
    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const ExprNode& node = expr.nodes[node_idx];
        if (node.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
            depends_on_ragged_extent[node_idx] = true;
            continue;
        }

        auto parent_depends = [&](uint32_t parent_idx) {
            if (parent_idx == UINT32_MAX) {
                return false;
            }
            if (parent_idx >= depends_on_ragged_extent.size()) {
                throw std::runtime_error("ragged runtime extent consumer references an out-of-range parent node.");
            }
            return static_cast<bool>(depends_on_ragged_extent[parent_idx]);
        };
        bool depends = parent_depends(node.lhs) || parent_depends(node.rhs) || parent_depends(node.aux) ||
                       parent_depends(node.alpha_node) || parent_depends(node.beta_node) ||
                       parent_depends(node.matmul_epilogue_aux) || parent_depends(node.attention_seq_len_q_node) ||
                       parent_depends(node.attention_seq_len_kv_node) || parent_depends(node.attention_ragged_offset_q_node) ||
                       parent_depends(node.attention_ragged_offset_kv_node) || parent_depends(node.attention_page_table_k_node) ||
                       parent_depends(node.attention_page_table_v_node) || parent_depends(node.attention_dropout_seed_node) ||
                       parent_depends(node.attention_dropout_offset_node);
        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            for (uint32_t input_node : node.cuda_kernel_input_nodes) {
                depends = depends || parent_depends(input_node);
            }
        }
        depends_on_ragged_extent[node_idx] = depends;
        if (!depends) {
            continue;
        }

        if (isStageBoundaryOp(node.op) || node.op == ExprOp::ROPE || node.op == ExprOp::TRANSPOSE ||
            node.op == ExprOp::TAKE_ALONG_AXIS) {
            throw std::runtime_error(
                "an expression carrying ragged runtime extent reached an unsupported operation; use an explicit ragged operation or adapter.");
        }
    }
}

static uint32_t peelExplicitTransposeChain(const PhysicalExpression& expr, uint32_t node_idx, bool& transpose_toggled) {
    transpose_toggled = false;
    uint32_t current = node_idx;
    while (current != UINT32_MAX && current < expr.nodes.size() && expr.nodes[current].op == ExprOp::TRANSPOSE) {
        const ExprNode& transpose = expr.nodes[current];
        if (transpose.lhs == UINT32_MAX || transpose.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Transpose node missing valid lhs input while peeling matmul/gemm transpose dependency.");
        }
        transpose_toggled = !transpose_toggled;
        current = transpose.lhs;
    }
    return current;
}

static void collectExternalValueIds(const PhysicalExpression& expr,
                                    const std::unordered_set<uint32_t>& region_nodes,
                                    const std::unordered_map<uint32_t, uint32_t>& node_output_value_id,
                                    std::unordered_set<uint32_t>& external_value_ids) {
    auto addExternalValue = [&](uint32_t parent_idx) {
        if (parent_idx == UINT32_MAX)
            return;

        if (region_nodes.contains(parent_idx))
            return;

        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("External dependency node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        if (parent.op == ExprOp::INPUT || parent.op == ExprOp::RUNTIME_SCALAR || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            external_value_ids.insert(parent.input_slot);
            return;
        }

        if (isStageBoundaryOp(parent.op)) {
            auto it = node_output_value_id.find(parent_idx);
            if (it == node_output_value_id.end()) {
                throw std::runtime_error("Missing value id for boundary dependency.");
            }
            external_value_ids.insert(it->second);
            return;
        }

        throw std::runtime_error("Found non-boundary dependency outside fusable region.");
    };

    for (uint32_t node_idx : region_nodes) {
        const ExprNode& node = expr.nodes[node_idx];

        if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            external_value_ids.insert(node.input_slot);
            continue;
        }

        if (Expression::isLeafOp(node.op)) {
            continue;
        }

        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            for (uint32_t input_node : node.cuda_kernel_input_nodes) {
                addExternalValue(input_node);
            }
            continue;
        }

        addExternalValue(node.lhs);

        if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
            addExternalValue(node.rhs);
        }
        if (Expression::isTernaryOp(node.op)) {
            addExternalValue(node.aux);
        }
        if (node.op == ExprOp::ATTENTION && node.attention_use_bias) {
            addExternalValue(node.alpha_node);
        }
        if (node.op == ExprOp::ATTENTION && node.attention_use_padding_mask) {
            addExternalValue(node.attention_seq_len_q_node);
            addExternalValue(node.attention_seq_len_kv_node);
        }
        if (node.op == ExprOp::ATTENTION && node.attention_use_ragged_offsets) {
            addExternalValue(node.attention_ragged_offset_q_node);
            addExternalValue(node.attention_ragged_offset_kv_node);
        }
        if (node.op == ExprOp::ATTENTION && node.attention_use_paged_kv_cache) {
            addExternalValue(node.attention_page_table_k_node);
            addExternalValue(node.attention_page_table_v_node);
        }
        if (node.op == ExprOp::ATTENTION && node.attention_dropout_probability > 0.0f) {
            addExternalValue(node.attention_dropout_seed_node);
            addExternalValue(node.attention_dropout_offset_node);
        }
        if (node.op == ExprOp::ATTENTION && node.attention_use_fp8_forward_scaling) {
            addExternalValue(node.attention_descale_q_node);
            addExternalValue(node.attention_descale_k_node);
            addExternalValue(node.attention_descale_v_node);
            addExternalValue(node.attention_descale_s_node);
            addExternalValue(node.attention_scale_s_node);
            addExternalValue(node.attention_scale_o_node);
            addExternalValue(node.attention_amax_s_node);
            addExternalValue(node.attention_amax_o_node);
        }
        if (isAttentionBackwardOp(node.op)) {
            addExternalValue(node.alpha_node);
            if (node.attention_use_bias) {
                addExternalValue(node.beta_node);
            }
            if (node.attention_use_padding_mask) {
                addExternalValue(node.attention_seq_len_q_node);
                addExternalValue(node.attention_seq_len_kv_node);
            }
            if (node.attention_use_ragged_offsets) {
                addExternalValue(node.attention_ragged_offset_q_node);
                addExternalValue(node.attention_ragged_offset_kv_node);
            }
            if (node.attention_use_paged_kv_cache) {
                addExternalValue(node.attention_page_table_k_node);
                addExternalValue(node.attention_page_table_v_node);
            }
            if (node.attention_dropout_probability > 0.0f) {
                addExternalValue(node.attention_dropout_seed_node);
                addExternalValue(node.attention_dropout_offset_node);
            }
        }
    }
}

static bool setsOverlap(const std::unordered_set<uint32_t>& a, const std::unordered_set<uint32_t>& b) {
    if (a.size() > b.size()) {
        return setsOverlap(b, a);
    }

    for (uint32_t v : a) {
        if (b.contains(v)) {
            return true;
        }
    }

    return false;
}

static const char* fusedOpTag(ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
            return "IN";
        case ExprOp::SCALAR_FP:
            return "F";
        case ExprOp::RUNTIME_SCALAR:
            return "RSC";
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            return "TRSC";
        case ExprOp::ADD:
            return "ADD";
        case ExprOp::SUB:
            return "SUB";
        case ExprOp::MUL:
            return "MUL";
        case ExprOp::DIV:
            return "DIV";
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
        case ExprOp::CAST:
            return "CAST";
        case ExprOp::RAGGED_VALUEWISE_EXTENT:
            return "RAGGED_EXTENT";
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
        case ExprOp::EXP2:
            return "EXP2";
        case ExprOp::EXP10:
            return "EXP10";
        case ExprOp::LN:
            return "LOG";
        case ExprOp::LOG1P:
            return "LOG1P";
        case ExprOp::LOG2:
            return "LOG2";
        case ExprOp::LOG10:
            return "LOG10";
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
            return "VIEW_BW";
        case ExprOp::UNSQUEEZE:
            return "UNSQ";
        case ExprOp::SQUEEZE:
            return "SQZ";
        case ExprOp::TRANSPOSE:
            return "TRANSPOSE";
        case ExprOp::TAKE_ALONG_AXIS:
            return "TAKE";
        case ExprOp::POW:
            return "POW";
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
        case ExprOp::SCAN_MIN_BACKWARD:
            return "SCAN_MIN_BW";
        case ExprOp::SCAN_MAX_BACKWARD:
            return "SCAN_MAX_BW";
        case ExprOp::SEGMENTED_SCAN_MIN_BACKWARD:
            return "SEG_SCAN_MIN_BW";
        case ExprOp::SEGMENTED_SCAN_MAX_BACKWARD:
            return "SEG_SCAN_MAX_BW";
        case ExprOp::REDUCE_AVG:
            return "RAVG";
        case ExprOp::REDUCE_NORM1:
            return "RNORM1";
        case ExprOp::REDUCE_NORM2:
            return "RNORM2";
        case ExprOp::SCAN:
            return "SCAN";
        case ExprOp::SEGMENTED_SCAN:
            return "SEGMENTED_SCAN";
        case ExprOp::SEGMENTED_REDUCE_SUM:
            return "SEG_REDUCE_SUM";
        case ExprOp::SEGMENTED_REDUCE_MIN:
            return "SEG_REDUCE_MIN";
        case ExprOp::SEGMENTED_REDUCE_MAX:
            return "SEG_REDUCE_MAX";
        case ExprOp::RMSNORM:
            return "RMSNORM";
        case ExprOp::EMBEDDING_LOOKUP:
            return "EMBEDDING_LOOKUP";
        case ExprOp::MATMUL:
            return "MATMUL";
        case ExprOp::GEMM:
            return "GEMM";
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
        default:
            throw std::runtime_error("Unsupported op in fusedRegionSignature, value: " + to_string((int)op));
    }
}

static std::string optionalDTypeSignature(const std::optional<DataType>& dtype) {
    if (!dtype.has_value()) {
        return "none";
    }
    return TensorDescriptor::getElementTypeName(dtype.value());
}

static void appendNodeDTypeSignature(std::string& s, const ExprNode& node) {
    s += ";input=" + optionalDTypeSignature(node.input_tensor_dtype);
    s += ";out=" + optionalDTypeSignature(node.output_dtype);
    s += ";compute=" + optionalDTypeSignature(node.compute_dtype);
    s += ";bwd_out=" + optionalDTypeSignature(node.backward_output_dtype);
    s += ";bwd_compute=" + optionalDTypeSignature(node.backward_compute_dtype);
}

static std::string fusedRegionSignatureRec(const PhysicalExpression& expr, uint32_t node_idx);

static const char* matmulEpilogueSignatureName(MatmulEpilogue epilogue) {
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

static const char* matmulBackwardEpilogueSignatureName(MatmulBackwardEpilogue epilogue) {
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

static std::string uintVecSignature(const std::vector<uint64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += std::to_string(v[i]);
        if (i + 1 < v.size()) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

static std::string doubleVecSignature(const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += std::to_string(scalarBits(v[i]));
        if (i + 1 < v.size()) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

static std::string gemmScaleSignature(const PhysicalExpression& expr, uint32_t node_idx, double scale_fp) {
    if (node_idx == UINT32_MAX) {
        return std::to_string(scalarBits(scale_fp));
    }
    return fusedRegionSignatureRec(expr, node_idx) + "*" + std::to_string(scalarBits(scale_fp));
}

static std::string fusedRegionSignatureRec(const PhysicalExpression& expr, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size()) {
        throw std::runtime_error("fusedRegionSignatureRec node_idx out of range.");
    }

    const ExprNode& node = expr.nodes[node_idx];

    switch (node.op) {
        case ExprOp::INPUT: {
            std::string s = std::string("IN(") + std::to_string(node.input_slot) + ")";
            appendNodeDTypeSignature(s, node);
            return s;
        }

        case ExprOp::RUNTIME_SCALAR: {
            std::string s = std::string("RIN(") + std::to_string(node.input_slot) + ")";
            appendNodeDTypeSignature(s, node);
            return s;
        }

        case ExprOp::TENSOR_RUNTIME_SCALAR: {
            std::string s = std::string("TRIN(") + std::to_string(node.input_slot) + ")";
            appendNodeDTypeSignature(s, node);
            return s;
        }

        case ExprOp::SCALAR_FP: {
            std::string s = std::string("F(") + std::to_string(scalarBits(node.scalar_fp)) + ")";
            appendNodeDTypeSignature(s, node);
            return s;
        }

        case ExprOp::FILL: {
            std::string s =
                std::string("FILL(") + std::to_string(scalarBits(node.scalar_fp)) + ",dims=" + uintVecSignature(node.fill_dims) + ")";
            appendNodeDTypeSignature(s, node);
            return s;
        }

        default:
            break;
    }

    if (Expression::isLeafOp(node.op)) {
        std::string s = std::string(fusedOpTag(node.op));
        appendNodeDTypeSignature(s, node);
        return s;
    }

    const std::string lhs = fusedRegionSignatureRec(expr, node.lhs);

    if (isStageBoundaryOp(node.op)) {
        std::string s;

        if (isTransposeOp(node.op)) {
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ")";
        } else if (isReduceMinMaxBackwardOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",rhs=" + rhs + ",axes=" + uintVecSignature(node.reduction_axes) +
                ",squeeze=" + uintVecSignature(node.squeeze_axes) + ")";
        } else if (isScanMinMaxBackwardOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",grad=" + rhs;
            if (node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
                s += ",offsets=" + fusedRegionSignatureRec(expr, node.aux);
            }
            s += ",mode=" + std::to_string(static_cast<int>(node.scan_mode)) +
                 ",axis=" + std::to_string(node.scan_axis) +
                 ",reverse=" + std::to_string(node.scan_reverse ? 1 : 0) + ")";
        } else if (isSoftmaxOp(node.op)) {
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs +
                ",algorithm=" + std::to_string(static_cast<int>(node.softmax_algorithm)) +
                ",mode=" + std::to_string(static_cast<int>(node.softmax_mode)) + ")";
        } else if (isSegmentedReduceOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",offsets=" + rhs + ")";
        } else if (isScanOp(node.op)) {
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs;
            if (node.op == ExprOp::SEGMENTED_SCAN) {
                const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
                s += ",offsets=" + rhs;
            }
            s += ",op=" + std::to_string(static_cast<int>(node.scan_op)) +
                 ",mode=" + std::to_string(static_cast<int>(node.scan_mode)) +
                 ",axis=" + std::to_string(node.scan_axis) +
                 ",reverse=" + std::to_string(node.scan_reverse ? 1 : 0) + ")";
        } else if (isRmsNormOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",scale=" + rhs +
                ",hidden=" + std::to_string(node.rms_norm_normalized_feature_count) +
                ",epsilon=" + std::to_string(scalarBits(node.rms_norm_epsilon)) +
                ",fused=" + std::string(toString(node.rms_norm_fused_activation)) + ")";
        } else if (isMatmulOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);

            if (node.op == ExprOp::MATMUL) {
                s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",rhs=" + rhs + ",ta=" + std::to_string(node.transpose_lhs ? 1 : 0) +
                    ",tb=" + std::to_string(node.transpose_rhs ? 1 : 0) +
                    ",epilogue=" + std::string(matmulEpilogueSignatureName(node.matmul_epilogue)) +
                    ",backward_epilogue=" + std::string(matmulBackwardEpilogueSignatureName(node.matmul_backward_epilogue));
                if (node.matmul_epilogue_aux != UINT32_MAX) {
                    s += ",epilogue_aux=" + fusedRegionSignatureRec(expr, node.matmul_epilogue_aux);
                }
                s += ")";
            } else {
                const std::string aux = fusedRegionSignatureRec(expr, node.aux);
                s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",rhs=" + rhs + ",aux=" + aux +
                    ",ta=" + std::to_string(node.transpose_lhs ? 1 : 0) + ",tb=" + std::to_string(node.transpose_rhs ? 1 : 0) +
                    ",tc=" + std::to_string(node.transpose_aux ? 1 : 0) +
                    ",alpha=" + gemmScaleSignature(expr, node.alpha_node, node.alpha_fp) +
                    ",beta=" + gemmScaleSignature(expr, node.beta_node, node.beta_fp) +
                    ",epilogue=" + std::string(matmulEpilogueSignatureName(node.matmul_epilogue)) +
                    ",backward_epilogue=" + std::string(matmulBackwardEpilogueSignatureName(node.matmul_backward_epilogue));
                if (node.matmul_epilogue_aux != UINT32_MAX) {
                    s += ",epilogue_aux=" + fusedRegionSignatureRec(expr, node.matmul_epilogue_aux);
                }
                s += ")";
            }
        } else if (isAttentionOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            const std::string aux = fusedRegionSignatureRec(expr, node.aux);
            s = std::string(fusedOpTag(node.op)) + "(q=" + lhs + ",k=" + rhs + ",v=" + aux +
                ",qLayout=" + std::to_string(static_cast<int>(node.attention_q_layout)) +
                ",kLayout=" + std::to_string(static_cast<int>(node.attention_k_layout)) +
                ",vLayout=" + std::to_string(static_cast<int>(node.attention_v_layout)) +
                ",oLayout=" + std::to_string(static_cast<int>(node.attention_o_layout)) +
                ",mask=" + std::to_string(static_cast<int>(node.attention_mask_kind)) +
                ",left=" + std::to_string(node.attention_diagonal_left_bound) +
                ",right=" + std::to_string(node.attention_diagonal_right_bound) +
                ",hasScale=" + std::to_string(node.attention_has_scale ? 1 : 0) + ",scale=" + std::to_string(node.attention_scale) +
                ",alibi=" + std::to_string(node.attention_use_alibi_mask ? 1 : 0) +
                ",bias=" + std::to_string(node.attention_use_bias ? 1 : 0) +
                ",padding=" + std::to_string(node.attention_use_padding_mask ? 1 : 0) +
                ",ragged=" + std::to_string(node.attention_use_ragged_offsets ? 1 : 0) +
                ",dropout=" + formatFloatCanonical(node.attention_dropout_probability);
            if (node.attention_use_bias && node.alpha_node != UINT32_MAX) {
                s += ",biasNode=" + fusedRegionSignatureRec(expr, node.alpha_node);
            }
            if (node.attention_use_ragged_offsets) {
                s += ",raggedQ=" + fusedRegionSignatureRec(expr, node.attention_ragged_offset_q_node);
                s += ",raggedKV=" + fusedRegionSignatureRec(expr, node.attention_ragged_offset_kv_node);
            }
            if (node.attention_use_paged_kv_cache) {
                s += ",pagedMax=" + std::to_string(node.attention_paged_kv_max_sequence_length);
                s += ",pageK=" + fusedRegionSignatureRec(expr, node.attention_page_table_k_node);
                s += ",pageV=" + fusedRegionSignatureRec(expr, node.attention_page_table_v_node);
            }
            if (node.attention_dropout_probability > 0.0f) {
                s += ",dropoutSeed=" + fusedRegionSignatureRec(expr, node.attention_dropout_seed_node);
                s += ",dropoutOffset=" + fusedRegionSignatureRec(expr, node.attention_dropout_offset_node);
            }
            s += ")";
        } else if (isAttentionBackwardOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            const std::string aux = fusedRegionSignatureRec(expr, node.aux);
            const std::string dO = fusedRegionSignatureRec(expr, node.alpha_node);
            s = std::string(fusedOpTag(node.op)) + "(q=" + lhs + ",k=" + rhs + ",v=" + aux + ",dO=" + dO +
                ",qLayout=" + std::to_string(static_cast<int>(node.attention_q_layout)) +
                ",kLayout=" + std::to_string(static_cast<int>(node.attention_k_layout)) +
                ",vLayout=" + std::to_string(static_cast<int>(node.attention_v_layout)) +
                ",oLayout=" + std::to_string(static_cast<int>(node.attention_o_layout)) +
                ",mask=" + std::to_string(static_cast<int>(node.attention_mask_kind)) +
                ",left=" + std::to_string(node.attention_diagonal_left_bound) +
                ",right=" + std::to_string(node.attention_diagonal_right_bound) +
                ",hasScale=" + std::to_string(node.attention_has_scale ? 1 : 0) + ",scale=" + std::to_string(node.attention_scale) +
                ",alibi=" + std::to_string(node.attention_use_alibi_mask ? 1 : 0) +
                ",bias=" + std::to_string(node.attention_use_bias ? 1 : 0) +
                ",padding=" + std::to_string(node.attention_use_padding_mask ? 1 : 0) +
                ",ragged=" + std::to_string(node.attention_use_ragged_offsets ? 1 : 0) +
                ",dropout=" + formatFloatCanonical(node.attention_dropout_probability);
            if (node.attention_use_bias && node.beta_node != UINT32_MAX) {
                s += ",biasNode=" + fusedRegionSignatureRec(expr, node.beta_node);
            }
            if (node.attention_use_ragged_offsets) {
                s += ",raggedQ=" + fusedRegionSignatureRec(expr, node.attention_ragged_offset_q_node);
                s += ",raggedKV=" + fusedRegionSignatureRec(expr, node.attention_ragged_offset_kv_node);
            }
            if (node.attention_use_paged_kv_cache) {
                s += ",pagedMax=" + std::to_string(node.attention_paged_kv_max_sequence_length);
                s += ",pageK=" + fusedRegionSignatureRec(expr, node.attention_page_table_k_node);
                s += ",pageV=" + fusedRegionSignatureRec(expr, node.attention_page_table_v_node);
            }
            if (node.attention_dropout_probability > 0.0f) {
                s += ",dropoutSeed=" + fusedRegionSignatureRec(expr, node.attention_dropout_seed_node);
                s += ",dropoutOffset=" + fusedRegionSignatureRec(expr, node.attention_dropout_offset_node);
            }
            s += ")";
        } else if (isConvolutionOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",rhs=" + rhs + ",sh=" + std::to_string(node.conv_stride_h) +
                ",sw=" + std::to_string(node.conv_stride_w) + ",ph=" + std::to_string(node.conv_pad_h) +
                ",pw=" + std::to_string(node.conv_pad_w) + ")";
        } else {
            s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",axes=" + uintVecSignature(node.reduction_axes) +
                ",squeeze=" + uintVecSignature(node.squeeze_axes) + ")";
        }

        appendNodeDTypeSignature(s, node);
        return s;
    }

    if (!Expression::isBinaryOp(node.op) && !Expression::isTernaryOp(node.op)) {
        std::string s;
        if (node.op == ExprOp::RESHAPE) {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ",dims=" + uintVecSignature(node.reshape_dims) + ")";
        } else if (node.op == ExprOp::UNSQUEEZE) {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ",axes=" + uintVecSignature(node.unsqueeze_axes) + ")";
        } else if (node.op == ExprOp::SQUEEZE) {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ",axes=" + uintVecSignature(node.squeeze_axes) + ")";
        } else if (node.op == ExprOp::ROPE) {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ",seqAxis=" + std::to_string(node.rope_sequence_axis) +
                ",dimAxis=" + std::to_string(node.rope_head_dim_axis) + ",rotaryDim=" + std::to_string(node.rope_rotary_dim) +
                ",base=" + std::to_string(scalarBits(node.rope_base)) + ",offset=" + std::to_string(node.rope_position_offset) +
                ",interleaved=" + std::to_string(node.rope_interleaved ? 1 : 0) + ",inverse=" + std::to_string(node.rope_inverse ? 1 : 0) +
                ",scaling=" + std::to_string(static_cast<int>(node.rope_scaling_kind)) +
                ",factor=" + std::to_string(scalarBits(node.rope_scaling_factor)) +
                ",originalMax=" + std::to_string(node.rope_original_max_position_embeddings) +
                ",attentionFactor=" + std::to_string(scalarBits(node.rope_attention_factor)) +
                ",yarnBetaFast=" + std::to_string(scalarBits(node.rope_yarn_beta_fast)) +
                ",yarnBetaSlow=" + std::to_string(scalarBits(node.rope_yarn_beta_slow)) +
                ",llama3LowFreq=" + std::to_string(scalarBits(node.rope_llama3_low_freq_factor)) +
                ",llama3HighFreq=" + std::to_string(scalarBits(node.rope_llama3_high_freq_factor)) +
                ",longRopeShort=" + doubleVecSignature(node.rope_long_rope_short_factors) +
                ",longRopeLong=" + doubleVecSignature(node.rope_long_rope_long_factors) +
                ",allowInPlace=" + std::to_string(node.rope_allow_in_place_materialization ? 1 : 0) + ")";
        } else {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ")";
        }
        appendNodeDTypeSignature(s, node);
        return s;
    }

    if (node.op == ExprOp::WHERE) {
        const std::string true_value = fusedRegionSignatureRec(expr, node.rhs);
        const std::string false_value = fusedRegionSignatureRec(expr, node.aux);
        std::string s = std::string(fusedOpTag(node.op)) + "(" + lhs + "," + true_value + "," + false_value + ")";
        appendNodeDTypeSignature(s, node);
        return s;
    }

    std::string rhs = fusedRegionSignatureRec(expr, node.rhs);

    if (node.op == ExprOp::TAKE_ALONG_AXIS) {
        std::string s = std::string(fusedOpTag(node.op)) + "(" + lhs + "," + rhs + ",axis=" + uintVecSignature(node.reduction_axes) + ")";
        appendNodeDTypeSignature(s, node);
        return s;
    }

    if (node.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
        return std::string(fusedOpTag(node.op)) + "(" + lhs + "," + rhs + ",batch=" +
               std::to_string(node.ragged_runtime_batch_size) + ",maxActive=" +
               std::to_string(node.ragged_runtime_max_active_values) + ",elementsPerValue=" +
               std::to_string(node.ragged_runtime_elements_per_value) + ")";
    }

    if (isCommutativeStageOp(node.op) && rhs < lhs) {
        std::string s = std::string(fusedOpTag(node.op)) + "(" + rhs + "," + lhs + ")";
        appendNodeDTypeSignature(s, node);
        return s;
    }

    std::string s = std::string(fusedOpTag(node.op)) + "(" + lhs + "," + rhs + ")";
    appendNodeDTypeSignature(s, node);
    return s;
}

static std::string fusedRegionSignature(const PhysicalExpression& expr, uint32_t root_idx) {
    return fusedRegionSignatureRec(expr, root_idx);
}

shared_ptr<CompiledEquation> EquationCompiler::loadCubin(const EquationCacheKey& key,
                                                         const vector<char>& cubin,
                                                         const string& kernel_name,
                                                         const vector<string>& input_names,
                                                         const std::vector<NamedInput::Kind>& input_kinds,
                                                         const std::vector<DataType>& input_dtypes,
                                                         const std::vector<DataType>& output_dtypes,
                                                         int device_num) {
    CUmodule module;
    CUfunction fn;

    CU_CHECK(cuModuleLoadData(&module, cubin.data()));
    CU_CHECK(cuModuleGetFunction(&fn, module, kernel_name.c_str()));

    auto out = make_shared<CompiledEquation>();
    out->key = key;
    out->module = module;
    out->kernel = fn;
    out->kernel_name = kernel_name;
    out->input_names = input_names;
    out->input_kinds = input_kinds;
    out->input_dtypes = input_dtypes;
    out->output_dtypes = output_dtypes;
    out->deviceNum = device_num;

    return out;
}

static std::vector<DataType> collectCompiledInputDTypes(const PhysicalExpression& expr) {
    std::vector<DataType> input_dtypes(expr.numInputs(), DataType::FP32);
    std::vector<uint8_t> seen(expr.numInputs(), 0);

    for (const ExprNode& node : expr.nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }
        if (!node.input_tensor_dtype.has_value()) {
            throw runtime_error("Fused stage input node is missing resolved input_tensor_dtype.");
        }
        if (node.input_slot >= input_dtypes.size()) {
            throw runtime_error("Input slot out of range while collecting compiled input dtypes.");
        }

        const DataType dtype = node.input_tensor_dtype.value();
        if (seen[node.input_slot]) {
            if (input_dtypes[node.input_slot] != dtype) {
                throw runtime_error("Inconsistent fused stage input dtype for local input slot.");
            }
        } else {
            input_dtypes[node.input_slot] = dtype;
            seen[node.input_slot] = 1;
        }
    }

    for (uint32_t slot = 0; slot < input_dtypes.size(); ++slot) {
        if (!seen[slot]) {
            throw runtime_error("Unused or unresolved fused stage input slot.");
        }
    }

    return input_dtypes;
}

static std::vector<DataType> collectCompiledOutputDTypes(const PhysicalExecutionStage& stage) {
    std::vector<DataType> output_dtypes;
    output_dtypes.reserve(stage.outputs.size());

    for (const CompiledStageOutput& output : stage.outputs) {
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Stage output local_node_idx out of range while collecting output dtypes.");
        }
        const ExprNode& node = stage.expr.nodes[output.local_node_idx];
        if (!node.output_dtype.has_value()) {
            throw runtime_error("Fused stage output node is missing resolved output_dtype.");
        }
        output_dtypes.push_back(node.output_dtype.value());
    }

    return output_dtypes;
}

static bool stageHasTransposedMaterializedOutput(const std::vector<CompiledStageOutput>& outputs) {
    return std::any_of(outputs.begin(), outputs.end(), [](const CompiledStageOutput& output) {
        return output.materialized_layout == MaterializedTensorLayout::Transposed;
    });
}

vector<char> EquationCompiler::linkToCubin(const vector<char>& ltoir, const EquationSignature& sig) {
    string arch = "-arch=sm_" + to_string(sig.sm_major) + to_string(sig.sm_minor);

    const char* opts[] = {arch.c_str(), "-lto", "-O3"};

    nvJitLinkHandle handle;
    NVJITLINK_CHECK(handle, nvJitLinkCreate(&handle, 3, opts));

    NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void*)ltoir.data(), ltoir.size(), "fused.ltoir"));

    NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));

    size_t cubin_size = 0;
    NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
    vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));

    nvJitLinkDestroy(&handle);
    return cubin;
}

constexpr bool PRINT_KERNELS = false;

vector<char> EquationCompiler::compileToLtoIr(const string& src, const string& kernel_name, const EquationSignature& sig) {
    if (PRINT_KERNELS) {
        printf("%s\n", src.c_str());
        fflush(stdout);
    }

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "fused.cu", 0, nullptr, nullptr));

    string arch = "--gpu-architecture=compute_" + to_string(sig.sm_major) + to_string(sig.sm_minor);

    vector<string> option_strings = {arch, "-dlto", "--std=c++23", "-fmad=true"};

    const string bundled_headers_dir = getNvrtcBundledHeadersDir();
    if (!bundled_headers_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(bundled_headers_dir, ec);
        if (ec) {
            throw runtime_error("Failed to create NVRTC bundled CUDA headers directory " + bundled_headers_dir + ": " + ec.message());
        }
        option_strings.emplace_back("--use-bundled-headers=" + bundled_headers_dir);
    } else {
        const vector<string> cuda_include_dirs = getCudaIncludeDirs();
        for (const string& include_dir : cuda_include_dirs)
            option_strings.emplace_back(std::string("--include-path=") + include_dir);
    }

    if (sig.use_fast_math)
        option_strings.emplace_back("--use_fast_math");

    vector<const char*> opts;
    opts.reserve(option_strings.size());
    for (const string& option : option_strings)
        opts.push_back(option.c_str());

    NVRTC_COMPILE_CHECK(prog, (int)opts.size(), opts.data());

    size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size));
    vector<char> ltoir(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog, ltoir.data()));

    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return ltoir;
}

static bool expressionUsesDeviceRaggedRuntimeExtent(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::RAGGED_VALUEWISE_EXTENT;
    });
}

shared_ptr<CompiledEquation> EquationCompiler::compileFusedStage(const PhysicalExecutionStage& stage,
                                                                 const EquationSignature& sig,
                                                                 bool use_uint32_index_math) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("compileFusedStage called on non-fused stage.");
    }

    ensureCudaContextCurrent(sig.device_num);

    EquationCacheKey key(canonicalize(stage), sig, use_uint32_index_math);
    shared_ptr<CompiledEquation> hit = cacheLookup(key);
    if (hit)
        return hit;

    vector<string> input_names;
    std::vector<NamedInput::Kind> input_kinds;
    input_names.reserve(stage.expr.inputs.size());
    input_kinds.reserve(stage.expr.inputs.size());
    for (const NamedInput& input : stage.expr.inputs) {
        input_names.push_back(input.name);
        input_kinds.push_back(input.kind);
    }
    const std::vector<DataType> input_dtypes = collectCompiledInputDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectCompiledOutputDTypes(stage);

    const bool uses_device_ragged_runtime_extent = expressionUsesDeviceRaggedRuntimeExtent(stage.expr);

    // RoPE/index-aware fused stages need runtime output dimensions to compute logical
    // coordinates. They are always launched through the specialized-broadcast path.
    // Ragged runtime extent is deliberately limited to ordinary flat valuewise kernels
    // in this stabilization chunk; unsupported combinations reject rather than silently
    // processing padded capacity.
    if (expressionHasIndexAwareOps(stage.expr)) {
        if (uses_device_ragged_runtime_extent) {
            throw std::runtime_error("ragged runtime extent is not supported with index-aware fused operations.");
        }
        auto compiled = std::make_shared<CompiledEquation>();
        compiled->key = key;
        compiled->kernel_name = "fused_kernel";
        compiled->deviceNum = sig.device_num;
        compiled->input_names = std::move(input_names);
        compiled->input_kinds = std::move(input_kinds);
        compiled->input_dtypes = input_dtypes;
        compiled->output_dtypes = output_dtypes;
        compiled->launch_kind = CompiledEquation::LaunchKind::BroadcastGrouped;
        compiled->elements_per_thread = 1;
        compiled->uses_uint32_numel_arg = false;
        compiled->uses_device_runtime_extent = uses_device_ragged_runtime_extent;
        cacheInsert(key, compiled);
        return compiled;
    }

    string kernel_name = "fused_kernel";
    const std::string cuda_src = CudaSourceEmitter::emitFlat(stage, kernel_name, use_uint32_index_math);

    vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    vector<char> cubin = linkToCubin(ltoir, sig);
    auto compiled = loadCubin(key, cubin, kernel_name, input_names, input_kinds, input_dtypes, output_dtypes, sig.device_num);
    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        compiled->launch_kind = CompiledEquation::LaunchKind::FusedTiledTranspose;
        compiled->elements_per_thread = 1;
        compiled->tiled_transpose_pack_scalars = CudaSourceEmitter::tiledTransposePackScalars(stage);
        compiled->uses_uint32_numel_arg = false;
        compiled->uses_uint32_tiled_transpose_index_math = use_uint32_index_math;
    } else {
        compiled->elements_per_thread = CudaSourceEmitter::flatElementsPerThread(stage);
        compiled->uses_uint32_numel_arg = use_uint32_index_math;
    }
    compiled->uses_device_runtime_extent = uses_device_ragged_runtime_extent;

    cacheInsert(key, compiled);
    return compiled;
}

shared_ptr<CompiledReduction> EquationCompiler::compileReduction(const PhysicalExpression& expr) {
    if (expr.numInputs() != 1) {
        throw std::runtime_error("Reduction stage must have exactly one input.");
    }

    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Reduction stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isValueReductionOp(node.op)) {
        throw std::runtime_error("Reduction stage output node is not a supported value reduction op.");
    }

    if (node.lhs == UINT32_MAX) {
        throw std::runtime_error("Reduction node is missing its input.");
    }

    if (node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Reduction node lhs is out of range.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    if (input_node.op != ExprOp::INPUT) {
        throw std::runtime_error("Reduction stage input must be a local INPUT node.");
    }

    if (!input_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("Reduction input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Reduction node missing resolved output_dtype.");
    }
    if (node.output_dtype.value() != DataType::FP32) {
        throw std::runtime_error("Floating-point reduction stages must materialize FP32 output; add an explicit cast afterward.");
    }
    if (!node.compute_dtype.has_value() || node.compute_dtype.value() != DataType::FP32) {
        throw std::runtime_error("Floating-point reduction stages must compute in FP32.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.value());

    return make_shared<CompiledReduction>(
        node.op, node.reduction_axes, node.squeeze_axes, supported_input_dtype, node.output_dtype.value(), node.compute_dtype);
}

shared_ptr<CompiledArgMinMax> EquationCompiler::compileArgMinMax(const PhysicalExpression& expr) {
    if (expr.numInputs() != 1) {
        throw std::runtime_error("ArgMinMax stage must have exactly one input.");
    }

    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("ArgMinMax stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isArgMinMaxOp(node.op)) {
        throw std::runtime_error("ArgMinMax stage output node is not a supported arg min/max op.");
    }

    if (node.lhs == UINT32_MAX) {
        throw std::runtime_error("ArgMinMax node is missing its input.");
    }

    if (node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("ArgMinMax node lhs is out of range.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    if (input_node.op != ExprOp::INPUT) {
        throw std::runtime_error("ArgMinMax stage input must be a local INPUT node.");
    }

    if (!input_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("ArgMinMax input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("ArgMinMax node missing resolved output_dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.value());

    return make_shared<CompiledArgMinMax>(
        node.op, node.reduction_axes, node.squeeze_axes, supported_input_dtype, node.output_dtype.value(), node.compute_dtype);
}

shared_ptr<CompiledSegmentedReduction> EquationCompiler::compileSegmentedReduction(const PhysicalExpression& expr) {
    if (expr.numInputs() != 2) {
        throw std::runtime_error("Segmented-reduction stage must have exactly two inputs: values and offsets.");
    }
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Segmented-reduction stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isSegmentedReduceOp(node.op)) {
        throw std::runtime_error("Segmented-reduction stage output node is not a supported segmented reduction op.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("Segmented-reduction node is missing values or offsets input.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    const ExprNode& offsets_node = expr.nodes[node.rhs];
    if (input_node.op != ExprOp::INPUT || offsets_node.op != ExprOp::INPUT) {
        throw std::runtime_error("Segmented-reduction stage inputs must be local INPUT nodes.");
    }
    if (!input_node.input_tensor_dtype.has_value() || !offsets_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("Segmented-reduction input nodes missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Segmented-reduction node missing resolved output_dtype.");
    }
    if (!CubSegmentedReduction::isOffsetDataTypeSupported(offsets_node.input_tensor_dtype.value())) {
        throw std::runtime_error("Expression segmented-reduction offsets dtype is not supported by the central CUB reducer.");
    }

    const DataType input_dtype = input_node.input_tensor_dtype.value();
    if (input_dtype != node.output_dtype.value()) {
        throw std::runtime_error("Expression segmented-reduction currently requires input/output dtypes to match.");
    }
    if (!CubSegmentedReduction::isInputDataTypeSupported(input_dtype)) {
        throw std::runtime_error("Expression segmented-reduction input dtype is not supported by the central CUB reducer.");
    }
    switch (node.op) {
        case ExprOp::SEGMENTED_REDUCE_SUM:
        case ExprOp::SEGMENTED_REDUCE_MIN:
        case ExprOp::SEGMENTED_REDUCE_MAX:
            break;
        default:
            throw std::runtime_error("Unsupported segmented-reduction op.");
    }

    return make_shared<CompiledSegmentedReduction>(node.op, input_dtype, node.output_dtype.value(), offsets_node.input_tensor_dtype.value());
}

shared_ptr<CompiledScan> EquationCompiler::compileScan(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Scan stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isScanOp(node.op)) {
        throw std::runtime_error("Scan stage output node is not SCAN or SEGMENTED_SCAN.");
    }
    const bool segmented_by_offsets = node.op == ExprOp::SEGMENTED_SCAN;
    const uint64_t expected_inputs = segmented_by_offsets ? 2 : 1;
    if (expr.numInputs() != expected_inputs) {
        throw std::runtime_error("Scan stage has an unexpected number of inputs.");
    }
    if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Scan node is missing its input.");
    }
    if (segmented_by_offsets && (node.rhs == UINT32_MAX || node.rhs >= expr.nodes.size())) {
        throw std::runtime_error("segmented_scan node is missing its offsets input.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    if (input_node.op != ExprOp::INPUT) {
        throw std::runtime_error("Scan stage input must be a local INPUT node.");
    }

    if (!input_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("Scan input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Scan node missing resolved output_dtype.");
    }
    const bool arg_scan = node.scan_op == ScanOp::ArgMin || node.scan_op == ScanOp::ArgMax;
    if (arg_scan) {
        if (node.output_dtype.value() != DataType::UINT32) {
            throw std::runtime_error("Expression arg scan output dtype must be UINT32.");
        }
    } else if (input_node.input_tensor_dtype.value() != node.output_dtype.value()) {
        throw std::runtime_error("Expression scan currently requires input and output dtypes to match.");
    }
    if (!isCubScanDTypeSupported(input_node.input_tensor_dtype.value())) {
        throw std::runtime_error("Expression scan dtype is not supported by the CUB scan backend.");
    }

    std::optional<DataType> offset_dtype = std::nullopt;
    if (segmented_by_offsets) {
        const ExprNode& offsets_node = expr.nodes[node.rhs];
        if (offsets_node.op != ExprOp::INPUT) {
            throw std::runtime_error("segmented_scan offsets must be a local INPUT node in the scan stage.");
        }
        if (!offsets_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error("segmented_scan offsets node missing resolved input_tensor_dtype.");
        }
        if (!isCubSegmentOffsetDTypeSupported(offsets_node.input_tensor_dtype.value())) {
            throw std::runtime_error("Expression segmented_scan offsets dtype is not supported by the CUB segmented-scan backend.");
        }
        offset_dtype = offsets_node.input_tensor_dtype.value();
    }

    return make_shared<CompiledScan>(node.scan_op,
                                     node.scan_mode,
                                     node.scan_axis,
                                     node.scan_reverse,
                                     segmented_by_offsets,
                                     input_node.input_tensor_dtype.value(),
                                     node.output_dtype.value(),
                                     offset_dtype);
}

shared_ptr<CompiledSoftmax> EquationCompiler::compileSoftmax(const PhysicalExpression& expr) {
    if (expr.numInputs() != 1) {
        throw std::runtime_error("Softmax stage must have exactly one input.");
    }

    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Softmax stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isSoftmaxOp(node.op)) {
        throw std::runtime_error("Softmax stage output node is not SOFTMAX.");
    }

    if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Softmax node is missing its input.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    if (input_node.op != ExprOp::INPUT) {
        throw std::runtime_error("Softmax stage input must be a local INPUT node.");
    }

    if (!input_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("Softmax input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Softmax node missing resolved output_dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.value());
    return make_shared<CompiledSoftmax>(node.softmax_algorithm, node.softmax_mode, supported_input_dtype, node.output_dtype.value());
}

shared_ptr<CompiledRmsNorm> EquationCompiler::compileRmsNorm(const PhysicalExpression& expr) {
    if (expr.numInputs() != 2) {
        throw std::runtime_error("RMSNorm stage must have exactly two inputs: feature input and scale.");
    }
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("RMSNorm stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isRmsNormOp(node.op)) {
        throw std::runtime_error("RMSNorm stage output node is not RMSNORM.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("RMSNorm node is missing input or scale.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    const ExprNode& scale_node = expr.nodes[node.rhs];
    if (input_node.op != ExprOp::INPUT || scale_node.op != ExprOp::INPUT) {
        throw std::runtime_error("RMSNorm stage inputs must be local INPUT nodes.");
    }
    if (!input_node.input_tensor_dtype.has_value() || !scale_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("RMSNorm stage input nodes are missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value() || !node.compute_dtype.has_value()) {
        throw std::runtime_error("RMSNorm node is missing resolved dtype metadata.");
    }
    if (node.rms_norm_normalized_feature_count == 0) {
        throw std::runtime_error("RMSNorm node has zero normalized feature count.");
    }
    if (!(node.rms_norm_epsilon > 0.0)) {
        throw std::runtime_error("RMSNorm node epsilon must be > 0.");
    }

    const DataType input_dtype = input_node.input_tensor_dtype.value();
    const DataType scale_dtype = scale_node.input_tensor_dtype.value();
    const DataType output_dtype = node.output_dtype.value();
    const DataType compute_dtype = toSupportedComputeDType(node.op, node.compute_dtype.value());

    auto compiled = make_shared<CompiledRmsNorm>();
    compiled->normalized_feature_count = node.rms_norm_normalized_feature_count;
    compiled->epsilon = node.rms_norm_epsilon;
    compiled->input_dtype = input_dtype;
    compiled->scale_dtype = scale_dtype;
    compiled->output_dtype = output_dtype;
    compiled->compute_dtype = compute_dtype;
    compiled->fused_activation = node.rms_norm_fused_activation;
    compiled->debug_name =
        node.rms_norm_fused_activation == CudnnRmsNormFusedActivation::SWISH ? "thor_expr_rms_norm_swish" : "thor_expr_rms_norm";

    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = 1;
    descriptor.normalizedFeatureCount = compiled->normalized_feature_count;
    descriptor.inputDataType = input_dtype;
    descriptor.parameterDataType = scale_dtype;
    descriptor.outputDataType = output_dtype;
    descriptor.computeDataType = compute_dtype;
    descriptor.epsilon = static_cast<float>(compiled->epsilon);
    descriptor.training = false;
    descriptor.fusedActivation = compiled->fused_activation;
    descriptor.debugName = compiled->debug_name;
    descriptor.validateForward();
    return compiled;
}


static uint32_t findLocalEmbeddingLookupNode(const PhysicalExpression& expr) {
    uint32_t found = UINT32_MAX;
    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        if (isEmbeddingLookupOp(expr.nodes[i].op)) {
            if (found != UINT32_MAX) {
                throw std::runtime_error("EmbeddingLookup stage may contain only one embedding root.");
            }
            found = i;
        }
    }
    if (found == UINT32_MAX) {
        throw std::runtime_error("EmbeddingLookup stage is missing its embedding root.");
    }
    return found;
}

static std::string embeddingScalarLiteral(double value) {
    std::ostringstream out;
    out.setf(std::ios::scientific);
    out.precision(17);
    out << "ValueT(" << value << "f)";
    return out.str();
}

static std::string compileEmbeddingEpilogueExpression(const PhysicalExpression& expr,
                                                      uint32_t local_idx,
                                                      uint32_t local_embedding_idx,
                                                      std::unordered_set<uint32_t>& visiting) {
    if (local_idx >= expr.nodes.size()) {
        throw std::runtime_error("Embedding epilogue node index out of range.");
    }
    if (!visiting.insert(local_idx).second) {
        throw std::runtime_error("Embedding epilogue expression contains a cycle.");
    }

    const ExprNode& node = expr.nodes[local_idx];
    std::string result;
    if (local_idx == local_embedding_idx) {
        result = "v";
    } else if (node.op == ExprOp::INPUT) {
        if (node.input_slot < 2) {
            throw std::runtime_error("Embedding epilogue may not reference raw indices or weights inputs.");
        }
        result = "arg" + std::to_string(node.input_slot - 2) + "[linear]";
    } else if (node.op == ExprOp::SCALAR_FP) {
        result = embeddingScalarLiteral(node.scalar_fp);
    } else if (node.op == ExprOp::NEG) {
        result = "(-" + compileEmbeddingEpilogueExpression(expr, node.lhs, local_embedding_idx, visiting) + ")";
    } else if (node.op == ExprOp::ADD || node.op == ExprOp::SUB || node.op == ExprOp::MUL || node.op == ExprOp::DIV) {
        const std::string lhs = compileEmbeddingEpilogueExpression(expr, node.lhs, local_embedding_idx, visiting);
        const std::string rhs = compileEmbeddingEpilogueExpression(expr, node.rhs, local_embedding_idx, visiting);
        const char* op = nullptr;
        switch (node.op) {
            case ExprOp::ADD: op = " + "; break;
            case ExprOp::SUB: op = " - "; break;
            case ExprOp::MUL: op = " * "; break;
            case ExprOp::DIV: op = " / "; break;
            default: break;
        }
        result = "(" + lhs + op + rhs + ")";
    } else {
        throw std::runtime_error("Embedding epilogue supports only add/sub/mul/div/neg with same-shape tensor inputs and scalar constants.");
    }

    visiting.erase(local_idx);
    return result;
}

shared_ptr<CompiledEmbeddingLookup> EquationCompiler::compileEmbeddingLookup(const PhysicalExpression& expr) {
    if (expr.numInputs() < 2) {
        throw std::runtime_error("EmbeddingLookup stage must have at least two inputs: indices and weights.");
    }
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("EmbeddingLookup stage output_node is out of range.");
    }

    const uint32_t embedding_node_idx = findLocalEmbeddingLookupNode(expr);
    const ExprNode& lookup_node = expr.nodes[embedding_node_idx];
    if (lookup_node.lhs == UINT32_MAX || lookup_node.rhs == UINT32_MAX || lookup_node.lhs >= expr.nodes.size() ||
        lookup_node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("EmbeddingLookup node is missing indices or weights.");
    }

    const ExprNode& indices_node = expr.nodes[lookup_node.lhs];
    const ExprNode& weights_node = expr.nodes[lookup_node.rhs];
    if (indices_node.op != ExprOp::INPUT || weights_node.op != ExprOp::INPUT || indices_node.input_slot != 0 ||
        weights_node.input_slot != 1) {
        throw std::runtime_error("EmbeddingLookup stage inputs must begin with local INPUT nodes for indices and weights.");
    }
    if (!indices_node.input_tensor_dtype.has_value() || !weights_node.input_tensor_dtype.has_value() ||
        !lookup_node.output_dtype.has_value()) {
        throw std::runtime_error("EmbeddingLookup stage input/output nodes are missing resolved dtype metadata.");
    }
    const ExprNode& output_node = expr.nodes[expr.output_node];
    if (!output_node.output_dtype.has_value()) {
        throw std::runtime_error("EmbeddingLookup stage output node is missing dtype metadata.");
    }

    auto compiled = make_shared<CompiledEmbeddingLookup>();
    compiled->has_padding_index = lookup_node.embedding_has_padding_index;
    compiled->padding_index = lookup_node.embedding_padding_index;
    compiled->index_dtype = indices_node.input_tensor_dtype.value();
    compiled->weights_dtype = weights_node.input_tensor_dtype.value();
    compiled->output_dtype = output_node.output_dtype.value();
    compiled->debug_name = expr.output_node == embedding_node_idx ? "thor_expr_embedding_lookup" : "thor_expr_embedding_lookup_fused";

    if (compiled->index_dtype != DataType::UINT8 && compiled->index_dtype != DataType::UINT16 &&
        compiled->index_dtype != DataType::UINT32 && compiled->index_dtype != DataType::UINT64) {
        throw std::runtime_error("EmbeddingLookup indices dtype must be uint8, uint16, uint32, or uint64.");
    }
    if (compiled->weights_dtype != DataType::FP16 && compiled->weights_dtype != DataType::BF16 &&
        compiled->weights_dtype != DataType::FP32) {
        throw std::runtime_error("EmbeddingLookup weights dtype must be fp16, bf16, or fp32.");
    }
    if (lookup_node.output_dtype.value() != compiled->weights_dtype || compiled->output_dtype != compiled->weights_dtype) {
        throw std::runtime_error("EmbeddingLookup root fusion currently requires lookup, epilogue, and output dtypes to match weights dtype.");
    }

    if (expr.output_node != embedding_node_idx) {
        EmbeddingForwardEpilogue epilogue;
        std::unordered_set<uint32_t> visiting;
        epilogue.expression = compileEmbeddingEpilogueExpression(expr, expr.output_node, embedding_node_idx, visiting);
        for (const NamedInput& input : expr.inputs) {
            if (input.slot < 2) {
                continue;
            }
            bool found = false;
            for (const ExprNode& node : expr.nodes) {
                if (node.op == ExprOp::INPUT && node.input_slot == input.slot) {
                    if (!node.input_tensor_dtype.has_value()) {
                        throw std::runtime_error("Embedding epilogue input is missing dtype metadata.");
                    }
                    if (node.input_tensor_dtype.value() != compiled->output_dtype) {
                        throw std::runtime_error("Embedding epilogue tensor inputs must match output dtype.");
                    }
                    epilogue.extra_input_dtypes.push_back(node.input_tensor_dtype.value());
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("Embedding epilogue input slot missing local INPUT node.");
            }
        }
        compiled->epilogue = std::move(epilogue);
    }

    return compiled;
}

shared_ptr<CompiledMatmul> EquationCompiler::compileMatmul(const PhysicalExpression& expr,
                                                           const std::vector<CompiledStageOutput>& outputs) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Matmul stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isMatmulOp(node.op)) {
        throw std::runtime_error("Matmul stage output node is not a supported matmul/gemm op.");
    }

    const uint32_t min_expected_inputs = node.op == ExprOp::MATMUL ? 2u : 3u;
    if (expr.numInputs() < min_expected_inputs) {
        throw std::runtime_error("Matmul stage must have at least " + std::to_string(min_expected_inputs) + " inputs.");
    }

    auto validate_local_input = [&](uint32_t local_idx, const char* label) -> const ExprNode& {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Matmul stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT) {
            throw std::runtime_error(std::string("Matmul stage ") + label + " input must be a local INPUT node.");
        }
        if (!input_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error(std::string("Matmul stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    auto resolve_dynamic_scale_input_slot = [&](uint32_t local_idx, const char* label, double& scale_fp) -> uint32_t {
        if (local_idx == UINT32_MAX) {
            return UINT32_MAX;
        }
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Matmul stage ") + label + " scale node index is out of range.");
        }
        const ExprNode& scale_node = expr.nodes[local_idx];
        if (scale_node.op == ExprOp::SCALAR_FP) {
            scale_fp *= scale_node.scalar_fp;
            return UINT32_MAX;
        }
        if (scale_node.op != ExprOp::INPUT && scale_node.op != ExprOp::RUNTIME_SCALAR && scale_node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            throw std::runtime_error(std::string("Matmul stage ") + label +
                                     " dynamic scale must be a local INPUT, RUNTIME_SCALAR, TENSOR_RUNTIME_SCALAR, or SCALAR_FP node.");
        }
        if (scale_node.input_slot >= expr.inputs.size()) {
            throw std::runtime_error(std::string("Matmul stage ") + label + " dynamic scale input slot is out of range.");
        }
        return scale_node.input_slot;
    };

    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || (node.op == ExprOp::GEMM && node.aux == UINT32_MAX)) {
        throw std::runtime_error("Matmul/gemm node is missing required input(s).");
    }

    const ExprNode& lhs_input = validate_local_input(node.lhs, "lhs");
    const ExprNode& rhs_input = validate_local_input(node.rhs, "rhs");
    const ExprNode* aux_input = nullptr;
    if (node.op == ExprOp::GEMM) {
        aux_input = &validate_local_input(node.aux, "aux");
    }

    uint32_t epilogue_aux_input_slot = UINT32_MAX;
    std::optional<DataType> epilogue_aux_dtype = std::nullopt;
    if (node.matmul_backward_epilogue != MatmulBackwardEpilogue::Default) {
        if (node.matmul_epilogue_aux == UINT32_MAX) {
            throw std::runtime_error("Matmul backward epilogue requires an aux input node.");
        }
        const ExprNode& epilogue_aux_input = validate_local_input(node.matmul_epilogue_aux, "backward epilogue aux");
        epilogue_aux_input_slot = epilogue_aux_input.input_slot;
        epilogue_aux_dtype = epilogue_aux_input.input_tensor_dtype.value();
    }

    std::optional<DataType> bgrad_output_dtype = std::nullopt;
    for (const CompiledStageOutput& output : outputs) {
        if (output.local_node_idx == UINT32_MAX || output.local_node_idx == expr.output_node) {
            continue;
        }
        if (output.local_node_idx >= expr.nodes.size()) {
            throw std::runtime_error("Matmul stage output local node index is out of range.");
        }
        const ExprNode& output_node = expr.nodes.at(output.local_node_idx);
        if (output_node.op != ExprOp::REDUCE_SUM || output_node.lhs != expr.output_node || output_node.reduction_axes.size() != 1 ||
            output_node.reduction_axes[0] != 0 || output_node.squeeze_axes.size() != 1 || output_node.squeeze_axes[0] != 0) {
            throw std::runtime_error("Matmul stage secondary output must be a reduce_sum(axis=0) bias-gradient output.");
        }
        if (!output_node.output_dtype.has_value()) {
            throw std::runtime_error("Matmul stage bias-gradient output missing resolved output_dtype.");
        }
        if (bgrad_output_dtype.has_value() && bgrad_output_dtype.value() != output_node.output_dtype.value()) {
            throw std::runtime_error("Matmul stage has incompatible bias-gradient output dtypes.");
        }
        bgrad_output_dtype = output_node.output_dtype.value();
    }

    if (bgrad_output_dtype.has_value() && node.matmul_backward_epilogue == MatmulBackwardEpilogue::Default) {
        throw std::runtime_error("Matmul stage bias-gradient output requires a backward epilogue matmul plan.");
    }

    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Matmul/gemm node missing resolved output_dtype.");
    }

    std::vector<DataType> input_dtypes;
    input_dtypes.push_back(lhs_input.input_tensor_dtype.value());
    input_dtypes.push_back(rhs_input.input_tensor_dtype.value());
    if (aux_input != nullptr) {
        input_dtypes.push_back(aux_input->input_tensor_dtype.value());
    }

    const DataType logical_input_dtype = promoteTensorValueDTypes(input_dtypes);
    const DataType supported_input_dtype = toSupportedInputDType(node.op, logical_input_dtype);
    const DataType supported_lhs_dtype = toSupportedInputDType(node.op, lhs_input.input_tensor_dtype.value());
    const DataType supported_rhs_dtype = toSupportedInputDType(node.op, rhs_input.input_tensor_dtype.value());
    const DataType raw_supported_aux_dtype =
        aux_input != nullptr ? toSupportedInputDType(node.op, aux_input->input_tensor_dtype.value()) : node.output_dtype.value();

    auto is_fp8_dtype = [](DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; };
    auto supports_fp8_matmul_plan = [&](DataType lhs_dtype, DataType rhs_dtype, DataType aux_dtype, DataType output_dtype) {
        if (!is_fp8_dtype(lhs_dtype) || !is_fp8_dtype(rhs_dtype)) {
            return false;
        }
        if (lhs_dtype == DataType::FP8_E5M2 && rhs_dtype == DataType::FP8_E5M2) {
            return false;
        }
        if (aux_dtype == DataType::FP32) {
            return output_dtype == DataType::FP32;
        }
        if (aux_dtype == DataType::BF16 || aux_dtype == DataType::FP16) {
            if (output_dtype == aux_dtype || output_dtype == DataType::FP8_E4M3) {
                return true;
            }
            if (output_dtype == DataType::FP8_E5M2) {
                return lhs_dtype != DataType::FP8_E4M3 || rhs_dtype != DataType::FP8_E4M3;
            }
        }
        return false;
    };
    auto supports_regular_matmul_plan = [](DataType lhs_dtype, DataType rhs_dtype, DataType output_dtype) {
        if (lhs_dtype != rhs_dtype) {
            return false;
        }
        if (lhs_dtype == DataType::FP32) {
            return output_dtype == DataType::FP32;
        }
        if (lhs_dtype == DataType::FP16 || lhs_dtype == DataType::BF16) {
            return output_dtype == lhs_dtype || output_dtype == DataType::FP32;
        }
        return false;
    };

    DataType compiled_lhs_dtype = supported_lhs_dtype;
    DataType compiled_rhs_dtype = supported_rhs_dtype;
    DataType compiled_aux_dtype = node.output_dtype.value();

    if (supports_fp8_matmul_plan(supported_lhs_dtype, supported_rhs_dtype, raw_supported_aux_dtype, node.output_dtype.value())) {
        compiled_aux_dtype = raw_supported_aux_dtype;
    } else if (!supports_regular_matmul_plan(supported_lhs_dtype, supported_rhs_dtype, node.output_dtype.value())) {
        throw std::runtime_error(
            "Matmul requested an unsupported operand/output dtype combination and Thor will not implicitly convert matrix "
            "operands. Add an explicit expression conversion or choose a directly supported plan. lhs=" +
            TensorDescriptor::getElementTypeName(lhs_input.input_tensor_dtype.value()) + ", rhs=" +
            TensorDescriptor::getElementTypeName(rhs_input.input_tensor_dtype.value()) + ", output=" +
            TensorDescriptor::getElementTypeName(node.output_dtype.value()) + ".");
    }

    double alpha_scale = node.alpha_fp;
    double beta_scale = node.beta_fp;
    const uint32_t alpha_input_slot = resolve_dynamic_scale_input_slot(node.alpha_node, "alpha", alpha_scale);
    const uint32_t beta_input_slot = resolve_dynamic_scale_input_slot(node.beta_node, "beta", beta_scale);

    return make_shared<CompiledMatmul>(node.op,
                                       node.transpose_lhs,
                                       node.transpose_rhs,
                                       node.transpose_aux,
                                       alpha_scale,
                                       beta_scale,
                                       alpha_input_slot,
                                       beta_input_slot,
                                       supported_input_dtype,
                                       compiled_lhs_dtype,
                                       compiled_rhs_dtype,
                                       compiled_aux_dtype,
                                       node.output_dtype.value(),
                                       node.compute_dtype,
                                       node.matmul_epilogue,
                                       node.matmul_backward_epilogue,
                                       epilogue_aux_input_slot,
                                       epilogue_aux_dtype,
                                       bgrad_output_dtype);
}

static bool isCudnnAttentionTensorDType(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
}

static bool isFp8AttentionTensorDType(DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; }

static bool isFp16OrBf16AttentionTensorDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }

shared_ptr<CompiledAttention> EquationCompiler::compileAttention(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Attention stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isAttentionOp(node.op)) {
        throw std::runtime_error("Attention stage output node is not ATTENTION.");
    }
    const uint32_t expected_attention_inputs =
        3u + (node.attention_use_bias ? 1u : 0u) + (node.attention_use_padding_mask ? 2u : 0u) +
        (node.attention_use_ragged_offsets ? 2u : 0u) + (node.attention_use_paged_kv_cache ? 2u : 0u) +
        (node.attention_dropout_probability > 0.0f ? 2u : 0u) + (node.attention_use_fp8_forward_scaling ? 8u : 0u);
    if (expr.numInputs() != expected_attention_inputs) {
        throw std::runtime_error(
            "Attention stage input count mismatch for q/k/v plus optional bias, optional q/kv sequence lengths, optional ragged offsets, "
            "optional paged-KV page tables, optional dropout seed/offset, and optional FP8 scale/descale/amax tensors.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.aux == UINT32_MAX) {
        throw std::runtime_error("Attention node is missing q/k/v inputs.");
    }

    auto validate_local_input = [&](uint32_t local_idx, const char* label) -> const ExprNode& {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Attention stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT) {
            throw std::runtime_error(std::string("Attention stage ") + label + " input must be a local INPUT node.");
        }
        if (!input_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error(std::string("Attention stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    auto validate_local_dropout_scalar = [&](uint32_t local_idx, const char* label) -> DataType {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Attention stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT && input_node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            throw std::runtime_error(std::string("Attention stage ") + label +
                                     " input must be a local INPUT or TENSOR_RUNTIME_SCALAR node.");
        }
        std::optional<DataType> dtype = input_node.op == ExprOp::INPUT ? input_node.input_tensor_dtype : input_node.output_dtype;
        if (!dtype.has_value()) {
            throw std::runtime_error(std::string("Attention stage ") + label + " input missing resolved dtype.");
        }
        return dtype.value();
    };

    const ExprNode& q_input = validate_local_input(node.lhs, "q");
    const ExprNode& k_input = validate_local_input(node.rhs, "k");
    const ExprNode& v_input = validate_local_input(node.aux, "v");
    const bool has_fp8_tensor =
        q_input.input_tensor_dtype.value() == DataType::FP8_E4M3 || q_input.input_tensor_dtype.value() == DataType::FP8_E5M2 ||
        k_input.input_tensor_dtype.value() == DataType::FP8_E4M3 || k_input.input_tensor_dtype.value() == DataType::FP8_E5M2 ||
        v_input.input_tensor_dtype.value() == DataType::FP8_E4M3 || v_input.input_tensor_dtype.value() == DataType::FP8_E5M2 ||
        (node.output_dtype.has_value() &&
         (node.output_dtype.value() == DataType::FP8_E4M3 || node.output_dtype.value() == DataType::FP8_E5M2));
    if (has_fp8_tensor && !node.attention_use_fp8_forward_scaling) {
        throw std::runtime_error(
            "FP8 attention tensors require explicit FP8 forward scale/descale/amax inputs; use scaledDotProductAttentionFp8Forward().");
    }
    if (node.attention_use_bias) {
        if (node.alpha_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using bias but is missing the bias input.");
        }
        (void)validate_local_input(node.alpha_node, "bias");
    }
    if (node.attention_use_padding_mask) {
        if (node.attention_seq_len_q_node == UINT32_MAX || node.attention_seq_len_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using padding mask but is missing q/kv sequence length inputs.");
        }
        const ExprNode& q_len = validate_local_input(node.attention_seq_len_q_node, "q_seq_len");
        const ExprNode& kv_len = validate_local_input(node.attention_seq_len_kv_node, "kv_seq_len");
        if (q_len.input_tensor_dtype.value() != DataType::INT32 || kv_len.input_tensor_dtype.value() != DataType::INT32) {
            throw std::runtime_error("Attention padding-mask sequence length inputs must be INT32 tensors.");
        }
    }
    if (node.attention_use_ragged_offsets) {
        if (node.attention_ragged_offset_q_node == UINT32_MAX || node.attention_ragged_offset_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using ragged offsets but is missing q/kv ragged offset inputs.");
        }
        const ExprNode& q_offsets = validate_local_input(node.attention_ragged_offset_q_node, "q_ragged_offsets");
        const ExprNode& kv_offsets = validate_local_input(node.attention_ragged_offset_kv_node, "kv_ragged_offsets");
        if (!isCudnnRaggedOffsetDataType(q_offsets.input_tensor_dtype.value()) ||
            !isCudnnRaggedOffsetDataType(kv_offsets.input_tensor_dtype.value())) {
            throw std::runtime_error("Attention ragged offset inputs must be INT32 tensors.");
        }
    }
    if (node.attention_use_paged_kv_cache) {
        if (node.attention_page_table_k_node == UINT32_MAX || node.attention_page_table_v_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using paged KV cache but is missing page-table inputs.");
        }
        const ExprNode& page_k = validate_local_input(node.attention_page_table_k_node, "page_table_k");
        const ExprNode& page_v = validate_local_input(node.attention_page_table_v_node, "page_table_v");
        if (page_k.input_tensor_dtype.value() != DataType::INT32 || page_v.input_tensor_dtype.value() != DataType::INT32) {
            throw std::runtime_error("Attention paged KV page-table inputs must be INT32 tensors.");
        }
    }
    if (node.attention_dropout_probability > 0.0f) {
        if (node.attention_dropout_seed_node == UINT32_MAX || node.attention_dropout_offset_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using dropout but is missing seed/offset inputs.");
        }
        const DataType seed_dtype = validate_local_dropout_scalar(node.attention_dropout_seed_node, "dropout_seed");
        const DataType offset_dtype = validate_local_dropout_scalar(node.attention_dropout_offset_node, "dropout_offset");
        if (seed_dtype != DataType::INT64 || offset_dtype != DataType::INT64) {
            throw std::runtime_error("Attention dropout seed/offset inputs must be INT64 tensors.");
        }
    }
    if (node.attention_use_fp8_forward_scaling) {
        auto validate_fp8_scalar = [&](uint32_t local_idx, const char* label) {
            if (local_idx == UINT32_MAX) {
                throw std::runtime_error(std::string("Attention FP8 forward is missing required '") + label + "' scale/descale/amax input.");
            }
            const ExprNode& input_node = validate_local_input(local_idx, label);
            if (input_node.input_tensor_dtype.value() != DataType::FP32) {
                throw std::runtime_error(std::string("Attention FP8 scalar input '") + label + "' must be an FP32 tensor.");
            }
        };
        validate_fp8_scalar(node.attention_descale_q_node, "descale_q");
        validate_fp8_scalar(node.attention_descale_k_node, "descale_k");
        validate_fp8_scalar(node.attention_descale_v_node, "descale_v");
        validate_fp8_scalar(node.attention_descale_s_node, "descale_s");
        validate_fp8_scalar(node.attention_scale_s_node, "scale_s");
        validate_fp8_scalar(node.attention_scale_o_node, "scale_o");
        validate_fp8_scalar(node.attention_amax_s_node, "amax_s");
        validate_fp8_scalar(node.attention_amax_o_node, "amax_o");
    }

    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Attention node missing resolved output_dtype.");
    }
    if (!node.compute_dtype.has_value()) {
        throw std::runtime_error("Attention node missing resolved compute_dtype.");
    }

    const DataType q_dtype = q_input.input_tensor_dtype.value();
    const DataType k_dtype = k_input.input_tensor_dtype.value();
    const DataType v_dtype = v_input.input_tensor_dtype.value();
    const DataType output_dtype = node.output_dtype.value();
    if (node.compute_dtype.value() != DataType::FP32) {
        throw std::runtime_error("cuDNN attention expression stages require FP32 compute dtype.");
    }
    if (node.attention_use_fp8_forward_scaling) {
        if (!isFp8AttentionTensorDType(q_dtype) || q_dtype != k_dtype || q_dtype != v_dtype || output_dtype != q_dtype) {
            throw std::runtime_error(
                "FP8 attention forward requires q/k/v/output to use the same FP8 dtype in Thor's experimental cuDNN path.");
        }
    } else {
        if (!isFp16OrBf16AttentionTensorDType(q_dtype) || q_dtype != k_dtype || q_dtype != v_dtype || output_dtype != q_dtype) {
            throw std::runtime_error(
                "non-FP8 attention requires q/k/v/output to use the same FP16 or BF16 dtype; insert explicit cast() nodes before attention "
                "instead of relying on hidden attention-stage conversion.");
        }
    }
    if (node.compute_dtype.value() != DataType::FP32) {
        throw std::runtime_error("cuDNN attention expression stages require FP32 compute dtype.");
    }
    if (!isCudnnAttentionTensorDType(node.output_dtype.value())) {
        throw std::runtime_error(
            "cuDNN attention expression stages require FP16, BF16, FP8_E4M3, or FP8_E5M2 output dtype. "
            "Specify AttentionOptions::output_dtype when q/k/v promotion would otherwise resolve to FP32.");
    }

    auto compiled = make_shared<CompiledAttention>();
    compiled->q_layout = node.attention_q_layout;
    compiled->k_layout = node.attention_k_layout;
    compiled->v_layout = node.attention_v_layout;
    compiled->o_layout = node.attention_o_layout;
    compiled->mask_kind = node.attention_mask_kind;
    compiled->diagonal_left_bound = node.attention_diagonal_left_bound;
    compiled->diagonal_right_bound = node.attention_diagonal_right_bound;
    compiled->attention_scale = node.attention_has_scale ? std::optional<float>(node.attention_scale) : std::nullopt;
    compiled->use_alibi_mask = node.attention_use_alibi_mask;
    compiled->use_bias = node.attention_use_bias;
    compiled->use_padding_mask = node.attention_use_padding_mask;
    compiled->use_ragged_offsets = node.attention_use_ragged_offsets;
    compiled->use_paged_kv_cache = node.attention_use_paged_kv_cache;
    compiled->paged_kv_max_sequence_length = node.attention_paged_kv_max_sequence_length;
    compiled->dropout_probability = node.attention_dropout_probability;
    compiled->use_fp8_forward_scaling = node.attention_use_fp8_forward_scaling;
    compiled->compute_dtype = node.compute_dtype.value();
    compiled->output_dtype = node.output_dtype.value();
    return compiled;
}

shared_ptr<CompiledAttentionBackward> EquationCompiler::compileAttentionBackward(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Attention-backward stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isAttentionBackwardOp(node.op)) {
        throw std::runtime_error("Attention-backward stage output node is not an attention backward op.");
    }
    const uint32_t expected_attention_backward_inputs =
        4u + (node.attention_use_bias ? 1u : 0u) + (node.attention_use_padding_mask ? 2u : 0u) +
        (node.attention_use_ragged_offsets ? 2u : 0u) + (node.attention_use_paged_kv_cache ? 2u : 0u) +
        (node.attention_dropout_probability > 0.0f ? 2u : 0u);
    if (expr.numInputs() != expected_attention_backward_inputs) {
        throw std::runtime_error(
            "Attention-backward stage input count mismatch for q/k/v/dO plus optional bias, optional q/kv sequence lengths, optional "
            "ragged offsets, optional paged-KV page tables, and optional dropout seed/offset.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.aux == UINT32_MAX || node.alpha_node == UINT32_MAX) {
        throw std::runtime_error("Attention-backward node is missing q/k/v/dO input(s).");
    }
    if (node.op == ExprOp::ATTENTION_BACKWARD_BIAS && !node.attention_use_bias) {
        throw std::runtime_error("Attention-backward dBias output requested for an unbiased attention node.");
    }
    if (node.attention_use_ragged_offsets && node.attention_use_bias && !experimentalCudnnRaggedBiasBackwardProbeEnabled()) {
        throw std::runtime_error(
            "cuDNN primary SDPA backward does not support ragged offsets with additive bias; ragged additive bias is forward-only "
            "until a supported dBias/backward path is implemented. Set THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD=1 "
            "to bypass this guard for cuDNN support-surface probing only.");
    }

    auto validate_local_input = [&](uint32_t local_idx, const char* label) -> const ExprNode& {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Attention-backward stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT) {
            throw std::runtime_error(std::string("Attention-backward stage ") + label + " input must be a local INPUT node.");
        }
        if (!input_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error(std::string("Attention-backward stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    auto validate_local_dropout_scalar = [&](uint32_t local_idx, const char* label) -> DataType {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Attention-backward stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT && input_node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            throw std::runtime_error(std::string("Attention-backward stage ") + label +
                                     " input must be a local INPUT or TENSOR_RUNTIME_SCALAR node.");
        }
        std::optional<DataType> dtype = input_node.op == ExprOp::INPUT ? input_node.input_tensor_dtype : input_node.output_dtype;
        if (!dtype.has_value()) {
            throw std::runtime_error(std::string("Attention-backward stage ") + label + " input missing resolved dtype.");
        }
        return dtype.value();
    };

    const ExprNode& q = validate_local_input(node.lhs, "q");
    const ExprNode& k = validate_local_input(node.rhs, "k");
    const ExprNode& v = validate_local_input(node.aux, "v");
    const ExprNode& dO = validate_local_input(node.alpha_node, "dO");
    if (node.attention_use_bias) {
        if (node.beta_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using bias but is missing the bias input.");
        }
        (void)validate_local_input(node.beta_node, "bias");
    }
    if (node.attention_use_padding_mask) {
        if (node.attention_seq_len_q_node == UINT32_MAX || node.attention_seq_len_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using padding mask but is missing q/kv sequence length inputs.");
        }
        const ExprNode& q_len = validate_local_input(node.attention_seq_len_q_node, "q_seq_len");
        const ExprNode& kv_len = validate_local_input(node.attention_seq_len_kv_node, "kv_seq_len");
        if (q_len.input_tensor_dtype.value() != DataType::INT32 || kv_len.input_tensor_dtype.value() != DataType::INT32) {
            throw std::runtime_error("Attention-backward padding-mask sequence length inputs must be INT32 tensors.");
        }
    }
    if (node.attention_use_ragged_offsets) {
        if (node.attention_ragged_offset_q_node == UINT32_MAX || node.attention_ragged_offset_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using ragged offsets but is missing q/kv ragged offset inputs.");
        }
        const ExprNode& q_offsets = validate_local_input(node.attention_ragged_offset_q_node, "q_ragged_offsets");
        const ExprNode& kv_offsets = validate_local_input(node.attention_ragged_offset_kv_node, "kv_ragged_offsets");
        if (!isCudnnRaggedOffsetDataType(q_offsets.input_tensor_dtype.value()) ||
            !isCudnnRaggedOffsetDataType(kv_offsets.input_tensor_dtype.value())) {
            throw std::runtime_error("Attention-backward ragged offset inputs must be INT32 tensors.");
        }
    }
    if (node.attention_use_paged_kv_cache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throw std::runtime_error(
            "Attention-backward with paged KV cache is not enabled; the paged KV path is inference-only until training semantics are "
            "defined.");
    }
    if (node.attention_dropout_probability > 0.0f) {
        if (node.attention_dropout_seed_node == UINT32_MAX || node.attention_dropout_offset_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using dropout but is missing seed/offset inputs.");
        }
        const DataType seed_dtype = validate_local_dropout_scalar(node.attention_dropout_seed_node, "dropout_seed");
        const DataType offset_dtype = validate_local_dropout_scalar(node.attention_dropout_offset_node, "dropout_offset");
        if (seed_dtype != DataType::INT64 || offset_dtype != DataType::INT64) {
            throw std::runtime_error("Attention-backward dropout seed/offset inputs must be INT64 tensors.");
        }
    }

    if (!node.compute_dtype.has_value()) {
        throw std::runtime_error("Attention-backward node missing resolved compute_dtype.");
    }
    if (node.compute_dtype.value() != DataType::FP32) {
        throw std::runtime_error("cuDNN attention-backward expression stages require FP32 compute dtype.");
    }
    const DataType q_dtype = q.input_tensor_dtype.value();
    const DataType k_dtype = k.input_tensor_dtype.value();
    const DataType v_dtype = v.input_tensor_dtype.value();
    const DataType dO_dtype = dO.input_tensor_dtype.value();
    if (!isFp16OrBf16AttentionTensorDType(q_dtype) || q_dtype != k_dtype || q_dtype != v_dtype || q_dtype != dO_dtype) {
        throw std::runtime_error(
            "attention backward requires q/k/v/dO to use the same FP16 or BF16 dtype; insert explicit cast() nodes outside "
            "attention instead of relying on hidden attention-stage conversion.");
    }

    auto compiled = make_shared<CompiledAttentionBackward>();
    compiled->q_layout = node.attention_q_layout;
    compiled->k_layout = node.attention_k_layout;
    compiled->v_layout = node.attention_v_layout;
    compiled->o_layout = node.attention_o_layout;
    compiled->mask_kind = node.attention_mask_kind;
    compiled->diagonal_left_bound = node.attention_diagonal_left_bound;
    compiled->diagonal_right_bound = node.attention_diagonal_right_bound;
    compiled->attention_scale = node.attention_has_scale ? std::optional<float>(node.attention_scale) : std::nullopt;
    compiled->use_alibi_mask = node.attention_use_alibi_mask;
    compiled->use_bias = node.attention_use_bias;
    compiled->use_padding_mask = node.attention_use_padding_mask;
    compiled->use_ragged_offsets = node.attention_use_ragged_offsets;
    compiled->use_paged_kv_cache = node.attention_use_paged_kv_cache;
    compiled->paged_kv_max_sequence_length = node.attention_paged_kv_max_sequence_length;
    compiled->dropout_probability = node.attention_dropout_probability;
    compiled->compute_dtype = node.compute_dtype.value();
    compiled->dQ_dtype = q_dtype;
    compiled->dK_dtype = k_dtype;
    compiled->dV_dtype = v_dtype;
    return compiled;
}

shared_ptr<CompiledConvolution> EquationCompiler::compileConvolution(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Convolution stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isConvolutionForwardOp(node.op)) {
        throw std::runtime_error("Convolution stage output node is not a supported convolution forward op.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("Convolution forward node is missing required inputs.");
    }
    if (expr.numInputs() < 2) {
        throw std::runtime_error("Convolution stage must have at least two inputs.");
    }

    auto validate_local_input = [&](uint32_t local_idx, const char* label) -> const ExprNode& {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Convolution stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT) {
            throw std::runtime_error(std::string("Convolution stage ") + label + " input must be a local INPUT node.");
        }
        if (!input_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error(std::string("Convolution stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    const ExprNode& input_node = validate_local_input(node.lhs, "lhs");
    const ExprNode& filter_node = validate_local_input(node.rhs, "rhs");

    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Convolution forward node missing resolved output_dtype.");
    }

    if (!node.compute_dtype.has_value()) {
        throw std::runtime_error("Convolution forward node missing resolved compute_dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.value());
    const DataType supported_filter_dtype = toSupportedInputDType(node.op, filter_node.input_tensor_dtype.value());
    const DataType output_dtype = node.output_dtype.value();
    const DataType compute_dtype = node.compute_dtype.value();

    if (!isThorCudnnConvolutionFloatingDType(supported_input_dtype) || !isThorCudnnConvolutionFloatingDType(supported_filter_dtype) ||
        !isThorCudnnConvolutionFloatingDType(output_dtype)) {
        throw std::runtime_error(std::string(fusedOpTag(node.op)) + " staged path uses cuDNN Frontend; " +
                                 thorCudnnConvolutionFloatingDTypesMessage() + ".");
    }
    if (!convolutionComputeDTypeIsCompatibleWithTensorDTypes({supported_input_dtype, supported_filter_dtype, output_dtype},
                                                             compute_dtype)) {
        throw std::runtime_error(std::string(fusedOpTag(node.op)) +
                                 " staged path received an unsupported cuDNN compute dtype for the tensor dtypes. "
                                 "FP8, BF16, and FP32 tensors require FP32 compute; FP16 tensors support FP16 or FP32 compute.");
    }

    return make_shared<CompiledConvolution>(node.op == ExprOp::CONV3D,
                                            node.conv_stride_d,
                                            node.conv_stride_h,
                                            node.conv_stride_w,
                                            node.conv_pad_d,
                                            node.conv_pad_h,
                                            node.conv_pad_w,
                                            supported_input_dtype,
                                            supported_filter_dtype,
                                            output_dtype,
                                            compute_dtype);
}

shared_ptr<CompiledConvolutionBackward> EquationCompiler::compileConvolutionBackward(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Convolution-backward stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isConvolutionBackwardOp(node.op)) {
        throw std::runtime_error("Convolution-backward stage output node is not a supported convolution backward op.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("Convolution backward node is missing required inputs.");
    }
    if (expr.numInputs() != 2) {
        throw std::runtime_error("Convolution backward stage must have exactly two inputs.");
    }

    auto validate_local_input = [&](uint32_t local_idx, const char* label) -> const ExprNode& {
        if (local_idx >= expr.nodes.size()) {
            throw std::runtime_error(std::string("Convolution backward stage ") + label + " input index is out of range.");
        }
        const ExprNode& input_node = expr.nodes[local_idx];
        if (input_node.op != ExprOp::INPUT) {
            throw std::runtime_error(std::string("Convolution backward stage ") + label + " input must be a local INPUT node.");
        }
        if (!input_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error(std::string("Convolution backward stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    const ExprNode& input_node = validate_local_input(node.lhs, "lhs");
    const ExprNode& grad_node = validate_local_input(node.rhs, "rhs");

    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("Convolution backward node missing resolved output_dtype.");
    }

    if (!node.compute_dtype.has_value()) {
        throw std::runtime_error("Convolution backward node missing resolved compute_dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.value());
    const DataType supported_grad_output_dtype = toSupportedInputDType(node.op, grad_node.input_tensor_dtype.value());
    const DataType output_dtype = node.output_dtype.value();
    const DataType compute_dtype = node.compute_dtype.value();

    if (!isThorCudnnConvolutionFloatingDType(supported_input_dtype) || !isThorCudnnConvolutionFloatingDType(supported_grad_output_dtype) ||
        !isThorCudnnConvolutionFloatingDType(output_dtype)) {
        throw std::runtime_error(std::string(fusedOpTag(node.op)) + " staged path uses cuDNN Frontend; " +
                                 thorCudnnConvolutionFloatingDTypesMessage() + ".");
    }
    if (!convolutionComputeDTypeIsCompatibleWithTensorDTypes({supported_input_dtype, supported_grad_output_dtype, output_dtype},
                                                             compute_dtype)) {
        throw std::runtime_error(std::string(fusedOpTag(node.op)) +
                                 " staged path received an unsupported cuDNN compute dtype for the tensor dtypes. "
                                 "FP8, BF16, and FP32 tensors require FP32 compute; FP16 tensors support FP16 or FP32 compute.");
    }

    return make_shared<CompiledConvolutionBackward>(node.op,
                                                    node.conv_stride_d,
                                                    node.conv_stride_h,
                                                    node.conv_stride_w,
                                                    node.conv_pad_d,
                                                    node.conv_pad_h,
                                                    node.conv_pad_w,
                                                    supported_input_dtype,
                                                    supported_grad_output_dtype,
                                                    output_dtype,
                                                    compute_dtype,
                                                    node.fill_dims);
}

shared_ptr<CompiledReduceMinMaxBackward> EquationCompiler::compileReduceMinMaxBackward(const PhysicalExpression& expr) {
    if (expr.numInputs() != 2) {
        throw std::runtime_error("ReduceMinMaxBackward stage must have exactly two inputs.");
    }

    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("ReduceMinMaxBackward stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (node.op != ExprOp::REDUCE_MIN_BACKWARD && node.op != ExprOp::REDUCE_MAX_BACKWARD) {
        throw std::runtime_error("ReduceMinMaxBackward stage output node is not a supported min/max backward op.");
    }

    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("ReduceMinMaxBackward node is missing an input.");
    }
    if (node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("ReduceMinMaxBackward input node is out of range.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    const ExprNode& grad_node = expr.nodes[node.rhs];
    if (input_node.op != ExprOp::INPUT || grad_node.op != ExprOp::INPUT) {
        throw std::runtime_error("ReduceMinMaxBackward stage inputs must be local INPUT nodes.");
    }

    if (!input_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("ReduceMinMaxBackward input node missing resolved input_tensor_dtype.");
    }
    if (!grad_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("ReduceMinMaxBackward grad input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("ReduceMinMaxBackward node missing resolved output_dtype.");
    }

    const DataType input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.value());

    return make_shared<CompiledReduceMinMaxBackward>(node.op,
                                                     node.reduction_axes,
                                                     node.squeeze_axes,
                                                     input_dtype,
                                                     grad_node.input_tensor_dtype.value(),
                                                     node.output_dtype.value(),
                                                     node.compute_dtype);
}


shared_ptr<CompiledScanMinMaxBackward> EquationCompiler::compileScanMinMaxBackward(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("ScanMinMaxBackward stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (!isScanMinMaxBackwardOp(node.op)) {
        throw std::runtime_error("ScanMinMaxBackward stage output node is not a supported scan min/max backward op.");
    }

    const bool segmented = node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD;
    const uint32_t expected_inputs = segmented ? 3 : 2;
    if (expr.numInputs() != expected_inputs) {
        throw std::runtime_error("ScanMinMaxBackward stage has an unexpected number of inputs.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("ScanMinMaxBackward node is missing input or grad input.");
    }
    if (segmented && (node.aux == UINT32_MAX || node.aux >= expr.nodes.size())) {
        throw std::runtime_error("Segmented ScanMinMaxBackward node is missing offsets input.");
    }

    const ExprNode& input_node = expr.nodes[node.lhs];
    const ExprNode& grad_node = expr.nodes[node.rhs];
    if (input_node.op != ExprOp::INPUT || grad_node.op != ExprOp::INPUT) {
        throw std::runtime_error("ScanMinMaxBackward stage input and grad nodes must be local INPUT nodes.");
    }
    if (!input_node.input_tensor_dtype.has_value() || !grad_node.input_tensor_dtype.has_value()) {
        throw std::runtime_error("ScanMinMaxBackward local input nodes missing resolved dtypes.");
    }
    if (!node.output_dtype.has_value()) {
        throw std::runtime_error("ScanMinMaxBackward node missing resolved output_dtype.");
    }

    const ScanOp value_op = (node.op == ExprOp::SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD) ? ScanOp::Min : ScanOp::Max;
    if (!isCubScanDTypeSupported(input_node.input_tensor_dtype.value())) {
        throw std::runtime_error("ScanMinMaxBackward input dtype is not supported by the CUB arg-scan backend.");
    }
    if (grad_node.input_tensor_dtype.value() != node.output_dtype.value()) {
        throw std::runtime_error("ScanMinMaxBackward grad-output dtype must match output dtype.");
    }
    if (node.output_dtype.value() != DataType::FP32) {
        throw std::runtime_error("ScanMinMaxBackward currently supports only FP32 gradients for the no-atomic flat scatter-add backend.");
    }

    std::optional<DataType> offset_dtype = std::nullopt;
    if (segmented) {
        const ExprNode& offsets_node = expr.nodes[node.aux];
        if (offsets_node.op != ExprOp::INPUT || !offsets_node.input_tensor_dtype.has_value()) {
            throw std::runtime_error("Segmented ScanMinMaxBackward offsets must be a local INPUT node with a resolved dtype.");
        }
        if (!isCubSegmentOffsetDTypeSupported(offsets_node.input_tensor_dtype.value())) {
            throw std::runtime_error("Segmented ScanMinMaxBackward offsets dtype is not supported by CUB.");
        }
        offset_dtype = offsets_node.input_tensor_dtype.value();
    }

    return make_shared<CompiledScanMinMaxBackward>(value_op,
                                                   node.scan_mode,
                                                   node.scan_axis,
                                                   node.scan_reverse,
                                                   segmented,
                                                   input_node.input_tensor_dtype.value(),
                                                   grad_node.input_tensor_dtype.value(),
                                                   node.output_dtype.value(),
                                                   offset_dtype);
}


static bool inputRequiresMaterialization(const ExprNode& node) {
    if (node.op != ExprOp::INPUT) {
        return false;
    }
    if (!node.input_tensor_dtype.has_value() || !node.output_dtype.has_value()) {
        return false;
    }
    return node.input_tensor_dtype.value() != node.output_dtype.value();
}

static void collectFusableRegionStoppingAt(const PhysicalExpression& expr,
                                           uint32_t root_idx,
                                           const std::unordered_set<uint32_t>& forced_boundary_nodes,
                                           std::unordered_set<uint32_t>& region_nodes) {
    if (forced_boundary_nodes.count(root_idx)) {
        throw std::runtime_error("collectFusableRegionStoppingAt root cannot be a forced boundary node.");
    }

    std::vector<uint32_t> stack{root_idx};

    while (!stack.empty()) {
        uint32_t node_idx = stack.back();
        stack.pop_back();

        if (forced_boundary_nodes.count(node_idx)) {
            continue;
        }

        if (!region_nodes.insert(node_idx).second) {
            continue;
        }

        const ExprNode& node = expr.nodes[node_idx];
        if (isStageBoundaryOp(node.op)) {
            throw std::runtime_error("collectFusableRegion called on stage-boundary root.");
        }

        if (Expression::isLeafOp(node.op)) {
            continue;
        }

        uint32_t lhs_idx = node.lhs;
        if (lhs_idx >= expr.nodes.size()) {
            throw std::runtime_error("Invalid lhs node index in expression.");
        }

        const ExprNode& lhs = expr.nodes[lhs_idx];
        if (!isStageBoundaryOp(lhs.op) && !forced_boundary_nodes.count(lhs_idx)) {
            stack.push_back(lhs_idx);
        }

        if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
            uint32_t rhs_idx = node.rhs;
            if (rhs_idx >= expr.nodes.size()) {
                throw std::runtime_error("Invalid rhs node index in expression.");
            }

            const ExprNode& rhs = expr.nodes[rhs_idx];
            if (!isStageBoundaryOp(rhs.op) && !forced_boundary_nodes.count(rhs_idx)) {
                stack.push_back(rhs_idx);
            }
        }

        if (Expression::isTernaryOp(node.op)) {
            uint32_t aux_idx = node.aux;
            if (aux_idx >= expr.nodes.size()) {
                throw std::runtime_error("Invalid aux node index in expression.");
            }

            const ExprNode& aux = expr.nodes[aux_idx];
            if (!isStageBoundaryOp(aux.op) && !forced_boundary_nodes.count(aux_idx)) {
                stack.push_back(aux_idx);
            }
        }
    }
}

static void collectFusableRegion(const PhysicalExpression& expr, uint32_t root_idx, std::unordered_set<uint32_t>& region_nodes) {
    static const std::unordered_set<uint32_t> no_forced_boundaries;
    collectFusableRegionStoppingAt(expr, root_idx, no_forced_boundaries, region_nodes);
}

static bool subgraphContainsLogicalTranspose(const PhysicalExpression& expr, uint32_t root_idx) {
    if (root_idx >= expr.nodes.size()) {
        throw std::runtime_error("subgraphContainsLogicalTranspose root index out of range.");
    }

    std::vector<uint32_t> stack{root_idx};
    std::unordered_set<uint32_t> visited;
    while (!stack.empty()) {
        const uint32_t node_idx = stack.back();
        stack.pop_back();
        if (!visited.insert(node_idx).second) {
            continue;
        }

        const ExprNode& node = expr.nodes[node_idx];
        if (node.op == ExprOp::TRANSPOSE) {
            return true;
        }
        if (Expression::isLeafOp(node.op)) {
            continue;
        }
        if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed expression while searching for logical transpose.");
        }
        stack.push_back(node.lhs);
        if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
            if (node.rhs == UINT32_MAX || node.rhs >= expr.nodes.size()) {
                throw std::runtime_error("Malformed rhs expression while searching for logical transpose.");
            }
            stack.push_back(node.rhs);
        }
        if (Expression::isTernaryOp(node.op)) {
            if (node.aux == UINT32_MAX || node.aux >= expr.nodes.size()) {
                throw std::runtime_error("Malformed aux expression while searching for logical transpose.");
            }
            stack.push_back(node.aux);
        }
    }
    return false;
}

static void collectReachableLogicalTransposeNodes(const PhysicalExpression& expr,
                                                  uint32_t root_idx,
                                                  std::unordered_set<uint32_t>& visited,
                                                  std::vector<uint32_t>& transpose_nodes) {
    if (root_idx >= expr.nodes.size()) {
        throw std::runtime_error("collectReachableLogicalTransposeNodes root index out of range.");
    }
    if (!visited.insert(root_idx).second) {
        return;
    }

    const ExprNode& node = expr.nodes[root_idx];
    if (node.op == ExprOp::TRANSPOSE) {
        transpose_nodes.push_back(root_idx);
    }
    if (Expression::isLeafOp(node.op)) {
        return;
    }
    if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Malformed expression while collecting logical transposes.");
    }
    collectReachableLogicalTransposeNodes(expr, node.lhs, visited, transpose_nodes);
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        if (node.rhs == UINT32_MAX || node.rhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed rhs expression while collecting logical transposes.");
        }
        collectReachableLogicalTransposeNodes(expr, node.rhs, visited, transpose_nodes);
    }
    if (Expression::isTernaryOp(node.op)) {
        if (node.aux == UINT32_MAX || node.aux >= expr.nodes.size()) {
            throw std::runtime_error("Malformed aux expression while collecting logical transposes.");
        }
        collectReachableLogicalTransposeNodes(expr, node.aux, visited, transpose_nodes);
    }
}

static std::vector<uint32_t> collectReachableLogicalTransposeNodes(const PhysicalExpression& expr, uint32_t root_idx) {
    std::unordered_set<uint32_t> visited;
    std::vector<uint32_t> transpose_nodes;
    collectReachableLogicalTransposeNodes(expr, root_idx, visited, transpose_nodes);
    return transpose_nodes;
}

static void collectUnsupportedLogicalTransposeBoundariesImpl(const PhysicalExpression& expr,
                                                             uint32_t root_idx,
                                                             std::unordered_set<uint32_t>& visited,
                                                             std::unordered_set<uint32_t>& forced_boundaries) {
    if (root_idx >= expr.nodes.size()) {
        throw std::runtime_error("collectUnsupportedLogicalTransposeBoundaries root index out of range.");
    }
    if (!visited.insert(root_idx).second) {
        return;
    }

    const ExprNode& node = expr.nodes[root_idx];
    if (node.op == ExprOp::TRANSPOSE) {
        if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed transpose while collecting unsupported logical transpose boundaries.");
        }
        if (subgraphContainsLogicalTranspose(expr, node.lhs)) {
            forced_boundaries.insert(root_idx);
            return;
        }
    }

    if (Expression::isLeafOp(node.op)) {
        return;
    }
    if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Malformed expression while collecting unsupported logical transpose boundaries.");
    }
    collectUnsupportedLogicalTransposeBoundariesImpl(expr, node.lhs, visited, forced_boundaries);
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        if (node.rhs == UINT32_MAX || node.rhs >= expr.nodes.size()) {
            throw std::runtime_error("Malformed rhs expression while collecting unsupported logical transpose boundaries.");
        }
        collectUnsupportedLogicalTransposeBoundariesImpl(expr, node.rhs, visited, forced_boundaries);
    }
    if (Expression::isTernaryOp(node.op)) {
        if (node.aux == UINT32_MAX || node.aux >= expr.nodes.size()) {
            throw std::runtime_error("Malformed aux expression while collecting unsupported logical transpose boundaries.");
        }
        collectUnsupportedLogicalTransposeBoundariesImpl(expr, node.aux, visited, forced_boundaries);
    }
}

static std::unordered_set<uint32_t> collectUnsupportedLogicalTransposeBoundaries(const PhysicalExpression& expr, uint32_t root_idx) {
    std::unordered_set<uint32_t> visited;
    std::unordered_set<uint32_t> forced_boundaries;
    collectUnsupportedLogicalTransposeBoundariesImpl(expr, root_idx, visited, forced_boundaries);
    return forced_boundaries;
}

static void collectBoundaryDependencies(const PhysicalExpression& expr,
                                        const std::unordered_set<uint32_t>& region_nodes,
                                        std::unordered_set<uint32_t>& boundary_nodes) {
    for (uint32_t node_idx : region_nodes) {
        const ExprNode& node = expr.nodes[node_idx];

        if (Expression::isLeafOp(node.op)) {
            continue;
        }

        uint32_t lhs_idx = node.lhs;
        if (lhs_idx >= expr.nodes.size()) {
            throw std::runtime_error("Invalid lhs node index in expression.");
        }
        if (!region_nodes.count(lhs_idx) && isStageBoundaryOp(expr.nodes[lhs_idx].op)) {
            boundary_nodes.insert(lhs_idx);
        }

        if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
            uint32_t rhs_idx = node.rhs;
            if (rhs_idx >= expr.nodes.size()) {
                throw std::runtime_error("Invalid rhs node index in expression.");
            }
            if (!region_nodes.count(rhs_idx) && isStageBoundaryOp(expr.nodes[rhs_idx].op)) {
                boundary_nodes.insert(rhs_idx);
            }
        }

        if (Expression::isTernaryOp(node.op)) {
            uint32_t aux_idx = node.aux;
            if (aux_idx >= expr.nodes.size()) {
                throw std::runtime_error("Invalid aux node index in expression.");
            }
            if (!region_nodes.count(aux_idx) && isStageBoundaryOp(expr.nodes[aux_idx].op)) {
                boundary_nodes.insert(aux_idx);
            }
        }
    }
}

struct RequestedStageOutput {
    std::string name;
    uint32_t old_root_idx;
    uint32_t value_id;
    MaterializedTensorLayout materialized_layout = MaterializedTensorLayout::RowMajor;
};

static PhysicalExecutionStage buildFusedStage(const PhysicalExpression& expr,
                                              const std::unordered_set<uint32_t>& region_nodes,
                                              const std::vector<RequestedStageOutput>& requested_outputs,
                                              const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    if (requested_outputs.empty()) {
        throw std::runtime_error("buildFusedStage requires at least one requested output.");
    }

    std::vector<uint32_t> sorted_nodes(region_nodes.begin(), region_nodes.end());
    std::sort(sorted_nodes.begin(), sorted_nodes.end());

    PhysicalExpression stage_expr;
    stage_expr.nodes.reserve(sorted_nodes.size());

    std::unordered_map<uint32_t, uint32_t> old_to_new_node_idx;

    std::vector<uint32_t> stage_input_value_ids;
    std::unordered_map<uint32_t, uint32_t> value_id_to_local_input_slot;

    auto getOrCreateLocalInputSlot =
        [&](uint32_t value_id, NamedInput::Kind kind, const std::optional<std::string>& preferred_name = std::nullopt) -> uint32_t {
        auto it = value_id_to_local_input_slot.find(value_id);
        if (it != value_id_to_local_input_slot.end()) {
            return it->second;
        }

        uint32_t local_slot = static_cast<uint32_t>(stage_input_value_ids.size());
        stage_input_value_ids.push_back(value_id);
        value_id_to_local_input_slot.emplace(value_id, local_slot);

        NamedInput input;
        input.name = preferred_name.has_value() ? *preferred_name : ("__arg" + std::to_string(local_slot));
        input.slot = local_slot;
        input.kind = kind;
        stage_expr.inputs.push_back(std::move(input));

        return local_slot;
    };

    for (uint32_t old_idx : sorted_nodes) {
        ExprNode new_node = expr.nodes[old_idx];

        if (new_node.op == ExprOp::INPUT || new_node.op == ExprOp::RUNTIME_SCALAR || new_node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            uint32_t value_id = new_node.input_slot;
            const NamedInput& root_input = expr.inputs.at(value_id);
            new_node.input_slot = getOrCreateLocalInputSlot(value_id, root_input.kind, root_input.name);
        } else {
            auto remapParent = [&](uint32_t old_parent, const char* field_name) -> uint32_t {
                auto it = old_to_new_node_idx.find(old_parent);
                if (it != old_to_new_node_idx.end()) {
                    return it->second;
                }

                auto out_it = node_output_value_id.find(old_parent);
                if (out_it == node_output_value_id.end()) {
                    throw std::runtime_error(std::string("Missing value id for fused stage ") + field_name + " boundary input.");
                }

                ExprNode input_node;
                input_node.op = ExprOp::INPUT;
                input_node.input_slot = getOrCreateLocalInputSlot(out_it->second, NamedInput::Kind::Tensor);

                // This local INPUT stands in for the already-materialized parent value
                // crossing a stage boundary, so it should inherit that value's dtype semantics.
                input_node.input_tensor_dtype = expr.nodes[old_parent].output_dtype;
                input_node.output_dtype = expr.nodes[old_parent].output_dtype;
                input_node.compute_dtype = expr.nodes[old_parent].compute_dtype;
                input_node.backward_output_dtype = expr.nodes[old_parent].backward_output_dtype;
                input_node.backward_compute_dtype = expr.nodes[old_parent].backward_compute_dtype;

                uint32_t new_input_idx = static_cast<uint32_t>(stage_expr.nodes.size());
                stage_expr.nodes.push_back(std::move(input_node));
                old_to_new_node_idx[old_parent] = new_input_idx;
                return new_input_idx;
            };

            if (!Expression::isLeafOp(new_node.op)) {
                new_node.lhs = remapParent(new_node.lhs, "lhs");
            }

            if (Expression::isBinaryOp(new_node.op) || Expression::isTernaryOp(new_node.op)) {
                new_node.rhs = remapParent(new_node.rhs, "rhs");
            }

            if (Expression::isTernaryOp(new_node.op)) {
                new_node.aux = remapParent(new_node.aux, "aux");
            }
        }

        uint32_t new_idx = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(new_node));
        old_to_new_node_idx[old_idx] = new_idx;
    }

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.reserve(requested_outputs.size());

    for (const RequestedStageOutput& requested : requested_outputs) {
        auto it = old_to_new_node_idx.find(requested.old_root_idx);
        if (it == old_to_new_node_idx.end()) {
            throw std::runtime_error("Failed to remap fused stage output node.");
        }

        stage_outputs.push_back(CompiledStageOutput{
            .name = requested.name,
            .local_node_idx = it->second,
            .value_id = requested.value_id,
            .materialized_layout = requested.materialized_layout,
        });
    }

    if (!stage_outputs.empty()) {
        stage_expr.output_node = stage_outputs.front().local_node_idx;
    }

    deduplicateFusedStageExpr(stage_expr, stage_outputs);
    compactFusedStageInputs(stage_expr, stage_input_value_ids);

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(stage_input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static PhysicalExecutionStage buildCudaKernelStage(const PhysicalExpression& expr,
                                                   const std::vector<RequestedStageOutput>& requested_outputs,
                                                   const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    if (requested_outputs.empty()) {
        throw std::runtime_error("buildCudaKernelStage requires at least one requested output.");
    }

    const ExprNode& first_output = expr.nodes.at(requested_outputs.front().old_root_idx);
    if (first_output.op != ExprOp::CUDA_KERNEL_OUTPUT) {
        throw std::runtime_error("buildCudaKernelStage called on non-CUDA-kernel output node.");
    }
    const uint32_t spec_idx = first_output.cuda_kernel_spec_index;
    if (spec_idx >= expr.cuda_kernel_expressions.size() || !expr.cuda_kernel_expressions[spec_idx]) {
        throw std::runtime_error("buildCudaKernelStage references missing CUDA kernel spec.");
    }

    PhysicalExpression stage_expr;
    stage_expr.cuda_kernel_expressions.push_back(expr.cuda_kernel_expressions[spec_idx]);

    std::vector<uint32_t> stage_input_value_ids;
    std::vector<uint32_t> local_input_node_indices;
    local_input_node_indices.reserve(first_output.cuda_kernel_input_nodes.size());

    for (uint32_t input_idx : first_output.cuda_kernel_input_nodes) {
        if (input_idx >= expr.nodes.size()) {
            throw std::runtime_error("buildCudaKernelStage input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[input_idx];
        uint32_t value_id = UINT32_MAX;
        NamedInput::Kind input_kind = NamedInput::Kind::Tensor;
        std::string local_name = "__arg" + std::to_string(stage_input_value_ids.size());
        if ((parent.op == ExprOp::INPUT || parent.op == ExprOp::RUNTIME_SCALAR || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR) &&
            !inputRequiresMaterialization(parent)) {
            value_id = parent.input_slot;
            if (value_id >= expr.inputs.size()) {
                throw std::runtime_error("buildCudaKernelStage root input slot out of range.");
            }
            input_kind = expr.inputs[value_id].kind;
            local_name = expr.inputs[value_id].name;
        } else {
            auto value_it = node_output_value_id.find(input_idx);
            if (value_it == node_output_value_id.end()) {
                throw std::runtime_error("buildCudaKernelStage missing value id for kernel input dependency.");
            }
            value_id = value_it->second;
        }

        const uint32_t local_slot = static_cast<uint32_t>(stage_input_value_ids.size());
        stage_input_value_ids.push_back(value_id);
        stage_expr.inputs.push_back(NamedInput{local_name, local_slot, input_kind});

        ExprNode local_input;
        local_input.op = parent.op == ExprOp::TENSOR_RUNTIME_SCALAR ? ExprOp::TENSOR_RUNTIME_SCALAR
                         : parent.op == ExprOp::RUNTIME_SCALAR      ? ExprOp::RUNTIME_SCALAR
                                                                    : ExprOp::INPUT;
        local_input.input_slot = local_slot;
        local_input.input_tensor_dtype = parent.output_dtype;
        local_input.output_dtype = parent.output_dtype;
        local_input.compute_dtype = parent.compute_dtype;
        local_input.backward_output_dtype = parent.backward_output_dtype;
        local_input.backward_compute_dtype = parent.backward_compute_dtype;

        const uint32_t local_node_idx = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(local_input));
        local_input_node_indices.push_back(local_node_idx);
    }

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.reserve(requested_outputs.size());
    for (const RequestedStageOutput& requested : requested_outputs) {
        const ExprNode& old_node = expr.nodes.at(requested.old_root_idx);
        if (old_node.op != ExprOp::CUDA_KERNEL_OUTPUT || old_node.cuda_kernel_spec_index != spec_idx) {
            throw std::runtime_error("buildCudaKernelStage requested output does not belong to the same CUDA kernel spec.");
        }
        if (old_node.cuda_kernel_input_nodes != first_output.cuda_kernel_input_nodes) {
            throw std::runtime_error("buildCudaKernelStage requested outputs have mismatched input ABI nodes.");
        }

        ExprNode new_node = old_node;
        new_node.cuda_kernel_spec_index = 0;
        new_node.cuda_kernel_input_nodes = local_input_node_indices;
        const uint32_t local_node_idx = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(new_node));
        stage_outputs.push_back(CompiledStageOutput{
            .name = requested.name,
            .local_node_idx = local_node_idx,
            .value_id = requested.value_id,
            .materialized_layout = requested.materialized_layout,
        });
    }

    return PhysicalExecutionStage{
        PhysicalExecutionStage::Kind::CudaKernel, std::move(stage_expr), std::move(stage_input_value_ids), std::move(stage_outputs), {}};
}

static PhysicalExecutionStage buildReductionStage(const PhysicalExpression& expr,
                                                  uint32_t node_idx,
                                                  uint32_t output_value_id,
                                                  const std::string& output_name,
                                                  const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isReductionOp(node.op)) {
        throw std::runtime_error("buildReductionStage called on non-reduction node.");
    }

    if (node.lhs == UINT32_MAX) {
        throw std::runtime_error("Reduction node missing lhs input.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});

    ExprNode reduction = node;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(1);
    std::optional<DataType> actual_input_dtype = std::nullopt;

    uint32_t parent_idx = reduction.lhs;
    if (parent_idx >= expr.nodes.size()) {
        throw std::runtime_error("Reduction input node index out of range.");
    }

    const ExprNode& parent = expr.nodes[parent_idx];
    auto out_it = node_output_value_id.find(parent_idx);
    if (out_it != node_output_value_id.end()) {
        input_value_ids.push_back(out_it->second);
        actual_input_dtype = parent.output_dtype;
    } else if (parent.op == ExprOp::INPUT) {
        input_value_ids.push_back(parent.input_slot);
        actual_input_dtype = parent.input_tensor_dtype;
    } else {
        throw std::runtime_error("Missing value id for reduction input.");
    }

    if (!parent.output_dtype.has_value()) {
        throw std::runtime_error("Reduction parent node is missing resolved output_dtype.");
    }
    if (!actual_input_dtype.has_value()) {
        throw std::runtime_error("Reduction parent node is missing resolved actual input dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, actual_input_dtype.value());

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;

    // This local INPUT node represents the already-materialized value produced by the parent expression feeding the
    // reduction. Dense value and arg reductions consume that storage dtype directly; FP32 conversion happens lazily
    // inside the central CUB reduction iterator rather than through a materialized compatibility tensor.
    input_node.input_tensor_dtype = supported_input_dtype;
    input_node.output_dtype = supported_input_dtype;
    input_node.compute_dtype = defaultComputeDType(supported_input_dtype);
    input_node.backward_output_dtype = supported_input_dtype;
    input_node.backward_compute_dtype = defaultComputeDType(supported_input_dtype);

    stage_expr.nodes.push_back(std::move(input_node));

    reduction.lhs = 0;
    reduction.rhs = UINT32_MAX;
    reduction.reduction_axes = node.reduction_axes;
    reduction.squeeze_axes = node.squeeze_axes;
    reduction.compute_dtype = node.compute_dtype;

    stage_expr.nodes.push_back(std::move(reduction));
    stage_expr.output_node = 1;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = 1,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = isArgMinMaxOp(node.op) ? PhysicalExecutionStage::Kind::ArgMinMax : PhysicalExecutionStage::Kind::Reduction,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static bool isArgScanExpressionOp(ScanOp op) { return op == ScanOp::ArgMin || op == ScanOp::ArgMax; }

static bool isValueScanExpressionOp(ScanOp op) { return op == ScanOp::Min || op == ScanOp::Max; }

static bool scanOpsAreValueIndexPair(ScanOp a, ScanOp b) {
    return (a == ScanOp::Min && b == ScanOp::ArgMin) || (a == ScanOp::ArgMin && b == ScanOp::Min) ||
           (a == ScanOp::Max && b == ScanOp::ArgMax) || (a == ScanOp::ArgMax && b == ScanOp::Max);
}

static bool scanNodesCanShareValueIndexStage(const ExprNode& a, const ExprNode& b) {
    if (!isScanOp(a.op) || !isScanOp(b.op) || a.op != b.op) {
        return false;
    }
    if (!scanOpsAreValueIndexPair(a.scan_op, b.scan_op)) {
        return false;
    }
    return a.lhs == b.lhs && a.rhs == b.rhs && a.scan_mode == b.scan_mode && a.scan_axis == b.scan_axis &&
           a.scan_reverse == b.scan_reverse;
}

static uint32_t findPairedScanNode(const PhysicalExpression& expr, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size()) {
        return UINT32_MAX;
    }
    const ExprNode& root = expr.nodes[node_idx];
    if (!isScanOp(root.op) || (!isValueScanExpressionOp(root.scan_op) && !isArgScanExpressionOp(root.scan_op))) {
        return UINT32_MAX;
    }
    for (uint32_t candidate_idx = 0; candidate_idx < expr.nodes.size(); ++candidate_idx) {
        if (candidate_idx == node_idx) {
            continue;
        }
        if (scanNodesCanShareValueIndexStage(root, expr.nodes[candidate_idx])) {
            return candidate_idx;
        }
    }
    return UINT32_MAX;
}

static PhysicalExecutionStage buildSegmentedReductionStage(const PhysicalExpression& expr,
                                                              uint32_t node_idx,
                                                              uint32_t output_value_id,
                                                              const std::string& output_name,
                                                              const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes.at(node_idx);
    if (!isSegmentedReduceOp(node.op)) {
        throw std::runtime_error("buildSegmentedReductionStage called on non-segmented-reduction node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("Segmented-reduction node missing values or offsets input.");
    }

    auto resolve_stage_parent = [&](uint32_t parent_idx, const char* what) -> std::pair<uint32_t, DataType> {
        const ExprNode& parent = expr.nodes[parent_idx];
        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            if (!parent.output_dtype.has_value()) {
                throw std::runtime_error(std::string("Segmented-reduction ") + what + " parent node is missing output dtype.");
            }
            return {out_it->second, parent.output_dtype.value()};
        }
        if (parent.op == ExprOp::INPUT) {
            if (!parent.input_tensor_dtype.has_value()) {
                throw std::runtime_error(std::string("Segmented-reduction ") + what + " input node is missing input dtype.");
            }
            return {parent.input_slot, parent.input_tensor_dtype.value()};
        }
        throw std::runtime_error(std::string("Missing value id for segmented-reduction ") + what + ".");
    };

    const auto [values_value_id, values_dtype] = resolve_stage_parent(node.lhs, "values");
    const auto [offsets_value_id, offsets_dtype] = resolve_stage_parent(node.rhs, "offsets");
    if (!isCubSegmentOffsetDTypeSupported(offsets_dtype)) {
        throw std::runtime_error("Expression segmented-reduction offsets dtype is not supported by CUB.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});
    ExprNode values_node;
    values_node.op = ExprOp::INPUT;
    values_node.input_slot = 0;
    values_node.input_tensor_dtype = values_dtype;
    values_node.output_dtype = values_dtype;
    values_node.compute_dtype = values_dtype;
    values_node.backward_output_dtype = values_dtype;
    values_node.backward_compute_dtype = values_dtype;
    stage_expr.nodes.push_back(std::move(values_node));

    stage_expr.inputs.push_back(NamedInput{"__arg1", 1});
    ExprNode offsets_node;
    offsets_node.op = ExprOp::INPUT;
    offsets_node.input_slot = 1;
    offsets_node.input_tensor_dtype = offsets_dtype;
    offsets_node.output_dtype = offsets_dtype;
    offsets_node.compute_dtype = offsets_dtype;
    offsets_node.backward_output_dtype = offsets_dtype;
    offsets_node.backward_compute_dtype = offsets_dtype;
    stage_expr.nodes.push_back(std::move(offsets_node));

    ExprNode reduction = node;
    reduction.lhs = 0;
    reduction.rhs = 1;
    const uint32_t local_node_idx = static_cast<uint32_t>(stage_expr.nodes.size());
    stage_expr.nodes.push_back(std::move(reduction));
    stage_expr.output_node = local_node_idx;

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::SegmentedReduction,
        .expr = std::move(stage_expr),
        .input_value_ids = {values_value_id, offsets_value_id},
        .outputs = {CompiledStageOutput{.name = output_name, .local_node_idx = local_node_idx, .value_id = output_value_id}},
    };
}

static PhysicalExecutionStage buildScanStage(const PhysicalExpression& expr,
                                            const std::vector<RequestedStageOutput>& requested_outputs,
                                            const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    if (requested_outputs.empty() || requested_outputs.size() > 2) {
        throw std::runtime_error("buildScanStage requires one scan output, or a paired value/index scan output.");
    }

    const ExprNode& first_node = expr.nodes.at(requested_outputs.front().old_root_idx);
    if (!isScanOp(first_node.op)) {
        throw std::runtime_error("buildScanStage called on non-scan node.");
    }
    if (requested_outputs.size() == 2) {
        const ExprNode& second_node = expr.nodes.at(requested_outputs[1].old_root_idx);
        if (!scanNodesCanShareValueIndexStage(first_node, second_node)) {
            throw std::runtime_error("buildScanStage paired outputs must be matching min/arg_min or max/arg_max scans.");
        }
    }

    const ExprNode& node = first_node;
    if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Scan node missing lhs input.");
    }
    if (node.op == ExprOp::SEGMENTED_SCAN && (node.rhs == UINT32_MAX || node.rhs >= expr.nodes.size())) {
        throw std::runtime_error("segmented_scan node missing offsets input.");
    }

    auto resolve_stage_parent = [&](uint32_t parent_idx, const char* what) -> std::pair<uint32_t, DataType> {
        const ExprNode& parent = expr.nodes[parent_idx];
        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            if (!parent.output_dtype.has_value()) {
                throw std::runtime_error(std::string("Scan ") + what + " parent node is missing resolved output dtype.");
            }
            return {out_it->second, parent.output_dtype.value()};
        }
        if (parent.op == ExprOp::INPUT) {
            if (!parent.input_tensor_dtype.has_value()) {
                throw std::runtime_error(std::string("Scan ") + what + " input node is missing resolved input dtype.");
            }
            return {parent.input_slot, parent.input_tensor_dtype.value()};
        }
        throw std::runtime_error(std::string("Missing value id for scan ") + what + ".");
    };

    const auto [input_value_id, actual_input_dtype] = resolve_stage_parent(node.lhs, "input");

    for (const RequestedStageOutput& requested : requested_outputs) {
        const ExprNode& requested_node = expr.nodes.at(requested.old_root_idx);
        if (!requested_node.output_dtype.has_value()) {
            throw std::runtime_error("Scan node is missing resolved output dtype.");
        }
        const bool arg_scan = requested_node.scan_op == ScanOp::ArgMin || requested_node.scan_op == ScanOp::ArgMax;
        if (arg_scan) {
            if (requested_node.output_dtype.value() != DataType::UINT32) {
                throw std::runtime_error("Expression arg scan output dtype must be UINT32.");
            }
        } else if (actual_input_dtype != requested_node.output_dtype.value()) {
            throw std::runtime_error("Expression scan currently requires input and output dtypes to match.");
        }
    }

    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(node.op == ExprOp::SEGMENTED_SCAN ? 2 : 1);
    input_value_ids.push_back(input_value_id);

    std::optional<DataType> offsets_dtype = std::nullopt;
    if (node.op == ExprOp::SEGMENTED_SCAN) {
        const auto [offsets_value_id, actual_offsets_dtype] = resolve_stage_parent(node.rhs, "offsets");
        if (!isCubSegmentOffsetDTypeSupported(actual_offsets_dtype)) {
            throw std::runtime_error("Expression segmented_scan offsets dtype is not supported by the CUB segmented-scan backend.");
        }
        input_value_ids.push_back(offsets_value_id);
        offsets_dtype = actual_offsets_dtype;
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;
    input_node.input_tensor_dtype = actual_input_dtype;
    input_node.output_dtype = actual_input_dtype;
    input_node.compute_dtype = actual_input_dtype;
    input_node.backward_output_dtype = actual_input_dtype;
    input_node.backward_compute_dtype = actual_input_dtype;
    stage_expr.nodes.push_back(std::move(input_node));

    if (node.op == ExprOp::SEGMENTED_SCAN) {
        stage_expr.inputs.push_back(NamedInput{"__arg1", 1});
        ExprNode offsets_node;
        offsets_node.op = ExprOp::INPUT;
        offsets_node.input_slot = 1;
        offsets_node.input_tensor_dtype = offsets_dtype.value();
        offsets_node.output_dtype = offsets_dtype.value();
        offsets_node.compute_dtype = offsets_dtype.value();
        offsets_node.backward_output_dtype = offsets_dtype.value();
        offsets_node.backward_compute_dtype = offsets_dtype.value();
        stage_expr.nodes.push_back(std::move(offsets_node));
    }

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.reserve(requested_outputs.size());
    uint32_t arg_local_node_idx = UINT32_MAX;

    for (const RequestedStageOutput& requested : requested_outputs) {
        ExprNode scan = expr.nodes.at(requested.old_root_idx);
        scan.lhs = 0;
        scan.rhs = node.op == ExprOp::SEGMENTED_SCAN ? 1 : UINT32_MAX;
        const uint32_t local_node_idx = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(scan));
        if (isArgScanExpressionOp(stage_expr.nodes.back().scan_op)) {
            arg_local_node_idx = local_node_idx;
        }
        stage_outputs.push_back(CompiledStageOutput{
            .name = requested.name,
            .local_node_idx = local_node_idx,
            .value_id = requested.value_id,
        });
    }

    if (requested_outputs.size() == 2 && arg_local_node_idx == UINT32_MAX) {
        throw std::runtime_error("Paired scan stage is missing its arg scan output.");
    }
    stage_expr.output_node = requested_outputs.size() == 2 ? arg_local_node_idx : stage_outputs.front().local_node_idx;

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::Scan,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static PhysicalExecutionStage buildScanStage(const PhysicalExpression& expr,
                                            uint32_t node_idx,
                                            uint32_t output_value_id,
                                            const std::string& output_name,
                                            const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    return buildScanStage(expr,
                          std::vector<RequestedStageOutput>{RequestedStageOutput{
                              .name = output_name,
                              .old_root_idx = node_idx,
                              .value_id = output_value_id,
                          }},
                          node_output_value_id);
}

static PhysicalExecutionStage buildSoftmaxStage(const PhysicalExpression& expr,
                                                uint32_t node_idx,
                                                uint32_t output_value_id,
                                                const std::string& output_name,
                                                const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isSoftmaxOp(node.op)) {
        throw std::runtime_error("buildSoftmaxStage called on non-softmax node.");
    }

    if (node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Softmax node missing lhs input.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});

    ExprNode softmax = node;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(1);
    std::optional<DataType> actual_input_dtype = std::nullopt;

    const uint32_t parent_idx = softmax.lhs;
    const ExprNode& parent = expr.nodes[parent_idx];
    auto out_it = node_output_value_id.find(parent_idx);
    if (out_it != node_output_value_id.end()) {
        input_value_ids.push_back(out_it->second);
        actual_input_dtype = parent.output_dtype;
    } else if (parent.op == ExprOp::INPUT) {
        input_value_ids.push_back(parent.input_slot);
        actual_input_dtype = parent.input_tensor_dtype;
    } else {
        throw std::runtime_error("Missing value id for softmax input.");
    }

    if (!parent.output_dtype.has_value()) {
        throw std::runtime_error("Softmax parent node is missing resolved output_dtype.");
    }
    if (!actual_input_dtype.has_value()) {
        throw std::runtime_error("Softmax parent node is missing resolved actual input dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, actual_input_dtype.value());

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;
    input_node.input_tensor_dtype = supported_input_dtype;
    input_node.output_dtype = supported_input_dtype;
    input_node.compute_dtype = defaultComputeDType(supported_input_dtype);
    input_node.backward_output_dtype = supported_input_dtype;
    input_node.backward_compute_dtype = defaultComputeDType(supported_input_dtype);

    stage_expr.nodes.push_back(std::move(input_node));

    softmax.lhs = 0;
    softmax.rhs = UINT32_MAX;
    stage_expr.nodes.push_back(std::move(softmax));
    stage_expr.output_node = 1;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = 1,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::Softmax,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static PhysicalExecutionStage buildRmsNormStage(const PhysicalExpression& expr,
                                                uint32_t node_idx,
                                                uint32_t output_value_id,
                                                const std::string& output_name,
                                                const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isRmsNormOp(node.op)) {
        throw std::runtime_error("buildRmsNormStage called on non-RMSNorm node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("RMSNorm node missing input or scale.");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(2);
    auto inputNameForSlot = [](uint32_t slot) { return std::string("__arg") + std::to_string(slot); };

    auto add_local_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        const ExprNode& parent = expr.nodes[parent_idx];
        uint32_t value_id = UINT32_MAX;
        std::optional<DataType> actual_input_dtype = std::nullopt;
        std::optional<DataType> output_dtype = std::nullopt;
        std::optional<DataType> compute_dtype = std::nullopt;
        std::optional<DataType> backward_output_dtype = std::nullopt;
        std::optional<DataType> backward_compute_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            value_id = out_it->second;
            actual_input_dtype = parent.output_dtype;
            output_dtype = parent.output_dtype;
            compute_dtype = parent.compute_dtype;
            backward_output_dtype = parent.backward_output_dtype;
            backward_compute_dtype = parent.backward_compute_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            value_id = parent.input_slot;
            actual_input_dtype = parent.input_tensor_dtype;
            output_dtype = parent.output_dtype;
            compute_dtype = parent.compute_dtype;
            backward_output_dtype = parent.backward_output_dtype;
            backward_compute_dtype = parent.backward_compute_dtype;
        } else {
            throw std::runtime_error("Missing value id for RMSNorm input.");
        }

        if (!actual_input_dtype.has_value() || !output_dtype.has_value()) {
            throw std::runtime_error("RMSNorm input parent is missing resolved dtype metadata.");
        }

        input_value_ids.push_back(value_id);
        stage_expr.inputs.push_back(NamedInput{inputNameForSlot(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = output_dtype.value();
        input_node.compute_dtype = compute_dtype.value_or(defaultComputeDType(actual_input_dtype.value(), output_dtype.value()));
        input_node.backward_output_dtype = backward_output_dtype.value_or(output_dtype.value());
        input_node.backward_compute_dtype = backward_compute_dtype.value_or(input_node.compute_dtype.value());
        stage_expr.nodes.push_back(std::move(input_node));
    };

    add_local_input(node.lhs, 0);
    add_local_input(node.rhs, 1);

    ExprNode rms_norm = node;
    rms_norm.lhs = 0;
    rms_norm.rhs = 1;
    rms_norm.aux = UINT32_MAX;
    stage_expr.nodes.push_back(std::move(rms_norm));
    stage_expr.output_node = 2;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = 2,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::RmsNorm,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}


static bool isEmbeddingRootFusionOp(ExprOp op) {
    switch (op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::NEG:
            return true;
        default:
            return false;
    }
}

static bool collectEmbeddingRootFusionNodes(const PhysicalExpression& expr,
                                            uint32_t node_idx,
                                            uint32_t& embedding_node_idx,
                                            std::unordered_set<uint32_t>& region) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (isEmbeddingLookupOp(node.op)) {
        if (embedding_node_idx != UINT32_MAX && embedding_node_idx != node_idx) {
            return false;
        }
        embedding_node_idx = node_idx;
        region.insert(node_idx);
        return true;
    }

    if (node.op == ExprOp::INPUT || node.op == ExprOp::SCALAR_FP) {
        return true;
    }

    if (!isEmbeddingRootFusionOp(node.op)) {
        return false;
    }

    bool ok = true;
    bool saw_embedding = false;
    auto visit_parent = [&](uint32_t parent_idx) {
        if (parent_idx == UINT32_MAX) {
            ok = false;
            return;
        }
        const uint32_t before = embedding_node_idx;
        if (!collectEmbeddingRootFusionNodes(expr, parent_idx, embedding_node_idx, region)) {
            ok = false;
            return;
        }
        if (embedding_node_idx != UINT32_MAX && (before == UINT32_MAX || before == embedding_node_idx)) {
            saw_embedding = true;
        }
    };

    visit_parent(node.lhs);
    if (node.op != ExprOp::NEG) {
        visit_parent(node.rhs);
    }
    if (!ok || embedding_node_idx == UINT32_MAX || !saw_embedding) {
        return false;
    }

    region.insert(node_idx);
    return true;
}

static bool canBuildEmbeddingRootFusedStage(const PhysicalExpression& expr, uint32_t root_idx, uint32_t& embedding_node_idx) {
    embedding_node_idx = UINT32_MAX;
    std::unordered_set<uint32_t> region;
    if (!collectEmbeddingRootFusionNodes(expr, root_idx, embedding_node_idx, region)) {
        return false;
    }
    return embedding_node_idx != UINT32_MAX && embedding_node_idx != root_idx;
}

static PhysicalExecutionStage buildEmbeddingRootFusedStage(const PhysicalExpression& expr,
                                                          uint32_t root_idx,
                                                          uint32_t output_value_id,
                                                          const std::string& output_name,
                                                          const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    uint32_t embedding_node_idx = UINT32_MAX;
    std::unordered_set<uint32_t> region;
    if (!collectEmbeddingRootFusionNodes(expr, root_idx, embedding_node_idx, region) || embedding_node_idx == UINT32_MAX ||
        embedding_node_idx == root_idx) {
        throw std::runtime_error("buildEmbeddingRootFusedStage called for a non-fusible embedding-root expression.");
    }

    const ExprNode& embedding = expr.nodes[embedding_node_idx];
    if (embedding.lhs == UINT32_MAX || embedding.rhs == UINT32_MAX || embedding.lhs >= expr.nodes.size() || embedding.rhs >= expr.nodes.size()) {
        throw std::runtime_error("Embedding-root fused stage has malformed EmbeddingLookup inputs.");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    std::unordered_map<uint32_t, uint32_t> local_node_by_old;
    std::unordered_map<uint32_t, uint32_t> local_input_slot_by_old;

    auto add_local_input = [&](uint32_t parent_idx, uint32_t forced_slot) -> uint32_t {
        auto existing = local_node_by_old.find(parent_idx);
        if (existing != local_node_by_old.end()) {
            return existing->second;
        }
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Embedding-root fused stage input parent is out of range.");
        }
        const ExprNode& parent = expr.nodes[parent_idx];
        if (parent.op != ExprOp::INPUT) {
            throw std::runtime_error("Embedding-root fusion currently supports only direct tensor inputs as external operands.");
        }
        if (!parent.input_tensor_dtype.has_value() || !parent.output_dtype.has_value()) {
            throw std::runtime_error("Embedding-root fusion input is missing dtype metadata.");
        }
        const uint32_t local_slot = forced_slot == UINT32_MAX ? static_cast<uint32_t>(stage_expr.inputs.size()) : forced_slot;
        if (local_slot != stage_expr.inputs.size()) {
            throw std::runtime_error("Embedding-root fusion internal input slots must be appended in order.");
        }
        uint32_t value_id = parent.input_slot;
        auto value_it = node_output_value_id.find(parent_idx);
        if (value_it != node_output_value_id.end()) {
            value_id = value_it->second;
        }
        input_value_ids.push_back(value_id);
        stage_expr.inputs.push_back(NamedInput{std::string("__arg") + std::to_string(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = parent.input_tensor_dtype.value();
        input_node.output_dtype = parent.output_dtype.value();
        input_node.compute_dtype = parent.compute_dtype.value_or(parent.output_dtype.value());
        input_node.backward_output_dtype = parent.backward_output_dtype.value_or(parent.output_dtype.value());
        input_node.backward_compute_dtype = parent.backward_compute_dtype.value_or(input_node.compute_dtype.value());
        const uint32_t local_idx = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(input_node));
        local_node_by_old[parent_idx] = local_idx;
        local_input_slot_by_old[parent_idx] = local_slot;
        return local_idx;
    };

    add_local_input(embedding.lhs, 0);
    add_local_input(embedding.rhs, 1);

    std::function<uint32_t(uint32_t)> copy_node = [&](uint32_t old_idx) -> uint32_t {
        auto existing = local_node_by_old.find(old_idx);
        if (existing != local_node_by_old.end()) {
            return existing->second;
        }
        if (old_idx >= expr.nodes.size()) {
            throw std::runtime_error("Embedding-root fusion node index out of range.");
        }
        const ExprNode& old = expr.nodes[old_idx];
        if (old.op == ExprOp::INPUT) {
            return add_local_input(old_idx, UINT32_MAX);
        }
        if (old.op == ExprOp::SCALAR_FP) {
            ExprNode scalar = old;
            const uint32_t local_idx = static_cast<uint32_t>(stage_expr.nodes.size());
            stage_expr.nodes.push_back(std::move(scalar));
            local_node_by_old[old_idx] = local_idx;
            return local_idx;
        }
        if (old_idx == embedding_node_idx) {
            ExprNode lookup = old;
            lookup.lhs = local_node_by_old.at(embedding.lhs);
            lookup.rhs = local_node_by_old.at(embedding.rhs);
            lookup.aux = UINT32_MAX;
            const uint32_t local_idx = static_cast<uint32_t>(stage_expr.nodes.size());
            stage_expr.nodes.push_back(std::move(lookup));
            local_node_by_old[old_idx] = local_idx;
            return local_idx;
        }
        if (!isEmbeddingRootFusionOp(old.op)) {
            throw std::runtime_error("Embedding-root fusion encountered an unsupported epilogue op.");
        }
        ExprNode copy = old;
        copy.lhs = copy_node(old.lhs);
        if (old.op != ExprOp::NEG) {
            copy.rhs = copy_node(old.rhs);
        } else {
            copy.rhs = UINT32_MAX;
        }
        copy.aux = UINT32_MAX;
        const uint32_t local_idx = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(copy));
        local_node_by_old[old_idx] = local_idx;
        return local_idx;
    };

    const uint32_t local_root = copy_node(root_idx);
    stage_expr.output_node = local_root;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{.name = output_name, .local_node_idx = local_root, .value_id = output_value_id});

    return PhysicalExecutionStage{.kind = PhysicalExecutionStage::Kind::EmbeddingLookup,
                                  .expr = std::move(stage_expr),
                                  .input_value_ids = std::move(input_value_ids),
                                  .outputs = std::move(stage_outputs)};
}

static PhysicalExecutionStage buildEmbeddingLookupStage(const PhysicalExpression& expr,
                                                        uint32_t node_idx,
                                                        uint32_t output_value_id,
                                                        const std::string& output_name,
                                                        const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isEmbeddingLookupOp(node.op)) {
        throw std::runtime_error("buildEmbeddingLookupStage called on non-EmbeddingLookup node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
        throw std::runtime_error("EmbeddingLookup node missing indices or weights input.");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(2);

    auto add_local_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        const ExprNode& parent = expr.nodes[parent_idx];
        uint32_t value_id = UINT32_MAX;
        std::optional<DataType> actual_input_dtype = std::nullopt;
        std::optional<DataType> output_dtype = std::nullopt;
        std::optional<DataType> compute_dtype = std::nullopt;
        std::optional<DataType> backward_output_dtype = std::nullopt;
        std::optional<DataType> backward_compute_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            value_id = out_it->second;
            actual_input_dtype = parent.output_dtype;
            output_dtype = parent.output_dtype;
            compute_dtype = parent.compute_dtype;
            backward_output_dtype = parent.backward_output_dtype;
            backward_compute_dtype = parent.backward_compute_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            value_id = parent.input_slot;
            actual_input_dtype = parent.input_tensor_dtype;
            output_dtype = parent.output_dtype;
            compute_dtype = parent.compute_dtype;
            backward_output_dtype = parent.backward_output_dtype;
            backward_compute_dtype = parent.backward_compute_dtype;
        } else {
            throw std::runtime_error("Missing value id for EmbeddingLookup input.");
        }
        if (!actual_input_dtype.has_value() || !output_dtype.has_value()) {
            throw std::runtime_error("EmbeddingLookup parent is missing resolved dtype metadata.");
        }

        input_value_ids.push_back(value_id);
        stage_expr.inputs.push_back(NamedInput{std::string("__arg") + std::to_string(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = output_dtype.value();
        input_node.compute_dtype = compute_dtype.value_or(output_dtype.value());
        input_node.backward_output_dtype = backward_output_dtype.value_or(output_dtype.value());
        input_node.backward_compute_dtype = backward_compute_dtype.value_or(input_node.compute_dtype.value());
        stage_expr.nodes.push_back(std::move(input_node));
    };

    add_local_input(node.lhs, 0);
    add_local_input(node.rhs, 1);

    ExprNode lookup = node;
    lookup.lhs = 0;
    lookup.rhs = 1;
    lookup.aux = UINT32_MAX;
    stage_expr.nodes.push_back(std::move(lookup));
    stage_expr.output_node = 2;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{.name = output_name, .local_node_idx = 2, .value_id = output_value_id});

    return PhysicalExecutionStage{.kind = PhysicalExecutionStage::Kind::EmbeddingLookup,
                                  .expr = std::move(stage_expr),
                                  .input_value_ids = std::move(input_value_ids),
                                  .outputs = std::move(stage_outputs)};
}

static PhysicalExecutionStage buildInputTransposedMaterializationStage(const PhysicalExpression& expr,
                                                                       uint32_t parent_idx,
                                                                       uint32_t output_value_id,
                                                                       const std::string& output_name,
                                                                       const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    if (parent_idx >= expr.nodes.size()) {
        throw std::runtime_error("buildInputTransposedMaterializationStage parent index out of range.");
    }

    const ExprNode& parent = expr.nodes[parent_idx];

    uint32_t input_value_id = UINT32_MAX;
    std::optional<DataType> actual_input_dtype = std::nullopt;
    std::optional<DataType> output_dtype = std::nullopt;
    std::optional<DataType> compute_dtype = std::nullopt;
    std::optional<DataType> backward_output_dtype = std::nullopt;
    std::optional<DataType> backward_compute_dtype = std::nullopt;

    auto out_it = node_output_value_id.find(parent_idx);
    if (out_it != node_output_value_id.end()) {
        input_value_id = out_it->second;
        actual_input_dtype = parent.output_dtype;
        output_dtype = parent.output_dtype;
        compute_dtype = parent.compute_dtype;
        backward_output_dtype = parent.backward_output_dtype;
        backward_compute_dtype = parent.backward_compute_dtype;
    } else if (parent.op == ExprOp::INPUT) {
        input_value_id = parent.input_slot;
        actual_input_dtype = parent.input_tensor_dtype;
        output_dtype = parent.output_dtype;
        compute_dtype = parent.compute_dtype;
        backward_output_dtype = parent.backward_output_dtype;
        backward_compute_dtype = parent.backward_compute_dtype;
    } else {
        throw std::runtime_error("Missing materialized value id for transposed input materialization parent.");
    }

    if (!actual_input_dtype.has_value() || !output_dtype.has_value()) {
        throw std::runtime_error("Transposed input materialization parent is missing resolved dtype metadata.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0, NamedInput::Kind::Tensor});

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;
    input_node.input_tensor_dtype = actual_input_dtype.value();
    input_node.output_dtype = output_dtype;
    input_node.compute_dtype = compute_dtype;
    input_node.backward_output_dtype = backward_output_dtype;
    input_node.backward_compute_dtype = backward_compute_dtype;
    stage_expr.nodes.push_back(std::move(input_node));
    stage_expr.output_node = 0;

    std::vector<uint32_t> input_value_ids{input_value_id};
    std::vector<CompiledStageOutput> stage_outputs{CompiledStageOutput{
        .name = output_name,
        .local_node_idx = 0,
        .value_id = output_value_id,
        .materialized_layout = MaterializedTensorLayout::Transposed,
    }};

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

struct PendingMatmulBgradOutput {
    uint32_t reduce_node_idx = UINT32_MAX;
    uint32_t value_id = UINT32_MAX;
    std::string name;
};

static PhysicalExecutionStage buildMatmulStage(const PhysicalExpression& expr,
                                               uint32_t node_idx,
                                               uint32_t output_value_id,
                                               const std::string& output_name,
                                               const std::unordered_map<uint32_t, uint32_t>& node_output_value_id,
                                               const std::vector<PendingMatmulBgradOutput>& bgrad_outputs = {}) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isMatmulOp(node.op)) {
        throw std::runtime_error("buildMatmulStage called on non-matmul node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || (node.op == ExprOp::GEMM && node.aux == UINT32_MAX)) {
        throw std::runtime_error("Matmul/gemm node missing required input(s).");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve((node.op == ExprOp::MATMUL ? 4u : 5u) + (node.matmul_epilogue_aux != UINT32_MAX ? 1u : 0u));

    auto inputNameForSlot = [](uint32_t slot) { return std::string("__arg") + std::to_string(slot); };

    auto bind_parent_to_local_tensor_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Matmul/gemm input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.input_tensor_dtype;
        } else {
            throw std::runtime_error("Missing value id for matmul/gemm input.");
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("Matmul/gemm parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("Matmul/gemm parent node missing resolved actual input dtype.");
        }

        stage_expr.inputs.push_back(NamedInput{inputNameForSlot(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
    };

    auto bind_parent_to_local_scalar = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx == UINT32_MAX) {
            return;
        }
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Matmul/gemm scalar node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        if (parent.op == ExprOp::SCALAR_FP) {
            ExprNode scalar_node = parent;
            scalar_node.input_slot = UINT32_MAX;
            stage_expr.nodes.push_back(std::move(scalar_node));
            return;
        }
        if (parent.op == ExprOp::RUNTIME_SCALAR || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            if (parent.input_slot >= expr.inputs.size()) {
                throw std::runtime_error("Matmul/gemm dynamic scale parent input slot is out of range.");
            }
            input_value_ids.push_back(parent.input_slot);

            const NamedInput::Kind input_kind =
                (parent.op == ExprOp::RUNTIME_SCALAR) ? NamedInput::Kind::RuntimeScalarFp32 : NamedInput::Kind::TensorRuntimeScalar;

            stage_expr.inputs.push_back(NamedInput{expr.inputs[parent.input_slot].name, local_slot, input_kind});

            ExprNode input_node;
            input_node.op = parent.op;
            input_node.input_slot = local_slot;
            input_node.input_tensor_dtype = DataType::FP32;
            input_node.output_dtype = parent.output_dtype.has_value() ? parent.output_dtype.value() : DataType::FP32;
            input_node.compute_dtype = parent.compute_dtype;
            input_node.backward_output_dtype = parent.backward_output_dtype;
            input_node.backward_compute_dtype = parent.backward_compute_dtype;
            stage_expr.nodes.push_back(std::move(input_node));
            return;
        }

        // Arbitrary scalar subexpressions are materialized ahead of the GEMM stage as 1-element tensor values,
        // and direct tensor inputs may also be used as GEMM scales when they resolve to a single element.
        bind_parent_to_local_tensor_input(parent_idx, local_slot);
    };

    bool absorbed_lhs_transpose = false;
    bool absorbed_rhs_transpose = false;
    uint32_t effective_lhs_parent = node.lhs;
    uint32_t effective_rhs_parent = node.rhs;
    if (node.matmul_backward_epilogue == MatmulBackwardEpilogue::Default) {
        effective_lhs_parent = peelExplicitTransposeChain(expr, node.lhs, absorbed_lhs_transpose);
        effective_rhs_parent = peelExplicitTransposeChain(expr, node.rhs, absorbed_rhs_transpose);
    }

    bind_parent_to_local_tensor_input(effective_lhs_parent, static_cast<uint32_t>(stage_expr.inputs.size()));
    const uint32_t lhs_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    bind_parent_to_local_tensor_input(effective_rhs_parent, static_cast<uint32_t>(stage_expr.inputs.size()));
    const uint32_t rhs_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    uint32_t aux_local = UINT32_MAX;
    if (node.op == ExprOp::GEMM) {
        bind_parent_to_local_tensor_input(node.aux, static_cast<uint32_t>(stage_expr.inputs.size()));
        aux_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    }

    uint32_t alpha_local = UINT32_MAX;
    if (node.alpha_node != UINT32_MAX) {
        bind_parent_to_local_scalar(node.alpha_node, static_cast<uint32_t>(stage_expr.inputs.size()));
        alpha_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    }
    uint32_t beta_local = UINT32_MAX;
    if (node.beta_node != UINT32_MAX) {
        bind_parent_to_local_scalar(node.beta_node, static_cast<uint32_t>(stage_expr.inputs.size()));
        beta_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    }

    uint32_t epilogue_aux_local = UINT32_MAX;
    if (node.matmul_epilogue_aux != UINT32_MAX) {
        bind_parent_to_local_tensor_input(node.matmul_epilogue_aux, static_cast<uint32_t>(stage_expr.inputs.size()));
        epilogue_aux_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    }

    ExprNode route = node;
    route.lhs = lhs_local;
    route.rhs = rhs_local;
    route.aux = aux_local;
    route.alpha_node = alpha_local;
    route.beta_node = beta_local;
    route.matmul_epilogue_aux = epilogue_aux_local;
    route.transpose_lhs = node.transpose_lhs ^ absorbed_lhs_transpose;
    route.transpose_rhs = node.transpose_rhs ^ absorbed_rhs_transpose;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = static_cast<uint32_t>(stage_expr.nodes.size() - 1);

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = stage_expr.output_node,
        .value_id = output_value_id,
    });

    for (const PendingMatmulBgradOutput& bgrad : bgrad_outputs) {
        if (bgrad.reduce_node_idx >= expr.nodes.size()) {
            throw std::runtime_error("Matmul bias-gradient output node index is out of range.");
        }
        ExprNode local_bgrad = expr.nodes.at(bgrad.reduce_node_idx);
        if (local_bgrad.op != ExprOp::REDUCE_SUM || local_bgrad.lhs != node_idx || local_bgrad.reduction_axes.size() != 1 ||
            local_bgrad.reduction_axes[0] != 0 || local_bgrad.squeeze_axes.size() != 1 || local_bgrad.squeeze_axes[0] != 0) {
            throw std::runtime_error("Matmul bias-gradient output must be reduce_sum(matmul_output, axis=0).");
        }
        local_bgrad.lhs = stage_expr.output_node;
        stage_expr.nodes.push_back(std::move(local_bgrad));
        stage_outputs.push_back(CompiledStageOutput{
            .name = bgrad.name,
            .local_node_idx = static_cast<uint32_t>(stage_expr.nodes.size() - 1),
            .value_id = bgrad.value_id,
        });
    }

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::Matmul,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

std::shared_ptr<CompiledInPlaceRope> EquationCompiler::compileInPlaceRope(const PhysicalExpression& expr,
                                                                          const std::vector<CompiledStageOutput>& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("In-place RoPE stage requires at least one output.");
    }
    auto compiled = std::make_shared<CompiledInPlaceRope>();
    compiled->tensors.reserve(outputs.size());

    for (const CompiledStageOutput& output : outputs) {
        if (output.local_node_idx >= expr.nodes.size()) {
            throw std::runtime_error("In-place RoPE output node index out of range.");
        }
        const ExprNode& rope = expr.nodes.at(output.local_node_idx);
        if (rope.op != ExprOp::ROPE || rope.lhs == UINT32_MAX || rope.lhs >= expr.nodes.size()) {
            throw std::runtime_error("In-place RoPE stage output must be a ROPE node.");
        }
        const ExprNode& reshape = expr.nodes.at(rope.lhs);
        if (reshape.op != ExprOp::RESHAPE || reshape.lhs == UINT32_MAX || reshape.lhs >= expr.nodes.size()) {
            throw std::runtime_error("In-place RoPE stage expects ROPE(RESHAPE(INPUT)).");
        }
        const ExprNode& input = expr.nodes.at(reshape.lhs);
        if (input.op != ExprOp::INPUT || input.input_slot == UINT32_MAX) {
            throw std::runtime_error("In-place RoPE stage reshape source must be an INPUT node.");
        }
        if (!input.input_tensor_dtype.has_value() || !rope.output_dtype.has_value()) {
            throw std::runtime_error("In-place RoPE stage nodes must have resolved dtypes.");
        }
        if (input.input_tensor_dtype.value() != rope.output_dtype.value()) {
            throw std::runtime_error("In-place RoPE requires input and output dtype to match.");
        }
        RotaryPositionEmbeddingOptions options;
        options.sequence_axis = rope.rope_sequence_axis;
        options.head_dim_axis = rope.rope_head_dim_axis;
        options.rotary_dim = rope.rope_rotary_dim;
        options.base = rope.rope_base;
        options.position_offset = rope.rope_position_offset;
        options.interleaved = rope.rope_interleaved;
        options.inverse = rope.rope_inverse;
        options.scaling_kind = rope.rope_scaling_kind;
        options.scaling_factor = rope.rope_scaling_factor;
        options.original_max_position_embeddings = rope.rope_original_max_position_embeddings;
        options.attention_factor = rope.rope_attention_factor;
        options.yarn_beta_fast = rope.rope_yarn_beta_fast;
        options.yarn_beta_slow = rope.rope_yarn_beta_slow;
        options.llama3_low_freq_factor = rope.rope_llama3_low_freq_factor;
        options.llama3_high_freq_factor = rope.rope_llama3_high_freq_factor;
        options.long_rope_short_factors = rope.rope_long_rope_short_factors;
        options.long_rope_long_factors = rope.rope_long_rope_long_factors;
        options.output_dtype = rope.output_dtype;
        options.compute_dtype = rope.compute_dtype;
        compiled->tensors.push_back(CompiledInPlaceRopeTensor{
            .input_slot = input.input_slot,
            .logical_dims = reshape.reshape_dims,
            .options = options,
            .dtype = rope.output_dtype.value(),
        });
    }
    return compiled;
}

static void forceAttentionSeqLenLocalInputDType(PhysicalExpression& stage_expr, uint32_t local_idx, const char* label) {
    if (local_idx >= stage_expr.nodes.size()) {
        throw std::runtime_error(std::string("Attention sequence-length local input index out of range for ") + label + ".");
    }
    ExprNode& input_node = stage_expr.nodes.at(local_idx);
    if (input_node.op != ExprOp::INPUT) {
        throw std::runtime_error(std::string("Attention sequence-length local node must be an INPUT for ") + label + ".");
    }

    // Runtime sequence lengths are metadata tensors consumed by cuDNN SDPA, not floating-point expression values.
    // Keep both storage and logical metadata pinned to INT32 so dtype resolution for cloned/specialized backward
    // graphs cannot accidentally inherit the surrounding attention compute dtype.
    input_node.input_tensor_dtype = DataType::INT32;
    input_node.output_dtype = DataType::INT32;
    input_node.compute_dtype = DataType::INT32;
    input_node.backward_output_dtype = DataType::INT32;
    input_node.backward_compute_dtype = DataType::INT32;
}

static void forceAttentionFp8ScalarLocalInputDType(PhysicalExpression& stage_expr, uint32_t local_idx, const char* label) {
    if (local_idx >= stage_expr.nodes.size()) {
        throw std::runtime_error(std::string("Attention FP8 scalar local input index out of range for ") + label + ".");
    }
    ExprNode& input_node = stage_expr.nodes.at(local_idx);
    if (input_node.op != ExprOp::INPUT) {
        throw std::runtime_error(std::string("Attention FP8 scalar local node must be an INPUT for ") + label + ".");
    }
    input_node.input_tensor_dtype = DataType::FP32;
    input_node.output_dtype = DataType::FP32;
    input_node.compute_dtype = DataType::FP32;
    input_node.backward_output_dtype = DataType::FP32;
    input_node.backward_compute_dtype = DataType::FP32;
}

static void forceAttentionDropoutLocalInputDType(PhysicalExpression& stage_expr, uint32_t local_idx, const char* label) {
    if (local_idx >= stage_expr.nodes.size()) {
        throw std::runtime_error(std::string("Attention dropout local input index out of range for ") + label + ".");
    }
    ExprNode& input_node = stage_expr.nodes.at(local_idx);
    if (input_node.op != ExprOp::INPUT && input_node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
        throw std::runtime_error(std::string("Attention dropout local node must be an INPUT or TENSOR_RUNTIME_SCALAR for ") + label + ".");
    }

    // cuDNN SDPA Philox dropout seed/offset are scalar INT64 metadata tensors, not floating expression values.
    // Pin local stage metadata so cloned/specialized backward graphs cannot inherit floating attention dtypes.
    if (input_node.op == ExprOp::INPUT) {
        input_node.input_tensor_dtype = DataType::INT64;
    }
    input_node.output_dtype = DataType::INT64;
    input_node.compute_dtype = DataType::INT64;
    input_node.backward_output_dtype = DataType::INT64;
    input_node.backward_compute_dtype = DataType::INT64;
}

static PhysicalExecutionStage buildAttentionStage(const PhysicalExpression& expr,
                                                  uint32_t node_idx,
                                                  uint32_t output_value_id,
                                                  const std::string& output_name,
                                                  const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isAttentionOp(node.op)) {
        throw std::runtime_error("buildAttentionStage called on non-attention node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.aux == UINT32_MAX) {
        throw std::runtime_error("Attention node missing q/k/v input(s).");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(3 + (node.attention_use_bias ? 1 : 0) + (node.attention_use_padding_mask ? 2 : 0) +
                            (node.attention_use_ragged_offsets ? 2 : 0) + (node.attention_use_paged_kv_cache ? 2 : 0) +
                            (node.attention_dropout_probability > 0.0f ? 2 : 0) + (node.attention_use_fp8_forward_scaling ? 8 : 0));

    auto inputNameForSlot = [](uint32_t slot) { return std::string("__arg") + std::to_string(slot); };

    auto bind_parent_to_local_tensor_input = [&](uint32_t parent_idx, uint32_t local_slot) -> uint32_t {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Attention input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        if (parent.op == ExprOp::INPUT || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.op == ExprOp::INPUT ? parent.input_tensor_dtype : parent.output_dtype;
        } else {
            auto out_it = node_output_value_id.find(parent_idx);
            if (out_it == node_output_value_id.end()) {
                throw std::runtime_error("Missing value id for attention input.");
            }
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("Attention parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("Attention parent node missing resolved actual input dtype.");
        }

        const bool is_tensor_runtime_scalar = parent.op == ExprOp::TENSOR_RUNTIME_SCALAR;
        stage_expr.inputs.push_back(
            NamedInput{inputNameForSlot(local_slot),
                       local_slot,
                       is_tensor_runtime_scalar ? NamedInput::Kind::TensorRuntimeScalar : NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = is_tensor_runtime_scalar ? ExprOp::TENSOR_RUNTIME_SCALAR : ExprOp::INPUT;
        input_node.input_slot = local_slot;
        if (!is_tensor_runtime_scalar) {
            input_node.input_tensor_dtype = actual_input_dtype.value();
        }
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
        return static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    };

    const uint32_t q_local = bind_parent_to_local_tensor_input(node.lhs, 0);
    const uint32_t k_local = bind_parent_to_local_tensor_input(node.rhs, 1);
    const uint32_t v_local = bind_parent_to_local_tensor_input(node.aux, 2);
    uint32_t next_local_slot = 3;
    uint32_t bias_local = UINT32_MAX;
    if (node.attention_use_bias) {
        if (node.alpha_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using bias but missing bias input.");
        }
        bias_local = bind_parent_to_local_tensor_input(node.alpha_node, next_local_slot++);
    }
    uint32_t q_seq_len_local = UINT32_MAX;
    uint32_t kv_seq_len_local = UINT32_MAX;
    if (node.attention_use_padding_mask) {
        if (node.attention_seq_len_q_node == UINT32_MAX || node.attention_seq_len_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using padding mask but missing q/kv sequence length inputs.");
        }
        q_seq_len_local = bind_parent_to_local_tensor_input(node.attention_seq_len_q_node, next_local_slot++);
        kv_seq_len_local = bind_parent_to_local_tensor_input(node.attention_seq_len_kv_node, next_local_slot++);
        forceAttentionSeqLenLocalInputDType(stage_expr, q_seq_len_local, "q_seq_len");
        forceAttentionSeqLenLocalInputDType(stage_expr, kv_seq_len_local, "kv_seq_len");
    }
    uint32_t q_ragged_offset_local = UINT32_MAX;
    uint32_t kv_ragged_offset_local = UINT32_MAX;
    if (node.attention_use_ragged_offsets) {
        if (node.attention_ragged_offset_q_node == UINT32_MAX || node.attention_ragged_offset_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using ragged offsets but missing q/kv ragged offset inputs.");
        }
        q_ragged_offset_local = bind_parent_to_local_tensor_input(node.attention_ragged_offset_q_node, next_local_slot++);
        kv_ragged_offset_local = bind_parent_to_local_tensor_input(node.attention_ragged_offset_kv_node, next_local_slot++);
        forceAttentionSeqLenLocalInputDType(stage_expr, q_ragged_offset_local, "q_ragged_offsets");
        forceAttentionSeqLenLocalInputDType(stage_expr, kv_ragged_offset_local, "kv_ragged_offsets");
    }
    uint32_t page_table_k_local = UINT32_MAX;
    uint32_t page_table_v_local = UINT32_MAX;
    if (node.attention_use_paged_kv_cache) {
        if (node.attention_page_table_k_node == UINT32_MAX || node.attention_page_table_v_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using paged KV cache but missing page-table inputs.");
        }
        page_table_k_local = bind_parent_to_local_tensor_input(node.attention_page_table_k_node, next_local_slot++);
        page_table_v_local = bind_parent_to_local_tensor_input(node.attention_page_table_v_node, next_local_slot++);
        forceAttentionSeqLenLocalInputDType(stage_expr, page_table_k_local, "page_table_k");
        forceAttentionSeqLenLocalInputDType(stage_expr, page_table_v_local, "page_table_v");
    }

    uint32_t dropout_seed_local = UINT32_MAX;
    uint32_t dropout_offset_local = UINT32_MAX;
    if (node.attention_dropout_probability > 0.0f) {
        if (node.attention_dropout_seed_node == UINT32_MAX || node.attention_dropout_offset_node == UINT32_MAX) {
            throw std::runtime_error("Attention node marked as using dropout but missing seed/offset inputs.");
        }
        dropout_seed_local = bind_parent_to_local_tensor_input(node.attention_dropout_seed_node, next_local_slot++);
        dropout_offset_local = bind_parent_to_local_tensor_input(node.attention_dropout_offset_node, next_local_slot++);
        forceAttentionDropoutLocalInputDType(stage_expr, dropout_seed_local, "dropout_seed");
        forceAttentionDropoutLocalInputDType(stage_expr, dropout_offset_local, "dropout_offset");
    }

    uint32_t descale_q_local = UINT32_MAX;
    uint32_t descale_k_local = UINT32_MAX;
    uint32_t descale_v_local = UINT32_MAX;
    uint32_t descale_s_local = UINT32_MAX;
    uint32_t scale_s_local = UINT32_MAX;
    uint32_t scale_o_local = UINT32_MAX;
    uint32_t amax_s_local = UINT32_MAX;
    uint32_t amax_o_local = UINT32_MAX;
    if (node.attention_use_fp8_forward_scaling) {
        descale_q_local = bind_parent_to_local_tensor_input(node.attention_descale_q_node, next_local_slot++);
        descale_k_local = bind_parent_to_local_tensor_input(node.attention_descale_k_node, next_local_slot++);
        descale_v_local = bind_parent_to_local_tensor_input(node.attention_descale_v_node, next_local_slot++);
        descale_s_local = bind_parent_to_local_tensor_input(node.attention_descale_s_node, next_local_slot++);
        scale_s_local = bind_parent_to_local_tensor_input(node.attention_scale_s_node, next_local_slot++);
        scale_o_local = bind_parent_to_local_tensor_input(node.attention_scale_o_node, next_local_slot++);
        amax_s_local = bind_parent_to_local_tensor_input(node.attention_amax_s_node, next_local_slot++);
        amax_o_local = bind_parent_to_local_tensor_input(node.attention_amax_o_node, next_local_slot++);
        forceAttentionFp8ScalarLocalInputDType(stage_expr, descale_q_local, "descale_q");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, descale_k_local, "descale_k");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, descale_v_local, "descale_v");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, descale_s_local, "descale_s");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, scale_s_local, "scale_s");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, scale_o_local, "scale_o");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, amax_s_local, "amax_s");
        forceAttentionFp8ScalarLocalInputDType(stage_expr, amax_o_local, "amax_o");
    }

    ExprNode route = node;
    route.lhs = q_local;
    route.rhs = k_local;
    route.aux = v_local;
    route.alpha_node = bias_local;
    route.beta_node = UINT32_MAX;
    route.attention_seq_len_q_node = q_seq_len_local;
    route.attention_seq_len_kv_node = kv_seq_len_local;
    route.attention_ragged_offset_q_node = q_ragged_offset_local;
    route.attention_ragged_offset_kv_node = kv_ragged_offset_local;
    route.attention_page_table_k_node = page_table_k_local;
    route.attention_page_table_v_node = page_table_v_local;
    route.attention_dropout_seed_node = dropout_seed_local;
    route.attention_dropout_offset_node = dropout_offset_local;
    route.attention_descale_q_node = descale_q_local;
    route.attention_descale_k_node = descale_k_local;
    route.attention_descale_v_node = descale_v_local;
    route.attention_descale_s_node = descale_s_local;
    route.attention_scale_s_node = scale_s_local;
    route.attention_scale_o_node = scale_o_local;
    route.attention_amax_s_node = amax_s_local;
    route.attention_amax_o_node = amax_o_local;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = static_cast<uint32_t>(stage_expr.nodes.size() - 1);

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = stage_expr.output_node,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::Attention,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static PhysicalExecutionStage buildAttentionBackwardStage(const PhysicalExpression& expr,
                                                          uint32_t node_idx,
                                                          uint32_t output_value_id,
                                                          const std::string& output_name,
                                                          const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isAttentionBackwardOp(node.op)) {
        throw std::runtime_error("buildAttentionBackwardStage called on non-attention-backward node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.aux == UINT32_MAX || node.alpha_node == UINT32_MAX) {
        throw std::runtime_error("Attention-backward node missing q/k/v/dO input(s).");
    }
    if (node.op == ExprOp::ATTENTION_BACKWARD_BIAS && !node.attention_use_bias) {
        throw std::runtime_error("Attention-backward dBias output requested for an unbiased attention node.");
    }
    if (node.attention_use_ragged_offsets && node.attention_use_bias && !experimentalCudnnRaggedBiasBackwardProbeEnabled()) {
        throw std::runtime_error(
            "cuDNN primary SDPA backward does not support ragged offsets with additive bias; ragged additive bias is forward-only "
            "until a supported dBias/backward path is implemented. Set THOR_EXPERIMENTAL_CUDNN_RAGGED_BIAS_BACKWARD=1 "
            "to bypass this guard for cuDNN support-surface probing only.");
    }
    if (node.attention_use_paged_kv_cache && !experimentalCudnnAttentionSupportSurfaceProbeEnabled()) {
        throw std::runtime_error(
            "Attention-backward with paged KV cache is not enabled; the paged KV path is inference-only until training semantics are "
            "defined.");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(4 + (node.attention_use_bias ? 1 : 0) + (node.attention_use_padding_mask ? 2 : 0) +
                            (node.attention_use_ragged_offsets ? 2 : 0) + (node.attention_use_paged_kv_cache ? 2 : 0) +
                            (node.attention_dropout_probability > 0.0f ? 2 : 0));

    auto inputNameForSlot = [](uint32_t slot) { return std::string("__arg") + std::to_string(slot); };

    auto bind_parent_to_local_tensor_input = [&](uint32_t parent_idx, uint32_t local_slot) -> uint32_t {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Attention-backward input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        if (parent.op == ExprOp::INPUT || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.op == ExprOp::INPUT ? parent.input_tensor_dtype : parent.output_dtype;
        } else {
            auto out_it = node_output_value_id.find(parent_idx);
            if (out_it == node_output_value_id.end()) {
                throw std::runtime_error("Missing value id for attention-backward input.");
            }
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("Attention-backward parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("Attention-backward parent node missing resolved actual input dtype.");
        }

        const bool is_tensor_runtime_scalar = parent.op == ExprOp::TENSOR_RUNTIME_SCALAR;
        stage_expr.inputs.push_back(
            NamedInput{inputNameForSlot(local_slot),
                       local_slot,
                       is_tensor_runtime_scalar ? NamedInput::Kind::TensorRuntimeScalar : NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = is_tensor_runtime_scalar ? ExprOp::TENSOR_RUNTIME_SCALAR : ExprOp::INPUT;
        input_node.input_slot = local_slot;
        if (!is_tensor_runtime_scalar) {
            input_node.input_tensor_dtype = actual_input_dtype.value();
        }
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
        return static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    };

    const uint32_t q_local = bind_parent_to_local_tensor_input(node.lhs, 0);
    const uint32_t k_local = bind_parent_to_local_tensor_input(node.rhs, 1);
    const uint32_t v_local = bind_parent_to_local_tensor_input(node.aux, 2);
    const uint32_t dO_local = bind_parent_to_local_tensor_input(node.alpha_node, 3);
    uint32_t next_local_slot = 4;
    uint32_t bias_local = UINT32_MAX;
    if (node.attention_use_bias) {
        if (node.beta_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using bias but missing bias input.");
        }
        bias_local = bind_parent_to_local_tensor_input(node.beta_node, next_local_slot++);
    }
    uint32_t q_seq_len_local = UINT32_MAX;
    uint32_t kv_seq_len_local = UINT32_MAX;
    if (node.attention_use_padding_mask) {
        if (node.attention_seq_len_q_node == UINT32_MAX || node.attention_seq_len_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using padding mask but missing q/kv sequence length inputs.");
        }
        q_seq_len_local = bind_parent_to_local_tensor_input(node.attention_seq_len_q_node, next_local_slot++);
        kv_seq_len_local = bind_parent_to_local_tensor_input(node.attention_seq_len_kv_node, next_local_slot++);
        forceAttentionSeqLenLocalInputDType(stage_expr, q_seq_len_local, "q_seq_len");
        forceAttentionSeqLenLocalInputDType(stage_expr, kv_seq_len_local, "kv_seq_len");
    }
    uint32_t q_ragged_offset_local = UINT32_MAX;
    uint32_t kv_ragged_offset_local = UINT32_MAX;
    if (node.attention_use_ragged_offsets) {
        if (node.attention_ragged_offset_q_node == UINT32_MAX || node.attention_ragged_offset_kv_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using ragged offsets but missing q/kv ragged offset inputs.");
        }
        q_ragged_offset_local = bind_parent_to_local_tensor_input(node.attention_ragged_offset_q_node, next_local_slot++);
        kv_ragged_offset_local = bind_parent_to_local_tensor_input(node.attention_ragged_offset_kv_node, next_local_slot++);
        forceAttentionSeqLenLocalInputDType(stage_expr, q_ragged_offset_local, "q_ragged_offsets");
        forceAttentionSeqLenLocalInputDType(stage_expr, kv_ragged_offset_local, "kv_ragged_offsets");
    }

    uint32_t dropout_seed_local = UINT32_MAX;
    uint32_t dropout_offset_local = UINT32_MAX;
    if (node.attention_dropout_probability > 0.0f) {
        if (node.attention_dropout_seed_node == UINT32_MAX || node.attention_dropout_offset_node == UINT32_MAX) {
            throw std::runtime_error("Attention-backward node marked as using dropout but missing seed/offset inputs.");
        }
        dropout_seed_local = bind_parent_to_local_tensor_input(node.attention_dropout_seed_node, next_local_slot++);
        dropout_offset_local = bind_parent_to_local_tensor_input(node.attention_dropout_offset_node, next_local_slot++);
        forceAttentionDropoutLocalInputDType(stage_expr, dropout_seed_local, "dropout_seed");
        forceAttentionDropoutLocalInputDType(stage_expr, dropout_offset_local, "dropout_offset");
    }

    ExprNode route = node;
    route.lhs = q_local;
    route.rhs = k_local;
    route.aux = v_local;
    route.alpha_node = dO_local;
    route.beta_node = bias_local;
    if (route.op == ExprOp::ATTENTION_BACKWARD_BIAS) {
        // cuDNN writes dBias in the same native dtype as the Q path for Thor's attention stages.
        // Keep the stage output dtype pinned to the local Q dtype even when the original additive-bias
        // tensor is FP32, so merged backward stages do not hand an FP32 output tensor to cuDNN.
        route.output_dtype = stage_expr.nodes.at(q_local).output_dtype;
    }
    route.attention_seq_len_q_node = q_seq_len_local;
    route.attention_seq_len_kv_node = kv_seq_len_local;
    route.attention_ragged_offset_q_node = q_ragged_offset_local;
    route.attention_ragged_offset_kv_node = kv_ragged_offset_local;
    route.attention_dropout_seed_node = dropout_seed_local;
    route.attention_dropout_offset_node = dropout_offset_local;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = static_cast<uint32_t>(stage_expr.nodes.size() - 1);

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = stage_expr.output_node,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::AttentionBackward,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static std::string attentionBackwardMergeKey(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::AttentionBackward || stage.expr.output_node >= stage.expr.nodes.size()) {
        throw std::runtime_error("attentionBackwardMergeKey expected an attention-backward stage.");
    }
    const ExprNode& node = stage.expr.nodes[stage.expr.output_node];
    std::string key = "attn_bwd";
    for (uint32_t value_id : stage.input_value_ids) {
        key += ":" + std::to_string(value_id);
    }
    key += ";qL=" + std::to_string(static_cast<int>(node.attention_q_layout));
    key += ";kL=" + std::to_string(static_cast<int>(node.attention_k_layout));
    key += ";vL=" + std::to_string(static_cast<int>(node.attention_v_layout));
    key += ";oL=" + std::to_string(static_cast<int>(node.attention_o_layout));
    key += ";mask=" + std::to_string(static_cast<int>(node.attention_mask_kind));
    key += ";left=" + std::to_string(node.attention_diagonal_left_bound);
    key += ";right=" + std::to_string(node.attention_diagonal_right_bound);
    key += ";hasScale=" + std::to_string(node.attention_has_scale ? 1 : 0);
    key += ";scale=" + formatFloatCanonical(node.attention_scale);
    key += ";alibi=" + std::to_string(node.attention_use_alibi_mask ? 1 : 0);
    key += ";bias=" + std::to_string(node.attention_use_bias ? 1 : 0);
    key += ";padding=" + std::to_string(node.attention_use_padding_mask ? 1 : 0);
    key += ";ragged=" + std::to_string(node.attention_use_ragged_offsets ? 1 : 0);
    key += ";paged=" + std::to_string(node.attention_use_paged_kv_cache ? 1 : 0);
    key += ";pagedMax=" + std::to_string(node.attention_paged_kv_max_sequence_length);
    key += ";dropout=" + formatFloatCanonical(node.attention_dropout_probability);
    key += ";compute=" + optionalDTypeSignature(node.compute_dtype);
    return key;
}

static void mergeAttentionBackwardStages(std::vector<PhysicalExecutionStage>& stages) {
    std::unordered_map<std::string, size_t> first_by_key;
    std::vector<PhysicalExecutionStage> merged;
    merged.reserve(stages.size());

    for (PhysicalExecutionStage& stage : stages) {
        if (stage.kind != PhysicalExecutionStage::Kind::AttentionBackward) {
            merged.push_back(std::move(stage));
            continue;
        }

        const std::string key = attentionBackwardMergeKey(stage);
        auto it = first_by_key.find(key);
        if (it == first_by_key.end()) {
            first_by_key.emplace(key, merged.size());
            merged.push_back(std::move(stage));
            continue;
        }

        PhysicalExecutionStage& target = merged.at(it->second);
        if (stage.outputs.size() != 1 || stage.expr.output_node >= stage.expr.nodes.size()) {
            throw std::runtime_error("Attention-backward stage merge expected one source output.");
        }
        ExprNode route = stage.expr.nodes.at(stage.expr.output_node);
        const uint32_t new_local_idx = static_cast<uint32_t>(target.expr.nodes.size());
        target.expr.nodes.push_back(std::move(route));
        CompiledStageOutput out = stage.outputs.front();
        out.local_node_idx = new_local_idx;
        target.outputs.push_back(std::move(out));
    }

    stages = std::move(merged);
}

static PhysicalExecutionStage buildConvolutionStage(const PhysicalExpression& expr,
                                                    uint32_t node_idx,
                                                    uint32_t output_value_id,
                                                    const std::string& output_name,
                                                    const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isConvolutionOp(node.op)) {
        throw std::runtime_error("buildConvolutionStage called on non-convolution node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("Convolution node missing required inputs.");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(2u);

    auto inputNameForSlot = [](uint32_t slot) { return std::string("__arg") + std::to_string(slot); };

    auto bind_parent_to_local_tensor_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Convolution input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.input_tensor_dtype;
        } else {
            throw std::runtime_error("Missing value id for convolution input.");
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("Convolution parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("Convolution parent node missing resolved actual input dtype.");
        }

        stage_expr.inputs.push_back(NamedInput{inputNameForSlot(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
    };

    bind_parent_to_local_tensor_input(node.lhs, static_cast<uint32_t>(stage_expr.inputs.size()));
    const uint32_t lhs_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);
    bind_parent_to_local_tensor_input(node.rhs, static_cast<uint32_t>(stage_expr.inputs.size()));
    const uint32_t rhs_local = static_cast<uint32_t>(stage_expr.nodes.size() - 1);

    ExprNode route = node;
    route.lhs = lhs_local;
    route.rhs = rhs_local;
    route.aux = UINT32_MAX;
    route.alpha_node = UINT32_MAX;
    route.beta_node = UINT32_MAX;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = static_cast<uint32_t>(stage_expr.nodes.size() - 1);

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(
        CompiledStageOutput{.name = output_name, .local_node_idx = stage_expr.output_node, .value_id = output_value_id});

    return PhysicalExecutionStage{.kind = PhysicalExecutionStage::Kind::Convolution,
                                  .expr = std::move(stage_expr),
                                  .input_value_ids = std::move(input_value_ids),
                                  .outputs = std::move(stage_outputs)};
}

static PhysicalExecutionStage buildConvolutionBackwardStage(const PhysicalExpression& expr,
                                                            uint32_t node_idx,
                                                            uint32_t output_value_id,
                                                            const std::string& output_name,
                                                            const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isConvolutionBackwardOp(node.op)) {
        throw std::runtime_error("buildConvolutionBackwardStage called on unsupported node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("ConvolutionBackward node missing lhs or rhs input.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});
    stage_expr.inputs.push_back(NamedInput{"__arg1", 1});

    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(2);

    auto bind_parent_to_local_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("ConvolutionBackward input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.input_tensor_dtype;
        } else {
            throw std::runtime_error("Missing value id for ConvolutionBackward input.");
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("ConvolutionBackward parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("ConvolutionBackward parent node missing resolved actual input dtype.");
        }

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
    };

    bind_parent_to_local_input(node.lhs, 0);
    bind_parent_to_local_input(node.rhs, 1);

    ExprNode route = node;
    route.lhs = 0;
    route.rhs = 1;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = 2;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = 2,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::ConvolutionBackward,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static PhysicalExecutionStage buildReduceMinMaxBackwardStage(const PhysicalExpression& expr,
                                                             uint32_t node_idx,
                                                             uint32_t output_value_id,
                                                             const std::string& output_name,
                                                             const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isReduceMinMaxBackwardOp(node.op)) {
        throw std::runtime_error("buildReduceMinMaxBackwardStage called on unsupported node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("ReduceMinMaxBackward node missing lhs or rhs input.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});
    stage_expr.inputs.push_back(NamedInput{"__arg1", 1});

    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(2);

    auto bind_parent_to_local_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("ReduceMinMaxBackward input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.input_tensor_dtype;
        } else {
            throw std::runtime_error("Missing value id for ReduceMinMaxBackward input.");
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("ReduceMinMaxBackward parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("ReduceMinMaxBackward parent node missing resolved actual input dtype.");
        }

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
    };

    bind_parent_to_local_input(node.lhs, 0);
    bind_parent_to_local_input(node.rhs, 1);

    ExprNode route = node;
    route.lhs = 0;
    route.rhs = 1;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = 2;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = 2,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::ReduceMinMaxBackward,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}


static PhysicalExecutionStage buildScanMinMaxBackwardStage(const PhysicalExpression& expr,
                                                           uint32_t node_idx,
                                                           uint32_t output_value_id,
                                                           const std::string& output_name,
                                                           const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isScanMinMaxBackwardOp(node.op)) {
        throw std::runtime_error("buildScanMinMaxBackwardStage called on unsupported node.");
    }
    const bool segmented = node.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD;
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || (segmented && node.aux == UINT32_MAX)) {
        throw std::runtime_error("ScanMinMaxBackward node missing input, grad, or offsets input.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0});
    stage_expr.inputs.push_back(NamedInput{"__arg1", 1});
    if (segmented) {
        stage_expr.inputs.push_back(NamedInput{"__arg2", 2});
    }

    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(segmented ? 3 : 2);

    auto bind_parent_to_local_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("ScanMinMaxBackward input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        std::optional<DataType> actual_input_dtype = std::nullopt;

        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it != node_output_value_id.end()) {
            input_value_ids.push_back(out_it->second);
            actual_input_dtype = parent.output_dtype;
        } else if (parent.op == ExprOp::INPUT) {
            input_value_ids.push_back(parent.input_slot);
            actual_input_dtype = parent.input_tensor_dtype;
        } else {
            throw std::runtime_error("Missing value id for ScanMinMaxBackward input.");
        }

        if (!parent.output_dtype.has_value()) {
            throw std::runtime_error("ScanMinMaxBackward parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.has_value()) {
            throw std::runtime_error("ScanMinMaxBackward parent node missing resolved actual input dtype.");
        }

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.value();
        input_node.output_dtype = parent.output_dtype;
        input_node.compute_dtype = parent.compute_dtype;
        input_node.backward_output_dtype = parent.backward_output_dtype;
        input_node.backward_compute_dtype = parent.backward_compute_dtype;
        stage_expr.nodes.push_back(std::move(input_node));
    };

    bind_parent_to_local_input(node.lhs, 0);
    bind_parent_to_local_input(node.rhs, 1);
    if (segmented) {
        bind_parent_to_local_input(node.aux, 2);
    }

    ExprNode route = node;
    route.lhs = 0;
    route.rhs = 1;
    route.aux = segmented ? 2 : UINT32_MAX;
    stage_expr.nodes.push_back(std::move(route));
    stage_expr.output_node = segmented ? 3 : 2;

    std::vector<CompiledStageOutput> stage_outputs;
    stage_outputs.push_back(CompiledStageOutput{
        .name = output_name,
        .local_node_idx = stage_expr.output_node,
        .value_id = output_value_id,
    });

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::ScanMinMaxBackward,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

struct PlannedExecution {
    std::vector<PhysicalExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
    std::vector<CompiledValueAlias> value_aliases;
};

static bool regionContainsShapeOnlyOp(const PhysicalExpression& expr, const std::unordered_set<uint32_t>& region_nodes) {
    for (uint32_t node_idx : region_nodes) {
        if (node_idx >= expr.nodes.size()) {
            throw std::runtime_error("regionContainsShapeOnlyOp node index out of range.");
        }
        const ExprNode& node = expr.nodes[node_idx];
        if (node.op == ExprOp::RESHAPE || node.op == ExprOp::STRIDED_VIEW || node.op == ExprOp::UNSQUEEZE || node.op == ExprOp::SQUEEZE) {
            return true;
        }
    }
    return false;
}

static bool regionSupportsTiledTransposeMaterialization(const PhysicalExpression& expr, const std::unordered_set<uint32_t>& region_nodes) {
    if (region_nodes.empty()) {
        return false;
    }
    if (regionContainsShapeOnlyOp(expr, region_nodes)) {
        return false;
    }

    // The tiled transpose materializer evaluates the fused expression in the
    // producer's logical row-major index space and only changes the final store
    // pattern.  That is valid for ordinary flat elementwise regions with any
    // number of same-shaped tensor inputs: all tensor operands are still read at
    // the same logical flat index, and the computed scalar is staged through the
    // padded shared-memory tile before being written transposed.
    //
    // Broadcast / mixed-shape fused regions are accepted here too.  At stamp
    // time they are compiled through the specialized broadcast emitter, which
    // uses the pre-transpose logical output index space for input-offset math
    // and still stores through the tiled transposed materializer.
    return true;
}

static bool isStorageAliasOp(ExprOp op) { return op == ExprOp::RESHAPE || op == ExprOp::STRIDED_VIEW; }

static bool reshapeAliasPreservesDType(const PhysicalExpression& expr, uint32_t reshape_idx) {
    if (reshape_idx >= expr.nodes.size()) {
        throw std::runtime_error("reshapeAliasPreservesDType node index out of range.");
    }
    const ExprNode& reshape_node = expr.nodes[reshape_idx];
    if (!isStorageAliasOp(reshape_node.op)) {
        return false;
    }
    if (reshape_node.lhs == UINT32_MAX || reshape_node.lhs >= expr.nodes.size()) {
        throw std::runtime_error("Storage alias node is missing a valid source while checking dtype preservation.");
    }
    if (!reshape_node.output_dtype.has_value()) {
        throw std::runtime_error("Storage alias node missing resolved output dtype.");
    }
    const ExprNode& source_node = expr.nodes[reshape_node.lhs];
    if (!source_node.output_dtype.has_value()) {
        throw std::runtime_error("Storage alias source node missing resolved output dtype.");
    }
    return reshape_node.output_dtype.value() == source_node.output_dtype.value();
}

static bool isMatmulBiasGradientReduction(const PhysicalExpression& expr, uint32_t node_idx, uint32_t& matmul_idx) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }
    const ExprNode& node = expr.nodes.at(node_idx);
    if (node.op != ExprOp::REDUCE_SUM || node.lhs == UINT32_MAX || node.lhs >= expr.nodes.size()) {
        return false;
    }
    if (node.reduction_axes.size() != 1 || node.reduction_axes[0] != 0 || node.squeeze_axes.size() != 1 || node.squeeze_axes[0] != 0) {
        return false;
    }
    const ExprNode& source = expr.nodes.at(node.lhs);
    if (!isMatmulOp(source.op) || source.matmul_backward_epilogue == MatmulBackwardEpilogue::Default) {
        return false;
    }
    matmul_idx = node.lhs;
    return true;
}

static std::vector<uint32_t> computeNodeUseCounts(const PhysicalExpression& expr) {
    std::vector<uint32_t> use_counts(expr.nodes.size(), 0);
    auto bump = [&](uint32_t idx) {
        if (idx != UINT32_MAX && idx < use_counts.size()) {
            ++use_counts[idx];
        }
    };
    for (const ExprNode& node : expr.nodes) {
        if (!Expression::isLeafOp(node.op)) {
            bump(node.lhs);
        }
        if (Expression::isBinaryOp(node.op)) {
            bump(node.rhs);
        }
        if (node.op == ExprOp::WHERE) {
            bump(node.rhs);
            bump(node.aux);
        }
        if (node.op == ExprOp::GEMM || node.op == ExprOp::ATTENTION_BACKWARD_Q || node.op == ExprOp::ATTENTION_BACKWARD_K ||
            node.op == ExprOp::ATTENTION_BACKWARD_V || node.op == ExprOp::ATTENTION_BACKWARD_BIAS) {
            bump(node.aux);
        }
        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            for (uint32_t input_node : node.cuda_kernel_input_nodes) {
                bump(input_node);
            }
        }
        bump(node.alpha_node);
        bump(node.beta_node);
        bump(node.matmul_epilogue_aux);
        bump(node.attention_seq_len_q_node);
        bump(node.attention_seq_len_kv_node);
        bump(node.attention_ragged_offset_q_node);
        bump(node.attention_ragged_offset_kv_node);
        bump(node.attention_page_table_k_node);
        bump(node.attention_page_table_v_node);
        bump(node.attention_dropout_seed_node);
        bump(node.attention_dropout_offset_node);
    }
    return use_counts;
}

struct InPlaceRopeMaterializationCandidate {
    uint32_t rope_root = UINT32_MAX;
    uint32_t source_node = UINT32_MAX;
    std::vector<uint64_t> logical_dims;
};

static std::optional<InPlaceRopeMaterializationCandidate> classifySplitProjectionInPlaceRopeCandidate(
    const PhysicalExpression& expr, uint32_t rope_root, const std::vector<uint32_t>& node_use_counts) {
    if (rope_root >= expr.nodes.size()) {
        return std::nullopt;
    }
    const ExprNode& rope = expr.nodes.at(rope_root);
    if (rope.op != ExprOp::ROPE || rope.lhs == UINT32_MAX || rope.lhs >= expr.nodes.size()) {
        return std::nullopt;
    }
    if (!rope.rope_allow_in_place_materialization) {
        return std::nullopt;
    }
    if (!rope.output_dtype.has_value()) {
        return std::nullopt;
    }
    const ExprNode& reshape = expr.nodes.at(rope.lhs);
    if (reshape.op != ExprOp::RESHAPE || reshape.lhs == UINT32_MAX || reshape.lhs >= expr.nodes.size()) {
        return std::nullopt;
    }
    if (!reshapeAliasPreservesDType(expr, rope.lhs) || reshape.reshape_dims.size() != 4) {
        return std::nullopt;
    }
    if (node_use_counts.at(rope.lhs) != 1) {
        return std::nullopt;
    }
    const ExprNode& source = expr.nodes.at(reshape.lhs);
    if (!isMatmulOp(source.op) || !source.output_dtype.has_value() || source.output_dtype.value() != rope.output_dtype.value()) {
        return std::nullopt;
    }
    // This optimization mutates the projection output in-place. Only use it for private split-Q/K projection
    // materializations whose dense reshape is the sole consumer. Bias/GEMM and packed-QKV strided views intentionally
    // fall back to the normal out-of-place grouped RoPE path.
    if (source.op != ExprOp::MATMUL || node_use_counts.at(reshape.lhs) != 1) {
        return std::nullopt;
    }
    return InPlaceRopeMaterializationCandidate{.rope_root = rope_root, .source_node = reshape.lhs, .logical_dims = reshape.reshape_dims};
}

static bool sameRopeOptionsForInPlaceGrouping(const PhysicalExpression& expr,
                                              const std::vector<InPlaceRopeMaterializationCandidate>& candidates) {
    if (candidates.empty()) {
        return false;
    }
    const ExprNode& first = expr.nodes.at(candidates.front().rope_root);
    for (const InPlaceRopeMaterializationCandidate& candidate : candidates) {
        const ExprNode& node = expr.nodes.at(candidate.rope_root);
        if (node.rope_sequence_axis != first.rope_sequence_axis || node.rope_head_dim_axis != first.rope_head_dim_axis ||
            node.rope_rotary_dim != first.rope_rotary_dim || node.rope_base != first.rope_base ||
            node.rope_position_offset != first.rope_position_offset || node.rope_interleaved != first.rope_interleaved ||
            node.rope_inverse != first.rope_inverse || node.rope_scaling_kind != first.rope_scaling_kind ||
            node.rope_scaling_factor != first.rope_scaling_factor ||
            node.rope_original_max_position_embeddings != first.rope_original_max_position_embeddings ||
            node.rope_attention_factor != first.rope_attention_factor || node.rope_yarn_beta_fast != first.rope_yarn_beta_fast ||
            node.rope_yarn_beta_slow != first.rope_yarn_beta_slow ||
            node.rope_llama3_low_freq_factor != first.rope_llama3_low_freq_factor ||
            node.rope_llama3_high_freq_factor != first.rope_llama3_high_freq_factor ||
            node.rope_long_rope_short_factors != first.rope_long_rope_short_factors ||
            node.rope_long_rope_long_factors != first.rope_long_rope_long_factors ||
            node.rope_allow_in_place_materialization != first.rope_allow_in_place_materialization) {
            return false;
        }
    }
    return true;
}

static PhysicalExecutionStage buildInPlaceRopeStage(const PhysicalExpression& expr,
                                                    const std::vector<InPlaceRopeMaterializationCandidate>& candidates,
                                                    const std::unordered_map<uint32_t, uint32_t>& node_output_value_id,
                                                    const std::unordered_map<uint32_t, uint32_t>& rope_output_value_ids) {
    if (candidates.empty()) {
        throw std::runtime_error("buildInPlaceRopeStage requires at least one candidate.");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    std::vector<CompiledStageOutput> stage_outputs;
    input_value_ids.reserve(candidates.size());
    stage_outputs.reserve(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        const InPlaceRopeMaterializationCandidate& candidate = candidates[i];
        auto value_it = node_output_value_id.find(candidate.source_node);
        if (value_it == node_output_value_id.end()) {
            throw std::runtime_error("In-place RoPE candidate source has not been materialized.");
        }
        input_value_ids.push_back(value_it->second);

        const ExprNode& source = expr.nodes.at(candidate.source_node);
        stage_expr.inputs.push_back(NamedInput{"__arg" + std::to_string(i), static_cast<uint32_t>(i), NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = static_cast<uint32_t>(i);
        input_node.input_tensor_dtype = source.output_dtype;
        input_node.output_dtype = source.output_dtype;
        input_node.compute_dtype = source.compute_dtype;
        const uint32_t input_local = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(input_node));

        ExprNode reshape_node;
        reshape_node.op = ExprOp::RESHAPE;
        reshape_node.lhs = input_local;
        reshape_node.reshape_dims = candidate.logical_dims;
        reshape_node.output_dtype = source.output_dtype;
        reshape_node.compute_dtype = source.compute_dtype;
        const uint32_t reshape_local = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(reshape_node));

        ExprNode rope_node = expr.nodes.at(candidate.rope_root);
        rope_node.lhs = reshape_local;
        const uint32_t rope_local = static_cast<uint32_t>(stage_expr.nodes.size());
        stage_expr.nodes.push_back(std::move(rope_node));

        auto out_it = rope_output_value_ids.find(candidate.rope_root);
        if (out_it == rope_output_value_ids.end()) {
            throw std::runtime_error("In-place RoPE candidate missing output value id.");
        }
        stage_outputs.push_back(CompiledStageOutput{.name = "", .local_node_idx = rope_local, .value_id = out_it->second});
    }

    stage_expr.output_node = stage_outputs.front().local_node_idx;
    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::InPlaceRope,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

static PlannedExecution planExecution(const PhysicalOutputs& outputs) {
    if (!outputs.expr) {
        throw std::runtime_error("Cannot split null PhysicalOutputs expression.");
    }
    if (outputs.outputs.empty()) {
        throw std::runtime_error("Cannot split empty PhysicalOutputs.");
    }

    PhysicalExpression expr = *outputs.expr;
    if (expr.nodes.empty()) {
        throw std::runtime_error("Cannot split empty PhysicalExpression.");
    }

    for (const NamedOutput& output : outputs.outputs) {
        if (output.node_idx >= expr.nodes.size()) {
            throw std::runtime_error("PhysicalOutputs contains output node_idx out of range.");
        }
    }
    validateRaggedRuntimeExtentConsumers(expr);

    const std::vector<uint32_t> node_use_counts = computeNodeUseCounts(expr);

    std::unordered_map<uint32_t, uint32_t> node_output_value_id;
    std::map<std::string, uint32_t> fused_region_value_id;
    std::map<std::string, uint32_t> stage_boundary_value_id;

    struct TerminalFusedGroup {
        std::unordered_set<uint32_t> region_nodes;
        std::unordered_set<uint32_t> dependency_value_ids;
        std::vector<RequestedStageOutput> outputs;
        std::map<std::string, uint32_t> exact_region_value_id;
        bool emitted = false;
    };

    std::vector<std::optional<TerminalFusedGroup>> terminal_groups;
    std::map<std::string, size_t> pending_terminal_region_to_group;

    PlannedExecution planned;
    uint32_t next_value_id = expr.numInputs();

    std::unordered_map<uint32_t, std::string> final_output_name_by_node;
    for (const NamedOutput& output : outputs.outputs) {
        final_output_name_by_node.emplace(output.node_idx, output.name);
    }

    std::unordered_map<uint32_t, std::vector<PendingMatmulBgradOutput>> pending_matmul_bgrad_outputs;
    for (const NamedOutput& output : outputs.outputs) {
        uint32_t matmul_idx = UINT32_MAX;
        if (isMatmulBiasGradientReduction(expr, output.node_idx, matmul_idx)) {
            pending_matmul_bgrad_outputs[matmul_idx].push_back(PendingMatmulBgradOutput{
                .reduce_node_idx = output.node_idx,
                .value_id = UINT32_MAX,
                .name = output.name,
            });
        }
    }

    auto takePendingMatmulBgradOutputs = [&](uint32_t matmul_idx) -> std::vector<PendingMatmulBgradOutput> {
        std::vector<PendingMatmulBgradOutput> result;
        auto pending_it = pending_matmul_bgrad_outputs.find(matmul_idx);
        if (pending_it == pending_matmul_bgrad_outputs.end()) {
            return result;
        }
        std::unordered_set<uint32_t> seen_reduce_nodes;
        for (PendingMatmulBgradOutput& output : pending_it->second) {
            if (!seen_reduce_nodes.insert(output.reduce_node_idx).second) {
                continue;
            }
            auto existing_it = node_output_value_id.find(output.reduce_node_idx);
            if (existing_it != node_output_value_id.end()) {
                output.value_id = existing_it->second;
            } else {
                output.value_id = next_value_id++;
                node_output_value_id[output.reduce_node_idx] = output.value_id;
            }
            result.push_back(output);
        }
        return result;
    };

    std::function<void(size_t)> materializeTerminalGroup;
    std::function<void(uint32_t)> emitForDependency;
    std::function<void(uint32_t)> emitCudaKernelStage;
    std::function<uint32_t(uint32_t, std::optional<uint32_t>)> emitStorageAlias;
    std::function<bool(uint32_t, uint32_t, const std::string&)> tryEmitTiledTransposeMaterializedFusedStage;
    std::function<std::unordered_set<uint32_t>(const std::vector<uint32_t>&)> tryEmitGroupedRopeMaterialization;

    materializeTerminalGroup = [&](size_t group_idx) {
        if (group_idx >= terminal_groups.size() || !terminal_groups[group_idx].has_value()) {
            throw std::runtime_error("materializeTerminalGroup group_idx out of range or inactive.");
        }

        TerminalFusedGroup& group = *terminal_groups[group_idx];
        if (group.emitted) {
            return;
        }

        if (group.outputs.empty()) {
            throw std::runtime_error("materializeTerminalGroup found empty terminal group.");
        }

        planned.stages.push_back(buildFusedStage(expr, group.region_nodes, group.outputs, node_output_value_id));

        for (const auto& [region_key, value_id] : group.exact_region_value_id) {
            fused_region_value_id[region_key] = value_id;
        }

        group.emitted = true;
    };

    emitStorageAlias = [&](uint32_t reshape_idx, std::optional<uint32_t> forced_value_id) -> uint32_t {
        auto existing_it = node_output_value_id.find(reshape_idx);
        if (existing_it != node_output_value_id.end()) {
            return existing_it->second;
        }
        if (reshape_idx >= expr.nodes.size()) {
            throw std::runtime_error("Reshape alias node index out of range.");
        }
        const ExprNode& reshape_node = expr.nodes[reshape_idx];
        if (!isStorageAliasOp(reshape_node.op)) {
            throw std::runtime_error("emitStorageAlias called on a non-alias node.");
        }
        if (!reshapeAliasPreservesDType(expr, reshape_idx)) {
            throw std::runtime_error("emitStorageAlias called on a reshape that requires dtype conversion.");
        }
        if (reshape_node.lhs == UINT32_MAX || reshape_node.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Reshape alias node is missing a valid source.");
        }
        const std::vector<uint64_t>& alias_dims =
            reshape_node.op == ExprOp::STRIDED_VIEW ? reshape_node.view_dims : reshape_node.reshape_dims;
        if (alias_dims.empty()) {
            throw std::runtime_error("Storage alias node is missing output dimensions.");
        }
        std::vector<uint64_t> alias_strides;
        uint64_t alias_offset = 0;
        if (reshape_node.op == ExprOp::STRIDED_VIEW) {
            alias_strides = reshape_node.view_strides;
            alias_offset = reshape_node.view_element_offset;
            if (alias_strides.size() != alias_dims.size()) {
                throw std::runtime_error("Strided-view alias dimensions and strides must have the same rank.");
            }
        }

        const uint32_t source_node_idx = reshape_node.lhs;
        const ExprNode& source_node = expr.nodes[source_node_idx];
        uint32_t source_value_id = UINT32_MAX;
        if (isStorageAliasOp(source_node.op) && reshapeAliasPreservesDType(expr, source_node_idx)) {
            source_value_id = emitStorageAlias(source_node_idx, std::nullopt);
        } else if (source_node.op == ExprOp::INPUT && !inputRequiresMaterialization(source_node)) {
            source_value_id = source_node.input_slot;
        } else {
            emitForDependency(source_node_idx);
            auto source_it = node_output_value_id.find(source_node_idx);
            if (source_it == node_output_value_id.end()) {
                throw std::runtime_error("Failed to materialize source value for reshape alias.");
            }
            source_value_id = source_it->second;
        }

        const uint32_t alias_value_id = forced_value_id.has_value() ? forced_value_id.value() : next_value_id++;
        node_output_value_id[reshape_idx] = alias_value_id;
        planned.value_aliases.push_back(CompiledValueAlias{
            .value_id = alias_value_id,
            .source_value_id = source_value_id,
            .dimensions = alias_dims,
            .strides = std::move(alias_strides),
            .element_offset = alias_offset,
        });
        return alias_value_id;
    };

    tryEmitTiledTransposeMaterializedFusedStage = [&](uint32_t transpose_idx, uint32_t output_value_id, const std::string& output_name) {
        if (transpose_idx >= expr.nodes.size()) {
            throw std::runtime_error("Transpose node index out of range while planning fused transposed materialization.");
        }
        const ExprNode& transpose_node = expr.nodes[transpose_idx];
        if (!isTransposeOp(transpose_node.op)) {
            return false;
        }
        if (transpose_node.lhs == UINT32_MAX || transpose_node.lhs >= expr.nodes.size()) {
            throw std::runtime_error("Transpose node missing lhs while planning fused transposed materialization.");
        }

        const uint32_t materialized_parent_idx = transpose_node.lhs;
        const ExprNode& materialized_parent = expr.nodes[materialized_parent_idx];
        if (isStageBoundaryOp(materialized_parent.op)) {
            emitForDependency(materialized_parent_idx);
            if (node_output_value_id.find(materialized_parent_idx) == node_output_value_id.end()) {
                throw std::runtime_error("Failed to materialize transpose parent boundary stage.");
            }
            node_output_value_id[transpose_idx] = output_value_id;
            planned.stages.push_back(buildInputTransposedMaterializationStage(
                expr, materialized_parent_idx, output_value_id, output_name, node_output_value_id));
            return true;
        }

        std::unordered_set<uint32_t> forced_transpose_boundaries;
        for (uint32_t logical_transpose_idx : collectReachableLogicalTransposeNodes(expr, materialized_parent_idx)) {
            forced_transpose_boundaries.insert(logical_transpose_idx);
        }
        for (uint32_t forced_transpose_idx : forced_transpose_boundaries) {
            emitForDependency(forced_transpose_idx);
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegionStoppingAt(expr, materialized_parent_idx, forced_transpose_boundaries, region);
        if (regionContainsShapeOnlyOp(expr, region)) {
            return false;
        }

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        if (!regionSupportsTiledTransposeMaterialization(expr, region)) {
            return false;
        }

        node_output_value_id[transpose_idx] = output_value_id;
        std::vector<RequestedStageOutput> requested_outputs{RequestedStageOutput{
            .name = output_name,
            .old_root_idx = materialized_parent_idx,
            .value_id = output_value_id,
            .materialized_layout = MaterializedTensorLayout::Transposed,
        }};
        planned.stages.push_back(buildFusedStage(expr, region, requested_outputs, node_output_value_id));
        return true;
    };

    tryEmitGroupedRopeMaterialization = [&](const std::vector<uint32_t>& roots) -> std::unordered_set<uint32_t> {
        std::vector<uint32_t> rope_roots;
        rope_roots.reserve(roots.size());
        for (uint32_t root_candidate : roots) {
            if (root_candidate == UINT32_MAX || root_candidate >= expr.nodes.size()) {
                continue;
            }
            if (node_output_value_id.find(root_candidate) != node_output_value_id.end()) {
                continue;
            }
            if (expr.nodes[root_candidate].op == ExprOp::ROPE) {
                rope_roots.push_back(root_candidate);
            }
        }
        if (rope_roots.size() < 2) {
            return {};
        }

        std::vector<InPlaceRopeMaterializationCandidate> in_place_candidates;
        in_place_candidates.reserve(rope_roots.size());
        bool all_in_place_candidates = true;
        for (uint32_t rope_root : rope_roots) {
            std::optional<InPlaceRopeMaterializationCandidate> candidate =
                classifySplitProjectionInPlaceRopeCandidate(expr, rope_root, node_use_counts);
            if (!candidate.has_value()) {
                all_in_place_candidates = false;
                break;
            }
            in_place_candidates.push_back(std::move(candidate.value()));
        }
        if (all_in_place_candidates && sameRopeOptionsForInPlaceGrouping(expr, in_place_candidates)) {
            for (const InPlaceRopeMaterializationCandidate& candidate : in_place_candidates) {
                emitForDependency(candidate.source_node);
            }

            std::unordered_map<uint32_t, uint32_t> rope_output_value_ids;
            std::unordered_set<uint32_t> emitted_roots;
            for (uint32_t rope_root : rope_roots) {
                if (node_output_value_id.find(rope_root) != node_output_value_id.end()) {
                    return {};
                }
                const uint32_t value_id = next_value_id++;
                node_output_value_id[rope_root] = value_id;
                rope_output_value_ids.emplace(rope_root, value_id);
                emitted_roots.insert(rope_root);
            }

            planned.stages.push_back(buildInPlaceRopeStage(expr, in_place_candidates, node_output_value_id, rope_output_value_ids));
            return emitted_roots;
        }

        std::unordered_set<uint32_t> merged_region;
        std::unordered_set<uint32_t> boundary_nodes;
        std::vector<std::string> region_sigs;
        region_sigs.reserve(rope_roots.size());

        for (uint32_t rope_root : rope_roots) {
            const std::string region_sig = fusedRegionSignature(expr, rope_root);
            if (fused_region_value_id.find(region_sig) != fused_region_value_id.end() ||
                pending_terminal_region_to_group.find(region_sig) != pending_terminal_region_to_group.end()) {
                return {};
            }

            std::unordered_set<uint32_t> region;
            collectFusableRegion(expr, rope_root, region);
            std::unordered_set<uint32_t> deps;
            collectBoundaryDependencies(expr, region, deps);

            for (uint32_t dep : deps) {
                if (dep >= expr.nodes.size()) {
                    throw std::runtime_error("Grouped RoPE dependency node index out of range.");
                }
                // Packed-QKV views are intentionally not part of this optimization path. Fused kernels still
                // cannot consume non-dense STRIDED_VIEW dependencies directly; packed QKV remains a dormant
                // experimental path and should not receive split-path RoPE upgrades by accident.
                if (expr.nodes[dep].op == ExprOp::STRIDED_VIEW) {
                    return {};
                }
            }

            merged_region.insert(region.begin(), region.end());
            boundary_nodes.insert(deps.begin(), deps.end());
            region_sigs.push_back(region_sig);
        }

        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        std::vector<RequestedStageOutput> requested_outputs;
        requested_outputs.reserve(rope_roots.size());
        std::unordered_set<uint32_t> emitted_roots;
        for (size_t i = 0; i < rope_roots.size(); ++i) {
            const uint32_t rope_root = rope_roots[i];
            if (node_output_value_id.find(rope_root) != node_output_value_id.end()) {
                return {};
            }
            const uint32_t value_id = next_value_id++;
            node_output_value_id[rope_root] = value_id;
            fused_region_value_id.emplace(region_sigs[i], value_id);
            requested_outputs.push_back(RequestedStageOutput{
                .name = "",
                .old_root_idx = rope_root,
                .value_id = value_id,
            });
            emitted_roots.insert(rope_root);
        }

        planned.stages.push_back(buildFusedStage(expr, merged_region, requested_outputs, node_output_value_id));
        return emitted_roots;
    };

    emitCudaKernelStage = [&](uint32_t root_idx) {
        if (root_idx >= expr.nodes.size()) {
            throw std::runtime_error("emitCudaKernelStage root index out of range.");
        }
        const ExprNode& root = expr.nodes[root_idx];
        if (root.op != ExprOp::CUDA_KERNEL_OUTPUT) {
            throw std::runtime_error("emitCudaKernelStage called on non-CUDA-kernel output node.");
        }

        const uint32_t spec_idx = root.cuda_kernel_spec_index;
        if (spec_idx >= expr.cuda_kernel_expressions.size() || !expr.cuda_kernel_expressions[spec_idx]) {
            throw std::runtime_error("emitCudaKernelStage references missing CUDA kernel spec.");
        }

        bool already_emitted = true;
        std::vector<uint32_t> kernel_output_nodes;
        for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
            const ExprNode& candidate = expr.nodes[node_idx];
            if (candidate.op == ExprOp::CUDA_KERNEL_OUTPUT && candidate.cuda_kernel_spec_index == spec_idx &&
                candidate.cuda_kernel_input_nodes == root.cuda_kernel_input_nodes) {
                kernel_output_nodes.push_back(node_idx);
                if (node_output_value_id.find(node_idx) == node_output_value_id.end()) {
                    already_emitted = false;
                }
            }
        }
        if (already_emitted) {
            return;
        }

        for (uint32_t input_node_idx : root.cuda_kernel_input_nodes) {
            if (input_node_idx >= expr.nodes.size()) {
                throw std::runtime_error("CudaKernelExpression input dependency node index out of range.");
            }
            const ExprNode& parent = expr.nodes[input_node_idx];
            if ((parent.op != ExprOp::INPUT && parent.op != ExprOp::RUNTIME_SCALAR && parent.op != ExprOp::TENSOR_RUNTIME_SCALAR) ||
                inputRequiresMaterialization(parent)) {
                emitForDependency(input_node_idx);
            }
        }

        std::sort(kernel_output_nodes.begin(), kernel_output_nodes.end(), [&](uint32_t a, uint32_t b) {
            return expr.nodes[a].cuda_kernel_output_index < expr.nodes[b].cuda_kernel_output_index;
        });

        std::vector<RequestedStageOutput> requested_outputs;
        requested_outputs.reserve(kernel_output_nodes.size());
        for (uint32_t node_idx : kernel_output_nodes) {
            uint32_t value_id;
            auto existing_it = node_output_value_id.find(node_idx);
            if (existing_it != node_output_value_id.end()) {
                value_id = existing_it->second;
            } else {
                value_id = next_value_id++;
                node_output_value_id[node_idx] = value_id;
            }
            auto final_name_it = final_output_name_by_node.find(node_idx);
            requested_outputs.push_back(RequestedStageOutput{
                .name = final_name_it == final_output_name_by_node.end() ? std::string{} : final_name_it->second,
                .old_root_idx = node_idx,
                .value_id = value_id,
            });
        }

        planned.stages.push_back(buildCudaKernelStage(expr, requested_outputs, node_output_value_id));
    };

    emitForDependency = [&](uint32_t root_idx) {
        if (node_output_value_id.find(root_idx) != node_output_value_id.end()) {
            return;
        }

        const ExprNode& root = expr.nodes[root_idx];
        if (isStorageAliasOp(root.op) && reshapeAliasPreservesDType(expr, root_idx)) {
            emitStorageAlias(root_idx, std::nullopt);
            return;
        }
        if (isTransposeOp(root.op)) {
            const uint32_t stage_out_id = next_value_id++;
            if (tryEmitTiledTransposeMaterializedFusedStage(root_idx, stage_out_id, "")) {
                return;
            }
            --next_value_id;
        }

        if (root.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            emitCudaKernelStage(root_idx);
            return;
        }

        uint32_t embedding_root_idx = UINT32_MAX;
        if (canBuildEmbeddingRootFusedStage(expr, root_idx, embedding_root_idx)) {
            const uint32_t out_id = next_value_id++;
            node_output_value_id[root_idx] = out_id;
            planned.stages.push_back(buildEmbeddingRootFusedStage(expr, root_idx, out_id, "", node_output_value_id));
            return;
        }

        if (isStageBoundaryOp(root.op)) {
            const std::string boundary_sig = fusedRegionSignature(expr, root_idx);
            auto emitted_boundary_it = stage_boundary_value_id.find(boundary_sig);
            if (emitted_boundary_it != stage_boundary_value_id.end()) {
                node_output_value_id[root_idx] = emitted_boundary_it->second;
                return;
            }

            auto ensureBoundaryParentEmitted = [&](uint32_t parent_idx, const char* label) {
                if (parent_idx >= expr.nodes.size()) {
                    throw std::runtime_error(std::string("Stage-boundary ") + label + " out of range.");
                }
                const ExprNode& parent = expr.nodes[parent_idx];
                if (parent.op != ExprOp::INPUT || inputRequiresMaterialization(parent)) {
                    emitForDependency(parent_idx);
                }
            };

            auto ensureScaleDependencyEmitted = [&](uint32_t parent_idx, const char* label) {
                if (parent_idx == UINT32_MAX) {
                    return;
                }
                if (parent_idx >= expr.nodes.size()) {
                    throw std::runtime_error(std::string("Stage-boundary ") + label + " out of range.");
                }
                const ExprNode& parent = expr.nodes[parent_idx];
                if (parent.op == ExprOp::INPUT || parent.op == ExprOp::RUNTIME_SCALAR || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR ||
                    parent.op == ExprOp::SCALAR_FP) {
                    return;
                }
                emitForDependency(parent_idx);
            };

            uint32_t lhs_dependency_idx = root.lhs;
            uint32_t rhs_dependency_idx = root.rhs;
            if (isMatmulOp(root.op) && root.matmul_backward_epilogue == MatmulBackwardEpilogue::Default) {
                bool ignored = false;
                lhs_dependency_idx = peelExplicitTransposeChain(expr, root.lhs, ignored);
                rhs_dependency_idx = peelExplicitTransposeChain(expr, root.rhs, ignored);
            }

            std::unordered_set<uint32_t> grouped_rope_roots;
            if (root.op == ExprOp::ATTENTION) {
                grouped_rope_roots = tryEmitGroupedRopeMaterialization({root.lhs, root.rhs});
            }

            if (!grouped_rope_roots.contains(lhs_dependency_idx)) {
                ensureBoundaryParentEmitted(lhs_dependency_idx, "lhs");
            }
            if (isReduceMinMaxBackwardOp(root.op) || isScanMinMaxBackwardOp(root.op) || isMatmulOp(root.op) || isEmbeddingLookupOp(root.op) ||
                root.op == ExprOp::SEGMENTED_SCAN || isSegmentedReduceOp(root.op) || isAttentionOp(root.op) || isAttentionBackwardOp(root.op) ||
                isConvolutionOp(root.op)) {
                if (!grouped_rope_roots.contains(rhs_dependency_idx)) {
                    ensureBoundaryParentEmitted(rhs_dependency_idx, "rhs");
                }
            }
            if (isAttentionOp(root.op) || isAttentionBackwardOp(root.op)) {
                ensureBoundaryParentEmitted(root.aux, "aux");
            }
            if (root.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || root.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
                ensureBoundaryParentEmitted(root.aux, "offsets");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_bias) {
                ensureBoundaryParentEmitted(root.alpha_node, "bias");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_padding_mask) {
                ensureBoundaryParentEmitted(root.attention_seq_len_q_node, "q_seq_len");
                ensureBoundaryParentEmitted(root.attention_seq_len_kv_node, "kv_seq_len");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_ragged_offsets) {
                ensureBoundaryParentEmitted(root.attention_ragged_offset_q_node, "q_ragged_offsets");
                ensureBoundaryParentEmitted(root.attention_ragged_offset_kv_node, "kv_ragged_offsets");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_dropout_probability > 0.0f) {
                ensureBoundaryParentEmitted(root.attention_dropout_seed_node, "dropout_seed");
                ensureBoundaryParentEmitted(root.attention_dropout_offset_node, "dropout_offset");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_fp8_forward_scaling) {
                ensureBoundaryParentEmitted(root.attention_descale_q_node, "descale_q");
                ensureBoundaryParentEmitted(root.attention_descale_k_node, "descale_k");
                ensureBoundaryParentEmitted(root.attention_descale_v_node, "descale_v");
                ensureBoundaryParentEmitted(root.attention_descale_s_node, "descale_s");
                ensureBoundaryParentEmitted(root.attention_scale_s_node, "scale_s");
                ensureBoundaryParentEmitted(root.attention_scale_o_node, "scale_o");
                ensureBoundaryParentEmitted(root.attention_amax_s_node, "amax_s");
                ensureBoundaryParentEmitted(root.attention_amax_o_node, "amax_o");
            }
            if (isAttentionBackwardOp(root.op)) {
                ensureBoundaryParentEmitted(root.alpha_node, "dO");
                if (root.attention_use_bias) {
                    ensureBoundaryParentEmitted(root.beta_node, "bias");
                }
                if (root.attention_use_padding_mask) {
                    ensureBoundaryParentEmitted(root.attention_seq_len_q_node, "q_seq_len");
                    ensureBoundaryParentEmitted(root.attention_seq_len_kv_node, "kv_seq_len");
                }
                if (root.attention_use_ragged_offsets) {
                    ensureBoundaryParentEmitted(root.attention_ragged_offset_q_node, "q_ragged_offsets");
                    ensureBoundaryParentEmitted(root.attention_ragged_offset_kv_node, "kv_ragged_offsets");
                }
                if (root.attention_dropout_probability > 0.0f) {
                    ensureBoundaryParentEmitted(root.attention_dropout_seed_node, "dropout_seed");
                    ensureBoundaryParentEmitted(root.attention_dropout_offset_node, "dropout_offset");
                }
            }
            if (root.op == ExprOp::GEMM) {
                ensureBoundaryParentEmitted(root.aux, "aux");
                ensureScaleDependencyEmitted(root.alpha_node, "alpha");
                ensureScaleDependencyEmitted(root.beta_node, "beta");
            }
            if (isMatmulOp(root.op) && root.matmul_epilogue_aux != UINT32_MAX) {
                ensureBoundaryParentEmitted(root.matmul_epilogue_aux, "backward epilogue aux");
            }

            if (isScanOp(root.op)) {
                const uint32_t paired_node_idx = findPairedScanNode(expr, root_idx);
                if (paired_node_idx != UINT32_MAX && node_output_value_id.find(paired_node_idx) == node_output_value_id.end()) {
                    const uint32_t first_value_id = next_value_id++;
                    const uint32_t second_value_id = next_value_id++;
                    node_output_value_id[root_idx] = first_value_id;
                    node_output_value_id[paired_node_idx] = second_value_id;
                    planned.stages.push_back(buildScanStage(expr,
                                                            std::vector<RequestedStageOutput>{
                                                                RequestedStageOutput{
                                                                    .name = "",
                                                                    .old_root_idx = root_idx,
                                                                    .value_id = first_value_id,
                                                                },
                                                                RequestedStageOutput{
                                                                    .name = "",
                                                                    .old_root_idx = paired_node_idx,
                                                                    .value_id = second_value_id,
                                                                },
                                                            },
                                                            node_output_value_id));
                    stage_boundary_value_id.emplace(boundary_sig, first_value_id);
                    stage_boundary_value_id.emplace(fusedRegionSignature(expr, paired_node_idx), second_value_id);
                    return;
                }
            }

            uint32_t stage_out_id = next_value_id++;
            node_output_value_id[root_idx] = stage_out_id;
            if (isReduceMinMaxBackwardOp(root.op)) {
                planned.stages.push_back(buildReduceMinMaxBackwardStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isScanMinMaxBackwardOp(root.op)) {
                planned.stages.push_back(buildScanMinMaxBackwardStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isSegmentedReduceOp(root.op)) {
                planned.stages.push_back(buildSegmentedReductionStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isScanOp(root.op)) {
                planned.stages.push_back(buildScanStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isSoftmaxOp(root.op)) {
                planned.stages.push_back(buildSoftmaxStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isRmsNormOp(root.op)) {
                planned.stages.push_back(buildRmsNormStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isMatmulOp(root.op)) {
                std::vector<PendingMatmulBgradOutput> bgrad_outputs = takePendingMatmulBgradOutputs(root_idx);
                planned.stages.push_back(buildMatmulStage(expr, root_idx, stage_out_id, "", node_output_value_id, bgrad_outputs));
            } else if (isAttentionOp(root.op)) {
                planned.stages.push_back(buildAttentionStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isAttentionBackwardOp(root.op)) {
                planned.stages.push_back(buildAttentionBackwardStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isConvolutionBackwardOp(root.op)) {
                planned.stages.push_back(buildConvolutionBackwardStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isConvolutionForwardOp(root.op)) {
                planned.stages.push_back(buildConvolutionStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isTransposeOp(root.op)) {
                throw std::runtime_error("Internal error: explicit transpose was not lowered to fused tiled-transpose materialization.");
            } else {
                planned.stages.push_back(buildReductionStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            }
            stage_boundary_value_id.emplace(boundary_sig, stage_out_id);
            return;
        }

        std::unordered_set<uint32_t> forced_transpose_boundaries = collectUnsupportedLogicalTransposeBoundaries(expr, root_idx);
        for (uint32_t forced_transpose_idx : forced_transpose_boundaries) {
            emitForDependency(forced_transpose_idx);
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegionStoppingAt(expr, root_idx, forced_transpose_boundaries, region);

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        std::string region_sig = fusedRegionSignature(expr, root_idx);

        if (forced_transpose_boundaries.empty()) {
            auto emitted_it = fused_region_value_id.find(region_sig);
            if (emitted_it != fused_region_value_id.end()) {
                node_output_value_id[root_idx] = emitted_it->second;
                return;
            }

            auto pending_it = pending_terminal_region_to_group.find(region_sig);
            if (pending_it != pending_terminal_region_to_group.end()) {
                materializeTerminalGroup(pending_it->second);

                auto fused_it = fused_region_value_id.find(region_sig);
                if (fused_it == fused_region_value_id.end()) {
                    throw std::runtime_error("Pending terminal region was materialized but no fused region value id was recorded.");
                }

                node_output_value_id[root_idx] = fused_it->second;
                return;
            }
        }

        uint32_t out_id = next_value_id++;
        node_output_value_id[root_idx] = out_id;
        if (forced_transpose_boundaries.empty()) {
            fused_region_value_id.emplace(region_sig, out_id);
        }

        std::vector<RequestedStageOutput> requested_outputs{RequestedStageOutput{
            .name = "",
            .old_root_idx = root_idx,
            .value_id = out_id,
        }};

        planned.stages.push_back(buildFusedStage(expr, region, requested_outputs, node_output_value_id));
    };

    auto addOrMergeTerminalGroup = [&](std::unordered_set<uint32_t> region,
                                       std::unordered_set<uint32_t> dependency_value_ids,
                                       RequestedStageOutput requested_output,
                                       const std::string& region_sig) {
        std::vector<size_t> overlapping_groups;
        for (size_t i = 0; i < terminal_groups.size(); ++i) {
            if (!terminal_groups[i].has_value() || terminal_groups[i]->emitted) {
                continue;
            }
            if (setsOverlap(terminal_groups[i]->dependency_value_ids, dependency_value_ids)) {
                overlapping_groups.push_back(i);
            }
        }

        if (overlapping_groups.empty()) {
            TerminalFusedGroup new_group;
            new_group.region_nodes = std::move(region);
            new_group.dependency_value_ids = std::move(dependency_value_ids);
            new_group.outputs.push_back(requested_output);
            new_group.exact_region_value_id.emplace(region_sig, requested_output.value_id);

            size_t new_idx = terminal_groups.size();
            terminal_groups.push_back(std::move(new_group));
            pending_terminal_region_to_group[region_sig] = new_idx;
            return;
        }

        size_t target = overlapping_groups.front();
        TerminalFusedGroup& target_group = *terminal_groups[target];

        target_group.region_nodes.insert(region.begin(), region.end());
        target_group.dependency_value_ids.insert(dependency_value_ids.begin(), dependency_value_ids.end());
        target_group.outputs.push_back(requested_output);
        target_group.exact_region_value_id[region_sig] = requested_output.value_id;
        pending_terminal_region_to_group[region_sig] = target;

        for (size_t k = 1; k < overlapping_groups.size(); ++k) {
            size_t src_idx = overlapping_groups[k];
            if (!terminal_groups[src_idx].has_value()) {
                continue;
            }

            TerminalFusedGroup& src_group = *terminal_groups[src_idx];

            target_group.region_nodes.insert(src_group.region_nodes.begin(), src_group.region_nodes.end());
            target_group.dependency_value_ids.insert(src_group.dependency_value_ids.begin(), src_group.dependency_value_ids.end());
            target_group.outputs.insert(target_group.outputs.end(), src_group.outputs.begin(), src_group.outputs.end());

            for (const auto& [key, value_id] : src_group.exact_region_value_id) {
                target_group.exact_region_value_id[key] = value_id;
                pending_terminal_region_to_group[key] = target;
            }

            terminal_groups[src_idx].reset();
        }
    };

    for (const NamedOutput& named_output : outputs.outputs) {
        const ExprNode& root = expr.nodes[named_output.node_idx];

        auto existing_value_it = node_output_value_id.find(named_output.node_idx);
        if (existing_value_it != node_output_value_id.end()) {
            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = existing_value_it->second,
            });
            continue;
        }

        uint32_t bgrad_matmul_idx = UINT32_MAX;
        if (isMatmulBiasGradientReduction(expr, named_output.node_idx, bgrad_matmul_idx)) {
            emitForDependency(bgrad_matmul_idx);
            auto bgrad_value_it = node_output_value_id.find(named_output.node_idx);
            if (bgrad_value_it == node_output_value_id.end()) {
                throw std::runtime_error("Matmul bias-gradient reduction was not exposed as a matmul stage output.");
            }
            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = bgrad_value_it->second,
            });
            continue;
        }

        if (isStorageAliasOp(root.op) && reshapeAliasPreservesDType(expr, named_output.node_idx)) {
            const uint32_t alias_value_id = emitStorageAlias(named_output.node_idx, std::nullopt);
            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = alias_value_id,
            });
            continue;
        }

        if (isTransposeOp(root.op)) {
            const uint32_t stage_out_id = next_value_id++;
            if (tryEmitTiledTransposeMaterializedFusedStage(named_output.node_idx, stage_out_id, named_output.name)) {
                planned.final_outputs.push_back(CompiledStageOutput{
                    .name = named_output.name,
                    .local_node_idx = UINT32_MAX,
                    .value_id = stage_out_id,
                });
                continue;
            }
            --next_value_id;
        }

        if (root.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            emitCudaKernelStage(named_output.node_idx);
            auto value_it = node_output_value_id.find(named_output.node_idx);
            if (value_it == node_output_value_id.end()) {
                throw std::runtime_error("CudaKernelExpression final output missing value id after stage emission.");
            }
            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = value_it->second,
            });
            continue;
        }

        uint32_t embedding_root_idx = UINT32_MAX;
        if (canBuildEmbeddingRootFusedStage(expr, named_output.node_idx, embedding_root_idx)) {
            const uint32_t out_id = next_value_id++;
            node_output_value_id[named_output.node_idx] = out_id;
            planned.stages.push_back(
                buildEmbeddingRootFusedStage(expr, named_output.node_idx, out_id, named_output.name, node_output_value_id));
            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = out_id,
            });
            continue;
        }

        if (isStageBoundaryOp(root.op)) {
            auto already_emitted_it = node_output_value_id.find(named_output.node_idx);
            if (already_emitted_it != node_output_value_id.end()) {
                planned.final_outputs.push_back(CompiledStageOutput{
                    .name = named_output.name,
                    .local_node_idx = UINT32_MAX,
                    .value_id = already_emitted_it->second,
                });
                continue;
            }

            const std::string boundary_sig = fusedRegionSignature(expr, named_output.node_idx);
            auto emitted_boundary_it = stage_boundary_value_id.find(boundary_sig);
            if (emitted_boundary_it != stage_boundary_value_id.end()) {
                node_output_value_id[named_output.node_idx] = emitted_boundary_it->second;
                planned.final_outputs.push_back(CompiledStageOutput{
                    .name = named_output.name,
                    .local_node_idx = UINT32_MAX,
                    .value_id = emitted_boundary_it->second,
                });
                continue;
            }

            auto ensureBoundaryParentEmitted = [&](uint32_t parent_idx, const char* label) {
                if (parent_idx >= expr.nodes.size()) {
                    throw std::runtime_error(std::string("Stage-boundary ") + label + " out of range.");
                }
                const ExprNode& parent = expr.nodes[parent_idx];
                if (parent.op != ExprOp::INPUT || inputRequiresMaterialization(parent)) {
                    emitForDependency(parent_idx);
                }
            };

            auto ensureScaleDependencyEmitted = [&](uint32_t parent_idx, const char* label) {
                if (parent_idx == UINT32_MAX) {
                    return;
                }
                if (parent_idx >= expr.nodes.size()) {
                    throw std::runtime_error(std::string("Stage-boundary ") + label + " out of range.");
                }
                const ExprNode& parent = expr.nodes[parent_idx];
                if (parent.op == ExprOp::INPUT || parent.op == ExprOp::RUNTIME_SCALAR || parent.op == ExprOp::TENSOR_RUNTIME_SCALAR ||
                    parent.op == ExprOp::SCALAR_FP) {
                    return;
                }
                emitForDependency(parent_idx);
            };

            uint32_t lhs_dependency_idx = root.lhs;
            uint32_t rhs_dependency_idx = root.rhs;
            if (isMatmulOp(root.op) && root.matmul_backward_epilogue == MatmulBackwardEpilogue::Default) {
                bool ignored = false;
                lhs_dependency_idx = peelExplicitTransposeChain(expr, root.lhs, ignored);
                rhs_dependency_idx = peelExplicitTransposeChain(expr, root.rhs, ignored);
            }

            std::unordered_set<uint32_t> grouped_rope_roots;
            if (root.op == ExprOp::ATTENTION) {
                grouped_rope_roots = tryEmitGroupedRopeMaterialization({root.lhs, root.rhs});
            }

            if (!grouped_rope_roots.contains(lhs_dependency_idx)) {
                ensureBoundaryParentEmitted(lhs_dependency_idx, "lhs");
            }
            if (isReduceMinMaxBackwardOp(root.op) || isScanMinMaxBackwardOp(root.op) || isMatmulOp(root.op) || isEmbeddingLookupOp(root.op) ||
                root.op == ExprOp::SEGMENTED_SCAN || isSegmentedReduceOp(root.op) || isAttentionOp(root.op) || isAttentionBackwardOp(root.op) ||
                isConvolutionOp(root.op)) {
                if (!grouped_rope_roots.contains(rhs_dependency_idx)) {
                    ensureBoundaryParentEmitted(rhs_dependency_idx, "rhs");
                }
            }
            if (isAttentionOp(root.op) || isAttentionBackwardOp(root.op)) {
                ensureBoundaryParentEmitted(root.aux, "aux");
            }
            if (root.op == ExprOp::SEGMENTED_SCAN_MIN_BACKWARD || root.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
                ensureBoundaryParentEmitted(root.aux, "offsets");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_bias) {
                ensureBoundaryParentEmitted(root.alpha_node, "bias");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_padding_mask) {
                ensureBoundaryParentEmitted(root.attention_seq_len_q_node, "q_seq_len");
                ensureBoundaryParentEmitted(root.attention_seq_len_kv_node, "kv_seq_len");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_ragged_offsets) {
                ensureBoundaryParentEmitted(root.attention_ragged_offset_q_node, "q_ragged_offsets");
                ensureBoundaryParentEmitted(root.attention_ragged_offset_kv_node, "kv_ragged_offsets");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_dropout_probability > 0.0f) {
                ensureBoundaryParentEmitted(root.attention_dropout_seed_node, "dropout_seed");
                ensureBoundaryParentEmitted(root.attention_dropout_offset_node, "dropout_offset");
            }
            if (root.op == ExprOp::ATTENTION && root.attention_use_fp8_forward_scaling) {
                ensureBoundaryParentEmitted(root.attention_descale_q_node, "descale_q");
                ensureBoundaryParentEmitted(root.attention_descale_k_node, "descale_k");
                ensureBoundaryParentEmitted(root.attention_descale_v_node, "descale_v");
                ensureBoundaryParentEmitted(root.attention_descale_s_node, "descale_s");
                ensureBoundaryParentEmitted(root.attention_scale_s_node, "scale_s");
                ensureBoundaryParentEmitted(root.attention_scale_o_node, "scale_o");
                ensureBoundaryParentEmitted(root.attention_amax_s_node, "amax_s");
                ensureBoundaryParentEmitted(root.attention_amax_o_node, "amax_o");
            }
            if (isAttentionBackwardOp(root.op)) {
                ensureBoundaryParentEmitted(root.alpha_node, "dO");
                if (root.attention_use_bias) {
                    ensureBoundaryParentEmitted(root.beta_node, "bias");
                }
                if (root.attention_use_padding_mask) {
                    ensureBoundaryParentEmitted(root.attention_seq_len_q_node, "q_seq_len");
                    ensureBoundaryParentEmitted(root.attention_seq_len_kv_node, "kv_seq_len");
                }
                if (root.attention_use_ragged_offsets) {
                    ensureBoundaryParentEmitted(root.attention_ragged_offset_q_node, "q_ragged_offsets");
                    ensureBoundaryParentEmitted(root.attention_ragged_offset_kv_node, "kv_ragged_offsets");
                }
                if (root.attention_dropout_probability > 0.0f) {
                    ensureBoundaryParentEmitted(root.attention_dropout_seed_node, "dropout_seed");
                    ensureBoundaryParentEmitted(root.attention_dropout_offset_node, "dropout_offset");
                }
            }
            if (root.op == ExprOp::GEMM) {
                ensureBoundaryParentEmitted(root.aux, "aux");
                ensureScaleDependencyEmitted(root.alpha_node, "alpha");
                ensureScaleDependencyEmitted(root.beta_node, "beta");
            }
            if (isMatmulOp(root.op) && root.matmul_epilogue_aux != UINT32_MAX) {
                ensureBoundaryParentEmitted(root.matmul_epilogue_aux, "backward epilogue aux");
            }

            if (isScanOp(root.op)) {
                const uint32_t paired_node_idx = findPairedScanNode(expr, named_output.node_idx);
                auto paired_final_name_it = paired_node_idx == UINT32_MAX ? final_output_name_by_node.end()
                                                                          : final_output_name_by_node.find(paired_node_idx);
                if (paired_final_name_it != final_output_name_by_node.end() &&
                    node_output_value_id.find(paired_node_idx) == node_output_value_id.end()) {
                    const uint32_t first_value_id = next_value_id++;
                    const uint32_t second_value_id = next_value_id++;
                    node_output_value_id[named_output.node_idx] = first_value_id;
                    node_output_value_id[paired_node_idx] = second_value_id;
                    std::vector<RequestedStageOutput> requested_scan_outputs;
                    requested_scan_outputs.push_back(RequestedStageOutput{
                        .name = named_output.name,
                        .old_root_idx = named_output.node_idx,
                        .value_id = first_value_id,
                    });
                    requested_scan_outputs.push_back(RequestedStageOutput{
                        .name = paired_final_name_it->second,
                        .old_root_idx = paired_node_idx,
                        .value_id = second_value_id,
                    });
                    planned.stages.push_back(buildScanStage(expr, requested_scan_outputs, node_output_value_id));
                    stage_boundary_value_id.emplace(boundary_sig, first_value_id);
                    stage_boundary_value_id.emplace(fusedRegionSignature(expr, paired_node_idx), second_value_id);
                    planned.final_outputs.push_back(CompiledStageOutput{
                        .name = named_output.name,
                        .local_node_idx = UINT32_MAX,
                        .value_id = first_value_id,
                    });
                    continue;
                }
            }

            uint32_t stage_out_id = next_value_id++;
            node_output_value_id[named_output.node_idx] = stage_out_id;
            if (isReduceMinMaxBackwardOp(root.op)) {
                planned.stages.push_back(
                    buildReduceMinMaxBackwardStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isScanMinMaxBackwardOp(root.op)) {
                planned.stages.push_back(
                    buildScanMinMaxBackwardStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isSegmentedReduceOp(root.op)) {
                planned.stages.push_back(
                    buildSegmentedReductionStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isScanOp(root.op)) {
                planned.stages.push_back(
                    buildScanStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isSoftmaxOp(root.op)) {
                planned.stages.push_back(
                    buildSoftmaxStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isRmsNormOp(root.op)) {
                planned.stages.push_back(
                    buildRmsNormStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isEmbeddingLookupOp(root.op)) {
                planned.stages.push_back(
                    buildEmbeddingLookupStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isMatmulOp(root.op)) {
                std::vector<PendingMatmulBgradOutput> bgrad_outputs = takePendingMatmulBgradOutputs(named_output.node_idx);
                planned.stages.push_back(
                    buildMatmulStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id, bgrad_outputs));
            } else if (isAttentionOp(root.op)) {
                planned.stages.push_back(
                    buildAttentionStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isAttentionBackwardOp(root.op)) {
                planned.stages.push_back(
                    buildAttentionBackwardStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isConvolutionBackwardOp(root.op)) {
                planned.stages.push_back(
                    buildConvolutionBackwardStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isConvolutionForwardOp(root.op)) {
                planned.stages.push_back(
                    buildConvolutionStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isTransposeOp(root.op)) {
                throw std::runtime_error(
                    "Internal error: explicit transpose output was not lowered to fused tiled-transpose materialization.");
            } else {
                planned.stages.push_back(
                    buildReductionStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            }

            stage_boundary_value_id.emplace(boundary_sig, stage_out_id);

            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = stage_out_id,
            });
            continue;
        }

        std::unordered_set<uint32_t> forced_transpose_boundaries =
            collectUnsupportedLogicalTransposeBoundaries(expr, named_output.node_idx);
        for (uint32_t forced_transpose_idx : forced_transpose_boundaries) {
            emitForDependency(forced_transpose_idx);
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegionStoppingAt(expr, named_output.node_idx, forced_transpose_boundaries, region);

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        std::string region_sig = fusedRegionSignature(expr, named_output.node_idx);

        if (forced_transpose_boundaries.empty()) {
            auto emitted_it = fused_region_value_id.find(region_sig);
            if (emitted_it != fused_region_value_id.end()) {
                node_output_value_id[named_output.node_idx] = emitted_it->second;
                planned.final_outputs.push_back(CompiledStageOutput{
                    .name = named_output.name,
                    .local_node_idx = UINT32_MAX,
                    .value_id = emitted_it->second,
                });
                continue;
            }

            auto pending_it = pending_terminal_region_to_group.find(region_sig);
            if (pending_it != pending_terminal_region_to_group.end()) {
                size_t group_idx = pending_it->second;
                if (group_idx >= terminal_groups.size() || !terminal_groups[group_idx].has_value()) {
                    throw std::runtime_error("Pending terminal region points to invalid group.");
                }

                uint32_t existing_value_id = terminal_groups[group_idx]->exact_region_value_id.at(region_sig);
                node_output_value_id[named_output.node_idx] = existing_value_id;

                planned.final_outputs.push_back(CompiledStageOutput{
                    .name = named_output.name,
                    .local_node_idx = UINT32_MAX,
                    .value_id = existing_value_id,
                });
                continue;
            }
        }

        uint32_t out_id = next_value_id++;
        node_output_value_id[named_output.node_idx] = out_id;

        RequestedStageOutput requested_output{
            .name = named_output.name,
            .old_root_idx = named_output.node_idx,
            .value_id = out_id,
        };

        if (forced_transpose_boundaries.empty()) {
            std::unordered_set<uint32_t> dependency_value_ids;
            collectExternalValueIds(expr, region, node_output_value_id, dependency_value_ids);
            addOrMergeTerminalGroup(std::move(region), std::move(dependency_value_ids), requested_output, region_sig);
        } else {
            std::vector<RequestedStageOutput> requested_outputs{requested_output};
            planned.stages.push_back(buildFusedStage(expr, region, requested_outputs, node_output_value_id));
        }

        planned.final_outputs.push_back(CompiledStageOutput{
            .name = named_output.name,
            .local_node_idx = UINT32_MAX,
            .value_id = out_id,
        });
    }

    for (size_t i = 0; i < terminal_groups.size(); ++i) {
        if (!terminal_groups[i].has_value()) {
            continue;
        }
        materializeTerminalGroup(i);
    }

    mergeAttentionBackwardStages(planned.stages);
    return planned;
}

std::vector<PhysicalExecutionStage> EquationCompiler::splitAtReductionBoundaries(const PhysicalOutputs& outputs) {
    return planExecution(outputs).stages;
}

std::shared_ptr<CompiledOutputs> EquationCompiler::compile(const PhysicalOutputs& outputs,
                                                           const EquationSignature& sig,
                                                           bool broadcast_support) {
    if (!outputs.expr) {
        throw std::runtime_error("Cannot compile Outputs with null expression graph.");
    }

    if (outputs.outputs.empty()) {
        throw std::runtime_error("Cannot compile Outputs with no requested outputs.");
    }

    ensureCudaContextCurrent(sig.device_num);

    auto compiled = std::make_shared<CompiledOutputs>();
    compiled->signature = sig;
    compiled->broadcast_support = broadcast_support;

    PlannedExecution planned = planExecution(outputs);
    compiled->stages.reserve(planned.stages.size());

    for (const PhysicalExecutionStage& stage : planned.stages) {
        std::shared_ptr<CompiledEquation> flat;
        std::shared_ptr<CompiledReduction> reduction;
        switch (stage.kind) {
            case PhysicalExecutionStage::Kind::FusedKernel:
                flat = compileFusedStage(stage, sig);
                compiled->stages.emplace_back(stage.expr, flat, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            case PhysicalExecutionStage::Kind::CudaKernel: {
                if (stage.expr.cuda_kernel_expressions.size() != 1 || !stage.expr.cuda_kernel_expressions[0]) {
                    throw std::runtime_error("CudaKernel physical stage requires exactly one kernel spec.");
                }
                std::shared_ptr<const CudaKernelExpression> cuda_expr = stage.expr.cuda_kernel_expressions[0];
                std::shared_ptr<CompiledCudaKernel> compiled_kernel = cuda_expr->compile(sig.device_num);
                std::vector<DataType> output_dtypes;
                output_dtypes.reserve(stage.outputs.size());
                for (const CompiledStageOutput& output : stage.outputs) {
                    if (output.local_node_idx >= stage.expr.nodes.size()) {
                        throw std::runtime_error("CudaKernel physical stage output node index out of range.");
                    }
                    const ExprNode& node = stage.expr.nodes[output.local_node_idx];
                    if (!node.output_dtype.has_value()) {
                        throw std::runtime_error("CudaKernel physical stage output node missing dtype.");
                    }
                    output_dtypes.push_back(node.output_dtype.value());
                }
                compiled->stages.emplace_back(stage.expr,
                                              cuda_expr,
                                              compiled_kernel,
                                              output_dtypes,
                                              stage.input_value_ids,
                                              stage.outputs,
                                              stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Reduction:
                reduction = compileReduction(stage.expr);
                compiled->stages.emplace_back(reduction, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            case PhysicalExecutionStage::Kind::ArgMinMax: {
                std::shared_ptr<CompiledArgMinMax> arg_minmax = compileArgMinMax(stage.expr);
                compiled->stages.emplace_back(arg_minmax, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::SegmentedReduction: {
                std::shared_ptr<CompiledSegmentedReduction> segmented_reduction = compileSegmentedReduction(stage.expr);
                compiled->stages.emplace_back(segmented_reduction, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Scan: {
                std::shared_ptr<CompiledScan> scan = compileScan(stage.expr);
                compiled->stages.emplace_back(stage.expr, scan, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Softmax: {
                std::shared_ptr<CompiledSoftmax> softmax = compileSoftmax(stage.expr);
                compiled->stages.emplace_back(softmax, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::RmsNorm: {
                std::shared_ptr<CompiledRmsNorm> rms_norm = compileRmsNorm(stage.expr);
                compiled->stages.emplace_back(rms_norm, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::EmbeddingLookup: {
                std::shared_ptr<CompiledEmbeddingLookup> embedding_lookup = compileEmbeddingLookup(stage.expr);
                compiled->stages.emplace_back(embedding_lookup, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Matmul: {
                std::shared_ptr<CompiledMatmul> matmul = compileMatmul(stage.expr, stage.outputs);
                compiled->stages.emplace_back(matmul, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::InPlaceRope: {
                std::shared_ptr<CompiledInPlaceRope> in_place_rope = compileInPlaceRope(stage.expr, stage.outputs);
                compiled->stages.emplace_back(in_place_rope, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Attention: {
                std::shared_ptr<CompiledAttention> attention = compileAttention(stage.expr);
                compiled->stages.emplace_back(attention, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::AttentionBackward: {
                std::shared_ptr<CompiledAttentionBackward> attention_backward = compileAttentionBackward(stage.expr);
                compiled->stages.emplace_back(
                    stage.expr, attention_backward, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Convolution: {
                std::shared_ptr<CompiledConvolution> convolution = compileConvolution(stage.expr);
                compiled->stages.emplace_back(convolution, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::ConvolutionBackward: {
                std::shared_ptr<CompiledConvolutionBackward> convolution_backward = compileConvolutionBackward(stage.expr);
                compiled->stages.emplace_back(convolution_backward, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::ReduceMinMaxBackward: {
                std::shared_ptr<CompiledReduceMinMaxBackward> reduce_minmax_backward = compileReduceMinMaxBackward(stage.expr);
                compiled->stages.emplace_back(reduce_minmax_backward, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::ScanMinMaxBackward: {
                std::shared_ptr<CompiledScanMinMaxBackward> scan_minmax_backward = compileScanMinMaxBackward(stage.expr);
                compiled->stages.emplace_back(scan_minmax_backward, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            default:
                throw std::runtime_error("Unknown stage kind in EquationCompiler::compile(PhysicalOutputs).");
        }
    }

    compiled->final_outputs = std::move(planned.final_outputs);
    compiled->value_aliases = std::move(planned.value_aliases);
    return compiled;
}

shared_ptr<CompiledEquation> EquationCompiler::compileSpecializedBroadcastStage(const CompiledExecutionStage& stage,
                                                                                const EquationSignature& sig,
                                                                                const std::vector<SpecializedBroadcastGroup>& groups) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("compileSpecializedBroadcastStage called on non-fused stage.");
    }
    if (groups.empty()) {
        throw std::runtime_error("compileSpecializedBroadcastStage requires at least one broadcast group.");
    }
    if (expressionUsesDeviceRaggedRuntimeExtent(stage.expr)) {
        throw std::runtime_error(
            "ragged runtime extent is not supported by specialized broadcast kernels; ragged valuewise operands must have identical capacity shapes.");
    }

    ensureCudaContextCurrent(sig.device_num);

    const std::string kernel_name = "fused_kernel";
    const std::string cuda_src = CudaSourceEmitter::emitSpecializedBroadcast(stage, groups, kernel_name);

    const std::string cache_key = makeSpecializedBroadcastCacheKey(cuda_src, sig);
    optional<shared_ptr<CompiledEquation>> hit = specializedBroadcastCache.get(cache_key);
    if (hit.has_value()) {
        return hit.value();
    }

    std::vector<std::string> input_names;
    std::vector<NamedInput::Kind> input_kinds;
    input_names.reserve(stage.expr.inputs.size());
    input_kinds.reserve(stage.expr.inputs.size());
    for (const NamedInput& input : stage.expr.inputs) {
        input_names.push_back(input.name);
        input_kinds.push_back(input.kind);
    }
    const std::vector<DataType> input_dtypes = collectCompiledInputDTypes(stage.expr);
    std::vector<DataType> output_dtypes;
    output_dtypes.reserve(stage.outputs.size());
    for (const CompiledStageOutput& output : stage.outputs) {
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw std::runtime_error("Stage output local_node_idx out of range.");
        }
        const ExprNode& node = stage.expr.nodes[output.local_node_idx];
        if (!node.output_dtype.has_value()) {
            throw std::runtime_error("Specialized broadcast output node missing resolved output_dtype.");
        }
        output_dtypes.push_back(node.output_dtype.value());
    }

    std::vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    std::vector<char> cubin = linkToCubin(ltoir, sig);

    shared_ptr<CompiledEquation> compiled = loadCubin(EquationCacheKey(canonicalize(stage.expr), sig),
                                                      cubin,
                                                      kernel_name,
                                                      input_names,
                                                      input_kinds,
                                                      input_dtypes,
                                                      output_dtypes,
                                                      sig.device_num);
    compiled->uses_uint32_numel_arg = false;

    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        compiled->launch_kind = CompiledEquation::LaunchKind::FusedTiledTranspose;
        compiled->elements_per_thread = 1u;
        compiled->tiled_transpose_pack_scalars = CudaSourceEmitter::tiledTransposePackScalars(stage);
        compiled->uses_uint32_tiled_transpose_index_math = CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath(groups);
    } else if (CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, groups)) {
        compiled->launch_kind = CompiledEquation::LaunchKind::FusedTiledTranspose;
        compiled->elements_per_thread = 1u;
        compiled->tiled_transpose_pack_scalars = CudaSourceEmitter::tiledLogicalTransposeConsumerPackScalars(stage, groups);
        compiled->uses_uint32_tiled_transpose_index_math = CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath(groups);
        compiled->uses_tiled_logical_transpose_consumer = true;
        compiled->tiled_logical_transpose_slot_bytes = CudaSourceEmitter::tiledLogicalTransposeConsumerSlotBytes(stage, groups);
        compiled->tiled_logical_transpose_dense_packed_input_load_count =
            CudaSourceEmitter::tiledLogicalTransposeConsumerDensePackedInputLoadCount(stage, groups);
        compiled->tiled_logical_transpose_vectorized_output_count =
            CudaSourceEmitter::tiledLogicalTransposeConsumerVectorizedOutputCount(stage, groups);
    } else {
        const std::optional<DataType> vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
        compiled->elements_per_thread = vectorized_dtype.has_value() ? 2u : 1u;

        if (groups.size() > 1) {
            compiled->launch_kind = CompiledEquation::LaunchKind::BroadcastGrouped;
            compiled->num_broadcast_groups = static_cast<uint32_t>(groups.size());
        }
    }

    specializedBroadcastCache.put(cache_key, compiled);
    return compiled;
}

}  // namespace ThorImplementation
