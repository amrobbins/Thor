#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"

#include "CudaSourceEmitter.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

namespace ThorImplementation {

using DataType = TensorDescriptor::DataType;

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

string EquationCompiler::getCudaIncludeDir() {
    if (const char* p = std::getenv("THOR_CUDA_INCLUDE_DIR")) {
        if (*p)
            return std::string(p);
    }

    if (const char* p = std::getenv("CUDA_HOME")) {
        return std::string(p) + "/include";
    }

    if (const char* p = std::getenv("CUDA_PATH")) {
        return std::string(p) + "/include";
    }

    return THOR_CUDA_INCLUDE_DIR;  // compile-time fallback from CMake
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
    int32_t conv_stride_h = 1;
    int32_t conv_stride_w = 1;
    int32_t conv_pad_h = 0;
    int32_t conv_pad_w = 0;
    int32_t input_tensor_dtype = -1;
    int32_t output_dtype = -1;
    int32_t compute_dtype = -1;
    int32_t backward_output_dtype = -1;
    int32_t backward_compute_dtype = -1;
    std::vector<uint64_t> reduction_axes;
    std::vector<uint64_t> squeeze_axes;
    std::vector<uint64_t> unsqueeze_axes;
    std::vector<uint64_t> fill_dims;

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
        hashCombine(h, std::hash<int32_t>{}(k.conv_stride_h));
        hashCombine(h, std::hash<int32_t>{}(k.conv_stride_w));
        hashCombine(h, std::hash<int32_t>{}(k.conv_pad_h));
        hashCombine(h, std::hash<int32_t>{}(k.conv_pad_w));
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
        return h;
    }
};

static bool isCommutativeStageOp(ExprOp op) { return op == ExprOp::ADD || op == ExprOp::MUL || op == ExprOp::MIN || op == ExprOp::MAX; }

static uint64_t scalarBits(double x) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(x));
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int32_t optionalDTypeTag(const Optional<DataType>& dtype) { return dtype.isPresent() ? static_cast<int32_t>(dtype.get()) : -1; }

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
    key.conv_stride_h = n.conv_stride_h;
    key.conv_stride_w = n.conv_stride_w;
    key.conv_pad_h = n.conv_pad_h;
    key.conv_pad_w = n.conv_pad_w;
    key.alpha_bits = scalarBits(n.alpha_fp);
    key.beta_bits = scalarBits(n.beta_fp);
    key.alpha_node = n.alpha_node;
    key.beta_node = n.beta_node;

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
static bool isConvolutionForwardOp(ExprOp op) { return op == ExprOp::CONV2D; }
static bool isConvolutionBackwardOp(ExprOp op) { return op == ExprOp::CONV2D_BACKWARD_DATA || op == ExprOp::CONV2D_BACKWARD_FILTER; }
static bool isConvolutionOp(ExprOp op) { return isConvolutionForwardOp(op) || isConvolutionBackwardOp(op); }
static bool isReduceMinMaxBackwardOp(ExprOp op) { return op == ExprOp::REDUCE_MIN_BACKWARD || op == ExprOp::REDUCE_MAX_BACKWARD; }
static bool isTransposeOp(ExprOp op) { return op == ExprOp::TRANSPOSE; }

static bool isStageBoundaryOp(ExprOp op) {
    return isCudnnReduceOp(op) || isMatmulOp(op) || isConvolutionOp(op) || isReduceMinMaxBackwardOp(op) || isTransposeOp(op);
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

        addExternalValue(node.lhs);

        if (Expression::isBinaryOp(node.op)) {
            addExternalValue(node.rhs);
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

static std::vector<uint32_t> makeRegionKey(const std::unordered_set<uint32_t>& region) {
    std::vector<uint32_t> region_key(region.begin(), region.end());
    std::sort(region_key.begin(), region_key.end());
    return region_key;
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
        case ExprOp::NEG:
            return "NEG";
        case ExprOp::ABS:
            return "ABS";
        case ExprOp::EXP:
            return "EXP";
        case ExprOp::EXP2:
            return "EXP2";
        case ExprOp::EXP10:
            return "EXP10";
        case ExprOp::LN:
            return "LOG";
        case ExprOp::LOG2:
            return "LOG2";
        case ExprOp::LOG10:
            return "LOG10";
        case ExprOp::SQRT:
            return "SQRT";
        case ExprOp::FILL:
            return "FILL";
        case ExprOp::UNSQUEEZE:
            return "UNSQ";
        case ExprOp::SQUEEZE:
            return "SQZ";
        case ExprOp::TRANSPOSE:
            return "TRANSPOSE";
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
        case ExprOp::REDUCE_AVG:
            return "RAVG";
        case ExprOp::REDUCE_NORM1:
            return "RNORM1";
        case ExprOp::REDUCE_NORM2:
            return "RNORM2";
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
        default:
            throw std::runtime_error("Unsupported op in fusedRegionSignature, value: " + to_string((int)op));
    }
}

static std::string optionalDTypeSignature(const Optional<DataType>& dtype) {
    if (!dtype.isPresent()) {
        return "none";
    }
    return TensorDescriptor::getElementTypeName(dtype.get());
}

static void appendNodeDTypeSignature(std::string& s, const ExprNode& node) {
    s += ";input=" + optionalDTypeSignature(node.input_tensor_dtype);
    s += ";out=" + optionalDTypeSignature(node.output_dtype);
    s += ";compute=" + optionalDTypeSignature(node.compute_dtype);
    s += ";bwd_out=" + optionalDTypeSignature(node.backward_output_dtype);
    s += ";bwd_compute=" + optionalDTypeSignature(node.backward_compute_dtype);
}

static std::string fusedRegionSignatureRec(const PhysicalExpression& expr, uint32_t node_idx);

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
        } else if (isMatmulOp(node.op)) {
            const std::string rhs = fusedRegionSignatureRec(expr, node.rhs);

            if (node.op == ExprOp::MATMUL) {
                s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",rhs=" + rhs + ",ta=" + std::to_string(node.transpose_lhs ? 1 : 0) +
                    ",tb=" + std::to_string(node.transpose_rhs ? 1 : 0) + ")";
            } else {
                const std::string aux = fusedRegionSignatureRec(expr, node.aux);
                s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",rhs=" + rhs + ",aux=" + aux +
                    ",ta=" + std::to_string(node.transpose_lhs ? 1 : 0) + ",tb=" + std::to_string(node.transpose_rhs ? 1 : 0) +
                    ",tc=" + std::to_string(node.transpose_aux ? 1 : 0) +
                    ",alpha=" + gemmScaleSignature(expr, node.alpha_node, node.alpha_fp) +
                    ",beta=" + gemmScaleSignature(expr, node.beta_node, node.beta_fp) + ")";
            }
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

    if (!Expression::isBinaryOp(node.op)) {
        std::string s;
        if (node.op == ExprOp::UNSQUEEZE) {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ",axes=" + uintVecSignature(node.unsqueeze_axes) + ")";
        } else if (node.op == ExprOp::SQUEEZE) {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ",axes=" + uintVecSignature(node.squeeze_axes) + ")";
        } else {
            s = std::string(fusedOpTag(node.op)) + "(" + lhs + ")";
        }
        appendNodeDTypeSignature(s, node);
        return s;
    }

    std::string rhs = fusedRegionSignatureRec(expr, node.rhs);

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
                                                         const std::vector<TensorDescriptor::DataType>& input_dtypes,
                                                         const std::vector<TensorDescriptor::DataType>& output_dtypes,
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

static std::vector<NamedInput::Kind> collectCompiledInputKinds(const PhysicalExpression& expr) {
    std::vector<NamedInput::Kind> input_kinds(expr.numInputs(), NamedInput::Kind::Tensor);
    std::vector<uint8_t> seen(expr.numInputs(), 0);

    for (const ExprNode& node : expr.nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }
        if (node.input_slot >= input_kinds.size()) {
            throw runtime_error("Input slot out of range while collecting compiled input kinds.");
        }

        const NamedInput::Kind kind =
            node.op == ExprOp::INPUT
                ? NamedInput::Kind::Tensor
                : (node.op == ExprOp::RUNTIME_SCALAR ? NamedInput::Kind::RuntimeScalarFp32 : NamedInput::Kind::TensorRuntimeScalar);
        if (seen[node.input_slot]) {
            if (input_kinds[node.input_slot] != kind) {
                throw runtime_error("Inconsistent fused stage input kind for local input slot.");
            }
        } else {
            input_kinds[node.input_slot] = kind;
            seen[node.input_slot] = 1;
        }
    }

    for (uint32_t slot = 0; slot < input_kinds.size(); ++slot) {
        if (!seen[slot]) {
            throw runtime_error("Unused or unresolved fused stage input slot.");
        }
    }

    return input_kinds;
}

static std::vector<DataType> collectCompiledInputDTypes(const PhysicalExpression& expr) {
    std::vector<DataType> input_dtypes(expr.numInputs(), DataType::FP32);
    std::vector<uint8_t> seen(expr.numInputs(), 0);

    for (const ExprNode& node : expr.nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }
        if (!node.input_tensor_dtype.isPresent()) {
            throw runtime_error("Fused stage input node is missing resolved input_tensor_dtype.");
        }
        if (node.input_slot >= input_dtypes.size()) {
            throw runtime_error("Input slot out of range while collecting compiled input dtypes.");
        }

        const DataType dtype = node.input_tensor_dtype.get();
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
        if (!node.output_dtype.isPresent()) {
            throw runtime_error("Fused stage output node is missing resolved output_dtype.");
        }
        output_dtypes.push_back(node.output_dtype.get());
    }

    return output_dtypes;
}

static bool stageHasTransposedMaterializedOutput(const std::vector<CompiledStageOutput>& outputs) {
    return std::any_of(outputs.begin(), outputs.end(), [](const CompiledStageOutput& output) {
        return output.materialized_layout == MaterializedTensorLayout::Transposed;
    });
}

static DataType resolveHomogeneousFusedStageDType(const PhysicalExpression& expr) {
    Optional<DataType> stage_output_dtype = Optional<DataType>::empty();

    for (const ExprNode& node : expr.nodes) {
        if (node.op == ExprOp::SCALAR_FP) {
            continue;
        }

        if (!node.output_dtype.isPresent()) {
            throw runtime_error("Fused stage node is missing resolved output_dtype.");
        }
        if (!node.compute_dtype.isPresent()) {
            throw runtime_error("Fused stage node is missing resolved compute_dtype.");
        }

        const DataType node_output_dtype = node.output_dtype.get();
        const DataType expected_compute_dtype = defaultComputeDType(node_output_dtype);

        if (!stage_output_dtype.isPresent()) {
            stage_output_dtype = node_output_dtype;
        } else if (stage_output_dtype.get() != node_output_dtype) {
            throw runtime_error("Current fused kernel codegen still requires a single resolved output_dtype per fused stage.");
        }
        if (node.compute_dtype.get() != expected_compute_dtype) {
            throw runtime_error(
                "Current fused kernel codegen still requires compute_dtype to follow the default policy for the "
                "resolved stage dtype. Per-node compute dtypes are the next patch.");
        }
    }

    if (!stage_output_dtype.isPresent()) {
        return DataType::FP32;
    }

    return stage_output_dtype.get();
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

    const std::string cuda_include_dir = getCudaIncludeDir();
    const std::string cuda_include_path = std::string("--include-path=") + cuda_include_dir;

    vector<const char*> opts = {arch.c_str(), "-dlto", "--std=c++17", "-fmad=true", cuda_include_path.c_str()};
    if (sig.use_fast_math)
        opts.push_back("--use_fast_math");

    NVRTC_COMPILE_CHECK(prog, (int)opts.size(), opts.data());

    size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size));
    vector<char> ltoir(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog, ltoir.data()));

    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return ltoir;
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

    string kernel_name = "fused_kernel";
    const std::string cuda_src = CudaSourceEmitter::emitFlat(stage, kernel_name, use_uint32_index_math);

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

    vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    vector<char> cubin = linkToCubin(ltoir, sig);
    auto compiled = loadCubin(key, cubin, kernel_name, input_names, input_kinds, input_dtypes, output_dtypes, sig.device_num);
    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        compiled->launch_kind = CompiledEquation::LaunchKind::FusedTiledTranspose;
        compiled->elements_per_thread = 1;
        compiled->tiled_transpose_pack_scalars = CudaSourceEmitter::tiledTransposePackScalars(stage);
        compiled->uses_uint32_numel_arg = false;
    } else {
        compiled->elements_per_thread = CudaSourceEmitter::flatElementsPerThread(stage);
        compiled->uses_uint32_numel_arg = use_uint32_index_math;
    }

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
    if (!isCudnnReduceOp(node.op) || isArgMinMaxOp(node.op)) {
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

    if (!input_node.input_tensor_dtype.isPresent()) {
        throw std::runtime_error("Reduction input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("Reduction node missing resolved output_dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.get());

    return make_shared<CompiledReduction>(
        node.op, node.reduction_axes, node.squeeze_axes, supported_input_dtype, node.output_dtype.get(), node.compute_dtype);
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

    if (!input_node.input_tensor_dtype.isPresent()) {
        throw std::runtime_error("ArgMinMax input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("ArgMinMax node missing resolved output_dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, input_node.input_tensor_dtype.get());

    return make_shared<CompiledArgMinMax>(
        node.op, node.reduction_axes, node.squeeze_axes, supported_input_dtype, node.output_dtype.get(), node.compute_dtype);
}

shared_ptr<CompiledMatmul> EquationCompiler::compileMatmul(const PhysicalExpression& expr) {
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
        if (!input_node.input_tensor_dtype.isPresent()) {
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

    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("Matmul/gemm node missing resolved output_dtype.");
    }

    std::vector<DataType> input_dtypes;
    input_dtypes.push_back(lhs_input.input_tensor_dtype.get());
    input_dtypes.push_back(rhs_input.input_tensor_dtype.get());
    if (aux_input != nullptr) {
        input_dtypes.push_back(aux_input->input_tensor_dtype.get());
    }

    const DataType logical_input_dtype = promoteTensorValueDTypes(input_dtypes);
    const DataType supported_input_dtype = toSupportedInputDType(node.op, logical_input_dtype);
    const DataType supported_lhs_dtype = toSupportedInputDType(node.op, lhs_input.input_tensor_dtype.get());
    const DataType supported_rhs_dtype = toSupportedInputDType(node.op, rhs_input.input_tensor_dtype.get());
    const DataType raw_supported_aux_dtype =
        aux_input != nullptr ? toSupportedInputDType(node.op, aux_input->input_tensor_dtype.get()) : node.output_dtype.get();

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
    DataType compiled_aux_dtype = node.output_dtype.get();

    if (supports_fp8_matmul_plan(supported_lhs_dtype, supported_rhs_dtype, raw_supported_aux_dtype, node.output_dtype.get())) {
        compiled_aux_dtype = raw_supported_aux_dtype;
    } else if (!supports_regular_matmul_plan(supported_lhs_dtype, supported_rhs_dtype, node.output_dtype.get())) {
        // Preserve the old safe behavior for combinations cublasLt does not directly expose as mixed A/B types:
        // cast both matrix inputs and the optional addend to the resolved output dtype so A/B/C/D form a regular plan.
        compiled_lhs_dtype = node.output_dtype.get();
        compiled_rhs_dtype = node.output_dtype.get();
        compiled_aux_dtype = node.output_dtype.get();
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
                                       node.output_dtype.get(),
                                       node.compute_dtype);
}

shared_ptr<CompiledConvolution> EquationCompiler::compileConvolution(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("Convolution stage output_node is out of range.");
    }

    const ExprNode& node = expr.nodes[expr.output_node];
    if (node.op != ExprOp::CONV2D) {
        throw std::runtime_error("Convolution stage output node is not CONV2D.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX) {
        throw std::runtime_error("CONV2D node is missing required inputs.");
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
        if (!input_node.input_tensor_dtype.isPresent()) {
            throw std::runtime_error(std::string("Convolution stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    const ExprNode& input_node = validate_local_input(node.lhs, "lhs");
    const ExprNode& filter_node = validate_local_input(node.rhs, "rhs");

    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("CONV2D node missing resolved output_dtype.");
    }

    const DataType logical_input_dtype =
        promoteTensorValueDTypes(std::vector<DataType>{input_node.input_tensor_dtype.get(), filter_node.input_tensor_dtype.get()});
    const DataType supported_input_dtype = toSupportedInputDType(node.op, logical_input_dtype);

    if (supported_input_dtype != DataType::FP16 || node.output_dtype.get() != DataType::FP16) {
        throw std::runtime_error("CONV2D staged path currently supports FP16 input/filter/output tensors only.");
    }

    return make_shared<CompiledConvolution>(node.conv_stride_h,
                                            node.conv_stride_w,
                                            node.conv_pad_h,
                                            node.conv_pad_w,
                                            supported_input_dtype,
                                            node.output_dtype.get(),
                                            node.compute_dtype);
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
        if (!input_node.input_tensor_dtype.isPresent()) {
            throw std::runtime_error(std::string("Convolution backward stage ") + label + " input missing resolved input_tensor_dtype.");
        }
        return input_node;
    };

    const ExprNode& input_node = validate_local_input(node.lhs, "lhs");
    const ExprNode& grad_node = validate_local_input(node.rhs, "rhs");

    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("Convolution backward node missing resolved output_dtype.");
    }

    const DataType logical_input_dtype =
        promoteTensorValueDTypes(std::vector<DataType>{input_node.input_tensor_dtype.get(), grad_node.input_tensor_dtype.get()});
    const DataType supported_input_dtype = toSupportedInputDType(node.op, logical_input_dtype);

    if (supported_input_dtype != DataType::FP16 || node.output_dtype.get() != DataType::FP16) {
        throw std::runtime_error("Convolution backward staged path currently supports FP16 input/grad/output tensors only.");
    }

    return make_shared<CompiledConvolutionBackward>(node.op,
                                                    node.conv_stride_h,
                                                    node.conv_stride_w,
                                                    node.conv_pad_h,
                                                    node.conv_pad_w,
                                                    supported_input_dtype,
                                                    grad_node.input_tensor_dtype.get(),
                                                    node.output_dtype.get(),
                                                    node.compute_dtype,
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

    if (!input_node.input_tensor_dtype.isPresent()) {
        throw std::runtime_error("ReduceMinMaxBackward input node missing resolved input_tensor_dtype.");
    }
    if (!grad_node.input_tensor_dtype.isPresent()) {
        throw std::runtime_error("ReduceMinMaxBackward grad input node missing resolved input_tensor_dtype.");
    }
    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("ReduceMinMaxBackward node missing resolved output_dtype.");
    }

    const ExprOp reduce_op = node.op == ExprOp::REDUCE_MIN_BACKWARD ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX;
    const DataType supported_input_dtype = toSupportedInputDType(reduce_op, input_node.input_tensor_dtype.get());

    return make_shared<CompiledReduceMinMaxBackward>(node.op,
                                                     node.reduction_axes,
                                                     node.squeeze_axes,
                                                     supported_input_dtype,
                                                     grad_node.input_tensor_dtype.get(),
                                                     node.output_dtype.get(),
                                                     node.compute_dtype);
}

static bool inputRequiresMaterialization(const ExprNode& node) {
    if (node.op != ExprOp::INPUT) {
        return false;
    }
    if (!node.input_tensor_dtype.isPresent() || !node.output_dtype.isPresent()) {
        return false;
    }
    return node.input_tensor_dtype.get() != node.output_dtype.get();
}

static void collectFusableRegion(const PhysicalExpression& expr, uint32_t root_idx, std::unordered_set<uint32_t>& region_nodes) {
    std::vector<uint32_t> stack{root_idx};

    while (!stack.empty()) {
        uint32_t node_idx = stack.back();
        stack.pop_back();

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
        if (!isStageBoundaryOp(lhs.op)) {
            stack.push_back(lhs_idx);
        }

        if (Expression::isBinaryOp(node.op)) {
            uint32_t rhs_idx = node.rhs;
            if (rhs_idx >= expr.nodes.size()) {
                throw std::runtime_error("Invalid rhs node index in expression.");
            }

            const ExprNode& rhs = expr.nodes[rhs_idx];
            if (!isStageBoundaryOp(rhs.op)) {
                stack.push_back(rhs_idx);
            }
        }
    }
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

        if (Expression::isBinaryOp(node.op)) {
            uint32_t rhs_idx = node.rhs;
            if (rhs_idx >= expr.nodes.size()) {
                throw std::runtime_error("Invalid rhs node index in expression.");
            }
            if (!region_nodes.count(rhs_idx) && isStageBoundaryOp(expr.nodes[rhs_idx].op)) {
                boundary_nodes.insert(rhs_idx);
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
            if (!Expression::isLeafOp(new_node.op)) {
                uint32_t old_parent = new_node.lhs;
                auto it = old_to_new_node_idx.find(old_parent);
                if (it != old_to_new_node_idx.end()) {
                    new_node.lhs = it->second;
                } else {
                    auto out_it = node_output_value_id.find(old_parent);
                    if (out_it == node_output_value_id.end()) {
                        throw std::runtime_error("Missing value id for fused stage lhs boundary input.");
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
                    new_node.lhs = new_input_idx;
                }
            }

            if (Expression::isBinaryOp(new_node.op)) {
                uint32_t old_parent = new_node.rhs;
                auto it = old_to_new_node_idx.find(old_parent);
                if (it != old_to_new_node_idx.end()) {
                    new_node.rhs = it->second;
                } else {
                    auto out_it = node_output_value_id.find(old_parent);
                    if (out_it == node_output_value_id.end()) {
                        throw std::runtime_error("Missing value id for fused stage rhs boundary input.");
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
                    new_node.rhs = new_input_idx;
                }
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

static PhysicalExecutionStage buildReductionStage(const PhysicalExpression& expr,
                                                  uint32_t node_idx,
                                                  uint32_t output_value_id,
                                                  const std::string& output_name,
                                                  const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isCudnnReduceOp(node.op)) {
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
    Optional<DataType> actual_input_dtype = Optional<DataType>::empty();

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

    if (!parent.output_dtype.isPresent()) {
        throw std::runtime_error("Reduction parent node is missing resolved output_dtype.");
    }
    if (!actual_input_dtype.isPresent()) {
        throw std::runtime_error("Reduction parent node is missing resolved actual input dtype.");
    }

    const DataType supported_input_dtype = toSupportedInputDType(node.op, actual_input_dtype.get());

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;

    // This local INPUT node represents the already-materialized value produced by
    // the parent expression feeding the reduction. Reductions may need a narrower
    // supported library input dtype than the logical parent value dtype, so record
    // the normalized input dtype here and let stamp-time adapt the concrete tensor
    // if needed.
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
    Optional<DataType> actual_input_dtype = Optional<DataType>::empty();
    Optional<DataType> output_dtype = Optional<DataType>::empty();
    Optional<DataType> compute_dtype = Optional<DataType>::empty();
    Optional<DataType> backward_output_dtype = Optional<DataType>::empty();
    Optional<DataType> backward_compute_dtype = Optional<DataType>::empty();

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

    if (!actual_input_dtype.isPresent() || !output_dtype.isPresent()) {
        throw std::runtime_error("Transposed input materialization parent is missing resolved dtype metadata.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"__arg0", 0, NamedInput::Kind::Tensor});

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;
    input_node.input_tensor_dtype = actual_input_dtype.get();
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

static PhysicalExecutionStage buildMatmulStage(const PhysicalExpression& expr,
                                               uint32_t node_idx,
                                               uint32_t output_value_id,
                                               const std::string& output_name,
                                               const std::unordered_map<uint32_t, uint32_t>& node_output_value_id) {
    const ExprNode& node = expr.nodes[node_idx];
    if (!isMatmulOp(node.op)) {
        throw std::runtime_error("buildMatmulStage called on non-matmul node.");
    }
    if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || (node.op == ExprOp::GEMM && node.aux == UINT32_MAX)) {
        throw std::runtime_error("Matmul/gemm node missing required input(s).");
    }

    PhysicalExpression stage_expr;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(node.op == ExprOp::MATMUL ? 4u : 5u);

    auto inputNameForSlot = [](uint32_t slot) { return std::string("__arg") + std::to_string(slot); };

    auto bind_parent_to_local_tensor_input = [&](uint32_t parent_idx, uint32_t local_slot) {
        if (parent_idx >= expr.nodes.size()) {
            throw std::runtime_error("Matmul/gemm input node index out of range.");
        }

        const ExprNode& parent = expr.nodes[parent_idx];
        Optional<DataType> actual_input_dtype = Optional<DataType>::empty();

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

        if (!parent.output_dtype.isPresent()) {
            throw std::runtime_error("Matmul/gemm parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.isPresent()) {
            throw std::runtime_error("Matmul/gemm parent node missing resolved actual input dtype.");
        }

        stage_expr.inputs.push_back(NamedInput{inputNameForSlot(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.get();
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
            input_node.output_dtype = parent.output_dtype.isPresent() ? parent.output_dtype.get() : DataType::FP32;
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
    const uint32_t effective_lhs_parent = peelExplicitTransposeChain(expr, node.lhs, absorbed_lhs_transpose);
    const uint32_t effective_rhs_parent = peelExplicitTransposeChain(expr, node.rhs, absorbed_rhs_transpose);

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

    ExprNode route = node;
    route.lhs = lhs_local;
    route.rhs = rhs_local;
    route.aux = aux_local;
    route.alpha_node = alpha_local;
    route.beta_node = beta_local;
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

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::Matmul,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
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
        Optional<DataType> actual_input_dtype = Optional<DataType>::empty();

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

        if (!parent.output_dtype.isPresent()) {
            throw std::runtime_error("Convolution parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.isPresent()) {
            throw std::runtime_error("Convolution parent node missing resolved actual input dtype.");
        }

        stage_expr.inputs.push_back(NamedInput{inputNameForSlot(local_slot), local_slot, NamedInput::Kind::Tensor});

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.get();
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
        Optional<DataType> actual_input_dtype = Optional<DataType>::empty();

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

        if (!parent.output_dtype.isPresent()) {
            throw std::runtime_error("ConvolutionBackward parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.isPresent()) {
            throw std::runtime_error("ConvolutionBackward parent node missing resolved actual input dtype.");
        }

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.get();
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
        Optional<DataType> actual_input_dtype = Optional<DataType>::empty();

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

        if (!parent.output_dtype.isPresent()) {
            throw std::runtime_error("ReduceMinMaxBackward parent node missing resolved output_dtype.");
        }
        if (!actual_input_dtype.isPresent()) {
            throw std::runtime_error("ReduceMinMaxBackward parent node missing resolved actual input dtype.");
        }

        ExprNode input_node;
        input_node.op = ExprOp::INPUT;
        input_node.input_slot = local_slot;
        input_node.input_tensor_dtype = actual_input_dtype.get();
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

struct PlannedExecution {
    std::vector<PhysicalExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

static bool regionContainsShapeOnlyOp(const PhysicalExpression& expr, const std::unordered_set<uint32_t>& region_nodes) {
    for (uint32_t node_idx : region_nodes) {
        if (node_idx >= expr.nodes.size()) {
            throw std::runtime_error("regionContainsShapeOnlyOp node index out of range.");
        }
        const ExprNode& node = expr.nodes[node_idx];
        if (node.op == ExprOp::UNSQUEEZE || node.op == ExprOp::SQUEEZE) {
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
    // Broadcast / shape-changing fused regions are intentionally not accepted
    // here yet.  They are rejected at stamp time by fusedStageRequiresBroadcastLaunch
    // when a transposed materialized output is present, because their descriptor
    // indexing needs a separate layout-aware implementation.
    return true;
}

static void forceReductionProducerOutputDTypeIfNeeded(PhysicalExpression& expr, uint32_t producer_idx) {
    if (producer_idx >= expr.nodes.size()) {
        throw std::runtime_error("forceReductionProducerOutputDTypeIfNeeded producer_idx out of range.");
    }

    ExprNode& producer = expr.nodes[producer_idx];
    if (!producer.output_dtype.isPresent()) {
        throw std::runtime_error("Reduction producer missing resolved output_dtype.");
    }

    const DataType output_dtype = producer.output_dtype.get();
    if (output_dtype == DataType::FP16 || output_dtype == DataType::FP32) {
        return;
    }

    producer.output_dtype = DataType::FP16;
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

    std::unordered_map<uint32_t, uint32_t> node_output_value_id;
    std::map<std::string, uint32_t> fused_region_value_id;

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

    std::function<void(size_t)> materializeTerminalGroup;
    std::function<void(uint32_t)> emitForDependency;
    std::function<bool(uint32_t, uint32_t, const std::string&)> tryEmitTiledTransposeMaterializedFusedStage;

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

        std::unordered_set<uint32_t> region;
        collectFusableRegion(expr, materialized_parent_idx, region);
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

    emitForDependency = [&](uint32_t root_idx) {
        if (node_output_value_id.find(root_idx) != node_output_value_id.end()) {
            return;
        }

        const ExprNode& root = expr.nodes[root_idx];
        if (isTransposeOp(root.op)) {
            const uint32_t stage_out_id = next_value_id++;
            if (tryEmitTiledTransposeMaterializedFusedStage(root_idx, stage_out_id, "")) {
                return;
            }
            --next_value_id;
        }

        if (isStageBoundaryOp(root.op)) {
            auto ensureBoundaryParentEmitted = [&](uint32_t parent_idx, const char* label, bool force_supported_reduction_input_dtype) {
                if (parent_idx >= expr.nodes.size()) {
                    throw std::runtime_error(std::string("Stage-boundary ") + label + " out of range.");
                }
                const ExprNode& parent = expr.nodes[parent_idx];
                if (parent.op != ExprOp::INPUT || inputRequiresMaterialization(parent)) {
                    if (force_supported_reduction_input_dtype) {
                        forceReductionProducerOutputDTypeIfNeeded(expr, parent_idx);
                    }
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
            if (isMatmulOp(root.op)) {
                bool ignored = false;
                lhs_dependency_idx = peelExplicitTransposeChain(expr, root.lhs, ignored);
                rhs_dependency_idx = peelExplicitTransposeChain(expr, root.rhs, ignored);
            }

            ensureBoundaryParentEmitted(lhs_dependency_idx, "lhs", isCudnnReduceOp(root.op));
            if (isReduceMinMaxBackwardOp(root.op) || isMatmulOp(root.op) || isConvolutionOp(root.op)) {
                ensureBoundaryParentEmitted(rhs_dependency_idx, "rhs", false);
            }
            if (root.op == ExprOp::GEMM) {
                ensureBoundaryParentEmitted(root.aux, "aux", false);
                ensureScaleDependencyEmitted(root.alpha_node, "alpha");
                ensureScaleDependencyEmitted(root.beta_node, "beta");
            }

            uint32_t stage_out_id = next_value_id++;
            node_output_value_id[root_idx] = stage_out_id;
            if (isReduceMinMaxBackwardOp(root.op)) {
                planned.stages.push_back(buildReduceMinMaxBackwardStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isMatmulOp(root.op)) {
                planned.stages.push_back(buildMatmulStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isConvolutionBackwardOp(root.op)) {
                planned.stages.push_back(buildConvolutionBackwardStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isConvolutionForwardOp(root.op)) {
                planned.stages.push_back(buildConvolutionStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            } else if (isTransposeOp(root.op)) {
                throw std::runtime_error("Internal error: explicit transpose was not lowered to fused tiled-transpose materialization.");
            } else {
                planned.stages.push_back(buildReductionStage(expr, root_idx, stage_out_id, "", node_output_value_id));
            }
            return;
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegion(expr, root_idx, region);

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        std::string region_sig = fusedRegionSignature(expr, root_idx);

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

        uint32_t out_id = next_value_id++;
        node_output_value_id[root_idx] = out_id;
        fused_region_value_id.emplace(region_sig, out_id);

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

        if (isStageBoundaryOp(root.op)) {
            auto ensureBoundaryParentEmitted = [&](uint32_t parent_idx, const char* label, bool force_supported_reduction_input_dtype) {
                if (parent_idx >= expr.nodes.size()) {
                    throw std::runtime_error(std::string("Stage-boundary ") + label + " out of range.");
                }
                const ExprNode& parent = expr.nodes[parent_idx];
                if (parent.op != ExprOp::INPUT || inputRequiresMaterialization(parent)) {
                    if (force_supported_reduction_input_dtype) {
                        forceReductionProducerOutputDTypeIfNeeded(expr, parent_idx);
                    }
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
            if (isMatmulOp(root.op)) {
                bool ignored = false;
                lhs_dependency_idx = peelExplicitTransposeChain(expr, root.lhs, ignored);
                rhs_dependency_idx = peelExplicitTransposeChain(expr, root.rhs, ignored);
            }

            ensureBoundaryParentEmitted(lhs_dependency_idx, "lhs", isCudnnReduceOp(root.op));
            if (isReduceMinMaxBackwardOp(root.op) || isMatmulOp(root.op) || isConvolutionOp(root.op)) {
                ensureBoundaryParentEmitted(rhs_dependency_idx, "rhs", false);
            }
            if (root.op == ExprOp::GEMM) {
                ensureBoundaryParentEmitted(root.aux, "aux", false);
                ensureScaleDependencyEmitted(root.alpha_node, "alpha");
                ensureScaleDependencyEmitted(root.beta_node, "beta");
            }

            uint32_t stage_out_id = next_value_id++;
            node_output_value_id[named_output.node_idx] = stage_out_id;
            if (isReduceMinMaxBackwardOp(root.op)) {
                planned.stages.push_back(
                    buildReduceMinMaxBackwardStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
            } else if (isMatmulOp(root.op)) {
                planned.stages.push_back(
                    buildMatmulStage(expr, named_output.node_idx, stage_out_id, named_output.name, node_output_value_id));
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

            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = stage_out_id,
            });
            continue;
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegion(expr, named_output.node_idx, region);

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        std::string region_sig = fusedRegionSignature(expr, named_output.node_idx);

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

        uint32_t out_id = next_value_id++;
        node_output_value_id[named_output.node_idx] = out_id;

        std::unordered_set<uint32_t> dependency_value_ids;
        collectExternalValueIds(expr, region, node_output_value_id, dependency_value_ids);

        RequestedStageOutput requested_output{
            .name = named_output.name,
            .old_root_idx = named_output.node_idx,
            .value_id = out_id,
        };

        addOrMergeTerminalGroup(std::move(region), std::move(dependency_value_ids), requested_output, region_sig);

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
            case PhysicalExecutionStage::Kind::Reduction:
                reduction = compileReduction(stage.expr);
                compiled->stages.emplace_back(reduction, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            case PhysicalExecutionStage::Kind::ArgMinMax: {
                std::shared_ptr<CompiledArgMinMax> arg_minmax = compileArgMinMax(stage.expr);
                compiled->stages.emplace_back(arg_minmax, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
                break;
            }
            case PhysicalExecutionStage::Kind::Matmul: {
                std::shared_ptr<CompiledMatmul> matmul = compileMatmul(stage.expr);
                compiled->stages.emplace_back(matmul, stage.input_value_ids, stage.outputs, stage.parameter_fan_overrides);
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
            default:
                throw std::runtime_error("Unknown stage kind in EquationCompiler::compile(PhysicalOutputs).");
        }
    }

    compiled->final_outputs = std::move(planned.final_outputs);
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
        if (!node.output_dtype.isPresent()) {
            throw std::runtime_error("Specialized broadcast output node missing resolved output_dtype.");
        }
        output_dtypes.push_back(node.output_dtype.get());
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
    const Optional<DataType> vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    compiled->elements_per_thread = vectorized_dtype.isPresent() ? 2u : 1u;

    compiled->uses_uint32_numel_arg = false;

    if (groups.size() > 1) {
        compiled->launch_kind = CompiledEquation::LaunchKind::BroadcastGrouped;
        compiled->num_broadcast_groups = static_cast<uint32_t>(groups.size());
    }

    specializedBroadcastCache.put(cache_key, compiled);
    return compiled;
}

}  // namespace ThorImplementation
