#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/ExpressionDTypeResolution.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

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

// static unordered_map<EquationCacheKey, shared_ptr<CompiledEquation>> compiledEquationCache;
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
    uint32_t input_slot = UINT32_MAX;
    uint64_t scalar_bits = 0;
    int32_t output_dtype = -1;
    int32_t compute_dtype = -1;
    int32_t backward_output_dtype = -1;
    int32_t backward_compute_dtype = -1;

    bool operator==(const StageNodeKey& other) const = default;
};

struct StageNodeKeyHash {
    size_t operator()(const StageNodeKey& k) const noexcept {
        size_t h = std::hash<int>{}(static_cast<int>(k.op));
        hashCombine(h, std::hash<uint32_t>{}(k.lhs));
        hashCombine(h, std::hash<uint32_t>{}(k.rhs));
        hashCombine(h, std::hash<uint32_t>{}(k.input_slot));
        hashCombine(h, std::hash<uint64_t>{}(k.scalar_bits));
        hashCombine(h, std::hash<int32_t>{}(k.output_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.compute_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.backward_output_dtype));
        hashCombine(h, std::hash<int32_t>{}(k.backward_compute_dtype));
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
    key.output_dtype = optionalDTypeTag(n.output_dtype);
    key.compute_dtype = optionalDTypeTag(n.compute_dtype);
    key.backward_output_dtype = optionalDTypeTag(n.backward_output_dtype);
    key.backward_compute_dtype = optionalDTypeTag(n.backward_compute_dtype);

    switch (n.op) {
        case ExprOp::INPUT:
            key.input_slot = n.input_slot;
            break;

        case ExprOp::SCALAR_FP:
            key.scalar_bits = scalarBits(n.scalar_fp);
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

static bool isStageBoundaryOp(ExprOp op) { return isCudnnReduceOp(op); }

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
        if (parent.op == ExprOp::INPUT) {
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

        if (node.op == ExprOp::INPUT) {
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
        case ExprOp::POW:
            return "POW";
        case ExprOp::MIN:
            return "MIN";
        case ExprOp::MAX:
            return "MAX";
        case ExprOp::REDUCE_SUM:
            return "RSUM";
        case ExprOp::REDUCE_PROD:
            return "RPROD";
        case ExprOp::REDUCE_MIN:
            return "RMIN";
        case ExprOp::REDUCE_MAX:
            return "RMAX";
        case ExprOp::REDUCE_AVG:
            return "RAVG";
        case ExprOp::REDUCE_NORM1:
            return "RNORM1";
        case ExprOp::REDUCE_NORM2:
            return "RNORM2";
        default:
            throw std::runtime_error("Unsupported op in fusedRegionSignature.");
    }
}

static std::string optionalDTypeSignature(const Optional<DataType>& dtype) {
    if (!dtype.isPresent()) {
        return "none";
    }
    return TensorDescriptor::getElementTypeName(dtype.get());
}

static void appendNodeDTypeSignature(std::string& s, const ExprNode& node) {
    s += ";out=" + optionalDTypeSignature(node.output_dtype);
    s += ";compute=" + optionalDTypeSignature(node.compute_dtype);
    s += ";bwd_out=" + optionalDTypeSignature(node.backward_output_dtype);
    s += ";bwd_compute=" + optionalDTypeSignature(node.backward_compute_dtype);
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

        case ExprOp::SCALAR_FP: {
            std::string s = std::string("F(") + std::to_string(scalarBits(node.scalar_fp)) + ")";
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
        std::string s = std::string(fusedOpTag(node.op)) + "(lhs=" + lhs + ",axes=" + uintVecSignature(node.reduction_axes) +
                        ",squeeze=" + uintVecSignature(node.squeeze_axes) + ")";
        appendNodeDTypeSignature(s, node);
        return s;
    }

    if (!Expression::isBinaryOp(node.op)) {
        std::string s = std::string(fusedOpTag(node.op)) + "(" + lhs + ")";
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
    out->input_dtypes = input_dtypes;
    out->output_dtypes = output_dtypes;
    out->deviceNum = device_num;

    return out;
}

static std::vector<DataType> collectCompiledInputDTypes(const PhysicalExpression& expr) {
    std::vector<DataType> input_dtypes(expr.numInputs(), DataType::FP32);
    std::vector<uint8_t> seen(expr.numInputs(), 0);

    for (const ExprNode& node : expr.nodes) {
        if (node.op != ExprOp::INPUT) {
            continue;
        }
        if (!node.output_dtype.isPresent()) {
            throw runtime_error("Fused stage input node is missing resolved output_dtype.");
        }
        if (node.input_slot >= input_dtypes.size()) {
            throw runtime_error("Input slot out of range while collecting compiled input dtypes.");
        }

        const DataType dtype = node.output_dtype.get();
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

    std::string cuda_include_path = std::string("--include-path=") + THOR_CUDA_INCLUDE_DIR;
    vector<const char*> opts = {arch.c_str(), "-dlto", "--std=c++17", "-fmad=true", cuda_include_path.c_str()};
    if (sig.use_fast_math)
        opts.push_back("--use_fast_math");

    nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    string log(log_size, '\0');
    if (log_size > 1)
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));

    if (res != NVRTC_SUCCESS)
        throw runtime_error("NVRTC compile failed:\n" + log);

    size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size));
    vector<char> ltoir(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog, ltoir.data()));

    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return ltoir;
}

shared_ptr<CompiledEquation> EquationCompiler::compileFusedStage(const PhysicalExecutionStage& stage, const EquationSignature& sig) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("compileFusedStage called on non-fused stage.");
    }

    ensureCudaContextCurrent(sig.device_num);

    EquationCacheKey key(canonicalize(stage), sig);
    shared_ptr<CompiledEquation> hit = cacheLookup(key);
    if (hit)
        return hit;

    string kernel_name = "fused_kernel";
    const std::string cuda_src = CudaSourceEmitter::emitFlat(stage, kernel_name);

    vector<string> input_names;
    input_names.reserve(stage.expr.inputs.size());
    for (const NamedInput& input : stage.expr.inputs) {
        input_names.push_back(input.name);
    }
    const std::vector<DataType> input_dtypes = collectCompiledInputDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectCompiledOutputDTypes(stage);

    vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    vector<char> cubin = linkToCubin(ltoir, sig);
    auto compiled = loadCubin(key, cubin, kernel_name, input_names, input_dtypes, output_dtypes, sig.device_num);
    const Optional<DataType> vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    compiled->elements_per_thread = vectorized_dtype.isPresent() ? 2u : 1u;

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
    if (!isCudnnReduceOp(node.op)) {
        throw std::runtime_error("Reduction stage output node is not a cuDNN reduction op.");
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

    if (!input_node.output_dtype.isPresent()) {
        throw std::runtime_error("Reduction input node missing resolved output_dtype.");
    }
    if (!node.output_dtype.isPresent()) {
        throw std::runtime_error("Reduction node missing resolved output_dtype.");
    }

    return make_shared<CompiledReduction>(
        node.op, node.reduction_axes, node.squeeze_axes, input_node.output_dtype.get(), node.output_dtype.get(), node.compute_dtype);
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
            throw std::runtime_error("collectFusableRegion called on reduction root.");
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

    auto getOrCreateLocalInputSlot = [&](uint32_t value_id) -> uint32_t {
        auto it = value_id_to_local_input_slot.find(value_id);
        if (it != value_id_to_local_input_slot.end()) {
            return it->second;
        }

        uint32_t local_slot = static_cast<uint32_t>(stage_input_value_ids.size());
        stage_input_value_ids.push_back(value_id);
        value_id_to_local_input_slot.emplace(value_id, local_slot);

        NamedInput input;
        input.name = "arg" + std::to_string(local_slot);
        input.slot = local_slot;
        stage_expr.inputs.push_back(std::move(input));

        return local_slot;
    };

    for (uint32_t old_idx : sorted_nodes) {
        ExprNode new_node = expr.nodes[old_idx];

        if (new_node.op == ExprOp::INPUT) {
            uint32_t value_id = new_node.input_slot;
            new_node.input_slot = getOrCreateLocalInputSlot(value_id);
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
                    input_node.input_slot = getOrCreateLocalInputSlot(out_it->second);

                    // This local INPUT stands in for the already-materialized parent value
                    // crossing a stage boundary, so it should inherit that value's dtype semantics.
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
                    input_node.input_slot = getOrCreateLocalInputSlot(out_it->second);

                    // This local INPUT stands in for the already-materialized parent value
                    // crossing a stage boundary, so it should inherit that value's dtype semantics.
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
        });
    }

    if (!stage_outputs.empty()) {
        stage_expr.output_node = stage_outputs.front().local_node_idx;
    }

    deduplicateFusedStageExpr(stage_expr, stage_outputs);

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
    if (!isStageBoundaryOp(node.op)) {
        throw std::runtime_error("buildReductionStage called on non-reduction node.");
    }

    if (node.lhs == UINT32_MAX) {
        throw std::runtime_error("Reduction node missing lhs input.");
    }

    PhysicalExpression stage_expr;
    stage_expr.inputs.push_back(NamedInput{"arg0", 0});

    ExprNode reduction = node;
    std::vector<uint32_t> input_value_ids;
    input_value_ids.reserve(1);

    uint32_t parent_idx = reduction.lhs;
    if (parent_idx >= expr.nodes.size()) {
        throw std::runtime_error("Reduction input node index out of range.");
    }

    const ExprNode& parent = expr.nodes[parent_idx];
    if (parent.op == ExprOp::INPUT) {
        input_value_ids.push_back(parent.input_slot);
    } else {
        auto out_it = node_output_value_id.find(parent_idx);
        if (out_it == node_output_value_id.end()) {
            throw std::runtime_error("Missing value id for reduction input.");
        }
        input_value_ids.push_back(out_it->second);
    }

    if (!parent.output_dtype.isPresent()) {
        throw std::runtime_error("Reduction parent node is missing resolved output_dtype.");
    }

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;

    // This local INPUT node represents the already-materialized value produced by
    // the parent expression feeding the reduction. It should inherit the parent's
    // value dtype semantics.
    input_node.output_dtype = parent.output_dtype;
    input_node.compute_dtype = parent.compute_dtype;
    input_node.backward_output_dtype = parent.backward_output_dtype;
    input_node.backward_compute_dtype = parent.backward_compute_dtype;

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
        .kind = PhysicalExecutionStage::Kind::Reduction,
        .expr = std::move(stage_expr),
        .input_value_ids = std::move(input_value_ids),
        .outputs = std::move(stage_outputs),
    };
}

struct PlannedExecution {
    std::vector<PhysicalExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

static PlannedExecution planExecution(const PhysicalOutputs& outputs) {
    if (!outputs.expr) {
        throw std::runtime_error("Cannot split null PhysicalOutputs expression.");
    }
    if (outputs.outputs.empty()) {
        throw std::runtime_error("Cannot split empty PhysicalOutputs.");
    }

    const PhysicalExpression& expr = *outputs.expr;
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

    emitForDependency = [&](uint32_t root_idx) {
        if (node_output_value_id.find(root_idx) != node_output_value_id.end()) {
            return;
        }

        const ExprNode& root = expr.nodes[root_idx];
        if (isStageBoundaryOp(root.op)) {
            uint32_t parent_idx = root.lhs;
            if (parent_idx >= expr.nodes.size()) {
                throw std::runtime_error("Reduction lhs out of range.");
            }

            const ExprNode& parent = expr.nodes[parent_idx];
            if (parent.op != ExprOp::INPUT) {
                emitForDependency(parent_idx);
            }

            uint32_t reduce_out_id = next_value_id++;
            node_output_value_id[root_idx] = reduce_out_id;
            planned.stages.push_back(buildReductionStage(expr, root_idx, reduce_out_id, "", node_output_value_id));
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

        if (isStageBoundaryOp(root.op)) {
            uint32_t parent_idx = root.lhs;
            if (parent_idx >= expr.nodes.size()) {
                throw std::runtime_error("Reduction lhs out of range.");
            }

            const ExprNode& parent = expr.nodes[parent_idx];
            if (parent.op != ExprOp::INPUT) {
                emitForDependency(parent_idx);
            }

            uint32_t reduce_out_id = next_value_id++;
            node_output_value_id[named_output.node_idx] = reduce_out_id;
            planned.stages.push_back(
                buildReductionStage(expr, named_output.node_idx, reduce_out_id, named_output.name, node_output_value_id));

            planned.final_outputs.push_back(CompiledStageOutput{
                .name = named_output.name,
                .local_node_idx = UINT32_MAX,
                .value_id = reduce_out_id,
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
                compiled->stages.emplace_back(stage.expr, flat, stage.input_value_ids, stage.outputs);
                break;
            case PhysicalExecutionStage::Kind::Reduction:
                reduction = compileReduction(stage.expr);
                compiled->stages.emplace_back(reduction, stage.input_value_ids, stage.outputs);
                break;
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
    input_names.reserve(stage.expr.inputs.size());
    for (const NamedInput& input : stage.expr.inputs) {
        input_names.push_back(input.name);
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

    shared_ptr<CompiledEquation> compiled = loadCubin(
        EquationCacheKey(canonicalize(stage.expr), sig), cubin, kernel_name, input_names, input_dtypes, output_dtypes, sig.device_num);
    const Optional<DataType> vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    compiled->elements_per_thread = vectorized_dtype.isPresent() ? 2u : 1u;

    if (groups.size() > 1) {
        compiled->launch_kind = CompiledEquation::LaunchKind::BroadcastGrouped;
        compiled->num_broadcast_groups = static_cast<uint32_t>(groups.size());
    }

    specializedBroadcastCache.put(cache_key, compiled);
    return compiled;
}

}  // namespace ThorImplementation
