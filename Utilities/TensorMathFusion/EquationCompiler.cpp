#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "CudaSourceEmitter.h"

using namespace std;

namespace ThorImplementation {

static unordered_map<EquationCacheKey, shared_ptr<CompiledEquation>> compiledEquationCache;

static shared_ptr<CompiledEquation> cacheLookup(const EquationCacheKey& key) {
    auto it = compiledEquationCache.find(key);
    if (it == compiledEquationCache.end()) {
        return nullptr;
    }
    return it->second;
}

static void cacheInsert(const EquationCacheKey& key, shared_ptr<CompiledEquation>& compiledEquation) {
    compiledEquationCache[key] = compiledEquation;
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

shared_ptr<CompiledEquation> EquationCompiler::loadCubin(const EquationCacheKey& key,
                                                         const vector<char>& cubin,
                                                         const string& kernel_name,
                                                         const vector<string>& input_names,
                                                         TensorDescriptor::DataType dtype,
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
    out->dtype = dtype;
    out->deviceNum = device_num;

    return out;
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

vector<char> EquationCompiler::compileToLtoIr(const string& src, const string& kernel_name, const EquationSignature& sig) {
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

shared_ptr<CompiledEquation> EquationCompiler::compile(const PhysicalExpression& expr,
                                                       const EquationSignature& sig,
                                                       const bool broadcast_support) {
    ensureCudaContextCurrent(sig.device_num);

    EquationCacheKey key(canonicalize(expr), sig, broadcast_support);
    shared_ptr<CompiledEquation> hit = cacheLookup(key);
    if (hit)
        return hit;

    string kernel_name = "fused_kernel";
    string cuda_src = CudaSourceEmitter::emit(expr, sig.dtype, kernel_name, broadcast_support);

    vector<string> input_names;
    input_names.reserve(expr.inputs.size());
    for (const NamedInput& input : expr.inputs) {
        input_names.push_back(input.name);
    }

    vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    vector<char> cubin = linkToCubin(ltoir, sig);
    auto compiled = loadCubin(key, cubin, kernel_name, input_names, sig.dtype, sig.device_num);

    cacheInsert(key, compiled);
    return compiled;
}

shared_ptr<CompiledEquation> EquationCompiler::compileFusedStage(const PhysicalExecutionStage& stage,
                                                                 const EquationSignature& sig,
                                                                 const bool broadcast_support) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("compileFusedStage called on non-fused stage.");
    }

    ensureCudaContextCurrent(sig.device_num);

    EquationCacheKey key(canonicalize(stage), sig, broadcast_support);
    shared_ptr<CompiledEquation> hit = cacheLookup(key);
    if (hit)
        return hit;

    string kernel_name = "fused_kernel";
    // string cuda_src = CudaSourceEmitter::emit(stage, sig.dtype, kernel_name, broadcast_support);
    std::string cuda_src;
    if (stage.outputs.size() == 1) {
        // FIXME: broadcast support is not there yet for multi-output
        cuda_src = CudaSourceEmitter::emit(stage.expr, sig.dtype, kernel_name, broadcast_support);
    } else {
        cuda_src = CudaSourceEmitter::emit(stage, sig.dtype, kernel_name, broadcast_support);
    }

    vector<string> input_names;
    input_names.reserve(stage.expr.inputs.size());
    for (const NamedInput& input : stage.expr.inputs) {
        input_names.push_back(input.name);
    }

    vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    vector<char> cubin = linkToCubin(ltoir, sig);
    auto compiled = loadCubin(key, cubin, kernel_name, input_names, sig.dtype, sig.device_num);

    cacheInsert(key, compiled);
    return compiled;
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

    std::vector<PhysicalExecutionStage> planned_stages = splitAtReductionBoundaries(outputs);
    compiled->stages.reserve(planned_stages.size());

    for (const PhysicalExecutionStage& stage : planned_stages) {
        std::shared_ptr<CompiledEquation> flat;
        std::shared_ptr<CompiledEquation> broadcast;
        std::shared_ptr<CompiledReduction> reduction;
        switch (stage.kind) {
            case PhysicalExecutionStage::Kind::FusedKernel:
                flat = compileFusedStage(stage, sig, false);
                broadcast = compileFusedStage(stage, sig, true);
                compiled->stages.emplace_back(flat, broadcast, stage.input_value_ids, stage.outputs);
                break;
            case PhysicalExecutionStage::Kind::Reduction:
                reduction = compileReduction(stage.expr, sig.dtype);
                compiled->stages.emplace_back(reduction, stage.input_value_ids, stage.outputs);
                break;
            default:
                throw std::runtime_error("Unknown stage kind in EquationCompiler::compile(PhysicalOutputs).");
        }
    }

    compiled->final_outputs.reserve(outputs.outputs.size());
    for (const NamedOutput& requested : outputs.outputs) {
        bool found = false;

        for (const CompiledExecutionStage& stage : compiled->stages) {
            for (const CompiledStageOutput& produced : stage.outputs) {
                if (produced.name == requested.name) {
                    compiled->final_outputs.push_back(produced);
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Failed to resolve final output value id for output: " + requested.name);
        }
    }

    return compiled;
}

shared_ptr<CompiledReduction> EquationCompiler::compileReduction(const PhysicalExpression& expr, TensorDescriptor::DataType inout_dtype) {
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

    return make_shared<CompiledReduction>(node.op, node.reduction_axes, node.squeeze_axes, inout_dtype, node.compute_dtype);
}

static bool isStageBoundaryOp(ExprOp op) { return isCudnnReduceOp(op); }

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

    ExprNode input_node;
    input_node.op = ExprOp::INPUT;
    input_node.input_slot = 0;
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

std::vector<PhysicalExecutionStage> EquationCompiler::splitAtReductionBoundaries(const PhysicalOutputs& outputs) {
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
    std::vector<PhysicalExecutionStage> stages;
    uint32_t next_value_id = expr.numInputs();

    std::function<void(uint32_t)> emitForDependency = [&](uint32_t root_idx) {
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
                if (isStageBoundaryOp(parent.op)) {
                    emitForDependency(parent_idx);
                } else {
                    std::unordered_set<uint32_t> region;
                    collectFusableRegion(expr, parent_idx, region);

                    std::unordered_set<uint32_t> boundary_nodes;
                    collectBoundaryDependencies(expr, region, boundary_nodes);
                    for (uint32_t boundary_root : boundary_nodes) {
                        emitForDependency(boundary_root);
                    }

                    uint32_t fused_out_id = next_value_id++;
                    node_output_value_id[parent_idx] = fused_out_id;

                    std::vector<RequestedStageOutput> requested_outputs{RequestedStageOutput{
                        .name = "",
                        .old_root_idx = parent_idx,
                        .value_id = fused_out_id,
                    }};

                    stages.push_back(buildFusedStage(expr, region, requested_outputs, node_output_value_id));
                }
            }

            uint32_t reduce_out_id = next_value_id++;
            node_output_value_id[root_idx] = reduce_out_id;
            stages.push_back(buildReductionStage(expr, root_idx, reduce_out_id, "", node_output_value_id));
            return;
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegion(expr, root_idx, region);

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        uint32_t out_id = next_value_id++;
        node_output_value_id[root_idx] = out_id;

        std::vector<RequestedStageOutput> requested_outputs{RequestedStageOutput{
            .name = "",
            .old_root_idx = root_idx,
            .value_id = out_id,
        }};

        stages.push_back(buildFusedStage(expr, region, requested_outputs, node_output_value_id));
    };

    struct OutputRegionGroup {
        std::vector<uint32_t> region_nodes_sorted;
        std::vector<RequestedStageOutput> outputs;
    };

    std::map<std::vector<uint32_t>, OutputRegionGroup> fused_terminal_groups;

    for (const NamedOutput& named_output : outputs.outputs) {
        const ExprNode& root = expr.nodes[named_output.node_idx];

        if (isStageBoundaryOp(root.op)) {
            uint32_t parent_idx = root.lhs;
            if (parent_idx >= expr.nodes.size()) {
                throw std::runtime_error("Reduction lhs out of range.");
            }

            const ExprNode& parent = expr.nodes[parent_idx];
            if (parent.op != ExprOp::INPUT) {
                if (isStageBoundaryOp(parent.op)) {
                    emitForDependency(parent_idx);
                } else {
                    std::unordered_set<uint32_t> region;
                    collectFusableRegion(expr, parent_idx, region);

                    std::unordered_set<uint32_t> boundary_nodes;
                    collectBoundaryDependencies(expr, region, boundary_nodes);
                    for (uint32_t boundary_root : boundary_nodes) {
                        emitForDependency(boundary_root);
                    }

                    if (node_output_value_id.find(parent_idx) == node_output_value_id.end()) {
                        uint32_t fused_out_id = next_value_id++;
                        node_output_value_id[parent_idx] = fused_out_id;

                        std::vector<RequestedStageOutput> dependency_outputs{RequestedStageOutput{
                            .name = "",
                            .old_root_idx = parent_idx,
                            .value_id = fused_out_id,
                        }};
                        stages.push_back(buildFusedStage(expr, region, dependency_outputs, node_output_value_id));
                    }
                }
            }

            uint32_t reduce_out_id = next_value_id++;
            node_output_value_id[named_output.node_idx] = reduce_out_id;
            stages.push_back(buildReductionStage(expr, named_output.node_idx, reduce_out_id, named_output.name, node_output_value_id));
            continue;
        }

        std::unordered_set<uint32_t> region;
        collectFusableRegion(expr, named_output.node_idx, region);

        std::unordered_set<uint32_t> boundary_nodes;
        collectBoundaryDependencies(expr, region, boundary_nodes);
        for (uint32_t boundary_root : boundary_nodes) {
            emitForDependency(boundary_root);
        }

        uint32_t out_id = next_value_id++;
        node_output_value_id[named_output.node_idx] = out_id;

        std::vector<uint32_t> region_key(region.begin(), region.end());
        std::sort(region_key.begin(), region_key.end());

        auto& group = fused_terminal_groups[region_key];
        group.region_nodes_sorted = region_key;
        group.outputs.push_back(RequestedStageOutput{
            .name = named_output.name,
            .old_root_idx = named_output.node_idx,
            .value_id = out_id,
        });
    }

    for (const auto& [region_key, group] : fused_terminal_groups) {
        std::unordered_set<uint32_t> region_nodes(group.region_nodes_sorted.begin(), group.region_nodes_sorted.end());
        stages.push_back(buildFusedStage(expr, region_nodes, group.outputs, node_output_value_id));
    }

    return stages;
}

}  // namespace ThorImplementation
