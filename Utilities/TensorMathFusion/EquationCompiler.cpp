#include "Utilities/TensorMathFusion/EquationCompiler.h"

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
                                                         uint32_t num_inputs,
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
    out->num_inputs = num_inputs;
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

    vector<const char*> opts = {arch.c_str(), "-dlto", "--std=c++17", "-fmad=true"};
    if (sig.use_fast_math)
        opts.push_back("--use_fast_math");

    nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

    // collect log either way
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

shared_ptr<CompiledEquation> EquationCompiler::compile(const PhysicalExpression& expr, const EquationSignature& sig) {
    ensureCudaContextCurrent(sig.device_num);

    EquationCacheKey key(canonicalize(expr), sig);
    if (auto hit = cacheLookup(key))
        return hit;

    string kernel_name = "fused_kernel";
    string cuda_src = CudaSourceEmitter::emit(expr, kernel_name, false);

    vector<char> ltoir = compileToLtoIr(cuda_src, kernel_name, sig);
    vector<char> cubin = linkToCubin(ltoir, sig);
    auto compiled = loadCubin(key, cubin, kernel_name, expr.num_inputs, TensorDescriptor::DataType::FP32, sig.device_num);
    compiled->num_inputs = expr.num_inputs;

    cacheInsert(key, compiled);
    return compiled;
}

}  // namespace ThorImplementation
