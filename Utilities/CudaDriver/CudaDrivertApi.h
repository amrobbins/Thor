#pragma once

#include <cuda.h>
#include <dlfcn.h>

#include <stdexcept>
#include <string>

class CudaDriverApi {
   public:
    static CudaDriverApi& instance() {
        static CudaDriverApi api;
        return api;
    }

    CudaDriverApi(const CudaDriverApi&) = delete;
    CudaDriverApi& operator=(const CudaDriverApi&) = delete;

    CUresult cuInit(unsigned int flags) { return p_cuInit(flags); }

    CUresult cuDeviceGet(CUdevice* device, int ordinal) { return p_cuDeviceGet(device, ordinal); }

    CUresult cuCtxGetCurrent(CUcontext* ctx) { return p_cuCtxGetCurrent(ctx); }

    CUresult cuCtxSetCurrent(CUcontext ctx) { return p_cuCtxSetCurrent(ctx); }

    CUresult cuCtxGetDevice(CUdevice* device) { return p_cuCtxGetDevice(device); }

    CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, CUdevice device) { return p_cuDevicePrimaryCtxRetain(ctx, device); }

    CUresult cuModuleLoadData(CUmodule* module, const void* image) { return p_cuModuleLoadData(module, image); }

    CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
        return p_cuModuleGetFunction(function, module, name);
    }

    CUresult cuModuleUnload(CUmodule module) { return p_cuModuleUnload(module); }

    CUresult cuLaunchKernel(CUfunction function,
                            unsigned int gridDimX,
                            unsigned int gridDimY,
                            unsigned int gridDimZ,
                            unsigned int blockDimX,
                            unsigned int blockDimY,
                            unsigned int blockDimZ,
                            unsigned int sharedMemBytes,
                            CUstream stream,
                            void** kernelParams,
                            void** extra) {
        return p_cuLaunchKernel(
            function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
    }

    CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction function, void** kernelParams, void** extra) {
        return p_cuLaunchKernelEx(config, function, kernelParams, extra);
    }

    CUresult graphKernelNodeGetParams(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS* params) {
        return p_cuGraphKernelNodeGetParams_v2(node, params);
    }

    CUresult graphExecKernelNodeSetParams(CUgraphExec graphExec, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS* params) {
        return p_cuGraphExecKernelNodeSetParams_v2(graphExec, node, params);
    }

    CUresult cuGetErrorName(CUresult error, const char** pStr) { return p_cuGetErrorName(error, pStr); }

    CUresult cuGetErrorString(CUresult error, const char** pStr) { return p_cuGetErrorString(error, pStr); }

   private:
    void* handle = nullptr;

    using cuInit_t = CUresult (*)(unsigned int);
    using cuDeviceGet_t = CUresult (*)(CUdevice*, int);
    using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);
    using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
    using cuCtxGetDevice_t = CUresult (*)(CUdevice*);
    using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
    using cuModuleLoadData_t = CUresult (*)(CUmodule*, const void*);
    using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
    using cuModuleUnload_t = CUresult (*)(CUmodule);
    using cuLaunchKernel_t = CUresult (*)(CUfunction,
                                          unsigned int,
                                          unsigned int,
                                          unsigned int,
                                          unsigned int,
                                          unsigned int,
                                          unsigned int,
                                          unsigned int,
                                          CUstream,
                                          void**,
                                          void**);
    using cuLaunchKernelEx_t = CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**);
    using cuGraphKernelNodeGetParams_v2_t = CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS*);
    using cuGraphExecKernelNodeSetParams_v2_t = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*);
    using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
    using cuGetErrorString_t = CUresult (*)(CUresult, const char**);

    cuInit_t p_cuInit = nullptr;
    cuDeviceGet_t p_cuDeviceGet = nullptr;
    cuCtxGetCurrent_t p_cuCtxGetCurrent = nullptr;
    cuCtxSetCurrent_t p_cuCtxSetCurrent = nullptr;
    cuCtxGetDevice_t p_cuCtxGetDevice = nullptr;
    cuDevicePrimaryCtxRetain_t p_cuDevicePrimaryCtxRetain = nullptr;
    cuModuleLoadData_t p_cuModuleLoadData = nullptr;
    cuModuleGetFunction_t p_cuModuleGetFunction = nullptr;
    cuModuleUnload_t p_cuModuleUnload = nullptr;
    cuLaunchKernel_t p_cuLaunchKernel = nullptr;
    cuLaunchKernelEx_t p_cuLaunchKernelEx = nullptr;
    cuGraphKernelNodeGetParams_v2_t p_cuGraphKernelNodeGetParams_v2 = nullptr;
    cuGraphExecKernelNodeSetParams_v2_t p_cuGraphExecKernelNodeSetParams_v2 = nullptr;
    cuGetErrorName_t p_cuGetErrorName = nullptr;
    cuGetErrorString_t p_cuGetErrorString = nullptr;

    CudaDriverApi() {
        handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            throw std::runtime_error(std::string("CUDA driver library libcuda.so.1 is not available: ") + dlerror());
        }

        p_cuInit = load<cuInit_t>("cuInit");
        p_cuDeviceGet = load<cuDeviceGet_t>("cuDeviceGet");
        p_cuCtxGetCurrent = load<cuCtxGetCurrent_t>("cuCtxGetCurrent");
        p_cuCtxSetCurrent = load<cuCtxSetCurrent_t>("cuCtxSetCurrent");
        p_cuCtxGetDevice = load<cuCtxGetDevice_t>("cuCtxGetDevice");
        p_cuDevicePrimaryCtxRetain = load<cuDevicePrimaryCtxRetain_t>("cuDevicePrimaryCtxRetain");

        p_cuModuleLoadData = load<cuModuleLoadData_t>("cuModuleLoadData");
        p_cuModuleGetFunction = load<cuModuleGetFunction_t>("cuModuleGetFunction");
        p_cuModuleUnload = load<cuModuleUnload_t>("cuModuleUnload");
        p_cuLaunchKernel = load<cuLaunchKernel_t>("cuLaunchKernel");
        p_cuLaunchKernelEx = load<cuLaunchKernelEx_t>("cuLaunchKernelEx");
        p_cuGraphKernelNodeGetParams_v2 = load<cuGraphKernelNodeGetParams_v2_t>("cuGraphKernelNodeGetParams_v2");
        p_cuGraphExecKernelNodeSetParams_v2 =
            load<cuGraphExecKernelNodeSetParams_v2_t>("cuGraphExecKernelNodeSetParams_v2");

        p_cuGetErrorName = load<cuGetErrorName_t>("cuGetErrorName");
        p_cuGetErrorString = load<cuGetErrorString_t>("cuGetErrorString");
    }

    ~CudaDriverApi() {
        // I would intentionally not dlclose libcuda here.
        // CUDA contexts/modules may still be alive during static destruction.
    }

    template <typename Fn>
    Fn load(const char* name) {
        dlerror();
        void* sym = dlsym(handle, name);
        const char* err = dlerror();

        if (err || !sym) {
            throw std::runtime_error(std::string("Missing CUDA driver symbol: ") + name);
        }

        return reinterpret_cast<Fn>(sym);
    }
};
