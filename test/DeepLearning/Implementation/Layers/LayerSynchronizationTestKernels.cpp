#include "test/DeepLearning/Implementation/Layers/LayerSynchronizationTestKernels.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/CudaDriver/CudaDrivertApi.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ThorImplementation::Test {
namespace {

class StreamMemoryOperations {
   public:
    using WaitValue32 = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);

    static StreamMemoryOperations& instance() {
        static StreamMemoryOperations operations;
        return operations;
    }

    CUresult wait(CUstream stream, CUdeviceptr address, cuuint32_t value) const {
        return waitValue32(stream, address, value, CU_STREAM_WAIT_VALUE_EQ);
    }

   private:
    WaitValue32 waitValue32;

    StreamMemoryOperations()
        : waitValue32(load<WaitValue32>("cuStreamWaitValue32_v2", "cuStreamWaitValue32")) {}

    template <typename Function>
    static Function load(const char* preferredName, const char* fallbackName) {
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (driver == nullptr)
            driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (driver == nullptr)
            throw std::runtime_error(std::string("CUDA driver library libcuda.so.1 is not available: ") + dlerror());

        dlerror();
        void* symbol = dlsym(driver, preferredName);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr) {
            dlerror();
            symbol = dlsym(driver, fallbackName);
            error = dlerror();
        }
        if (error != nullptr || symbol == nullptr)
            throw std::runtime_error(std::string("Missing CUDA driver stream-memory-operation symbol: ") + preferredName);

        return reinterpret_cast<Function>(symbol);
    }
};

CUstream asDriverStream(const Stream& stream) {
    return reinterpret_cast<CUstream>(stream.getStream());
}

CUdeviceptr asDevicePointer(const uint32_t* pointer) {
    return static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer));
}

void checkDriverResult(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS)
        return;

    auto& driver = CudaDriverApi::instance();
    const char* name = nullptr;
    const char* description = nullptr;
    (void)driver.cuGetErrorName(status, &name);
    (void)driver.cuGetErrorString(status, &description);
    throw std::runtime_error(std::string(operation) + " failed with " + (name != nullptr ? name : "<unknown>") + ": " +
                             (description != nullptr ? description : "<no description>"));
}

}  // namespace

DeviceStreamGate::DeviceStreamGate(int32_t gpuNum) : gpuNum(gpuNum) {
    ScopedGpu scopedGpu(gpuNum);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&released_h), sizeof(uint32_t), cudaHostAllocMapped));
    void* mappedDevicePointer = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&mappedDevicePointer, released_h, 0));
    released_d = static_cast<uint32_t*>(mappedDevicePointer);
    std::atomic_ref<uint32_t>(*released_h).store(0u, std::memory_order_release);
}

DeviceStreamGate::~DeviceStreamGate() {
    releaseNoThrow();

    if (gatedStream.has_value()) {
        ScopedGpu scopedGpu(gpuNum);
        (void)cudaStreamSynchronize(gatedStream->getStream());
    }

    if (released_h != nullptr) {
        ScopedGpu scopedGpu(gpuNum);
        (void)cudaFreeHost(released_h);
        released_h = nullptr;
        released_d = nullptr;
    }
}

void DeviceStreamGate::enqueue(const Stream& stream) {
    THOR_THROW_IF_FALSE(!gatedStream.has_value());
    THOR_THROW_IF_FALSE(stream.getGpuNum() == gpuNum);

    ScopedGpu scopedGpu(gpuNum);
    checkDriverResult(StreamMemoryOperations::instance().wait(asDriverStream(stream), asDevicePointer(released_d), 1u),
                      "cuStreamWaitValue32");
    gatedStream = stream;
    completionEvent = stream.putEvent(/*enableTiming=*/false,
                                      /*expectingHostToWaitOnThisOne=*/true);
}

void DeviceStreamGate::release() {
    if (released)
        return;

    THOR_THROW_IF_FALSE(released_h != nullptr);
    std::atomic_ref<uint32_t>(*released_h).store(1u, std::memory_order_release);
    released = true;
}

bool DeviceStreamGate::isComplete() {
    THOR_THROW_IF_FALSE(completionEvent.isInitialized());

    ScopedGpu scopedGpu(gpuNum);
    cudaError_t status = cudaEventQuery(completionEvent.getEvent());
    if (status == cudaErrorNotReady)
        return false;
    CUDA_CHECK(status);
    return true;
}

void DeviceStreamGate::releaseNoThrow() noexcept {
    if (released || released_h == nullptr)
        return;

    std::atomic_ref<uint32_t>(*released_h).store(1u, std::memory_order_release);
    released = true;
}

}  // namespace ThorImplementation::Test
