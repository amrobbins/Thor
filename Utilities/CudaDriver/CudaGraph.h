#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

#include "Utilities/Common/Stream.h"

namespace ThorImplementation {

struct CudaGraphKernelLaunch {
    CUfunction function = nullptr;
    dim3 gridDim{1, 1, 1};
    dim3 blockDim{1, 1, 1};
    uint32_t sharedMemBytes = 0;
    void** kernelParams = nullptr;
    void** extra = nullptr;

    void validate() const;
};

class DeviceUpdatableKernelNode {
   public:
    DeviceUpdatableKernelNode() = default;
    explicit DeviceUpdatableKernelNode(CUgraphDeviceNode node) : driverNode_(node) {}

    bool isInitialized() const { return driverNode_ != nullptr; }
    explicit operator bool() const { return isInitialized(); }

    CUgraphDeviceNode driverNode() const { return driverNode_; }

    cudaGraphDeviceNode_t runtimeNode() const {
        return reinterpret_cast<cudaGraphDeviceNode_t>(driverNode_);
    }

   private:
    CUgraphDeviceNode driverNode_ = nullptr;
};

class DeviceUpdatableKernelNodeDeviceHandle {
   public:
    DeviceUpdatableKernelNodeDeviceHandle() = default;
    explicit DeviceUpdatableKernelNodeDeviceHandle(int gpuNum);

    DeviceUpdatableKernelNodeDeviceHandle(const DeviceUpdatableKernelNodeDeviceHandle&) = delete;
    DeviceUpdatableKernelNodeDeviceHandle& operator=(const DeviceUpdatableKernelNodeDeviceHandle&) = delete;

    DeviceUpdatableKernelNodeDeviceHandle(DeviceUpdatableKernelNodeDeviceHandle&& other) noexcept;
    DeviceUpdatableKernelNodeDeviceHandle& operator=(DeviceUpdatableKernelNodeDeviceHandle&& other) noexcept;

    ~DeviceUpdatableKernelNodeDeviceHandle();

    bool isInitialized() const { return deviceNode_ != nullptr; }
    explicit operator bool() const { return isInitialized(); }

    int getGpuNum() const { return gpuNum_; }
    const cudaGraphDeviceNode_t* devicePtr() const { return deviceNode_; }

    void upload(DeviceUpdatableKernelNode node, Stream stream) const;

   private:
    void reset() noexcept;

    cudaGraphDeviceNode_t* deviceNode_ = nullptr;
    int gpuNum_ = -1;
};

class CudaGraphExecutable {
   public:
    CudaGraphExecutable() = default;
    CudaGraphExecutable(cudaGraphExec_t graphExec, int gpuNum, bool containsDeviceUpdatableNodes);

    CudaGraphExecutable(const CudaGraphExecutable&) = delete;
    CudaGraphExecutable& operator=(const CudaGraphExecutable&) = delete;

    CudaGraphExecutable(CudaGraphExecutable&& other) noexcept;
    CudaGraphExecutable& operator=(CudaGraphExecutable&& other) noexcept;

    ~CudaGraphExecutable();

    bool isInitialized() const { return graphExec_ != nullptr; }
    bool containsDeviceUpdatableNodes() const { return containsDeviceUpdatableNodes_; }
    bool isUploaded() const { return uploaded_; }
    int getGpuNum() const { return gpuNum_; }

    void upload(Stream stream);
    void launch(Stream stream) const;

   private:
    void reset() noexcept;

    cudaGraphExec_t graphExec_ = nullptr;
    int gpuNum_ = -1;
    bool containsDeviceUpdatableNodes_ = false;
    bool uploaded_ = false;
};

class CudaGraph {
   public:
    CudaGraph() = default;
    CudaGraph(cudaGraph_t graph, int gpuNum, bool containsDeviceUpdatableNodes);

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    CudaGraph(CudaGraph&& other) noexcept;
    CudaGraph& operator=(CudaGraph&& other) noexcept;

    ~CudaGraph();

    bool isInitialized() const { return graph_ != nullptr; }
    bool containsDeviceUpdatableNodes() const { return containsDeviceUpdatableNodes_; }
    int getGpuNum() const { return gpuNum_; }

    CudaGraphExecutable instantiate();

   private:
    void reset() noexcept;

    cudaGraph_t graph_ = nullptr;
    int gpuNum_ = -1;
    bool containsDeviceUpdatableNodes_ = false;
    bool instantiated_ = false;
};

class CudaGraphCaptureBuilder {
   public:
    explicit CudaGraphCaptureBuilder(Stream stream, cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal);

    CudaGraphCaptureBuilder(const CudaGraphCaptureBuilder&) = delete;
    CudaGraphCaptureBuilder& operator=(const CudaGraphCaptureBuilder&) = delete;

    CudaGraphCaptureBuilder(CudaGraphCaptureBuilder&& other) noexcept;
    CudaGraphCaptureBuilder& operator=(CudaGraphCaptureBuilder&& other) noexcept;

    ~CudaGraphCaptureBuilder();

    Stream stream() const { return stream_; }
    bool isCapturing() const { return active_; }
    bool containsDeviceUpdatableNodes() const { return containsDeviceUpdatableNodes_; }

    void captureKernel(const CudaGraphKernelLaunch& launch);
    DeviceUpdatableKernelNode captureDeviceUpdatableKernel(const CudaGraphKernelLaunch& launch);

    CudaGraph endCapture();
    CudaGraphExecutable endCaptureAndInstantiate(Stream uploadStream, bool upload = true);

   private:
    void ensureActive() const;
    DeviceUpdatableKernelNode captureKernelImpl(const CudaGraphKernelLaunch& launch, bool deviceUpdatable);
    void abortCaptureNoThrow() noexcept;

    Stream stream_;
    bool active_ = false;
    bool containsDeviceUpdatableNodes_ = false;
};

}  // namespace ThorImplementation
