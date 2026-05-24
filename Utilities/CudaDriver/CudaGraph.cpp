#include "Utilities/CudaDriver/CudaGraph.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {
namespace {

CUstream asDriverStream(cudaStream_t stream) {
    return reinterpret_cast<CUstream>(stream);
}

unsigned int checkedDim(const char* name, uint32_t value) {
    if (value == 0) {
        throw std::invalid_argument(std::string("CUDA graph kernel launch ") + name + " must be non-zero.");
    }
    return value;
}

CUlaunchConfig makeLaunchConfig(const CudaGraphKernelLaunch& launch, Stream stream) {
    CUlaunchConfig config{};
    config.gridDimX = checkedDim("gridDim.x", launch.gridDim.x);
    config.gridDimY = checkedDim("gridDim.y", launch.gridDim.y);
    config.gridDimZ = checkedDim("gridDim.z", launch.gridDim.z);
    config.blockDimX = checkedDim("blockDim.x", launch.blockDim.x);
    config.blockDimY = checkedDim("blockDim.y", launch.blockDim.y);
    config.blockDimZ = checkedDim("blockDim.z", launch.blockDim.z);
    config.sharedMemBytes = launch.sharedMemBytes;
    config.hStream = asDriverStream(stream.getStream());
    config.attrs = nullptr;
    config.numAttrs = 0;
    return config;
}

}  // namespace


DeviceUpdatableKernelNodeDeviceHandle::DeviceUpdatableKernelNodeDeviceHandle(int gpuNum) : gpuNum_(gpuNum) {
    if (gpuNum < 0) {
        throw std::invalid_argument("Device-updatable kernel-node device handle requires a non-negative GPU number.");
    }
    ScopedGpu scopedGpu(gpuNum_);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNode_), sizeof(cudaGraphDeviceNode_t)));
    CUDA_CHECK(cudaMemset(deviceNode_, 0, sizeof(cudaGraphDeviceNode_t)));
}

DeviceUpdatableKernelNodeDeviceHandle::DeviceUpdatableKernelNodeDeviceHandle(DeviceUpdatableKernelNodeDeviceHandle&& other) noexcept {
    *this = std::move(other);
}

DeviceUpdatableKernelNodeDeviceHandle& DeviceUpdatableKernelNodeDeviceHandle::operator=(DeviceUpdatableKernelNodeDeviceHandle&& other) noexcept {
    if (this != &other) {
        reset();
        deviceNode_ = other.deviceNode_;
        gpuNum_ = other.gpuNum_;
        other.deviceNode_ = nullptr;
        other.gpuNum_ = -1;
    }
    return *this;
}

DeviceUpdatableKernelNodeDeviceHandle::~DeviceUpdatableKernelNodeDeviceHandle() { reset(); }

void DeviceUpdatableKernelNodeDeviceHandle::reset() noexcept {
    if (deviceNode_ != nullptr) {
        try {
            ScopedGpu scopedGpu(gpuNum_);
            (void)cudaFree(deviceNode_);
        } catch (...) {
        }
        deviceNode_ = nullptr;
    }
    gpuNum_ = -1;
}

void DeviceUpdatableKernelNodeDeviceHandle::upload(DeviceUpdatableKernelNode node, Stream stream) const {
    if (deviceNode_ == nullptr) {
        throw std::runtime_error("Cannot upload into an uninitialized device-updatable kernel-node device handle.");
    }
    if (!node) {
        throw std::invalid_argument("Cannot upload an uninitialized device-updatable kernel node handle.");
    }
    if (stream.getGpuNum() != gpuNum_) {
        throw std::invalid_argument("Device-updatable kernel-node handle upload stream must be on the handle GPU.");
    }

    cudaGraphDeviceNode_t runtimeNode = node.runtimeNode();
    ScopedGpu scopedGpu(gpuNum_);
    CUDA_CHECK(cudaMemcpyAsync(deviceNode_, &runtimeNode, sizeof(runtimeNode), cudaMemcpyHostToDevice, stream.getStream()));
}

void CudaGraphKernelLaunch::validate() const {
    if (function == nullptr) {
        throw std::invalid_argument("CUDA graph kernel launch requires a non-null CUfunction.");
    }
    if (kernelParams != nullptr && extra != nullptr) {
        throw std::invalid_argument("CUDA graph kernel launch cannot provide both kernelParams and extra.");
    }
    checkedDim("gridDim.x", gridDim.x);
    checkedDim("gridDim.y", gridDim.y);
    checkedDim("gridDim.z", gridDim.z);
    checkedDim("blockDim.x", blockDim.x);
    checkedDim("blockDim.y", blockDim.y);
    checkedDim("blockDim.z", blockDim.z);
}

CudaGraphExecutable::CudaGraphExecutable(cudaGraphExec_t graphExec, int gpuNum, bool containsDeviceUpdatableNodes)
    : graphExec_(graphExec), gpuNum_(gpuNum), containsDeviceUpdatableNodes_(containsDeviceUpdatableNodes) {}

CudaGraphExecutable::CudaGraphExecutable(CudaGraphExecutable&& other) noexcept { *this = std::move(other); }

CudaGraphExecutable& CudaGraphExecutable::operator=(CudaGraphExecutable&& other) noexcept {
    if (this != &other) {
        reset();
        graphExec_ = other.graphExec_;
        gpuNum_ = other.gpuNum_;
        containsDeviceUpdatableNodes_ = other.containsDeviceUpdatableNodes_;
        uploaded_ = other.uploaded_;
        other.graphExec_ = nullptr;
        other.gpuNum_ = -1;
        other.containsDeviceUpdatableNodes_ = false;
        other.uploaded_ = false;
    }
    return *this;
}

CudaGraphExecutable::~CudaGraphExecutable() { reset(); }

void CudaGraphExecutable::reset() noexcept {
    if (graphExec_ != nullptr) {
        (void)cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;
    }
    gpuNum_ = -1;
    containsDeviceUpdatableNodes_ = false;
    uploaded_ = false;
}

void CudaGraphExecutable::upload(Stream stream) {
    if (graphExec_ == nullptr) {
        throw std::runtime_error("Cannot upload an uninitialized CUDA graph executable.");
    }
    if (stream.getGpuNum() != gpuNum_) {
        throw std::invalid_argument("CUDA graph executable upload stream must be on the same GPU as the captured graph.");
    }
    ScopedGpu scopedGpu(stream.getGpuNum());
    CUDA_CHECK(cudaGraphUpload(graphExec_, stream.getStream()));
    uploaded_ = true;
}

void CudaGraphExecutable::launch(Stream stream) const {
    if (graphExec_ == nullptr) {
        throw std::runtime_error("Cannot launch an uninitialized CUDA graph executable.");
    }
    if (containsDeviceUpdatableNodes_ && !uploaded_) {
        throw std::runtime_error(
            "CUDA graphs containing device-updatable kernel nodes must be uploaded before launch so device-side graph updates "
            "can target the uploaded executable.");
    }
    if (stream.getGpuNum() != gpuNum_) {
        throw std::invalid_argument("CUDA graph executable launch stream must be on the same GPU as the captured graph.");
    }
    ScopedGpu scopedGpu(stream.getGpuNum());
    CUDA_CHECK(cudaGraphLaunch(graphExec_, stream.getStream()));
}

CudaGraph::CudaGraph(cudaGraph_t graph, int gpuNum, bool containsDeviceUpdatableNodes)
    : graph_(graph), gpuNum_(gpuNum), containsDeviceUpdatableNodes_(containsDeviceUpdatableNodes) {}

CudaGraph::CudaGraph(CudaGraph&& other) noexcept { *this = std::move(other); }

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = other.graph_;
        gpuNum_ = other.gpuNum_;
        containsDeviceUpdatableNodes_ = other.containsDeviceUpdatableNodes_;
        instantiated_ = other.instantiated_;
        other.graph_ = nullptr;
        other.gpuNum_ = -1;
        other.containsDeviceUpdatableNodes_ = false;
        other.instantiated_ = false;
    }
    return *this;
}

CudaGraph::~CudaGraph() { reset(); }

void CudaGraph::reset() noexcept {
    if (graph_ != nullptr) {
        (void)cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    gpuNum_ = -1;
    containsDeviceUpdatableNodes_ = false;
    instantiated_ = false;
}

CudaGraphExecutable CudaGraph::instantiate() {
    if (graph_ == nullptr) {
        throw std::runtime_error("Cannot instantiate an uninitialized CUDA graph.");
    }
    if (containsDeviceUpdatableNodes_ && instantiated_) {
        throw std::runtime_error(
            "CUDA graphs containing device-updatable kernel nodes cannot be instantiated more than once. Re-capture the graph "
            "to create another executable.");
    }

    cudaGraphExec_t graphExec = nullptr;
    ScopedGpu scopedGpu(gpuNum_);
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph_, 0));
    instantiated_ = true;
    return CudaGraphExecutable(graphExec, gpuNum_, containsDeviceUpdatableNodes_);
}

CudaGraphCaptureBuilder::CudaGraphCaptureBuilder(Stream stream, cudaStreamCaptureMode mode) : stream_(stream) {
    if (!stream_.isInitialized()) {
        throw std::invalid_argument("CUDA graph capture requires an initialized stream.");
    }
    ScopedGpu scopedGpu(stream_.getGpuNum());
    CUDA_CHECK(cudaStreamBeginCapture(stream_.getStream(), mode));
    active_ = true;
}

CudaGraphCaptureBuilder::CudaGraphCaptureBuilder(CudaGraphCaptureBuilder&& other) noexcept { *this = std::move(other); }

CudaGraphCaptureBuilder& CudaGraphCaptureBuilder::operator=(CudaGraphCaptureBuilder&& other) noexcept {
    if (this != &other) {
        abortCaptureNoThrow();
        stream_ = other.stream_;
        active_ = other.active_;
        containsDeviceUpdatableNodes_ = other.containsDeviceUpdatableNodes_;
        other.active_ = false;
        other.containsDeviceUpdatableNodes_ = false;
    }
    return *this;
}

CudaGraphCaptureBuilder::~CudaGraphCaptureBuilder() { abortCaptureNoThrow(); }

void CudaGraphCaptureBuilder::ensureActive() const {
    if (!active_) {
        throw std::runtime_error("CUDA graph capture builder is not actively capturing.");
    }
}

void CudaGraphCaptureBuilder::captureKernel(const CudaGraphKernelLaunch& launch) { (void)captureKernelImpl(launch, false); }

DeviceUpdatableKernelNode CudaGraphCaptureBuilder::captureDeviceUpdatableKernel(const CudaGraphKernelLaunch& launch) {
    return captureKernelImpl(launch, true);
}

DeviceUpdatableKernelNode CudaGraphCaptureBuilder::captureKernelImpl(const CudaGraphKernelLaunch& launch, bool deviceUpdatable) {
    ensureActive();
    launch.validate();

    ScopedGpu scopedGpu(stream_.getGpuNum());

    CUlaunchConfig config = makeLaunchConfig(launch, stream_);
    CUlaunchAttribute deviceUpdatableAttribute{};
    if (deviceUpdatable) {
        deviceUpdatableAttribute.id = CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
        deviceUpdatableAttribute.value.deviceUpdatableKernelNode.deviceUpdatable = 1;
        deviceUpdatableAttribute.value.deviceUpdatableKernelNode.devNode = nullptr;
        config.attrs = &deviceUpdatableAttribute;
        config.numAttrs = 1;
    }

    CU_CHECK(cuLaunchKernelEx(&config, launch.function, launch.kernelParams, launch.extra));

    if (!deviceUpdatable) {
        return DeviceUpdatableKernelNode();
    }

    containsDeviceUpdatableNodes_ = true;
    return DeviceUpdatableKernelNode(deviceUpdatableAttribute.value.deviceUpdatableKernelNode.devNode);
}

CudaGraph CudaGraphCaptureBuilder::endCapture() {
    ensureActive();
    ScopedGpu scopedGpu(stream_.getGpuNum());

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(stream_.getStream(), &graph));
    active_ = false;

    return CudaGraph(graph, stream_.getGpuNum(), containsDeviceUpdatableNodes_);
}

CudaGraphExecutable CudaGraphCaptureBuilder::endCaptureAndInstantiate(Stream uploadStream, bool upload) {
    CudaGraph graph = endCapture();
    CudaGraphExecutable executable = graph.instantiate();
    if (upload || executable.containsDeviceUpdatableNodes()) {
        executable.upload(uploadStream);
    }
    return executable;
}

void CudaGraphCaptureBuilder::abortCaptureNoThrow() noexcept {
    if (!active_) {
        return;
    }

    cudaGraph_t graph = nullptr;
    cudaError_t status = cudaStreamEndCapture(stream_.getStream(), &graph);
    if (status == cudaSuccess && graph != nullptr) {
        (void)cudaGraphDestroy(graph);
    }
    active_ = false;
    containsDeviceUpdatableNodes_ = false;
}

}  // namespace ThorImplementation
