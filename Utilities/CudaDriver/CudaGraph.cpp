#include "Utilities/CudaDriver/CudaGraph.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

CudaGraphExecutable::CudaGraphExecutable(cudaGraphExec_t graphExec,
                                           int gpuNum,
                                           bool containsDeviceUpdatableNodes,
                                           cudaGraph_t retainedSourceGraph)
    : graphExec_(graphExec),
      retainedSourceGraph_(retainedSourceGraph),
      gpuNum_(gpuNum),
      containsDeviceUpdatableNodes_(containsDeviceUpdatableNodes) {}

CudaGraphExecutable::CudaGraphExecutable(CudaGraphExecutable&& other) noexcept { *this = std::move(other); }

CudaGraphExecutable& CudaGraphExecutable::operator=(CudaGraphExecutable&& other) noexcept {
    if (this != &other) {
        reset();
        graphExec_ = other.graphExec_;
        retainedSourceGraph_ = other.retainedSourceGraph_;
        gpuNum_ = other.gpuNum_;
        containsDeviceUpdatableNodes_ = other.containsDeviceUpdatableNodes_;
        uploaded_ = other.uploaded_;
        other.graphExec_ = nullptr;
        other.retainedSourceGraph_ = nullptr;
        other.gpuNum_ = -1;
        other.containsDeviceUpdatableNodes_ = false;
        other.uploaded_ = false;
    }
    return *this;
}

CudaGraphExecutable::~CudaGraphExecutable() { reset(); }

void CudaGraphExecutable::reset() noexcept {
    if (graphExec_ != nullptr || retainedSourceGraph_ != nullptr) {
        try {
            ScopedGpu scopedGpu(gpuNum_);
            if (graphExec_ != nullptr) {
                cudaError_t status = cudaGraphExecDestroy(graphExec_);
                if (status != cudaSuccess) {
                    // Destructors cannot report CUDA failures, but they also must not leave a
                    // swallowed failure in CUDA's per-thread last-error slot for the next test
                    // or operation to trip over.
                    (void)cudaGetLastError();
                }
                graphExec_ = nullptr;
            }
            if (retainedSourceGraph_ != nullptr) {
                cudaError_t status = cudaGraphDestroy(retainedSourceGraph_);
                if (status != cudaSuccess) {
                    (void)cudaGetLastError();
                }
                retainedSourceGraph_ = nullptr;
            }
        } catch (...) {
            // Best effort cleanup only; this is a noexcept destructor/reset path.
            graphExec_ = nullptr;
            retainedSourceGraph_ = nullptr;
        }
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

void CudaGraphExecutable::setKernelNodeParams(cudaGraphNode_t sourceNode, const cudaKernelNodeParams& params) {
    if (graphExec_ == nullptr) {
        throw std::runtime_error("Cannot update an uninitialized CUDA graph executable.");
    }
    if (sourceNode == nullptr) {
        throw std::invalid_argument("Cannot update an uninitialized CUDA graph kernel node.");
    }
    ScopedGpu scopedGpu(gpuNum_);
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graphExec_, sourceNode, &params));
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
        try {
            ScopedGpu scopedGpu(gpuNum_);
            cudaError_t status = cudaGraphDestroy(graph_);
            if (status != cudaSuccess) {
                // Do not leak swallowed destructor/reset errors into CUDA's sticky last-error state.
                (void)cudaGetLastError();
            }
        } catch (...) {
        }
        graph_ = nullptr;
    }
    gpuNum_ = -1;
    containsDeviceUpdatableNodes_ = false;
    instantiated_ = false;
}

std::vector<cudaGraphNode_t> CudaGraph::nodes() const {
    if (graph_ == nullptr) {
        throw std::runtime_error("Cannot inspect an uninitialized CUDA graph.");
    }

    ScopedGpu scopedGpu(gpuNum_);

    size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
    std::vector<cudaGraphNode_t> out(count);
    if (count != 0) {
        CUDA_CHECK(cudaGraphGetNodes(graph_, out.data(), &count));
        out.resize(count);
    }
    return out;
}

cudaGraphNode_t CudaGraph::findSingleKernelNodeByFunction(const void* function) const {
    if (function == nullptr) {
        throw std::invalid_argument("Cannot find a CUDA graph kernel node for a null function pointer.");
    }

    ScopedGpu scopedGpu(gpuNum_);

    cudaGraphNode_t match = nullptr;
    for (cudaGraphNode_t node : nodes()) {
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) {
            continue;
        }

        cudaKernelNodeParams params{};
        cudaError_t paramsStatus = cudaGraphKernelNodeGetParams(node, &params);
        if (paramsStatus == cudaErrorInvalidDeviceFunction) {
            // CUDA can report invalid-device-function when runtime graph introspection is
            // asked to decode kernel nodes captured through driver-only launch paths, for
            // example device-updatable reducer nodes captured with cuLaunchKernelEx. Those
            // nodes cannot be the ordinary runtime kernel node we are looking for here, so
            // skip them instead of failing before reaching later inspectable nodes. Because
            // later kernel launch checks often use cudaGetLastError(), clear the swallowed
            // status from CUDA's per-thread last-error slot before continuing.
            (void)cudaGetLastError();
            continue;
        }
        CUDA_CHECK(paramsStatus);
        if (params.func != function) {
            continue;
        }
        if (match != nullptr) {
            throw std::runtime_error("CUDA graph contains multiple matching kernel nodes for the requested function.");
        }
        match = node;
    }

    if (match == nullptr) {
        throw std::runtime_error("CUDA graph does not contain a kernel node for the requested function.");
    }
    return match;
}

CudaGraphExecutable CudaGraph::instantiate(bool retainSourceGraph) {
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

    cudaGraph_t retainedSourceGraph = nullptr;
    if (retainSourceGraph) {
        retainedSourceGraph = graph_;
        graph_ = nullptr;
    }
    return CudaGraphExecutable(graphExec, gpuNum_, containsDeviceUpdatableNodes_, retainedSourceGraph);
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

void CudaGraphCaptureBuilder::captureKernel(const CudaGraphKernelLaunch& launch) { (void)captureKernelImpl(launch, false, stream_); }

void CudaGraphCaptureBuilder::captureKernelOnStream(const CudaGraphKernelLaunch& launch, Stream stream) {
    (void)captureKernelImpl(launch, false, stream);
}

DeviceUpdatableKernelNode CudaGraphCaptureBuilder::captureDeviceUpdatableKernel(const CudaGraphKernelLaunch& launch) {
    return captureKernelImpl(launch, true, stream_);
}

DeviceUpdatableKernelNode CudaGraphCaptureBuilder::captureDeviceUpdatableKernelOnStream(const CudaGraphKernelLaunch& launch,
                                                                                        Stream stream) {
    return captureKernelImpl(launch, true, stream);
}

DeviceUpdatableKernelNode CudaGraphCaptureBuilder::captureKernelImpl(const CudaGraphKernelLaunch& launch,
                                                                     bool deviceUpdatable,
                                                                     Stream stream) {
    ensureActive();
    launch.validate();
    if (!stream.isInitialized()) {
        throw std::invalid_argument("CUDA graph capture target stream must be initialized.");
    }
    if (stream.getGpuNum() != stream_.getGpuNum()) {
        throw std::invalid_argument("CUDA graph capture target stream must be on the same GPU as the captured graph.");
    }

    ScopedGpu scopedGpu(stream_.getGpuNum());

    CUlaunchConfig config = makeLaunchConfig(launch, stream);
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

    try {
        ScopedGpu scopedGpu(stream_.getGpuNum());
        cudaGraph_t graph = nullptr;
        cudaError_t status = cudaStreamEndCapture(stream_.getStream(), &graph);
        if (status == cudaSuccess && graph != nullptr) {
            cudaError_t destroyStatus = cudaGraphDestroy(graph);
            if (destroyStatus != cudaSuccess) {
                (void)cudaGetLastError();
            }
        } else if (status != cudaSuccess) {
            (void)cudaGetLastError();
        }
    } catch (...) {
    }
    active_ = false;
    containsDeviceUpdatableNodes_ = false;
}

}  // namespace ThorImplementation
