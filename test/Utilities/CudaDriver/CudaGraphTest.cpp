#include "Utilities/CudaDriver/CudaGraph.h"
#include "Utilities/CudaDriver/CudaGraphDynamicGrid.h"
#include "DeepLearning/Implementation/Layers/Optimizers/SparseRowGradient.h"
#include "Utilities/Expression/SparseRowUpdate.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/FusedEquation.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

const char* kCudaGraphTestKernels = R"CUDA(
extern "C" __global__ void write_block_x(unsigned int* out) {
    out[blockIdx.x] = blockIdx.x + 1U;
}

extern "C" __global__ void write_block_xy(unsigned int* out, unsigned int stride) {
    out[blockIdx.y * stride + blockIdx.x] = (blockIdx.y + 1U) * 100U + blockIdx.x + 1U;
}

extern "C" __global__ void write_ordinary_capture(unsigned int* out) {
    out[blockIdx.x] = 10U + blockIdx.x;
}

extern "C" __global__ void write_offset(unsigned int* out, unsigned int offset, unsigned int base) {
    out[offset + blockIdx.x] = base + blockIdx.x;
}
)CUDA";

struct TestCudaModule {
    CUmodule module = nullptr;
    CUfunction writeBlockX = nullptr;
    CUfunction writeBlockXY = nullptr;
    CUfunction writeOrdinaryCapture = nullptr;
    CUfunction writeOffset = nullptr;

    explicit TestCudaModule(int deviceNum) {
        ScopedGpu scoped(deviceNum);
        CUDA_CHECK(cudaFree(nullptr));
        EquationSignature signature = FusedEquation::buildSignature(0, deviceNum, false);
        std::vector<char> lto = EquationCompiler::compileToLtoIr(kCudaGraphTestKernels, "write_block_x", signature);
        std::vector<char> cubin = EquationCompiler::linkToCubin(lto, signature);
        CU_CHECK(cuModuleLoadData(&module, cubin.data()));
        CU_CHECK(cuModuleGetFunction(&writeBlockX, module, "write_block_x"));
        CU_CHECK(cuModuleGetFunction(&writeBlockXY, module, "write_block_xy"));
        CU_CHECK(cuModuleGetFunction(&writeOrdinaryCapture, module, "write_ordinary_capture"));
        CU_CHECK(cuModuleGetFunction(&writeOffset, module, "write_offset"));
    }

    TestCudaModule(const TestCudaModule&) = delete;
    TestCudaModule& operator=(const TestCudaModule&) = delete;

    ~TestCudaModule() {
        if (module != nullptr) {
            try {
                CU_CHECK(cuModuleUnload(module));
            } catch (...) {
            }
        }
    }
};

void writeGpuScalar(Tensor& tensor, uint64_t value, Stream stream) {
    Tensor host(cpuPlacement, tensor.getDescriptor());
    switch (tensor.getDataType()) {
        case DataType::UINT16:
            host.getMemPtr<uint16_t>()[0] = static_cast<uint16_t>(value);
            break;
        case DataType::UINT32:
            host.getMemPtr<uint32_t>()[0] = static_cast<uint32_t>(value);
            break;
        case DataType::UINT64:
            host.getMemPtr<uint64_t>()[0] = static_cast<uint64_t>(value);
            break;
        default:
            FAIL() << "Unsupported scalar dtype.";
    }
    tensor.copyFromAsync(host, stream);
}

std::vector<uint32_t> readGpuUint32Tensor(const Tensor& tensor, Stream stream) {
    Tensor host = tensor.clone(cpuPlacement);
    host.copyFromAsync(tensor, stream);
    stream.synchronize();

    std::vector<uint32_t> out(host.getTotalNumElements());
    const uint32_t* ptr = host.getMemPtr<uint32_t>();
    for (uint64_t i = 0; i < out.size(); ++i)
        out[i] = ptr[i];
    return out;
}

void expectPrefixAndZeros(const std::vector<uint32_t>& values, uint32_t prefixLength) {
    ASSERT_LE(prefixLength, values.size());
    for (uint32_t i = 0; i < prefixLength; ++i)
        EXPECT_EQ(values[i], i + 1U) << "Mismatch at active block " << i;
    for (uint64_t i = prefixLength; i < values.size(); ++i)
        EXPECT_EQ(values[i], 0U) << "Inactive block " << i << " should not have launched.";
}

Tensor makeGpuFp32Tensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("makeGpuFp32Tensor values size does not match tensor shape.");
    }
    float* hostPtr = host.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        hostPtr[i] = values[i];

    Tensor gpu(gpuPlacement, host.getDescriptor());
    gpu.copyFromAsync(host, stream);
    return gpu;
}

Tensor makeGpuUint32Tensor(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& values, Stream stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT32, dims));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("makeGpuUint32Tensor values size does not match tensor shape.");
    }
    uint32_t* hostPtr = host.getMemPtr<uint32_t>();
    for (uint64_t i = 0; i < values.size(); ++i)
        hostPtr[i] = values[i];

    Tensor gpu(gpuPlacement, host.getDescriptor());
    gpu.copyFromAsync(host, stream);
    return gpu;
}

void writeGpuFp32Tensor(Tensor& gpu, const std::vector<float>& values, Stream stream) {
    Tensor host(cpuPlacement, gpu.getDescriptor());
    ASSERT_EQ(host.getDataType(), DataType::FP32);
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("makeGpuFp32Tensor values size does not match tensor shape.");
    }
    float* hostPtr = host.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        hostPtr[i] = values[i];
    gpu.copyFromAsync(host, stream);
}

void writeGpuUint32Tensor(Tensor& gpu, const std::vector<uint32_t>& values, Stream stream) {
    Tensor host(cpuPlacement, gpu.getDescriptor());
    ASSERT_EQ(host.getDataType(), DataType::UINT32);
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("makeGpuUint32Tensor values size does not match tensor shape.");
    }
    uint32_t* hostPtr = host.getMemPtr<uint32_t>();
    for (uint64_t i = 0; i < values.size(); ++i)
        hostPtr[i] = values[i];
    gpu.copyFromAsync(host, stream);
}

void writeGpuRowTensor(Tensor& gpu, const std::vector<uint64_t>& values, Stream stream) {
    Tensor host(cpuPlacement, gpu.getDescriptor());
    ASSERT_EQ(host.getTotalNumElements(), values.size());
    switch (gpu.getDataType()) {
        case DataType::UINT16: {
            uint16_t* ptr = host.getMemPtr<uint16_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<uint16_t>(values[i]);
            break;
        }
        case DataType::UINT32: {
            uint32_t* ptr = host.getMemPtr<uint32_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<uint32_t>(values[i]);
            break;
        }
        case DataType::UINT64: {
            uint64_t* ptr = host.getMemPtr<uint64_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        default:
            FAIL() << "Unsupported row dtype.";
    }
    gpu.copyFromAsync(host, stream);
}

std::vector<float> readGpuFp32Tensor(const Tensor& gpu, Stream stream) {
    Tensor host = gpu.clone(cpuPlacement);
    host.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(host.getTotalNumElements());
    const float* ptr = host.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];
    return values;
}

std::vector<uint64_t> readGpuRowTensor(const Tensor& gpu, Stream stream) {
    Tensor host = gpu.clone(cpuPlacement);
    host.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<uint64_t> values(host.getTotalNumElements());
    switch (gpu.getDataType()) {
        case DataType::UINT16: {
            const uint16_t* ptr = host.getMemPtr<uint16_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<uint64_t>(ptr[i]);
            break;
        }
        case DataType::UINT32: {
            const uint32_t* ptr = host.getMemPtr<uint32_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<uint64_t>(ptr[i]);
            break;
        }
        case DataType::UINT64: {
            const uint64_t* ptr = host.getMemPtr<uint64_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported row dtype.";
            break;
    }
    return values;
}

void expectFloatVectorNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], atol) << "Mismatch at element " << i;
}

}  // namespace

TEST(CudaGraphTest, CapturesAndLaunchesOrdinaryDriverKernel) {
    constexpr int deviceNum = 0;
    Stream stream(deviceNum);
    TestCudaModule module(deviceNum);
    Tensor out(gpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    out.memsetAsync(stream, 0);
    stream.synchronize();

    void* outPtr = out.getMemPtr();
    void* args[] = {&outPtr};

    CudaGraphCaptureBuilder builder(stream);
    builder.captureKernel(CudaGraphKernelLaunch{module.writeOrdinaryCapture, dim3(4, 1, 1), dim3(1, 1, 1), 0, args, nullptr});
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);

    executable.launch(stream);
    std::vector<uint32_t> values = readGpuUint32Tensor(out, stream);
    EXPECT_EQ(values, (std::vector<uint32_t>{10U, 11U, 12U, 13U}));
}


TEST(CudaGraphTest, CapturesSiblingBranchesOnExplicitStreams) {
    constexpr int deviceNum = 0;
    Stream stream(deviceNum);
    Stream helperStream0(deviceNum);
    Stream helperStream1(deviceNum);
    TestCudaModule module(deviceNum);
    Tensor out(gpuPlacement, TensorDescriptor(DataType::UINT32, {9}));
    out.memsetAsync(stream, 0);
    stream.synchronize();

    void* outPtr = out.getMemPtr();
    uint32_t lowOffset = 0;
    uint32_t highOffset = 3;
    uint32_t ultraOffset = 6;
    uint32_t lowBase = 10;
    uint32_t highBase = 20;
    uint32_t ultraBase = 30;
    void* lowArgs[] = {&outPtr, &lowOffset, &lowBase};
    void* highArgs[] = {&outPtr, &highOffset, &highBase};
    void* ultraArgs[] = {&outPtr, &ultraOffset, &ultraBase};

    CudaGraphCaptureBuilder builder(stream);
    Event ready = stream.putEvent(false);
    helperStream0.waitEvent(ready);
    helperStream1.waitEvent(ready);

    builder.captureKernel(CudaGraphKernelLaunch{module.writeOffset, dim3(3, 1, 1), dim3(1, 1, 1), 0, lowArgs, nullptr});
    DeviceUpdatableKernelNode highNode = builder.captureDeviceUpdatableKernelOnStream(
        CudaGraphKernelLaunch{module.writeOffset, dim3(3, 1, 1), dim3(1, 1, 1), 0, highArgs, nullptr}, helperStream0);
    DeviceUpdatableKernelNode ultraNode = builder.captureDeviceUpdatableKernelOnStream(
        CudaGraphKernelLaunch{module.writeOffset, dim3(3, 1, 1), dim3(1, 1, 1), 0, ultraArgs, nullptr}, helperStream1);
    ASSERT_TRUE(highNode.isInitialized());
    ASSERT_TRUE(ultraNode.isInitialized());

    stream.waitEvent(helperStream0.putEvent(false));
    stream.waitEvent(helperStream1.putEvent(false));
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);

    executable.launch(stream);
    std::vector<uint32_t> values = readGpuUint32Tensor(out, stream);
    EXPECT_EQ(values, (std::vector<uint32_t>{10U, 11U, 12U, 20U, 21U, 22U, 30U, 31U, 32U}));
}

TEST(CudaGraphTest, DeviceUpdatedOneDimensionalGridUsesRuntimeDeviceScalarAcrossLaunches) {
    constexpr int deviceNum = 0;
    Stream stream(deviceNum);
    TestCudaModule module(deviceNum);
    Tensor count(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor out(gpuPlacement, TensorDescriptor(DataType::UINT32, {8}));
    DeviceUpdatableKernelNodeDeviceHandle targetHandle(deviceNum);

    writeGpuScalar(count, 5, stream);
    out.memsetAsync(stream, 0);
    stream.synchronize();

    void* outPtr = out.getMemPtr();
    void* args[] = {&outPtr};

    CudaGraphCaptureBuilder builder(stream);
    launchUpdateDeviceGrid1DFromScalar(DynamicGrid1DFromScalarDescriptor{&targetHandle, count, 1, 1, 1, 8}, stream);
    DeviceUpdatableKernelNode targetNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{module.writeBlockX, dim3(1, 1, 1), dim3(1, 1, 1), 0, args, nullptr});
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    targetHandle.upload(targetNode, stream);

    executable.launch(stream);
    expectPrefixAndZeros(readGpuUint32Tensor(out, stream), 5);

    writeGpuScalar(count, 3, stream);
    out.memsetAsync(stream, 0);
    executable.launch(stream);
    expectPrefixAndZeros(readGpuUint32Tensor(out, stream), 3);
}

TEST(CudaGraphTest, DeviceUpdatedOneDimensionalGridHonorsElementsPerCountAndBlockSize) {
    constexpr int deviceNum = 0;
    Stream stream(deviceNum);
    TestCudaModule module(deviceNum);
    Tensor count(gpuPlacement, TensorDescriptor(DataType::UINT16, {1}));
    Tensor out(gpuPlacement, TensorDescriptor(DataType::UINT32, {10}));
    DeviceUpdatableKernelNodeDeviceHandle targetHandle(deviceNum);

    writeGpuScalar(count, 5, stream);
    out.memsetAsync(stream, 0);
    stream.synchronize();

    void* outPtr = out.getMemPtr();
    void* args[] = {&outPtr};

    CudaGraphCaptureBuilder builder(stream);
    // ceil(5 * 3 / 2) = 8 blocks.
    launchUpdateDeviceGrid1DFromScalar(DynamicGrid1DFromScalarDescriptor{&targetHandle, count, 3, 2, 1, 10}, stream);
    DeviceUpdatableKernelNode targetNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{module.writeBlockX, dim3(1, 1, 1), dim3(1, 1, 1), 0, args, nullptr});
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    targetHandle.upload(targetNode, stream);

    executable.launch(stream);
    expectPrefixAndZeros(readGpuUint32Tensor(out, stream), 8);
}

TEST(CudaGraphTest, DeviceUpdatedTwoDimensionalGridUsesRuntimeDeviceScalar) {
    constexpr int deviceNum = 0;
    constexpr uint32_t stride = 4;
    Stream stream(deviceNum);
    TestCudaModule module(deviceNum);
    Tensor rows(gpuPlacement, TensorDescriptor(DataType::UINT64, {1}));
    Tensor out(gpuPlacement, TensorDescriptor(DataType::UINT32, {2 * stride}));
    DeviceUpdatableKernelNodeDeviceHandle targetHandle(deviceNum);

    writeGpuScalar(rows, 3, stream);
    out.memsetAsync(stream, 0);
    stream.synchronize();

    void* outPtr = out.getMemPtr();
    uint32_t strideArg = stride;
    void* args[] = {&outPtr, &strideArg};

    CudaGraphCaptureBuilder builder(stream);
    launchUpdateDeviceGrid2DFromScalar(DynamicGrid2DFromScalarDescriptor{&targetHandle, rows, 2, 1, 1, 4, 2}, stream);
    DeviceUpdatableKernelNode targetNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{module.writeBlockXY, dim3(1, 1, 1), dim3(1, 1, 1), 0, args, nullptr});
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    targetHandle.upload(targetNode, stream);

    executable.launch(stream);
    std::vector<uint32_t> values = readGpuUint32Tensor(out, stream);
    EXPECT_EQ(values, (std::vector<uint32_t>{101U, 102U, 103U, 0U, 201U, 202U, 203U, 0U}));
}

TEST(CudaGraphTest, DeviceUpdatableGraphsCannotBeInstantiatedTwice) {
    constexpr int deviceNum = 0;
    Stream stream(deviceNum);
    TestCudaModule module(deviceNum);
    Tensor out(gpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    void* outPtr = out.getMemPtr();
    void* args[] = {&outPtr};

    CudaGraphCaptureBuilder builder(stream);
    DeviceUpdatableKernelNode targetNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{module.writeBlockX, dim3(1, 1, 1), dim3(1, 1, 1), 0, args, nullptr});
    ASSERT_TRUE(targetNode.isInitialized());
    CudaGraph graph = builder.endCapture();

    CudaGraphExecutable executable = graph.instantiate();
    EXPECT_TRUE(executable.containsDeviceUpdatableNodes());
    EXPECT_THROW((void)graph.instantiate(), std::runtime_error);
}

TEST(CudaGraphTest, CapturedSparseRowUpdateUsesRuntimeNumRowsAcrossGraphLaunches) {
    constexpr int deviceNum = 0;
    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 1;
    constexpr uint64_t capacity = 4;
    Stream stream(deviceNum);

    Tensor rows(gpuPlacement, TensorDescriptor(DataType::UINT16, {capacity}));
    Tensor numRows(gpuPlacement, TensorDescriptor(DataType::UINT16, {1}));
    Tensor gradient = makeGpuFp32Tensor({capacity, embeddingDim}, {10.0f, 20.0f, 30.0f, 40.0f}, stream);
    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {vocabularySize, embeddingDim}));

    writeGpuRowTensor(rows, {2, 0, 4, 3}, stream);
    writeGpuRowTensor(numRows, {2}, stream);
    weights.memsetAsync(stream, 0);
    stream.synchronize();

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto step = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto outputs = Expression::outputs({{"weights", (w - step * g).withOutputDType(DataType::FP32)}});

    std::unordered_map<std::string, SparseRowUpdateTensorBinding> inputs;
    inputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    inputs["gradient"] = SparseRowUpdateTensorBinding{gradient, SparseRowUpdateTensorKind::DenseLogicalRows};
    std::unordered_map<std::string, Tensor> indexedOutputs;
    indexedOutputs["weights"] = weights;

    std::unique_ptr<SparseRowUpdatePlan> plan = SparseRowUpdatePlan::compile(outputs.physicalOutputs(), rows, numRows, inputs, indexedOutputs, deviceNum);

    CapturedSparseRowUpdate captured(deviceNum);
    CudaGraphCaptureBuilder builder(stream);
    plan->capture(builder, captured, {{"step", 1.0f}});
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    captured.uploadTargetNode(stream);

    executable.launch(stream);
    expectFloatVectorNear(readGpuFp32Tensor(weights, stream), { -20.0f, 0.0f, -10.0f, 0.0f, 0.0f });

    writeGpuRowTensor(numRows, {4}, stream);
    weights.memsetAsync(stream, 0);
    executable.launch(stream);
    expectFloatVectorNear(readGpuFp32Tensor(weights, stream), { -20.0f, 0.0f, -10.0f, -40.0f, -30.0f });
}

TEST(CudaGraphTest, CapturedEmbeddingSparseGradientUsesRuntimeReducedRowCountAcrossGraphLaunches) {
    constexpr int deviceNum = 0;
    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 1;
    Stream stream(deviceNum);

    Tensor indices = makeGpuUint32Tensor({6}, {3, 1, 3, 0, 2, 3}, stream);
    Tensor upstream = makeGpuFp32Tensor({6, embeddingDim}, {1.0f, 2.0f, 4.0f, 100.0f, 5.0f, -2.0f}, stream);
    stream.synchronize();

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/4,
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    ASSERT_EQ(gradient.rows.getDataType(), DataType::UINT16);
    auto prepared = prepareEmbeddingSparseGradient(indices, upstream, gradient, /*paddingIndex=*/0);

    CapturedEmbeddingSparseGradient captured(deviceNum);
    CudaGraphCaptureBuilder builder(stream);
    capturePreparedEmbeddingSparseGradient(builder, *prepared, indices, upstream, gradient, captured);
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    captured.uploadTargetNodes(stream);

    executable.launch(stream);
    std::vector<uint64_t> numRows = readGpuRowTensor(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);
    std::vector<uint64_t> rows = readGpuRowTensor(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3}));
    std::vector<float> values = readGpuFp32Tensor(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectFloatVectorNear(values, {2.0f, 5.0f, 3.0f});

    // Re-capture is intentionally not used here; update the same captured buffers to change the runtime number of valid rows.
    writeGpuUint32Tensor(indices, {1, 1, 0, 0, 0, 0}, stream);
    writeGpuFp32Tensor(upstream, {7.0f, 5.0f, 100.0f, 100.0f, 100.0f, 100.0f}, stream);
    executable.launch(stream);

    numRows = readGpuRowTensor(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 1u);
    rows = readGpuRowTensor(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1}));
    values = readGpuFp32Tensor(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectFloatVectorNear(values, {12.0f});

    // Expand again without recapture; the fused finalize+grid-update node must resize the reduce grid upward too.
    writeGpuUint32Tensor(indices, {3, 2, 1, 0, 3, 2}, stream);
    writeGpuFp32Tensor(upstream, {4.0f, 5.0f, 6.0f, 100.0f, -1.0f, 7.0f}, stream);
    executable.launch(stream);

    numRows = readGpuRowTensor(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);
    rows = readGpuRowTensor(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3}));
    values = readGpuFp32Tensor(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectFloatVectorNear(values, {6.0f, 12.0f, 3.0f});
}
