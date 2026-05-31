#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"
#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

class EmbeddingErrorSink final : public Layer {
   public:
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        (void)featureInput;
        (void)validationPass;
        (void)batchSize;
        THOR_THROW_IF_FALSE(running);
    }

   private:
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)stream;
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)stream;
    }
};

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t n = 1;
    for (uint64_t d : tensor.getDimensions())
        n *= d;
    return n;
}

void writeCpuFloatTensor(Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    switch (tensor.getDataType()) {
        case DataType::FP32: {
            float* ptr = tensor.getMemPtr<float>();
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        case DataType::FP16: {
            auto* ptr = static_cast<__half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2half(values[i]);
            break;
        }
        case DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2bfloat16(values[i]);
            break;
        }
        default:
            FAIL() << "Unsupported float dtype in writeCpuFloatTensor.";
    }
}

std::vector<float> readCpuFloatTensor(const Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    std::vector<float> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case DataType::FP32: {
            const float* ptr = tensor.getMemPtr<float>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        case DataType::FP16: {
            const auto* ptr = static_cast<const __half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __half2float(ptr[i]);
            break;
        }
        case DataType::BF16: {
            const auto* ptr = static_cast<const __nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __bfloat162float(ptr[i]);
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported float dtype in readCpuFloatTensor.";
            break;
    }
    return values;
}

void writeCpuUint32Tensor(Tensor& tensor, const std::vector<uint32_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT32);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    uint32_t* ptr = tensor.getMemPtr<uint32_t>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

void writeCpuUint64Tensor(Tensor& tensor, const std::vector<uint64_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT64);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    uint64_t* ptr = tensor.getMemPtr<uint64_t>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

std::vector<uint64_t> readCpuRowTensorAsUint64(const Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    std::vector<uint64_t> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case DataType::UINT16: {
            const uint16_t* ptr = tensor.getMemPtr<uint16_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<uint64_t>(ptr[i]);
            break;
        }
        case DataType::UINT32: {
            const uint32_t* ptr = tensor.getMemPtr<uint32_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<uint64_t>(ptr[i]);
            break;
        }
        case DataType::UINT64: {
            const uint64_t* ptr = tensor.getMemPtr<uint64_t>();
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported sparse row dtype in readCpuRowTensorAsUint64.";
            break;
    }
    return values;
}

Tensor makeCpuFloatTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<float>& values) {
    Tensor tensor(cpuPlacement, TensorDescriptor(dtype, dims));
    writeCpuFloatTensor(tensor, values);
    return tensor;
}

Tensor makeCpuUint32Tensor(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& values) {
    Tensor tensor(cpuPlacement, TensorDescriptor(DataType::UINT32, dims));
    writeCpuUint32Tensor(tensor, values);
    return tensor;
}

Tensor makeCpuUint64Tensor(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& values) {
    Tensor tensor(cpuPlacement, TensorDescriptor(DataType::UINT64, dims));
    writeCpuUint64Tensor(tensor, values);
    return tensor;
}

Tensor copyCpuToGpu(Tensor& cpuTensor, Stream& stream) {
    Tensor gpuTensor(gpuPlacement, cpuTensor.getDescriptor());
    gpuTensor.copyFromAsync(cpuTensor, stream);
    stream.synchronize();
    return gpuTensor;
}

void copyCpuToExistingGpu(Tensor& gpuTensor, Tensor& cpuTensor, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(cpuTensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), cpuTensor.getDataType());
    ASSERT_EQ(gpuTensor.getDimensions(), cpuTensor.getDimensions());
    gpuTensor.copyFromAsync(cpuTensor, stream);
    stream.synchronize();
}

std::vector<float> copyGpuFloatTensorToValues(const Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    Tensor host = gpuTensor.clone(cpuPlacement);
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    return readCpuFloatTensor(host);
}

std::vector<uint64_t> copyGpuRowTensorToUint64Values(const Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    Tensor host = gpuTensor.clone(cpuPlacement);
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    return readCpuRowTensorAsUint64(host);
}

void expectAllClose(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    float atol = 1e-5f,
                    float rtol = 1e-5f,
                    const std::string& label = "") {
    ASSERT_EQ(actual.size(), expected.size());
    uint32_t printed = 0;
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << label << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
        if (diff > tol && ++printed == 10)
            return;
    }
}

float computeStep(float lr, uint32_t batchSize) { return lr / (static_cast<float>(batchSize) * Loss::getLossScalingFactor()); }

struct SgdReferenceState {
    std::vector<float> weights;
    std::vector<float> velocity;
};

void applyEmbeddingSgdReferencePass(SgdReferenceState& state,
                                    const std::vector<uint64_t>& indices,
                                    const std::vector<float>& upstreamGradient,
                                    uint64_t vocabularySize,
                                    uint64_t embeddingDim,
                                    std::optional<uint64_t> paddingIndex,
                                    uint32_t batchSize,
                                    float lr,
                                    float momentum,
                                    bool useNesterovMomentum) {
    ASSERT_EQ(state.weights.size(), vocabularySize * embeddingDim);
    ASSERT_EQ(upstreamGradient.size(), indices.size() * embeddingDim);
    if (momentum != 0.0f)
        ASSERT_EQ(state.velocity.size(), state.weights.size());

    std::vector<float> denseGrad(vocabularySize * embeddingDim, 0.0f);
    std::vector<bool> touched(vocabularySize, false);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        const uint64_t row = indices[token];
        if (row >= vocabularySize || (paddingIndex.has_value() && row == paddingIndex.value()))
            continue;

        touched[row] = true;
        for (uint64_t d = 0; d < embeddingDim; ++d)
            denseGrad[row * embeddingDim + d] += upstreamGradient[token * embeddingDim + d];
    }

    const float step = computeStep(lr, batchSize);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        if (!touched[row])
            continue;

        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t i = row * embeddingDim + d;
            const float g = denseGrad[i];
            if (momentum == 0.0f) {
                state.weights[i] -= step * g;
            } else {
                const float vNext = momentum * state.velocity[i] - step * g;
                if (useNesterovMomentum) {
                    state.weights[i] += momentum * vNext - step * g;
                } else {
                    state.weights[i] += vNext;
                }
                state.velocity[i] = vNext;
            }
        }
    }
}

std::vector<uint64_t> toUint64(const std::vector<uint32_t>& values) {
    std::vector<uint64_t> out(values.size());
    for (uint64_t i = 0; i < values.size(); ++i)
        out[i] = values[i];
    return out;
}

struct EmbeddingNetworkFixture {
    std::shared_ptr<NetworkInput> input;
    std::shared_ptr<Embedding> embedding;
    std::shared_ptr<EmbeddingErrorSink> sink;
    std::shared_ptr<PhysicalParameter> weightsParameter;
    std::shared_ptr<Sgd> optimizer;
};

EmbeddingNetworkFixture makeEmbeddingNetwork(uint64_t vocabularySize,
                                             uint64_t embeddingDim,
                                             const std::vector<uint64_t>& indexDims,
                                             DataType indexDataType,
                                             DataType weightsDataType,
                                             std::optional<uint64_t> paddingIndex,
                                             float learningRate,
                                             float decay,
                                             float momentum,
                                             bool useNesterovMomentum) {
    EmbeddingNetworkFixture f;
    f.input = std::make_shared<NetworkInput>(gpuPlacement, indexDataType, indexDims);
    f.weightsParameter =
        std::make_shared<PhysicalParameter>("weights", true, std::vector<uint64_t>{vocabularySize, embeddingDim}, weightsDataType);
    f.optimizer = std::make_shared<Sgd>(1001, learningRate, decay, momentum, useNesterovMomentum);
    f.weightsParameter->setOptimizer(f.optimizer);
    f.embedding = std::make_shared<Embedding>(gpuPlacement,
                                              std::vector<std::shared_ptr<PhysicalParameter>>{f.weightsParameter},
                                              vocabularySize,
                                              embeddingDim,
                                              weightsDataType,
                                              paddingIndex,
                                              /*sparseGradients=*/true,
                                              /*inferenceOnly=*/false);
    f.sink = std::make_shared<EmbeddingErrorSink>();

    f.input->connectToNextLayer(f.embedding.get());
    f.embedding->connectToNextLayer(f.sink.get());

    f.input->compile();
    f.embedding->compile();
    f.sink->compile();

    f.input->initialize();
    f.embedding->initialize();
    f.sink->initialize();
    return f;
}

void runEmbeddingTrainingPass(EmbeddingNetworkFixture& f,
                              Tensor& cpuIndices,
                              const std::vector<float>& upstreamGradient,
                              uint32_t batchSize) {
    f.input->forward(cpuIndices, /*validationPass=*/false, batchSize);

    std::vector<Stream> dataStreams = f.embedding->getStreams();
    ASSERT_EQ(dataStreams.size(), 1u);
    Stream dataStream = dataStreams[0];

    std::optional<Tensor> errorTensorOpt = f.sink->getErrorOutput();
    ASSERT_TRUE(errorTensorOpt.has_value());
    Tensor errorTensor = errorTensorOpt.value();
    Tensor cpuError = makeCpuFloatTensor(errorTensor.getDataType(), errorTensor.getDimensions(), upstreamGradient);
    copyCpuToExistingGpu(errorTensor, cpuError, dataStream);

    f.embedding->backward(errorTensor, batchSize);
    ASSERT_TRUE(f.embedding->getGradientUpdateStream().has_value());
    f.embedding->getGradientUpdateStream().value().synchronize();
}

}  // namespace

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerReducesDuplicatesAndSkipsPaddingAndOutOfRangeRows) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 6;
    constexpr uint64_t embeddingDim = 3;
    const std::vector<uint32_t> indices{4, 2, 4, 1, 6, 2, 0, 1};
    const std::vector<float> upstream{
        1.0f,   2.0f,   3.0f,   4.0f,  5.0f,  6.0f,  7.0f,   8.0f,   9.0f,   10.0f, 11.0f, 12.0f,
        100.0f, 100.0f, 100.0f, 13.0f, 14.0f, 15.0f, 200.0f, 200.0f, 200.0f, 16.0f, 17.0f, 18.0f,
    };

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    ASSERT_EQ(gradient.rows.getDataType(), DataType::UINT16);
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, /*paddingIndex=*/0);
    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows.size(), 1u);
    ASSERT_EQ(numRows[0], 3u);

    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 4}));

    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectAllClose(values,
                   std::vector<float>{
                       26.0f,
                       28.0f,
                       30.0f,
                       17.0f,
                       19.0f,
                       21.0f,
                       8.0f,
                       10.0f,
                       12.0f,
                   },
                   1e-5f,
                   1e-5f,
                   "sparse gradient values");
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerRleWritesDirectlyIntoRowsWithSentinelScratchSlot) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 2;
    const std::vector<uint32_t> indices{0, 1, 2, 3, 4, 1, 3};
    const std::vector<float> upstream{
        1.0f,
        10.0f,
        2.0f,
        20.0f,
        3.0f,
        30.0f,
        4.0f,
        40.0f,
        100.0f,
        100.0f,
        5.0f,
        50.0f,
        6.0f,
        60.0f,
    };

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    ASSERT_EQ(gradient.capacity, vocabularySize);
    ASSERT_EQ(gradient.rows.getTotalNumElements(), vocabularySize + 1);

    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows.size(), 1u);
    ASSERT_EQ(numRows[0], vocabularySize);

    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    ASSERT_GE(rows.size(), vocabularySize + 1);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{0, 1, 2, 3}));

    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectAllClose(values,
                   std::vector<float>{
                       1.0f,
                       10.0f,
                       7.0f,
                       70.0f,
                       3.0f,
                       30.0f,
                       10.0f,
                       100.0f,
                   },
                   1e-5f,
                   1e-5f,
                   "sparse gradient values with direct RLE rows");
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerReducesSmallAndWideFixedEmbeddingDimensions) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 3;
    const std::vector<uint32_t> indices{2, 0, 2, 1};

    for (uint64_t embeddingDim : {4ULL, 8ULL, 512ULL, 768ULL, 1024ULL, 2048ULL, 4096ULL}) {
        std::vector<float> upstream(indices.size() * embeddingDim);
        for (uint64_t token = 0; token < indices.size(); ++token) {
            const float tokenBase = token == 0 ? 1.0f : token == 1 ? 10.0f : token == 2 ? 100.0f : 1000.0f;
            for (uint64_t d = 0; d < embeddingDim; ++d) {
                upstream[token * embeddingDim + d] = tokenBase + static_cast<float>(d);
            }
        }

        Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
        Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
        Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
        Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

        SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                                 /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                                 vocabularySize,
                                                                 embeddingDim,
                                                                 DataType::FP32,
                                                                 SparseRowGradient::chooseRowDataType(vocabularySize));
        auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
        launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
        stream.synchronize();

        const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
        ASSERT_EQ(numRows.size(), 1u);
        ASSERT_EQ(numRows[0], 3u) << "embeddingDim=" << embeddingDim;

        std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
        rows.resize(numRows[0]);
        EXPECT_EQ(rows, (std::vector<uint64_t>{0, 1, 2})) << "embeddingDim=" << embeddingDim;

        std::vector<float> expected;
        expected.reserve(numRows[0] * embeddingDim);
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(10.0f + static_cast<float>(d));
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(1000.0f + static_cast<float>(d));
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(101.0f + 2.0f * static_cast<float>(d));
        }

        std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
        values.resize(expected.size());
        expectAllClose(values, expected, 1e-5f, 1e-5f, "small/wide fixed-D sparse gradient values");
    }
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerHandlesOddLargeVectorizedJitEmbeddingDimensions) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    const std::vector<uint32_t> indices{2, 1, 2, 3};

    for (uint64_t embeddingDim : {4097ULL, 16385ULL}) {
        std::vector<float> upstream(indices.size() * embeddingDim);
        for (uint64_t token = 0; token < indices.size(); ++token) {
            const float tokenBase = token == 0 ? 1.0f : token == 1 ? 10.0f : token == 2 ? 100.0f : 1000.0f;
            for (uint64_t d = 0; d < embeddingDim; ++d) {
                upstream[token * embeddingDim + d] = tokenBase + static_cast<float>(d % 97) * 0.01f;
            }
        }

        Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
        Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
        Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
        Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

        SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                                 /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                                 vocabularySize,
                                                                 embeddingDim,
                                                                 DataType::FP32,
                                                                 SparseRowGradient::chooseRowDataType(vocabularySize));
        auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
        launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
        stream.synchronize();

        const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
        ASSERT_EQ(numRows.size(), 1u);
        ASSERT_EQ(numRows[0], 3u) << "embeddingDim=" << embeddingDim;

        std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
        rows.resize(numRows[0]);
        EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3})) << "embeddingDim=" << embeddingDim;

        std::vector<float> expected;
        expected.reserve(numRows[0] * embeddingDim);
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(upstream[1ULL * embeddingDim + d]);
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(upstream[0ULL * embeddingDim + d] + upstream[2ULL * embeddingDim + d]);
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(upstream[3ULL * embeddingDim + d]);
        }

        std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
        values.resize(expected.size());
        expectAllClose(values, expected, 1e-5f, 1e-5f, "odd large-D sparse gradient values");
    }
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerBucketizesLowHighAndUltraHighRuns) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 16;
    std::vector<uint32_t> indices;
    auto appendRun = [&](uint32_t row, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            indices.push_back(row);
        }
    };
    appendRun(4, 1025);  // ultra-high bucket: count >= 1024, including a partial-tail chunk.
    appendRun(2, 16);    // low bucket: count <= 16.
    appendRun(1, 1);     // low bucket: singleton.
    appendRun(3, 17);    // high bucket: 17 <= count < 1024.

    std::vector<float> upstream(indices.size() * embeddingDim, 1.0f);
    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    EmbeddingSparseGradientProfileResult profile =
        profilePreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    EXPECT_EQ(profile.activeRows, 4u);
    EXPECT_EQ(profile.singletonRows, 1u);
    EXPECT_EQ(profile.duplicateRows, 3u);
    EXPECT_EQ(profile.lowRunRows, 2u);
    EXPECT_EQ(profile.highRunRows, 1u);
    EXPECT_EQ(profile.ultraHighRunRows, 1u);
    EXPECT_EQ(profile.lowRunTokens, 17u);
    EXPECT_EQ(profile.highRunTokens, 17u);
    EXPECT_EQ(profile.ultraHighRunTokens, 1025u);
    EXPECT_EQ(profile.maxRunCount, 1025u);

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 4u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3, 4}));

    std::vector<float> expected;
    for (float count : {1.0f, 16.0f, 17.0f, 1025.0f}) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expected.push_back(count);
        }
    }
    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(expected.size());
    expectAllClose(values, expected, 1e-5f, 1e-5f, "bucketized sparse gradient values");
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerBucketizesMoreThanOneLowRunBufferHalf) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 1300;
    constexpr uint64_t embeddingDim = 1;
    std::vector<uint32_t> indices;
    indices.reserve(vocabularySize);
    for (uint32_t row = 0; row < vocabularySize; ++row) {
        indices.push_back(row);
    }
    std::vector<float> upstream(indices.size() * embeddingDim, 1.0f);

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/vocabularySize,
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], vocabularySize);

    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    ASSERT_EQ(rows.size(), vocabularySize);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        EXPECT_EQ(rows[row], row);
    }

    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(vocabularySize * embeddingDim);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerBucketizesMoreThanOneHighRunBufferHalf) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 1100;
    constexpr uint64_t embeddingDim = 1;
    constexpr uint32_t repeatsPerRow = 17;
    std::vector<uint32_t> indices;
    indices.reserve(vocabularySize * repeatsPerRow);
    for (uint32_t row = 0; row < vocabularySize; ++row) {
        for (uint32_t i = 0; i < repeatsPerRow; ++i) {
            indices.push_back(row);
        }
    }
    std::vector<float> upstream(indices.size() * embeddingDim, 1.0f);

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/vocabularySize,
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], vocabularySize);

    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    ASSERT_EQ(rows.size(), vocabularySize);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        EXPECT_EQ(rows[row], row);
    }

    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(vocabularySize * embeddingDim);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, static_cast<float>(repeatsPerRow));
    }
}


TEST(EmbeddingSparseBackwardTest, CapturedSparseGradientTwoStageFinalizerHandlesLargeUniqueRowsAndRuntimeDisablesWhenSmall) {
    const uint32_t threshold = runtimeTwoStageEmbeddingSparseGradientFinalizeRunThreshold();
    if (threshold == 0U || threshold > 200000U) {
        GTEST_SKIP() << "two-stage embedding sparse-gradient finalizer threshold is disabled or too large for this unit test.";
    }

    constexpr int deviceNum = 0;
    Stream stream(deviceNum);

    const uint64_t vocabularySize = static_cast<uint64_t>(threshold) + 512ULL;
    const uint64_t embeddingDim = 1;
    std::vector<uint32_t> indices(static_cast<size_t>(vocabularySize));
    for (uint32_t row = 0; row < indices.size(); ++row) {
        indices[row] = row;
    }
    std::vector<float> upstream(indices.size(), 1.0f);

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/vocabularySize,
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    ASSERT_TRUE(useTwoStageEmbeddingSparseGradientFinalize(indices.size()));
    ASSERT_TRUE(preparedEmbeddingSparseGradientUsesTwoStageFinalize(*prepared));

    CapturedEmbeddingSparseGradient captured(deviceNum);
    CudaGraphCaptureBuilder builder(stream);
    capturePreparedEmbeddingSparseGradient(builder, *prepared, gpuIndices, gpuUpstream, gradient, captured);
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    captured.uploadTargetNodes(stream);

    executable.launch(stream);
    stream.synchronize();

    std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], vocabularySize);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    ASSERT_EQ(rows.size(), vocabularySize);
    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0]);
    for (uint64_t row = 0; row < vocabularySize; row += 9973ULL) {
        EXPECT_EQ(rows[row], row) << "sampled two-stage row mismatch at " << row;
        EXPECT_FLOAT_EQ(values[row], 1.0f) << "sampled two-stage value mismatch at row " << row;
    }
    EXPECT_EQ(rows.back(), vocabularySize - 1ULL);
    EXPECT_FLOAT_EQ(values.back(), 1.0f);

    // Reuse the same captured graph, but make the runtime RLE output much smaller than the two-stage threshold.
    // The first-stage finalizer must disable the two-stage classify/accumulate nodes and complete the finalize itself.
    std::fill(indices.begin(), indices.end(), 7U);
    std::fill(upstream.begin(), upstream.end(), 1.0f);
    Tensor smallCpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor smallCpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    copyCpuToExistingGpu(gpuIndices, smallCpuIndices, stream);
    copyCpuToExistingGpu(gpuUpstream, smallCpuUpstream, stream);

    executable.launch(stream);
    stream.synchronize();

    numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 1u);
    rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{7}));
    values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0]);
    EXPECT_FLOAT_EQ(values[0], static_cast<float>(indices.size()));
}

TEST(EmbeddingSparseBackwardTest, CapturedSparseGradientTwoStageFinalizerBucketizesAcrossStageBlocksAndTrimsSentinel) {
    const uint32_t threshold = runtimeTwoStageEmbeddingSparseGradientFinalizeRunThreshold();
    if (threshold == 0U || threshold > 200000U) {
        GTEST_SKIP() << "two-stage embedding sparse-gradient finalizer threshold is disabled or too large for this unit test.";
    }

    constexpr int deviceNum = 0;
    Stream stream(deviceNum);

    const uint32_t lowUniqueRows = threshold + 4U;
    const uint32_t highRow = lowUniqueRows;
    const uint32_t ultraRow = lowUniqueRows + 1U;
    const uint32_t sentinelRow = lowUniqueRows + 2U;
    const uint64_t vocabularySize = sentinelRow;
    constexpr uint64_t embeddingDim = 1;
    constexpr uint32_t highRepeats = 17U;
    constexpr uint32_t ultraRepeats = 1025U;
    constexpr uint32_t sentinelRepeats = 9U;

    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(lowUniqueRows) + highRepeats + ultraRepeats + sentinelRepeats);
    for (uint32_t row = 0; row < lowUniqueRows; ++row) {
        indices.push_back(row);
    }
    for (uint32_t i = 0; i < highRepeats; ++i) {
        indices.push_back(highRow);
    }
    for (uint32_t i = 0; i < ultraRepeats; ++i) {
        indices.push_back(ultraRow);
    }
    for (uint32_t i = 0; i < sentinelRepeats; ++i) {
        indices.push_back(sentinelRow);  // out-of-range sentinel row; finalizer must trim it.
    }

    std::vector<float> upstream(indices.size(), 1.0f);
    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    ASSERT_TRUE(useTwoStageEmbeddingSparseGradientFinalize(indices.size()));
    ASSERT_TRUE(preparedEmbeddingSparseGradientUsesTwoStageFinalize(*prepared));

    CapturedEmbeddingSparseGradient captured(deviceNum);
    CudaGraphCaptureBuilder builder(stream);
    capturePreparedEmbeddingSparseGradient(builder, *prepared, gpuIndices, gpuUpstream, gradient, captured);
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    captured.uploadTargetNodes(stream);

    executable.launch(stream);
    stream.synchronize();

    const uint64_t expectedRows = static_cast<uint64_t>(lowUniqueRows) + 2ULL;
    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], expectedRows);

    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    ASSERT_EQ(rows.size(), expectedRows);
    for (uint64_t row = 0; row < static_cast<uint64_t>(lowUniqueRows); row += 8191ULL) {
        EXPECT_EQ(rows[row], row) << "sampled low row mismatch at " << row;
    }
    EXPECT_EQ(rows[lowUniqueRows], static_cast<uint64_t>(highRow));
    EXPECT_EQ(rows[lowUniqueRows + 1ULL], static_cast<uint64_t>(ultraRow));

    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0]);
    for (uint64_t row = 0; row < static_cast<uint64_t>(lowUniqueRows); row += 8191ULL) {
        EXPECT_FLOAT_EQ(values[row], 1.0f) << "sampled low value mismatch at row " << row;
    }
    EXPECT_FLOAT_EQ(values[lowUniqueRows], static_cast<float>(highRepeats));
    EXPECT_FLOAT_EQ(values[lowUniqueRows + 1ULL], static_cast<float>(ultraRepeats));

}

TEST(EmbeddingSparseBackwardTest, CapturedFusedSparseRowUpdateUsesTwoStageFinalizerWithoutMaterializingValues) {
    const uint32_t threshold = runtimeTwoStageEmbeddingSparseGradientFinalizeRunThreshold();
    if (threshold == 0U || threshold > 200000U) {
        GTEST_SKIP() << "two-stage embedding sparse-gradient finalizer threshold is disabled or too large for this unit test.";
    }

    constexpr int deviceNum = 0;
    Stream stream(deviceNum);

    const uint64_t vocabularySize = static_cast<uint64_t>(threshold) + 256ULL;
    const uint64_t embeddingDim = 1;
    constexpr float step = 0.25f;

    std::vector<uint32_t> indices(static_cast<size_t>(vocabularySize));
    for (uint32_t row = 0; row < indices.size(); ++row) {
        indices[row] = row;
    }
    // Add duplicates to prove the captured two-stage bucket metadata still drives the fused update reducer.
    indices.push_back(3U);
    indices.push_back(3U);
    indices.push_back(11U);

    std::vector<float> upstream(indices.size(), 1.0f);
    std::vector<float> initialWeights(vocabularySize, 2.0f);

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);
    Tensor weights = copyCpuToGpu(cpuWeights, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    std::vector<float> sentinelValues(gradient.capacity * embeddingDim, -777.0f);
    Tensor cpuSentinelValues = makeCpuFloatTensor(DataType::FP32, {gradient.capacity, embeddingDim}, sentinelValues);
    copyCpuToExistingGpu(gradient.values, cpuSentinelValues, stream);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto stepScalar = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto updateOutputs = Expression::outputs({{"weights", (w - stepScalar * g).withOutputDType(DataType::FP32)}});

    std::unordered_map<std::string, SparseRowUpdateTensorBinding> updateInputs;
    updateInputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    updateInputs["gradient"] = SparseRowUpdateTensorBinding{gradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    std::unordered_map<std::string, Tensor> indexedUpdateOutputs;
    indexedUpdateOutputs["weights"] = weights;

    auto prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(gpuIndices,
                                                                      gpuUpstream,
                                                                      gradient,
                                                                      updateOutputs.physicalOutputs(),
                                                                      updateInputs,
                                                                      indexedUpdateOutputs,
                                                                      std::nullopt);
    ASSERT_TRUE(preparedEmbeddingSparseGradientHasSparseRowUpdate(*prepared));
    ASSERT_TRUE(useTwoStageEmbeddingSparseGradientFinalize(indices.size()));
    ASSERT_TRUE(preparedEmbeddingSparseGradientUsesTwoStageFinalize(*prepared));

    CapturedEmbeddingSparseGradient captured(deviceNum);
    CudaGraphCaptureBuilder builder(stream);
    capturePreparedEmbeddingSparseGradientWithSparseRowUpdate(builder,
                                                             *prepared,
                                                             gpuIndices,
                                                             gpuUpstream,
                                                             gradient,
                                                             {{"step", step}},
                                                             captured);
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);
    captured.uploadTargetNodes(stream);

    executable.launch(stream);
    stream.synchronize();

    std::vector<float> expectedWeights = initialWeights;
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        expectedWeights[row] -= step;
    }
    expectedWeights[3] -= 2.0f * step;
    expectedWeights[11] -= step;

    std::vector<float> weightsActual = copyGpuFloatTensorToValues(weights, stream);
    ASSERT_EQ(weightsActual.size(), expectedWeights.size());
    for (uint64_t row = 0; row < vocabularySize; row += 9973ULL) {
        EXPECT_FLOAT_EQ(weightsActual[row], expectedWeights[row]) << "sampled fused two-stage weight mismatch at row " << row;
    }
    EXPECT_FLOAT_EQ(weightsActual[3], expectedWeights[3]);
    EXPECT_FLOAT_EQ(weightsActual[11], expectedWeights[11]);
    EXPECT_FLOAT_EQ(weightsActual.back(), expectedWeights.back());

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], vocabularySize);
    expectAllClose(copyGpuFloatTensorToValues(gradient.values, stream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "captured fused two-stage should not materialize gradient values");
}


TEST(EmbeddingSparseBackwardTest, PreparedSparseGradientProducerReadsCurrentIndicesAcrossRelaunches) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 1;
    const std::vector<uint32_t> firstIndices{1, 1, 0, 2, 4};
    const std::vector<float> firstUpstream{2.0f, 3.0f, 100.0f, 5.0f, 7.0f};

    Tensor cpuIndices = makeCpuUint32Tensor({firstIndices.size()}, firstIndices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {firstIndices.size(), embeddingDim}, firstUpstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(firstIndices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, /*paddingIndex=*/0);

    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 4}));
    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectAllClose(values, std::vector<float>{5.0f, 5.0f, 7.0f}, 1e-5f, 1e-5f, "first prepared sparse gradient values");

    const std::vector<uint32_t> secondIndices{3, 0, 3, 3, 1};
    const std::vector<float> secondUpstream{11.0f, 100.0f, -2.0f, 4.0f, 9.0f};
    Tensor secondCpuIndices = makeCpuUint32Tensor({secondIndices.size()}, secondIndices);
    Tensor secondCpuUpstream = makeCpuFloatTensor(DataType::FP32, {secondIndices.size(), embeddingDim}, secondUpstream);
    copyCpuToExistingGpu(gpuIndices, secondCpuIndices, stream);
    copyCpuToExistingGpu(gpuUpstream, secondCpuUpstream, stream);

    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 2u);
    rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 3}));
    values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectAllClose(values, std::vector<float>{9.0f, 13.0f}, 1e-5f, 1e-5f, "second prepared sparse gradient values");
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerUsesUint32RowStorageForVocabulariesThatDoNotFitUint16) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1ULL;
    constexpr uint64_t embeddingDim = 1;
    const std::vector<uint32_t> indices{static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()), 1, 1, 7};
    const std::vector<float> upstream{2.0f, 3.0f, 5.0f, -11.0f};

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    ASSERT_EQ(gradient.rows.getDataType(), DataType::UINT32);
    auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
    launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
    stream.synchronize();

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);

    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 7, static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())}));

    std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
    values.resize(numRows[0] * embeddingDim);
    expectAllClose(values, std::vector<float>{8.0f, -11.0f, 2.0f}, 1e-5f, 1e-5f, "uint32 sparse gradient values");
}

TEST(EmbeddingSparseBackwardTest, SparseGradientProducerAccumulatesFp16AndBf16UpstreamIntoFp32Values) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 2;
    const std::vector<uint64_t> indices{3, 1, 3, 1, 2};
    const std::vector<float> upstream{
        1.5f,
        -2.0f,
        3.0f,
        4.5f,
        -0.5f,
        1.0f,
        2.0f,
        -1.5f,
        7.0f,
        -8.0f,
    };

    for (DataType upstreamDType : {DataType::FP16, DataType::BF16}) {
        Tensor cpuIndices = makeCpuUint64Tensor({indices.size()}, indices);
        Tensor cpuUpstream = makeCpuFloatTensor(upstreamDType, {indices.size(), embeddingDim}, upstream);
        Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
        Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);

        SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                                 /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                                 vocabularySize,
                                                                 embeddingDim,
                                                                 DataType::FP32,
                                                                 SparseRowGradient::chooseRowDataType(vocabularySize));
        auto prepared = prepareEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt);
        launchPreparedEmbeddingSparseGradient(*prepared, gpuIndices, gpuUpstream, gradient, stream);
        stream.synchronize();

        const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
        ASSERT_EQ(numRows[0], 3u);

        std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
        rows.resize(numRows[0]);
        EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3}));

        std::vector<float> values = copyGpuFloatTensorToValues(gradient.values, stream);
        values.resize(numRows[0] * embeddingDim);
        expectAllClose(values,
                       std::vector<float>{
                           5.0f,
                           3.0f,
                           7.0f,
                           -8.0f,
                           1.0f,
                           -1.0f,
                       },
                       upstreamDType == DataType::FP16 ? 1e-5f : 2e-2f,
                       upstreamDType == DataType::FP16 ? 1e-5f : 2e-2f,
                       upstreamDType == DataType::FP16 ? "fp16 sparse gradient values" : "bf16 sparse gradient values");
    }
}

TEST(EmbeddingSparseBackwardTest, FusedSparseRowUpdateReducesAndAppliesPlainSgdWithoutMaterializingValues) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 4096;
    constexpr float step = 0.25f;
    std::vector<uint32_t> indices;
    indices.reserve(36);
    for (uint32_t i = 0; i < 33; ++i) {
        indices.push_back(1);  // high-run bucket: count > 32.
    }
    indices.push_back(3);
    indices.push_back(0);  // padding row is skipped.
    indices.push_back(3);

    std::vector<float> upstream(indices.size() * embeddingDim);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            upstream[token * embeddingDim + d] = static_cast<float>(token * 100 + d + 1);
        }
    }

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = static_cast<float>(1000 + row * 100 + d);
        }
    }

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);
    Tensor weights = copyCpuToGpu(cpuWeights, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    std::vector<float> sentinelValues(gradient.capacity * embeddingDim, -777.0f);
    Tensor cpuSentinelValues = makeCpuFloatTensor(DataType::FP32, {gradient.capacity, embeddingDim}, sentinelValues);
    copyCpuToExistingGpu(gradient.values, cpuSentinelValues, stream);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto stepScalar = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto updateOutputs = Expression::outputs({{"weights", (w - stepScalar * g).withOutputDType(DataType::FP32)}});

    std::unordered_map<std::string, SparseRowUpdateTensorBinding> updateInputs;
    updateInputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    updateInputs["gradient"] = SparseRowUpdateTensorBinding{gradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    std::unordered_map<std::string, Tensor> indexedUpdateOutputs;
    indexedUpdateOutputs["weights"] = weights;

    auto prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(gpuIndices,
                                                                      gpuUpstream,
                                                                      gradient,
                                                                      updateOutputs.physicalOutputs(),
                                                                      updateInputs,
                                                                      indexedUpdateOutputs,
                                                                      /*paddingIndex=*/0);
    ASSERT_TRUE(preparedEmbeddingSparseGradientHasSparseRowUpdate(*prepared));

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*prepared, gpuIndices, gpuUpstream, gradient, {{"step", step}}, stream);
    stream.synchronize();

    std::vector<float> expectedWeights = initialWeights;
    auto applyToken = [&](uint64_t token) {
        const uint64_t row = indices[token];
        if (row == 0 || row >= vocabularySize) {
            return;
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expectedWeights[row * embeddingDim + d] -= step * upstream[token * embeddingDim + d];
        }
    };
    for (uint64_t token = 0; token < indices.size(); ++token) {
        applyToken(token);
    }

    expectAllClose(copyGpuFloatTensorToValues(weights, stream), expectedWeights, 2e-5f, 2e-5f, "fused sparse SGD weights");

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 2u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 3}));

    expectAllClose(copyGpuFloatTensorToValues(gradient.values, stream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "fused sparse SGD should not materialize gradient values");
}

TEST(EmbeddingSparseBackwardTest, SparseOptimizerFusionCapabilityExtendsToLargeJitDimensions) {
    EXPECT_FALSE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(0));
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(4096));
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(4097));
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(16385));
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(32768));
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(65536));      // 256 KiB FP32 row.
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(262144));     // 1 MiB FP32 row.
    EXPECT_TRUE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(1048576));    // 4 MiB FP32 row.
    EXPECT_FALSE(supportsEmbeddingSparseGradientFusedSparseRowUpdate(1048577));
}

TEST(EmbeddingSparseBackwardTest, FusedSparseRowUpdateSupportsOddLargeVectorizedJitEmbeddingDimension) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 16385;
    constexpr float step = 0.5f;
    const std::vector<uint32_t> indices{1, 2, 1, 0, 3};  // row 0 is padding and must be skipped.

    std::vector<float> upstream(indices.size() * embeddingDim);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        const float tokenBase = static_cast<float>(1 + token * 1000);
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            upstream[token * embeddingDim + d] = tokenBase + static_cast<float>(d % 103) * 0.01f;
        }
    }

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        const float rowBase = static_cast<float>(10000 + row * 1000);
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = rowBase + static_cast<float>(d % 127) * 0.01f;
        }
    }

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);
    Tensor weights = copyCpuToGpu(cpuWeights, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    std::vector<float> sentinelValues(gradient.capacity * embeddingDim, -777.0f);
    Tensor cpuSentinelValues = makeCpuFloatTensor(DataType::FP32, {gradient.capacity, embeddingDim}, sentinelValues);
    copyCpuToExistingGpu(gradient.values, cpuSentinelValues, stream);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto stepScalar = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto updateOutputs = Expression::outputs({{"weights", (w - stepScalar * g).withOutputDType(DataType::FP32)}});

    std::unordered_map<std::string, SparseRowUpdateTensorBinding> updateInputs;
    updateInputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    updateInputs["gradient"] = SparseRowUpdateTensorBinding{gradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    std::unordered_map<std::string, Tensor> indexedUpdateOutputs;
    indexedUpdateOutputs["weights"] = weights;

    auto prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(gpuIndices,
                                                                      gpuUpstream,
                                                                      gradient,
                                                                      updateOutputs.physicalOutputs(),
                                                                      updateInputs,
                                                                      indexedUpdateOutputs,
                                                                      /*paddingIndex=*/0);
    ASSERT_TRUE(preparedEmbeddingSparseGradientHasSparseRowUpdate(*prepared));

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*prepared, gpuIndices, gpuUpstream, gradient, {{"step", step}}, stream);
    stream.synchronize();

    std::vector<float> expectedWeights = initialWeights;
    for (uint64_t token = 0; token < indices.size(); ++token) {
        const uint64_t row = indices[token];
        if (row == 0 || row >= vocabularySize) {
            continue;
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expectedWeights[row * embeddingDim + d] -= step * upstream[token * embeddingDim + d];
        }
    }
    expectAllClose(copyGpuFloatTensorToValues(weights, stream), expectedWeights, 1e-5f, 1e-5f, "odd large-D fused sparse SGD weights");

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3}));

    expectAllClose(copyGpuFloatTensorToValues(gradient.values, stream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "odd large-D fused sparse SGD should not materialize gradient values");
}

TEST(EmbeddingSparseBackwardTest, FusedSparseRowUpdateSupportsVeryLargeVectorizedJitEmbeddingDimension) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 3;
    constexpr uint64_t embeddingDim = 262144;
    constexpr float step = 0.5f;
    const std::vector<uint32_t> indices{1, 2, 1, 0};  // row 0 is padding and must be skipped.

    std::vector<float> upstream(indices.size() * embeddingDim);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        const float tokenBase = static_cast<float>(1 + token * 1000);
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            upstream[token * embeddingDim + d] = tokenBase + static_cast<float>(d);
        }
    }

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        const float rowBase = static_cast<float>(10000 + row * 1000);
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = rowBase + static_cast<float>(d);
        }
    }

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);
    Tensor weights = copyCpuToGpu(cpuWeights, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    std::vector<float> sentinelValues(gradient.capacity * embeddingDim, -777.0f);
    Tensor cpuSentinelValues = makeCpuFloatTensor(DataType::FP32, {gradient.capacity, embeddingDim}, sentinelValues);
    copyCpuToExistingGpu(gradient.values, cpuSentinelValues, stream);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto stepScalar = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto updateOutputs = Expression::outputs({{"weights", (w - stepScalar * g).withOutputDType(DataType::FP32)}});

    std::unordered_map<std::string, SparseRowUpdateTensorBinding> updateInputs;
    updateInputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    updateInputs["gradient"] = SparseRowUpdateTensorBinding{gradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    std::unordered_map<std::string, Tensor> indexedUpdateOutputs;
    indexedUpdateOutputs["weights"] = weights;

    auto prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(gpuIndices,
                                                                      gpuUpstream,
                                                                      gradient,
                                                                      updateOutputs.physicalOutputs(),
                                                                      updateInputs,
                                                                      indexedUpdateOutputs,
                                                                      /*paddingIndex=*/0);
    ASSERT_TRUE(preparedEmbeddingSparseGradientHasSparseRowUpdate(*prepared));

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*prepared, gpuIndices, gpuUpstream, gradient, {{"step", step}}, stream);
    stream.synchronize();

    std::vector<float> expectedWeights = initialWeights;
    for (uint64_t token = 0; token < indices.size(); ++token) {
        const uint64_t row = indices[token];
        if (row == 0 || row >= vocabularySize) {
            continue;
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expectedWeights[row * embeddingDim + d] -= step * upstream[token * embeddingDim + d];
        }
    }
    expectAllClose(copyGpuFloatTensorToValues(weights, stream), expectedWeights, 1e-5f, 1e-5f, "large-D fused sparse SGD weights");

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 2u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2}));

    expectAllClose(copyGpuFloatTensorToValues(gradient.values, stream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "large-D fused sparse SGD should not materialize gradient values");
}

TEST(EmbeddingSparseBackwardTest, FusedSparseRowUpdateHandlesUltraHighRunsWithoutMaterializingValues) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 16;
    constexpr float step = 0.125f;
    std::vector<uint32_t> indices;
    indices.reserve(4097 + 33 + 3);
    for (uint32_t i = 0; i < 4097; ++i) {
        indices.push_back(4);  // ultra-high bucket, split into multiple partial chunks plus a tail.
    }
    for (uint32_t i = 0; i < 33; ++i) {
        indices.push_back(1);  // high-run bucket.
    }
    indices.push_back(3);
    indices.push_back(0);  // padding row is skipped.
    indices.push_back(3);

    std::vector<float> upstream(indices.size() * embeddingDim, 1.0f);
    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = static_cast<float>(1000 + row * 100 + d);
        }
    }

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);
    Tensor weights = copyCpuToGpu(cpuWeights, stream);

    SparseRowGradient gradient = SparseRowGradient::allocate(gpuPlacement,
                                                             /*capacity=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                             vocabularySize,
                                                             embeddingDim,
                                                             DataType::FP32,
                                                             SparseRowGradient::chooseRowDataType(vocabularySize));
    std::vector<float> sentinelValues(gradient.capacity * embeddingDim, -777.0f);
    Tensor cpuSentinelValues = makeCpuFloatTensor(DataType::FP32, {gradient.capacity, embeddingDim}, sentinelValues);
    copyCpuToExistingGpu(gradient.values, cpuSentinelValues, stream);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto stepScalar = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto updateOutputs = Expression::outputs({{"weights", (w - stepScalar * g).withOutputDType(DataType::FP32)}});

    std::unordered_map<std::string, SparseRowUpdateTensorBinding> updateInputs;
    updateInputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    updateInputs["gradient"] = SparseRowUpdateTensorBinding{gradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    std::unordered_map<std::string, Tensor> indexedUpdateOutputs;
    indexedUpdateOutputs["weights"] = weights;

    auto prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(gpuIndices,
                                                                      gpuUpstream,
                                                                      gradient,
                                                                      updateOutputs.physicalOutputs(),
                                                                      updateInputs,
                                                                      indexedUpdateOutputs,
                                                                      /*paddingIndex=*/0);
    ASSERT_TRUE(preparedEmbeddingSparseGradientHasSparseRowUpdate(*prepared));

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*prepared, gpuIndices, gpuUpstream, gradient, {{"step", step}}, stream);
    stream.synchronize();

    std::vector<float> expectedWeights = initialWeights;
    auto applyRun = [&](uint64_t row, float count) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            expectedWeights[row * embeddingDim + d] -= step * count;
        }
    };
    applyRun(1, 33.0f);
    applyRun(3, 2.0f);
    applyRun(4, 4097.0f);

    expectAllClose(copyGpuFloatTensorToValues(weights, stream), expectedWeights, 2e-5f, 2e-5f, "fused ultra-high sparse SGD weights");

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 3, 4}));

    expectAllClose(copyGpuFloatTensorToValues(gradient.values, stream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "fused ultra-high sparse SGD should not materialize gradient values");
}

TEST(EmbeddingSparseBackwardTest, FusedSparseRowUpdateReducesAndAppliesAdamWithoutMaterializingValues) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 16;
    constexpr uint32_t batchSize = 5;
    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.9f;
    constexpr float epsilon = 1e-5f;
    const std::vector<uint32_t> indices{2, 1, 2, 0, 3, 1};

    std::vector<float> upstream(indices.size() * embeddingDim);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            upstream[token * embeddingDim + d] = 0.01f * static_cast<float>((token + 1) * (d + 2));
        }
    }

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = static_cast<float>(1.0 + 0.1 * row + 0.01 * d);
        }
    }

    Tensor cpuIndices = makeCpuUint32Tensor({indices.size()}, indices);
    Tensor cpuUpstream = makeCpuFloatTensor(DataType::FP32, {indices.size(), embeddingDim}, upstream);
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor gpuIndices = copyCpuToGpu(cpuIndices, stream);
    Tensor gpuUpstream = copyCpuToGpu(cpuUpstream, stream);
    Tensor weights = copyCpuToGpu(cpuWeights, stream);

    Adam adam(/*id=*/2222, alpha, beta1, beta2, epsilon);
    SparseRowGradient gradient = adam.compileSparseRows(weights,
                                                        /*maxSparseRows=*/std::min<uint64_t>(indices.size(), vocabularySize),
                                                        stream);
    std::vector<float> sentinelValues(gradient.capacity * embeddingDim, -1234.0f);
    Tensor cpuSentinelValues = makeCpuFloatTensor(DataType::FP32, {gradient.capacity, embeddingDim}, sentinelValues);
    copyCpuToExistingGpu(gradient.values, cpuSentinelValues, stream);

    SparseRowOptimizerExpression updateExpression = adam.toSparseRowUpdateExpression(weights, gradient);
    auto prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(gpuIndices,
                                                                      gpuUpstream,
                                                                      gradient,
                                                                      updateExpression.outputs,
                                                                      updateExpression.inputs,
                                                                      updateExpression.indexedOutputs,
                                                                      /*paddingIndex=*/0);
    ASSERT_TRUE(preparedEmbeddingSparseGradientHasSparseRowUpdate(*prepared));

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(
        *prepared, gpuIndices, gpuUpstream, gradient, adam.sparseRowUpdateRuntimeScalars(batchSize), stream);
    stream.synchronize();

    std::vector<float> denseGrad(vocabularySize * embeddingDim, 0.0f);
    std::vector<bool> touched(vocabularySize, false);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        const uint64_t row = indices[token];
        if (row == 0 || row >= vocabularySize) {
            continue;
        }
        touched[row] = true;
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            denseGrad[row * embeddingDim + d] += upstream[token * embeddingDim + d];
        }
    }

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const float alphaT = static_cast<float>(static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), 1.0)) /
                                            (1.0 - std::pow(static_cast<double>(beta1), 1.0)));
    std::vector<float> expectedWeights = initialWeights;
    std::vector<float> expectedM(vocabularySize * embeddingDim, 0.0f);
    std::vector<float> expectedV(vocabularySize * embeddingDim, 0.0f);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        if (!touched[row]) {
            continue;
        }
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t i = row * embeddingDim + d;
            const float g = denseGrad[i] * invBatchLossScale;
            expectedM[i] = (1.0f - beta1) * g;
            expectedV[i] = (1.0f - beta2) * g * g;
            expectedWeights[i] -= alphaT * expectedM[i] / (std::sqrt(expectedV[i]) + epsilon);
        }
    }

    expectAllClose(copyGpuFloatTensorToValues(weights, stream), expectedWeights, 2e-5f, 2e-5f, "fused sparse Adam weights");
    expectAllClose(
        copyGpuFloatTensorToValues(adam.getOptimizerParameterTensor("m"), stream), expectedM, 2e-6f, 2e-5f, "fused sparse Adam m");
    expectAllClose(
        copyGpuFloatTensorToValues(adam.getOptimizerParameterTensor("v"), stream), expectedV, 2e-6f, 2e-5f, "fused sparse Adam v");

    const std::vector<uint64_t> numRows = copyGpuRowTensorToUint64Values(gradient.numRows, stream);
    ASSERT_EQ(numRows[0], 3u);
    std::vector<uint64_t> rows = copyGpuRowTensorToUint64Values(gradient.rows, stream);
    rows.resize(numRows[0]);
    EXPECT_EQ(rows, (std::vector<uint64_t>{1, 2, 3}));

    expectAllClose(copyGpuFloatTensorToValues(gradient.values, stream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "fused sparse Adam should not materialize gradient values");
    EXPECT_FLOAT_EQ(adam.getT(), 1.0f);
}

TEST(EmbeddingSparseBackwardTest, EndToEndPlainSgdUpdatesDuplicateRowsAndSkipsPaddingRow) {
    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 3;
    constexpr uint32_t batchSize = 2;
    constexpr float learningRate = 0.2f;
    const std::optional<uint64_t> paddingIndex = 0;

    EmbeddingNetworkFixture f = makeEmbeddingNetwork(vocabularySize,
                                                     embeddingDim,
                                                     /*indexDims=*/{batchSize, 4},
                                                     DataType::UINT32,
                                                     DataType::FP32,
                                                     paddingIndex,
                                                     learningRate,
                                                     /*decay=*/0.0f,
                                                     /*momentum=*/0.0f,
                                                     /*useNesterovMomentum=*/false);

    const std::vector<float> initialWeights{
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f,
        9.0f,
        10.0f,
        11.0f,
        12.0f,
        13.0f,
        14.0f,
        15.0f,
    };
    ASSERT_TRUE(f.weightsParameter->getStorage().has_value());
    Stream stream = f.embedding->getStreams()[0];
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor weights = f.weightsParameter->getStorage().value();
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    const std::vector<uint32_t> indices32{2, 0, 2, 4, 1, 2, 0, 1};
    Tensor cpuIndices = makeCpuUint32Tensor({batchSize, 4}, indices32);
    const std::vector<float> upstream{
        1.0f,  -2.0f, 3.0f,  100.0f, 100.0f, 100.0f, 4.0f,   5.0f,   -6.0f,  7.0f, -8.0f, 9.0f,
        -1.0f, 2.0f,  -3.0f, 10.0f,  -11.0f, 12.0f,  200.0f, 200.0f, 200.0f, 5.0f, -4.0f, 3.0f,
    };

    SgdReferenceState expected;
    expected.weights = initialWeights;
    applyEmbeddingSgdReferencePass(expected,
                                   toUint64(indices32),
                                   upstream,
                                   vocabularySize,
                                   embeddingDim,
                                   paddingIndex,
                                   batchSize,
                                   learningRate,
                                   /*momentum=*/0.0f,
                                   /*useNesterovMomentum=*/false);

    runEmbeddingTrainingPass(f, cpuIndices, upstream, batchSize);

    Stream gradientStream = f.embedding->getGradientUpdateStream().value();
    expectAllClose(copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 2e-5f, 2e-5f, "weights");
    EXPECT_FALSE(f.optimizer->getWeightsGradient().has_value()) << "Embedding sparse path must not allocate a dense weights gradient.";
    ASSERT_TRUE(f.optimizer->getSparseRowGradient().has_value());
    SparseRowGradient sparseGradient = f.optimizer->getSparseRowGradient().value();
    EXPECT_EQ(sparseGradient.rows.getDataType(), DataType::UINT16);
    EXPECT_EQ(copyGpuRowTensorToUint64Values(sparseGradient.numRows, gradientStream)[0], 3u);
    std::vector<uint64_t> sparseRows = copyGpuRowTensorToUint64Values(sparseGradient.rows, gradientStream);
    sparseRows.resize(3);
    EXPECT_EQ(sparseRows, (std::vector<uint64_t>{1, 2, 4}));
}

TEST(EmbeddingSparseBackwardTest, EndToEndPlainSgdFixedWidthFusesOptimizerUpdateWithoutMaterializingValues) {
    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 16;
    constexpr uint32_t batchSize = 2;
    constexpr float learningRate = 0.125f;
    const std::optional<uint64_t> paddingIndex = 0;

    EmbeddingNetworkFixture f = makeEmbeddingNetwork(vocabularySize,
                                                     embeddingDim,
                                                     /*indexDims=*/{batchSize, 3},
                                                     DataType::UINT32,
                                                     DataType::FP32,
                                                     paddingIndex,
                                                     learningRate,
                                                     /*decay=*/0.0f,
                                                     /*momentum=*/0.0f,
                                                     /*useNesterovMomentum=*/false);

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = static_cast<float>(1.0 + 0.5 * row + 0.01 * d);
        }
    }

    Stream stream = f.embedding->getStreams()[0];
    Tensor weights = f.weightsParameter->getStorage().value();
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    ASSERT_TRUE(f.optimizer->getSparseRowGradient().has_value());
    SparseRowGradient sparseGradient = f.optimizer->getSparseRowGradient().value();
    std::vector<float> sentinelValues(sparseGradient.capacity * embeddingDim, -991.0f);
    Tensor cpuSentinel = makeCpuFloatTensor(DataType::FP32, {sparseGradient.capacity, embeddingDim}, sentinelValues);
    Stream gradientStream = f.embedding->getGradientUpdateStream().value();
    copyCpuToExistingGpu(sparseGradient.values, cpuSentinel, gradientStream);

    const std::vector<uint32_t> indices{2, 0, 2, 4, 1, 2};
    std::vector<float> upstream(indices.size() * embeddingDim);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            upstream[token * embeddingDim + d] = 0.25f * static_cast<float>((token + 1) * (d + 1));
        }
    }

    SgdReferenceState expected;
    expected.weights = initialWeights;
    applyEmbeddingSgdReferencePass(expected,
                                   toUint64(indices),
                                   upstream,
                                   vocabularySize,
                                   embeddingDim,
                                   paddingIndex,
                                   batchSize,
                                   learningRate,
                                   /*momentum=*/0.0f,
                                   /*useNesterovMomentum=*/false);

    Tensor cpuIndices = makeCpuUint32Tensor({batchSize, 3}, indices);
    runEmbeddingTrainingPass(f, cpuIndices, upstream, batchSize);

    expectAllClose(copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 3e-5f, 3e-5f, "fused fixed-width SGD weights");
    expectAllClose(copyGpuFloatTensorToValues(sparseGradient.values, gradientStream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "fixed-width fused SGD path must not materialize SparseRowGradient::values");

    EXPECT_EQ(copyGpuRowTensorToUint64Values(sparseGradient.numRows, gradientStream)[0], 3u);
    std::vector<uint64_t> sparseRows = copyGpuRowTensorToUint64Values(sparseGradient.rows, gradientStream);
    sparseRows.resize(3);
    EXPECT_EQ(sparseRows, (std::vector<uint64_t>{1, 2, 4}));
}


TEST(EmbeddingSparseBackwardTest, EndToEndPlainSgdGenericWidthFusesOptimizerUpdateWithoutMaterializingValues) {
    constexpr uint64_t vocabularySize = 5;
    constexpr uint32_t batchSize = 2;
    constexpr float learningRate = 0.03125f;
    const std::optional<uint64_t> paddingIndex = 0;

    for (uint64_t embeddingDim : std::vector<uint64_t>{96ULL, 97ULL, 160ULL, 192ULL, 320ULL, 768ULL}) {
        SCOPED_TRACE(::testing::Message() << "embeddingDim=" << embeddingDim);

    EmbeddingNetworkFixture f = makeEmbeddingNetwork(vocabularySize,
                                                     embeddingDim,
                                                     /*indexDims=*/{batchSize, 3},
                                                     DataType::UINT32,
                                                     DataType::FP32,
                                                     paddingIndex,
                                                     learningRate,
                                                     /*decay=*/0.0f,
                                                     /*momentum=*/0.0f,
                                                     /*useNesterovMomentum=*/false);

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = static_cast<float>(1.0 + 0.25 * row + 0.001 * d);
        }
    }

    Stream stream = f.embedding->getStreams()[0];
    Tensor weights = f.weightsParameter->getStorage().value();
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    ASSERT_TRUE(f.optimizer->getSparseRowGradient().has_value());
    SparseRowGradient sparseGradient = f.optimizer->getSparseRowGradient().value();
    std::vector<float> sentinelValues(sparseGradient.capacity * embeddingDim, -313.0f);
    Tensor cpuSentinel = makeCpuFloatTensor(DataType::FP32, {sparseGradient.capacity, embeddingDim}, sentinelValues);
    Stream gradientStream = f.embedding->getGradientUpdateStream().value();
    copyCpuToExistingGpu(sparseGradient.values, cpuSentinel, gradientStream);

    const std::vector<uint32_t> indices{2, 0, 2, 4, 1, 2};
    std::vector<float> upstream(indices.size() * embeddingDim);
    for (uint64_t token = 0; token < indices.size(); ++token) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            upstream[token * embeddingDim + d] = 0.05f * static_cast<float>((token + 1) * ((d % 17U) + 1U));
        }
    }

    SgdReferenceState expected;
    expected.weights = initialWeights;
    applyEmbeddingSgdReferencePass(expected,
                                   toUint64(indices),
                                   upstream,
                                   vocabularySize,
                                   embeddingDim,
                                   paddingIndex,
                                   batchSize,
                                   learningRate,
                                   /*momentum=*/0.0f,
                                   /*useNesterovMomentum=*/false);

    Tensor cpuIndices = makeCpuUint32Tensor({batchSize, 3}, indices);
    runEmbeddingTrainingPass(f, cpuIndices, upstream, batchSize);

    expectAllClose(copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 5e-5f, 5e-5f, "wide fixed-width fused SGD weights");
    expectAllClose(copyGpuFloatTensorToValues(sparseGradient.values, gradientStream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "wide fixed-width fused SGD path must not materialize SparseRowGradient::values");

    EXPECT_EQ(copyGpuRowTensorToUint64Values(sparseGradient.numRows, gradientStream)[0], 3u);
    std::vector<uint64_t> sparseRows = copyGpuRowTensorToUint64Values(sparseGradient.rows, gradientStream);
    sparseRows.resize(3);
    EXPECT_EQ(sparseRows, (std::vector<uint64_t>{1, 2, 4}));
    }
}

TEST(EmbeddingSparseBackwardTest, EndToEndMomentumSgdFixedWidthFusesOptimizerUpdateAndVelocityWithoutMaterializingValues) {
    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 16;
    constexpr uint32_t batchSize = 2;
    constexpr float learningRate = 0.08f;
    constexpr float momentum = 0.65f;
    const std::optional<uint64_t> paddingIndex = 4;

    EmbeddingNetworkFixture f = makeEmbeddingNetwork(vocabularySize,
                                                     embeddingDim,
                                                     /*indexDims=*/{batchSize, 3},
                                                     DataType::UINT64,
                                                     DataType::FP32,
                                                     paddingIndex,
                                                     learningRate,
                                                     /*decay=*/0.0f,
                                                     momentum,
                                                     /*useNesterovMomentum=*/false);

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t row = 0; row < vocabularySize; ++row) {
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            initialWeights[row * embeddingDim + d] = static_cast<float>(-2.0 + row + 0.02 * d);
        }
    }

    Stream stream = f.embedding->getStreams()[0];
    Tensor weights = f.weightsParameter->getStorage().value();
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    ASSERT_TRUE(f.optimizer->getSparseRowGradient().has_value());
    SparseRowGradient sparseGradient = f.optimizer->getSparseRowGradient().value();
    std::vector<float> sentinelValues(sparseGradient.capacity * embeddingDim, 551.0f);
    Tensor cpuSentinel = makeCpuFloatTensor(DataType::FP32, {sparseGradient.capacity, embeddingDim}, sentinelValues);
    Stream gradientStream = f.embedding->getGradientUpdateStream().value();
    copyCpuToExistingGpu(sparseGradient.values, cpuSentinel, gradientStream);

    struct Pass {
        std::vector<uint64_t> indices;
        std::vector<float> upstream;
    };
    std::vector<Pass> passes;
    for (const std::vector<uint64_t>& indices : {std::vector<uint64_t>{3, 1, 3, 4, 0, 1}, std::vector<uint64_t>{2, 3, 2, 1, 4, 0}}) {
        Pass pass;
        pass.indices = indices;
        pass.upstream.resize(indices.size() * embeddingDim);
        const uint64_t passIndex = passes.size();
        for (uint64_t token = 0; token < indices.size(); ++token) {
            for (uint64_t d = 0; d < embeddingDim; ++d) {
                pass.upstream[token * embeddingDim + d] =
                    0.1f * static_cast<float>((passIndex + 1) * (token + 2)) - 0.03f * static_cast<float>(d);
            }
        }
        passes.push_back(std::move(pass));
    }

    SgdReferenceState expected;
    expected.weights = initialWeights;
    expected.velocity.assign(initialWeights.size(), 0.0f);

    for (const Pass& pass : passes) {
        Tensor cpuIndices = makeCpuUint64Tensor({batchSize, 3}, pass.indices);
        applyEmbeddingSgdReferencePass(expected,
                                       pass.indices,
                                       pass.upstream,
                                       vocabularySize,
                                       embeddingDim,
                                       paddingIndex,
                                       batchSize,
                                       learningRate,
                                       momentum,
                                       /*useNesterovMomentum=*/false);
        runEmbeddingTrainingPass(f, cpuIndices, pass.upstream, batchSize);
    }

    expectAllClose(
        copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 4e-5f, 4e-5f, "fused fixed-width momentum SGD weights");
    Tensor velocity = f.optimizer->getOptimizerParameterTensor("velocity");
    expectAllClose(
        copyGpuFloatTensorToValues(velocity, gradientStream), expected.velocity, 4e-5f, 4e-5f, "fused fixed-width momentum SGD velocity");
    expectAllClose(copyGpuFloatTensorToValues(sparseGradient.values, gradientStream),
                   sentinelValues,
                   1e-5f,
                   1e-5f,
                   "fixed-width fused momentum SGD path must not materialize SparseRowGradient::values");
}

TEST(EmbeddingSparseBackwardTest, EndToEndMomentumSgdThreePassesMatchUniqueRowReferenceAndCarryVelocity) {
    constexpr uint64_t vocabularySize = 6;
    constexpr uint64_t embeddingDim = 2;
    constexpr uint32_t batchSize = 2;
    constexpr float learningRate = 0.12f;
    constexpr float decay = 0.0f;
    constexpr float momentum = 0.7f;
    const std::optional<uint64_t> paddingIndex = 5;

    EmbeddingNetworkFixture f = makeEmbeddingNetwork(vocabularySize,
                                                     embeddingDim,
                                                     /*indexDims=*/{batchSize, 3},
                                                     DataType::UINT64,
                                                     DataType::FP32,
                                                     paddingIndex,
                                                     learningRate,
                                                     decay,
                                                     momentum,
                                                     /*useNesterovMomentum=*/false);

    const std::vector<float> initialWeights{
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        3.0f,
        -3.0f,
        4.0f,
        -4.0f,
        5.0f,
        -5.0f,
        6.0f,
        -6.0f,
    };
    ASSERT_TRUE(f.weightsParameter->getStorage().has_value());
    Stream stream = f.embedding->getStreams()[0];
    Tensor weights = f.weightsParameter->getStorage().value();
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    struct Pass {
        std::vector<uint64_t> indices;
        std::vector<float> upstream;
    };
    const std::vector<Pass> passes{
        Pass{{1, 3, 1, 5, 0, 3},
             {
                 1.0f,
                 2.0f,
                 -3.0f,
                 4.0f,
                 5.0f,
                 -6.0f,
                 100.0f,
                 100.0f,
                 -2.0f,
                 1.0f,
                 7.0f,
                 -8.0f,
             }},
        Pass{{4, 1, 4, 0, 5, 1},
             {
                 0.5f,
                 -1.5f,
                 2.5f,
                 -3.5f,
                 -4.0f,
                 6.0f,
                 1.0f,
                 -2.0f,
                 200.0f,
                 200.0f,
                 -1.0f,
                 3.0f,
             }},
        Pass{{3, 2, 3, 2, 1, 5},
             {
                 4.0f,
                 -5.0f,
                 6.0f,
                 7.0f,
                 -8.0f,
                 9.0f,
                 1.0f,
                 -2.0f,
                 3.0f,
                 -4.0f,
                 300.0f,
                 300.0f,
             }},
    };

    SgdReferenceState expected;
    expected.weights = initialWeights;
    expected.velocity.assign(initialWeights.size(), 0.0f);

    for (const Pass& pass : passes) {
        Tensor cpuIndices = makeCpuUint64Tensor({batchSize, 3}, pass.indices);
        applyEmbeddingSgdReferencePass(expected,
                                       pass.indices,
                                       pass.upstream,
                                       vocabularySize,
                                       embeddingDim,
                                       paddingIndex,
                                       batchSize,
                                       learningRate,
                                       momentum,
                                       /*useNesterovMomentum=*/false);
        runEmbeddingTrainingPass(f, cpuIndices, pass.upstream, batchSize);
    }

    Stream gradientStream = f.embedding->getGradientUpdateStream().value();
    expectAllClose(copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 3e-5f, 3e-5f, "weights");
    Tensor velocity = f.optimizer->getOptimizerParameterTensor("velocity");
    expectAllClose(copyGpuFloatTensorToValues(velocity, gradientStream), expected.velocity, 3e-5f, 3e-5f, "velocity");
}

TEST(EmbeddingSparseBackwardTest, EndToEndNesterovSgdTwoPassesMatchUniqueRowReference) {
    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 2;
    constexpr uint32_t batchSize = 1;
    constexpr float learningRate = 0.05f;
    constexpr float momentum = 0.6f;

    EmbeddingNetworkFixture f = makeEmbeddingNetwork(vocabularySize,
                                                     embeddingDim,
                                                     /*indexDims=*/{batchSize, 5},
                                                     DataType::UINT32,
                                                     DataType::FP32,
                                                     std::nullopt,
                                                     learningRate,
                                                     /*decay=*/0.0f,
                                                     momentum,
                                                     /*useNesterovMomentum=*/true);

    const std::vector<float> initialWeights{
        2.0f,
        -1.0f,
        3.0f,
        -2.0f,
        4.0f,
        -3.0f,
        5.0f,
        -4.0f,
    };
    Stream stream = f.embedding->getStreams()[0];
    Tensor weights = f.weightsParameter->getStorage().value();
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    const std::vector<std::vector<uint32_t>> indicesPasses{{0, 2, 0, 1, 2}, {3, 1, 3, 0, 1}};
    const std::vector<std::vector<float>> upstreamPasses{
        {
            1.0f,
            -2.0f,
            3.0f,
            4.0f,
            -5.0f,
            6.0f,
            7.0f,
            -8.0f,
            9.0f,
            10.0f,
        },
        {
            -1.0f,
            2.0f,
            -3.0f,
            4.0f,
            5.0f,
            -6.0f,
            7.0f,
            8.0f,
            -9.0f,
            10.0f,
        },
    };

    SgdReferenceState expected;
    expected.weights = initialWeights;
    expected.velocity.assign(initialWeights.size(), 0.0f);

    for (uint64_t pass = 0; pass < indicesPasses.size(); ++pass) {
        Tensor cpuIndices = makeCpuUint32Tensor({batchSize, 5}, indicesPasses[pass]);
        applyEmbeddingSgdReferencePass(expected,
                                       toUint64(indicesPasses[pass]),
                                       upstreamPasses[pass],
                                       vocabularySize,
                                       embeddingDim,
                                       std::nullopt,
                                       batchSize,
                                       learningRate,
                                       momentum,
                                       /*useNesterovMomentum=*/true);
        runEmbeddingTrainingPass(f, cpuIndices, upstreamPasses[pass], batchSize);
    }

    Stream gradientStream = f.embedding->getGradientUpdateStream().value();
    expectAllClose(copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 3e-5f, 3e-5f, "weights");
    Tensor velocity = f.optimizer->getOptimizerParameterTensor("velocity");
    expectAllClose(copyGpuFloatTensorToValues(velocity, gradientStream), expected.velocity, 3e-5f, 3e-5f, "velocity");
}
