#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

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

float computeStep(float lr, uint32_t batchSize) {
    return lr / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
}

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
    f.weightsParameter = std::make_shared<PhysicalParameter>("weights", true, std::vector<uint64_t>{vocabularySize, embeddingDim}, weightsDataType);
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
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
        100.0f, 100.0f, 100.0f,
        13.0f, 14.0f, 15.0f,
        200.0f, 200.0f, 200.0f,
        16.0f, 17.0f, 18.0f,
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
    launchEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, /*paddingIndex=*/0, stream);
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
                       26.0f, 28.0f, 30.0f,
                       17.0f, 19.0f, 21.0f,
                       8.0f, 10.0f, 12.0f,
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
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
        100.0f, 100.0f,
        5.0f, 50.0f,
        6.0f, 60.0f,
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

    launchEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt, stream);
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
                       1.0f, 10.0f,
                       7.0f, 70.0f,
                       3.0f, 30.0f,
                       10.0f, 100.0f,
                   },
                   1e-5f,
                   1e-5f,
                   "sparse gradient values with direct RLE rows");
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
    launchEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt, stream);
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
        1.5f, -2.0f,
        3.0f, 4.5f,
        -0.5f, 1.0f,
        2.0f, -1.5f,
        7.0f, -8.0f,
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
        launchEmbeddingSparseGradient(gpuIndices, gpuUpstream, gradient, std::nullopt, stream);
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
                           5.0f, 3.0f,
                           7.0f, -8.0f,
                           1.0f, -1.0f,
                       },
                       upstreamDType == DataType::FP16 ? 1e-5f : 2e-2f,
                       upstreamDType == DataType::FP16 ? 1e-5f : 2e-2f,
                       upstreamDType == DataType::FP16 ? "fp16 sparse gradient values" : "bf16 sparse gradient values");
    }
}

TEST(EmbeddingSparseBackwardTest, FusedSparseRowUpdateReducesAndAppliesPlainSgdWithoutMaterializingValues) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 4;
    constexpr uint64_t embeddingDim = 16;
    constexpr float step = 0.25f;
    const std::vector<uint32_t> indices{1, 3, 1, 0, 3};

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

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*prepared,
                                                             gpuIndices,
                                                             gpuUpstream,
                                                             gradient,
                                                             {{"step", step}},
                                                             stream);
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

    launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*prepared,
                                                             gpuIndices,
                                                             gpuUpstream,
                                                             gradient,
                                                             adam.sparseRowUpdateRuntimeScalars(batchSize),
                                                             stream);
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
    expectAllClose(copyGpuFloatTensorToValues(adam.getOptimizerParameterTensor("m"), stream), expectedM, 2e-6f, 2e-5f, "fused sparse Adam m");
    expectAllClose(copyGpuFloatTensorToValues(adam.getOptimizerParameterTensor("v"), stream), expectedV, 2e-6f, 2e-5f, "fused sparse Adam v");

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
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f,
    };
    ASSERT_TRUE(f.weightsParameter->getStorage().has_value());
    Stream stream = f.embedding->getStreams()[0];
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    Tensor weights = f.weightsParameter->getStorage().value();
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    const std::vector<uint32_t> indices32{2, 0, 2, 4, 1, 2, 0, 1};
    Tensor cpuIndices = makeCpuUint32Tensor({batchSize, 4}, indices32);
    const std::vector<float> upstream{
        1.0f, -2.0f, 3.0f,
        100.0f, 100.0f, 100.0f,
        4.0f, 5.0f, -6.0f,
        7.0f, -8.0f, 9.0f,
        -1.0f, 2.0f, -3.0f,
        10.0f, -11.0f, 12.0f,
        200.0f, 200.0f, 200.0f,
        5.0f, -4.0f, 3.0f,
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
    for (const std::vector<uint64_t>& indices : {std::vector<uint64_t>{3, 1, 3, 4, 0, 1},
                                                 std::vector<uint64_t>{2, 3, 2, 1, 4, 0}}) {
        Pass pass;
        pass.indices = indices;
        pass.upstream.resize(indices.size() * embeddingDim);
        const uint64_t passIndex = passes.size();
        for (uint64_t token = 0; token < indices.size(); ++token) {
            for (uint64_t d = 0; d < embeddingDim; ++d) {
                pass.upstream[token * embeddingDim + d] = 0.1f * static_cast<float>((passIndex + 1) * (token + 2)) -
                                                         0.03f * static_cast<float>(d);
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

    expectAllClose(copyGpuFloatTensorToValues(weights, gradientStream), expected.weights, 4e-5f, 4e-5f, "fused fixed-width momentum SGD weights");
    Tensor velocity = f.optimizer->getOptimizerParameterTensor("velocity");
    expectAllClose(copyGpuFloatTensorToValues(velocity, gradientStream), expected.velocity, 4e-5f, 4e-5f, "fused fixed-width momentum SGD velocity");
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
        1.0f, -1.0f,
        2.0f, -2.0f,
        3.0f, -3.0f,
        4.0f, -4.0f,
        5.0f, -5.0f,
        6.0f, -6.0f,
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
                 1.0f, 2.0f,
                 -3.0f, 4.0f,
                 5.0f, -6.0f,
                 100.0f, 100.0f,
                 -2.0f, 1.0f,
                 7.0f, -8.0f,
             }},
        Pass{{4, 1, 4, 0, 5, 1},
             {
                 0.5f, -1.5f,
                 2.5f, -3.5f,
                 -4.0f, 6.0f,
                 1.0f, -2.0f,
                 200.0f, 200.0f,
                 -1.0f, 3.0f,
             }},
        Pass{{3, 2, 3, 2, 1, 5},
             {
                 4.0f, -5.0f,
                 6.0f, 7.0f,
                 -8.0f, 9.0f,
                 1.0f, -2.0f,
                 3.0f, -4.0f,
                 300.0f, 300.0f,
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
        2.0f, -1.0f,
        3.0f, -2.0f,
        4.0f, -3.0f,
        5.0f, -4.0f,
    };
    Stream stream = f.embedding->getStreams()[0];
    Tensor weights = f.weightsParameter->getStorage().value();
    Tensor cpuWeights = makeCpuFloatTensor(DataType::FP32, {vocabularySize, embeddingDim}, initialWeights);
    copyCpuToExistingGpu(weights, cpuWeights, stream);

    const std::vector<std::vector<uint32_t>> indicesPasses{{0, 2, 0, 1, 2}, {3, 1, 3, 0, 1}};
    const std::vector<std::vector<float>> upstreamPasses{
        {
            1.0f, -2.0f,
            3.0f, 4.0f,
            -5.0f, 6.0f,
            7.0f, -8.0f,
            9.0f, 10.0f,
        },
        {
            -1.0f, 2.0f,
            -3.0f, 4.0f,
            5.0f, -6.0f,
            7.0f, 8.0f,
            -9.0f, 10.0f,
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
