#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Add.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include <cuda_bf16.h>
#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t d : tensor.getDimensions())
        numel *= d;
    return numel;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

void writeCpuTensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    switch (tensor.getDataType()) {
        case DataType::FP16: {
            auto* ptr = static_cast<half*>(tensor.getMemPtr());
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
        case DataType::FP32: {
            auto* ptr = static_cast<float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        default:
            FAIL() << "Unsupported tensor dtype in writeCpuTensor.";
    }
}

void writeCpuInt32Tensor(Impl::Tensor& tensor, const vector<int32_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::INT32);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    auto* ptr = static_cast<int32_t*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

void writeCpuUint32Tensor(Impl::Tensor& tensor, const vector<uint32_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT32);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    auto* ptr = static_cast<uint32_t*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

void writeCpuUint64Tensor(Impl::Tensor& tensor, const vector<uint64_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT64);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    auto* ptr = static_cast<uint64_t*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    vector<float> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case DataType::FP16: {
            const auto* ptr = static_cast<const half*>(tensor.getMemPtr());
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
        case DataType::FP32: {
            const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported tensor dtype in readCpuTensor.";
            break;
    }
    return values;
}

Impl::Tensor copyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

vector<uint8_t> readTensorBytes(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = copyTensorToCpu(tensor, stream);
    vector<uint8_t> bytes(cpuTensor.getArraySizeInBytes());
    std::memcpy(bytes.data(), cpuTensor.getMemPtr(), bytes.size());
    return bytes;
}

void expectAllClose(const vector<float>& actual, const vector<float>& expected, float atol = 6e-2f, float rtol = 6e-2f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << "mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

void expectNotAllClose(const vector<float>& lhs, const vector<float>& rhs, float atol = 6e-2f, float rtol = 6e-2f) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (uint64_t i = 0; i < lhs.size(); ++i) {
        const float diff = fabs(lhs[i] - rhs[i]);
        const float tol = atol + rtol * fabs(rhs[i]);
        if (diff > tol) {
            return;
        }
    }
    FAIL() << "sentinel references are too close; this test would not catch the intended feature mismatch";
}

float castToStorage(float value, DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
            return __half2float(__float2half(value));
        case DataType::BF16:
            return __bfloat162float(__float2bfloat16(value));
        case DataType::FP32:
            return value;
        default:
            throw std::runtime_error("Unsupported dtype in castToStorage.");
    }
}

vector<float> castVectorToStorage(vector<float> values, DataType dataType) {
    for (float& value : values)
        value = castToStorage(value, dataType);
    return values;
}

void setParameterTensor(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor deviceTensor = parameter->getStorage().value();
    Impl::Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

struct PlacedAttentionFixture {
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    Impl::StampedNetwork* stampedNetwork = nullptr;
    shared_ptr<Impl::NetworkInput> physicalInput;
    shared_ptr<Impl::NetworkOutput> physicalOutput;
    shared_ptr<Impl::CustomLayer> physicalAttention;
};

PlacedAttentionFixture placeSingleAttentionNetwork(Api::Network& network,
                                                   const Api::NetworkInput& apiInput,
                                                   const Api::NetworkOutput& apiOutput,
                                                   const Api::Attention& apiAttention,
                                                   uint32_t batchSize,
                                                   bool inferenceOnly) {
    vector<Event> initDoneEvents;
    PlacedAttentionFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, inferenceOnly);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    fixture.stampedNetwork = &fixture.placedNetwork->getStampedNetwork(0);

    fixture.physicalInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiInput.getId()));
    fixture.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalAttention =
        dynamic_pointer_cast<Impl::CustomLayer>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiAttention.getId()));

    EXPECT_NE(fixture.physicalInput, nullptr);
    EXPECT_NE(fixture.physicalOutput, nullptr);
    EXPECT_NE(fixture.physicalAttention, nullptr);
    return fixture;
}

vector<float> runForward(Impl::NetworkInput& physicalInput,
                         Impl::NetworkOutput& physicalOutput,
                         Impl::Tensor& featureInHost,
                         uint32_t batchSize,
                         bool validationPass = false) {
    physicalInput.forward(featureInHost, validationPass, batchSize);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
}

std::filesystem::path makeUniqueTestArchiveDir(const std::string& testName) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() / (testName + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

template <typename LayerT>
std::shared_ptr<LayerT> findOnlyLayerOfType(Api::Network& network) {
    std::shared_ptr<LayerT> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        std::shared_ptr<LayerT> candidate = std::dynamic_pointer_cast<LayerT>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1u);
    return found;
}

vector<float> scaledIdentity(uint32_t width, float scale) {
    vector<float> values(static_cast<uint64_t>(width) * width, 0.0f);
    for (uint32_t i = 0; i < width; ++i) {
        values[static_cast<uint64_t>(i) * width + i] = scale;
    }
    return values;
}

vector<float> runForwardWithMetadata(Impl::NetworkInput& physicalInput,
                                     Impl::NetworkInput& physicalSequenceLengthsInput,
                                     Impl::NetworkOutput& physicalOutput,
                                     Impl::Tensor& featureInHost,
                                     Impl::Tensor& sequenceLengthsHost,
                                     uint32_t batchSize) {
    physicalInput.forward(featureInHost, false, batchSize);
    physicalSequenceLengthsInput.forward(sequenceLengthsHost, false, batchSize);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
}

void forwardPhysicalRowPartitionOffsets(Impl::NetworkInput& physicalRaggedOffsetsInput,
                                        Impl::Tensor& raggedOffsetsHost,
                                        uint32_t batchSize,
                                        uint64_t maxTotalValues,
                                        uint64_t activeRows) {
    if (!physicalRaggedOffsetsInput.getFeatureOutput().has_value()) {
        throw std::runtime_error("Ragged attention direct-physical test requires a placed offsets input.");
    }
    Impl::Tensor offsets = physicalRaggedOffsetsInput.getFeatureOutput().value();
    if (!Impl::RowPartitionDescriptor::isValidOffsetsDataType(offsets.getDataType()) ||
        raggedOffsetsHost.getDataType() != offsets.getDataType()) {
        throw std::runtime_error("Ragged attention direct-physical test requires matching canonical UINT32/UINT64 offsets.");
    }
    physicalRaggedOffsetsInput.forwardRowPartitionOffsets(
        raggedOffsetsHost,
        /*validationPass=*/false,
        Impl::RowPartitionDescriptor(batchSize, maxTotalValues, offsets.getDataType()),
        activeRows,
        batchSize);
}

vector<float> runForwardWithRaggedRowPartitionRuntime(Impl::NetworkInput& physicalInput,
                                           Impl::NetworkInput& physicalRaggedOffsetsInput,
                                           Impl::NetworkOutput& physicalOutput,
                                           Impl::Tensor& featureInHost,
                                           Impl::Tensor& raggedOffsetsHost,
                                           uint32_t batchSize) {
    if (!physicalInput.getFeatureOutput().has_value() || !physicalRaggedOffsetsInput.getFeatureOutput().has_value()) {
        throw std::runtime_error("Ragged attention direct-physical helper requires placed values and offsets inputs.");
    }

    uint64_t activeRows = 0;
    if (raggedOffsetsHost.getDataType() == DataType::UINT32) {
        activeRows = raggedOffsetsHost.getMemPtr<uint32_t>()[batchSize];
    } else if (raggedOffsetsHost.getDataType() == DataType::UINT64) {
        activeRows = raggedOffsetsHost.getMemPtr<uint64_t>()[batchSize];
    } else {
        throw std::runtime_error("Ragged attention test requires canonical UINT32/UINT64 offsets.");
    }

    const uint64_t maxTotalValues = physicalInput.getFeatureOutput()->getDimensions().front();

    physicalInput.forwardRaggedValues(featureInHost, false, activeRows, batchSize);
    forwardPhysicalRowPartitionOffsets(
        physicalRaggedOffsetsInput, raggedOffsetsHost, batchSize, maxTotalValues, activeRows);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
}

uint64_t idx3(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t dim1, uint32_t dim2) {
    return (static_cast<uint64_t>(d0) * dim1 + d1) * dim2 + d2;
}

uint64_t idx4(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t dim1, uint32_t dim2, uint32_t dim3) {
    return ((static_cast<uint64_t>(d0) * dim1 + d1) * dim2 + d2) * dim3 + d3;
}

uint64_t idxBshd(uint32_t b, uint32_t s, uint32_t h, uint32_t d, uint32_t sequenceLength, uint32_t heads, uint32_t dim) {
    return ((static_cast<uint64_t>(b) * sequenceLength + s) * heads + h) * dim + d;
}

struct AttentionReferenceCase {
    uint32_t batchSize;
    uint32_t sequenceLength;
    uint32_t inputFeatures;
    uint32_t outputFeatures;
    uint32_t numHeads;
    uint32_t numKeyValueHeads;
    uint32_t headDim;
    uint32_t valueDim;
    bool hasBias = false;
    bool useRope = false;
    Impl::RotaryPositionEmbeddingOptions ropeOptions;
    optional<int64_t> queryRopePositionOffset;
    optional<int64_t> keyRopePositionOffset;
    Impl::AttentionMaskKind maskKind = Impl::AttentionMaskKind::None;
    int64_t diagonalLeftBound = 0;
    int64_t diagonalRightBound = 0;
    bool useAlibiMask = false;
    float attentionScale = 1.0f;
    vector<int32_t> sequenceLengths;
    DataType dataType = DataType::FP16;
};

struct AttentionReferenceInputs {
    vector<float> featureInput;
    vector<float> queryWeights;
    vector<float> keyWeights;
    vector<float> valueWeights;
    vector<float> outputWeights;
    vector<float> queryBias;
    vector<float> keyBias;
    vector<float> valueBias;
    vector<float> outputBias;
};

uint32_t qWidth(const AttentionReferenceCase& c) { return c.numHeads * c.headDim; }
uint32_t kWidth(const AttentionReferenceCase& c) { return c.numKeyValueHeads * c.headDim; }
uint32_t vWidth(const AttentionReferenceCase& c) { return c.numKeyValueHeads * c.valueDim; }
uint32_t qkvWidth(const AttentionReferenceCase& c) { return qWidth(c) + kWidth(c) + vWidth(c); }
uint32_t mergedWidth(const AttentionReferenceCase& c) { return c.numHeads * c.valueDim; }

uint32_t effectiveSequenceLength(const AttentionReferenceCase& c, uint32_t batch) {
    if (c.sequenceLengths.empty())
        return c.sequenceLength;
    return static_cast<uint32_t>(c.sequenceLengths.at(batch));
}

constexpr bool attentionUsesPackedQkv(bool useRope) {
    if constexpr (!Api::Attention::USE_PACKED_QKV_PROJECTION) {
        return false;
    } else {
        return !useRope;
    }
}

bool attentionUsesPackedQkv(const AttentionReferenceCase& c) { return attentionUsesPackedQkv(c.useRope); }

vector<float> packQkvWeights(const AttentionReferenceInputs& inputs, const AttentionReferenceCase& c) {
    vector<float> qkv(static_cast<uint64_t>(c.inputFeatures) * qkvWidth(c), 0.0f);
    for (uint32_t f = 0; f < c.inputFeatures; ++f) {
        const uint64_t packedRow = static_cast<uint64_t>(f) * qkvWidth(c);
        const uint64_t qRow = static_cast<uint64_t>(f) * qWidth(c);
        const uint64_t kRow = static_cast<uint64_t>(f) * kWidth(c);
        const uint64_t vRow = static_cast<uint64_t>(f) * vWidth(c);
        std::copy(inputs.queryWeights.begin() + qRow, inputs.queryWeights.begin() + qRow + qWidth(c), qkv.begin() + packedRow);
        std::copy(inputs.keyWeights.begin() + kRow, inputs.keyWeights.begin() + kRow + kWidth(c), qkv.begin() + packedRow + qWidth(c));
        std::copy(inputs.valueWeights.begin() + vRow,
                  inputs.valueWeights.begin() + vRow + vWidth(c),
                  qkv.begin() + packedRow + qWidth(c) + kWidth(c));
    }
    return qkv;
}

vector<float> packQkvBias(const AttentionReferenceInputs& inputs, const AttentionReferenceCase& c) {
    vector<float> qkv(qkvWidth(c), 0.0f);
    std::copy(inputs.queryBias.begin(), inputs.queryBias.end(), qkv.begin());
    std::copy(inputs.keyBias.begin(), inputs.keyBias.end(), qkv.begin() + qWidth(c));
    std::copy(inputs.valueBias.begin(), inputs.valueBias.end(), qkv.begin() + qWidth(c) + kWidth(c));
    return qkv;
}

vector<float> makePatternVector(uint64_t count, float scale, int64_t a, int64_t b, int64_t modulus) {
    vector<float> values(count, 0.0f);
    for (uint64_t i = 0; i < count; ++i) {
        const int64_t centered = static_cast<int64_t>((a * static_cast<int64_t>(i) + b) % modulus) - (modulus / 2);
        values[i] = scale * static_cast<float>(centered);
    }
    return values;
}

AttentionReferenceInputs makeAttentionReferenceInputs(const AttentionReferenceCase& c) {
    AttentionReferenceInputs inputs;
    inputs.featureInput.resize(static_cast<uint64_t>(c.batchSize) * c.sequenceLength * c.inputFeatures, 0.0f);
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t s = 0; s < c.sequenceLength; ++s) {
            for (uint32_t f = 0; f < c.inputFeatures; ++f) {
                const float signedFeature = static_cast<float>(static_cast<int32_t>(f % 9) - 4);
                inputs.featureInput[idx3(b, s, f, c.sequenceLength, c.inputFeatures)] =
                    0.18f + 0.071f * static_cast<float>(b) + 0.093f * static_cast<float>(s) + 0.011f * signedFeature;
            }
        }
    }

    inputs.queryWeights = makePatternVector(static_cast<uint64_t>(c.inputFeatures) * qWidth(c), 0.010f, 37, 5, 17);
    inputs.keyWeights = makePatternVector(static_cast<uint64_t>(c.inputFeatures) * kWidth(c), 0.011f, 29, 3, 19);
    inputs.valueWeights = makePatternVector(static_cast<uint64_t>(c.inputFeatures) * vWidth(c), 0.013f, 31, 7, 23);
    inputs.outputWeights = makePatternVector(static_cast<uint64_t>(mergedWidth(c)) * c.outputFeatures, 0.009f, 41, 11, 29);

    // Add identity-like diagonals on top of the deterministic background.  This keeps the reference numerically
    // well-conditioned and makes head/sequence mixups produce large, easy-to-debug differences.
    for (uint32_t h = 0; h < c.numHeads; ++h) {
        for (uint32_t d = 0; d < c.headDim; ++d) {
            const uint32_t inFeature = (h * c.valueDim + d) % c.inputFeatures;
            inputs.queryWeights[static_cast<uint64_t>(inFeature) * qWidth(c) + h * c.headDim + d] += 0.19f + 0.003f * d;
        }
    }
    for (uint32_t kvh = 0; kvh < c.numKeyValueHeads; ++kvh) {
        const uint32_t firstQueryHeadForKv = kvh * (c.numHeads / c.numKeyValueHeads);
        for (uint32_t d = 0; d < c.headDim; ++d) {
            const uint32_t inFeature = (firstQueryHeadForKv * c.valueDim + d) % c.inputFeatures;
            inputs.keyWeights[static_cast<uint64_t>(inFeature) * kWidth(c) + kvh * c.headDim + d] += 0.17f + 0.002f * d;
        }
        for (uint32_t d = 0; d < c.valueDim; ++d) {
            const uint32_t inFeature = (firstQueryHeadForKv * c.valueDim + d) % c.inputFeatures;
            inputs.valueWeights[static_cast<uint64_t>(inFeature) * vWidth(c) + kvh * c.valueDim + d] += 0.82f;
        }
    }
    for (uint32_t i = 0; i < std::min<uint32_t>(mergedWidth(c), c.outputFeatures); ++i)
        inputs.outputWeights[static_cast<uint64_t>(i) * c.outputFeatures + i] += 0.74f;

    if (c.hasBias) {
        inputs.queryBias = makePatternVector(qWidth(c), 0.006f, 5, 1, 13);
        inputs.keyBias = makePatternVector(kWidth(c), 0.005f, 7, 2, 11);
        inputs.valueBias = makePatternVector(vWidth(c), 0.007f, 11, 3, 17);
        inputs.outputBias = makePatternVector(c.outputFeatures, 0.008f, 13, 4, 19);
    }

    inputs.featureInput = castVectorToStorage(std::move(inputs.featureInput), c.dataType);
    inputs.queryWeights = castVectorToStorage(std::move(inputs.queryWeights), c.dataType);
    inputs.keyWeights = castVectorToStorage(std::move(inputs.keyWeights), c.dataType);
    inputs.valueWeights = castVectorToStorage(std::move(inputs.valueWeights), c.dataType);
    inputs.outputWeights = castVectorToStorage(std::move(inputs.outputWeights), c.dataType);
    if (c.hasBias) {
        inputs.queryBias = castVectorToStorage(std::move(inputs.queryBias), c.dataType);
        inputs.keyBias = castVectorToStorage(std::move(inputs.keyBias), c.dataType);
        inputs.valueBias = castVectorToStorage(std::move(inputs.valueBias), c.dataType);
        inputs.outputBias = castVectorToStorage(std::move(inputs.outputBias), c.dataType);
    }
    return inputs;
}

AttentionReferenceInputs makeRopeLayoutSentinelInputs(const AttentionReferenceCase& c) {
    if (c.numHeads != c.numKeyValueHeads || c.headDim != c.valueDim || c.inputFeatures != qWidth(c) || c.outputFeatures != mergedWidth(c) ||
        c.hasBias) {
        throw std::runtime_error("RoPE layout sentinel input helper expects bias-free MHA with identity-sized projections.");
    }

    AttentionReferenceInputs inputs;
    inputs.featureInput.resize(static_cast<uint64_t>(c.batchSize) * c.sequenceLength * c.inputFeatures, 0.0f);
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t s = 0; s < c.sequenceLength; ++s) {
            for (uint32_t h = 0; h < c.numHeads; ++h) {
                for (uint32_t d = 0; d < c.headDim; ++d) {
                    float value = 0.20f * static_cast<float>(b + 1) + 0.70f * static_cast<float>(s + 1) +
                                  0.11f * static_cast<float>(h + 1) + 0.03f * static_cast<float>(d + 1);
                    if ((d & 1U) != 0U) {
                        value *= -0.80f;
                    }
                    inputs.featureInput[idx3(b, s, h * c.headDim + d, c.sequenceLength, c.inputFeatures)] = value;
                }
            }
        }
    }

    inputs.queryWeights.resize(static_cast<uint64_t>(c.inputFeatures) * qWidth(c), 0.0f);
    inputs.keyWeights.resize(static_cast<uint64_t>(c.inputFeatures) * kWidth(c), 0.0f);
    inputs.valueWeights.resize(static_cast<uint64_t>(c.inputFeatures) * vWidth(c), 0.0f);
    inputs.outputWeights.resize(static_cast<uint64_t>(mergedWidth(c)) * c.outputFeatures, 0.0f);
    for (uint32_t i = 0; i < c.inputFeatures; ++i) {
        inputs.queryWeights[static_cast<uint64_t>(i) * qWidth(c) + i] = 1.0f;
        inputs.keyWeights[static_cast<uint64_t>(i) * kWidth(c) + i] = 1.0f;
        inputs.valueWeights[static_cast<uint64_t>(i) * vWidth(c) + i] = 1.0f;
        inputs.outputWeights[static_cast<uint64_t>(i) * c.outputFeatures + i] = 1.0f;
    }

    inputs.featureInput = castVectorToStorage(std::move(inputs.featureInput), c.dataType);
    inputs.queryWeights = castVectorToStorage(std::move(inputs.queryWeights), c.dataType);
    inputs.keyWeights = castVectorToStorage(std::move(inputs.keyWeights), c.dataType);
    inputs.valueWeights = castVectorToStorage(std::move(inputs.valueWeights), c.dataType);
    inputs.outputWeights = castVectorToStorage(std::move(inputs.outputWeights), c.dataType);
    return inputs;
}

AttentionReferenceInputs makeAlibiSentinelInputs(const AttentionReferenceCase& c) {
    if (c.numHeads != c.numKeyValueHeads || c.headDim != c.valueDim || c.inputFeatures < mergedWidth(c) ||
        c.outputFeatures > mergedWidth(c) || c.hasBias || c.useRope) {
        throw std::runtime_error(
            "ALiBi sentinel input helper expects bias-free non-RoPE MHA with input/output widths compatible with the merged head width.");
    }

    AttentionReferenceInputs inputs;
    inputs.featureInput.resize(static_cast<uint64_t>(c.batchSize) * c.sequenceLength * c.inputFeatures, 0.0f);
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t s = 0; s < c.sequenceLength; ++s) {
            for (uint32_t h = 0; h < c.numHeads; ++h) {
                for (uint32_t d = 0; d < c.valueDim; ++d) {
                    const uint32_t f = h * c.valueDim + d;
                    inputs.featureInput[idx3(b, s, f, c.sequenceLength, c.inputFeatures)] =
                        0.05f * static_cast<float>(b + 1) + 0.40f * static_cast<float>(s + 1) + 0.03f * static_cast<float>(h + 1) +
                        0.002f * static_cast<float>(d + 1);
                }
            }
        }
    }

    inputs.queryWeights.resize(static_cast<uint64_t>(c.inputFeatures) * qWidth(c), 0.0f);
    inputs.keyWeights.resize(static_cast<uint64_t>(c.inputFeatures) * kWidth(c), 0.0f);
    inputs.valueWeights.resize(static_cast<uint64_t>(c.inputFeatures) * vWidth(c), 0.0f);
    inputs.outputWeights.resize(static_cast<uint64_t>(mergedWidth(c)) * c.outputFeatures, 0.0f);

    // Keep Q/K logits at zero so the no-ALiBi reference is a uniform average over the causal prefix.
    // Value/output identity projections make the ALiBi preference for recent keys directly visible in feature_output.
    for (uint32_t i = 0; i < mergedWidth(c); ++i) {
        inputs.valueWeights[static_cast<uint64_t>(i) * vWidth(c) + i] = 1.0f;
        if (i < c.outputFeatures) {
            inputs.outputWeights[static_cast<uint64_t>(i) * c.outputFeatures + i] = 1.0f;
        }
    }

    inputs.featureInput = castVectorToStorage(std::move(inputs.featureInput), c.dataType);
    inputs.queryWeights = castVectorToStorage(std::move(inputs.queryWeights), c.dataType);
    inputs.keyWeights = castVectorToStorage(std::move(inputs.keyWeights), c.dataType);
    inputs.valueWeights = castVectorToStorage(std::move(inputs.valueWeights), c.dataType);
    inputs.outputWeights = castVectorToStorage(std::move(inputs.outputWeights), c.dataType);
    return inputs;
}

vector<float> projectToBhsd(const vector<float>& featureInput,
                            const vector<float>& weights,
                            const vector<float>* bias,
                            uint32_t batchSize,
                            uint32_t sequenceLength,
                            uint32_t inputFeatures,
                            uint32_t heads,
                            uint32_t dim,
                            DataType dataType) {
    const uint32_t width = heads * dim;
    vector<float> out(static_cast<uint64_t>(batchSize) * heads * sequenceLength * dim, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < sequenceLength; ++s) {
            for (uint32_t h = 0; h < heads; ++h) {
                for (uint32_t d = 0; d < dim; ++d) {
                    const uint32_t o = h * dim + d;
                    float sum = bias == nullptr ? 0.0f : (*bias)[o];
                    for (uint32_t f = 0; f < inputFeatures; ++f) {
                        sum += featureInput[idx3(b, s, f, sequenceLength, inputFeatures)] * weights[static_cast<uint64_t>(f) * width + o];
                    }
                    out[idx4(b, h, s, d, heads, sequenceLength, dim)] = castToStorage(sum, dataType);
                }
            }
        }
    }
    return out;
}

void applyRopeInPlace(vector<float>& bhsd,
                      const AttentionReferenceCase& c,
                      uint32_t heads,
                      uint32_t dim,
                      int64_t positionOffset) {
    Impl::RotaryPositionEmbeddingOptions opts = c.ropeOptions;
    opts.position_offset = positionOffset;
    const uint64_t rotaryDim = opts.rotary_dim == 0 ? dim : opts.rotary_dim;
    ASSERT_TRUE(rotaryDim > 0 && rotaryDim <= dim && (rotaryDim % 2) == 0);

    const vector<float> in = bhsd;
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t s = 0; s < c.sequenceLength; ++s) {
                float ropePosition = static_cast<float>(s) + static_cast<float>(opts.position_offset);
                if (opts.scaling_kind == Impl::RotaryScalingKind::Linear) {
                    ropePosition /= static_cast<float>(opts.scaling_factor);
                }
                float ropeBase = static_cast<float>(opts.base);
                const uint32_t logicalBatchMaxSequenceLength =
                    c.sequenceLengths.empty()
                        ? c.sequenceLength
                        : static_cast<uint32_t>(*std::max_element(c.sequenceLengths.begin(), c.sequenceLengths.end()));
                const int64_t queryOffset = c.queryRopePositionOffset.value_or(c.ropeOptions.position_offset);
                const int64_t keyOffset = c.keyRopePositionOffset.value_or(c.ropeOptions.position_offset);
                const float ropeSeqLen = std::max(
                    std::max(static_cast<float>(logicalBatchMaxSequenceLength) +
                                 static_cast<float>(std::max<int64_t>(0, queryOffset)),
                             static_cast<float>(logicalBatchMaxSequenceLength) +
                                 static_cast<float>(std::max<int64_t>(0, keyOffset))),
                    1.0f);
                if (opts.scaling_kind == Impl::RotaryScalingKind::DynamicNTK) {
                    const float ropeOriginalMax = static_cast<float>(opts.original_max_position_embeddings);
                    if (ropeSeqLen > ropeOriginalMax && rotaryDim > 2) {
                        const float ratio = static_cast<float>(opts.scaling_factor) * ropeSeqLen / ropeOriginalMax -
                                            (static_cast<float>(opts.scaling_factor) - 1.0f);
                        ropeBase *= powf(ratio, static_cast<float>(rotaryDim) / static_cast<float>(rotaryDim - 2));
                    }
                }
                for (uint32_t d = 0; d < dim; ++d) {
                    const uint64_t outIndex = idx4(b, h, s, d, heads, c.sequenceLength, dim);
                    if (d >= rotaryDim) {
                        bhsd[outIndex] = in[outIndex];
                        continue;
                    }
                    const uint64_t halfDim = rotaryDim / 2;
                    const bool firstLane = opts.interleaved ? ((d & 1U) == 0U) : (d < halfDim);
                    const uint64_t pairIndex = opts.interleaved ? (d >> 1U) : (d < halfDim ? d : d - halfDim);
                    const uint64_t peerDelta = opts.interleaved ? 1U : halfDim;
                    const uint32_t peerD = static_cast<uint32_t>(firstLane ? d + peerDelta : d - peerDelta);
                    float ropeFreq =
                        powf(ropeBase, -2.0f * static_cast<float>(pairIndex) / static_cast<float>(rotaryDim));
                    if (opts.scaling_kind == Impl::RotaryScalingKind::LongRope) {
                        const bool useLongFactors = ropeSeqLen > static_cast<float>(opts.original_max_position_embeddings);
                        const float factor = static_cast<float>(useLongFactors ? opts.long_rope_long_factors.at(pairIndex)
                                                                             : opts.long_rope_short_factors.at(pairIndex));
                        ropeFreq /= factor;
                    }
                    const float theta = ropePosition * ropeFreq;
                    float sTheta = sinf(theta);
                    const float cTheta = cosf(theta);
                    if (opts.inverse)
                        sTheta = -sTheta;
                    const float current = in[outIndex];
                    const float peer = in[idx4(b, h, s, peerD, heads, c.sequenceLength, dim)];
                    const float rotated = firstLane ? (current * cTheta - peer * sTheta) : (peer * sTheta + current * cTheta);
                    bhsd[outIndex] = castToStorage(rotated, c.dataType);
                }
            }
        }
    }
}

bool attentionKeyAllowed(const AttentionReferenceCase& c, uint32_t queryIndex, uint32_t keyIndex) {
    const int64_t q = static_cast<int64_t>(queryIndex);
    const int64_t k = static_cast<int64_t>(keyIndex);

    switch (c.maskKind) {
        case Impl::AttentionMaskKind::None:
            return true;
        case Impl::AttentionMaskKind::CausalTopLeft:
            return k <= (q + c.diagonalRightBound);
        case Impl::AttentionMaskKind::CausalBottomRight:
            // Attention layer is self-attention today, so query length and KV length are the same; bottom-right
            // diagonal alignment numerically matches top-left alignment, but still exercises the cuDNN option path.
            return k <= q;
        case Impl::AttentionMaskKind::SlidingWindowTopLeft:
            return k > (q - c.diagonalLeftBound) && k <= (q + c.diagonalRightBound);
        case Impl::AttentionMaskKind::SlidingWindowBottomRight:
            return k > (q - c.diagonalLeftBound) && k <= (q + c.diagonalRightBound);
        default:
            throw std::runtime_error("Unsupported mask kind in Attention API CPU reference test.");
    }
}

float alibiSlope(uint32_t numHeads, uint32_t head) {
    const uint32_t closestPowerOfTwo = 1U << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(numHeads))));
    if (head < closestPowerOfTwo) {
        const float base = std::pow(2.0f, -8.0f / static_cast<float>(closestPowerOfTwo));
        return std::pow(base, static_cast<float>(head + 1));
    }

    const float extraBase = std::pow(2.0f, -4.0f / static_cast<float>(closestPowerOfTwo));
    const uint32_t extraIndex = head - closestPowerOfTwo;
    return std::pow(extraBase, static_cast<float>(1 + 2 * extraIndex));
}

float alibiBias(uint32_t numHeads, uint32_t head, uint32_t queryIndex, uint32_t keyIndex) {
    return alibiSlope(numHeads, head) * (static_cast<float>(static_cast<int64_t>(keyIndex) - static_cast<int64_t>(queryIndex)));
}

vector<float> sdpaReference(const vector<float>& q, const vector<float>& k, const vector<float>& v, const AttentionReferenceCase& c) {
    vector<float> out(static_cast<uint64_t>(c.batchSize) * c.numHeads * c.sequenceLength * c.valueDim, 0.0f);
    const uint32_t headsPerKvHead = c.numHeads / c.numKeyValueHeads;
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        const uint32_t validLength = effectiveSequenceLength(c, b);
        for (uint32_t h = 0; h < c.numHeads; ++h) {
            const uint32_t kvHead = h / headsPerKvHead;
            for (uint32_t sq = 0; sq < c.sequenceLength; ++sq) {
                if (sq >= validLength)
                    continue;
                vector<float> scores(c.sequenceLength, -std::numeric_limits<float>::infinity());
                float maxScore = -std::numeric_limits<float>::infinity();
                for (uint32_t sk = 0; sk < c.sequenceLength; ++sk) {
                    if (sk >= validLength || !attentionKeyAllowed(c, sq, sk))
                        continue;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < c.headDim; ++d) {
                        dot += q[idx4(b, h, sq, d, c.numHeads, c.sequenceLength, c.headDim)] *
                               k[idx4(b, kvHead, sk, d, c.numKeyValueHeads, c.sequenceLength, c.headDim)];
                    }
                    scores[sk] = dot * c.attentionScale;
                    if (c.useAlibiMask)
                        scores[sk] += alibiBias(c.numHeads, h, sq, sk);
                    maxScore = std::max(maxScore, scores[sk]);
                }

                float denom = 0.0f;
                for (uint32_t sk = 0; sk < c.sequenceLength; ++sk) {
                    if (scores[sk] == -std::numeric_limits<float>::infinity())
                        continue;
                    scores[sk] = expf(scores[sk] - maxScore);
                    denom += scores[sk];
                }
                for (uint32_t dv = 0; dv < c.valueDim; ++dv) {
                    float sum = 0.0f;
                    for (uint32_t sk = 0; sk < c.sequenceLength; ++sk) {
                        if (scores[sk] == -std::numeric_limits<float>::infinity())
                            continue;
                        sum += (scores[sk] / denom) * v[idx4(b, kvHead, sk, dv, c.numKeyValueHeads, c.sequenceLength, c.valueDim)];
                    }
                    out[idx4(b, h, sq, dv, c.numHeads, c.sequenceLength, c.valueDim)] = castToStorage(sum, c.dataType);
                }
            }
        }
    }
    return out;
}

vector<float> bhsdSemanticToBshdStorage(
    const vector<float>& bhsd, uint32_t batchSize, uint32_t sequenceLength, uint32_t heads, uint32_t dim) {
    vector<float> storage(static_cast<uint64_t>(batchSize) * sequenceLength * heads * dim, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t s = 0; s < sequenceLength; ++s) {
                for (uint32_t d = 0; d < dim; ++d) {
                    storage[idxBshd(b, s, h, d, sequenceLength, heads, dim)] = bhsd[idx4(b, h, s, d, heads, sequenceLength, dim)];
                }
            }
        }
    }
    return storage;
}

vector<float> bshdStorageToBhsdSemantic(
    const vector<float>& storage, uint32_t batchSize, uint32_t sequenceLength, uint32_t heads, uint32_t dim) {
    vector<float> bhsd(static_cast<uint64_t>(batchSize) * heads * sequenceLength * dim, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t s = 0; s < sequenceLength; ++s) {
                for (uint32_t d = 0; d < dim; ++d) {
                    bhsd[idx4(b, h, s, d, heads, sequenceLength, dim)] = storage[idxBshd(b, s, h, d, sequenceLength, heads, dim)];
                }
            }
        }
    }
    return bhsd;
}

vector<uint32_t> canonicalRaggedRowOffsets(const vector<int32_t>& lengths) {
    vector<uint32_t> offsets(lengths.size() + 1, 0);
    uint64_t cursor = 0;
    constexpr uint64_t MAX_UINT32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    for (uint64_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] < 0)
            throw std::invalid_argument("Ragged sequence lengths must be non-negative.");

        const uint64_t length = static_cast<uint64_t>(lengths[i]);
        if (length > MAX_UINT32 - cursor)
            throw std::overflow_error("Canonical ragged row offsets exceed UINT32 capacity.");

        cursor += length;
        offsets[i + 1] = static_cast<uint32_t>(cursor);
    }
    return offsets;
}

vector<float> packBsfRaggedStorage(const vector<float>& dense,
                                   const vector<int32_t>& lengths,
                                   uint32_t batchSize,
                                   uint32_t sequenceLength,
                                   uint32_t width) {
    vector<float> packed(static_cast<uint64_t>(batchSize) * sequenceLength * width, 0.0f);
    uint64_t cursor = 0;
    for (uint32_t b = 0; b < batchSize; ++b) {
        const uint32_t valid = static_cast<uint32_t>(lengths.at(b));
        for (uint32_t s = 0; s < valid; ++s) {
            const uint64_t src = idx3(b, s, 0, sequenceLength, width);
            std::copy(dense.begin() + src, dense.begin() + src + width, packed.begin() + cursor);
            cursor += width;
        }
    }
    return packed;
}

vector<float> packedBsfRaggedValidValues(const vector<float>& storage, const vector<int32_t>& lengths, uint32_t width) {
    vector<float> values;
    for (int32_t length : lengths) {
        values.resize(values.size() + static_cast<uint64_t>(length) * width);
    }
    std::copy(storage.begin(), storage.begin() + values.size(), values.begin());
    return values;
}

vector<float> mergeBhsdToBsd(const vector<float>& bhsd, const AttentionReferenceCase& c) {
    vector<float> merged(static_cast<uint64_t>(c.batchSize) * c.sequenceLength * mergedWidth(c), 0.0f);
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t s = 0; s < c.sequenceLength; ++s) {
            for (uint32_t h = 0; h < c.numHeads; ++h) {
                for (uint32_t d = 0; d < c.valueDim; ++d) {
                    merged[idx3(b, s, h * c.valueDim + d, c.sequenceLength, mergedWidth(c))] =
                        bhsd[idx4(b, h, s, d, c.numHeads, c.sequenceLength, c.valueDim)];
                }
            }
        }
    }
    return merged;
}

vector<float> outputProjectionReference(const vector<float>& merged,
                                        const AttentionReferenceInputs& inputs,
                                        const AttentionReferenceCase& c) {
    vector<float> out(static_cast<uint64_t>(c.batchSize) * c.sequenceLength * c.outputFeatures, 0.0f);
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t s = 0; s < c.sequenceLength; ++s) {
            for (uint32_t o = 0; o < c.outputFeatures; ++o) {
                float sum = c.hasBias ? inputs.outputBias[o] : 0.0f;
                for (uint32_t i = 0; i < mergedWidth(c); ++i) {
                    sum += merged[idx3(b, s, i, c.sequenceLength, mergedWidth(c))] *
                           inputs.outputWeights[static_cast<uint64_t>(i) * c.outputFeatures + o];
                }
                out[idx3(b, s, o, c.sequenceLength, c.outputFeatures)] = castToStorage(sum, c.dataType);
            }
        }
    }
    return out;
}

vector<float> attentionLayerReference(const AttentionReferenceInputs& inputs, const AttentionReferenceCase& c) {
    const vector<float>* qBias = c.hasBias ? &inputs.queryBias : nullptr;
    const vector<float>* kBias = c.hasBias ? &inputs.keyBias : nullptr;
    const vector<float>* vBias = c.hasBias ? &inputs.valueBias : nullptr;
    vector<float> q = projectToBhsd(
        inputs.featureInput, inputs.queryWeights, qBias, c.batchSize, c.sequenceLength, c.inputFeatures, c.numHeads, c.headDim, c.dataType);
    vector<float> k = projectToBhsd(inputs.featureInput,
                                    inputs.keyWeights,
                                    kBias,
                                    c.batchSize,
                                    c.sequenceLength,
                                    c.inputFeatures,
                                    c.numKeyValueHeads,
                                    c.headDim,
                                    c.dataType);
    vector<float> v = projectToBhsd(inputs.featureInput,
                                    inputs.valueWeights,
                                    vBias,
                                    c.batchSize,
                                    c.sequenceLength,
                                    c.inputFeatures,
                                    c.numKeyValueHeads,
                                    c.valueDim,
                                    c.dataType);
    if (c.useRope) {
        applyRopeInPlace(q, c, c.numHeads, c.headDim, c.queryRopePositionOffset.value_or(c.ropeOptions.position_offset));
        applyRopeInPlace(k, c, c.numKeyValueHeads, c.headDim, c.keyRopePositionOffset.value_or(c.ropeOptions.position_offset));
    }
    return outputProjectionReference(mergeBhsdToBsd(sdpaReference(q, k, v, c), c), inputs, c);
}

vector<float> attentionLayerReferenceWithRopeAppliedAfterBadBshdReinterpret(const AttentionReferenceInputs& inputs,
                                                                            const AttentionReferenceCase& c) {
    const vector<float>* qBias = c.hasBias ? &inputs.queryBias : nullptr;
    const vector<float>* kBias = c.hasBias ? &inputs.keyBias : nullptr;
    const vector<float>* vBias = c.hasBias ? &inputs.valueBias : nullptr;
    vector<float> q = projectToBhsd(
        inputs.featureInput, inputs.queryWeights, qBias, c.batchSize, c.sequenceLength, c.inputFeatures, c.numHeads, c.headDim, c.dataType);
    vector<float> k = projectToBhsd(inputs.featureInput,
                                    inputs.keyWeights,
                                    kBias,
                                    c.batchSize,
                                    c.sequenceLength,
                                    c.inputFeatures,
                                    c.numKeyValueHeads,
                                    c.headDim,
                                    c.dataType);
    vector<float> v = projectToBhsd(inputs.featureInput,
                                    inputs.valueWeights,
                                    vBias,
                                    c.batchSize,
                                    c.sequenceLength,
                                    c.inputFeatures,
                                    c.numKeyValueHeads,
                                    c.valueDim,
                                    c.dataType);

    vector<float> qStorage = bhsdSemanticToBshdStorage(q, c.batchSize, c.sequenceLength, c.numHeads, c.headDim);
    vector<float> kStorage = bhsdSemanticToBshdStorage(k, c.batchSize, c.sequenceLength, c.numKeyValueHeads, c.headDim);

    // This intentionally models the layout bug we want the end-to-end test to catch: RoPE is applied by a
    // generic dense op after BSHD projection storage has been reinterpreted as dense [B,H,S,D] metadata.
    // The subsequent SDPA call still reads the buffer through BSHD strides, so the sequence/head positions
    // seen by RoPE and attention disagree.
    applyRopeInPlace(qStorage, c, c.numHeads, c.headDim, c.queryRopePositionOffset.value_or(c.ropeOptions.position_offset));
    applyRopeInPlace(kStorage, c, c.numKeyValueHeads, c.headDim, c.keyRopePositionOffset.value_or(c.ropeOptions.position_offset));

    q = bshdStorageToBhsdSemantic(qStorage, c.batchSize, c.sequenceLength, c.numHeads, c.headDim);
    k = bshdStorageToBhsdSemantic(kStorage, c.batchSize, c.sequenceLength, c.numKeyValueHeads, c.headDim);
    return outputProjectionReference(mergeBhsdToBsd(sdpaReference(q, k, v, c), c), inputs, c);
}

void setAttentionParameters(const shared_ptr<Impl::CustomLayer>& physicalAttention,
                            const AttentionReferenceInputs& inputs,
                            const AttentionReferenceCase& c,
                            Stream& stream) {
    if (attentionUsesPackedQkv(c)) {
        setParameterTensor(physicalAttention->getParameter("qkv_weights"), packQkvWeights(inputs, c), stream);
    } else {
        setParameterTensor(physicalAttention->getParameter("query_weights"), inputs.queryWeights, stream);
        setParameterTensor(physicalAttention->getParameter("key_weights"), inputs.keyWeights, stream);
        setParameterTensor(physicalAttention->getParameter("value_weights"), inputs.valueWeights, stream);
    }
    setParameterTensor(physicalAttention->getParameter("output_weights"), inputs.outputWeights, stream);
    if (c.hasBias) {
        if (attentionUsesPackedQkv(c)) {
            setParameterTensor(physicalAttention->getParameter("qkv_bias"), packQkvBias(inputs, c), stream);
        } else {
            setParameterTensor(physicalAttention->getParameter("query_bias"), inputs.queryBias, stream);
            setParameterTensor(physicalAttention->getParameter("key_bias"), inputs.keyBias, stream);
            setParameterTensor(physicalAttention->getParameter("value_bias"), inputs.valueBias, stream);
        }
        setParameterTensor(physicalAttention->getParameter("output_bias"), inputs.outputBias, stream);
    }
    stream.synchronize();
}

void runAttentionApiReferenceCaseWithInputs(const std::string& networkName,
                                            const AttentionReferenceCase& c,
                                            const AttentionReferenceInputs& inputs,
                                            float atol = 9e-2f,
                                            float rtol = 9e-2f) {
    Api::Network network(networkName);
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({c.sequenceLength, c.inputFeatures})
                                  .dataType(c.dataType)
                                  .build();
    Api::Attention::Builder builder;
    builder.network(network)
        .featureInput(input.getFeatureOutput().value())
        .numHeads(c.numHeads)
        .numKeyValueHeads(c.numKeyValueHeads)
        .headDim(c.headDim)
        .valueDim(c.valueDim)
        .outputFeatures(c.outputFeatures)
        .hasBias(c.hasBias)
        .maskKind(c.maskKind)
        .diagonalLeftBound(c.diagonalLeftBound)
        .diagonalRightBound(c.diagonalRightBound)
        .useAlibiMask(c.useAlibiMask)
        .weightsDataType(c.dataType)
        .computeDataType(DataType::FP32)
        .outputDataType(c.dataType)
        .attentionScale(c.attentionScale);
    if (c.useRope) {
        builder.ropeOptions(c.ropeOptions);
        if (c.queryRopePositionOffset.has_value()) {
            builder.queryRopePositionOffset(c.queryRopePositionOffset.value());
        }
        if (c.keyRopePositionOffset.has_value()) {
            builder.keyRopePositionOffset(c.keyRopePositionOffset.value());
        }
    }
    Api::Attention attention = builder.build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getOutput("feature_output"))
                                    .dataType(c.dataType)
                                    .build();

    PlacedAttentionFixture fixture = placeSingleAttentionNetwork(network, input, output, attention, c.batchSize, true);
    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, inputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, inputs.featureInput);

    const vector<float> expected = attentionLayerReference(inputs, c);
    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, c.batchSize);
    expectAllClose(actual, expected, atol, rtol);
}

void runAttentionApiReferenceCase(const std::string& networkName, const AttentionReferenceCase& c, float atol = 9e-2f, float rtol = 9e-2f) {
    runAttentionApiReferenceCaseWithInputs(networkName, c, makeAttentionReferenceInputs(c), atol, rtol);
}

vector<float> deterministicValues(uint64_t count, float scale, float phase) {
    vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        values[i] = scale * std::sin((static_cast<float>(i) + 1.0f) * (0.17f + phase));
    }
    return values;
}

float deterministicParameterPhase(const string& parameterName) {
    uint32_t accumulator = 0;
    for (unsigned char c : parameterName) {
        accumulator = accumulator * 131U + c;
    }
    return 0.001f * static_cast<float>(1U + accumulator % 251U);
}

struct ResidualAttentionTrainingResult {
    vector<float> output;
    vector<float> queryGradient;
    optional<vector<float>> contextGradient;
    optional<vector<float>> residualGradient;
    unordered_map<string, vector<float>> parametersAfter;
    vector<float> residualValues;
    vector<float> upstreamGradient;
};

ResidualAttentionTrainingResult runResidualAttentionTrainingCase(const string& networkName,
                                                                 bool fused,
                                                                 bool crossAttention,
                                                                 float dropoutProbability = 0.0f,
                                                                 bool trainingDropoutEnabled = true) {
    auto require = [](bool condition, const string& message) {
        if (!condition) {
            throw runtime_error(message);
        }
    };

    constexpr uint32_t batchSize = 2;
    constexpr uint32_t querySequenceLength = 3;
    constexpr uint32_t keyValueSequenceLength = 4;
    constexpr uint32_t queryFeatures = 8;
    constexpr uint32_t contextFeatures = 12;
    constexpr uint32_t outputFeatures = 8;
    constexpr uint32_t numHeads = 2;
    constexpr uint32_t numKeyValueHeads = 1;
    constexpr uint32_t headDim = 8;
    constexpr uint32_t valueDim = 8;
    constexpr float learningRate = 0.01f;
    const DataType dataType = DataType::BF16;

    const vector<float> queryValues = deterministicValues(
        static_cast<uint64_t>(batchSize) * querySequenceLength * queryFeatures, 0.25f, 0.03f);
    const vector<float> contextValues = deterministicValues(
        static_cast<uint64_t>(batchSize) * keyValueSequenceLength * contextFeatures, 0.20f, 0.11f);
    const vector<float> residualValues = deterministicValues(
        static_cast<uint64_t>(batchSize) * querySequenceLength * outputFeatures, 0.15f, 0.19f);
    const vector<float> upstreamGradient = deterministicValues(
        static_cast<uint64_t>(batchSize) * querySequenceLength * outputFeatures, 0.10f, 0.29f);

    shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();

    Api::Network network(networkName);
    Api::NetworkInput query = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .dimensions({querySequenceLength, queryFeatures})
                                  .dataType(dataType)
                                  .build();
    optional<Api::NetworkInput> context;
    if (crossAttention) {
        context = Api::NetworkInput::Builder()
                      .network(network)
                      .name("context")
                      .dimensions({keyValueSequenceLength, contextFeatures})
                      .dataType(dataType)
                      .build();
    }
    Api::NetworkInput residual = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("residual")
                                     .dimensions({querySequenceLength, outputFeatures})
                                     .dataType(dataType)
                                     .build();

    Api::GradientRivet queryRivet = Api::GradientRivet::Builder().network(network).tensor(query.getFeatureOutput().value()).build();
    optional<Api::GradientRivet> contextRivet;
    if (crossAttention) {
        contextRivet = Api::GradientRivet::Builder().network(network).tensor(context->getFeatureOutput().value()).build();
    }
    Api::GradientRivet residualRivet =
        Api::GradientRivet::Builder().network(network).tensor(residual.getFeatureOutput().value()).build();

    Api::Attention::Builder builder;
    builder.network(network)
        .featureInput(queryRivet.getFeatureOutput().value())
        .numHeads(numHeads)
        .numKeyValueHeads(numKeyValueHeads)
        .headDim(headDim)
        .valueDim(valueDim)
        .outputFeatures(outputFeatures)
        .hasBias(false)
        .weightsDataType(dataType)
        .computeDataType(DataType::FP32)
        .outputDataType(dataType)
        .optimizer(sgd);
    if (crossAttention) {
        builder.contextInput(contextRivet->getFeatureOutput().value());
    }
    if (dropoutProbability > 0.0f) {
        builder.dropout(dropoutProbability, 1234, 5678);
    }
    if (fused) {
        Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, dataType);
        Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, dataType);
        builder.epilogueInput("residual", residualRivet.getFeatureOutput().value()).epilogue(attentionOutput + residualInput);
    }
    Api::Attention attention = builder.build();

    optional<Api::CustomLayer> residualAdd;
    Api::Tensor transformerOutput = attention.getFeatureOutput().value();
    if (!fused) {
        Impl::Expression attentionInput = Impl::Expression::input("attention", DataType::FP32, dataType);
        Impl::Expression residualInput = Impl::Expression::input("residual", DataType::FP32, dataType);
        Impl::ExpressionDefinition definition =
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"output", attentionInput + residualInput}}));
        residualAdd.emplace(Api::CustomLayer::Builder()
                                .network(network)
                                .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                .inputNames({"attention", "residual"})
                                .outputNames({"output"})
                                .inputInterface({{"attention", transformerOutput},
                                                 {"residual", residualRivet.getFeatureOutput().value()}})
                                .build());
        transformerOutput = residualAdd->getOutput("output");
    }

    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(transformerOutput).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    synchronizeEvents(initDoneEvents);
    placedNetwork->setTrainingDropoutEnabled(trainingDropoutEnabled);
    require(placedNetwork->getNumTrainingDropoutControllableLayers() == 1,
            "Residual attention test expected one training-dropout-controllable physical layer.");
    require(placedNetwork->isTrainingDropoutEnabled() == trainingDropoutEnabled,
            "Residual attention test observed the wrong placed training-dropout policy.");
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalQuery = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(query.getId()));
    auto physicalContext = crossAttention
                               ? dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(context->getId()))
                               : nullptr;
    auto physicalResidual =
        dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(residual.getId()));
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalAttention = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(attention.getId()));
    auto physicalResidualAdd = !fused
                                   ? dynamic_pointer_cast<Impl::CustomLayer>(
                                         stampedNetwork.getPhysicalLayerFromApiLayer(residualAdd->getId()))
                                   : nullptr;
    require(physicalQuery != nullptr, "Residual attention test failed to place query NetworkInput.");
    require(physicalOutput != nullptr, "Residual attention test failed to place NetworkOutput.");
    require(physicalAttention != nullptr, "Residual attention test failed to place Attention.");
    if (crossAttention) require(physicalContext != nullptr, "Residual attention test failed to place context NetworkInput.");
    require(physicalResidual != nullptr, "Residual attention test failed to place residual NetworkInput.");
    if (!fused) require(physicalResidualAdd != nullptr, "Residual attention test failed to place unfused residual add.");

    Stream stream = physicalAttention->getStreams()[0];
    require(physicalAttention->getGradientUpdateStream().has_value(),
            "Residual attention training test requires a gradient update stream.");
    Stream gradientStream = physicalAttention->getGradientUpdateStream().value();

    for (const string& parameterName : physicalAttention->listParameters()) {
        const shared_ptr<Impl::PhysicalParameter> parameter = physicalAttention->getParameter(parameterName);
        require(parameter != nullptr, "Residual attention test encountered a null parameter.");
        require(parameter->getStorage().has_value(), "Residual attention test parameter has no storage.");
        const uint64_t count = tensorNumel(parameter->getStorage().value());
        vector<float> values = deterministicValues(count, 0.04f, deterministicParameterPhase(parameterName));
        setParameterTensor(parameter, values, stream);
    }
    stream.synchronize();

    Impl::Tensor queryHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, querySequenceLength, queryFeatures}));
    writeCpuTensor(queryHost, queryValues);
    physicalQuery->forward(queryHost, false, batchSize);
    optional<Impl::Tensor> contextHost;
    if (crossAttention) {
        contextHost.emplace(cpuPlacement,
                            Impl::TensorDescriptor(dataType, {batchSize, keyValueSequenceLength, contextFeatures}));
        writeCpuTensor(contextHost.value(), contextValues);
        physicalContext->forward(contextHost.value(), false, batchSize);
    }
    Impl::Tensor residualHost(cpuPlacement,
                              Impl::TensorDescriptor(dataType, {batchSize, querySequenceLength, outputFeatures}));
    writeCpuTensor(residualHost, residualValues);
    physicalResidual->forward(residualHost, false, batchSize);
    physicalOutput->getOutputReadyEvent().synchronize();

    ResidualAttentionTrainingResult result;
    result.output = readCpuTensor(physicalOutput->getFeatureOutput().value());
    result.residualValues = residualValues;
    result.upstreamGradient = upstreamGradient;

    Impl::CustomLayer* physicalTerminal = fused ? physicalAttention.get() : physicalResidualAdd.get();
    Stream terminalStream = physicalTerminal->getStreams()[0];
    require(physicalTerminal->getErrorInputs().size() == 1U,
            "Residual attention test expected exactly one downstream error input.");
    require(physicalTerminal->getErrorInputs().front().has_value(),
            "Residual attention test downstream error input was not allocated.");
    const size_t expectedAttentionErrorOutputs = crossAttention ? (fused ? 3U : 2U) : (fused ? 2U : 1U);
    require(physicalAttention->getErrorOutputs().size() == expectedAttentionErrorOutputs,
            "Residual attention test produced an unexpected number of Attention upstream gradients.");
    for (const optional<Impl::Tensor>& errorOutput : physicalAttention->getErrorOutputs()) {
        require(errorOutput.has_value(), "Residual attention test Attention upstream gradient was not allocated.");
    }
    if (!fused) {
        require(physicalResidualAdd->getErrorOutputs().size() == 2U,
                "Unfused residual add must produce Attention and residual gradients.");
        for (const optional<Impl::Tensor>& errorOutput : physicalResidualAdd->getErrorOutputs()) {
            require(errorOutput.has_value(), "Unfused residual add upstream gradient was not allocated.");
        }
    }

    Impl::Tensor errorInput = physicalTerminal->getErrorInputs().front().value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, upstreamGradient);
    errorInput.copyFromAsync(errorInputHost, terminalStream);
    physicalTerminal->backward(errorInput, batchSize);

    result.queryGradient = readCpuTensor(copyTensorToCpu(physicalAttention->getErrorOutputs().at(0).value(), stream));
    size_t nextErrorOutput = 1;
    if (crossAttention) {
        result.contextGradient =
            readCpuTensor(copyTensorToCpu(physicalAttention->getErrorOutputs().at(nextErrorOutput++).value(), stream));
    }
    result.residualGradient = fused
                                  ? readCpuTensor(copyTensorToCpu(
                                        physicalAttention->getErrorOutputs().at(nextErrorOutput).value(), stream))
                                  : readCpuTensor(copyTensorToCpu(
                                        physicalResidualAdd->getErrorOutputs().at(1).value(), terminalStream));

    for (const string& parameterName : physicalAttention->listParameters()) {
        result.parametersAfter.emplace(
            parameterName,
            readCpuTensor(copyTensorToCpu(physicalAttention->getParameter(parameterName)->getStorage().value(), gradientStream)));
    }
    stream.synchronize();
    gradientStream.synchronize();
    return result;
}

void expectResidualAttentionTrainingMatchesUnfused(bool crossAttention) {
    const string kind = crossAttention ? "cross" : "self";
    ResidualAttentionTrainingResult unfused =
        runResidualAttentionTrainingCase("attention_api_unfused_" + kind + "_training_reference", false, crossAttention);
    ResidualAttentionTrainingResult fused =
        runResidualAttentionTrainingCase("attention_api_fused_" + kind + "_training_reference", true, crossAttention);

    ASSERT_EQ(fused.output.size(), unfused.output.size());
    expectAllClose(fused.output, unfused.output, 1.8e-1f, 1.8e-1f);
    expectAllClose(fused.queryGradient, unfused.queryGradient, 1.8e-1f, 1.8e-1f);

    if (crossAttention) {
        ASSERT_TRUE(fused.contextGradient.has_value());
        ASSERT_TRUE(unfused.contextGradient.has_value());
        expectAllClose(fused.contextGradient.value(), unfused.contextGradient.value(), 1.8e-1f, 1.8e-1f);
    }

    ASSERT_TRUE(fused.residualGradient.has_value());
    ASSERT_TRUE(unfused.residualGradient.has_value());
    expectAllClose(fused.residualGradient.value(), unfused.residualGradient.value(), 1.8e-1f, 1.8e-1f);
    expectAllClose(fused.residualGradient.value(), fused.upstreamGradient, 1.8e-1f, 1.8e-1f);

    ASSERT_EQ(fused.parametersAfter.size(), unfused.parametersAfter.size());
    for (const auto& [parameterName, fusedValues] : fused.parametersAfter) {
        ASSERT_TRUE(unfused.parametersAfter.contains(parameterName));
        expectAllClose(fusedValues, unfused.parametersAfter.at(parameterName), 1.8e-1f, 1.8e-1f);
    }
}

}  // namespace

TEST(AttentionApi, BuildsComposedCausalSelfAttention) {
    Api::Network network("attention_api_builds_composed_causal_self_attention");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({16, 64}).dataType(DataType::FP16).build();

    Api::Attention attention =
        Api::Attention::Builder().network(network).featureInput(input.getFeatureOutput().value()).numHeads(4).causal().build();

    EXPECT_EQ(attention.getLayerType(), "Attention");
    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"feature_input"}));
    EXPECT_EQ(attention.getOutputNames(), (std::vector<std::string>{"feature_output"}));
    EXPECT_EQ(attention.getOutput("feature_output").getDataType(), DataType::FP16);
    EXPECT_EQ(attention.getOutput("feature_output").getDimensions(), (std::vector<uint64_t>{16, 64}));
    EXPECT_EQ(attention.getNumHeads(), 4U);
    EXPECT_EQ(attention.getNumKeyValueHeads(), 4U);
    EXPECT_EQ(attention.getHeadDim(), 16U);
    EXPECT_EQ(attention.getValueDim(), 16U);
    EXPECT_EQ(attention.getOutputFeatures(), 64U);
    EXPECT_EQ(attention.getMaskKind(), Impl::AttentionMaskKind::CausalTopLeft);
    EXPECT_FALSE(attention.hasEpilogue());
    EXPECT_TRUE(attention.getEpilogueInputBindings().empty());
    EXPECT_TRUE(attention.architectureJson().at("epilogue").is_null());
    EXPECT_TRUE(attention.architectureJson().at("epilogue_inputs").empty());
}

TEST(AttentionApi, DeserializeAcceptsPreEpilogueVersionOneMetadata) {
    Api::Network network("attention_api_deserialize_accepts_pre_epilogue_v1_metadata");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(2)
                                   .headDim(4)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .build();

    nlohmann::json legacyArchitecture = attention.architectureJson();
    legacyArchitecture.erase("epilogue");
    legacyArchitecture.erase("epilogue_inputs");

    const uint32_t previousTrainableLayerCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, legacyArchitecture, &network);
    ASSERT_EQ(network.getNumTrainableLayers(), previousTrainableLayerCount + 1);
    auto restored = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(previousTrainableLayerCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->hasEpilogue());
    EXPECT_TRUE(restored->getEpilogueInputBindings().empty());
    EXPECT_EQ(restored->getInputNames(), (std::vector<std::string>{"feature_input"}));
}

TEST(AttentionApi, BuildsSelfAttentionWithResidualAddEpilogue) {
    Api::Network network("attention_api_builds_self_attention_with_residual_add_epilogue");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({5, 8}).dataType(DataType::BF16).build();

    Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, DataType::BF16);
    Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(2)
                                   .headDim(4)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .epilogueInput("residual", residual.getFeatureOutput().value())
                                   .epilogue(attentionOutput + residualInput)
                                   .build();

    EXPECT_TRUE(attention.hasEpilogue());
    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"feature_input", "residual"}));
    ASSERT_EQ(attention.getEpilogueInputBindings().size(), 1U);
    EXPECT_EQ(attention.getEpilogueInputBindings().front().first, "residual");
    EXPECT_EQ(attention.getEpilogueInputBindings().front().second.getOriginalId(), residual.getFeatureOutput()->getOriginalId());
    EXPECT_EQ(attention.getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{5, 8}));
    EXPECT_EQ(attention.getFeatureOutput()->getDataType(), DataType::BF16);
}

TEST(AttentionApi, BuildsCrossAttentionWithResidualAddEpilogue) {
    Api::Network network("attention_api_builds_cross_attention_with_residual_add_epilogue");
    Api::NetworkInput query =
        Api::NetworkInput::Builder().network(network).name("query").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput context =
        Api::NetworkInput::Builder().network(network).name("context").dimensions({7, 12}).dataType(DataType::BF16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({5, 8}).dataType(DataType::BF16).build();

    Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, DataType::BF16);
    Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(query.getFeatureOutput().value())
                                   .contextInput(context.getFeatureOutput().value())
                                   .numHeads(2)
                                   .numKeyValueHeads(1)
                                   .headDim(4)
                                   .valueDim(4)
                                   .outputFeatures(8)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .epilogueInput("residual", residual.getFeatureOutput().value())
                                   .epilogue(attentionOutput + residualInput)
                                   .build();

    EXPECT_TRUE(attention.getUseCrossAttention());
    EXPECT_TRUE(attention.hasEpilogue());
    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"feature_input", "context_input", "residual"}));
    ASSERT_EQ(attention.getEpilogueInputBindings().size(), 1U);
    EXPECT_EQ(attention.getEpilogueInputBindings().front().first, "residual");
    EXPECT_EQ(attention.getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{5, 8}));
    EXPECT_EQ(attention.getFeatureOutput()->getDataType(), DataType::BF16);
}

TEST(AttentionApi, SelfAttentionResidualEpilogueDeserializeRoundTrip) {
    Api::Network network("attention_api_self_residual_epilogue_deserialize_round_trip");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({5, 8}).dataType(DataType::BF16).build();

    Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, DataType::BF16);
    Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(2)
                                   .headDim(4)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .epilogueInput("residual", residual.getFeatureOutput().value())
                                   .epilogue(attentionOutput + residualInput)
                                   .build();

    const uint32_t previousTrainableLayerCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, attention.architectureJson(), &network);
    ASSERT_EQ(network.getNumTrainableLayers(), previousTrainableLayerCount + 1);
    auto restored = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(previousTrainableLayerCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->getUseCrossAttention());
    EXPECT_TRUE(restored->hasEpilogue());
    ASSERT_EQ(restored->getEpilogueInputBindings().size(), 1U);
    EXPECT_EQ(restored->getEpilogueInputBindings().front().first, "residual");
    EXPECT_EQ(restored->getEpilogueInputBindings().front().second.getOriginalId(),
              residual.getFeatureOutput()->getOriginalId());
}

TEST(AttentionApi, ResidualEpilogueArchitectureJsonAndDeserializeRoundTrip) {
    Api::Network network("attention_api_residual_epilogue_architecture_round_trip");
    Api::NetworkInput query =
        Api::NetworkInput::Builder().network(network).name("query").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput context =
        Api::NetworkInput::Builder().network(network).name("context").dimensions({7, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({5, 8}).dataType(DataType::BF16).build();

    Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, DataType::BF16);
    Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(query.getFeatureOutput().value())
                                   .contextInput(context.getFeatureOutput().value())
                                   .numHeads(2)
                                   .headDim(4)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .epilogueInput("residual", residual.getFeatureOutput().value())
                                   .epilogue(attentionOutput + residualInput)
                                   .build();

    const nlohmann::json arch = attention.architectureJson();
    ASSERT_TRUE(arch.contains("epilogue"));
    ASSERT_FALSE(arch.at("epilogue").is_null());
    const std::vector<std::string> serializedEpilogueInputNames =
        arch.at("epilogue").at("expected_input_names").get<std::vector<std::string>>();
    const std::set<std::string> serializedEpilogueInputs(
        serializedEpilogueInputNames.begin(), serializedEpilogueInputNames.end());
    EXPECT_EQ(serializedEpilogueInputs,
              (std::set<std::string>{Api::Attention::epilogueInputName(), "residual"}));
    ASSERT_TRUE(arch.contains("epilogue_inputs"));
    ASSERT_EQ(arch.at("epilogue_inputs").size(), 1U);
    EXPECT_EQ(arch.at("epilogue_inputs").at(0).at("name").get<std::string>(), "residual");
    EXPECT_EQ(arch.at("epilogue_inputs").at(0).at("tensor").at("id").get<uint64_t>(),
              residual.getFeatureOutput()->getOriginalId());

    const uint32_t previousTrainableLayerCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    ASSERT_EQ(network.getNumTrainableLayers(), previousTrainableLayerCount + 1);
    auto restored = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(previousTrainableLayerCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->hasEpilogue());
    EXPECT_TRUE(restored->getUseCrossAttention());
    ASSERT_EQ(restored->getEpilogueInputBindings().size(), 1U);
    EXPECT_EQ(restored->getEpilogueInputBindings().front().first, "residual");
    EXPECT_EQ(restored->getEpilogueInputBindings().front().second.getOriginalId(),
              residual.getFeatureOutput()->getOriginalId());
    EXPECT_FALSE(restored->architectureJson().at("epilogue").is_null());
}

TEST(AttentionApi, RejectsInvalidResidualEpilogueBindings) {
    Api::Network network("attention_api_rejects_invalid_residual_epilogue_bindings");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput badShape =
        Api::NetworkInput::Builder().network(network).name("bad_shape").dimensions({4, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput badDtype =
        Api::NetworkInput::Builder().network(network).name("bad_dtype").dimensions({5, 8}).dataType(DataType::FP32).build();

    Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, DataType::BF16);
    Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(2)
                     .headDim(4)
                     .outputDataType(DataType::BF16)
                     .epilogue(attentionOutput + residualInput),
                 std::invalid_argument);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(2)
                     .headDim(4)
                     .outputDataType(DataType::BF16)
                     .epilogueInput("residual", residual.getFeatureOutput().value())
                     .epilogue(residualInput),
                 std::invalid_argument);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(2)
                     .headDim(4)
                     .outputDataType(DataType::BF16)
                     .epilogueInput("residual", badShape.getFeatureOutput().value())
                     .epilogue(attentionOutput + residualInput)
                     .build(),
                 std::invalid_argument);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(2)
                     .headDim(4)
                     .outputDataType(DataType::BF16)
                     .epilogueInput("residual", badDtype.getFeatureOutput().value())
                     .epilogue(attentionOutput + residualInput)
                     .build(),
                 std::invalid_argument);

    Impl::Expression wrongResidualStorageAnnotation =
        Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::FP16);
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(2)
                     .headDim(4)
                     .outputDataType(DataType::BF16)
                     .epilogueInput("residual", residual.getFeatureOutput().value())
                     .epilogue(attentionOutput + wrongResidualStorageAnnotation)
                     .build(),
                 std::invalid_argument);

    Api::Attention::Builder duplicateBuilder;
    duplicateBuilder.network(network)
        .featureInput(input.getFeatureOutput().value())
        .numHeads(2)
        .headDim(4)
        .epilogueInput("residual", residual.getFeatureOutput().value());
    EXPECT_THROW(duplicateBuilder.epilogueInput("residual", residual.getFeatureOutput().value()), std::invalid_argument);

    EXPECT_THROW(static_cast<void>(
                     Api::Attention::epilogueAuxInput("feature_input", DataType::FP32, DataType::BF16)),
                 std::invalid_argument);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(2)
                     .headDim(4)
                     .outputDataType(DataType::BF16)
                     .epilogue(attentionOutput.reshape({1})),
                 std::invalid_argument);
}

TEST(AttentionApi, DeserializeRejectsInvalidResidualEpilogueMetadata) {
    Api::Network network("attention_api_deserialize_rejects_invalid_residual_epilogue_metadata");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({5, 8}).dataType(DataType::BF16).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({5, 8}).dataType(DataType::BF16).build();
    Impl::Expression attentionOutput = Api::Attention::epilogueInput(DataType::FP32, DataType::BF16);
    Impl::Expression residualInput = Api::Attention::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(2)
                                   .headDim(4)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .epilogueInput("residual", residual.getFeatureOutput().value())
                                   .epilogue(attentionOutput + residualInput)
                                   .build();

    std::shared_ptr<thor_file::TarReader> archiveReader;
    nlohmann::json missingExpression = attention.architectureJson();
    missingExpression["epilogue"] = nullptr;
    EXPECT_THROW(Api::Attention::deserialize(archiveReader, missingExpression, &network), std::runtime_error);

    nlohmann::json duplicateBinding = attention.architectureJson();
    duplicateBinding["epilogue_inputs"].push_back(duplicateBinding["epilogue_inputs"].at(0));
    EXPECT_THROW(Api::Attention::deserialize(archiveReader, duplicateBinding, &network), std::runtime_error);

    nlohmann::json missingBinding = attention.architectureJson();
    missingBinding["epilogue_inputs"] = nlohmann::json::array();
    EXPECT_THROW(Api::Attention::deserialize(archiveReader, missingBinding, &network), std::invalid_argument);
}

TEST(AttentionApi, ResidualEpilogueForwardBackwardMatchesUnfusedSelfAttention) {
    expectResidualAttentionTrainingMatchesUnfused(false);
}

TEST(AttentionApi, ResidualEpilogueForwardBackwardMatchesUnfusedCrossAttention) {
    expectResidualAttentionTrainingMatchesUnfused(true);
}

TEST(AttentionApi, DisabledTrainingDropoutUsesDeterministicForwardAndMatchingBackwardVariant) {
    const ResidualAttentionTrainingResult noDropout =
        runResidualAttentionTrainingCase("attention_api_training_dropout_disabled_control",
                                         /*fused=*/true,
                                         /*crossAttention=*/false,
                                         /*dropoutProbability=*/0.0f,
                                         /*trainingDropoutEnabled=*/true);
    const ResidualAttentionTrainingResult configuredButDisabled =
        runResidualAttentionTrainingCase("attention_api_training_dropout_configured_but_disabled",
                                         /*fused=*/true,
                                         /*crossAttention=*/false,
                                         /*dropoutProbability=*/0.5f,
                                         /*trainingDropoutEnabled=*/false);

    expectAllClose(configuredButDisabled.output, noDropout.output, 1.8e-1f, 1.8e-1f);
    expectAllClose(configuredButDisabled.queryGradient, noDropout.queryGradient, 1.8e-1f, 1.8e-1f);
    ASSERT_EQ(configuredButDisabled.parametersAfter.size(), noDropout.parametersAfter.size());
    for (const auto& [parameterName, configuredValues] : configuredButDisabled.parametersAfter) {
        ASSERT_TRUE(noDropout.parametersAfter.contains(parameterName));
        expectAllClose(configuredValues, noDropout.parametersAfter.at(parameterName), 1.8e-1f, 1.8e-1f);
    }
}

TEST(AttentionApi, DynamicNtkRejectsSequenceLengthMetadataBeyondFp32ExactIntegerRange) {
    constexpr uint64_t maxExactFp32Integer = uint64_t{1} << 24;

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.scaling_kind = Impl::RotaryScalingKind::DynamicNTK;
    rope.scaling_factor = 2.0;
    rope.original_max_position_embeddings = 4096;
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    Api::Network boundaryNetwork("attention_api_dynamic_ntk_fp32_exact_boundary");
    Api::NetworkInput boundaryInput = Api::NetworkInput::Builder()
                                          .network(boundaryNetwork)
                                          .name("tokens")
                                          .dimensions({64, 8})
                                          .dataType(DataType::BF16)
                                          .build();
    rope.position_offset = static_cast<int64_t>(maxExactFp32Integer - 64);
    EXPECT_NO_THROW((Api::Attention::Builder()
                         .network(boundaryNetwork)
                         .featureInput(boundaryInput.getFeatureOutput().value())
                         .numHeads(1)
                         .headDim(8)
                         .ropeOptions(rope)
                         .build()));

    Api::Network overflowNetwork("attention_api_dynamic_ntk_fp32_exact_overflow");
    Api::NetworkInput overflowInput = Api::NetworkInput::Builder()
                                          .network(overflowNetwork)
                                          .name("tokens")
                                          .dimensions({64, 8})
                                          .dataType(DataType::BF16)
                                          .build();
    rope.position_offset = static_cast<int64_t>(maxExactFp32Integer - 63);
    EXPECT_THROW((Api::Attention::Builder()
                      .network(overflowNetwork)
                      .featureInput(overflowInput.getFeatureOutput().value())
                      .numHeads(1)
                      .headDim(8)
                      .ropeOptions(rope)
                      .build()),
                 std::invalid_argument);

    Api::Network independentOverflowNetwork("attention_api_dynamic_ntk_independent_query_offset_fp32_overflow");
    Api::NetworkInput independentOverflowInput = Api::NetworkInput::Builder()
                                                     .network(independentOverflowNetwork)
                                                     .name("tokens")
                                                     .dimensions({64, 8})
                                                     .dataType(DataType::BF16)
                                                     .build();
    rope.position_offset = 0;
    EXPECT_THROW((Api::Attention::Builder()
                      .network(independentOverflowNetwork)
                      .featureInput(independentOverflowInput.getFeatureOutput().value())
                      .numHeads(1)
                      .headDim(8)
                      .ropeOptions(rope)
                      .queryRopePositionOffset(static_cast<int64_t>(maxExactFp32Integer - 63))
                      .keyRopePositionOffset(0)
                      .build()),
                 std::invalid_argument);
}

TEST(AttentionApi, RaggedDynamicNtkRejectsCapacityBeyondFp32ExactIntegerRange) {
    constexpr uint64_t maxExactFp32Integer = uint64_t{1} << 24;

    Api::Network network("attention_api_ragged_dynamic_ntk_fp32_capacity_guard");
    Api::RaggedTensor tokens = Api::RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("tokens")
                                   .valuesDataType(DataType::BF16)
                                   .trailingDimensions({8})
                                   .batchSize(2)
                                   .maxTotalValues(maxExactFp32Integer + 1)
                                   .build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.scaling_kind = Impl::RotaryScalingKind::DynamicNTK;
    rope.scaling_factor = 2.0;
    rope.original_max_position_embeddings = 4096;
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    EXPECT_THROW((Api::Attention::Builder()
                      .network(network)
                      .featureInput(tokens)
                      .numHeads(1)
                      .headDim(8)
                      .ropeOptions(rope)
                      .build()),
                 std::invalid_argument);
}

TEST(AttentionApi, LongRopeRejectsOriginalMaxBeyondFp32ExactIntegerRangeAndDeserializeRevalidates) {
    constexpr uint64_t maxExactFp32Integer = uint64_t{1} << 24;

    Api::Network network("attention_api_long_rope_fp32_original_max_guard");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 8}).dataType(DataType::BF16).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.scaling_kind = Impl::RotaryScalingKind::LongRope;
    rope.scaling_factor = 2.0;
    rope.original_max_position_embeddings = maxExactFp32Integer;
    rope.long_rope_short_factors = {1.0, 1.0, 1.0, 1.0};
    rope.long_rope_long_factors = {2.0, 2.0, 2.0, 2.0};
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(1)
                                   .headDim(8)
                                   .ropeOptions(rope)
                                   .build();

    Impl::RotaryPositionEmbeddingOptions invalidRope = rope;
    invalidRope.original_max_position_embeddings = maxExactFp32Integer + 1;
    Api::Network invalidNetwork("attention_api_long_rope_fp32_original_max_reject");
    Api::NetworkInput invalidInput = Api::NetworkInput::Builder()
                                         .network(invalidNetwork)
                                         .name("tokens")
                                         .dimensions({8, 8})
                                         .dataType(DataType::BF16)
                                         .build();
    EXPECT_THROW((Api::Attention::Builder()
                      .network(invalidNetwork)
                      .featureInput(invalidInput.getFeatureOutput().value())
                      .numHeads(1)
                      .headDim(8)
                      .ropeOptions(invalidRope)
                      .build()),
                 std::invalid_argument);

    nlohmann::json invalidArchive = attention.architectureJson();
    invalidArchive["rope_options"]["original_max_position_embeddings"] = maxExactFp32Integer + 1;
    std::shared_ptr<thor_file::TarReader> archiveReader;
    EXPECT_THROW(Api::Attention::deserialize(archiveReader, invalidArchive, &network), std::runtime_error);
}

TEST(AttentionApi, BuildsComposedGqaAttentionWithExplicitDimsBiasAndRope) {
    Api::Network network("attention_api_builds_composed_gqa_attention_with_explicit_dims_bias_and_rope");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 96}).dataType(DataType::BF16).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 16;
    rope.sequence_axis = 1;
    rope.head_dim_axis = 3;
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(6)
                                   .numKeyValueHeads(2)
                                   .headDim(16)
                                   .valueDim(12)
                                   .outputFeatures(80)
                                   .hasBias(true)
                                   .ropeOptions(rope)
                                   .ropeInPlace(true)
                                   .attentionScale(0.25)
                                   .dropout(0.125f, 123456789LL, 987654321LL)
                                   .outputDataType(DataType::BF16)
                                   .build();

    EXPECT_EQ(attention.getOutput("feature_output").getDimensions(), (std::vector<uint64_t>{8, 80}));
    EXPECT_EQ(attention.getNumHeads(), 6U);
    EXPECT_EQ(attention.getNumKeyValueHeads(), 2U);
    EXPECT_EQ(attention.getHeadDim(), 16U);
    EXPECT_EQ(attention.getValueDim(), 12U);
    EXPECT_EQ(attention.getOutputFeatures(), 80U);
    EXPECT_TRUE(attention.getHasBias());
    EXPECT_TRUE(attention.getUseRope());
    EXPECT_TRUE(attention.getRopeInPlace());
    ASSERT_TRUE(attention.getAttentionScale().has_value());
    EXPECT_DOUBLE_EQ(attention.getAttentionScale().value(), 0.25);
    EXPECT_FLOAT_EQ(attention.getDropoutProbability(), 0.125f);
    EXPECT_EQ(attention.getDropoutSeed(), 123456789LL);
    EXPECT_EQ(attention.getDropoutOffset(), 987654321LL);
}

TEST(AttentionApi, RopePositionOffsetsSupportSharedConvenienceAndIndependentQueryKeyOrigins) {
    Api::Network network("attention_api_independent_rope_position_offsets");
    Api::NetworkInput query =
        Api::NetworkInput::Builder().network(network).name("query").dimensions({3, 32}).dataType(DataType::BF16).build();
    Api::NetworkInput context =
        Api::NetworkInput::Builder().network(network).name("context").dimensions({5, 32}).dataType(DataType::BF16).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.position_offset = 7;
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    Api::Attention legacyShared = Api::Attention::Builder()
                                      .network(network)
                                      .featureInput(query.getFeatureOutput().value())
                                      .contextInput(context.getFeatureOutput().value())
                                      .numHeads(4)
                                      .headDim(8)
                                      .ropeOptions(rope)
                                      .build();
    EXPECT_EQ(legacyShared.getQueryRopePositionOffset(), 7);
    EXPECT_EQ(legacyShared.getKeyRopePositionOffset(), 7);

    Api::Attention explicitShared = Api::Attention::Builder()
                                        .network(network)
                                        .featureInput(query.getFeatureOutput().value())
                                        .contextInput(context.getFeatureOutput().value())
                                        .numHeads(4)
                                        .headDim(8)
                                        .ropeOptions(rope)
                                        .ropePositionOffset(11)
                                        .build();
    EXPECT_EQ(explicitShared.getQueryRopePositionOffset(), 11);
    EXPECT_EQ(explicitShared.getKeyRopePositionOffset(), 11);
    EXPECT_EQ(explicitShared.architectureJson().at("rope_options").at("position_offset").get<int64_t>(), 11);

    Api::Attention independent = Api::Attention::Builder()
                                     .network(network)
                                     .featureInput(query.getFeatureOutput().value())
                                     .contextInput(context.getFeatureOutput().value())
                                     .numHeads(4)
                                     .headDim(8)
                                     .ropeOptions(rope)
                                     .queryRopePositionOffset(100)
                                     .keyRopePositionOffset(0)
                                     .build();
    EXPECT_EQ(independent.getQueryRopePositionOffset(), 100);
    EXPECT_EQ(independent.getKeyRopePositionOffset(), 0);
    const nlohmann::json arch = independent.architectureJson();
    EXPECT_EQ(arch.at("rope_query_position_offset").get<int64_t>(), 100);
    EXPECT_EQ(arch.at("rope_key_position_offset").get<int64_t>(), 0);

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    auto restored = dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->getQueryRopePositionOffset(), 100);
    EXPECT_EQ(restored->getKeyRopePositionOffset(), 0);
}


TEST(AttentionApi, ArchitectureJsonAndDeserializePreserveReleaseCriticalOptions) {
    Api::Network network("attention_api_architecture_json_and_deserialize_preserve_release_critical_options");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({12, 96}).dataType(DataType::BF16).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 16;
    rope.base = 2048.0;
    rope.position_offset = 7;
    rope.interleaved = true;
    rope.scaling_kind = Impl::RotaryScalingKind::LongRope;
    rope.scaling_factor = 4.0;
    rope.original_max_position_embeddings = 8;
    rope.attention_factor = 1.125;
    rope.long_rope_short_factors = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
    rope.long_rope_long_factors = {2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7};
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(6)
                                   .numKeyValueHeads(2)
                                   .headDim(16)
                                   .valueDim(12)
                                   .outputFeatures(80)
                                   .hasBias(true)
                                   .ropeOptions(rope)
                                   .maskKind(Impl::AttentionMaskKind::SlidingWindowTopLeft)
                                   .diagonalLeftBound(3)
                                   .useAlibiMask(true)
                                   .attentionScale(0.25)
                                   .dropout(0.2f, 424242LL, 31337LL)
                                   .weightsDataType(DataType::BF16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::BF16)
                                   .build();

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_EQ(arch.at("layer_type").get<string>(), "attention");
    EXPECT_NE(arch.at("layer_type").get<string>(), "custom_layer");
    EXPECT_EQ(arch.at("num_heads").get<uint32_t>(), 6U);
    EXPECT_EQ(arch.at("num_key_value_heads").get<uint32_t>(), 2U);
    EXPECT_EQ(arch.at("head_dim").get<uint32_t>(), 16U);
    EXPECT_EQ(arch.at("value_dim").get<uint32_t>(), 12U);
    EXPECT_EQ(arch.at("output_features").get<uint32_t>(), 80U);
    EXPECT_TRUE(arch.at("has_bias").get<bool>());
    EXPECT_TRUE(arch.at("use_rope").get<bool>());
    EXPECT_FALSE(arch.at("rope_in_place").get<bool>());
    EXPECT_EQ(arch.at("rope_query_position_offset").get<int64_t>(), 7);
    EXPECT_EQ(arch.at("rope_key_position_offset").get<int64_t>(), 7);
    EXPECT_EQ(arch.at("mask_kind").get<string>(), "sliding_window_top_left");
    EXPECT_EQ(arch.at("diagonal_left_bound").get<int64_t>(), 3);
    EXPECT_TRUE(arch.at("use_alibi_mask").get<bool>());
    EXPECT_DOUBLE_EQ(arch.at("attention_scale").get<double>(), 0.25);
    EXPECT_FLOAT_EQ(arch.at("dropout_probability").get<float>(), 0.2f);
    EXPECT_EQ(arch.at("dropout_seed").get<int64_t>(), 424242LL);
    EXPECT_EQ(arch.at("dropout_offset").get<int64_t>(), 31337LL);
    EXPECT_EQ(arch.at("parameters").size(), 8U);

    const nlohmann::json ropeJson = arch.at("rope_options");
    EXPECT_EQ(ropeJson.at("rotary_dim").get<uint64_t>(), 16U);
    EXPECT_DOUBLE_EQ(ropeJson.at("base").get<double>(), 2048.0);
    EXPECT_EQ(ropeJson.at("position_offset").get<int64_t>(), 7);
    EXPECT_TRUE(ropeJson.at("interleaved").get<bool>());
    EXPECT_EQ(ropeJson.at("scaling_kind").get<string>(), "longrope");
    EXPECT_DOUBLE_EQ(ropeJson.at("scaling_factor").get<double>(), 4.0);
    EXPECT_EQ(ropeJson.at("original_max_position_embeddings").get<uint64_t>(), 8U);
    EXPECT_DOUBLE_EQ(ropeJson.at("attention_factor").get<double>(), 1.125);
    EXPECT_EQ(ropeJson.at("long_rope_short_factors").get<vector<double>>(), rope.long_rope_short_factors);
    EXPECT_EQ(ropeJson.at("long_rope_long_factors").get<vector<double>>(), rope.long_rope_long_factors);

    const nlohmann::json networkArch = network.architectureJson();
    ASSERT_EQ(networkArch.at("layers").size(), 2U);
    EXPECT_EQ(networkArch.at("layers").at(1).at("layer_type").get<string>(), "attention");

    const uint32_t previousTrainableLayerCount = network.getNumTrainableLayers();
    shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    ASSERT_EQ(network.getNumTrainableLayers(), previousTrainableLayerCount + 1);
    auto restored = dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(previousTrainableLayerCount));
    ASSERT_NE(restored, nullptr);

    EXPECT_EQ(restored->getNumHeads(), attention.getNumHeads());
    EXPECT_EQ(restored->getNumKeyValueHeads(), attention.getNumKeyValueHeads());
    EXPECT_EQ(restored->getHeadDim(), attention.getHeadDim());
    EXPECT_EQ(restored->getValueDim(), attention.getValueDim());
    EXPECT_EQ(restored->getOutputFeatures(), attention.getOutputFeatures());
    EXPECT_EQ(restored->getHasBias(), attention.getHasBias());
    EXPECT_EQ(restored->getUseRope(), attention.getUseRope());
    EXPECT_EQ(restored->getQueryRopePositionOffset(), 7);
    EXPECT_EQ(restored->getKeyRopePositionOffset(), 7);
    EXPECT_EQ(restored->getMaskKind(), attention.getMaskKind());
    EXPECT_EQ(restored->getDiagonalLeftBound(), attention.getDiagonalLeftBound());
    EXPECT_EQ(restored->getDiagonalRightBound(), attention.getDiagonalRightBound());
    EXPECT_EQ(restored->getUseAlibiMask(), attention.getUseAlibiMask());
    ASSERT_TRUE(restored->getAttentionScale().has_value());
    EXPECT_DOUBLE_EQ(restored->getAttentionScale().value(), 0.25);
    EXPECT_FLOAT_EQ(restored->getDropoutProbability(), attention.getDropoutProbability());
    EXPECT_EQ(restored->getDropoutSeed(), attention.getDropoutSeed());
    EXPECT_EQ(restored->getDropoutOffset(), attention.getDropoutOffset());

    const Impl::RotaryPositionEmbeddingOptions& restoredRope = restored->getRopeOptions();
    EXPECT_EQ(restoredRope.rotary_dim, rope.rotary_dim);
    EXPECT_DOUBLE_EQ(restoredRope.base, rope.base);
    EXPECT_EQ(restoredRope.position_offset, rope.position_offset);
    EXPECT_EQ(restoredRope.interleaved, rope.interleaved);
    EXPECT_EQ(restoredRope.scaling_kind, rope.scaling_kind);
    EXPECT_DOUBLE_EQ(restoredRope.scaling_factor, rope.scaling_factor);
    EXPECT_EQ(restoredRope.original_max_position_embeddings, rope.original_max_position_embeddings);
    ASSERT_TRUE(restoredRope.attention_factor.has_value());
    EXPECT_DOUBLE_EQ(restoredRope.attention_factor.value(), rope.attention_factor.value());
    EXPECT_EQ(restoredRope.long_rope_short_factors, rope.long_rope_short_factors);
    EXPECT_EQ(restoredRope.long_rope_long_factors, rope.long_rope_long_factors);

    nlohmann::json legacyArch = arch;
    legacyArch.erase("rope_query_position_offset");
    legacyArch.erase("rope_key_position_offset");
    const uint32_t beforeLegacyRestoreCount = network.getNumTrainableLayers();
    Api::Attention::deserialize(archiveReader, legacyArch, &network);
    auto legacyRestored = dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(beforeLegacyRestoreCount));
    ASSERT_NE(legacyRestored, nullptr);
    EXPECT_EQ(legacyRestored->getQueryRopePositionOffset(), rope.position_offset);
    EXPECT_EQ(legacyRestored->getKeyRopePositionOffset(), rope.position_offset);
}

TEST(AttentionApi, RejectsProjectionStorageDtypeMismatchInsteadOfDeferringToExpressionCompilation) {
    Api::Network network("attention_api_rejects_projection_storage_dtype_mismatch");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::BF16).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .weightsDataType(DataType::FP16)
                     .outputDataType(DataType::BF16)
                     .build(),
                 std::invalid_argument);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .weightsDataType(DataType::BF16)
                     .outputDataType(DataType::FP16)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsInvalidHeadConfiguration) {
    Api::Network network("attention_api_rejects_invalid_head_configuration");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(6)
                     .numKeyValueHeads(4)
                     .headDim(16)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsInvalidDropoutProbability) {
    Api::Network network("attention_api_rejects_invalid_dropout_probability");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .dropoutProbability(-0.01f)
                     .build(),
                 std::invalid_argument);

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .dropoutProbability(1.0f)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsRank3FeatureInputForComposedAttention) {
    Api::Network network("attention_api_rejects_rank3_feature_input_for_composed_attention");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({2, 8, 64}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::Attention::Builder().network(network).featureInput(input.getFeatureOutput().value()).numHeads(4).build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsBottomRightMaskWithAlibi) {
    Api::Network network("attention_api_rejects_bottom_right_mask_with_alibi");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .maskKind(Impl::AttentionMaskKind::SlidingWindowBottomRight)
                     .diagonalLeftBound(4)
                     .useAlibiMask()
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsBottomRightMaskWithDropout) {
    Api::Network network("attention_api_rejects_bottom_right_mask_with_dropout");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .maskKind(Impl::AttentionMaskKind::SlidingWindowBottomRight)
                     .diagonalLeftBound(4)
                     .dropout(0.1f, 7, 11)
                     .build(),
                 std::invalid_argument);
}


TEST(AttentionApi, BuildsComposedAttentionWithCanonicalRaggedTensorInput) {
    Api::Network network("attention_api_builds_composed_attention_with_canonical_ragged_tensor_input");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({64})
                                  .maxTotalValues(8)
                                  .batchSize(2)
                                  .build();

    Api::Attention attention = Api::Attention::Builder().network(network).featureInput(input).numHeads(4).headDim(16).build();

    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"feature_input", "query_row_partition", "key_value_row_partition"}));
    EXPECT_FALSE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRagged());
    ASSERT_TRUE(attention.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getRaggedFeatureInput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{8, 64}));
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getBatchSize(), 2u);
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getMaxTotalValues(), 8u);
}

TEST(AttentionApi, ArchitectureJsonAndDeserializePreserveCanonicalRaggedTensorInput) {
    Api::Network network("attention_api_architecture_json_and_deserialize_preserve_canonical_ragged_tensor_input");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT64)
                                  .trailingDimensions({64})
                                  .maxTotalValues(8)
                                  .batchSize(2)
                                  .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input)
                                   .numHeads(4)
                                   .headDim(16)
                                   .attentionScale(0.25)
                                   .build();

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_FALSE(arch.at("use_sequence_lengths").get<bool>());
    EXPECT_TRUE(arch.at("use_ragged").get<bool>());
    EXPECT_FALSE(arch.contains("use_ragged_offsets"));
    EXPECT_FALSE(arch.contains("query_ragged_offsets_input"));
    EXPECT_FALSE(arch.contains("key_value_ragged_offsets_input"));
    ASSERT_TRUE(arch.contains("ragged_feature_input"));
    ASSERT_TRUE(arch.contains("ragged_feature_output"));
    EXPECT_EQ(arch.at("ragged_feature_input").at("offsets").at("data_type").get<DataType>(), DataType::UINT64);

    const uint32_t previousTrainableLayerCount = network.getNumTrainableLayers();
    shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    ASSERT_EQ(network.getNumTrainableLayers(), previousTrainableLayerCount + 1);
    auto restored = dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(previousTrainableLayerCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->getUseSequenceLengths());
    EXPECT_TRUE(restored->getUseRagged());
    ASSERT_TRUE(restored->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(restored->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(restored->getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(restored->getRaggedFeatureOutput()->getOffsets(), restored->getRaggedFeatureInput()->getOffsets());
    EXPECT_EQ(restored->getInputNames(),
              (std::vector<std::string>{"feature_input", "query_row_partition", "key_value_row_partition"}));
}

TEST(AttentionApi, DeserializeRejectsRemovedRawRaggedMetadataFields) {
    Api::Network network("attention_api_deserialize_rejects_removed_raw_ragged_metadata_fields");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({64})
                                  .maxTotalValues(8)
                                  .batchSize(2)
                                  .build();
    Api::Attention attention = Api::Attention::Builder().network(network).featureInput(input).numHeads(4).headDim(16).build();
    const nlohmann::json arch = attention.architectureJson();
    shared_ptr<thor_file::TarReader> archiveReader;

    for (const char* removedField : {"ragged_offsets_input",
                                     "use_separate_ragged_offsets",
                                     "use_ragged_offsets",
                                     "query_ragged_offsets_input",
                                     "key_value_ragged_offsets_input"}) {
        nlohmann::json invalid = arch;
        invalid[removedField] = true;
        EXPECT_THROW(Api::Attention::deserialize(archiveReader, invalid, &network), std::runtime_error) << removedField;
    }
}

TEST(AttentionApi, BuildsRaggedCrossAttentionWithIndependentPartitionsWithoutRope) {
    Api::Network network("attention_api_builds_ragged_cross_attention_with_independent_partitions_without_rope");
    Api::RaggedTensor decoder = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("decoder")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT32)
                                    .trailingDimensions({32})
                                    .maxTotalValues(5)
                                    .batchSize(2)
                                    .build();
    Api::RaggedTensor encoder = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("encoder")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT64)
                                    .trailingDimensions({48})
                                    .maxTotalValues(7)
                                    .batchSize(2)
                                    .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(decoder)
                                   .contextInput(encoder)
                                   .numHeads(4)
                                   .numKeyValueHeads(2)
                                   .headDim(8)
                                   .valueDim(12)
                                   .outputFeatures(40)
                                   .dropout(0.125f, 17, 23)
                                   .build();

    EXPECT_TRUE(attention.getUseCrossAttention());
    EXPECT_TRUE(attention.getUseRagged());
    ASSERT_TRUE(attention.getRaggedContextInput().has_value());
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT32);
    EXPECT_EQ(attention.getRaggedContextInput()->getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{5, 40}));
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getOffsets(), decoder.getOffsets());
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"feature_input", "context_input", "query_row_partition", "key_value_row_partition"}));
}


TEST(AttentionApi, DenseQueryRaggedKvInfersMixedModeAndRoundTripsArchitecture) {
    Api::Network network("attention_api_dense_query_ragged_kv_round_trip");
    Api::NetworkInput query = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .dimensions({5, 32})
                                  .dataType(DataType::FP16)
                                  .build();
    Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT64)
                                    .trailingDimensions({32})
                                    .maxTotalValues(11)
                                    .batchSize(3)
                                    .build();
    Api::NetworkInput keyOrigins =
        Api::NetworkInput::Builder().network(network).name("key_origins").dimensions({1}).dataType(DataType::INT32).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.base = 10000.0;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(query.getFeatureOutput().value())
                                   .contextInput(context)
                                   .numHeads(4)
                                   .headDim(8)
                                   .ropeOptions(rope)
                                   .queryRopePositionOffset(371)
                                   .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                                   .build();

    EXPECT_TRUE(attention.getUseRagged());
    EXPECT_FALSE(attention.getQueryRagged());
    EXPECT_TRUE(attention.getKeyValueRagged());
    EXPECT_FALSE(attention.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(attention.getRaggedContextInput().has_value());
    EXPECT_FALSE(attention.getRaggedFeatureOutput().has_value());
    ASSERT_TRUE(attention.getFeatureOutput().has_value());
    EXPECT_EQ(attention.getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{5, 32}));
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"feature_input", "context_input", "key_value_row_partition", "key_rope_position_offsets"}));

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_TRUE(arch.at("use_ragged").get<bool>());
    EXPECT_FALSE(arch.at("query_ragged").get<bool>());
    EXPECT_TRUE(arch.at("key_value_ragged").get<bool>());
    EXPECT_FALSE(arch.contains("ragged_feature_input"));
    EXPECT_FALSE(arch.contains("ragged_feature_output"));
    EXPECT_TRUE(arch.contains("ragged_context_input"));
    EXPECT_EQ(arch.at("rope_query_position_offset").get<int64_t>(), 371);

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    auto restored = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->getQueryRagged());
    EXPECT_TRUE(restored->getKeyValueRagged());
    EXPECT_FALSE(restored->getRaggedFeatureOutput().has_value());
    ASSERT_TRUE(restored->getRaggedContextInput().has_value());
    EXPECT_EQ(restored->getRaggedContextInput()->getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(restored->getQueryRopePositionOffset(), 371);
}

TEST(AttentionApi, RaggedQueryDenseKvInfersMixedModeAndRoundTripsArchitecture) {
    Api::Network network("attention_api_ragged_query_dense_kv_round_trip");
    Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({32})
                                  .maxTotalValues(9)
                                  .batchSize(3)
                                  .build();
    Api::NetworkInput context = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .dimensions({7, 32})
                                    .dataType(DataType::FP16)
                                    .build();
    Api::NetworkInput queryOrigins =
        Api::NetworkInput::Builder().network(network).name("query_origins").dimensions({1}).dataType(DataType::INT32).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.base = 10000.0;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(query)
                                   .contextInput(context.getFeatureOutput().value())
                                   .numHeads(4)
                                   .headDim(8)
                                   .ropeOptions(rope)
                                   .queryRopePositionOffsetsInput(queryOrigins.getFeatureOutput().value())
                                   .keyRopePositionOffset(40)
                                   .build();

    EXPECT_TRUE(attention.getUseRagged());
    EXPECT_TRUE(attention.getQueryRagged());
    EXPECT_FALSE(attention.getKeyValueRagged());
    ASSERT_TRUE(attention.getRaggedFeatureInput().has_value());
    EXPECT_FALSE(attention.getRaggedContextInput().has_value());
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getOffsets(), query.getOffsets());
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"feature_input", "context_input", "query_row_partition", "query_rope_position_offsets"}));

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_TRUE(arch.at("query_ragged").get<bool>());
    EXPECT_FALSE(arch.at("key_value_ragged").get<bool>());
    EXPECT_TRUE(arch.contains("ragged_feature_input"));
    EXPECT_TRUE(arch.contains("ragged_feature_output"));
    EXPECT_FALSE(arch.contains("ragged_context_input"));

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    auto restored = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->getQueryRagged());
    EXPECT_FALSE(restored->getKeyValueRagged());
    ASSERT_TRUE(restored->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(restored->getRaggedFeatureOutput()->getOffsets(), restored->getRaggedFeatureInput()->getOffsets());
    EXPECT_EQ(restored->getKeyRopePositionOffset(), 40);
}


TEST(AttentionApi, MixedCrossAttentionModesPlaceWithRopeForTraining) {
    auto placeDenseQueryRaggedKv = []() {
        Api::Network network("attention_api_dense_query_ragged_kv_places_training");
        shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
        Api::NetworkInput query = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("query")
                                      .dimensions({4, 16})
                                      .dataType(DataType::FP16)
                                      .build();
        Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                        .network(network)
                                        .name("context")
                                        .valuesDataType(DataType::FP16)
                                        .offsetsDataType(DataType::UINT64)
                                        .trailingDimensions({16})
                                        .maxTotalValues(7)
                                        .batchSize(2)
                                        .build();
        Api::NetworkInput keyOrigins = Api::NetworkInput::Builder()
                                           .network(network)
                                           .name("key_origins")
                                           .dimensions({1})
                                           .dataType(DataType::INT32)
                                           .build();
        Impl::RotaryPositionEmbeddingOptions rope;
        rope.rotary_dim = 16;
        rope.compute_dtype = DataType::FP32;
        rope.output_dtype = DataType::FP16;
        Api::Attention attention = Api::Attention::Builder()
                                       .network(network)
                                       .featureInput(query.getFeatureOutput().value())
                                       .contextInput(context)
                                       .numHeads(1)
                                       .headDim(16)
                                       .ropeOptions(rope)
                                       .queryRopePositionOffset(371)
                                       .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                                       .optimizer(sgd)
                                       .build();
        Api::GradientRivet rivet =
            Api::GradientRivet::Builder().network(network).tensor(attention.getFeatureOutput().value()).build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(rivet.getFeatureOutput().value())
            .dataType(DataType::FP16)
            .build();
        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed = network.place(2, initDoneEvents, /*inferenceOnly=*/false);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
    };

    auto placeRaggedQueryDenseKv = []() {
        Api::Network network("attention_api_ragged_query_dense_kv_places_training");
        shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
        Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("query")
                                      .valuesDataType(DataType::FP16)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({16})
                                      .maxTotalValues(6)
                                      .batchSize(2)
                                      .build();
        Api::NetworkInput context = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("context")
                                        .dimensions({5, 16})
                                        .dataType(DataType::FP16)
                                        .build();
        Api::NetworkInput queryOrigins = Api::NetworkInput::Builder()
                                             .network(network)
                                             .name("query_origins")
                                             .dimensions({1})
                                             .dataType(DataType::INT32)
                                             .build();
        Impl::RotaryPositionEmbeddingOptions rope;
        rope.rotary_dim = 16;
        rope.compute_dtype = DataType::FP32;
        rope.output_dtype = DataType::FP16;
        Api::Attention attention = Api::Attention::Builder()
                                       .network(network)
                                       .featureInput(query)
                                       .contextInput(context.getFeatureOutput().value())
                                       .numHeads(1)
                                       .headDim(16)
                                       .ropeOptions(rope)
                                       .queryRopePositionOffsetsInput(queryOrigins.getFeatureOutput().value())
                                       .keyRopePositionOffset(40)
                                       .optimizer(sgd)
                                       .build();
        Api::GradientRivet rivet =
            Api::GradientRivet::Builder().network(network).tensor(attention.getRaggedFeatureOutput()->getValues()).build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(rivet.getFeatureOutput().value())
            .dataType(DataType::FP16)
            .build();
        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed = network.place(2, initDoneEvents, /*inferenceOnly=*/false);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
    };

    placeDenseQueryRaggedKv();
    placeRaggedQueryDenseKv();
}


TEST(AttentionApi, DenseQueryRaggedKvRopeMatchesUniformRaggedQueryReference) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t features = 16;
    constexpr uint64_t queryLength = 2;
    constexpr uint64_t queryCapacity = batchSize * queryLength;
    constexpr uint64_t contextCapacity = 7;
    constexpr int64_t historyBoundary = 371;

    Api::Network network("attention_api_dense_query_ragged_kv_rope_matches_uniform_ragged_query");
    Api::NetworkInput denseQuery = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("dense_query")
                                       .dimensions({queryLength, features})
                                       .dataType(DataType::FP16)
                                       .build();
    Api::RaggedTensor raggedQuery = Api::RaggedNetworkInput::Builder()
                                        .network(network)
                                        .name("ragged_query")
                                        .valuesDataType(DataType::FP16)
                                        .offsetsDataType(DataType::UINT32)
                                        .trailingDimensions({features})
                                        .maxTotalValues(queryCapacity)
                                        .batchSize(batchSize)
                                        .build();
    Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT64)
                                    .trailingDimensions({features})
                                    .maxTotalValues(contextCapacity)
                                    .batchSize(batchSize)
                                    .build();
    Api::NetworkInput raggedQueryOrigins = Api::NetworkInput::Builder()
                                               .network(network)
                                               .name("ragged_query_origins")
                                               .dimensions({1})
                                               .dataType(DataType::INT32)
                                               .build();
    Api::NetworkInput keyOrigins = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("key_origins")
                                       .dimensions({1})
                                       .dataType(DataType::INT32)
                                       .build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = features;
    rope.base = 10000.0;
    rope.compute_dtype = DataType::FP32;
    rope.output_dtype = DataType::FP16;

    Api::Attention mixed = Api::Attention::Builder()
                               .network(network)
                               .featureInput(denseQuery.getFeatureOutput().value())
                               .contextInput(context)
                               .numHeads(1)
                               .headDim(features)
                               .valueDim(features)
                               .outputFeatures(features)
                               .hasBias(false)
                               .ropeOptions(rope)
                               .queryRopePositionOffset(historyBoundary)
                               .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                               .weightsDataType(DataType::FP16)
                               .computeDataType(DataType::FP32)
                               .outputDataType(DataType::FP16)
                               .build();
    Api::Attention raggedReference = Api::Attention::Builder()
                                         .network(network)
                                         .featureInput(raggedQuery)
                                         .contextInput(context)
                                         .numHeads(1)
                                         .headDim(features)
                                         .valueDim(features)
                                         .outputFeatures(features)
                                         .hasBias(false)
                                         .ropeOptions(rope)
                                         .queryRopePositionOffsetsInput(raggedQueryOrigins.getFeatureOutput().value())
                                         .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                                         .weightsDataType(DataType::FP16)
                                         .computeDataType(DataType::FP32)
                                         .outputDataType(DataType::FP16)
                                         .build();

    Api::NetworkOutput mixedOutput = Api::NetworkOutput::Builder()
                                         .network(network)
                                         .name("mixed_output")
                                         .inputTensor(mixed.getFeatureOutput().value())
                                         .dataType(DataType::FP16)
                                         .build();
    Api::NetworkOutput raggedOutput = Api::NetworkOutput::Builder()
                                          .network(network)
                                          .name("ragged_output")
                                          .inputTensor(raggedReference.getRaggedFeatureOutput()->getValues())
                                          .dataType(DataType::FP16)
                                          .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);

    auto physicalMixed = std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(mixed.getId()));
    auto physicalReference =
        std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(raggedReference.getId()));
    auto physicalMixedOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(mixedOutput.getId()));
    auto physicalRaggedOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(raggedOutput.getId()));
    auto denseQueryInput = stamped.getNamedInput("dense_query");
    auto raggedQueryValuesInput = stamped.getNamedInput("ragged_query.values");
    auto raggedQueryOffsetsInput = stamped.getNamedInput("ragged_query.offsets");
    auto contextValuesInput = stamped.getNamedInput("context.values");
    auto contextOffsetsInput = stamped.getNamedInput("context.offsets");
    auto raggedQueryOriginsInput = stamped.getNamedInput("ragged_query_origins");
    auto keyOriginsInput = stamped.getNamedInput("key_origins");
    ASSERT_NE(physicalMixed, nullptr);
    ASSERT_NE(physicalReference, nullptr);
    ASSERT_NE(physicalMixedOutput, nullptr);
    ASSERT_NE(physicalRaggedOutput, nullptr);
    ASSERT_NE(denseQueryInput, nullptr);
    ASSERT_NE(raggedQueryValuesInput, nullptr);
    ASSERT_NE(raggedQueryOffsetsInput, nullptr);
    ASSERT_NE(contextValuesInput, nullptr);
    ASSERT_NE(contextOffsetsInput, nullptr);
    ASSERT_NE(raggedQueryOriginsInput, nullptr);
    ASSERT_NE(keyOriginsInput, nullptr);

    Stream stream = physicalMixed->getStreams()[0];
    const std::vector<float> identity = scaledIdentity(features, 1.0f);
    for (const auto& layer : {physicalMixed, physicalReference}) {
        setParameterTensor(layer->getParameter("query_weights"), identity, stream);
        setParameterTensor(layer->getParameter("key_weights"), identity, stream);
        setParameterTensor(layer->getParameter("value_weights"), identity, stream);
        setParameterTensor(layer->getParameter("output_weights"), identity, stream);
    }
    stream.synchronize();

    std::vector<float> queryValues(batchSize * queryLength * features, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint64_t q = 0; q < queryLength; ++q) {
            const uint64_t row = static_cast<uint64_t>(b) * queryLength + q;
            queryValues[row * features + 0] = 1.0f + static_cast<float>(b);
            queryValues[row * features + 1] = 0.25f * static_cast<float>(q + 1);
            queryValues[row * features + 4] = -0.5f + 0.1f * static_cast<float>(row);
        }
    }
    std::vector<float> contextValues(contextCapacity * features, 123.0f);
    const std::vector<uint64_t> contextLengths{2, 3};
    uint64_t packedRow = 0;
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint64_t t = 0; t < contextLengths[b]; ++t, ++packedRow) {
            std::fill(contextValues.begin() + packedRow * features,
                      contextValues.begin() + (packedRow + 1) * features,
                      0.0f);
            contextValues[packedRow * features + 0] = 0.5f + static_cast<float>(t);
            contextValues[packedRow * features + 1] = static_cast<float>(b + 1);
            contextValues[packedRow * features + 4] = 0.2f * static_cast<float>(packedRow + 1);
        }
    }

    Impl::Tensor denseQueryHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, queryLength, features}));
    Impl::Tensor raggedQueryValuesHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {queryCapacity, features}));
    Impl::Tensor raggedQueryOffsetsHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    Impl::Tensor contextValuesHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {contextCapacity, features}));
    Impl::Tensor contextOffsetsHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    Impl::Tensor raggedQueryOriginsHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {batchSize, 1}));
    Impl::Tensor keyOriginsHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {batchSize, 1}));
    writeCpuTensor(denseQueryHost, queryValues);
    writeCpuTensor(raggedQueryValuesHost, queryValues);
    writeCpuUint32Tensor(raggedQueryOffsetsHost, {0U, 2U, 4U});
    writeCpuTensor(contextValuesHost, contextValues);
    writeCpuUint64Tensor(contextOffsetsHost, {0ULL, 2ULL, 5ULL});
    writeCpuInt32Tensor(raggedQueryOriginsHost, {historyBoundary, historyBoundary});
    writeCpuInt32Tensor(keyOriginsHost,
                        {static_cast<int32_t>(historyBoundary - static_cast<int64_t>(contextLengths[0])),
                         static_cast<int32_t>(historyBoundary - static_cast<int64_t>(contextLengths[1]))});

    // This test drives physical NetworkInput layers directly rather than using
    // PlacedNetwork::infer(). Use the explicit ragged boundaries so the offsets
    // payload is materialized before its host cache is published.
    denseQueryInput->forward(denseQueryHost, false, batchSize);
    raggedQueryValuesInput->forwardRaggedValues(
        raggedQueryValuesHost, false, queryCapacity, batchSize);
    forwardPhysicalRowPartitionOffsets(
        *raggedQueryOffsetsInput,
        raggedQueryOffsetsHost,
        batchSize,
        queryCapacity,
        queryCapacity);
    contextValuesInput->forward(contextValuesHost, false, batchSize);
    contextOffsetsInput->forward(contextOffsetsHost, false, batchSize);
    raggedQueryOriginsInput->forward(raggedQueryOriginsHost, false, batchSize);
    keyOriginsInput->forward(keyOriginsHost, false, batchSize);
    physicalMixedOutput->getOutputReadyEvent().synchronize();
    physicalRaggedOutput->getOutputReadyEvent().synchronize();

    const std::vector<float> mixedValues = readCpuTensor(physicalMixedOutput->getFeatureOutput().value());
    const std::vector<float> referenceValues = readCpuTensor(physicalRaggedOutput->getFeatureOutput().value());
    ASSERT_EQ(mixedValues.size(), referenceValues.size());
    expectAllClose(mixedValues, referenceValues, 5.0e-2f, 5.0e-2f);
}


TEST(AttentionApi, DenseQueryRaggedKvMatchesRightAlignedPaddedMaskedReference) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t features = 16;
    constexpr uint64_t queryLength = 3;
    constexpr uint64_t historyExtent = 371;
    constexpr uint64_t raggedHistoryCapacity = 600;
    const std::vector<uint64_t> historyLengths{371, 187};

    Api::Network network("attention_api_dense_query_ragged_kv_matches_right_aligned_padded_reference");
    Api::NetworkInput query = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .dimensions({queryLength, features})
                                  .dataType(DataType::FP16)
                                  .build();
    Api::RaggedTensor raggedContext = Api::RaggedNetworkInput::Builder()
                                          .network(network)
                                          .name("ragged_context")
                                          .valuesDataType(DataType::FP16)
                                          .offsetsDataType(DataType::UINT64)
                                          .trailingDimensions({features})
                                          .maxTotalValues(raggedHistoryCapacity)
                                          .batchSize(batchSize)
                                          .build();
    Api::NetworkInput paddedContext = Api::NetworkInput::Builder()
                                          .network(network)
                                          .name("padded_context")
                                          .dimensions({historyExtent, features})
                                          .dataType(DataType::FP16)
                                          .build();
    Api::NetworkInput paddedMask = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("padded_mask")
                                       .dimensions({1, queryLength, historyExtent})
                                       .dataType(DataType::FP32)
                                       .build();
    Api::NetworkInput keyOrigins = Api::NetworkInput::Builder()
                                       .network(network)
                                       .name("key_origins")
                                       .dimensions({1})
                                       .dataType(DataType::INT32)
                                       .build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = features;
    rope.base = 10000.0;
    rope.compute_dtype = DataType::FP32;
    rope.output_dtype = DataType::FP16;

    Api::Attention nativeRagged = Api::Attention::Builder()
                                      .network(network)
                                      .featureInput(query.getFeatureOutput().value())
                                      .contextInput(raggedContext)
                                      .numHeads(1)
                                      .headDim(features)
                                      .valueDim(features)
                                      .outputFeatures(features)
                                      .hasBias(false)
                                      .ropeOptions(rope)
                                      .queryRopePositionOffset(historyExtent)
                                      .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                                      .dropoutProbability(0.0f)
                                      .weightsDataType(DataType::FP16)
                                      .computeDataType(DataType::FP32)
                                      .outputDataType(DataType::FP16)
                                      .build();
    Api::Attention paddedReference = Api::Attention::Builder()
                                         .network(network)
                                         .featureInput(query.getFeatureOutput().value())
                                         .contextInput(paddedContext.getFeatureOutput().value())
                                         .scoreBiasInput(paddedMask.getFeatureOutput().value())
                                         .numHeads(1)
                                         .headDim(features)
                                         .valueDim(features)
                                         .outputFeatures(features)
                                         .hasBias(false)
                                         .ropeOptions(rope)
                                         .queryRopePositionOffset(historyExtent)
                                         .keyRopePositionOffset(0)
                                         .dropoutProbability(0.0f)
                                         .weightsDataType(DataType::FP16)
                                         .computeDataType(DataType::FP32)
                                         .outputDataType(DataType::FP16)
                                         .build();

    Api::NetworkOutput nativeOutput = Api::NetworkOutput::Builder()
                                          .network(network)
                                          .name("native_output")
                                          .inputTensor(nativeRagged.getFeatureOutput().value())
                                          .dataType(DataType::FP16)
                                          .build();
    Api::NetworkOutput referenceOutput = Api::NetworkOutput::Builder()
                                             .network(network)
                                             .name("reference_output")
                                             .inputTensor(paddedReference.getFeatureOutput().value())
                                             .dataType(DataType::FP16)
                                             .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);

    auto physicalNative =
        std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(nativeRagged.getId()));
    auto physicalReference =
        std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(paddedReference.getId()));
    auto physicalNativeOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(nativeOutput.getId()));
    auto physicalReferenceOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(referenceOutput.getId()));
    auto queryInput = stamped.getNamedInput("query");
    auto raggedContextValuesInput = stamped.getNamedInput("ragged_context.values");
    auto raggedContextOffsetsInput = stamped.getNamedInput("ragged_context.offsets");
    auto paddedContextInput = stamped.getNamedInput("padded_context");
    auto paddedMaskInput = stamped.getNamedInput("padded_mask");
    auto keyOriginsInput = stamped.getNamedInput("key_origins");
    ASSERT_NE(physicalNative, nullptr);
    ASSERT_NE(physicalReference, nullptr);
    ASSERT_NE(physicalNativeOutput, nullptr);
    ASSERT_NE(physicalReferenceOutput, nullptr);
    ASSERT_NE(queryInput, nullptr);
    ASSERT_NE(raggedContextValuesInput, nullptr);
    ASSERT_NE(raggedContextOffsetsInput, nullptr);
    ASSERT_NE(paddedContextInput, nullptr);
    ASSERT_NE(paddedMaskInput, nullptr);
    ASSERT_NE(keyOriginsInput, nullptr);

    Stream stream = physicalNative->getStreams()[0];
    const std::vector<float> identity = scaledIdentity(features, 1.0f);
    for (const auto& layer : {physicalNative, physicalReference}) {
        setParameterTensor(layer->getParameter("query_weights"), identity, stream);
        setParameterTensor(layer->getParameter("key_weights"), identity, stream);
        setParameterTensor(layer->getParameter("value_weights"), identity, stream);
        setParameterTensor(layer->getParameter("output_weights"), identity, stream);
    }
    stream.synchronize();

    std::vector<float> queryValues(batchSize * queryLength * features, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint64_t q = 0; q < queryLength; ++q) {
            const uint64_t row = static_cast<uint64_t>(b) * queryLength + q;
            queryValues[row * features + 0] = 0.35f * static_cast<float>(row + 1);
            queryValues[row * features + 1] = 1.0f + 0.2f * static_cast<float>(b);
            queryValues[row * features + 4] = -0.3f + 0.1f * static_cast<float>(q);
        }
    }

    // The padded reference is the old representation: each history is right-aligned inside
    // [0, historyExtent), and a dense additive mask excludes the left padding. Padding values
    // are deliberately large so any mask regression becomes numerically obvious.
    std::vector<float> paddedContextValues(batchSize * historyExtent * features, 256.0f);
    std::vector<float> raggedContextValues(raggedHistoryCapacity * features, -512.0f);
    uint64_t packedRow = 0;
    for (uint32_t b = 0; b < batchSize; ++b) {
        const uint64_t firstValid = historyExtent - historyLengths[b];
        for (uint64_t t = 0; t < historyLengths[b]; ++t, ++packedRow) {
            const uint64_t denseRow = static_cast<uint64_t>(b) * historyExtent + firstValid + t;
            for (uint32_t f = 0; f < features; ++f) {
                paddedContextValues[denseRow * features + f] = 0.0f;
                raggedContextValues[packedRow * features + f] = 0.0f;
            }
            const float token = static_cast<float>(1 + b * 10 + t);
            paddedContextValues[denseRow * features + 0] = 0.2f * token;
            paddedContextValues[denseRow * features + 1] = -0.15f * token;
            paddedContextValues[denseRow * features + 4] = 0.05f * token;
            raggedContextValues[packedRow * features + 0] = 0.2f * token;
            raggedContextValues[packedRow * features + 1] = -0.15f * token;
            raggedContextValues[packedRow * features + 4] = 0.05f * token;
        }
    }
    ASSERT_EQ(packedRow, historyLengths[0] + historyLengths[1]);

    std::vector<float> maskValues(batchSize * queryLength * historyExtent, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        const uint64_t firstValid = historyExtent - historyLengths[b];
        for (uint64_t q = 0; q < queryLength; ++q) {
            for (uint64_t k = 0; k < firstValid; ++k) {
                const uint64_t index = (static_cast<uint64_t>(b) * queryLength + q) * historyExtent + k;
                maskValues[index] = -10000.0f;
            }
        }
    }

    Impl::Tensor queryHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, queryLength, features}));
    Impl::Tensor raggedContextHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {raggedHistoryCapacity, features}));
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    Impl::Tensor paddedContextHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, historyExtent, features}));
    Impl::Tensor paddedMaskHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1, queryLength, historyExtent}));
    Impl::Tensor keyOriginsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {batchSize, 1}));
    writeCpuTensor(queryHost, queryValues);
    writeCpuTensor(raggedContextHost, raggedContextValues);
    writeCpuUint64Tensor(raggedOffsetsHost, {0ULL, historyLengths[0], historyLengths[0] + historyLengths[1]});
    writeCpuTensor(paddedContextHost, paddedContextValues);
    writeCpuTensor(paddedMaskHost, maskValues);
    writeCpuInt32Tensor(keyOriginsHost,
                        {static_cast<int32_t>(historyExtent - historyLengths[0]),
                         static_cast<int32_t>(historyExtent - historyLengths[1])});

    queryInput->forward(queryHost, false, batchSize);
    raggedContextValuesInput->forward(raggedContextHost, false, batchSize);
    raggedContextOffsetsInput->forward(raggedOffsetsHost, false, batchSize);
    paddedContextInput->forward(paddedContextHost, false, batchSize);
    paddedMaskInput->forward(paddedMaskHost, false, batchSize);
    keyOriginsInput->forward(keyOriginsHost, false, batchSize);
    physicalNativeOutput->getOutputReadyEvent().synchronize();
    physicalReferenceOutput->getOutputReadyEvent().synchronize();

    const std::vector<float> nativeValues = readCpuTensor(physicalNativeOutput->getFeatureOutput().value());
    const std::vector<float> referenceValues = readCpuTensor(physicalReferenceOutput->getFeatureOutput().value());
    expectAllClose(nativeValues, referenceValues, 6.0e-2f, 6.0e-2f);
}

TEST(AttentionApi, RaggedQueryDenseKvMatchesRightAlignedPaddedQueryReference) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t features = 16;
    constexpr uint64_t queryExtent = 371;
    constexpr uint64_t raggedQueryCapacity = 600;
    constexpr uint64_t keyValueLength = 3;
    const std::vector<uint64_t> queryLengths{371, 187};

    Api::Network network("attention_api_ragged_query_dense_kv_matches_right_aligned_padded_reference");
    Api::RaggedTensor raggedQuery = Api::RaggedNetworkInput::Builder()
                                        .network(network)
                                        .name("ragged_query")
                                        .valuesDataType(DataType::FP16)
                                        .offsetsDataType(DataType::UINT32)
                                        .trailingDimensions({features})
                                        .maxTotalValues(raggedQueryCapacity)
                                        .batchSize(batchSize)
                                        .build();
    Api::NetworkInput paddedQuery = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("padded_query")
                                        .dimensions({queryExtent, features})
                                        .dataType(DataType::FP16)
                                        .build();
    Api::NetworkInput context = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .dimensions({keyValueLength, features})
                                    .dataType(DataType::FP16)
                                    .build();
    Api::NetworkInput queryOrigins = Api::NetworkInput::Builder()
                                         .network(network)
                                         .name("query_origins")
                                         .dimensions({1})
                                         .dataType(DataType::INT32)
                                         .build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = features;
    rope.base = 10000.0;
    rope.compute_dtype = DataType::FP32;
    rope.output_dtype = DataType::FP16;

    Api::Attention nativeRagged = Api::Attention::Builder()
                                      .network(network)
                                      .featureInput(raggedQuery)
                                      .contextInput(context.getFeatureOutput().value())
                                      .numHeads(1)
                                      .headDim(features)
                                      .valueDim(features)
                                      .outputFeatures(features)
                                      .hasBias(false)
                                      .ropeOptions(rope)
                                      .queryRopePositionOffsetsInput(queryOrigins.getFeatureOutput().value())
                                      .keyRopePositionOffset(queryExtent)
                                      .dropoutProbability(0.0f)
                                      .weightsDataType(DataType::FP16)
                                      .computeDataType(DataType::FP32)
                                      .outputDataType(DataType::FP16)
                                      .build();
    Api::Attention paddedReference = Api::Attention::Builder()
                                         .network(network)
                                         .featureInput(paddedQuery.getFeatureOutput().value())
                                         .contextInput(context.getFeatureOutput().value())
                                         .numHeads(1)
                                         .headDim(features)
                                         .valueDim(features)
                                         .outputFeatures(features)
                                         .hasBias(false)
                                         .ropeOptions(rope)
                                         .queryRopePositionOffset(0)
                                         .keyRopePositionOffset(queryExtent)
                                         .dropoutProbability(0.0f)
                                         .weightsDataType(DataType::FP16)
                                         .computeDataType(DataType::FP32)
                                         .outputDataType(DataType::FP16)
                                         .build();

    Api::NetworkOutput nativeOutput = Api::NetworkOutput::Builder()
                                          .network(network)
                                          .name("native_output")
                                          .inputTensor(nativeRagged.getRaggedFeatureOutput()->getValues())
                                          .dataType(DataType::FP16)
                                          .build();
    Api::NetworkOutput referenceOutput = Api::NetworkOutput::Builder()
                                             .network(network)
                                             .name("reference_output")
                                             .inputTensor(paddedReference.getFeatureOutput().value())
                                             .dataType(DataType::FP16)
                                             .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);

    auto physicalNative =
        std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(nativeRagged.getId()));
    auto physicalReference =
        std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(paddedReference.getId()));
    auto physicalNativeOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(nativeOutput.getId()));
    auto physicalReferenceOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(referenceOutput.getId()));
    auto raggedQueryValuesInput = stamped.getNamedInput("ragged_query.values");
    auto raggedQueryOffsetsInput = stamped.getNamedInput("ragged_query.offsets");
    auto paddedQueryInput = stamped.getNamedInput("padded_query");
    auto contextInput = stamped.getNamedInput("context");
    auto queryOriginsInput = stamped.getNamedInput("query_origins");
    ASSERT_NE(physicalNative, nullptr);
    ASSERT_NE(physicalReference, nullptr);
    ASSERT_NE(physicalNativeOutput, nullptr);
    ASSERT_NE(physicalReferenceOutput, nullptr);
    ASSERT_NE(raggedQueryValuesInput, nullptr);
    ASSERT_NE(raggedQueryOffsetsInput, nullptr);
    ASSERT_NE(paddedQueryInput, nullptr);
    ASSERT_NE(contextInput, nullptr);
    ASSERT_NE(queryOriginsInput, nullptr);

    Stream stream = physicalNative->getStreams()[0];
    const std::vector<float> identity = scaledIdentity(features, 1.0f);
    for (const auto& layer : {physicalNative, physicalReference}) {
        setParameterTensor(layer->getParameter("query_weights"), identity, stream);
        setParameterTensor(layer->getParameter("key_weights"), identity, stream);
        setParameterTensor(layer->getParameter("value_weights"), identity, stream);
        setParameterTensor(layer->getParameter("output_weights"), identity, stream);
    }
    stream.synchronize();

    std::vector<float> paddedQueryValues(batchSize * queryExtent * features, 384.0f);
    std::vector<float> raggedQueryValues(raggedQueryCapacity * features, -640.0f);
    uint64_t packedRow = 0;
    for (uint32_t b = 0; b < batchSize; ++b) {
        const uint64_t firstValid = queryExtent - queryLengths[b];
        for (uint64_t t = 0; t < queryLengths[b]; ++t, ++packedRow) {
            const uint64_t denseRow = static_cast<uint64_t>(b) * queryExtent + firstValid + t;
            std::fill(paddedQueryValues.begin() + denseRow * features,
                      paddedQueryValues.begin() + (denseRow + 1) * features,
                      0.0f);
            std::fill(raggedQueryValues.begin() + packedRow * features,
                      raggedQueryValues.begin() + (packedRow + 1) * features,
                      0.0f);
            const float token = static_cast<float>(1 + b * 10 + t);
            paddedQueryValues[denseRow * features + 0] = 0.1f * token;
            paddedQueryValues[denseRow * features + 1] = 0.25f + 0.05f * token;
            paddedQueryValues[denseRow * features + 4] = -0.08f * token;
            raggedQueryValues[packedRow * features + 0] = 0.1f * token;
            raggedQueryValues[packedRow * features + 1] = 0.25f + 0.05f * token;
            raggedQueryValues[packedRow * features + 4] = -0.08f * token;
        }
    }
    ASSERT_EQ(packedRow, queryLengths[0] + queryLengths[1]);

    std::vector<float> contextValues(batchSize * keyValueLength * features, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint64_t k = 0; k < keyValueLength; ++k) {
            const uint64_t row = static_cast<uint64_t>(b) * keyValueLength + k;
            const float token = static_cast<float>(1 + b * 7 + k);
            contextValues[row * features + 0] = 0.2f * token;
            contextValues[row * features + 1] = -0.1f * token;
            contextValues[row * features + 4] = 0.07f * token;
        }
    }

    Impl::Tensor raggedQueryHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {raggedQueryCapacity, features}));
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    Impl::Tensor paddedQueryHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, queryExtent, features}));
    Impl::Tensor contextHost(
        cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, keyValueLength, features}));
    Impl::Tensor queryOriginsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {batchSize, 1}));
    writeCpuTensor(raggedQueryHost, raggedQueryValues);
    writeCpuUint32Tensor(raggedOffsetsHost, {0U,
                                             static_cast<uint32_t>(queryLengths[0]),
                                             static_cast<uint32_t>(queryLengths[0] + queryLengths[1])});
    writeCpuTensor(paddedQueryHost, paddedQueryValues);
    writeCpuTensor(contextHost, contextValues);
    writeCpuInt32Tensor(queryOriginsHost,
                        {static_cast<int32_t>(queryExtent - queryLengths[0]),
                         static_cast<int32_t>(queryExtent - queryLengths[1])});

    const uint64_t activeQueryRows = queryLengths[0] + queryLengths[1];
    raggedQueryValuesInput->forwardRaggedValues(
        raggedQueryHost, false, activeQueryRows, batchSize);
    forwardPhysicalRowPartitionOffsets(
        *raggedQueryOffsetsInput,
        raggedOffsetsHost,
        batchSize,
        raggedQueryCapacity,
        activeQueryRows);
    paddedQueryInput->forward(paddedQueryHost, false, batchSize);
    contextInput->forward(contextHost, false, batchSize);
    queryOriginsInput->forward(queryOriginsHost, false, batchSize);
    physicalNativeOutput->getOutputReadyEvent().synchronize();
    physicalReferenceOutput->getOutputReadyEvent().synchronize();

    const std::vector<float> nativeValues = readCpuTensor(physicalNativeOutput->getFeatureOutput().value());
    const std::vector<float> referenceValues = readCpuTensor(physicalReferenceOutput->getFeatureOutput().value());
    ASSERT_EQ(nativeValues.size(), raggedQueryCapacity * features);
    ASSERT_EQ(referenceValues.size(), batchSize * queryExtent * features);

    uint64_t logicalRow = 0;
    for (uint32_t b = 0; b < batchSize; ++b) {
        const uint64_t firstValid = queryExtent - queryLengths[b];
        for (uint64_t t = 0; t < queryLengths[b]; ++t, ++logicalRow) {
            const uint64_t denseRow = static_cast<uint64_t>(b) * queryExtent + firstValid + t;
            const std::vector<float> nativeRow(nativeValues.begin() + logicalRow * features,
                                               nativeValues.begin() + (logicalRow + 1) * features);
            const std::vector<float> referenceRow(referenceValues.begin() + denseRow * features,
                                                  referenceValues.begin() + (denseRow + 1) * features);
            expectAllClose(nativeRow, referenceRow, 6.0e-2f, 6.0e-2f);
        }
    }
}

TEST(AttentionApi, RaggedRopeAllowsSelfAttentionAndIndependentCrossPartitions) {
    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 16;
    rope.base = 10000.0;

    {
        Api::Network network("attention_api_ragged_rope_allows_self_attention");
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP16)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({64})
                                      .maxTotalValues(8)
                                      .batchSize(2)
                                      .build();
        EXPECT_NO_THROW(Api::Attention::Builder()
                            .network(network)
                            .featureInput(input)
                            .numHeads(4)
                            .headDim(16)
                            .ropeOptions(rope)
                            .build());
    }

    {
        Api::Network network("attention_api_ragged_rope_allows_independent_cross_partitions");
        Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("query")
                                      .valuesDataType(DataType::FP16)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({64})
                                      .maxTotalValues(8)
                                      .batchSize(2)
                                      .build();
        Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                        .network(network)
                                        .name("context")
                                        .valuesDataType(DataType::FP16)
                                        .offsetsDataType(DataType::UINT64)
                                        .trailingDimensions({64})
                                        .maxTotalValues(10)
                                        .batchSize(2)
                                        .build();
        Api::Attention attention = Api::Attention::Builder()
                                       .network(network)
                                       .featureInput(query)
                                       .contextInput(context)
                                       .numHeads(4)
                                       .headDim(16)
                                       .ropeOptions(rope)
                                       .build();
        EXPECT_EQ(attention.getInputNames(),
                  (std::vector<std::string>{"feature_input",
                                            "context_input",
                                            "query_row_partition",
                                            "key_value_row_partition"}));
        EXPECT_EQ(attention.getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT32);
        EXPECT_EQ(attention.getRaggedContextInput()->getOffsetsDataType(), DataType::UINT64);
    }
}

TEST(AttentionApi, RaggedRopePerRowOriginsArePublicInputsAndRoundTripThroughArchitectureJson) {
    Api::Network network("attention_api_ragged_rope_per_row_origins_round_trip");
    Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({32})
                                  .maxTotalValues(5)
                                  .batchSize(2)
                                  .build();
    Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT64)
                                    .trailingDimensions({32})
                                    .maxTotalValues(7)
                                    .batchSize(2)
                                    .build();
    Api::NetworkInput queryOrigins =
        Api::NetworkInput::Builder().network(network).name("query_origins").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput keyOrigins =
        Api::NetworkInput::Builder().network(network).name("key_origins").dimensions({1}).dataType(DataType::INT32).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.base = 10000.0;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(query)
                                   .contextInput(context)
                                   .numHeads(4)
                                   .headDim(8)
                                   .ropeOptions(rope)
                                   .queryRopePositionOffsetsInput(queryOrigins.getFeatureOutput().value())
                                   .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                                   .build();

    ASSERT_TRUE(attention.getQueryRopePositionOffsetsInput().has_value());
    ASSERT_TRUE(attention.getKeyRopePositionOffsetsInput().has_value());
    EXPECT_EQ(attention.getQueryRopePositionOffsetsInput().value(), queryOrigins.getFeatureOutput().value());
    EXPECT_EQ(attention.getKeyRopePositionOffsetsInput().value(), keyOrigins.getFeatureOutput().value());
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"feature_input",
                                        "context_input",
                                        "query_row_partition",
                                        "key_value_row_partition",
                                        "query_rope_position_offsets",
                                        "key_rope_position_offsets"}));

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_TRUE(arch.at("use_query_rope_position_offsets").get<bool>());
    EXPECT_TRUE(arch.at("use_key_rope_position_offsets").get<bool>());
    EXPECT_EQ(arch.at("query_rope_position_offsets_input").at("id").get<uint64_t>(),
              queryOrigins.getFeatureOutput()->getOriginalId());
    EXPECT_EQ(arch.at("key_rope_position_offsets_input").at("id").get<uint64_t>(),
              keyOrigins.getFeatureOutput()->getOriginalId());

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    auto restored = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    ASSERT_TRUE(restored->getQueryRopePositionOffsetsInput().has_value());
    ASSERT_TRUE(restored->getKeyRopePositionOffsetsInput().has_value());
    EXPECT_EQ(restored->getQueryRopePositionOffsetsInput().value(), queryOrigins.getFeatureOutput().value());
    EXPECT_EQ(restored->getKeyRopePositionOffsetsInput().value(), keyOrigins.getFeatureOutput().value());
    EXPECT_EQ(restored->getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT32);
    EXPECT_EQ(restored->getRaggedContextInput()->getOffsetsDataType(), DataType::UINT64);
}

TEST(AttentionApi, RaggedRopePerRowOriginsRejectInvalidInputsAndDenseMode) {
    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;

    {
        Api::Network network("attention_api_dense_rejects_per_row_rope_origins");
        Api::NetworkInput tokens =
            Api::NetworkInput::Builder().network(network).name("tokens").dimensions({4, 32}).dataType(DataType::FP16).build();
        Api::NetworkInput origins =
            Api::NetworkInput::Builder().network(network).name("origins").dimensions({1}).dataType(DataType::INT32).build();
        EXPECT_THROW(Api::Attention::Builder()
                         .network(network)
                         .featureInput(tokens.getFeatureOutput().value())
                         .numHeads(4)
                         .headDim(8)
                         .ropeOptions(rope)
                         .queryRopePositionOffsetsInput(origins.getFeatureOutput().value())
                         .build(),
                     std::invalid_argument);
    }

    {
        Api::Network network("attention_api_ragged_rejects_invalid_per_row_rope_origins");
        Api::RaggedTensor tokens = Api::RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("tokens")
                                       .valuesDataType(DataType::FP16)
                                       .trailingDimensions({32})
                                       .maxTotalValues(6)
                                       .batchSize(2)
                                       .build();
        Api::NetworkInput badDtype =
            Api::NetworkInput::Builder().network(network).name("bad_dtype").dimensions({1}).dataType(DataType::FP16).build();
        Api::NetworkInput badShape =
            Api::NetworkInput::Builder().network(network).name("bad_shape").dimensions({2}).dataType(DataType::INT32).build();

        EXPECT_THROW(Api::Attention::Builder()
                         .network(network)
                         .featureInput(tokens)
                         .numHeads(4)
                         .headDim(8)
                         .ropeOptions(rope)
                         .queryRopePositionOffsetsInput(badDtype.getFeatureOutput().value())
                         .build(),
                     std::invalid_argument);
        EXPECT_THROW(Api::Attention::Builder()
                         .network(network)
                         .featureInput(tokens)
                         .numHeads(4)
                         .headDim(8)
                         .ropeOptions(rope)
                         .queryRopePositionOffsetsInput(badShape.getFeatureOutput().value())
                         .build(),
                     std::invalid_argument);
    }

    {
        Api::Network network("attention_api_ragged_rope_requires_matching_logical_row_counts");
        Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("query")
                                      .valuesDataType(DataType::FP16)
                                      .trailingDimensions({32})
                                      .maxTotalValues(6)
                                      .batchSize(2)
                                      .build();
        Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                        .network(network)
                                        .name("context")
                                        .valuesDataType(DataType::FP16)
                                        .trailingDimensions({32})
                                        .maxTotalValues(9)
                                        .batchSize(3)
                                        .build();

        EXPECT_THROW(Api::Attention::Builder()
                         .network(network)
                         .featureInput(query)
                         .contextInput(context)
                         .numHeads(4)
                         .headDim(8)
                         .ropeOptions(rope)
                         .build(),
                     std::invalid_argument);
    }
}

TEST(AttentionApi, RaggedCrossAttentionRopePerRowOriginsExecuteWithIndependentPartitions) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t features = 16;
    constexpr uint64_t queryCapacity = 3;
    constexpr uint64_t keyCapacity = 6;

    Api::Network network("attention_api_ragged_cross_attention_rope_per_row_origins_execute");
    Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({features})
                                  .maxTotalValues(queryCapacity)
                                  .batchSize(batchSize)
                                  .build();
    Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT64)
                                    .trailingDimensions({features})
                                    .maxTotalValues(keyCapacity)
                                    .batchSize(batchSize)
                                    .build();
    Api::NetworkInput queryOrigins =
        Api::NetworkInput::Builder().network(network).name("query_origins").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput keyOrigins =
        Api::NetworkInput::Builder().network(network).name("key_origins").dimensions({1}).dataType(DataType::INT32).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = features;
    rope.base = 10000.0;
    rope.output_dtype = DataType::FP16;
    rope.compute_dtype = DataType::FP32;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(query)
                                   .contextInput(context)
                                   .numHeads(1)
                                   .headDim(features)
                                   .valueDim(features)
                                   .outputFeatures(features)
                                   .hasBias(false)
                                   .ropeOptions(rope)
                                   .queryRopePositionOffsetsInput(queryOrigins.getFeatureOutput().value())
                                   .keyRopePositionOffsetsInput(keyOrigins.getFeatureOutput().value())
                                   .attentionScale(2.0)
                                   .weightsDataType(DataType::FP16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::FP16)
                                   .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getRaggedFeatureOutput()->getValues())
                                    .dataType(DataType::FP16)
                                    .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);

    auto physicalAttention =
        std::dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(attention.getId()));
    auto physicalOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    auto queryValuesInput = stamped.getNamedInput("query.values");
    auto queryOffsetsInput = stamped.getNamedInput("query.offsets");
    auto contextValuesInput = stamped.getNamedInput("context.values");
    auto contextOffsetsInput = stamped.getNamedInput("context.offsets");
    auto queryOriginsInput = stamped.getNamedInput("query_origins");
    auto keyOriginsInput = stamped.getNamedInput("key_origins");
    ASSERT_NE(physicalAttention, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(queryValuesInput, nullptr);
    ASSERT_NE(queryOffsetsInput, nullptr);
    ASSERT_NE(contextValuesInput, nullptr);
    ASSERT_NE(contextOffsetsInput, nullptr);
    ASSERT_NE(queryOriginsInput, nullptr);
    ASSERT_NE(keyOriginsInput, nullptr);

    Stream stream = physicalAttention->getStreams()[0];
    const std::vector<float> identity = scaledIdentity(features, 1.0f);
    setParameterTensor(physicalAttention->getParameter("query_weights"), identity, stream);
    setParameterTensor(physicalAttention->getParameter("key_weights"), identity, stream);
    setParameterTensor(physicalAttention->getParameter("value_weights"), identity, stream);
    setParameterTensor(physicalAttention->getParameter("output_weights"), identity, stream);
    stream.synchronize();

    std::vector<float> queryValues(queryCapacity * features, 0.0f);
    // The two logical query rows are intentionally identical. Correct row-local RoPE therefore produces
    // identical outputs for them when their key/value rows and origins are likewise identical.
    queryValues[0 * features + 0] = 1.0f;
    queryValues[1 * features + 0] = 1.0f;

    std::vector<float> contextValues(keyCapacity * features, 0.0f);
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t first = row * 2;
        const uint64_t second = first + 1;
        contextValues[first * features + 0] = 1.0f;
        contextValues[second * features + 0] = 1.0f;
        // Q has zero in this rotary lane, so this does not affect Q·K. It does make the V mixture
        // visibly change when the Q origin moves relative to the two key positions.
        contextValues[second * features + 1] = 4.0f;
    }

    Impl::Tensor queryValuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {queryCapacity, features}));
    Impl::Tensor contextValuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {keyCapacity, features}));
    Impl::Tensor queryOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    Impl::Tensor contextOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    Impl::Tensor queryOriginsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {batchSize, 1}));
    Impl::Tensor keyOriginsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {batchSize, 1}));
    writeCpuTensor(queryValuesHost, queryValues);
    writeCpuTensor(contextValuesHost, contextValues);
    writeCpuUint32Tensor(queryOffsetsHost, {0U, 1U, 2U});
    writeCpuUint64Tensor(contextOffsetsHost, {0ULL, 2ULL, 4ULL});

    auto runWithOrigins = [&](const std::vector<int32_t>& queryOriginValues,
                              const std::vector<int32_t>& keyOriginValues) -> std::vector<float> {
        writeCpuInt32Tensor(queryOriginsHost, queryOriginValues);
        writeCpuInt32Tensor(keyOriginsHost, keyOriginValues);
        queryValuesInput->forwardRaggedValues(queryValuesHost, false, /*activeValueCount=*/2, batchSize);
        contextValuesInput->forward(contextValuesHost, false, batchSize);
        forwardPhysicalRowPartitionOffsets(
            *queryOffsetsInput,
            queryOffsetsHost,
            batchSize,
            queryCapacity,
            /*activeRows=*/2);
        contextOffsetsInput->forward(contextOffsetsHost, false, batchSize);
        queryOriginsInput->forward(queryOriginsHost, false, batchSize);
        keyOriginsInput->forward(keyOriginsHost, false, batchSize);
        Event ready = physicalOutput->getOutputReadyEvent();
        ready.synchronize();
        return readCpuTensor(physicalOutput->getFeatureOutput().value());
    };

    const std::vector<float> zeroOrigins = runWithOrigins({0, 0}, {0, 0});
    const std::vector<float> commonPerRowShift = runWithOrigins({5, 11}, {5, 11});
    const std::vector<float> queryShiftedByOne = runWithOrigins({1, 1}, {0, 0});

    ASSERT_GE(zeroOrigins.size(), static_cast<size_t>(2 * features));
    // Distinct packed starts are Q:[0,1,2] and K:[0,2,4]. If packed indices leaked into RoPE positions,
    // the otherwise-identical second row would not match the first.
    for (uint32_t d = 0; d < features; ++d) {
        EXPECT_NEAR(zeroOrigins[d], zeroOrigins[features + d], 5.0e-2f) << "feature " << d;
    }

    const std::vector<float> zeroActive(zeroOrigins.begin(), zeroOrigins.begin() + 2 * features);
    const std::vector<float> commonShiftActive(commonPerRowShift.begin(), commonPerRowShift.begin() + 2 * features);
    const std::vector<float> queryShiftedActive(queryShiftedByOne.begin(), queryShiftedByOne.begin() + 2 * features);
    // RoPE depends only on Q/K relative phase, so a common per-row absolute shift must leave attention unchanged.
    expectAllClose(commonShiftActive, zeroActive, 5.0e-2f, 5.0e-2f);
    // Moving only Q by one position changes the relative phase and must visibly change the value mixture.
    expectNotAllClose(queryShiftedActive, zeroActive, 8.0e-2f, 8.0e-2f);
}

TEST(AttentionApi, RaggedCrossAttentionWithRopeAllowsSharedPartition) {
    Api::Network network("attention_api_ragged_cross_attention_with_rope_allows_shared_partition");
    Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .valuesDataType(DataType::FP16)
                                  .trailingDimensions({32})
                                  .maxTotalValues(8)
                                  .batchSize(2)
                                  .build();
    Api::RaggedTensor context(query.getValues(), query.getOffsets());

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 8;
    rope.base = 10000.0;
    EXPECT_NO_THROW(Api::Attention::Builder()
                        .network(network)
                        .featureInput(query)
                        .contextInput(context)
                        .numHeads(4)
                        .headDim(8)
                        .ropeOptions(rope)
                        .build());
}

TEST(AttentionApi, RejectsInvalidVariableLengthInputs) {
    Api::Network network("attention_api_rejects_invalid_variable_length_inputs");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput badSequenceLengthsDtype =
        Api::NetworkInput::Builder().network(network).name("bad_sequence_lengths_dtype").dimensions({1}).dataType(DataType::FP16).build();
    Api::NetworkInput badSequenceLengthsShape =
        Api::NetworkInput::Builder().network(network).name("bad_sequence_lengths_shape").dimensions({2}).dataType(DataType::INT32).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .querySequenceLengthsInput(badSequenceLengthsDtype.getFeatureOutput().value())
                     .keyValueSequenceLengthsInput(badSequenceLengthsDtype.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .querySequenceLengthsInput(badSequenceLengthsShape.getFeatureOutput().value())
                     .keyValueSequenceLengthsInput(badSequenceLengthsShape.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);

    Api::RaggedTensor ragged = Api::RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("ragged")
                                   .valuesDataType(DataType::FP16)
                                   .trailingDimensions({64})
                                   .maxTotalValues(8)
                                   .batchSize(2)
                                   .build();
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(ragged)
                     .querySequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                     .keyValueSequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, ForwardWithCanonicalRaggedTensorMatchesPackedReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 4;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 24;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.21f;
    c.sequenceLengths = {4, 2};
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs denseInputs = makeAttentionReferenceInputs(c);
    AttentionReferenceInputs packedInputs = denseInputs;
    packedInputs.featureInput =
        packBsfRaggedStorage(denseInputs.featureInput, c.sequenceLengths, c.batchSize, c.sequenceLength, c.inputFeatures);
    const vector<float> expectedDense = attentionLayerReference(denseInputs, c);
    const vector<float> expectedPacked =
        packBsfRaggedStorage(expectedDense, c.sequenceLengths, c.batchSize, c.sequenceLength, c.outputFeatures);

    Api::Network network("attention_api_forward_with_canonical_ragged_tensor_matches_packed_reference");
    const uint64_t maxTotalValues = static_cast<uint64_t>(c.batchSize) * c.sequenceLength;
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(c.dataType)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({c.inputFeatures})
                                  .maxTotalValues(maxTotalValues)
                                  .batchSize(c.batchSize)
                                  .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input)
                                   .numHeads(c.numHeads)
                                   .numKeyValueHeads(c.numKeyValueHeads)
                                   .headDim(c.headDim)
                                   .valueDim(c.valueDim)
                                   .outputFeatures(c.outputFeatures)
                                   .hasBias(c.hasBias)
                                   .weightsDataType(c.dataType)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(c.dataType)
                                   .attentionScale(c.attentionScale)
                                   .build();
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getRaggedFeatureOutput()->getValues())
                                    .dataType(c.dataType)
                                    .build();

    std::shared_ptr<Api::NetworkInput> valuesApiInput;
    for (const auto& candidate : network.getExternalNetworkInputs()) {
        if (candidate != nullptr && candidate->getName() == "tokens.values") {
            valuesApiInput = candidate;
            break;
        }
    }
    ASSERT_NE(valuesApiInput, nullptr);

    PlacedAttentionFixture fixture = placeSingleAttentionNetwork(network, *valuesApiInput, output, attention, c.batchSize, true);
    auto physicalRaggedOffsetsInput = fixture.stampedNetwork->getNamedInput("tokens.offsets");
    ASSERT_NE(physicalRaggedOffsetsInput, nullptr);

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, denseInputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {maxTotalValues, c.inputFeatures}));
    writeCpuTensor(featureInHost, packedInputs.featureInput);

    const vector<uint32_t> offsets = canonicalRaggedRowOffsets(c.sequenceLengths);
    ASSERT_FALSE(offsets.empty());
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {c.batchSize + 1}));
    writeCpuUint32Tensor(raggedOffsetsHost, offsets);

    const vector<float> actual = runForwardWithRaggedRowPartitionRuntime(*fixture.physicalInput,
                                                             *physicalRaggedOffsetsInput,
                                                             *fixture.physicalOutput,
                                                             featureInHost,
                                                             raggedOffsetsHost,
                                                             c.batchSize);
    ASSERT_TRUE(fixture.physicalAttention->getFeatureOutput().has_value());
    expectAllClose(packedBsfRaggedValidValues(actual, c.sequenceLengths, c.outputFeatures),
                   packedBsfRaggedValidValues(expectedPacked, c.sequenceLengths, c.outputFeatures),
                   1.2e-1f,
                   1.2e-1f);
}


TEST(AttentionApi, RaggedAttentionResidualAddUsesOffsetsRuntimeWithoutValuesMetadata) {
    constexpr uint32_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint32_t features = 32;

    Api::Network network("attention_api_ragged_attention_residual_add_uses_offsets_runtime");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({features})
                                  .maxTotalValues(maxTotalValues)
                                  .batchSize(batchSize)
                                  .build();
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input)
                                   .numHeads(2)
                                   .headDim(16)
                                   .hasBias(false)
                                   .weightsDataType(DataType::FP16)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(DataType::FP16)
                                   .build();
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    Api::Add residual = Api::Add::Builder()
                            .network(network)
                            .left(attention.getRaggedFeatureOutput().value())
                            .right(input)
                            .build();
    ASSERT_TRUE(residual.getRaggedFeatureOutput().has_value());
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(residual.getRaggedFeatureOutput()->getValues())
                                    .dataType(DataType::FP16)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);

    auto physicalValuesInput = stamped.getNamedInput("tokens.values");
    auto physicalOffsetsInput = stamped.getNamedInput("tokens.offsets");
    auto physicalAttention = dynamic_pointer_cast<Impl::CustomLayer>(stamped.getPhysicalLayerFromApiLayer(attention.getId()));
    auto physicalResidual = dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamped.getPhysicalLayerFromApiLayer(residual.getId()));
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalValuesInput, nullptr);
    ASSERT_NE(physicalOffsetsInput, nullptr);
    ASSERT_NE(physicalAttention, nullptr);
    ASSERT_NE(physicalResidual, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_TRUE(physicalValuesInput->getFeatureOutput().has_value());
    ASSERT_TRUE(physicalOffsetsInput->getFeatureOutput().has_value());

    vector<float> packed(maxTotalValues * features, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint32_t column = 0; column < features; ++column) {
            packed[row * features + column] = 0.01f * static_cast<float>(1 + row * features + column);
        }
    }
    for (uint64_t row = activeRows; row < maxTotalValues; ++row) {
        for (uint32_t column = 0; column < features; ++column) {
            packed[row * features + column] = 4096.0f;
        }
    }

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {maxTotalValues, features}));
    writeCpuTensor(featureInHost, packed);

    Impl::Tensor offsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    writeCpuUint32Tensor(offsetsHost, {0, 2, 5});

    // A normal logical batch owns its runtime cache on the source offsets tensor.
    // StampedNetwork must explicitly transfer that cache to the statically placed
    // physical offsets allocation before forwarding the offsets payload.
    Impl::Tensor logicalValuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {maxTotalValues, features}));
    writeCpuTensor(logicalValuesHost, packed);
    Impl::Tensor logicalOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    writeCpuUint32Tensor(logicalOffsetsHost, {0, 2, 5});
    Impl::RaggedTensor logicalInput(logicalValuesHost, logicalOffsetsHost);
    logicalInput.getRowPartitionRuntime().setHostActiveValueCount(activeRows);
    Batch logicalBatch;
    logicalBatch.insert("tokens", logicalInput);
    const auto logicalOutputs = placed->infer(logicalBatch);
    ASSERT_TRUE(logicalOutputs.contains("output"));
    const std::optional<uint64_t> placedActiveRows =
        Impl::RowPartitionRuntime(
            physicalOffsetsInput->getFeatureOutput().value(),
            Impl::RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32))
            .getHostActiveValueCountIfAvailable();
    ASSERT_EQ(placedActiveRows, std::optional<uint64_t>(activeRows));

    // Drive the physical graph directly without any values-owned metadata.
    // The dedicated ragged NetworkInput boundaries canonicalize the values tail
    // and publish the offsets-owned host cache only after the offsets payload is
    // materialized. Attention and the downstream ragged Add must both obtain
    // their extent solely from that shared row-partition runtime.
    physicalValuesInput->forwardRaggedValues(featureInHost, false, activeRows, batchSize);
    forwardPhysicalRowPartitionOffsets(
        *physicalOffsetsInput,
        offsetsHost,
        batchSize,
        maxTotalValues,
        activeRows);
    Event outputReady = physicalOutput->getOutputReadyEvent();
    outputReady.synchronize();

    ASSERT_TRUE(physicalAttention->getFeatureOutput().has_value());
    ASSERT_TRUE(physicalResidual->getFeatureOutput().has_value());

    const vector<float> actual = readCpuTensor(physicalOutput->getFeatureOutput().value());
    ASSERT_EQ(actual.size(), maxTotalValues * features);
    for (uint64_t i = activeRows * features; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], 0.0f) << "inactive packed tail index " << i;
    }
}

TEST(AttentionApi, RaggedDynamicNtkUsesLongestLogicalRowNotPackedCapacity) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 8;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 16;
    c.ropeOptions.base = 37.0;
    c.ropeOptions.scaling_kind = Impl::RotaryScalingKind::DynamicNTK;
    c.ropeOptions.scaling_factor = 8.0;
    c.ropeOptions.original_max_position_embeddings = 4;
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.sequenceLengths = {4, 3};
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs denseInputs = makeRopeLayoutSentinelInputs(c);
    AttentionReferenceInputs packedInputs = denseInputs;
    packedInputs.featureInput =
        packBsfRaggedStorage(denseInputs.featureInput, c.sequenceLengths, c.batchSize, c.sequenceLength, c.inputFeatures);
    const vector<float> expectedDense = attentionLayerReference(denseInputs, c);
    const vector<float> expectedPacked =
        packBsfRaggedStorage(expectedDense, c.sequenceLengths, c.batchSize, c.sequenceLength, c.outputFeatures);

    Api::Network network("attention_api_ragged_dynamic_ntk_uses_longest_logical_row");
    // Capacity is 16 tokens while the active rows contain only 4 and 3 tokens. The old implementation
    // incorrectly used 16 for Dynamic-NTK and therefore crossed original_max_position_embeddings=4.
    // The deliberately small base/high scaling factor makes that wrong basis numerically obvious.
    const uint64_t maxTotalValues = static_cast<uint64_t>(c.batchSize) * c.sequenceLength;
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(c.dataType)
                                  .offsetsDataType(DataType::UINT64)
                                  .trailingDimensions({c.inputFeatures})
                                  .maxTotalValues(maxTotalValues)
                                  .batchSize(c.batchSize)
                                  .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input)
                                   .numHeads(c.numHeads)
                                   .numKeyValueHeads(c.numKeyValueHeads)
                                   .headDim(c.headDim)
                                   .valueDim(c.valueDim)
                                   .outputFeatures(c.outputFeatures)
                                   .hasBias(c.hasBias)
                                   .ropeOptions(c.ropeOptions)
                                   .weightsDataType(c.dataType)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(c.dataType)
                                   .attentionScale(c.attentionScale)
                                   .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getRaggedFeatureOutput()->getValues())
                                    .dataType(c.dataType)
                                    .build();

    std::shared_ptr<Api::NetworkInput> valuesApiInput;
    for (const auto& candidate : network.getExternalNetworkInputs()) {
        if (candidate != nullptr && candidate->getName() == "tokens.values") {
            valuesApiInput = candidate;
            break;
        }
    }
    ASSERT_NE(valuesApiInput, nullptr);

    PlacedAttentionFixture fixture = placeSingleAttentionNetwork(network, *valuesApiInput, output, attention, c.batchSize, true);
    auto physicalRaggedOffsetsInput = fixture.stampedNetwork->getNamedInput("tokens.offsets");
    ASSERT_NE(physicalRaggedOffsetsInput, nullptr);

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, denseInputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {maxTotalValues, c.inputFeatures}));
    writeCpuTensor(featureInHost, packedInputs.featureInput);
    const vector<uint32_t> offsets32 = canonicalRaggedRowOffsets(c.sequenceLengths);
    const vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT64, {c.batchSize + 1}));
    writeCpuUint64Tensor(raggedOffsetsHost, offsets);

    const vector<float> actual = runForwardWithRaggedRowPartitionRuntime(*fixture.physicalInput,
                                                             *physicalRaggedOffsetsInput,
                                                             *fixture.physicalOutput,
                                                             featureInHost,
                                                             raggedOffsetsHost,
                                                             c.batchSize);
    expectAllClose(packedBsfRaggedValidValues(actual, c.sequenceLengths, c.outputFeatures),
                   packedBsfRaggedValidValues(expectedPacked, c.sequenceLengths, c.outputFeatures),
                   1.2e-1f,
                   1.2e-1f);
}

TEST(AttentionApi, ForwardWithSequenceLengthsMatchesPaddingMaskReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 4;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.21f;
    c.sequenceLengths = {4, 2};
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs inputs = makeAttentionReferenceInputs(c);

    Api::Network network("attention_api_forward_with_sequence_lengths_matches_padding_mask_reference");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({c.sequenceLength, c.inputFeatures})
                                  .dataType(c.dataType)
                                  .build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .querySequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                     .keyValueSequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .numHeads(c.numHeads)
                                   .numKeyValueHeads(c.numKeyValueHeads)
                                   .headDim(c.headDim)
                                   .valueDim(c.valueDim)
                                   .outputFeatures(c.outputFeatures)
                                   .hasBias(c.hasBias)
                                   .weightsDataType(c.dataType)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(c.dataType)
                                   .attentionScale(c.attentionScale)
                                   .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getOutput("feature_output"))
                                    .dataType(c.dataType)
                                    .build();

    PlacedAttentionFixture fixture = placeSingleAttentionNetwork(network, input, output, attention, c.batchSize, true);
    auto physicalSequenceLengthsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(sequenceLengths.getId()));
    ASSERT_NE(physicalSequenceLengthsInput, nullptr);

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, inputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, inputs.featureInput);
    Impl::Tensor sequenceLengthsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 1}));
    writeCpuInt32Tensor(sequenceLengthsHost, c.sequenceLengths);

    const vector<float> expected = attentionLayerReference(inputs, c);
    const vector<float> actual = runForwardWithMetadata(
        *fixture.physicalInput, *physicalSequenceLengthsInput, *fixture.physicalOutput, featureInHost, sequenceLengthsHost, c.batchSize);
    expectAllClose(actual, expected, 1.2e-1f, 1.2e-1f);
}


TEST(AttentionApi, DenseVariableLengthLongRopeUsesActiveMaximumNotPaddedSequenceExtent) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 8;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 8;
    c.ropeOptions.base = 37.0;
    c.ropeOptions.scaling_kind = Impl::RotaryScalingKind::LongRope;
    c.ropeOptions.scaling_factor = 2.0;
    c.ropeOptions.original_max_position_embeddings = 4;
    c.ropeOptions.attention_factor = 1.0;
    c.ropeOptions.long_rope_short_factors = {1.0, 1.0, 1.0, 1.0};
    c.ropeOptions.long_rope_long_factors = {4.0, 4.0, 4.0, 4.0};
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.sequenceLengths = {4, 3};
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs inputs = makeRopeLayoutSentinelInputs(c);

    Api::Network network("attention_api_dense_variable_length_longrope_uses_active_maximum");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({c.sequenceLength, c.inputFeatures})
                                  .dataType(c.dataType)
                                  .build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .querySequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .keyValueSequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .numHeads(c.numHeads)
                                   .numKeyValueHeads(c.numKeyValueHeads)
                                   .headDim(c.headDim)
                                   .valueDim(c.valueDim)
                                   .outputFeatures(c.outputFeatures)
                                   .hasBias(c.hasBias)
                                   .ropeOptions(c.ropeOptions)
                                   .weightsDataType(c.dataType)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(c.dataType)
                                   .attentionScale(c.attentionScale)
                                   .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getOutput("feature_output"))
                                    .dataType(c.dataType)
                                    .build();

    PlacedAttentionFixture fixture = placeSingleAttentionNetwork(network, input, output, attention, c.batchSize, true);
    auto physicalSequenceLengthsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(sequenceLengths.getId()));
    ASSERT_NE(physicalSequenceLengthsInput, nullptr);

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, inputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, inputs.featureInput);
    Impl::Tensor sequenceLengthsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 1}));
    writeCpuInt32Tensor(sequenceLengthsHost, c.sequenceLengths);

    const vector<float> expected = attentionLayerReference(inputs, c);
    const vector<float> actual = runForwardWithMetadata(
        *fixture.physicalInput, *physicalSequenceLengthsInput, *fixture.physicalOutput, featureInHost, sequenceLengthsHost, c.batchSize);
    expectAllClose(actual, expected, 1.2e-1f, 1.2e-1f);
}

TEST(AttentionApi, DropoutIsTrainingOnlyForValidationAndInference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 4;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.useRope = false;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs referenceInputs = makeAttentionReferenceInputs(c);
    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, referenceInputs.featureInput);

    auto runLayer = [&](const string& networkName,
                        float dropoutProbability,
                        bool inferenceOnly,
                        bool validationPass,
                        uint32_t forwardCount) {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .dimensions({c.sequenceLength, c.inputFeatures})
                                      .dataType(c.dataType)
                                      .build();
        Api::Attention::Builder builder;
        builder.network(network)
            .featureInput(input.getFeatureOutput().value())
            .numHeads(c.numHeads)
            .numKeyValueHeads(c.numKeyValueHeads)
            .headDim(c.headDim)
            .valueDim(c.valueDim)
            .outputFeatures(c.outputFeatures)
            .hasBias(c.hasBias)
            .weightsDataType(c.dataType)
            .computeDataType(DataType::FP32)
            .outputDataType(c.dataType)
            .attentionScale(c.attentionScale);
        if (dropoutProbability > 0.0f) {
            builder.dropout(dropoutProbability, 1234, 5678);
        }
        Api::Attention attention = builder.build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(attention.getOutput("feature_output"))
                                        .dataType(c.dataType)
                                        .build();

        if (!inferenceOnly) {
            // Validation and training-mode forward are exercised without a
            // backward/update step. Keep the training-capable graph while
            // explicitly freezing Attention's projection parameters.
            network.freezeTraining();
        }
        PlacedAttentionFixture fixture =
            placeSingleAttentionNetwork(network, input, output, attention, c.batchSize, inferenceOnly);
        Stream stream = fixture.physicalAttention->getStreams()[0];
        setAttentionParameters(fixture.physicalAttention, referenceInputs, c, stream);

        vector<vector<float>> results;
        for (uint32_t i = 0; i < forwardCount; ++i) {
            results.push_back(runForward(
                *fixture.physicalInput, *fixture.physicalOutput, featureInHost, c.batchSize, validationPass));
        }
        return results;
    };

    const vector<vector<float>> validationRuns =
        runLayer("attention_dropout_validation_bypass", 0.5f, false, true, 2);
    const vector<vector<float>> inferenceRuns =
        runLayer("attention_dropout_inference_bypass", 0.5f, true, false, 1);
    const vector<vector<float>> noDropoutControl =
        runLayer("attention_dropout_zero_control", 0.0f, true, false, 1);
    const vector<vector<float>> trainingRuns =
        runLayer("attention_dropout_training_enabled", 0.5f, false, false, 1);

    ASSERT_EQ(validationRuns.size(), 2u);
    ASSERT_EQ(inferenceRuns.size(), 1u);
    ASSERT_EQ(noDropoutControl.size(), 1u);
    ASSERT_EQ(trainingRuns.size(), 1u);

    expectAllClose(validationRuns[0], validationRuns[1], 1.0e-3f, 1.0e-3f);
    expectAllClose(validationRuns[0], inferenceRuns[0], 1.0e-3f, 1.0e-3f);
    expectAllClose(validationRuns[0], noDropoutControl[0], 1.0e-3f, 1.0e-3f);
    expectNotAllClose(trainingRuns[0], noDropoutControl[0], 1.0e-3f, 1.0e-3f);
}

TEST(AttentionApi, ForwardUniformAttentionMatchesBshdProjectionLayoutReference) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t sequenceLength = 3;
    constexpr uint32_t numHeads = 2;
    constexpr uint32_t headDim = 16;
    constexpr uint32_t valueDim = 16;
    constexpr uint32_t inputFeatures = numHeads * valueDim;
    constexpr uint32_t outputFeatures = numHeads * valueDim;
    const DataType dataType = DataType::FP16;

    Api::Network network("attention_api_forward_uniform_attention_matches_bshd_projection_layout_reference");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({sequenceLength, inputFeatures}).dataType(dataType).build();
    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(numHeads)
                                   .numKeyValueHeads(numHeads)
                                   .headDim(headDim)
                                   .valueDim(valueDim)
                                   .outputFeatures(outputFeatures)
                                   .hasBias(false)
                                   .weightsDataType(dataType)
                                   .computeDataType(DataType::FP32)
                                   .outputDataType(dataType)
                                   .attentionScale(1.0)
                                   .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(attention.getOutput("feature_output"))
                                    .dataType(dataType)
                                    .build();

    PlacedAttentionFixture fixture = placeSingleAttentionNetwork(network, input, output, attention, batchSize, true);
    ASSERT_EQ(fixture.stampedNetwork->getNumTrainableLayers(), 1u);
    if constexpr (Api::Attention::USE_PACKED_QKV_PROJECTION) {
        ASSERT_EQ(fixture.physicalAttention->listParameters(), (vector<string>{"qkv_weights", "output_weights"}));
    } else {
        ASSERT_EQ(fixture.physicalAttention->listParameters(),
                  (vector<string>{"query_weights", "key_weights", "value_weights", "output_weights"}));
    }

    vector<float> queryWeights(inputFeatures * numHeads * headDim, 0.0f);
    vector<float> keyWeights(inputFeatures * numHeads * headDim, 0.0f);
    vector<float> valueWeights(inputFeatures * numHeads * valueDim, 0.0f);
    vector<float> outputWeights(outputFeatures * outputFeatures, 0.0f);
    for (uint32_t i = 0; i < outputFeatures; ++i) {
        valueWeights[i * outputFeatures + i] = 1.0f;
        outputWeights[i * outputFeatures + i] = 1.0f;
    }

    AttentionReferenceCase parameterCase;
    parameterCase.batchSize = batchSize;
    parameterCase.sequenceLength = sequenceLength;
    parameterCase.inputFeatures = inputFeatures;
    parameterCase.outputFeatures = outputFeatures;
    parameterCase.numHeads = numHeads;
    parameterCase.numKeyValueHeads = numHeads;
    parameterCase.headDim = headDim;
    parameterCase.valueDim = valueDim;
    parameterCase.hasBias = false;
    parameterCase.useRope = false;
    parameterCase.dataType = dataType;

    AttentionReferenceInputs parameterInputs;
    parameterInputs.queryWeights = queryWeights;
    parameterInputs.keyWeights = keyWeights;
    parameterInputs.valueWeights = valueWeights;
    parameterInputs.outputWeights = outputWeights;

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, parameterInputs, parameterCase, stream);

    vector<float> inputValues(batchSize * sequenceLength * inputFeatures, 0.0f);
    auto inputIndex = [=](uint32_t b, uint32_t s, uint32_t feature) { return (b * sequenceLength + s) * inputFeatures + feature; };
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < sequenceLength; ++s) {
            for (uint32_t h = 0; h < numHeads; ++h) {
                for (uint32_t d = 0; d < valueDim; ++d) {
                    const uint32_t feature = h * valueDim + d;
                    inputValues[inputIndex(b, s, feature)] = 0.25f * static_cast<float>(b + 1) + 0.10f * static_cast<float>(s) +
                                                             0.03f * static_cast<float>(h) + 0.001f * static_cast<float>(d);
                }
            }
        }
    }

    vector<float> expected(batchSize * sequenceLength * outputFeatures, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < sequenceLength; ++s) {
            for (uint32_t h = 0; h < numHeads; ++h) {
                for (uint32_t d = 0; d < valueDim; ++d) {
                    const uint32_t feature = h * valueDim + d;
                    float sum = 0.0f;
                    for (uint32_t t = 0; t < sequenceLength; ++t)
                        sum += inputValues[inputIndex(b, t, feature)];
                    expected[inputIndex(b, s, feature)] = sum / static_cast<float>(sequenceLength);
                }
            }
        }
    }

    // The sentinel must vary across the true sequence axis.  Otherwise a layout
    // bug that swaps sequence/head semantics could still produce the same answer.
    ASSERT_NE(expected[inputIndex(0, 0, 0)], inputValues[inputIndex(0, 0, 0)]);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, sequenceLength, inputFeatures}));
    writeCpuTensor(featureInHost, inputValues);

    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    expectAllClose(actual, expected);
}

TEST(AttentionApi, ForwardMhaNoBiasMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 3;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_mha_no_bias_matches_full_cpu_reference", c, 1.1e-1f, 1.1e-1f);
}

TEST(AttentionApi, ForwardGqaWithBiasAndCausalMaskMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 3;
    c.numHeads = 4;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 64;
    c.outputFeatures = 24;
    c.hasBias = true;
    c.maskKind = Impl::AttentionMaskKind::CausalTopLeft;
    c.attentionScale = 0.20f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_gqa_with_bias_and_causal_mask_matches_full_cpu_reference", c, 1.2e-1f, 1.2e-1f);
}

TEST(AttentionApi, ForwardMqaWithBiasMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 3;
    c.numHeads = 4;
    c.numKeyValueHeads = 1;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 64;
    c.outputFeatures = 40;
    c.hasBias = true;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.20f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_mqa_with_bias_matches_full_cpu_reference", c, 1.2e-1f, 1.2e-1f);
}

TEST(AttentionApi, ForwardRopeMhaMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 3;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 8;
    c.ropeOptions.base = 1000.0;
    c.ropeOptions.position_offset = 1;
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_rope_mha_matches_full_cpu_reference", c, 1.2e-1f, 1.2e-1f);
}

TEST(AttentionApi, ForwardRopeIndependentQueryKeyPositionOffsetsMatchFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 4;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 16;
    c.ropeOptions.base = 37.0;
    c.ropeOptions.position_offset = 3;
    c.ropeOptions.scaling_kind = Impl::RotaryScalingKind::DynamicNTK;
    c.ropeOptions.scaling_factor = 2.0;
    c.ropeOptions.original_max_position_embeddings = 4;
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.queryRopePositionOffset = 11;
    c.keyRopePositionOffset = 0;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase(
        "attention_api_forward_rope_independent_query_key_offsets_match_full_cpu_reference", c, 1.2e-1f, 1.2e-1f);
}

TEST(AttentionApi, ForwardRopeLayoutSentinelMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 5;
    c.numHeads = 4;
    c.numKeyValueHeads = 4;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 64;
    c.outputFeatures = 64;
    c.hasBias = false;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 16;
    c.ropeOptions.base = 37.0;
    c.ropeOptions.position_offset = 1;
    c.ropeOptions.interleaved = true;
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.25f;
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs inputs = makeRopeLayoutSentinelInputs(c);
    const vector<float> expected = attentionLayerReference(inputs, c);
    const vector<float> badLayoutReference = attentionLayerReferenceWithRopeAppliedAfterBadBshdReinterpret(inputs, c);
    expectNotAllClose(expected, badLayoutReference, 2.5e-1f, 2.5e-1f);

    runAttentionApiReferenceCaseWithInputs(
        "attention_api_forward_rope_layout_sentinel_matches_full_cpu_reference", c, inputs, 2.0e-1f, 2.0e-1f);
}

TEST(AttentionApi, ForwardBf16MhaWithBiasMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 4;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 24;
    c.hasBias = true;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.17f;
    c.dataType = DataType::BF16;

    runAttentionApiReferenceCase("attention_api_forward_bf16_mha_with_bias_matches_full_cpu_reference", c, 1.8e-1f, 1.8e-1f);
}

TEST(AttentionApi, PlacedSaveLoadRoundTripRestoresBf16ProjectionBytesAndExecution) {
    constexpr uint32_t batchSize = 1;
    constexpr uint32_t sequenceLength = 2;
    constexpr uint32_t featureWidth = 16;
    const string networkName = "attention_bf16_parameter_round_trip";
    const filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    vector<float> inputValues(static_cast<uint64_t>(batchSize) * sequenceLength * featureWidth);
    for (uint64_t i = 0; i < inputValues.size(); ++i) {
        inputValues[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
    }

    const vector<string> parameterNames = {
        "query_weights",
        "key_weights",
        "value_weights",
        "output_weights",
        "query_bias",
        "key_bias",
        "value_bias",
        "output_bias",
    };
    const vector<vector<float>> parameterValues = {
        scaledIdentity(featureWidth, 0.5f),
        scaledIdentity(featureWidth, 0.75f),
        scaledIdentity(featureWidth, 1.25f),
        scaledIdentity(featureWidth, 0.8f),
        vector<float>(featureWidth, 0.03125f),
        vector<float>(featureWidth, -0.046875f),
        vector<float>(featureWidth, 0.0625f),
        vector<float>(featureWidth, -0.015625f),
    };

    try {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .dimensions({sequenceLength, featureWidth})
                                      .dataType(DataType::BF16)
                                      .build();
        Api::Attention attention = Api::Attention::Builder()
                                       .network(network)
                                       .featureInput(input.getFeatureOutput().value())
                                       .numHeads(1)
                                       .headDim(featureWidth)
                                       .valueDim(featureWidth)
                                       .outputFeatures(featureWidth)
                                       .hasBias(true)
                                       .weightsDataType(DataType::BF16)
                                       .computeDataType(DataType::FP32)
                                       .outputDataType(DataType::BF16)
                                       .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(attention.getFeatureOutput().value())
                                        .dataType(DataType::BF16)
                                        .build();

        PlacedAttentionFixture source =
            placeSingleAttentionNetwork(network, input, output, attention, batchSize, true);
        Stream sourceStream = source.physicalAttention->getStreams()[0];
        vector<vector<uint8_t>> sourceParameterBytes;
        sourceParameterBytes.reserve(parameterNames.size());
        for (uint64_t i = 0; i < parameterNames.size(); ++i) {
            auto parameter = source.physicalAttention->getParameter(parameterNames[i]);
            ASSERT_NE(parameter, nullptr);
            ASSERT_TRUE(parameter->getStorage().has_value());
            ASSERT_EQ(parameter->getStorage()->getDataType(), DataType::BF16);
            setParameterTensor(parameter, parameterValues[i], sourceStream);
        }
        sourceStream.synchronize();
        for (const string& parameterName : parameterNames) {
            auto parameter = source.physicalAttention->getParameter(parameterName);
            sourceParameterBytes.push_back(readTensorBytes(parameter->getStorage().value(), sourceStream));
        }

        Impl::Tensor featureInHost(
            cpuPlacement, Impl::TensorDescriptor(DataType::BF16, {batchSize, sequenceLength, featureWidth}));
        writeCpuTensor(featureInHost, inputValues);
        const vector<float> sourceOutput =
            runForward(*source.physicalInput, *source.physicalOutput, featureInHost, batchSize);
        EXPECT_TRUE(std::any_of(sourceOutput.begin(), sourceOutput.end(), [](float value) { return std::fabs(value) > 1.0e-3f; }));
        source.placedNetwork->save(archiveDir.string(), true, false);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        auto loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        auto loadedAttention = findOnlyLayerOfType<Api::Attention>(loadedNetwork);
        auto loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedAttention, nullptr);
        ASSERT_NE(loadedOutput, nullptr);

        const nlohmann::json loadedArchitecture = loadedAttention->architectureJson();
        EXPECT_EQ(loadedArchitecture.at("weights_data_type").get<DataType>(), DataType::BF16);
        EXPECT_EQ(loadedArchitecture.at("output_data_type").get<DataType>(), DataType::BF16);
        for (const string& parameterName : parameterNames) {
            EXPECT_EQ(loadedArchitecture.at("parameters").at(parameterName).at("dtype").get<DataType>(), DataType::BF16);
        }

        PlacedAttentionFixture loaded =
            placeSingleAttentionNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedAttention, batchSize, true);
        Stream loadedStream = loaded.physicalAttention->getStreams()[0];
        for (uint64_t i = 0; i < parameterNames.size(); ++i) {
            auto parameter = loaded.physicalAttention->getParameter(parameterNames[i]);
            ASSERT_NE(parameter, nullptr);
            ASSERT_TRUE(parameter->getStorage().has_value());
            ASSERT_EQ(parameter->getStorage()->getDataType(), DataType::BF16);
            EXPECT_EQ(readTensorBytes(parameter->getStorage().value(), loadedStream), sourceParameterBytes[i]);
        }

        const vector<float> loadedOutputValues =
            runForward(*loaded.physicalInput, *loaded.physicalOutput, featureInHost, batchSize);
        expectAllClose(loadedOutputValues, sourceOutput, 2e-2f, 2e-2f);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}

TEST(AttentionApi, ForwardCausalBottomRightNoBiasMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 5;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.maskKind = Impl::AttentionMaskKind::CausalBottomRight;
    c.attentionScale = 0.21f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_causal_bottom_right_no_bias_matches_full_cpu_reference", c, 1.2e-1f, 1.2e-1f);
}

TEST(AttentionApi, ForwardSlidingWindowTopLeftWithRightBoundMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 5;
    c.numHeads = 4;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 64;
    c.outputFeatures = 48;
    c.hasBias = true;
    c.maskKind = Impl::AttentionMaskKind::SlidingWindowTopLeft;
    c.diagonalLeftBound = 2;
    c.diagonalRightBound = 1;
    c.attentionScale = 0.19f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase(
        "attention_api_forward_sliding_window_top_left_with_right_bound_matches_full_cpu_reference", c, 1.3e-1f, 1.3e-1f);
}

TEST(AttentionApi, RejectsCausalTopLeftAlibiWithPositiveRightBound) {
    Api::Network network("attention_api_rejects_causal_top_left_alibi_positive_right_bound");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(4)
                     .maskKind(Impl::AttentionMaskKind::CausalTopLeft)
                     .diagonalRightBound(1)
                     .useAlibiMask()
                     .build(),
                 std::invalid_argument);
}


TEST(AttentionApi, ForwardCausalTopLeftWithAlibiMatchesFullCpuReferenceAndDiffersFromPlainMask) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 5;
    c.numHeads = 4;
    c.numKeyValueHeads = 4;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 64;
    c.outputFeatures = 40;
    c.hasBias = false;
    c.maskKind = Impl::AttentionMaskKind::CausalTopLeft;
    c.useAlibiMask = true;
    c.attentionScale = 0.18f;
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs inputs = makeAlibiSentinelInputs(c);
    AttentionReferenceCase plainMaskCase = c;
    plainMaskCase.useAlibiMask = false;
    const vector<float> expected = attentionLayerReference(inputs, c);
    const vector<float> plainMaskExpected = attentionLayerReference(inputs, plainMaskCase);
    expectNotAllClose(expected, plainMaskExpected, 2.0e-2f, 2.0e-2f);

    runAttentionApiReferenceCaseWithInputs(
        "attention_api_forward_causal_top_left_with_alibi_matches_full_cpu_reference", c, inputs, 1.5e-1f, 1.5e-1f);
}

TEST(AttentionApi, ForwardRopeLinearInverseGqaWithBiasMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 4;
    c.numHeads = 4;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 64;
    c.outputFeatures = 48;
    c.hasBias = true;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 8;
    c.ropeOptions.base = 256.0;
    c.ropeOptions.position_offset = 2;
    c.ropeOptions.inverse = true;
    c.ropeOptions.scaling_kind = Impl::RotaryScalingKind::Linear;
    c.ropeOptions.scaling_factor = 2.0;
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.18f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_rope_linear_inverse_gqa_with_bias_matches_full_cpu_reference", c, 1.5e-1f, 1.5e-1f);
}

TEST(AttentionApi, ForwardRopeDynamicNtkInterleavedMhaMatchesFullCpuReference) {
    AttentionReferenceCase c;
    c.batchSize = 2;
    c.sequenceLength = 6;
    c.numHeads = 2;
    c.numKeyValueHeads = 2;
    c.headDim = 16;
    c.valueDim = 16;
    c.inputFeatures = 32;
    c.outputFeatures = 32;
    c.hasBias = false;
    c.useRope = true;
    c.ropeOptions.rotary_dim = 8;
    c.ropeOptions.base = 10000.0;
    c.ropeOptions.position_offset = 1;
    c.ropeOptions.interleaved = true;
    c.ropeOptions.scaling_kind = Impl::RotaryScalingKind::DynamicNTK;
    c.ropeOptions.scaling_factor = 2.0;
    c.ropeOptions.original_max_position_embeddings = 4;
    c.ropeOptions.output_dtype = DataType::FP16;
    c.ropeOptions.compute_dtype = DataType::FP32;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.22f;
    c.dataType = DataType::FP16;

    runAttentionApiReferenceCase("attention_api_forward_rope_dynamic_ntk_interleaved_mha_matches_full_cpu_reference", c, 1.6e-1f, 1.6e-1f);
}
