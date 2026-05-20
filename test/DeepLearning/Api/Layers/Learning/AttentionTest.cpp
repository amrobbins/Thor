#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

#include <cuda_bf16.h>
#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::TensorDescriptor::DataType;

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
                         uint32_t batchSize) {
    physicalInput.forward(featureInHost, false, batchSize);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
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

vector<float> runForwardWithMetadata(Impl::NetworkInput& physicalInput,
                                     Impl::NetworkInput& physicalSequenceLengthsInput,
                                     Impl::NetworkInput& physicalRaggedOffsetsInput,
                                     Impl::NetworkOutput& physicalOutput,
                                     Impl::Tensor& featureInHost,
                                     Impl::Tensor& sequenceLengthsHost,
                                     Impl::Tensor& raggedOffsetsHost,
                                     uint32_t batchSize) {
    physicalInput.forward(featureInHost, false, batchSize);
    physicalSequenceLengthsInput.forward(sequenceLengthsHost, false, batchSize);
    physicalRaggedOffsetsInput.forward(raggedOffsetsHost, false, batchSize);
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

void applyRopeInPlace(vector<float>& bhsd, const AttentionReferenceCase& c, uint32_t heads, uint32_t dim) {
    Impl::RotaryPositionEmbeddingOptions opts = c.ropeOptions;
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
                if (opts.scaling_kind == Impl::RotaryScalingKind::DynamicNTK) {
                    const float ropeSeqLen = std::max(
                        static_cast<float>(c.sequenceLength) + static_cast<float>(std::max<int64_t>(0, opts.position_offset)), 1.0f);
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
                    const float theta =
                        ropePosition * powf(ropeBase, -2.0f * static_cast<float>(pairIndex) / static_cast<float>(rotaryDim));
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
            return k <= q;
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

vector<int32_t> raggedElementOffsets(const vector<int32_t>& lengths, uint32_t width) {
    vector<int32_t> offsets(lengths.size() + 1, 0);
    int64_t cursor = 0;
    for (uint64_t i = 0; i < lengths.size(); ++i) {
        cursor += static_cast<int64_t>(lengths[i]) * static_cast<int64_t>(width);
        offsets[i + 1] = static_cast<int32_t>(cursor);
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
        applyRopeInPlace(q, c, c.numHeads, c.headDim);
        applyRopeInPlace(k, c, c.numKeyValueHeads, c.headDim);
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
    applyRopeInPlace(qStorage, c, c.numHeads, c.headDim);
    applyRopeInPlace(kStorage, c, c.numKeyValueHeads, c.headDim);

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


TEST(AttentionApi, BuildsComposedAttentionWithPublicVariableLengthInputs) {
    Api::Network network("attention_api_builds_composed_attention_with_public_variable_length_inputs");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
                                   .numHeads(4)
                                   .headDim(16)
                                   .build();

    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"feature_input", "sequence_lengths", "ragged_offsets"}));
    EXPECT_TRUE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRaggedOffsets());
    ASSERT_TRUE(attention.getSequenceLengthsInput().has_value());
    ASSERT_TRUE(attention.getRaggedOffsetsInput().has_value());
    EXPECT_EQ(attention.getSequenceLengthsInput()->getDimensions(), (std::vector<uint64_t>{1}));
    EXPECT_EQ(attention.getRaggedOffsetsInput()->getDimensions(), (std::vector<uint64_t>{2}));
}

TEST(AttentionApi, ArchitectureJsonAndDeserializePreserveVariableLengthInputs) {
    Api::Network network("attention_api_architecture_json_and_deserialize_preserve_variable_length_inputs");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
                                   .numHeads(4)
                                   .headDim(16)
                                   .attentionScale(0.25)
                                   .build();

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_TRUE(arch.at("use_sequence_lengths").get<bool>());
    EXPECT_TRUE(arch.at("use_ragged_offsets").get<bool>());
    ASSERT_TRUE(arch.contains("sequence_lengths_input"));
    ASSERT_TRUE(arch.contains("ragged_offsets_input"));

    const uint32_t previousTrainableLayerCount = network.getNumTrainableLayers();
    shared_ptr<thor_file::TarReader> archiveReader;
    Api::Attention::deserialize(archiveReader, arch, &network);
    ASSERT_EQ(network.getNumTrainableLayers(), previousTrainableLayerCount + 1);
    auto restored = dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(previousTrainableLayerCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->getUseSequenceLengths());
    EXPECT_TRUE(restored->getUseRaggedOffsets());
    EXPECT_EQ(restored->getInputNames(), (std::vector<std::string>{"feature_input", "sequence_lengths", "ragged_offsets"}));
}

TEST(AttentionApi, BuildsCrossAttentionWithSeparateRaggedMetadata) {
    Api::Network network("attention_api_builds_cross_attention_with_separate_ragged_metadata");
    Api::NetworkInput decoder =
        Api::NetworkInput::Builder().network(network).name("decoder_tokens").dimensions({5, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput encoder =
        Api::NetworkInput::Builder().network(network).name("encoder_tokens").dimensions({7, 48}).dataType(DataType::FP16).build();
    Api::NetworkInput querySequenceLengths =
        Api::NetworkInput::Builder().network(network).name("query_sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput keyValueSequenceLengths = Api::NetworkInput::Builder()
                                                    .network(network)
                                                    .name("key_value_sequence_lengths")
                                                    .dimensions({1})
                                                    .dataType(DataType::INT32)
                                                    .build();
    Api::NetworkInput queryRaggedOffsets =
        Api::NetworkInput::Builder().network(network).name("query_ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();
    Api::NetworkInput keyValueRaggedOffsets = Api::NetworkInput::Builder()
                                                 .network(network)
                                                 .name("key_value_ragged_offsets")
                                                 .dimensions({2})
                                                 .dataType(DataType::INT32)
                                                 .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(decoder.getFeatureOutput().value())
                                   .contextInput(encoder.getFeatureOutput().value())
                                   .querySequenceLengthsInput(querySequenceLengths.getFeatureOutput().value())
                                   .keyValueSequenceLengthsInput(keyValueSequenceLengths.getFeatureOutput().value())
                                   .queryRaggedOffsetsInput(queryRaggedOffsets.getFeatureOutput().value())
                                   .keyValueRaggedOffsetsInput(keyValueRaggedOffsets.getFeatureOutput().value())
                                   .numHeads(4)
                                   .numKeyValueHeads(2)
                                   .headDim(8)
                                   .valueDim(8)
                                   .outputFeatures(40)
                                   .dropout(0.125f, 17, 23)
                                   .build();

    EXPECT_TRUE(attention.getUseCrossAttention());
    EXPECT_TRUE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRaggedOffsets());
    EXPECT_FALSE(attention.getRaggedOffsetsInput().has_value());
    ASSERT_TRUE(attention.getQueryRaggedOffsetsInput().has_value());
    ASSERT_TRUE(attention.getKeyValueRaggedOffsetsInput().has_value());
    EXPECT_EQ(attention.getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{5, 40}));
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"feature_input",
                                        "context_input",
                                        "query_sequence_lengths",
                                        "key_value_sequence_lengths",
                                        "query_ragged_offsets",
                                        "key_value_ragged_offsets"}));

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_TRUE(arch.at("use_cross_attention").get<bool>());
    EXPECT_TRUE(arch.at("use_sequence_lengths").get<bool>());
    EXPECT_TRUE(arch.at("use_separate_sequence_lengths").get<bool>());
    EXPECT_TRUE(arch.at("use_ragged_offsets").get<bool>());
    EXPECT_TRUE(arch.at("use_separate_ragged_offsets").get<bool>());
    EXPECT_FALSE(arch.contains("ragged_offsets_input"));
    ASSERT_TRUE(arch.contains("query_ragged_offsets_input"));
    ASSERT_TRUE(arch.contains("key_value_ragged_offsets_input"));
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
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();
    Api::NetworkInput badRaggedOffsetsDtype =
        Api::NetworkInput::Builder().network(network).name("bad_ragged_offsets_dtype").dimensions({2}).dataType(DataType::FP16).build();
    Api::NetworkInput badRaggedOffsetsShape =
        Api::NetworkInput::Builder().network(network).name("bad_ragged_offsets_shape").dimensions({1}).dataType(DataType::INT32).build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .sequenceLengthsInput(badSequenceLengthsDtype.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .sequenceLengthsInput(badSequenceLengthsShape.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                     .raggedOffsetsInput(badRaggedOffsetsDtype.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                     .raggedOffsetsInput(badRaggedOffsetsShape.getFeatureOutput().value())
                     .numHeads(4)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, BuildsRaggedOffsetsWithDropoutAndRope) {
    Api::Network network("attention_api_builds_ragged_offsets_with_dropout_and_rope");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("tokens").dimensions({8, 64}).dataType(DataType::FP16).build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 16;
    rope.base = 10000.0;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
                                   .numHeads(4)
                                   .headDim(16)
                                   .ropeOptions(rope)
                                   .dropout(0.125f, 7, 11)
                                   .build();

    EXPECT_TRUE(attention.getUseRaggedOffsets());
    EXPECT_TRUE(attention.getUseRope());
    EXPECT_FLOAT_EQ(attention.getDropoutProbability(), 0.125f);
    EXPECT_EQ(attention.getDropoutSeed(), 7LL);
    EXPECT_EQ(attention.getDropoutOffset(), 11LL);
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
                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
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

TEST(AttentionApi, ForwardWithRaggedOffsetsMatchesPackedReference) {
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

    const AttentionReferenceInputs denseInputs = makeAttentionReferenceInputs(c);
    AttentionReferenceInputs packedInputs = denseInputs;
    packedInputs.featureInput = packBsfRaggedStorage(denseInputs.featureInput, c.sequenceLengths, c.batchSize, c.sequenceLength, c.inputFeatures);
    const vector<float> expectedDense = attentionLayerReference(denseInputs, c);
    const vector<float> expectedPacked = packBsfRaggedStorage(expectedDense, c.sequenceLengths, c.batchSize, c.sequenceLength, c.outputFeatures);

    Api::Network network("attention_api_forward_with_ragged_offsets_matches_packed_reference");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({c.sequenceLength, c.inputFeatures})
                                  .dataType(c.dataType)
                                  .build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
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
    auto physicalRaggedOffsetsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(raggedOffsets.getId()));
    ASSERT_NE(physicalSequenceLengthsInput, nullptr);
    ASSERT_NE(physicalRaggedOffsetsInput, nullptr);

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, denseInputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, packedInputs.featureInput);
    Impl::Tensor sequenceLengthsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 1}));
    writeCpuInt32Tensor(sequenceLengthsHost, c.sequenceLengths);

    vector<int32_t> raggedHostValues(static_cast<uint64_t>(c.batchSize) * 2, 0);
    const vector<int32_t> offsets = raggedElementOffsets(c.sequenceLengths, c.numHeads * c.headDim);
    std::copy(offsets.begin(), offsets.end(), raggedHostValues.begin());
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 2}));
    writeCpuInt32Tensor(raggedOffsetsHost, raggedHostValues);

    const vector<float> actual = runForwardWithMetadata(*fixture.physicalInput,
                                                        *physicalSequenceLengthsInput,
                                                        *physicalRaggedOffsetsInput,
                                                        *fixture.physicalOutput,
                                                        featureInHost,
                                                        sequenceLengthsHost,
                                                        raggedOffsetsHost,
                                                        c.batchSize);
    expectAllClose(packedBsfRaggedValidValues(actual, c.sequenceLengths, c.outputFeatures),
                   packedBsfRaggedValidValues(expectedPacked, c.sequenceLengths, c.outputFeatures),
                   1.2e-1f,
                   1.2e-1f);
}

TEST(AttentionApi, ForwardWithRaggedOffsetsAndRopeMatchesPackedReference) {
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
    c.ropeOptions.base = 10000.0;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.21f;
    c.sequenceLengths = {4, 2};
    c.dataType = DataType::FP16;

    const AttentionReferenceInputs denseInputs = makeAttentionReferenceInputs(c);
    AttentionReferenceInputs packedInputs = denseInputs;
    packedInputs.featureInput = packBsfRaggedStorage(denseInputs.featureInput, c.sequenceLengths, c.batchSize, c.sequenceLength, c.inputFeatures);
    const vector<float> expectedDense = attentionLayerReference(denseInputs, c);
    const vector<float> expectedPacked = packBsfRaggedStorage(expectedDense, c.sequenceLengths, c.batchSize, c.sequenceLength, c.outputFeatures);

    Api::Network network("attention_api_forward_with_ragged_offsets_and_rope_matches_packed_reference");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({c.sequenceLength, c.inputFeatures})
                                  .dataType(c.dataType)
                                  .build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                   .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
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
    auto physicalRaggedOffsetsInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(raggedOffsets.getId()));
    ASSERT_NE(physicalSequenceLengthsInput, nullptr);
    ASSERT_NE(physicalRaggedOffsetsInput, nullptr);

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setAttentionParameters(fixture.physicalAttention, denseInputs, c, stream);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, packedInputs.featureInput);
    Impl::Tensor sequenceLengthsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 1}));
    writeCpuInt32Tensor(sequenceLengthsHost, c.sequenceLengths);

    vector<int32_t> raggedHostValues(static_cast<uint64_t>(c.batchSize) * 2, 0);
    const vector<int32_t> offsets = raggedElementOffsets(c.sequenceLengths, c.numHeads * c.headDim);
    std::copy(offsets.begin(), offsets.end(), raggedHostValues.begin());
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 2}));
    writeCpuInt32Tensor(raggedOffsetsHost, raggedHostValues);

    const vector<float> actual = runForwardWithMetadata(*fixture.physicalInput,
                                                        *physicalSequenceLengthsInput,
                                                        *physicalRaggedOffsetsInput,
                                                        *fixture.physicalOutput,
                                                        featureInHost,
                                                        sequenceLengthsHost,
                                                        raggedOffsetsHost,
                                                        c.batchSize);
    expectAllClose(packedBsfRaggedValidValues(actual, c.sequenceLengths, c.outputFeatures),
                   packedBsfRaggedValidValues(expectedPacked, c.sequenceLengths, c.outputFeatures),
                   1.5e-1f,
                   1.5e-1f);
}

TEST(AttentionApi, ForwardWithRaggedOffsetsDropoutAndRopeAdvancesPhiloxOffset) {
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
    c.ropeOptions.base = 10000.0;
    c.maskKind = Impl::AttentionMaskKind::None;
    c.attentionScale = 0.21f;
    c.sequenceLengths = {4, 2};
    c.dataType = DataType::FP16;

    constexpr float dropoutProbability = 0.5f;
    constexpr int64_t dropoutSeed = 1234;
    constexpr int64_t initialDropoutOffset = 5678;
    const int64_t expectedOffsetAdvance = static_cast<int64_t>(c.batchSize) * static_cast<int64_t>(c.numHeads) *
                                          static_cast<int64_t>(c.sequenceLength) * static_cast<int64_t>(c.sequenceLength);
    const int64_t advancedDropoutOffset = initialDropoutOffset + expectedOffsetAdvance;

    const AttentionReferenceInputs denseInputs = makeAttentionReferenceInputs(c);
    AttentionReferenceInputs packedInputs = denseInputs;
    packedInputs.featureInput = packBsfRaggedStorage(denseInputs.featureInput, c.sequenceLengths, c.batchSize, c.sequenceLength, c.inputFeatures);

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(c.dataType, {c.batchSize, c.sequenceLength, c.inputFeatures}));
    writeCpuTensor(featureInHost, packedInputs.featureInput);
    Impl::Tensor sequenceLengthsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 1}));
    writeCpuInt32Tensor(sequenceLengthsHost, c.sequenceLengths);

    vector<int32_t> raggedHostValues(static_cast<uint64_t>(c.batchSize) * 2, 0);
    const vector<int32_t> offsets = raggedElementOffsets(c.sequenceLengths, c.numHeads * c.headDim);
    std::copy(offsets.begin(), offsets.end(), raggedHostValues.begin());
    Impl::Tensor raggedOffsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::INT32, {c.batchSize, 2}));
    writeCpuInt32Tensor(raggedOffsetsHost, raggedHostValues);

    auto runPublicLayer = [&](const std::string& networkName, int64_t dropoutOffset, uint32_t forwardCount) {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .dimensions({c.sequenceLength, c.inputFeatures})
                                      .dataType(c.dataType)
                                      .build();
        Api::NetworkInput sequenceLengths = Api::NetworkInput::Builder()
                                                .network(network)
                                                .name("sequence_lengths")
                                                .dimensions({1})
                                                .dataType(DataType::INT32)
                                                .build();
        Api::NetworkInput raggedOffsets = Api::NetworkInput::Builder()
                                              .network(network)
                                              .name("ragged_offsets")
                                              .dimensions({2})
                                              .dataType(DataType::INT32)
                                              .build();

        Api::Attention attention = Api::Attention::Builder()
                                       .network(network)
                                       .featureInput(input.getFeatureOutput().value())
                                       .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                       .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
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
                                       .dropout(dropoutProbability, dropoutSeed, dropoutOffset)
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
        auto physicalRaggedOffsetsInput =
            dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(raggedOffsets.getId()));
        EXPECT_NE(physicalSequenceLengthsInput, nullptr);
        EXPECT_NE(physicalRaggedOffsetsInput, nullptr);
        if (physicalSequenceLengthsInput == nullptr || physicalRaggedOffsetsInput == nullptr) {
            return vector<vector<float>>{};
        }

        Stream stream = fixture.physicalAttention->getStreams()[0];
        setAttentionParameters(fixture.physicalAttention, denseInputs, c, stream);

        vector<vector<float>> validOutputs;
        for (uint32_t i = 0; i < forwardCount; ++i) {
            const vector<float> actual = runForwardWithMetadata(*fixture.physicalInput,
                                                                *physicalSequenceLengthsInput,
                                                                *physicalRaggedOffsetsInput,
                                                                *fixture.physicalOutput,
                                                                featureInHost,
                                                                sequenceLengthsHost,
                                                                raggedOffsetsHost,
                                                                c.batchSize);
            validOutputs.push_back(packedBsfRaggedValidValues(actual, c.sequenceLengths, c.outputFeatures));
        }
        return validOutputs;
    };

    const vector<vector<float>> managedRuns =
        runPublicLayer("attention_api_ragged_dropout_rope_managed_offset", initialDropoutOffset, 2);
    ASSERT_EQ(managedRuns.size(), 2u);

    const vector<vector<float>> initialOffsetControl =
        runPublicLayer("attention_api_ragged_dropout_rope_initial_offset_control", initialDropoutOffset, 1);
    ASSERT_EQ(initialOffsetControl.size(), 1u);

    const vector<vector<float>> advancedOffsetControl =
        runPublicLayer("attention_api_ragged_dropout_rope_advanced_offset_control", advancedDropoutOffset, 1);
    ASSERT_EQ(advancedOffsetControl.size(), 1u);

    // Fixed Philox seed/offset is deterministic across independently placed public Attention layers.
    expectAllClose(managedRuns[0], initialOffsetControl[0], 1.0e-3f, 1.0e-3f);

    // The public layer must advance its managed Philox offset between forward executions.  The second managed
    // execution should therefore match a fresh layer whose initial offset is exactly the first offset plus the
    // public Attention layer's conservative per-forward score-count advance.
    expectAllClose(managedRuns[1], advancedOffsetControl[0], 1.0e-3f, 1.0e-3f);

    // Guard the test fixture: if these controls are numerically identical, the equality checks above would not
    // prove that offset advancement is observable for this configuration.
    expectNotAllClose(initialOffsetControl[0], advancedOffsetControl[0], 1.0e-3f, 1.0e-3f);
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
