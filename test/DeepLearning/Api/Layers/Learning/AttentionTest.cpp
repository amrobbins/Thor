#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
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
    FAIL() << "sentinel references are too close; this test would not catch a RoPE layout mismatch";
}

float castToStorage(float value, DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
            return __half2float(__float2half(value));
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

    fixture.physicalInput = dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiInput.getId()));
    fixture.physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalAttention = dynamic_pointer_cast<Impl::CustomLayer>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiAttention.getId()));

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
    float attentionScale = 1.0f;
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
uint32_t mergedWidth(const AttentionReferenceCase& c) { return c.numHeads * c.valueDim; }

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
    if (c.numHeads != c.numKeyValueHeads || c.headDim != c.valueDim || c.inputFeatures != qWidth(c) ||
        c.outputFeatures != mergedWidth(c) || c.hasBias) {
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
                    const float ropeSeqLen = std::max(static_cast<float>(c.sequenceLength) +
                                                          static_cast<float>(std::max<int64_t>(0, opts.position_offset)),
                                                      1.0f);
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
                    const float theta = ropePosition * powf(ropeBase, -2.0f * static_cast<float>(pairIndex) / static_cast<float>(rotaryDim));
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
    switch (c.maskKind) {
        case Impl::AttentionMaskKind::None:
            return true;
        case Impl::AttentionMaskKind::CausalTopLeft:
            return keyIndex <= queryIndex;
        default:
            throw std::runtime_error("Unsupported mask kind in Attention API CPU reference test.");
    }
}

vector<float> sdpaReference(const vector<float>& q, const vector<float>& k, const vector<float>& v, const AttentionReferenceCase& c) {
    vector<float> out(static_cast<uint64_t>(c.batchSize) * c.numHeads * c.sequenceLength * c.valueDim, 0.0f);
    const uint32_t headsPerKvHead = c.numHeads / c.numKeyValueHeads;
    for (uint32_t b = 0; b < c.batchSize; ++b) {
        for (uint32_t h = 0; h < c.numHeads; ++h) {
            const uint32_t kvHead = h / headsPerKvHead;
            for (uint32_t sq = 0; sq < c.sequenceLength; ++sq) {
                vector<float> scores(c.sequenceLength, -std::numeric_limits<float>::infinity());
                float maxScore = -std::numeric_limits<float>::infinity();
                for (uint32_t sk = 0; sk < c.sequenceLength; ++sk) {
                    if (!attentionKeyAllowed(c, sq, sk))
                        continue;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < c.headDim; ++d) {
                        dot += q[idx4(b, h, sq, d, c.numHeads, c.sequenceLength, c.headDim)] *
                               k[idx4(b, kvHead, sk, d, c.numKeyValueHeads, c.sequenceLength, c.headDim)];
                    }
                    scores[sk] = dot * c.attentionScale;
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

vector<float> bhsdSemanticToBshdStorage(const vector<float>& bhsd,
                                           uint32_t batchSize,
                                           uint32_t sequenceLength,
                                           uint32_t heads,
                                           uint32_t dim) {
    vector<float> storage(static_cast<uint64_t>(batchSize) * sequenceLength * heads * dim, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t s = 0; s < sequenceLength; ++s) {
                for (uint32_t d = 0; d < dim; ++d) {
                    storage[idxBshd(b, s, h, d, sequenceLength, heads, dim)] =
                        bhsd[idx4(b, h, s, d, heads, sequenceLength, dim)];
                }
            }
        }
    }
    return storage;
}

vector<float> bshdStorageToBhsdSemantic(const vector<float>& storage,
                                         uint32_t batchSize,
                                         uint32_t sequenceLength,
                                         uint32_t heads,
                                         uint32_t dim) {
    vector<float> bhsd(static_cast<uint64_t>(batchSize) * heads * sequenceLength * dim, 0.0f);
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t s = 0; s < sequenceLength; ++s) {
                for (uint32_t d = 0; d < dim; ++d) {
                    bhsd[idx4(b, h, s, d, heads, sequenceLength, dim)] =
                        storage[idxBshd(b, s, h, d, sequenceLength, heads, dim)];
                }
            }
        }
    }
    return bhsd;
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

vector<float> outputProjectionReference(const vector<float>& merged, const AttentionReferenceInputs& inputs, const AttentionReferenceCase& c) {
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
    vector<float> q = projectToBhsd(inputs.featureInput,
                                    inputs.queryWeights,
                                    qBias,
                                    c.batchSize,
                                    c.sequenceLength,
                                    c.inputFeatures,
                                    c.numHeads,
                                    c.headDim,
                                    c.dataType);
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
    vector<float> q = projectToBhsd(inputs.featureInput,
                                    inputs.queryWeights,
                                    qBias,
                                    c.batchSize,
                                    c.sequenceLength,
                                    c.inputFeatures,
                                    c.numHeads,
                                    c.headDim,
                                    c.dataType);
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
    setParameterTensor(physicalAttention->getParameter("query_weights"), inputs.queryWeights, stream);
    setParameterTensor(physicalAttention->getParameter("key_weights"), inputs.keyWeights, stream);
    setParameterTensor(physicalAttention->getParameter("value_weights"), inputs.valueWeights, stream);
    setParameterTensor(physicalAttention->getParameter("output_weights"), inputs.outputWeights, stream);
    if (c.hasBias) {
        setParameterTensor(physicalAttention->getParameter("query_bias"), inputs.queryBias, stream);
        setParameterTensor(physicalAttention->getParameter("key_bias"), inputs.keyBias, stream);
        setParameterTensor(physicalAttention->getParameter("value_bias"), inputs.valueBias, stream);
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
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({16, 64})
                                  .dataType(DataType::FP16)
                                  .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(4)
                                   .causal()
                                   .build();

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
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({8, 96})
                                  .dataType(DataType::BF16)
                                  .build();

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
                                   .attentionScale(0.25)
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
    ASSERT_TRUE(attention.getAttentionScale().has_value());
    EXPECT_DOUBLE_EQ(attention.getAttentionScale().value(), 0.25);
}

TEST(AttentionApi, RejectsInvalidHeadConfiguration) {
    Api::Network network("attention_api_rejects_invalid_head_configuration");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({8, 64})
                                  .dataType(DataType::FP16)
                                  .build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(6)
                     .numKeyValueHeads(4)
                     .headDim(16)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsRank3FeatureInputForComposedAttention) {
    Api::Network network("attention_api_rejects_rank3_feature_input_for_composed_attention");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({2, 8, 64})
                                  .dataType(DataType::FP16)
                                  .build();

    EXPECT_THROW(Api::Attention::Builder().network(network).featureInput(input.getFeatureOutput().value()).numHeads(4).build(),
                 std::invalid_argument);
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
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({sequenceLength, inputFeatures})
                                  .dataType(dataType)
                                  .build();
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
    ASSERT_EQ(fixture.physicalAttention->listParameters(),
              (vector<string>{"query_weights", "key_weights", "value_weights", "output_weights"}));

    vector<float> queryWeights(inputFeatures * numHeads * headDim, 0.0f);
    vector<float> keyWeights(inputFeatures * numHeads * headDim, 0.0f);
    vector<float> valueWeights(inputFeatures * numHeads * valueDim, 0.0f);
    vector<float> outputWeights(outputFeatures * outputFeatures, 0.0f);
    for (uint32_t i = 0; i < outputFeatures; ++i) {
        valueWeights[i * outputFeatures + i] = 1.0f;
        outputWeights[i * outputFeatures + i] = 1.0f;
    }

    Stream stream = fixture.physicalAttention->getStreams()[0];
    setParameterTensor(fixture.physicalAttention->getParameter("query_weights"), queryWeights, stream);
    setParameterTensor(fixture.physicalAttention->getParameter("key_weights"), keyWeights, stream);
    setParameterTensor(fixture.physicalAttention->getParameter("value_weights"), valueWeights, stream);
    setParameterTensor(fixture.physicalAttention->getParameter("output_weights"), outputWeights, stream);
    stream.synchronize();

    vector<float> inputValues(batchSize * sequenceLength * inputFeatures, 0.0f);
    auto inputIndex = [=](uint32_t b, uint32_t s, uint32_t feature) {
        return (b * sequenceLength + s) * inputFeatures + feature;
    };
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < sequenceLength; ++s) {
            for (uint32_t h = 0; h < numHeads; ++h) {
                for (uint32_t d = 0; d < valueDim; ++d) {
                    const uint32_t feature = h * valueDim + d;
                    inputValues[inputIndex(b, s, feature)] = 0.25f * static_cast<float>(b + 1) +
                                                             0.10f * static_cast<float>(s) +
                                                             0.03f * static_cast<float>(h) +
                                                             0.001f * static_cast<float>(d);
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
