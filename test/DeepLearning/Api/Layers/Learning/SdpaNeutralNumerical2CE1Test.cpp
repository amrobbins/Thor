#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

constexpr uint64_t kMq = 4;
constexpr uint64_t kMkv = 6;
constexpr uint64_t kHq = 4;
constexpr uint64_t kHkv = 2;
constexpr uint64_t kD = 32;

std::filesystem::path archive2CE1(const char* label) {
    return std::filesystem::temp_directory_path() /
        (std::string(label) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

std::vector<uint64_t> lengthsToOffsets2CE1(uint32_t batch, bool query) {
    std::vector<uint64_t> offsets(batch + 1, 0);
    for (uint32_t row = 0; row < batch; ++row) {
        // The B=7 placement has empty query rows, but active query rows always
        // have a nonempty KV sequence.  All-empty KV semantics are a separate
        // cuDNN compatibility case, not silently skipped by this test.
        const uint64_t length = query ? (row % 4 == 0 ? 0u : 1u + row % 3)
                                      : (1u + (row * 3u + 1u) % 5u);
        offsets[row + 1] = offsets[row] + length;
    }
    return offsets;
}

float sample2CE1(int which, uint64_t token, uint64_t head, uint64_t dim) {
    const float value = 0.11f * static_cast<float>(which + 1) +
        0.043f * static_cast<float>((token * 7 + head * 5 + dim * 3 + which) % 17) -
        0.28f + 0.014f * static_cast<float>(dim % 4);
    // Reference uses precisely the FP16 value passed to the physical kernel.
    return __half2float(__float2half(value));
}

Impl::Tensor values2CE1(uint32_t b, uint64_t m, uint64_t heads,
                        const std::vector<uint64_t>& offsets, int which) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor tensor(cpu, Impl::TensorDescriptor(DataType::FP16, {b * m, heads, kD}));
    auto* data = tensor.getMemPtr<half>();
    for (uint64_t token = 0; token < b * m; ++token) {
        for (uint64_t head = 0; head < heads; ++head) {
            for (uint64_t dim = 0; dim < kD; ++dim) {
                const uint64_t index = (token * heads + head) * kD + dim;
                data[index] = token < offsets.back()
                    ? __float2half(sample2CE1(which, token, head, dim))
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
            }
        }
    }
    return tensor;
}

Impl::Tensor offsets2CE1(uint32_t b, DataType type, const std::vector<uint64_t>& offsets) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor tensor(cpu, Impl::TensorDescriptor(type, {static_cast<uint64_t>(b) + 1}));
    if (type == DataType::UINT32) {
        auto* data = tensor.getMemPtr<uint32_t>();
        for (uint32_t i = 0; i <= b; ++i) data[i] = static_cast<uint32_t>(offsets[i]);
    } else {
        auto* data = tensor.getMemPtr<uint64_t>();
        for (uint32_t i = 0; i <= b; ++i) data[i] = offsets[i];
    }
    return tensor;
}

Batch batch2CE1(uint32_t b, const Api::RaggedTensor& q,
               const Api::RaggedTensor& kv, const Api::RaggedTensor& v) {
    const auto qOffsets = lengthsToOffsets2CE1(b, true);
    const auto kvOffsets = lengthsToOffsets2CE1(b, false);
    Impl::Tensor physicalQOffsets = offsets2CE1(b, DataType::UINT32, qOffsets);
    Impl::Tensor physicalKVOffsets = offsets2CE1(b, DataType::UINT64, kvOffsets);
    Impl::RowPartitionRuntime qPartition(physicalQOffsets, q.getDescriptorForPlacement(b).getRowPartition());
    Impl::RowPartitionRuntime kvPartition(physicalKVOffsets, kv.getDescriptorForPlacement(b).getRowPartition());
    qPartition.setHostOffsets(qOffsets);
    kvPartition.setHostOffsets(kvOffsets);
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(values2CE1(b, kMq, kHq, qOffsets, 0), qPartition));
    batch.insert("kv", Impl::RaggedTensor(values2CE1(b, kMkv, kHkv, kvOffsets, 1), kvPartition));
    batch.insert("v", Impl::RaggedTensor(values2CE1(b, kMkv, kHkv, kvOffsets, 2), kvPartition));
    batch.setValidExampleCount(b);
    return batch;
}

std::shared_ptr<Api::ScaledDotProductAttention> findSdpa2CE1(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto layer = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getLayer(i));
        if (layer) return layer;
    }
    return {};
}

void checkReference2CE1(const Impl::RaggedTensor& output, uint32_t b,
                        bool causalTopLeft) {
    const auto qOffsets = lengthsToOffsets2CE1(b, true);
    const auto kvOffsets = lengthsToOffsets2CE1(b, false);
    ASSERT_TRUE(output.getHostOffsetsIfAvailable().has_value());
    EXPECT_EQ(output.getHostOffsetsIfAvailable().value(), qOffsets);
    ASSERT_EQ(output.getValues().getDataType(), DataType::FP16);
    EXPECT_EQ(output.getValues().getDimensions(), (std::vector<uint64_t>{b*kMq, kHq, kD}));
    const auto* actual = output.getValues().getMemPtr<half>();
    const double scale = 1.0 / std::sqrt(static_cast<double>(kD));
    uint64_t checked = 0;
    for (uint32_t row = 0; row < b; ++row) {
        for (uint64_t qi = qOffsets[row]; qi < qOffsets[row+1]; ++qi) {
            const uint64_t queryPosition = qi - qOffsets[row];
            for (uint64_t h = 0; h < kHq; ++h) {
                const uint64_t kh = h / (kHq/kHkv);
                std::vector<double> scores;
                std::vector<uint64_t> keyTokens;
                for (uint64_t ki = kvOffsets[row]; ki < kvOffsets[row+1]; ++ki) {
                    if (causalTopLeft && ki - kvOffsets[row] > queryPosition) continue;
                    double dot = 0.0;
                    for (uint64_t dim = 0; dim < kD; ++dim)
                        dot += static_cast<double>(sample2CE1(0, qi, h, dim)) *
                               sample2CE1(1, ki, kh, dim);
                    scores.push_back(dot * scale);
                    keyTokens.push_back(ki);
                }
                ASSERT_FALSE(scores.empty()) << "Nonempty KV is required for active query rows";
                const double peak = *std::max_element(scores.begin(), scores.end());
                double normalizer = 0.0;
                for (double& score : scores) {
                    score = std::exp(score - peak);
                    normalizer += score;
                }
                for (uint64_t dim = 0; dim < kD; ++dim) {
                    double expected = 0.0;
                    for (size_t k = 0; k < keyTokens.size(); ++k)
                        expected += (scores[k] / normalizer) *
                                    sample2CE1(2, keyTokens[k], kh, dim);
                    const uint64_t outputIndex = (qi*kHq + h)*kD + dim;
                    const float got = __half2float(actual[outputIndex]);
                    ASSERT_TRUE(std::isfinite(got)) << "active output index " << outputIndex;
                    EXPECT_NEAR(got, expected, 0.06)
                        << "row=" << row << " q=" << queryPosition
                        << " head=" << h << " dim=" << dim;
                    ++checked;
                }
            }
        }
    }
    EXPECT_GT(checked, 0u) << "Numerical test must execute active SDPA outputs";
}

void runSavedCross2CE1(bool causalTopLeft) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA required for numerical SDPA forward";
    const auto archive = archive2CE1(causalTopLeft ? "sdpa_neutral_e1_causal" : "sdpa_neutral_e1_cross");
    std::filesystem::remove_all(archive);
    {
        Api::Network network("sdpa_neutral_e1_saved");
        const auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
            .trailingDimensions({kHq,kD}).maxValuesPerRow(kMq).build();
        const auto kv = Api::RaggedNetworkInput::Builder().network(network).name("kv")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({kHkv,kD}).maxValuesPerRow(kMkv).build();
        const auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({kHkv,kD}).partition(kv).build();
        Api::ScaledDotProductAttention::Builder builder;
        builder.network(network).queryInput(q).keyInput(kv).valueInput(v)
            .computeDataType(DataType::FP32);
        if (causalTopLeft) builder.maskKind(Impl::AttentionMaskKind::CausalTopLeft);
        const auto sdpa = builder.build();
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(q));
        (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
            .inputTensor(*sdpa.getRaggedFeatureOutput()).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b << ", causal=" << causalTopLeft);
        Api::Network loaded("sdpa_neutral_e1_saved");
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto layer = findSdpa2CE1(loaded);
        ASSERT_NE(layer, nullptr);
        ASSERT_EQ(layer->architectureJson().at("version"), "2.2.0");
        ASSERT_TRUE(layer->getQueryRaggedInput().has_value());
        ASSERT_TRUE(layer->getKeyRaggedInput().has_value());
        ASSERT_TRUE(layer->getValueRaggedInput().has_value());
        EXPECT_FALSE(layer->getQueryRaggedInput()->sharesPartitionWith(*layer->getKeyRaggedInput()));
        EXPECT_TRUE(layer->getKeyRaggedInput()->sharesPartitionWith(*layer->getValueRaggedInput()));
        const auto before = loaded.architectureJson();
        std::vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{b*kMq,kHq,kD}));
        auto result = placed->inferLogical(batch2CE1(
            b, *layer->getQueryRaggedInput(), *layer->getKeyRaggedInput(), *layer->getValueRaggedInput()));
        ASSERT_TRUE(result.contains("out"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(result.at("out")));
        checkReference2CE1(std::get<Impl::RaggedTensor>(result.at("out")), b, causalTopLeft);
        EXPECT_EQ(loaded.architectureJson(), before);
    }
    std::filesystem::remove_all(archive);
}
}  // namespace

TEST(SdpaNeutralNumerical2CE1, SavedCrossAttentionB2B7MatchesCpuReferenceWithPoisonedInactiveCapacity) {
    runSavedCross2CE1(false);
}

TEST(SdpaNeutralNumerical2CE1, SavedTopLeftCausalCrossAttentionB2B7MatchesCpuReference) {
    runSavedCross2CE1(true);
}

// 2C-G3: CUDA runtime/placement reuse.  The placed-network API performs input
// staging and host-visible output materialization, and is NOT an external CUDA
// stream-capture API.  These tests exercise the existing stamp-owned cuDNN
// execution on successive submissions with changing device row offsets; they
// do not claim that inferLogical() itself is capturable as one CUDA graph.
namespace {

std::vector<uint64_t> varyingOffsets2CG3(uint32_t b, bool query, uint32_t pattern) {
    static constexpr uint64_t qLens[3][4] = {
        {2, 1, 3, 0}, {1, 3, 0, 2}, {3, 0, 2, 1}};
    static constexpr uint64_t kvLens[3][4] = {
        {3, 2, 5, 1}, {2, 5, 1, 4}, {5, 1, 3, 2}};
    std::vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row) {
        const auto length = query ? qLens[pattern][row % 4] : kvLens[pattern][row % 4];
        offsets[row + 1] = offsets[row] + length;
    }
    return offsets;
}

Batch varyingBatch2CG3(uint32_t b, uint32_t pattern,
                       const Api::RaggedTensor& q, const Api::RaggedTensor& k,
                       const Api::RaggedTensor& v) {
    const auto qo = varyingOffsets2CG3(b, true, pattern);
    const auto ko = varyingOffsets2CG3(b, false, pattern);
    Impl::RowPartitionRuntime qp(offsets2CE1(b, DataType::UINT32, qo),
                                  q.getDescriptorForPlacement(b).getRowPartition());
    Impl::RowPartitionRuntime kp(offsets2CE1(b, DataType::UINT64, ko),
                                  k.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(values2CE1(b, kMq, kHq, qo, 0), qp));
    batch.insert("kv", Impl::RaggedTensor(values2CE1(b, kMkv, kHkv, ko, 1), kp));
    batch.insert("v", Impl::RaggedTensor(values2CE1(b, kMkv, kHkv, ko, 2), kp));
    batch.setValidExampleCount(b);
    return batch;
}

std::vector<float> verifyRuntimeReference2CG3(const Impl::RaggedTensor& output,
                                             uint32_t b, uint32_t pattern,
                                             bool causalTopLeft) {
    const auto qo = varyingOffsets2CG3(b, true, pattern);
    const auto ko = varyingOffsets2CG3(b, false, pattern);
    EXPECT_EQ(output.getValues().getDimensions(), (std::vector<uint64_t>{b*kMq,kHq,kD}));
    const auto mirror = output.getHostOffsetsIfAvailable();
    EXPECT_TRUE(mirror.has_value());
    if (mirror.has_value()) EXPECT_EQ(*mirror, qo);
    const half* actual = output.getValues().getMemPtr<half>();
    const double scale = 1.0 / std::sqrt(static_cast<double>(kD));
    std::vector<float> active;
    active.reserve(qo.back()*kHq*kD);
    for (uint32_t row = 0; row < b; ++row) {
        for (uint64_t qi = qo[row]; qi < qo[row+1]; ++qi) {
            const uint64_t qPosition = qi-qo[row];
            for (uint64_t h = 0; h < kHq; ++h) {
                const uint64_t kh = h/(kHq/kHkv);
                std::vector<double> scores;
                std::vector<uint64_t> keys;
                for (uint64_t ki = ko[row]; ki < ko[row+1]; ++ki) {
                    if (causalTopLeft && ki-ko[row]>qPosition) continue;
                    double dot=0.0;
                    for (uint64_t d=0; d<kD; ++d)
                        dot+=sample2CE1(0,qi,h,d)*sample2CE1(1,ki,kh,d);
                    scores.push_back(dot*scale);
                    keys.push_back(ki);
                }
                if (scores.empty()) {
                    ADD_FAILURE() << "Nonempty query row has no unmasked keys";
                    return active;
                }
                const double maximum=*std::max_element(scores.begin(),scores.end());
                double denom=0.0;
                for (double& score: scores) {score=std::exp(score-maximum);denom+=score;}
                for (uint64_t d=0; d<kD; ++d) {
                    double expected=0.0;
                    for (size_t j=0; j<keys.size(); ++j)
                        expected+=(scores[j]/denom)*sample2CE1(2,keys[j],kh,d);
                    const float value=__half2float(actual[(qi*kHq+h)*kD+d]);
                    EXPECT_TRUE(std::isfinite(value)) << "active Q=" << qi << " h=" << h << " d=" << d;
                    EXPECT_NEAR(value, expected, 0.06) << "pattern=" << pattern << " Q=" << qi
                        << " h=" << h << " d=" << d;
                    active.push_back(value);
                }
            }
        }
    }
    EXPECT_EQ(active.size(), qo.back()*kHq*kD);
    return active;
}

void runRuntimeReuse2CG3(bool causalTopLeft) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)
        GTEST_SKIP() << "CUDA device required for attention runtime execution";

    const auto archive=archive2CE1(causalTopLeft ? "sdpa_g3_causal" : "sdpa_g3_unmasked");
    std::filesystem::remove_all(archive);
    {
        Api::Network net("sdpa_g3_runtime");
        const auto q=Api::RaggedNetworkInput::Builder().network(net).name("q")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
            .trailingDimensions({kHq,kD}).maxValuesPerRow(kMq).build();
        const auto kv=Api::RaggedNetworkInput::Builder().network(net).name("kv")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({kHkv,kD}).maxValuesPerRow(kMkv).build();
        const auto v=Api::RaggedNetworkInput::Builder().network(net).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({kHkv,kD}).partition(kv).build();
        Api::ScaledDotProductAttention::Builder builder;
        builder.network(net).queryInput(q).keyInput(kv).valueInput(v)
            .computeDataType(DataType::FP32);
        if (causalTopLeft) builder.maskKind(Impl::AttentionMaskKind::CausalTopLeft);
        const auto sdpa=builder.build();
        ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
        (void)Api::RaggedNetworkOutput::Builder().network(net).name("out")
            .inputTensor(*sdpa.getRaggedFeatureOutput()).build();
        net.save(archive.string(),true);
    }

    for (uint32_t b: {2u,7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b << " causal=" << causalTopLeft);
        Api::Network loaded("sdpa_g3_runtime");
        loaded.load(archive.string());
        auto sdpa=findSdpa2CE1(loaded);
        ASSERT_NE(sdpa,nullptr);
        ASSERT_TRUE(sdpa->getQueryRaggedInput().has_value());
        ASSERT_TRUE(sdpa->getKeyRaggedInput().has_value());
        ASSERT_TRUE(sdpa->getValueRaggedInput().has_value());
        const auto original=loaded.architectureJson();
        std::vector<Event> events;
        auto placed=loaded.place(b,events,true);
        ASSERT_NE(placed,nullptr);
        for (auto& event: events) event.synchronize();
        auto physical=std::dynamic_pointer_cast<Impl::CustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(sdpa->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        const void* physicalOutput=physical->getFeatureOutput()->getMemPtr<void>();
        ASSERT_NE(physicalOutput,nullptr);
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{b*kMq,kHq,kD}));

        // No restamp between A -> B -> C -> A: the same device capacity and
        // cuDNN attention executable must consume newly submitted row offsets.
        std::vector<float> initial;
        for (uint32_t pattern: {0u,1u,2u,0u}) {
            SCOPED_TRACE(::testing::Message() << "pattern=" << pattern);
            auto outputs=placed->inferLogical(varyingBatch2CG3(
                b,pattern,*sdpa->getQueryRaggedInput(),
                *sdpa->getKeyRaggedInput(),*sdpa->getValueRaggedInput()));
            ASSERT_TRUE(outputs.contains("out"));
            ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("out")));
            const auto active=verifyRuntimeReference2CG3(
                std::get<Impl::RaggedTensor>(outputs.at("out")),b,pattern,causalTopLeft);
            ASSERT_GT(active.size(),0u);
            if (pattern==0u) {
                if (initial.empty()) initial=active;
                else EXPECT_EQ(active,initial) << "same active inputs must replay exactly on same stamp";
            }
            ASSERT_TRUE(physical->getFeatureOutput().has_value());
            EXPECT_EQ(physical->getFeatureOutput()->getMemPtr<void>(),physicalOutput)
                << "stamp-owned physical output allocation changed between submissions";
            EXPECT_EQ(loaded.architectureJson(),original)
                << "runtime row lengths must not rewrite the artifact";
        }
    }
    std::filesystem::remove_all(archive);
}
}  // namespace

TEST(SdpaNeutralRuntime2CG3, UnmaskedRepeatedExecutionChangesOffsetsWithoutRestamp) {
    runRuntimeReuse2CG3(false);
}

TEST(SdpaNeutralRuntime2CG3, CausalRepeatedExecutionChangesOffsetsWithoutRestamp) {
    runRuntimeReuse2CG3(true);
}
