#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
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

constexpr uint64_t kMq2 = 4;
constexpr uint64_t kMkv2 = 6;
constexpr uint64_t kHq2 = 4;
constexpr uint64_t kHkv2 = 2;
constexpr uint64_t kD2 = 32;

std::filesystem::path archive2CE2(const char* label) {
    return std::filesystem::temp_directory_path() /
        (std::string(label) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

std::vector<uint64_t> offsets2CE2(uint32_t b, bool query) {
    std::vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row) {
        // Q contains an empty row. KV is intentionally nonempty for each
        // active dense query; empty-KV output semantics need a separate gate.
        const uint64_t len = query ? (row % 4 == 0 ? 0u : 1u + row % 3)
                                   : (2u + row % 4);
        offsets[row + 1] = offsets[row] + len;
    }
    return offsets;
}

float scalar2CE2(int which, uint64_t token, uint64_t h, uint64_t d) {
    const float value = 0.075f * static_cast<float>(which + 1) +
        0.039f * static_cast<float>((token * 11 + h * 7 + d * 3 + which) % 19) -
        0.27f + 0.015f * static_cast<float>(d % 5);
    return __half2float(__float2half(value));
}

Impl::Tensor denseValues2CE2(uint32_t b, uint64_t seq, uint64_t heads, int which) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(DataType::FP16, {b, seq, heads, kD2}));
    auto* ptr = result.getMemPtr<half>();
    for (uint64_t token = 0; token < static_cast<uint64_t>(b) * seq; ++token)
        for (uint64_t h = 0; h < heads; ++h)
            for (uint64_t d = 0; d < kD2; ++d)
                ptr[(token * heads + h) * kD2 + d] =
                    __float2half(scalar2CE2(which, token, h, d));
    return result;
}

Impl::Tensor packedValues2CE2(uint32_t b, uint64_t m, uint64_t heads,
                              const std::vector<uint64_t>& offsets, int which) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(DataType::FP16, {b*m, heads, kD2}));
    auto* ptr = result.getMemPtr<half>();
    for (uint64_t token = 0; token < static_cast<uint64_t>(b) * m; ++token)
        for (uint64_t h = 0; h < heads; ++h)
            for (uint64_t d = 0; d < kD2; ++d)
                ptr[(token * heads + h) * kD2 + d] = token < offsets.back()
                    ? __float2half(scalar2CE2(which, token, h, d))
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
    return result;
}

Impl::Tensor offsetTensor2CE2(uint32_t b, DataType dtype,
                              const std::vector<uint64_t>& offsets) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(dtype, {static_cast<uint64_t>(b) + 1}));
    if (dtype == DataType::UINT32) {
        auto* ptr = result.getMemPtr<uint32_t>();
        for (uint32_t i = 0; i <= b; ++i) ptr[i] = static_cast<uint32_t>(offsets[i]);
    } else {
        auto* ptr = result.getMemPtr<uint64_t>();
        for (uint32_t i = 0; i <= b; ++i) ptr[i] = offsets[i];
    }
    return result;
}

Impl::RaggedTensor raggedValues2CE2(uint32_t b, const Api::RaggedTensor& logical,
                                    uint64_t m, uint64_t heads, int which,
                                    const std::vector<uint64_t>& offsets,
                                    DataType offsetType) {
    Impl::Tensor physicalOffsets = offsetTensor2CE2(b, offsetType, offsets);
    Impl::RowPartitionRuntime partition(
        physicalOffsets, logical.getDescriptorForPlacement(b).getRowPartition());
    partition.setHostOffsets(offsets);
    return Impl::RaggedTensor(packedValues2CE2(b, m, heads, offsets, which), partition);
}

std::shared_ptr<Api::ScaledDotProductAttention> findSdpa2CE2(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto layer = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getLayer(i));
        if (layer) return layer;
    }
    return {};
}

// Compare real physical output with CPU scaled-dot-product attention, using
// token-major BSHD indexing for each input's actual dense/packed domain.
void reference2CE2(const Impl::Tensor& values, uint32_t b,
                   bool raggedQuery, bool causalTopLeft) {
    const auto qOffsets = offsets2CE2(b, true);
    const auto kvOffsets = offsets2CE2(b, false);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kD2));
    const auto* actual = values.getMemPtr<half>();
    ASSERT_EQ(values.getDataType(), DataType::FP16);
    const auto expectedDims = raggedQuery
        ? std::vector<uint64_t>{static_cast<uint64_t>(b)*kMq2,kHq2,kD2}
        : std::vector<uint64_t>{b,kMq2,kHq2,kD2};
    ASSERT_EQ(values.getDimensions(), expectedDims);
    uint64_t checked = 0;
    for (uint32_t row = 0; row < b; ++row) {
        const uint64_t qCount = raggedQuery ? qOffsets[row + 1] - qOffsets[row] : kMq2;
        const uint64_t kvCount = raggedQuery ? kMkv2 : kvOffsets[row + 1] - kvOffsets[row];
        ASSERT_GT(kvCount, 0u);
        for (uint64_t qp = 0; qp < qCount; ++qp) {
            const uint64_t qt = raggedQuery ? qOffsets[row] + qp : row*kMq2 + qp;
            for (uint64_t h = 0; h < kHq2; ++h) {
                const uint64_t kh = h / (kHq2/kHkv2);
                std::vector<double> scores;
                std::vector<uint64_t> keys;
                for (uint64_t kp = 0; kp < kvCount; ++kp) {
                    if (causalTopLeft && kp > qp) continue;
                    const uint64_t kt = raggedQuery ? row*kMkv2 + kp : kvOffsets[row] + kp;
                    double dot = 0.0;
                    for (uint64_t d = 0; d < kD2; ++d)
                        dot += static_cast<double>(scalar2CE2(0, qt, h, d)) *
                               scalar2CE2(1, kt, kh, d);
                    scores.push_back(dot * scale);
                    keys.push_back(kt);
                }
                ASSERT_FALSE(scores.empty());
                const double peak = *std::max_element(scores.begin(), scores.end());
                double divisor = 0.0;
                for (double& score : scores) {
                    score = std::exp(score - peak);
                    divisor += score;
                }
                for (uint64_t d = 0; d < kD2; ++d) {
                    double expected = 0.0;
                    for (size_t i = 0; i < scores.size(); ++i)
                        expected += (scores[i] / divisor) * scalar2CE2(2, keys[i], kh, d);
                    const uint64_t index = (qt*kHq2 + h)*kD2 + d;
                    const float got = __half2float(actual[index]);
                    ASSERT_TRUE(std::isfinite(got)) << "active index " << index;
                    EXPECT_NEAR(got, expected, 0.06)
                        << "row=" << row << " query=" << qp << " key_count=" << kvCount
                        << " head=" << h << " d=" << d;
                    ++checked;
                }
            }
        }
    }
    EXPECT_GT(checked, 0u);
}

void runMixed2CE2(bool raggedQuery, bool causalTopLeft) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA required for actual SDPA numerical forward";
    const auto archive = archive2CE2(raggedQuery ?
        (causalTopLeft ? "sdpa_e2_rq_causal" : "sdpa_e2_rq") : "sdpa_e2_rkv");
    std::filesystem::remove_all(archive);
    {
        Api::Network network("sdpa_neutral_e2_mixed");
        if (raggedQuery) {
            auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
                .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
                .maxValuesPerRow(kMq2).trailingDimensions({kHq2,kD2}).build();
            auto k = Api::NetworkInput::Builder().network(network).name("k")
                .dataType(DataType::FP16).dimensions({kMkv2,kHkv2,kD2}).build();
            auto v = Api::NetworkInput::Builder().network(network).name("v")
                .dataType(DataType::FP16).dimensions({kMkv2,kHkv2,kD2}).build();
            Api::ScaledDotProductAttention::Builder builder;
            builder.network(network).queryInput(q).keyInput(*k.getFeatureOutput())
                .valueInput(*v.getFeatureOutput()).computeDataType(DataType::FP32);
            if (causalTopLeft) builder.maskKind(Impl::AttentionMaskKind::CausalTopLeft);
            auto sdpa = builder.build();
            ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
            EXPECT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(q));
            (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
                .inputTensor(*sdpa.getRaggedFeatureOutput()).build();
        } else {
            auto q = Api::NetworkInput::Builder().network(network).name("q")
                .dataType(DataType::FP16).dimensions({kMq2,kHq2,kD2}).build();
            auto kv = Api::RaggedNetworkInput::Builder().network(network).name("kv")
                .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
                .maxValuesPerRow(kMkv2).trailingDimensions({kHkv2,kD2}).build();
            auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
                .valuesDataType(DataType::FP16).trailingDimensions({kHkv2,kD2})
                .partition(kv).build();
            auto sdpa = Api::ScaledDotProductAttention::Builder().network(network)
                .queryInput(*q.getFeatureOutput()).keyInput(kv).valueInput(v)
                .computeDataType(DataType::FP32).build();
            ASSERT_FALSE(sdpa.getRaggedFeatureOutput().has_value());
            (void)Api::NetworkOutput::Builder().network(network).name("out")
                .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
        }
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b << " raggedQ=" << raggedQuery
                                           << " causal=" << causalTopLeft);
        Api::Network loaded("sdpa_neutral_e2_mixed");
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto sdpa = findSdpa2CE2(loaded);
        ASSERT_NE(sdpa, nullptr);
        EXPECT_EQ(sdpa->architectureJson().at("version"), "2.2.0");
        const auto before = loaded.architectureJson();
        std::vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(sdpa->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        const auto expectedDims = raggedQuery
            ? std::vector<uint64_t>{static_cast<uint64_t>(b)*kMq2,kHq2,kD2}
            : std::vector<uint64_t>{b,kMq2,kHq2,kD2};
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), expectedDims);
        Batch batch;
        if (raggedQuery) {
            const auto qOff = offsets2CE2(b, true);
            ASSERT_TRUE(sdpa->getQueryRaggedInput().has_value());
            batch.insert("q", raggedValues2CE2(b, *sdpa->getQueryRaggedInput(),
                                                kMq2,kHq2,0,qOff,DataType::UINT32));
            batch.insert("k", denseValues2CE2(b,kMkv2,kHkv2,1));
            batch.insert("v", denseValues2CE2(b,kMkv2,kHkv2,2));
        } else {
            const auto kvOff = offsets2CE2(b, false);
            ASSERT_TRUE(sdpa->getKeyRaggedInput().has_value());
            ASSERT_TRUE(sdpa->getValueRaggedInput().has_value());
            EXPECT_TRUE(sdpa->getKeyRaggedInput()->sharesPartitionWith(
                *sdpa->getValueRaggedInput()));
            batch.insert("q", denseValues2CE2(b,kMq2,kHq2,0));
            batch.insert("kv", raggedValues2CE2(b, *sdpa->getKeyRaggedInput(),
                                                 kMkv2,kHkv2,1,kvOff,DataType::UINT64));
            batch.insert("v", raggedValues2CE2(b, *sdpa->getValueRaggedInput(),
                                                kMkv2,kHkv2,2,kvOff,DataType::UINT64));
        }
        batch.setValidExampleCount(b);
        auto outputs = placed->inferLogical(std::move(batch));
        ASSERT_TRUE(outputs.contains("out"));
        if (raggedQuery) {
            ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("out")));
            const auto& output = std::get<Impl::RaggedTensor>(outputs.at("out"));
            ASSERT_TRUE(output.getHostOffsetsIfAvailable().has_value());
            EXPECT_EQ(*output.getHostOffsetsIfAvailable(), offsets2CE2(b, true));
            reference2CE2(output.getValues(), b, true, causalTopLeft);
        } else {
            ASSERT_TRUE(std::holds_alternative<Impl::Tensor>(outputs.at("out")));
            reference2CE2(std::get<Impl::Tensor>(outputs.at("out")), b, false, false);
        }
        EXPECT_EQ(loaded.architectureJson(), before);
    }
    std::filesystem::remove_all(archive);
}
}  // namespace

TEST(SdpaNeutralNumerical2CE2, RaggedQueryDenseKeyValueSavedB2B7MatchesCpuReference) {
    runMixed2CE2(true, false);
}

TEST(SdpaNeutralNumerical2CE2, DenseQueryRaggedKeyValueSavedB2B7MatchesCpuReference) {
    runMixed2CE2(false, false);
}

TEST(SdpaNeutralNumerical2CE2, RaggedQueryDenseKeyValueTopLeftCausalSavedB2B7MatchesCpuReference) {
    runMixed2CE2(true, true);
}
