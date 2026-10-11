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

constexpr uint64_t kMqE3 = 4;
constexpr uint64_t kMkvE3 = 6;
constexpr uint64_t kHqE3 = 4;
constexpr uint64_t kHkvE3 = 2;
constexpr uint64_t kDE3 = 32;

std::filesystem::path archiveE3(bool raggedQ) {
    return std::filesystem::temp_directory_path() /
        (std::string(raggedQ ? "sdpa_e3_rq_empty_kv_" : "sdpa_e3_denseq_empty_kv_") +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

std::vector<uint64_t> qOffsetsE3(uint32_t batch) {
    std::vector<uint64_t> offsets(batch + 1, 0);
    for (uint32_t row = 0; row < batch; ++row) {
        // Active queries intentionally coincide with zero-length KV rows.
        const uint64_t n = row % 5 == 4 ? 0u : 1u + row % 3;
        offsets[row+1] = offsets[row] + n;
    }
    return offsets;
}

std::vector<uint64_t> kvOffsetsE3(uint32_t batch) {
    std::vector<uint64_t> offsets(batch + 1, 0);
    for (uint32_t row = 0; row < batch; ++row) {
        // Includes an empty KV row even at B=2, next to a nonempty control.
        const uint64_t n = row % 3 == 0 ? 0u : 2u + row % 3;
        offsets[row+1] = offsets[row] + n;
    }
    return offsets;
}

float inputScalarE3(int which, uint64_t token, uint64_t head, uint64_t d) {
    const float value = 0.09f * static_cast<float>(which + 1) +
        0.037f * static_cast<float>((token * 7 + head * 11 + d * 3 + which) % 19) -
        0.28f + 0.013f * static_cast<float>(d % 5);
    return __half2float(__float2half(value));
}

Impl::Tensor offsetTensorE3(uint32_t b, DataType dtype, const std::vector<uint64_t>& offsets) {
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

Impl::Tensor packedValuesE3(uint32_t b, uint64_t m, uint64_t heads,
                            const std::vector<uint64_t>& offsets, int which) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(DataType::FP16, {b*m, heads, kDE3}));
    auto* ptr = result.getMemPtr<half>();
    for (uint64_t token = 0; token < static_cast<uint64_t>(b) * m; ++token)
        for (uint64_t h = 0; h < heads; ++h)
            for (uint64_t d = 0; d < kDE3; ++d)
                ptr[(token * heads + h)*kDE3 + d] = token < offsets.back()
                    ? __float2half(inputScalarE3(which, token, h, d))
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
    return result;
}

Impl::Tensor denseQueriesE3(uint32_t b) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(DataType::FP16, {b,kMqE3,kHqE3,kDE3}));
    auto* ptr = result.getMemPtr<half>();
    for (uint64_t token = 0; token < static_cast<uint64_t>(b) * kMqE3; ++token)
        for (uint64_t h = 0; h < kHqE3; ++h)
            for (uint64_t d = 0; d < kDE3; ++d)
                ptr[(token*kHqE3 + h)*kDE3 + d] =
                    __float2half(inputScalarE3(0, token, h, d));
    return result;
}

Impl::RaggedTensor raggedValuesE3(uint32_t b, const Api::RaggedTensor& logical,
                                  uint64_t m, uint64_t heads, int which,
                                  const std::vector<uint64_t>& offsets, DataType dtype) {
    Impl::RowPartitionRuntime partition(offsetTensorE3(b, dtype, offsets),
                                        logical.getDescriptorForPlacement(b).getRowPartition());
    partition.setHostOffsets(offsets);
    return Impl::RaggedTensor(packedValuesE3(b, m, heads, offsets, which), partition);
}

std::shared_ptr<Api::ScaledDotProductAttention> findSdpaE3(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto sdpa = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getLayer(i));
        if (sdpa) return sdpa;
    }
    return {};
}

void compareEmptyKvReferenceE3(const Impl::Tensor& output, uint32_t b, bool raggedQ) {
    const auto queryOffsets = qOffsetsE3(b);
    const auto keyOffsets = kvOffsetsE3(b);
    const std::vector<uint64_t> expectedDims = raggedQ
        ? std::vector<uint64_t>{b*kMqE3,kHqE3,kDE3}
        : std::vector<uint64_t>{b,kMqE3,kHqE3,kDE3};
    ASSERT_EQ(output.getDimensions(), expectedDims);
    ASSERT_EQ(output.getDataType(), DataType::FP16);
    const auto* actual = output.getMemPtr<half>();
    const double scale = 1.0 / std::sqrt(static_cast<double>(kDE3));
    uint64_t emptyChecked = 0;
    uint64_t nonemptyChecked = 0;
    for (uint32_t row = 0; row < b; ++row) {
        const uint64_t qCount = raggedQ ? queryOffsets[row+1] - queryOffsets[row] : kMqE3;
        const uint64_t kvCount = keyOffsets[row+1] - keyOffsets[row];
        for (uint64_t qp = 0; qp < qCount; ++qp) {
            const uint64_t qt = raggedQ ? queryOffsets[row] + qp : row*kMqE3 + qp;
            for (uint64_t h = 0; h < kHqE3; ++h) {
                const uint64_t kh = h / (kHqE3/kHkvE3);
                std::vector<double> weights;
                for (uint64_t kp = 0; kp < kvCount; ++kp) {
                    double dot = 0.0;
                    for (uint64_t d = 0; d < kDE3; ++d)
                        dot += static_cast<double>(inputScalarE3(0, qt, h, d)) *
                               inputScalarE3(1, keyOffsets[row] + kp, kh, d);
                    weights.push_back(dot * scale);
                }
                double denominator = 0.0;
                if (!weights.empty()) {
                    const double peak = *std::max_element(weights.begin(), weights.end());
                    for (double& weight : weights) {
                        weight = std::exp(weight - peak);
                        denominator += weight;
                    }
                }
                for (uint64_t d = 0; d < kDE3; ++d) {
                    double expected = 0.0;
                    // Explicit desired empty-KV policy: zero output for an
                    // active query with no keys. Whether cuDNN provides this
                    // behavior is what this execution gate determines.
                    for (uint64_t kp = 0; kp < kvCount; ++kp)
                        expected += weights[kp] / denominator *
                                    inputScalarE3(2, keyOffsets[row] + kp, kh, d);
                    const uint64_t index = (qt*kHqE3 + h)*kDE3 + d;
                    const float got = __half2float(actual[index]);
                    ASSERT_TRUE(std::isfinite(got)) << "row=" << row << " q=" << qp
                        << " kv_count=" << kvCount << " active_index=" << index;
                    EXPECT_NEAR(got, expected, kvCount == 0 ? 0.01 : 0.06)
                        << "row=" << row << " q=" << qp << " h=" << h
                        << " d=" << d << " kv_count=" << kvCount;
                    if (kvCount == 0) ++emptyChecked;
                    else ++nonemptyChecked;
                }
            }
        }
    }
    EXPECT_GT(emptyChecked, 0u) << "Must execute active Q with empty KV";
    EXPECT_GT(nonemptyChecked, 0u) << "Must retain nonempty-KV numerical control";
}

void runEmptyKvE3(bool raggedQ) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA required for numerical SDPA empty-KV forward";
    const auto archive = archiveE3(raggedQ);
    std::filesystem::remove_all(archive);
    {
        Api::Network network("sdpa_neutral_e3_empty_kv");
        auto kv = Api::RaggedNetworkInput::Builder().network(network).name("kv")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .maxValuesPerRow(kMkvE3).trailingDimensions({kHkvE3,kDE3}).build();
        auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({kHkvE3,kDE3})
            .partition(kv).build();
        if (raggedQ) {
            auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
                .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
                .maxValuesPerRow(kMqE3).trailingDimensions({kHqE3,kDE3}).build();
            auto sdpa = Api::ScaledDotProductAttention::Builder().network(network)
                .queryInput(q).keyInput(kv).valueInput(v)
                .computeDataType(DataType::FP32).build();
            ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
            EXPECT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(q));
            (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
                .inputTensor(*sdpa.getRaggedFeatureOutput()).build();
        } else {
            auto q = Api::NetworkInput::Builder().network(network).name("q")
                .dataType(DataType::FP16).dimensions({kMqE3,kHqE3,kDE3}).build();
            auto sdpa = Api::ScaledDotProductAttention::Builder().network(network)
                .queryInput(*q.getFeatureOutput()).keyInput(kv).valueInput(v)
                .computeDataType(DataType::FP32).build();
            ASSERT_FALSE(sdpa.getRaggedFeatureOutput().has_value());
            (void)Api::NetworkOutput::Builder().network(network).name("out")
                .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
        }
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b << " raggedQ=" << raggedQ);
        Api::Network loaded("sdpa_neutral_e3_empty_kv");
        ASSERT_NO_THROW(loaded.load(archive.string()));
        auto sdpa = findSdpaE3(loaded);
        ASSERT_NE(sdpa, nullptr);
        ASSERT_EQ(sdpa->architectureJson().at("version"), "2.2.0");
        const auto before = loaded.architectureJson();
        std::vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(sdpa->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        const std::vector<uint64_t> expected = raggedQ
            ? std::vector<uint64_t>{b*kMqE3,kHqE3,kDE3}
            : std::vector<uint64_t>{b,kMqE3,kHqE3,kDE3};
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), expected);
        ASSERT_TRUE(sdpa->getKeyRaggedInput().has_value());
        ASSERT_TRUE(sdpa->getValueRaggedInput().has_value());
        EXPECT_TRUE(sdpa->getKeyRaggedInput()->sharesPartitionWith(*sdpa->getValueRaggedInput()));
        const auto kvOffsets = kvOffsetsE3(b);
        Batch batch;
        batch.insert("kv", raggedValuesE3(b, *sdpa->getKeyRaggedInput(),
            kMkvE3,kHkvE3,1,kvOffsets,DataType::UINT64));
        batch.insert("v", raggedValuesE3(b, *sdpa->getValueRaggedInput(),
            kMkvE3,kHkvE3,2,kvOffsets,DataType::UINT64));
        if (raggedQ) {
            ASSERT_TRUE(sdpa->getQueryRaggedInput().has_value());
            batch.insert("q", raggedValuesE3(b, *sdpa->getQueryRaggedInput(),
                kMqE3,kHqE3,0,qOffsetsE3(b),DataType::UINT32));
        } else {
            batch.insert("q", denseQueriesE3(b));
        }
        batch.setValidExampleCount(b);
        const auto outputs = placed->inferLogical(std::move(batch));
        ASSERT_TRUE(outputs.contains("out"));
        if (raggedQ) {
            ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("out")));
            const auto& output = std::get<Impl::RaggedTensor>(outputs.at("out"));
            ASSERT_TRUE(output.getHostOffsetsIfAvailable().has_value());
            EXPECT_EQ(*output.getHostOffsetsIfAvailable(), qOffsetsE3(b));
            compareEmptyKvReferenceE3(output.getValues(), b, true);
        } else {
            ASSERT_TRUE(std::holds_alternative<Impl::Tensor>(outputs.at("out")));
            compareEmptyKvReferenceE3(std::get<Impl::Tensor>(outputs.at("out")), b, false);
        }
        EXPECT_EQ(loaded.architectureJson(), before);
    }
    std::filesystem::remove_all(archive);
}
} // namespace

TEST(SdpaNeutralNumerical2CE3, RaggedQueryRaggedKeyValueEmptyKvRowMustProduceFiniteZeros) {
    runEmptyKvE3(true);
}

TEST(SdpaNeutralNumerical2CE3, DenseQueryRaggedKeyValueEmptyKvRowMustProduceFiniteZeros) {
    runEmptyKvE3(false);
}
