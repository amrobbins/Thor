#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

#define REQUIRE_CUDA_B2()                                                            \
    do {                                                                             \
        int count = 0;                                                               \
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {                \
            GTEST_SKIP() << "CUDA device required for SDPA physical placement";    \
        }                                                                            \
    } while (false)

Api::RaggedTensor raggedInput(Api::Network& network, const char* name,
                              uint64_t m, uint64_t heads, uint64_t width,
                              DataType offsetsType = DataType::UINT64) {
    return Api::RaggedNetworkInput::Builder()
        .network(network).name(name).valuesDataType(DataType::FP16)
        .offsetsDataType(offsetsType).maxValuesPerRow(m)
        .trailingDimensions({heads, width}).build();
}

void verifyPlacement(Api::Network& network, const Api::ScaledDotProductAttention& sdpa,
                     uint32_t batch, const std::vector<uint64_t>& expectedOutput) {
    const auto output = Api::NetworkOutput::Builder()
                            .network(network).name("attention_out")
                            .inputTensor(sdpa.getOutput("output"))
                            .dataType(DataType::FP16).build();
    std::vector<Event> ready;
    auto placed = network.place(batch, ready, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : ready) event.synchronize();
    auto& stamped = placed->getStampedNetwork(0);
    auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
        stamped.getPhysicalLayerFromApiLayer(sdpa.getId()));
    ASSERT_NE(physical, nullptr);
    auto physicalOutput = stamped.getPhysicalLayerFromApiLayer(output.getId());
    ASSERT_NE(physicalOutput, nullptr);
    const auto physicalOut = physicalOutput->getFeatureInput();
    ASSERT_TRUE(physicalOut.has_value());
    EXPECT_EQ(physicalOut->getDimensions(), expectedOutput);
    auto requirePhysicalOffsets = [&](const std::optional<Api::RaggedTensor>& logicalRagged) {
        if (!logicalRagged.has_value()) return;
        auto offsets = stamped.getManagedPartitionOffsetsInputForTest(logicalRagged->getRowPartitionId());
        ASSERT_NE(offsets, nullptr);
        ASSERT_TRUE(offsets->getFeatureOutput().has_value());
        EXPECT_EQ(offsets->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(batch) + 1}));
    };
    requirePhysicalOffsets(sdpa.getQueryRaggedInput());
    requirePhysicalOffsets(sdpa.getKeyRaggedInput());
}

} // namespace

TEST(SdpaNeutralFoundationB2, DistinctQKvPartitionsPlaceAtB2AndB7) {
    REQUIRE_CUDA_B2();
    for (uint32_t batch : {2u, 7u}) {
        for (bool kvFirst : {false, true}) {
            Api::Network network("sdpa_b2_cross");
            Api::RaggedTensor q;
            Api::RaggedTensor kv;
            if (kvFirst) {
                kv = raggedInput(network, "kv", 7, 2, 32, DataType::UINT64);
                q = raggedInput(network, "q", 3, 4, 32, DataType::UINT32);
            } else {
                q = raggedInput(network, "q", 3, 4, 32, DataType::UINT32);
                kv = raggedInput(network, "kv", 7, 2, 32, DataType::UINT64);
            }
            ASSERT_FALSE(q.sharesPartitionWith(kv));
            auto sdpa = Api::ScaledDotProductAttention::Builder()
                            .network(network).queryInput(q).keyInput(kv).valueInput(kv).build();
            verifyPlacement(network, sdpa, batch, {batch * 3u, 4, 32});
        }
    }
}

TEST(SdpaNeutralFoundationB2, RaggedQueryDenseKvPlacesPackedOutputAtB2AndB7) {
    REQUIRE_CUDA_B2();
    for (uint32_t batch : {2u, 7u}) {
        Api::Network network("sdpa_b2_ragged_query");
        const auto q = raggedInput(network, "q", 3, 4, 32);
        const auto k = Api::NetworkInput::Builder().network(network).name("k")
                           .dataType(DataType::FP16).dimensions({7, 2, 32}).build();
        const auto v = Api::NetworkInput::Builder().network(network).name("v")
                           .dataType(DataType::FP16).dimensions({7, 2, 48}).build();
        auto sdpa = Api::ScaledDotProductAttention::Builder()
                        .network(network).queryInput(q)
                        .keyInput(k.getFeatureOutput().value())
                        .valueInput(v.getFeatureOutput().value()).build();
        verifyPlacement(network, sdpa, batch, {batch * 3u, 4, 48});
    }
}

TEST(SdpaNeutralFoundationB2, DenseQueryRaggedKvKeepsOrdinaryDenseBatchOutput) {
    REQUIRE_CUDA_B2();
    for (uint32_t batch : {2u, 7u}) {
        Api::Network network("sdpa_b2_ragged_kv");
        const auto q = Api::NetworkInput::Builder().network(network).name("q")
                           .dataType(DataType::FP16).dimensions({4, 4, 32}).build();
        const auto kv = raggedInput(network, "kv", 7, 2, 32);
        auto sdpa = Api::ScaledDotProductAttention::Builder()
                        .network(network).queryInput(q.getFeatureOutput().value())
                        .keyInput(kv).valueInput(kv).build();
        verifyPlacement(network, sdpa, batch, {batch, 4, 4, 32});
    }
}
