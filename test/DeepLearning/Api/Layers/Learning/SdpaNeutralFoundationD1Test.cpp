#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
using json = nlohmann::json;

#define REQUIRE_CUDA_D1() do {                                               \
    int devices = 0;                                                         \
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {       \
        GTEST_SKIP() << "CUDA required for physical SDPA stamping";        \
    }                                                                        \
} while (false)

Api::RaggedTensor raggedD1(Api::Network& n, const char* name, uint64_t m,
                           uint64_t heads, uint64_t width, DataType offsetType) {
    return Api::RaggedNetworkInput::Builder().network(n).name(name)
        .valuesDataType(DataType::FP16).offsetsDataType(offsetType)
        .maxValuesPerRow(m).trailingDimensions({heads, width}).build();
}

std::shared_ptr<Api::ScaledDotProductAttention> findSdpaD1(Api::Network& n) {
    std::shared_ptr<Api::ScaledDotProductAttention> found;
    for (uint32_t i = 0; i < n.getNumLayers(); ++i) {
        auto candidate = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(n.getLayer(i));
        if (!candidate) continue;
        EXPECT_EQ(found, nullptr);
        found = std::move(candidate);
    }
    return found;
}

std::filesystem::path temporaryD1(const char* label) {
    return std::filesystem::temp_directory_path() /
        (std::string(label) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

void checkPhysicalD1(Api::Network& n, const Api::ScaledDotProductAttention& layer,
                     uint32_t b, const std::vector<uint64_t>& expected,
                     bool qRagged, bool kvRagged) {
    std::vector<Event> ready;
    auto placed = n.place(b, ready, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (auto& event : ready) event.synchronize();
    auto& stamp = placed->getStampedNetwork(0);
    auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
        stamp.getPhysicalLayerFromApiLayer(layer.getId()));
    ASSERT_NE(physical, nullptr);
    ASSERT_TRUE(physical->getFeatureOutput().has_value());
    EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), expected);
    EXPECT_EQ(physical->getFeatureOutput()->getDataType(), DataType::FP16);
    // network.place() compiles the real CustomLayer expression, including its
    // descriptor comparison between the prepared equation output and the
    // preallocated public output.  This is not a throwaway compile or run.
    auto checkOffsets = [&](const std::optional<Api::RaggedTensor>& r, DataType dtype) {
        ASSERT_TRUE(r.has_value());
        auto owner = stamp.getManagedPartitionOffsetsInputForTest(r->getRowPartitionId());
        ASSERT_NE(owner, nullptr);
        ASSERT_TRUE(owner->getFeatureOutput().has_value());
        EXPECT_EQ(owner->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(b) + 1}));
        EXPECT_EQ(owner->getFeatureOutput()->getDataType(), dtype);
    };
    if (qRagged)
        checkOffsets(layer.getQueryRaggedInput(), DataType::UINT32);
    if (kvRagged)
        checkOffsets(layer.getKeyRaggedInput(), DataType::UINT64);
}

void expectNeutralD1(const json& arch) {
    size_t sdpaCount = 0;
    for (const auto& layer : arch.at("layers")) {
        if (layer.value("layer_type", std::string{}) != "scaled_dot_product_attention") continue;
        ++sdpaCount;
        EXPECT_EQ(layer.at("version"), "2.2.0");
        EXPECT_FALSE(layer.contains("batch_size"));
        EXPECT_FALSE(layer.contains("max_total_values"));
        for (const char* key : {"query_ragged_input", "key_ragged_input", "value_ragged_input"}) {
            if (!layer.contains(key)) continue;
            EXPECT_FALSE(layer.at(key).contains("geometry_mode"));
            EXPECT_EQ(layer.at(key).at("offsets").at("dimensions"), json::array({1}));
            EXPECT_FALSE(layer.at(key).contains("batch_size"));
            EXPECT_FALSE(layer.at(key).contains("max_total_values"));
        }
    }
    EXPECT_EQ(sdpaCount, 1u);
}
} // namespace

TEST(SdpaNeutralFoundationD1, SavedSeparateKeyAndValueShapesRePlaceAndCompileAtB2B7) {
    REQUIRE_CUDA_D1();
    for (bool kvFirst : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "kvFirst=" << kvFirst);
        const auto archive = temporaryD1("sdpa_foundation_d_cross");
        std::filesystem::remove_all(archive);
        json original;
        {
            Api::Network n("sdpa_foundation_d_cross");
            Api::RaggedTensor q, kv;
            if (kvFirst) {
                kv = raggedD1(n, "kv", 9, 2, 32, DataType::UINT64);
                q = raggedD1(n, "q", 5, 4, 32, DataType::UINT32);
            } else {
                q = raggedD1(n, "q", 5, 4, 32, DataType::UINT32);
                kv = raggedD1(n, "kv", 9, 2, 32, DataType::UINT64);
            }
            const auto v = Api::RaggedNetworkInput::Builder().network(n).name("v")
                .valuesDataType(DataType::FP16).trailingDimensions({2, 48})
                .partition(kv).build();
            const auto sdpa = Api::ScaledDotProductAttention::Builder()
                .network(n).queryInput(q).keyInput(kv).valueInput(v).build();
            (void)Api::NetworkOutput::Builder().network(n).name("out")
                .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
            EXPECT_TRUE(kv.sharesPartitionWith(v));
            EXPECT_FALSE(q.sharesPartitionWith(kv));
            EXPECT_EQ(sdpa.getOutput("output").getDimensions(),
                      (std::vector<uint64_t>{5, 4, 48}));
            original = n.architectureJson();
            expectNeutralD1(original);
            n.save(archive.string(), true);
        }
        // Reversing placement order catches accidental persistent-B ownership.
        for (uint32_t batch : {7u, 2u}) {
            SCOPED_TRACE(::testing::Message() << "B=" << batch);
            Api::Network loaded("sdpa_foundation_d_cross");
            ASSERT_NO_THROW(loaded.load(archive.string()));
            // Independently loaded networks receive fresh graph IDs and
            // autogenerated layer names.  Placement must preserve the loaded
            // instance's own logical architecture, not the source's IDs.
            const json loadedArchitecture = loaded.architectureJson();
            expectNeutralD1(loadedArchitecture);
            ASSERT_EQ(loadedArchitecture.at("layers").size(), original.at("layers").size());
            ASSERT_EQ(loadedArchitecture.at("ragged_network_inputs").size(),
                      original.at("ragged_network_inputs").size());
            auto sdpa = findSdpaD1(loaded);
            ASSERT_NE(sdpa, nullptr);
            ASSERT_TRUE(sdpa->getQueryRaggedInput().has_value());
            ASSERT_TRUE(sdpa->getKeyRaggedInput().has_value());
            ASSERT_TRUE(sdpa->getValueRaggedInput().has_value());
            EXPECT_FALSE(sdpa->getQueryRaggedInput()->sharesPartitionWith(*sdpa->getKeyRaggedInput()));
            EXPECT_TRUE(sdpa->getKeyRaggedInput()->sharesPartitionWith(*sdpa->getValueRaggedInput()));
            EXPECT_TRUE(sdpa->getRaggedFeatureOutput()->sharesPartitionWith(*sdpa->getQueryRaggedInput()));
            checkPhysicalD1(loaded, *sdpa, batch, {static_cast<uint64_t>(batch) * 5, 4, 48}, true, true);
            EXPECT_EQ(loaded.architectureJson(), loadedArchitecture);
        }
        std::filesystem::remove_all(archive);
    }
}

TEST(SdpaNeutralFoundationD1, SavedMixedDomainsRePlaceAndCompileWithoutPackingDenseOutput) {
    REQUIRE_CUDA_D1();
    for (bool raggedQuery : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "raggedQuery=" << raggedQuery);
        const auto archive = temporaryD1("sdpa_foundation_d_mixed");
        std::filesystem::remove_all(archive);
        json original;
        {
            Api::Network n("sdpa_foundation_d_mixed");
            Api::ScaledDotProductAttention sdpa = [&]() {
                if (raggedQuery) {
                    auto q = raggedD1(n, "q", 5, 4, 32, DataType::UINT32);
                    auto k = Api::NetworkInput::Builder().network(n).name("k")
                        .dataType(DataType::FP16).dimensions({9, 2, 32}).build();
                    auto v = Api::NetworkInput::Builder().network(n).name("v")
                        .dataType(DataType::FP16).dimensions({9, 2, 48}).build();
                    return Api::ScaledDotProductAttention::Builder().network(n)
                        .queryInput(q).keyInput(*k.getFeatureOutput())
                        .valueInput(*v.getFeatureOutput()).build();
                }
                auto q = Api::NetworkInput::Builder().network(n).name("q")
                    .dataType(DataType::FP16).dimensions({5, 4, 32}).build();
                auto kv = raggedD1(n, "kv", 9, 2, 32, DataType::UINT64);
                auto v = Api::RaggedNetworkInput::Builder().network(n).name("v")
                    .valuesDataType(DataType::FP16).trailingDimensions({2, 48})
                    .partition(kv).build();
                return Api::ScaledDotProductAttention::Builder().network(n)
                    .queryInput(*q.getFeatureOutput()).keyInput(kv).valueInput(v).build();
            }();
            (void)Api::NetworkOutput::Builder().network(n).name("out")
                .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
            original = n.architectureJson();
            expectNeutralD1(original);
            n.save(archive.string(), true);
        }
        for (uint32_t b : {2u, 7u}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b);
            Api::Network loaded("sdpa_foundation_d_mixed");
            ASSERT_NO_THROW(loaded.load(archive.string()));
            // Independently loaded networks receive fresh graph IDs and
            // autogenerated layer names.  Placement must preserve the loaded
            // instance's own logical architecture, not the source's IDs.
            const json loadedArchitecture = loaded.architectureJson();
            expectNeutralD1(loadedArchitecture);
            ASSERT_EQ(loadedArchitecture.at("layers").size(), original.at("layers").size());
            ASSERT_EQ(loadedArchitecture.at("ragged_network_inputs").size(),
                      original.at("ragged_network_inputs").size());
            auto sdpa = findSdpaD1(loaded);
            ASSERT_NE(sdpa, nullptr);
            const std::vector<uint64_t> dims = raggedQuery
                ? std::vector<uint64_t>{static_cast<uint64_t>(b) * 5, 4, 48}
                : std::vector<uint64_t>{b, 5, 4, 48};
            checkPhysicalD1(loaded, *sdpa, b, dims, raggedQuery, !raggedQuery);
            EXPECT_EQ(loaded.architectureJson(), loadedArchitecture);
        }
        std::filesystem::remove_all(archive);
    }
}

TEST(SdpaNeutralFoundationD1, DenseAndLegacyFixedRaggedPlacementRemainUnchanged) {
    REQUIRE_CUDA_D1();
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        {
            Api::Network n("sdpa_foundation_d_dense_control");
            const auto q = Api::NetworkInput::Builder().network(n).name("q")
                .dataType(DataType::FP16).dimensions({4, 5, 32}).build();
            auto sdpa = Api::ScaledDotProductAttention::Builder()
                .network(n).selfInput(*q.getFeatureOutput()).build();
            (void)Api::NetworkOutput::Builder().network(n).name("out")
                .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
            checkPhysicalD1(n, sdpa, b, {b, 4, 5, 32}, false, false);
        }
        {
            Api::Network n("sdpa_foundation_d_fixed_control");
            auto q = Api::RaggedNetworkInput::Builder().network(n).name("q")
                .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
                .maxValuesPerRow(5).trailingDimensions({4, 32}).build();
            auto sdpa = Api::ScaledDotProductAttention::Builder()
                .network(n).selfInput(q).build();
            (void)Api::NetworkOutput::Builder().network(n).name("out")
                .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
            // The placed ragged output has B * M capacity. No archived
            // batch geometry participates in constructing the logical input.
            checkPhysicalD1(n, sdpa, b, {static_cast<uint64_t>(b) * 5, 4, 32}, true, false);
        }
    }
}
