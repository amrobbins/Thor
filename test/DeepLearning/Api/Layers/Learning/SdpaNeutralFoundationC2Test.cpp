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
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
using json = nlohmann::json;

Api::RaggedTensor neutralC2(Api::Network& network, const char* name,
                            uint64_t maxValues, uint64_t heads,
                            DataType offsetsType = DataType::UINT64) {
    return Api::RaggedNetworkInput::Builder().network(network).name(name)
        .valuesDataType(DataType::FP16).offsetsDataType(offsetsType)
        .maxValuesPerRow(maxValues).trailingDimensions({heads, 32}).build();
}

std::shared_ptr<Api::ScaledDotProductAttention> findSdpaC2(Api::Network& network) {
    std::shared_ptr<Api::ScaledDotProductAttention> found;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto candidate = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getLayer(i));
        if (candidate != nullptr) {
            EXPECT_EQ(found, nullptr) << "Expected one SDPA layer";
            found = std::move(candidate);
        }
    }
    return found;
}

std::filesystem::path archiveC2(const char* name) {
    return std::filesystem::temp_directory_path() /
        (std::string(name) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

void expectNoRuntimeGeometryC2(const json& architecture) {
    const json& layers = architecture.at("layers");
    size_t sdpaCount = 0;
    for (const json& layer : layers) {
        if (layer.value("layer_type", std::string{}) != "scaled_dot_product_attention") continue;
        ++sdpaCount;
        EXPECT_EQ(layer.at("version"), "2.2.0");
        EXPECT_FALSE(layer.contains("batch_size"));
        EXPECT_FALSE(layer.contains("max_total_values"));
        for (const char* key : {"query_ragged_input", "key_ragged_input", "value_ragged_input"}) {
            if (!layer.contains(key)) continue;
            EXPECT_FALSE(layer.at(key).contains("batch_size"));
            EXPECT_FALSE(layer.at(key).contains("max_total_values"));
            EXPECT_EQ(layer.at(key).at("offsets").at("dimensions"), json::array({1}));
        }
    }
    EXPECT_GT(sdpaCount, 0u);
}

std::map<std::string, std::vector<char>> archiveBytesC2(const std::filesystem::path& directory) {
    std::map<std::string, std::vector<char>> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::ifstream input(entry.path(), std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open saved SDPA archive");
        files[entry.path().filename().string()] = std::vector<char>(
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    return files;
}

#define REQUIRE_CUDA_C2() do {                                              \
    int count = 0;                                                          \
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {          \
        GTEST_SKIP() << "CUDA device required for saved SDPA placement";   \
    }                                                                       \
} while (false)
} // namespace

TEST(SdpaNeutralFoundationC2, SavedCrossAttentionReloadsIndependentlyAtB2AndB7) {
    REQUIRE_CUDA_C2();
    const auto archive = archiveC2("sdpa_neutral_c2_saved_cross");
    std::filesystem::remove_all(archive);
    json savedArchitecture;
    {
        Api::Network network("sdpa_neutral_c2_saved_cross");
        const auto q = neutralC2(network, "q", 3, 4, DataType::UINT32);
        const auto kv = neutralC2(network, "kv", 7, 2, DataType::UINT64);
        const auto sdpa = Api::ScaledDotProductAttention::Builder()
            .network(network).queryInput(q).keyInput(kv).valueInput(kv).build();
        (void)Api::NetworkOutput::Builder().network(network).name("attention_out")
            .inputTensor(sdpa.getOutput("output")).dataType(DataType::FP16).build();
        savedArchitecture = network.architectureJson();
        expectNoRuntimeGeometryC2(savedArchitecture);
        network.save(archive.string(), true);
    }

    const auto exactSavedBytes = archiveBytesC2(archive);
    ASSERT_FALSE(exactSavedBytes.empty());
    for (uint32_t batch : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << batch);
        Api::Network network("sdpa_neutral_c2_saved_cross");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto layer = findSdpaC2(network);
        ASSERT_NE(layer, nullptr);
        ASSERT_TRUE(layer->getQueryRaggedInput().has_value());
        ASSERT_TRUE(layer->getKeyRaggedInput().has_value());
        ASSERT_TRUE(layer->getValueRaggedInput().has_value());
        const auto q = layer->getQueryRaggedInput().value();
        const auto k = layer->getKeyRaggedInput().value();
        const auto v = layer->getValueRaggedInput().value();
        EXPECT_TRUE(k.sharesPartitionWith(v));
        EXPECT_FALSE(q.sharesPartitionWith(k));
        EXPECT_EQ(q.getMaxValuesPerRow(), 3u);
        EXPECT_EQ(k.getMaxValuesPerRow(), 7u);
        EXPECT_TRUE(layer->getRaggedFeatureOutput()->sharesPartitionWith(q));
        const json loadedArchitecture = network.architectureJson();
        expectNoRuntimeGeometryC2(loadedArchitecture);
        std::vector<Event> ready;
        auto placed = network.place(batch, ready, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : ready) event.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        const auto physical = std::dynamic_pointer_cast<Impl::CustomLayer>(
            stamp.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(batch) * 3u, 4, 32}));
        for (auto partitionId : {q.getRowPartitionId(), k.getRowPartitionId()}) {
            const auto offsets = stamp.getManagedPartitionOffsetsInputForTest(partitionId);
            ASSERT_NE(offsets, nullptr);
            ASSERT_TRUE(offsets->getFeatureOutput().has_value());
            EXPECT_EQ(offsets->getFeatureOutput()->getDimensions(),
                      (std::vector<uint64_t>{static_cast<uint64_t>(batch) + 1u}));
        }
        EXPECT_EQ(network.architectureJson(), loadedArchitecture);
        EXPECT_EQ(archiveBytesC2(archive), exactSavedBytes);
    }
    std::filesystem::remove_all(archive);
}

TEST(SdpaNeutralFoundationC2, CloneSubgraphRemapsIndependentQKvPartitionsAndOutput) {
    Api::Network source("sdpa_neutral_c2_source");
    const auto q = neutralC2(source, "q", 3, 4, DataType::UINT32);
    const auto kv = neutralC2(source, "kv", 7, 2, DataType::UINT64);
    const auto sourceSdpa = Api::ScaledDotProductAttention::Builder()
        .network(source).queryInput(q).keyInput(kv).valueInput(kv).build();
    (void)Api::NetworkOutput::Builder().network(source).name("attention_out")
        .inputTensor(sourceSdpa.getOutput("output")).dataType(DataType::FP16).build();
    const json sourceArchitecture = source.architectureJson();

    Api::Network destination("sdpa_neutral_c2_destination");
    const auto destinationQ = neutralC2(destination, "q", 3, 4, DataType::UINT32);
    const auto destinationKv = neutralC2(destination, "kv", 7, 2, DataType::UINT64);
    Api::ApiTensorRemap remap;
    remap.map(q.getValues(), destinationQ.getValues());
    remap.map(q.getRowPartitionToken(), destinationQ.getRowPartitionToken());
    remap.map(kv.getValues(), destinationKv.getValues());
    remap.map(kv.getRowPartitionToken(), destinationKv.getRowPartitionToken());
    Api::ApiSubgraphCloneOptions options;
    options.namePrefix = "copy/";
    options.inferenceOnly = true;
    const auto cloned = destination.cloneSubgraphInto(source, {"attention_out"}, remap, options);
    ASSERT_EQ(cloned.outputTensorsByName.size(), 1u);
    EXPECT_EQ(cloned.outputTensorsByName.at("attention_out").getDimensions(),
              (std::vector<uint64_t>{3, 4, 32}));
    EXPECT_NE(cloned.outputTensorsByName.at("attention_out"), sourceSdpa.getOutput("output"));
    (void)Api::NetworkOutput::Builder().network(destination).name("cloned_attention_out")
        .inputTensor(cloned.outputTensorsByName.at("attention_out"))
        .dataType(DataType::FP16).build();
    const auto clonedSdpa = findSdpaC2(destination);
    ASSERT_NE(clonedSdpa, nullptr);
    ASSERT_TRUE(clonedSdpa->getQueryRaggedInput().has_value());
    ASSERT_TRUE(clonedSdpa->getKeyRaggedInput().has_value());
    EXPECT_TRUE(clonedSdpa->getQueryRaggedInput()->sharesPartitionWith(destinationQ));
    EXPECT_TRUE(clonedSdpa->getKeyRaggedInput()->sharesPartitionWith(destinationKv));
    EXPECT_FALSE(clonedSdpa->getQueryRaggedInput()->sharesPartitionWith(destinationKv));
    EXPECT_TRUE(clonedSdpa->getRaggedFeatureOutput()->sharesPartitionWith(destinationQ));
    EXPECT_EQ(clonedSdpa->architectureJson().at("version"), "2.2.0");
    EXPECT_EQ(source.architectureJson(), sourceArchitecture);
    expectNoRuntimeGeometryC2(destination.architectureJson());
}
