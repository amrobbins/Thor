#include "DeepLearning/Api/Layers/Learning/Embedding.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

template <typename LayerT>
shared_ptr<LayerT> findOnlyLayerOfType(Network& network) {
    shared_ptr<LayerT> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<LayerT> candidate = dynamic_pointer_cast<LayerT>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1u);
    return found;
}

filesystem::path uniqueArchivePath(const string& name) {
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    filesystem::path path = filesystem::temp_directory_path() / (name + "_" + to_string(nonce));
    filesystem::remove_all(path);
    return path;
}

}  // namespace

TEST(EmbeddingApi, RaggedBuildPreservesPartitionAndUsesPackedCapacityMemoryAccounting) {
    constexpr uint64_t maxTotalValues = 11;
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t embeddingDim = 5;

    Network network("ragged_embedding_build");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::UINT32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({})
                             .maxTotalValues(maxTotalValues)
                             .maxValuesPerRow(7)
                             .batchSize(batchSize)
                             .build();

    Embedding embedding = Embedding::Builder()
                              .network(network)
                              .featureInput(input)
                              .vocabularySize(32)
                              .embeddingDim(embeddingDim)
                              .weightsDataType(DataType::FP32)
                              .build();

    ASSERT_TRUE(embedding.getUseRagged());
    ASSERT_TRUE(embedding.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(embedding.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = embedding.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{maxTotalValues, embeddingDim}));
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getBatchSize(), batchSize);
    EXPECT_EQ(output.getMaxTotalValues(), maxTotalValues);
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 7u);

    const vector<Tensor> physicalInputs = embedding.getFeatureInputs();
    ASSERT_EQ(physicalInputs.size(), 2u);
    EXPECT_EQ(physicalInputs[0], input.getValues());
    EXPECT_EQ(physicalInputs[1], input.getOffsets());
    EXPECT_EQ(embedding.getOutputTensorBytes(batchSize), output.getValues().getTotalSizeInBytes());

    const json architecture = embedding.architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    ASSERT_EQ(architecture.at("ragged_inputs").size(), 1u);
    ASSERT_EQ(architecture.at("ragged_outputs").size(), 1u);
    EXPECT_EQ(architecture.at("ragged_inputs").at(0).at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_outputs").at(0).at("offsets").at("id").get<uint64_t>());
}

TEST(EmbeddingApi, RaggedMultipleApplicationsPlaceWithIndependentCanonicalPartitions) {
    constexpr uint32_t batchSize = 3;
    Network network("ragged_embedding_multiple_applications");
    RaggedTensor a = RaggedNetworkInput::Builder()
                         .network(network)
                         .name("a")
                         .valuesDataType(DataType::UINT32)
                         .offsetsDataType(DataType::UINT32)
                         .trailingDimensions({})
                         .maxTotalValues(8)
                         .batchSize(batchSize)
                         .build();
    RaggedTensor b = RaggedNetworkInput::Builder()
                         .network(network)
                         .name("b")
                         .valuesDataType(DataType::UINT32)
                         .offsetsDataType(DataType::UINT32)
                         .trailingDimensions({})
                         .maxTotalValues(8)
                         .batchSize(batchSize)
                         .build();

    Embedding embedding = Embedding::Builder()
                              .network(network)
                              .featureInput(a)
                              .featureInput(b)
                              .vocabularySize(16)
                              .embeddingDim(4)
                              .weightsDataType(DataType::FP32)
                              .build();
    ASSERT_TRUE(embedding.getRaggedFeatureOutput(0).has_value());
    ASSERT_TRUE(embedding.getRaggedFeatureOutput(1).has_value());
    EXPECT_EQ(embedding.getRaggedFeatureOutput(0)->getOffsets(), a.getOffsets());
    EXPECT_EQ(embedding.getRaggedFeatureOutput(1)->getOffsets(), b.getOffsets());
    ASSERT_EQ(embedding.getFeatureInputs().size(), 4u);

    (void)RaggedNetworkOutput::Builder()
        .network(network)
        .name("a_out")
        .inputTensor(embedding.getRaggedFeatureOutput(0).value())
        .build();
    (void)RaggedNetworkOutput::Builder()
        .network(network)
        .name("b_out")
        .inputTensor(embedding.getRaggedFeatureOutput(1).value())
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true));
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();
}

TEST(EmbeddingApi, RaggedArchitectureSaveLoadPreservesCapacityAndPartitionMetadata) {
    constexpr uint64_t maxTotalValues = 9;
    constexpr uint64_t maxValuesPerRow = 5;
    constexpr uint32_t batchSize = 3;
    const string name = "ragged_embedding_round_trip";
    const filesystem::path archive = uniqueArchivePath(name);

    try {
        Network network(name);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::UINT32)
                                 .offsetsDataType(DataType::UINT64)
                                 .trailingDimensions({})
                                 .maxTotalValues(maxTotalValues)
                                 .maxValuesPerRow(maxValuesPerRow)
                                 .batchSize(batchSize)
                                 .build();
        Embedding embedding = Embedding::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .vocabularySize(24)
                                  .embeddingDim(6)
                                  .weightsDataType(DataType::FP32)
                                  .build();
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(embedding.getRaggedFeatureOutput().value())
            .build();

        network.save(archive.string(), true);

        Network loaded(name);
        loaded.load(archive.string());
        shared_ptr<Embedding> loadedEmbedding = findOnlyLayerOfType<Embedding>(loaded);
        ASSERT_NE(loadedEmbedding, nullptr);
        ASSERT_TRUE(loadedEmbedding->getUseRagged());
        ASSERT_TRUE(loadedEmbedding->getRaggedFeatureInput().has_value());
        ASSERT_TRUE(loadedEmbedding->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getBatchSize(), batchSize);
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getMaxTotalValues(), maxTotalValues);
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT64);
        ASSERT_TRUE(loadedEmbedding->getRaggedFeatureInput()->hasMaxValuesPerRow());
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getMaxValuesPerRow(), maxValuesPerRow);
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureOutput()->getOffsets(), loadedEmbedding->getRaggedFeatureInput()->getOffsets());
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{maxTotalValues, 6}));
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}
