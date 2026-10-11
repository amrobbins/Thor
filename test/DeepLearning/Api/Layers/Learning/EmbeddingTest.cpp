#include "DeepLearning/Api/Layers/Learning/Embedding.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Stream.h"

#include <cuda_runtime.h>

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdint>
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
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t embeddingDim = 5;

    Network network("ragged_embedding_build");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::UINT32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({})
                             .maxValuesPerRow(7)
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
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{7, embeddingDim}));
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getMaxValuesPerRow(), input.getMaxValuesPerRow());
    EXPECT_EQ(output.getOffsetsDimensions(), (vector<uint64_t>{1}));
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 7u);

    const vector<Tensor> physicalInputs = embedding.getFeatureInputs();
    ASSERT_EQ(physicalInputs.size(), 2u);
    EXPECT_EQ(physicalInputs[0], input.getValues());
    EXPECT_EQ(physicalInputs[1], input.getOffsets());
    EXPECT_TRUE(embedding.outputTensorDimensionsIncludeBatch(output.getValues()));
    EXPECT_EQ(embedding.getOutputTensorBytes(batchSize),
              batchSize * output.getValues().getTotalSizeInBytes());

    const json architecture = embedding.architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    ASSERT_EQ(architecture.at("ragged_inputs").size(), 1u);
    ASSERT_EQ(architecture.at("ragged_outputs").size(), 1u);
    EXPECT_EQ(architecture.at("ragged_inputs").at(0).at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_outputs").at(0).at("offsets").at("id").get<uint64_t>());
}


TEST(EmbeddingApi, RaggedValuesOnlyOutputUsesManagedActiveCountWithoutFullOffsets) {
    constexpr uint32_t batchSize = 2;
    Network network("ragged_embedding_active_count_physicalization");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::UINT32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({})
                             .maxValuesPerRow(3)  // B=2 placement below reserves six values.
                             .build();
    Embedding embedding = Embedding::Builder()
                              .network(network)
                              .featureInput(input)
                              .vocabularySize(16)
                              .embeddingDim(4)
                              .weightsDataType(DataType::FP32)
                              .build();
    ASSERT_TRUE(embedding.getRaggedFeatureOutput().has_value());
    NetworkOutput::Builder()
        .network(network)
        .name("embedded_values")
        .inputTensor(embedding.getRaggedFeatureOutput()->getValues())
        .dataType(DataType::FP32)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true));
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();

    const auto& stamp = placed->getStampedNetwork(0);
    EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId()), nullptr);
    auto activeCount = stamp.getManagedPartitionActiveCountInputForTest(input.getRowPartitionId());
    ASSERT_NE(activeCount, nullptr);
    ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
    EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(activeCount->getFeatureOutput()->getDataType(), DataType::UINT64);
}

TEST(EmbeddingApi, RaggedMultipleApplicationsPlaceWithIndependentCanonicalPartitions) {
    constexpr uint64_t semanticM = 4;
    Network network("ragged_embedding_multiple_applications");
    RaggedTensor a = RaggedNetworkInput::Builder()
                         .network(network)
                         .name("a")
                         .valuesDataType(DataType::UINT32)
                         .offsetsDataType(DataType::UINT32)
                         .trailingDimensions({})
                         .maxValuesPerRow(semanticM)
                         .build();
    RaggedTensor b = RaggedNetworkInput::Builder()
                         .network(network)
                         .name("b")
                         .valuesDataType(DataType::UINT32)
                         .offsetsDataType(DataType::UINT32)
                         .trailingDimensions({})
                         .maxValuesPerRow(semanticM)
                         .build();

    EXPECT_FALSE(a.sharesPartitionWith(b));
    EXPECT_EQ(a.getValuesDimensions(), (vector<uint64_t>{semanticM}));
    EXPECT_EQ(b.getOffsetsDimensions(), (vector<uint64_t>{1}));
    for (uint32_t placementB : {2u, 7u}) {
        EXPECT_EQ(a.getDescriptorForPlacement(placementB).getMaxTotalValues(), placementB * semanticM);
        EXPECT_EQ(b.getDescriptorForPlacement(placementB).getMaxTotalValues(), placementB * semanticM);
    }

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
    EXPECT_TRUE(embedding.getRaggedFeatureOutput(0)->sharesPartitionWith(a));
    EXPECT_TRUE(embedding.getRaggedFeatureOutput(1)->sharesPartitionWith(b));

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

    for (uint32_t placementB : {2u, 7u}) {
        vector<Event> initDoneEvents;
        shared_ptr<PlacedNetwork> placed;
        ASSERT_NO_THROW(placed = network.place(placementB, initDoneEvents, /*inferenceOnly=*/true));
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();
    }
}

TEST(EmbeddingApi, RaggedArchitectureSaveLoadPreservesCapacityAndPartitionMetadata) {
    constexpr uint64_t maxValuesPerRow = 5;
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
                                 .maxValuesPerRow(maxValuesPerRow)
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
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getValuesDimensions().front(), maxValuesPerRow);
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getOffsetsDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT64);
        ASSERT_TRUE(loadedEmbedding->getRaggedFeatureInput()->hasMaxValuesPerRow());
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureInput()->getMaxValuesPerRow(), maxValuesPerRow);
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureOutput()->getOffsets(), loadedEmbedding->getRaggedFeatureInput()->getOffsets());
        EXPECT_EQ(loadedEmbedding->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 6}));
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}


namespace {

RaggedTensor neutralTokens1U(Network& network, uint64_t m = 4) {
    return RaggedNetworkInput::Builder().network(network).name("tokens")
        .valuesDataType(DataType::UINT32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({}).maxValuesPerRow(m).build();
}

Batch embeddingBatch1U(const RaggedTensor& tokens, uint32_t b) {
    namespace Impl = ThorImplementation;
    const auto descriptor = tokens.getDescriptorForPlacement(b);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor values(cpu, descriptor.getValuesDescriptor());
    // Includes empty rows and a padding index (0). The unused packed capacity
    // is deliberately invalid to catch kernels that touch inactive indices.
    vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row) {
        const uint64_t count = row % 3 == 0 ? 0 : (row % 3 == 1 ? 2 : 1);
        offsets[row + 1] = offsets[row] + count;
    }
    const uint64_t capacity = static_cast<uint64_t>(b) * tokens.getMaxValuesPerRow();
    for (uint64_t i = 0; i < capacity; ++i) {
        const uint32_t index = i >= offsets.back() ? UINT32_MAX : (i % 3 == 0 ? 0u : 3u);
        values.getMemPtr<uint32_t>()[i] = index;
    }
    Impl::Tensor gpuOffsets(cpu, descriptor.getOffsetsDescriptor());
    for (uint32_t i = 0; i <= b; ++i)
        gpuOffsets.getMemPtr<uint64_t>()[i] = offsets[i];
    Impl::RowPartitionRuntime partition(gpuOffsets, descriptor.getRowPartition());
    partition.setHostOffsets(offsets);
    Batch result;
    result.insert("tokens", Impl::RaggedTensor(values, partition));
    result.setValidExampleCount(b);
    return result;
}

} // namespace

TEST(RaggedEmbeddingNeutral1U, SemanticGeometryMemoryAndStrictReferences) {
    Network network("neutral_embedding_1u_schema");
    const RaggedTensor input = neutralTokens1U(network);
    Embedding embedding = Embedding::Builder().network(network).featureInput(input)
        .vocabularySize(16).embeddingDim(3).paddingIndex(0).weightsDataType(DataType::FP32).build();
    ASSERT_TRUE(embedding.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = embedding.getRaggedFeatureOutput().value();
    EXPECT_TRUE(output.sharesPartitionWith(input));
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{4, 3}));
    EXPECT_EQ(output.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_NE(output.getValues().getOriginalId(), input.getValues().getOriginalId());
    EXPECT_EQ(embedding.getRaggedPartitionRequirementForInput(input.getRowPartitionToken()),
              ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u, 7u}) {
        EXPECT_EQ(embedding.getOutputTensorBytes(b), b * 4u * 3u * sizeof(float));
        EXPECT_EQ(embedding.getNonFirstInstanceMemRequirementInBytes(b, cpu),
                  embedding.getOutputTensorBytes(b));
        EXPECT_EQ(embedding.getFirstInstanceMemRequirementInBytes(b, cpu),
                  embedding.getOutputTensorBytes(b) + 16u * 3u * sizeof(float));
    }
    EXPECT_ANY_THROW((void)embedding.getOutputTensorBytes(0));
    const json record = embedding.architectureJson();
    EXPECT_EQ(record.at("version"), "1.1.0");
    for (const char* key : {"ragged_inputs", "ragged_outputs"}) {
        const json& r = record.at(key).at(0);
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
        EXPECT_EQ(r.at("offsets").at("dimensions"), (vector<uint64_t>{1}));
    }
    std::shared_ptr<thor_file::TarReader> reader;
    auto rejected = [&](const json& corrupt) { EXPECT_ANY_THROW(Embedding::deserialize(reader, corrupt, &network)); };
    // A previous writer emitted 1.2.0 only to distinguish the canonical
    // logical grammar from the bugged packed representation. It remains a
    // readable historical input version, not a different execution mode.
    json priorCanonical = record;
    priorCanonical["version"] = "1.2.0";
    // Earlier canonical archives had a redundant mode marker; it is accepted
    // only for reading and does not alter the logical M or partition.
    for (const char* key : {"ragged_inputs", "ragged_outputs"}) {
        for (auto& reference : priorCanonical[key]) {
            reference["geometry_mode"] = "placement_neutral";
        }
    }
    ASSERT_NO_THROW(Embedding::deserialize(reader, priorCanonical, &network));
    ASSERT_NO_THROW(Embedding::deserialize(reader, record, &network));
    json bad = record; bad["version"] = "1.0.0"; rejected(bad);
    bad = record; bad["version"] = "1.3.0"; rejected(bad);
    bad = record; bad["max_total_values"] = 28; rejected(bad);
    bad = record; bad["ragged_inputs"][0]["batch_size"] = 7; rejected(bad);
    bad = record; bad["ragged_outputs"][0]["max_total_values"] = 28; rejected(bad);
    bad = record; bad["ragged_inputs"][0]["values"]["data_type"] = DataType::UINT16; rejected(bad);
    bad = record; bad["ragged_inputs"][0]["values"]["dimensions"] = vector<uint64_t>{7}; rejected(bad);
    bad = record; bad["ragged_inputs"][0]["offsets"]["id"] = input.getValues().getOriginalId(); rejected(bad);
    bad = record; bad["ragged_outputs"][0]["offsets"]["id"] = input.getValues().getOriginalId(); rejected(bad);
    bad = record; bad["ragged_outputs"][0]["values"]["data_type"] = DataType::FP16; rejected(bad);
    bad = record; bad["ragged_outputs"][0]["values"]["dimensions"] = vector<uint64_t>{4,4}; rejected(bad);
    bad = record; bad["ragged_outputs"][0]["max_values_per_row"] = 8; rejected(bad);
    bad = record; bad["outputs"][0]["dimensions"] = vector<uint64_t>{4, 4}; rejected(bad);
    bad = record; bad["inputs"][0]["dimensions"] = vector<uint64_t>{9}; rejected(bad);
    bad = record; bad["parameters"]["weights"]["shape"] = vector<uint64_t>{16, 4}; rejected(bad);
    bad = record; bad["padding_index"] = 20; rejected(bad);
    bad = record; bad["extra_runtime_batch_size"] = 7; rejected(bad);

    auto historical = [](json ref) {
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref["batch_size"] = 91;      // obsolete execution hint, not semantic M
        ref["max_total_values"] = 999;
        ref["values"]["dimensions"][0] = 123;
        ref["offsets"]["dimensions"] = vector<uint64_t>{92};
        return ref;
    };
    json historicalLayer = record;
    historicalLayer["version"] = "1.1.0";
    historicalLayer["ragged_inputs"][0] = historical(historicalLayer["ragged_inputs"][0]);
    historicalLayer["ragged_outputs"][0] = historical(historicalLayer["ragged_outputs"][0]);
    historicalLayer["inputs"][0]["dimensions"][0] = 123;
    historicalLayer["outputs"][0]["dimensions"][0] = 123;
    bad = historicalLayer; bad["version"] = "1.2.0"; rejected(bad);
    ASSERT_NO_THROW(Embedding::deserialize(reader, historicalLayer, &network));
    auto recovered = dynamic_pointer_cast<Embedding>(network.getLayer(network.getNumLayers()-1));
    ASSERT_NE(recovered, nullptr);
    ASSERT_TRUE(recovered->getRaggedFeatureOutput().has_value());
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(recovered->getRaggedFeatureOutput()->getDescriptorForPlacement(b).getMaxTotalValues(), b*4u);
    bad = historicalLayer; bad["ragged_outputs"][0]["offsets"]["id"] = input.getValues().getOriginalId(); rejected(bad);
}

TEST(RaggedEmbeddingNeutral1U, SavedNetworkReplacesAtB2AndB7WithEmptyRowsAndPadding) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const auto archive = uniqueArchivePath("neutral_embedding_1u_saved");
    {
        Network network("neutral_embedding_1u_run");
        const auto input = neutralTokens1U(network);
        const auto embedding = Embedding::Builder().network(network).featureInput(input)
            .vocabularySize(16).embeddingDim(3).paddingIndex(0).weightsDataType(DataType::FP32).build();
        (void)RaggedNetworkOutput::Builder().network(network).name("embedded")
            .inputTensor(embedding.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network("neutral_embedding_1u_run");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto inputs = network.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 1u);
        const auto input = inputs.front().raggedTensor;
        const auto embedding = findOnlyLayerOfType<Embedding>(network);
        ASSERT_NE(embedding, nullptr);
        ASSERT_TRUE(embedding->getRaggedFeatureOutput().has_value());
        EXPECT_TRUE(embedding->getRaggedFeatureOutput()->sharesPartitionWith(input));
        const json before = network.architectureJson();
        vector<Event> events;
        shared_ptr<PlacedNetwork> placed;
        ASSERT_NO_THROW(placed = network.place(b, events, true));
        ASSERT_NE(placed, nullptr);
        for (Event& e : events) e.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        ASSERT_EQ(stamped.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId()), nullptr);
        auto activeCount = stamped.getManagedPartitionActiveCountInputForTest(input.getRowPartitionId());
        ASSERT_NE(activeCount, nullptr);
        EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        auto physical = dynamic_pointer_cast<ThorImplementation::Embedding>(
            stamped.getPhysicalLayerFromApiLayer(embedding->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * 4u, 3}));
        EXPECT_EQ(network.architectureJson(), before);
        const auto outputMap = placed->inferLogical(embeddingBatch1U(input, b));
        ASSERT_TRUE(outputMap.contains("embedded"));
        ASSERT_TRUE(holds_alternative<ThorImplementation::RaggedTensor>(outputMap.at("embedded")));
        const auto result = get<ThorImplementation::RaggedTensor>(outputMap.at("embedded"));
        const auto ends = result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        ASSERT_EQ(ends->size(), b + 1u);
        for (uint32_t row = 0; row < b; ++row) {
            const uint64_t expected = row % 3 == 0 ? 0 : (row % 3 == 1 ? 2 : 1);
            EXPECT_EQ(ends->at(row + 1) - ends->at(row), expected);
        }
        const float* out = result.getValues().getMemPtr<float>();
        for (uint64_t i = 0; i < ends->back(); ++i) {
            if (i % 3 == 0) {
                for (uint32_t c = 0; c < 3; ++c) EXPECT_NEAR(out[i*3+c], 0.0f, 1e-6f);
            }
            for (uint32_t c = 0; c < 3; ++c) EXPECT_TRUE(std::isfinite(out[i*3+c]));
        }
    }
    filesystem::remove_all(archive);
}


TEST(RaggedEmbeddingNeutral1U, SavedTrainingBackpropagatesSparseWeightsWithoutPaddingOrInactiveRows) {
    namespace Impl = ThorImplementation;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const auto archive = uniqueArchivePath("neutral_embedding_1u_train");
    {
        Network network("neutral_embedding_1u_train");
        (void)Sgd::Builder().network(network).initialLearningRate(0.01f)
            .decay(0.0f).momentum(0.0f).build();
        const auto tokens = neutralTokens1U(network);
        const auto labels = RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({3}).partition(tokens).build();
        const auto embedding = Embedding::Builder().network(network).featureInput(tokens)
            .vocabularySize(16).embeddingDim(3).paddingIndex(0).weightsDataType(DataType::FP32).build();
        auto prediction = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
        auto label = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
        auto delta = prediction - label;
        auto loss = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss", (delta * delta).withOutputDType(DataType::FP32)}})));
        auto grad = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad", (delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(embedding.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network("neutral_embedding_1u_train");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto allInputs = network.getExternalRaggedNetworkInputs();
        ASSERT_EQ(allInputs.size(), 2u);
        const auto tokensIt = find_if(allInputs.begin(), allInputs.end(),
            [](const auto& x) { return x.name == "tokens"; });
        const auto labelsIt = find_if(allInputs.begin(), allInputs.end(),
            [](const auto& x) { return x.name == "labels"; });
        ASSERT_NE(tokensIt, allInputs.end());
        ASSERT_NE(labelsIt, allInputs.end());
        const auto tokens = tokensIt->raggedTensor;
        const auto labels = labelsIt->raggedTensor;
        EXPECT_TRUE(tokens.sharesPartitionWith(labels));
        const auto embedding = findOnlyLayerOfType<Embedding>(network);
        ASSERT_NE(embedding, nullptr);
        const json before = network.architectureJson();
        vector<Event> events;
        auto placed = network.place(b, events, /*inferenceOnly=*/false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = dynamic_pointer_cast<Impl::Embedding>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(embedding->getId()));
        ASSERT_NE(physical, nullptr);
        const auto weights = physical->getParameterStorage("weights");
        auto downloadWeights = [&]() {
            Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
            auto copy = weights.clone(cpu);
            Impl::Tensor original = weights;
            Stream stream = Stream::getNextDownloadStream(weights.getPlacement().getDeviceNum());
            copy.downloadSection(original, stream, 0, 0, 16u * 3u * sizeof(float));
            stream.synchronize();
            return vector<float>(copy.getMemPtr<float>(), copy.getMemPtr<float>() + 16u * 3u);
        };
        const auto prior = downloadWeights();
        const auto desc = tokens.getDescriptorForPlacement(b);
        const auto labelDesc = labels.getDescriptorForPlacement(b);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor indices(cpu, desc.getValuesDescriptor());
        Impl::Tensor target(cpu, labelDesc.getValuesDescriptor());
        Impl::Tensor ends(cpu, desc.getOffsetsDescriptor());
        vector<uint64_t> offsets(b + 1u, 0);
        for (uint32_t i = 0; i < b; ++i) offsets[i+1] = offsets[i] + (i % 2 == 0 ? 0u : 2u);
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0u);
        for (uint64_t t = 0; t < static_cast<uint64_t>(b) * 4u; ++t) {
            indices.getMemPtr<uint32_t>()[t] = t >= active ? UINT32_MAX : (t % 2 == 0 ? 0u : 3u);
            for (uint32_t c = 0; c < 3; ++c) target.getMemPtr<float>()[3*t+c] = 0.f;
        }
        for (uint32_t i = 0; i <= b; ++i) ends.getMemPtr<uint64_t>()[i] = offsets[i];
        Impl::RowPartitionRuntime partition(ends, desc.getRowPartition());
        partition.setHostOffsets(offsets);
        Batch training;
        training.insert("tokens", Impl::RaggedTensor(indices, partition));
        training.insert("labels", Impl::RaggedTensor(target, partition));
        training.setValidExampleCount(b);
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, training, outputs, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        EXPECT_EQ(outputs.at("raw_loss").getDimensions(), (vector<uint64_t>{b*4u,3}));
        const auto updated = downloadWeights();
        bool changedRow3 = false;
        for (uint32_t c = 0; c < 3; ++c) {
            changedRow3 |= fabs(updated[3u*3u+c] - prior[3u*3u+c]) > 1e-8f;
            EXPECT_NEAR(updated[c], prior[c], 1e-7f) << "padding row changed";
            EXPECT_NEAR(updated[6u*3u+c], prior[6u*3u+c], 1e-7f) << "unused row changed";
        }
        EXPECT_TRUE(changedRow3) << "sparse gradient did not update the active non-padding vocabulary row";
        EXPECT_EQ(network.architectureJson(), before);
    }
    filesystem::remove_all(archive);
}
