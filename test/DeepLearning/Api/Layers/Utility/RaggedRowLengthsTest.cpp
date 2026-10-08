#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedRowLengths.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
using std::shared_ptr;
using std::vector;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events) event.synchronize();
    events.clear();
}

shared_ptr<Api::RaggedRowLengths> findRowLengths(Api::Network& network) {
    for (uint32_t layerIndex = 0; layerIndex < network.getNumLayers(); ++layerIndex) {
        auto rowLengths = std::dynamic_pointer_cast<Api::RaggedRowLengths>(network.getLayer(layerIndex));
        if (rowLengths != nullptr) return rowLengths;
    }
    return nullptr;
}

Batch makePlacementNeutralRowLengthsBatch(const std::string& inputName,
                                           const Api::RaggedTensor& logicalInput,
                                           uint32_t batchSize,
                                           vector<int32_t>& expectedLengths) {
    const Impl::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    Impl::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    auto* valuesPtr = values.getMemPtr<float>();
    const uint64_t valuesElements = descriptor.getValuesDescriptor().getArraySizeInBytes() / sizeof(float);
    for (uint64_t i = 0; i < valuesElements; ++i) valuesPtr[i] = static_cast<float>(i + 1);

    expectedLengths.clear();
    expectedLengths.reserve(batchSize);
    std::vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    uint64_t running = 0;
    for (uint32_t row = 0; row < batchSize; ++row) {
        const uint64_t length = row % (logicalInput.getMaxValuesPerRow() + 1);
        expectedLengths.push_back(static_cast<int32_t>(length));
        running += length;
        hostOffsets[row + 1] = running;
    }
    if (running > descriptor.getMaxTotalValues()) {
        throw std::runtime_error("RaggedRowLengths test offsets exceed placement capacity.");
    }

    Impl::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    if (logicalInput.getOffsetsDataType() == DataType::UINT32) {
        uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);
    } else {
        if (logicalInput.getOffsetsDataType() != DataType::UINT64) {
            throw std::runtime_error("RaggedRowLengths test offsets must be uint32 or uint64.");
        }
        uint64_t* offsetsPtr = offsets.getMemPtr<uint64_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = hostOffsets[i];
    }

    Impl::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, Impl::RaggedTensor(values, partition));
    return batch;
}

}  // namespace

TEST(RaggedRowLengthsApi, PlacementNeutralBuilderUsesOnlyPartitionAndPlacementBatchForDenseOutput) {
    constexpr uint64_t maxValuesPerRow = 4;
    Api::Network network("ragged_row_lengths_neutral_builder");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({3})
                                  .maxValuesPerRow(maxValuesPerRow)
                                  .build();
    ASSERT_TRUE(input.isPlacementNeutral());

    Api::RaggedRowLengths rowLengths = Api::RaggedRowLengths::Builder()
                                           .network(network)
                                           .featureInput(input)
                                           .build();
    ASSERT_TRUE(rowLengths.getFeatureInput().has_value());
    ASSERT_TRUE(rowLengths.getFeatureOutput().has_value());
    EXPECT_EQ(rowLengths.getFeatureInput().value(), input.getRowPartitionToken());
    ASSERT_EQ(rowLengths.getAllInputTensors().size(), 1U);
    EXPECT_EQ(rowLengths.getAllInputTensors().front(), input.getRowPartitionToken());
    EXPECT_NE(rowLengths.getAllInputTensors().front(), input.getValues());
    EXPECT_EQ(rowLengths.getFeatureOutput()->getDataType(), DataType::INT32);
    EXPECT_EQ(rowLengths.getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));

    EXPECT_EQ(rowLengths.getOutputTensorBytes(2), 2U * sizeof(int32_t));
    EXPECT_EQ(rowLengths.getOutputTensorBytes(7), 7U * sizeof(int32_t));
    EXPECT_EQ(rowLengths.getFirstInstanceMemRequirementInBytes(2, cpuPlacement),
              3U * sizeof(uint32_t) + 2U * sizeof(int32_t));
    EXPECT_EQ(rowLengths.getFirstInstanceMemRequirementInBytes(7, cpuPlacement),
              8U * sizeof(uint32_t) + 7U * sizeof(int32_t));
}

TEST(RaggedRowLengthsApi, PlacementNeutralSerializationIsSemanticAndRejectsVersionGeometryOrIdentityMismatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    Api::Network neutralNetwork("ragged_row_lengths_neutral_json");
    Api::RaggedTensor neutralInput = Api::RaggedNetworkInput::Builder()
                                         .network(neutralNetwork)
                                         .name("tokens")
                                         .valuesDataType(DataType::FP32)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({2})
                                         .maxValuesPerRow(maxValuesPerRow)
                                         .build();
    Api::RaggedRowLengths neutralRowLengths = Api::RaggedRowLengths::Builder()
                                                  .network(neutralNetwork)
                                                  .featureInput(neutralInput)
                                                  .build();

    const nlohmann::json neutralJson = neutralRowLengths.architectureJson();
    EXPECT_EQ(neutralJson.at("version").get<std::string>(), "1.1.0");
    const nlohmann::json& raggedJson = neutralJson.at("ragged_feature_input");
    EXPECT_EQ(raggedJson.at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
    EXPECT_EQ(raggedJson.at("geometry_mode").get<std::string>(), "placement_neutral");
    EXPECT_EQ(raggedJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
    EXPECT_FALSE(raggedJson.contains("batch_size"));
    EXPECT_FALSE(raggedJson.contains("max_total_values"));
    EXPECT_EQ(raggedJson.at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_EQ(raggedJson.at("offsets").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
    EXPECT_EQ(neutralJson.at("offsets_input").at("id").get<uint64_t>(),
              neutralInput.getRowPartitionToken().getId());
    EXPECT_EQ(neutralJson.at("offsets_input").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
    EXPECT_EQ(neutralJson.at("feature_output").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));

    nlohmann::json neutralAsFixed = neutralJson;
    neutralAsFixed["version"] = "1.0.0";
    EXPECT_THROW(Api::RaggedRowLengths::deserialize(neutralAsFixed, &neutralNetwork), std::runtime_error);

    nlohmann::json wrongOffsetsIdentity = neutralJson;
    wrongOffsetsIdentity["offsets_input"]["id"] = neutralInput.getValues().getId();
    EXPECT_THROW(Api::RaggedRowLengths::deserialize(wrongOffsetsIdentity, &neutralNetwork), std::runtime_error);

    nlohmann::json wrongOffsetsShape = neutralJson;
    wrongOffsetsShape["offsets_input"]["dimensions"] = vector<uint64_t>{2};
    EXPECT_THROW(Api::RaggedRowLengths::deserialize(wrongOffsetsShape, &neutralNetwork), std::runtime_error);

    nlohmann::json wrongOutputShape = neutralJson;
    wrongOutputShape["feature_output"]["dimensions"] = vector<uint64_t>{2};
    EXPECT_THROW(Api::RaggedRowLengths::deserialize(wrongOutputShape, &neutralNetwork), std::runtime_error);

    Api::Network fixedNetwork("ragged_row_lengths_fixed_json");
    Api::RaggedTensor fixedInput = Api::RaggedNetworkInput::Builder()
                                       .network(fixedNetwork)
                                       .name("history")
                                       .valuesDataType(DataType::FP32)
                                       .offsetsDataType(DataType::UINT32)
                                       .trailingDimensions({2})
                                       .batchSize(3)
                                       .maxTotalValues(9)
                                       .build();
    Api::RaggedRowLengths fixedRowLengths = Api::RaggedRowLengths::Builder()
                                                .network(fixedNetwork)
                                                .featureInput(fixedInput)
                                                .build();
    nlohmann::json fixedAsNeutral = fixedRowLengths.architectureJson();
    ASSERT_EQ(fixedAsNeutral.at("version").get<std::string>(), "1.0.0");
    fixedAsNeutral["version"] = "1.1.0";
    EXPECT_THROW(Api::RaggedRowLengths::deserialize(fixedAsNeutral, &fixedNetwork), std::runtime_error);
}

TEST(RaggedRowLengthsApi, PlacementNeutralSaveLoadReplacesPlacementGeometryPerBatchAndExecutesEmptyRows) {
    constexpr uint64_t maxValuesPerRow = 4;
    const std::string networkName = "ragged_row_lengths_placement_neutral_round_trip";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_ragged_row_lengths_neutral_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedValuesOriginalId = 0;
    uint64_t savedPartitionTokenOriginalId = 0;
    uint64_t savedOutputOriginalId = 0;
    {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({1})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
        Api::RaggedRowLengths rowLengths = Api::RaggedRowLengths::Builder()
                                               .network(network)
                                               .featureInput(input)
                                               .build();
        // RaggedRowLengths intentionally consumes only the row-partition token. Explicitly discard
        // the independent packed-values boundary so inference graph validation does not treat it as dangling.
        (void)Api::Stub::Builder().network(network).inputTensor(input.getValues()).build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("lengths")
            .inputTensor(rowLengths.getFeatureOutput().value())
            .dataType(DataType::INT32)
            .build();

        savedValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionTokenOriginalId = input.getRowPartitionToken().getOriginalId();
        savedOutputOriginalId = rowLengths.getFeatureOutput()->getOriginalId();

        const nlohmann::json layerJson = rowLengths.architectureJson();
        EXPECT_EQ(layerJson.at("version").get<std::string>(), "1.1.0");
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("batch_size"));
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("max_total_values"));
        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));

        const auto loadedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedInputs.size(), 1U);
        const Api::RaggedTensor loadedInput = loadedInputs.front().raggedTensor;
        ASSERT_TRUE(loadedInput.isPlacementNeutral());
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
        EXPECT_EQ(loadedInput.getMaxValuesPerRow(), maxValuesPerRow);

        shared_ptr<Api::RaggedRowLengths> loadedRowLengths = findRowLengths(loaded);
        ASSERT_NE(loadedRowLengths, nullptr);
        ASSERT_TRUE(loadedRowLengths->getFeatureInput().has_value());
        ASSERT_TRUE(loadedRowLengths->getFeatureOutput().has_value());
        EXPECT_EQ(loadedRowLengths->getFeatureInput().value(), loadedInput.getRowPartitionToken());
        EXPECT_EQ(loadedRowLengths->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedRowLengths->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        synchronizeEvents(initDoneEvents);

        Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
        const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(loadedInput.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_EQ(requirements.value(), Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);

        const Impl::RaggedTensorDescriptor placedDescriptor =
            loadedInput.getDescriptorForPlacement(placementBatchSize);
        EXPECT_EQ(placedDescriptor.getBatchSize(), placementBatchSize);
        EXPECT_EQ(placedDescriptor.getMaxTotalValues(),
                  static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow);
        EXPECT_EQ(placedDescriptor.getOffsetsDescriptor().getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        shared_ptr<Impl::NetworkInput> physicalOffsets =
            stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(physicalOffsets, nullptr);
        ASSERT_TRUE(physicalOffsets->getFeatureOutput().has_value());
        EXPECT_EQ(physicalOffsets->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        auto physicalRowLengths = std::dynamic_pointer_cast<Impl::CustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(loadedRowLengths->getId()));
        ASSERT_NE(physicalRowLengths, nullptr);
        ASSERT_TRUE(physicalRowLengths->getFeatureOutput().has_value());
        EXPECT_EQ(physicalRowLengths->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{placementBatchSize, 1}));

        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        vector<int32_t> expectedLengths;
        const Batch batch = makePlacementNeutralRowLengthsBatch(
            "tokens", loadedInput, placementBatchSize, expectedLengths);
        const std::map<std::string, Impl::Tensor> outputs = placed->infer(batch);
        ASSERT_TRUE(outputs.contains("lengths"));
        const Impl::Tensor& output = outputs.at("lengths");
        ASSERT_EQ(output.getDataType(), DataType::INT32);
        ASSERT_EQ(output.getDimensions(), (vector<uint64_t>{placementBatchSize, 1}));
        const int32_t* outputValues = output.getMemPtr<int32_t>();
        ASSERT_EQ(expectedLengths.size(), placementBatchSize);
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            EXPECT_EQ(outputValues[row], expectedLengths[row]) << "row " << row;
        }
        ASSERT_FALSE(expectedLengths.empty());
        EXPECT_EQ(expectedLengths.front(), 0);
    };

    try {
        loadPlaceAndRun(2);
        loadPlaceAndRun(7);
    } catch (...) {
        std::filesystem::remove_all(archive);
        throw;
    }
    std::filesystem::remove_all(archive);
}

TEST(RaggedRowLengthsApi, FixedSerializationRemainsVersionOneAndLoads) {
    const std::string networkName = "ragged_row_lengths_fixed_serialization_compatibility";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_ragged_row_lengths_fixed_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("history")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({2})
                                      .batchSize(3)
                                      .maxTotalValues(9)
                                      .build();
        Api::RaggedRowLengths rowLengths = Api::RaggedRowLengths::Builder()
                                               .network(network)
                                               .featureInput(input)
                                               .build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("lengths")
            .inputTensor(rowLengths.getFeatureOutput().value())
            .dataType(DataType::INT32)
            .build();

        const nlohmann::json serialized = rowLengths.architectureJson();
        EXPECT_EQ(serialized.at("version").get<std::string>(), "1.0.0");
        EXPECT_EQ(serialized.at("ragged_feature_input").at("batch_size").get<uint64_t>(), 3U);
        EXPECT_EQ(serialized.at("ragged_feature_input").at("max_total_values").get<uint64_t>(), 9U);
        EXPECT_FALSE(serialized.at("ragged_feature_input").contains("layer_ragged_reference_version"));
        EXPECT_EQ(serialized.at("offsets_input").at("dimensions").get<vector<uint64_t>>(),
                  (vector<uint64_t>{4}));
        network.save(archive.string(), true);
    }

    try {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        shared_ptr<Api::RaggedRowLengths> loadedRowLengths = findRowLengths(loaded);
        ASSERT_NE(loadedRowLengths, nullptr);
        EXPECT_FALSE(loadedRowLengths->getRaggedFeatureInput().isPlacementNeutral());
        EXPECT_EQ(loadedRowLengths->getRaggedFeatureInput().getBatchSize(), 3U);
        EXPECT_EQ(loadedRowLengths->getRaggedFeatureInput().getMaxTotalValues(), 9U);
        EXPECT_EQ(loadedRowLengths->architectureJson().at("version").get<std::string>(), "1.0.0");
    } catch (...) {
        std::filesystem::remove_all(archive);
        throw;
    }
    std::filesystem::remove_all(archive);
}
