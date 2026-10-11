#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Layers/Utility/Transpose.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

const nlohmann::json& findLayerJson(const nlohmann::json& architecture, const std::string& layerType) {
    for (const nlohmann::json& layerJson : architecture.at("layers")) {
        if (layerJson.at("layer_type").get<std::string>() == layerType) return layerJson;
    }
    throw std::runtime_error("Missing layer JSON for " + layerType);
}

template <typename LayerT>
std::shared_ptr<LayerT> findOnlyLayerOfType(Api::Network& network) {
    std::shared_ptr<LayerT> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        std::shared_ptr<LayerT> candidate = std::dynamic_pointer_cast<LayerT>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1U);
    return found;
}

Batch makeSliceTransposeBatch(const std::string& inputName,
                              const Api::RaggedTensor& logicalInput,
                              uint32_t batchSize) {
    constexpr uint64_t activeValuesPerRow = 2;
    constexpr uint64_t elementsPerValue = 8;
    const Impl::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    const uint64_t capacity = descriptor.getMaxTotalValues();
    const uint64_t activeValues = static_cast<uint64_t>(batchSize) * activeValuesPerRow;
    THOR_THROW_IF_FALSE(logicalInput.getTrailingDimensions() == (std::vector<uint64_t>{2, 4}));
    THOR_THROW_IF_FALSE(logicalInput.getMaxValuesPerRow() >= activeValuesPerRow);
    THOR_THROW_IF_FALSE(activeValues <= capacity);

    Impl::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t i = 0; i < capacity * elementsPerValue; ++i) valuesPtr[i] = -1000.0F;
    for (uint32_t row = 0; row < batchSize; ++row) {
        for (uint64_t valueInRow = 0; valueInRow < activeValuesPerRow; ++valueInRow) {
            const uint64_t packedValue = static_cast<uint64_t>(row) * activeValuesPerRow + valueInRow;
            const float base = static_cast<float>(row * 100 + valueInRow * 10);
            for (uint64_t element = 0; element < elementsPerValue; ++element) {
                valuesPtr[packedValue * elementsPerValue + element] = base + static_cast<float>(element);
            }
        }
    }

    Impl::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    std::vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    for (uint32_t row = 0; row <= batchSize; ++row) {
        hostOffsets[row] = static_cast<uint64_t>(row) * activeValuesPerRow;
    }
    THOR_THROW_IF_FALSE(logicalInput.getOffsetsDataType() == Api::DataType::UINT32);
    uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
    for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);

    Impl::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, Impl::RaggedTensor(values, partition));
    return batch;
}

void expectNeutralReference(const nlohmann::json& raggedJson, uint64_t maxValuesPerRow) {
    EXPECT_EQ(raggedJson.at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
    EXPECT_FALSE(raggedJson.contains("geometry_mode"));
    EXPECT_EQ(raggedJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
    EXPECT_FALSE(raggedJson.contains("batch_size"));
    EXPECT_FALSE(raggedJson.contains("max_total_values"));
    EXPECT_EQ(raggedJson.at("values").at("dimensions").at(0).get<uint64_t>(), maxValuesPerRow);
    EXPECT_EQ(raggedJson.at("offsets").at("dimensions").get<std::vector<uint64_t>>(),
              (std::vector<uint64_t>{1}));
}

void expectPreservedPartition(const Api::RaggedTensor& tensor, uint64_t partitionTokenOriginalId) {
    EXPECT_EQ(tensor.getRowPartitionToken().getOriginalId(), partitionTokenOriginalId);
}

}  // namespace

TEST(UtilityApiLayers, PlacementNeutralSliceActiveCountManagedCarrierSuppliesPlacementDescriptorForStamp) {
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint32_t placementBatchSize = 7;

    Api::Network network("slice_active_count_managed_carrier_metadata");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .offsetsDataType(Api::DataType::UINT32)
                                  .trailingDimensions({2, 4})
                                  .maxValuesPerRow(maxValuesPerRow)
                                  .build();
    Api::Slice slice = Api::Slice::Builder()
                           .network(network)
                           .featureInput(input)
                           .axis(1)
                           .start(1)
                           .length(3)
                           .build();
    ASSERT_TRUE(slice.getRaggedFeatureOutput().has_value());
    (void)Api::RaggedNetworkOutput::Builder()
        .network(network)
        .name("result")
        .inputTensor(slice.getRaggedFeatureOutput().value())
        .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(placementBatchSize, initDoneEvents, true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(input.getRowPartitionId());
    ASSERT_TRUE(requirements.has_value());
    EXPECT_EQ(requirements.value(), Impl::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);

    std::shared_ptr<Impl::NetworkInput> activeCountInput =
        stamped.getManagedPartitionActiveCountInputForTest(input.getRowPartitionId());
    ASSERT_NE(activeCountInput, nullptr);
    ASSERT_TRUE(activeCountInput->getFeatureOutput().has_value());
    EXPECT_EQ(activeCountInput->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{1}));
    EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId()), nullptr);

    std::shared_ptr<Impl::TensorFanout> managedCarrier;
    for (const std::shared_ptr<Impl::Layer>& layer : stamped.getOtherLayers()) {
        auto fanout = std::dynamic_pointer_cast<Impl::TensorFanout>(layer);
        if (fanout == nullptr || !fanout->getRowPartitionDescriptorForStamp().has_value()) continue;
        ASSERT_EQ(managedCarrier, nullptr) << "Expected exactly one managed row-partition fanout for active-count-only Slice.";
        managedCarrier = fanout;
    }
    ASSERT_NE(managedCarrier, nullptr);
    ASSERT_TRUE(managedCarrier->getFeatureOutput().has_value());
    EXPECT_EQ(managedCarrier->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{1}));

    const Impl::RaggedTensorDescriptor expected = input.getDescriptorForPlacement(placementBatchSize);
    const std::optional<Impl::RowPartitionDescriptor> attached = managedCarrier->getRowPartitionDescriptorForStamp();
    ASSERT_TRUE(attached.has_value());
    EXPECT_EQ(attached.value(), expected.getRowPartition());

    const Impl::RaggedTensorDescriptor fromActiveCountDriver =
        Api::SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            input, managedCarrier, "Slice");
    EXPECT_EQ(fromActiveCountDriver, expected);
}

TEST(UtilityApiLayers, PlacementNeutralSliceTransposeChainSaveLoadReplacesPlacementGeometryPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint32_t batchOne = 2;
    constexpr uint32_t batchTwo = 7;
    const std::string networkName = "placement_neutral_slice_transpose_chain";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_neutral_slice_transpose_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedPartitionTokenOriginalId = 0;
    {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(Api::DataType::FP32)
                                      .offsetsDataType(Api::DataType::UINT32)
                                      .trailingDimensions({2, 4})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
        Api::Slice slice = Api::Slice::Builder()
                               .network(network)
                               .featureInput(input)
                               .axis(1)
                               .start(1)
                               .length(3)
                               .build();
        ASSERT_TRUE(slice.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor sliced = slice.getRaggedFeatureOutput().value();
        Api::Transpose transpose = Api::Transpose::Builder().network(network).featureInput(sliced).build();
        ASSERT_TRUE(transpose.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor transposed = transpose.getRaggedFeatureOutput().value();
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("result")
            .inputTensor(transposed)
            .build();

        savedPartitionTokenOriginalId = input.getRowPartitionToken().getOriginalId();
        for (const Api::RaggedTensor& tensor : {input, sliced, transposed}) {
            EXPECT_EQ(tensor.getRowPartitionToken(), input.getRowPartitionToken());
            EXPECT_EQ(tensor.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
            ASSERT_TRUE(tensor.hasMaxValuesPerRow());
            EXPECT_EQ(tensor.getMaxValuesPerRow(), maxValuesPerRow);
        }
        EXPECT_EQ(input.getValuesDimensions(), (std::vector<uint64_t>{4, 2, 4}));
        EXPECT_EQ(sliced.getValuesDimensions(), (std::vector<uint64_t>{4, 2, 3}));
        EXPECT_EQ(transposed.getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));
        ASSERT_TRUE(slice.getFeatureOutput().has_value());
        ASSERT_TRUE(transpose.getFeatureOutput().has_value());
        EXPECT_FALSE(slice.outputTensorDimensionsIncludeBatch(slice.getFeatureOutput().value()));
        EXPECT_FALSE(transpose.outputTensorDimensionsIncludeBatch(transpose.getFeatureOutput().value()));

        const nlohmann::json architecture = network.architectureJson();
        const nlohmann::json& sliceJson = findLayerJson(architecture, "slice");
        const nlohmann::json& transposeJson = findLayerJson(architecture, "transpose");
        for (const nlohmann::json* layerJson : {&sliceJson, &transposeJson}) {
            EXPECT_EQ(layerJson->at("version").get<std::string>(), "1.1.0");
            ASSERT_TRUE(layerJson->at("use_ragged").get<bool>());
            expectNeutralReference(layerJson->at("ragged_feature_input"), maxValuesPerRow);
            expectNeutralReference(layerJson->at("ragged_feature_output"), maxValuesPerRow);
            EXPECT_EQ(layerJson->at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
                      layerJson->at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());
        }
        EXPECT_EQ(sliceJson.at("axis").get<uint64_t>(), 1U);
        EXPECT_EQ(sliceJson.at("start").get<int64_t>(), 1);
        EXPECT_EQ(sliceJson.at("length").get<uint64_t>(), 3U);
        EXPECT_EQ(sliceJson.at("ragged_feature_output").at("values").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{4, 2, 3}));
        EXPECT_TRUE(transposeJson.at("epilogue").is_null());
        EXPECT_EQ(transposeJson.at("ragged_feature_output").at("values").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{4, 3, 2}));

        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));

        const auto loadedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedInputs.size(), 1U);
        const Api::RaggedTensor loadedInput = loadedInputs.front().raggedTensor;
        expectPreservedPartition(loadedInput, savedPartitionTokenOriginalId);

        const auto loadedOutputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(loadedOutputs.size(), 1U);
        const Api::RaggedTensor loadedBoundaryOutput = loadedOutputs.front().raggedTensor;
        expectPreservedPartition(loadedBoundaryOutput, savedPartitionTokenOriginalId);
        EXPECT_EQ(loadedBoundaryOutput.getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));

        const auto loadedSlice = findOnlyLayerOfType<Api::Slice>(loaded);
        const auto loadedTranspose = findOnlyLayerOfType<Api::Transpose>(loaded);
        ASSERT_NE(loadedSlice, nullptr);
        ASSERT_NE(loadedTranspose, nullptr);
        ASSERT_TRUE(loadedSlice->getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(loadedTranspose->getRaggedFeatureOutput().has_value());
        expectPreservedPartition(loadedSlice->getRaggedFeatureOutput().value(), savedPartitionTokenOriginalId);
        expectPreservedPartition(loadedTranspose->getRaggedFeatureOutput().value(), savedPartitionTokenOriginalId);
        EXPECT_EQ(loadedSlice->getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{4, 2, 3}));
        EXPECT_EQ(loadedTranspose->getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
        const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(loadedInput.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_EQ(requirements.value(), Impl::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
        std::shared_ptr<Impl::NetworkInput> activeCountInput =
            stamped.getManagedPartitionActiveCountInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(activeCountInput, nullptr);
        ASSERT_TRUE(activeCountInput->getFeatureOutput().has_value());
        EXPECT_EQ(activeCountInput->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{1}));
        EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId()), nullptr);

        std::shared_ptr<Impl::NetworkInput> valuesInput = stamped.getNamedInput("tokens.values");
        ASSERT_NE(valuesInput, nullptr);
        ASSERT_TRUE(valuesInput->getFeatureOutput().has_value());
        EXPECT_EQ(valuesInput->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{expectedCapacity, 2, 4}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const std::map<std::string, Api::InferenceOutputValue> outputs =
            placed->inferLogical(makeSliceTransposeBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("result"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("result")));
        const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("result"));
        EXPECT_EQ(result.getValues().getDataType(), Api::DataType::FP32);
        EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{expectedCapacity, 3, 2}));
        EXPECT_EQ(result.getOffsets().getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        std::vector<uint64_t> expectedOffsets(static_cast<size_t>(placementBatchSize) + 1, 0);
        for (uint32_t row = 0; row <= placementBatchSize; ++row) expectedOffsets[row] = static_cast<uint64_t>(row) * 2;
        EXPECT_EQ(result.getHostOffsetsIfAvailable(), std::optional<std::vector<uint64_t>>(expectedOffsets));
        ASSERT_EQ(result.getHostActiveValueCountIfAvailable(),
                  std::optional<uint64_t>(static_cast<uint64_t>(placementBatchSize) * 2));

        const float* resultValues = result.getValues().getMemPtr<float>();
        constexpr uint64_t outputElementsPerValue = 6;
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            for (uint64_t valueInRow = 0; valueInRow < 2; ++valueInRow) {
                const uint64_t packedValue = static_cast<uint64_t>(row) * 2 + valueInRow;
                const float base = static_cast<float>(row * 100 + valueInRow * 10);
                const std::vector<float> expected{base + 1.0F,
                                                  base + 5.0F,
                                                  base + 2.0F,
                                                  base + 6.0F,
                                                  base + 3.0F,
                                                  base + 7.0F};
                for (uint64_t element = 0; element < outputElementsPerValue; ++element) {
                    EXPECT_FLOAT_EQ(resultValues[packedValue * outputElementsPerValue + element], expected[element])
                        << "row=" << row << " packed=" << valueInRow << " element=" << element;
                }
            }
        }
    };

    loadPlaceAndRun(batchOne);
    loadPlaceAndRun(batchTwo);
    std::filesystem::remove_all(archive);
}

TEST(UtilityApiLayers, HistoricalSliceTransposeReadersDiscardPackedGeometryAndPlaceAtTwoBatches) {
    constexpr uint64_t m = 4;
    const std::string networkName = "historical_slice_transpose_reconstruction";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_historical_slice_transpose_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    Api::Network network(networkName);
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network).name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .offsetsDataType(Api::DataType::UINT32)
                                  .trailingDimensions({2, 4}).maxValuesPerRow(m).build();
    Api::Slice slice = Api::Slice::Builder().network(network).featureInput(input)
                           .axis(1).start(1).length(3).build();
    Api::Transpose transpose = Api::Transpose::Builder().network(network)
                                   .featureInput(slice.getRaggedFeatureOutput().value()).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("result")
        .inputTensor(transpose.getRaggedFeatureOutput().value()).build();

    nlohmann::json oldArtifact = network.architectureJson();
    uint32_t changed = 0;
    for (auto& layerJson : oldArtifact.at("layers")) {
        const std::string layerType = layerJson.at("layer_type").get<std::string>();
        if (layerType == "slice" || layerType == "transpose") {
            layerJson["version"] = "1.0.0";
            for (const char* key : {"ragged_feature_input", "ragged_feature_output"}) {
                auto& reference = layerJson.at(key);
                reference.erase("layer_ragged_reference_version");
                reference.erase("geometry_mode");
                reference.erase("max_values_per_row");
                reference["batch_size"] = 91; // Ignored obsolete metadata.
                reference["max_total_values"] = 999;
                reference["values"]["dimensions"][0] = 12; // Old physical [B*M].
                reference["offsets"]["dimensions"][0] = 4;
            }
            layerJson["feature_input"]["dimensions"][0] = 12;
            layerJson["feature_output"]["dimensions"][0] = 12;
            ++changed;
        } else if (layerType == "network_input") {
            const std::string name = layerJson.at("name").get<std::string>();
            if (name != "tokens.values" && name != "__thor_row_partition.tokens") continue;
            const uint64_t physicalLeading = name == "tokens.values" ? 12 : 4;
            layerJson["dimensions"][0] = physicalLeading;
            for (const char* key : {"feature_input", "feature_output"})
                layerJson[key]["dimensions"][0] = physicalLeading;
        }
    }
    ASSERT_EQ(changed, 2u);
    const std::string encoded = oldArtifact.dump(4);
    Impl::Tensor bytes(cpuPlacement, Impl::TensorDescriptor(Api::DataType::UINT8, {encoded.size()}));
    std::memcpy(bytes.getMemPtr<void>(), encoded.data(), encoded.size());
    thor_file::TarWriter writer(networkName);
    writer.addArchiveFile(networkName + ".thor.json", bytes);
    writer.createArchive(archive, true);

    for (const uint32_t b : {2u, 7u}) {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto loadedSlice = findOnlyLayerOfType<Api::Slice>(loaded);
        const auto loadedTranspose = findOnlyLayerOfType<Api::Transpose>(loaded);
        ASSERT_NE(loadedSlice, nullptr);
        ASSERT_NE(loadedTranspose, nullptr);
        ASSERT_TRUE(loadedSlice->getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(loadedTranspose->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedSlice->getRaggedFeatureOutput()->getValuesDimensions(),
                  (std::vector<uint64_t>{m, 2, 3}));
        EXPECT_EQ(loadedTranspose->getRaggedFeatureOutput()->getValuesDimensions(),
                  (std::vector<uint64_t>{m, 3, 2}));
        EXPECT_EQ(loadedSlice->getRaggedFeatureOutput()->getRowPartitionToken(),
                  loadedTranspose->getRaggedFeatureOutput()->getRowPartitionToken());
        EXPECT_EQ(loadedTranspose->getRaggedFeatureOutput()->getDescriptorForPlacement(b).getValuesDimensions(),
                  (std::vector<uint64_t>{b * m, 3, 2}));
        std::vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : events) event.synchronize();
    }
    std::filesystem::remove_all(archive);
}
