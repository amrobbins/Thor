#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include "gtest/gtest.h"

#include <cuda_fp16.h>

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
using std::vector;

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

Batch makeNeutralUnaryBatch(const std::string& inputName,
                            const Api::RaggedTensor& logicalInput,
                            uint32_t batchSize) {
    const Impl::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    const uint64_t capacity = descriptor.getMaxTotalValues();
    constexpr uint64_t activeValuesPerRow = 2;
    constexpr uint64_t elementsPerValue = 6;
    const uint64_t activeValues = static_cast<uint64_t>(batchSize) * activeValuesPerRow;
    THOR_THROW_IF_FALSE(logicalInput.getTrailingDimensions() == (std::vector<uint64_t>{2, 3}));
    THOR_THROW_IF_FALSE(logicalInput.getMaxValuesPerRow() >= activeValuesPerRow);
    THOR_THROW_IF_FALSE(activeValues <= capacity);

    Impl::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t i = 0; i < capacity * elementsPerValue; ++i) valuesPtr[i] = -1000.0F;
    for (uint32_t row = 0; row < batchSize; ++row) {
        for (uint64_t valueInRow = 0; valueInRow < activeValuesPerRow; ++valueInRow) {
            const uint64_t packedValue = static_cast<uint64_t>(row) * activeValuesPerRow + valueInRow;
            for (uint64_t element = 0; element < elementsPerValue; ++element) {
                valuesPtr[packedValue * elementsPerValue + element] =
                    static_cast<float>(row * 32 + valueInRow * 8 + element + 1);
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

TEST(UtilityApiLayers, PlacementNeutralUnaryFiveLayerChainSaveLoadReplacesPlacementGeometryPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint32_t batchOne = 2;
    constexpr uint32_t batchTwo = 7;
    const std::string networkName = "placement_neutral_unary_five_layer_chain";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_neutral_unary_five_layer_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedPartitionTokenOriginalId = 0;
    {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(Api::DataType::FP32)
                                      .offsetsDataType(Api::DataType::UINT32)
                                      .trailingDimensions({2, 3})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
        Api::TypeConverter converter = Api::TypeConverter::Builder()
                                           .network(network)
                                           .featureInput(input)
                                           .newDataType(Api::DataType::FP16)
                                           .build();
        ASSERT_TRUE(converter.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor converted = converter.getRaggedFeatureOutput().value();
        Api::Flatten flatten = Api::Flatten::Builder()
                                   .network(network)
                                   .featureInput(converted)
                                   .numOutputDimensions(1)
                                   .build();
        ASSERT_TRUE(flatten.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor flattened = flatten.getRaggedFeatureOutput().value();
        Api::Reshape reshape = Api::Reshape::Builder()
                                   .network(network)
                                   .featureInput(flattened)
                                   .newDimensions({3, 2})
                                   .build();
        ASSERT_TRUE(reshape.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor reshaped = reshape.getRaggedFeatureOutput().value();
        Api::StopGradient stop = Api::StopGradient::Builder().network(network).featureInput(reshaped).build();
        ASSERT_TRUE(stop.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor stopped = stop.getRaggedFeatureOutput().value();
        Api::ScaleGradient scale =
            Api::ScaleGradient::Builder().network(network).featureInput(stopped).scale(0.5F).build();
        ASSERT_TRUE(scale.getRaggedFeatureOutput().has_value());
        Api::RaggedTensor scaled = scale.getRaggedFeatureOutput().value();
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("result")
            .inputTensor(scaled)
            .build();

        savedPartitionTokenOriginalId = input.getRowPartitionToken().getOriginalId();
        const std::vector<Api::RaggedTensor> tensors{input, converted, flattened, reshaped, stopped, scaled};
        for (const Api::RaggedTensor& tensor : tensors) {
            EXPECT_EQ(tensor.getRowPartitionToken(), input.getRowPartitionToken());
            EXPECT_EQ(tensor.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
            ASSERT_TRUE(tensor.hasMaxValuesPerRow());
            EXPECT_EQ(tensor.getMaxValuesPerRow(), maxValuesPerRow);
        }
        EXPECT_EQ(input.getValuesDimensions(), (std::vector<uint64_t>{4, 2, 3}));
        EXPECT_EQ(converted.getValuesDimensions(), (std::vector<uint64_t>{4, 2, 3}));
        EXPECT_EQ(flattened.getValuesDimensions(), (std::vector<uint64_t>{4, 6}));
        EXPECT_EQ(reshaped.getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));
        EXPECT_EQ(stopped.getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));
        EXPECT_EQ(scaled.getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));
        EXPECT_FALSE(flatten.outputTensorDimensionsIncludeBatch(flatten.getFeatureOutput().value()));
        EXPECT_FALSE(reshape.outputTensorDimensionsIncludeBatch(reshape.getFeatureOutput().value()));
        EXPECT_FALSE(stop.outputTensorDimensionsIncludeBatch(stop.getFeatureOutput().value()));
        EXPECT_FALSE(scale.outputTensorDimensionsIncludeBatch(scale.getFeatureOutput().value()));

        const nlohmann::json architecture = network.architectureJson();
        for (const std::string& layerType : {"type_converter", "flatten", "reshape", "stop_gradient", "scale_gradient"}) {
            const nlohmann::json& layerJson = findLayerJson(architecture, layerType);
            EXPECT_EQ(layerJson.at("version").get<std::string>(), "1.1.0") << layerType;
            EXPECT_TRUE(layerJson.at("use_ragged").get<bool>()) << layerType;
            expectNeutralReference(layerJson.at("ragged_feature_input"), maxValuesPerRow);
            expectNeutralReference(layerJson.at("ragged_feature_output"), maxValuesPerRow);
            EXPECT_EQ(layerJson.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
                      layerJson.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>()) << layerType;
        }
        EXPECT_FLOAT_EQ(findLayerJson(architecture, "scale_gradient").at("scale").get<float>(), 0.5F);
        EXPECT_EQ(findLayerJson(architecture, "type_converter").at("ragged_feature_output").at("values").at("data_type").get<std::string>(),
                  "fp16");
        EXPECT_EQ(findLayerJson(architecture, "flatten").at("ragged_feature_output").at("values").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{4, 6}));
        EXPECT_EQ(findLayerJson(architecture, "reshape").at("ragged_feature_output").at("values").at("dimensions").get<std::vector<uint64_t>>(),
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
        EXPECT_EQ(loadedBoundaryOutput.getValuesDataType(), Api::DataType::FP16);

        const auto loadedConverter = findOnlyLayerOfType<Api::TypeConverter>(loaded);
        const auto loadedFlatten = findOnlyLayerOfType<Api::Flatten>(loaded);
        const auto loadedReshape = findOnlyLayerOfType<Api::Reshape>(loaded);
        const auto loadedStop = findOnlyLayerOfType<Api::StopGradient>(loaded);
        const auto loadedScale = findOnlyLayerOfType<Api::ScaleGradient>(loaded);
        ASSERT_NE(loadedConverter, nullptr);
        ASSERT_NE(loadedFlatten, nullptr);
        ASSERT_NE(loadedReshape, nullptr);
        ASSERT_NE(loadedStop, nullptr);
        ASSERT_NE(loadedScale, nullptr);
        ASSERT_TRUE(loadedConverter->getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(loadedFlatten->getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(loadedReshape->getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(loadedStop->getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(loadedScale->getRaggedFeatureOutput().has_value());
        for (const Api::RaggedTensor& tensor : {loadedConverter->getRaggedFeatureOutput().value(),
                                               loadedFlatten->getRaggedFeatureOutput().value(),
                                               loadedReshape->getRaggedFeatureOutput().value(),
                                               loadedStop->getRaggedFeatureOutput().value(),
                                               loadedScale->getRaggedFeatureOutput().value()}) {
            expectPreservedPartition(tensor, savedPartitionTokenOriginalId);
        }
        EXPECT_EQ(loadedFlatten->getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{4, 6}));
        EXPECT_EQ(loadedReshape->getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{4, 3, 2}));
        EXPECT_FLOAT_EQ(loadedScale->getScale(), 0.5F);

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        std::shared_ptr<Impl::NetworkInput> valuesInput = placed->getStampedNetwork(0).getNamedInput("tokens.values");
        ASSERT_NE(valuesInput, nullptr);
        ASSERT_TRUE(valuesInput->getFeatureOutput().has_value());
        EXPECT_EQ(valuesInput->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{expectedCapacity, 2, 3}));

        // This chain consumes only DEVICE_ACTIVE_COUNT (via TypeConverter), not DEVICE_OFFSETS.
        // Do not manufacture a hidden [B+1] physical offsets input merely because the logical
        // partition later appears on the RaggedNetworkOutput. Logical inference still carries
        // and validates the authoritative offsets below.
        EXPECT_EQ(placed->getStampedNetwork(0).getManagedPartitionOffsetsInputForTest(
                      loadedInput.getRowPartitionId()),
                  nullptr);
        std::shared_ptr<Impl::NetworkInput> activeCountInput =
            placed->getStampedNetwork(0).getManagedPartitionActiveCountInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(activeCountInput, nullptr);
        ASSERT_TRUE(activeCountInput->getFeatureOutput().has_value());
        EXPECT_EQ(activeCountInput->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{1}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const std::map<std::string, Api::InferenceOutputValue> outputs =
            placed->inferLogical(makeNeutralUnaryBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("result"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("result")));
        const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("result"));
        EXPECT_EQ(result.getValues().getDataType(), Api::DataType::FP16);
        EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{expectedCapacity, 3, 2}));
        EXPECT_EQ(result.getOffsets().getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        std::vector<uint64_t> expectedOffsets(static_cast<size_t>(placementBatchSize) + 1, 0);
        for (uint32_t row = 0; row <= placementBatchSize; ++row) expectedOffsets[row] = static_cast<uint64_t>(row) * 2;
        EXPECT_EQ(result.getHostOffsetsIfAvailable(), std::optional<std::vector<uint64_t>>(expectedOffsets));
        ASSERT_EQ(result.getHostActiveValueCountIfAvailable(),
                  std::optional<uint64_t>(static_cast<uint64_t>(placementBatchSize) * 2));

        const __half* resultValues = result.getValues().getMemPtr<__half>();
        constexpr uint64_t elementsPerValue = 6;
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            for (uint64_t valueInRow = 0; valueInRow < 2; ++valueInRow) {
                const uint64_t packedValue = static_cast<uint64_t>(row) * 2 + valueInRow;
                for (uint64_t element = 0; element < elementsPerValue; ++element) {
                    const float expected = static_cast<float>(row * 32 + valueInRow * 8 + element + 1);
                    EXPECT_FLOAT_EQ(__half2float(resultValues[packedValue * elementsPerValue + element]), expected)
                        << "row=" << row << " packed=" << valueInRow << " element=" << element;
                }
            }
        }
    };

    loadPlaceAndRun(batchOne);
    loadPlaceAndRun(batchTwo);
    std::filesystem::remove_all(archive);
}

TEST(UtilityApiLayers, TypeConverterActiveCountManagedCarrierSuppliesPlacementDescriptorForStamp) {
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint32_t placementBatchSize = 7;

    Api::Network network("type_converter_active_count_managed_carrier_metadata");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .offsetsDataType(Api::DataType::UINT32)
                                  .trailingDimensions({2, 3})
                                  .maxValuesPerRow(maxValuesPerRow)
                                  .build();
    Api::TypeConverter converter = Api::TypeConverter::Builder()
                                       .network(network)
                                       .featureInput(input)
                                       .newDataType(Api::DataType::FP16)
                                       .build();
    Api::RaggedNetworkOutput::Builder()
        .network(network)
        .name("result")
        .inputTensor(converter.getRaggedFeatureOutput().value())
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
        ASSERT_EQ(managedCarrier, nullptr) << "Expected exactly one managed row-partition fanout for active-count-only TypeConverter.";
        managedCarrier = fanout;
    }
    ASSERT_NE(managedCarrier, nullptr);
    ASSERT_TRUE(managedCarrier->getFeatureOutput().has_value());
    EXPECT_EQ(managedCarrier->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{1}));

    const Impl::RaggedTensorDescriptor expected = input.getDescriptorForPlacement(placementBatchSize);
    const std::optional<Impl::RowPartitionDescriptor> attached = managedCarrier->getRowPartitionDescriptorForStamp();
    ASSERT_TRUE(attached.has_value());
    EXPECT_EQ(attached.value(), expected.getRowPartition());
    EXPECT_EQ(attached->getBatchSize(), placementBatchSize);
    ASSERT_TRUE(attached->hasMaxValuesPerRow());
    EXPECT_EQ(attached->getMaxValuesPerRow(), maxValuesPerRow);
    EXPECT_EQ(attached->getMaxTotalValues(), static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow);
    EXPECT_EQ(attached->getOffsetsDataType(), Api::DataType::UINT32);

    const Impl::RaggedTensorDescriptor fromActiveCountDriver =
        Api::SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            input, managedCarrier, "TypeConverter");
    EXPECT_EQ(fromActiveCountDriver, expected);
}

TEST(UtilityApiLayers, HistoricalUnaryReferencesAndPackedOutputsReconstructLogicalAtTwoBatches) {
    // This replaces the old fixed-writer test: current builders must never
    // produce fixed model geometry. Only historical *readers* accept that JSON.
    constexpr uint64_t m = 4;
    const std::string networkName = "historical_unary_grammar_reconstruction";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
        ("thor_historical_unary_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    Api::Network network(networkName);
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
        .network(network).name("tokens").valuesDataType(Api::DataType::FP32)
        .offsetsDataType(Api::DataType::UINT32).trailingDimensions({2, 3})
        .maxValuesPerRow(m).build();
    Api::Flatten flatten = Api::Flatten::Builder().network(network)
        .featureInput(input).numOutputDimensions(1).build();
    Api::Reshape reshape = Api::Reshape::Builder().network(network)
        .featureInput(flatten.getRaggedFeatureOutput().value()).newDimensions({3, 2}).build();
    Api::StopGradient stop = Api::StopGradient::Builder().network(network)
        .featureInput(reshape.getRaggedFeatureOutput().value()).build();
    Api::ScaleGradient scale = Api::ScaleGradient::Builder().network(network)
        .featureInput(stop.getRaggedFeatureOutput().value()).scale(0.5F).build();
    Api::DropOut dropout = Api::DropOut::Builder().network(network)
        .featureInput(scale.getRaggedFeatureOutput().value()).dropProportion(0.0F).build();
    Api::TypeConverter converter = Api::TypeConverter::Builder().network(network)
        .featureInput(dropout.getRaggedFeatureOutput().value())
        .newDataType(Api::DataType::FP16).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("result")
        .inputTensor(converter.getRaggedFeatureOutput().value()).build();

    nlohmann::json oldArtifact = network.architectureJson();
    uint32_t migratedCount = 0;
    for (auto& layerJson : oldArtifact.at("layers")) {
        const std::string layerType = layerJson.at("layer_type").get<std::string>();
        if (layerType != "flatten" && layerType != "reshape" &&
            layerType != "stop_gradient" && layerType != "scale_gradient" &&
            layerType != "drop_out" && layerType != "type_converter") continue;
        // Old layers serialized physical packed output leading extents and
        // carried a separate old-format reference. M comes from the boundary.
        layerJson["version"] = "1.0.0";
        for (const char* field : {"ragged_feature_input", "ragged_feature_output"}) {
            auto& reference = layerJson.at(field);
            reference.erase("layer_ragged_reference_version");
            reference.erase("geometry_mode");
            reference.erase("max_values_per_row");
            reference["batch_size"] = 99;  // Deliberately false obsolete metadata.
            reference["max_total_values"] = 777;
            reference["values"]["dimensions"][0] = 12;
            reference["offsets"]["dimensions"][0] = 4;
        }
        layerJson["feature_input"]["dimensions"][0] = 12;
        layerJson["feature_output"]["dimensions"][0] = 12;
        ++migratedCount;
    }
    ASSERT_EQ(migratedCount, 6u);
    // U1's source importer also has to discard archived physical [B*M] and
    // [B+1] without altering ordinary dense source handling.
    for (auto& layerJson : oldArtifact.at("layers")) {
        if (layerJson.at("layer_type") != "network_input") continue;
        const std::string name = layerJson.at("name").get<std::string>();
        if (name != "tokens.values" && name != "__thor_row_partition.tokens") continue;
        const uint64_t archivedLeading = name == "tokens.values" ? 12 : 4;
        layerJson["dimensions"][0] = archivedLeading;
        for (const char* field : {"feature_input", "feature_output"})
            layerJson[field]["dimensions"][0] = archivedLeading;
    }
    // An actual Thor archive, not a direct helper call or a rewritten live graph.
    const std::string encoded = oldArtifact.dump(4);
    Impl::Tensor bytes(cpuPlacement, Impl::TensorDescriptor(Api::DataType::UINT8, {encoded.size()}));
    std::memcpy(bytes.getMemPtr<void>(), encoded.data(), encoded.size());
    thor_file::TarWriter writer(networkName);
    writer.addArchiveFile(networkName + ".thor.json", bytes);
    writer.createArchive(archive, true);

    for (const uint32_t batchSize : {2u, 7u}) {
        Api::Network restored(networkName);
        ASSERT_NO_THROW(restored.load(archive.string()));
        auto boundary = restored.getExternalRaggedNetworkInputs();
        ASSERT_EQ(boundary.size(), 1u);
        EXPECT_EQ(boundary[0].raggedTensor.getValuesDimensions(), (std::vector<uint64_t>{m, 2, 3}));
        EXPECT_EQ(boundary[0].raggedTensor.getOffsetsDimensions(), (std::vector<uint64_t>{1}));
        const auto restoredScale = findOnlyLayerOfType<Api::ScaleGradient>(restored);
        ASSERT_NE(restoredScale, nullptr);
        ASSERT_TRUE(restoredScale->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(restoredScale->getRaggedFeatureOutput()->getValuesDimensions(),
                  (std::vector<uint64_t>{m, 3, 2}));
        const auto restoredDropout = findOnlyLayerOfType<Api::DropOut>(restored);
        ASSERT_NE(restoredDropout, nullptr);
        ASSERT_TRUE(restoredDropout->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(restoredDropout->getRaggedFeatureOutput()->getValuesDimensions(),
                  (std::vector<uint64_t>{m, 3, 2}));
        EXPECT_EQ(restoredDropout->getRaggedFeatureOutput()->getDescriptorForPlacement(batchSize).getValuesDimensions(),
                  (std::vector<uint64_t>{batchSize * m, 3, 2}));
        EXPECT_EQ(restoredDropout->getRaggedFeatureOutput()->getRowPartitionToken(),
                  boundary[0].raggedTensor.getRowPartitionToken());
        const auto restoredConverter = findOnlyLayerOfType<Api::TypeConverter>(restored);
        ASSERT_NE(restoredConverter, nullptr);
        ASSERT_TRUE(restoredConverter->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(restoredConverter->getRaggedFeatureOutput()->getValuesDimensions(),
                  (std::vector<uint64_t>{m, 3, 2}));
        EXPECT_EQ(restoredConverter->getRaggedFeatureOutput()->getValuesDataType(), Api::DataType::FP16);
        EXPECT_EQ(restoredConverter->getRaggedFeatureOutput()->getDescriptorForPlacement(batchSize).getValuesDimensions(),
                  (std::vector<uint64_t>{batchSize * m, 3, 2}));
        EXPECT_EQ(restoredConverter->getRaggedFeatureOutput()->getRowPartitionToken(),
                  boundary[0].raggedTensor.getRowPartitionToken());
        // The same historical artifact is independently placeable at either B.
        std::vector<Event> events;
        auto placed = restored.place(batchSize, events, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : events) event.synchronize();
    }
    std::filesystem::remove_all(archive);
}
