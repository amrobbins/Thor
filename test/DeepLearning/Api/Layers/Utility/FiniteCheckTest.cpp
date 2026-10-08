#include "DeepLearning/Api/Layers/Utility/FiniteCheck.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <variant>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, FiniteCheckBuildsAsLogicalIdentity) {
    Network network("finiteCheckBuilds");
    Tensor featureInput(DataType::BF16, {3, 5});

    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(network)
                                  .featureInput(featureInput)
                                  .tensorLabel("after_projection")
                                  .enabled(false)
                                  .checkForward(true)
                                  .checkBackward(false)
                                  .failOnNonFinite(false)
                                  .maxReportedIndices(4)
                                  .build();

    ASSERT_TRUE(finiteCheck.isInitialized());
    ASSERT_TRUE(finiteCheck.getFeatureInput().has_value());
    ASSERT_TRUE(finiteCheck.getFeatureOutput().has_value());
    EXPECT_EQ(finiteCheck.getFeatureInput().value(), featureInput);
    EXPECT_NE(finiteCheck.getFeatureOutput().value(), featureInput);
    EXPECT_EQ(finiteCheck.getFeatureOutput().value().getDataType(), DataType::BF16);
    EXPECT_EQ(finiteCheck.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{3, 5}));
    EXPECT_EQ(finiteCheck.getTensorLabel(), "after_projection");
    EXPECT_FALSE(finiteCheck.getEnabled());
    EXPECT_TRUE(finiteCheck.getCheckForward());
    EXPECT_FALSE(finiteCheck.getCheckBackward());
    EXPECT_FALSE(finiteCheck.getFailOnNonFinite());
    EXPECT_EQ(finiteCheck.getMaxReportedIndices(), 4U);
}

TEST(UtilityApiLayers, FiniteCheckRejectsInvalidConfiguration) {
    Network network("finiteCheckInvalid");
    Tensor featureInput(DataType::FP32, {4});

    EXPECT_THROW(FiniteCheck::Builder()
                     .network(network)
                     .featureInput(featureInput)
                     .checkForward(false)
                     .checkBackward(false)
                     .build(),
                 invalid_argument);

    EXPECT_THROW(FiniteCheck::Builder()
                     .network(network)
                     .featureInput(featureInput)
                     .maxReportedIndices(ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES + 1)
                     .build(),
                 invalid_argument);
}

TEST(UtilityApiLayers, FiniteCheckArchitecturePersistsDiagnosticPolicy) {
    Network network("finiteCheckArchitecture");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({4})
                             .dataType(DataType::FP32)
                             .build();
    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .tensorLabel("encoder_output")
                                  .enabled(false)
                                  .checkForward(true)
                                  .checkBackward(true)
                                  .failOnNonFinite(true)
                                  .maxReportedIndices(7)
                                  .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(finiteCheck.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();

    const json architecture = finiteCheck.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "finite_check");
    EXPECT_EQ(architecture.at("tensor_label").get<string>(), "encoder_output");
    EXPECT_FALSE(architecture.at("enabled").get<bool>());
    EXPECT_TRUE(architecture.at("check_forward").get<bool>());
    EXPECT_TRUE(architecture.at("check_backward").get<bool>());
    EXPECT_TRUE(architecture.at("fail_on_non_finite").get<bool>());
    EXPECT_EQ(architecture.at("max_reported_indices").get<uint32_t>(), 7U);
}

TEST(UtilityApiLayers, RaggedFiniteCheckPreservesExactPartitionAndSerializes) {
    Network network("raggedFiniteCheckBuilds");
    Tensor values(DataType::FP32, {8, 2});
    Tensor offsets(DataType::UINT64, {4});
    RaggedTensor input(values, offsets, 5);

    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .tensorLabel("ragged_history")
                                  .enabled(false)
                                  .build();

    ASSERT_TRUE(finiteCheck.getUseRagged());
    ASSERT_TRUE(finiteCheck.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(finiteCheck.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = finiteCheck.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getBatchSize(), input.getBatchSize());
    EXPECT_EQ(output.getMaxTotalValues(), input.getMaxTotalValues());
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 5u);
    EXPECT_EQ(output.getValues().getDimensions(), values.getDimensions());
    EXPECT_NE(output.getValues(), values);

    const json architecture = finiteCheck.architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(), offsets.getId());
    EXPECT_EQ(architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(), offsets.getId());

    auto cloned = dynamic_pointer_cast<FiniteCheck>(finiteCheck.clone());
    ASSERT_NE(cloned, nullptr);
    ASSERT_TRUE(cloned->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(cloned->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(cloned->getRaggedFeatureInput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(cloned->getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
}

TEST(UtilityApiLayers, PlacementNeutralFiniteCheckSerializesSemanticGeometryAndRejectsMismatches) {
    constexpr uint64_t maxValuesPerRow = 4;
    Network neutralNetwork("placement_neutral_finite_check_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(neutralNetwork)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(maxValuesPerRow)
                             .build();
    ASSERT_TRUE(input.isPlacementNeutral());

    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(neutralNetwork)
                                  .featureInput(input)
                                  .tensorLabel("neutral_tokens")
                                  .enabled(true)
                                  .checkForward(true)
                                  .checkBackward(false)
                                  .failOnNonFinite(true)
                                  .maxReportedIndices(3)
                                  .build();
    ASSERT_TRUE(finiteCheck.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(finiteCheck.getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
    EXPECT_EQ(finiteCheck.getRaggedFeatureOutput()->getValues().getDimensions(),
              (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_FALSE(finiteCheck.outputTensorDimensionsIncludeBatch(finiteCheck.getFeatureOutput().value()));
    EXPECT_EQ(finiteCheck.getOutputTensorBytes(2), 2U * maxValuesPerRow * 2U * sizeof(float));
    EXPECT_EQ(finiteCheck.getOutputTensorBytes(7), 7U * maxValuesPerRow * 2U * sizeof(float));
    EXPECT_EQ(finiteCheck.getRaggedPartitionRequirementForInput(input.getRowPartitionToken()),
              ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);

    const json architecture = finiteCheck.architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.2.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("tensor_label").get<string>(), "neutral_tokens");
    EXPECT_TRUE(architecture.at("enabled").get<bool>());
    EXPECT_TRUE(architecture.at("check_forward").get<bool>());
    EXPECT_FALSE(architecture.at("check_backward").get<bool>());
    EXPECT_TRUE(architecture.at("fail_on_non_finite").get<bool>());
    EXPECT_EQ(architecture.at("max_reported_indices").get<uint32_t>(), 3U);

    const json& inputJson = architecture.at("ragged_feature_input");
    const json& outputJson = architecture.at("ragged_feature_output");
    EXPECT_EQ(inputJson.at("layer_ragged_reference_version").get<string>(), "1.0.0");
    EXPECT_EQ(inputJson.at("geometry_mode").get<string>(), "placement_neutral");
    EXPECT_EQ(inputJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
    EXPECT_FALSE(inputJson.contains("batch_size"));
    EXPECT_FALSE(inputJson.contains("max_total_values"));
    EXPECT_FALSE(outputJson.contains("batch_size"));
    EXPECT_FALSE(outputJson.contains("max_total_values"));
    EXPECT_EQ(inputJson.at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_EQ(inputJson.at("offsets").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
    EXPECT_EQ(outputJson.at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), inputJson.at("offsets").at("id").get<uint64_t>());
    EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(),
              architecture.at("feature_output").at("id").get<uint64_t>());

    json neutralAsFixed = architecture;
    neutralAsFixed["version"] = "1.1.0";
    EXPECT_THROW(FiniteCheck::deserialize(neutralAsFixed, &neutralNetwork), runtime_error);

    json neutralAsDense = architecture;
    neutralAsDense["use_ragged"] = false;
    EXPECT_THROW(FiniteCheck::deserialize(neutralAsDense, &neutralNetwork), runtime_error);

    json wrongPartition = architecture;
    wrongPartition["ragged_feature_output"]["offsets"]["id"] = input.getValues().getId();
    EXPECT_THROW(FiniteCheck::deserialize(wrongPartition, &neutralNetwork), runtime_error);

    json wrongOutputValues = architecture;
    wrongOutputValues["ragged_feature_output"]["values"]["id"] = input.getValues().getId();
    EXPECT_THROW(FiniteCheck::deserialize(wrongOutputValues, &neutralNetwork), runtime_error);

    Network fixedNetwork("fixed_finite_check_geometry_mismatch");
    RaggedTensor fixedInput = RaggedNetworkInput::Builder()
                                  .network(fixedNetwork)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2})
                                  .batchSize(3)
                                  .maxTotalValues(9)
                                  .maxValuesPerRow(4)
                                  .build();
    FiniteCheck fixedCheck = FiniteCheck::Builder()
                                 .network(fixedNetwork)
                                 .featureInput(fixedInput)
                                 .enabled(false)
                                 .build();
    json fixedAsNeutral = fixedCheck.architectureJson();
    ASSERT_EQ(fixedAsNeutral.at("version").get<string>(), "1.1.0");
    fixedAsNeutral["version"] = "1.2.0";
    EXPECT_THROW(FiniteCheck::deserialize(fixedAsNeutral, &fixedNetwork), runtime_error);
}

TEST(UtilityApiLayers, PlacementNeutralFiniteCheckSaveLoadReplacesRuntimeConfigurationPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint64_t elementsPerValue = 2;
    const string networkName = "placement_neutral_finite_check_round_trip";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_finite_check_neutral_" + to_string(nonce));
    filesystem::remove_all(archive);

    uint64_t savedValuesOriginalId = 0;
    uint64_t savedPartitionOriginalId = 0;
    uint64_t savedOutputOriginalId = 0;
    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({elementsPerValue})
                                 .maxValuesPerRow(maxValuesPerRow)
                                 .build();
        FiniteCheck finiteCheck = FiniteCheck::Builder()
                                      .network(network)
                                      .featureInput(input)
                                      .tensorLabel("tokens")
                                      .enabled(true)
                                      .checkForward(true)
                                      .checkBackward(false)
                                      .failOnNonFinite(true)
                                      .maxReportedIndices(4)
                                      .build();
        ASSERT_TRUE(finiteCheck.getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("checked")
            .inputTensor(finiteCheck.getRaggedFeatureOutput().value())
            .build();

        savedValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionOriginalId = input.getRowPartitionToken().getOriginalId();
        savedOutputOriginalId = finiteCheck.getFeatureOutput()->getOriginalId();
        const json layerJson = finiteCheck.architectureJson();
        EXPECT_EQ(layerJson.at("version").get<string>(), "1.2.0");
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("batch_size"));
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("max_total_values"));
        network.save(archive.string(), true);
    }

    auto makeBatch = [&](const RaggedTensor& logicalInput, uint32_t batchSize, bool activeNan) {
        const ThorImplementation::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
        const uint64_t capacity = descriptor.getMaxTotalValues();
        ThorImplementation::Tensor values(
            ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU),
            descriptor.getValuesDescriptor());
        float* valuesPtr = values.getMemPtr<float>();

        vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
        uint64_t activeValues = 0;
        for (uint32_t row = 0; row < batchSize; ++row) {
            const uint64_t rowLength = row == 0 ? 0 : 1 + ((row - 1) % 2);
            activeValues += rowLength;
            hostOffsets[row + 1] = activeValues;
        }
        if (activeValues > capacity) throw runtime_error("FiniteCheck test active values exceed placement capacity.");

        for (uint64_t value = 0; value < capacity; ++value) {
            for (uint64_t element = 0; element < elementsPerValue; ++element) {
                valuesPtr[value * elementsPerValue + element] =
                    value < activeValues ? static_cast<float>(value * 10 + element + 1)
                                         : numeric_limits<float>::quiet_NaN();
            }
        }
        if (activeNan && activeValues != 0) valuesPtr[0] = numeric_limits<float>::quiet_NaN();

        ThorImplementation::Tensor offsets(
            ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU),
            descriptor.getOffsetsDescriptor());
        if (logicalInput.getOffsetsDataType() != DataType::UINT32)
            throw runtime_error("FiniteCheck neutral runtime test expects UINT32 offsets.");
        uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);

        ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
        partition.setHostOffsets(hostOffsets);
        Batch batch;
        batch.insert("tokens", ThorImplementation::RaggedTensor(values, partition));
        return batch;
    };

    auto findFiniteCheck = [](Network& network) {
        shared_ptr<FiniteCheck> result;
        for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
            auto candidate = dynamic_pointer_cast<FiniteCheck>(network.getLayer(i));
            if (candidate != nullptr) {
                EXPECT_EQ(result, nullptr);
                result = candidate;
            }
        }
        return result;
    };

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto externalInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(externalInputs.size(), 1U);
        const RaggedTensor loadedInput = externalInputs.front().raggedTensor;
        ASSERT_TRUE(loadedInput.isPlacementNeutral());
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionOriginalId);

        shared_ptr<FiniteCheck> loadedFiniteCheck = findFiniteCheck(loaded);
        ASSERT_NE(loadedFiniteCheck, nullptr);
        ASSERT_TRUE(loadedFiniteCheck->getRaggedFeatureInput().has_value());
        ASSERT_TRUE(loadedFiniteCheck->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedFiniteCheck->getFeatureInput().value(), loadedInput.getValues());
        EXPECT_EQ(loadedFiniteCheck->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedFiniteCheck->getRaggedFeatureOutput()->getRowPartitionToken(),
                  loadedInput.getRowPartitionToken());
        EXPECT_EQ(loadedFiniteCheck->getRaggedFeatureOutput()->getValues().getDimensions(),
                  (vector<uint64_t>{maxValuesPerRow, elementsPerValue}));
        EXPECT_FALSE(loadedFiniteCheck->outputTensorDimensionsIncludeBatch(loadedFiniteCheck->getFeatureOutput().value()));

        const json architectureBeforePlacement = loaded.architectureJson();
        vector<Event> initDoneEvents;
        shared_ptr<PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        auto& stamped = placed->getStampedNetwork(0);
        const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(loadedInput.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_TRUE(ThorImplementation::hasRaggedPartitionRequirement(
            requirements.value(), ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT));
        shared_ptr<ThorImplementation::NetworkInput> activeCount =
            stamped.getManagedPartitionActiveCountInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(activeCount, nullptr);
        ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
        EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId()), nullptr);

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        shared_ptr<ThorImplementation::NetworkInput> physicalValues = stamped.getNamedInput("tokens.values");
        ASSERT_NE(physicalValues, nullptr);
        ASSERT_TRUE(physicalValues->getFeatureOutput().has_value());
        EXPECT_EQ(physicalValues->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, elementsPerValue}));

        auto physicalFiniteCheck = dynamic_pointer_cast<ThorImplementation::FiniteCheck>(
            stamped.getPhysicalLayerFromApiLayer(loadedFiniteCheck->getId()));
        ASSERT_NE(physicalFiniteCheck, nullptr);
        ASSERT_TRUE(physicalFiniteCheck->getFeatureOutput().has_value());
        EXPECT_EQ(physicalFiniteCheck->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, elementsPerValue}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        map<string, InferenceOutputValue> outputs;
        ASSERT_NO_THROW(outputs = placed->inferLogical(makeBatch(loadedInput, placementBatchSize, false)));
        ASSERT_TRUE(outputs.contains("checked"));
        ASSERT_TRUE(holds_alternative<ThorImplementation::RaggedTensor>(outputs.at("checked")));
        const ThorImplementation::RaggedTensor checked =
            get<ThorImplementation::RaggedTensor>(outputs.at("checked"));
        EXPECT_EQ(checked.getValues().getDimensions(), (vector<uint64_t>{expectedCapacity, elementsPerValue}));
        EXPECT_EQ(checked.getOffsets().getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));
        const auto hostOffsets = checked.getHostOffsetsIfAvailable();
        ASSERT_TRUE(hostOffsets.has_value());
        ASSERT_EQ(hostOffsets->size(), static_cast<size_t>(placementBatchSize) + 1);
        EXPECT_EQ(hostOffsets->at(0), 0U);
        EXPECT_EQ(hostOffsets->at(1), 0U) << "first row must remain empty";
    };

    try {
        loadPlaceAndRun(2);
        loadPlaceAndRun(7);

        // Active non-finite data must still be diagnosed; only undefined inactive capacity is ignored.
        Network failing(networkName);
        ASSERT_NO_THROW(failing.load(archive.string()));
        const RaggedTensor failingInput = failing.getExternalRaggedNetworkInputs().front().raggedTensor;
        vector<Event> initDoneEvents;
        shared_ptr<PlacedNetwork> failingPlaced = failing.place(2, initDoneEvents, true);
        ASSERT_NE(failingPlaced, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();
        EXPECT_THROW((void)failingPlaced->inferLogical(makeBatch(failingInput, 2, true)), runtime_error);
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

TEST(UtilityApiLayers, FiniteCheckDenseAndFixedRaggedVersionOneOneRemainLoadable) {
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();

    const string denseName = "finite_check_dense_compatibility";
    const filesystem::path denseArchive = filesystem::temp_directory_path() /
                                          ("thor_finite_check_dense_" + to_string(nonce));
    filesystem::remove_all(denseArchive);
    {
        Network network(denseName);
        NetworkInput input = NetworkInput::Builder()
                                 .network(network)
                                 .name("input")
                                 .dimensions({3})
                                 .dataType(DataType::FP32)
                                 .build();
        FiniteCheck finiteCheck = FiniteCheck::Builder()
                                      .network(network)
                                      .featureInput(input.getFeatureOutput().value())
                                      .enabled(false)
                                      .build();
        (void)NetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(finiteCheck.getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();
        const json serialized = finiteCheck.architectureJson();
        EXPECT_EQ(serialized.at("version").get<string>(), "1.1.0");
        EXPECT_FALSE(serialized.at("use_ragged").get<bool>());
        network.save(denseArchive.string(), true);
    }
    {
        Network loaded(denseName);
        ASSERT_NO_THROW(loaded.load(denseArchive.string()));
    }
    filesystem::remove_all(denseArchive);

    const string raggedName = "finite_check_fixed_ragged_compatibility";
    const filesystem::path raggedArchive = filesystem::temp_directory_path() /
                                           ("thor_finite_check_fixed_ragged_" + to_string(nonce));
    filesystem::remove_all(raggedArchive);
    {
        Network network(raggedName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("history")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .batchSize(3)
                                 .maxTotalValues(9)
                                 .maxValuesPerRow(4)
                                 .build();
        FiniteCheck finiteCheck = FiniteCheck::Builder()
                                      .network(network)
                                      .featureInput(input)
                                      .enabled(false)
                                      .build();
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(finiteCheck.getRaggedFeatureOutput().value())
            .build();
        const json serialized = finiteCheck.architectureJson();
        EXPECT_EQ(serialized.at("version").get<string>(), "1.1.0");
        EXPECT_TRUE(serialized.at("use_ragged").get<bool>());
        EXPECT_EQ(serialized.at("ragged_feature_input").at("batch_size").get<uint64_t>(), 3U);
        EXPECT_EQ(serialized.at("ragged_feature_input").at("max_total_values").get<uint64_t>(), 9U);
        EXPECT_FALSE(serialized.at("ragged_feature_input").contains("layer_ragged_reference_version"));
        network.save(raggedArchive.string(), true);
    }
    {
        Network loaded(raggedName);
        ASSERT_NO_THROW(loaded.load(raggedArchive.string()));
        shared_ptr<FiniteCheck> loadedFiniteCheck;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            loadedFiniteCheck = dynamic_pointer_cast<FiniteCheck>(loaded.getLayer(i));
            if (loadedFiniteCheck != nullptr) break;
        }
        ASSERT_NE(loadedFiniteCheck, nullptr);
        ASSERT_TRUE(loadedFiniteCheck->getRaggedFeatureInput().has_value());
        EXPECT_FALSE(loadedFiniteCheck->getRaggedFeatureInput()->isPlacementNeutral());
        EXPECT_EQ(loadedFiniteCheck->getRaggedFeatureInput()->getBatchSize(), 3U);
        EXPECT_EQ(loadedFiniteCheck->getRaggedFeatureInput()->getMaxTotalValues(), 9U);
        EXPECT_EQ(loadedFiniteCheck->architectureJson().at("version").get<string>(), "1.1.0");
    }
    filesystem::remove_all(raggedArchive);
}
