#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedLogSoftmax.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedSoftmax.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "test/DeepLearning/RaggedTestUtils.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using std::vector;

namespace {

const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

Impl::Tensor makeDenseRows() {
    Impl::Tensor rows(cpuPlacement, Impl::TensorDescriptor(Api::DataType::FP32, {3, 3}));
    float* p = rows.getMemPtr<float>();
    const float values[] = {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F, 70.0F, 80.0F, 90.0F};
    for (uint32_t i = 0; i < 9; ++i) p[i] = values[i];
    return rows;
}

Impl::Tensor makePlacementNeutralBroadcastDenseRows(uint32_t batchSize) {
    Impl::Tensor rows(cpuPlacement, Impl::TensorDescriptor(Api::DataType::FP32, {batchSize, 3}));
    float* p = rows.getMemPtr<float>();
    for (uint32_t row = 0; row < batchSize; ++row) {
        p[row * 3] = static_cast<float>(row * 10 + 1);
        p[row * 3 + 1] = static_cast<float>(row * 10 + 2);
        p[row * 3 + 2] = static_cast<float>(row * 10 + 3);
    }
    return rows;
}

Impl::RaggedTensor makeRaggedInput(Api::DataType offsetsDtype, const vector<uint64_t>& offsets) {
    constexpr uint64_t maxValuesPerRow = 3;
    const uint64_t capacity = (offsets.size() - 1) * maxValuesPerRow;
    Impl::Tensor values(cpuPlacement, Impl::TensorDescriptor(Api::DataType::FP32, {capacity, 2}));
    float* p = values.getMemPtr<float>();
    const float activeValues[] = {0.0F, 1.0F, 2.0F, 3.0F, 1.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2.0F};
    const uint64_t activeValueCount = offsets.back();
    for (uint64_t i = 0; i < activeValueCount * 2; ++i) p[i] = activeValues[i];
    ThorTest::poisonInactiveElements(p,
                                     activeValueCount * 2,
                                     capacity * 2,
                                     ThorTest::RaggedInactivePoison::NaN);

    Impl::Tensor offsetTensor(cpuPlacement, Impl::TensorDescriptor(offsetsDtype, {4}));
    if (offsetsDtype == Api::DataType::UINT32) {
        uint32_t* q = offsetTensor.getMemPtr<uint32_t>();
        for (uint32_t i = 0; i < offsets.size(); ++i) q[i] = static_cast<uint32_t>(offsets[i]);
    } else {
        uint64_t* q = offsetTensor.getMemPtr<uint64_t>();
        for (uint32_t i = 0; i < offsets.size(); ++i) q[i] = offsets[i];
    }
    return Impl::RaggedTensor(values, offsetTensor, maxValuesPerRow);
}

Batch makePlacementNeutralSegmentedBatch(const std::string& inputName,
                                                const Api::RaggedTensor& logicalInput,
                                                uint32_t batchSize) {
    const Impl::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    const uint64_t capacity = descriptor.getMaxTotalValues();
    const uint64_t activeValues = static_cast<uint64_t>(batchSize) * 2;
    THOR_THROW_IF_FALSE(logicalInput.getTrailingDimensions() == (std::vector<uint64_t>{1}));
    THOR_THROW_IF_FALSE(logicalInput.getMaxValuesPerRow() >= 2);
    THOR_THROW_IF_FALSE(activeValues <= capacity);

    Impl::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint32_t row = 0; row < batchSize; ++row) {
        valuesPtr[row * 2] = 0.0F;
        valuesPtr[row * 2 + 1] = 1.0F;
    }
    ThorTest::poisonInactiveElements(
        valuesPtr, activeValues, capacity, ThorTest::RaggedInactivePoison::NaN);

    Impl::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    std::vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    for (uint32_t row = 0; row <= batchSize; ++row) hostOffsets[row] = static_cast<uint64_t>(row) * 2;
    if (logicalInput.getOffsetsDataType() == Api::DataType::UINT32) {
        uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);
    } else {
        THOR_THROW_IF_FALSE(logicalInput.getOffsetsDataType() == Api::DataType::UINT64);
        uint64_t* offsetsPtr = offsets.getMemPtr<uint64_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = hostOffsets[i];
    }

    Impl::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, Impl::RaggedTensor(values, partition));
    return batch;
}

void expectActiveValuesNear(const Impl::RaggedTensor& actual,
                            const vector<float>& expected,
                            uint64_t elementsPerValue = 2,
                            float tolerance = 2.0e-5F) {
    ASSERT_EQ(actual.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(expected.size() / elementsPerValue));
    const float* p = actual.getValues().getMemPtr<float>();
    for (uint64_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(p[i], expected[i], tolerance) << "element " << i;
}

void runPublicRuntimeCase(Api::DataType offsetsDtype) {
    constexpr uint32_t batchSize = 3;
    Api::Network network("segmented_primitive_public_runtime");
    Api::RaggedTensor tokens = Api::RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("tokens")
                                   .valuesDataType(Api::DataType::FP32)
                                   .offsetsDataType(offsetsDtype)
                                   .trailingDimensions({2})
                                   .maxValuesPerRow(3).build();
    Api::NetworkInput rowValues = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("row_values")
                                      .dimensions({3})
                                      .dataType(Api::DataType::FP32)
                                      .build();

    Api::SegmentedSoftmax softmax =
        Api::SegmentedSoftmax::Builder().network(network).featureInput(tokens).build();
    Api::SegmentedLogSoftmax logSoftmax =
        Api::SegmentedLogSoftmax::Builder().network(network).featureInput(tokens).build();
    Api::SegmentedBroadcast broadcast = Api::SegmentedBroadcast::Builder()
                                            .network(network)
                                            .featureInput(rowValues.getFeatureOutput().value())
                                            .partitionInput(tokens)
                                            .build();

    EXPECT_EQ(softmax.getRaggedFeatureOutput().getOffsets(), tokens.getOffsets());
    EXPECT_EQ(logSoftmax.getRaggedFeatureOutput().getOffsets(), tokens.getOffsets());
    EXPECT_EQ(broadcast.getRaggedFeatureOutput().getOffsets(), tokens.getOffsets());
    EXPECT_EQ(broadcast.getRaggedFeatureOutput().getTrailingDimensions(), (vector<uint64_t>{3}));

    (void)Api::RaggedNetworkOutput::Builder()
        .network(network)
        .name("softmax")
        .inputTensor(softmax.getRaggedFeatureOutput())
        .build();
    (void)Api::RaggedNetworkOutput::Builder()
        .network(network)
        .name("log_softmax")
        .inputTensor(logSoftmax.getRaggedFeatureOutput())
        .build();
    (void)Api::RaggedNetworkOutput::Builder()
        .network(network)
        .name("broadcast")
        .inputTensor(broadcast.getRaggedFeatureOutput())
        .build();

    vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();

    Batch batch;
    batch.insert("tokens", makeRaggedInput(offsetsDtype, {0, 2, 2, 5}));
    batch.insert("row_values", makeDenseRows());
    std::map<std::string, Api::InferenceOutputValue> outputs = placed->inferLogical(batch);

    const Impl::RaggedTensor soft = std::get<Impl::RaggedTensor>(outputs.at("softmax"));
    const Impl::RaggedTensor logSoft = std::get<Impl::RaggedTensor>(outputs.at("log_softmax"));
    const Impl::RaggedTensor broad = std::get<Impl::RaggedTensor>(outputs.at("broadcast"));

    const float row0Low = 1.0F / (1.0F + std::exp(2.0F));
    const float row0High = 1.0F - row0Low;
    const float denom = 1.0F + std::exp(1.0F) + std::exp(2.0F);
    const vector<float> expectedSoft{
        row0Low, row0Low, row0High, row0High,
        1.0F / 3.0F, 1.0F / denom,
        1.0F / 3.0F, std::exp(1.0F) / denom,
        1.0F / 3.0F, std::exp(2.0F) / denom};
    expectActiveValuesNear(soft, expectedSoft);

    ASSERT_EQ(logSoft.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));
    const float* logValues = logSoft.getValues().getMemPtr<float>();
    for (uint64_t i = 0; i < expectedSoft.size(); ++i) {
        EXPECT_NEAR(std::exp(logValues[i]), expectedSoft[i], 3.0e-5F) << "log-softmax element " << i;
    }

    expectActiveValuesNear(broad,
                           {10.0F, 20.0F, 30.0F, 10.0F, 20.0F, 30.0F,
                            70.0F, 80.0F, 90.0F, 70.0F, 80.0F, 90.0F, 70.0F, 80.0F, 90.0F},
                           3);

    // Reuse the same placed executable for an all-empty partition. No primitive
    // may inspect the NaN-poisoned packed capacity when offsets[B] == 0.
    Batch emptyBatch;
    emptyBatch.insert("tokens", makeRaggedInput(offsetsDtype, {0, 0, 0, 0}));
    emptyBatch.insert("row_values", makeDenseRows());
    outputs = placed->inferLogical(emptyBatch);
    EXPECT_EQ(std::get<Impl::RaggedTensor>(outputs.at("softmax")).getHostActiveValueCountIfAvailable(),
              std::optional<uint64_t>(0));
    EXPECT_EQ(std::get<Impl::RaggedTensor>(outputs.at("log_softmax")).getHostActiveValueCountIfAvailable(),
              std::optional<uint64_t>(0));
    EXPECT_EQ(std::get<Impl::RaggedTensor>(outputs.at("broadcast")).getHostActiveValueCountIfAvailable(),
              std::optional<uint64_t>(0));
}

}  // namespace

TEST(SegmentedPrimitiveApi, CommonLayerRaggedJsonHelperRoundTripsLogicalReferences) {
    Api::Network neutralNetwork("segmented_common_helper_neutral");
    Api::RaggedTensor neutral = Api::RaggedNetworkInput::Builder()
                                    .network(neutralNetwork)
                                    .name("neutral_tokens")
                                    .valuesDataType(Api::DataType::FP32)
                                    .offsetsDataType(Api::DataType::UINT32)
                                    .trailingDimensions({3})
                                    .maxValuesPerRow(5)
                                    .build();
    const nlohmann::json neutralJson = Api::SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(neutral);
    EXPECT_EQ(neutralJson.at("layer_ragged_reference_version"), "1.0.0");
    EXPECT_FALSE(neutralJson.contains("geometry_mode"));
    EXPECT_FALSE(neutralJson.contains("batch_size"));
    EXPECT_FALSE(neutralJson.contains("max_total_values"));
    EXPECT_EQ(neutralJson.at("max_values_per_row"), 5U);
    EXPECT_EQ(neutralJson.at("values").at("id"), neutral.getValues().getId());
    EXPECT_EQ(neutralJson.at("offsets").at("id"), neutral.getRowPartitionToken().getId());
    EXPECT_EQ(neutralJson.at("values").at("dimensions"), (vector<uint64_t>{5, 3}));
    EXPECT_EQ(neutralJson.at("offsets").at("dimensions"), (vector<uint64_t>{1}));

    Api::RaggedTensor neutralRoundTrip =
        Api::SegmentedPrimitiveDetail::reconstructInput(neutralJson, &neutralNetwork, "SegmentedPrimitiveHelperTest");
    EXPECT_EQ(neutralRoundTrip.getMaxValuesPerRow(), 5U);
    EXPECT_EQ(neutralRoundTrip.getValues().getOriginalId(), neutral.getValues().getOriginalId());
    EXPECT_EQ(neutralRoundTrip.getRowPartitionToken().getOriginalId(), neutral.getRowPartitionToken().getOriginalId());
    EXPECT_TRUE(neutralRoundTrip.sharesPartitionWith(neutral));

    // The reference version describes an input grammar, not an alternate model
    // geometry. An older archive's redundant marker remains readable.
    nlohmann::json oldCanonical = neutralJson;
    oldCanonical["geometry_mode"] = "placement_neutral";
    const auto oldCanonicalRoundTrip = Api::SegmentedPrimitiveDetail::reconstructInput(
        oldCanonical, &neutralNetwork, "SegmentedPrimitiveHelperTest");
    EXPECT_TRUE(oldCanonicalRoundTrip.sharesPartitionWith(neutral));
    EXPECT_EQ(oldCanonicalRoundTrip.getMaxValuesPerRow(), 5U);
    oldCanonical["geometry_mode"] = "fixed";
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(
                     oldCanonical, &neutralNetwork, "SegmentedPrimitiveHelperTest")),
                 std::runtime_error);

    // Every reader using the shared reconstruction helper must reject unknown
    // model fields, even when its owning layer has no local schema checker.
    // The old six-field canonical marker remains accepted above.
    auto malformedCanonical = neutralJson;
    malformedCanonical["batch_size"] = 2;
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(
                     malformedCanonical, &neutralNetwork, "SegmentedPrimitiveHelperTest")),
                 std::runtime_error);
    malformedCanonical = neutralJson;
    malformedCanonical["max_total_values"] = 10;
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(
                     malformedCanonical, &neutralNetwork, "SegmentedPrimitiveHelperTest")),
                 std::runtime_error);
    malformedCanonical = neutralJson;
    malformedCanonical["unrecognized_field"] = 1;
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(
                     malformedCanonical, &neutralNetwork, "SegmentedPrimitiveHelperTest")),
                 std::runtime_error);
    malformedCanonical = neutralJson;
    malformedCanonical.erase("max_values_per_row");
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(
                     malformedCanonical, &neutralNetwork, "SegmentedPrimitiveHelperTest")),
                 std::runtime_error);

    EXPECT_NO_THROW(Api::SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        neutralJson, neutralJson, neutralRoundTrip, "SegmentedPrimitiveHelperTest"));
    nlohmann::json mismatchedPartitionMetadata = neutralJson;
    mismatchedPartitionMetadata["max_values_per_row"] = 6;
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
                     mismatchedPartitionMetadata, neutralJson, neutralRoundTrip, "SegmentedPrimitiveHelperTest")),
                 std::runtime_error);
}

TEST(SegmentedPrimitiveApi, HistoricalRaggedReferenceGrammarUsesLiveLogicalMNotArchivedBatch) {
    Api::Network network("segmented_common_helper_legacy_normalization");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .trailingDimensions({2})
                                  .maxValuesPerRow(4)
                                  .build();
    auto historical = Api::SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(input);
    historical.erase("layer_ragged_reference_version");
    historical.erase("geometry_mode");
    historical.erase("max_values_per_row"); // Recoverable from original graph source.
    historical["batch_size"] = 2;
    historical["max_total_values"] = 8;
    historical["values"]["dimensions"] = vector<uint64_t>{8, 2};
    historical["offsets"]["dimensions"] = vector<uint64_t>{3};

    auto loaded = Api::SegmentedPrimitiveDetail::reconstructInput(historical, &network, "LegacyGrammar");
    EXPECT_TRUE(loaded.sharesPartitionWith(input));
    EXPECT_EQ(loaded.getValues().getOriginalId(), input.getValues().getOriginalId());
    EXPECT_EQ(loaded.getMaxValuesPerRow(), 4U);
    EXPECT_EQ(loaded.getDescriptorForPlacement(2).getMaxTotalValues(), 8U);
    EXPECT_EQ(loaded.getDescriptorForPlacement(7).getMaxTotalValues(), 28U);

    // Only the archived physical hints vary; the reconstructed logical model must not.
    historical["batch_size"] = 99;
    historical["max_total_values"] = 999;
    historical["values"]["dimensions"][0] = 999;
    historical["offsets"]["dimensions"][0] = 100;
    auto reloaded = Api::SegmentedPrimitiveDetail::reconstructInput(historical, &network, "LegacyGrammar");
    EXPECT_EQ(reloaded.getMaxValuesPerRow(), 4U);
    EXPECT_TRUE(reloaded.sharesPartitionWith(input));
    EXPECT_NO_THROW(Api::SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        historical, historical, reloaded, "LegacyGrammar"));

    auto malformed = historical;
    malformed["max_values_per_row"] = 3;
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(malformed, &network, "LegacyGrammar")),
                 std::runtime_error);
    malformed = historical;
    malformed["offsets"]["id"] = input.getValues().getOriginalId();
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(malformed, &network, "LegacyGrammar")),
                 std::exception);
    malformed = historical;
    malformed["values"]["dimensions"][1] = 3;
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(malformed, &network, "LegacyGrammar")),
                 std::runtime_error);
    malformed = historical;
    malformed["offsets"]["data_type"] = "fp32";
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(malformed, &network, "LegacyGrammar")),
                 std::runtime_error);
    malformed = historical;
    malformed["layer_ragged_reference_version"] = "1.0.0";
    EXPECT_THROW((Api::SegmentedPrimitiveDetail::reconstructInput(malformed, &network, "LegacyGrammar")),
                 std::runtime_error);
}

TEST(SegmentedPrimitiveApi, HistoricalSoftmaxAndLogSoftmaxLayerReadersNormalizePhysicalReferences) {
    for (bool logMode : {false, true}) {
        Api::Network network(logMode ? "historical_log_softmax" : "historical_softmax");
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network).name("tokens")
                                      .valuesDataType(Api::DataType::FP32)
                                      .offsetsDataType(Api::DataType::UINT32)
                                      .trailingDimensions({1}).maxValuesPerRow(4).build();
        nlohmann::json layerJson;
        if (logMode) {
            auto layer = Api::SegmentedLogSoftmax::Builder().network(network).featureInput(input).build();
            layerJson = layer.architectureJson();
        } else {
            auto layer = Api::SegmentedSoftmax::Builder().network(network).featureInput(input).build();
            layerJson = layer.architectureJson();
        }
        layerJson["version"] = "1.0.0";
        for (const char* field : {"ragged_feature_input", "ragged_feature_output"}) {
            auto& record = layerJson[field];
            record.erase("geometry_mode");
            record.erase("layer_ragged_reference_version");
            record["batch_size"] = 2;
            record["max_total_values"] = 8;
            record["values"]["dimensions"] = vector<uint64_t>{8, 1};
            record["offsets"]["dimensions"] = vector<uint64_t>{3};
        }
        ASSERT_NO_THROW({
            if (logMode) Api::SegmentedLogSoftmax::deserialize(layerJson, &network);
            else Api::SegmentedSoftmax::deserialize(layerJson, &network);
        });
        auto malformed = layerJson;
        malformed["ragged_feature_output"]["offsets"]["id"] = input.getValues().getOriginalId();
        if (logMode) EXPECT_THROW(Api::SegmentedLogSoftmax::deserialize(malformed, &network), std::exception);
        else EXPECT_THROW(Api::SegmentedSoftmax::deserialize(malformed, &network), std::exception);
    }
}

TEST(SegmentedPrimitiveApi, PlacementNeutralSegmentedSoftmaxSaveLoadReplacesPlacementGeometryPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    const std::string networkName = "segmented_softmax_placement_neutral_round_trip";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_segmented_softmax_neutral_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedInputValuesOriginalId = 0;
    uint64_t savedSoftmaxOutputValuesOriginalId = 0;
    uint64_t savedBoundaryOutputValuesOriginalId = 0;
    uint64_t savedPartitionTokenOriginalId = 0;
    {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(Api::DataType::FP32)
                                      .offsetsDataType(Api::DataType::UINT32)
                                      .trailingDimensions({1})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
        Api::SegmentedSoftmax softmax =
            Api::SegmentedSoftmax::Builder().network(network).featureInput(input).build();
        EXPECT_TRUE(softmax.getRaggedFeatureOutput().sharesPartitionWith(input));
        Api::RaggedNetworkOutput boundaryOutput = Api::RaggedNetworkOutput::Builder()
                                                      .network(network)
                                                      .name("softmax")
                                                      .inputTensor(softmax.getRaggedFeatureOutput())
                                                      .build();

        savedInputValuesOriginalId = input.getValues().getOriginalId();
        savedSoftmaxOutputValuesOriginalId = softmax.getRaggedFeatureOutput().getValues().getOriginalId();
        savedBoundaryOutputValuesOriginalId = boundaryOutput.getFeatureOutput().getValues().getOriginalId();
        savedPartitionTokenOriginalId = input.getRowPartitionToken().getOriginalId();
        EXPECT_NE(savedSoftmaxOutputValuesOriginalId, savedBoundaryOutputValuesOriginalId);
        EXPECT_EQ(boundaryOutput.getFeatureOutput().getRowPartitionToken().getOriginalId(),
                  savedPartitionTokenOriginalId);

        const nlohmann::json architecture = network.architectureJson();
        const nlohmann::json* softmaxJson = nullptr;
        for (const nlohmann::json& layerJson : architecture.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_softmax") {
                softmaxJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(softmaxJson, nullptr);
        EXPECT_EQ(softmaxJson->at("version").get<std::string>(), "1.1.0");
        const nlohmann::json& inputJson = softmaxJson->at("ragged_feature_input");
        const nlohmann::json& outputJson = softmaxJson->at("ragged_feature_output");
        for (const nlohmann::json* raggedJson : {&inputJson, &outputJson}) {
            EXPECT_EQ(raggedJson->at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
            EXPECT_FALSE(raggedJson->contains("geometry_mode"));
            EXPECT_EQ(raggedJson->at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
            EXPECT_FALSE(raggedJson->contains("batch_size"));
            EXPECT_FALSE(raggedJson->contains("max_total_values"));
            EXPECT_EQ(raggedJson->at("values").at("dimensions").get<std::vector<uint64_t>>(),
                      (std::vector<uint64_t>{maxValuesPerRow, 1}));
            EXPECT_EQ(raggedJson->at("offsets").at("dimensions").get<std::vector<uint64_t>>(),
                      (std::vector<uint64_t>{1}));
        }
        EXPECT_EQ(inputJson.at("values").at("id").get<uint64_t>(), input.getValues().getId());
        EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(), softmax.getRaggedFeatureOutput().getValues().getId());
        EXPECT_EQ(inputJson.at("offsets").at("id").get<uint64_t>(), input.getRowPartitionToken().getId());
        EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), input.getRowPartitionToken().getId());

        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));

        const auto loadedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedInputs.size(), 1U);
        const Api::RaggedTensor loadedInput = loadedInputs.front().raggedTensor;
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedInputValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
        const auto loadedOutputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(loadedOutputs.size(), 1U);
        const Api::RaggedTensor loadedOutput = loadedOutputs.front().raggedTensor;
        EXPECT_EQ(loadedOutput.getValues().getOriginalId(), savedBoundaryOutputValuesOriginalId);
        EXPECT_EQ(loadedOutput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);

        std::shared_ptr<Api::SegmentedSoftmax> loadedSoftmax;
        for (uint32_t layerIndex = 0; layerIndex < loaded.getNumLayers(); ++layerIndex) {
            loadedSoftmax = std::dynamic_pointer_cast<Api::SegmentedSoftmax>(loaded.getLayer(layerIndex));
            if (loadedSoftmax != nullptr) break;
        }
        ASSERT_NE(loadedSoftmax, nullptr);
        const Api::RaggedTensor loadedSoftmaxOutput = loadedSoftmax->getRaggedFeatureOutput();
        EXPECT_EQ(loadedSoftmaxOutput.getValues().getOriginalId(), savedSoftmaxOutputValuesOriginalId);
        EXPECT_EQ(loadedSoftmaxOutput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
        EXPECT_NE(loadedSoftmaxOutput.getValues().getId(), loadedOutput.getValues().getId());

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        const nlohmann::json* loadedSoftmaxJson = nullptr;
        for (const nlohmann::json& layerJson : architectureBeforePlacement.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_softmax") {
                loadedSoftmaxJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(loadedSoftmaxJson, nullptr);
        EXPECT_EQ(loadedSoftmaxJson->at("ragged_feature_input").at("values").at("id").get<uint64_t>(),
                  loadedInput.getValues().getId());
        EXPECT_EQ(loadedSoftmaxJson->at("ragged_feature_output").at("values").at("id").get<uint64_t>(),
                  loadedSoftmaxOutput.getValues().getId());
        EXPECT_EQ(loadedSoftmaxJson->at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
                  loadedInput.getRowPartitionToken().getId());
        EXPECT_EQ(loadedSoftmaxJson->at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(),
                  loadedInput.getRowPartitionToken().getId());

        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed =
            loaded.place(placementBatchSize, initDoneEvents, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        std::shared_ptr<Impl::NetworkInput> valuesInput =
            placed->getStampedNetwork(0).getNamedInput("tokens.values");
        ASSERT_NE(valuesInput, nullptr);
        ASSERT_TRUE(valuesInput->getFeatureOutput().has_value());
        EXPECT_EQ(valuesInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{expectedCapacity, 1}));

        std::shared_ptr<Impl::NetworkInput> offsetsInput =
            placed->getStampedNetwork(0).getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(offsetsInput, nullptr);
        ASSERT_TRUE(offsetsInput->getFeatureOutput().has_value());
        EXPECT_EQ(offsetsInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const std::map<std::string, Api::InferenceOutputValue> outputs =
            placed->inferLogical(makePlacementNeutralSegmentedBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("softmax"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("softmax")));
        const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("softmax"));
        EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{expectedCapacity, 1}));
        EXPECT_EQ(result.getOffsets().getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        std::vector<uint64_t> expectedOffsets(static_cast<size_t>(placementBatchSize) + 1, 0);
        for (uint32_t row = 0; row <= placementBatchSize; ++row) expectedOffsets[row] = static_cast<uint64_t>(row) * 2;
        EXPECT_EQ(result.getHostOffsetsIfAvailable(), std::optional<std::vector<uint64_t>>(expectedOffsets));
        ASSERT_EQ(result.getHostActiveValueCountIfAvailable(),
                  std::optional<uint64_t>(static_cast<uint64_t>(placementBatchSize) * 2));
        const float expectedLow = 1.0F / (1.0F + std::exp(1.0F));
        const float expectedHigh = 1.0F - expectedLow;
        const float* resultValues = result.getValues().getMemPtr<float>();
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            EXPECT_NEAR(resultValues[row * 2], expectedLow, 2.0e-5F) << "row " << row;
            EXPECT_NEAR(resultValues[row * 2 + 1], expectedHigh, 2.0e-5F) << "row " << row;
        }
    };

    loadPlaceAndRun(2);
    loadPlaceAndRun(7);
    std::filesystem::remove_all(archive);
}

TEST(SegmentedPrimitiveApi, PlacementNeutralSegmentedLogSoftmaxSaveLoadReplacesPlacementGeometryPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    const std::string networkName = "segmented_log_softmax_placement_neutral_round_trip";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_segmented_log_softmax_neutral_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedInputValuesOriginalId = 0;
    uint64_t savedLogSoftmaxOutputValuesOriginalId = 0;
    uint64_t savedBoundaryOutputValuesOriginalId = 0;
    uint64_t savedPartitionTokenOriginalId = 0;
    {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(Api::DataType::FP32)
                                      .offsetsDataType(Api::DataType::UINT32)
                                      .trailingDimensions({1})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
        Api::SegmentedLogSoftmax logSoftmax =
            Api::SegmentedLogSoftmax::Builder().network(network).featureInput(input).build();
        EXPECT_TRUE(logSoftmax.getRaggedFeatureOutput().sharesPartitionWith(input));
        Api::RaggedNetworkOutput boundaryOutput = Api::RaggedNetworkOutput::Builder()
                                                      .network(network)
                                                      .name("log_softmax")
                                                      .inputTensor(logSoftmax.getRaggedFeatureOutput())
                                                      .build();

        savedInputValuesOriginalId = input.getValues().getOriginalId();
        savedLogSoftmaxOutputValuesOriginalId = logSoftmax.getRaggedFeatureOutput().getValues().getOriginalId();
        savedBoundaryOutputValuesOriginalId = boundaryOutput.getFeatureOutput().getValues().getOriginalId();
        savedPartitionTokenOriginalId = input.getRowPartitionToken().getOriginalId();
        EXPECT_NE(savedLogSoftmaxOutputValuesOriginalId, savedBoundaryOutputValuesOriginalId);
        EXPECT_EQ(boundaryOutput.getFeatureOutput().getRowPartitionToken().getOriginalId(),
                  savedPartitionTokenOriginalId);

        const nlohmann::json architecture = network.architectureJson();
        const nlohmann::json* logSoftmaxJson = nullptr;
        for (const nlohmann::json& layerJson : architecture.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_log_softmax") {
                logSoftmaxJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(logSoftmaxJson, nullptr);
        EXPECT_EQ(logSoftmaxJson->at("version").get<std::string>(), "1.1.0");
        const nlohmann::json& inputJson = logSoftmaxJson->at("ragged_feature_input");
        const nlohmann::json& outputJson = logSoftmaxJson->at("ragged_feature_output");
        for (const nlohmann::json* raggedJson : {&inputJson, &outputJson}) {
            EXPECT_EQ(raggedJson->at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
            EXPECT_FALSE(raggedJson->contains("geometry_mode"));
            EXPECT_EQ(raggedJson->at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
            EXPECT_FALSE(raggedJson->contains("batch_size"));
            EXPECT_FALSE(raggedJson->contains("max_total_values"));
            EXPECT_EQ(raggedJson->at("values").at("dimensions").get<std::vector<uint64_t>>(),
                      (std::vector<uint64_t>{maxValuesPerRow, 1}));
            EXPECT_EQ(raggedJson->at("offsets").at("dimensions").get<std::vector<uint64_t>>(),
                      (std::vector<uint64_t>{1}));
        }
        EXPECT_EQ(inputJson.at("values").at("id").get<uint64_t>(), input.getValues().getId());
        EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(), logSoftmax.getRaggedFeatureOutput().getValues().getId());
        EXPECT_EQ(inputJson.at("offsets").at("id").get<uint64_t>(), input.getRowPartitionToken().getId());
        EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), input.getRowPartitionToken().getId());

        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));

        const auto loadedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedInputs.size(), 1U);
        const Api::RaggedTensor loadedInput = loadedInputs.front().raggedTensor;
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedInputValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
        const auto loadedOutputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(loadedOutputs.size(), 1U);
        const Api::RaggedTensor loadedOutput = loadedOutputs.front().raggedTensor;
        EXPECT_EQ(loadedOutput.getValues().getOriginalId(), savedBoundaryOutputValuesOriginalId);
        EXPECT_EQ(loadedOutput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);

        std::shared_ptr<Api::SegmentedLogSoftmax> loadedLogSoftmax;
        for (uint32_t layerIndex = 0; layerIndex < loaded.getNumLayers(); ++layerIndex) {
            loadedLogSoftmax = std::dynamic_pointer_cast<Api::SegmentedLogSoftmax>(loaded.getLayer(layerIndex));
            if (loadedLogSoftmax != nullptr) break;
        }
        ASSERT_NE(loadedLogSoftmax, nullptr);
        const Api::RaggedTensor loadedLogSoftmaxOutput = loadedLogSoftmax->getRaggedFeatureOutput();
        EXPECT_EQ(loadedLogSoftmaxOutput.getValues().getOriginalId(), savedLogSoftmaxOutputValuesOriginalId);
        EXPECT_EQ(loadedLogSoftmaxOutput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
        EXPECT_NE(loadedLogSoftmaxOutput.getValues().getId(), loadedOutput.getValues().getId());

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        const nlohmann::json* loadedLogSoftmaxJson = nullptr;
        for (const nlohmann::json& layerJson : architectureBeforePlacement.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_log_softmax") {
                loadedLogSoftmaxJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(loadedLogSoftmaxJson, nullptr);
        EXPECT_EQ(loadedLogSoftmaxJson->at("ragged_feature_input").at("values").at("id").get<uint64_t>(),
                  loadedInput.getValues().getId());
        EXPECT_EQ(loadedLogSoftmaxJson->at("ragged_feature_output").at("values").at("id").get<uint64_t>(),
                  loadedLogSoftmaxOutput.getValues().getId());
        EXPECT_EQ(loadedLogSoftmaxJson->at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
                  loadedInput.getRowPartitionToken().getId());
        EXPECT_EQ(loadedLogSoftmaxJson->at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(),
                  loadedInput.getRowPartitionToken().getId());

        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed =
            loaded.place(placementBatchSize, initDoneEvents, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        std::shared_ptr<Impl::NetworkInput> valuesInput =
            placed->getStampedNetwork(0).getNamedInput("tokens.values");
        ASSERT_NE(valuesInput, nullptr);
        ASSERT_TRUE(valuesInput->getFeatureOutput().has_value());
        EXPECT_EQ(valuesInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{expectedCapacity, 1}));

        std::shared_ptr<Impl::NetworkInput> offsetsInput =
            placed->getStampedNetwork(0).getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(offsetsInput, nullptr);
        ASSERT_TRUE(offsetsInput->getFeatureOutput().has_value());
        EXPECT_EQ(offsetsInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const std::map<std::string, Api::InferenceOutputValue> outputs =
            placed->inferLogical(makePlacementNeutralSegmentedBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("log_softmax"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("log_softmax")));
        const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("log_softmax"));
        EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{expectedCapacity, 1}));
        EXPECT_EQ(result.getOffsets().getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        std::vector<uint64_t> expectedOffsets(static_cast<size_t>(placementBatchSize) + 1, 0);
        for (uint32_t row = 0; row <= placementBatchSize; ++row) expectedOffsets[row] = static_cast<uint64_t>(row) * 2;
        EXPECT_EQ(result.getHostOffsetsIfAvailable(), std::optional<std::vector<uint64_t>>(expectedOffsets));
        ASSERT_EQ(result.getHostActiveValueCountIfAvailable(),
                  std::optional<uint64_t>(static_cast<uint64_t>(placementBatchSize) * 2));
        const float logDenominator = std::log1p(std::exp(1.0F));
        const float expectedLow = -logDenominator;
        const float expectedHigh = 1.0F - logDenominator;
        const float* resultValues = result.getValues().getMemPtr<float>();
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            EXPECT_NEAR(resultValues[row * 2], expectedLow, 2.0e-5F) << "row " << row;
            EXPECT_NEAR(resultValues[row * 2 + 1], expectedHigh, 2.0e-5F) << "row " << row;
        }
    };

    loadPlaceAndRun(2);
    loadPlaceAndRun(7);
    std::filesystem::remove_all(archive);
}

TEST(SegmentedPrimitiveApi, PlacementNeutralSegmentedBroadcastBatchRecoveryCoversDenseAndOffsetsDrivers) {
    constexpr uint64_t maxValuesPerRow = 4;
    Api::Network network("segmented_broadcast_batch_recovery");
    Api::RaggedTensor partition = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(Api::DataType::FP32)
                                      .offsetsDataType(Api::DataType::UINT32)
                                      .trailingDimensions({1})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
    Api::NetworkInput dense = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("rows")
                                  .dimensions({3})
                                  .dataType(Api::DataType::FP32)
                                  .build();

    const auto denseDriver = std::make_shared<Impl::NetworkInput>(
        cpuPlacement, Api::DataType::FP32, std::vector<unsigned long>{7, 3});
    EXPECT_EQ(Api::SegmentedPrimitivePlacementDetail::placementBatchSizeForStamp(
                  dense.getFeatureOutput().value(),
                  partition,
                  denseDriver,
                  dense.getFeatureOutput().value(),
                  "SegmentedBroadcast"),
              7U);

    const auto offsetsDriver = std::make_shared<Impl::NetworkInput>(
        cpuPlacement, Api::DataType::UINT32, std::vector<unsigned long>{8});
    EXPECT_EQ(Api::SegmentedPrimitivePlacementDetail::placementBatchSizeForStamp(
                  dense.getFeatureOutput().value(),
                  partition,
                  offsetsDriver,
                  partition.getOffsets(),
                  "SegmentedBroadcast"),
              7U);
}

TEST(SegmentedPrimitiveApi, PlacementNeutralSegmentedBroadcastBuildsAndExecutesAtDifferentBatches) {
    constexpr uint64_t maxValuesPerRow = 4;

    auto buildPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network network("segmented_broadcast_placement_neutral_" + std::to_string(placementBatchSize));
        Api::RaggedTensor partition = Api::RaggedNetworkInput::Builder()
                                          .network(network)
                                          .name("tokens")
                                          .valuesDataType(Api::DataType::FP32)
                                          .offsetsDataType(Api::DataType::UINT32)
                                          .trailingDimensions({1})
                                          .maxValuesPerRow(maxValuesPerRow)
                                          .build();
        Api::NetworkInput dense = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("rows")
                                      .dimensions({3})
                                      .dataType(Api::DataType::FP32)
                                      .build();
        (void)Api::Stub::Builder().network(network).inputTensor(partition.getValues()).build();
        Api::SegmentedBroadcast broadcast = Api::SegmentedBroadcast::Builder()
                                                .network(network)
                                                .featureInput(dense.getFeatureOutput().value())
                                                .partitionInput(partition)
                                                .build();
        const Api::RaggedTensor logicalOutput = broadcast.getRaggedFeatureOutput();
        EXPECT_EQ(logicalOutput.getValuesDimensions(), (std::vector<uint64_t>{maxValuesPerRow, 3}));
        EXPECT_EQ(logicalOutput.getRowPartitionToken(), partition.getRowPartitionToken());
        EXPECT_TRUE(logicalOutput.sharesPartitionWith(partition));

        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("broadcast")
            .inputTensor(logicalOutput)
            .build();

        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed =
            network.place(placementBatchSize, initDoneEvents, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        std::shared_ptr<Impl::NetworkInput> valuesInput =
            placed->getStampedNetwork(0).getNamedInput("tokens.values");
        ASSERT_NE(valuesInput, nullptr);
        ASSERT_TRUE(valuesInput->getFeatureOutput().has_value());
        EXPECT_EQ(valuesInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{expectedCapacity, 1}));

        std::shared_ptr<Impl::NetworkInput> offsetsInput =
            placed->getStampedNetwork(0).getManagedPartitionOffsetsInputForTest(partition.getRowPartitionId());
        ASSERT_NE(offsetsInput, nullptr);
        ASSERT_TRUE(offsetsInput->getFeatureOutput().has_value());
        EXPECT_EQ(offsetsInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));
        EXPECT_EQ(broadcast.getRaggedFeatureOutput().getValuesDimensions(),
                  (std::vector<uint64_t>{maxValuesPerRow, 3}));

        Batch batch = makePlacementNeutralSegmentedBatch("tokens", partition, placementBatchSize);
        batch.insert("rows", makePlacementNeutralBroadcastDenseRows(placementBatchSize));
        const std::map<std::string, Api::InferenceOutputValue> outputs = placed->inferLogical(batch);
        ASSERT_TRUE(outputs.contains("broadcast"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("broadcast")));
        const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("broadcast"));
        EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{expectedCapacity, 3}));
        EXPECT_EQ(result.getOffsets().getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        std::vector<uint64_t> expectedOffsets(static_cast<size_t>(placementBatchSize) + 1, 0);
        for (uint32_t row = 0; row <= placementBatchSize; ++row) {
            expectedOffsets[row] = static_cast<uint64_t>(row) * 2;
        }
        EXPECT_EQ(result.getHostOffsetsIfAvailable(), std::optional<std::vector<uint64_t>>(expectedOffsets));
        ASSERT_EQ(result.getHostActiveValueCountIfAvailable(),
                  std::optional<uint64_t>(static_cast<uint64_t>(placementBatchSize) * 2));

        const float* resultValues = result.getValues().getMemPtr<float>();
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            for (uint32_t token = 0; token < 2; ++token) {
                const uint64_t base = (static_cast<uint64_t>(row) * 2 + token) * 3;
                EXPECT_NEAR(resultValues[base], static_cast<float>(row * 10 + 1), 1.0e-6F);
                EXPECT_NEAR(resultValues[base + 1], static_cast<float>(row * 10 + 2), 1.0e-6F);
                EXPECT_NEAR(resultValues[base + 2], static_cast<float>(row * 10 + 3), 1.0e-6F);
            }
        }
        EXPECT_EQ(broadcast.getRaggedFeatureOutput().getValuesDimensions(),
                  (std::vector<uint64_t>{maxValuesPerRow, 3}));
    };

    buildPlaceAndRun(2);
    buildPlaceAndRun(7);
}

TEST(SegmentedPrimitiveApi, PlacementNeutralSegmentedBroadcastSaveLoadReplacesPlacementGeometryPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    const std::string networkName = "segmented_broadcast_placement_neutral_round_trip";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_segmented_broadcast_neutral_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedPartitionValuesOriginalId = 0;
    uint64_t savedDenseValuesOriginalId = 0;
    uint64_t savedBroadcastOutputValuesOriginalId = 0;
    uint64_t savedBoundaryOutputValuesOriginalId = 0;
    uint64_t savedPartitionTokenOriginalId = 0;
    {
        Api::Network network(networkName);
        Api::RaggedTensor partition = Api::RaggedNetworkInput::Builder()
                                          .network(network)
                                          .name("tokens")
                                          .valuesDataType(Api::DataType::FP32)
                                          .offsetsDataType(Api::DataType::UINT32)
                                          .trailingDimensions({1})
                                          .maxValuesPerRow(maxValuesPerRow)
                                          .build();
        Api::NetworkInput dense = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("rows")
                                      .dimensions({3})
                                      .dataType(Api::DataType::FP32)
                                      .build();
        (void)Api::Stub::Builder().network(network).inputTensor(partition.getValues()).build();
        Api::SegmentedBroadcast broadcast = Api::SegmentedBroadcast::Builder()
                                                .network(network)
                                                .featureInput(dense.getFeatureOutput().value())
                                                .partitionInput(partition)
                                                .build();
        EXPECT_EQ(broadcast.getRaggedFeatureOutput().getValuesDimensions(),
                  (std::vector<uint64_t>{maxValuesPerRow, 3}));
        EXPECT_TRUE(broadcast.getRaggedFeatureOutput().sharesPartitionWith(partition));
        Api::RaggedNetworkOutput boundaryOutput = Api::RaggedNetworkOutput::Builder()
                                                      .network(network)
                                                      .name("broadcast")
                                                      .inputTensor(broadcast.getRaggedFeatureOutput())
                                                      .build();

        savedPartitionValuesOriginalId = partition.getValues().getOriginalId();
        savedDenseValuesOriginalId = dense.getFeatureOutput()->getOriginalId();
        savedBroadcastOutputValuesOriginalId = broadcast.getRaggedFeatureOutput().getValues().getOriginalId();
        savedBoundaryOutputValuesOriginalId = boundaryOutput.getFeatureOutput().getValues().getOriginalId();
        savedPartitionTokenOriginalId = partition.getRowPartitionToken().getOriginalId();
        EXPECT_NE(savedBroadcastOutputValuesOriginalId, savedBoundaryOutputValuesOriginalId);
        EXPECT_EQ(boundaryOutput.getFeatureOutput().getRowPartitionToken().getOriginalId(),
                  savedPartitionTokenOriginalId);

        const nlohmann::json architecture = network.architectureJson();
        const nlohmann::json* broadcastJson = nullptr;
        for (const nlohmann::json& layerJson : architecture.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_broadcast") {
                broadcastJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(broadcastJson, nullptr);
        EXPECT_EQ(broadcastJson->at("version").get<std::string>(), "1.1.0");
        EXPECT_EQ(broadcastJson->at("feature_input").at("id").get<uint64_t>(), dense.getFeatureOutput()->getId());
        EXPECT_EQ(broadcastJson->at("feature_input").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{3}));

        const nlohmann::json& partitionJson = broadcastJson->at("partition_input");
        const nlohmann::json& outputJson = broadcastJson->at("ragged_feature_output");
        EXPECT_EQ(partitionJson.at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
        EXPECT_EQ(outputJson.at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
        EXPECT_FALSE(partitionJson.contains("geometry_mode"));
        EXPECT_FALSE(outputJson.contains("geometry_mode"));
        EXPECT_EQ(partitionJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
        EXPECT_EQ(outputJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
        EXPECT_FALSE(partitionJson.contains("batch_size"));
        EXPECT_FALSE(partitionJson.contains("max_total_values"));
        EXPECT_FALSE(outputJson.contains("batch_size"));
        EXPECT_FALSE(outputJson.contains("max_total_values"));
        EXPECT_EQ(partitionJson.at("values").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{maxValuesPerRow, 1}));
        EXPECT_EQ(outputJson.at("values").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{maxValuesPerRow, 3}));
        EXPECT_EQ(partitionJson.at("offsets").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{1}));
        EXPECT_EQ(outputJson.at("offsets").at("dimensions").get<std::vector<uint64_t>>(),
                  (std::vector<uint64_t>{1}));
        EXPECT_EQ(partitionJson.at("values").at("id").get<uint64_t>(), partition.getValues().getId());
        EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(), broadcast.getRaggedFeatureOutput().getValues().getId());
        EXPECT_EQ(partitionJson.at("offsets").at("id").get<uint64_t>(), partition.getRowPartitionToken().getId());
        EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), partition.getRowPartitionToken().getId());

        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));

        const auto loadedRaggedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedRaggedInputs.size(), 1U);
        const Api::RaggedTensor loadedPartition = loadedRaggedInputs.front().raggedTensor;
        EXPECT_EQ(loadedPartition.getValues().getOriginalId(), savedPartitionValuesOriginalId);
        EXPECT_EQ(loadedPartition.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);

        std::shared_ptr<Api::NetworkInput> loadedDenseInput;
        for (const std::shared_ptr<Api::NetworkInput>& candidate : loaded.getExternalNetworkInputs()) {
            if (candidate->getName() == "rows") {
                loadedDenseInput = candidate;
                break;
            }
        }
        ASSERT_NE(loadedDenseInput, nullptr);
        ASSERT_TRUE(loadedDenseInput->getFeatureOutput().has_value());
        EXPECT_EQ(loadedDenseInput->getFeatureOutput()->getOriginalId(), savedDenseValuesOriginalId);
        EXPECT_EQ(loadedDenseInput->getDimensions(), (std::vector<uint64_t>{3}));

        const auto loadedOutputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(loadedOutputs.size(), 1U);
        const Api::RaggedTensor loadedBoundaryOutput = loadedOutputs.front().raggedTensor;
        EXPECT_EQ(loadedBoundaryOutput.getValues().getOriginalId(), savedBoundaryOutputValuesOriginalId);
        EXPECT_EQ(loadedBoundaryOutput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);

        std::shared_ptr<Api::SegmentedBroadcast> loadedBroadcast;
        for (uint32_t layerIndex = 0; layerIndex < loaded.getNumLayers(); ++layerIndex) {
            loadedBroadcast = std::dynamic_pointer_cast<Api::SegmentedBroadcast>(loaded.getLayer(layerIndex));
            if (loadedBroadcast != nullptr) break;
        }
        ASSERT_NE(loadedBroadcast, nullptr);
        const Api::RaggedTensor loadedBroadcastOutput = loadedBroadcast->getRaggedFeatureOutput();
        EXPECT_EQ(loadedBroadcast->getDenseFeatureInput().getOriginalId(), savedDenseValuesOriginalId);
        EXPECT_EQ(loadedBroadcastOutput.getValues().getOriginalId(), savedBroadcastOutputValuesOriginalId);
        EXPECT_EQ(loadedBroadcastOutput.getRowPartitionToken().getOriginalId(), savedPartitionTokenOriginalId);
        EXPECT_NE(loadedBroadcastOutput.getValues().getId(), loadedBoundaryOutput.getValues().getId());
        EXPECT_EQ(loadedBroadcastOutput.getValuesDimensions(), (std::vector<uint64_t>{maxValuesPerRow, 3}));

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        const nlohmann::json* loadedBroadcastJson = nullptr;
        for (const nlohmann::json& layerJson : architectureBeforePlacement.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_broadcast") {
                loadedBroadcastJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(loadedBroadcastJson, nullptr);
        EXPECT_EQ(loadedBroadcastJson->at("version").get<std::string>(), "1.1.0");
        EXPECT_EQ(loadedBroadcastJson->at("feature_input").at("id").get<uint64_t>(),
                  loadedBroadcast->getDenseFeatureInput().getId());
        EXPECT_EQ(loadedBroadcastJson->at("partition_input").at("values").at("id").get<uint64_t>(),
                  loadedPartition.getValues().getId());
        EXPECT_EQ(loadedBroadcastJson->at("partition_input").at("offsets").at("id").get<uint64_t>(),
                  loadedPartition.getRowPartitionToken().getId());
        EXPECT_EQ(loadedBroadcastJson->at("ragged_feature_output").at("values").at("id").get<uint64_t>(),
                  loadedBroadcastOutput.getValues().getId());
        EXPECT_EQ(loadedBroadcastJson->at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(),
                  loadedPartition.getRowPartitionToken().getId());

        std::vector<Event> initDoneEvents;
        std::shared_ptr<Api::PlacedNetwork> placed =
            loaded.place(placementBatchSize, initDoneEvents, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        std::shared_ptr<Impl::NetworkInput> valuesInput =
            placed->getStampedNetwork(0).getNamedInput("tokens.values");
        ASSERT_NE(valuesInput, nullptr);
        ASSERT_TRUE(valuesInput->getFeatureOutput().has_value());
        EXPECT_EQ(valuesInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{expectedCapacity, 1}));

        std::shared_ptr<Impl::NetworkInput> offsetsInput =
            placed->getStampedNetwork(0).getManagedPartitionOffsetsInputForTest(loadedPartition.getRowPartitionId());
        ASSERT_NE(offsetsInput, nullptr);
        ASSERT_TRUE(offsetsInput->getFeatureOutput().has_value());
        EXPECT_EQ(offsetsInput->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        Batch batch = makePlacementNeutralSegmentedBatch("tokens", loadedPartition, placementBatchSize);
        batch.insert("rows", makePlacementNeutralBroadcastDenseRows(placementBatchSize));
        const std::map<std::string, Api::InferenceOutputValue> outputs = placed->inferLogical(batch);
        ASSERT_TRUE(outputs.contains("broadcast"));
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("broadcast")));
        const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("broadcast"));
        EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{expectedCapacity, 3}));
        EXPECT_EQ(result.getOffsets().getDimensions(),
                  (std::vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        std::vector<uint64_t> expectedOffsets(static_cast<size_t>(placementBatchSize) + 1, 0);
        for (uint32_t row = 0; row <= placementBatchSize; ++row) {
            expectedOffsets[row] = static_cast<uint64_t>(row) * 2;
        }
        EXPECT_EQ(result.getHostOffsetsIfAvailable(), std::optional<std::vector<uint64_t>>(expectedOffsets));
        ASSERT_EQ(result.getHostActiveValueCountIfAvailable(),
                  std::optional<uint64_t>(static_cast<uint64_t>(placementBatchSize) * 2));

        const float* resultValues = result.getValues().getMemPtr<float>();
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            for (uint32_t token = 0; token < 2; ++token) {
                const uint64_t base = (static_cast<uint64_t>(row) * 2 + token) * 3;
                EXPECT_NEAR(resultValues[base], static_cast<float>(row * 10 + 1), 1.0e-6F);
                EXPECT_NEAR(resultValues[base + 1], static_cast<float>(row * 10 + 2), 1.0e-6F);
                EXPECT_NEAR(resultValues[base + 2], static_cast<float>(row * 10 + 3), 1.0e-6F);
            }
        }
    };

    loadPlaceAndRun(2);
    loadPlaceAndRun(7);
    std::filesystem::remove_all(archive);
}

TEST(SegmentedPrimitiveApi, BuildersPreservePartitionAndRejectFp64) {
    Api::Network network("segmented_primitive_builder_contract");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .trailingDimensions({4})
                                  .maxValuesPerRow(3).build();
    Api::NetworkInput dense = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("rows")
                                  .dimensions({5})
                                  .dataType(Api::DataType::FP32)
                                  .build();
    Api::SegmentedSoftmax softmax = Api::SegmentedSoftmax::Builder().network(network).featureInput(input).build();
    Api::SegmentedLogSoftmax logSoftmax =
        Api::SegmentedLogSoftmax::Builder().network(network).featureInput(input).build();
    Api::SegmentedBroadcast broadcast = Api::SegmentedBroadcast::Builder()
                                            .network(network)
                                            .featureInput(dense.getFeatureOutput().value())
                                            .partitionInput(input)
                                            .build();
    EXPECT_EQ(softmax.getRaggedFeatureOutput().getOffsets(), input.getOffsets());
    EXPECT_EQ(logSoftmax.getRaggedFeatureOutput().getOffsets(), input.getOffsets());
    EXPECT_EQ(broadcast.getRaggedFeatureOutput().getOffsets(), input.getOffsets());
    EXPECT_EQ(broadcast.getRaggedFeatureOutput().getValuesDimensions(), (vector<uint64_t>{3, 5}));

    Api::Network fp64Network("segmented_primitive_fp64_rejected");
    Api::RaggedTensor fp64Ragged = Api::RaggedNetworkInput::Builder()
                                       .network(fp64Network)
                                       .name("fp64_tokens")
                                       .valuesDataType(Api::DataType::FP64)
                                       .trailingDimensions({2})
                                       .maxValuesPerRow(2).build();
    Api::NetworkInput fp64Dense = Api::NetworkInput::Builder()
                                      .network(fp64Network)
                                      .name("fp64_rows")
                                      .dimensions({2})
                                      .dataType(Api::DataType::FP64)
                                      .build();
    EXPECT_THROW((Api::SegmentedSoftmax::Builder().network(fp64Network).featureInput(fp64Ragged).build()), std::invalid_argument);
    EXPECT_THROW((Api::SegmentedLogSoftmax::Builder().network(fp64Network).featureInput(fp64Ragged).build()), std::invalid_argument);
    EXPECT_THROW((Api::SegmentedBroadcast::Builder()
                      .network(fp64Network)
                      .featureInput(fp64Dense.getFeatureOutput().value())
                      .partitionInput(fp64Ragged)
                      .build()),
                 std::invalid_argument);
}


TEST(SegmentedPrimitiveApi, HistoricalThreeLayerArchiveLoadsLogicalAtTwoPlacements) {
    const std::string name = "segmented_primitive_historical_round_trip";
    constexpr uint64_t m = 4;
    Api::Network network(name);
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network).name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .offsetsDataType(Api::DataType::UINT64)
                                  .trailingDimensions({2}).maxValuesPerRow(m).build();
    Api::NetworkInput dense = Api::NetworkInput::Builder()
                                  .network(network).name("rows")
                                  .dimensions({3}).dataType(Api::DataType::FP32).build();
    auto softmax = Api::SegmentedSoftmax::Builder().network(network).featureInput(input).build();
    auto logSoftmax = Api::SegmentedLogSoftmax::Builder().network(network).featureInput(input).build();
    auto broadcast = Api::SegmentedBroadcast::Builder()
                         .network(network).featureInput(dense.getFeatureOutput().value())
                         .partitionInput(input).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("softmax")
        .inputTensor(softmax.getRaggedFeatureOutput()).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("log_softmax")
        .inputTensor(logSoftmax.getRaggedFeatureOutput()).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("broadcast")
        .inputTensor(broadcast.getRaggedFeatureOutput()).build();

    nlohmann::json historical = network.architectureJson();
    uint32_t migrated = 0;
    for (auto& layer : historical.at("layers")) {
        const std::string type = layer.at("layer_type").get<std::string>();
        if (type == "segmented_softmax" || type == "segmented_log_softmax" || type == "segmented_broadcast") {
            layer["version"] = "1.0.0";
            for (const char* key : {"ragged_feature_input", "partition_input", "ragged_feature_output"}) {
                if (!layer.contains(key)) continue;
                auto& ref = layer.at(key);
                ref.erase("layer_ragged_reference_version");
                ref.erase("geometry_mode");
                ref.erase("max_values_per_row");
                ref["batch_size"] = 91;
                ref["max_total_values"] = 999;
                ref["values"]["dimensions"][0] = 12;
                ref["offsets"]["dimensions"][0] = 4;
            }
            ++migrated;
        } else if (type == "network_input") {
            const std::string inputName = layer.at("name").get<std::string>();
            if (inputName != "tokens.values" && inputName != "__thor_row_partition.tokens") continue;
            const uint64_t leading = inputName == "tokens.values" ? 12 : 4;
            layer["dimensions"][0] = leading;
            for (const char* key : {"feature_input", "feature_output"})
                layer[key]["dimensions"][0] = leading;
        }
    }
    ASSERT_EQ(migrated, 3u);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
        ("thor_historical_seg_primitives_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);
    const std::string encoded = historical.dump(4);
    Impl::Tensor bytes(cpuPlacement, Impl::TensorDescriptor(Api::DataType::UINT8, {encoded.size()}));
    std::memcpy(bytes.getMemPtr<void>(), encoded.data(), encoded.size());
    thor_file::TarWriter writer(name);
    writer.addArchiveFile(name + ".thor.json", bytes);
    writer.createArchive(archive, true);

    for (const uint32_t b : {2u, 7u}) {
        Api::Network loaded(name);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto outputs = loaded.getExternalRaggedNetworkOutputs();
        ASSERT_EQ(outputs.size(), 3u);
        for (const auto& output : outputs) {
            EXPECT_EQ(output.raggedTensor.getOffsetsDataType(), Api::DataType::UINT64);
            EXPECT_EQ(output.raggedTensor.getValuesDimensions().front(), m);
            EXPECT_EQ(output.raggedTensor.getOffsetsDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(output.raggedTensor.getDescriptorForPlacement(b).getMaxTotalValues(), b * m);
        }
        vector<Event> events;
        auto placed = loaded.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : events) event.synchronize();
    }
    std::filesystem::remove_all(archive);
}

TEST(SegmentedPrimitiveApi, PublicRuntimeSupportsUint32Offsets) { runPublicRuntimeCase(Api::DataType::UINT32); }
TEST(SegmentedPrimitiveApi, PublicRuntimeSupportsUint64Offsets) { runPublicRuntimeCase(Api::DataType::UINT64); }
