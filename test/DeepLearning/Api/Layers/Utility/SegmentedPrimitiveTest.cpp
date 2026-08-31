#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedLogSoftmax.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedSoftmax.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "test/DeepLearning/RaggedTestUtils.h"

#include "gtest/gtest.h"

#include <chrono>
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

Impl::RaggedTensor makeRaggedInput(Api::DataType offsetsDtype, const vector<uint64_t>& offsets) {
    constexpr uint64_t capacity = 8;
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
    return Impl::RaggedTensor(values, offsetTensor);
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
                                   .maxTotalValues(8)
                                   .batchSize(batchSize)
                                   .build();
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

TEST(SegmentedPrimitiveApi, BuildersPreservePartitionAndRejectFp64) {
    Api::Network network("segmented_primitive_builder_contract");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .trailingDimensions({4})
                                  .maxTotalValues(9)
                                  .batchSize(3)
                                  .build();
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
    EXPECT_EQ(broadcast.getRaggedFeatureOutput().getValuesDimensions(), (vector<uint64_t>{9, 5}));

    Api::Network fp64Network("segmented_primitive_fp64_rejected");
    Api::RaggedTensor fp64Ragged = Api::RaggedNetworkInput::Builder()
                                       .network(fp64Network)
                                       .name("fp64_tokens")
                                       .valuesDataType(Api::DataType::FP64)
                                       .trailingDimensions({2})
                                       .maxTotalValues(4)
                                       .batchSize(2)
                                       .build();
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


TEST(SegmentedPrimitiveApi, ArchitectureSaveLoadRoundTripsAllThreeLayers) {
    const std::string name = "segmented_primitive_architecture_round_trip";
    Api::Network network(name);
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(Api::DataType::FP32)
                                  .offsetsDataType(Api::DataType::UINT64)
                                  .trailingDimensions({2})
                                  .maxTotalValues(8)
                                  .batchSize(3)
                                  .build();
    Api::NetworkInput dense = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("rows")
                                  .dimensions({3})
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
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("softmax").inputTensor(softmax.getRaggedFeatureOutput()).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("log_softmax").inputTensor(logSoftmax.getRaggedFeatureOutput()).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("broadcast").inputTensor(broadcast.getRaggedFeatureOutput()).build();

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_segmented_primitive_round_trip_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);
    network.save(archive.string(), true);

    Api::Network loaded(name);
    ASSERT_NO_THROW(loaded.load(archive.string()));
    const nlohmann::json architecture = loaded.architectureJson();
    std::set<std::string> layerTypes;
    for (const auto& layer : architecture.at("layers")) layerTypes.insert(layer.at("layer_type").get<std::string>());
    EXPECT_TRUE(layerTypes.contains("segmented_softmax"));
    EXPECT_TRUE(layerTypes.contains("segmented_log_softmax"));
    EXPECT_TRUE(layerTypes.contains("segmented_broadcast"));
    ASSERT_EQ(loaded.getExternalRaggedNetworkOutputs().size(), 3U);
    for (const auto& output : loaded.getExternalRaggedNetworkOutputs()) {
        EXPECT_EQ(output.raggedTensor.getOffsetsDataType(), Api::DataType::UINT64);
        EXPECT_EQ(output.raggedTensor.getBatchSize(), 3U);
        EXPECT_EQ(output.raggedTensor.getMaxTotalValues(), 8U);
    }

    std::filesystem::remove_all(archive);
}

TEST(SegmentedPrimitiveApi, PublicRuntimeSupportsUint32Offsets) { runPublicRuntimeCase(Api::DataType::UINT32); }
TEST(SegmentedPrimitiveApi, PublicRuntimeSupportsUint64Offsets) { runPublicRuntimeCase(Api::DataType::UINT64); }
