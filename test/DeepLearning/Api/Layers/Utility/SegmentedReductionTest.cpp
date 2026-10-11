#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedReduction.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdint>
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

uint64_t numel(const Impl::Tensor& tensor) {
    uint64_t n = 1;
    for (uint64_t dim : tensor.getDimensions()) n *= dim;
    return n;
}

void writeCpuFloat(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(numel(tensor), values.size());
    auto* ptr = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i) ptr[i] = values[i];
}

vector<float> copyFloatToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor host = tensor.clone(cpuPlacement);
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    const auto* ptr = static_cast<const float*>(host.getMemPtr());
    return vector<float>(ptr, ptr + numel(host));
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events) event.synchronize();
    events.clear();
}

shared_ptr<Api::NetworkInput> findInput(Api::Network& network, const std::string& name) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto input = std::dynamic_pointer_cast<Api::NetworkInput>(network.getLayer(i));
        if (input != nullptr && input->getName() == name) return input;
    }
    return nullptr;
}

Batch makePlacementNeutralReductionBatch(const std::string& inputName,
                                         const Api::RaggedTensor& logicalInput,
                                         uint32_t batchSize) {
    const Impl::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    if (logicalInput.getTrailingDimensions() != (std::vector<uint64_t>{1}) ||
        logicalInput.getMaxValuesPerRow() < 2) {
        throw std::invalid_argument("Placement-neutral SegmentedReduction test input must be scalar with M >= 2.");
    }

    const uint64_t activeValues = static_cast<uint64_t>(batchSize) * 2;
    Impl::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint32_t row = 0; row < batchSize; ++row) {
        valuesPtr[row * 2] = static_cast<float>(row * 10 + 1);
        valuesPtr[row * 2 + 1] = static_cast<float>(row * 10 + 3);
    }
    for (uint64_t value = activeValues; value < descriptor.getMaxTotalValues(); ++value) {
        valuesPtr[value] = -1000.0F;
    }

    Impl::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    std::vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    for (uint32_t row = 0; row <= batchSize; ++row) {
        hostOffsets[row] = static_cast<uint64_t>(row) * 2;
    }
    if (logicalInput.getOffsetsDataType() == DataType::UINT32) {
        uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);
    } else {
        if (logicalInput.getOffsetsDataType() != DataType::UINT64) {
            throw std::invalid_argument("Placement-neutral SegmentedReduction test offsets must be uint32 or uint64.");
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

struct ReductionFixture {
    shared_ptr<Api::Network> network;
    Api::SegmentedReduction reduction;
    Api::NetworkOutput output;
    shared_ptr<Api::NetworkInput> valuesInput;
    Impl::RowPartitionId rowPartitionId = 0;
};

ReductionFixture makeReductionFixture(Api::SegmentedReduction::Type type) {
    auto network = std::make_shared<Api::Network>("segmented_reduction_api");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(*network)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2})
                                  .maxValuesPerRow(4)
                                  .build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(*network).tensor(input.getValues()).build();
    Api::RaggedTensor rivetedInput = input.withValues(inputRivet.getFeatureOutput().value());
    Api::SegmentedReduction reduction = Api::SegmentedReduction::Builder()
                                            .network(*network)
                                            .featureInput(rivetedInput)
                                            .reductionType(type)
                                            .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(*network).tensor(reduction.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(*network)
                                    .name("reduced")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    return {network,
            reduction,
            output,
            findInput(*network, "history.values"),
            input.getRowPartitionId()};
}

struct ExpectedCase {
    vector<float> forward;
    vector<float> backward;
    bool assertEmptyForward;
};

ExpectedCase expectedFor(Api::SegmentedReduction::Type type) {
    switch (type) {
        case Api::SegmentedReduction::Type::SUM:
            return {{6.0F, 30.0F, 0.0F, 0.0F, 23.0F, 33.0F},
                    {10.0F, 20.0F, 10.0F, 20.0F, 10.0F, 20.0F,
                     50.0F, 60.0F, 50.0F, 60.0F, 50.0F, 60.0F, 50.0F, 60.0F},
                    true};
        case Api::SegmentedReduction::Type::MEAN:
            return {{2.0F, 10.0F, 0.0F, 0.0F, 5.75F, 8.25F},
                    {10.0F / 3.0F, 20.0F / 3.0F, 10.0F / 3.0F, 20.0F / 3.0F, 10.0F / 3.0F, 20.0F / 3.0F,
                     12.5F, 15.0F, 12.5F, 15.0F, 12.5F, 15.0F, 12.5F, 15.0F},
                    true};
        case Api::SegmentedReduction::Type::MIN:
            return {{1.0F, 8.0F, 0.0F, 0.0F, 4.0F, 6.0F},
                    {10.0F, 0.0F, 0.0F, 20.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 50.0F, 0.0F, 0.0F, 60.0F, 0.0F, 0.0F},
                    false};
        case Api::SegmentedReduction::Type::MAX:
            return {{3.0F, 12.0F, 0.0F, 0.0F, 8.0F, 11.0F},
                    {0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 20.0F,
                     0.0F, 0.0F, 0.0F, 0.0F, 50.0F, 0.0F, 0.0F, 60.0F},
                    false};
    }
    return {};
}

void runForwardBackwardCase(Api::SegmentedReduction::Type type) {
    constexpr uint32_t batchSize = 3;
    ReductionFixture fixture = makeReductionFixture(type);
    ASSERT_NE(fixture.valuesInput, nullptr);
    EXPECT_EQ(fixture.reduction.getFeatureOutput()->getDimensions(), (vector<uint64_t>{2}));

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = fixture.network->place(batchSize, initDoneEvents, false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalValues = std::dynamic_pointer_cast<Impl::NetworkInput>(
        stamped.getPhysicalLayerFromApiLayer(fixture.valuesInput->getId()));
    auto physicalOffsets = stamped.getManagedPartitionOffsetsInputForTest(fixture.rowPartitionId);
    auto physicalReduction = std::dynamic_pointer_cast<Impl::CustomLayer>(
        stamped.getPhysicalLayerFromApiLayer(fixture.reduction.getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(
        stamped.getPhysicalLayerFromApiLayer(fixture.output.getId()));
    ASSERT_NE(physicalValues, nullptr);
    ASSERT_NE(physicalOffsets, nullptr);
    ASSERT_NE(physicalReduction, nullptr);
    ASSERT_NE(physicalOutput, nullptr);

    // offsets = [0,3,3,7]: unequal lengths and an empty middle row.
    // M=4 at B=3 provides 12 physical value slots; the final five inactive
    // slots are poison and must not participate in the reduction/gradient.
    const vector<float> values{
        1.0F, 10.0F, 3.0F, 8.0F, 2.0F, 12.0F,
        5.0F, 9.0F, 4.0F, 7.0F, 8.0F, 6.0F, 6.0F, 11.0F,
        -1000.0F, 1000.0F, 1000.0F, -1000.0F, -1000.0F, 1000.0F,
        1000.0F, -1000.0F, -1000.0F, 1000.0F};
    const vector<uint32_t> offsets{0U, 3U, 3U, 7U};
    Impl::Tensor valuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {12, 2}));
    writeCpuFloat(valuesHost, values);
    physicalValues->forward(valuesHost, false, batchSize);
    physicalOffsets->forwardManagedRowPartitionOffsets(
        false,
        Impl::RowPartitionDescriptor(batchSize, 12, DataType::UINT32, 4),
        batchSize,
        std::vector<uint64_t>(offsets.begin(), offsets.end()),
        fixture.rowPartitionId);
    physicalOutput->getOutputReadyEvent().synchronize();

    const ExpectedCase expected = expectedFor(type);
    const auto forward = static_cast<const float*>(physicalOutput->getFeatureOutput()->getMemPtr());
    ASSERT_EQ(physicalOutput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{3, 2}));
    EXPECT_NEAR(forward[0], expected.forward[0], 1.0e-5F);
    EXPECT_NEAR(forward[1], expected.forward[1], 1.0e-5F);
    if (expected.assertEmptyForward) {
        EXPECT_NEAR(forward[2], expected.forward[2], 1.0e-5F);
        EXPECT_NEAR(forward[3], expected.forward[3], 1.0e-5F);
    }
    EXPECT_NEAR(forward[4], expected.forward[4], 1.0e-5F);
    EXPECT_NEAR(forward[5], expected.forward[5], 1.0e-5F);

    ASSERT_GE(physicalReduction->getErrorInputs().size(), 1U);
    ASSERT_TRUE(physicalReduction->getErrorInputs()[0].has_value());
    ASSERT_GE(physicalReduction->getErrorOutputs().size(), 1U);
    ASSERT_TRUE(physicalReduction->getErrorOutputs()[0].has_value());
    Stream stream = physicalReduction->getStreams()[0];
    Impl::Tensor errorInput = physicalReduction->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    const vector<float> upstream{10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F};
    writeCpuFloat(errorInputHost, upstream);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalReduction->backward(errorInput, batchSize);
    const vector<float> backward = copyFloatToCpu(physicalReduction->getErrorOutputs()[0].value(), stream);
    ASSERT_GE(backward.size(), expected.backward.size());
    for (uint64_t i = 0; i < expected.backward.size(); ++i) {
        EXPECT_NEAR(backward[i], expected.backward[i], 2.0e-5F) << "gradient index " << i;
    }
}

}  // namespace

TEST(SegmentedReductionApi, BuilderProducesNormalDensePerExampleShape) {
    Api::Network network("segmented_reduction_shape");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({2, 3})
                                  .maxValuesPerRow(3)
                                  .build();
    Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                       .network(network)
                                       .featureInput(input)
                                       .reductionType(Api::SegmentedReduction::Type::MEAN)
                                       .build();
    EXPECT_EQ(mean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{2, 3}));
    EXPECT_EQ(mean.getFeatureInput().value(), input.getValues());

    Api::Network scalarNetwork("segmented_reduction_scalar_shape");
    Api::RaggedTensor scalarInput = Api::RaggedNetworkInput::Builder()
                                        .network(scalarNetwork)
                                        .name("scalars")
                                        .valuesDataType(DataType::FP32)
                                        .trailingDimensions({})
                                        .maxValuesPerRow(3)
                                        .build();
    Api::SegmentedReduction scalarMean = Api::SegmentedReduction::Builder()
                                             .network(scalarNetwork)
                                             .featureInput(scalarInput)
                                             .reductionType(Api::SegmentedReduction::Type::MEAN)
                                             .build();
    EXPECT_EQ(scalarMean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
}

TEST(SegmentedReductionApi, PlacementNeutralBuilderProducesDensePerExampleShapeWithoutBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    Api::Network network("segmented_reduction_neutral_shape");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2, 3})
                                  .maxValuesPerRow(maxValuesPerRow)
                                  .build();
    EXPECT_EQ(input.getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 2, 3}));

    Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                       .network(network)
                                       .featureInput(input)
                                       .reductionType(Api::SegmentedReduction::Type::MEAN)
                                       .build();
    EXPECT_EQ(mean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{2, 3}));
    EXPECT_EQ(mean.getFeatureInput().value(), input.getValues());

    Api::Network scalarNetwork("segmented_reduction_neutral_scalar_shape");
    Api::RaggedTensor scalarInput = Api::RaggedNetworkInput::Builder()
                                        .network(scalarNetwork)
                                        .name("scalars")
                                        .valuesDataType(DataType::FP32)
                                        .offsetsDataType(DataType::UINT32)
                                        .trailingDimensions({})
                                        .maxValuesPerRow(maxValuesPerRow)
                                        .build();
    Api::SegmentedReduction scalarMean = Api::SegmentedReduction::Builder()
                                             .network(scalarNetwork)
                                             .featureInput(scalarInput)
                                             .reductionType(Api::SegmentedReduction::Type::MEAN)
                                             .build();
    EXPECT_EQ(scalarMean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
}

TEST(SegmentedReductionApi, PlacementNeutralReductionBuildsAndExecutesAtDifferentBatches) {
    constexpr uint64_t maxValuesPerRow = 4;

    auto buildPlaceAndRun = [&](uint32_t placementBatchSize) {
        Api::Network network("segmented_reduction_neutral_runtime_" + std::to_string(placementBatchSize));
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({1})
                                      .maxValuesPerRow(maxValuesPerRow)
                                      .build();
        Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                           .network(network)
                                           .featureInput(input)
                                           .reductionType(Api::SegmentedReduction::Type::MEAN)
                                           .build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("mean")
            .inputTensor(mean.getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();

        ASSERT_EQ(mean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = network.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        synchronizeEvents(initDoneEvents);

        Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        shared_ptr<Impl::NetworkInput> physicalValues = stamped.getNamedInput("tokens.values");
        ASSERT_NE(physicalValues, nullptr);
        ASSERT_TRUE(physicalValues->getFeatureOutput().has_value());
        EXPECT_EQ(physicalValues->getFeatureOutput()->getDimensions(), (vector<uint64_t>{expectedCapacity, 1}));

        shared_ptr<Impl::NetworkInput> physicalOffsets =
            stamped.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId());
        ASSERT_NE(physicalOffsets, nullptr);
        ASSERT_TRUE(physicalOffsets->getFeatureOutput().has_value());
        EXPECT_EQ(physicalOffsets->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        auto physicalMean = std::dynamic_pointer_cast<Impl::CustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(mean.getId()));
        ASSERT_NE(physicalMean, nullptr);
        ASSERT_TRUE(physicalMean->getFeatureOutput().has_value());
        EXPECT_EQ(physicalMean->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{placementBatchSize, 1}));

        const Batch batch = makePlacementNeutralReductionBatch("tokens", input, placementBatchSize);
        const std::map<std::string, Impl::Tensor> outputs = placed->infer(batch);
        ASSERT_TRUE(outputs.contains("mean"));
        const Impl::Tensor& output = outputs.at("mean");
        ASSERT_EQ(output.getDimensions(), (vector<uint64_t>{placementBatchSize, 1}));
        const float* outputValues = output.getMemPtr<float>();
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            EXPECT_NEAR(outputValues[row], static_cast<float>(row * 10 + 2), 1.0e-5F) << "row " << row;
        }

        EXPECT_EQ(mean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(input.getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 1}));
    };

    buildPlaceAndRun(2);
    buildPlaceAndRun(7);
}

TEST(SegmentedReductionApi, PlacementNeutralManagedOffsetsCarrierDrivesPhysicalDescriptorRecovery) {
    constexpr uint32_t placementBatchSize = 7;
    constexpr uint64_t maxValuesPerRow = 4;
    Api::Network network("segmented_reduction_neutral_offsets_driver");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({1})
                                  .maxValuesPerRow(maxValuesPerRow)
                                  .build();
    Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                       .network(network)
                                       .featureInput(input)
                                       .reductionType(Api::SegmentedReduction::Type::MEAN)
                                       .build();
    (void)Api::NetworkOutput::Builder()
        .network(network)
        .name("mean")
        .inputTensor(mean.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(placementBatchSize, initDoneEvents, true);
    ASSERT_NE(placed, nullptr);
    synchronizeEvents(initDoneEvents);

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(input.getRowPartitionId());
    ASSERT_TRUE(requirements.has_value());
    EXPECT_EQ(requirements.value(), Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);

    shared_ptr<Impl::NetworkInput> offsetsInput =
        stamped.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId());
    ASSERT_NE(offsetsInput, nullptr);
    ASSERT_TRUE(offsetsInput->getFeatureOutput().has_value());
    EXPECT_EQ(offsetsInput->getFeatureOutput()->getDimensions(),
              (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

    const Impl::RaggedTensorDescriptor expected = input.getDescriptorForPlacement(placementBatchSize);
    shared_ptr<Impl::TensorFanout> managedOffsetsCarrier;
    for (const shared_ptr<Impl::Layer>& layer : stamped.getOtherLayers()) {
        auto fanout = std::dynamic_pointer_cast<Impl::TensorFanout>(layer);
        if (fanout == nullptr || !fanout->getRowPartitionDescriptorForStamp().has_value() ||
            !fanout->getFeatureOutput().has_value()) {
            continue;
        }
        if (fanout->getRowPartitionDescriptorForStamp().value() != expected.getRowPartition() ||
            fanout->getFeatureOutput()->getDimensions() !=
                (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1})) {
            continue;
        }
        ASSERT_EQ(managedOffsetsCarrier, nullptr) << "Expected exactly one managed [B+1] offsets fanout.";
        managedOffsetsCarrier = fanout;
    }
    ASSERT_NE(managedOffsetsCarrier, nullptr);

    const Impl::RaggedTensorDescriptor recovered =
        Api::SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            input, managedOffsetsCarrier, "SegmentedReduction");
    EXPECT_EQ(recovered, expected);
    EXPECT_EQ(recovered.getBatchSize(), placementBatchSize);
    EXPECT_EQ(recovered.getMaxValuesPerRow(), maxValuesPerRow);
    EXPECT_EQ(recovered.getMaxTotalValues(),
              static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow);
}

TEST(SegmentedReductionApi, PlacementNeutralReductionSaveLoadReplacesPlacementGeometryPerBatch) {
    constexpr uint64_t maxValuesPerRow = 4;
    const std::string networkName = "segmented_reduction_placement_neutral_round_trip";
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archive = std::filesystem::temp_directory_path() /
                                          ("thor_segmented_reduction_neutral_" + std::to_string(nonce));
    std::filesystem::remove_all(archive);

    uint64_t savedInputValuesOriginalId = 0;
    uint64_t savedPartitionTokenOriginalId = 0;
    uint64_t savedReductionOutputOriginalId = 0;
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
        Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                           .network(network)
                                           .featureInput(input)
                                           .reductionType(Api::SegmentedReduction::Type::MEAN)
                                           .build();
        (void)Api::NetworkOutput::Builder()
            .network(network)
            .name("mean")
            .inputTensor(mean.getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();

        savedInputValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionTokenOriginalId = input.getRowPartitionToken().getOriginalId();
        savedReductionOutputOriginalId = mean.getFeatureOutput()->getOriginalId();

        const nlohmann::json architecture = network.architectureJson();
        const nlohmann::json* reductionJson = nullptr;
        for (const nlohmann::json& layerJson : architecture.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_reduction") {
                reductionJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(reductionJson, nullptr);
        EXPECT_EQ(reductionJson->at("version").get<std::string>(), "1.1.0");
        EXPECT_EQ(reductionJson->at("reduction_type").get<std::string>(), "mean");

        const nlohmann::json& inputJson = reductionJson->at("ragged_feature_input");
        EXPECT_EQ(inputJson.at("layer_ragged_reference_version").get<std::string>(), "1.0.0");
        EXPECT_FALSE(inputJson.contains("geometry_mode"));
        EXPECT_EQ(inputJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
        EXPECT_FALSE(inputJson.contains("batch_size"));
        EXPECT_FALSE(inputJson.contains("max_total_values"));
        EXPECT_EQ(inputJson.at("values").at("dimensions").get<vector<uint64_t>>(),
                  (vector<uint64_t>{maxValuesPerRow, 1}));
        EXPECT_EQ(inputJson.at("offsets").at("dimensions").get<vector<uint64_t>>(),
                  (vector<uint64_t>{1}));
        EXPECT_EQ(inputJson.at("values").at("id").get<uint64_t>(), input.getValues().getId());
        EXPECT_EQ(inputJson.at("offsets").at("id").get<uint64_t>(), input.getRowPartitionToken().getId());

        const nlohmann::json& outputJson = reductionJson->at("feature_output");
        EXPECT_EQ(outputJson.at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
        EXPECT_EQ(outputJson.at("id").get<uint64_t>(), mean.getFeatureOutput()->getId());

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
        EXPECT_EQ(loadedInput.getMaxValuesPerRow(), maxValuesPerRow);

        shared_ptr<Api::SegmentedReduction> loadedReduction;
        for (uint32_t layerIndex = 0; layerIndex < loaded.getNumLayers(); ++layerIndex) {
            loadedReduction = std::dynamic_pointer_cast<Api::SegmentedReduction>(loaded.getLayer(layerIndex));
            if (loadedReduction != nullptr) break;
        }
        ASSERT_NE(loadedReduction, nullptr);
        EXPECT_EQ(loadedReduction->getReductionType(), Api::SegmentedReduction::Type::MEAN);
        ASSERT_TRUE(loadedReduction->getFeatureOutput().has_value());
        EXPECT_EQ(loadedReduction->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(loadedReduction->getFeatureOutput()->getOriginalId(), savedReductionOutputOriginalId);

        const nlohmann::json architectureBeforePlacement = loaded.architectureJson();
        const nlohmann::json* loadedReductionJson = nullptr;
        for (const nlohmann::json& layerJson : architectureBeforePlacement.at("layers")) {
            if (layerJson.at("layer_type").get<std::string>() == "segmented_reduction") {
                loadedReductionJson = &layerJson;
                break;
            }
        }
        ASSERT_NE(loadedReductionJson, nullptr);
        EXPECT_EQ(loadedReductionJson->at("version").get<std::string>(), "1.1.0");
        EXPECT_EQ(loadedReductionJson->at("ragged_feature_input").at("values").at("id").get<uint64_t>(),
                  loadedInput.getValues().getId());
        EXPECT_EQ(loadedReductionJson->at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
                  loadedInput.getRowPartitionToken().getId());
        EXPECT_FALSE(loadedReductionJson->at("ragged_feature_input").contains("batch_size"));
        EXPECT_FALSE(loadedReductionJson->at("ragged_feature_input").contains("max_total_values"));

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        synchronizeEvents(initDoneEvents);

        Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        shared_ptr<Impl::NetworkInput> physicalValues = stamped.getNamedInput("tokens.values");
        ASSERT_NE(physicalValues, nullptr);
        ASSERT_TRUE(physicalValues->getFeatureOutput().has_value());
        EXPECT_EQ(physicalValues->getFeatureOutput()->getDimensions(), (vector<uint64_t>{expectedCapacity, 1}));

        shared_ptr<Impl::NetworkInput> physicalOffsets =
            stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(physicalOffsets, nullptr);
        ASSERT_TRUE(physicalOffsets->getFeatureOutput().has_value());
        EXPECT_EQ(physicalOffsets->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));

        auto physicalReduction = std::dynamic_pointer_cast<Impl::CustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(loadedReduction->getId()));
        ASSERT_NE(physicalReduction, nullptr);
        ASSERT_TRUE(physicalReduction->getFeatureOutput().has_value());
        EXPECT_EQ(physicalReduction->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{placementBatchSize, 1}));

        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const Batch batch = makePlacementNeutralReductionBatch("tokens", loadedInput, placementBatchSize);
        const std::map<std::string, Impl::Tensor> outputs = placed->infer(batch);
        ASSERT_TRUE(outputs.contains("mean"));
        const Impl::Tensor& output = outputs.at("mean");
        ASSERT_EQ(output.getDimensions(), (vector<uint64_t>{placementBatchSize, 1}));
        const float* outputValues = output.getMemPtr<float>();
        for (uint32_t row = 0; row < placementBatchSize; ++row) {
            EXPECT_NEAR(outputValues[row], static_cast<float>(row * 10 + 2), 1.0e-5F) << "row " << row;
        }
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

TEST(SegmentedReductionApi, HistoricalVersionIgnoresObsoletePhysicalBatchMetadata) {
    Api::Network network("segmented_reduction_legacy_logical_reconstruction");
    constexpr uint64_t m = 4;
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2})
                                  .maxValuesPerRow(m)
                                  .build();

    // A genuine historical reference grammar can carry physical leading
    // extents. Its source identities, trailing dimensions and dtypes remain
    // meaningful, but the archived B and total capacity do not.
    nlohmann::json legacyValues = input.getValues().architectureJson();
    nlohmann::json legacyOffsets = input.getRowPartitionToken().architectureJson();
    legacyValues["dimensions"] = std::vector<uint64_t>{15, 2};
    legacyOffsets["dimensions"] = std::vector<uint64_t>{4};
    nlohmann::json legacyReference = {
        {"ragged_rank", 1},
        {"values", legacyValues},
        {"offsets", legacyOffsets},
        {"max_values_per_row", m},
        {"batch_size", 3},
        {"max_total_values", 15}
    };
    Api::Tensor output(DataType::FP32, {2});
    nlohmann::json record = {
        {"factory", "layer"},
        {"version", "1.0.0"},
        {"layer_type", "segmented_reduction"},
        {"reduction_type", "sum"},
        {"ragged_feature_input", legacyReference},
        {"feature_output", output.architectureJson()}
    };

    ASSERT_NO_THROW(Api::SegmentedReduction::deserialize(record, &network));
    std::shared_ptr<Api::SegmentedReduction> restored;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        restored = std::dynamic_pointer_cast<Api::SegmentedReduction>(network.getLayer(i));
        if (restored) break;
    }
    ASSERT_NE(restored, nullptr);
    const Api::RaggedTensor logical = restored->getRaggedFeatureInput();
    EXPECT_EQ(logical.getMaxValuesPerRow(), m);
    EXPECT_EQ(logical.getValues().getOriginalId(), input.getValues().getOriginalId());
    EXPECT_EQ(logical.getRowPartitionToken().getOriginalId(), input.getRowPartitionToken().getOriginalId());
    EXPECT_EQ(logical.getDescriptorForPlacement(2).getMaxTotalValues(), 2 * m);
    EXPECT_EQ(logical.getDescriptorForPlacement(7).getMaxTotalValues(), 7 * m);
    EXPECT_FALSE(restored->architectureJson().at("ragged_feature_input").contains("batch_size"));
    EXPECT_FALSE(restored->architectureJson().at("ragged_feature_input").contains("max_total_values"));

    // Obsolete metadata must not override semantic identities or M. The
    // reader still rejects genuine semantic corruption.
    auto assertRejected = [&](const nlohmann::json& invalid) {
        EXPECT_ANY_THROW(Api::SegmentedReduction::deserialize(invalid, &network));
    };
    nlohmann::json invalidM = record;
    invalidM["ragged_feature_input"]["max_values_per_row"] = m + 1;
    assertRejected(invalidM);
    nlohmann::json invalidDtype = record;
    invalidDtype["ragged_feature_input"]["values"]["data_type"] = "fp16";
    assertRejected(invalidDtype);
    nlohmann::json invalidPartition = record;
    invalidPartition["ragged_feature_input"]["offsets"]["id"] = 99999999ULL;
    assertRejected(invalidPartition);
}

TEST(SegmentedReductionApi, RaggedSliceFeedsBothValuesAndOffsetsIntoSegmentedReduction) {
    constexpr uint32_t batchSize = 3;
    Api::Network network("ragged_slice_segmented_reduction_wiring");
    Api::RaggedTensor history = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("history")
                                    .valuesDataType(DataType::FP32)
                                    .offsetsDataType(DataType::UINT32)
                                    .trailingDimensions({3})
                                    .maxValuesPerRow(3)
                                    .build();
    Api::Slice channel0 = Api::Slice::Builder()
                              .network(network)
                              .featureInput(history)
                              .axis(0)
                              .start(0)
                              .length(1)
                              .build();
    ASSERT_TRUE(channel0.getRaggedFeatureOutput().has_value());
    const vector<Api::Tensor> sliceInputs = channel0.getAllInputTensors();
    ASSERT_EQ(sliceInputs.size(), 2U);
    EXPECT_EQ(sliceInputs[0], history.getValues());
    EXPECT_EQ(sliceInputs[1], history.getOffsets());

    Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                       .network(network)
                                       .featureInput(channel0.getRaggedFeatureOutput().value())
                                       .reductionType(Api::SegmentedReduction::Type::MEAN)
                                       .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("mean")
                                    .inputTensor(mean.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalSlice = std::dynamic_pointer_cast<Impl::CustomLayer>(
        stamped.getPhysicalLayerFromApiLayer(channel0.getId()));
    auto physicalMean = std::dynamic_pointer_cast<Impl::CustomLayer>(
        stamped.getPhysicalLayerFromApiLayer(mean.getId()));
    ASSERT_NE(physicalSlice, nullptr);
    ASSERT_NE(physicalMean, nullptr);
}

TEST(SegmentedReductionApi, SingleReductionPlacesForInferenceWithoutFanout) {
    constexpr uint32_t batchSize = 3;
    Api::Network network("segmented_reduction_single_inference");
    Api::RaggedTensor history = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("history")
                                    .valuesDataType(DataType::FP32)
                                    .offsetsDataType(DataType::UINT32)
                                    .trailingDimensions({2})
                                    .maxValuesPerRow(3)
                                    .build();
    Api::SegmentedReduction mean = Api::SegmentedReduction::Builder()
                                       .network(network)
                                       .featureInput(history)
                                       .reductionType(Api::SegmentedReduction::Type::MEAN)
                                       .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("mean")
                                    .inputTensor(mean.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
}

TEST(SegmentedReductionApi, SharedOffsetsFanoutFeedsPortOneOfIndependentReductions) {
    constexpr uint32_t batchSize = 3;
    Api::Network network("segmented_reduction_shared_offsets_fanout");
    Api::RaggedTensor history = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("history")
                                    .valuesDataType(DataType::FP32)
                                    .offsetsDataType(DataType::UINT32)
                                    .trailingDimensions({2})
                                    .maxValuesPerRow(3)
                                    .build();

    Api::RaggedTensor alternate = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("alternate_values")
                                      .valuesDataType(DataType::FP32)
                                      .trailingDimensions({2})
                                      .partition(history)
                                      .build();
    EXPECT_TRUE(alternate.sharesPartitionWith(history));

    Api::SegmentedReduction historyMean = Api::SegmentedReduction::Builder()
                                              .network(network)
                                              .featureInput(history)
                                              .reductionType(Api::SegmentedReduction::Type::MEAN)
                                              .build();
    Api::SegmentedReduction alternateMean = Api::SegmentedReduction::Builder()
                                                .network(network)
                                                .featureInput(alternate)
                                                .reductionType(Api::SegmentedReduction::Type::MEAN)
                                                .build();
    Api::NetworkOutput historyOutput = Api::NetworkOutput::Builder()
                                           .network(network)
                                           .name("history_mean")
                                           .inputTensor(historyMean.getFeatureOutput().value())
                                           .dataType(DataType::FP32)
                                           .build();
    Api::NetworkOutput alternateOutput = Api::NetworkOutput::Builder()
                                             .network(network)
                                             .name("alternate_mean")
                                             .inputTensor(alternateMean.getFeatureOutput().value())
                                             .dataType(DataType::FP32)
                                             .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
}

TEST(SegmentedReductionApi, SumForwardAndBackwardUseOnlyEachLogicalRow) {
    runForwardBackwardCase(Api::SegmentedReduction::Type::SUM);
}

TEST(SegmentedReductionApi, MeanForwardAndBackwardUseOnlyEachLogicalRow) {
    runForwardBackwardCase(Api::SegmentedReduction::Type::MEAN);
}

TEST(SegmentedReductionApi, MinForwardAndBackwardUseOnlyEachLogicalRow) {
    runForwardBackwardCase(Api::SegmentedReduction::Type::MIN);
}

TEST(SegmentedReductionApi, MaxForwardAndBackwardUseOnlyEachLogicalRow) {
    runForwardBackwardCase(Api::SegmentedReduction::Type::MAX);
}
