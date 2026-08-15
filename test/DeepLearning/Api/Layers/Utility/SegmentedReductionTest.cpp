#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedReduction.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <memory>
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

void writeCpuU32(Impl::Tensor& tensor, const vector<uint32_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT32);
    ASSERT_EQ(numel(tensor), values.size());
    auto* ptr = static_cast<uint32_t*>(tensor.getMemPtr());
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

struct ReductionFixture {
    shared_ptr<Api::Network> network;
    Api::SegmentedReduction reduction;
    Api::NetworkOutput output;
    shared_ptr<Api::NetworkInput> valuesInput;
    shared_ptr<Api::NetworkInput> offsetsInput;
};

ReductionFixture makeReductionFixture(Api::SegmentedReduction::Type type) {
    auto network = std::make_shared<Api::Network>("segmented_reduction_api");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(*network)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2})
                                  .maxTotalValues(9)
                                  .batchSize(3)
                                  .build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(*network).tensor(input.getValues()).build();
    Api::RaggedTensor rivetedInput(inputRivet.getFeatureOutput().value(), input.getOffsets());
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
    return {network, reduction, output, findInput(*network, "history.values"), findInput(*network, "history.offsets")};
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
    ASSERT_NE(fixture.offsetsInput, nullptr);
    EXPECT_EQ(fixture.reduction.getFeatureOutput()->getDimensions(), (vector<uint64_t>{2}));

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = fixture.network->place(batchSize, initDoneEvents, false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalValues = std::dynamic_pointer_cast<Impl::NetworkInput>(
        stamped.getPhysicalLayerFromApiLayer(fixture.valuesInput->getId()));
    auto physicalOffsets = std::dynamic_pointer_cast<Impl::NetworkInput>(
        stamped.getPhysicalLayerFromApiLayer(fixture.offsetsInput->getId()));
    auto physicalReduction = std::dynamic_pointer_cast<Impl::CustomLayer>(
        stamped.getPhysicalLayerFromApiLayer(fixture.reduction.getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(
        stamped.getPhysicalLayerFromApiLayer(fixture.output.getId()));
    ASSERT_NE(physicalValues, nullptr);
    ASSERT_NE(physicalOffsets, nullptr);
    ASSERT_NE(physicalReduction, nullptr);
    ASSERT_NE(physicalOutput, nullptr);

    // offsets = [0,3,3,7]: unequal lengths and an empty middle row. The final two
    // packed rows are deliberately extreme poison and must never participate.
    const vector<float> values{
        1.0F, 10.0F, 3.0F, 8.0F, 2.0F, 12.0F,
        5.0F, 9.0F, 4.0F, 7.0F, 8.0F, 6.0F, 6.0F, 11.0F,
        -1000.0F, 1000.0F, 1000.0F, -1000.0F};
    const vector<uint32_t> offsets{0U, 3U, 3U, 7U};
    Impl::Tensor valuesHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {9, 2}));
    Impl::Tensor offsetsHost(cpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {4}));
    writeCpuFloat(valuesHost, values);
    writeCpuU32(offsetsHost, offsets);
    physicalValues->forward(valuesHost, false, batchSize);
    physicalOffsets->forward(offsetsHost, false, batchSize);
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
                                  .maxTotalValues(9)
                                  .batchSize(3)
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
                                        .maxTotalValues(9)
                                        .batchSize(3)
                                        .build();
    Api::SegmentedReduction scalarMean = Api::SegmentedReduction::Builder()
                                             .network(scalarNetwork)
                                             .featureInput(scalarInput)
                                             .reductionType(Api::SegmentedReduction::Type::MEAN)
                                             .build();
    EXPECT_EQ(scalarMean.getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
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
                                    .maxTotalValues(9)
                                    .batchSize(batchSize)
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
                                    .maxTotalValues(9)
                                    .batchSize(batchSize)
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
                                    .maxTotalValues(9)
                                    .batchSize(batchSize)
                                    .build();

    Api::NetworkInput alternateValues = Api::NetworkInput::Builder()
                                            .network(network)
                                            .name("alternate_values")
                                            .dimensions({9, 2})
                                            .dataType(DataType::FP32)
                                            .dimensionsIncludeBatch(true)
                                            .build();
    Api::RaggedTensor alternate(alternateValues.getFeatureOutput().value(), history.getOffsets());

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
