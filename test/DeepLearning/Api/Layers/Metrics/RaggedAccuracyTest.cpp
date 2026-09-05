#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Api/Layers/Metrics/CategoricalAccuracy.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

namespace {

void writeOffsets(ThorImplementation::Tensor& offsets, DataType dtype, const vector<uint64_t>& values) {
    if (dtype == DataType::UINT32) {
        uint32_t* dst = offsets.getMemPtr<uint32_t>();
        for (size_t i = 0; i < values.size(); ++i) dst[i] = static_cast<uint32_t>(values[i]);
        return;
    }
    THOR_THROW_IF_FALSE(dtype == DataType::UINT64);
    copy(values.begin(), values.end(), offsets.getMemPtr<uint64_t>());
}

float copyScalarToHost(const ThorImplementation::Tensor& tensor) {
    ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor host = tensor.clone(cpu);
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    return *host.getMemPtr<float>();
}

RaggedTensor buildPredictions(Network& network,
                              const string& name,
                              DataType offsetsDType,
                              vector<uint64_t> trailing) {
    return RaggedNetworkInput::Builder()
        .network(network)
        .name(name)
        .valuesDataType(DataType::FP32)
        .offsetsDataType(offsetsDType)
        .trailingDimensions(std::move(trailing))
        .batchSize(4)
        .maxTotalValues(9)
        .maxValuesPerRow(4)
        .build();
}

}  // namespace

TEST(RaggedAccuracyApi, R10OBinaryBuildsAsTokenRatioAndSerializesPartition) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r10o_binary_build_" + ThorImplementation::TensorDescriptor::getElementTypeName(offsetsDType));
        RaggedTensor predictions = buildPredictions(network, "predictions", offsetsDType, {1});
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::UINT8)
                                  .trailingDimensions({1})
                                  .partition(predictions)
                                  .build();
        BinaryAccuracy metric = BinaryAccuracy::Builder().network(network).predictions(predictions).labels(labels).build();

        EXPECT_TRUE(metric.getUseRagged());
        EXPECT_EQ(metric.getAggregation(), MetricAggregation::RATIO);
        ASSERT_EQ(metric.getAllInputTensors().size(), 3U);
        EXPECT_EQ(metric.getConnectionType(predictions.getOffsets()),
                  static_cast<int>(ThorImplementation::Metric::ConnectionType::STRUCTURAL));
        ASSERT_TRUE(metric.getRaggedPredictions().has_value());
        ASSERT_TRUE(metric.getRaggedLabels().has_value());
        EXPECT_EQ(metric.getRaggedPredictions()->getOffsets(), metric.getRaggedLabels()->getOffsets());

        const json j = metric.architectureJson();
        EXPECT_EQ(j.at("aggregation").get<MetricAggregation>(), MetricAggregation::RATIO);
        EXPECT_TRUE(j.contains("ragged_predictions"));
        EXPECT_TRUE(j.contains("ragged_labels"));
        EXPECT_FALSE(j.contains("predictions"));
        EXPECT_FALSE(j.contains("labels"));

        shared_ptr<Layer> cloneLayer = metric.clone();
        BinaryAccuracy* clone = dynamic_cast<BinaryAccuracy*>(cloneLayer.get());
        ASSERT_NE(clone, nullptr);
        ASSERT_TRUE(clone->getRaggedPredictions().has_value());
        ASSERT_TRUE(clone->getRaggedLabels().has_value());
        EXPECT_EQ(clone->getRaggedPredictions()->getOffsets(), clone->getRaggedLabels()->getOffsets());
    }
}

TEST(RaggedAccuracyApi, R10OCategoricalBuildsIndexAndPerClassContracts) {
    for (bool indexLabels : {false, true}) {
        Network network(indexLabels ? "r10o_cat_index_build" : "r10o_cat_per_class_build");
        RaggedTensor predictions = buildPredictions(network, "predictions", DataType::UINT32, {3});
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(indexLabels ? DataType::UINT32 : DataType::FP32)
                                  .trailingDimensions(indexLabels ? vector<uint64_t>{1} : vector<uint64_t>{3})
                                  .partition(predictions)
                                  .build();
        CategoricalAccuracy::Builder builder =
            CategoricalAccuracy::Builder().network(network).predictions(predictions).labels(labels);
        if (indexLabels)
            builder.receivesClassIndexLabels(3);
        else
            builder.receivesOneHotLabels();
        CategoricalAccuracy metric = builder.build();

        EXPECT_TRUE(metric.getUseRagged());
        EXPECT_EQ(metric.getAggregation(), MetricAggregation::RATIO);
        EXPECT_EQ(metric.getConnectionType(predictions.getOffsets()),
                  static_cast<int>(ThorImplementation::Metric::ConnectionType::STRUCTURAL));
        const json j = metric.architectureJson();
        EXPECT_EQ(j.at("aggregation").get<MetricAggregation>(), MetricAggregation::RATIO);
        EXPECT_EQ(j.at("label_type").get<CategoricalAccuracy::LabelType>(),
                  indexLabels ? CategoricalAccuracy::LabelType::INDEX : CategoricalAccuracy::LabelType::ONE_HOT);
        EXPECT_TRUE(j.contains("ragged_predictions"));
        EXPECT_TRUE(j.contains("ragged_labels"));
    }
}

TEST(RaggedAccuracyApi, R10ORejectsDifferentPartitionsAndConfusedClassAxis) {
    Network network("r10o_reject");
    RaggedTensor binaryPredictions = buildPredictions(network, "binary_predictions", DataType::UINT32, {1});
    RaggedTensor separateBinaryLabels = RaggedNetworkInput::Builder()
                                            .network(network)
                                            .name("separate_binary_labels")
                                            .valuesDataType(DataType::UINT8)
                                            .trailingDimensions({1})
                                            .batchSize(4)
                                            .maxTotalValues(9)
                                            .maxValuesPerRow(4)
                                            .build();
    EXPECT_THROW(BinaryAccuracy::Builder()
                     .network(network)
                     .predictions(binaryPredictions)
                     .labels(separateBinaryLabels)
                     .build(),
                 std::invalid_argument);

    RaggedTensor categoricalPredictions = buildPredictions(network, "cat_predictions", DataType::UINT32, {3});
    RaggedTensor wrongIndexLabels = RaggedNetworkInput::Builder()
                                        .network(network)
                                        .name("wrong_index_labels")
                                        .valuesDataType(DataType::UINT32)
                                        .trailingDimensions({3})
                                        .partition(categoricalPredictions)
                                        .build();
    EXPECT_THROW(CategoricalAccuracy::Builder()
                     .network(network)
                     .predictions(categoricalPredictions)
                     .labels(wrongIndexLabels)
                     .receivesClassIndexLabels(3)
                     .build(),
                 std::invalid_argument);
}

TEST(RaggedAccuracyApi, R10OBinaryPartialTailUsesOffsetsAtValidLogicalRowCount) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "R10O ragged BinaryAccuracy execution requires a GPU";

    constexpr uint32_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 9;
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r10o_binary_partial_" + ThorImplementation::TensorDescriptor::getElementTypeName(offsetsDType));
        RaggedTensor predictions = buildPredictions(network, "predictions", offsetsDType, {1});
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::UINT8)
                                  .trailingDimensions({1})
                                  .partition(predictions)
                                  .build();
        BinaryAccuracy metric = BinaryAccuracy::Builder().network(network).predictions(predictions).labels(labels).build();
        NetworkOutput::Builder().network(network).name("accuracy").inputTensor(metric.getMetric()).dataType(DataType::FP32).build();

        vector<Event> initializationDone;
        shared_ptr<PlacedNetwork> placed = network.place(
            batchSize, initializationDone, /*inferenceOnly=*/true, vector<int32_t>{0}, /*forcedNumStampsPerGpu=*/1);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initializationDone) event.synchronize();
        placed->preallocateInputSlots(1);
        placed->preallocateOutputSlots(1);
        placed->synchronize();

        ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::Tensor packedPredictions(
            cpu, ThorImplementation::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
        ThorImplementation::Tensor packedLabels(
            cpu, ThorImplementation::TensorDescriptor(DataType::UINT8, {maxTotalValues, 1}));
        float* predictionData = packedPredictions.getMemPtr<float>();
        uint8_t* labelData = packedLabels.getMemPtr<uint8_t>();
        fill(predictionData, predictionData + maxTotalValues, numeric_limits<float>::quiet_NaN());
        fill(labelData, labelData + maxTotalValues, static_cast<uint8_t>(255));
        // Valid rows 0-1 stop at offsets[2] == 2: one correct, one wrong.
        predictionData[0] = 0.9f; labelData[0] = 1;
        predictionData[1] = 0.1f; labelData[1] = 1;
        // Populated invalid rows would make the answer very different if read.
        predictionData[2] = 0.9f; labelData[2] = 1;
        predictionData[3] = 0.9f; labelData[3] = 1;
        predictionData[4] = 0.9f; labelData[4] = 1;
        predictionData[5] = 0.9f; labelData[5] = 1;
        predictionData[6] = 0.9f; labelData[6] = 1;

        ThorImplementation::Tensor offsets(
            cpu, ThorImplementation::TensorDescriptor(offsetsDType, {batchSize + 1}));
        writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 7});

        Batch batch;
        batch.insert("predictions", ThorImplementation::RaggedTensor(packedPredictions, offsets, 4));
        batch.insert("labels", packedLabels);
        batch.setValidExampleCount(2);

        map<string, ThorImplementation::Tensor> outputs;
        map<string, Event> outputReadyEvents;
        Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/true);
        done.synchronize();
        outputReadyEvents.at("accuracy").synchronize();
        EXPECT_NEAR(copyScalarToHost(outputs.at("accuracy")), 0.5f, 1.0e-6f);

        map<string, ThorImplementation::MetricBatchStatisticTensors> statistics =
            placed->getMetricBatchStatisticTensorsForSlot(0, 0);
        ASSERT_TRUE(statistics.count("accuracy"));
        auto& stat = statistics.at("accuracy");
        EXPECT_EQ(stat.aggregation, MetricAggregation::RATIO);
        ASSERT_TRUE(stat.numerator.has_value());
        ASSERT_TRUE(stat.denominator.has_value());
        stat.readyEvent.synchronize();
        EXPECT_FLOAT_EQ(*stat.numerator->getMemPtr<float>(), 1.0f);
        EXPECT_FLOAT_EQ(*stat.denominator->getMemPtr<float>(), 2.0f);
        placed->synchronize();
    }
}
