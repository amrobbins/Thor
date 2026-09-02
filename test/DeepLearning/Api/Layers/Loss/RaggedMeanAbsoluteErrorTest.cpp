#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
namespace Impl = ThorImplementation;

namespace {

struct Inputs {
    RaggedTensor predictions;
    RaggedTensor labels;
};

Inputs makeSharedPartitionInputs(Network& network,
                                 DataType valuesDType = DataType::FP32,
                                 DataType offsetsDType = DataType::UINT32) {
    RaggedTensor predictions = RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .valuesDataType(valuesDType)
                                   .offsetsDataType(offsetsDType)
                                   .trailingDimensions({2})
                                   .batchSize(4)
                                   .maxTotalValues(11)
                                   .maxValuesPerRow(5)
                                   .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(valuesDType)
                              .trailingDimensions({2})
                              .partition(predictions)
                              .build();
    return {predictions, labels};
}

uint32_t countLayerType(Network& network, const string& type) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        if (network.getLayer(i)->getLayerType() == type) ++count;
    }
    return count;
}

bool cudaAvailable() {
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

void writeOffsets(Impl::Tensor& offsetsTensor, DataType dtype, const vector<uint64_t>& offsets) {
    if (dtype == DataType::UINT32) {
        uint32_t* values = offsetsTensor.getMemPtr<uint32_t>();
        for (size_t i = 0; i < offsets.size(); ++i) values[i] = static_cast<uint32_t>(offsets[i]);
        return;
    }
    THOR_THROW_IF_FALSE(dtype == DataType::UINT64);
    uint64_t* values = offsetsTensor.getMemPtr<uint64_t>();
    copy(offsets.begin(), offsets.end(), values);
}

vector<float> copyFp32ToHost(const Impl::Tensor& tensor) {
    THOR_THROW_IF_FALSE(tensor.getDataType() == DataType::FP32);
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = tensor.clone(cpuPlacement);
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + host.getTotalNumElements());
}

}  // namespace

TEST(RaggedMAEApi, RawPreservesExactPartitionAndCreatesTrainingRoot) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("ragged_mae_raw");
        Inputs inputs = makeSharedPartitionInputs(network, DataType::FP32, offsetsDType);

        MAE loss = MAE::Builder()
                       .network(network)
                       .predictions(inputs.predictions)
                       .labels(inputs.labels)
                       .reportsRawLoss()
                       .build();

        ASSERT_TRUE(loss.isRagged());
        EXPECT_EQ(loss.getRaggedPredictions(), inputs.predictions);
        EXPECT_EQ(loss.getRaggedLabels(), inputs.labels);
        RaggedTensor raw = loss.getRaggedRawLoss();
        EXPECT_EQ(raw.getOffsets(), inputs.predictions.getOffsets());
        EXPECT_EQ(raw.getBatchSize(), 4u);
        EXPECT_EQ(raw.getMaxTotalValues(), 11u);
        EXPECT_EQ(raw.getTrailingDimensions(), (vector<uint64_t>{2}));
        EXPECT_EQ(raw.getValuesDataType(), DataType::FP32);
        EXPECT_EQ(loss.getRaggedLoss(), raw);

        ASSERT_EQ(network.getLossRootTensors().size(), 1u);
        EXPECT_EQ(network.getLossRootTensors()[0], raw.getValues());
        EXPECT_EQ(countLayerType(network, "RaggedCustomLoss"), 1u);
        EXPECT_EQ(countLayerType(network, "RaggedLossShaper"), 0u);
    }
}

TEST(RaggedMAEApi, PerExampleAndBatchReportDenseScalars) {
    for (Loss::LossShape shape : {Loss::LossShape::PER_EXAMPLE, Loss::LossShape::BATCH}) {
        Network network(shape == Loss::LossShape::PER_EXAMPLE ? "ragged_mae_per_example" : "ragged_mae_batch");
        Inputs inputs = makeSharedPartitionInputs(network, DataType::FP16, DataType::UINT64);
        MAE::Builder builder;
        builder.network(network).predictions(inputs.predictions).labels(inputs.labels).lossDataType(DataType::FP32);
        if (shape == Loss::LossShape::PER_EXAMPLE)
            builder.reportsPerExampleLoss();
        else
            builder.reportsBatchLoss();
        MAE loss = builder.build();

        EXPECT_TRUE(loss.isRagged());
        EXPECT_EQ(loss.getRawLoss(), loss.getRaggedRawLoss().getValues());
        EXPECT_EQ(loss.getLoss().getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(loss.getLoss().getDataType(), DataType::FP32);
        EXPECT_THROW((void)loss.getRaggedLoss(), runtime_error);
        EXPECT_EQ(countLayerType(network, "RaggedCustomLoss"), 1u);
        EXPECT_EQ(countLayerType(network, "RaggedLossShaper"), 1u);
    }
}

TEST(RaggedMAEApi, NoneKeepsRawTrainingRootButDoesNotExposeReportedLoss) {
    Network network("ragged_mae_none");
    Inputs inputs = makeSharedPartitionInputs(network);
    MAE loss = MAE::Builder()
                   .network(network)
                   .predictions(inputs.predictions)
                   .labels(inputs.labels)
                   .reportsNoLoss()
                   .build();

    EXPECT_TRUE(loss.isRagged());
    EXPECT_THROW((void)loss.getLoss(), runtime_error);
    ASSERT_EQ(network.getLossRootTensors().size(), 1u);
    EXPECT_EQ(network.getLossRootTensors()[0], loss.getRaggedRawLoss().getValues());
}

TEST(RaggedMAEApi, RejectsDifferentPartitionPerOutputAndInvalidExampleWeights) {
    Network network("ragged_mae_validation");
    RaggedTensor predictions = RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .valuesDataType(DataType::FP32)
                                   .trailingDimensions({2})
                                   .batchSize(3)
                                   .maxTotalValues(8)
                                   .build();
    RaggedTensor labelsDifferentPartition = RaggedNetworkInput::Builder()
                                                .network(network)
                                                .name("labels")
                                                .valuesDataType(DataType::FP32)
                                                .trailingDimensions({2})
                                                .batchSize(3)
                                                .maxTotalValues(8)
                                                .build();
    EXPECT_THROW((void)MAE::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labelsDifferentPartition)
                     .reportsBatchLoss()
                     .build(),
                 invalid_argument);

    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels_shared")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({2})
                              .partition(predictions)
                              .build();
    EXPECT_THROW((void)MAE::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .reportsPerOutputLoss()
                     .build(),
                 invalid_argument);

    NetworkInput weightsInput = NetworkInput::Builder()
                                    .network(network)
                                    .name("weights")
                                    .dimensions({1})
                                    .dataType(DataType::FP32)
                                    .build();
    EXPECT_NO_THROW((void)MAE::Builder()
                        .network(network)
                        .predictions(predictions)
                        .labels(labels)
                        .exampleWeights(weightsInput.getFeatureOutput().value())
                        .reportsBatchLoss()
                        .build());

    Tensor wrongShapeWeights(DataType::FP32, {2});
    EXPECT_THROW((void)MAE::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .exampleWeights(wrongShapeWeights)
                     .reportsBatchLoss()
                     .build(),
                 invalid_argument);

    Tensor wrongDTypeWeights(DataType::UINT32, {1});
    EXPECT_THROW((void)MAE::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .exampleWeights(wrongDTypeWeights)
                     .reportsBatchLoss()
                     .build(),
                 exception);
}


TEST(RaggedMAEApi, MatchesDenseRegressionPredictionDTypeContractAndDefaultLossStorage) {
    const vector<pair<DataType, DataType>> cases{
        {DataType::FP8_E4M3, DataType::FP32},
        {DataType::FP8_E5M2, DataType::FP32},
        {DataType::FP16, DataType::FP16},
        {DataType::BF16, DataType::FP32},
        {DataType::FP32, DataType::FP32},
    };
    for (const auto& [predictionDType, expectedLossDType] : cases) {
        Network network("ragged_mae_prediction_dtype_parity");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(predictionDType)
                                       .trailingDimensions({2})
                                       .batchSize(3)
                                       .maxTotalValues(8)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({2})
                                  .partition(predictions)
                                  .build();
        MAE loss = MAE::Builder().network(network).predictions(predictions).labels(labels).reportsRawLoss().build();
        EXPECT_EQ(loss.getRaggedRawLoss().getValuesDataType(), expectedLossDType);
    }
}

TEST(RaggedMAEApi, MatchesDenseRegressionLabelDTypeContract) {
    const vector<DataType> labelDTypes{DataType::BOOLEAN, DataType::INT8,  DataType::INT16, DataType::INT32, DataType::INT64,
                                       DataType::UINT8,   DataType::UINT16, DataType::UINT32, DataType::UINT64,
                                       DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32};
    for (DataType labelDType : labelDTypes) {
        Network network("ragged_mae_label_dtype_parity");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::BF16)
                                       .trailingDimensions({2})
                                       .batchSize(3)
                                       .maxTotalValues(8)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(labelDType)
                                  .trailingDimensions({2})
                                  .partition(predictions)
                                  .build();
        EXPECT_NO_THROW((void)MAE::Builder()
                            .network(network)
                            .predictions(predictions)
                            .labels(labels)
                            .lossDataType(DataType::FP32)
                            .reportsRawLoss()
                            .build());
    }
}

TEST(RaggedMAEApi, MatchesDenseRegressionExampleWeightDTypeContractForPerRowWeights) {
    for (DataType weightDType :
         {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32}) {
        Network network("ragged_mae_weight_dtype_parity");
        Inputs inputs = makeSharedPartitionInputs(network);
        NetworkInput weightsInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("weights")
                                        .dimensions({1})
                                        .dataType(weightDType)
                                        .build();
        MAE loss = MAE::Builder()
                       .network(network)
                       .predictions(inputs.predictions)
                       .labels(inputs.labels)
                       .exampleWeights(weightsInput.getFeatureOutput().value())
                       .reportsRawLoss()
                       .build();
        ASSERT_TRUE(loss.getExampleWeights().has_value());
        EXPECT_EQ(loss.getExampleWeights().value(), weightsInput.getFeatureOutput().value());
        EXPECT_EQ(countLayerType(network, "TypeConverter"), 1u);
        EXPECT_EQ(countLayerType(network, "SegmentedBroadcast"), 1u);
        EXPECT_EQ(countLayerType(network, "RaggedCustomLoss"), 1u);
    }
}

TEST(RaggedMAEApi, PerRowExampleWeightsPlaceForScalarAndMultiAxisTrailingValues) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for ragged MAE weighted placement coverage.";

    for (const vector<uint64_t>& trailingDimensions : {vector<uint64_t>{}, vector<uint64_t>{2, 3}}) {
        Network network(trailingDimensions.empty() ? "ragged_mae_weighted_scalar" : "ragged_mae_weighted_multi_axis");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP32)
                                       .offsetsDataType(DataType::UINT32)
                                       .trailingDimensions(trailingDimensions)
                                       .batchSize(3)
                                       .maxTotalValues(7)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions(trailingDimensions)
                                  .partition(predictions)
                                  .build();
        NetworkInput weightsInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("weights")
                                        .dimensions({1})
                                        .dataType(DataType::FP32)
                                        .build();
        MAE loss = MAE::Builder()
                       .network(network)
                       .predictions(predictions)
                       .labels(labels)
                       .exampleWeights(weightsInput.getFeatureOutput().value())
                       .reportsBatchLoss()
                       .build();
        (void)NetworkOutput::Builder()
            .network(network)
            .name("loss")
            .inputTensor(loss.getLoss())
            .dataType(DataType::FP32)
            .build();

        vector<Event> initializationDone;
        shared_ptr<PlacedNetwork> placed = network.place(3, initializationDone, /*inferenceOnly=*/false);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initializationDone) event.synchronize();
        placed->synchronize();
    }
}

TEST(RaggedMAEApi, RejectsTheSameUnsupportedRegressionDTypesAsDenseMAE) {
    {
        Network network("ragged_mae_bad_prediction_dtype");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP64)
                                       .trailingDimensions({2})
                                       .batchSize(3)
                                       .maxTotalValues(8)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({2})
                                  .partition(predictions)
                                  .build();
        EXPECT_THROW((void)MAE::Builder().network(network).predictions(predictions).labels(labels).build(), exception);
    }
    {
        Network network("ragged_mae_bad_label_dtype");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP32)
                                       .trailingDimensions({2})
                                       .batchSize(3)
                                       .maxTotalValues(8)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP64)
                                  .trailingDimensions({2})
                                  .partition(predictions)
                                  .build();
        EXPECT_THROW((void)MAE::Builder().network(network).predictions(predictions).labels(labels).build(), exception);
    }
    {
        Network network("ragged_mae_bad_loss_dtype");
        Inputs inputs = makeSharedPartitionInputs(network);
        EXPECT_THROW((void)MAE::Builder()
                         .network(network)
                         .predictions(inputs.predictions)
                         .labels(inputs.labels)
                         .lossDataType(DataType::BF16)
                         .build(),
                     exception);
    }
}


TEST(RaggedMAEApi, PlacedBatchReportingAndBackwardUseLogicalRowsAndActivePrefix) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for ragged MAE placement/runtime coverage.";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t trailingWidth = 2;
    constexpr float lossWeight = 0.5f;

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network(offsetsDType == DataType::UINT32 ? "ragged_mae_runtime_u32" : "ragged_mae_runtime_u64");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP32)
                                       .offsetsDataType(offsetsDType)
                                       .trailingDimensions({trailingWidth})
                                       .batchSize(batchSize)
                                       .maxTotalValues(maxTotalValues)
                                       .maxValuesPerRow(3)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({trailingWidth})
                                  .partition(predictions)
                                  .build();
        MAE loss = MAE::Builder()
                       .network(network)
                       .predictions(predictions)
                       .labels(labels)
                       .lossWeight(lossWeight)
                       .reportsBatchLoss()
                       .build();
        (void)NetworkOutput::Builder()
            .network(network)
            .name("loss")
            .inputTensor(loss.getLoss())
            .dataType(DataType::FP32)
            .build();

        vector<Event> initializationDone;
        shared_ptr<PlacedNetwork> placed = network.place(batchSize, initializationDone, /*inferenceOnly=*/false);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initializationDone) event.synchronize();

        shared_ptr<Impl::RaggedCustomLoss> physicalLoss;
        for (const shared_ptr<Impl::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
            auto candidate = dynamic_pointer_cast<Impl::RaggedCustomLoss>(layer);
            if (candidate == nullptr) continue;
            ASSERT_EQ(physicalLoss, nullptr);
            physicalLoss = std::move(candidate);
        }
        ASSERT_NE(physicalLoss, nullptr);

        const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor predictionValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, trailingWidth}));
        Impl::Tensor labelValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, trailingWidth}));
        Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));

        float* p = predictionValues.getMemPtr<float>();
        float* y = labelValues.getMemPtr<float>();
        fill(p, p + maxTotalValues * trailingWidth, numeric_limits<float>::quiet_NaN());
        fill(y, y + maxTotalValues * trailingWidth, numeric_limits<float>::quiet_NaN());
        const vector<float> activePredictions{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f};
        copy(activePredictions.begin(), activePredictions.end(), p);
        fill(y, y + activePredictions.size(), 0.0f);
        writeOffsets(offsets, offsetsDType, {0, 2, 2, 5});

        Batch batch;
        batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, /*maxValuesPerRow=*/3));
        // labels is a shared-partition logical input, so only its packed values are submitted.
        batch.insert("labels", labelValues);

        map<string, Impl::Tensor> outputs;
        map<string, Event> outputReadyEvents;
        Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
        done.synchronize();
        outputReadyEvents.at("loss").synchronize();
        placed->synchronize();

        const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
        ASSERT_EQ(reported.size(), 1u);
        // Row sums are 10, 0, and 45. Global loss_weight=0.5 is applied to
        // the raw objective, then BATCH averages row sums across all 3 valid rows.
        EXPECT_NEAR(reported[0], (55.0f * lossWeight) / 3.0f, 1.0e-5f);

        ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
        const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
        ASSERT_EQ(gradient.size(), maxTotalValues * trailingWidth);
        const float magnitude = Impl::Loss::getLossScalingFactor() * lossWeight;
        for (size_t i = 0; i < activePredictions.size(); ++i) {
            EXPECT_FLOAT_EQ(gradient[i], activePredictions[i] > 0.0f ? magnitude : -magnitude) << "index=" << i;
        }
        // Deliberately do not inspect the inactive gradient tail. R10A's contract
        // leaves packed capacity beyond offsets[B] undefined rather than zeroing it.
    }
}

TEST(RaggedMAEApi, PerRowExampleWeightsScaleActiveLossAndGradientAndPreserveLogicalRowNormalization) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for ragged MAE weighted runtime coverage.";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t trailingWidth = 2;

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network(offsetsDType == DataType::UINT32 ? "ragged_mae_weighted_u32" : "ragged_mae_weighted_u64");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP32)
                                       .offsetsDataType(offsetsDType)
                                       .trailingDimensions({trailingWidth})
                                       .batchSize(batchSize)
                                       .maxTotalValues(maxTotalValues)
                                       .maxValuesPerRow(3)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({trailingWidth})
                                  .partition(predictions)
                                  .build();
        NetworkInput weightsInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("weights")
                                        .dimensions({1})
                                        .dataType(DataType::FP32)
                                        .build();
        MAE loss = MAE::Builder()
                       .network(network)
                       .predictions(predictions)
                       .labels(labels)
                       .exampleWeights(weightsInput.getFeatureOutput().value())
                       .reportsBatchLoss()
                       .build();
        (void)NetworkOutput::Builder()
            .network(network)
            .name("loss")
            .inputTensor(loss.getLoss())
            .dataType(DataType::FP32)
            .build();

        vector<Event> initializationDone;
        shared_ptr<PlacedNetwork> placed = network.place(batchSize, initializationDone, /*inferenceOnly=*/false);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initializationDone) event.synchronize();

        shared_ptr<Impl::RaggedCustomLoss> physicalLoss;
        for (const shared_ptr<Impl::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
            auto candidate = dynamic_pointer_cast<Impl::RaggedCustomLoss>(layer);
            if (candidate == nullptr) continue;
            ASSERT_EQ(physicalLoss, nullptr);
            physicalLoss = std::move(candidate);
        }
        ASSERT_NE(physicalLoss, nullptr);
        ASSERT_TRUE(physicalLoss->getExampleWeightsInput().has_value());

        const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor predictionValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, trailingWidth}));
        Impl::Tensor labelValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, trailingWidth}));
        Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));
        Impl::Tensor weights(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1}));

        float* p = predictionValues.getMemPtr<float>();
        float* y = labelValues.getMemPtr<float>();
        fill(p, p + maxTotalValues * trailingWidth, numeric_limits<float>::quiet_NaN());
        fill(y, y + maxTotalValues * trailingWidth, numeric_limits<float>::quiet_NaN());
        const vector<float> activePredictions{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f};
        copy(activePredictions.begin(), activePredictions.end(), p);
        fill(y, y + activePredictions.size(), 0.0f);
        writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});
        float* w = weights.getMemPtr<float>();
        w[0] = 0.5f;
        w[1] = 7.0f;   // Empty but valid row: contributes zero loss but still counts in the BATCH denominator.
        w[2] = 2.0f;
        w[3] = 100.0f; // Invalid tail row: canonical offsets keep it empty.

        Batch batch;
        batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, /*maxValuesPerRow=*/3));
        batch.insert("labels", labelValues);
        batch.insert("weights", weights);
        batch.setValidExampleCount(validExamples);

        map<string, Impl::Tensor> outputs;
        map<string, Event> outputReadyEvents;
        Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
        done.synchronize();
        outputReadyEvents.at("loss").synchronize();
        placed->synchronize();

        const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
        ASSERT_EQ(reported.size(), 1u);
        // Unweighted row sums are 10, 0, and 45. Apply row weights 0.5, 7, and 2,
        // then divide by the three valid logical rows, not by active token count or weight sum.
        EXPECT_NEAR(reported[0], 95.0f / 3.0f, 1.0e-5f);

        ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
        const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
        ASSERT_EQ(gradient.size(), maxTotalValues * trailingWidth);
        const float scale = Impl::Loss::getLossScalingFactor();
        for (size_t i = 0; i < activePredictions.size(); ++i) {
            const float rowWeight = i < 4 ? 0.5f : 2.0f;
            EXPECT_FLOAT_EQ(gradient[i], (activePredictions[i] > 0.0f ? 1.0f : -1.0f) * rowWeight * scale) << "index=" << i;
        }
    }
}

TEST(RaggedMAEApi, SupportLayersSaveLoadWithCanonicalSharedPartition) {
    const string networkName = "ragged_mae_round_trip";
    Network network(networkName);
    Inputs inputs = makeSharedPartitionInputs(network, DataType::FP32, DataType::UINT64);
    NetworkInput weightsInput = NetworkInput::Builder()
                                    .network(network)
                                    .name("weights")
                                    .dimensions({1})
                                    .dataType(DataType::BF16)
                                    .build();
    MAE loss = MAE::Builder()
                   .network(network)
                   .predictions(inputs.predictions)
                   .labels(inputs.labels)
                   .exampleWeights(weightsInput.getFeatureOutput().value())
                   .lossWeight(0.5f)
                   .reportsPerExampleLoss()
                   .build();
    (void)loss;

    const auto now = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archiveDir = filesystem::temp_directory_path() /
        (string("thor_ragged_mae_round_trip_") + to_string(now));
    filesystem::remove_all(archiveDir);
    network.save(archiveDir.string(), /*overwrite=*/true);

    Network loaded(networkName);
    ASSERT_NO_THROW(loaded.load(archiveDir.string()));
    EXPECT_EQ(countLayerType(loaded, "TypeConverter"), 1u);
    EXPECT_EQ(countLayerType(loaded, "SegmentedBroadcast"), 1u);
    EXPECT_EQ(countLayerType(loaded, "RaggedCustomLoss"), 1u);
    EXPECT_EQ(countLayerType(loaded, "RaggedLossShaper"), 1u);
    ASSERT_EQ(loaded.getLossRootTensors().size(), 1u);

    shared_ptr<RaggedCustomLoss> loadedRaw;
    for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
        loadedRaw = dynamic_pointer_cast<RaggedCustomLoss>(loaded.getLayer(i));
        if (loadedRaw != nullptr) break;
    }
    ASSERT_NE(loadedRaw, nullptr);
    EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedLabels().getOffsets());
    EXPECT_EQ(loadedRaw->getRaggedRawLoss().getOffsets(), loadedRaw->getRaggedPredictions().getOffsets());
    ASSERT_TRUE(loadedRaw->getRaggedExampleWeights().has_value());
    EXPECT_EQ(loadedRaw->getRaggedExampleWeights()->getOffsets(), loadedRaw->getRaggedPredictions().getOffsets());
    EXPECT_EQ(loadedRaw->getRaggedExampleWeights()->getTrailingDimensions(), (vector<uint64_t>{1}));
    ASSERT_TRUE(loadedRaw->getLossWeight().has_value());
    EXPECT_FLOAT_EQ(loadedRaw->getLossWeight().value(), 0.5f);

    filesystem::remove_all(archiveDir);
}
