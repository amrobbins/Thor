#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
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
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
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
                                 DataType predictionsDType = DataType::FP32,
                                 DataType labelsDType = DataType::FP32,
                                 DataType offsetsDType = DataType::UINT32,
                                 vector<uint64_t> trailingDimensions = {2},
                                 uint32_t batchSize = 4,
                                 uint64_t maxTotalValues = 11) {
    RaggedTensor predictions = RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .valuesDataType(predictionsDType)
                                   .offsetsDataType(offsetsDType)
                                   .trailingDimensions(trailingDimensions)
                                   .batchSize(batchSize)
                                   .maxTotalValues(maxTotalValues)
                                   .maxValuesPerRow(5)
                                   .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(labelsDType)
                              .trailingDimensions(trailingDimensions)
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

enum class LossKind { MSE, MEAN_POWER };

void runRuntimeCase(LossKind kind, DataType offsetsDType, bool weighted) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t trailingWidth = 2;
    constexpr float exponent = 1.5f;

    const string kindName = kind == LossKind::MSE ? "mse" : "mean_power";
    Network network("ragged_r10e_" + kindName + (weighted ? "_weighted" : "_unweighted") +
                    (offsetsDType == DataType::UINT32 ? "_u32" : "_u64"));
    Inputs inputs = makeSharedPartitionInputs(
        network, DataType::FP32, DataType::FP32, offsetsDType, {trailingWidth}, batchSize, maxTotalValues);

    optional<Tensor> weightsTensor;
    if (weighted) {
        NetworkInput weightsInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("weights")
                                        .dimensions({1})
                                        .dataType(DataType::FP32)
                                        .build();
        weightsTensor = weightsInput.getFeatureOutput().value();
    }

    Tensor reportedLoss;
    if (kind == LossKind::MSE) {
        MSE::Builder builder;
        builder.network(network).predictions(inputs.predictions).labels(inputs.labels).reportsBatchLoss();
        if (weightsTensor.has_value()) builder.exampleWeights(weightsTensor.value());
        MSE loss = builder.build();
        reportedLoss = loss.getLoss();
    } else {
        MeanPowerError::Builder builder;
        builder.network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .exponent(exponent)
            .reportsBatchLoss();
        if (weightsTensor.has_value()) builder.exampleWeights(weightsTensor.value());
        MeanPowerError loss = builder.build();
        reportedLoss = loss.getLoss();
    }

    (void)NetworkOutput::Builder()
        .network(network)
        .name("loss")
        .inputTensor(reportedLoss)
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
    const vector<float> activePredictions{1.0f, -4.0f, 9.0f, -16.0f, 25.0f, -36.0f, 49.0f, -64.0f, 81.0f, -100.0f};
    copy(activePredictions.begin(), activePredictions.end(), p);
    fill(y, y + activePredictions.size(), 0.0f);
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, inputs.predictions.getMaxValuesPerRow()));
    batch.insert("labels", labelValues);
    if (weighted) {
        Impl::Tensor weights(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1}));
        float* w = weights.getMemPtr<float>();
        w[0] = 0.5f;
        w[1] = 7.0f;    // Empty valid row: zero contribution but still part of denominator.
        w[2] = 2.0f;
        w[3] = 100.0f;  // Invalid tail row: canonical offsets keep it empty.
        batch.insert("weights", weights);
    }
    batch.setValidExampleCount(validExamples);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    outputReadyEvents.at("loss").synchronize();
    placed->synchronize();

    double row0 = 0.0;
    double row2 = 0.0;
    for (size_t i = 0; i < activePredictions.size(); ++i) {
        const double absValue = std::abs(static_cast<double>(activePredictions[i]));
        const double contribution = kind == LossKind::MSE ? absValue * absValue : std::pow(absValue, exponent);
        if (i < 4)
            row0 += contribution;
        else
            row2 += contribution;
    }
    const double numerator = weighted ? row0 * 0.5 + row2 * 2.0 : row0 + row2;
    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    const double expectedBatchLoss = numerator / validExamples;
    const double fp32Tolerance =
        max(1.0e-3, 4.0 * static_cast<double>(numeric_limits<float>::epsilon()) * abs(expectedBatchLoss));
    EXPECT_NEAR(reported[0], expectedBatchLoss, fp32Tolerance);

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    ASSERT_EQ(gradient.size(), maxTotalValues * trailingWidth);
    const float scale = Impl::Loss::getLossScalingFactor();
    for (size_t i = 0; i < activePredictions.size(); ++i) {
        const float prediction = activePredictions[i];
        const float rowWeight = weighted ? (i < 4 ? 0.5f : 2.0f) : 1.0f;
        const float expected = kind == LossKind::MSE
                                   ? 2.0f * prediction * rowWeight * scale
                                   : (prediction > 0.0f ? 1.0f : -1.0f) * exponent * std::sqrt(std::abs(prediction)) * rowWeight * scale;
        EXPECT_NEAR(gradient[i], expected, 1.0e-3f) << "index=" << i;
    }
    // The inactive packed tail is intentionally left undefined; NaNs there must
    // not affect the active loss or gradient calculation.
}

}  // namespace

TEST(RaggedRegressionR10E, MSESupportsAllRaggedReportingShapesExceptPerOutput) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network rawNetwork("r10e_mse_raw");
        Inputs rawInputs = makeSharedPartitionInputs(rawNetwork, DataType::FP16, DataType::INT32, offsetsDType);
        MSE raw = MSE::Builder()
                      .network(rawNetwork)
                      .predictions(rawInputs.predictions)
                      .labels(rawInputs.labels)
                      .lossDataType(DataType::FP32)
                      .reportsRawLoss()
                      .build();
        EXPECT_TRUE(raw.isRagged());
        EXPECT_EQ(raw.getRaggedRawLoss().getOffsets(), rawInputs.predictions.getOffsets());
        EXPECT_EQ(raw.getRaggedRawLoss().getTrailingDimensions(), (vector<uint64_t>{2}));
        EXPECT_EQ(raw.getRaggedRawLoss().getValuesDataType(), DataType::FP32);

        Network perExampleNetwork("r10e_mse_per_example");
        Inputs perExampleInputs = makeSharedPartitionInputs(perExampleNetwork, DataType::FP32, DataType::FP32, offsetsDType);
        MSE perExample = MSE::Builder()
                             .network(perExampleNetwork)
                             .predictions(perExampleInputs.predictions)
                             .labels(perExampleInputs.labels)
                             .reportsPerExampleLoss()
                             .build();
        EXPECT_EQ(perExample.getLoss().getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);

        Network batchNetwork("r10e_mse_batch");
        Inputs batchInputs = makeSharedPartitionInputs(batchNetwork, DataType::FP32, DataType::FP32, offsetsDType);
        MSE batch = MSE::Builder()
                        .network(batchNetwork)
                        .predictions(batchInputs.predictions)
                        .labels(batchInputs.labels)
                        .reportsBatchLoss()
                        .build();
        EXPECT_EQ(batch.getLoss().getDimensions(), (vector<uint64_t>{1}));

        Network noneNetwork("r10e_mse_none");
        Inputs noneInputs = makeSharedPartitionInputs(noneNetwork, DataType::FP32, DataType::FP32, offsetsDType);
        MSE none = MSE::Builder()
                       .network(noneNetwork)
                       .predictions(noneInputs.predictions)
                       .labels(noneInputs.labels)
                       .reportsNoLoss()
                       .build();
        EXPECT_THROW((void)none.getLoss(), runtime_error);
        ASSERT_EQ(noneNetwork.getLossRootTensors().size(), 1u);

        Network rejectNetwork("r10e_mse_reject_per_output");
        Inputs rejectInputs = makeSharedPartitionInputs(rejectNetwork, DataType::FP32, DataType::FP32, offsetsDType);
        EXPECT_THROW((void)MSE::Builder()
                         .network(rejectNetwork)
                         .predictions(rejectInputs.predictions)
                         .labels(rejectInputs.labels)
                         .reportsPerOutputLoss()
                         .build(),
                     invalid_argument);
    }
}

TEST(RaggedRegressionR10E, MeanPowerErrorSupportsRaggedReportingAndExponent) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r10e_mean_power_raw");
        Inputs inputs = makeSharedPartitionInputs(network, DataType::BF16, DataType::UINT16, offsetsDType);
        MeanPowerError loss = MeanPowerError::Builder()
                                  .network(network)
                                  .predictions(inputs.predictions)
                                  .labels(inputs.labels)
                                  .exponent(1.5f)
                                  .lossDataType(DataType::FP32)
                                  .reportsRawLoss()
                                  .build();
        EXPECT_TRUE(loss.isRagged());
        EXPECT_FLOAT_EQ(loss.getExponent(), 1.5f);
        EXPECT_EQ(loss.getRaggedRawLoss().getOffsets(), inputs.predictions.getOffsets());
        EXPECT_EQ(loss.getRaggedRawLoss().getValuesDataType(), DataType::FP32);

        Network rejectNetwork("r10e_mean_power_reject_per_output");
        Inputs rejectInputs = makeSharedPartitionInputs(rejectNetwork, DataType::FP32, DataType::FP32, offsetsDType);
        EXPECT_THROW((void)MeanPowerError::Builder()
                         .network(rejectNetwork)
                         .predictions(rejectInputs.predictions)
                         .labels(rejectInputs.labels)
                         .exponent(1.25f)
                         .reportsPerOutputLoss()
                         .build(),
                     invalid_argument);
    }
}

TEST(RaggedRegressionR10E, BothLossesRejectDifferentPartitionsAndAcceptDensePerRowWeights) {
    for (LossKind kind : {LossKind::MSE, LossKind::MEAN_POWER}) {
        Network network(kind == LossKind::MSE ? "r10e_mse_partition" : "r10e_mean_power_partition");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP32)
                                       .trailingDimensions({2})
                                       .batchSize(3)
                                       .maxTotalValues(8)
                                       .maxValuesPerRow(4)
                                       .build();
        RaggedTensor differentLabels = RaggedNetworkInput::Builder()
                                           .network(network)
                                           .name("different_labels")
                                           .valuesDataType(DataType::FP32)
                                           .trailingDimensions({2})
                                           .batchSize(3)
                                           .maxTotalValues(8)
                                           .maxValuesPerRow(4)
                                           .build();
        if (kind == LossKind::MSE) {
            EXPECT_THROW((void)MSE::Builder().network(network).predictions(predictions).labels(differentLabels).build(), invalid_argument);
        } else {
            EXPECT_THROW((void)MeanPowerError::Builder()
                             .network(network)
                             .predictions(predictions)
                             .labels(differentLabels)
                             .build(),
                         invalid_argument);
        }

        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({2})
                                  .partition(predictions)
                                  .build();
        NetworkInput weightsInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("weights")
                                        .dimensions({1})
                                        .dataType(DataType::BF16)
                                        .build();
        if (kind == LossKind::MSE) {
            EXPECT_NO_THROW((void)MSE::Builder()
                                .network(network)
                                .predictions(predictions)
                                .labels(labels)
                                .exampleWeights(weightsInput.getFeatureOutput().value())
                                .reportsRawLoss()
                                .build());
        } else {
            EXPECT_NO_THROW((void)MeanPowerError::Builder()
                                .network(network)
                                .predictions(predictions)
                                .labels(labels)
                                .exampleWeights(weightsInput.getFeatureOutput().value())
                                .reportsRawLoss()
                                .build());
        }
    }
}

TEST(RaggedRegressionR10E, MatchesDenseRegressionPredictionDTypeAndDefaultLossStorage) {
    const vector<pair<DataType, DataType>> cases{{DataType::FP8_E4M3, DataType::FP32},
                                                 {DataType::FP8_E5M2, DataType::FP32},
                                                 {DataType::FP16, DataType::FP16},
                                                 {DataType::BF16, DataType::FP32},
                                                 {DataType::FP32, DataType::FP32}};
    for (LossKind kind : {LossKind::MSE, LossKind::MEAN_POWER}) {
        for (const auto& [predictionDType, expectedLossDType] : cases) {
            Network network(kind == LossKind::MSE ? "r10e_mse_dtype" : "r10e_mean_power_dtype");
            Inputs inputs = makeSharedPartitionInputs(network, predictionDType, DataType::INT32);
            if (kind == LossKind::MSE) {
                MSE loss = MSE::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build();
                EXPECT_EQ(loss.getRaggedRawLoss().getValuesDataType(), expectedLossDType);
            } else {
                MeanPowerError loss = MeanPowerError::Builder()
                                          .network(network)
                                          .predictions(inputs.predictions)
                                          .labels(inputs.labels)
                                          .reportsRawLoss()
                                          .build();
                EXPECT_EQ(loss.getRaggedRawLoss().getValuesDataType(), expectedLossDType);
            }
        }
    }
}

TEST(RaggedRegressionR10E, WeightedAndUnweightedForwardBackwardUseOnlyActivePrefix) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for R10E ragged regression runtime coverage.";
    for (LossKind kind : {LossKind::MSE, LossKind::MEAN_POWER}) {
        for (bool weighted : {false, true}) {
            for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
                runRuntimeCase(kind, offsetsDType, weighted);
            }
        }
    }
}

TEST(RaggedRegressionR10E, WeightedSupportLayersSaveLoadWithCanonicalPartition) {
    for (LossKind kind : {LossKind::MSE, LossKind::MEAN_POWER}) {
        const string networkName = kind == LossKind::MSE ? "r10e_mse_round_trip" : "r10e_mean_power_round_trip";
        Network network(networkName);
        Inputs inputs = makeSharedPartitionInputs(network, DataType::FP32, DataType::FP32, DataType::UINT64);
        NetworkInput weightsInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("weights")
                                        .dimensions({1})
                                        .dataType(DataType::BF16)
                                        .build();
        if (kind == LossKind::MSE) {
            (void)MSE::Builder()
                .network(network)
                .predictions(inputs.predictions)
                .labels(inputs.labels)
                .exampleWeights(weightsInput.getFeatureOutput().value())
                .reportsPerExampleLoss()
                .build();
        } else {
            (void)MeanPowerError::Builder()
                .network(network)
                .predictions(inputs.predictions)
                .labels(inputs.labels)
                .exampleWeights(weightsInput.getFeatureOutput().value())
                .exponent(1.25f)
                .reportsPerExampleLoss()
                .build();
        }

        const auto now = chrono::steady_clock::now().time_since_epoch().count();
        const filesystem::path archiveDir = filesystem::temp_directory_path() /
            (string("thor_r10e_") + networkName + "_" + to_string(now));
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

        filesystem::remove_all(archiveDir);
    }
}

TEST(RaggedRegressionR10E, PerRowWeightsPlaceForScalarAndMultiAxisTrailingValues) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for R10E weighted shape placement coverage.";

    for (LossKind kind : {LossKind::MSE, LossKind::MEAN_POWER}) {
        for (const vector<uint64_t>& trailingDimensions : {vector<uint64_t>{}, vector<uint64_t>{2, 3}}) {
            Network network(kind == LossKind::MSE ? "r10e_mse_weight_shape" : "r10e_mean_power_weight_shape");
            Inputs inputs = makeSharedPartitionInputs(
                network, DataType::FP32, DataType::FP32, DataType::UINT32, trailingDimensions, 3, 7);
            NetworkInput weightsInput = NetworkInput::Builder()
                                            .network(network)
                                            .name("weights")
                                            .dimensions({1})
                                            .dataType(DataType::FP32)
                                            .build();
            Tensor reportedLoss;
            if (kind == LossKind::MSE) {
                MSE loss = MSE::Builder()
                               .network(network)
                               .predictions(inputs.predictions)
                               .labels(inputs.labels)
                               .exampleWeights(weightsInput.getFeatureOutput().value())
                               .reportsBatchLoss()
                               .build();
                reportedLoss = loss.getLoss();
            } else {
                MeanPowerError loss = MeanPowerError::Builder()
                                          .network(network)
                                          .predictions(inputs.predictions)
                                          .labels(inputs.labels)
                                          .exampleWeights(weightsInput.getFeatureOutput().value())
                                          .exponent(1.5f)
                                          .reportsBatchLoss()
                                          .build();
                reportedLoss = loss.getLoss();
            }
            (void)NetworkOutput::Builder()
                .network(network)
                .name("loss")
                .inputTensor(reportedLoss)
                .dataType(DataType::FP32)
                .build();

            vector<Event> initializationDone;
            shared_ptr<PlacedNetwork> placed = network.place(3, initializationDone, /*inferenceOnly=*/false);
            ASSERT_NE(placed, nullptr);
            for (Event& event : initializationDone) event.synchronize();
            placed->synchronize();
        }
    }
}
