#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"
#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
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
#include <filesystem>
#include <cmath>
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

Inputs makeInputs(Network& network,
                  DataType predictionDType = DataType::FP32,
                  DataType labelDType = DataType::FP32,
                  DataType offsetsDType = DataType::UINT32,
                  vector<uint64_t> trailingDimensions = {1},
                  uint32_t batchSize = 4,
                  uint64_t maxTotalValues = 8) {
    RaggedTensor predictions = RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .valuesDataType(predictionDType)
                                   .offsetsDataType(offsetsDType)
                                   .trailingDimensions(trailingDimensions)
                                   .batchSize(batchSize)
                                   .maxTotalValues(maxTotalValues)
                                   .maxValuesPerRow(5)
                                   .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(labelDType)
                              .trailingDimensions(trailingDimensions)
                              .partition(predictions)
                              .build();
    return {predictions, labels};
}

uint32_t countLayerType(Network& network, const string& type) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i)
        if (network.getLayer(i)->getLayerType() == type) ++count;
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
    ASSERT_EQ(dtype, DataType::UINT64);
    copy(offsets.begin(), offsets.end(), offsetsTensor.getMemPtr<uint64_t>());
}

vector<float> copyFp32ToHost(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = tensor.clone(cpuPlacement);
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + host.getTotalNumElements());
}

enum class Kind { QUANTILE, EXPECTILE, ASYMMETRIC_POWER };

pair<float, float> referenceLossGradient(Kind kind, float prediction, float label) {
    constexpr float level = 0.8f;
    constexpr float exponent = 1.5f;
    const float error = label - prediction;
    const float predictionError = prediction - label;
    const float scale = Impl::Loss::getLossScalingFactor();

    if (kind == Kind::QUANTILE) {
        if (error > 0.0f) return {level * error, -level * scale};
        if (error < 0.0f) return {(level - 1.0f) * error, (1.0f - level) * scale};
        return {0.0f, 0.0f};
    }

    const float asymmetricWeight = error > 0.0f ? 2.0f * level : 2.0f * (1.0f - level);
    if (kind == Kind::EXPECTILE) {
        return {asymmetricWeight * error * error, 2.0f * asymmetricWeight * predictionError * scale};
    }

    const float absError = std::abs(error);
    const float loss = asymmetricWeight * std::pow(absError, exponent);
    float gradient = 0.0f;
    if (predictionError != 0.0f) {
        const float sign = predictionError > 0.0f ? 1.0f : -1.0f;
        gradient = asymmetricWeight * exponent * sign * std::pow(std::abs(predictionError), exponent - 1.0f) * scale;
    }
    return {loss, gradient};
}

Tensor buildLoss(Kind kind, Network& network, const Inputs& inputs, optional<Tensor> weights) {
    if (kind == Kind::QUANTILE) {
        QuantileLoss::Builder builder;
        builder.network(network).predictions(inputs.predictions).labels(inputs.labels).quantile(0.8f).reportsBatchLoss();
        if (weights.has_value()) builder.exampleWeights(weights.value());
        return builder.build().getLoss();
    }
    if (kind == Kind::EXPECTILE) {
        ExpectileLoss::Builder builder;
        builder.network(network).predictions(inputs.predictions).labels(inputs.labels).expectile(0.8f).reportsBatchLoss();
        if (weights.has_value()) builder.exampleWeights(weights.value());
        return builder.build().getLoss();
    }
    AsymmetricPowerLoss::Builder builder;
    builder.network(network)
        .predictions(inputs.predictions)
        .labels(inputs.labels)
        .level(0.8f)
        .exponent(1.5f)
        .reportsBatchLoss();
    if (weights.has_value()) builder.exampleWeights(weights.value());
    return builder.build().getLoss();
}

void runRuntimeCase(Kind kind, DataType offsetsDType, bool weighted) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    Network network("ragged_r10g_runtime");
    Inputs inputs = makeInputs(network, DataType::FP32, DataType::FP32, offsetsDType, {1}, batchSize, maxTotalValues);

    optional<Tensor> weights;
    if (weighted) {
        NetworkInput input = NetworkInput::Builder().network(network).name("weights").dimensions({1}).dataType(DataType::FP32).build();
        weights = input.getFeatureOutput().value();
    }

    Tensor reportedLoss = buildLoss(kind, network, inputs, weights);
    (void)NetworkOutput::Builder().network(network).name("loss").inputTensor(reportedLoss).dataType(DataType::FP32).build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(batchSize, initializationDone, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initializationDone) event.synchronize();

    shared_ptr<Impl::RaggedCustomLoss> physicalLoss;
    for (const shared_ptr<Impl::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
        auto candidate = dynamic_pointer_cast<Impl::RaggedCustomLoss>(layer);
        if (candidate == nullptr) continue;
        ASSERT_EQ(physicalLoss, nullptr);
        physicalLoss = candidate;
    }
    ASSERT_NE(physicalLoss, nullptr);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor predictionValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor labelValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));
    float* predictions = predictionValues.getMemPtr<float>();
    float* labels = labelValues.getMemPtr<float>();
    fill(predictions, predictions + maxTotalValues, numeric_limits<float>::quiet_NaN());
    fill(labels, labels + maxTotalValues, numeric_limits<float>::quiet_NaN());

    const vector<float> activePredictions{0.0f, 2.0f, -3.0f, 4.0f, -5.0f};
    const vector<float> activeLabels{1.0f, 1.0f, -3.0f, 2.0f, -3.0f};
    copy(activePredictions.begin(), activePredictions.end(), predictions);
    copy(activeLabels.begin(), activeLabels.end(), labels);
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, inputs.predictions.getMaxValuesPerRow()));
    batch.insert("labels", labelValues);
    if (weighted) {
        Impl::Tensor hostWeights(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1}));
        float* w = hostWeights.getMemPtr<float>();
        w[0] = 0.5f;
        w[1] = 7.0f;    // valid empty row
        w[2] = 2.0f;
        w[3] = 99.0f;   // invalid canonical-empty tail row
        batch.insert("weights", hostWeights);
    }
    batch.setValidExampleCount(validExamples);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    outputReadyEvents.at("loss").synchronize();
    placed->synchronize();

    double numerator = 0.0;
    vector<float> expectedGradients;
    for (size_t i = 0; i < activePredictions.size(); ++i) {
        auto [loss, gradient] = referenceLossGradient(kind, activePredictions[i], activeLabels[i]);
        const float rowWeight = !weighted ? 1.0f : (i < 2 ? 0.5f : 2.0f);
        numerator += static_cast<double>(loss) * rowWeight;
        expectedGradients.push_back(gradient * rowWeight);
    }

    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    const double expectedBatchLoss = numerator / validExamples;
    const double tolerance = max(1.0e-5, 8.0 * static_cast<double>(numeric_limits<float>::epsilon()) * abs(expectedBatchLoss));
    EXPECT_NEAR(reported[0], expectedBatchLoss, tolerance);

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    ASSERT_EQ(gradient.size(), maxTotalValues);
    for (size_t i = 0; i < expectedGradients.size(); ++i)
        EXPECT_NEAR(gradient[i], expectedGradients[i], 2.0e-4f) << "active index " << i;
}

}  // namespace

TEST(RaggedRegressionR10G, ReportingAndWeightContracts) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::QUANTILE, Kind::EXPECTILE, Kind::ASYMMETRIC_POWER}) {
            Network rawNetwork("r10g_raw");
            Inputs inputs = makeInputs(rawNetwork, DataType::BF16, DataType::INT32, offsetsDType, {2});
            RaggedTensor raw;
            if (kind == Kind::QUANTILE) {
                QuantileLoss loss = QuantileLoss::Builder().network(rawNetwork).predictions(inputs.predictions).labels(inputs.labels).quantile(0.8f).reportsRawLoss().build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedRawLoss();
            } else if (kind == Kind::EXPECTILE) {
                ExpectileLoss loss = ExpectileLoss::Builder().network(rawNetwork).predictions(inputs.predictions).labels(inputs.labels).expectile(0.8f).reportsRawLoss().build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedRawLoss();
            } else {
                AsymmetricPowerLoss loss = AsymmetricPowerLoss::Builder().network(rawNetwork).predictions(inputs.predictions).labels(inputs.labels).level(0.8f).exponent(1.5f).reportsRawLoss().build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedRawLoss();
            }
            EXPECT_EQ(raw.getOffsets(), inputs.predictions.getOffsets());
            EXPECT_EQ(raw.getValuesDataType(), DataType::FP32);
            EXPECT_EQ(countLayerType(rawNetwork, "RaggedCustomLoss"), 1u);

            Network perExampleNetwork("r10g_per_example");
            Inputs perExampleInputs = makeInputs(perExampleNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            Tensor perExample;
            if (kind == Kind::QUANTILE)
                perExample = QuantileLoss::Builder().network(perExampleNetwork).predictions(perExampleInputs.predictions).labels(perExampleInputs.labels).reportsPerExampleLoss().build().getLoss();
            else if (kind == Kind::EXPECTILE)
                perExample = ExpectileLoss::Builder().network(perExampleNetwork).predictions(perExampleInputs.predictions).labels(perExampleInputs.labels).reportsPerExampleLoss().build().getLoss();
            else
                perExample = AsymmetricPowerLoss::Builder().network(perExampleNetwork).predictions(perExampleInputs.predictions).labels(perExampleInputs.labels).reportsPerExampleLoss().build().getLoss();
            EXPECT_EQ(perExample.getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);

            Network noneNetwork("r10g_none");
            Inputs noneInputs = makeInputs(noneNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            if (kind == Kind::QUANTILE) {
                QuantileLoss loss = QuantileLoss::Builder().network(noneNetwork).predictions(noneInputs.predictions).labels(noneInputs.labels).reportsNoLoss().build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            } else if (kind == Kind::EXPECTILE) {
                ExpectileLoss loss = ExpectileLoss::Builder().network(noneNetwork).predictions(noneInputs.predictions).labels(noneInputs.labels).reportsNoLoss().build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            } else {
                AsymmetricPowerLoss loss = AsymmetricPowerLoss::Builder().network(noneNetwork).predictions(noneInputs.predictions).labels(noneInputs.labels).reportsNoLoss().build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            }
            ASSERT_EQ(noneNetwork.getLossRootTensors().size(), 1u);

            Network rejectNetwork("r10g_reject");
            Inputs reject = makeInputs(rejectNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            if (kind == Kind::QUANTILE)
                EXPECT_THROW((void)QuantileLoss::Builder().network(rejectNetwork).predictions(reject.predictions).labels(reject.labels).reportsPerOutputLoss().build(), invalid_argument);
            else if (kind == Kind::EXPECTILE)
                EXPECT_THROW((void)ExpectileLoss::Builder().network(rejectNetwork).predictions(reject.predictions).labels(reject.labels).reportsPerOutputLoss().build(), invalid_argument);
            else
                EXPECT_THROW((void)AsymmetricPowerLoss::Builder().network(rejectNetwork).predictions(reject.predictions).labels(reject.labels).reportsPerOutputLoss().build(), invalid_argument);
        }
    }
}

TEST(RaggedRegressionR10G, RequiresExactSharedPartition) {
    for (Kind kind : {Kind::QUANTILE, Kind::EXPECTILE, Kind::ASYMMETRIC_POWER}) {
        Network network("r10g_partition");
        RaggedTensor predictions = RaggedNetworkInput::Builder().network(network).name("predictions").valuesDataType(DataType::FP32).trailingDimensions({1}).batchSize(3).maxTotalValues(7).maxValuesPerRow(4).build();
        RaggedTensor labels = RaggedNetworkInput::Builder().network(network).name("labels").valuesDataType(DataType::FP32).trailingDimensions({1}).batchSize(3).maxTotalValues(7).maxValuesPerRow(4).build();
        if (kind == Kind::QUANTILE)
            EXPECT_THROW((void)QuantileLoss::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
        else if (kind == Kind::EXPECTILE)
            EXPECT_THROW((void)ExpectileLoss::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
        else
            EXPECT_THROW((void)AsymmetricPowerLoss::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
    }
}

TEST(RaggedRegressionR10G, WeightedAndUnweightedForwardBackwardUseOnlyActivePrefix) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::QUANTILE, Kind::EXPECTILE, Kind::ASYMMETRIC_POWER}) {
            runRuntimeCase(kind, offsetsDType, false);
            runRuntimeCase(kind, offsetsDType, true);
        }
    }
}


TEST(RaggedRegressionR10G, DenseDTypeParityAndIntegerLabels) {
    const vector<pair<DataType, DataType>> cases{
        {DataType::FP8_E4M3, DataType::FP32},
        {DataType::FP8_E5M2, DataType::FP32},
        {DataType::FP16, DataType::FP16},
        {DataType::BF16, DataType::FP32},
        {DataType::FP32, DataType::FP32},
    };
    for (Kind kind : {Kind::QUANTILE, Kind::EXPECTILE, Kind::ASYMMETRIC_POWER}) {
        for (auto [predictionDType, expectedLossDType] : cases) {
            Network network("r10g_dtype");
            Inputs inputs = makeInputs(network, predictionDType, DataType::INT32, DataType::UINT64, {2}, 3, 7);
            RaggedTensor raw;
            if (kind == Kind::QUANTILE)
                raw = QuantileLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build().getRaggedRawLoss();
            else if (kind == Kind::EXPECTILE)
                raw = ExpectileLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build().getRaggedRawLoss();
            else
                raw = AsymmetricPowerLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build().getRaggedRawLoss();
            EXPECT_EQ(raw.getValuesDataType(), expectedLossDType);
        }
    }
}

TEST(RaggedRegressionR10G, WeightedSupportLayersSaveLoadWithCanonicalPartition) {
    for (Kind kind : {Kind::QUANTILE, Kind::EXPECTILE, Kind::ASYMMETRIC_POWER}) {
        Network network("r10g_round_trip");
        Inputs inputs = makeInputs(network, DataType::FP32, DataType::FP32, DataType::UINT64);
        NetworkInput weightsInput = NetworkInput::Builder().network(network).name("weights").dimensions({1}).dataType(DataType::BF16).build();
        Tensor weights = weightsInput.getFeatureOutput().value();
        if (kind == Kind::QUANTILE)
            (void)QuantileLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).quantile(0.8f).exampleWeights(weights).reportsPerExampleLoss().build();
        else if (kind == Kind::EXPECTILE)
            (void)ExpectileLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).expectile(0.8f).exampleWeights(weights).reportsPerExampleLoss().build();
        else
            (void)AsymmetricPowerLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).level(0.8f).exponent(1.5f).exampleWeights(weights).reportsPerExampleLoss().build();

        const auto now = chrono::steady_clock::now().time_since_epoch().count();
        const filesystem::path archiveDir = filesystem::temp_directory_path() / (string("thor_r10g_") + to_string(now));
        filesystem::remove_all(archiveDir);
        network.save(archiveDir.string(), /*overwrite=*/true);
        Network loaded("r10g_round_trip");
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
        ASSERT_TRUE(loadedRaw->getRaggedExampleWeights().has_value());
        EXPECT_EQ(loadedRaw->getRaggedExampleWeights()->getOffsets(), loadedRaw->getRaggedPredictions().getOffsets());
        filesystem::remove_all(archiveDir);
    }
}

TEST(RaggedRegressionR10G, PerRowWeightsPlaceForScalarAndMultiAxisTrailingValues) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required";
    for (Kind kind : {Kind::QUANTILE, Kind::EXPECTILE, Kind::ASYMMETRIC_POWER}) {
        for (const vector<uint64_t>& trailing : {vector<uint64_t>{}, vector<uint64_t>{2, 3}}) {
            Network network("r10g_weight_shape");
            Inputs inputs = makeInputs(network, DataType::FP32, DataType::FP32, DataType::UINT32, trailing, 3, 7);
            NetworkInput weightsInput = NetworkInput::Builder().network(network).name("weights").dimensions({1}).dataType(DataType::FP32).build();
            Tensor weights = weightsInput.getFeatureOutput().value();
            Tensor reported;
            if (kind == Kind::QUANTILE)
                reported = QuantileLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).exampleWeights(weights).reportsBatchLoss().build().getLoss();
            else if (kind == Kind::EXPECTILE)
                reported = ExpectileLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).exampleWeights(weights).reportsBatchLoss().build().getLoss();
            else
                reported = AsymmetricPowerLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).exampleWeights(weights).reportsBatchLoss().build().getLoss();
            (void)NetworkOutput::Builder().network(network).name("loss").inputTensor(reported).dataType(DataType::FP32).build();
            vector<Event> initializationDone;
            shared_ptr<PlacedNetwork> placed = network.place(3, initializationDone, /*inferenceOnly=*/false);
            ASSERT_NE(placed, nullptr);
            for (Event& event : initializationDone) event.synchronize();
            placed->synchronize();
        }
    }
}
