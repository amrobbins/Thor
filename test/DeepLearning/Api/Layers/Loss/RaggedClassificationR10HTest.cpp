#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
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

enum class Kind { BCE, FOCAL };

float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

pair<float, float> referenceLossGradient(Kind kind, float logit, float label, float gamma = 2.0f, float alpha = 0.25f) {
    const float bce = max(logit, 0.0f) - logit * label + log1p(exp(-abs(logit)));
    const float bceGradient = sigmoid(logit) - label;
    const float scale = Impl::Loss::getLossScalingFactor();
    if (kind == Kind::BCE) return {bce, bceGradient * scale};

    const float alphaFactor = label * alpha + (1.0f - label) * (1.0f - alpha);
    if (gamma == 0.0f) return {alphaFactor * bce, alphaFactor * bceGradient * scale};
    const float pt = exp(-bce);
    const float oneMinusPt = max(1.0f - pt, 1.0e-7f);
    const float focalWeight = pow(oneMinusPt, gamma);
    const float derivativeTerm = gamma * bce * pt * pow(oneMinusPt, gamma - 1.0f);
    return {alphaFactor * bce * focalWeight, alphaFactor * bceGradient * (focalWeight + derivativeTerm) * scale};
}

Tensor buildBatchLoss(Kind kind, Network& network, const Inputs& inputs, float gamma = 2.0f, float alpha = 0.25f) {
    if (kind == Kind::BCE)
        return BinaryCrossEntropy::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsBatchLoss().build().getLoss();
    return BinaryFocalLoss::Builder()
        .network(network)
        .predictions(inputs.predictions)
        .labels(inputs.labels)
        .focusingParameter(gamma)
        .alpha(alpha)
        .reportsBatchLoss()
        .build()
        .getLoss();
}

void runRuntimeCase(Kind kind, DataType offsetsDType, float gamma = 2.0f, float alpha = 0.25f) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    Network network("ragged_r10h_runtime");
    Inputs inputs = makeInputs(network, DataType::FP32, DataType::FP32, offsetsDType, {1}, batchSize, maxTotalValues);
    Tensor reportedLoss = buildBatchLoss(kind, network, inputs, gamma, alpha);
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

    const vector<float> activePredictions{-2.0f, -0.25f, 0.0f, 1.5f, 3.0f};
    const vector<float> activeLabels{0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    copy(activePredictions.begin(), activePredictions.end(), predictions);
    copy(activeLabels.begin(), activeLabels.end(), labels);
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, inputs.predictions.getMaxValuesPerRow()));
    batch.insert("labels", labelValues);
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
        auto [loss, gradient] = referenceLossGradient(kind, activePredictions[i], activeLabels[i], gamma, alpha);
        numerator += loss;
        expectedGradients.push_back(gradient);
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
        EXPECT_NEAR(gradient[i], expectedGradients[i], 3.0e-4f) << "active index " << i;
}

}  // namespace

TEST(RaggedClassificationR10H, ReportingContractsAndExactPartition) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::BCE, Kind::FOCAL}) {
            Network rawNetwork("r10h_raw");
            Inputs inputs = makeInputs(rawNetwork, DataType::FP16, DataType::UINT8, offsetsDType, {2});
            RaggedTensor raw;
            if (kind == Kind::BCE) {
                BinaryCrossEntropy loss = BinaryCrossEntropy::Builder()
                                              .network(rawNetwork)
                                              .predictions(inputs.predictions)
                                              .labels(inputs.labels)
                                              .reportsRawLoss()
                                              .build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedLoss();
            } else {
                BinaryFocalLoss loss = BinaryFocalLoss::Builder()
                                           .network(rawNetwork)
                                           .predictions(inputs.predictions)
                                           .labels(inputs.labels)
                                           .reportsRawLoss()
                                           .build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedLoss();
            }
            EXPECT_EQ(raw.getOffsets(), inputs.predictions.getOffsets());
            EXPECT_EQ(raw.getValuesDataType(), kind == Kind::BCE ? DataType::FP32 : DataType::FP16);
            EXPECT_EQ(countLayerType(rawNetwork, "RaggedCustomLoss"), 1u);

            Network perExampleNetwork("r10h_per_example");
            Inputs perInputs = makeInputs(perExampleNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            Tensor perExample = kind == Kind::BCE
                                    ? BinaryCrossEntropy::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss()
                                    : BinaryFocalLoss::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss();
            EXPECT_EQ(perExample.getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);

            Network noneNetwork("r10h_none");
            Inputs noneInputs = makeInputs(noneNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            if (kind == Kind::BCE) {
                BinaryCrossEntropy loss = BinaryCrossEntropy::Builder()
                                              .network(noneNetwork)
                                              .predictions(noneInputs.predictions)
                                              .labels(noneInputs.labels)
                                              .reportsNoLoss()
                                              .build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            } else {
                BinaryFocalLoss loss = BinaryFocalLoss::Builder()
                                           .network(noneNetwork)
                                           .predictions(noneInputs.predictions)
                                           .labels(noneInputs.labels)
                                           .reportsNoLoss()
                                           .build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            }
            ASSERT_EQ(noneNetwork.getLossRootTensors().size(), 1u);
        }
    }

    Network network("r10h_partition");
    RaggedTensor predictions = RaggedNetworkInput::Builder().network(network).name("p").valuesDataType(DataType::FP32).trailingDimensions({1}).batchSize(3).maxTotalValues(7).maxValuesPerRow(4).build();
    RaggedTensor labels = RaggedNetworkInput::Builder().network(network).name("l").valuesDataType(DataType::FP32).trailingDimensions({1}).batchSize(3).maxTotalValues(7).maxValuesPerRow(4).build();
    EXPECT_THROW((void)BinaryCrossEntropy::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
    EXPECT_THROW((void)BinaryFocalLoss::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);

    Network perOutputNetwork("r10h_per_output");
    Inputs same = makeInputs(perOutputNetwork);
    EXPECT_THROW((void)BinaryCrossEntropy::Builder().network(perOutputNetwork).predictions(same.predictions).labels(same.labels).reportsPerOutputLoss().build(), invalid_argument);
}

TEST(RaggedClassificationR10H, ForwardBackwardUseOnlyActivePrefix) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        runRuntimeCase(Kind::BCE, offsetsDType);
        runRuntimeCase(Kind::FOCAL, offsetsDType, 2.0f, 0.25f);
        runRuntimeCase(Kind::FOCAL, offsetsDType, 0.0f, 0.35f);
    }
}

TEST(RaggedClassificationR10H, FocalGammaZeroIsAlphaWeightedBce) {
    constexpr float alpha = 0.35f;
    for (float label : {0.0f, 1.0f}) {
        for (float logit : {-2.0f, -0.25f, 0.0f, 1.5f}) {
            const auto [bceLoss, bceGradient] = referenceLossGradient(Kind::BCE, logit, label);
            const auto [focalLoss, focalGradient] = referenceLossGradient(Kind::FOCAL, logit, label, 0.0f, alpha);
            const float alphaFactor = label * alpha + (1.0f - label) * (1.0f - alpha);
            EXPECT_NEAR(focalLoss, alphaFactor * bceLoss, 1.0e-6f);
            EXPECT_NEAR(focalGradient, alphaFactor * bceGradient, 1.0e-6f);
        }
    }
}

TEST(RaggedClassificationR10H, SupportLayersSaveLoadWithCanonicalPartition) {
    for (Kind kind : {Kind::BCE, Kind::FOCAL}) {
        Network network("r10h_round_trip");
        Inputs inputs = makeInputs(network, DataType::FP32, DataType::UINT8, DataType::UINT64);
        if (kind == Kind::BCE)
            (void)BinaryCrossEntropy::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsPerExampleLoss().build();
        else
            (void)BinaryFocalLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).focusingParameter(1.5f).alpha(0.35f).reportsPerExampleLoss().build();

        const auto now = chrono::steady_clock::now().time_since_epoch().count();
        const filesystem::path archiveDir = filesystem::temp_directory_path() / (string("thor_r10h_") + to_string(now));
        filesystem::remove_all(archiveDir);
        network.save(archiveDir.string(), /*overwrite=*/true);
        Network loaded("r10h_round_trip");
        ASSERT_NO_THROW(loaded.load(archiveDir.string()));
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
        filesystem::remove_all(archiveDir);
    }
}
