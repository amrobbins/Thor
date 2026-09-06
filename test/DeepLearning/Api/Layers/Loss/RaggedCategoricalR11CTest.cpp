#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Loss/SoftTargetCrossEntropy.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
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

enum class Kind { SOFT_TARGET, CATEGORICAL_FOCAL, CATEGORICAL_CE };

struct Inputs {
    RaggedTensor predictions;
    RaggedTensor labels;
};

Inputs makeInputs(Network& network,
                  DataType offsetsDType = DataType::UINT32,
                  DataType valuesDType = DataType::FP32,
                  uint32_t batchSize = 4,
                  uint64_t maxTotalValues = 8,
                  vector<uint64_t> trailingDimensions = {3}) {
    RaggedTensor predictions = RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .valuesDataType(valuesDType)
                                   .offsetsDataType(offsetsDType)
                                   .trailingDimensions(trailingDimensions)
                                   .batchSize(batchSize)
                                   .maxTotalValues(maxTotalValues)
                                   .maxValuesPerRow(5)
                                   .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(valuesDType)
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

vector<float> softmax3(const float* logits) {
    const float m = max({logits[0], logits[1], logits[2]});
    vector<float> e{exp(logits[0] - m), exp(logits[1] - m), exp(logits[2] - m)};
    const float z = e[0] + e[1] + e[2];
    for (float& v : e) v /= z;
    return e;
}

struct Reference {
    vector<float> rawLoss;
    vector<float> gradient;
};

Reference referenceToken(Kind kind, const float* logits, const float* labels, float gamma = 1.5f, float alpha = 0.35f) {
    const vector<float> p = softmax3(logits);
    vector<float> resultLoss(3);
    vector<float> resultGradient(3);
    const float lossScale = Impl::Loss::getLossScalingFactor();

    if (kind == Kind::SOFT_TARGET || kind == Kind::CATEGORICAL_CE) {
        for (size_t c = 0; c < 3; ++c) {
            resultLoss[c] = -labels[c] * log(p[c]);
            resultGradient[c] = (p[c] - labels[c]) * lossScale;
        }
        return {resultLoss, resultGradient};
    }

    vector<float> dLossDProbability(3);
    vector<float> weighted(3);
    float weightedSum = 0.0f;
    for (size_t c = 0; c < 3; ++c) {
        const float probabilitySafe = max(p[c], 1.0e-7f);
        const float oneMinusProbability = max(1.0f - p[c], 1.0e-7f);
        const float focalWeight = gamma == 0.0f ? 1.0f : pow(oneMinusProbability, gamma);
        resultLoss[c] = -alpha * labels[c] * focalWeight * log(p[c]);
        if (gamma == 0.0f) {
            dLossDProbability[c] = -alpha * labels[c] / probabilitySafe;
        } else {
            const float focalDerivative = gamma * pow(oneMinusProbability, gamma - 1.0f) * log(p[c]);
            dLossDProbability[c] = alpha * labels[c] * (focalDerivative - focalWeight / probabilitySafe);
        }
        weighted[c] = p[c] * dLossDProbability[c];
        weightedSum += weighted[c];
    }
    for (size_t c = 0; c < 3; ++c)
        resultGradient[c] = (weighted[c] - p[c] * weightedSum) * lossScale;
    return {resultLoss, resultGradient};
}

Tensor buildBatchLoss(Kind kind, Network& network, const Inputs& inputs, float gamma = 1.5f, float alpha = 0.35f) {
    if (kind == Kind::SOFT_TARGET)
        return SoftTargetCrossEntropy::Builder()
            .network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .reportsBatchLoss()
            .build()
            .getLoss();
    if (kind == Kind::CATEGORICAL_FOCAL)
        return CategoricalFocalLoss::Builder()
            .network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .focusingParameter(gamma)
            .alpha(alpha)
            .reportsBatchLoss()
            .build()
            .getLoss();
    return CategoricalCrossEntropy::Builder()
        .network(network)
        .predictions(inputs.predictions)
        .labels(inputs.labels)
        .reportsBatchLoss()
        .build()
        .getLoss();
}

RaggedTensor buildRawLoss(Kind kind, Network& network, const Inputs& inputs) {
    if (kind == Kind::SOFT_TARGET)
        return SoftTargetCrossEntropy::Builder()
            .network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .reportsRawLoss()
            .build()
            .getRaggedLoss();
    if (kind == Kind::CATEGORICAL_FOCAL)
        return CategoricalFocalLoss::Builder()
            .network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .reportsRawLoss()
            .build()
            .getRaggedLoss();
    return CategoricalCrossEntropy::Builder()
        .network(network)
        .predictions(inputs.predictions)
        .labels(inputs.labels)
        .reportsRawLoss()
        .build()
        .getRaggedLoss();
}

void runRuntimeCase(Kind kind, DataType offsetsDType) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    Network network("ragged_r11c_runtime");
    Inputs inputs = makeInputs(network, offsetsDType, DataType::FP32, batchSize, maxTotalValues);
    Tensor reportedLoss = buildBatchLoss(kind, network, inputs);
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
    Impl::Tensor predictionValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 3}));
    Impl::Tensor labelValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 3}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));
    float* predictions = predictionValues.getMemPtr<float>();
    float* labels = labelValues.getMemPtr<float>();
    fill(predictions, predictions + maxTotalValues * 3, numeric_limits<float>::quiet_NaN());
    fill(labels, labels + maxTotalValues * 3, numeric_limits<float>::quiet_NaN());

    const vector<float> activePredictions{
        1.0f, -0.5f, 0.25f,
        -1.0f, 0.5f, 1.5f,
        0.0f, 0.0f, 0.0f,
        2.0f, 0.25f, -1.0f,
        -0.25f, 1.25f, 0.5f,
    };
    const vector<float> activeLabels{
        0.70f, 0.20f, 0.10f,
        0.00f, 1.00f, 0.00f,
        0.25f, 0.25f, 0.50f,
        1.00f, 0.00f, 0.00f,
        0.10f, 0.60f, 0.30f,
    };
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

    double lossNumerator = 0.0;
    vector<float> expectedGradient;
    for (size_t token = 0; token < 5; ++token) {
        const Reference reference = referenceToken(kind, &activePredictions[token * 3], &activeLabels[token * 3]);
        for (float value : reference.rawLoss) lossNumerator += value;
        expectedGradient.insert(expectedGradient.end(), reference.gradient.begin(), reference.gradient.end());
    }

    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    EXPECT_NEAR(reported[0], lossNumerator / validExamples, 4.0e-5);

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    ASSERT_EQ(gradient.size(), maxTotalValues * 3);
    ASSERT_EQ(expectedGradient.size(), 15u);
    for (size_t i = 0; i < expectedGradient.size(); ++i)
        EXPECT_NEAR(gradient[i], expectedGradient[i], 8.0e-4f) << "active element " << i;
}

}  // namespace

TEST(RaggedCategoricalR11C, PublicContractsPreservePartitionAndRejectUndefinedGeometry) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::SOFT_TARGET, Kind::CATEGORICAL_FOCAL, Kind::CATEGORICAL_CE}) {
            Network network("r11c_raw");
            Inputs inputs = makeInputs(network, offsetsDType, DataType::FP16);
            RaggedTensor raw = buildRawLoss(kind, network, inputs);
            EXPECT_TRUE(raw.sharesPartitionWith(inputs.predictions));
            EXPECT_EQ(raw.getTrailingDimensions(), (vector<uint64_t>{3}));
            EXPECT_EQ(countLayerType(network, "RaggedCustomLoss"), 1u);

            Network perExampleNetwork("r11c_per_example");
            Inputs perInputs = makeInputs(perExampleNetwork, offsetsDType);
            Tensor perExample;
            if (kind == Kind::SOFT_TARGET)
                perExample = SoftTargetCrossEntropy::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss();
            else if (kind == Kind::CATEGORICAL_FOCAL)
                perExample = CategoricalFocalLoss::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss();
            else
                perExample = CategoricalCrossEntropy::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss();
            EXPECT_EQ(perExample.getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);
        }
    }

    Network mismatchNetwork("r11c_mismatch");
    Inputs same = makeInputs(mismatchNetwork);
    RaggedTensor different = RaggedNetworkInput::Builder()
                                 .network(mismatchNetwork)
                                 .name("different")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({3})
                                 .batchSize(4)
                                 .maxTotalValues(8)
                                 .maxValuesPerRow(5)
                                 .build();
    EXPECT_THROW((void)SoftTargetCrossEntropy::Builder().network(mismatchNetwork).predictions(same.predictions).labels(different).build(), invalid_argument);
    EXPECT_THROW((void)CategoricalFocalLoss::Builder().network(mismatchNetwork).predictions(same.predictions).labels(different).build(), invalid_argument);
    EXPECT_THROW((void)CategoricalCrossEntropy::Builder().network(mismatchNetwork).predictions(same.predictions).labels(different).build(), invalid_argument);

    Network perOutputNetwork("r11c_per_output");
    Inputs perOutputInputs = makeInputs(perOutputNetwork);
    EXPECT_THROW((void)SoftTargetCrossEntropy::Builder().network(perOutputNetwork).predictions(perOutputInputs.predictions).labels(perOutputInputs.labels).reportsPerOutputLoss().build(), invalid_argument);
    EXPECT_THROW((void)CategoricalFocalLoss::Builder().network(perOutputNetwork).predictions(perOutputInputs.predictions).labels(perOutputInputs.labels).reportsPerOutputLoss().build(), invalid_argument);
    EXPECT_THROW((void)CategoricalCrossEntropy::Builder().network(perOutputNetwork).predictions(perOutputInputs.predictions).labels(perOutputInputs.labels).reportsPerOutputLoss().build(), invalid_argument);

    Network rankNetwork("r11c_rank");
    Inputs rankInputs = makeInputs(rankNetwork, DataType::UINT32, DataType::FP32, 4, 8, {2, 3});
    EXPECT_THROW((void)SoftTargetCrossEntropy::Builder().network(rankNetwork).predictions(rankInputs.predictions).labels(rankInputs.labels).build(), logic_error);
    EXPECT_THROW((void)CategoricalFocalLoss::Builder().network(rankNetwork).predictions(rankInputs.predictions).labels(rankInputs.labels).build(), logic_error);
    EXPECT_THROW((void)CategoricalCrossEntropy::Builder().network(rankNetwork).predictions(rankInputs.predictions).labels(rankInputs.labels).build(), invalid_argument);
}

TEST(RaggedCategoricalR11C, ForwardBackwardUseTokenwiseClassAxisAndIgnoreInactiveCapacity) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        runRuntimeCase(Kind::SOFT_TARGET, offsetsDType);
        runRuntimeCase(Kind::CATEGORICAL_FOCAL, offsetsDType);
        runRuntimeCase(Kind::CATEGORICAL_CE, offsetsDType);
    }
}

TEST(RaggedCategoricalR11C, SupportLayersSaveLoadWithCanonicalPartition) {
    for (Kind kind : {Kind::SOFT_TARGET, Kind::CATEGORICAL_FOCAL, Kind::CATEGORICAL_CE}) {
        Network network("r11c_round_trip");
        Inputs inputs = makeInputs(network, DataType::UINT64);
        if (kind == Kind::SOFT_TARGET)
            (void)SoftTargetCrossEntropy::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsPerExampleLoss().build();
        else if (kind == Kind::CATEGORICAL_FOCAL)
            (void)CategoricalFocalLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).focusingParameter(1.25f).alpha(0.4f).reportsPerExampleLoss().build();
        else
            (void)CategoricalCrossEntropy::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsPerExampleLoss().build();

        const auto now = chrono::steady_clock::now().time_since_epoch().count();
        const filesystem::path archiveDir = filesystem::temp_directory_path() / (string("thor_r11c_") + to_string(now));
        filesystem::remove_all(archiveDir);
        network.save(archiveDir.string(), /*overwrite=*/true);
        Network loaded("r11c_round_trip");
        ASSERT_NO_THROW(loaded.load(archiveDir.string()));
        EXPECT_EQ(countLayerType(loaded, "RaggedCustomLoss"), 1u);
        EXPECT_EQ(countLayerType(loaded, "RaggedLossShaper"), 1u);
        ASSERT_EQ(loaded.getLossRootTensors().size(), 1u);
        filesystem::remove_all(archiveDir);
    }
}
