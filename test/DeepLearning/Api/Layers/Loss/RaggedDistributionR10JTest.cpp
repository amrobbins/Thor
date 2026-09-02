#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/LaplaceNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
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
#include <optional>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
namespace Impl = ThorImplementation;

namespace {

enum class Kind { GAUSSIAN, LAPLACE };

struct Inputs {
    RaggedTensor primary;
    RaggedTensor labels;
    RaggedTensor parameter;
};

Inputs makeInputs(Network& network,
                  Kind kind,
                  DataType offsetsDType = DataType::UINT32,
                  DataType valueDType = DataType::FP32,
                  vector<uint64_t> trailingDimensions = {1},
                  uint32_t batchSize = 4,
                  uint64_t maxTotalValues = 8) {
    (void)kind;
    RaggedTensor primary = RaggedNetworkInput::Builder()
                               .network(network)
                               .name("predictions")
                               .valuesDataType(valueDType)
                               .offsetsDataType(offsetsDType)
                               .trailingDimensions(trailingDimensions)
                               .batchSize(batchSize)
                               .maxTotalValues(maxTotalValues)
                               .maxValuesPerRow(5)
                               .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(valueDType)
                              .trailingDimensions(trailingDimensions)
                              .partition(primary)
                              .build();
    RaggedTensor parameter = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("parameter")
                                 .valuesDataType(valueDType)
                                 .trailingDimensions(trailingDimensions)
                                 .partition(primary)
                                 .build();
    return {primary, labels, parameter};
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

struct Reference {
    double loss;
    double primaryGradient;
    double parameterGradient;
};

Reference gaussianReference(double meanInput, double target, double varianceInput, bool logVariance, bool full) {
    constexpr double eps = 1.0e-6;
    constexpr double logTwoPi = 1.837877066409345483560659472811;
    const double diff = meanInput - target;
    const double variance = logVariance ? exp(varianceInput) : max(varianceInput, eps);
    const double loss = 0.5 * (log(variance) + diff * diff / variance) + (full ? 0.5 * logTwoPi : 0.0);
    const double meanGradient = diff / variance * Impl::Loss::getLossScalingFactor();
    const double varianceGradient = (logVariance ? 0.5 * (1.0 - diff * diff / variance)
                                                 : 0.5 * (1.0 / variance - diff * diff / (variance * variance))) *
                                    Impl::Loss::getLossScalingFactor();
    return {loss, meanGradient, varianceGradient};
}

Reference laplaceReference(double location, double target, double scaleInput, bool logScale) {
    constexpr double eps = 1.0e-8;
    constexpr double logTwo = 0.693147180559945309417232121458;
    const double diff = location - target;
    const double absDiff = abs(diff);
    const double scale = logScale ? exp(scaleInput) : max(scaleInput, eps);
    const double sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
    const double loss = logTwo + log(scale) + absDiff / scale;
    const double locationGradient = sign / scale * Impl::Loss::getLossScalingFactor();
    const double scaleGradient = (logScale ? 1.0 - absDiff / scale
                                           : 1.0 / scale - absDiff / (scale * scale)) *
                                 Impl::Loss::getLossScalingFactor();
    return {loss, locationGradient, scaleGradient};
}

Tensor buildBatchLoss(Kind kind,
                      Network& network,
                      const Inputs& inputs,
                      bool logParameter,
                      optional<Tensor> weights) {
    if (kind == Kind::GAUSSIAN) {
        GaussianNLLLoss::Builder builder;
        builder.network(network)
            .mean(inputs.primary)
            .target(inputs.labels)
            .variance(inputs.parameter)
            .logVariance(logParameter)
            .full(true)
            .reportsBatchLoss();
        if (weights.has_value()) builder.exampleWeights(weights.value());
        return builder.build().getLoss();
    }

    LaplaceNLLLoss::Builder builder;
    builder.network(network)
        .location(inputs.primary)
        .target(inputs.labels)
        .scale(inputs.parameter)
        .logScale(logParameter)
        .reportsBatchLoss();
    if (weights.has_value()) builder.exampleWeights(weights.value());
    return builder.build().getLoss();
}

void runRuntimeCase(Kind kind, DataType offsetsDType, bool logParameter, bool weighted) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    Network network("ragged_r10j_runtime");
    Inputs inputs = makeInputs(network, kind, offsetsDType);

    optional<Tensor> weights;
    if (weighted) {
        NetworkInput input = NetworkInput::Builder()
                                 .network(network)
                                 .name("weights")
                                 .dimensions({1})
                                 .dataType(DataType::FP32)
                                 .build();
        weights = input.getFeatureOutput().value();
    }

    Tensor reportedLoss = buildBatchLoss(kind, network, inputs, logParameter, weights);
    (void)NetworkOutput::Builder().network(network).name("loss").inputTensor(reportedLoss).dataType(DataType::FP32).build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(batchSize, initializationDone, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initializationDone) event.synchronize();

    shared_ptr<Impl::RaggedCustomLoss> physicalLoss;
    for (const shared_ptr<Impl::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
        auto candidate = dynamic_pointer_cast<Impl::RaggedCustomLoss>(layer);
        if (candidate != nullptr) {
            physicalLoss = candidate;
            break;
        }
    }
    ASSERT_NE(physicalLoss, nullptr);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor predictionValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor labelValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor parameterValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));

    float* predictions = predictionValues.getMemPtr<float>();
    float* labels = labelValues.getMemPtr<float>();
    float* parameters = parameterValues.getMemPtr<float>();
    fill(predictions, predictions + maxTotalValues, numeric_limits<float>::quiet_NaN());
    fill(labels, labels + maxTotalValues, numeric_limits<float>::quiet_NaN());
    fill(parameters, parameters + maxTotalValues, numeric_limits<float>::quiet_NaN());

    const vector<float> activePredictions = {0.5f, 1.2f, 2.0f, -0.4f, 3.0f};
    const vector<float> activeLabels = {0.25f, 1.2f, 1.5f, 0.4f, 4.0f};
    const vector<float> positiveParameters = kind == Kind::GAUSSIAN
                                                  ? vector<float>{0.4f, 1.25f, 0.8f, 2.0f, 0.55f}
                                                  : vector<float>{0.3f, 1.1f, 0.75f, 1.8f, 0.45f};
    vector<float> activeParameters(positiveParameters.size());
    transform(positiveParameters.begin(), positiveParameters.end(), activeParameters.begin(), [&](float value) {
        return logParameter ? log(value) : value;
    });
    copy(activePredictions.begin(), activePredictions.end(), predictions);
    copy(activeLabels.begin(), activeLabels.end(), labels);
    copy(activeParameters.begin(), activeParameters.end(), parameters);
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, inputs.primary.getMaxValuesPerRow()));
    batch.insert("labels", labelValues);
    batch.insert("parameter", parameterValues);
    if (weighted) {
        Impl::Tensor hostWeights(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1}));
        float* values = hostWeights.getMemPtr<float>();
        values[0] = 0.5f;
        values[1] = 7.0f;   // valid empty row
        values[2] = 2.0f;
        values[3] = 99.0f;  // invalid canonical-empty tail row
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
    vector<float> expectedPrimaryGradients;
    vector<float> expectedParameterGradients;
    for (size_t i = 0; i < activePredictions.size(); ++i) {
        const double rowWeight = !weighted ? 1.0 : (i < 2 ? 0.5 : 2.0);
        const Reference reference = kind == Kind::GAUSSIAN
                                        ? gaussianReference(activePredictions[i], activeLabels[i], activeParameters[i], logParameter, true)
                                        : laplaceReference(activePredictions[i], activeLabels[i], activeParameters[i], logParameter);
        numerator += rowWeight * reference.loss;
        expectedPrimaryGradients.push_back(static_cast<float>(rowWeight * reference.primaryGradient));
        expectedParameterGradients.push_back(static_cast<float>(rowWeight * reference.parameterGradient));
    }

    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    const double expectedBatchLoss = numerator / validExamples;
    const double fp32Tolerance = max(1.0e-4, 8.0 * static_cast<double>(numeric_limits<float>::epsilon()) * abs(expectedBatchLoss));
    EXPECT_NEAR(reported[0], expectedBatchLoss, fp32Tolerance);

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> primaryGradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    ASSERT_EQ(primaryGradient.size(), maxTotalValues);
    for (size_t i = 0; i < expectedPrimaryGradients.size(); ++i)
        EXPECT_NEAR(primaryGradient[i], expectedPrimaryGradients[i], 5.0e-4f) << "primary active index " << i;

    ASSERT_TRUE(physicalLoss->getSecondaryErrorOutput().has_value());
    const vector<float> parameterGradient = copyFp32ToHost(physicalLoss->getSecondaryErrorOutput().value());
    ASSERT_EQ(parameterGradient.size(), maxTotalValues);
    for (size_t i = 0; i < expectedParameterGradients.size(); ++i)
        EXPECT_NEAR(parameterGradient[i], expectedParameterGradients[i], 5.0e-4f) << "parameter active index " << i;
}

}  // namespace

TEST(RaggedDistributionR10J, ReportingContractsAndExactPartition) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::GAUSSIAN, Kind::LAPLACE}) {
            Network rawNetwork("r10j_raw");
            Inputs inputs = makeInputs(rawNetwork, kind, offsetsDType, DataType::FP16, {2});
            RaggedTensor raw;
            if (kind == Kind::GAUSSIAN) {
                GaussianNLLLoss loss = GaussianNLLLoss::Builder()
                                           .network(rawNetwork)
                                           .mean(inputs.primary)
                                           .target(inputs.labels)
                                           .variance(inputs.parameter)
                                           .reportsRawLoss()
                                           .build();
                EXPECT_TRUE(loss.isRagged());
                EXPECT_EQ(loss.getRaggedVariance().getOffsets(), inputs.primary.getOffsets());
                raw = loss.getRaggedLoss();
            } else {
                LaplaceNLLLoss loss = LaplaceNLLLoss::Builder()
                                          .network(rawNetwork)
                                          .location(inputs.primary)
                                          .target(inputs.labels)
                                          .scale(inputs.parameter)
                                          .reportsRawLoss()
                                          .build();
                EXPECT_TRUE(loss.isRagged());
                EXPECT_EQ(loss.getRaggedScale().getOffsets(), inputs.primary.getOffsets());
                raw = loss.getRaggedLoss();
            }
            EXPECT_EQ(raw.getOffsets(), inputs.primary.getOffsets());
            EXPECT_EQ(raw.getValuesDataType(), DataType::FP16);
            EXPECT_EQ(countLayerType(rawNetwork, "RaggedCustomLoss"), 1u);

            Network perExampleNetwork("r10j_per_example");
            Inputs perInputs = makeInputs(perExampleNetwork, kind, offsetsDType);
            Tensor perExample = kind == Kind::GAUSSIAN
                                    ? GaussianNLLLoss::Builder()
                                          .network(perExampleNetwork)
                                          .mean(perInputs.primary)
                                          .target(perInputs.labels)
                                          .variance(perInputs.parameter)
                                          .reportsPerExampleLoss()
                                          .build()
                                          .getLoss()
                                    : LaplaceNLLLoss::Builder()
                                          .network(perExampleNetwork)
                                          .location(perInputs.primary)
                                          .target(perInputs.labels)
                                          .scale(perInputs.parameter)
                                          .reportsPerExampleLoss()
                                          .build()
                                          .getLoss();
            EXPECT_EQ(perExample.getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);
        }
    }

    Network partitionNetwork("r10j_partition");
    Inputs inputs = makeInputs(partitionNetwork, Kind::GAUSSIAN);
    RaggedTensor different = RaggedNetworkInput::Builder()
                                 .network(partitionNetwork)
                                 .name("different")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({1})
                                 .batchSize(4)
                                 .maxTotalValues(8)
                                 .maxValuesPerRow(5)
                                 .build();
    EXPECT_THROW((void)GaussianNLLLoss::Builder()
                         .network(partitionNetwork)
                         .mean(inputs.primary)
                         .target(inputs.labels)
                         .variance(different)
                         .build(),
                 invalid_argument);
    EXPECT_THROW((void)LaplaceNLLLoss::Builder()
                         .network(partitionNetwork)
                         .location(inputs.primary)
                         .target(inputs.labels)
                         .scale(different)
                         .build(),
                 invalid_argument);

    EXPECT_THROW((void)GaussianNLLLoss::Builder()
                         .network(partitionNetwork)
                         .mean(inputs.primary)
                         .target(inputs.labels)
                         .variance(inputs.parameter)
                         .reportsPerOutputLoss()
                         .build(),
                 invalid_argument);
    EXPECT_THROW((void)LaplaceNLLLoss::Builder()
                         .network(partitionNetwork)
                         .location(inputs.primary)
                         .target(inputs.labels)
                         .scale(inputs.parameter)
                         .reportsPerOutputLoss()
                         .build(),
                 invalid_argument);
}

TEST(RaggedDistributionR10J, ForwardBackwardQualifiesDirectAndLogParameters) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::GAUSSIAN, Kind::LAPLACE}) {
            for (bool logParameter : {false, true}) {
                runRuntimeCase(kind, offsetsDType, logParameter, false);
                runRuntimeCase(kind, offsetsDType, logParameter, true);
            }
        }
    }
}

TEST(RaggedDistributionR10J, SupportLayersSaveLoadPreserveSecondaryInputAndWeights) {
    for (Kind kind : {Kind::GAUSSIAN, Kind::LAPLACE}) {
        const string networkName = kind == Kind::GAUSSIAN ? "r10j_gaussian_round_trip" : "r10j_laplace_round_trip";
        Network network(networkName);
        Inputs inputs = makeInputs(network, kind, DataType::UINT64);
        NetworkInput weightInput = NetworkInput::Builder()
                                       .network(network)
                                       .name("weights")
                                       .dimensions({1})
                                       .dataType(DataType::FP32)
                                       .build();
        if (kind == Kind::GAUSSIAN) {
            (void)GaussianNLLLoss::Builder()
                .network(network)
                .mean(inputs.primary)
                .target(inputs.labels)
                .variance(inputs.parameter)
                .exampleWeights(weightInput.getFeatureOutput().value())
                .reportsPerExampleLoss()
                .build();
        } else {
            (void)LaplaceNLLLoss::Builder()
                .network(network)
                .location(inputs.primary)
                .target(inputs.labels)
                .scale(inputs.parameter)
                .exampleWeights(weightInput.getFeatureOutput().value())
                .reportsPerExampleLoss()
                .build();
        }

        const auto now = chrono::steady_clock::now().time_since_epoch().count();
        const filesystem::path archiveDir = filesystem::temp_directory_path() /
                                            (string("thor_r10j_") + to_string(static_cast<int>(kind)) + "_" + to_string(now));
        filesystem::remove_all(archiveDir);
        network.save(archiveDir.string(), /*overwrite=*/true);

        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archiveDir.string()));
        EXPECT_EQ(countLayerType(loaded, "RaggedCustomLoss"), 1u);
        EXPECT_EQ(countLayerType(loaded, "RaggedLossShaper"), 1u);

        shared_ptr<RaggedCustomLoss> loadedRaw;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            loadedRaw = dynamic_pointer_cast<RaggedCustomLoss>(loaded.getLayer(i));
            if (loadedRaw != nullptr) break;
        }
        ASSERT_NE(loadedRaw, nullptr);
        ASSERT_TRUE(loadedRaw->getRaggedSecondaryInput().has_value());
        ASSERT_TRUE(loadedRaw->getRaggedExampleWeights().has_value());
        EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedLabels().getOffsets());
        EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedSecondaryInput()->getOffsets());
        EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedExampleWeights()->getOffsets());
        filesystem::remove_all(archiveDir);
    }
}
