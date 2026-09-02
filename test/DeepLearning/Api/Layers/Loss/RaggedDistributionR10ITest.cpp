#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/PoissonNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
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

enum class Kind { POISSON, TWEEDIE, GAMMA };

struct Inputs {
    RaggedTensor predictions;
    RaggedTensor labels;
    optional<RaggedTensor> dispersion;
};

Inputs makeInputs(Network& network,
                  Kind kind,
                  DataType predictionDType = DataType::FP32,
                  DataType labelDType = DataType::FP32,
                  DataType offsetsDType = DataType::UINT32,
                  vector<uint64_t> trailingDimensions = {1},
                  uint32_t batchSize = 4,
                  uint64_t maxTotalValues = 8,
                  bool withDispersion = false) {
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
    optional<RaggedTensor> dispersion;
    if (kind == Kind::GAMMA && withDispersion) {
        dispersion = RaggedNetworkInput::Builder()
                         .network(network)
                         .name("dispersion")
                         .valuesDataType(predictionDType)
                         .trailingDimensions(trailingDimensions)
                         .partition(predictions)
                         .build();
    }
    return {predictions, labels, dispersion};
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

pair<double, double> poissonReference(double prediction, double label) {
    const double loss = exp(prediction) - label * prediction;
    const double gradient = (exp(prediction) - label) * Impl::Loss::getLossScalingFactor();
    return {loss, gradient};
}

pair<double, double> tweedieReference(double prediction, double label) {
    constexpr double power = 1.5;
    constexpr double eps = 1.0e-6;
    const double mean = max(prediction, eps);
    const double target = max(label, 0.0);
    const double safeTarget = max(target, eps);
    const double oneMinusP = 1.0 - power;
    const double twoMinusP = 2.0 - power;
    const double loss = 2.0 * (pow(safeTarget, twoMinusP) / (oneMinusP * twoMinusP) -
                               target * pow(mean, oneMinusP) / oneMinusP +
                               pow(mean, twoMinusP) / twoMinusP);
    const double gradient = 2.0 * (pow(mean, 1.0 - power) - target * pow(mean, -power)) *
                            Impl::Loss::getLossScalingFactor();
    return {loss, gradient};
}

double gammaLossReference(double meanInput, double target, double dispersion) {
    constexpr double eps = 1.0e-6;
    const double mean = max(meanInput, eps);
    const double phi = max(dispersion, eps);
    const double concentration = 1.0 / phi;
    const double safeTarget = max(target, eps);
    return lgamma(concentration) + concentration * (log(mean) + log(phi)) -
           (concentration - 1.0) * log(safeTarget) + target / (mean * phi);
}

pair<double, double> gammaReference(double meanInput, double target, double dispersion) {
    constexpr double eps = 1.0e-6;
    const double mean = max(meanInput, eps);
    const double phi = max(dispersion, eps);
    const double concentration = 1.0 / phi;
    const double meanGradient = concentration * (1.0 / mean - target / (mean * mean)) *
                                Impl::Loss::getLossScalingFactor();

    // Use a numerical reference for dLoss/dDispersion so this test is independent
    // of Thor's digamma implementation while still qualifying the second ragged
    // differentiable input.
    const double h = 1.0e-4 * max(1.0, phi);
    const double dispersionGradient =
        (gammaLossReference(meanInput, target, phi + h) - gammaLossReference(meanInput, target, phi - h)) /
        (2.0 * h) * Impl::Loss::getLossScalingFactor();
    return {meanGradient, dispersionGradient};
}

Tensor buildBatchLoss(Kind kind, Network& network, const Inputs& inputs, optional<Tensor> weights) {
    if (kind == Kind::POISSON) {
        PoissonNLLLoss::Builder builder;
        builder.network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .logInput(true)
            .full(false)
            .reportsBatchLoss();
        if (weights.has_value()) builder.exampleWeights(weights.value());
        return builder.build().getLoss();
    }
    if (kind == Kind::TWEEDIE) {
        TweedieLoss::Builder builder;
        builder.network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .power(1.5f)
            .reportsBatchLoss();
        if (weights.has_value()) builder.exampleWeights(weights.value());
        return builder.build().getLoss();
    }
    GammaNLLLoss::Builder builder;
    builder.network(network).mean(inputs.predictions).labels(inputs.labels).reportsBatchLoss();
    if (inputs.dispersion.has_value()) builder.dispersion(inputs.dispersion.value());
    if (weights.has_value()) builder.exampleWeights(weights.value());
    return builder.build().getLoss();
}

void runRuntimeCase(Kind kind, DataType offsetsDType, bool weighted) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    Network network("ragged_r10i_runtime");
    Inputs inputs = makeInputs(network,
                               kind,
                               DataType::FP32,
                               DataType::FP32,
                               offsetsDType,
                               {1},
                               batchSize,
                               maxTotalValues,
                               kind == Kind::GAMMA);

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

    Tensor reportedLoss = buildBatchLoss(kind, network, inputs, weights);
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

    vector<float> activePredictions;
    vector<float> activeLabels;
    vector<float> activeDispersion;
    if (kind == Kind::POISSON) {
        activePredictions = {-0.2f, 0.4f, 0.0f, 0.8f, -0.7f};
        activeLabels = {0.0f, 2.0f, 1.0f, 3.0f, 1.0f};
    } else if (kind == Kind::TWEEDIE) {
        activePredictions = {0.5f, 1.25f, 2.0f, 0.75f, 3.0f};
        activeLabels = {0.0f, 1.5f, 4.0f, 0.25f, 2.0f};
    } else {
        activePredictions = {0.7f, 1.4f, 2.2f, 0.9f, 3.1f};
        activeLabels = {0.25f, 1.8f, 1.2f, 2.1f, 4.0f};
        activeDispersion = {0.4f, 0.75f, 1.25f, 0.55f, 1.7f};
    }
    copy(activePredictions.begin(), activePredictions.end(), predictions);
    copy(activeLabels.begin(), activeLabels.end(), labels);
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(predictionValues, offsets, inputs.predictions.getMaxValuesPerRow()));
    batch.insert("labels", labelValues);

    optional<Impl::Tensor> dispersionValues;
    if (kind == Kind::GAMMA) {
        dispersionValues.emplace(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
        float* dispersion = dispersionValues->getMemPtr<float>();
        fill(dispersion, dispersion + maxTotalValues, numeric_limits<float>::quiet_NaN());
        copy(activeDispersion.begin(), activeDispersion.end(), dispersion);
        batch.insert("dispersion", dispersionValues.value());
    }

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
    vector<float> expectedPrimaryGradients;
    vector<float> expectedSecondaryGradients;
    for (size_t i = 0; i < activePredictions.size(); ++i) {
        const double rowWeight = !weighted ? 1.0 : (i < 2 ? 0.5 : 2.0);
        if (kind == Kind::POISSON) {
            auto [loss, gradient] = poissonReference(activePredictions[i], activeLabels[i]);
            numerator += rowWeight * loss;
            expectedPrimaryGradients.push_back(static_cast<float>(rowWeight * gradient));
        } else if (kind == Kind::TWEEDIE) {
            auto [loss, gradient] = tweedieReference(activePredictions[i], activeLabels[i]);
            numerator += rowWeight * loss;
            expectedPrimaryGradients.push_back(static_cast<float>(rowWeight * gradient));
        } else {
            const double loss = gammaLossReference(activePredictions[i], activeLabels[i], activeDispersion[i]);
            auto [meanGradient, dispersionGradient] = gammaReference(activePredictions[i], activeLabels[i], activeDispersion[i]);
            numerator += rowWeight * loss;
            expectedPrimaryGradients.push_back(static_cast<float>(rowWeight * meanGradient));
            expectedSecondaryGradients.push_back(static_cast<float>(rowWeight * dispersionGradient));
        }
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
        EXPECT_NEAR(primaryGradient[i], expectedPrimaryGradients[i], 4.0e-4f) << "active index " << i;

    if (kind == Kind::GAMMA) {
        ASSERT_TRUE(physicalLoss->getSecondaryErrorOutput().has_value());
        const vector<float> secondaryGradient = copyFp32ToHost(physicalLoss->getSecondaryErrorOutput().value());
        ASSERT_EQ(secondaryGradient.size(), maxTotalValues);
        for (size_t i = 0; i < expectedSecondaryGradients.size(); ++i)
            EXPECT_NEAR(secondaryGradient[i], expectedSecondaryGradients[i], 2.0e-3f) << "dispersion active index " << i;
    } else {
        EXPECT_FALSE(physicalLoss->getSecondaryErrorOutput().has_value());
    }
}

}  // namespace

TEST(RaggedDistributionR10I, ReportingContractsAndExactPartition) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::POISSON, Kind::TWEEDIE, Kind::GAMMA}) {
            Network rawNetwork("r10i_raw");
            const DataType labelsDType = kind == Kind::POISSON ? DataType::UINT16 : DataType::FP16;
            Inputs inputs = makeInputs(rawNetwork, kind, DataType::FP16, labelsDType, offsetsDType, {2}, 4, 8, kind == Kind::GAMMA);
            RaggedTensor raw;
            if (kind == Kind::POISSON) {
                PoissonNLLLoss loss = PoissonNLLLoss::Builder()
                                          .network(rawNetwork)
                                          .predictions(inputs.predictions)
                                          .labels(inputs.labels)
                                          .reportsRawLoss()
                                          .build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedLoss();
            } else if (kind == Kind::TWEEDIE) {
                TweedieLoss loss = TweedieLoss::Builder()
                                       .network(rawNetwork)
                                       .predictions(inputs.predictions)
                                       .labels(inputs.labels)
                                       .reportsRawLoss()
                                       .build();
                EXPECT_TRUE(loss.isRagged());
                raw = loss.getRaggedLoss();
            } else {
                GammaNLLLoss loss = GammaNLLLoss::Builder()
                                        .network(rawNetwork)
                                        .mean(inputs.predictions)
                                        .labels(inputs.labels)
                                        .dispersion(inputs.dispersion.value())
                                        .reportsRawLoss()
                                        .build();
                EXPECT_TRUE(loss.isRagged());
                ASSERT_TRUE(loss.getRaggedDispersion().has_value());
                EXPECT_EQ(loss.getRaggedDispersion()->getOffsets(), inputs.predictions.getOffsets());
                raw = loss.getRaggedLoss();
            }
            EXPECT_EQ(raw.getOffsets(), inputs.predictions.getOffsets());
            EXPECT_EQ(raw.getValuesDataType(), DataType::FP16);
            EXPECT_EQ(countLayerType(rawNetwork, "RaggedCustomLoss"), 1u);

            Network perExampleNetwork("r10i_per_example");
            Inputs perInputs = makeInputs(perExampleNetwork, kind, DataType::FP32, DataType::FP32, offsetsDType, {1}, 4, 8, kind == Kind::GAMMA);
            Tensor perExample;
            if (kind == Kind::POISSON)
                perExample = PoissonNLLLoss::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss();
            else if (kind == Kind::TWEEDIE)
                perExample = TweedieLoss::Builder().network(perExampleNetwork).predictions(perInputs.predictions).labels(perInputs.labels).reportsPerExampleLoss().build().getLoss();
            else
                perExample = GammaNLLLoss::Builder().network(perExampleNetwork).mean(perInputs.predictions).labels(perInputs.labels).dispersion(perInputs.dispersion.value()).reportsPerExampleLoss().build().getLoss();
            EXPECT_EQ(perExample.getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);
        }
    }

    Network partitionNetwork("r10i_partition");
    RaggedTensor predictions = RaggedNetworkInput::Builder().network(partitionNetwork).name("p").valuesDataType(DataType::FP32).trailingDimensions({1}).batchSize(3).maxTotalValues(7).maxValuesPerRow(4).build();
    RaggedTensor labels = RaggedNetworkInput::Builder().network(partitionNetwork).name("l").valuesDataType(DataType::FP32).trailingDimensions({1}).batchSize(3).maxTotalValues(7).maxValuesPerRow(4).build();
    EXPECT_THROW((void)PoissonNLLLoss::Builder().network(partitionNetwork).predictions(predictions).labels(labels).build(), invalid_argument);
    EXPECT_THROW((void)TweedieLoss::Builder().network(partitionNetwork).predictions(predictions).labels(labels).build(), invalid_argument);
    EXPECT_THROW((void)GammaNLLLoss::Builder().network(partitionNetwork).mean(predictions).labels(labels).build(), invalid_argument);

    Network perOutputNetwork("r10i_per_output");
    Inputs same = makeInputs(perOutputNetwork, Kind::POISSON);
    EXPECT_THROW((void)PoissonNLLLoss::Builder().network(perOutputNetwork).predictions(same.predictions).labels(same.labels).reportsPerOutputLoss().build(), invalid_argument);
}

TEST(RaggedDistributionR10I, GammaDispersionMustSharePartitionAndGeometry) {
    Network network("r10i_gamma_dispersion_validation");
    Inputs inputs = makeInputs(network, Kind::GAMMA, DataType::FP32, DataType::FP32, DataType::UINT32, {2}, 3, 7, false);
    RaggedTensor differentPartition = RaggedNetworkInput::Builder()
                                          .network(network)
                                          .name("different_dispersion")
                                          .valuesDataType(DataType::FP32)
                                          .trailingDimensions({2})
                                          .batchSize(3)
                                          .maxTotalValues(7)
                                          .maxValuesPerRow(4)
                                          .build();
    EXPECT_THROW((void)GammaNLLLoss::Builder()
                         .network(network)
                         .mean(inputs.predictions)
                         .labels(inputs.labels)
                         .dispersion(differentPartition)
                         .build(),
                 invalid_argument);
}

TEST(RaggedDistributionR10I, ForwardBackwardUseOnlyActivePrefix) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (Kind kind : {Kind::POISSON, Kind::TWEEDIE, Kind::GAMMA}) {
            runRuntimeCase(kind, offsetsDType, false);
            runRuntimeCase(kind, offsetsDType, true);
        }
    }
}

TEST(RaggedDistributionR10I, SupportLayersSaveLoadIncludingGammaSecondaryInput) {
    Network network("r10i_round_trip");
    Inputs inputs = makeInputs(network, Kind::GAMMA, DataType::FP32, DataType::FP32, DataType::UINT64, {1}, 4, 8, true);
    NetworkInput weightInput = NetworkInput::Builder().network(network).name("weights").dimensions({1}).dataType(DataType::FP32).build();
    (void)GammaNLLLoss::Builder()
        .network(network)
        .mean(inputs.predictions)
        .labels(inputs.labels)
        .dispersion(inputs.dispersion.value())
        .exampleWeights(weightInput.getFeatureOutput().value())
        .reportsPerExampleLoss()
        .build();

    const auto now = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archiveDir = filesystem::temp_directory_path() / (string("thor_r10i_") + to_string(now));
    filesystem::remove_all(archiveDir);
    network.save(archiveDir.string(), /*overwrite=*/true);

    Network loaded("r10i_round_trip");
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
    ASSERT_TRUE(loadedRaw->getRaggedSecondaryInput().has_value());
    ASSERT_TRUE(loadedRaw->getRaggedExampleWeights().has_value());
    EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedLabels().getOffsets());
    EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedSecondaryInput()->getOffsets());
    EXPECT_EQ(loadedRaw->getRaggedPredictions().getOffsets(), loadedRaw->getRaggedExampleWeights()->getOffsets());
    filesystem::remove_all(archiveDir);
}
