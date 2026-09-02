#include "DeepLearning/Api/Layers/Loss/NegativeBinomialNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/StudentTNLLLoss.h"
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
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
namespace Impl = ThorImplementation;

namespace {

struct Inputs {
    RaggedTensor primary;
    RaggedTensor labels;
    RaggedTensor parameter;
};

Inputs makeInputs(Network& network, const string& prefix, DataType offsetsDType = DataType::UINT32) {
    RaggedTensor primary = RaggedNetworkInput::Builder()
                               .network(network)
                               .name(prefix + "_primary")
                               .valuesDataType(DataType::FP32)
                               .offsetsDataType(offsetsDType)
                               .trailingDimensions({2})
                               .batchSize(4)
                               .maxTotalValues(9)
                               .maxValuesPerRow(5)
                               .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name(prefix + "_labels")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({2})
                              .partition(primary)
                              .build();
    RaggedTensor parameter = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name(prefix + "_parameter")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({2})
                                 .partition(primary)
                                 .build();
    return {primary, labels, parameter};
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

double digammaReference(double x) {
    double result = 0.0;
    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    result += log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0));
    return result;
}

struct StudentReference {
    double loss;
    double locationGradient;
    double logScaleGradient;
    double logDegreesOfFreedomGradient;
};

StudentReference studentReference(double location,
                                  double target,
                                  double logScale,
                                  double learnedLogDof,
                                  double minimumDegreesOfFreedom) {
    constexpr double logPi = 1.1447298858494001741434273513531;
    const double nuExcess = exp(learnedLogDof);
    const double nu = minimumDegreesOfFreedom + nuExcess;
    const double inverseScale = exp(-logScale);
    const double standardizedResidual = (location - target) * inverseScale;
    const double residualSquared = standardizedResidual * standardizedResidual;
    const double denominator = nu + residualSquared;
    const double halfNu = 0.5 * nu;
    const double halfNuPlusOne = 0.5 * (nu + 1.0);
    const double loss = logScale + lgamma(halfNu) - lgamma(halfNuPlusOne) +
                        0.5 * (log(nu) + logPi) + halfNuPlusOne * log1p(residualSquared / nu);
    const double locationGradient = (nu + 1.0) * standardizedResidual * inverseScale / denominator;
    const double logScaleGradient = 1.0 - (nu + 1.0) * residualSquared / denominator;
    const double dLossDNu = 0.5 * (digammaReference(halfNu) - digammaReference(halfNuPlusOne) + 1.0 / nu +
                                   log1p(residualSquared / nu) -
                                   (nu + 1.0) * residualSquared / (nu * denominator));
    return {loss, locationGradient, logScaleGradient, nuExcess * dLossDNu};
}

struct NegativeBinomialReference {
    double loss;
    double meanGradient;
    double dispersionGradient;
};

NegativeBinomialReference negativeBinomialReference(double meanInput,
                                                    double label,
                                                    double dispersionInput,
                                                    bool logMean,
                                                    bool logDispersion) {
    constexpr double eps = 1.0e-8;
    const double mean = logMean ? exp(meanInput) : max(meanInput, eps);
    const double dispersion = logDispersion ? exp(dispersionInput) : max(dispersionInput, eps);
    const double logMeanValue = logMean ? meanInput : log(mean);
    const double logDispersionValue = logDispersion ? dispersionInput : log(dispersion);
    const double concentration = 1.0 / dispersion;
    const double onePlusDispersionMean = 1.0 + dispersion * mean;
    const double logOnePlusDispersionMean = log1p(dispersion * mean);
    const double loss = lgamma(concentration) + lgamma(label + 1.0) - lgamma(label + concentration) +
                        (concentration + label) * logOnePlusDispersionMean - label * logDispersionValue -
                        label * logMeanValue;
    const double logMeanGradient = (mean - label) / onePlusDispersionMean;
    const double meanGradient = logMean ? logMeanGradient : logMeanGradient / mean;
    const double concentrationTerm = digammaReference(concentration) - digammaReference(label + concentration) +
                                     logOnePlusDispersionMean;
    const double logDispersionGradient = -concentration * concentrationTerm +
                                         (mean - label) / onePlusDispersionMean;
    const double dispersionGradient = logDispersion ? logDispersionGradient : logDispersionGradient / dispersion;
    return {loss, meanGradient, dispersionGradient};
}

shared_ptr<Impl::RaggedCustomLoss> findPhysicalRaggedCustomLoss(const shared_ptr<PlacedNetwork>& placed) {
    for (const shared_ptr<Impl::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
        auto candidate = dynamic_pointer_cast<Impl::RaggedCustomLoss>(layer);
        if (candidate != nullptr) return candidate;
    }
    return nullptr;
}

void runStudentRuntimeCase(DataType offsetsDType) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    constexpr float minimumDof = 2.0f;
    // Runtime numerical qualification uses scalar trailing width so that each active
    // packed value corresponds to exactly one reference contribution.
    Network scalarNetwork("r10k_student_runtime_scalar");
    Inputs scalarInputs;
    scalarInputs.primary = RaggedNetworkInput::Builder()
                               .network(scalarNetwork)
                               .name("location")
                               .valuesDataType(DataType::FP32)
                               .offsetsDataType(offsetsDType)
                               .trailingDimensions({1})
                               .batchSize(batchSize)
                               .maxTotalValues(maxTotalValues)
                               .maxValuesPerRow(5)
                               .build();
    scalarInputs.labels = RaggedNetworkInput::Builder()
                              .network(scalarNetwork)
                              .name("target")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({1})
                              .partition(scalarInputs.primary)
                              .build();
    scalarInputs.parameter = RaggedNetworkInput::Builder()
                                 .network(scalarNetwork)
                                 .name("log_scale")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({1})
                                 .partition(scalarInputs.primary)
                                 .build();
    RaggedTensor scalarLogDof = RaggedNetworkInput::Builder()
                                    .network(scalarNetwork)
                                    .name("log_dof")
                                    .valuesDataType(DataType::FP32)
                                    .trailingDimensions({1})
                                    .partition(scalarInputs.primary)
                                    .build();
    NetworkInput weightInput = NetworkInput::Builder()
                                   .network(scalarNetwork)
                                   .name("weights")
                                   .dimensions({1})
                                   .dataType(DataType::FP32)
                                   .build();
    Tensor loss = StudentTNLLLoss::Builder()
                      .network(scalarNetwork)
                      .location(scalarInputs.primary)
                      .logScale(scalarInputs.parameter)
                      .target(scalarInputs.labels)
                      .logDegreesOfFreedom(scalarLogDof)
                      .minimumDegreesOfFreedom(minimumDof)
                      .exampleWeights(weightInput.getFeatureOutput().value())
                      .reportsBatchLoss()
                      .build()
                      .getLoss();
    (void)NetworkOutput::Builder().network(scalarNetwork).name("loss").inputTensor(loss).dataType(DataType::FP32).build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = scalarNetwork.place(batchSize, initializationDone, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initializationDone) event.synchronize();
    shared_ptr<Impl::RaggedCustomLoss> physicalLoss = findPhysicalRaggedCustomLoss(placed);
    ASSERT_NE(physicalLoss, nullptr);
    ASSERT_EQ(physicalLoss->getNumSecondaryInputs(), 2u);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor locationValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor targetValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor scaleValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor dofValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));
    for (Impl::Tensor* tensor : {&locationValues, &targetValues, &scaleValues, &dofValues})
        fill(tensor->getMemPtr<float>(), tensor->getMemPtr<float>() + maxTotalValues, numeric_limits<float>::quiet_NaN());

    const vector<float> locations = {0.5f, 1.2f, 2.0f, -0.4f, 3.0f};
    const vector<float> targets = {0.25f, 1.2f, 1.5f, 0.4f, 4.0f};
    const vector<float> logScales = {log(0.7f), log(1.1f), log(0.8f), log(1.4f), log(0.55f)};
    const vector<float> logDofs = {log(2.0f), log(3.0f), log(5.0f), log(1.5f), log(4.0f)};
    copy(locations.begin(), locations.end(), locationValues.getMemPtr<float>());
    copy(targets.begin(), targets.end(), targetValues.getMemPtr<float>());
    copy(logScales.begin(), logScales.end(), scaleValues.getMemPtr<float>());
    copy(logDofs.begin(), logDofs.end(), dofValues.getMemPtr<float>());
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Impl::Tensor hostWeights(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1}));
    float* weights = hostWeights.getMemPtr<float>();
    weights[0] = 0.5f;
    weights[1] = 7.0f;   // valid empty row
    weights[2] = 2.0f;
    weights[3] = 99.0f;  // invalid canonical-empty tail row

    Batch batch;
    batch.insert("location", Impl::RaggedTensor(locationValues, offsets, scalarInputs.primary.getMaxValuesPerRow()));
    batch.insert("target", targetValues);
    batch.insert("log_scale", scaleValues);
    batch.insert("log_dof", dofValues);
    batch.insert("weights", hostWeights);
    batch.setValidExampleCount(validExamples);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    outputReadyEvents.at("loss").synchronize();
    placed->synchronize();

    double numerator = 0.0;
    vector<float> expectedLocationGradients;
    vector<float> expectedScaleGradients;
    vector<float> expectedDofGradients;
    for (size_t i = 0; i < locations.size(); ++i) {
        const double rowWeight = i < 2 ? 0.5 : 2.0;
        const StudentReference reference = studentReference(locations[i], targets[i], logScales[i], logDofs[i], minimumDof);
        numerator += rowWeight * reference.loss;
        const double scale = rowWeight * Impl::Loss::getLossScalingFactor();
        expectedLocationGradients.push_back(static_cast<float>(scale * reference.locationGradient));
        expectedScaleGradients.push_back(static_cast<float>(scale * reference.logScaleGradient));
        expectedDofGradients.push_back(static_cast<float>(scale * reference.logDegreesOfFreedomGradient));
    }
    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    EXPECT_NEAR(reported[0], numerator / validExamples, 1.5e-4);

    const vector<float> locationGradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    const vector<float> scaleGradient = copyFp32ToHost(physicalLoss->getSecondaryErrorOutput(0).value());
    const vector<float> dofGradient = copyFp32ToHost(physicalLoss->getSecondaryErrorOutput(1).value());
    for (size_t i = 0; i < locations.size(); ++i) {
        EXPECT_NEAR(locationGradient[i], expectedLocationGradients[i], 7.5e-4f) << "location active index " << i;
        EXPECT_NEAR(scaleGradient[i], expectedScaleGradients[i], 7.5e-4f) << "log_scale active index " << i;
        EXPECT_NEAR(dofGradient[i], expectedDofGradients[i], 1.5e-3f) << "log_dof active index " << i;
    }
}

void runNegativeBinomialRuntimeCase(DataType offsetsDType, bool logMean, bool logDispersion) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    Network network("r10k_nb_runtime");
    RaggedTensor mean = RaggedNetworkInput::Builder()
                            .network(network)
                            .name("mean")
                            .valuesDataType(DataType::FP32)
                            .offsetsDataType(offsetsDType)
                            .trailingDimensions({1})
                            .batchSize(batchSize)
                            .maxTotalValues(maxTotalValues)
                            .maxValuesPerRow(5)
                            .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({1})
                              .partition(mean)
                              .build();
    RaggedTensor dispersion = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("dispersion")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({1})
                                  .partition(mean)
                                  .build();
    NetworkInput weightInput = NetworkInput::Builder()
                                   .network(network)
                                   .name("weights")
                                   .dimensions({1})
                                   .dataType(DataType::FP32)
                                   .build();
    Tensor loss = NegativeBinomialNLLLoss::Builder()
                      .network(network)
                      .mean(mean)
                      .dispersion(dispersion)
                      .labels(labels)
                      .logMean(logMean)
                      .logDispersion(logDispersion)
                      .exampleWeights(weightInput.getFeatureOutput().value())
                      .reportsBatchLoss()
                      .build()
                      .getLoss();
    (void)NetworkOutput::Builder().network(network).name("loss").inputTensor(loss).dataType(DataType::FP32).build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(batchSize, initializationDone, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initializationDone) event.synchronize();
    shared_ptr<Impl::RaggedCustomLoss> physicalLoss = findPhysicalRaggedCustomLoss(placed);
    ASSERT_NE(physicalLoss, nullptr);
    ASSERT_EQ(physicalLoss->getNumSecondaryInputs(), 1u);

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor meanValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor labelValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor dispersionValues(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, 1}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {batchSize + 1}));
    for (Impl::Tensor* tensor : {&meanValues, &labelValues, &dispersionValues})
        fill(tensor->getMemPtr<float>(), tensor->getMemPtr<float>() + maxTotalValues, numeric_limits<float>::quiet_NaN());

    const vector<float> positiveMeans = {1.5f, 2.0f, 4.0f, 0.75f, 6.0f};
    const vector<float> activeLabels = {0.0f, 1.0f, 3.0f, 2.0f, 7.0f};
    const vector<float> positiveDispersions = {0.4f, 0.7f, 0.25f, 1.2f, 0.55f};
    vector<float> activeMeans(positiveMeans.size());
    vector<float> activeDispersions(positiveDispersions.size());
    transform(positiveMeans.begin(), positiveMeans.end(), activeMeans.begin(), [&](float value) { return logMean ? log(value) : value; });
    transform(positiveDispersions.begin(), positiveDispersions.end(), activeDispersions.begin(),
              [&](float value) { return logDispersion ? log(value) : value; });
    copy(activeMeans.begin(), activeMeans.end(), meanValues.getMemPtr<float>());
    copy(activeLabels.begin(), activeLabels.end(), labelValues.getMemPtr<float>());
    copy(activeDispersions.begin(), activeDispersions.end(), dispersionValues.getMemPtr<float>());
    writeOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

    Impl::Tensor hostWeights(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 1}));
    float* weights = hostWeights.getMemPtr<float>();
    weights[0] = 0.5f;
    weights[1] = 7.0f;
    weights[2] = 2.0f;
    weights[3] = 99.0f;

    Batch batch;
    batch.insert("mean", Impl::RaggedTensor(meanValues, offsets, mean.getMaxValuesPerRow()));
    batch.insert("labels", labelValues);
    batch.insert("dispersion", dispersionValues);
    batch.insert("weights", hostWeights);
    batch.setValidExampleCount(validExamples);

    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    outputReadyEvents.at("loss").synchronize();
    placed->synchronize();

    double numerator = 0.0;
    vector<float> expectedMeanGradients;
    vector<float> expectedDispersionGradients;
    for (size_t i = 0; i < activeMeans.size(); ++i) {
        const double rowWeight = i < 2 ? 0.5 : 2.0;
        const NegativeBinomialReference reference = negativeBinomialReference(
            activeMeans[i], activeLabels[i], activeDispersions[i], logMean, logDispersion);
        numerator += rowWeight * reference.loss;
        const double scale = rowWeight * Impl::Loss::getLossScalingFactor();
        expectedMeanGradients.push_back(static_cast<float>(scale * reference.meanGradient));
        expectedDispersionGradients.push_back(static_cast<float>(scale * reference.dispersionGradient));
    }
    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    EXPECT_NEAR(reported[0], numerator / validExamples, 2.0e-4);

    const vector<float> meanGradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    const vector<float> dispersionGradient = copyFp32ToHost(physicalLoss->getSecondaryErrorOutput(0).value());
    for (size_t i = 0; i < activeMeans.size(); ++i) {
        EXPECT_NEAR(meanGradient[i], expectedMeanGradients[i], 1.0e-3f) << "mean active index " << i;
        EXPECT_NEAR(dispersionGradient[i], expectedDispersionGradients[i], 2.0e-3f) << "dispersion active index " << i;
    }
}

shared_ptr<RaggedCustomLoss> findRaggedCustomLoss(Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto layer = dynamic_pointer_cast<RaggedCustomLoss>(network.getLayer(i));
        if (layer != nullptr) return layer;
    }
    return nullptr;
}

}  // namespace

TEST(RaggedDistributionR10K, StudentTSupportsFixedAndLearnedDegreesOfFreedomWithExactPartition) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network fixedNetwork("r10k_student_fixed");
        Inputs fixedInputs = makeInputs(fixedNetwork, "fixed", offsetsDType);
        StudentTNLLLoss fixed = StudentTNLLLoss::Builder()
                                    .network(fixedNetwork)
                                    .location(fixedInputs.primary)
                                    .logScale(fixedInputs.parameter)
                                    .target(fixedInputs.labels)
                                    .degreesOfFreedom(4.5f)
                                    .reportsRawLoss()
                                    .build();
        EXPECT_TRUE(fixed.isRagged());
        EXPECT_EQ(fixed.getRaggedLoss().getOffsets(), fixedInputs.primary.getOffsets());
        shared_ptr<RaggedCustomLoss> fixedRaw = findRaggedCustomLoss(fixedNetwork);
        ASSERT_NE(fixedRaw, nullptr);
        EXPECT_EQ(fixedRaw->getRaggedSecondaryInputs().size(), 1u);

        Network learnedNetwork("r10k_student_learned");
        Inputs learnedInputs = makeInputs(learnedNetwork, "learned", offsetsDType);
        RaggedTensor learnedLogDf = RaggedNetworkInput::Builder()
                                        .network(learnedNetwork)
                                        .name("learned_log_df")
                                        .valuesDataType(DataType::FP32)
                                        .trailingDimensions({2})
                                        .partition(learnedInputs.primary)
                                        .build();
        Tensor weights = NetworkInput::Builder()
                             .network(learnedNetwork)
                             .name("weights")
                             .dimensions({1})
                             .dataType(DataType::FP16)
                             .build()
                             .getFeatureOutput()
                             .value();
        StudentTNLLLoss learned = StudentTNLLLoss::Builder()
                                      .network(learnedNetwork)
                                      .location(learnedInputs.primary)
                                      .logScale(learnedInputs.parameter)
                                      .target(learnedInputs.labels)
                                      .logDegreesOfFreedom(learnedLogDf)
                                      .minimumDegreesOfFreedom(2.0f)
                                      .exampleWeights(weights)
                                      .reportsRawLoss()
                                      .build();
        EXPECT_TRUE(learned.isRagged());
        ASSERT_TRUE(learned.getRaggedLearnedLogDegreesOfFreedom().has_value());
        EXPECT_EQ(learned.getRaggedLearnedLogDegreesOfFreedom()->getOffsets(), learnedInputs.primary.getOffsets());
        shared_ptr<RaggedCustomLoss> learnedRaw = findRaggedCustomLoss(learnedNetwork);
        ASSERT_NE(learnedRaw, nullptr);
        EXPECT_EQ(learnedRaw->getRaggedSecondaryInputs().size(), 2u);
    }
}

TEST(RaggedDistributionR10K, NegativeBinomialSupportsRaggedParametersAndReporting) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r10k_negative_binomial");
        Inputs inputs = makeInputs(network, "nb", offsetsDType);
        NegativeBinomialNLLLoss loss = NegativeBinomialNLLLoss::Builder()
                                           .network(network)
                                           .mean(inputs.primary)
                                           .dispersion(inputs.parameter)
                                           .labels(inputs.labels)
                                           .logMean(false)
                                           .logDispersion(false)
                                           .reportsRawLoss()
                                           .build();
        EXPECT_TRUE(loss.isRagged());
        EXPECT_EQ(loss.getRaggedDispersion().getOffsets(), inputs.primary.getOffsets());
        EXPECT_EQ(loss.getRaggedLoss().getOffsets(), inputs.primary.getOffsets());
        shared_ptr<RaggedCustomLoss> raw = findRaggedCustomLoss(network);
        ASSERT_NE(raw, nullptr);
        EXPECT_EQ(raw->getRaggedSecondaryInputs().size(), 1u);
    }
}

TEST(RaggedDistributionR10K, RejectsPerOutputAndMismatchedTokenVaryingPartitions) {
    Network network("r10k_reject");
    Inputs inputs = makeInputs(network, "base");
    RaggedTensor different = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("different")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({2})
                                 .batchSize(4)
                                 .maxTotalValues(9)
                                 .maxValuesPerRow(5)
                                 .build();

    EXPECT_THROW((void)StudentTNLLLoss::Builder()
                         .network(network)
                         .location(inputs.primary)
                         .logScale(inputs.parameter)
                         .target(inputs.labels)
                         .logDegreesOfFreedom(different)
                         .build(),
                 invalid_argument);
    EXPECT_THROW((void)NegativeBinomialNLLLoss::Builder()
                         .network(network)
                         .mean(inputs.primary)
                         .dispersion(different)
                         .labels(inputs.labels)
                         .build(),
                 invalid_argument);
    EXPECT_THROW((void)StudentTNLLLoss::Builder()
                         .network(network)
                         .location(inputs.primary)
                         .logScale(inputs.parameter)
                         .target(inputs.labels)
                         .reportsPerOutputLoss()
                         .build(),
                 invalid_argument);
    EXPECT_THROW((void)NegativeBinomialNLLLoss::Builder()
                         .network(network)
                         .mean(inputs.primary)
                         .dispersion(inputs.parameter)
                         .labels(inputs.labels)
                         .reportsPerOutputLoss()
                         .build(),
                 invalid_argument);
}


TEST(RaggedDistributionR10K, ForwardBackwardQualifiesEveryDifferentiableStudentTParameter) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) runStudentRuntimeCase(offsetsDType);
}

TEST(RaggedDistributionR10K, ForwardBackwardQualifiesNegativeBinomialDirectAndLogParameters) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64})
        for (bool logMean : {false, true})
            for (bool logDispersion : {false, true})
                runNegativeBinomialRuntimeCase(offsetsDType, logMean, logDispersion);
}


TEST(RaggedDistributionR10K, SupportLayersSaveLoadPreserveAllDifferentiableSecondaryInputs) {
    Network network("r10k_student_round_trip");
    Inputs inputs = makeInputs(network, "round_trip", DataType::UINT64);
    RaggedTensor learnedLogDof = RaggedNetworkInput::Builder()
                                     .network(network)
                                     .name("round_trip_log_dof")
                                     .valuesDataType(DataType::FP32)
                                     .trailingDimensions({2})
                                     .partition(inputs.primary)
                                     .build();
    NetworkInput weights = NetworkInput::Builder()
                               .network(network)
                               .name("round_trip_weights")
                               .dimensions({1})
                               .dataType(DataType::FP32)
                               .build();
    (void)StudentTNLLLoss::Builder()
        .network(network)
        .location(inputs.primary)
        .logScale(inputs.parameter)
        .target(inputs.labels)
        .logDegreesOfFreedom(learnedLogDof)
        .minimumDegreesOfFreedom(2.0f)
        .exampleWeights(weights.getFeatureOutput().value())
        .reportsPerExampleLoss()
        .build();

    const auto now = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archiveDir = filesystem::temp_directory_path() /
                                        (string("thor_r10k_student_") + to_string(now));
    filesystem::remove_all(archiveDir);
    network.save(archiveDir.string(), /*overwrite=*/true);

    Network loaded("r10k_student_round_trip");
    ASSERT_NO_THROW(loaded.load(archiveDir.string()));
    shared_ptr<RaggedCustomLoss> loadedRaw = findRaggedCustomLoss(loaded);
    ASSERT_NE(loadedRaw, nullptr);
    ASSERT_EQ(loadedRaw->getRaggedSecondaryInputs().size(), 2u);
    ASSERT_TRUE(loadedRaw->getRaggedExampleWeights().has_value());
    for (const RaggedTensor& secondary : loadedRaw->getRaggedSecondaryInputs())
        EXPECT_EQ(secondary.getOffsets(), loadedRaw->getRaggedPredictions().getOffsets());
    EXPECT_EQ(loadedRaw->getRaggedExampleWeights()->getOffsets(), loadedRaw->getRaggedPredictions().getOffsets());
    filesystem::remove_all(archiveDir);
}
