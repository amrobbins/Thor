#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/HuberLoss.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Loss/SmoothL1Loss.h"
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
#include <tuple>
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
                  DataType predictionsDType = DataType::FP32,
                  DataType labelsDType = DataType::FP32,
                  DataType offsetsDType = DataType::UINT32,
                  vector<uint64_t> trailingDimensions = {1},
                  uint32_t batchSize = 4,
                  uint64_t maxTotalValues = 8) {
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

bool cudaAvailable() {
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

uint32_t countLayerType(Network& network, const string& type) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        if (network.getLayer(i)->getLayerType() == type) ++count;
    }
    return count;
}

void writeOffsets(Impl::Tensor& offsetsTensor, DataType dtype, const vector<uint64_t>& offsets) {
    if (dtype == DataType::UINT32) {
        uint32_t* values = offsetsTensor.getMemPtr<uint32_t>();
        for (size_t i = 0; i < offsets.size(); ++i) values[i] = static_cast<uint32_t>(offsets[i]);
    } else {
        ASSERT_EQ(dtype, DataType::UINT64);
        uint64_t* values = offsetsTensor.getMemPtr<uint64_t>();
        copy(offsets.begin(), offsets.end(), values);
    }
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

enum class LossKind { MAPE, HUBER, SMOOTH_L1 };

Tensor buildBatchLoss(Network& network, const Inputs& inputs, LossKind kind, float parameter) {
    if (kind == LossKind::MAPE) {
        return MAPE::Builder()
            .network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .lossDataType(DataType::FP32)
            .reportsBatchLoss()
            .build()
            .getLoss();
    }
    if (kind == LossKind::HUBER) {
        return HuberLoss::Builder()
            .network(network)
            .predictions(inputs.predictions)
            .labels(inputs.labels)
            .delta(parameter)
            .lossDataType(DataType::FP32)
            .reportsBatchLoss()
            .build()
            .getLoss();
    }
    return SmoothL1Loss::Builder()
        .network(network)
        .predictions(inputs.predictions)
        .labels(inputs.labels)
        .beta(parameter)
        .lossDataType(DataType::FP32)
        .reportsBatchLoss()
        .build()
        .getLoss();
}

float huberLoss(float diff, float delta) {
    const float a = fabsf(diff);
    return a <= delta ? 0.5f * diff * diff : delta * (a - 0.5f * delta);
}

float huberGradient(float diff, float delta) {
    if (fabsf(diff) <= delta) return diff;
    return diff > 0.0f ? delta : -delta;
}

float smoothLoss(float diff, float beta) {
    const float a = fabsf(diff);
    return a < beta ? 0.5f * diff * diff / beta : a - 0.5f * beta;
}

float smoothGradient(float diff, float beta) {
    if (fabsf(diff) < beta) return diff / beta;
    if (diff > 0.0f) return 1.0f;
    if (diff < 0.0f) return -1.0f;
    return 0.0f;
}

pair<float, float> mapeLossGradient(float prediction, float label) {
    if (prediction == label) return {0.0f, 0.0f};
    float effectiveLabel = label;
    constexpr float epsilon = 0.0001f;
    if (effectiveLabel < 0.0f) {
        if (effectiveLabel > -epsilon) effectiveLabel = -epsilon;
    } else if (effectiveLabel <= epsilon) {
        effectiveLabel = epsilon;
    }
    const float diff = prediction - effectiveLabel;
    const float loss = min(1000.0f, fabsf(diff / effectiveLabel) * 100.0f);
    float gradient = 0.0f;
    if (diff > 0.0f) gradient = 100.0f / fabsf(effectiveLabel);
    if (diff < 0.0f) gradient = -100.0f / fabsf(effectiveLabel);
    gradient = max(-1000.0f, min(1000.0f, gradient));
    return {loss, gradient};
}

void runRuntimeCase(LossKind kind, DataType offsetsDType) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t maxTotalValues = 8;
    constexpr float parameter = 0.75f;

    const string kindName = kind == LossKind::MAPE ? "mape" : kind == LossKind::HUBER ? "huber" : "smooth";
    Network network("ragged_r10f_" + kindName + (offsetsDType == DataType::UINT32 ? "_u32" : "_u64"));
    Inputs inputs = makeInputs(network, DataType::FP32, DataType::FP32, offsetsDType, {1}, batchSize, maxTotalValues);
    Tensor reportedLoss = buildBatchLoss(network, inputs, kind, parameter);
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
    float* p = predictionValues.getMemPtr<float>();
    float* y = labelValues.getMemPtr<float>();
    fill(p, p + maxTotalValues, numeric_limits<float>::quiet_NaN());
    fill(y, y + maxTotalValues, numeric_limits<float>::quiet_NaN());

    vector<float> predictions;
    vector<float> labels;
    if (kind == LossKind::MAPE) {
        predictions = {0.0f, 0.0002f, 0.0f, 3.0f, 100.0f};
        labels = {0.0f, 0.00005f, -0.00005f, 2.0f, 0.1f};
    } else {
        predictions = {0.25f, -0.75f, 1.5f, -2.0f, 0.0f};
        labels.assign(predictions.size(), 0.0f);
    }
    copy(predictions.begin(), predictions.end(), p);
    copy(labels.begin(), labels.end(), y);
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

    float numerator = 0.0f;
    vector<float> expectedGradients;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float raw = 0.0f;
        float grad = 0.0f;
        if (kind == LossKind::MAPE) {
            tie(raw, grad) = mapeLossGradient(predictions[i], labels[i]);
        } else {
            const float diff = predictions[i] - labels[i];
            if (kind == LossKind::HUBER) {
                raw = huberLoss(diff, parameter);
                grad = huberGradient(diff, parameter);
            } else {
                raw = smoothLoss(diff, parameter);
                grad = smoothGradient(diff, parameter);
            }
        }
        numerator += raw;
        expectedGradients.push_back(grad * Impl::Loss::getLossScalingFactor());
    }

    const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
    ASSERT_EQ(reported.size(), 1u);
    EXPECT_NEAR(reported[0], numerator / validExamples, 2.0e-3f);

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    ASSERT_EQ(gradient.size(), maxTotalValues);
    for (size_t i = 0; i < expectedGradients.size(); ++i)
        EXPECT_NEAR(gradient[i], expectedGradients[i], kind == LossKind::MAPE ? 2.0e-2f : 1.0e-5f) << "active index " << i;
}

}  // namespace

TEST(RaggedRegressionR10F, ReportingShapesPreservePartitionAndRejectPerOutput) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (LossKind kind : {LossKind::MAPE, LossKind::HUBER, LossKind::SMOOTH_L1}) {
            Network rawNetwork("r10f_raw");
            Inputs rawInputs = makeInputs(rawNetwork, DataType::FP16, DataType::UINT32, offsetsDType, {2});
            if (kind == LossKind::MAPE) {
                MAPE loss = MAPE::Builder().network(rawNetwork).predictions(rawInputs.predictions).labels(rawInputs.labels).reportsRawLoss().build();
                EXPECT_TRUE(loss.isRagged());
                EXPECT_EQ(loss.getRaggedRawLoss().getOffsets(), rawInputs.predictions.getOffsets());
                EXPECT_EQ(loss.getRaggedRawLoss().getValuesDataType(), DataType::FP16);
            } else if (kind == LossKind::HUBER) {
                HuberLoss loss = HuberLoss::Builder().network(rawNetwork).predictions(rawInputs.predictions).labels(rawInputs.labels).reportsRawLoss().build();
                EXPECT_TRUE(loss.isRagged());
                EXPECT_EQ(loss.getRaggedRawLoss().getOffsets(), rawInputs.predictions.getOffsets());
            } else {
                SmoothL1Loss loss = SmoothL1Loss::Builder().network(rawNetwork).predictions(rawInputs.predictions).labels(rawInputs.labels).reportsRawLoss().build();
                EXPECT_TRUE(loss.isRagged());
                EXPECT_EQ(loss.getRaggedRawLoss().getOffsets(), rawInputs.predictions.getOffsets());
            }

            Network perExampleNetwork("r10f_per_example");
            Inputs perExampleInputs = makeInputs(perExampleNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            Tensor perExample;
            if (kind == LossKind::MAPE)
                perExample = MAPE::Builder().network(perExampleNetwork).predictions(perExampleInputs.predictions).labels(perExampleInputs.labels).reportsPerExampleLoss().build().getLoss();
            else if (kind == LossKind::HUBER)
                perExample = HuberLoss::Builder().network(perExampleNetwork).predictions(perExampleInputs.predictions).labels(perExampleInputs.labels).reportsPerExampleLoss().build().getLoss();
            else
                perExample = SmoothL1Loss::Builder().network(perExampleNetwork).predictions(perExampleInputs.predictions).labels(perExampleInputs.labels).reportsPerExampleLoss().build().getLoss();
            EXPECT_EQ(perExample.getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(countLayerType(perExampleNetwork, "RaggedLossShaper"), 1u);

            Network noneNetwork("r10f_none");
            Inputs noneInputs = makeInputs(noneNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            if (kind == LossKind::MAPE) {
                MAPE loss = MAPE::Builder().network(noneNetwork).predictions(noneInputs.predictions).labels(noneInputs.labels).reportsNoLoss().build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            } else if (kind == LossKind::HUBER) {
                HuberLoss loss = HuberLoss::Builder().network(noneNetwork).predictions(noneInputs.predictions).labels(noneInputs.labels).reportsNoLoss().build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            } else {
                SmoothL1Loss loss = SmoothL1Loss::Builder().network(noneNetwork).predictions(noneInputs.predictions).labels(noneInputs.labels).reportsNoLoss().build();
                EXPECT_THROW((void)loss.getLoss(), runtime_error);
            }
            ASSERT_EQ(noneNetwork.getLossRootTensors().size(), 1u);

            Network rejectNetwork("r10f_reject");
            Inputs rejectInputs = makeInputs(rejectNetwork, DataType::FP32, DataType::FP32, offsetsDType);
            if (kind == LossKind::MAPE)
                EXPECT_THROW((void)MAPE::Builder().network(rejectNetwork).predictions(rejectInputs.predictions).labels(rejectInputs.labels).reportsPerOutputLoss().build(), invalid_argument);
            else if (kind == LossKind::HUBER)
                EXPECT_THROW((void)HuberLoss::Builder().network(rejectNetwork).predictions(rejectInputs.predictions).labels(rejectInputs.labels).reportsPerOutputLoss().build(), invalid_argument);
            else
                EXPECT_THROW((void)SmoothL1Loss::Builder().network(rejectNetwork).predictions(rejectInputs.predictions).labels(rejectInputs.labels).reportsPerOutputLoss().build(), invalid_argument);
        }
    }
}

TEST(RaggedRegressionR10F, RequiresExactSharedPartition) {
    for (LossKind kind : {LossKind::MAPE, LossKind::HUBER, LossKind::SMOOTH_L1}) {
        Network network("r10f_partition");
        RaggedTensor predictions = RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("predictions")
                                       .valuesDataType(DataType::FP32)
                                       .trailingDimensions({1})
                                       .batchSize(3)
                                       .maxTotalValues(7)
                                       .maxValuesPerRow(4)
                                       .build();
        RaggedTensor labels = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("labels")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({1})
                                  .batchSize(3)
                                  .maxTotalValues(7)
                                  .maxValuesPerRow(4)
                                  .build();
        if (kind == LossKind::MAPE)
            EXPECT_THROW((void)MAPE::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
        else if (kind == LossKind::HUBER)
            EXPECT_THROW((void)HuberLoss::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
        else
            EXPECT_THROW((void)SmoothL1Loss::Builder().network(network).predictions(predictions).labels(labels).build(), invalid_argument);
    }
}

TEST(RaggedRegressionR10F, MatchesDenseDTypeContracts) {
    const vector<DataType> labelTypes{DataType::BOOLEAN, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::FP16, DataType::FP32};
    for (LossKind kind : {LossKind::MAPE, LossKind::HUBER, LossKind::SMOOTH_L1}) {
        for (DataType predictionType : {DataType::FP16, DataType::FP32}) {
            for (DataType labelType : labelTypes) {
                Network network("r10f_dtype");
                Inputs inputs = makeInputs(network, predictionType, labelType);
                if (kind == LossKind::MAPE)
                    EXPECT_NO_THROW((void)MAPE::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build());
                else if (kind == LossKind::HUBER)
                    EXPECT_NO_THROW((void)HuberLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build());
                else
                    EXPECT_NO_THROW((void)SmoothL1Loss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsRawLoss().build());
            }
        }
    }
}

TEST(RaggedRegressionR10F, RejectsTheSameUnsupportedValueDTypesAsDenseCounterparts) {
    for (LossKind kind : {LossKind::MAPE, LossKind::HUBER, LossKind::SMOOTH_L1}) {
        Network predictionNetwork("r10f_reject_prediction_dtype");
        Inputs predictionInputs = makeInputs(predictionNetwork, DataType::BF16, DataType::FP32);
        if (kind == LossKind::MAPE)
            EXPECT_THROW((void)MAPE::Builder().network(predictionNetwork).predictions(predictionInputs.predictions).labels(predictionInputs.labels).build(), exception);
        else if (kind == LossKind::HUBER)
            EXPECT_THROW((void)HuberLoss::Builder().network(predictionNetwork).predictions(predictionInputs.predictions).labels(predictionInputs.labels).build(), exception);
        else
            EXPECT_THROW((void)SmoothL1Loss::Builder().network(predictionNetwork).predictions(predictionInputs.predictions).labels(predictionInputs.labels).build(), exception);

        Network labelNetwork("r10f_reject_label_dtype");
        Inputs labelInputs = makeInputs(labelNetwork, DataType::FP32, DataType::INT32);
        if (kind == LossKind::MAPE)
            EXPECT_THROW((void)MAPE::Builder().network(labelNetwork).predictions(labelInputs.predictions).labels(labelInputs.labels).build(), exception);
        else if (kind == LossKind::HUBER)
            EXPECT_THROW((void)HuberLoss::Builder().network(labelNetwork).predictions(labelInputs.predictions).labels(labelInputs.labels).build(), exception);
        else
            EXPECT_THROW((void)SmoothL1Loss::Builder().network(labelNetwork).predictions(labelInputs.predictions).labels(labelInputs.labels).build(), exception);
    }
}

TEST(RaggedRegressionR10F, PiecewiseAndMapeStabilityForwardBackwardUseOnlyActivePrefix) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for R10F runtime coverage.";
    for (LossKind kind : {LossKind::MAPE, LossKind::HUBER, LossKind::SMOOTH_L1}) {
        for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) runRuntimeCase(kind, offsetsDType);
    }
}

TEST(RaggedRegressionR10F, SupportLayersRoundTripWithCanonicalPartition) {
    for (LossKind kind : {LossKind::MAPE, LossKind::HUBER, LossKind::SMOOTH_L1}) {
        Network network("r10f_round_trip");
        Inputs inputs = makeInputs(network, DataType::FP32, DataType::FP32, DataType::UINT64);
        if (kind == LossKind::MAPE)
            (void)MAPE::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).reportsPerExampleLoss().build();
        else if (kind == LossKind::HUBER)
            (void)HuberLoss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).delta(0.625f).reportsPerExampleLoss().build();
        else
            (void)SmoothL1Loss::Builder().network(network).predictions(inputs.predictions).labels(inputs.labels).beta(0.625f).reportsPerExampleLoss().build();

        const auto now = chrono::steady_clock::now().time_since_epoch().count();
        const filesystem::path archiveDir = filesystem::temp_directory_path() / ("thor_r10f_" + to_string(now));
        filesystem::remove_all(archiveDir);
        network.save(archiveDir.string(), /*overwrite=*/true);
        Network loaded("r10f_round_trip");
        ASSERT_NO_THROW(loaded.load(archiveDir.string()));
        EXPECT_EQ(countLayerType(loaded, "RaggedCustomLoss"), 1u);
        EXPECT_EQ(countLayerType(loaded, "RaggedLossShaper"), 1u);

        shared_ptr<RaggedCustomLoss> raw;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            raw = dynamic_pointer_cast<RaggedCustomLoss>(loaded.getLayer(i));
            if (raw != nullptr) break;
        }
        ASSERT_NE(raw, nullptr);
        EXPECT_EQ(raw->getRaggedPredictions().getOffsets(), raw->getRaggedLabels().getOffsets());
        EXPECT_EQ(raw->getRaggedRawLoss().getOffsets(), raw->getRaggedPredictions().getOffsets());
        filesystem::remove_all(archiveDir);
    }
}
