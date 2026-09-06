#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/SparseCategoricalCrossEntropyWithLogits.h"
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

struct SparseInputs {
    RaggedTensor predictions;
    RaggedTensor labels;
    RaggedTensor mask;
};

SparseInputs makeSparseInputs(Network& network,
                              DataType offsetsDType = DataType::UINT32,
                              vector<uint64_t> labelTrailing = {},
                              vector<uint64_t> maskTrailing = {},
                              DataType labelsDType = DataType::UINT32,
                              DataType maskDType = DataType::UINT8,
                              uint32_t batchSize = 3,
                              uint64_t capacity = 8,
                              uint64_t maxValuesPerRow = 5) {
    RaggedTensor predictions = RaggedNetworkInput::Builder()
                                   .network(network)
                                   .name("predictions")
                                   .valuesDataType(DataType::FP32)
                                   .offsetsDataType(offsetsDType)
                                   .trailingDimensions({3})
                                   .batchSize(batchSize)
                                   .maxTotalValues(capacity)
                                   .maxValuesPerRow(maxValuesPerRow)
                                   .build();
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(labelsDType)
                              .trailingDimensions(labelTrailing)
                              .partition(predictions)
                              .build();
    RaggedTensor mask = RaggedNetworkInput::Builder()
                            .network(network)
                            .name("mask")
                            .valuesDataType(maskDType)
                            .trailingDimensions(maskTrailing)
                            .partition(predictions)
                            .build();
    return {predictions, labels, mask};
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

void writeLabels(Impl::Tensor& labelsTensor, DataType dtype, const vector<uint32_t>& activeLabels) {
    const uint64_t capacity = labelsTensor.getTotalNumElements();
    if (dtype == DataType::UINT8) {
        uint8_t* values = labelsTensor.getMemPtr<uint8_t>();
        fill(values, values + capacity, numeric_limits<uint8_t>::max());
        for (size_t i = 0; i < activeLabels.size(); ++i) values[i] = static_cast<uint8_t>(activeLabels[i]);
        return;
    }
    if (dtype == DataType::UINT16) {
        uint16_t* values = labelsTensor.getMemPtr<uint16_t>();
        fill(values, values + capacity, numeric_limits<uint16_t>::max());
        for (size_t i = 0; i < activeLabels.size(); ++i) values[i] = static_cast<uint16_t>(activeLabels[i]);
        return;
    }
    ASSERT_EQ(dtype, DataType::UINT32);
    uint32_t* values = labelsTensor.getMemPtr<uint32_t>();
    fill(values, values + capacity, numeric_limits<uint32_t>::max());
    copy(activeLabels.begin(), activeLabels.end(), values);
}

vector<float> copyFp32ToHost(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    if (tensor.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU) {
        const float* values = tensor.getMemPtr<float>();
        return vector<float>(values, values + tensor.getTotalNumElements());
    }
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = tensor.clone(cpuPlacement);
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + host.getTotalNumElements());
}

float sparseCe3(const float* logits, uint32_t label) {
    const float m = max({logits[0], logits[1], logits[2]});
    const float sum = exp(logits[0] - m) + exp(logits[1] - m) + exp(logits[2] - m);
    return m + log(sum) - logits[label];
}

vector<float> sparseGradient3(const float* logits, uint32_t label, float scale = 1.0f) {
    const float m = max({logits[0], logits[1], logits[2]});
    const float e0 = exp(logits[0] - m);
    const float e1 = exp(logits[1] - m);
    const float e2 = exp(logits[2] - m);
    const float inv = 1.0f / (e0 + e1 + e2);
    vector<float> gradient{e0 * inv, e1 * inv, e2 * inv};
    gradient[label] -= 1.0f;
    for (float& value : gradient) value *= scale;
    return gradient;
}

shared_ptr<Impl::SparseCategoricalCrossEntropyWithLogits> findOnlyPhysicalSparseCe(PlacedNetwork& placed) {
    shared_ptr<Impl::SparseCategoricalCrossEntropyWithLogits> result;
    for (const shared_ptr<Impl::Layer>& layer : placed.getStampedNetwork(0).getOtherLayers()) {
        auto candidate = dynamic_pointer_cast<Impl::SparseCategoricalCrossEntropyWithLogits>(layer);
        if (candidate == nullptr) continue;
        if (result != nullptr) throw logic_error("R11C.1 test found multiple physical sparse CE layers.");
        result = candidate;
    }
    return result;
}

Batch makeBatch(DataType offsetsDType,
                uint64_t maxValuesPerRow,
                const vector<uint64_t>& offsets,
                const vector<float>& activeLogits,
                const vector<uint32_t>& activeLabels,
                const vector<uint8_t>& activeMask,
                DataType labelsDType = DataType::UINT32,
                uint32_t validExampleCount = 0,
                uint64_t capacity = 8) {
    if (offsets.size() < 2 || activeLogits.size() != activeLabels.size() * 3 ||
        activeMask.size() != activeLabels.size() || activeLabels.size() > capacity)
        throw invalid_argument("R11C.1 sparse test batch geometry is inconsistent.");
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor logits(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {capacity, 3}));
    Impl::Tensor labels(cpuPlacement, Impl::TensorDescriptor(labelsDType, {capacity}));
    Impl::Tensor mask(cpuPlacement, Impl::TensorDescriptor(DataType::UINT8, {capacity}));
    Impl::Tensor physicalOffsets(cpuPlacement, Impl::TensorDescriptor(offsetsDType, {offsets.size()}));

    fill(logits.getMemPtr<float>(), logits.getMemPtr<float>() + capacity * 3, numeric_limits<float>::quiet_NaN());
    fill(mask.getMemPtr<uint8_t>(), mask.getMemPtr<uint8_t>() + capacity, uint8_t{0xFF});
    copy(activeLogits.begin(), activeLogits.end(), logits.getMemPtr<float>());
    writeLabels(labels, labelsDType, activeLabels);
    copy(activeMask.begin(), activeMask.end(), mask.getMemPtr<uint8_t>());
    writeOffsets(physicalOffsets, offsetsDType, offsets);

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(logits, physicalOffsets, maxValuesPerRow));
    batch.insert("labels", labels);
    batch.insert("mask", mask);
    if (validExampleCount != 0) batch.setValidExampleCount(validExampleCount);
    return batch;
}

}  // namespace

TEST(RaggedSparseCategoricalR11C1, PublicApiPreservesPartitionScalarGeometryMaskAndCloneIdentity) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r11c1_sparse_public");
        SparseInputs inputs = makeSparseInputs(network, offsetsDType);
        SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                                 .network(network)
                                                 .predictions(inputs.predictions)
                                                 .labels(inputs.labels)
                                                 .numClasses(3)
                                                 .ignoreIndex(2)
                                                 .mask(inputs.mask)
                                                 .reportsRawLoss()
                                                 .build();

        ASSERT_TRUE(loss.isRagged());
        EXPECT_EQ(loss.getRaggedPredictions(), inputs.predictions);
        EXPECT_EQ(loss.getRaggedLabels(), inputs.labels);
        ASSERT_TRUE(loss.getRaggedMask().has_value());
        EXPECT_EQ(loss.getRaggedMask().value(), inputs.mask);
        RaggedTensor raw = loss.getRaggedLoss();
        EXPECT_TRUE(raw.sharesPartitionWith(inputs.predictions));
        EXPECT_TRUE(raw.getTrailingDimensions().empty());
        EXPECT_EQ(raw.getValuesDimensions(), (vector<uint64_t>{8}));
        EXPECT_EQ(countLayerType(network, "SparseCategoricalCrossEntropy"), 1u);

        shared_ptr<Layer> clonedBase = loss.clone();
        auto cloned = dynamic_pointer_cast<SparseCategoricalCrossEntropy>(clonedBase);
        ASSERT_NE(cloned, nullptr);
        EXPECT_EQ(cloned->getRaggedPredictions(), inputs.predictions);
        EXPECT_EQ(cloned->getRaggedLabels(), inputs.labels);
        ASSERT_TRUE(cloned->getRaggedMask().has_value());
        EXPECT_EQ(cloned->getRaggedMask().value(), inputs.mask);
        EXPECT_EQ(cloned->getRaggedRawLoss(), raw);
    }

    Network singletonNetwork("r11c1_sparse_singletons");
    SparseInputs singleton = makeSparseInputs(singletonNetwork, DataType::UINT32, {1}, {1});
    EXPECT_NO_THROW((void)SparseCategoricalCrossEntropy::Builder()
                        .network(singletonNetwork)
                        .predictions(singleton.predictions)
                        .labels(singleton.labels)
                        .numClasses(3)
                        .mask(singleton.mask)
                        .reportsRawLoss()
                        .build());
}

TEST(RaggedSparseCategoricalR11C1, RejectsUndefinedOrMismatchedRaggedContracts) {
    Network perOutputNetwork("r11c1_sparse_per_output");
    SparseInputs perOutput = makeSparseInputs(perOutputNetwork);
    EXPECT_THROW((void)SparseCategoricalCrossEntropy::Builder()
                     .network(perOutputNetwork)
                     .predictions(perOutput.predictions)
                     .labels(perOutput.labels)
                     .numClasses(3)
                     .reportsPerOutputLoss()
                     .build(),
                 invalid_argument);

    Network labelShapeNetwork("r11c1_sparse_label_shape");
    SparseInputs badShape = makeSparseInputs(labelShapeNetwork, DataType::UINT32, {2});
    EXPECT_THROW((void)SparseCategoricalCrossEntropy::Builder()
                     .network(labelShapeNetwork)
                     .predictions(badShape.predictions)
                     .labels(badShape.labels)
                     .numClasses(3)
                     .build(),
                 invalid_argument);

    Network partitionNetwork("r11c1_sparse_partition");
    SparseInputs same = makeSparseInputs(partitionNetwork);
    RaggedTensor differentLabels = RaggedNetworkInput::Builder()
                                       .network(partitionNetwork)
                                       .name("different_labels")
                                       .valuesDataType(DataType::UINT32)
                                       .trailingDimensions({})
                                       .batchSize(3)
                                       .maxTotalValues(8)
                                       .maxValuesPerRow(5)
                                       .build();
    EXPECT_THROW((void)SparseCategoricalCrossEntropy::Builder()
                     .network(partitionNetwork)
                     .predictions(same.predictions)
                     .labels(differentLabels)
                     .numClasses(3)
                     .build(),
                 invalid_argument);

    RaggedTensor differentMask = RaggedNetworkInput::Builder()
                                     .network(partitionNetwork)
                                     .name("different_mask")
                                     .valuesDataType(DataType::UINT8)
                                     .trailingDimensions({})
                                     .batchSize(3)
                                     .maxTotalValues(8)
                                     .maxValuesPerRow(5)
                                     .build();
    EXPECT_THROW((void)SparseCategoricalCrossEntropy::Builder()
                     .network(partitionNetwork)
                     .predictions(same.predictions)
                     .labels(same.labels)
                     .numClasses(3)
                     .mask(differentMask)
                     .build(),
                 invalid_argument);
}

TEST(RaggedSparseCategoricalR11C1, RawAndNoneRequireOnlyActiveCountWhileReportedReductionsRequireOffsets) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    for (Loss::LossShape shape : {Loss::LossShape::RAW, Loss::LossShape::NONE, Loss::LossShape::PER_EXAMPLE, Loss::LossShape::BATCH}) {
        Network network("r11c1_sparse_partition_requirement");
        SparseInputs inputs = makeSparseInputs(network, DataType::UINT64);
        SparseCategoricalCrossEntropy::Builder builder;
        builder.network(network).predictions(inputs.predictions).labels(inputs.labels).numClasses(3).mask(inputs.mask);
        if (shape == Loss::LossShape::RAW)
            builder.reportsRawLoss();
        else if (shape == Loss::LossShape::NONE)
            builder.reportsNoLoss();
        else if (shape == Loss::LossShape::PER_EXAMPLE)
            builder.reportsPerExampleLoss();
        else
            builder.reportsBatchLoss();
        SparseCategoricalCrossEntropy loss = builder.build();

        if (shape == Loss::LossShape::RAW) {
            NetworkOutput::Builder()
                .network(network)
                .name("raw_values")
                .inputTensor(loss.getRaggedLoss().getValues())
                .dataType(DataType::FP32)
                .build();
        } else if (shape != Loss::LossShape::NONE) {
            NetworkOutput::Builder().network(network).name("loss").inputTensor(loss.getLoss()).dataType(DataType::FP32).build();
        }

        vector<Event> initDone;
        shared_ptr<PlacedNetwork> placed = network.place(3, initDone, /*inferenceOnly=*/true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDone) event.synchronize();
        const auto& stamp = placed->getStampedNetwork(0);
        const auto requirements = stamp.getExternalRowPartitionRequirementsForTest(inputs.predictions.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());

        const Impl::RaggedPartitionRequirement expected =
            (shape == Loss::LossShape::RAW || shape == Loss::LossShape::NONE)
                ? Impl::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT
                : Impl::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT | Impl::RaggedPartitionRequirement::DEVICE_OFFSETS;
        EXPECT_EQ(requirements.value(), expected);
        ASSERT_NE(stamp.getManagedPartitionActiveCountInputForTest(inputs.predictions.getRowPartitionId()), nullptr);
        if (shape == Loss::LossShape::RAW || shape == Loss::LossShape::NONE)
            EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(inputs.predictions.getRowPartitionId()), nullptr);
        else
            EXPECT_NE(stamp.getManagedPartitionOffsetsInputForTest(inputs.predictions.getRowPartitionId()), nullptr);
    }
}

TEST(RaggedSparseCategoricalR11C1, QualificationForwardBackwardCoversOffsetsLabelWidthsPartialBatchInvalidIgnoreMaskAndPoison) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr uint32_t batchSize = 4;
    constexpr uint32_t validExamples = 3;
    constexpr uint64_t capacity = 8;
    const vector<float> activeLogits{
        1.0f, -0.5f, 0.25f,
        -1.0f, 0.5f, 1.5f,
        0.0f, 0.0f, 0.0f,
        2.0f, 0.25f, -1.0f,
        -0.25f, 1.25f, 0.5f,
    };
    // token 2 is an out-of-range class, token 3 is ignore_index, and token 4
    // is masked. All three are active packed tokens and must produce zero
    // loss/gradient without reading any inactive poisoned capacity.
    const vector<uint32_t> activeLabels{0, 1, 7, 2, 1};
    const vector<uint8_t> activeMask{1, 1, 1, 1, 0};

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        for (DataType labelsDType : {DataType::UINT8, DataType::UINT16, DataType::UINT32}) {
            Network network("r11c1_sparse_qualification_backward");
            SparseInputs inputs = makeSparseInputs(network,
                                                   offsetsDType,
                                                   {},
                                                   {},
                                                   labelsDType,
                                                   DataType::UINT8,
                                                   batchSize,
                                                   capacity,
                                                   5);
            SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                                     .network(network)
                                                     .predictions(inputs.predictions)
                                                     .labels(inputs.labels)
                                                     .numClasses(3)
                                                     .ignoreIndex(2)
                                                     .mask(inputs.mask)
                                                     .reportsBatchLoss()
                                                     .build();
            NetworkOutput::Builder()
                .network(network)
                .name("loss")
                .inputTensor(loss.getLoss())
                .dataType(DataType::FP32)
                .build();

            vector<Event> initDone;
            shared_ptr<PlacedNetwork> placed = network.place(batchSize, initDone, /*inferenceOnly=*/false);
            ASSERT_NE(placed, nullptr);
            for (Event& event : initDone) event.synchronize();

            shared_ptr<Impl::SparseCategoricalCrossEntropyWithLogits> physicalLoss = findOnlyPhysicalSparseCe(*placed);
            ASSERT_NE(physicalLoss, nullptr);

            Batch batch = makeBatch(offsetsDType,
                                    5,
                                    {0, 2, 2, 5, 5},
                                    activeLogits,
                                    activeLabels,
                                    activeMask,
                                    labelsDType,
                                    validExamples,
                                    capacity);
            map<string, Impl::Tensor> outputs;
            map<string, Event> outputReadyEvents;
            Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
            done.synchronize();
            outputReadyEvents.at("loss").synchronize();
            placed->synchronize();

            const vector<float> reported = copyFp32ToHost(outputs.at("loss"));
            ASSERT_EQ(reported.size(), 1u);
            const float expectedLoss = (sparseCe3(&activeLogits[0], 0) + sparseCe3(&activeLogits[3], 1)) /
                                       static_cast<float>(validExamples);
            EXPECT_NEAR(reported[0], expectedLoss, 4.0e-5f)
                << "offsets dtype=" << static_cast<int>(offsetsDType)
                << " labels dtype=" << static_cast<int>(labelsDType);

            ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
            const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
            ASSERT_EQ(gradient.size(), capacity * 3);
            const float gradientScale = static_cast<float>(Impl::Loss::getLossScalingFactor());
            const vector<float> expected0 = sparseGradient3(&activeLogits[0], 0, gradientScale);
            const vector<float> expected1 = sparseGradient3(&activeLogits[3], 1, gradientScale);
            for (size_t c = 0; c < 3; ++c) {
                EXPECT_NEAR(gradient[c], expected0[c], 8.0e-4f);
                EXPECT_NEAR(gradient[3 + c], expected1[c], 8.0e-4f);
            }
            for (size_t token = 2; token < 5; ++token) {
                for (size_t c = 0; c < 3; ++c)
                    EXPECT_NEAR(gradient[token * 3 + c], 0.0f, 1.0e-6f)
                        << "token=" << token << " class=" << c;
            }
        }
    }
}

TEST(RaggedSparseCategoricalR11C1, QualificationNoneRemainsTrainingObjectiveWithoutFullOffsets) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    Network network("r11c1_sparse_qualification_none");
    SparseInputs inputs = makeSparseInputs(network, DataType::UINT32);
    SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                             .network(network)
                                             .predictions(inputs.predictions)
                                             .labels(inputs.labels)
                                             .numClasses(3)
                                             .mask(inputs.mask)
                                             .reportsNoLoss()
                                             .build();
    EXPECT_THROW((void)loss.getLoss(), runtime_error);
    ASSERT_EQ(network.getLossRootTensors().size(), 1u);
    EXPECT_EQ(network.getLossRootTensors()[0], loss.getRaggedRawLoss().getValues());

    vector<Event> initDone;
    shared_ptr<PlacedNetwork> placed = network.place(3, initDone, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDone) event.synchronize();

    const auto& stamp = placed->getStampedNetwork(0);
    const auto requirements = stamp.getExternalRowPartitionRequirementsForTest(inputs.predictions.getRowPartitionId());
    ASSERT_TRUE(requirements.has_value());
    EXPECT_EQ(requirements.value(), Impl::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(inputs.predictions.getRowPartitionId()), nullptr);

    shared_ptr<Impl::SparseCategoricalCrossEntropyWithLogits> physicalLoss = findOnlyPhysicalSparseCe(*placed);
    ASSERT_NE(physicalLoss, nullptr);
    const vector<float> logits{
        1.0f, 0.0f, -1.0f,
        -0.5f, 0.25f, 1.5f,
    };
    const vector<uint32_t> labels{0, 2};
    const vector<uint8_t> mask{1, 1};
    Batch batch = makeBatch(DataType::UINT32, 5, {0, 1, 1, 2}, logits, labels, mask);
    map<string, Impl::Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/false);
    done.synchronize();
    placed->synchronize();
    EXPECT_TRUE(outputs.empty());

    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> gradient = copyFp32ToHost(physicalLoss->getErrorOutput().value());
    const float scale = static_cast<float>(Impl::Loss::getLossScalingFactor());
    const vector<float> expected0 = sparseGradient3(&logits[0], labels[0], scale);
    const vector<float> expected1 = sparseGradient3(&logits[3], labels[1], scale);
    for (size_t c = 0; c < 3; ++c) {
        EXPECT_NEAR(gradient[c], expected0[c], 8.0e-4f);
        EXPECT_NEAR(gradient[3 + c], expected1[c], 8.0e-4f);
    }
}

TEST(RaggedSparseCategoricalR11C1, QualificationBatchReportingReusesShortLongShortAndAllEmptyPartitions) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr DataType offsetsDType = DataType::UINT64;
    Network network("r11c1_sparse_qualification_reuse");
    SparseInputs inputs = makeSparseInputs(network, offsetsDType, {}, {}, DataType::UINT16);
    SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                             .network(network)
                                             .predictions(inputs.predictions)
                                             .labels(inputs.labels)
                                             .numClasses(3)
                                             .mask(inputs.mask)
                                             .reportsBatchLoss()
                                             .build();
    NetworkOutput::Builder().network(network).name("loss").inputTensor(loss.getLoss()).dataType(DataType::FP32).build();

    vector<Event> initDone;
    shared_ptr<PlacedNetwork> placed = network.place(3, initDone, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDone) event.synchronize();

    struct RuntimeCase {
        vector<uint64_t> offsets;
        vector<float> logits;
        vector<uint32_t> labels;
    };
    const vector<RuntimeCase> cases{
        {{0, 1, 1, 2}, {1.0f, 0.0f, -1.0f, -0.5f, 0.25f, 1.5f}, {0, 2}},
        {{0, 2, 2, 5},
         {1.0f, -0.5f, 0.25f,
          -1.0f, 0.5f, 1.5f,
          0.0f, 0.0f, 0.0f,
          2.0f, 0.25f, -1.0f,
          -0.25f, 1.25f, 0.5f},
         {0, 1, 2, 0, 1}},
        {{0, 0, 1, 1}, {0.5f, 1.0f, -0.5f}, {1}},
        {{0, 0, 0, 0}, {}, {}},
    };

    for (size_t pass = 0; pass < cases.size(); ++pass) {
        const RuntimeCase& runtime = cases[pass];
        vector<uint8_t> mask(runtime.labels.size(), uint8_t{1});
        Batch batch = makeBatch(offsetsDType,
                                5,
                                runtime.offsets,
                                runtime.logits,
                                runtime.labels,
                                mask,
                                DataType::UINT16);
        const vector<float> reported = copyFp32ToHost(placed->infer(batch).at("loss"));
        ASSERT_EQ(reported.size(), 1u);
        float expected = 0.0f;
        for (size_t token = 0; token < runtime.labels.size(); ++token)
            expected += sparseCe3(&runtime.logits[token * 3], runtime.labels[token]);
        expected /= 3.0f;
        EXPECT_NEAR(reported[0], expected, 4.0e-5f) << "reuse pass=" << pass;
    }
}

TEST(RaggedSparseCategoricalR11C1, QualificationPerExampleReportingHonorsEmptyRows) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    Network network("r11c1_sparse_qualification_per_example");
    SparseInputs inputs = makeSparseInputs(network, DataType::UINT32, {}, {}, DataType::UINT8);
    SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                             .network(network)
                                             .predictions(inputs.predictions)
                                             .labels(inputs.labels)
                                             .numClasses(3)
                                             .mask(inputs.mask)
                                             .reportsPerExampleLoss()
                                             .build();
    NetworkOutput::Builder().network(network).name("loss").inputTensor(loss.getLoss()).dataType(DataType::FP32).build();

    vector<Event> initDone;
    shared_ptr<PlacedNetwork> placed = network.place(3, initDone, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDone) event.synchronize();

    const vector<float> logits{
        1.0f, -0.5f, 0.25f,
        -1.0f, 0.5f, 1.5f,
        0.0f, 0.0f, 0.0f,
        2.0f, 0.25f, -1.0f,
        -0.25f, 1.25f, 0.5f,
    };
    const vector<uint32_t> labels{0, 1, 2, 0, 1};
    const vector<uint8_t> mask(labels.size(), uint8_t{1});
    const vector<float> reported = copyFp32ToHost(
        placed->infer(makeBatch(DataType::UINT32, 5, {0, 2, 2, 5}, logits, labels, mask, DataType::UINT8)).at("loss"));
    ASSERT_EQ(reported.size(), 3u);
    EXPECT_NEAR(reported[0], sparseCe3(&logits[0], 0) + sparseCe3(&logits[3], 1), 4.0e-5f);
    EXPECT_NEAR(reported[1], 0.0f, 1.0e-6f);
    EXPECT_NEAR(reported[2],
                sparseCe3(&logits[6], 2) + sparseCe3(&logits[9], 0) + sparseCe3(&logits[12], 1),
                4.0e-5f);
}

TEST(RaggedSparseCategoricalR11C1, QualificationRawInferencePreservesExactRuntimePartitionAndAllEmptyReuse) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    Network network("r11c1_sparse_qualification_raw_partition");
    SparseInputs inputs = makeSparseInputs(network, DataType::UINT64);
    SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                             .network(network)
                                             .predictions(inputs.predictions)
                                             .labels(inputs.labels)
                                             .numClasses(3)
                                             .mask(inputs.mask)
                                             .reportsRawLoss()
                                             .build();
    RaggedNetworkOutput::Builder().network(network).name("raw").inputTensor(loss.getRaggedLoss()).build();

    vector<Event> initDone;
    shared_ptr<PlacedNetwork> placed = network.place(3, initDone, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDone) event.synchronize();

    const vector<float> logits{
        1.0f, -0.5f, 0.25f,
        -1.0f, 0.5f, 1.5f,
        0.0f, 0.0f, 0.0f,
        2.0f, 0.25f, -1.0f,
        -0.25f, 1.25f, 0.5f,
    };
    const vector<uint32_t> labels{0, 1, 2, 0, 1};
    const vector<uint8_t> mask(labels.size(), uint8_t{1});
    map<string, InferenceOutputValue> outputs = placed->inferLogical(
        makeBatch(DataType::UINT64, 5, {0, 2, 2, 5}, logits, labels, mask));
    ASSERT_TRUE(outputs.contains("raw"));
    ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("raw")));
    const Impl::RaggedTensor raw = get<Impl::RaggedTensor>(outputs.at("raw"));
    EXPECT_EQ(raw.getHostActiveValueCountIfAvailable(), optional<uint64_t>(5));
    EXPECT_EQ(raw.getHostOffsetsIfAvailable(), (optional<vector<uint64_t>>{{0, 2, 2, 5}}));
    const vector<float> rawValues = copyFp32ToHost(raw.getValues());
    ASSERT_GE(rawValues.size(), 5u);
    for (size_t token = 0; token < 5; ++token)
        EXPECT_NEAR(rawValues[token], sparseCe3(&logits[token * 3], labels[token]), 4.0e-5f);

    outputs = placed->inferLogical(makeBatch(DataType::UINT64, 5, {0, 0, 0, 0}, {}, {}, {}));
    ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("raw")));
    const Impl::RaggedTensor empty = get<Impl::RaggedTensor>(outputs.at("raw"));
    EXPECT_EQ(empty.getHostActiveValueCountIfAvailable(), optional<uint64_t>(0));
    EXPECT_EQ(empty.getHostOffsetsIfAvailable(), (optional<vector<uint64_t>>{{0, 0, 0, 0}}));
}

TEST(RaggedSparseCategoricalR11C1, QualificationSubgraphCloneRemapsValuesAndSharedPartitionToken) {
    Network source("r11c1_sparse_clone_source");
    SparseInputs sourceInputs =
        makeSparseInputs(source, DataType::UINT64, {}, {}, DataType::UINT16, DataType::FP32);
    SparseCategoricalCrossEntropy sourceLoss = SparseCategoricalCrossEntropy::Builder()
                                                   .network(source)
                                                   .predictions(sourceInputs.predictions)
                                                   .labels(sourceInputs.labels)
                                                   .numClasses(3)
                                                   .ignoreIndex(2)
                                                   .mask(sourceInputs.mask)
                                                   .reportsRawLoss()
                                                   .build();
    NetworkOutput::Builder()
        .network(source)
        .name("raw_values")
        .inputTensor(sourceLoss.getRaggedLoss().getValues())
        .dataType(DataType::FP32)
        .build();

    Network destination("r11c1_sparse_clone_destination");
    SparseInputs destinationInputs =
        makeSparseInputs(destination, DataType::UINT64, {}, {}, DataType::UINT16, DataType::FP32);
    ApiTensorRemap remap;
    remap.map(sourceInputs.predictions.getValues(), destinationInputs.predictions.getValues());
    remap.map(sourceInputs.predictions.getRowPartitionToken(), destinationInputs.predictions.getRowPartitionToken());
    remap.map(sourceInputs.labels.getValues(), destinationInputs.labels.getValues());
    remap.map(sourceInputs.mask.getValues(), destinationInputs.mask.getValues());

    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    ApiSubgraphCloneResult clone = destination.cloneSubgraphInto(source, {"raw_values"}, remap, options);
    ASSERT_TRUE(clone.outputTensorsByName.contains("raw_values"));
    EXPECT_EQ(clone.outputTensorsByName.at("raw_values").getDimensions(), (vector<uint64_t>{8}));
    EXPECT_EQ(countLayerType(destination, "SparseCategoricalCrossEntropy"), 1u);

    shared_ptr<SparseCategoricalCrossEntropy> clonedLoss;
    for (uint32_t i = 0; i < destination.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<SparseCategoricalCrossEntropy>(destination.getLayer(i));
        if (candidate == nullptr) continue;
        ASSERT_EQ(clonedLoss, nullptr);
        clonedLoss = candidate;
    }
    ASSERT_NE(clonedLoss, nullptr);
    ASSERT_TRUE(clonedLoss->isRagged());
    EXPECT_EQ(clonedLoss->getRaggedPredictions().getValues(), destinationInputs.predictions.getValues());
    EXPECT_EQ(clonedLoss->getRaggedLabels().getValues(), destinationInputs.labels.getValues());
    EXPECT_TRUE(clonedLoss->getRaggedPredictions().sharesPartitionWith(destinationInputs.predictions));
    EXPECT_TRUE(clonedLoss->getRaggedLabels().sharesPartitionWith(destinationInputs.predictions));
    ASSERT_TRUE(clonedLoss->getRaggedMask().has_value());
    EXPECT_EQ(clonedLoss->getRaggedMask()->getValues(), destinationInputs.mask.getValues());
    EXPECT_TRUE(clonedLoss->getRaggedMask()->sharesPartitionWith(destinationInputs.predictions));
    EXPECT_TRUE(clonedLoss->getRaggedRawLoss().sharesPartitionWith(destinationInputs.predictions));
}

TEST(RaggedSparseCategoricalR11C1, SaveLoadPreservesInputsAndAcceptsChangedRuntimePartition) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device unavailable";

    constexpr DataType offsetsDType = DataType::UINT64;
    Network network("r11c1_sparse_round_trip");
    SparseInputs inputs = makeSparseInputs(network, offsetsDType);
    SparseCategoricalCrossEntropy loss = SparseCategoricalCrossEntropy::Builder()
                                             .network(network)
                                             .predictions(inputs.predictions)
                                             .labels(inputs.labels)
                                             .numClasses(3)
                                             .ignoreIndex(2)
                                             .mask(inputs.mask)
                                             .reportsBatchLoss()
                                             .build();
    NetworkOutput::Builder().network(network).name("loss").inputTensor(loss.getLoss()).dataType(DataType::FP32).build();

    const auto now = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archiveDir = filesystem::temp_directory_path() / (string("thor_r11c1_sparse_") + to_string(now));
    filesystem::remove_all(archiveDir);
    network.save(archiveDir.string(), /*overwrite=*/true);

    Network loaded("r11c1_sparse_round_trip");
    ASSERT_NO_THROW(loaded.load(archiveDir.string()));
    EXPECT_EQ(countLayerType(loaded, "SparseCategoricalCrossEntropy"), 1u);
    EXPECT_EQ(countLayerType(loaded, "RaggedLossShaper"), 1u);
    ASSERT_EQ(loaded.getExternalRaggedNetworkInputs().size(), 3u);
    map<string, RaggedNetworkInputReference> loadedInputs;
    for (const RaggedNetworkInputReference& input : loaded.getExternalRaggedNetworkInputs()) loadedInputs.emplace(input.name, input);
    ASSERT_TRUE(loadedInputs.contains("predictions"));
    ASSERT_TRUE(loadedInputs.contains("labels"));
    ASSERT_TRUE(loadedInputs.contains("mask"));
    EXPECT_TRUE(loadedInputs.at("predictions").raggedTensor.sharesPartitionWith(loadedInputs.at("labels").raggedTensor));
    EXPECT_TRUE(loadedInputs.at("predictions").raggedTensor.sharesPartitionWith(loadedInputs.at("mask").raggedTensor));
    EXPECT_TRUE(loadedInputs.at("labels").raggedTensor.getTrailingDimensions().empty());
    EXPECT_TRUE(loadedInputs.at("mask").raggedTensor.getTrailingDimensions().empty());

    vector<Event> initDone;
    shared_ptr<PlacedNetwork> placed = loaded.place(3, initDone, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDone) event.synchronize();

    const vector<float> logitsA{
        1.0f, -0.5f, 0.25f,
        -1.0f, 0.5f, 1.5f,
        0.0f, 0.0f, 0.0f,
        2.0f, 0.25f, -1.0f,
        -0.25f, 1.25f, 0.5f,
    };
    const vector<uint32_t> labelsA{0, 1, 2, 0, 1};
    const vector<uint8_t> maskA{1, 1, 1, 0, 1};
    Batch batchA = makeBatch(offsetsDType, 5, {0, 2, 2, 5}, logitsA, labelsA, maskA);
    map<string, Impl::Tensor> outputsA = placed->infer(batchA);
    const vector<float> hostA = copyFp32ToHost(outputsA.at("loss"));
    ASSERT_EQ(hostA.size(), 1u);
    const float expectedA = (sparseCe3(&logitsA[0], 0) + sparseCe3(&logitsA[3], 1) + sparseCe3(&logitsA[12], 1)) / 3.0f;
    EXPECT_NEAR(hostA[0], expectedA, 4.0e-5f);

    // Same loaded placement and capacity, but a different row partition and a
    // shorter active prefix. This proves the deserialized loss still consumes
    // the runtime partition carrier rather than baking the save-time offsets.
    const vector<float> logitsB{
        0.25f, 1.5f, -0.75f,
        1.25f, -0.25f, 0.5f,
        -0.5f, 0.25f, 1.0f,
    };
    const vector<uint32_t> labelsB{1, 0, 1};
    const vector<uint8_t> maskB{1, 1, 0};
    Batch batchB = makeBatch(offsetsDType, 5, {0, 1, 3, 3}, logitsB, labelsB, maskB);
    map<string, Impl::Tensor> outputsB = placed->infer(batchB);
    const vector<float> hostB = copyFp32ToHost(outputsB.at("loss"));
    ASSERT_EQ(hostB.size(), 1u);
    const float expectedB = (sparseCe3(&logitsB[0], 1) + sparseCe3(&logitsB[3], 0)) / 3.0f;
    EXPECT_NEAR(hostB[0], expectedB, 4.0e-5f);

    filesystem::remove_all(archiveDir);
}
