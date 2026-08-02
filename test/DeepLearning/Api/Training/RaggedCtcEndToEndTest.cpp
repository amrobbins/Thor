#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/DatasetAccessPolicy.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Api/Data/ExampleType.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Training/DatasetInputBindings.h"
#include "DeepLearning/Api/Training/Trainer.h"
#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

constexpr uint64_t BATCH_SIZE = 2;
constexpr uint64_t MAX_TOTAL_LABELS = 2;
constexpr uint64_t TIME_STEPS = 2;
constexpr uint64_t NUM_CLASSES = 2;

class ScopedTempDirectory {
   public:
    explicit ScopedTempDirectory(const std::string& prefix) {
        static std::atomic<uint64_t> serial{0};
        const uint64_t tick =
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() /
                (prefix + "-" + std::to_string(tick) + "-" +
                 std::to_string(serial.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(path_);
    }

    ~ScopedTempDirectory() {
        std::error_code errorCode;
        std::filesystem::remove_all(path_, errorCode);
    }

    const std::filesystem::path& path() const { return path_; }

   private:
    std::filesystem::path path_;
};

DatasetLayout ctcDatasetLayout() {
    return DatasetLayout::fromTensorShapes(
        {
            DatasetLayout::TensorShape(
                "logits", {TIME_STEPS, NUM_CLASSES}, Impl::DataType::FP32),
            DatasetLayout::TensorShape(
                "input_lengths", {1}, Impl::DataType::INT32),
        },
        {
            DatasetLayout::RaggedTensorShape(
                "labels", {}, Impl::DataType::INT32),
        });
}

void writeCtcDataset(const std::filesystem::path& datasetPath) {
    DatasetLayout layout = ctcDatasetLayout();
    DatasetWriter writer(datasetPath, layout, /*examplesPerShard=*/2);

    // Four examples. Training uses rows 0,1,2 so batch size 2 produces one
    // exact one-example tail; validation uses row 3.
    const std::vector<float> logits(4 * TIME_STEPS * NUM_CLASSES, 0.0f);
    const std::vector<int32_t> inputLengths{2, 2, 2, 2};

    // Targets: [1], [], [1], [1].
    const std::vector<int32_t> labels{1, 1, 1};
    const std::vector<uint64_t> labelOffsets{0, 1, 1, 2, 3};

    writer.writeIndexedExamples(
        {
            {"logits",
             DatasetWriter::TensorBatchView{
                 .dataType = Impl::DataType::FP32,
                 .dimensions = {4, TIME_STEPS, NUM_CLASSES},
                 .data = logits.data(),
                 .numBytes = logits.size() * sizeof(float)}},
            {"input_lengths",
             DatasetWriter::TensorBatchView{
                 .dataType = Impl::DataType::INT32,
                 .dimensions = {4, 1},
                 .data = inputLengths.data(),
                 .numBytes = inputLengths.size() * sizeof(int32_t)}},
        },
        {
            {"labels",
             DatasetWriter::RaggedTensorBatchView{
                 .dataType = Impl::DataType::INT32,
                 .dimensions = {labels.size()},
                 .data = labels.data(),
                 .numBytes = labels.size() * sizeof(int32_t),
                 .offsetsDataType = Impl::DataType::UINT64,
                 .offsets = labelOffsets.data(),
                 .count = 4}},
        });
    writer.close();
}

struct CtcNetwork {
    std::shared_ptr<Api::Network> network;
    Api::CtcLoss loss;
};

CtcNetwork buildCtcNetwork(const std::string& name) {
    auto network = std::make_shared<Api::Network>(name);

    Api::NetworkInput logits =
        Api::NetworkInput::Builder()
            .network(*network)
            .name("logits")
            .dimensions({TIME_STEPS, NUM_CLASSES})
            .dataType(Impl::DataType::FP32)
            .build();
    Api::RaggedTensor labels =
        Api::RaggedNetworkInput::Builder()
            .network(*network)
            .name("labels")
            .valuesDataType(Impl::DataType::INT32)
            .offsetsDataType(Impl::DataType::UINT64)
            .trailingDimensions({})
            .maxTotalValues(MAX_TOTAL_LABELS)
            .batchSize(BATCH_SIZE)
            .build();
    Api::NetworkInput inputLengths =
        Api::NetworkInput::Builder()
            .network(*network)
            .name("input_lengths")
            .dimensions({1})
            .dataType(Impl::DataType::INT32)
            .build();

    std::shared_ptr<Api::Initializer> zeroInitializer =
        Api::UniformRandom::Builder().minValue(0.0f).maxValue(0.0f).build();
    Api::FullyConnected logitsProjection =
        Api::FullyConnected::Builder()
            .network(*network)
            .featureInput(logits.getFeatureOutput().value())
            .numOutputFeatures(NUM_CLASSES)
            .hasBias(true)
            .preserveInputPrefixDimensions(true)
            .weightsInitializer(zeroInitializer)
            .biasInitializer(zeroInitializer)
            .weightsDataType(Impl::DataType::FP32)
            .computeDataType(Impl::DataType::FP32)
            .outputDataType(Impl::DataType::FP32)
            .noActivation()
            .build();

    std::shared_ptr<Api::Sgd> optimizer =
        Api::Sgd::Builder().network(*network).initialLearningRate(0.01f).build();
    optimizer->setConstantLearningRate(0.0f, nullptr);

    Api::CtcLoss loss =
        Api::CtcLoss::Builder()
            .network(*network)
            .logits(logitsProjection.getFeatureOutput().value())
            .labels(labels)
            .inputLengths(inputLengths.getFeatureOutput().value())
            .reportsRawLoss()
            .build();

    Api::NetworkOutput::Builder()
        .network(*network)
        .name("ctc_loss")
        .inputTensor(loss.getLoss())
        .dataType(Impl::DataType::FP32)
        .build();

    return CtcNetwork{.network = std::move(network), .loss = loss};
}

std::shared_ptr<Api::TrainingData> makeTrainingData(
    const std::shared_ptr<Api::FileDataset>& dataset) {
    return std::make_shared<Api::TrainingData>(
        dataset,
        Api::DatasetSplitManifest(*dataset, {0, 1, 2}, {3}),
        Api::BatchPolicy(BATCH_SIZE, /*randomizeTrain=*/false),
        Api::DatasetAccessPolicy{.deviceStorage = Api::DeviceDatasetStorage::OFF},
        "ragged_ctc_e2e");
}

std::vector<float> copyFp32TensorToHost(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getDescriptor().getDataType(), Impl::DataType::FP32);
    const uint64_t numElements = tensor.getTotalNumElements();
    if (tensor.getPlacement().getMemDevice() == Impl::TensorPlacement::MemDevices::CPU) {
        const float* values = tensor.getMemPtr<float>();
        return std::vector<float>(values, values + numElements);
    }

    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = tensor.clone(cpuPlacement);
    Stream stream(tensor.getPlacement().getDeviceNum());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();

    const float* values = host.getMemPtr<float>();
    return std::vector<float>(values, values + numElements);
}

std::vector<uint64_t> raggedOffsetsAsUint64(const Impl::RaggedTensor& ragged) {
    const Impl::Tensor offsets = ragged.getOffsets();
    std::vector<uint64_t> result(ragged.getBatchSize() + 1, 0);
    if (ragged.getOffsetsDataType() == Impl::DataType::UINT32) {
        const uint32_t* values = offsets.getMemPtr<uint32_t>();
        for (uint64_t i = 0; i < result.size(); ++i)
            result[i] = values[i];
        return result;
    }
    EXPECT_EQ(ragged.getOffsetsDataType(), Impl::DataType::UINT64);
    const uint64_t* values = offsets.getMemPtr<uint64_t>();
    for (uint64_t i = 0; i < result.size(); ++i)
        result[i] = values[i];
    return result;
}

void expectInactiveGradientTailZero(const std::vector<float>& gradient,
                                    uint64_t firstInactiveBatchRow) {
    const uint64_t elementsPerRow = TIME_STEPS * NUM_CLASSES;
    ASSERT_EQ(gradient.size(), BATCH_SIZE * elementsPerRow);
    for (uint64_t row = firstInactiveBatchRow; row < BATCH_SIZE; ++row) {
        for (uint64_t element = 0; element < elementsPerRow; ++element) {
            EXPECT_FLOAT_EQ(gradient[row * elementsPerRow + element], 0.0f)
                << "row=" << row << " element=" << element;
        }
    }
}

}  // namespace

TEST(RaggedCtcEndToEnd, FileDatasetFeedsCanonicalRaggedCtcForwardBackwardAndExactTail) {
    ScopedTempDirectory temp("thor-ragged-ctc-e2e");
    const std::filesystem::path datasetPath = temp.path() / "dataset";
    writeCtcDataset(datasetPath);

    std::shared_ptr<Api::FileDataset> dataset = Api::FileDataset::open(datasetPath);
    ASSERT_NE(dataset, nullptr);

    CtcNetwork graph = buildCtcNetwork("ragged_ctc_e2e_direct");
    Api::DatasetInputBindings bindings =
        Api::DatasetInputBindings::byExactName(*graph.network, *dataset);
    Api::CompiledDatasetInputBindings compiled =
        bindings.compile(*graph.network, *dataset, BATCH_SIZE);

    auto data = makeTrainingData(dataset);
    std::shared_ptr<Api::BatchSession> session =
        data->openSession(/*maxInFlightBatches=*/2, compiled.fieldRequirements);
    ASSERT_NE(session, nullptr);

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed =
        graph.network->place(BATCH_SIZE, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents)
        event.synchronize();
    ASSERT_NE(placed, nullptr);

    std::shared_ptr<Impl::CtcLoss> physicalCtc;
    for (const std::shared_ptr<Impl::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
        std::shared_ptr<Impl::CtcLoss> candidate = std::dynamic_pointer_cast<Impl::CtcLoss>(layer);
        if (candidate == nullptr) {
            continue;
        }
        ASSERT_EQ(physicalCtc, nullptr) << "Expected exactly one physical CTC loss layer.";
        physicalCtc = std::move(candidate);
    }
    ASSERT_NE(physicalCtc, nullptr);

    uint64_t batchNum = 99;
    {
        Api::BatchLease lease = session->leaseBatch(ExampleType::TRAIN, batchNum);
        ASSERT_FALSE(lease.empty());
        EXPECT_EQ(batchNum, 0u);
        EXPECT_FALSE(lease.get().getValidExampleCount().has_value());
        ASSERT_TRUE(lease.get().isRaggedTensor("labels"));
        EXPECT_EQ(
            raggedOffsetsAsUint64(lease.get().getRaggedTensor("labels")),
            (std::vector<uint64_t>{0, 1, 1}));

        std::map<std::string, Impl::Tensor> outputs;
        std::map<std::string, Event> outputReadyEvents;
        Event done = placed->submitBatch(
            0, lease.get(), outputs, outputReadyEvents, /*isInferenceOnly=*/false);
        done.synchronize();

        ASSERT_EQ(outputs.count("ctc_loss"), 1u);
        const std::vector<float> loss = copyFp32TensorToHost(outputs.at("ctc_loss"));
        ASSERT_EQ(loss.size(), BATCH_SIZE);
        EXPECT_NEAR(loss[0], -std::log(0.75f), 2.0e-4f);
        EXPECT_NEAR(loss[1], -std::log(0.25f), 2.0e-4f);

        ASSERT_TRUE(physicalCtc->getErrorOutput().has_value());
        const std::vector<float> gradient =
            copyFp32TensorToHost(physicalCtc->getErrorOutput().value());
        ASSERT_EQ(gradient.size(), BATCH_SIZE * TIME_STEPS * NUM_CLASSES);
        bool anyNonZero = false;
        for (float value : gradient) {
            EXPECT_TRUE(std::isfinite(value));
            anyNonZero = anyNonZero || std::abs(value) > 1.0e-6f;
        }
        EXPECT_TRUE(anyNonZero);
    }

    {
        Api::BatchLease lease = session->leaseBatch(ExampleType::TRAIN, batchNum);
        ASSERT_FALSE(lease.empty());
        EXPECT_EQ(batchNum, 1u);
        ASSERT_TRUE(lease.get().getValidExampleCount().has_value());
        EXPECT_EQ(lease.get().getValidExampleCount().value(), 1u);
        ASSERT_TRUE(lease.get().isRaggedTensor("labels"));
        EXPECT_EQ(
            raggedOffsetsAsUint64(lease.get().getRaggedTensor("labels")),
            (std::vector<uint64_t>{0, 1, 1}));

        std::map<std::string, Impl::Tensor> outputs;
        std::map<std::string, Event> outputReadyEvents;
        Event done = placed->submitBatch(
            0, lease.get(), outputs, outputReadyEvents, /*isInferenceOnly=*/false);
        done.synchronize();

        const std::vector<float> loss = copyFp32TensorToHost(outputs.at("ctc_loss"));
        ASSERT_EQ(loss.size(), BATCH_SIZE);
        EXPECT_NEAR(loss[0], -std::log(0.75f), 2.0e-4f);
        EXPECT_FLOAT_EQ(loss[1], 0.0f);

        ASSERT_TRUE(physicalCtc->getErrorOutput().has_value());
        const std::vector<float> gradient =
            copyFp32TensorToHost(physicalCtc->getErrorOutput().value());
        expectInactiveGradientTailZero(gradient, /*firstInactiveBatchRow=*/1);
    }

    placed->synchronize();
    session->cancel();
}

TEST(RaggedCtcEndToEnd, SerializedNetworkTrainsFromCurrentDatasetThroughLogicalBinding) {
    ScopedTempDirectory temp("thor-ragged-ctc-serialized");
    const std::filesystem::path datasetPath = temp.path() / "dataset";
    const std::filesystem::path modelPath = temp.path() / "model";
    std::filesystem::create_directories(modelPath);
    writeCtcDataset(datasetPath);

    CtcNetwork source = buildCtcNetwork("ragged_ctc_e2e_serialized");
    source.network->save(modelPath.string(), /*overwrite=*/true);

    auto loaded = std::make_shared<Api::Network>("ragged_ctc_e2e_serialized");
    loaded->load(modelPath.string());
    EXPECT_TRUE(loaded->hasRaggedNetworkInput("labels"));
    const std::vector<Api::RaggedNetworkInputReference> raggedInputs =
        loaded->getExternalRaggedNetworkInputs();
    ASSERT_EQ(raggedInputs.size(), 1u);
    EXPECT_EQ(raggedInputs.front().name, "labels");
    EXPECT_EQ(raggedInputs.front().raggedTensor.getOffsetsDataType(), Impl::DataType::UINT64);
    EXPECT_EQ(raggedInputs.front().raggedTensor.getBatchSize(), BATCH_SIZE);
    EXPECT_EQ(raggedInputs.front().raggedTensor.getMaxTotalValues(), MAX_TOTAL_LABELS);

    const std::vector<Api::NetworkLossReference> reportableLosses = loaded->getReportableLosses();
    ASSERT_EQ(reportableLosses.size(), 1u);
    EXPECT_EQ(reportableLosses.front().lossName, "ctc_loss");
    EXPECT_EQ(reportableLosses.front().targetInputName, "labels");
    EXPECT_EQ(std::count(reportableLosses.front().requiredInputNames.begin(),
                         reportableLosses.front().requiredInputNames.end(),
                         "labels"),
              1);
    EXPECT_EQ(std::count(reportableLosses.front().requiredInputNames.begin(),
                         reportableLosses.front().requiredInputNames.end(),
                         "labels.values"),
              0);
    EXPECT_EQ(std::count(reportableLosses.front().requiredInputNames.begin(),
                         reportableLosses.front().requiredInputNames.end(),
                         "labels.offsets"),
              0);

    std::shared_ptr<Api::FileDataset> dataset = Api::FileDataset::open(datasetPath);
    auto data = makeTrainingData(dataset);
    Api::DatasetInputBindings bindings =
        Api::DatasetInputBindings::byExactName(*loaded, *dataset);

    Api::Trainer trainer =
        Api::Trainer::Builder()
            .network(loaded)
            .data(data)
            .inputBindings(bindings)
            .debugSynchronousExecutor()
            .observer(std::make_shared<Api::NullTrainingObserver>())
            .maxInFlightBatches(2)
            .build();

    const Api::TrainingRunResult result = trainer.fit(1);
    EXPECT_TRUE(result.completed());
    ASSERT_TRUE(result.completedEpoch.has_value());
    EXPECT_EQ(result.completedEpoch.value(), 1u);
    ASSERT_TRUE(result.finalTrainingStats.has_value());
    EXPECT_EQ(result.finalTrainingStats->samplesProcessed, 3u);
    EXPECT_EQ(result.finalTrainingStats->samplesProcessedInEpoch, 3u);
    EXPECT_EQ(result.finalTrainingStats->batchSize, BATCH_SIZE);
    EXPECT_EQ(result.finalTrainingStats->validExamplesInBatch, 1u);
}
