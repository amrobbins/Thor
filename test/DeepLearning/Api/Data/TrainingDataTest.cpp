#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Api/Data/FileDataset.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using Thor::BatchLease;
using ThorImplementation::DataType;
using ThorImplementation::Tensor;

namespace {

template <typename T>
concept HasPublicRelease = requires(T value) { value.release(); };

static_assert(!HasPublicRelease<Thor::BatchLease>);

std::filesystem::path makeDatasetPath(const std::string &name) {
    static uint64_t counter = 0;
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("thor_training_data_" + name + "_" + std::to_string(counter++));
    std::filesystem::remove_all(path);
    return path;
}

DatasetLayout layout() {
    return DatasetLayout::fromTensorShapes(
        std::vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("features", {1}, DataType::FP32)});
}

void writeDataset(const std::filesystem::path &path) {
    DatasetWriter writer(path, layout(), 2);
    for (uint64_t i = 0; i < 6; ++i) {
        float value = static_cast<float>(i);
        DatasetWriter::TensorView view{
            .dataType = DataType::FP32,
            .dimensions = {1},
            .data = &value,
            .numBytes = sizeof(value),
        };
        writer.writeIndexedExample({{"features", view}});
    }
    writer.close();
}

std::vector<float> values(const Batch &batch) {
    const Tensor &tensor = batch.getTensor("features");
    const float *data = tensor.getMemPtr<float>();
    return {data, data + tensor.getDescriptor().getTotalNumElements()};
}

}  // namespace

TEST(TrainingData, OpensIndependentSessionsOverOneImmutableRecipe) {
    const std::filesystem::path path = makeDatasetPath("independent_sessions");
    writeDataset(path);
    auto dataset = Thor::FileDataset::open(path);
    Thor::TrainingData data(dataset,
                            Thor::DatasetSplitManifest(*dataset, {0, 1, 2, 3}, {4, 5}),
                            Thor::BatchPolicy(2, false),
                            Thor::DatasetAccessPolicy{},
                            "shared_examples");

    std::shared_ptr<Thor::BatchSession> first = data.openSession(2);
    std::shared_ptr<Thor::BatchSession> second = data.openSession(2);
    ASSERT_NE(first.get(), second.get());
    EXPECT_EQ(first->getDatasetName(), "shared_examples");
    EXPECT_EQ(second->getDatasetName(), "shared_examples");

    uint64_t firstBatchNum = 99;
    uint64_t secondBatchNum = 99;
    BatchLease firstBatchLease = first->leaseBatch(ExampleType::TRAIN, firstBatchNum);

    const Batch& firstBatch = firstBatchLease.get();
    BatchLease secondBatchLease = second->leaseBatch(ExampleType::TRAIN, secondBatchNum);

    const Batch& secondBatch = secondBatchLease.get();
    EXPECT_EQ(firstBatchNum, 0);
    EXPECT_EQ(secondBatchNum, 0);
    EXPECT_EQ(values(firstBatch), (std::vector<float>{0.0f, 1.0f}));
    EXPECT_EQ(values(secondBatch), (std::vector<float>{0.0f, 1.0f}));

    firstBatchLease.reset();
    secondBatchLease.reset();
    std::filesystem::remove_all(path);
}

TEST(TrainingData, FixedSeedRandomizationIsSessionLocal) {
    const std::filesystem::path path = makeDatasetPath("randomization");
    writeDataset(path);
    auto dataset = Thor::FileDataset::open(path);
    Thor::TrainingData data(dataset,
                            Thor::DatasetSplitManifest(*dataset, {0, 1, 2, 3, 4, 5}, {}),
                            Thor::BatchPolicy(2, true, 1234));

    auto first = data.openSession(2);
    auto second = data.openSession(2);
    for (uint64_t i = 0; i < 4; ++i) {
        uint64_t firstBatchNum = 0;
        uint64_t secondBatchNum = 0;
        BatchLease firstBatchLease = first->leaseBatch(ExampleType::TRAIN, firstBatchNum);

        const Batch& firstBatch = firstBatchLease.get();
        BatchLease secondBatchLease = second->leaseBatch(ExampleType::TRAIN, secondBatchNum);

        const Batch& secondBatch = secondBatchLease.get();
        EXPECT_EQ(firstBatchNum, secondBatchNum);
        EXPECT_EQ(values(firstBatch), values(secondBatch));
        firstBatchLease.reset();
        secondBatchLease.reset();
    }
    std::filesystem::remove_all(path);
}

TEST(TrainingData, BatchLeaseReturnsBuffersAndCancellationIsSessionLocal) {
    const std::filesystem::path path = makeDatasetPath("lease_cancel");
    writeDataset(path);
    auto dataset = Thor::FileDataset::open(path);
    Thor::TrainingData data(dataset,
                            Thor::DatasetSplitManifest(*dataset, {0, 1, 2, 3}, {4, 5}),
                            Thor::BatchPolicy(2, false));

    auto cancelled = data.openSession(1);
    auto survivor = data.openSession(1);
    {
        uint64_t batchNum = 0;
        Thor::BatchLease lease = survivor->leaseBatch(ExampleType::TRAIN, batchNum);
        EXPECT_EQ(values(lease.get()), (std::vector<float>{0.0f, 1.0f}));
    }

    cancelled->cancel();
    uint64_t cancelledBatchNum = 0;
    EXPECT_THROW(cancelled->leaseBatch(ExampleType::TRAIN, cancelledBatchNum), std::runtime_error);

    uint64_t survivorBatchNum = 0;
    BatchLease batchLease = survivor->leaseBatch(ExampleType::TRAIN, survivorBatchNum);

    const Batch& batch = batchLease.get();
    EXPECT_EQ(values(batch), (std::vector<float>{2.0f, 3.0f}));
    batchLease.reset();
    std::filesystem::remove_all(path);
}

TEST(TrainingData, AllowsEmptyPartitionsAndEnforcesThemAtOperationBoundaries) {
    const std::filesystem::path path = makeDatasetPath("validation");
    writeDataset(path);
    auto dataset = Thor::FileDataset::open(path);
    Thor::TrainingData data(dataset,
                            Thor::DatasetSplitManifest(*dataset, {}, {0}, std::vector<uint64_t>{1}),
                            Thor::BatchPolicy(1, false));
    EXPECT_THROW((void)data.openSession(0), std::runtime_error);
    EXPECT_THROW(data.requireNonEmptyPartition(ExampleType::TRAIN, "fit probe"), std::runtime_error);
    EXPECT_NO_THROW(data.requireNonEmptyPartition(ExampleType::VALIDATE, "validate probe"));
    EXPECT_NO_THROW(data.requireNonEmptyPartition(ExampleType::TEST, "test probe"));

    std::shared_ptr<Thor::BatchSession> session = data.openSession(1);
    EXPECT_EQ(session->getNumExamples(ExampleType::TRAIN), 0);
    EXPECT_EQ(session->getNumExamples(ExampleType::VALIDATE), 1);
    EXPECT_EQ(session->getNumExamples(ExampleType::TEST), 1);
    std::filesystem::remove_all(path);
}

TEST(TrainingData, BatchLeaseKeepsOwningSessionAliveUntilRecycled) {
    const std::filesystem::path path = makeDatasetPath("lease_owns_session");
    writeDataset(path);
    auto dataset = Thor::FileDataset::open(path);
    Thor::TrainingData data(dataset,
                            Thor::DatasetSplitManifest(*dataset, {0, 1}, {2}),
                            Thor::BatchPolicy(1, false));

    std::shared_ptr<Thor::BatchSession> session = data.openSession(1);
    std::weak_ptr<Thor::BatchSession> weakSession = session;
    uint64_t batchNum = 0;
    BatchLease lease = session->leaseBatch(ExampleType::TRAIN, batchNum);
    session.reset();
    EXPECT_FALSE(weakSession.expired());
    lease.reset();
    EXPECT_TRUE(weakSession.expired());

    std::filesystem::remove_all(path);
}

TEST(TrainingData, OwnsDeviceAccessPolicyIndependentlyOfSessions) {
    const std::filesystem::path path = makeDatasetPath("access_policy");
    writeDataset(path);
    auto dataset = Thor::FileDataset::open(path);
    Thor::DatasetSplitManifest splits(*dataset, {0, 1, 2, 3}, {4, 5});
    Thor::BatchPolicy batching(2, false);

    Thor::TrainingData defaultData(dataset, splits, batching);
    Thor::TrainingData strictData(
        dataset,
        splits,
        batching,
        Thor::DatasetAccessPolicy{
            .deviceStorage = Thor::DeviceDatasetStorage::STRICT});

    EXPECT_EQ(defaultData.getAccessPolicy().deviceStorage,
              Thor::DeviceDatasetStorage::BEST_EFFORT);
    EXPECT_EQ(strictData.getAccessPolicy().deviceStorage,
              Thor::DeviceDatasetStorage::STRICT);

    std::shared_ptr<Thor::BatchSession> first = strictData.openSession(1);
    std::shared_ptr<Thor::BatchSession> second = strictData.openSession(1);
    ASSERT_NE(first, second);
    EXPECT_EQ(strictData.getAccessPolicy().deviceStorage,
              Thor::DeviceDatasetStorage::STRICT);
    std::filesystem::remove_all(path);
}
