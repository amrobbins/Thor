#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Data/BatchSession.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::RaggedTensor;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;
using Thor::DeviceBatchMaterializer;
using Thor::DeviceBatchReference;

namespace {

Tensor makeCpuTensor(DataType dataType, std::vector<uint64_t> dimensions) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    return Tensor(cpuPlacement, TensorDescriptor(dataType, std::move(dimensions)));
}

class RecycleTokenTestSession final : public Thor::BatchSession {
   public:
    void attachToken(Batch& batch, std::shared_ptr<void> token) {
        setBatchRecycleToken(batch, std::move(token));
    }

    std::shared_ptr<void> takeToken(Batch& batch) {
        return takeBatchRecycleToken(batch);
    }

    void attachSource(
        Batch& batch,
        std::set<std::string> fieldNames,
        Thor::BatchSourceOwner owner) {
        addBatchSourceResource(
            batch,
            std::move(fieldNames),
            std::move(owner));
    }

    uint64_t getNumBatchesPerEpoch(ExampleType) override { return 0; }
    uint64_t getNumExamples(ExampleType) override { return 0; }
    uint64_t getNextBatchNum(ExampleType) override { return 0; }

   private:
    Batch acquireBatch(ExampleType, uint64_t&) override { return {}; }
    void recycleBatch(ExampleType, Batch&&) override {}
};

class DescriptorOnlyDeviceBatchMaterializer : public DeviceBatchMaterializer {
   public:
    DescriptorOnlyDeviceBatchMaterializer(TensorDescriptor descriptor, TensorPlacement placement)
        : descriptor(std::move(descriptor)), placement(placement) {}

    TensorDescriptor getOutputDescriptor() const override { return descriptor; }
    TensorPlacement getOutputPlacement() const override { return placement; }

    void enqueueMaterialization(Tensor& destination, Stream& destinationStream) const override {
        (void)destination;
        (void)destinationStream;
        throw std::logic_error("DescriptorOnlyDeviceBatchMaterializer cannot enqueue.");
    }

   private:
    TensorDescriptor descriptor;
    TensorPlacement placement;
};

}  // namespace

TEST(Batch, StoresAndRetrievesDenseTensorValues) {
    Tensor examples = makeCpuTensor(DataType::FP32, {4, 3});
    Tensor labels = makeCpuTensor(DataType::INT32, {4});

    Batch batch;
    batch.insert("examples", examples);
    batch.insert("labels", labels);

    EXPECT_EQ(batch.size(), 2u);
    EXPECT_TRUE(batch.contains("examples"));
    EXPECT_TRUE(batch.isTensor("examples"));
    EXPECT_FALSE(batch.isRaggedTensor("examples"));
    EXPECT_TRUE(batch.isDenseOnly());
    EXPECT_EQ(batch.getTensor("examples"), examples);
    EXPECT_EQ(batch.getTensor("labels"), labels);
}

TEST(Batch, WrapsDenseTensorMapWithoutDeepCopyingHandles) {
    Tensor examples = makeCpuTensor(DataType::FP32, {2, 8});
    Tensor labels = makeCpuTensor(DataType::INT32, {2});
    std::map<std::string, Tensor> tensorMap{{"examples", examples}, {"labels", labels}};

    Batch batch = batchFromTensorMap(std::move(tensorMap));
    std::map<std::string, Tensor> dense = denseTensorMapFromBatchOrThrow(batch, "test");

    ASSERT_EQ(dense.size(), 2u);
    EXPECT_EQ(dense.at("examples"), examples);
    EXPECT_EQ(dense.at("labels"), labels);
}

TEST(Batch, StoresAndRetrievesRaggedTensorValues) {
    Tensor values = makeCpuTensor(DataType::INT32, {7});
    Tensor offsets = makeCpuTensor(DataType::UINT32, {4});
    RaggedTensor ragged(values, offsets);

    Batch batch;
    batch.insert("labels", ragged);

    EXPECT_EQ(batch.size(), 1u);
    EXPECT_TRUE(batch.contains("labels"));
    EXPECT_FALSE(batch.isDenseOnly());
    EXPECT_FALSE(batch.isTensor("labels"));
    EXPECT_TRUE(batch.isRaggedTensor("labels"));
    EXPECT_EQ(batch.getRaggedTensor("labels"), ragged);
    EXPECT_THROW(batch.getTensor("labels"), std::runtime_error);
}

TEST(Batch, DenseTensorMapConversionRejectsRaggedValues) {
    Tensor values = makeCpuTensor(DataType::INT32, {7});
    Tensor offsets = makeCpuTensor(DataType::UINT32, {4});
    RaggedTensor ragged(values, offsets);

    Batch batch;
    batch.insert("labels", ragged);

    EXPECT_THROW((denseTensorMapFromBatchOrThrow(batch, "test execution path")), std::runtime_error);
}

TEST(Batch, RejectsDuplicateNames) {
    Tensor examples = makeCpuTensor(DataType::FP32, {4, 3});

    Batch batch;
    batch.insert("examples", examples);

    EXPECT_THROW(batch.insert("examples", examples), std::logic_error);
}


TEST(Batch, StoresAndRetrievesDeviceBatchReferences) {
    TensorDescriptor descriptor(DataType::FP32, {4, 3});
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    auto materializer = std::make_shared<DescriptorOnlyDeviceBatchMaterializer>(descriptor, gpuPlacement);
    DeviceBatchReference reference(materializer, 4);

    Batch batch;
    batch.insert("examples", reference);

    EXPECT_EQ(batch.size(), 1u);
    EXPECT_TRUE(batch.contains("examples"));
    EXPECT_FALSE(batch.isDenseOnly());
    EXPECT_FALSE(batch.isTensor("examples"));
    EXPECT_FALSE(batch.isRaggedTensor("examples"));
    EXPECT_TRUE(batch.isDeviceBatchReference("examples"));
    EXPECT_EQ(batch.getDeviceBatchReference("examples").getBatchSize(), 4u);
    EXPECT_EQ(batch.getDeviceBatchReference("examples").getOutputDescriptor(), descriptor);
    EXPECT_EQ(batch.getDeviceBatchReference("examples").getOutputPlacement(), gpuPlacement);
    EXPECT_THROW(batch.getTensor("examples"), std::runtime_error);
    EXPECT_THROW((denseTensorMapFromBatchOrThrow(batch, "test execution path")), std::runtime_error);
}


TEST(Batch, RecycleTokenMovesWithLeaseButIsNotCopiedIntoDerivedBatches) {
    RecycleTokenTestSession session;
    Batch leasedBatch;
    leasedBatch.insert("examples", makeCpuTensor(DataType::FP32, {2, 3}));

    auto token = std::make_shared<int>(7);
    session.attachToken(leasedBatch, token);

    Batch copiedBatch = leasedBatch;
    EXPECT_EQ(session.takeToken(copiedBatch), nullptr);
    EXPECT_EQ(session.takeToken(leasedBatch), token);

    session.attachToken(leasedBatch, token);
    Batch movedBatch = std::move(leasedBatch);
    EXPECT_EQ(session.takeToken(leasedBatch), nullptr);
    EXPECT_EQ(session.takeToken(movedBatch), token);
}

TEST(Batch, ReportsWhetherEveryFieldHasTrackedSourceStorage) {
    RecycleTokenTestSession session;
    Batch batch;
    batch.insert("examples", makeCpuTensor(DataType::FP32, {2, 3}));
    batch.insert("labels", makeCpuTensor(DataType::FP32, {2, 1}));

    EXPECT_FALSE(batch.allFieldsHaveSourceReferences());

    Thor::BatchSourceOwner examplesOwner([](std::vector<Event>) {});
    session.attachSource(batch, {"examples"}, std::move(examplesOwner));
    EXPECT_FALSE(batch.allFieldsHaveSourceReferences());

    Thor::BatchSourceOwner labelsOwner([](std::vector<Event>) {});
    session.attachSource(batch, {"labels"}, std::move(labelsOwner));
    EXPECT_TRUE(batch.allFieldsHaveSourceReferences());
    EXPECT_TRUE(batch.ownsSourceResourceLifecycle());

    Batch copiedBatch = batch;
    EXPECT_TRUE(copiedBatch.allFieldsHaveSourceReferences());
    EXPECT_FALSE(copiedBatch.ownsSourceResourceLifecycle());

    Batch movedBatch = std::move(batch);
    EXPECT_TRUE(movedBatch.ownsSourceResourceLifecycle());
}
