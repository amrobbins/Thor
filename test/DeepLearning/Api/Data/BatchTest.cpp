#include "DeepLearning/Api/Data/Batch.h"

#include "gtest/gtest.h"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::RaggedTensor;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

Tensor makeCpuTensor(DataType dataType, std::vector<uint64_t> dimensions) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    return Tensor(cpuPlacement, TensorDescriptor(dataType, std::move(dimensions)));
}

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
