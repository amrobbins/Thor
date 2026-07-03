#include "Utilities/Loaders/LocalNamedBatchAssembler.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

std::filesystem::path makeTempDatasetPath(const std::string &name) {
    static uint64_t counter = 0;
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("thor_local_named_batch_assembler_" + name + "_" + std::to_string(counter++));
    std::filesystem::remove_all(path);
    return path;
}

LocalNamedExampleLayout testLayout() {
    return LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"seasonality_inputs", {2}}, {"monotone_inputs", {3}}, {"daily_weight", {1}}},
        DataType::FP32);
}

LocalNamedExampleDatasetWriter::TensorView tensorView(vector<float> &values, vector<uint64_t> dimensions) {
    return LocalNamedExampleDatasetWriter::TensorView{.dataType = DataType::FP32,
                                                      .dimensions = std::move(dimensions),
                                                      .data = values.data(),
                                                      .numBytes = values.size() * sizeof(float)};
}

map<string, LocalNamedExampleDatasetWriter::TensorView> exampleViews(vector<float> &seasonality,
                                                                     vector<float> &monotone,
                                                                     vector<float> &weight) {
    return {{"seasonality_inputs", tensorView(seasonality, {2})},
            {"monotone_inputs", tensorView(monotone, {3})},
            {"daily_weight", tensorView(weight, {1})}};
}

void writeExample(LocalNamedExampleDatasetWriter &writer, ExampleType exampleType, float base) {
    vector<float> seasonality{base, base + 1.0f};
    vector<float> monotone{base + 10.0f, base + 11.0f, base + 12.0f};
    vector<float> weight{base + 100.0f};
    writer.writeExample(exampleType, exampleViews(seasonality, monotone, weight));
}

std::vector<std::shared_ptr<Shard>> openShards(const std::filesystem::path &datasetPath) {
    std::vector<std::shared_ptr<Shard>> shards;
    for (const auto &entry : std::filesystem::directory_iterator(datasetPath)) {
        if (entry.path().extension() != ".shard") {
            continue;
        }
        auto shard = std::make_shared<Shard>();
        shard->openShard(entry.path().string());
        shards.push_back(std::move(shard));
    }
    std::sort(shards.begin(), shards.end(), [](const std::shared_ptr<Shard> &a, const std::shared_ptr<Shard> &b) {
        return a->getFilename() < b->getFilename();
    });
    return shards;
}

void expectTensorValues(const Tensor &tensor, const vector<float> &expected) {
    ASSERT_EQ(tensor.getDescriptor().getArraySizeInBytes(), expected.size() * sizeof(float));
    const float *actual = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "i=" << i;
    }
}

}  // namespace

TEST(LocalNamedBatchAssemblerTest, AssemblesSequentialNamedBatches) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("sequential");
    LocalNamedExampleLayout layout = testLayout();
    {
        LocalNamedExampleDatasetWriter writer(datasetPath, layout, 8);
        writeExample(writer, ExampleType::TRAIN, 0.0f);
        writeExample(writer, ExampleType::TRAIN, 20.0f);
        writeExample(writer, ExampleType::TRAIN, 40.0f);
        writer.close();
    }

    LocalNamedBatchAssembler assembler(openShards(datasetPath), ExampleType::TRAIN, layout, 2, 2, false);
    EXPECT_EQ(assembler.getNumExamples(), 3);
    EXPECT_EQ(assembler.getNumBatchesPerEpoch(), 2);

    map<string, Tensor> tensors;
    uint64_t batchNum = 99;
    assembler.getBatch(tensors, batchNum);
    EXPECT_EQ(batchNum, 0);
    ASSERT_EQ(tensors.size(), 3);
    expectTensorValues(tensors.at("seasonality_inputs"), {0.0f, 1.0f, 20.0f, 21.0f});
    expectTensorValues(tensors.at("monotone_inputs"), {10.0f, 11.0f, 12.0f, 30.0f, 31.0f, 32.0f});
    expectTensorValues(tensors.at("daily_weight"), {100.0f, 120.0f});
    assembler.returnBuffers(tensors);

    assembler.getBatch(tensors, batchNum);
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(tensors.at("seasonality_inputs"), {40.0f, 41.0f, 0.0f, 1.0f});
    assembler.returnBuffers(tensors);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedBatchAssemblerTest, FixedSeedRandomizationIsDeterministic) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("fixed_seed");
    LocalNamedExampleLayout layout = testLayout();
    {
        LocalNamedExampleDatasetWriter writer(datasetPath, layout, 16);
        for (uint64_t i = 0; i < 8; ++i) {
            writeExample(writer, ExampleType::TRAIN, static_cast<float>(i * 10));
        }
        writer.close();
    }

    map<string, Tensor> tensorsA;
    map<string, Tensor> tensorsB;
    uint64_t batchNumA = 0;
    uint64_t batchNumB = 0;
    vector<float> firstRun;
    vector<float> secondRun;

    {
        LocalNamedBatchAssembler assembler(openShards(datasetPath), ExampleType::TRAIN, layout, 4, 2, true, 1234);
        assembler.getBatch(tensorsA, batchNumA);
        const float *values = tensorsA.at("seasonality_inputs").getMemPtr<float>();
        firstRun.assign(values, values + 8);
        assembler.returnBuffers(tensorsA);
    }
    {
        LocalNamedBatchAssembler assembler(openShards(datasetPath), ExampleType::TRAIN, layout, 4, 2, true, 1234);
        assembler.getBatch(tensorsB, batchNumB);
        const float *values = tensorsB.at("seasonality_inputs").getMemPtr<float>();
        secondRun.assign(values, values + 8);
        assembler.returnBuffers(tensorsB);
    }

    EXPECT_EQ(batchNumA, 0);
    EXPECT_EQ(batchNumB, 0);
    EXPECT_EQ(firstRun, secondRun);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedBatchAssemblerTest, RejectsReturnedTensorMapMissingName) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_returned_tensor");
    LocalNamedExampleLayout layout = testLayout();
    {
        LocalNamedExampleDatasetWriter writer(datasetPath, layout, 8);
        writeExample(writer, ExampleType::TRAIN, 0.0f);
        writeExample(writer, ExampleType::TRAIN, 20.0f);
        writer.close();
    }

    LocalNamedBatchAssembler assembler(openShards(datasetPath), ExampleType::TRAIN, layout, 2, 2, false);
    map<string, Tensor> tensors;
    uint64_t batchNum = 0;
    assembler.getBatch(tensors, batchNum);
    tensors.erase("daily_weight");
    EXPECT_THROW(assembler.returnBuffers(tensors), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}
