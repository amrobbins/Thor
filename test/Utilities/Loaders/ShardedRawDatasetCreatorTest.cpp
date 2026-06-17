#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/BatchAssembler.h"
#include "Utilities/Loaders/NoOpDataProcessor.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::array;
using std::set;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using namespace std::filesystem;

namespace {
constexpr uint64_t RAW_EXAMPLE_BYTES = 4;
constexpr uint64_t NUM_CLASSES = 4;
constexpr uint32_t SHARD_FORMAT_VERSION = 1;
constexpr uint32_t SHARD_METADATA_LAYOUT_COMPACT = 1;
constexpr uint32_t SHARD_METADATA_LAYOUT_BYTE_CORPUS = 2;
constexpr uint32_t SHARD_HEADER_BYTES = 88;

struct RawExampleSpec {
    const char *split;
    const char *label;
    const char *filename;
    array<uint8_t, RAW_EXAMPLE_BYTES> bytes;
};

const vector<RawExampleSpec> &rawExampleSpecs() {
    static const vector<RawExampleSpec> specs = {
        {"train", "black", "train_black_0.bin", {0, 1, 2, 3}},
        {"train", "blue", "train_blue_0.bin", {10, 11, 12, 13}},
        {"train", "green", "train_green_0.bin", {20, 21, 22, 23}},
        {"train", "white", "train_white_0.bin", {30, 31, 32, 33}},
        {"validate", "blue", "validate_blue_0.bin", {40, 41, 42, 43}},
        {"validate", "green", "validate_green_0.bin", {50, 51, 52, 53}},
        {"test", "black", "test_black_0.bin", {60, 61, 62, 63}},
        {"test", "black", "test_black_1.bin", {70, 71, 72, 73}},
        {"test", "blue", "test_blue_0.bin", {80, 81, 82, 83}},
        {"test", "green", "test_green_0.bin", {90, 91, 92, 93}},
        {"test", "green", "test_green_1.bin", {100, 101, 102, 103}},
        {"test", "white", "test_white_0.bin", {110, 111, 112, 113}},
        {"test", "white", "test_white_1.bin", {120, 121, 122, 123}},
    };
    return specs;
}


void writeExact(std::ostream &stream, const void *data, uint64_t numBytes) {
    stream.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(numBytes));
    ASSERT_TRUE(stream.good());
}

void writeUint32(std::ostream &stream, uint32_t value) { writeExact(stream, &value, sizeof(value)); }

void writeUint64(std::ostream &stream, uint64_t value) { writeExact(stream, &value, sizeof(value)); }

void writeString(std::ostream &stream, const std::string &value) {
    writeUint64(stream, value.size());
    if (!value.empty()) {
        writeExact(stream, value.data(), value.size());
    }
}

void writeCompactHeader(std::fstream &stream,
                        uint64_t trainCount,
                        uint64_t validateCount,
                        uint64_t testCount,
                        uint64_t metadataOffset,
                        uint64_t metadataBytes) {
    static const array<char, 16> magic = {'T', 'H', 'O', 'R', '_', 'R', 'A', 'W', '_', 'S', 'H', 'A', 'R', 'D', '\0', '\0'};
    stream.seekp(0, std::ios::beg);
    writeExact(stream, magic.data(), magic.size());
    writeUint32(stream, SHARD_FORMAT_VERSION);
    writeUint32(stream, SHARD_HEADER_BYTES);
    writeUint64(stream, RAW_EXAMPLE_BYTES);
    writeUint32(stream, static_cast<uint32_t>(ThorImplementation::DataType::UINT8));
    writeUint32(stream, SHARD_METADATA_LAYOUT_COMPACT);
    writeUint64(stream, trainCount);
    writeUint64(stream, validateCount);
    writeUint64(stream, testCount);
    writeUint64(stream, 1);
    writeUint64(stream, metadataOffset);
    writeUint64(stream, metadataBytes);
    ASSERT_EQ(stream.tellp(), static_cast<std::streampos>(SHARD_HEADER_BYTES));
}

void writeByteCorpusHeader(std::fstream &stream,
                           uint64_t recordBytes,
                           uint64_t trainCount,
                           uint64_t validateCount,
                           uint64_t testCount,
                           uint64_t metadataOffset,
                           uint64_t metadataBytes) {
    static const array<char, 16> magic = {'T', 'H', 'O', 'R', '_', 'R', 'A', 'W', '_', 'S', 'H', 'A', 'R', 'D', '\0', '\0'};
    stream.seekp(0, std::ios::beg);
    writeExact(stream, magic.data(), magic.size());
    writeUint32(stream, SHARD_FORMAT_VERSION);
    writeUint32(stream, SHARD_HEADER_BYTES);
    writeUint64(stream, recordBytes);
    writeUint32(stream, static_cast<uint32_t>(ThorImplementation::DataType::UINT8));
    writeUint32(stream, SHARD_METADATA_LAYOUT_BYTE_CORPUS);
    writeUint64(stream, trainCount);
    writeUint64(stream, validateCount);
    writeUint64(stream, testCount);
    writeUint64(stream, 1);
    writeUint64(stream, metadataOffset);
    writeUint64(stream, metadataBytes);
    ASSERT_EQ(stream.tellp(), static_cast<std::streampos>(SHARD_HEADER_BYTES));
}

void writeCompactShard(const path &shardPath) {
    const vector<array<uint8_t, RAW_EXAMPLE_BYTES>> train = {{{1, 2, 3, 4}}, {{5, 6, 7, 8}}};
    const vector<array<uint8_t, RAW_EXAMPLE_BYTES>> validate = {{{9, 10, 11, 12}}};
    const vector<array<uint8_t, RAW_EXAMPLE_BYTES>> test = {{{13, 14, 15, 16}}, {{17, 18, 19, 20}}};

    std::fstream file(shardPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file.write(std::string(SHARD_HEADER_BYTES, '\0').data(), SHARD_HEADER_BYTES);

    const uint64_t trainOffset = static_cast<uint64_t>(file.tellp());
    for (const auto &record : train) writeExact(file, record.data(), record.size());
    const uint64_t validateOffset = static_cast<uint64_t>(file.tellp());
    for (const auto &record : validate) writeExact(file, record.data(), record.size());
    const uint64_t testOffset = static_cast<uint64_t>(file.tellp());
    for (const auto &record : test) writeExact(file, record.data(), record.size());

    const uint64_t metadataOffset = static_cast<uint64_t>(file.tellp());
    writeString(file, "sequence");
    writeUint64(file, trainOffset);
    writeUint64(file, train.size());
    writeUint64(file, validateOffset);
    writeUint64(file, validate.size());
    writeUint64(file, testOffset);
    writeUint64(file, test.size());
    const uint64_t metadataEnd = static_cast<uint64_t>(file.tellp());
    writeCompactHeader(file, train.size(), validate.size(), test.size(), metadataOffset, metadataEnd - metadataOffset);
}

void writeByteCorpusShard(const path &shardPath) {
    constexpr uint64_t CONTEXT_BYTES = 4;
    constexpr uint64_t RECORD_BYTES = CONTEXT_BYTES + 1;
    constexpr uint64_t STRIDE_BYTES = 2;
    const vector<uint8_t> train = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    const vector<uint8_t> validate = {30, 31, 32, 33, 34};
    const vector<uint8_t> test = {40, 41, 42, 43, 44, 45, 46};
    const uint64_t trainCount = 1 + (train.size() - CONTEXT_BYTES - 1) / STRIDE_BYTES;
    const uint64_t validateCount = 1 + (validate.size() - CONTEXT_BYTES - 1) / STRIDE_BYTES;
    const uint64_t testCount = 1 + (test.size() - CONTEXT_BYTES - 1) / STRIDE_BYTES;

    std::fstream file(shardPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file.write(std::string(SHARD_HEADER_BYTES, '\0').data(), SHARD_HEADER_BYTES);

    const uint64_t trainOffset = static_cast<uint64_t>(file.tellp());
    writeExact(file, train.data(), train.size());
    const uint64_t validateOffset = static_cast<uint64_t>(file.tellp());
    writeExact(file, validate.data(), validate.size());
    const uint64_t testOffset = static_cast<uint64_t>(file.tellp());
    writeExact(file, test.data(), test.size());

    const uint64_t metadataOffset = static_cast<uint64_t>(file.tellp());
    writeString(file, "sequence");
    writeUint64(file, STRIDE_BYTES);
    writeUint64(file, trainOffset);
    writeUint64(file, trainCount);
    writeUint64(file, train.size());
    writeUint64(file, validateOffset);
    writeUint64(file, validateCount);
    writeUint64(file, validate.size());
    writeUint64(file, testOffset);
    writeUint64(file, testCount);
    writeUint64(file, test.size());
    const uint64_t metadataEnd = static_cast<uint64_t>(file.tellp());
    writeByteCorpusHeader(file, RECORD_BYTES, trainCount, validateCount, testCount, metadataOffset, metadataEnd - metadataOffset);
}

unordered_map<string, array<uint8_t, RAW_EXAMPLE_BYTES>> expectedBytesByFilename() {
    unordered_map<string, array<uint8_t, RAW_EXAMPLE_BYTES>> expected;
    for (const RawExampleSpec &spec : rawExampleSpecs()) {
        expected.emplace(spec.filename, spec.bytes);
    }
    return expected;
}

void writeRawDataset(const path &datasetDir) {
    for (const char *split : {"train", "validate", "test"}) {
        for (const char *label : {"black", "blue", "green", "white"}) {
            create_directories(datasetDir / split / label);
        }
    }

    for (const RawExampleSpec &spec : rawExampleSpecs()) {
        path filename = datasetDir / spec.split / spec.label / spec.filename;
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<const char *>(spec.bytes.data()), spec.bytes.size());
        ASSERT_TRUE(file.good());
    }
}

void verifyRawExamples(Shard &shard) {
    const unordered_map<string, array<uint8_t, RAW_EXAMPLE_BYTES>> expectedBytes = expectedBytesByFilename();
    uint8_t buffer[RAW_EXAMPLE_BYTES];
    string label;
    string filename;

    for (int exampleTypeInt = static_cast<int>(ExampleType::TRAIN); exampleTypeInt <= static_cast<int>(ExampleType::TEST);
         ++exampleTypeInt) {
        ExampleType exampleType = static_cast<ExampleType>(exampleTypeInt);
        set<string> exampleNames;
        for (uint32_t i = 0; i < shard.getNumExamples(exampleType); ++i) {
            shard.loadExample(buffer, label, filename, exampleType, i);
            string qualifiedExampleName = label + '/' + filename;
            ASSERT_EQ(exampleNames.count(qualifiedExampleName), 0u);
            exampleNames.insert(qualifiedExampleName);

            auto expectedIt = expectedBytes.find(filename);
            ASSERT_NE(expectedIt, expectedBytes.end());
            for (uint64_t byte = 0; byte < RAW_EXAMPLE_BYTES; ++byte) {
                ASSERT_EQ(buffer[byte], expectedIt->second[byte]);
            }
        }
    }
}

bool isExpectedTestPayload(const uint8_t *example) {
    for (const RawExampleSpec &spec : rawExampleSpecs()) {
        if (string(spec.split) != "test")
            continue;
        bool matches = true;
        for (uint64_t byte = 0; byte < RAW_EXAMPLE_BYTES; ++byte) {
            matches &= example[byte] == spec.bytes[byte];
        }
        if (matches)
            return true;
    }
    return false;
}

void verifyBatch(uint64_t batchSize,
                 const vector<ThorImplementation::Tensor> &batchTensors,
                 const vector<ThorImplementation::Tensor> &labelTensors) {
    ASSERT_GT(batchSize, 0u);
    ASSERT_GT(batchTensors.size(), 0u);
    ASSERT_EQ(batchTensors.size(), labelTensors.size());

    for (uint64_t i = 0; i < batchTensors.size(); ++i) {
        ASSERT_EQ(batchTensors[i].getDescriptor().getTotalNumElements(), batchSize * RAW_EXAMPLE_BYTES);
        ASSERT_EQ(labelTensors[i].getDescriptor().getTotalNumElements(), batchSize * NUM_CLASSES);

        const uint8_t *examples = static_cast<const uint8_t *>(batchTensors[i].getMemPtr());
        const float *labels = static_cast<const float *>(labelTensors[i].getMemPtr());
        for (uint64_t batchElement = 0; batchElement < batchSize; ++batchElement) {
            ASSERT_TRUE(isExpectedTestPayload(examples + batchElement * RAW_EXAMPLE_BYTES));

            uint64_t oneHotCount = 0;
            for (uint64_t k = 0; k < NUM_CLASSES; ++k) {
                float labelValue = labels[batchElement * NUM_CLASSES + k];
                ASSERT_TRUE(labelValue == 0.0f || labelValue == 1.0f);
                if (labelValue == 1.0f)
                    ++oneHotCount;
            }
            ASSERT_EQ(oneHotCount, 1u);
        }
    }
}

void verifyBatchAssembler(const vector<shared_ptr<Shard>> &shards) {
    srand(0);

    uint64_t batchSize = 3;

    BatchAssembler batchAssembler(
        shards,
        ExampleType::TEST,
        ThorImplementation::TensorDescriptor(ThorImplementation::DataType::UINT8, {RAW_EXAMPLE_BYTES}),
        ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, {NUM_CLASSES}),
        batchSize);

    ASSERT_EQ(batchAssembler.getNumExamples(), 7u);
    ASSERT_EQ(batchAssembler.getNumBatchesPerEpoch(), 3u);

    ThorImplementation::Tensor batchTensor;
    ThorImplementation::Tensor labelTensor;
    vector<ThorImplementation::Tensor> batchTensors;
    vector<ThorImplementation::Tensor> labelTensors;
    uint64_t batchNum;

    for (uint64_t i = 0; i < batchAssembler.getNumBatchesPerEpoch(); ++i) {
        batchAssembler.getBatch(batchTensor, labelTensor, batchNum);
        batchTensors.push_back(batchTensor);
        labelTensors.push_back(labelTensor);
    }

    verifyBatch(batchSize, batchTensors, labelTensors);

    // Next line just to cover that function; it does not return all buffers.
    batchAssembler.returnBuffer(batchTensor, labelTensor);
}
}  // namespace

TEST(ShardedRawDatasetCreator, evaluatesDataset) {
    string baseFilename = "testDataset";

    path tempDirectoryPath = temp_directory_path() / "ThorFrameworkDatasetTest";
    remove_all(tempDirectoryPath);
    create_directories(tempDirectoryPath);

    path sourceDatasetPath = tempDirectoryPath / "source";
    path destDatasetPath = tempDirectoryPath / "dest";
    create_directories(destDatasetPath);
    writeRawDataset(sourceDatasetPath);

    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert(sourceDatasetPath.native());
    destDirectories.insert(destDatasetPath.native());

    vector<shared_ptr<Shard>> shards;
    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename);
    creator.createDataset(
        std::unique_ptr<DataProcessor>(new NoOpDataProcessor(RAW_EXAMPLE_BYTES, ThorImplementation::DataType::UINT8)), shards);

    ASSERT_EQ(shards.size(), 1u);
    EXPECT_EQ(shards[0]->getNumExamples(ExampleType::TRAIN), 4u);
    EXPECT_EQ(shards[0]->getNumExamples(ExampleType::VALIDATE), 2u);
    EXPECT_EQ(shards[0]->getNumExamples(ExampleType::TEST), 7u);

    verifyBatchAssembler(shards);

    if (rand() % 2 == 0)
        shards.clear();

    Shard reopenedShard;
    path shardPath = destDatasetPath / (baseFilename + "_1_of_1.shard");
    reopenedShard.openShard(shardPath.native());
    verifyRawExamples(reopenedShard);

    remove_all(tempDirectoryPath);
}


TEST(Shard, opensCompactContiguousShardWithImplicitO1Offsets) {
    path tempDirectoryPath = temp_directory_path() / "ThorFrameworkCompactShardTest";
    remove_all(tempDirectoryPath);
    create_directories(tempDirectoryPath);
    path shardPath = tempDirectoryPath / "compact.shard";
    writeCompactShard(shardPath);

    Shard shard;
    shard.openShard(shardPath.native());
    ASSERT_EQ(shard.getNumExamples(ExampleType::TRAIN), 2u);
    ASSERT_EQ(shard.getNumExamples(ExampleType::VALIDATE), 1u);
    ASSERT_EQ(shard.getNumExamples(ExampleType::TEST), 2u);

    uint8_t buffer[RAW_EXAMPLE_BYTES] = {};
    string label;
    string filename;
    shard.loadExample(buffer, label, filename, ExampleType::TRAIN, 1);
    EXPECT_EQ(label, "sequence");
    EXPECT_TRUE(filename.empty());
    EXPECT_EQ(buffer[0], 5u);
    EXPECT_EQ(buffer[3], 8u);
    shard.loadExample(buffer, label, filename, ExampleType::TEST, 0);
    EXPECT_EQ(buffer[0], 13u);
    EXPECT_EQ(buffer[3], 16u);

    remove_all(tempDirectoryPath);
}

TEST(Shard, opensByteCorpusShardWithStrideO1Offsets) {
    path tempDirectoryPath = temp_directory_path() / "ThorFrameworkByteCorpusShardTest";
    remove_all(tempDirectoryPath);
    create_directories(tempDirectoryPath);
    path shardPath = tempDirectoryPath / "byte_corpus.shard";
    writeByteCorpusShard(shardPath);

    Shard shard;
    shard.openShard(shardPath.native());
    ASSERT_EQ(shard.getNumExamples(ExampleType::TRAIN), 3u);
    ASSERT_EQ(shard.getNumExamples(ExampleType::VALIDATE), 1u);
    ASSERT_EQ(shard.getNumExamples(ExampleType::TEST), 2u);

    uint8_t buffer[5] = {};
    string label;
    string filename;
    shard.loadExample(buffer, label, filename, ExampleType::TRAIN, 0);
    EXPECT_EQ(label, "sequence");
    EXPECT_TRUE(filename.empty());
    EXPECT_EQ(buffer[0], 10u);
    EXPECT_EQ(buffer[4], 14u);

    shard.loadExample(buffer, label, filename, ExampleType::TRAIN, 2);
    EXPECT_EQ(buffer[0], 14u);
    EXPECT_EQ(buffer[4], 18u);

    shard.loadExample(buffer, label, filename, ExampleType::TEST, 1);
    EXPECT_EQ(buffer[0], 42u);
    EXPECT_EQ(buffer[4], 46u);

    remove_all(tempDirectoryPath);
}

TEST(BatchAssembler, derivesShiftedUint8LabelsFromByteCorpusRecords) {
    srand(0);
    path tempDirectoryPath = temp_directory_path() / "ThorFrameworkByteCorpusBatchAssemblerTest";
    remove_all(tempDirectoryPath);
    create_directories(tempDirectoryPath);
    path shardPath = tempDirectoryPath / "byte_corpus.shard";
    writeByteCorpusShard(shardPath);

    {
        auto shard = std::make_shared<Shard>();
        shard->openShard(shardPath.native());
        vector<shared_ptr<Shard>> shards = {shard};

        BatchAssembler batchAssembler(
            shards,
            ExampleType::TRAIN,
            ThorImplementation::TensorDescriptor(ThorImplementation::DataType::UINT8, {4}),
            ThorImplementation::TensorDescriptor(ThorImplementation::DataType::UINT8, {4}),
            2);

        ThorImplementation::Tensor batchTensor;
        ThorImplementation::Tensor labelTensor;
        uint64_t batchNum;
        batchAssembler.getBatch(batchTensor, labelTensor, batchNum);

        const uint8_t *examples = static_cast<const uint8_t *>(batchTensor.getMemPtr());
        const uint8_t *labels = static_cast<const uint8_t *>(labelTensor.getMemPtr());
        for (uint64_t batchElement = 0; batchElement < 2; ++batchElement) {
            const uint8_t *example = examples + batchElement * 4;
            const uint8_t *label = labels + batchElement * 4;
            for (uint64_t i = 0; i < 4; ++i) {
                EXPECT_EQ(label[i], static_cast<uint8_t>(example[i] + 1));
            }
        }
        batchAssembler.returnBuffer(batchTensor, labelTensor);
    }
    remove_all(tempDirectoryPath);
}
