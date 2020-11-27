#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "omp.h"

using std::mutex;
using std::pair;
using std::set;
using std::string;
using std::unordered_set;
using std::vector;

using namespace boost::filesystem;

class TestDataProcessor : public DataProcessor {
    virtual uint64_t outputTensorSizeInBytes() { return 100; }

    virtual DataElement operator()(DataElement &input) { return input; }
};

void verifyImages(Shard &shard) {
    uint8_t buffer[224 * 224 * 3];
    string label;
    string filename;
    for (int exampleTypeInt = (int)ExampleType::TRAIN; exampleTypeInt <= (int)ExampleType::TEST; ++exampleTypeInt) {
        ExampleType exampleType = (ExampleType)exampleTypeInt;
        set<string> exampleNames;
        for (uint32_t i = 0; i < shard.getNumExamples(exampleType); ++i) {
            shard.loadExample(buffer, label, filename, exampleType, i);
            string qualifiedExampleName = label + '/' + filename;
            ASSERT_EQ(exampleNames.count(qualifiedExampleName), 0u);
            exampleNames.insert(qualifiedExampleName);

            uint8_t pixel[3];
            if (filename == "white.png") {
                pixel[0] = 255;
                pixel[1] = 255;
                pixel[2] = 255;
            } else if (filename == "black.png") {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
            } else if (filename == "red.png") {
                pixel[0] = 255;
                pixel[1] = 0;
                pixel[2] = 0;
            } else if (filename == "green.png") {
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
            } else if (filename == "blue.png") {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            } else {
                ASSERT_TRUE(false);
            }

            for (uint32_t p = 0; p < 224 * 224; ++p) {
                ASSERT_EQ(buffer[3 * p], pixel[0]);
                ASSERT_EQ(buffer[3 * p + 1], pixel[1]);
                ASSERT_EQ(buffer[3 * p + 2], pixel[2]);
            }
        }
    }
}

TEST(SharedRawDatasetCreator, evaluatesDataset) {
    string baseFilename = "testDataset";
    string testDatasetDir("test/DeepLearning/DataSet");

    path tempDirectoryPath = temp_directory_path();
    tempDirectoryPath /= "ThorFrameworkDatasetTest";
    remove_all(tempDirectoryPath);
    create_directory(tempDirectoryPath);

    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert(testDatasetDir);
    destDirectories.insert(tempDirectoryPath.native());

    std::vector<Shard> shards;
    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename);
    creator.createDataset(unique_ptr<ImageProcessor>(new ImageProcessor(0.05, 10, 224, 224, false)), shards);

    // load the dataset, ensure it contains the expected contents
    ASSERT_EQ(shards.size(), 1u);
    ASSERT_EQ(shards[0].getNumExamples(ExampleType::TRAIN), 3u);
    ASSERT_EQ(shards[0].getNumExamples(ExampleType::VALIDATE), 1u);
    ASSERT_EQ(shards[0].getNumExamples(ExampleType::TEST), 7u);

    if (rand() % 2 == 0)
        shards.clear();

    Shard reopenedShard;
    path shardPath;
    shardPath = tempDirectoryPath;
    shardPath /= (baseFilename + "_1_of_1");
    reopenedShard.openShard(shardPath.native());
    verifyImages(reopenedShard);

    remove_all(tempDirectoryPath);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
