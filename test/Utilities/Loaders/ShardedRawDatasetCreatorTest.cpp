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
using std::string;
using std::unordered_set;
using std::vector;

class TestDataProcessor : public DataProcessor {
    virtual uint64_t outputTensorSizeInBytes() { return 100; }

    virtual DataElement operator()(DataElement &input) { return input; }
};

TEST(SharedRawDatasetCreator, evaluatesDataset) {
    string baseFilename = "testDataset";
    unordered_set<string> sourceDirectories;
    unordered_set<string> destDirectories;

    sourceDirectories.insert("/home/andrew/ThorTestDataset/inputShard0");
    sourceDirectories.insert("/home/andrew/ThorTestDataset/inputShard1");
    destDirectories.insert("/home/andrew/ThorTestDataset/outputShard0");
    destDirectories.insert("/home/andrew/ThorTestDataset/outputShard1");

    ShardedRawDatasetCreator creator(sourceDirectories, destDirectories, baseFilename);
    creator.createDataset(unique_ptr<TestDataProcessor>(new TestDataProcessor()));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
