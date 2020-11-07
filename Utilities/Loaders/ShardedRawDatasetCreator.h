#pragma once

// TODO: Integrate with DALI
//       https://developer.nvidia.com/DALI
//       https://github.com/NVIDIA/DALI#compiling-dali-from-source-bare-metal
//       Will need a C++ api, currently that is experimental. Will need to talk to nVidia.

#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

#include <boost/filesystem.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include "omp.h"

#include <unistd.h>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::mutex;
using std::string;
using std::thread;

enum class ExampleType { TRAIN = 3, VALIDATE, TEST };

struct DataElement {
    ExampleType exampleType;
    string className;
    boost::filesystem::path destShard;

    uint64_t numDataBytes;
    std::shared_ptr<char> data;
};

class DataProcessor : public WorkQueueExecutorBase<DataElement, DataElement> {
   public:
    /**
     * raw data contains a whole file loaded to memory as is.
     * processData is the data that will be loaded byte for byte into the tensor.
     * returns true on success.
     */
    virtual uint64_t outputTensorSizeInBytes() = 0;
};

/**
 *  Processes the source data set and creates a raw data set that can be read directly into the input tensors byte for byte.
 *  Mutliple shards can be created. The intended use case is to place one shard per disk to maximize read bandwidth.
 */
class ShardedRawDatasetCreator {
   public:
    ShardedRawDatasetCreator(std::unordered_set<string> sourceDirectories,
                             std::unordered_set<string> destDirectories,
                             string baseDatasetFileName);

    bool createDataset(std::unique_ptr<DataProcessor> &&dataProcessor);

   private:
    std::unordered_set<string> sourceDirectories;
    std::unordered_set<string> destDirectories;
    string baseDatasetFileName;

    uint64_t numTrainExamples;
    uint64_t numValidateExamples;
    uint64_t numTestExamples;

    std::unordered_set<string> classes;
    // sourceDirectory -> (class -> [examplePath, examplePath, ...])
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> trainExamplesPerClass;
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> validateExamplesPerClass;
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> testExamplesPerClass;

    uint32_t numOutputShards;
    std::vector<boost::filesystem::path> destShardFiles;

    mutex mtx;

    uint64_t getNumExamples();
    void loadExamples(WorkQueueUnordered<DataElement, DataElement> &workQueue);
    void writeDataToShard(WorkQueueUnordered<DataElement, DataElement> *workQueue);
};
