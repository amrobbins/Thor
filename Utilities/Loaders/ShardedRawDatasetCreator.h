#pragma once

// TODO: Integrate with DALI
//       https://developer.nvidia.com/DALI
//       https://github.com/NVIDIA/DALI#compiling-dali-from-source-bare-metal
//       Will need a C++ api, currently that is experimental. Will need to talk to nVidia.

#include "Utilities/Common/Stream.h"
#include "Utilities/Loaders/MemMappedFileTypes.h"
#include "Utilities/Loaders/Shard.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

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

    bool createDataset(std::unique_ptr<DataProcessor> &&dataProcessor, std::vector<std::shared_ptr<Shard>> &shards);

   private:
    std::unordered_set<string> sourceDirectories;
    std::unordered_set<string> destDirectories;
    string baseDatasetFileName;

    uint64_t numTrainExamples;
    uint64_t numValidateExamples;
    uint64_t numTestExamples;
    uint64_t outputTensorSizeInBytes;

    std::unordered_set<string> classes;
    // sourceDirectory -> (class -> [examplePath, examplePath, ...])
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> trainExamplesPerClass;
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> validateExamplesPerClass;
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> testExamplesPerClass;

    uint32_t numOutputShards;
    std::vector<boost::filesystem::path> destShardFiles;

    mutex mtx;

    void getNumExamples(uint64_t &numTrainExamples,
                        uint64_t &numValidateExamples,
                        uint64_t &numTestExamples,
                        uint64_t &numFilenameChars,
                        std::set<string> &allClasses,
                        uint64_t &numClassNameChars);
    void loadExamples(WorkQueueUnordered<DataElement, DataElement> &workQueue);
    void writeDataToShard(WorkQueueUnordered<DataElement, DataElement> *workQueue, std::vector<std::shared_ptr<Shard>> *shards);

    atomic<uint64_t> destShardTrain;
    atomic<uint64_t> destShardValidate;
    atomic<uint64_t> destShardTest;
};
