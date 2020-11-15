#pragma once

// TODO: Integrate with DALI
//       https://developer.nvidia.com/DALI
//       https://github.com/NVIDIA/DALI#compiling-dali-from-source-bare-metal
//       Will need a C++ api, currently that is experimental. Will need to talk to nVidia.

#include "Utilities/Common/Stream.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

#include <boost/filesystem.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
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

typedef boost::interprocess::allocator<uint8_t, boost::interprocess::managed_mapped_file::segment_manager> file_vector_allocator_t;
typedef boost::interprocess::vector<uint8_t, file_vector_allocator_t> file_vector_t;

typedef boost::interprocess::allocator<boost::interprocess::string, boost::interprocess::managed_mapped_file::segment_manager>
    file_string_vector_allocator_t;
typedef boost::interprocess::vector<boost::interprocess::string, file_string_vector_allocator_t> file_string_vector_t;

enum class ExampleType { TRAIN = 3, VALIDATE, TEST };

struct DataElement {
    ExampleType exampleType;
    string className;
    string fileName;

    uint64_t numDataBytes;
    std::shared_ptr<uint8_t> data;
};

struct ShardMetadata {
    uint64_t exampleSizeInBytes;
};

class Shard {
   public:
    Shard() {}

    void createShard(
        string filename, uint64_t numTrainExamples, uint64_t numValidateExamples, uint64_t numTestExamples, uint64_t exampleSizeInBytes) {
        this->filename = filename;
        this->exampleSizeInBytes = exampleSizeInBytes;
        uint64_t shardSizeInBytes = (numTrainExamples + numValidateExamples + numTestExamples) * (exampleSizeInBytes + 300) + 1000000;
        mappedFile = boost::interprocess::managed_mapped_file(boost::interprocess::create_only, filename.c_str(), shardSizeInBytes);

        shardMetadata = mappedFile.construct<ShardMetadata>("shardMetadata")();
        shardMetadata->exampleSizeInBytes = exampleSizeInBytes;

        file_vector_allocator_t trainDataAllocator(mappedFile.get_segment_manager());
        trainData = mappedFile.construct<file_vector_t>("train")(trainDataAllocator);
        trainData->reserve(numTrainExamples * exampleSizeInBytes);

        file_vector_allocator_t validateDataAllocator(mappedFile.get_segment_manager());
        validateData = mappedFile.construct<file_vector_t>("validate")(validateDataAllocator);
        validateData->reserve(numValidateExamples * exampleSizeInBytes);

        file_vector_allocator_t testDataAllocator(mappedFile.get_segment_manager());
        testData = mappedFile.construct<file_vector_t>("test")(testDataAllocator);
        testData->reserve(numTestExamples * exampleSizeInBytes);

        file_string_vector_allocator_t trainLabelsAllocator(mappedFile.get_segment_manager());
        trainLabels = mappedFile.construct<file_string_vector_t>("trainLabels")(trainLabelsAllocator);
        trainLabels->reserve(numTrainExamples);

        file_string_vector_allocator_t validateLabelsAllocator(mappedFile.get_segment_manager());
        validateLabels = mappedFile.construct<file_string_vector_t>("validateLabels")(validateLabelsAllocator);
        validateLabels->reserve(numValidateExamples);

        file_string_vector_allocator_t testLabelsAllocator(mappedFile.get_segment_manager());
        testLabels = mappedFile.construct<file_string_vector_t>("testLabels")(testLabelsAllocator);
        testLabels->reserve(numTestExamples);
    }

    void openShard(string filename) {
        mappedFile = boost::interprocess::managed_mapped_file(boost::interprocess::open_only, filename.c_str());
        trainData = mappedFile.find<file_vector_t>("train").first;
        validateData = mappedFile.find<file_vector_t>("validate").first;
        testData = mappedFile.find<file_vector_t>("test").first;
        trainLabels = mappedFile.find<file_string_vector_t>("trainLabels").first;
        validateLabels = mappedFile.find<file_string_vector_t>("validateLabels").first;
        testLabels = mappedFile.find<file_string_vector_t>("testLabels").first;
        shardMetadata = mappedFile.find<ShardMetadata>("shardMetadata").first;
        exampleSizeInBytes = shardMetadata->exampleSizeInBytes;

        assert(trainData->size() % exampleSizeInBytes == 0);
        assert(testData->size() % exampleSizeInBytes == 0);
        assert(validateData->size() % exampleSizeInBytes == 0);
        assert(trainLabels->size() == trainData->size() / exampleSizeInBytes);
        assert(validateLabels->size() == validateData->size() / exampleSizeInBytes);
        assert(testLabels->size() == testData->size() / exampleSizeInBytes);
    }

    void writeExample(uint8_t *buffer, const string &label, ExampleType exampleType) {
        assert(buffer != nullptr);

        if (exampleType == ExampleType::TRAIN) {
            assert(trainData->capacity() > trainData->size() + exampleSizeInBytes);
            trainData->insert(trainData->end(), buffer, buffer + exampleSizeInBytes);
            assert(trainLabels->capacity() > trainLabels->size());
            boost::interprocess::string diskLabel(label.c_str());
            trainLabels->push_back(diskLabel);
        } else if (exampleType == ExampleType::VALIDATE) {
            assert(validateData->capacity() > validateData->size() + exampleSizeInBytes);
            validateData->insert(validateData->end(), buffer, buffer + exampleSizeInBytes);
            assert(validateLabels->capacity() > validateLabels->size());
            boost::interprocess::string diskLabel(label.c_str());
            validateLabels->push_back(diskLabel);
        } else if (exampleType == ExampleType::TEST) {
            assert(testData->capacity() > testData->size() + exampleSizeInBytes);
            testData->insert(testData->end(), buffer, buffer + exampleSizeInBytes);
            assert(testLabels->capacity() > testLabels->size());
            boost::interprocess::string diskLabel(label.c_str());
            testLabels->push_back(diskLabel);
        } else {
            assert(false);
        }
    }

    void loadExample(uint8_t *buffer, string &label, ExampleType exampleType, uint64_t exampleIndex) {
        assert(buffer != nullptr);

        uint8_t *data;
        if (exampleType == ExampleType::TRAIN) {
            uint64_t numExamples = trainData->size() / exampleSizeInBytes;
            assert(exampleIndex < numExamples);
            data = trainData->data();
            // fixme: put string loading on the stream
            label = (*trainLabels)[exampleIndex].c_str();
        } else if (exampleType == ExampleType::VALIDATE) {
            uint64_t numExamples = validateData->size() / exampleSizeInBytes;
            assert(exampleIndex < numExamples);
            data = validateData->data();
            label = (*validateLabels)[exampleIndex].c_str();
        } else if (exampleType == ExampleType::TEST) {
            uint64_t numExamples = testData->size() / exampleSizeInBytes;
            assert(exampleIndex < numExamples);
            data = testData->data();
            label = (*testLabels)[exampleIndex].c_str();
        } else {
            assert(false);
        }

        uint8_t *exampleStart = data + (exampleIndex * exampleSizeInBytes);
        memcpy(buffer, exampleStart, exampleSizeInBytes);
    }

    void loadExampleAsync(uint8_t *buffer, string &label, ExampleType exampleType, uint64_t exampleIndex, Stream stream) {
        assert(buffer != nullptr);
        cudaError_t cudaStatus;

        uint8_t *data;
        if (exampleType == ExampleType::TRAIN) {
            uint64_t numExamples = trainData->size() / exampleSizeInBytes;
            assert(exampleIndex < numExamples);
            data = trainData->data();
            label = (*trainLabels)[exampleIndex].c_str();
        } else if (exampleType == ExampleType::VALIDATE) {
            uint64_t numExamples = validateData->size() / exampleSizeInBytes;
            assert(exampleIndex < numExamples);
            data = validateData->data();
            label = (*validateLabels)[exampleIndex].c_str();
        } else if (exampleType == ExampleType::TEST) {
            uint64_t numExamples = testData->size() / exampleSizeInBytes;
            assert(exampleIndex < numExamples);
            data = testData->data();
            label = (*testLabels)[exampleIndex].c_str();
        } else {
            assert(false);
        }

        uint8_t *exampleStart = data + (exampleIndex * exampleSizeInBytes);
        cudaStatus = cudaMemcpyAsync(buffer, exampleStart, exampleSizeInBytes, cudaMemcpyHostToHost, stream);
        assert(cudaStatus == cudaSuccess);
    }

    void shrinkToFit() {
        trainData->shrink_to_fit();
        validateData->shrink_to_fit();
        testData->shrink_to_fit();
        trainLabels->shrink_to_fit();
        validateLabels->shrink_to_fit();
        testLabels->shrink_to_fit();
        bool status = boost::interprocess::managed_mapped_file::shrink_to_fit(filename.c_str());
        assert(status == true);
    }

   private:
    string filename;
    uint64_t exampleSizeInBytes;
    boost::interprocess::managed_mapped_file mappedFile;

    file_vector_t *trainData;
    file_vector_t *validateData;
    file_vector_t *testData;
    file_string_vector_t *trainLabels;
    file_string_vector_t *validateLabels;
    file_string_vector_t *testLabels;
    ShardMetadata *shardMetadata;
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
    uint64_t outputTensorSizeInBytes;

    std::unordered_set<string> classes;
    // sourceDirectory -> (class -> [examplePath, examplePath, ...])
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> trainExamplesPerClass;
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> validateExamplesPerClass;
    std::unordered_map<string, std::unordered_map<string, std::vector<boost::filesystem::path>>> testExamplesPerClass;

    uint32_t numOutputShards;
    std::vector<boost::filesystem::path> destShardFiles;
    std::vector<Shard> shards;

    mutex mtx;

    void getNumExamples(uint64_t &numTrainExamples, uint64_t &numValidateExamples, uint64_t &numTestExamples);
    void loadExamples(WorkQueueUnordered<DataElement, DataElement> &workQueue);
    void writeDataToShard(WorkQueueUnordered<DataElement, DataElement> *workQueue);

    atomic<uint64_t> destShardTrain;
    atomic<uint64_t> destShardValidate;
    atomic<uint64_t> destShardTest;
};
