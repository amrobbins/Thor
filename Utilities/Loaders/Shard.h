#pragma once

// TODO: Integrate with DALI
//       https://developer.nvidia.com/DALI
//       https://github.com/NVIDIA/DALI#compiling-dali-from-source-bare-metal
//       Will need a C++ api, currently that is experimental. Will need to talk to nVidia.

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

class UringDirect;

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "omp.h"

#include <unistd.h>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>


enum class ExampleType { TRAIN = 3, VALIDATE, TEST };

struct DataElement {
    ExampleType exampleType;
    std::string className;
    std::string fileName;

    uint64_t numDataBytes;
    std::shared_ptr<uint8_t[]> data;

    ThorImplementation::DataType dataType;
};

struct ShardExampleReadRequest {
    uint64_t fileOffsetBytes;
    uint64_t numBytes;
    std::string label;
    std::string filename;
};

struct ShardExampleRecord {
    uint64_t fileOffsetBytes;
    std::string label;
    std::string filename;
};

class Shard {
   public:
    Shard();
    ~Shard();

    void createShard(std::string filename,
                     uint64_t numTrainExamples,
                     uint64_t numValidateExamples,
                     uint64_t numTestExamples,
                     uint64_t exampleSizeInBytes,
                     ThorImplementation::DataType dataType,
                     uint64_t maxFilenameChars,
                     std::vector<std::string> &allClassesVector,
                     uint64_t maxClassNameChars);

    void openShard(std::string filename);
    void createCompactShard(std::string filename,
                            uint64_t numTrainExamples,
                            uint64_t numValidateExamples,
                            uint64_t numTestExamples,
                            uint64_t exampleSizeInBytes,
                            ThorImplementation::DataType dataType,
                            std::vector<std::string> &allClassesVector,
                            bool preallocate = false);
    bool isOpen();
    void writeExample(uint8_t *buffer, const std::string &label, const std::string &filename, ExampleType exampleType);
    void writeExamplesContiguous(uint8_t *buffer, uint64_t numExamples, ExampleType exampleType);
    ShardExampleReadRequest getExampleReadRequest(ExampleType exampleType, uint64_t exampleIndex);
    void loadExample(uint8_t *buffer, std::string &label, std::string &filename, ExampleType exampleType, uint64_t exampleIndex);
    void shrinkToFit();
    std::string getFilename();
    uint64_t getExampleSizeInBytes();
    ThorImplementation::DataType getDataType();
    uint64_t getNumExamples(ExampleType exampleType);
    const std::vector<std::string> &getAllClasses();

   private:
    std::string filename;
    uint64_t exampleSizeInBytes;
    ThorImplementation::DataType dataType;
    bool open;
    bool metadataFinalized;

    std::fstream shardFile;
    std::vector<ShardExampleRecord> trainExamples;
    std::vector<ShardExampleRecord> validateExamples;
    std::vector<ShardExampleRecord> testExamples;
    std::vector<std::string> allClasses;

    bool compactMetadata;
    uint64_t compactTrainOffsetBytes;
    uint64_t compactValidateOffsetBytes;
    uint64_t compactTestOffsetBytes;
    uint64_t compactTrainCount;
    uint64_t compactValidateCount;
    uint64_t compactTestCount;
    uint64_t compactTrainCapacity;
    uint64_t compactValidateCapacity;
    uint64_t compactTestCapacity;
    uint64_t compactTrainBytes;
    uint64_t compactValidateBytes;
    uint64_t compactTestBytes;
    uint64_t compactRecordStrideBytes;

    std::mutex mtx;
    std::mutex cachedReaderMtx;
    std::unique_ptr<UringDirect> cachedReader;

    void readExamplePayloadCached(uint8_t *buffer, uint64_t fileOffsetBytes);
    void writeHeader(uint64_t metadataOffsetBytes, uint64_t metadataBytes, uint32_t metadataLayout = 0);
    void writeMetadata();
    void readMetadata(uint32_t metadataLayout,
                      uint64_t metadataOffsetBytes,
                      uint64_t metadataBytes,
                      uint64_t trainCount,
                      uint64_t validateCount,
                      uint64_t testCount,
                      uint64_t classCount);
    uint64_t compactOffsetFor(ExampleType exampleType) const;
    uint64_t compactCountFor(ExampleType exampleType) const;
    uint64_t compactBytesFor(ExampleType exampleType) const;
    std::vector<ShardExampleRecord> &mutableRecordsFor(ExampleType exampleType);
    const std::vector<ShardExampleRecord> &recordsFor(ExampleType exampleType) const;
};
