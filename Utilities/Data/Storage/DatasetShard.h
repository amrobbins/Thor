#pragma once

#include "DeepLearning/Api/Data/ExampleType.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class UringDirect;

struct DatasetShardReadRequest {
    uint64_t fileOffsetBytes;
    uint64_t numBytes;
    std::string label;
    std::string filename;
};

struct DatasetShardRecord {
    uint64_t fileOffsetBytes;
    std::string label;
    std::string filename;
};

class DatasetShard {
   public:
    DatasetShard();
    ~DatasetShard();

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
    DatasetShardReadRequest getExampleReadRequest(ExampleType exampleType, uint64_t exampleIndex);
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
    std::vector<DatasetShardRecord> trainExamples;
    std::vector<DatasetShardRecord> validateExamples;
    std::vector<DatasetShardRecord> testExamples;
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
    std::vector<DatasetShardRecord> &mutableRecordsFor(ExampleType exampleType);
    const std::vector<DatasetShardRecord> &recordsFor(ExampleType exampleType) const;
};
