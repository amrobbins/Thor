#pragma once

// TODO: Integrate with DALI
//       https://developer.nvidia.com/DALI
//       https://github.com/NVIDIA/DALI#compiling-dali-from-source-bare-metal
//       Will need a C++ api, currently that is experimental. Will need to talk to nVidia.

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/Stream.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "Utilities/Loaders/MemMappedFileTypes.h"
#pragma GCC diagnostic pop
#include "Utilities/WorkQueue/AsyncQueue.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

class UringDirect;

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

enum class ExampleType { TRAIN = 3, VALIDATE, TEST };

struct DataElement {
    ExampleType exampleType;
    std::string className;
    std::string fileName;

    uint64_t numDataBytes;
    std::shared_ptr<uint8_t[]> data;

    ThorImplementation::DataType dataType;
};

struct ShardMetadata {
    uint64_t exampleSizeInBytes;
    ThorImplementation::DataType dataType;
};

struct ShardExampleReadRequest {
    uint64_t fileOffsetBytes;
    uint64_t numBytes;
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
    bool isOpen();
    void writeExample(uint8_t *buffer, const std::string &label, const std::string &filename, ExampleType exampleType);
    ShardExampleReadRequest getExampleReadRequest(ExampleType exampleType, uint64_t exampleIndex);
    void loadExample(uint8_t *buffer, std::string &label, std::string &filename, ExampleType exampleType, uint64_t exampleIndex);
    void loadExampleAsync(
        uint8_t *buffer, std::string &label, std::string &filename, ExampleType exampleType, uint64_t exampleIndex, Stream stream);
    void shrinkToFit();
    std::string getFilename();
    uint64_t getExampleSizeInBytes();
    ThorImplementation::DataType getDataType();
    uint64_t getNumExamples(ExampleType exampleType);
    file_string_vector_t *getAllClasses();

   private:
    std::string filename;
    uint64_t exampleSizeInBytes;
    ThorImplementation::DataType dataType;
    boost::interprocess::managed_mapped_file mappedFile;
    bool open;

    file_vector_t *trainData;
    file_vector_t *validateData;
    file_vector_t *testData;
    file_string_vector_t *trainLabels;
    file_string_vector_t *validateLabels;
    file_string_vector_t *testLabels;
    file_string_vector_t *trainFilenames;
    file_string_vector_t *validateFilenames;
    file_string_vector_t *testFilenames;
    file_string_vector_t *allClasses;
    ShardMetadata *shardMetadata;

    std::mutex mtx;
    std::mutex cachedReaderMtx;
    std::unique_ptr<UringDirect> cachedReader;

    std::shared_ptr<file_string_allocator_t> fileStringAllocator;

    struct LabelCallbackParams {
        uint64_t index;
        file_string_vector_t *labels;
        std::string *label;
        file_string_vector_t *filenames;
        std::string *filename;
    };

    void readExamplePayloadCached(uint8_t *buffer, uint64_t fileOffsetBytes);

    static void CUDART_CB getLabelCallback(void *data);
};
