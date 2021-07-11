#pragma once

// TODO: Integrate with DALI
//       https://developer.nvidia.com/DALI
//       https://github.com/NVIDIA/DALI#compiling-dali-from-source-bare-metal
//       Will need a C++ api, currently that is experimental. Will need to talk to nVidia.

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Loaders/MemMappedFileTypes.h"
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

enum class ExampleType { TRAIN = 3, VALIDATE, TEST };

struct DataElement {
    ExampleType exampleType;
    std::string className;
    std::string fileName;

    uint64_t numDataBytes;
    std::shared_ptr<uint8_t> data;

    ThorImplementation::TensorDescriptor::DataType dataType;
};

struct ShardMetadata {
    uint64_t exampleSizeInBytes;
    ThorImplementation::TensorDescriptor::DataType dataType;
};

class Shard {
   public:
    Shard();

    void createShard(std::string filename,
                     uint64_t numTrainExamples,
                     uint64_t numValidateExamples,
                     uint64_t numTestExamples,
                     uint64_t exampleSizeInBytes,
                     ThorImplementation::TensorDescriptor::DataType dataType,
                     uint64_t maxFilenameChars,
                     std::vector<std::string> &allClassesVector,
                     uint64_t maxClassNameChars);

    void openShard(std::string filename);
    bool isOpen();
    void writeExample(uint8_t *buffer, const std::string &label, const std::string &filename, ExampleType exampleType);
    void loadExample(uint8_t *buffer, std::string &label, std::string &filename, ExampleType exampleType, uint64_t exampleIndex);
    void loadExampleAsync(
        uint8_t *buffer, std::string &label, std::string &filename, ExampleType exampleType, uint64_t exampleIndex, Stream stream);
    void shrinkToFit();
    std::string getFilename();
    uint64_t getExampleSizeInBytes();
    ThorImplementation::TensorDescriptor::DataType getDataType();
    uint64_t getNumExamples(ExampleType exampleType);
    file_string_vector_t *getAllClasses();

   private:
    std::string filename;
    uint64_t exampleSizeInBytes;
    ThorImplementation::TensorDescriptor::DataType dataType;
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

    std::shared_ptr<std::mutex> mtx;

    std::shared_ptr<file_string_allocator_t> fileStringAllocator;

    struct LabelCallbackParams {
        uint64_t index;
        file_string_vector_t *labels;
        std::string *label;
        file_string_vector_t *filenames;
        std::string *filename;
    };

    static void CUDART_CB getLabelCallback(void *data);
};
