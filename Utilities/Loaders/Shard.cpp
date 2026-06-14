#include "Utilities/Loaders/Shard.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TarFile/UringDirect.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

using std::mutex;
using std::string;
using std::thread;

Shard::Shard() { open = false; }

Shard::~Shard() = default;

void Shard::createShard(string filename,
                        uint64_t numTrainExamples,
                        uint64_t numValidateExamples,
                        uint64_t numTestExamples,
                        uint64_t exampleSizeInBytes,
                        ThorImplementation::DataType dataType,
                        uint64_t maxFilenameChars,
                        std::vector<std::string> &allClassesVector,
                        uint64_t maxClassNameChars) {
    this->filename = filename;
    this->exampleSizeInBytes = exampleSizeInBytes;
    uint64_t numExamples = numTrainExamples + numValidateExamples + numTestExamples;
    uint64_t rawDataBytes = numExamples * exampleSizeInBytes;
    uint64_t labelBytes = numExamples * (maxClassNameChars + 32 + sizeof(file_string_t));
    uint64_t filenameBytes = numExamples * (maxFilenameChars + 32 + sizeof(file_string_t));
    uint64_t labelIndexBytes = allClassesVector.size() * (maxClassNameChars + 32 + sizeof(file_string_t));
    uint64_t shardSizeInBytes = rawDataBytes + labelBytes + filenameBytes + labelIndexBytes + 1000000;
    mappedFile = boost::interprocess::managed_mapped_file(boost::interprocess::create_only, filename.c_str(), shardSizeInBytes);

    shardMetadata = mappedFile.construct<ShardMetadata>("shardMetadata")();
    shardMetadata->exampleSizeInBytes = exampleSizeInBytes;
    shardMetadata->dataType = dataType;

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

    file_string_vector_allocator_t trainFilenamesAllocator(mappedFile.get_segment_manager());
    trainFilenames = mappedFile.construct<file_string_vector_t>("trainFilenames")(trainFilenamesAllocator);
    trainFilenames->reserve(numTrainExamples);

    file_string_vector_allocator_t validateFilenamesAllocator(mappedFile.get_segment_manager());
    validateFilenames = mappedFile.construct<file_string_vector_t>("validateFilenames")(validateFilenamesAllocator);
    validateFilenames->reserve(numValidateExamples);

    file_string_vector_allocator_t testFilenamesAllocator(mappedFile.get_segment_manager());
    testFilenames = mappedFile.construct<file_string_vector_t>("testFilenames")(testFilenamesAllocator);
    testFilenames->reserve(numTestExamples);

    fileStringAllocator = std::make_shared<file_string_allocator_t>(mappedFile.get_segment_manager());

    file_string_vector_allocator_t allClassesAllocator(mappedFile.get_segment_manager());
    allClasses = mappedFile.construct<file_string_vector_t>("allClasses")(allClassesAllocator);
    allClasses->reserve(allClassesVector.size());

    for (uint64_t i = 0; i < allClassesVector.size(); ++i) {
        file_string_t diskClassLabel(*fileStringAllocator);
        diskClassLabel = allClassesVector[i].c_str();
        diskClassLabel.shrink_to_fit();
        allClasses->push_back(diskClassLabel);
    }
    allClasses->shrink_to_fit();

    open = true;
}

void Shard::openShard(string filename) {
    this->filename = filename;
    mappedFile = boost::interprocess::managed_mapped_file(boost::interprocess::open_only, filename.c_str());
    trainData = mappedFile.find<file_vector_t>("train").first;
    validateData = mappedFile.find<file_vector_t>("validate").first;
    testData = mappedFile.find<file_vector_t>("test").first;
    trainLabels = mappedFile.find<file_string_vector_t>("trainLabels").first;
    validateLabels = mappedFile.find<file_string_vector_t>("validateLabels").first;
    testLabels = mappedFile.find<file_string_vector_t>("testLabels").first;
    trainFilenames = mappedFile.find<file_string_vector_t>("trainFilenames").first;
    validateFilenames = mappedFile.find<file_string_vector_t>("validateFilenames").first;
    testFilenames = mappedFile.find<file_string_vector_t>("testFilenames").first;
    shardMetadata = mappedFile.find<ShardMetadata>("shardMetadata").first;
    exampleSizeInBytes = shardMetadata->exampleSizeInBytes;
    dataType = shardMetadata->dataType;
    allClasses = mappedFile.find<file_string_vector_t>("allClasses").first;

    THOR_THROW_IF_FALSE(trainData->size() % exampleSizeInBytes == 0);
    THOR_THROW_IF_FALSE(testData->size() % exampleSizeInBytes == 0);
    THOR_THROW_IF_FALSE(validateData->size() % exampleSizeInBytes == 0);
    THOR_THROW_IF_FALSE(trainLabels->size() == trainData->size() / exampleSizeInBytes);
    THOR_THROW_IF_FALSE(validateLabels->size() == validateData->size() / exampleSizeInBytes);
    THOR_THROW_IF_FALSE(testLabels->size() == testData->size() / exampleSizeInBytes);
    THOR_THROW_IF_FALSE(trainFilenames->size() == trainData->size() / exampleSizeInBytes);
    THOR_THROW_IF_FALSE(validateFilenames->size() == validateData->size() / exampleSizeInBytes);
    THOR_THROW_IF_FALSE(testFilenames->size() == testData->size() / exampleSizeInBytes);

    open = true;
}

bool Shard::isOpen() { return open; }

void Shard::writeExample(uint8_t *buffer, const string &label, const string &filename, ExampleType exampleType) {
    THOR_THROW_IF_FALSE(buffer != nullptr);
    std::unique_lock<std::mutex> lck(mtx);

    if (exampleType == ExampleType::TRAIN) {
        THOR_THROW_IF_FALSE(trainData->capacity() > trainData->size() + exampleSizeInBytes);
        trainData->insert(trainData->end(), buffer, buffer + exampleSizeInBytes);
        THOR_THROW_IF_FALSE(trainLabels->capacity() > trainLabels->size());
        file_string_t diskLabel(*fileStringAllocator);
        diskLabel = label.c_str();
        diskLabel.shrink_to_fit();
        trainLabels->push_back(diskLabel);
        file_string_t diskFilename(*fileStringAllocator);
        diskFilename = filename.c_str();
        diskFilename.shrink_to_fit();
        trainFilenames->push_back(diskFilename);
    } else if (exampleType == ExampleType::VALIDATE) {
        THOR_THROW_IF_FALSE(validateData->capacity() > validateData->size() + exampleSizeInBytes);
        validateData->insert(validateData->end(), buffer, buffer + exampleSizeInBytes);
        THOR_THROW_IF_FALSE(validateLabels->capacity() > validateLabels->size());
        file_string_t diskLabel(*fileStringAllocator);
        diskLabel = label.c_str();
        diskLabel.shrink_to_fit();
        validateLabels->push_back(diskLabel);
        file_string_t diskFilename(*fileStringAllocator);
        diskFilename = filename.c_str();
        diskFilename.shrink_to_fit();
        validateFilenames->push_back(diskFilename);
    } else if (exampleType == ExampleType::TEST) {
        THOR_THROW_IF_FALSE(testData->capacity() > testData->size() + exampleSizeInBytes);
        testData->insert(testData->end(), buffer, buffer + exampleSizeInBytes);
        THOR_THROW_IF_FALSE(testLabels->capacity() > testLabels->size());
        file_string_t diskLabel(*fileStringAllocator);
        diskLabel = label.c_str();
        diskLabel.shrink_to_fit();
        testLabels->push_back(diskLabel);
        file_string_t diskFilename(*fileStringAllocator);
        diskFilename = filename.c_str();
        diskFilename.shrink_to_fit();
        testFilenames->push_back(diskFilename);
    } else {
        THOR_UNREACHABLE();
    }
}

ShardExampleReadRequest Shard::getExampleReadRequest(ExampleType exampleType, uint64_t exampleIndex) {
    THOR_THROW_IF_FALSE(isOpen());

    file_vector_t *data = nullptr;
    file_string_vector_t *labels = nullptr;
    file_string_vector_t *filenames = nullptr;

    if (exampleType == ExampleType::TRAIN) {
        data = trainData;
        labels = trainLabels;
        filenames = trainFilenames;
    } else if (exampleType == ExampleType::VALIDATE) {
        data = validateData;
        labels = validateLabels;
        filenames = validateFilenames;
    } else if (exampleType == ExampleType::TEST) {
        data = testData;
        labels = testLabels;
        filenames = testFilenames;
    } else {
        THOR_UNREACHABLE();
    }

    const uint64_t numExamples = data->size() / exampleSizeInBytes;
    THOR_THROW_IF_FALSE(exampleIndex < numExamples);
    THOR_THROW_IF_FALSE(labels->size() == numExamples);
    THOR_THROW_IF_FALSE(filenames->size() == numExamples);

    const auto baseAddress = reinterpret_cast<uintptr_t>(mappedFile.get_address());
    const auto dataAddress = reinterpret_cast<uintptr_t>(data->data());
    THOR_THROW_IF_FALSE(dataAddress >= baseAddress);

    const uint64_t mappedDataOffsetBytes = dataAddress - baseAddress;
    const uint64_t exampleOffsetBytes = mappedDataOffsetBytes + exampleIndex * exampleSizeInBytes;
    THOR_THROW_IF_FALSE(exampleOffsetBytes + exampleSizeInBytes <= mappedFile.get_size());

    ShardExampleReadRequest request;
    // managed_mapped_file::get_address() is Boost's real mapped-region base.
    // Object pointers returned from the managed segment are therefore already
    // file-offset-relative to that base. Do not add Boost's internal
    // ManagedOpenOrCreateUserOffset here; that would double-count the header
    // gap and shift every cached io_uring payload read forward.
    request.fileOffsetBytes = exampleOffsetBytes;
    request.numBytes = exampleSizeInBytes;
    request.label = (*labels)[exampleIndex].c_str();
    request.filename = (*filenames)[exampleIndex].c_str();
    return request;
}

void Shard::loadExample(uint8_t *buffer, string &label, string &filename, ExampleType exampleType, uint64_t exampleIndex) {
    THOR_THROW_IF_FALSE(buffer != nullptr);

    ShardExampleReadRequest request = getExampleReadRequest(exampleType, exampleIndex);
    readExamplePayloadCached(buffer, request.fileOffsetBytes);
    label = std::move(request.label);
    filename = std::move(request.filename);
}

void Shard::readExamplePayloadCached(uint8_t *buffer, uint64_t fileOffsetBytes) {
    THOR_THROW_IF_FALSE(isOpen());
    THOR_THROW_IF_FALSE(buffer != nullptr);

    std::unique_lock<std::mutex> lck(cachedReaderMtx);
    if (!cachedReader) {
        cachedReader = std::make_unique<UringDirect>(64);
        cachedReader->registerCachedLoadFile(this->filename);
    }

    uint64_t bytesDone = 0;
    while (bytesDone < exampleSizeInBytes) {
        const uint64_t remaining = exampleSizeInBytes - bytesDone;
        const uint32_t chunkBytes = static_cast<uint32_t>(
            std::min<uint64_t>(remaining, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));

        while (!cachedReader->submitReadCached(buffer + bytesDone, fileOffsetBytes + bytesDone, chunkBytes)) {
            cachedReader->submit();
        }

        cachedReader->submit();
        UringDirect::Completion completion = cachedReader->waitCompletionInOrder();
        if (completion.responseCode < 0) {
            throw std::runtime_error("cached io_uring read failed for shard '" + this->filename + "': " +
                                     std::strerror(-completion.responseCode));
        }
        THOR_THROW_IF_FALSE(completion.responseCode > 0);
        THOR_THROW_IF_FALSE(static_cast<uint64_t>(completion.responseCode) <= chunkBytes);
        bytesDone += static_cast<uint64_t>(completion.responseCode);
    }
}

void Shard::loadExampleAsync(
    uint8_t *buffer, string &label, string &filename, ExampleType exampleType, uint64_t exampleIndex, Stream stream) {
    THOR_THROW_IF_FALSE(isOpen());
    THOR_THROW_IF_FALSE(buffer != nullptr);

    uint8_t *data;
    LabelCallbackParams *labelCallbackParams = new LabelCallbackParams();
    if (exampleType == ExampleType::TRAIN) {
        uint64_t numExamples = trainData->size() / exampleSizeInBytes;
        THOR_THROW_IF_FALSE(exampleIndex < numExamples);
        data = trainData->data();
        labelCallbackParams->labels = trainLabels;
        labelCallbackParams->filenames = trainFilenames;
    } else if (exampleType == ExampleType::VALIDATE) {
        uint64_t numExamples = validateData->size() / exampleSizeInBytes;
        THOR_THROW_IF_FALSE(exampleIndex < numExamples);
        data = validateData->data();
        labelCallbackParams->labels = validateLabels;
        labelCallbackParams->filenames = validateFilenames;
    } else if (exampleType == ExampleType::TEST) {
        uint64_t numExamples = testData->size() / exampleSizeInBytes;
        THOR_THROW_IF_FALSE(exampleIndex < numExamples);
        data = testData->data();
        labelCallbackParams->labels = testLabels;
        labelCallbackParams->filenames = testFilenames;
    } else {
        THOR_UNREACHABLE();
    }

    uint8_t *exampleStart = data + (exampleIndex * exampleSizeInBytes);
    CUDA_CHECK(cudaMemcpyAsync(buffer, exampleStart, exampleSizeInBytes, cudaMemcpyHostToHost, stream));
    labelCallbackParams->index = exampleIndex;
    labelCallbackParams->label = &label;
    labelCallbackParams->filename = &filename;
    CUDA_CHECK(cudaLaunchHostFunc(stream, getLabelCallback, labelCallbackParams));
}

void Shard::shrinkToFit() {
    trainData->shrink_to_fit();
    validateData->shrink_to_fit();
    testData->shrink_to_fit();
    trainLabels->shrink_to_fit();
    validateLabels->shrink_to_fit();
    testLabels->shrink_to_fit();
    trainFilenames->shrink_to_fit();
    validateFilenames->shrink_to_fit();
    testFilenames->shrink_to_fit();
    bool status = boost::interprocess::managed_mapped_file::shrink_to_fit(filename.c_str());
    THOR_THROW_IF_FALSE(status == true);
}

string Shard::getFilename() { return filename; }

uint64_t Shard::getExampleSizeInBytes() { return exampleSizeInBytes; }

ThorImplementation::DataType Shard::getDataType() { return dataType; }

uint64_t Shard::getNumExamples(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return trainLabels->size();
    } else if (exampleType == ExampleType::VALIDATE) {
        return validateLabels->size();
    } else if (exampleType == ExampleType::TEST) {
        return testLabels->size();
    } else {
        THOR_UNREACHABLE();
    }
}

file_string_vector_t *Shard::getAllClasses() { return allClasses; }

void CUDART_CB Shard::getLabelCallback(void *data) {
    LabelCallbackParams *params = (LabelCallbackParams *)data;
    *(params->label) = (*(params->labels))[params->index].c_str();
    *(params->filename) = (*(params->filenames))[params->index].c_str();
    delete params;
}
