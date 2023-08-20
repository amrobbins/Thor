#include "Utilities/Loaders/Shard.h"

using std::mutex;
using std::string;
using std::thread;

Shard::Shard() { open = false; }

void Shard::createShard(string filename,
                        uint64_t numTrainExamples,
                        uint64_t numValidateExamples,
                        uint64_t numTestExamples,
                        uint64_t exampleSizeInBytes,
                        ThorImplementation::TensorDescriptor::DataType dataType,
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

    assert(trainData->size() % exampleSizeInBytes == 0);
    assert(testData->size() % exampleSizeInBytes == 0);
    assert(validateData->size() % exampleSizeInBytes == 0);
    assert(trainLabels->size() == trainData->size() / exampleSizeInBytes);
    assert(validateLabels->size() == validateData->size() / exampleSizeInBytes);
    assert(testLabels->size() == testData->size() / exampleSizeInBytes);
    assert(trainFilenames->size() == trainData->size() / exampleSizeInBytes);
    assert(validateFilenames->size() == validateData->size() / exampleSizeInBytes);
    assert(testFilenames->size() == testData->size() / exampleSizeInBytes);

    open = true;
}

bool Shard::isOpen() { return open; }

void Shard::writeExample(uint8_t *buffer, const string &label, const string &filename, ExampleType exampleType) {
    assert(buffer != nullptr);
    std::unique_lock<std::mutex> lck(mtx);

    if (exampleType == ExampleType::TRAIN) {
        assert(trainData->capacity() > trainData->size() + exampleSizeInBytes);
        trainData->insert(trainData->end(), buffer, buffer + exampleSizeInBytes);
        assert(trainLabels->capacity() > trainLabels->size());
        file_string_t diskLabel(*fileStringAllocator);
        diskLabel = label.c_str();
        diskLabel.shrink_to_fit();
        trainLabels->push_back(diskLabel);
        file_string_t diskFilename(*fileStringAllocator);
        diskFilename = filename.c_str();
        diskFilename.shrink_to_fit();
        trainFilenames->push_back(diskFilename);
    } else if (exampleType == ExampleType::VALIDATE) {
        assert(validateData->capacity() > validateData->size() + exampleSizeInBytes);
        validateData->insert(validateData->end(), buffer, buffer + exampleSizeInBytes);
        assert(validateLabels->capacity() > validateLabels->size());
        file_string_t diskLabel(*fileStringAllocator);
        diskLabel = label.c_str();
        diskLabel.shrink_to_fit();
        validateLabels->push_back(diskLabel);
        file_string_t diskFilename(*fileStringAllocator);
        diskFilename = filename.c_str();
        diskFilename.shrink_to_fit();
        validateFilenames->push_back(diskFilename);
    } else if (exampleType == ExampleType::TEST) {
        assert(testData->capacity() > testData->size() + exampleSizeInBytes);
        testData->insert(testData->end(), buffer, buffer + exampleSizeInBytes);
        assert(testLabels->capacity() > testLabels->size());
        file_string_t diskLabel(*fileStringAllocator);
        diskLabel = label.c_str();
        diskLabel.shrink_to_fit();
        testLabels->push_back(diskLabel);
        file_string_t diskFilename(*fileStringAllocator);
        diskFilename = filename.c_str();
        diskFilename.shrink_to_fit();
        testFilenames->push_back(diskFilename);
    } else {
        assert(false);
    }
}

void Shard::loadExample(uint8_t *buffer, string &label, string &filename, ExampleType exampleType, uint64_t exampleIndex) {
    assert(isOpen());
    assert(buffer != nullptr);

    uint8_t *data;
    if (exampleType == ExampleType::TRAIN) {
        uint64_t numExamples = trainData->size() / exampleSizeInBytes;
        assert(exampleIndex < numExamples);
        data = trainData->data();
        label = (*trainLabels)[exampleIndex].c_str();
        filename = (*trainFilenames)[exampleIndex].c_str();
    } else if (exampleType == ExampleType::VALIDATE) {
        uint64_t numExamples = validateData->size() / exampleSizeInBytes;
        assert(exampleIndex < numExamples);
        data = validateData->data();
        label = (*validateLabels)[exampleIndex].c_str();
        filename = (*validateFilenames)[exampleIndex].c_str();
    } else if (exampleType == ExampleType::TEST) {
        uint64_t numExamples = testData->size() / exampleSizeInBytes;
        assert(exampleIndex < numExamples);
        data = testData->data();
        label = (*testLabels)[exampleIndex].c_str();
        filename = (*testFilenames)[exampleIndex].c_str();
    } else {
        assert(false);
    }

    uint8_t *exampleStart = data + (exampleIndex * exampleSizeInBytes);
    memcpy(buffer, exampleStart, exampleSizeInBytes);
}

void Shard::loadExampleAsync(
    uint8_t *buffer, string &label, string &filename, ExampleType exampleType, uint64_t exampleIndex, Stream stream) {
    assert(isOpen());
    assert(buffer != nullptr);
    cudaError_t cudaStatus;

    uint8_t *data;
    LabelCallbackParams *labelCallbackParams = new LabelCallbackParams();
    if (exampleType == ExampleType::TRAIN) {
        uint64_t numExamples = trainData->size() / exampleSizeInBytes;
        assert(exampleIndex < numExamples);
        data = trainData->data();
        labelCallbackParams->labels = trainLabels;
        labelCallbackParams->filenames = trainFilenames;
    } else if (exampleType == ExampleType::VALIDATE) {
        uint64_t numExamples = validateData->size() / exampleSizeInBytes;
        assert(exampleIndex < numExamples);
        data = validateData->data();
        labelCallbackParams->labels = validateLabels;
        labelCallbackParams->filenames = validateFilenames;
    } else if (exampleType == ExampleType::TEST) {
        uint64_t numExamples = testData->size() / exampleSizeInBytes;
        assert(exampleIndex < numExamples);
        data = testData->data();
        labelCallbackParams->labels = testLabels;
        labelCallbackParams->filenames = testFilenames;
    } else {
        assert(false);
    }

    uint8_t *exampleStart = data + (exampleIndex * exampleSizeInBytes);
    cudaStatus = cudaMemcpyAsync(buffer, exampleStart, exampleSizeInBytes, cudaMemcpyHostToHost, stream);
    assert(cudaStatus == cudaSuccess);
    labelCallbackParams->index = exampleIndex;
    labelCallbackParams->label = &label;
    labelCallbackParams->filename = &filename;
    cudaStatus = cudaLaunchHostFunc(stream, getLabelCallback, labelCallbackParams);
    assert(cudaStatus == cudaSuccess);
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
    assert(status == true);
}

string Shard::getFilename() { return filename; }

uint64_t Shard::getExampleSizeInBytes() { return exampleSizeInBytes; }

ThorImplementation::TensorDescriptor::DataType Shard::getDataType() { return dataType; }

uint64_t Shard::getNumExamples(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return trainLabels->size();
    } else if (exampleType == ExampleType::VALIDATE) {
        return validateLabels->size();
    } else if (exampleType == ExampleType::TEST) {
        return testLabels->size();
    } else {
        assert(false);
    }
}

file_string_vector_t *Shard::getAllClasses() { return allClasses; }

void CUDART_CB Shard::getLabelCallback(void *data) {
    LabelCallbackParams *params = (LabelCallbackParams *)data;
    *(params->label) = (*(params->labels))[params->index].c_str();
    *(params->filename) = (*(params->filenames))[params->index].c_str();
    delete params;
}
