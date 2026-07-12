#include "Utilities/Data/Storage/DatasetShard.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TarFile/UringDirect.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

using std::mutex;
using std::string;

namespace {

constexpr std::array<char, 16> SHARD_MAGIC = {'T', 'H', 'O', 'R', '_', 'R', 'A', 'W', '_', 'S', 'H', 'A', 'R', 'D', '\0', '\0'};
constexpr uint32_t SHARD_FORMAT_VERSION = 1;
constexpr uint32_t SHARD_METADATA_LAYOUT_EXPLICIT = 0;
constexpr uint32_t SHARD_METADATA_LAYOUT_COMPACT = 1;
constexpr uint32_t SHARD_HEADER_BYTES = 88;

void writeExact(std::ostream &stream, const void *data, uint64_t numBytes) {
    stream.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(numBytes));
    THOR_THROW_IF_FALSE(stream.good());
}

void readExact(std::istream &stream, void *data, uint64_t numBytes) {
    stream.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(numBytes));
    THOR_THROW_IF_FALSE(stream.good());
}

void writeUint32(std::ostream &stream, uint32_t value) { writeExact(stream, &value, sizeof(value)); }

void writeUint64(std::ostream &stream, uint64_t value) { writeExact(stream, &value, sizeof(value)); }

uint32_t readUint32(std::istream &stream) {
    uint32_t value = 0;
    readExact(stream, &value, sizeof(value));
    return value;
}

uint64_t readUint64(std::istream &stream) {
    uint64_t value = 0;
    readExact(stream, &value, sizeof(value));
    return value;
}

void writeString(std::ostream &stream, const std::string &value) {
    writeUint64(stream, value.size());
    if (!value.empty()) {
        writeExact(stream, value.data(), value.size());
    }
}

std::string readString(std::istream &stream, uint64_t fileSizeBytes) {
    uint64_t size = readUint64(stream);
    THOR_THROW_IF_FALSE(size <= fileSizeBytes);
    std::string value(size, '\0');
    if (size != 0) {
        readExact(stream, value.data(), size);
    }
    return value;
}

void writeRecordVector(std::ostream &stream, const std::vector<DatasetShardRecord> &records) {
    for (const DatasetShardRecord &record : records) {
        writeUint64(stream, record.fileOffsetBytes);
        writeString(stream, record.label);
        writeString(stream, record.filename);
    }
}

void readRecordVector(std::istream &stream,
                      std::vector<DatasetShardRecord> &records,
                      uint64_t recordCount,
                      uint64_t fileSizeBytes,
                      uint64_t exampleSizeInBytes) {
    records.clear();
    records.reserve(recordCount);
    for (uint64_t i = 0; i < recordCount; ++i) {
        DatasetShardRecord record;
        record.fileOffsetBytes = readUint64(stream);
        THOR_THROW_IF_FALSE(record.fileOffsetBytes <= fileSizeBytes);
        THOR_THROW_IF_FALSE(exampleSizeInBytes <= fileSizeBytes - record.fileOffsetBytes);
        record.label = readString(stream, fileSizeBytes);
        record.filename = readString(stream, fileSizeBytes);
        records.push_back(std::move(record));
    }
}

uint64_t checkedMul(uint64_t a, uint64_t b, const char *context) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return a * b;
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const char *context) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return a + b;
}

void preallocateFileBytes(const std::string &filename, uint64_t numBytes) {
    if (numBytes == 0) {
        return;
    }
    const int fd = ::open(filename.c_str(), O_RDWR);
    if (fd < 0) {
        throw std::runtime_error("failed to open shard for preallocation '" + filename + "': " + std::strerror(errno));
    }
    const int rc = ::posix_fallocate(fd, 0, static_cast<off_t>(numBytes));
    const int savedErrno = errno;
    ::close(fd);
    if (rc != 0) {
        throw std::runtime_error("failed to preallocate shard '" + filename + "': " + std::strerror(rc == -1 ? savedErrno : rc));
    }
}

}  // namespace

DatasetShard::DatasetShard() {
    exampleSizeInBytes = 0;
    dataType = ThorImplementation::DataType::UINT8;
    open = false;
    metadataFinalized = false;
    compactMetadata = false;
    compactTrainOffsetBytes = 0;
    compactValidateOffsetBytes = 0;
    compactTestOffsetBytes = 0;
    compactTrainCount = 0;
    compactValidateCount = 0;
    compactTestCount = 0;
    compactTrainCapacity = 0;
    compactValidateCapacity = 0;
    compactTestCapacity = 0;
    compactTrainBytes = 0;
    compactValidateBytes = 0;
    compactTestBytes = 0;
}

DatasetShard::~DatasetShard() {
    if (shardFile.is_open()) {
        shardFile.close();
    }
}

void DatasetShard::createShard(string filename,
                        uint64_t numTrainExamples,
                        uint64_t numValidateExamples,
                        uint64_t numTestExamples,
                        uint64_t exampleSizeInBytes,
                        ThorImplementation::DataType dataType,
                        uint64_t maxFilenameChars,
                        std::vector<std::string> &allClassesVector,
                        uint64_t maxClassNameChars) {
    (void)maxFilenameChars;
    (void)maxClassNameChars;

    if (shardFile.is_open()) {
        shardFile.close();
    }
    cachedReader.reset();

    THOR_THROW_IF_FALSE(exampleSizeInBytes > 0);

    this->filename = filename;
    this->exampleSizeInBytes = exampleSizeInBytes;
    this->dataType = dataType;
    this->allClasses = allClassesVector;

    trainExamples.clear();
    validateExamples.clear();
    testExamples.clear();
    compactMetadata = false;
    compactTrainOffsetBytes = 0;
    compactValidateOffsetBytes = 0;
    compactTestOffsetBytes = 0;
    compactTrainCount = 0;
    compactValidateCount = 0;
    compactTestCount = 0;
    compactTrainCapacity = 0;
    compactValidateCapacity = 0;
    compactTestCapacity = 0;
    compactTrainBytes = 0;
    compactValidateBytes = 0;
    compactTestBytes = 0;
    trainExamples.reserve(numTrainExamples);
    validateExamples.reserve(numValidateExamples);
    testExamples.reserve(numTestExamples);

    shardFile.open(filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    THOR_THROW_IF_FALSE(shardFile.is_open());
    writeHeader(0, 0);
    shardFile.flush();
    THOR_THROW_IF_FALSE(shardFile.good());

    metadataFinalized = false;
    open = true;
}

void DatasetShard::createCompactShard(string filename,
                               uint64_t numTrainExamples,
                               uint64_t numValidateExamples,
                               uint64_t numTestExamples,
                               uint64_t exampleSizeInBytes,
                               ThorImplementation::DataType dataType,
                               std::vector<std::string> &allClassesVector,
                               bool preallocate) {
    if (shardFile.is_open()) {
        shardFile.close();
    }
    cachedReader.reset();

    THOR_THROW_IF_FALSE(exampleSizeInBytes > 0);

    this->filename = std::move(filename);
    this->exampleSizeInBytes = exampleSizeInBytes;
    this->dataType = dataType;
    this->allClasses = allClassesVector;

    trainExamples.clear();
    validateExamples.clear();
    testExamples.clear();
    compactMetadata = true;

    compactTrainCapacity = numTrainExamples;
    compactValidateCapacity = numValidateExamples;
    compactTestCapacity = numTestExamples;
    compactTrainCount = 0;
    compactValidateCount = 0;
    compactTestCount = 0;

    compactTrainOffsetBytes = SHARD_HEADER_BYTES;
    compactTrainBytes = 0;
    const uint64_t trainCapacityBytes = checkedMul(compactTrainCapacity, exampleSizeInBytes, "DatasetShard compact train capacity");
    compactValidateOffsetBytes = checkedAdd(compactTrainOffsetBytes, trainCapacityBytes, "DatasetShard compact validate offset");
    compactValidateBytes = 0;
    const uint64_t validateCapacityBytes = checkedMul(compactValidateCapacity, exampleSizeInBytes, "DatasetShard compact validate capacity");
    compactTestOffsetBytes = checkedAdd(compactValidateOffsetBytes, validateCapacityBytes, "DatasetShard compact test offset");
    compactTestBytes = 0;

    const uint64_t testCapacityBytes = checkedMul(compactTestCapacity, exampleSizeInBytes, "DatasetShard compact test capacity");
    uint64_t preallocateBytes = checkedAdd(compactTestOffsetBytes, testCapacityBytes, "DatasetShard compact payload capacity");
    uint64_t metadataReserveBytes = 6 * sizeof(uint64_t);
    for (const std::string &className : allClasses) {
        metadataReserveBytes = checkedAdd(metadataReserveBytes, sizeof(uint64_t), "DatasetShard compact metadata reserve");
        metadataReserveBytes = checkedAdd(metadataReserveBytes, className.size(), "DatasetShard compact metadata reserve");
    }
    preallocateBytes = checkedAdd(preallocateBytes, metadataReserveBytes, "DatasetShard compact preallocation size");

    shardFile.open(this->filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    THOR_THROW_IF_FALSE(shardFile.is_open());
    writeHeader(0, 0, SHARD_METADATA_LAYOUT_COMPACT);
    shardFile.flush();
    THOR_THROW_IF_FALSE(shardFile.good());

    if (preallocate) {
        preallocateFileBytes(this->filename, preallocateBytes);
    }

    metadataFinalized = false;
    open = true;
}

void DatasetShard::openShard(string filename) {
    if (shardFile.is_open()) {
        shardFile.close();
    }
    cachedReader.reset();
    trainExamples.clear();
    validateExamples.clear();
    testExamples.clear();
    allClasses.clear();
    compactMetadata = false;
    compactTrainOffsetBytes = 0;
    compactValidateOffsetBytes = 0;
    compactTestOffsetBytes = 0;
    compactTrainCount = 0;
    compactValidateCount = 0;
    compactTestCount = 0;
    compactTrainCapacity = 0;
    compactValidateCapacity = 0;
    compactTestCapacity = 0;
    compactTrainBytes = 0;
    compactValidateBytes = 0;
    compactTestBytes = 0;

    this->filename = filename;

    std::ifstream input(filename, std::ios::binary);
    THOR_THROW_IF_FALSE(input.is_open());

    std::array<char, 16> magic{};
    readExact(input, magic.data(), magic.size());
    THOR_THROW_IF_FALSE(magic == SHARD_MAGIC);

    const uint32_t version = readUint32(input);
    const uint32_t headerBytes = readUint32(input);
    THOR_THROW_IF_FALSE(version == SHARD_FORMAT_VERSION);
    THOR_THROW_IF_FALSE(headerBytes == SHARD_HEADER_BYTES);

    exampleSizeInBytes = readUint64(input);
    dataType = static_cast<ThorImplementation::DataType>(readUint32(input));
    const uint32_t metadataLayout = readUint32(input);
    THOR_THROW_IF_FALSE(metadataLayout == SHARD_METADATA_LAYOUT_EXPLICIT || metadataLayout == SHARD_METADATA_LAYOUT_COMPACT);

    const uint64_t trainCount = readUint64(input);
    const uint64_t validateCount = readUint64(input);
    const uint64_t testCount = readUint64(input);
    const uint64_t classCount = readUint64(input);
    const uint64_t metadataOffsetBytes = readUint64(input);
    const uint64_t metadataBytes = readUint64(input);

    const uint64_t fileSizeBytes = std::filesystem::file_size(filename);
    THOR_THROW_IF_FALSE(exampleSizeInBytes > 0);
    THOR_THROW_IF_FALSE(metadataOffsetBytes >= SHARD_HEADER_BYTES);
    THOR_THROW_IF_FALSE(metadataOffsetBytes <= fileSizeBytes);
    THOR_THROW_IF_FALSE(metadataBytes <= fileSizeBytes - metadataOffsetBytes);

    readMetadata(metadataLayout, metadataOffsetBytes, metadataBytes, trainCount, validateCount, testCount, classCount);

    if (compactMetadata) {
        THOR_THROW_IF_FALSE(compactTrainCount == trainCount);
        THOR_THROW_IF_FALSE(compactValidateCount == validateCount);
        THOR_THROW_IF_FALSE(compactTestCount == testCount);
    } else {
        THOR_THROW_IF_FALSE(trainExamples.size() == trainCount);
        THOR_THROW_IF_FALSE(validateExamples.size() == validateCount);
        THOR_THROW_IF_FALSE(testExamples.size() == testCount);
    }
    THOR_THROW_IF_FALSE(allClasses.size() == classCount);

    metadataFinalized = true;
    open = true;
}

bool DatasetShard::isOpen() { return open; }

void DatasetShard::writeExample(uint8_t *buffer, const string &label, const string &filename, ExampleType exampleType) {
    THOR_THROW_IF_FALSE(buffer != nullptr);
    THOR_THROW_IF_FALSE(isOpen());
    THOR_THROW_IF_FALSE(!metadataFinalized);

    std::unique_lock<std::mutex> lck(mtx);
    THOR_THROW_IF_FALSE(shardFile.is_open());

    shardFile.seekp(0, std::ios::end);
    THOR_THROW_IF_FALSE(shardFile.good());
    const std::streamoff payloadOffset = shardFile.tellp();
    THOR_THROW_IF_FALSE(payloadOffset >= 0);

    writeExact(shardFile, buffer, exampleSizeInBytes);

    DatasetShardRecord record;
    record.fileOffsetBytes = static_cast<uint64_t>(payloadOffset);
    record.label = label;
    record.filename = filename;
    mutableRecordsFor(exampleType).push_back(std::move(record));
}

void DatasetShard::writeExamplesContiguous(uint8_t *buffer, uint64_t numExamples, ExampleType exampleType) {
    THOR_THROW_IF_FALSE(buffer != nullptr);
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(isOpen());
    THOR_THROW_IF_FALSE(!metadataFinalized);
    THOR_THROW_IF_FALSE(compactMetadata);

    std::unique_lock<std::mutex> lck(mtx);
    THOR_THROW_IF_FALSE(shardFile.is_open());

    uint64_t *count = nullptr;
    uint64_t *bytes = nullptr;
    uint64_t capacity = 0;
    uint64_t baseOffsetBytes = 0;
    if (exampleType == ExampleType::TRAIN) {
        count = &compactTrainCount;
        bytes = &compactTrainBytes;
        capacity = compactTrainCapacity;
        baseOffsetBytes = compactTrainOffsetBytes;
    } else if (exampleType == ExampleType::VALIDATE) {
        count = &compactValidateCount;
        bytes = &compactValidateBytes;
        capacity = compactValidateCapacity;
        baseOffsetBytes = compactValidateOffsetBytes;
    } else if (exampleType == ExampleType::TEST) {
        count = &compactTestCount;
        bytes = &compactTestBytes;
        capacity = compactTestCapacity;
        baseOffsetBytes = compactTestOffsetBytes;
    } else {
        THOR_UNREACHABLE();
    }

    THOR_THROW_IF_FALSE(*count <= capacity);
    THOR_THROW_IF_FALSE(numExamples <= capacity - *count);
    const uint64_t writeOffsetBytes = checkedAdd(baseOffsetBytes, checkedMul(*count, exampleSizeInBytes, "DatasetShard compact write offset"),
                                                 "DatasetShard compact write offset");
    const uint64_t writeBytes = checkedMul(numExamples, exampleSizeInBytes, "DatasetShard compact write bytes");

    shardFile.seekp(static_cast<std::streamoff>(writeOffsetBytes), std::ios::beg);
    THOR_THROW_IF_FALSE(shardFile.good());
    writeExact(shardFile, buffer, writeBytes);

    *count += numExamples;
    *bytes = checkedMul(*count, exampleSizeInBytes, "DatasetShard compact payload bytes");
}

DatasetShardReadRequest DatasetShard::getExampleReadRequest(ExampleType exampleType, uint64_t exampleIndex) {
    THOR_THROW_IF_FALSE(isOpen());

    DatasetShardReadRequest request;
    request.numBytes = exampleSizeInBytes;

    if (compactMetadata) {
        THOR_THROW_IF_FALSE(exampleIndex < compactCountFor(exampleType));
        const uint64_t baseOffsetBytes = compactOffsetFor(exampleType);
        THOR_THROW_IF_FALSE(exampleIndex <= (std::numeric_limits<uint64_t>::max() - baseOffsetBytes) / exampleSizeInBytes);
        request.fileOffsetBytes = baseOffsetBytes + exampleIndex * exampleSizeInBytes;
        THOR_THROW_IF_FALSE(request.fileOffsetBytes >= baseOffsetBytes);
        THOR_THROW_IF_FALSE(exampleSizeInBytes <= compactBytesFor(exampleType));
        THOR_THROW_IF_FALSE(request.fileOffsetBytes - baseOffsetBytes <= compactBytesFor(exampleType) - exampleSizeInBytes);
        request.label = allClasses.empty() ? std::string() : allClasses.front();
        request.filename = std::string();
        return request;
    }

    const std::vector<DatasetShardRecord> &records = recordsFor(exampleType);
    THOR_THROW_IF_FALSE(exampleIndex < records.size());

    const DatasetShardRecord &record = records[exampleIndex];
    request.fileOffsetBytes = record.fileOffsetBytes;
    request.label = record.label;
    request.filename = record.filename;
    return request;
}

void DatasetShard::loadExample(uint8_t *buffer, string &label, string &filename, ExampleType exampleType, uint64_t exampleIndex) {
    THOR_THROW_IF_FALSE(buffer != nullptr);

    DatasetShardReadRequest request = getExampleReadRequest(exampleType, exampleIndex);
    readExamplePayloadCached(buffer, request.fileOffsetBytes);
    label = std::move(request.label);
    filename = std::move(request.filename);
}

void DatasetShard::readExamplePayloadCached(uint8_t *buffer, uint64_t fileOffsetBytes) {
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

void DatasetShard::shrinkToFit() {
    std::unique_lock<std::mutex> lck(mtx);
    if (metadataFinalized) {
        return;
    }
    THOR_THROW_IF_FALSE(shardFile.is_open());
    writeMetadata();
    shardFile.flush();
    THOR_THROW_IF_FALSE(shardFile.good());
    shardFile.close();
    metadataFinalized = true;
}

string DatasetShard::getFilename() { return filename; }

uint64_t DatasetShard::getExampleSizeInBytes() { return exampleSizeInBytes; }

ThorImplementation::DataType DatasetShard::getDataType() { return dataType; }

uint64_t DatasetShard::getNumExamples(ExampleType exampleType) {
    if (compactMetadata) {
        return compactCountFor(exampleType);
    }
    return recordsFor(exampleType).size();
}

const std::vector<std::string> &DatasetShard::getAllClasses() { return allClasses; }

void DatasetShard::writeHeader(uint64_t metadataOffsetBytes, uint64_t metadataBytes, uint32_t metadataLayout) {
    shardFile.seekp(0, std::ios::beg);
    THOR_THROW_IF_FALSE(shardFile.good());

    uint64_t trainCount = trainExamples.size();
    uint64_t validateCount = validateExamples.size();
    uint64_t testCount = testExamples.size();
    if (metadataLayout == SHARD_METADATA_LAYOUT_COMPACT) {
        trainCount = compactTrainCount;
        validateCount = compactValidateCount;
        testCount = compactTestCount;
    } else {
        THOR_THROW_IF_FALSE(metadataLayout == SHARD_METADATA_LAYOUT_EXPLICIT);
    }

    writeExact(shardFile, SHARD_MAGIC.data(), SHARD_MAGIC.size());
    writeUint32(shardFile, SHARD_FORMAT_VERSION);
    writeUint32(shardFile, SHARD_HEADER_BYTES);
    writeUint64(shardFile, exampleSizeInBytes);
    writeUint32(shardFile, static_cast<uint32_t>(dataType));
    writeUint32(shardFile, metadataLayout);
    writeUint64(shardFile, trainCount);
    writeUint64(shardFile, validateCount);
    writeUint64(shardFile, testCount);
    writeUint64(shardFile, allClasses.size());
    writeUint64(shardFile, metadataOffsetBytes);
    writeUint64(shardFile, metadataBytes);

    const std::streamoff headerEnd = shardFile.tellp();
    THOR_THROW_IF_FALSE(headerEnd == static_cast<std::streamoff>(SHARD_HEADER_BYTES));
}

void DatasetShard::writeMetadata() {
    uint32_t metadataLayout = SHARD_METADATA_LAYOUT_EXPLICIT;
    if (compactMetadata) {
        metadataLayout = SHARD_METADATA_LAYOUT_COMPACT;
        const uint64_t trainCapacityBytes = checkedMul(compactTrainCapacity, exampleSizeInBytes, "DatasetShard compact metadata offset");
        const uint64_t validateCapacityBytes = checkedMul(compactValidateCapacity, exampleSizeInBytes, "DatasetShard compact metadata offset");
        const uint64_t testCapacityBytes = checkedMul(compactTestCapacity, exampleSizeInBytes, "DatasetShard compact metadata offset");
        const uint64_t payloadCapacityBytes = checkedAdd(checkedAdd(trainCapacityBytes, validateCapacityBytes, "DatasetShard compact metadata offset"),
                                                        testCapacityBytes,
                                                        "DatasetShard compact metadata offset");
        const uint64_t metadataOffsetBytes = checkedAdd(SHARD_HEADER_BYTES, payloadCapacityBytes, "DatasetShard compact metadata offset");
        shardFile.seekp(static_cast<std::streamoff>(metadataOffsetBytes), std::ios::beg);
    } else {
        shardFile.seekp(0, std::ios::end);
    }
    THOR_THROW_IF_FALSE(shardFile.good());
    const std::streamoff metadataOffset = shardFile.tellp();
    THOR_THROW_IF_FALSE(metadataOffset >= 0);

    for (const std::string &className : allClasses) {
        writeString(shardFile, className);
    }
    if (compactMetadata) {
        writeUint64(shardFile, compactTrainOffsetBytes);
        writeUint64(shardFile, compactTrainCount);
        writeUint64(shardFile, compactValidateOffsetBytes);
        writeUint64(shardFile, compactValidateCount);
        writeUint64(shardFile, compactTestOffsetBytes);
        writeUint64(shardFile, compactTestCount);
    } else {
        writeRecordVector(shardFile, trainExamples);
        writeRecordVector(shardFile, validateExamples);
        writeRecordVector(shardFile, testExamples);
    }

    const std::streamoff metadataEnd = shardFile.tellp();
    THOR_THROW_IF_FALSE(metadataEnd >= metadataOffset);
    const uint64_t metadataOffsetBytes = static_cast<uint64_t>(metadataOffset);
    const uint64_t metadataBytes = static_cast<uint64_t>(metadataEnd - metadataOffset);

    writeHeader(metadataOffsetBytes, metadataBytes, metadataLayout);

    shardFile.flush();
    THOR_THROW_IF_FALSE(shardFile.good());
    std::filesystem::resize_file(filename, metadataOffsetBytes + metadataBytes);
}

void DatasetShard::readMetadata(uint32_t metadataLayout,
                         uint64_t metadataOffsetBytes,
                         uint64_t metadataBytes,
                         uint64_t trainCount,
                         uint64_t validateCount,
                         uint64_t testCount,
                         uint64_t classCount) {
    const uint64_t fileSizeBytes = std::filesystem::file_size(filename);
    THOR_THROW_IF_FALSE(metadataOffsetBytes <= fileSizeBytes);
    THOR_THROW_IF_FALSE(metadataBytes <= fileSizeBytes - metadataOffsetBytes);

    std::ifstream input(filename, std::ios::binary);
    THOR_THROW_IF_FALSE(input.is_open());
    input.seekg(static_cast<std::streamoff>(metadataOffsetBytes), std::ios::beg);
    THOR_THROW_IF_FALSE(input.good());

    allClasses.clear();
    allClasses.reserve(classCount);
    for (uint64_t i = 0; i < classCount; ++i) {
        allClasses.push_back(readString(input, fileSizeBytes));
    }

    if (metadataLayout == SHARD_METADATA_LAYOUT_COMPACT) {
        compactMetadata = true;

        compactTrainOffsetBytes = readUint64(input);
        compactTrainCount = readUint64(input);
        compactTrainCapacity = compactTrainCount;
        compactTrainBytes = checkedMul(compactTrainCount, exampleSizeInBytes, "DatasetShard compact train payload bytes");
        compactValidateOffsetBytes = readUint64(input);
        compactValidateCount = readUint64(input);
        compactValidateCapacity = compactValidateCount;
        compactValidateBytes = checkedMul(compactValidateCount, exampleSizeInBytes, "DatasetShard compact validate payload bytes");
        compactTestOffsetBytes = readUint64(input);
        compactTestCount = readUint64(input);
        compactTestCapacity = compactTestCount;
        compactTestBytes = checkedMul(compactTestCount, exampleSizeInBytes, "DatasetShard compact test payload bytes");

        THOR_THROW_IF_FALSE(compactTrainCount == trainCount);
        THOR_THROW_IF_FALSE(compactValidateCount == validateCount);
        THOR_THROW_IF_FALSE(compactTestCount == testCount);
        auto validateRange = [&](uint64_t offsetBytes, uint64_t count, uint64_t payloadBytes) {
            THOR_THROW_IF_FALSE(offsetBytes <= fileSizeBytes);
            THOR_THROW_IF_FALSE(payloadBytes <= fileSizeBytes - offsetBytes);
            THOR_THROW_IF_FALSE(offsetBytes + payloadBytes <= metadataOffsetBytes);
            if (count == 0) {
                THOR_THROW_IF_FALSE(payloadBytes < exampleSizeInBytes);
                return;
            }
            THOR_THROW_IF_FALSE(checkedMul(count, exampleSizeInBytes, "DatasetShard compact payload bytes") <= payloadBytes);
        };
        validateRange(compactTrainOffsetBytes, compactTrainCount, compactTrainBytes);
        validateRange(compactValidateOffsetBytes, compactValidateCount, compactValidateBytes);
        validateRange(compactTestOffsetBytes, compactTestCount, compactTestBytes);
    } else {
        THOR_THROW_IF_FALSE(metadataLayout == SHARD_METADATA_LAYOUT_EXPLICIT);
        compactMetadata = false;
        readRecordVector(input, trainExamples, trainCount, fileSizeBytes, exampleSizeInBytes);
        readRecordVector(input, validateExamples, validateCount, fileSizeBytes, exampleSizeInBytes);
        readRecordVector(input, testExamples, testCount, fileSizeBytes, exampleSizeInBytes);
    }

    const std::streamoff metadataEnd = input.tellg();
    THOR_THROW_IF_FALSE(metadataEnd >= 0);
    THOR_THROW_IF_FALSE(static_cast<uint64_t>(metadataEnd) == metadataOffsetBytes + metadataBytes);
}

uint64_t DatasetShard::compactOffsetFor(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return compactTrainOffsetBytes;
    } else if (exampleType == ExampleType::VALIDATE) {
        return compactValidateOffsetBytes;
    } else if (exampleType == ExampleType::TEST) {
        return compactTestOffsetBytes;
    } else {
        THOR_UNREACHABLE();
    }
}

uint64_t DatasetShard::compactCountFor(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return compactTrainCount;
    } else if (exampleType == ExampleType::VALIDATE) {
        return compactValidateCount;
    } else if (exampleType == ExampleType::TEST) {
        return compactTestCount;
    } else {
        THOR_UNREACHABLE();
    }
}

uint64_t DatasetShard::compactBytesFor(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return compactTrainBytes;
    } else if (exampleType == ExampleType::VALIDATE) {
        return compactValidateBytes;
    } else if (exampleType == ExampleType::TEST) {
        return compactTestBytes;
    } else {
        THOR_UNREACHABLE();
    }
}

std::vector<DatasetShardRecord> &DatasetShard::mutableRecordsFor(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return trainExamples;
    } else if (exampleType == ExampleType::VALIDATE) {
        return validateExamples;
    } else if (exampleType == ExampleType::TEST) {
        return testExamples;
    } else {
        THOR_UNREACHABLE();
    }
}

const std::vector<DatasetShardRecord> &DatasetShard::recordsFor(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return trainExamples;
    } else if (exampleType == ExampleType::VALIDATE) {
        return validateExamples;
    } else if (exampleType == ExampleType::TEST) {
        return testExamples;
    } else {
        THOR_UNREACHABLE();
    }
}
