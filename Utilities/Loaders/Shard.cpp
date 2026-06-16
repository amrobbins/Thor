#include "Utilities/Loaders/Shard.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TarFile/UringDirect.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>

using std::mutex;
using std::string;
using std::thread;

namespace {

constexpr std::array<char, 16> SHARD_MAGIC = {'T', 'H', 'O', 'R', '_', 'R', 'A', 'W', '_', 'S', 'H', 'A', 'R', 'D', '\0', '\0'};
constexpr uint32_t SHARD_FORMAT_VERSION = 2;
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

void writeRecordVector(std::ostream &stream, const std::vector<ShardExampleRecord> &records) {
    for (const ShardExampleRecord &record : records) {
        writeUint64(stream, record.fileOffsetBytes);
        writeString(stream, record.label);
        writeString(stream, record.filename);
    }
}

void readRecordVector(std::istream &stream,
                      std::vector<ShardExampleRecord> &records,
                      uint64_t recordCount,
                      uint64_t fileSizeBytes,
                      uint64_t exampleSizeInBytes) {
    records.clear();
    records.reserve(recordCount);
    for (uint64_t i = 0; i < recordCount; ++i) {
        ShardExampleRecord record;
        record.fileOffsetBytes = readUint64(stream);
        THOR_THROW_IF_FALSE(record.fileOffsetBytes <= fileSizeBytes);
        THOR_THROW_IF_FALSE(exampleSizeInBytes <= fileSizeBytes - record.fileOffsetBytes);
        record.label = readString(stream, fileSizeBytes);
        record.filename = readString(stream, fileSizeBytes);
        records.push_back(std::move(record));
    }
}

}  // namespace

Shard::Shard() {
    exampleSizeInBytes = 0;
    dataType = ThorImplementation::DataType::UINT8;
    open = false;
    metadataFinalized = false;
}

Shard::~Shard() {
    if (shardFile.is_open()) {
        shardFile.close();
    }
}

void Shard::createShard(string filename,
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

void Shard::openShard(string filename) {
    if (shardFile.is_open()) {
        shardFile.close();
    }
    cachedReader.reset();
    trainExamples.clear();
    validateExamples.clear();
    testExamples.clear();
    allClasses.clear();

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
    const uint32_t reserved = readUint32(input);
    THOR_THROW_IF_FALSE(reserved == 0);

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

    readMetadata(metadataOffsetBytes, metadataBytes, trainCount, validateCount, testCount, classCount);

    THOR_THROW_IF_FALSE(trainExamples.size() == trainCount);
    THOR_THROW_IF_FALSE(validateExamples.size() == validateCount);
    THOR_THROW_IF_FALSE(testExamples.size() == testCount);
    THOR_THROW_IF_FALSE(allClasses.size() == classCount);

    metadataFinalized = true;
    open = true;
}

bool Shard::isOpen() { return open; }

void Shard::writeExample(uint8_t *buffer, const string &label, const string &filename, ExampleType exampleType) {
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

    ShardExampleRecord record;
    record.fileOffsetBytes = static_cast<uint64_t>(payloadOffset);
    record.label = label;
    record.filename = filename;
    mutableRecordsFor(exampleType).push_back(std::move(record));
}

ShardExampleReadRequest Shard::getExampleReadRequest(ExampleType exampleType, uint64_t exampleIndex) {
    THOR_THROW_IF_FALSE(isOpen());

    const std::vector<ShardExampleRecord> &records = recordsFor(exampleType);
    THOR_THROW_IF_FALSE(exampleIndex < records.size());

    const ShardExampleRecord &record = records[exampleIndex];
    ShardExampleReadRequest request;
    request.fileOffsetBytes = record.fileOffsetBytes;
    request.numBytes = exampleSizeInBytes;
    request.label = record.label;
    request.filename = record.filename;
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

void Shard::shrinkToFit() {
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

string Shard::getFilename() { return filename; }

uint64_t Shard::getExampleSizeInBytes() { return exampleSizeInBytes; }

ThorImplementation::DataType Shard::getDataType() { return dataType; }

uint64_t Shard::getNumExamples(ExampleType exampleType) { return recordsFor(exampleType).size(); }

const std::vector<std::string> &Shard::getAllClasses() { return allClasses; }

void Shard::writeHeader(uint64_t metadataOffsetBytes, uint64_t metadataBytes) {
    shardFile.seekp(0, std::ios::beg);
    THOR_THROW_IF_FALSE(shardFile.good());

    writeExact(shardFile, SHARD_MAGIC.data(), SHARD_MAGIC.size());
    writeUint32(shardFile, SHARD_FORMAT_VERSION);
    writeUint32(shardFile, SHARD_HEADER_BYTES);
    writeUint64(shardFile, exampleSizeInBytes);
    writeUint32(shardFile, static_cast<uint32_t>(dataType));
    writeUint32(shardFile, 0);
    writeUint64(shardFile, trainExamples.size());
    writeUint64(shardFile, validateExamples.size());
    writeUint64(shardFile, testExamples.size());
    writeUint64(shardFile, allClasses.size());
    writeUint64(shardFile, metadataOffsetBytes);
    writeUint64(shardFile, metadataBytes);

    const std::streamoff headerEnd = shardFile.tellp();
    THOR_THROW_IF_FALSE(headerEnd == static_cast<std::streamoff>(SHARD_HEADER_BYTES));
}

void Shard::writeMetadata() {
    shardFile.seekp(0, std::ios::end);
    THOR_THROW_IF_FALSE(shardFile.good());
    const std::streamoff metadataOffset = shardFile.tellp();
    THOR_THROW_IF_FALSE(metadataOffset >= 0);

    for (const std::string &className : allClasses) {
        writeString(shardFile, className);
    }
    writeRecordVector(shardFile, trainExamples);
    writeRecordVector(shardFile, validateExamples);
    writeRecordVector(shardFile, testExamples);

    const std::streamoff metadataEnd = shardFile.tellp();
    THOR_THROW_IF_FALSE(metadataEnd >= metadataOffset);
    const uint64_t metadataOffsetBytes = static_cast<uint64_t>(metadataOffset);
    const uint64_t metadataBytes = static_cast<uint64_t>(metadataEnd - metadataOffset);

    writeHeader(metadataOffsetBytes, metadataBytes);
}

void Shard::readMetadata(uint64_t metadataOffsetBytes,
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

    readRecordVector(input, trainExamples, trainCount, fileSizeBytes, exampleSizeInBytes);
    readRecordVector(input, validateExamples, validateCount, fileSizeBytes, exampleSizeInBytes);
    readRecordVector(input, testExamples, testCount, fileSizeBytes, exampleSizeInBytes);

    const std::streamoff metadataEnd = input.tellg();
    THOR_THROW_IF_FALSE(metadataEnd >= 0);
    THOR_THROW_IF_FALSE(static_cast<uint64_t>(metadataEnd) == metadataOffsetBytes + metadataBytes);
}

std::vector<ShardExampleRecord> &Shard::mutableRecordsFor(ExampleType exampleType) {
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

const std::vector<ShardExampleRecord> &Shard::recordsFor(ExampleType exampleType) const {
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
