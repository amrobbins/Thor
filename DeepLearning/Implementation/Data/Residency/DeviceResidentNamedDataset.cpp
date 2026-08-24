#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedDataset.h"

#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Data/Storage/DatasetShard.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;
using json = nlohmann::json;

namespace {

uint64_t checkedAdd(uint64_t left, uint64_t right, const char *context) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::runtime_error(std::string(context) + " overflow while adding.");
    }
    return left + right;
}

uint64_t checkedMul(uint64_t left, uint64_t right, const char *context) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(std::string(context) + " overflow while multiplying.");
    }
    return left * right;
}

uint64_t scalarBytes(DataType dataType) {
    switch (dataType) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FP64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        default:
            break;
    }
    throw std::runtime_error("Unsupported compact resident scalar data type.");
}

std::vector<uint8_t> parseHexBytes(const std::string &hex, uint64_t expectedBytes) {
    if (hex.size() != expectedBytes * 2) {
        throw std::runtime_error("Compact resident window key has unexpected hex width.");
    }
    auto nibble = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
        if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
        throw std::runtime_error("Compact resident window key contains non-hex data.");
    };
    std::vector<uint8_t> bytes(expectedBytes);
    for (uint64_t i = 0; i < expectedBytes; ++i) {
        bytes.at(static_cast<size_t>(i)) =
            static_cast<uint8_t>((nibble(hex.at(static_cast<size_t>(2 * i))) << 4) |
                                 nibble(hex.at(static_cast<size_t>(2 * i + 1))));
    }
    return bytes;
}

uint64_t keyBitsFromHex(const std::string &hex, DataType keyDataType) {
    const uint64_t keyBytes = scalarBytes(keyDataType);
    if (keyBytes > sizeof(uint64_t)) {
        throw std::runtime_error("Compact resident window keys wider than 64 bits are unsupported.");
    }
    const std::vector<uint8_t> bytes = parseHexBytes(hex, keyBytes);
    uint64_t bits = 0;
    for (uint64_t i = 0; i < keyBytes; ++i) {
        bits |= static_cast<uint64_t>(bytes.at(static_cast<size_t>(i))) << (8 * i);
    }
    return bits;
}


uint64_t canonicalOffsetHost(const Tensor &offsets, uint64_t index) {
    THOR_THROW_IF_FALSE(offsets.isInitialized());
    THOR_THROW_IF_FALSE(
        offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    THOR_THROW_IF_FALSE(offsets.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(index < offsets.getDimensions().front());
    if (offsets.getDataType() == DataType::UINT32) {
        return offsets.getMemPtr<uint32_t>()[index];
    }
    if (offsets.getDataType() == DataType::UINT64) {
        return offsets.getMemPtr<uint64_t>()[index];
    }
    throw std::runtime_error(
        "Materialized ragged snapshot offsets must use canonical UINT32 or UINT64 dtype.");
}

void writeLittleEndianUint64(uint8_t *destination, uint64_t value) {
    THOR_THROW_IF_FALSE(destination != nullptr);
    for (uint64_t i = 0; i < sizeof(uint64_t); ++i) {
        destination[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xffu);
    }
}

uint64_t readLittleEndianUnsignedHost(const uint8_t *bytes, uint64_t numBytes) {
    if (bytes == nullptr || numBytes == 0 || numBytes > sizeof(uint64_t)) {
        throw std::runtime_error("Compact resident scalar byte width is invalid.");
    }
    uint64_t value = 0;
    for (uint64_t i = 0; i < numBytes; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

template <typename T>
T readScalarHost(const uint8_t *bytes) {
    T value{};
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

int64_t readIndexHost(const uint8_t *bytes, DataType dataType) {
    switch (dataType) {
        case DataType::INT8:
            return readScalarHost<int8_t>(bytes);
        case DataType::INT16:
            return readScalarHost<int16_t>(bytes);
        case DataType::INT32:
            return readScalarHost<int32_t>(bytes);
        case DataType::INT64:
            return readScalarHost<int64_t>(bytes);
        case DataType::UINT8:
            return readScalarHost<uint8_t>(bytes);
        case DataType::UINT16:
            return readScalarHost<uint16_t>(bytes);
        case DataType::UINT32:
            return readScalarHost<uint32_t>(bytes);
        case DataType::UINT64: {
            const uint64_t value = readScalarHost<uint64_t>(bytes);
            if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error(
                    "Compact resident window start is outside int64 range.");
            }
            return static_cast<int64_t>(value);
        }
        default:
            break;
    }
    throw std::runtime_error("Compact resident window index dtype must be integer.");
}

void validateWindowEnd(int64_t start, uint64_t length, const std::string &fieldName) {
    if (length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        start > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(length)) {
        throw std::runtime_error(
            "Compact resident window range overflows int64 for field '" + fieldName + "'.");
    }
}

int64_t checkedAffineStart(int64_t base,
                           int64_t stride,
                           int64_t fieldOffset,
                           uint64_t row,
                           const std::string &fieldName) {
    const __int128 value = static_cast<__int128>(base) +
                           static_cast<__int128>(row) * static_cast<__int128>(stride) +
                           static_cast<__int128>(fieldOffset);
    if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(
            "Compact resident affine reference overflows int64 for field '" + fieldName + "'.");
    }
    return static_cast<int64_t>(value);
}

std::set<uint64_t> sourceKeyBits(
    const DatasetLayout::WindowedTensorSourceSpec &source) {
    std::set<uint64_t> keys;
    for (const DatasetLayout::WindowedTensorSourceSequence &sequence :
         source.sourceSequences) {
        keys.insert(keyBitsFromHex(sequence.keyHex, source.keyDataType));
    }
    return keys;
}

json readDatasetManifest(const std::filesystem::path &datasetPath) {
    const std::filesystem::path manifestPath = datasetPath / DatasetWriter::MANIFEST_FILENAME;
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open compact resident dataset manifest: " +
                                 manifestPath.string());
    }
    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("Failed while reading compact resident dataset manifest: " +
                                 manifestPath.string());
    }
    return manifest;
}

enum class RequestedCompactFieldKind {
    DIRECT,
    WINDOW,
    RAGGED
};

struct RequestedCompactField {
    RequestedCompactFieldKind kind = RequestedCompactFieldKind::DIRECT;
    const DatasetLayout::TensorSpec *directSpec = nullptr;
    const DatasetLayout::WindowedTensorSpec *windowSpec = nullptr;
    const DatasetLayout::RaggedTensorSpec *raggedSpec = nullptr;
    bool mask = false;
};

std::map<std::string, RequestedCompactField> resolveRequestedCompactFields(
    const Thor::DatasetMaterializationDescription &description,
    const std::set<std::string> &fieldNamesToExpose) {
    if (description.source != Thor::DatasetMaterializationSource::FILE_DATASET) {
        throw std::runtime_error(
            "Compact device residency currently requires a file dataset.");
    }
    if (!description.layout.hasWindowedTensors() && !description.layout.hasRaggedTensors()) {
        throw std::runtime_error(
            "Compact device residency requires at least one windowed or ragged field.");
    }

    std::map<std::string, RequestedCompactField> available;
    for (const DatasetLayout::TensorSpec &spec : description.layout.tensors()) {
        available.emplace(
            spec.name,
            RequestedCompactField{
                .kind = RequestedCompactFieldKind::DIRECT,
                .directSpec = &spec});
    }
    for (const DatasetLayout::RaggedTensorSpec &spec : description.layout.raggedTensors()) {
        available.emplace(
            spec.name,
            RequestedCompactField{
                .kind = RequestedCompactFieldKind::RAGGED,
                .raggedSpec = &spec});
    }
    for (const DatasetLayout::WindowedTensorSpec &spec :
         description.layout.windowedTensors()) {
        available.emplace(
            spec.name,
            RequestedCompactField{
                .kind = RequestedCompactFieldKind::WINDOW,
                .windowSpec = &spec,
                .mask = false});
        if (spec.maskName.has_value()) {
            available.emplace(
                spec.maskName.value(),
                RequestedCompactField{
                    .kind = RequestedCompactFieldKind::WINDOW,
                    .windowSpec = &spec,
                    .mask = true});
        }
    }

    std::map<std::string, RequestedCompactField> requested;
    if (fieldNamesToExpose.empty()) {
        requested = available;
    } else {
        for (const std::string &name : fieldNamesToExpose) {
            const auto found = available.find(name);
            if (found == available.end()) {
                throw std::runtime_error(
                    "Compact device residency requested unknown field '" + name + "'.");
            }
            requested.emplace(name, found->second);
        }
    }
    if (requested.empty()) {
        throw std::runtime_error("Compact device residency requested no fields.");
    }
    bool hasVariableField = false;
    for (const auto &entry : requested) {
        hasVariableField = hasVariableField ||
                           entry.second.kind == RequestedCompactFieldKind::WINDOW ||
                           entry.second.kind == RequestedCompactFieldKind::RAGGED;
    }
    if (!hasVariableField) {
        throw std::runtime_error(
            "Compact device residency must expose at least one windowed or ragged field.");
    }
    return requested;
}

uint64_t affineSegmentCount(const json &manifest) {
    if (!manifest.contains("affine_window_reference_segments")) {
        return 0;
    }
    const json &segments = manifest.at("affine_window_reference_segments");
    if (!segments.is_array()) {
        throw std::runtime_error("Compact resident affine metadata must be an array.");
    }
    return static_cast<uint64_t>(segments.size());
}

uint64_t compactStorageBytes(
    const Thor::DatasetMaterializationDescription &description,
    const std::map<std::string, RequestedCompactField> &requested,
    const json &manifest) {
    bool needsRecords = false;
    uint64_t bytes = 0;
    std::set<std::string> sourceNames;
    std::set<std::string> sourcesNeedingSequenceMetadata;
    std::set<std::string> affineFieldNames;
    for (const auto &entry : requested) {
        if (entry.second.kind == RequestedCompactFieldKind::DIRECT) {
            needsRecords = true;
            continue;
        }
        if (entry.second.kind == RequestedCompactFieldKind::RAGGED) {
            needsRecords = true;
            const DatasetLayout::RaggedTensorSpec &spec = *entry.second.raggedSpec;
            if (spec.isWindowedSourceBacked()) {
                THOR_THROW_IF_FALSE(spec.sourceName.has_value());
                sourceNames.insert(spec.sourceName.value());
            } else {
                bytes = checkedAdd(
                    bytes,
                    spec.valuesNumBytes,
                    "Compact resident ragged values storage");
            }
            continue;
        }
        const DatasetLayout::WindowedTensorSpec &spec = *entry.second.windowSpec;
        sourceNames.insert(spec.sourceName);
        sourcesNeedingSequenceMetadata.insert(spec.sourceName);
        if (spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::INDEXED) {
            needsRecords = true;
        } else {
            affineFieldNames.insert(spec.name);
        }
    }

    if (needsRecords) {
        bytes = checkedAdd(
            bytes,
            checkedMul(description.numExamples,
                       description.layout.recordSizeBytes(),
                       "Compact resident record storage"),
            "Compact resident storage");
    }
    for (const std::string &sourceName : sourceNames) {
        const DatasetLayout::WindowedTensorSourceSpec &source =
            description.layout.windowedTensorSource(sourceName);
        bytes = checkedAdd(bytes, source.sourceNumBytes, "Compact resident source storage");
        if (sourcesNeedingSequenceMetadata.find(sourceName) !=
            sourcesNeedingSequenceMetadata.end()) {
            bytes = checkedAdd(
                bytes,
                checkedMul(static_cast<uint64_t>(source.sourceSequences.size()),
                           static_cast<uint64_t>(sizeof(DeviceResidentWindowSourceSequence)),
                           "Compact resident source metadata"),
                "Compact resident storage");
        }
    }
    const uint64_t segmentCount = affineSegmentCount(manifest);
    for (const std::string &fieldName : affineFieldNames) {
        (void)fieldName;
        bytes = checkedAdd(
            bytes,
            checkedMul(segmentCount,
                       static_cast<uint64_t>(sizeof(DeviceResidentAffineWindowSegment)),
                       "Compact resident affine metadata"),
            "Compact resident storage");
    }
    return bytes;
}

Tensor cpuByteTensor(uint64_t numBytes) {
    if (numBytes == 0) {
        return Tensor();
    }
    return Tensor(
        TensorPlacement(TensorPlacement::MemDevices::CPU),
        TensorDescriptor(DataType::UINT8, {numBytes}));
}

Tensor uploadTensor(const Tensor &host, TensorPlacement placement, Stream &stream) {
    THOR_THROW_IF_FALSE(host.isInitialized());
    Tensor device(placement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    return device;
}

Tensor readCompactRecords(
    const Thor::DatasetMaterializationDescription &description,
    const json &manifest) {
    const uint64_t recordSize = description.layout.recordSizeBytes();
    const uint64_t totalBytes = checkedMul(
        description.numExamples, recordSize, "Compact resident record bytes");
    if (totalBytes == 0) {
        return Tensor();
    }
    Tensor records = cpuByteTensor(totalBytes);
    uint8_t *destination = static_cast<uint8_t *>(records.getMemPtr());

    if (!manifest.contains("shards") || !manifest.at("shards").is_array()) {
        throw std::runtime_error("Compact resident dataset manifest is missing shards.");
    }
    uint64_t expectedGlobalStart = 0;
    for (const json &shardJson : manifest.at("shards")) {
        const uint64_t globalStart = shardJson.at("global_start").get<uint64_t>();
        const uint64_t numExamples = shardJson.at("num_examples").get<uint64_t>();
        if (globalStart != expectedGlobalStart) {
            throw std::runtime_error("Compact resident shard coverage is not contiguous.");
        }
        DatasetShard shard;
        shard.openShard((description.datasetPath /
                         shardJson.at("file").get<std::string>()).string());
        if (shard.getExampleSizeInBytes() != recordSize ||
            shard.getNumExamples(ExampleType::TRAIN) != numExamples) {
            throw std::runtime_error("Compact resident shard metadata does not match the dataset layout.");
        }
        for (uint64_t localRow = 0; localRow < numExamples; ++localRow) {
            std::string label;
            std::string filename;
            shard.loadExample(
                destination + (globalStart + localRow) * recordSize,
                label,
                filename,
                ExampleType::TRAIN,
                localRow);
        }
        expectedGlobalStart = checkedAdd(expectedGlobalStart, numExamples,
                                         "Compact resident shard coverage");
    }
    if (expectedGlobalStart != description.numExamples) {
        throw std::runtime_error("Compact resident shard coverage does not match num_examples.");
    }
    return records;
}

void validateIndexedReferences(
    const Thor::DatasetMaterializationDescription &description,
    const std::map<std::string, RequestedCompactField> &requested,
    const Tensor &records) {
    THOR_THROW_IF_FALSE(records.isInitialized());
    const uint8_t *recordBytes = records.getMemPtr<uint8_t>();
    const uint64_t recordSize = description.layout.recordSizeBytes();
    std::set<std::string> validatedFields;
    for (const auto &entry : requested) {
        if (entry.second.kind != RequestedCompactFieldKind::WINDOW) {
            continue;
        }
        const DatasetLayout::WindowedTensorSpec &spec = *entry.second.windowSpec;
        if (spec.referenceMode != DatasetLayout::WindowedTensorReferenceMode::INDEXED ||
            !validatedFields.insert(spec.name).second) {
            continue;
        }
        const DatasetLayout::WindowedTensorSourceSpec &source =
            description.layout.windowedTensorSource(spec.sourceName);
        const std::set<uint64_t> keys = sourceKeyBits(source);
        for (uint64_t row = 0; row < description.numExamples; ++row) {
            const uint8_t *reference =
                recordBytes + row * recordSize + spec.referenceOffsetBytes;
            const uint64_t keyBits =
                readLittleEndianUnsignedHost(reference, spec.keyNumBytes());
            if (keys.find(keyBits) == keys.end()) {
                throw std::runtime_error(
                    "Compact resident indexed reference for field '" + spec.name +
                    "' row " + std::to_string(row) +
                    " has no matching source sequence.");
            }
            const int64_t start =
                readIndexHost(reference + spec.keyNumBytes(), spec.indexDataType);
            validateWindowEnd(start, spec.windowLength(), spec.name);
        }
    }
}

Tensor readRaggedValuesBytes(
    const Thor::DatasetMaterializationDescription &description,
    const DatasetLayout::RaggedTensorSpec &spec) {
    if (!spec.valuesFilename.has_value() || spec.valuesFilename->empty()) {
        throw std::runtime_error(
            "Compact resident ragged field '" + spec.name + "' has no values filename.");
    }
    const std::filesystem::path path =
        description.datasetPath / spec.valuesFilename.value();
    if (!std::filesystem::exists(path) ||
        std::filesystem::file_size(path) != spec.valuesNumBytes) {
        throw std::runtime_error(
            "Compact resident ragged values size does not match its manifest for field '" +
            spec.name + "'.");
    }
    if (spec.valuesNumBytes == 0) {
        return Tensor();
    }
    Tensor bytes = cpuByteTensor(spec.valuesNumBytes);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            "Failed to open compact resident ragged values for field '" + spec.name + "'.");
    }
    in.read(static_cast<char *>(bytes.getMemPtr()),
            static_cast<std::streamsize>(spec.valuesNumBytes));
    if (in.gcount() != static_cast<std::streamsize>(spec.valuesNumBytes)) {
        throw std::runtime_error(
            "Compact resident ragged values were shorter than the manifest for field '" +
            spec.name + "'.");
    }
    return bytes;
}

std::map<std::string, std::vector<uint64_t>> readAndValidateRaggedValueCounts(
    const Thor::DatasetMaterializationDescription &description,
    const std::map<std::string, RequestedCompactField> &requested,
    const Tensor &records) {
    THOR_THROW_IF_FALSE(records.isInitialized());
    std::map<std::string, std::vector<uint64_t>> countsByField;
    const uint8_t *recordBytes = records.getMemPtr<uint8_t>();
    const uint64_t recordSize = description.layout.recordSizeBytes();
    for (const auto &entry : requested) {
        if (entry.second.kind != RequestedCompactFieldKind::RAGGED) continue;
        const DatasetLayout::RaggedTensorSpec &spec = *entry.second.raggedSpec;
        const uint64_t storedValueCount = spec.storedValueCount();
        if (spec.referenceNumBytes != 2 * sizeof(uint64_t) ||
            recordSize < spec.referenceNumBytes ||
            spec.referenceOffsetBytes > recordSize - spec.referenceNumBytes) {
            throw std::runtime_error(
                "Compact resident ragged reference layout is invalid for field '" +
                spec.name + "'.");
        }
        std::vector<uint64_t> counts(description.numExamples);
        for (uint64_t row = 0; row < description.numExamples; ++row) {
            const uint8_t *reference =
                recordBytes + row * recordSize + spec.referenceOffsetBytes;
            uint64_t startValue = 0;
            uint64_t valueCount = 0;
            std::memcpy(&startValue, reference, sizeof(uint64_t));
            std::memcpy(&valueCount, reference + sizeof(uint64_t), sizeof(uint64_t));
            if (startValue > storedValueCount ||
                valueCount > storedValueCount - startValue) {
                throw std::runtime_error(
                    "Compact resident ragged reference for field '" + spec.name +
                    "' row " + std::to_string(row) + " is outside its values sidecar.");
            }
            counts[row] = valueCount;
        }
        countsByField.emplace(spec.name, std::move(counts));
    }
    return countsByField;
}

Tensor readSourceBytes(const Thor::DatasetMaterializationDescription &description,
                       const DatasetLayout::WindowedTensorSourceSpec &source) {
    if (!source.sourceFilename.has_value() || source.sourceFilename->empty()) {
        throw std::runtime_error("Compact resident window source has no storage filename.");
    }
    Tensor bytes = cpuByteTensor(source.sourceNumBytes);
    if (!bytes.isInitialized()) {
        throw std::runtime_error("Compact resident window source is empty.");
    }
    const std::filesystem::path path = description.datasetPath / source.sourceFilename.value();
    if (!std::filesystem::exists(path) ||
        std::filesystem::file_size(path) != source.sourceNumBytes) {
        throw std::runtime_error(
            "Compact resident window source size does not match its manifest: " +
            path.string());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open compact resident window source: " + path.string());
    }
    in.read(static_cast<char *>(bytes.getMemPtr()),
            static_cast<std::streamsize>(source.sourceNumBytes));
    if (in.gcount() != static_cast<std::streamsize>(source.sourceNumBytes)) {
        throw std::runtime_error("Compact resident window source was shorter than its manifest size.");
    }
    return bytes;
}

Tensor sourceSequenceTensor(const DatasetLayout::WindowedTensorSourceSpec &source) {
    std::vector<DeviceResidentWindowSourceSequence> sequences;
    sequences.reserve(source.sourceSequences.size());
    for (const DatasetLayout::WindowedTensorSourceSequence &sequence :
         source.sourceSequences) {
        sequences.push_back(DeviceResidentWindowSourceSequence{
            .keyBits = keyBitsFromHex(sequence.keyHex, source.keyDataType),
            .startIndex = sequence.startIndex,
            .endIndexExclusive = sequence.endIndexExclusive,
            .offsetBytes = sequence.offsetBytes});
    }
    std::sort(sequences.begin(), sequences.end(),
              [](const auto &left, const auto &right) {
                  return left.keyBits < right.keyBits;
              });
    for (size_t i = 1; i < sequences.size(); ++i) {
        if (sequences[i - 1].keyBits == sequences[i].keyBits) {
            throw std::runtime_error("Compact resident window source contains duplicate keys.");
        }
    }
    const uint64_t numBytes = checkedMul(
        static_cast<uint64_t>(sequences.size()),
        static_cast<uint64_t>(sizeof(DeviceResidentWindowSourceSequence)),
        "Compact resident source metadata bytes");
    Tensor tensor = cpuByteTensor(numBytes);
    std::memcpy(tensor.getMemPtr(), sequences.data(), static_cast<size_t>(numBytes));
    return tensor;
}

Tensor affineSegmentTensor(
    const Thor::DatasetMaterializationDescription &description,
    const json &manifest,
    const DatasetLayout::WindowedTensorSpec &spec,
    uint64_t &segmentCountOut) {
    if (!manifest.contains("affine_window_reference_segments") ||
        !manifest.at("affine_window_reference_segments").is_array()) {
        throw std::runtime_error("Compact resident affine dataset is missing reference segments.");
    }
    const DatasetLayout::WindowedTensorSourceSpec &source =
        description.layout.windowedTensorSource(spec.sourceName);
    const std::set<uint64_t> sourceKeys = sourceKeyBits(source);
    std::vector<DeviceResidentAffineWindowSegment> segments;
    uint64_t expectedRowStart = 0;
    for (const json &segmentJson : manifest.at("affine_window_reference_segments")) {
        const uint64_t rowStart = segmentJson.at("row_start").get<uint64_t>();
        const uint64_t count = segmentJson.at("count").get<uint64_t>();
        if (rowStart != expectedRowStart || count == 0) {
            throw std::runtime_error("Compact resident affine segments must be contiguous and non-empty.");
        }
        const json &references = segmentJson.at("references");
        if (!references.contains(spec.name)) {
            throw std::runtime_error("Compact resident affine segment is missing field '" +
                                     spec.name + "'.");
        }
        const json &reference = references.at(spec.name);
        const uint64_t keyBits = keyBitsFromHex(
            reference.at("key_hex").get<std::string>(), source.keyDataType);
        if (sourceKeys.find(keyBits) == sourceKeys.end()) {
            throw std::runtime_error(
                "Compact resident affine reference for field '" + spec.name +
                "' has no matching source sequence.");
        }
        segments.push_back(DeviceResidentAffineWindowSegment{
            .rowStart = rowStart,
            .count = count,
            .keyBits = keyBits,
            .base = reference.at("base").get<int64_t>(),
            .stride = reference.at("stride").get<int64_t>(),
            .fieldOffset = reference.at("field_offset").get<int64_t>()});
        if (segments.back().stride <= 0) {
            throw std::runtime_error("Compact resident affine stride must be positive.");
        }
        const int64_t firstStart = checkedAffineStart(
            segments.back().base,
            segments.back().stride,
            segments.back().fieldOffset,
            0,
            spec.name);
        const int64_t lastStart = checkedAffineStart(
            segments.back().base,
            segments.back().stride,
            segments.back().fieldOffset,
            count - 1,
            spec.name);
        validateWindowEnd(firstStart, spec.windowLength(), spec.name);
        validateWindowEnd(lastStart, spec.windowLength(), spec.name);
        expectedRowStart = checkedAdd(expectedRowStart, count,
                                      "Compact resident affine coverage");
    }
    if (expectedRowStart != description.numExamples) {
        throw std::runtime_error("Compact resident affine segments do not cover dataset rows.");
    }
    segmentCountOut = static_cast<uint64_t>(segments.size());
    const uint64_t numBytes = checkedMul(
        segmentCountOut,
        static_cast<uint64_t>(sizeof(DeviceResidentAffineWindowSegment)),
        "Compact resident affine metadata bytes");
    Tensor tensor = cpuByteTensor(numBytes);
    std::memcpy(tensor.getMemPtr(), segments.data(), static_cast<size_t>(numBytes));
    return tensor;
}

}  // namespace

std::shared_ptr<DeviceResidentNamedDataset> DeviceResidentNamedDataset::fromSnapshot(
    const MaterializedNamedDatasetSnapshot &snapshot,
    TensorPlacement devicePlacement) {
    THOR_THROW_IF_FALSE(devicePlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (snapshot.numExamples == 0) {
        throw std::runtime_error("DeviceResidentNamedDataset requires at least one example.");
    }

    auto dataset = std::shared_ptr<DeviceResidentNamedDataset>(
        new DeviceResidentNamedDataset(
            snapshot.datasetId,
            snapshot.schema,
            snapshot.layout,
            snapshot.numExamples,
            devicePlacement));

    const auto start = std::chrono::steady_clock::now();
    Stream uploadStream(devicePlacement);
    std::vector<Tensor> hostRaggedReferenceBuffers;
    hostRaggedReferenceBuffers.reserve(snapshot.raggedFields.size());
    for (const auto &entry : snapshot.fields) {
        const Tensor &hostTensor = entry.second;
        Tensor deviceTensor(devicePlacement, hostTensor.getDescriptor());
        deviceTensor.copyFromAsync(hostTensor, uploadStream);
        dataset->fields.emplace(entry.first, std::move(deviceTensor));
    }

    for (const auto &[fieldId, source] : snapshot.raggedFields) {
        const Thor::DatasetField &field = snapshot.schema.getField(fieldId);
        if (field.kind != Thor::DatasetFieldKind::RAGGED) {
            throw std::runtime_error(
                "Materialized ragged snapshot field is not marked RAGGED: " + field.name);
        }
        const DatasetLayout::RaggedTensorSpec &spec = snapshot.layout.raggedTensor(field.name);
        if (source.valuesDataType != field.dataType ||
            source.trailingDimensions != field.dimensions ||
            source.valueBytes != spec.valueNumBytes()) {
            throw std::runtime_error(
                "Materialized ragged snapshot contract mismatch for field '" + field.name + "'.");
        }
        if (!source.offsets.isInitialized() ||
            source.offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU ||
            source.offsets.getDimensions() != std::vector<uint64_t>{snapshot.numExamples + 1}) {
            throw std::runtime_error(
                "Materialized ragged snapshot has invalid offsets for field '" + field.name + "'.");
        }
        if (source.storedValueCount != 0) {
            std::vector<uint64_t> expectedValueDimensions;
            expectedValueDimensions.reserve(field.dimensions.size() + 1);
            expectedValueDimensions.push_back(source.storedValueCount);
            expectedValueDimensions.insert(
                expectedValueDimensions.end(), field.dimensions.begin(), field.dimensions.end());
            if (!source.values.isInitialized() ||
                source.values.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU ||
                source.values.getDescriptor() != TensorDescriptor(field.dataType, expectedValueDimensions)) {
                throw std::runtime_error(
                    "Materialized ragged snapshot has invalid packed values for field '" + field.name + "'.");
            }
        } else if (source.values.isInitialized()) {
            throw std::runtime_error(
                "All-empty materialized ragged snapshot must not allocate packed values for field '" +
                field.name + "'.");
        }

        constexpr uint64_t referenceBytes = 2 * sizeof(uint64_t);
        const uint64_t referenceStorageBytes = checkedMul(
            snapshot.numExamples,
            referenceBytes,
            "Materialized ragged snapshot reference storage");
        Tensor hostReferences(
            TensorPlacement(TensorPlacement::MemDevices::CPU),
            TensorDescriptor(DataType::UINT8, {referenceStorageBytes}));
        std::vector<uint64_t> valueCounts(snapshot.numExamples, 0);
        uint8_t *referenceBytesPtr = hostReferences.getMemPtr<uint8_t>();
        uint64_t previous = canonicalOffsetHost(source.offsets, 0);
        if (previous != 0) {
            throw std::runtime_error(
                "Materialized ragged snapshot offsets must start at zero for field '" +
                field.name + "'.");
        }
        for (uint64_t row = 0; row < snapshot.numExamples; ++row) {
            const uint64_t next = canonicalOffsetHost(source.offsets, row + 1);
            if (next < previous || next > source.storedValueCount) {
                throw std::runtime_error(
                    "Materialized ragged snapshot offsets are invalid for field '" +
                    field.name + "'.");
            }
            valueCounts[row] = next - previous;
            uint8_t *reference = referenceBytesPtr + row * referenceBytes;
            writeLittleEndianUint64(reference, previous);
            writeLittleEndianUint64(reference + sizeof(uint64_t), next - previous);
            previous = next;
        }
        if (previous != source.storedValueCount) {
            throw std::runtime_error(
                "Materialized ragged snapshot final offset does not match packed values for field '" +
                field.name + "'.");
        }

        Tensor deviceReferences(devicePlacement, hostReferences.getDescriptor());
        deviceReferences.copyFromAsync(hostReferences, uploadStream);
        hostRaggedReferenceBuffers.push_back(hostReferences);
        Tensor deviceValues;
        if (source.storedValueCount != 0) {
            deviceValues = Tensor(devicePlacement, source.values.getDescriptor());
            deviceValues.copyFromAsync(source.values, uploadStream);
        }
        dataset->snapshotRaggedFields.emplace(
            field.name,
            SnapshotRaggedFieldStorage{
                .spec = spec,
                .references = std::move(deviceReferences),
                .values = std::move(deviceValues),
                .valueCounts = std::move(valueCounts),
                .storedValueCount = source.storedValueCount,
                .valueBytes = source.valueBytes});
    }

    uploadStream.synchronize();
    dataset->uploadSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return dataset;
}

uint64_t DeviceResidentNamedDataset::estimateCompactFileDatasetBytes(
    const Thor::DatasetMaterializationDescription &description,
    const std::set<std::string> &fieldNamesToExpose) {
    const auto requested = resolveRequestedCompactFields(description, fieldNamesToExpose);
    return compactStorageBytes(description, requested, readDatasetManifest(description.datasetPath));
}

std::shared_ptr<DeviceResidentNamedDataset>
DeviceResidentNamedDataset::fromCompactFileDataset(
    const Thor::DatasetMaterializationDescription &description,
    TensorPlacement devicePlacement,
    const std::set<std::string> &fieldNamesToExpose) {
    THOR_THROW_IF_FALSE(devicePlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (description.numExamples == 0) {
        throw std::runtime_error("Compact device residency requires at least one example.");
    }
    const auto requested = resolveRequestedCompactFields(description, fieldNamesToExpose);
    const json manifest = readDatasetManifest(description.datasetPath);

    auto dataset = std::shared_ptr<DeviceResidentNamedDataset>(
        new DeviceResidentNamedDataset(
            description.datasetId,
            description.schema,
            description.layout,
            description.numExamples,
            devicePlacement));
    dataset->compactFileStorage = true;

    bool needsRecords = false;
    std::set<std::string> sourceNames;
    std::set<std::string> sourcesNeedingSequenceMetadata;
    std::set<std::string> affineFieldNames;
    std::map<std::string, RequestedCompactField> requestedWindows;
    std::map<std::string, RequestedCompactField> requestedRagged;
    for (const auto &entry : requested) {
        dataset->compactFieldIds.insert(description.schema.getField(entry.first).id);
        if (entry.second.kind == RequestedCompactFieldKind::DIRECT) {
            needsRecords = true;
            dataset->compactDirectFields.emplace(
                entry.first,
                CompactDirectFieldStorage{*entry.second.directSpec});
            continue;
        }
        if (entry.second.kind == RequestedCompactFieldKind::RAGGED) {
            needsRecords = true;
            requestedRagged.emplace(entry.first, entry.second);
            const DatasetLayout::RaggedTensorSpec &spec = *entry.second.raggedSpec;
            if (spec.isWindowedSourceBacked()) {
                THOR_THROW_IF_FALSE(spec.sourceName.has_value());
                sourceNames.insert(spec.sourceName.value());
            }
            continue;
        }
        const DatasetLayout::WindowedTensorSpec &spec = *entry.second.windowSpec;
        requestedWindows.emplace(entry.first, entry.second);
        sourceNames.insert(spec.sourceName);
        sourcesNeedingSequenceMetadata.insert(spec.sourceName);
        needsRecords = needsRecords ||
                       spec.referenceMode ==
                           DatasetLayout::WindowedTensorReferenceMode::INDEXED;
        if (spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::AFFINE) {
            affineFieldNames.insert(spec.name);
        }
        dataset->compactWindowFields.emplace(
            entry.first,
            CompactWindowFieldStorage{spec, entry.second.mask});
    }

    const auto start = std::chrono::steady_clock::now();
    Stream uploadStream(devicePlacement);
    // cudaMemcpyAsync requires every host source allocation to remain alive until
    // the upload stream reaches the corresponding copy. Keep these compact host
    // tensors retained until the single synchronization below.
    std::vector<Tensor> uploadSources;
    std::map<std::string, std::vector<uint64_t>> raggedValueCounts;
    if (needsRecords) {
        Tensor recordsHost = readCompactRecords(description, manifest);
        validateIndexedReferences(description, requestedWindows, recordsHost);
        raggedValueCounts =
            readAndValidateRaggedValueCounts(description, requestedRagged, recordsHost);
        dataset->compactRecords = uploadTensor(recordsHost, devicePlacement, uploadStream);
        uploadSources.push_back(std::move(recordsHost));
    }

    for (const auto &entry : requestedRagged) {
        const DatasetLayout::RaggedTensorSpec &spec = *entry.second.raggedSpec;
        CompactRaggedFieldStorage storage;
        storage.spec = spec;
        storage.valueCounts = raggedValueCounts.at(entry.first);
        if (!spec.isWindowedSourceBacked()) {
            Tensor valuesHost = readRaggedValuesBytes(description, spec);
            if (valuesHost.isInitialized()) {
                storage.values = uploadTensor(valuesHost, devicePlacement, uploadStream);
                uploadSources.push_back(std::move(valuesHost));
            }
        }
        dataset->compactRaggedFields.emplace(entry.first, std::move(storage));
    }

    for (const std::string &sourceName : sourceNames) {
        const DatasetLayout::WindowedTensorSourceSpec &source =
            description.layout.windowedTensorSource(sourceName);
        Tensor sourceHost = readSourceBytes(description, source);
        CompactWindowSourceStorage storage;
        storage.spec = source;
        storage.bytes = uploadTensor(sourceHost, devicePlacement, uploadStream);
        if (sourcesNeedingSequenceMetadata.find(sourceName) !=
            sourcesNeedingSequenceMetadata.end()) {
            Tensor sequencesHost = sourceSequenceTensor(source);
            storage.sequences = uploadTensor(sequencesHost, devicePlacement, uploadStream);
            storage.sequenceCount = static_cast<uint64_t>(source.sourceSequences.size());
            uploadSources.push_back(std::move(sequencesHost));
        }
        dataset->compactSources.emplace(sourceName, std::move(storage));
        uploadSources.push_back(std::move(sourceHost));
    }

    for (const std::string &fieldName : affineFieldNames) {
        const DatasetLayout::WindowedTensorSpec &spec =
            description.layout.windowedTensor(fieldName);
        uint64_t segmentCount = 0;
        Tensor segmentsHost = affineSegmentTensor(
            description, manifest, spec, segmentCount);
        CompactAffineFieldStorage storage;
        storage.segments = uploadTensor(segmentsHost, devicePlacement, uploadStream);
        storage.segmentCount = segmentCount;
        dataset->compactAffineFields.emplace(fieldName, std::move(storage));
        uploadSources.push_back(std::move(segmentsHost));
    }

    uploadStream.synchronize();
    dataset->uploadSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    const uint64_t expectedBytes = compactStorageBytes(description, requested, manifest);
    if (dataset->totalBytes() != expectedBytes) {
        throw std::runtime_error(
            "Compact device residency allocation size did not match its estimate.");
    }
    return dataset;
}

uint64_t DeviceResidentNamedDataset::totalBytes() const {
    uint64_t bytes = 0;
    for (const auto &entry : fields) {
        bytes += entry.second.getArraySizeInBytes();
    }
    if (compactRecords.isInitialized()) {
        bytes += compactRecords.getArraySizeInBytes();
    }
    for (const auto &entry : compactRaggedFields) {
        if (entry.second.values.isInitialized()) {
            bytes += entry.second.values.getArraySizeInBytes();
        }
    }
    for (const auto &entry : snapshotRaggedFields) {
        bytes += entry.second.references.getArraySizeInBytes();
        if (entry.second.values.isInitialized()) {
            bytes += entry.second.values.getArraySizeInBytes();
        }
    }
    for (const auto &entry : compactSources) {
        bytes += entry.second.bytes.getArraySizeInBytes();
        if (entry.second.sequences.isInitialized()) {
            bytes += entry.second.sequences.getArraySizeInBytes();
        }
    }
    for (const auto &entry : compactAffineFields) {
        bytes += entry.second.segments.getArraySizeInBytes();
    }
    return bytes;
}

uint64_t DeviceResidentNamedDataset::compactRecordBytes() const {
    return compactRecords.isInitialized() ? compactRecords.getArraySizeInBytes() : 0;
}

uint64_t DeviceResidentNamedDataset::compactSourceBytes() const {
    uint64_t bytes = 0;
    for (const auto &entry : compactRaggedFields) {
        if (entry.second.values.isInitialized()) {
            bytes += entry.second.values.getArraySizeInBytes();
        }
    }
    for (const auto &entry : compactSources) {
        bytes += entry.second.bytes.getArraySizeInBytes();
    }
    return bytes;
}

uint64_t DeviceResidentNamedDataset::compactMetadataBytes() const {
    uint64_t bytes = 0;
    for (const auto &entry : compactSources) {
        if (entry.second.sequences.isInitialized()) {
            bytes += entry.second.sequences.getArraySizeInBytes();
        }
    }
    for (const auto &entry : compactAffineFields) {
        bytes += entry.second.segments.getArraySizeInBytes();
    }
    return bytes;
}

bool DeviceResidentNamedDataset::hasField(Thor::DatasetFieldId id) const {
    if (fields.find(id) != fields.end() || compactFieldIds.find(id) != compactFieldIds.end()) {
        return true;
    }
    for (const Thor::DatasetField &field : schema.getFields()) {
        if (field.id == id) {
            return snapshotRaggedFields.find(field.name) != snapshotRaggedFields.end();
        }
    }
    return false;
}

bool DeviceResidentNamedDataset::hasTensor(const std::string &name) const {
    return schema.contains(name) && fields.find(schema.getField(name).id) != fields.end();
}

bool DeviceResidentNamedDataset::hasCompactField(const std::string &name) const {
    return hasCompactDirectField(name) || hasCompactWindowField(name) ||
           hasCompactRaggedField(name);
}

bool DeviceResidentNamedDataset::hasCompactDirectField(const std::string &name) const {
    return compactDirectFields.find(name) != compactDirectFields.end();
}

bool DeviceResidentNamedDataset::hasCompactWindowField(const std::string &name) const {
    return compactWindowFields.find(name) != compactWindowFields.end();
}

bool DeviceResidentNamedDataset::hasCompactRaggedField(const std::string &name) const {
    return compactRaggedFields.find(name) != compactRaggedFields.end();
}

bool DeviceResidentNamedDataset::hasSnapshotRaggedField(const std::string &name) const {
    return snapshotRaggedFields.find(name) != snapshotRaggedFields.end();
}

std::optional<DeviceResidentNamedDataset::CompactWindowSourceAccess>
DeviceResidentNamedDataset::compactWindowSourceAccessForField(
    const std::string &fieldName) const {
    std::optional<std::string> sourceName;

    const auto windowIt = compactWindowFields.find(fieldName);
    if (windowIt != compactWindowFields.end()) {
        if (windowIt->second.materializeMask) return std::nullopt;
        sourceName = windowIt->second.spec.sourceName;
    } else {
        const auto raggedIt = compactRaggedFields.find(fieldName);
        if (raggedIt == compactRaggedFields.end() ||
            !raggedIt->second.spec.isWindowedSourceBacked()) {
            return std::nullopt;
        }
        THOR_THROW_IF_FALSE(raggedIt->second.spec.sourceName.has_value());
        sourceName = raggedIt->second.spec.sourceName.value();
    }

    const auto sourceIt = compactSources.find(sourceName.value());
    THOR_THROW_IF_FALSE(sourceIt != compactSources.end());
    const Tensor &source = sourceIt->second.bytes;
    THOR_THROW_IF_FALSE(source.isInitialized());
    THOR_THROW_IF_FALSE(source.getPlacement() == placement);
    THOR_THROW_IF_FALSE(
        source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    return CompactWindowSourceAccess{
        .sourceName = sourceIt->first,
        .tensorId = source.getTensorId(),
        .base = source.getMemPtr(),
        .bytes = source.getArraySizeInBytes()};
}

std::vector<DeviceResidentNamedDataset::CompactWindowSourceAccess>
DeviceResidentNamedDataset::compactWindowSourceAccesses() const {
    std::set<std::string> sourceNames;
    for (const auto &entry : compactWindowFields) {
        if (!entry.second.materializeMask) {
            sourceNames.insert(entry.second.spec.sourceName);
        }
    }
    for (const auto &entry : compactRaggedFields) {
        if (entry.second.spec.isWindowedSourceBacked()) {
            THOR_THROW_IF_FALSE(entry.second.spec.sourceName.has_value());
            sourceNames.insert(entry.second.spec.sourceName.value());
        }
    }

    std::vector<CompactWindowSourceAccess> accesses;
    accesses.reserve(sourceNames.size());
    std::set<uint64_t> seenTensorIds;
    for (const std::string &sourceName : sourceNames) {
        const auto sourceIt = compactSources.find(sourceName);
        THOR_THROW_IF_FALSE(sourceIt != compactSources.end());
        const Tensor &source = sourceIt->second.bytes;
        THOR_THROW_IF_FALSE(source.isInitialized());
        THOR_THROW_IF_FALSE(source.getPlacement() == placement);
        THOR_THROW_IF_FALSE(
            source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        if (!seenTensorIds.insert(source.getTensorId()).second) {
            continue;
        }
        accesses.push_back(CompactWindowSourceAccess{
            .sourceName = sourceName,
            .tensorId = source.getTensorId(),
            .base = source.getMemPtr(),
            .bytes = source.getArraySizeInBytes()});
    }
    return accesses;
}

const Tensor &DeviceResidentNamedDataset::field(Thor::DatasetFieldId id) const {
    const auto found = fields.find(id);
    if (found == fields.end()) {
        if (compactFieldIds.find(id) != compactFieldIds.end()) {
            throw std::runtime_error(
                "Compact resident fields are materialized by reference and have no canonical tensor.");
        }
        throw std::runtime_error("DeviceResidentNamedDataset does not contain requested field id.");
    }
    return found->second;
}

const Tensor &DeviceResidentNamedDataset::tensor(const std::string &name) const {
    return field(schema.getField(name).id);
}

RaggedBatchExtent DeviceResidentNamedDataset::validateCompactRaggedBatchCapacity(
    const std::string &fieldName,
    const Tensor &rowIndicesHost,
    uint64_t logicalRows,
    uint64_t maxTotalValues,
    uint64_t maxValuesPerRow) const {
    const auto found = compactRaggedFields.find(fieldName);
    if (found == compactRaggedFields.end()) {
        throw std::runtime_error(
            "DeviceResidentNamedDataset has no compact ragged field '" + fieldName + "'.");
    }
    THOR_THROW_IF_FALSE(rowIndicesHost.isInitialized());
    THOR_THROW_IF_FALSE(
        rowIndicesHost.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    THOR_THROW_IF_FALSE(rowIndicesHost.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(rowIndicesHost.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= rowIndicesHost.getDimensions().front());

    const uint64_t *rowIndices = rowIndicesHost.getMemPtr<uint64_t>();
    uint64_t active = 0;
    uint64_t maxActiveRowLength = 0;
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t sourceRow = rowIndices[row];
        if (sourceRow >= found->second.valueCounts.size()) {
            throw std::runtime_error(
                "Device resident ragged field '" + fieldName +
                "' selected invalid dataset row " + std::to_string(sourceRow) +
                " at batch row " + std::to_string(row) + ".");
        }
        const uint64_t count = found->second.valueCounts[sourceRow];
        if (maxValuesPerRow != 0 && count > maxValuesPerRow) {
            throw std::runtime_error(
                "Device resident ragged field '" + fieldName + "' row " + std::to_string(row) +
                " has length " + std::to_string(count) + ", exceeding maxValuesPerRow=" +
                std::to_string(maxValuesPerRow) + ".");
        }
        if (count > maxTotalValues - active) {
            throw std::runtime_error(
                "Device resident ragged field '" + fieldName + "' requires " +
                std::to_string(active + count) + " active values at row " +
                std::to_string(row) + ", exceeding maxTotalValues=" +
                std::to_string(maxTotalValues) + ".");
        }
        active += count;
        maxActiveRowLength = std::max(maxActiveRowLength, count);
    }
    return RaggedBatchExtent{.activeValueCount = active, .maxActiveRowLength = maxActiveRowLength};
}

void DeviceResidentNamedDataset::enqueueCompactRaggedFieldMaterialization(
    const std::string &fieldName,
    const Tensor &rowIndicesDevice,
    uint64_t logicalRows,
    ThorImplementation::RaggedTensor &destination,
    Stream &stream) const {
    const auto found = compactRaggedFields.find(fieldName);
    if (found == compactRaggedFields.end()) {
        throw std::runtime_error(
            "DeviceResidentNamedDataset has no compact ragged field '" + fieldName + "'.");
    }
    const DatasetLayout::RaggedTensorSpec &spec = found->second.spec;
    const ThorImplementation::RaggedTensorDescriptor descriptor = destination.getDescriptor();
    THOR_THROW_IF_FALSE(descriptor.getValuesDataType() == spec.dataType);
    THOR_THROW_IF_FALSE(descriptor.getTrailingDimensions() == spec.valueDimensions);
    THOR_THROW_IF_FALSE(descriptor.getBatchSize() == rowIndicesDevice.getDimensions().front());
    const Tensor *sourceValues = &found->second.values;
    if (spec.isWindowedSourceBacked()) {
        THOR_THROW_IF_FALSE(spec.sourceName.has_value());
        const auto sourceIt = compactSources.find(spec.sourceName.value());
        THOR_THROW_IF_FALSE(sourceIt != compactSources.end());
        sourceValues = &sourceIt->second.bytes;
    }
    // A completely empty ordinary ragged values sidecar is represented by an
    // uninitialized Tensor.  The CUDA materializer intentionally accepts that
    // representation when storedValueCount() == 0 and passes a null source
    // pointer to the gather kernel; every row then has count == 0, so the
    // pointer is never dereferenced.  Source-backed ragged windows still have a
    // real source allocation because empty window sources are rejected while
    // building compact residency.
    THOR_THROW_IF_FALSE(sourceValues->isInitialized() || spec.storedValueCount() == 0);

    Tensor values = destination.getValues();
    Tensor offsets = destination.getOffsets();
    launchDeviceResidentRaggedMaterializationKernel(
        compactRecords,
        *sourceValues,
        numExamples,
        layout.recordSizeBytes(),
        spec.referenceOffsetBytes,
        spec.storedValueCount(),
        spec.valueNumBytes(),
        logicalRows,
        values,
        offsets,
        rowIndicesDevice,
        stream);
}

RaggedBatchExtent DeviceResidentNamedDataset::validateSnapshotRaggedBatchCapacity(
    const std::string &fieldName,
    const Tensor &rowIndicesHost,
    uint64_t logicalRows,
    uint64_t maxTotalValues,
    uint64_t maxValuesPerRow) const {
    const auto found = snapshotRaggedFields.find(fieldName);
    if (found == snapshotRaggedFields.end()) {
        throw std::runtime_error(
            "DeviceResidentNamedDataset has no snapshot ragged field '" + fieldName + "'.");
    }
    THOR_THROW_IF_FALSE(rowIndicesHost.isInitialized());
    THOR_THROW_IF_FALSE(
        rowIndicesHost.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    THOR_THROW_IF_FALSE(rowIndicesHost.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(rowIndicesHost.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= rowIndicesHost.getDimensions().front());

    const uint64_t *rowIndices = rowIndicesHost.getMemPtr<uint64_t>();
    uint64_t active = 0;
    uint64_t maxActiveRowLength = 0;
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t sourceRow = rowIndices[row];
        if (sourceRow >= found->second.valueCounts.size()) {
            throw std::runtime_error(
                "Device resident ragged field '" + fieldName +
                "' selected invalid dataset row " + std::to_string(sourceRow) +
                " at batch row " + std::to_string(row) + ".");
        }
        const uint64_t count = found->second.valueCounts[sourceRow];
        if (maxValuesPerRow != 0 && count > maxValuesPerRow) {
            throw std::runtime_error(
                "Device resident ragged field '" + fieldName + "' row " + std::to_string(row) +
                " has length " + std::to_string(count) + ", exceeding maxValuesPerRow=" +
                std::to_string(maxValuesPerRow) + ".");
        }
        if (count > maxTotalValues - active) {
            throw std::runtime_error(
                "Device resident ragged field '" + fieldName + "' requires " +
                std::to_string(active + count) + " active values at row " +
                std::to_string(row) + ", exceeding maxTotalValues=" +
                std::to_string(maxTotalValues) + ".");
        }
        active += count;
        maxActiveRowLength = std::max(maxActiveRowLength, count);
    }
    return RaggedBatchExtent{.activeValueCount = active, .maxActiveRowLength = maxActiveRowLength};
}

void DeviceResidentNamedDataset::enqueueSnapshotRaggedFieldMaterialization(
    const std::string &fieldName,
    const Tensor &rowIndicesDevice,
    uint64_t logicalRows,
    ThorImplementation::RaggedTensor &destination,
    Stream &stream) const {
    const auto found = snapshotRaggedFields.find(fieldName);
    if (found == snapshotRaggedFields.end()) {
        throw std::runtime_error(
            "DeviceResidentNamedDataset has no snapshot ragged field '" + fieldName + "'.");
    }
    const SnapshotRaggedFieldStorage &storage = found->second;
    const ThorImplementation::RaggedTensorDescriptor descriptor = destination.getDescriptor();
    THOR_THROW_IF_FALSE(descriptor.getValuesDataType() == storage.spec.dataType);
    THOR_THROW_IF_FALSE(descriptor.getTrailingDimensions() == storage.spec.valueDimensions);
    THOR_THROW_IF_FALSE(descriptor.getBatchSize() == rowIndicesDevice.getDimensions().front());
    Tensor values = destination.getValues();
    Tensor offsets = destination.getOffsets();
    launchDeviceResidentRaggedMaterializationKernel(
        storage.references,
        storage.values,
        numExamples,
        2 * sizeof(uint64_t),
        0,
        storage.storedValueCount,
        storage.valueBytes,
        logicalRows,
        values,
        offsets,
        rowIndicesDevice,
        stream);
}

void DeviceResidentNamedDataset::enqueueCompactFieldMaterialization(
    const std::string &fieldName,
    const Tensor &rowIndicesDevice,
    Tensor &destination,
    Stream &stream) const {
    const auto directIt = compactDirectFields.find(fieldName);
    if (directIt != compactDirectFields.end()) {
        const DatasetLayout::TensorSpec &spec = directIt->second.spec;
        std::vector<uint64_t> expectedDimensions;
        expectedDimensions.push_back(rowIndicesDevice.getDimensions().front());
        expectedDimensions.insert(
            expectedDimensions.end(), spec.dimensions.begin(), spec.dimensions.end());
        THOR_THROW_IF_FALSE(
            destination.getDescriptor() == TensorDescriptor(spec.dataType, expectedDimensions));
        launchDeviceResidentDirectMaterializationKernel(
            compactRecords,
            numExamples,
            layout.recordSizeBytes(),
            spec.offsetBytes,
            spec.numBytes,
            destination,
            rowIndicesDevice,
            stream);
        return;
    }

    const auto fieldIt = compactWindowFields.find(fieldName);
    if (fieldIt == compactWindowFields.end()) {
        throw std::runtime_error(
            "DeviceResidentNamedDataset has no compact field '" + fieldName + "'.");
    }
    const CompactWindowFieldStorage &fieldStorage = fieldIt->second;
    const auto sourceIt = compactSources.find(fieldStorage.spec.sourceName);
    THOR_THROW_IF_FALSE(sourceIt != compactSources.end());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDimensions().size() == 1);

    std::vector<uint64_t> expectedDimensions;
    expectedDimensions.push_back(rowIndicesDevice.getDimensions().front());
    if (fieldStorage.materializeMask) {
        expectedDimensions.push_back(fieldStorage.spec.windowLength());
        THOR_THROW_IF_FALSE(
            destination.getDescriptor() == TensorDescriptor(DataType::UINT8, expectedDimensions));
    } else {
        expectedDimensions.insert(
            expectedDimensions.end(),
            fieldStorage.spec.dimensions.begin(),
            fieldStorage.spec.dimensions.end());
        THOR_THROW_IF_FALSE(
            destination.getDescriptor() ==
            TensorDescriptor(fieldStorage.spec.dataType, expectedDimensions));
    }

    Tensor noAffineSegments;
    const Tensor *affineSegments = &noAffineSegments;
    uint64_t affineSegmentCount = 0;
    if (fieldStorage.spec.referenceMode ==
        DatasetLayout::WindowedTensorReferenceMode::AFFINE) {
        const auto affineIt = compactAffineFields.find(fieldStorage.spec.name);
        THOR_THROW_IF_FALSE(affineIt != compactAffineFields.end());
        affineSegments = &affineIt->second.segments;
        affineSegmentCount = affineIt->second.segmentCount;
    }

    DeviceResidentWindowMaterializationSpec launchSpec;
    launchSpec.referenceMode = fieldStorage.spec.referenceMode;
    launchSpec.dataType = fieldStorage.spec.dataType;
    launchSpec.keyDataType = fieldStorage.spec.keyDataType;
    launchSpec.indexDataType = fieldStorage.spec.indexDataType;
    launchSpec.numExamples = numExamples;
    launchSpec.recordSizeBytes = layout.recordSizeBytes();
    launchSpec.referenceOffsetBytes = fieldStorage.spec.referenceOffsetBytes;
    launchSpec.windowLength = fieldStorage.spec.windowLength();
    launchSpec.sourceStepBytes = fieldStorage.spec.sourceStepNumBytes();
    launchSpec.padValue = fieldStorage.spec.padValue;
    launchSpec.materializeMask = fieldStorage.materializeMask;

    launchDeviceResidentWindowMaterializationKernel(
        compactRecords,
        sourceIt->second.bytes,
        sourceIt->second.sequences,
        sourceIt->second.sequenceCount,
        *affineSegments,
        affineSegmentCount,
        launchSpec,
        destination,
        rowIndicesDevice,
        stream);
}
