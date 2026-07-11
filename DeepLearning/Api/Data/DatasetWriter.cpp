#include "DeepLearning/Api/Data/DatasetWriter.h"

#include "Utilities/Loaders/Shard.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace {

std::string makeShardFilename(uint64_t shardIndex) {
    std::ostringstream out;
    out << "dataset_records_" << std::setw(6) << std::setfill('0') << shardIndex << ".shard";
    return out.str();
}

void ensureEmptyOrCreateDirectory(const std::filesystem::path &path) {
    if (std::filesystem::exists(path)) {
        if (!std::filesystem::is_directory(path)) {
            throw std::runtime_error("DatasetWriter path exists but is not a directory: " + path.string());
        }
        if (!std::filesystem::is_empty(path)) {
            throw std::runtime_error("DatasetWriter dataset directory must be empty: " + path.string());
        }
    } else {
        std::filesystem::create_directories(path);
    }
}

std::string shapeToString(const std::vector<uint64_t> &shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << shape[i];
    }
    out << ']';
    return out.str();
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const char *context) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return a + b;
}

uint64_t checkedMul(uint64_t a, uint64_t b, const char *context) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return a * b;
}

std::string bytesToHex(const void *data, uint64_t numBytes) {
    if (data == nullptr) {
        throw std::runtime_error("DatasetWriter cannot hex encode a null byte pointer.");
    }
    const auto *bytes = static_cast<const uint8_t *>(data);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint64_t i = 0; i < numBytes; ++i) {
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
}

std::filesystem::path windowedTensorSourceDirectory(const std::filesystem::path &datasetPath) {
    return datasetPath / "windowed_tensor_sources";
}

std::string makeWindowedTensorSourceFilename(uint64_t ordinal) {
    std::ostringstream out;
    out << "windowed_tensor_sources/windowed_tensor_" << std::setw(6) << std::setfill('0') << ordinal << ".bin";
    return out.str();
}

void checkedIndexBounds(int64_t startIndex, uint64_t numSteps, const std::string &tensorName) {
    if (numSteps > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("DatasetWriter windowed tensor '" + tensorName + "' source length is outside int64 range.");
    }
    const int64_t signedSteps = static_cast<int64_t>(numSteps);
    if (startIndex > std::numeric_limits<int64_t>::max() - signedSteps) {
        throw std::runtime_error("DatasetWriter windowed tensor '" + tensorName + "' source index range overflows int64.");
    }
}

}  // namespace

uint64_t DatasetWriter::ShardManifestEntry::remainingCapacity() const {
    if (numExamples > capacityExamples) {
        throw std::runtime_error("DatasetWriter shard entry exceeded its capacity.");
    }
    return capacityExamples - numExamples;
}

uint64_t DatasetWriter::ShardManifestEntry::numBytes(uint64_t recordSizeBytes) const {
    return checkedMul(numExamples, recordSizeBytes, "DatasetWriter shard byte count");
}

DatasetWriter::DatasetWriter(std::filesystem::path datasetPath,
                             DatasetLayout layout,
                             uint64_t examplesPerShard,
                             std::optional<uint64_t> expectedNumExamples,
                             bool preallocate)
    : datasetPath(std::move(datasetPath)),
      datasetId(Thor::DatasetId::generate()),
      layout(std::move(layout)),
      examplesPerShard(examplesPerShard),
      expectedNumExamples(expectedNumExamples),
      preallocate(preallocate),
      closed(false),
      nextShardIndex(0),
      totalExamples(0) {
    this->layout.validate();
    if (this->examplesPerShard == 0) {
        throw std::runtime_error("DatasetWriter examples_per_shard must be non-zero.");
    }
    if (this->preallocate && !this->expectedNumExamples.has_value()) {
        throw std::runtime_error("DatasetWriter preallocate=true requires expected_num_examples.");
    }
    ensureEmptyOrCreateDirectory(this->datasetPath);
    uint64_t ordinal = 0;
    for (const DatasetLayout::WindowedTensorSpec &spec : this->layout.windowedTensors()) {
        WindowedTensorSourceManifestEntry entry;
        entry.filename = makeWindowedTensorSourceFilename(ordinal++);
        windowedTensorSources.emplace(spec.name, std::move(entry));
    }
}

DatasetWriter::~DatasetWriter() {
    if (!closed) {
        try {
            close();
        } catch (...) {
        }
    }
}

void DatasetWriter::writeIndexedExample(const std::map<std::string, TensorView> &tensors) {
    if (layout.hasWindowedTensors()) {
        throw std::runtime_error(
            "DatasetWriter writeIndexedExample for a layout with windowed tensors requires windowed tensor references.");
    }
    validateWritable();
    validateTensorMapExact(tensors);
    std::vector<uint8_t> record = packRecord(tensors);
    writePackedIndexedRecords(record.data(), 1);
}

void DatasetWriter::writeIndexedExample(
    const std::map<std::string, TensorView> &tensors,
    const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences) {
    validateWritable();
    validateTensorMapExact(tensors);
    validateWindowedTensorReferenceMapExact(windowedTensorReferences);
    std::vector<uint8_t> record = packRecord(tensors, windowedTensorReferences);
    writePackedIndexedRecords(record.data(), 1);
}

void DatasetWriter::writeIndexedExamples(const std::map<std::string, TensorBatchView> &tensors) {
    if (layout.hasWindowedTensors()) {
        throw std::runtime_error(
            "DatasetWriter writeIndexedExamples for a layout with windowed tensors requires windowed tensor references.");
    }
    validateWritable();
    const uint64_t count = validateTensorBatchMapExact(tensors);
    std::vector<uint8_t> records = packRecords(tensors, count);
    writePackedIndexedRecords(records.data(), count);
}

void DatasetWriter::writeIndexedExamples(
    const std::map<std::string, TensorBatchView> &tensors,
    const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences) {
    validateWritable();
    const uint64_t count = validateTensorAndWindowedTensorReferenceBatchMapsExact(tensors, windowedTensorReferences);
    std::vector<uint8_t> records = packRecords(tensors, windowedTensorReferences, count);
    writePackedIndexedRecords(records.data(), count);
}

void DatasetWriter::writeWindowedTensorSource(std::string_view tensorName, const WindowedTensorSourceView &source) {
    validateWritable();
    const DatasetLayout::WindowedTensorSpec &spec = layout.windowedTensor(tensorName);
    auto manifestIt = windowedTensorSources.find(spec.name);
    if (manifestIt == windowedTensorSources.end()) {
        throw std::runtime_error("DatasetWriter missing source manifest entry for windowed tensor: " + spec.name);
    }
    WindowedTensorSourceManifestEntry &manifestEntry = manifestIt->second;
    if (source.key == nullptr) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' has null key.");
    }
    if (source.data == nullptr) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' has null data.");
    }
    if (source.dataType != spec.dataType) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' has wrong dtype.");
    }
    if (source.dimensions.size() != spec.dimensions.size()) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' shape " +
                                 shapeToString(source.dimensions) + " must be [N, *window_step_shape].");
    }
    if (source.dimensions.empty() || source.dimensions.front() == 0) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' must contain at least one step.");
    }
    std::vector<uint64_t> sourceStepShape(source.dimensions.begin() + 1, source.dimensions.end());
    if (sourceStepShape != spec.sourceStepDimensions()) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' shape " +
                                 shapeToString(source.dimensions) + " does not match windowed tensor step shape " +
                                 shapeToString(spec.sourceStepDimensions()) + ".");
    }
    const uint64_t numSteps = source.dimensions.front();
    const uint64_t expectedBytes = checkedMul(numSteps, spec.sourceStepNumBytes(), "DatasetWriter windowed source bytes");
    if (source.numBytes != expectedBytes) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' byte count " +
                                 std::to_string(source.numBytes) + " does not match expected byte count " +
                                 std::to_string(expectedBytes) + ".");
    }
    checkedIndexBounds(source.startIndex, numSteps, spec.name);
    const std::string keyHex = bytesToHex(source.key, spec.keyNumBytes());
    if (!manifestEntry.keyHexValues.insert(keyHex).second) {
        throw std::runtime_error("DatasetWriter windowed tensor source '" + spec.name + "' duplicate key.");
    }

    std::filesystem::create_directories(windowedTensorSourceDirectory(datasetPath));
    const std::filesystem::path sourcePath = datasetPath / manifestEntry.filename;
    std::ofstream out(sourcePath, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("DatasetWriter failed to open windowed tensor source for writing: " + sourcePath.string());
    }
    out.write(static_cast<const char *>(source.data), static_cast<std::streamsize>(source.numBytes));
    if (!out.good()) {
        throw std::runtime_error("DatasetWriter failed while writing windowed tensor source: " + sourcePath.string());
    }

    const uint64_t offsetBytes = manifestEntry.numBytes;
    manifestEntry.numBytes = checkedAdd(manifestEntry.numBytes, source.numBytes, "DatasetWriter windowed source bytes");
    const int64_t endIndexExclusive = source.startIndex + static_cast<int64_t>(numSteps);
    manifestEntry.sequences.push_back(DatasetLayout::WindowedTensorSourceSequence{.keyHex = keyHex,
                                                                       .startIndex = source.startIndex,
                                                                       .endIndexExclusive = endIndexExclusive,
                                                                       .offsetBytes = offsetBytes,
                                                                       .numSteps = numSteps,
                                                                       .numBytes = source.numBytes});
}

void DatasetWriter::writePackedIndexedRecords(const uint8_t *records, uint64_t count) {
    if (count == 0) {
        throw std::runtime_error("DatasetWriter writeIndexedExamples requires at least one example.");
    }
    if (records == nullptr) {
        throw std::runtime_error("DatasetWriter writeIndexedExamples received null records.");
    }
    if (expectedNumExamples.has_value()) {
        const uint64_t written = numExamples();
        if (written > expectedNumExamples.value() || count > expectedNumExamples.value() - written) {
            throw std::runtime_error("DatasetWriter write would exceed expected_num_examples.");
        }
    }

    uint64_t consumed = 0;
    while (consumed < count) {
        ensureCurrentShard();
        const uint64_t available = shardEntries.back().remainingCapacity();
        if (available == 0) {
            finalizeCurrentShard();
            continue;
        }
        const uint64_t toWrite = std::min<uint64_t>(count - consumed, available);
        currentShard->writeExamplesContiguous(const_cast<uint8_t *>(records + checkedMul(consumed, layout.recordSizeBytes(),
                                                                                         "DatasetWriter chunk offset")),
                                              toWrite,
                                              ExampleType::TRAIN);
        shardEntries.back().numExamples += toWrite;
        totalExamples += toWrite;
        consumed += toWrite;
    }
}

void DatasetWriter::close() {
    if (closed) {
        return;
    }
    if (expectedNumExamples.has_value() && numExamples() != expectedNumExamples.value()) {
        throw std::runtime_error("DatasetWriter wrote " + std::to_string(numExamples()) +
                                 " examples but expected_num_examples was " + std::to_string(expectedNumExamples.value()) + ".");
    }
    finalizeCurrentShard();
    writeManifest();
    closed = true;
}

bool DatasetWriter::isClosed() const { return closed; }

const std::filesystem::path &DatasetWriter::path() const { return datasetPath; }

std::filesystem::path DatasetWriter::manifestPath() const { return datasetPath / MANIFEST_FILENAME; }

uint64_t DatasetWriter::numExamples() const { return totalExamples; }

const DatasetLayout &DatasetWriter::getLayout() const { return layout; }

std::optional<uint64_t> DatasetWriter::getExpectedNumExamples() const { return expectedNumExamples; }

bool DatasetWriter::getPreallocate() const { return preallocate; }

void DatasetWriter::validateWritable() const {
    if (closed) {
        throw std::runtime_error("DatasetWriter is closed.");
    }
}

void DatasetWriter::validateTensorMapExact(const std::map<std::string, TensorView> &tensors) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("DatasetWriter tensor count " + std::to_string(tensors.size()) +
                                 " does not match layout tensor count " + std::to_string(layout.tensors().size()) + ".");
    }

    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("DatasetWriter missing tensor: " + spec.name);
        }
        const TensorView &view = it->second;
        if (view.data == nullptr) {
            throw std::runtime_error("DatasetWriter tensor '" + spec.name + "' has null data.");
        }
        if (view.dataType != spec.dataType) {
            throw std::runtime_error("DatasetWriter tensor '" + spec.name + "' has wrong dtype.");
        }
        if (view.dimensions != spec.dimensions) {
            throw std::runtime_error("DatasetWriter tensor '" + spec.name + "' shape " + shapeToString(view.dimensions) +
                                     " does not match layout shape " + shapeToString(spec.dimensions) + ".");
        }
        if (view.numBytes != spec.numBytes) {
            throw std::runtime_error("DatasetWriter tensor '" + spec.name + "' byte count " +
                                     std::to_string(view.numBytes) + " does not match layout byte count " +
                                     std::to_string(spec.numBytes) + ".");
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
}

uint64_t DatasetWriter::validateTensorBatchMapExact(const std::map<std::string, TensorBatchView> &tensors) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("DatasetWriter tensor count " + std::to_string(tensors.size()) +
                                 " does not match layout tensor count " + std::to_string(layout.tensors().size()) + ".");
    }

    bool haveCount = false;
    uint64_t count = 0;
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("DatasetWriter missing tensor: " + spec.name);
        }
        const TensorBatchView &view = it->second;
        if (view.data == nullptr) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' has null data.");
        }
        if (view.dataType != spec.dataType) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' has wrong dtype.");
        }
        if (view.dimensions.size() != spec.dimensions.size() + 1) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' shape " +
                                     shapeToString(view.dimensions) + " must be [N, *layout_shape].");
        }
        if (view.dimensions.front() == 0) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' must contain at least one example.");
        }
        if (!haveCount) {
            count = view.dimensions.front();
            haveCount = true;
        } else if (count != view.dimensions.front()) {
            throw std::runtime_error("DatasetWriter tensor batches must have the same leading dimension.");
        }
        std::vector<uint64_t> tensorShape(view.dimensions.begin() + 1, view.dimensions.end());
        if (tensorShape != spec.dimensions) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' shape " +
                                     shapeToString(view.dimensions) + " does not match layout shape [N," +
                                     shapeToString(spec.dimensions) + "].");
        }
        const uint64_t expectedBytes = checkedMul(count, spec.numBytes, "DatasetWriter tensor batch bytes");
        if (view.numBytes != expectedBytes) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' byte count " +
                                     std::to_string(view.numBytes) + " does not match expected byte count " +
                                     std::to_string(expectedBytes) + ".");
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
    return count;
}


void DatasetWriter::validateWindowedTensorReferenceMapExact(
    const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences) const {
    if (windowedTensorReferences.size() != layout.windowedTensors().size()) {
        throw std::runtime_error("DatasetWriter windowed tensor reference count " +
                                 std::to_string(windowedTensorReferences.size()) +
                                 " does not match layout windowed tensor count " + std::to_string(layout.windowedTensors().size()) + ".");
    }

    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        const auto it = windowedTensorReferences.find(spec.name);
        if (it == windowedTensorReferences.end()) {
            throw std::runtime_error("DatasetWriter missing windowed tensor reference: " + spec.name);
        }
        const WindowedTensorReferenceView &view = it->second;
        if (view.key == nullptr) {
            throw std::runtime_error("DatasetWriter windowed tensor reference '" + spec.name + "' has null key.");
        }
        if (view.start == nullptr) {
            throw std::runtime_error("DatasetWriter windowed tensor reference '" + spec.name + "' has null start.");
        }
        if (view.keyDataType != spec.keyDataType) {
            throw std::runtime_error("DatasetWriter windowed tensor reference '" + spec.name + "' has wrong key dtype.");
        }
        if (view.indexDataType != spec.indexDataType) {
            throw std::runtime_error("DatasetWriter windowed tensor reference '" + spec.name + "' has wrong index dtype.");
        }
    }
    for (const auto &entry : windowedTensorReferences) {
        (void)layout.windowedTensor(entry.first);
    }
}

uint64_t DatasetWriter::validateTensorAndWindowedTensorReferenceBatchMapsExact(
    const std::map<std::string, TensorBatchView> &tensors,
    const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("DatasetWriter tensor count " + std::to_string(tensors.size()) +
                                 " does not match layout tensor count " + std::to_string(layout.tensors().size()) + ".");
    }
    if (windowedTensorReferences.size() != layout.windowedTensors().size()) {
        throw std::runtime_error("DatasetWriter windowed tensor reference count " +
                                 std::to_string(windowedTensorReferences.size()) +
                                 " does not match layout windowed tensor count " + std::to_string(layout.windowedTensors().size()) + ".");
    }

    bool haveCount = false;
    uint64_t count = 0;
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("DatasetWriter missing tensor: " + spec.name);
        }
        const TensorBatchView &view = it->second;
        if (view.data == nullptr) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' has null data.");
        }
        if (view.dataType != spec.dataType) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' has wrong dtype.");
        }
        if (view.dimensions.size() != spec.dimensions.size() + 1) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' shape " +
                                     shapeToString(view.dimensions) + " must be [N, *layout_shape].");
        }
        if (view.dimensions.front() == 0) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' must contain at least one example.");
        }
        if (!haveCount) {
            count = view.dimensions.front();
            haveCount = true;
        } else if (count != view.dimensions.front()) {
            throw std::runtime_error("DatasetWriter tensor batches and windowed tensor references must have the same leading dimension.");
        }
        std::vector<uint64_t> tensorShape(view.dimensions.begin() + 1, view.dimensions.end());
        if (tensorShape != spec.dimensions) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' shape " +
                                     shapeToString(view.dimensions) + " does not match layout shape [N," +
                                     shapeToString(spec.dimensions) + "].");
        }
        const uint64_t expectedBytes = checkedMul(count, spec.numBytes, "DatasetWriter tensor batch bytes");
        if (view.numBytes != expectedBytes) {
            throw std::runtime_error("DatasetWriter tensor batch '" + spec.name + "' byte count " +
                                     std::to_string(view.numBytes) + " does not match expected byte count " +
                                     std::to_string(expectedBytes) + ".");
        }
    }

    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        const auto it = windowedTensorReferences.find(spec.name);
        if (it == windowedTensorReferences.end()) {
            throw std::runtime_error("DatasetWriter missing windowed tensor reference: " + spec.name);
        }
        const WindowedTensorReferenceBatchView &view = it->second;
        if (view.keys == nullptr) {
            throw std::runtime_error("DatasetWriter windowed tensor reference batch '" + spec.name + "' has null keys.");
        }
        if (view.starts == nullptr) {
            throw std::runtime_error("DatasetWriter windowed tensor reference batch '" + spec.name + "' has null starts.");
        }
        if (view.keyDataType != spec.keyDataType) {
            throw std::runtime_error("DatasetWriter windowed tensor reference batch '" + spec.name + "' has wrong key dtype.");
        }
        if (view.indexDataType != spec.indexDataType) {
            throw std::runtime_error("DatasetWriter windowed tensor reference batch '" + spec.name + "' has wrong index dtype.");
        }
        if (view.count == 0) {
            throw std::runtime_error("DatasetWriter windowed tensor reference batch '" + spec.name + "' must contain at least one example.");
        }
        if (!haveCount) {
            count = view.count;
            haveCount = true;
        } else if (count != view.count) {
            throw std::runtime_error("DatasetWriter tensor batches and windowed tensor references must have the same leading dimension.");
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
    for (const auto &entry : windowedTensorReferences) {
        (void)layout.windowedTensor(entry.first);
    }
    if (!haveCount) {
        throw std::runtime_error("DatasetWriter writeIndexedExamples requires at least one tensor or windowed tensor reference.");
    }
    return count;
}
std::vector<uint8_t> DatasetWriter::packRecord(const std::map<std::string, TensorView> &tensors) const {
    std::vector<uint8_t> record(layout.recordSizeBytes(), 0);
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        const TensorView &view = tensors.at(spec.name);
        std::memcpy(record.data() + spec.offsetBytes, view.data, spec.numBytes);
    }
    return record;
}

std::vector<uint8_t> DatasetWriter::packRecords(const std::map<std::string, TensorBatchView> &tensors,
                                                                 uint64_t count) const {
    std::vector<uint8_t> records(checkedMul(count, layout.recordSizeBytes(), "DatasetWriter packed records"), 0);
    for (uint64_t row = 0; row < count; ++row) {
        uint8_t *record = records.data() + checkedMul(row, layout.recordSizeBytes(), "DatasetWriter record offset");
        for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
            const TensorBatchView &view = tensors.at(spec.name);
            const uint8_t *source = static_cast<const uint8_t *>(view.data) + checkedMul(row, spec.numBytes,
                                                                                        "DatasetWriter tensor row offset");
            std::memcpy(record + spec.offsetBytes, source, spec.numBytes);
        }
    }
    return records;
}

std::vector<uint8_t> DatasetWriter::packRecord(
    const std::map<std::string, TensorView> &tensors,
    const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences) const {
    std::vector<uint8_t> record = packRecord(tensors);
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        const WindowedTensorReferenceView &view = windowedTensorReferences.at(spec.name);
        uint8_t *reference = record.data() + spec.referenceOffsetBytes;
        std::memcpy(reference, view.key, spec.keyNumBytes());
        std::memcpy(reference + spec.keyNumBytes(), view.start, spec.indexNumBytes());
    }
    return record;
}

std::vector<uint8_t> DatasetWriter::packRecords(
    const std::map<std::string, TensorBatchView> &tensors,
    const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences,
    uint64_t count) const {
    std::vector<uint8_t> records(checkedMul(count, layout.recordSizeBytes(), "DatasetWriter packed records"), 0);
    for (uint64_t row = 0; row < count; ++row) {
        uint8_t *record = records.data() + checkedMul(row, layout.recordSizeBytes(), "DatasetWriter record offset");
        for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
            const TensorBatchView &view = tensors.at(spec.name);
            const uint8_t *source = static_cast<const uint8_t *>(view.data) + checkedMul(row, spec.numBytes,
                                                                                        "DatasetWriter tensor row offset");
            std::memcpy(record + spec.offsetBytes, source, spec.numBytes);
        }
        for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
            const WindowedTensorReferenceBatchView &view = windowedTensorReferences.at(spec.name);
            const uint8_t *key = static_cast<const uint8_t *>(view.keys) + checkedMul(row,
                                                                                     spec.keyNumBytes(),
                                                                                     "DatasetWriter windowed key row offset");
            const uint8_t *start = static_cast<const uint8_t *>(view.starts) + checkedMul(row,
                                                                                         spec.indexNumBytes(),
                                                                                         "DatasetWriter windowed start row offset");
            uint8_t *reference = record + spec.referenceOffsetBytes;
            std::memcpy(reference, key, spec.keyNumBytes());
            std::memcpy(reference + spec.keyNumBytes(), start, spec.indexNumBytes());
        }
    }
    return records;
}

uint64_t DatasetWriter::nextShardCapacity() const {
    if (!expectedNumExamples.has_value()) {
        return examplesPerShard;
    }
    const uint64_t written = numExamples();
    if (written >= expectedNumExamples.value()) {
        return 0;
    }
    return std::min<uint64_t>(examplesPerShard, expectedNumExamples.value() - written);
}

void DatasetWriter::ensureCurrentShard() {
    if (currentShard && shardEntries.back().remainingCapacity() > 0) {
        return;
    }

    finalizeCurrentShard();

    const uint64_t capacity = nextShardCapacity();
    if (capacity == 0) {
        throw std::runtime_error("DatasetWriter cannot create another shard because expected_num_examples has been reached.");
    }

    ShardManifestEntry entry;
    entry.filename = makeShardFilename(nextShardIndex++);
    entry.globalStart = numExamples();
    entry.capacityExamples = capacity;

    std::vector<std::string> allClasses;
    currentShard = std::make_unique<Shard>();
    currentShard->createCompactShard((datasetPath / entry.filename).string(),
                                     capacity,
                                     0,
                                     0,
                                     layout.recordSizeBytes(),
                                     ThorImplementation::DataType::UINT8,
                                     allClasses,
                                     preallocate);
    shardEntries.push_back(std::move(entry));
}

void DatasetWriter::finalizeCurrentShard() {
    if (currentShard) {
        currentShard->shrinkToFit();
        currentShard.reset();
    }
}

void DatasetWriter::writeManifest() const {
    json root = layout.toJson();
    root["dataset_id"] = datasetId.str();
    root["storage_mode"] = STORAGE_MODE_INDEXED;
    root["num_examples"] = numExamples();
    if (expectedNumExamples.has_value()) {
        root["expected_num_examples"] = expectedNumExamples.value();
    }
    root["preallocated"] = preallocate;
    root["shards"] = json::array();

    if (!windowedTensorSources.empty()) {
        if (!root.contains("windowed_tensors") || !root.at("windowed_tensors").is_object()) {
            throw std::runtime_error("DatasetWriter internal error: missing windowed_tensors in layout manifest.");
        }
        for (const auto &entry : windowedTensorSources) {
            json sourceStorage{{"file", entry.second.filename}, {"num_bytes", entry.second.numBytes}, {"sequences", json::array()}};
            for (const DatasetLayout::WindowedTensorSourceSequence &sequence : entry.second.sequences) {
                sourceStorage["sequences"].push_back(json{{"key_hex", sequence.keyHex},
                                                           {"start_index", sequence.startIndex},
                                                           {"end_index_exclusive", sequence.endIndexExclusive},
                                                           {"offset_bytes", sequence.offsetBytes},
                                                           {"num_steps", sequence.numSteps},
                                                           {"num_bytes", sequence.numBytes}});
            }
            const std::filesystem::path sourcePath = datasetPath / entry.second.filename;
            if (!std::filesystem::exists(sourcePath)) {
                std::filesystem::create_directories(sourcePath.parent_path());
                std::ofstream emptySource(sourcePath, std::ios::binary | std::ios::app);
                if (!emptySource.is_open()) {
                    throw std::runtime_error("DatasetWriter failed to create empty windowed tensor source: " +
                                             sourcePath.string());
                }
            }
            root["windowed_tensors"].at(entry.first)["source_storage"] = std::move(sourceStorage);
        }
    }

    for (const ShardManifestEntry &entry : shardEntries) {
        json shard = json{{"file", entry.filename},
                          {"global_start", entry.globalStart},
                          {"num_examples", entry.numExamples},
                          {"capacity_examples", entry.capacityExamples},
                          {"num_bytes", entry.numBytes(layout.recordSizeBytes())}};
        root["shards"].push_back(std::move(shard));
    }

    std::ofstream out(manifestPath(), std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("DatasetWriter failed to open manifest for writing: " + manifestPath().string());
    }
    out << root.dump(2) << '\n';
    if (!out.good()) {
        throw std::runtime_error("DatasetWriter failed while writing manifest: " + manifestPath().string());
    }
}
