#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace {

std::string makeShardFilename(uint64_t shardIndex) {
    std::ostringstream out;
    out << "local_named_examples_" << std::setw(6) << std::setfill('0') << shardIndex << ".shard";
    return out.str();
}

void ensureEmptyOrCreateDirectory(const std::filesystem::path &path) {
    if (std::filesystem::exists(path)) {
        if (!std::filesystem::is_directory(path)) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter path exists but is not a directory: " + path.string());
        }
        if (!std::filesystem::is_empty(path)) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter dataset directory must be empty: " + path.string());
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

uint64_t checkedMul(uint64_t a, uint64_t b, const char *context) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string(context) + " overflow.");
    }
    return a * b;
}

}  // namespace

uint64_t LocalNamedExampleDatasetWriter::ShardManifestEntry::totalExamples() const {
    return trainExamples + validateExamples + testExamples;
}

uint64_t LocalNamedExampleDatasetWriter::ShardManifestEntry::examples(ExampleType exampleType) const {
    switch (exampleType) {
        case ExampleType::TRAIN:
            return trainExamples;
        case ExampleType::VALIDATE:
            return validateExamples;
        case ExampleType::TEST:
            return testExamples;
        default:
            break;
    }
    throw std::runtime_error("Unsupported ExampleType value: " + std::to_string(static_cast<int>(exampleType)));
}

uint64_t LocalNamedExampleDatasetWriter::ShardManifestEntry::remainingCapacity() const {
    if (totalExamples() > capacityExamples) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter shard entry exceeded its capacity.");
    }
    return capacityExamples - totalExamples();
}

uint64_t LocalNamedExampleDatasetWriter::ShardManifestEntry::numBytes(uint64_t recordSizeBytes) const {
    return checkedMul(totalExamples(), recordSizeBytes, "LocalNamedExampleDatasetWriter shard byte count");
}

void LocalNamedExampleDatasetWriter::ShardManifestEntry::increment(ExampleType exampleType) { incrementBy(exampleType, 1); }

void LocalNamedExampleDatasetWriter::ShardManifestEntry::incrementBy(ExampleType exampleType, uint64_t count) {
    switch (exampleType) {
        case ExampleType::TRAIN:
            trainExamples += count;
            return;
        case ExampleType::VALIDATE:
            validateExamples += count;
            return;
        case ExampleType::TEST:
            testExamples += count;
            return;
        default:
            break;
    }
    throw std::runtime_error("Unsupported ExampleType value: " + std::to_string(static_cast<int>(exampleType)));
}

LocalNamedExampleDatasetWriter::LocalNamedExampleDatasetWriter(std::filesystem::path datasetPath,
                                                               LocalNamedExampleLayout layout,
                                                               uint64_t examplesPerShard,
                                                               StorageMode storageMode,
                                                               std::optional<uint64_t> expectedNumExamples,
                                                               bool preallocate)
    : datasetPath(std::move(datasetPath)),
      layout(std::move(layout)),
      examplesPerShard(examplesPerShard),
      storageMode(storageMode),
      expectedNumExamples(expectedNumExamples),
      preallocate(preallocate),
      closed(false),
      nextShardIndex(0),
      totalTrainExamples(0),
      totalValidateExamples(0),
      totalTestExamples(0) {
    this->layout.validate();
    if (this->examplesPerShard == 0) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter examples_per_shard must be non-zero.");
    }
    (void)storageModeToString(this->storageMode);
    if (this->preallocate && !this->expectedNumExamples.has_value()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter preallocate=true requires expected_num_examples.");
    }
    if (this->expectedNumExamples.has_value() && this->storageMode != StorageMode::INDEXED) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter expected_num_examples is supported only for indexed storage_mode.");
    }
    ensureEmptyOrCreateDirectory(this->datasetPath);
}

LocalNamedExampleDatasetWriter::~LocalNamedExampleDatasetWriter() {
    if (!closed) {
        try {
            close();
        } catch (...) {
        }
    }
}

void LocalNamedExampleDatasetWriter::writeExample(ExampleType exampleType,
                                                  const std::map<std::string, TensorView> &tensors,
                                                  const std::string &label,
                                                  const std::string &filename) {
    if (storageMode != StorageMode::SPLIT) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter writeExample requires split storage_mode.");
    }
    writePackedRecord(exampleType, tensors, label, filename);
}

void LocalNamedExampleDatasetWriter::writeIndexedExample(const std::map<std::string, TensorView> &tensors,
                                                         const std::string &label,
                                                         const std::string &filename) {
    if (storageMode != StorageMode::INDEXED) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter writeIndexedExample requires indexed storage_mode.");
    }
    writePackedRecord(ExampleType::TRAIN, tensors, label, filename);
}

void LocalNamedExampleDatasetWriter::writeIndexedExamples(const std::map<std::string, TensorBatchView> &tensors) {
    if (storageMode != StorageMode::INDEXED) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter writeIndexedExamples requires indexed storage_mode.");
    }
    validateWritable();
    const uint64_t count = validateTensorBatchMapExact(tensors);
    std::vector<uint8_t> records = packRecords(tensors, count);
    writePackedIndexedRecords(records.data(), count);
}

void LocalNamedExampleDatasetWriter::writePackedRecord(ExampleType exampleType,
                                                       const std::map<std::string, TensorView> &tensors,
                                                       const std::string &label,
                                                       const std::string &filename) {
    validateWritable();
    validateTensorMapExact(tensors);

    std::vector<uint8_t> record = packRecord(tensors);
    if (storageMode == StorageMode::INDEXED) {
        (void)label;
        (void)filename;
        writePackedIndexedRecords(record.data(), 1);
        return;
    }

    ensureCurrentShard();
    currentShard->writeExample(record.data(), label, filename, exampleType);
    shardEntries.back().increment(exampleType);

    switch (exampleType) {
        case ExampleType::TRAIN:
            totalTrainExamples += 1;
            break;
        case ExampleType::VALIDATE:
            totalValidateExamples += 1;
            break;
        case ExampleType::TEST:
            totalTestExamples += 1;
            break;
        default:
            throw std::runtime_error("Unsupported ExampleType value: " + std::to_string(static_cast<int>(exampleType)));
    }
}

void LocalNamedExampleDatasetWriter::writePackedIndexedRecords(const uint8_t *records, uint64_t count) {
    if (count == 0) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter writeIndexedExamples requires at least one example.");
    }
    if (records == nullptr) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter writeIndexedExamples received null records.");
    }
    if (expectedNumExamples.has_value() && count > expectedNumExamples.value() - numExamples()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter write would exceed expected_num_examples.");
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
                                                                                         "LocalNamedExampleDatasetWriter chunk offset")),
                                              toWrite,
                                              ExampleType::TRAIN);
        shardEntries.back().incrementBy(ExampleType::TRAIN, toWrite);
        totalTrainExamples += toWrite;
        consumed += toWrite;
    }
}

void LocalNamedExampleDatasetWriter::close() {
    if (closed) {
        return;
    }
    if (expectedNumExamples.has_value() && numExamples() != expectedNumExamples.value()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter wrote " + std::to_string(numExamples()) +
                                 " examples but expected_num_examples was " + std::to_string(expectedNumExamples.value()) + ".");
    }
    finalizeCurrentShard();
    writeManifest();
    closed = true;
}

bool LocalNamedExampleDatasetWriter::isClosed() const { return closed; }

const std::filesystem::path &LocalNamedExampleDatasetWriter::path() const { return datasetPath; }

std::filesystem::path LocalNamedExampleDatasetWriter::manifestPath() const { return datasetPath / MANIFEST_FILENAME; }

uint64_t LocalNamedExampleDatasetWriter::numExamples() const { return totalTrainExamples + totalValidateExamples + totalTestExamples; }

uint64_t LocalNamedExampleDatasetWriter::numExamples(ExampleType exampleType) const {
    switch (exampleType) {
        case ExampleType::TRAIN:
            return totalTrainExamples;
        case ExampleType::VALIDATE:
            return totalValidateExamples;
        case ExampleType::TEST:
            return totalTestExamples;
        default:
            break;
    }
    throw std::runtime_error("Unsupported ExampleType value: " + std::to_string(static_cast<int>(exampleType)));
}

const LocalNamedExampleLayout &LocalNamedExampleDatasetWriter::getLayout() const { return layout; }

LocalNamedExampleDatasetWriter::StorageMode LocalNamedExampleDatasetWriter::getStorageMode() const { return storageMode; }

std::optional<uint64_t> LocalNamedExampleDatasetWriter::getExpectedNumExamples() const { return expectedNumExamples; }

bool LocalNamedExampleDatasetWriter::getPreallocate() const { return preallocate; }

LocalNamedExampleDatasetWriter::StorageMode LocalNamedExampleDatasetWriter::storageModeFromString(std::string_view value) {
    if (value == STORAGE_MODE_SPLIT) {
        return StorageMode::SPLIT;
    }
    if (value == STORAGE_MODE_INDEXED) {
        return StorageMode::INDEXED;
    }
    throw std::runtime_error("Unsupported local named example dataset storage_mode: " + std::string(value));
}

const char *LocalNamedExampleDatasetWriter::storageModeToString(StorageMode storageMode) {
    switch (storageMode) {
        case StorageMode::SPLIT:
            return STORAGE_MODE_SPLIT;
        case StorageMode::INDEXED:
            return STORAGE_MODE_INDEXED;
        default:
            break;
    }
    throw std::runtime_error("Unsupported LocalNamedExampleDatasetWriter StorageMode value: " +
                             std::to_string(static_cast<int>(storageMode)));
}

LocalNamedExampleDatasetWriter::StorageMode LocalNamedExampleDatasetWriter::readStorageMode(const std::filesystem::path &manifestPath) {
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter failed to open manifest for reading: " + manifestPath.string());
    }
    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter failed while reading manifest: " + manifestPath.string());
    }
    if (!manifest.contains("storage_mode")) {
        return StorageMode::SPLIT;
    }
    return storageModeFromString(manifest.at("storage_mode").get<std::string>());
}

void LocalNamedExampleDatasetWriter::validateWritable() const {
    if (closed) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter is closed.");
    }
}

void LocalNamedExampleDatasetWriter::validateTensorMapExact(const std::map<std::string, TensorView> &tensors) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter tensor count " + std::to_string(tensors.size()) +
                                 " does not match layout tensor count " + std::to_string(layout.tensors().size()) + ".");
    }

    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter missing tensor: " + spec.name);
        }
        const TensorView &view = it->second;
        if (view.data == nullptr) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor '" + spec.name + "' has null data.");
        }
        if (view.dataType != spec.dataType) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor '" + spec.name + "' has wrong dtype.");
        }
        if (view.dimensions != spec.dimensions) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor '" + spec.name + "' shape " + shapeToString(view.dimensions) +
                                     " does not match layout shape " + shapeToString(spec.dimensions) + ".");
        }
        if (view.numBytes != spec.numBytes) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor '" + spec.name + "' byte count " +
                                     std::to_string(view.numBytes) + " does not match layout byte count " +
                                     std::to_string(spec.numBytes) + ".");
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
}

uint64_t LocalNamedExampleDatasetWriter::validateTensorBatchMapExact(const std::map<std::string, TensorBatchView> &tensors) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter tensor count " + std::to_string(tensors.size()) +
                                 " does not match layout tensor count " + std::to_string(layout.tensors().size()) + ".");
    }

    bool haveCount = false;
    uint64_t count = 0;
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter missing tensor: " + spec.name);
        }
        const TensorBatchView &view = it->second;
        if (view.data == nullptr) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batch '" + spec.name + "' has null data.");
        }
        if (view.dataType != spec.dataType) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batch '" + spec.name + "' has wrong dtype.");
        }
        if (view.dimensions.size() != spec.dimensions.size() + 1) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batch '" + spec.name + "' shape " +
                                     shapeToString(view.dimensions) + " must be [N, *layout_shape].");
        }
        if (view.dimensions.front() == 0) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batch '" + spec.name + "' must contain at least one example.");
        }
        if (!haveCount) {
            count = view.dimensions.front();
            haveCount = true;
        } else if (count != view.dimensions.front()) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batches must have the same leading dimension.");
        }
        std::vector<uint64_t> tensorShape(view.dimensions.begin() + 1, view.dimensions.end());
        if (tensorShape != spec.dimensions) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batch '" + spec.name + "' shape " +
                                     shapeToString(view.dimensions) + " does not match layout shape [N," +
                                     shapeToString(spec.dimensions) + "].");
        }
        const uint64_t expectedBytes = checkedMul(count, spec.numBytes, "LocalNamedExampleDatasetWriter tensor batch bytes");
        if (view.numBytes != expectedBytes) {
            throw std::runtime_error("LocalNamedExampleDatasetWriter tensor batch '" + spec.name + "' byte count " +
                                     std::to_string(view.numBytes) + " does not match expected byte count " +
                                     std::to_string(expectedBytes) + ".");
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
    return count;
}

std::vector<uint8_t> LocalNamedExampleDatasetWriter::packRecord(const std::map<std::string, TensorView> &tensors) const {
    std::vector<uint8_t> record(layout.recordSizeBytes(), 0);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const TensorView &view = tensors.at(spec.name);
        std::memcpy(record.data() + spec.offsetBytes, view.data, spec.numBytes);
    }
    return record;
}

std::vector<uint8_t> LocalNamedExampleDatasetWriter::packRecords(const std::map<std::string, TensorBatchView> &tensors,
                                                                 uint64_t count) const {
    std::vector<uint8_t> records(checkedMul(count, layout.recordSizeBytes(), "LocalNamedExampleDatasetWriter packed records"), 0);
    for (uint64_t row = 0; row < count; ++row) {
        uint8_t *record = records.data() + checkedMul(row, layout.recordSizeBytes(), "LocalNamedExampleDatasetWriter record offset");
        for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
            const TensorBatchView &view = tensors.at(spec.name);
            const uint8_t *source = static_cast<const uint8_t *>(view.data) + checkedMul(row, spec.numBytes,
                                                                                        "LocalNamedExampleDatasetWriter tensor row offset");
            std::memcpy(record + spec.offsetBytes, source, spec.numBytes);
        }
    }
    return records;
}

uint64_t LocalNamedExampleDatasetWriter::nextShardCapacity() const {
    if (storageMode != StorageMode::INDEXED || !expectedNumExamples.has_value()) {
        return examplesPerShard;
    }
    const uint64_t written = numExamples();
    if (written >= expectedNumExamples.value()) {
        return 0;
    }
    return std::min<uint64_t>(examplesPerShard, expectedNumExamples.value() - written);
}

void LocalNamedExampleDatasetWriter::ensureCurrentShard() {
    if (currentShard && shardEntries.back().remainingCapacity() > 0) {
        return;
    }

    finalizeCurrentShard();

    const uint64_t capacity = nextShardCapacity();
    if (capacity == 0) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter cannot create another shard because expected_num_examples has been reached.");
    }

    ShardManifestEntry entry;
    entry.filename = makeShardFilename(nextShardIndex++);
    entry.globalStart = numExamples();
    entry.capacityExamples = capacity;

    std::vector<std::string> allClasses;
    currentShard = std::make_unique<Shard>();
    if (storageMode == StorageMode::INDEXED) {
        currentShard->createCompactShard((datasetPath / entry.filename).string(),
                                         capacity,
                                         0,
                                         0,
                                         layout.recordSizeBytes(),
                                         layout.dataType(),
                                         allClasses,
                                         preallocate);
    } else {
        currentShard->createShard((datasetPath / entry.filename).string(),
                                  examplesPerShard,
                                  examplesPerShard,
                                  examplesPerShard,
                                  layout.recordSizeBytes(),
                                  layout.dataType(),
                                  0,
                                  allClasses,
                                  0);
    }
    shardEntries.push_back(std::move(entry));
}

void LocalNamedExampleDatasetWriter::finalizeCurrentShard() {
    if (currentShard) {
        currentShard->shrinkToFit();
        currentShard.reset();
    }
}

void LocalNamedExampleDatasetWriter::writeManifest() const {
    json root = layout.toJson();
    root["storage_mode"] = storageModeToString(storageMode);
    root["num_examples"] = numExamples();
    if (expectedNumExamples.has_value()) {
        root["expected_num_examples"] = expectedNumExamples.value();
    }
    root["preallocated"] = preallocate;
    root["example_type_counts"] = json{{"train", totalTrainExamples}, {"validate", totalValidateExamples}, {"test", totalTestExamples}};
    root["shards"] = json::array();

    for (const ShardManifestEntry &entry : shardEntries) {
        json shard = json{{"file", entry.filename},
                          {"num_examples", entry.totalExamples()},
                          {"capacity_examples", entry.capacityExamples},
                          {"num_bytes", entry.numBytes(layout.recordSizeBytes())},
                          {"example_type_counts",
                           json{{"train", entry.trainExamples},
                                {"validate", entry.validateExamples},
                                {"test", entry.testExamples}}}};
        if (storageMode == StorageMode::INDEXED) {
            shard["global_start"] = entry.globalStart;
        }
        root["shards"].push_back(std::move(shard));
    }

    std::ofstream out(manifestPath(), std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter failed to open manifest for writing: " + manifestPath().string());
    }
    out << root.dump(2) << '\n';
    if (!out.good()) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter failed while writing manifest: " + manifestPath().string());
    }
}
