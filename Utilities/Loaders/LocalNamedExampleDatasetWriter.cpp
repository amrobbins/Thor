#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <iomanip>
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

void LocalNamedExampleDatasetWriter::ShardManifestEntry::increment(ExampleType exampleType) {
    switch (exampleType) {
        case ExampleType::TRAIN:
            trainExamples += 1;
            return;
        case ExampleType::VALIDATE:
            validateExamples += 1;
            return;
        case ExampleType::TEST:
            testExamples += 1;
            return;
        default:
            break;
    }
    throw std::runtime_error("Unsupported ExampleType value: " + std::to_string(static_cast<int>(exampleType)));
}

LocalNamedExampleDatasetWriter::LocalNamedExampleDatasetWriter(std::filesystem::path datasetPath,
                                                               LocalNamedExampleLayout layout,
                                                               uint64_t examplesPerShard)
    : datasetPath(std::move(datasetPath)),
      layout(std::move(layout)),
      examplesPerShard(examplesPerShard),
      closed(false),
      nextShardIndex(0),
      totalTrainExamples(0),
      totalValidateExamples(0),
      totalTestExamples(0) {
    this->layout.validate();
    if (this->examplesPerShard == 0) {
        throw std::runtime_error("LocalNamedExampleDatasetWriter examples_per_shard must be non-zero.");
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
    validateWritable();
    validateTensorMapExact(tensors);

    std::vector<uint8_t> record = packRecord(tensors);
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

void LocalNamedExampleDatasetWriter::close() {
    if (closed) {
        return;
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

std::vector<uint8_t> LocalNamedExampleDatasetWriter::packRecord(const std::map<std::string, TensorView> &tensors) const {
    std::vector<uint8_t> record(layout.recordSizeBytes(), 0);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const TensorView &view = tensors.at(spec.name);
        std::memcpy(record.data() + spec.offsetBytes, view.data, spec.numBytes);
    }
    return record;
}

void LocalNamedExampleDatasetWriter::ensureCurrentShard() {
    if (currentShard && shardEntries.back().totalExamples() < examplesPerShard) {
        return;
    }

    finalizeCurrentShard();

    ShardManifestEntry entry;
    entry.filename = makeShardFilename(nextShardIndex++);

    std::vector<std::string> allClasses;
    currentShard = std::make_unique<Shard>();
    currentShard->createShard((datasetPath / entry.filename).string(),
                              examplesPerShard,
                              examplesPerShard,
                              examplesPerShard,
                              layout.recordSizeBytes(),
                              layout.dataType(),
                              0,
                              allClasses,
                              0);
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
    root["num_examples"] = numExamples();
    root["example_type_counts"] = json{{"train", totalTrainExamples}, {"validate", totalValidateExamples}, {"test", totalTestExamples}};
    root["shards"] = json::array();

    for (const ShardManifestEntry &entry : shardEntries) {
        root["shards"].push_back(json{{"file", entry.filename},
                                      {"num_examples", entry.totalExamples()},
                                      {"example_type_counts",
                                       json{{"train", entry.trainExamples},
                                            {"validate", entry.validateExamples},
                                            {"test", entry.testExamples}}}});
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
