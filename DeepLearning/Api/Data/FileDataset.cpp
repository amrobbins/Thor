#include "DeepLearning/Api/Data/FileDataset.h"

#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Api/Loaders/IndexedNamedBatchSession.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {
namespace {

using nlohmann::json;
constexpr const char *DATASET_MANIFEST_FILENAME = "manifest.json";

DatasetSchema schemaFromLayout(const DatasetLayout &layout) {
    std::vector<DatasetField> fields;
    fields.reserve(layout.tensors().size() + 2 * layout.windowedTensors().size());
    DatasetFieldId nextId = 0;
    for (const DatasetLayout::TensorSpec &spec : layout.tensors()) {
        fields.push_back(DatasetField{.id = nextId++, .name = spec.name, .dataType = spec.dataType,
                                      .dimensions = spec.dimensions, .kind = DatasetFieldKind::DENSE});
    }
    for (const DatasetLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
        fields.push_back(DatasetField{.id = nextId++, .name = spec.name, .dataType = spec.dataType,
                                      .dimensions = spec.dimensions, .kind = DatasetFieldKind::WINDOWED});
        if (spec.maskName.has_value()) {
            fields.push_back(DatasetField{.id = nextId++, .name = spec.maskName.value(),
                                          .dataType = ThorImplementation::DataType::UINT8,
                                          .dimensions = {spec.windowLength()}, .kind = DatasetFieldKind::WINDOW_MASK});
        }
    }
    return DatasetSchema(std::move(fields));
}

DatasetId readIndexedDatasetIdentity(const std::filesystem::path &manifestPath) {
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("FileDataset failed to open manifest: " + manifestPath.string());
    }
    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("FileDataset failed while reading manifest: " + manifestPath.string());
    }
    if (!manifest.contains("storage_mode")) {
        throw std::runtime_error(
            "FileDataset rejected a legacy split dataset manifest without storage_mode. "
            "Rewrite the dataset with DatasetWriter and provide splits through DatasetSplitManifest.");
    }
    const std::string storageMode = manifest.at("storage_mode").get<std::string>();
    if (storageMode != DatasetWriter::STORAGE_MODE_INDEXED) {
        throw std::runtime_error(
            "FileDataset rejected legacy split dataset storage_mode='" + storageMode +
            "'. Rewrite the dataset with DatasetWriter and provide splits through DatasetSplitManifest.");
    }
    if (manifest.contains("dataset_id")) {
        return DatasetId(manifest.at("dataset_id").get<std::string>());
    }

    // Indexed manifests written before persisted dataset identities remain
    // readable until the explicit manifest-identity cleanup patch. Scope the
    // compatibility identity to the canonical manifest path so independent
    // copies cannot alias a residency-cache entry.
    const std::string stableMaterial =
        std::filesystem::absolute(manifestPath).lexically_normal().string() + "\n" + manifest.dump();
    return DatasetId::fromStableMaterial(stableMaterial);
}

}  // namespace

std::shared_ptr<FileDataset> FileDataset::open(const std::filesystem::path &datasetPath) {
    const std::filesystem::path manifestPath = datasetPath / DATASET_MANIFEST_FILENAME;
    DatasetId id = readIndexedDatasetIdentity(manifestPath);
    std::shared_ptr<IndexedLocalNamedExampleReader> reader = IndexedLocalNamedExampleReader::openDataset(datasetPath);
    DatasetSchema schema = schemaFromLayout(reader->getLayout());
    return std::shared_ptr<FileDataset>(
        new FileDataset(datasetPath, std::move(id), std::move(schema), std::move(reader)));
}

FileDataset::FileDataset(std::filesystem::path datasetPath,
                         DatasetId id,
                         DatasetSchema schema,
                         std::shared_ptr<IndexedLocalNamedExampleReader> reader)
    : datasetPath(std::move(datasetPath)), id(std::move(id)), schema(std::move(schema)), reader(std::move(reader)) {
    if (this->reader == nullptr) {
        throw std::runtime_error("FileDataset requires a reader.");
    }
}

uint64_t FileDataset::getNumExamples() const { return reader->getNumExamples(); }

const DatasetLayout &FileDataset::getLayout() const { return reader->getLayout(); }

void FileDataset::assertSchema(const DatasetSchema &expectedSchema) const {
    if (schema != expectedSchema) {
        throw std::runtime_error("FileDataset schema does not match the expected schema.");
    }
}

void FileDataset::assertLayout(const DatasetLayout &expectedLayout) const {
    getLayout().validateRequestedLayoutExact(expectedLayout);
}

std::shared_ptr<BatchSession> FileDataset::openBatchSession(
    const DatasetSplitManifest &splits,
    const BatchPolicy &batching,
    const DatasetAccessPolicy &,
    uint64_t maxInFlightBatches,
    const std::set<DatasetFieldId> &requiredFieldIds) const {
    std::shared_ptr<const FileDataset> self =
        std::dynamic_pointer_cast<const FileDataset>(shared_from_this());
    if (self == nullptr) {
        throw std::runtime_error("FileDataset must be owned by std::shared_ptr before opening a session.");
    }
    return std::make_shared<IndexedNamedBatchSession>(
        std::move(self), splits, batching, maxInFlightBatches, requiredFieldIds);
}

}  // namespace Thor
