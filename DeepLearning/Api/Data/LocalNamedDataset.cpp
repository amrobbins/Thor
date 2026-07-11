#include "DeepLearning/Api/Data/LocalNamedDataset.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {
namespace {

using nlohmann::json;
constexpr const char *LOCAL_NAMED_MANIFEST_FILENAME = "manifest.json";

DatasetSchema schemaFromLayout(const LocalNamedExampleLayout &layout) {
    std::vector<DatasetField> fields;
    fields.reserve(layout.tensors().size() + 2 * layout.windowedTensors().size());
    DatasetFieldId nextId = 0;
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        fields.push_back(DatasetField{.id = nextId++, .name = spec.name, .dataType = spec.dataType,
                                      .dimensions = spec.dimensions, .kind = DatasetFieldKind::DENSE});
    }
    for (const LocalNamedExampleLayout::WindowedTensorSpec &spec : layout.windowedTensors()) {
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

DatasetId readDatasetIdentity(const std::filesystem::path &manifestPath) {
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("LocalNamedDataset failed to open manifest: " + manifestPath.string());
    }
    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("LocalNamedDataset failed while reading manifest: " + manifestPath.string());
    }
    if (manifest.contains("dataset_id")) {
        return DatasetId(manifest.at("dataset_id").get<std::string>());
    }

    // Legacy manifests predate persisted dataset identities. Scope their
    // deterministic compatibility identity to the canonical manifest path so
    // two independent copies with the same schema/counts cannot alias a future
    // residency cache entry.
    const std::string stableMaterial =
        std::filesystem::absolute(manifestPath).lexically_normal().string() + "\n" + manifest.dump();
    return DatasetId::fromStableMaterial(stableMaterial);
}

}  // namespace

std::shared_ptr<LocalNamedDataset> LocalNamedDataset::open(const std::filesystem::path &datasetPath) {
    const std::filesystem::path manifestPath = datasetPath / LOCAL_NAMED_MANIFEST_FILENAME;
    DatasetId id = readDatasetIdentity(manifestPath);
    std::shared_ptr<IndexedLocalNamedExampleReader> reader = IndexedLocalNamedExampleReader::openDataset(datasetPath);
    DatasetSchema schema = schemaFromLayout(reader->getLayout());
    return std::shared_ptr<LocalNamedDataset>(
        new LocalNamedDataset(datasetPath, std::move(id), std::move(schema), std::move(reader)));
}

LocalNamedDataset::LocalNamedDataset(std::filesystem::path datasetPath,
                                     DatasetId id,
                                     DatasetSchema schema,
                                     std::shared_ptr<IndexedLocalNamedExampleReader> reader)
    : datasetPath(std::move(datasetPath)), id(std::move(id)), schema(std::move(schema)), reader(std::move(reader)) {
    if (this->reader == nullptr) {
        throw std::runtime_error("LocalNamedDataset requires a reader.");
    }
}

uint64_t LocalNamedDataset::getNumExamples() const { return reader->getNumExamples(); }

void LocalNamedDataset::assertSchema(const DatasetSchema &expectedSchema) const {
    if (schema != expectedSchema) {
        throw std::runtime_error("LocalNamedDataset schema does not match the expected schema.");
    }
}

void LocalNamedDataset::assertLayout(const LocalNamedExampleLayout &expectedLayout) const {
    getLayout().validateRequestedLayoutExact(expectedLayout);
}

}  // namespace Thor
