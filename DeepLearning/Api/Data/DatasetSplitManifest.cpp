#include "DeepLearning/Api/Data/DatasetSplitManifest.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <utility>

namespace Thor {
namespace {

using nlohmann::json;
constexpr const char *FORMAT = "thor.dataset_split_manifest.v1";

std::vector<uint64_t> readIndexArray(const json &partitions, const char *name) {
    if (!partitions.contains(name) || !partitions.at(name).is_array()) {
        throw std::runtime_error(std::string("DatasetSplitManifest partition '") + name + "' must be an array.");
    }
    return partitions.at(name).get<std::vector<uint64_t>>();
}

}  // namespace

ExampleIndexSet::ExampleIndexSet(std::vector<uint64_t> indices)
    : indices(std::make_shared<const std::vector<uint64_t>>(std::move(indices))) {}

std::shared_ptr<const ExampleIndexSet> DatasetSplitManifest::makeIndexSet(std::vector<uint64_t> indices,
                                                                          uint64_t numExamples,
                                                                          const char *partitionName) {
    std::unordered_set<uint64_t> unique;
    unique.reserve(indices.size());
    for (uint64_t index : indices) {
        if (index >= numExamples) {
            throw std::runtime_error(std::string("DatasetSplitManifest ") + partitionName +
                                     " contains row index outside dataset row count.");
        }
        if (!unique.insert(index).second) {
            throw std::runtime_error(std::string("DatasetSplitManifest ") + partitionName +
                                     " contains duplicate row index " + std::to_string(index) + ".");
        }
    }
    return std::make_shared<const ExampleIndexSet>(std::move(indices));
}

DatasetSplitManifest::DatasetSplitManifest(const NamedDataset &dataset,
                                           std::vector<uint64_t> trainIndices,
                                           std::vector<uint64_t> validateIndices,
                                           std::optional<std::vector<uint64_t>> testIndices)
    : DatasetSplitManifest(dataset.getId(),
                           dataset.getNumExamples(),
                           std::move(trainIndices),
                           std::move(validateIndices),
                           std::move(testIndices)) {}

DatasetSplitManifest::DatasetSplitManifest(DatasetId datasetId,
                                           uint64_t numExamples,
                                           std::vector<uint64_t> trainIndices,
                                           std::vector<uint64_t> validateIndices,
                                           std::optional<std::vector<uint64_t>> testIndices)
    : datasetId(std::move(datasetId)),
      numExamples(numExamples),
      train(makeIndexSet(std::move(trainIndices), numExamples, "train partition")),
      validate(makeIndexSet(std::move(validateIndices), numExamples, "validate partition")),
      explicitTestSplit(testIndices.has_value()) {
    if (explicitTestSplit) {
        test = makeIndexSet(std::move(testIndices.value()), numExamples, "test partition");
    } else {
        test = validate;
    }
}

void DatasetSplitManifest::validateAgainst(const NamedDataset &dataset) const {
    if (dataset.getId() != datasetId) {
        throw std::runtime_error("DatasetSplitManifest belongs to a different dataset identity.");
    }
    if (dataset.getNumExamples() != numExamples) {
        throw std::runtime_error("DatasetSplitManifest dataset row count does not match the opened dataset.");
    }
}

void DatasetSplitManifest::save(const std::filesystem::path &path) const {
    json manifest = {
        {"format", FORMAT},
        {"dataset_id", datasetId.str()},
        {"num_examples", numExamples},
        {"partitions",
         {{"train", train->getIndices()},
          {"validate", validate->getIndices()},
          {"test", explicitTestSplit ? json(test->getIndices()) : json{{"alias", "validate"}}}}},
    };

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("DatasetSplitManifest failed to open for writing: " + path.string());
    }
    out << manifest.dump(2) << '\n';
    if (!out.good()) {
        throw std::runtime_error("DatasetSplitManifest failed while writing: " + path.string());
    }
}

DatasetSplitManifest DatasetSplitManifest::load(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("DatasetSplitManifest failed to open: " + path.string());
    }

    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("DatasetSplitManifest failed while reading: " + path.string());
    }
    if (manifest.value("format", std::string()) != FORMAT) {
        throw std::runtime_error("DatasetSplitManifest has an unsupported format.");
    }

    DatasetId datasetId(manifest.at("dataset_id").get<std::string>());
    const uint64_t numExamples = manifest.at("num_examples").get<uint64_t>();
    const json &partitions = manifest.at("partitions");
    std::vector<uint64_t> trainIndices = readIndexArray(partitions, "train");
    std::vector<uint64_t> validateIndices = readIndexArray(partitions, "validate");

    std::optional<std::vector<uint64_t>> testIndices;
    if (!partitions.contains("test")) {
        throw std::runtime_error("DatasetSplitManifest is missing test partition metadata.");
    }
    const json &test = partitions.at("test");
    if (test.is_object()) {
        if (test.value("alias", std::string()) != "validate") {
            throw std::runtime_error("DatasetSplitManifest test alias must reference validate.");
        }
    } else if (test.is_array()) {
        testIndices = test.get<std::vector<uint64_t>>();
    } else {
        throw std::runtime_error("DatasetSplitManifest test partition must be an array or validate alias.");
    }

    return DatasetSplitManifest(std::move(datasetId),
                                numExamples,
                                std::move(trainIndices),
                                std::move(validateIndices),
                                std::move(testIndices));
}

bool DatasetSplitManifest::operator==(const DatasetSplitManifest &rhs) const {
    return datasetId == rhs.datasetId && numExamples == rhs.numExamples && *train == *rhs.train &&
           *validate == *rhs.validate && *test == *rhs.test && explicitTestSplit == rhs.explicitTestSplit;
}

}  // namespace Thor
