#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace {

uint64_t checkedAddUint64(uint64_t left, uint64_t right, const char *context) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::runtime_error(std::string(context) + " would overflow uint64_t.");
    }
    return left + right;
}

const char *splitNameForStats(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return "train";
    }
    if (exampleType == ExampleType::VALIDATE) {
        return "validate";
    }
    if (exampleType == ExampleType::TEST) {
        return "test";
    }
    return "unknown";
}

}  // namespace

IndexedLocalNamedBatchLoader::IndexedLocalNamedBatchLoader(std::filesystem::path datasetPath,
                                                           LocalNamedExampleLayout requestedLayout,
                                                           std::vector<uint64_t> trainIndices,
                                                           std::vector<uint64_t> validateIndices,
                                                           std::optional<std::vector<uint64_t>> testIndices,
                                                           uint64_t batchSize,
                                                           uint64_t batchQueueDepth,
                                                           bool randomizeTrain,
                                                           std::optional<uint64_t> seed)
    : datasetPath(std::move(datasetPath)),
      batchQueueDepth(batchQueueDepth),
      randomizeTrain(randomizeTrain),
      seed(seed),
      explicitTestSplit(testIndices.has_value()) {
    if (batchSize == 0) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader batch_size must be >= 1.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader batch_queue_depth must be >= 1.");
    }
    if (!randomizeTrain && seed.has_value()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader random_seed requires randomize_train=true.");
    }
    this->batchSize = batchSize;

    std::vector<uint64_t> resolvedTestIndices;
    if (explicitTestSplit) {
        resolvedTestIndices = std::move(testIndices.value());
    } else {
        resolvedTestIndices = validateIndices;
    }

    openDataset(requestedLayout);

    if (trainIndices.empty()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader train_indices must contain at least one row index.");
    }

    trainAssembler = createAssembler(std::move(trainIndices), "train", randomizeTrain, seed);
    validateAssembler = createAssembler(std::move(validateIndices), "validate", false, std::nullopt);
    testAssembler = createAssembler(std::move(resolvedTestIndices), "test", false, std::nullopt);
}

IndexedLocalNamedBatchLoader::~IndexedLocalNamedBatchLoader() = default;

std::vector<IndexedLocalNamedBatchLoader::IndexedShardManifestEntry> IndexedLocalNamedBatchLoader::readIndexedShardManifestEntries(
    const std::filesystem::path &manifestPath) {
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader failed to open manifest for reading: " + manifestPath.string());
    }

    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader failed while reading manifest: " + manifestPath.string());
    }

    const LocalNamedExampleDatasetWriter::StorageMode storageMode = LocalNamedExampleDatasetWriter::readStorageMode(manifestPath);
    if (storageMode != LocalNamedExampleDatasetWriter::StorageMode::INDEXED) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader requires an indexed local named dataset manifest.");
    }

    if (!manifest.contains("shards") || !manifest.at("shards").is_array()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader manifest shards field must be an array: " + manifestPath.string());
    }

    std::vector<IndexedShardManifestEntry> entries;
    entries.reserve(manifest.at("shards").size());
    uint64_t expectedGlobalStart = 0;
    for (const json &entryJson : manifest.at("shards")) {
        if (!entryJson.is_object()) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader indexed manifest shard entries must be objects.");
        }

        IndexedShardManifestEntry entry;
        entry.filename = entryJson.at("file").get<std::string>();
        entry.globalStart = entryJson.at("global_start").get<uint64_t>();
        entry.numExamples = entryJson.at("num_examples").get<uint64_t>();
        if (entry.filename.empty()) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader manifest contains an empty shard filename.");
        }
        if (entry.numExamples == 0) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader indexed shard entries must contain at least one example.");
        }
        if (entry.globalStart != expectedGlobalStart) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader indexed shard global_start values must be contiguous.");
        }
        expectedGlobalStart = checkedAddUint64(
            expectedGlobalStart, entry.numExamples, "IndexedLocalNamedBatchLoader indexed manifest row count");
        entries.push_back(std::move(entry));
    }

    if (entries.empty()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader dataset manifest contains no shards: " + manifestPath.string());
    }
    if (manifest.contains("num_examples") && manifest.at("num_examples").get<uint64_t>() != expectedGlobalStart) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader manifest num_examples does not match indexed shard ranges.");
    }

    return entries;
}

void IndexedLocalNamedBatchLoader::openDataset(const LocalNamedExampleLayout &requestedLayout) {
    const std::filesystem::path manifestPath = datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME;
    layout = LocalNamedExampleLayout::readManifest(manifestPath);
    layout.validateRequestedLayoutExact(requestedLayout);

    const std::vector<IndexedShardManifestEntry> shardEntries = readIndexedShardManifestEntries(manifestPath);

    for (const IndexedShardManifestEntry &entry : shardEntries) {
        auto shard = std::make_shared<Shard>();
        shard->openShard((datasetPath / entry.filename).string());
        if (shard->getExampleSizeInBytes() != layout.recordSizeBytes()) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader shard record size does not match manifest for shard: " + entry.filename);
        }
        if (shard->getDataType() != layout.dataType()) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader shard dtype does not match manifest for shard: " + entry.filename);
        }
        if (shard->getNumExamples(ExampleType::TRAIN) != entry.numExamples) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader shard TRAIN count does not match indexed manifest for shard: " +
                                     entry.filename);
        }
        if (shard->getNumExamples(ExampleType::VALIDATE) != 0 || shard->getNumExamples(ExampleType::TEST) != 0) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader indexed shards must not contain validate/test records: " + entry.filename);
        }

        shards.push_back(std::move(shard));
        shardGlobalStarts.push_back(entry.globalStart);
        shardTrainCounts.push_back(entry.numExamples);
        numDatasetExamples = checkedAddUint64(
            numDatasetExamples, entry.numExamples, "IndexedLocalNamedBatchLoader dataset row count");
    }
}

void IndexedLocalNamedBatchLoader::validateIndex(uint64_t index, const char *splitName) const {
    if (index >= numDatasetExamples) {
        throw std::runtime_error(std::string("IndexedLocalNamedBatchLoader ") + splitName +
                                 "_indices contains row index outside dataset row count.");
    }
}

void IndexedLocalNamedBatchLoader::validateIndices(const std::vector<uint64_t> &indices, const char *splitName) const {
    for (uint64_t index : indices) {
        validateIndex(index, splitName);
    }
}

std::unique_ptr<IndexedLocalNamedBatchAssembler> IndexedLocalNamedBatchLoader::createAssembler(std::vector<uint64_t> indices,
                                                                                                const char *splitName,
                                                                                                bool randomized,
                                                                                                std::optional<uint64_t> splitSeed) const {
    if (indices.empty()) {
        return nullptr;
    }
    validateIndices(indices, splitName);
    return std::make_unique<IndexedLocalNamedBatchAssembler>(shards,
                                                             shardGlobalStarts,
                                                             shardTrainCounts,
                                                             layout,
                                                             std::move(indices),
                                                             splitName,
                                                             batchSize,
                                                             batchQueueDepth,
                                                             randomized,
                                                             splitSeed);
}

IndexedLocalNamedBatchAssembler *IndexedLocalNamedBatchLoader::assemblerFor(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return trainAssembler.get();
    }
    if (exampleType == ExampleType::VALIDATE) {
        return validateAssembler.get();
    }
    if (exampleType == ExampleType::TEST) {
        return testAssembler.get();
    }
    throw std::runtime_error("Unsupported ExampleType");
}

const IndexedLocalNamedBatchAssembler *IndexedLocalNamedBatchLoader::assemblerFor(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return trainAssembler.get();
    }
    if (exampleType == ExampleType::VALIDATE) {
        return validateAssembler.get();
    }
    if (exampleType == ExampleType::TEST) {
        return testAssembler.get();
    }
    throw std::runtime_error("Unsupported ExampleType");
}

Batch IndexedLocalNamedBatchLoader::getBatch(ExampleType exampleType, uint64_t &batchNum) {
    IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler == nullptr) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader cannot get a batch from an empty split.");
    }

    std::map<std::string, ThorImplementation::Tensor> tensors;
    assembler->getBatch(tensors, batchNum);
    return batchFromTensorMap(std::move(tensors));
}

void IndexedLocalNamedBatchLoader::returnBatchBuffers(ExampleType exampleType, Batch &&batch) {
    IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler == nullptr) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader cannot return buffers to an empty split.");
    }

    std::map<std::string, ThorImplementation::Tensor> tensors =
        denseTensorMapFromBatchOrThrow(batch, "IndexedLocalNamedBatchLoader returned batch");
    assembler->returnBuffers(tensors);
}

uint64_t IndexedLocalNamedBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    const IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    return assembler == nullptr ? 0 : assembler->getNumBatchesPerEpoch();
}

uint64_t IndexedLocalNamedBatchLoader::getNumExamples(ExampleType exampleType) {
    const IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    return assembler == nullptr ? 0 : assembler->getNumExamples();
}

uint64_t IndexedLocalNamedBatchLoader::getNextBatchNum(ExampleType exampleType) {
    IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    return assembler == nullptr ? 0 : assembler->getNextBatchNum();
}

IndexedLocalNamedBatchAssemblerStats IndexedLocalNamedBatchLoader::getStatsSnapshot(ExampleType exampleType) {
    IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler != nullptr) {
        return assembler->getStatsSnapshot();
    }

    IndexedLocalNamedBatchAssemblerStats stats;
    stats.splitName = splitNameForStats(exampleType);
    stats.targetBatchQueueDepth = batchQueueDepth;
    stats.recordSizeBytes = layout.recordSizeBytes();
    stats.resolvedIoBackend = "empty";
    return stats;
}

const LocalNamedExampleLayout &IndexedLocalNamedBatchLoader::getLayout() const { return layout; }

const std::filesystem::path &IndexedLocalNamedBatchLoader::getDatasetPath() const { return datasetPath; }

uint64_t IndexedLocalNamedBatchLoader::getNumDatasetExamples() const { return numDatasetExamples; }

uint64_t IndexedLocalNamedBatchLoader::getBatchQueueDepth() const { return batchQueueDepth; }

bool IndexedLocalNamedBatchLoader::getRandomizeTrain() const { return randomizeTrain; }

std::optional<uint64_t> IndexedLocalNamedBatchLoader::getRandomSeed() const { return seed; }

bool IndexedLocalNamedBatchLoader::hasExplicitTestSplit() const { return explicitTestSplit; }
