#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;
using json = nlohmann::json;

namespace {

uint64_t batchesFor(uint64_t numExamples, uint64_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    return (numExamples + batchSize - 1) / batchSize;
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

    openDataset(requestedLayout);

    initializeSplit(train, std::move(trainIndices), "train", randomizeTrain, seed);
    initializeSplit(validate, std::move(validateIndices), "validate", false, std::nullopt);
    if (explicitTestSplit) {
        initializeSplit(test, std::move(testIndices.value()), "test", false, std::nullopt);
    } else {
        initializeSplit(test, validate.indices, "test", false, std::nullopt);
    }
}

IndexedLocalNamedBatchLoader::~IndexedLocalNamedBatchLoader() {
    closeSplitQueues(train);
    closeSplitQueues(validate);
    closeSplitQueues(test);
}

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
        expectedGlobalStart += entry.numExamples;
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
        numDatasetExamples += entry.numExamples;
    }

    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        std::vector<uint64_t> dimensions;
        dimensions.reserve(spec.dimensions.size() + 1);
        dimensions.push_back(batchSize);
        dimensions.insert(dimensions.end(), spec.dimensions.begin(), spec.dimensions.end());
        batchTensorDescriptors.emplace(spec.name, TensorDescriptor(spec.dataType, dimensions));
    }
}

void IndexedLocalNamedBatchLoader::validateIndex(uint64_t index, const char *splitName) const {
    if (index >= numDatasetExamples) {
        throw std::runtime_error(std::string("IndexedLocalNamedBatchLoader ") + splitName +
                                 "_indices contains row index outside dataset row count.");
    }
}

void IndexedLocalNamedBatchLoader::initializeSplit(Split &split,
                                                   std::vector<uint64_t> indices,
                                                   const char *splitName,
                                                   bool randomized,
                                                   std::optional<uint64_t> splitSeed) {
    if (indices.empty()) {
        if (std::string(splitName) == "train") {
            throw std::runtime_error("IndexedLocalNamedBatchLoader train_indices must contain at least one row index.");
        }

        split.indices = std::move(indices);
        split.nextBatchNum = 0;
        return;
    }

    for (uint64_t index : indices) {
        validateIndex(index, splitName);
    }
    split.indices = std::move(indices);
    split.nextBatchNum = 0;
    if (randomized) {
        split.randomizer = std::make_unique<FullPeriodRandom>(split.indices.size(), false);
        if (splitSeed.has_value()) {
            split.randomizer->reseed(splitSeed.value());
        }
    }
    initializeSplitQueues(split);
}

void IndexedLocalNamedBatchLoader::initializeSplitQueues(Split &split) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        auto queue = std::make_unique<AsyncTensorQueue>(batchQueueDepth, batchTensorDescriptors.at(spec.name), cpuPlacement);
        queue->open();
        split.queues.emplace(spec.name, std::move(queue));
    }
}

void IndexedLocalNamedBatchLoader::closeSplitQueues(Split &split) {
    for (auto &entry : split.queues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    split.queues.clear();
}

IndexedLocalNamedBatchLoader::Split &IndexedLocalNamedBatchLoader::mutableSplit(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return train;
    }
    if (exampleType == ExampleType::VALIDATE) {
        return validate;
    }
    if (exampleType == ExampleType::TEST) {
        return test;
    }
    throw std::runtime_error("Unsupported ExampleType");
}

const IndexedLocalNamedBatchLoader::Split &IndexedLocalNamedBatchLoader::immutableSplit(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return train;
    }
    if (exampleType == ExampleType::VALIDATE) {
        return validate;
    }
    if (exampleType == ExampleType::TEST) {
        return test;
    }
    throw std::runtime_error("Unsupported ExampleType");
}

void IndexedLocalNamedBatchLoader::loadGlobalRecord(uint64_t globalExampleIndex, std::vector<uint8_t> &record) {
    validateIndex(globalExampleIndex, "global");
    const auto it = std::upper_bound(shardGlobalStarts.begin(), shardGlobalStarts.end(), globalExampleIndex);
    THOR_THROW_IF_FALSE(it != shardGlobalStarts.begin());
    const uint64_t shardIndex = static_cast<uint64_t>(std::distance(shardGlobalStarts.begin(), it) - 1);
    THOR_THROW_IF_FALSE(shardIndex < shards.size());
    const uint64_t localIndex = globalExampleIndex - shardGlobalStarts.at(shardIndex);
    THOR_THROW_IF_FALSE(localIndex < shardTrainCounts.at(shardIndex));

    record.resize(layout.recordSizeBytes());
    std::string label;
    std::string filename;
    shards.at(shardIndex)->loadExample(record.data(), label, filename, ExampleType::TRAIN, localIndex);
}

Batch IndexedLocalNamedBatchLoader::getBatch(ExampleType exampleType, uint64_t &batchNum) {
    Split &split = mutableSplit(exampleType);
    const uint64_t batchesPerEpoch = getNumBatchesPerEpoch(exampleType);
    if (batchesPerEpoch == 0) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader cannot get a batch from an empty split.");
    }
    if (batchNum >= batchesPerEpoch) {
        batchNum = split.nextBatchNum;
    }

    std::map<std::string, Tensor> loadedTensors;
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        Tensor tensor;
        const bool queueOpen = split.queues.at(spec.name)->getBufferToLoad(tensor);
        THOR_THROW_IF_FALSE(queueOpen);
        loadedTensors.emplace(spec.name, tensor);
    }

    const uint64_t firstExample = batchNum * batchSize;
    const bool useRandomizer = exampleType == ExampleType::TRAIN && randomizeTrain;
    std::vector<uint8_t> record;
    record.reserve(layout.recordSizeBytes());
    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        const uint64_t logicalIndex = useRandomizer ? split.randomizer->getRandomNumber() : (firstExample + slot) % split.indices.size();
        const uint64_t globalExampleIndex = split.indices.at(logicalIndex);
        loadGlobalRecord(globalExampleIndex, record);

        for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
            Tensor &tensor = loadedTensors.at(spec.name);
            std::memcpy(static_cast<uint8_t *>(tensor.getMemPtr()) + spec.numBytes * slot,
                        record.data() + spec.offsetBytes,
                        spec.numBytes);
        }
    }

    split.nextBatchNum = (batchNum + 1) % batchesPerEpoch;

    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        Tensor &tensor = loadedTensors.at(spec.name);
        const bool queueOpen = split.queues.at(spec.name)->bufferLoaded(tensor);
        THOR_THROW_IF_FALSE(queueOpen);
    }
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        Tensor &tensor = loadedTensors.at(spec.name);
        const bool queueOpen = split.queues.at(spec.name)->getBufferToUnload(tensor);
        THOR_THROW_IF_FALSE(queueOpen);
    }

    return batchFromTensorMap(std::move(loadedTensors));
}

void IndexedLocalNamedBatchLoader::validateReturnedBatchExact(const Split &split, const Batch &batch) const {
    if (batch.size() != layout.tensors().size()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader returned batch has unexpected tensor count.");
    }
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        if (!batch.contains(spec.name)) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader returned batch is missing tensor '" + spec.name + "'.");
        }
        const Tensor &tensor = batch.getTensor(spec.name);
        if (tensor.getDescriptor() != batchTensorDescriptors.at(spec.name)) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader returned tensor has wrong descriptor for: " + spec.name);
        }
        (void)split.queues.at(spec.name);
    }
}

void IndexedLocalNamedBatchLoader::returnBatchBuffers(ExampleType exampleType, Batch &&batch) {
    Split &split = mutableSplit(exampleType);
    validateReturnedBatchExact(split, batch);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const bool queueOpen = split.queues.at(spec.name)->bufferUnloaded(batch.getTensor(spec.name));
        THOR_THROW_IF_FALSE(queueOpen);
    }
}

uint64_t IndexedLocalNamedBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    return batchesFor(immutableSplit(exampleType).indices.size(), batchSize);
}

uint64_t IndexedLocalNamedBatchLoader::getNumExamples(ExampleType exampleType) {
    return static_cast<uint64_t>(immutableSplit(exampleType).indices.size());
}

uint64_t IndexedLocalNamedBatchLoader::getNextBatchNum(ExampleType exampleType) { return immutableSplit(exampleType).nextBatchNum; }

const LocalNamedExampleLayout &IndexedLocalNamedBatchLoader::getLayout() const { return layout; }

const std::filesystem::path &IndexedLocalNamedBatchLoader::getDatasetPath() const { return datasetPath; }

uint64_t IndexedLocalNamedBatchLoader::getNumDatasetExamples() const { return numDatasetExamples; }

uint64_t IndexedLocalNamedBatchLoader::getBatchQueueDepth() const { return batchQueueDepth; }

bool IndexedLocalNamedBatchLoader::getRandomizeTrain() const { return randomizeTrain; }

std::optional<uint64_t> IndexedLocalNamedBatchLoader::getRandomSeed() const { return seed; }

bool IndexedLocalNamedBatchLoader::hasExplicitTestSplit() const { return explicitTestSplit; }
