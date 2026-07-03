#include "DeepLearning/Api/Loaders/LocalNamedBatchLoader.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <utility>

using ThorImplementation::Tensor;
using json = nlohmann::json;

namespace {

const char *exampleTypeName(ExampleType exampleType) {
    switch (exampleType) {
        case ExampleType::TRAIN:
            return "train";
        case ExampleType::VALIDATE:
            return "validate";
        case ExampleType::TEST:
            return "test";
        default:
            break;
    }
    return "unknown";
}

}  // namespace

LocalNamedBatchLoader::LocalNamedBatchLoader(std::filesystem::path datasetPath,
                                             LocalNamedExampleLayout requestedLayout,
                                             uint64_t batchSize,
                                             uint64_t batchQueueDepth,
                                             bool randomizeTrain,
                                             std::optional<uint64_t> seed)
    : datasetPath(std::move(datasetPath)) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(batchQueueDepth > 0);
    this->batchSize = batchSize;

    const std::filesystem::path manifestPath = this->datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME;
    layout = LocalNamedExampleLayout::readManifest(manifestPath);
    layout.validateRequestedLayoutExact(requestedLayout);

    const std::vector<std::string> shardFilenames = readShardFilenames(manifestPath);
    if (shardFilenames.empty()) {
        throw std::runtime_error("LocalNamedBatchLoader dataset manifest contains no shards: " + manifestPath.string());
    }

    for (const std::string &filename : shardFilenames) {
        auto shard = std::make_shared<Shard>();
        shard->openShard((this->datasetPath / filename).string());
        if (shard->getExampleSizeInBytes() != layout.recordSizeBytes()) {
            throw std::runtime_error("LocalNamedBatchLoader shard record size does not match manifest for shard: " + filename);
        }
        if (shard->getDataType() != layout.dataType()) {
            throw std::runtime_error("LocalNamedBatchLoader shard dtype does not match manifest for shard: " + filename);
        }
        shards.push_back(std::move(shard));
    }

    batchAssemblerTrain = maybeCreateAssembler(ExampleType::TRAIN, batchQueueDepth, randomizeTrain, seed);
    batchAssemblerValidate = maybeCreateAssembler(ExampleType::VALIDATE, batchQueueDepth, false, std::nullopt);
    batchAssemblerTest = maybeCreateAssembler(ExampleType::TEST, batchQueueDepth, false, std::nullopt);
}

std::vector<std::string> LocalNamedBatchLoader::readShardFilenames(const std::filesystem::path &manifestPath) {
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("LocalNamedBatchLoader failed to open manifest for reading: " + manifestPath.string());
    }

    json manifest;
    in >> manifest;
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("LocalNamedBatchLoader failed while reading manifest: " + manifestPath.string());
    }

    if (!manifest.contains("shards") || !manifest.at("shards").is_array()) {
        throw std::runtime_error("LocalNamedBatchLoader manifest shards field must be an array: " + manifestPath.string());
    }

    std::vector<std::string> filenames;
    filenames.reserve(manifest.at("shards").size());
    for (const json &entry : manifest.at("shards")) {
        if (entry.is_string()) {
            filenames.push_back(entry.get<std::string>());
        } else if (entry.is_object()) {
            filenames.push_back(entry.at("file").get<std::string>());
        } else {
            throw std::runtime_error("LocalNamedBatchLoader manifest shard entry must be a string or object.");
        }
        if (filenames.back().empty()) {
            throw std::runtime_error("LocalNamedBatchLoader manifest contains an empty shard filename.");
        }
    }
    return filenames;
}

std::shared_ptr<LocalNamedBatchAssembler> LocalNamedBatchLoader::maybeCreateAssembler(ExampleType exampleType,
                                                                                      uint64_t batchQueueDepth,
                                                                                      bool randomizeExamples,
                                                                                      std::optional<uint64_t> seed) {
    uint64_t totalExamples = 0;
    for (const std::shared_ptr<Shard> &shard : shards) {
        totalExamples += shard->getNumExamples(exampleType);
    }
    if (totalExamples == 0) {
        return nullptr;
    }

    return std::make_shared<LocalNamedBatchAssembler>(shards, exampleType, layout, batchSize, batchQueueDepth, randomizeExamples, seed);
}

LocalNamedBatchAssembler &LocalNamedBatchLoader::assemblerOrThrow(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN && batchAssemblerTrain) {
        return *batchAssemblerTrain;
    }
    if (exampleType == ExampleType::VALIDATE && batchAssemblerValidate) {
        return *batchAssemblerValidate;
    }
    if (exampleType == ExampleType::TEST && batchAssemblerTest) {
        return *batchAssemblerTest;
    }
    throw std::runtime_error(std::string("LocalNamedBatchLoader has no examples for split: ") + exampleTypeName(exampleType));
}

const LocalNamedBatchAssembler &LocalNamedBatchLoader::assemblerOrThrow(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN && batchAssemblerTrain) {
        return *batchAssemblerTrain;
    }
    if (exampleType == ExampleType::VALIDATE && batchAssemblerValidate) {
        return *batchAssemblerValidate;
    }
    if (exampleType == ExampleType::TEST && batchAssemblerTest) {
        return *batchAssemblerTest;
    }
    throw std::runtime_error(std::string("LocalNamedBatchLoader has no examples for split: ") + exampleTypeName(exampleType));
}

Batch LocalNamedBatchLoader::getBatch(ExampleType exampleType, uint64_t &batchNum) {
    std::map<std::string, Tensor> tensorMap;
    assemblerOrThrow(exampleType).getBatch(tensorMap, batchNum);
    return batchFromTensorMap(std::move(tensorMap));
}

void LocalNamedBatchLoader::returnBatchBuffers(ExampleType exampleType, Batch &&batch) {
    std::map<std::string, Tensor> tensorMap = denseTensorMapFromBatchOrThrow(batch, "LocalNamedBatchLoader::returnBatchBuffers");
    assemblerOrThrow(exampleType).returnBuffers(tensorMap);
}

uint64_t LocalNamedBatchLoader::getNumBatchesPerEpoch(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN && batchAssemblerTrain) {
        return batchAssemblerTrain->getNumBatchesPerEpoch();
    }
    if (exampleType == ExampleType::VALIDATE && batchAssemblerValidate) {
        return batchAssemblerValidate->getNumBatchesPerEpoch();
    }
    if (exampleType == ExampleType::TEST && batchAssemblerTest) {
        return batchAssemblerTest->getNumBatchesPerEpoch();
    }
    return 0;
}

uint64_t LocalNamedBatchLoader::getNumExamples(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN && batchAssemblerTrain) {
        return batchAssemblerTrain->getNumExamples();
    }
    if (exampleType == ExampleType::VALIDATE && batchAssemblerValidate) {
        return batchAssemblerValidate->getNumExamples();
    }
    if (exampleType == ExampleType::TEST && batchAssemblerTest) {
        return batchAssemblerTest->getNumExamples();
    }
    return 0;
}

uint64_t LocalNamedBatchLoader::getNextBatchNum(ExampleType exampleType) {
    return assemblerOrThrow(exampleType).getNextBatchNum();
}

const LocalNamedExampleLayout &LocalNamedBatchLoader::getLayout() const { return layout; }

const std::filesystem::path &LocalNamedBatchLoader::getDatasetPath() const { return datasetPath; }
