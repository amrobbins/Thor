#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <map>
#include <stdexcept>
#include <utility>

namespace {

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

void IndexedLocalNamedBatchLoader::openDataset(const LocalNamedExampleLayout &requestedLayout) {
    reader = IndexedLocalNamedExampleReader::openDataset(datasetPath, requestedLayout);
    layout = reader->getLayout();
    numDatasetExamples = reader->getNumExamples();
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
    return std::make_unique<IndexedLocalNamedBatchAssembler>(reader,
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
