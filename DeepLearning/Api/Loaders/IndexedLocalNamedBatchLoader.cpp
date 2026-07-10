#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <map>
#include <stdexcept>
#include <utility>

namespace {

std::shared_ptr<const Thor::LocalNamedDataset> openCompatibleDataset(
    const std::filesystem::path &datasetPath, const LocalNamedExampleLayout &requestedLayout) {
    std::shared_ptr<Thor::LocalNamedDataset> dataset = Thor::LocalNamedDataset::open(datasetPath);
    dataset->assertLayout(requestedLayout);
    return dataset;
}

Thor::DatasetSplitManifest makeSplitManifest(
    const std::shared_ptr<const Thor::LocalNamedDataset> &dataset,
    std::vector<uint64_t> trainIndices,
    std::vector<uint64_t> validateIndices,
    std::optional<std::vector<uint64_t>> testIndices) {
    if (dataset == nullptr) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader dataset must not be null.");
    }
    return Thor::DatasetSplitManifest(*dataset,
                                      std::move(trainIndices),
                                      std::move(validateIndices),
                                      std::move(testIndices));
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
    : IndexedLocalNamedBatchLoader(openCompatibleDataset(datasetPath, requestedLayout),
                                   std::move(trainIndices),
                                   std::move(validateIndices),
                                   std::move(testIndices),
                                   batchSize,
                                   batchQueueDepth,
                                   randomizeTrain,
                                   seed) {}

IndexedLocalNamedBatchLoader::IndexedLocalNamedBatchLoader(std::shared_ptr<const Thor::LocalNamedDataset> dataset,
                                                           std::vector<uint64_t> trainIndices,
                                                           std::vector<uint64_t> validateIndices,
                                                           std::optional<std::vector<uint64_t>> testIndices,
                                                           uint64_t batchSize,
                                                           uint64_t batchQueueDepth,
                                                           bool randomizeTrain,
                                                           std::optional<uint64_t> seed)
    : IndexedLocalNamedBatchLoader(dataset,
                                   makeSplitManifest(dataset,
                                                     std::move(trainIndices),
                                                     std::move(validateIndices),
                                                     std::move(testIndices)),
                                   Thor::BatchPolicy(batchSize, randomizeTrain, seed),
                                   batchQueueDepth) {}

IndexedLocalNamedBatchLoader::IndexedLocalNamedBatchLoader(std::shared_ptr<const Thor::LocalNamedDataset> dataset,
                                                           Thor::DatasetSplitManifest splits,
                                                           Thor::BatchPolicy batching,
                                                           uint64_t batchQueueDepth,
                                                           std::set<Thor::DatasetFieldId> requiredFieldIds)
    : dataset(std::move(dataset)),
      splitManifest(std::move(splits)),
      requiredFieldIds(std::move(requiredFieldIds)),
      batchQueueDepth(batchQueueDepth),
      randomizeTrain(batching.getRandomizeTrain()),
      seed(batching.getRandomSeed()) {
    if (this->dataset == nullptr) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader dataset must not be null.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader batch_queue_depth must be >= 1.");
    }

    splitManifest.validateAgainst(*this->dataset);
    if (this->requiredFieldIds.empty()) {
        for (const Thor::DatasetField& field : this->dataset->getSchema().getFields()) {
            this->requiredFieldIds.insert(field.id);
        }
    }
    for (Thor::DatasetFieldId fieldId : this->requiredFieldIds) {
        (void)this->dataset->getSchema().getField(fieldId);
    }
    this->batchSize = batching.getBatchSize();
    numDatasetExamples = this->dataset->getNumExamples();

    if (splitManifest.getTrain().empty()) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader train partition must contain at least one row index.");
    }

    trainAssembler = createAssembler(splitManifest.getTrain().getSharedIndices(), "train", randomizeTrain, seed);
    validateAssembler = createAssembler(splitManifest.getValidate().getSharedIndices(), "validate", false, std::nullopt);
    testAssembler = createAssembler(splitManifest.getTest().getSharedIndices(), "test", false, std::nullopt);
}

IndexedLocalNamedBatchLoader::~IndexedLocalNamedBatchLoader() = default;

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

std::unique_ptr<IndexedLocalNamedBatchAssembler> IndexedLocalNamedBatchLoader::createAssembler(
    std::shared_ptr<const std::vector<uint64_t>> indices,
    const char *splitName,
    bool randomized,
    std::optional<uint64_t> splitSeed) const {
    if (indices == nullptr || indices->empty()) {
        return nullptr;
    }
    validateIndices(*indices, splitName);
    return std::make_unique<IndexedLocalNamedBatchAssembler>(dataset->getReader(),
                                                             dataset->getLayout(),
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
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("IndexedNamedBatchSession has been cancelled.");
    }
    IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler == nullptr) {
        throw std::runtime_error("IndexedLocalNamedBatchLoader cannot get a batch from an empty split.");
    }

    std::map<std::string, ThorImplementation::Tensor> tensors;
    assembler->getBatch(tensors, batchNum);
    return batchFromTensorMap(std::move(tensors));
}

void IndexedLocalNamedBatchLoader::returnBatchBuffers(ExampleType exampleType, Batch &&batch) {
    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
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


bool IndexedLocalNamedBatchLoader::supportsDeviceDatasetMaterialization() const { return true; }

Thor::DatasetMaterializationDescription IndexedLocalNamedBatchLoader::describeDeviceDatasetMaterialization() const {
    return Thor::DatasetMaterializationDescription(
        dataset->getPath(), dataset->getId(), dataset->getSchema(), dataset->getLayout(), numDatasetExamples);
}

Thor::DeviceDatasetSessionDescription IndexedLocalNamedBatchLoader::describeDeviceDatasetSession() const {
    return Thor::DeviceDatasetSessionDescription(
        splitManifest, Thor::BatchPolicy(batchSize, randomizeTrain, seed), requiredFieldIds);
}

IndexedLocalNamedBatchAssemblerStats IndexedLocalNamedBatchLoader::getStatsSnapshot(ExampleType exampleType) {
    IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler != nullptr) {
        return assembler->getStatsSnapshot();
    }

    IndexedLocalNamedBatchAssemblerStats stats;
    stats.splitName = splitNameForStats(exampleType);
    stats.targetBatchQueueDepth = batchQueueDepth;
    stats.recordSizeBytes = dataset->getLayout().recordSizeBytes();
    stats.resolvedIoBackend = "empty";
    return stats;
}

const LocalNamedExampleLayout &IndexedLocalNamedBatchLoader::getLayout() const { return dataset->getLayout(); }

const std::filesystem::path &IndexedLocalNamedBatchLoader::getDatasetPath() const { return dataset->getPath(); }

uint64_t IndexedLocalNamedBatchLoader::getNumDatasetExamples() const { return numDatasetExamples; }

uint64_t IndexedLocalNamedBatchLoader::getBatchQueueDepth() const { return batchQueueDepth; }

bool IndexedLocalNamedBatchLoader::getRandomizeTrain() const { return randomizeTrain; }

std::optional<uint64_t> IndexedLocalNamedBatchLoader::getRandomSeed() const { return seed; }

const std::vector<uint64_t> &IndexedLocalNamedBatchLoader::getSplitIndices(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return splitManifest.getTrain().getIndices();
    }
    if (exampleType == ExampleType::VALIDATE) {
        return splitManifest.getValidate().getIndices();
    }
    if (exampleType == ExampleType::TEST) {
        return splitManifest.getTest().getIndices();
    }
    throw std::runtime_error("Unsupported ExampleType");
}

bool IndexedLocalNamedBatchLoader::hasExplicitTestSplit() const { return splitManifest.hasExplicitTestSplit(); }

void IndexedLocalNamedBatchLoader::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    trainAssembler.reset();
    validateAssembler.reset();
    testAssembler.reset();
}
