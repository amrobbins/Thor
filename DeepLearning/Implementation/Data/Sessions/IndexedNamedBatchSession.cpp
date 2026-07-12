#include "DeepLearning/Implementation/Data/Sessions/IndexedNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Sessions/IndexedNamedBatchSessionFactory.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Data/FileDatasetRuntimeAccess.h"

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

IndexedNamedBatchSession::IndexedNamedBatchSession(std::shared_ptr<const Thor::FileDataset> dataset,
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
        throw std::runtime_error("IndexedNamedBatchSession dataset must not be null.");
    }
    if (batchQueueDepth == 0) {
        throw std::runtime_error("IndexedNamedBatchSession batch_queue_depth must be >= 1.");
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


    trainAssembler = createAssembler(splitManifest.getSharedTrain(), "train", randomizeTrain, seed);
    validateAssembler = createAssembler(splitManifest.getSharedValidate(), "validate", false, std::nullopt);
    testAssembler = createAssembler(splitManifest.getSharedTest(), "test", false, std::nullopt);
}

IndexedNamedBatchSession::~IndexedNamedBatchSession() = default;

void IndexedNamedBatchSession::validateIndex(uint64_t index, const char *splitName) const {
    if (index >= numDatasetExamples) {
        throw std::runtime_error(std::string("IndexedNamedBatchSession ") + splitName +
                                 "_indices contains row index outside dataset row count.");
    }
}

void IndexedNamedBatchSession::validateIndices(const Thor::ExampleIndexSet &indices, const char *splitName) const {
    if (indices.isRangeBacked()) {
        for (const Thor::ExampleIndexRange &range : indices.getRanges()) {
            validateIndex(range.last(), splitName);
        }
        return;
    }
    for (uint64_t position = 0; position < indices.size(); ++position) {
        validateIndex(indices.at(position), splitName);
    }
}

std::unique_ptr<IndexedBatchAssembler> IndexedNamedBatchSession::createAssembler(
    std::shared_ptr<const Thor::ExampleIndexSet> indices,
    const char *splitName,
    bool randomized,
    std::optional<uint64_t> splitSeed) const {
    if (indices == nullptr || indices->empty()) {
        return nullptr;
    }
    validateIndices(*indices, splitName);
    const std::shared_ptr<IndexedDatasetReader> &reader =
        ThorImplementation::FileDatasetRuntimeAccess::reader(*dataset);
    const DatasetLayout &layout =
        ThorImplementation::FileDatasetRuntimeAccess::layout(*dataset);
    return std::make_unique<IndexedBatchAssembler>(
        reader,
        layout,
        std::move(indices),
        splitName,
        batchSize,
        batchQueueDepth,
        randomized,
        splitSeed);
}

IndexedBatchAssembler *IndexedNamedBatchSession::assemblerFor(ExampleType exampleType) {
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

const IndexedBatchAssembler *IndexedNamedBatchSession::assemblerFor(ExampleType exampleType) const {
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

Batch IndexedNamedBatchSession::acquireBatch(ExampleType exampleType, uint64_t &batchNum) {
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("IndexedNamedBatchSession has been cancelled.");
    }
    IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler == nullptr) {
        throw std::runtime_error("IndexedNamedBatchSession cannot get a batch from an empty split.");
    }

    std::map<std::string, ThorImplementation::Tensor> tensors;
    assembler->acquireBatch(tensors, batchNum);
    return batchFromTensorMap(std::move(tensors));
}

void IndexedNamedBatchSession::recycleBatch(ExampleType exampleType, Batch &&batch) {
    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
    IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler == nullptr) {
        throw std::runtime_error("IndexedNamedBatchSession cannot return buffers to an empty split.");
    }

    std::map<std::string, ThorImplementation::Tensor> tensors =
        denseTensorMapFromBatchOrThrow(batch, "IndexedNamedBatchSession returned batch");
    assembler->returnBuffers(tensors);
}

uint64_t IndexedNamedBatchSession::getNumBatchesPerEpoch(ExampleType exampleType) {
    const IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    return assembler == nullptr ? 0 : assembler->getNumBatchesPerEpoch();
}

uint64_t IndexedNamedBatchSession::getNumExamples(ExampleType exampleType) {
    const IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    return assembler == nullptr ? 0 : assembler->getNumExamples();
}

uint64_t IndexedNamedBatchSession::getNextBatchNum(ExampleType exampleType) {
    IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    return assembler == nullptr ? 0 : assembler->getNextBatchNum();
}



IndexedBatchAssemblerStats IndexedNamedBatchSession::getStatsSnapshot(ExampleType exampleType) {
    IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler != nullptr) {
        return assembler->getStatsSnapshot();
    }

    IndexedBatchAssemblerStats stats;
    stats.splitName = splitNameForStats(exampleType);
    stats.targetBatchQueueDepth = batchQueueDepth;
    stats.recordSizeBytes =
        ThorImplementation::FileDatasetRuntimeAccess::layout(*dataset).recordSizeBytes();
    stats.resolvedIoBackend = "empty";
    return stats;
}

const DatasetLayout &IndexedNamedBatchSession::getLayout() const {
    return ThorImplementation::FileDatasetRuntimeAccess::layout(*dataset);
}

uint64_t IndexedNamedBatchSession::getNumDatasetExamples() const { return numDatasetExamples; }

uint64_t IndexedNamedBatchSession::getBatchQueueDepth() const { return batchQueueDepth; }

bool IndexedNamedBatchSession::getRandomizeTrain() const { return randomizeTrain; }

std::optional<uint64_t> IndexedNamedBatchSession::getRandomSeed() const { return seed; }

const Thor::ExampleIndexSet &IndexedNamedBatchSession::getSplitIndices(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return splitManifest.getTrain();
    }
    if (exampleType == ExampleType::VALIDATE) {
        return splitManifest.getValidate();
    }
    if (exampleType == ExampleType::TEST) {
        return splitManifest.getTest();
    }
    throw std::runtime_error("Unsupported ExampleType");
}

bool IndexedNamedBatchSession::hasExplicitTestSplit() const { return splitManifest.hasExplicitTestSplit(); }

void IndexedNamedBatchSession::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    trainAssembler.reset();
    validateAssembler.reset();
    testAssembler.reset();
}

namespace ThorImplementation {

std::shared_ptr<Thor::BatchSession> openIndexedNamedBatchSession(
    std::shared_ptr<const Thor::FileDataset> dataset,
    const Thor::DatasetSplitManifest &splits,
    const Thor::BatchPolicy &batching,
    uint64_t maxInFlightBatches,
    const std::set<Thor::DatasetFieldId> &requiredFieldIds) {
    return std::make_shared<IndexedNamedBatchSession>(
        std::move(dataset), splits, batching, maxInFlightBatches, requiredFieldIds);
}

}  // namespace ThorImplementation
