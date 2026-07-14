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
    recyclerThread = std::thread(&IndexedNamedBatchSession::recyclerMain, this);
}

IndexedNamedBatchSession::~IndexedNamedBatchSession() { cancel(); }

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
    throwIfRecyclerFailed();
    IndexedBatchAssembler *assembler = assemblerFor(exampleType);
    if (assembler == nullptr) {
        throw std::runtime_error("IndexedNamedBatchSession cannot get a batch from an empty split.");
    }

    std::map<std::string, ThorImplementation::Tensor> tensors;
    assembler->acquireBatch(tensors, batchNum);
    auto sourceTensors = std::make_shared<std::map<std::string, ThorImplementation::Tensor>>(
        std::move(tensors));
    Batch batch = batchFromTensorMap(*sourceTensors);

    std::set<std::string> fieldNames;
    for (const auto &[name, tensor] : *sourceTensors) {
        (void)tensor;
        fieldNames.insert(name);
    }

    std::shared_ptr<IndexedNamedBatchSession> sharedSelf =
        std::dynamic_pointer_cast<IndexedNamedBatchSession>(shared_from_this());
    THOR_THROW_IF_FALSE(sharedSelf != nullptr);
    std::weak_ptr<IndexedNamedBatchSession> weakSelf = sharedSelf;
    Thor::BatchSourceOwner sourceOwner(
        [weakSelf, exampleType, sourceTensors](std::vector<Event> consumedEvents) mutable {
            if (std::shared_ptr<IndexedNamedBatchSession> session = weakSelf.lock()) {
                session->enqueueReturnedBuffers(
                    exampleType,
                    std::move(sourceTensors),
                    std::move(consumedEvents));
                return;
            }
            // A session normally remains alive through BatchLease ownership. If
            // teardown has already broken that relationship, still keep host
            // buffers alive until all asynchronous input copies have completed.
            try {
                for (Event &event : consumedEvents) {
                    event.synchronize();
                }
            } catch (...) {
            }
        });
    addBatchSourceResource(batch, std::move(fieldNames), std::move(sourceOwner));
    return batch;
}

void IndexedNamedBatchSession::recycleBatch(ExampleType exampleType, Batch &&batch) {
    if (batch.ownsSourceResourceLifecycle()) {
        // The source owner, rather than BatchLease destruction, returns the
        // assembler buffers after every NetworkInput upload has consumed them.
        // Batch copies retain source references but do not own this lifecycle,
        // so malformed copies continue through the exact legacy validation path.
        THOR_THROW_IF_FALSE(batch.allFieldsHaveSourceReferences());
        (void)denseTensorMapFromBatchOrThrow(
            batch,
            "IndexedNamedBatchSession returned source-tracked batch");
        batch.clear();
        return;
    }

    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
    throwIfRecyclerFailed();
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
    throwIfRecyclerFailed();
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

void IndexedNamedBatchSession::enqueueReturnedBuffers(
    ExampleType exampleType,
    std::shared_ptr<std::map<std::string, ThorImplementation::Tensor>> tensors,
    std::vector<Event> consumedEvents) noexcept {
    if (tensors == nullptr) {
        return;
    }

    // A manually released batch, or any batch that was never submitted to a
    // NetworkInput, has no asynchronous source consumers. Return those buffers
    // synchronously so the observable assembler counters and ready-buffer pool
    // preserve their historical immediate-after-reset behavior.
    if (consumedEvents.empty()) {
        std::lock_guard<std::mutex> guard(recyclerMutex);
        if (recyclerStopping || cancelled.load(std::memory_order_acquire)) {
            return;
        }
        try {
            IndexedBatchAssembler *assembler = assemblerFor(exampleType);
            if (assembler != nullptr) {
                assembler->returnBuffers(*tensors);
            }
        } catch (...) {
            if (recyclerFailure == nullptr) {
                recyclerFailure = std::current_exception();
            }
        }
        return;
    }

    try {
        std::lock_guard<std::mutex> guard(recyclerMutex);
        if (!recyclerStopping) {
            pendingReturnedBuffers.push_back(PendingReturnedBuffers{
                exampleType,
                tensors,
                consumedEvents});
            recyclerNotEmpty.notify_one();
            return;
        }
    } catch (...) {
    }

    // The recycler is already shutting down. Do not destroy host source
    // tensors until outstanding asynchronous copies have stopped reading them.
    try {
        for (Event &event : consumedEvents) {
            event.synchronize();
        }
    } catch (...) {
    }
}

void IndexedNamedBatchSession::recyclerMain() noexcept {
    while (true) {
        PendingReturnedBuffers pending;
        {
            std::unique_lock<std::mutex> lock(recyclerMutex);
            recyclerNotEmpty.wait(lock, [&] {
                return recyclerStopping || !pendingReturnedBuffers.empty();
            });
            if (pendingReturnedBuffers.empty()) {
                return;
            }
            pending = std::move(pendingReturnedBuffers.front());
            pendingReturnedBuffers.pop_front();
        }

        try {
            for (Event &event : pending.consumedEvents) {
                event.synchronize();
            }
            if (!cancelled.load(std::memory_order_acquire)) {
                IndexedBatchAssembler *assembler = assemblerFor(pending.exampleType);
                if (assembler != nullptr) {
                    assembler->returnBuffers(*pending.tensors);
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(recyclerMutex);
            if (recyclerFailure == nullptr) {
                recyclerFailure = std::current_exception();
            }
        }
    }
}

void IndexedNamedBatchSession::stopRecycler() noexcept {
    {
        std::lock_guard<std::mutex> guard(recyclerMutex);
        recyclerStopping = true;
    }
    recyclerNotEmpty.notify_all();
    if (recyclerThread.joinable()) {
        recyclerThread.join();
    }
}

void IndexedNamedBatchSession::throwIfRecyclerFailed() const {
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> guard(recyclerMutex);
        failure = recyclerFailure;
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}

void IndexedNamedBatchSession::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    stopRecycler();
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
