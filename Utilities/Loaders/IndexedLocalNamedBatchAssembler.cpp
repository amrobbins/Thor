#include "Utilities/Loaders/IndexedLocalNamedBatchAssembler.h"

#include "DeepLearning/Implementation/ThorError.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

uint64_t batchesFor(uint64_t numExamples, uint64_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    return (numExamples / batchSize) + ((numExamples % batchSize) == 0 ? 0 : 1);
}

uint64_t saturatedMultiplyUint64(uint64_t left, uint64_t right) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left * right;
}

uint64_t clampUint64(uint64_t value, uint64_t low, uint64_t high) {
    return std::max(low, std::min(value, high));
}

uint64_t computeShardReadQueueDepth(uint64_t exampleSizeInBytes) {
    constexpr uint64_t MIN_READS = 32;
    constexpr uint64_t MAX_READS = 1024;
    constexpr uint64_t TARGET_READ_BYTES = 8ull * 1024ull * 1024ull;
    const uint64_t safeExampleSize = std::max<uint64_t>(exampleSizeInBytes, 1);
    return clampUint64(TARGET_READ_BYTES / safeExampleSize, MIN_READS, MAX_READS);
}

uint64_t computeCompletedBatchQueueDepth(uint64_t batchQueueDepth) {
    constexpr uint64_t MIN_BATCHES = 1;
    constexpr uint64_t MAX_BATCHES = 4096;
    return clampUint64(batchQueueDepth, MIN_BATCHES, MAX_BATCHES);
}

uint64_t parsePositiveUint64Env(const char *primaryName, const char *secondaryName, uint64_t fallback) {
    const char *value = std::getenv(primaryName);
    if ((value == nullptr || value[0] == '\0') && secondaryName != nullptr) {
        value = std::getenv(secondaryName);
    }
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
        throw std::runtime_error(std::string(primaryName) + " must be a positive integer when set.");
    }
    return static_cast<uint64_t>(parsed);
}

uint64_t computeLoadWorkerThreadCount(uint64_t batchSize) {
    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const uint64_t conservativeHardwareDefault =
        hardwareThreads == 0 ? uint64_t{4} : std::max<uint64_t>(1, static_cast<uint64_t>(hardwareThreads) / 3);
    const uint64_t defaultWorkers = clampUint64(conservativeHardwareDefault, 1, std::min<uint64_t>(batchSize, 4));
    const uint64_t requestedWorkers = parsePositiveUint64Env(
        "THOR_INDEXED_LOCAL_NAMED_LOADER_LOAD_WORKERS", "THOR_INDEXED_LOCAL_NAMED_LOADER_WORKERS", defaultWorkers);
    return clampUint64(requestedWorkers, 1, batchSize);
}

uint64_t computeLoadWorkQueueDepth(uint64_t batchQueueDepth, uint64_t loadWorkerThreadCount) {
    constexpr uint64_t MIN_WORK_ITEMS = 1;
    constexpr uint64_t MAX_WORK_ITEMS = 4096;
    return clampUint64(saturatedMultiplyUint64(batchQueueDepth, saturatedMultiplyUint64(loadWorkerThreadCount, 2)),
                       MIN_WORK_ITEMS,
                       MAX_WORK_ITEMS);
}

uint32_t checkedQueueDepth(uint64_t depth, const char *context) {
    if (depth == 0 || depth > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(context) + " queue depth is outside uint32_t range.");
    }
    return static_cast<uint32_t>(depth);
}

bool statsLoggingEnabled() {
    static const bool enabled = [] {
        const char *specific = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_STATS");
        if (specific != nullptr && specific[0] != '\0') {
            return !(specific[0] == '0' && specific[1] == '\0');
        }
        const char *shared = std::getenv("THOR_TRAINING_QUEUE_DIAGNOSTICS");
        return shared != nullptr && shared[0] != '\0' && !(shared[0] == '0' && shared[1] == '\0');
    }();
    return enabled;
}

uint64_t statsLoggingEvery() {
    static const uint64_t every = [] {
        const char *value = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_STATS_EVERY");
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("THOR_TRAINING_QUEUE_DIAGNOSTICS_EVERY");
        }
        if (value == nullptr || value[0] == '\0') {
            return uint64_t{1};
        }
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || parsed == 0) {
            return uint64_t{1};
        }
        return static_cast<uint64_t>(parsed);
    }();
    return every;
}

bool shouldEmitStats(uint64_t index) {
    const uint64_t every = statsLoggingEvery();
    return index <= 3 || (every != 0 && (index % every) == 0);
}

}  // namespace

IndexedLocalNamedBatchAssembler::IndexedLocalNamedBatchAssembler(std::shared_ptr<IndexedLocalNamedExampleReader> reader,
                                                                 LocalNamedExampleLayout layout,
                                                                 std::vector<uint64_t> indices,
                                                                 std::string splitName,
                                                                 uint64_t batchSize,
                                                                 uint64_t batchQueueDepth,
                                                                 bool randomized,
                                                                 std::optional<uint64_t> seed)
    : reader(std::move(reader)),
      layout(std::move(layout)),
      indices(std::move(indices)),
      splitName(std::move(splitName)),
      batchSize(batchSize),
      batchQueueDepth(batchQueueDepth),
      shardReadQueueDepth(0),
      shardRequestQueueDepth(0),
      completedBatchQueueDepth(0),
      recordSizeBytes(this->reader == nullptr ? 0 : this->reader->getRecordSizeBytes()),
      batchesPerEpoch(batchesFor(this->indices.size(), batchSize)),
      numDatasetExamples(this->reader == nullptr ? 0 : this->reader->getNumExamples()),
      nextBatchNum(0),
      nextLogicalPosition(0),
      nextBatchOrdinal(0),
      nextPublishOrdinal(0),
      randomized(randomized),
      resolvedIoBackend("unresolved"),
      loadWorkerThreadCount(0),
      loadWorkQueueDepth(0),
      recordCopyThreadCount(0),
      recordBufferPoolDepth(0) {
    THOR_THROW_IF_FALSE(this->reader != nullptr);
    THOR_THROW_IF_FALSE(!this->indices.empty());
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(batchQueueDepth > 0);
    this->layout.validate();
    THOR_THROW_IF_FALSE(recordSizeBytes > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes == this->layout.recordSizeBytes());
    THOR_THROW_IF_FALSE(this->reader->getTensorCount() == this->layout.tensors().size());

    shardReadQueueDepth = computeShardReadQueueDepth(recordSizeBytes);
    loadWorkerThreadCount = computeLoadWorkerThreadCount(batchSize);
    loadWorkQueueDepth = computeLoadWorkQueueDepth(batchQueueDepth, loadWorkerThreadCount);
    shardRequestQueueDepth = loadWorkQueueDepth;
    completedBatchQueueDepth = computeCompletedBatchQueueDepth(batchQueueDepth);

    for (uint64_t index : this->indices) {
        validateGlobalIndex(index, this->splitName.c_str());
    }

    if (randomized) {
        randomizer = std::make_unique<FullPeriodRandom>(this->indices.size(), false);
        if (seed.has_value()) {
            randomizer->reseed(seed.value());
        }
    }

    layoutTensorOrdinals.reserve(this->layout.tensors().size());
    for (const LocalNamedExampleLayout::TensorSpec &spec : this->layout.tensors()) {
        std::vector<uint64_t> dimensions;
        dimensions.reserve(spec.dimensions.size() + 1);
        dimensions.push_back(batchSize);
        dimensions.insert(dimensions.end(), spec.dimensions.begin(), spec.dimensions.end());
        batchTensorDescriptors.emplace(spec.name, TensorDescriptor(spec.dataType, dimensions));
        layoutTensorOrdinals.push_back(this->reader->getLayoutTensorOrdinal(spec.name));
    }

    open();
}

IndexedLocalNamedBatchAssembler::~IndexedLocalNamedBatchAssembler() { close(); }

void IndexedLocalNamedBatchAssembler::open() {
    try {
        THOR_THROW_IF_FALSE(loadWorkerThreads.empty());
        THOR_THROW_IF_FALSE(batchTensorQueues.empty());

        loadWorkQueue.resize(checkedQueueDepth(loadWorkQueueDepth, "IndexedLocalNamedBatchAssembler load work"));
        loadWorkQueue.open();

        completedBatchQueue.resize(
            checkedQueueDepth(completedBatchQueueDepth, "IndexedLocalNamedBatchAssembler completed batch"));
        completedBatchQueue.open();

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
            auto queue = std::make_unique<AsyncTensorQueue>();
            queue->resize(batchQueueDepth, batchTensorDescriptors.at(spec.name), cpuPlacement);
            queue->open();
            batchTensorQueues.emplace(spec.name, std::move(queue));
        }

        batchNumQueue.resize(checkedQueueDepth(batchQueueDepth, "IndexedLocalNamedBatchAssembler batch number"));
        batchNumQueue.open();

        for (uint64_t i = 0; i < loadWorkerThreadCount; ++i) {
            loadWorkerThreads.emplace_back(&IndexedLocalNamedBatchAssembler::loadWorkerThread, this, i);
        }
        assemblerThread = std::thread(&IndexedLocalNamedBatchAssembler::batchAssemblerThread, this);
    } catch (...) {
        close();
        throw;
    }
}

void IndexedLocalNamedBatchAssembler::close() {
    loadWorkQueue.close();
    completedBatchQueue.close();
    for (auto &entry : batchTensorQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    batchNumQueue.close();

    for (std::thread &thread : loadWorkerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (assemblerThread.joinable()) {
        assemblerThread.join();
    }

    batchTensorQueues.clear();
    loadWorkerThreads.clear();
    {
        std::lock_guard<std::mutex> guard(pendingBatchesMutex);
        pendingBatches.clear();
    }
}

void IndexedLocalNamedBatchAssembler::recordWorkerException(std::exception_ptr exception) {
    {
        std::lock_guard<std::mutex> guard(workerExceptionMutex);
        if (workerException == nullptr) {
            workerException = exception;
        }
    }

    loadWorkQueue.close();
    completedBatchQueue.close();
    for (auto &entry : batchTensorQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    batchNumQueue.close();
}

void IndexedLocalNamedBatchAssembler::throwIfWorkerFailed() const {
    std::exception_ptr exception;
    {
        std::lock_guard<std::mutex> guard(workerExceptionMutex);
        exception = workerException;
    }
    if (exception != nullptr) {
        std::rethrow_exception(exception);
    }
}

void IndexedLocalNamedBatchAssembler::setResolvedIoBackend(const std::string &backendName) {
    std::lock_guard<std::mutex> guard(statsMutex);
    if (resolvedIoBackend == "unresolved" || resolvedIoBackend == backendName) {
        resolvedIoBackend = backendName;
    } else if (resolvedIoBackend.find(backendName) == std::string::npos) {
        resolvedIoBackend += "," + backendName;
    }
}

IndexedLocalNamedBatchAssemblerStats IndexedLocalNamedBatchAssembler::getStatsSnapshot() {
    IndexedLocalNamedBatchAssemblerStats stats;
    stats.splitName = splitName;
    stats.recordsRequested = statsRecordsRequested.load(std::memory_order_relaxed);
    stats.logicalRecordBytesRequested = statsLogicalRecordBytesRequested.load(std::memory_order_relaxed);
    stats.readCallsSubmitted = statsReadCallsSubmitted.load(std::memory_order_relaxed);
    stats.readBytesSubmitted = statsReadBytesSubmitted.load(std::memory_order_relaxed);
    stats.readCallsCompleted = statsReadCallsCompleted.load(std::memory_order_relaxed);
    stats.readBytesCompleted = statsReadBytesCompleted.load(std::memory_order_relaxed);
    stats.recordsCopied = statsRecordsCopied.load(std::memory_order_relaxed);
    stats.recordCopyBytes = statsRecordCopyBytes.load(std::memory_order_relaxed);
    stats.recordCopyMemcpyCalls = statsRecordCopyMemcpyCalls.load(std::memory_order_relaxed);
    stats.recordCopyActiveNanoseconds = statsRecordCopyActiveNanoseconds.load(std::memory_order_relaxed);
    stats.recordCopyPopWaitNanoseconds = statsRecordCopyPopWaitNanoseconds.load(std::memory_order_relaxed);
    stats.completedRecordQueuePushWaitNanoseconds = statsCompletedRecordQueuePushWaitNanoseconds.load(std::memory_order_relaxed);
    stats.copiedRecordQueuePushWaitNanoseconds = statsCopiedRecordQueuePushWaitNanoseconds.load(std::memory_order_relaxed);
    stats.recordBufferPoolCapacity = 0;
    stats.currentRecordBufferPoolDepth = 0;
    stats.batchesAssembled = statsBatchesAssembled.load(std::memory_order_relaxed);
    stats.batchesDelivered = statsBatchesDelivered.load(std::memory_order_relaxed);
    stats.batchBuffersReturned = statsBatchBuffersReturned.load(std::memory_order_relaxed);
    const int readyBatches = batchNumQueue.occupancy();
    stats.currentReadyBatches = readyBatches < 0 ? 0 : static_cast<uint64_t>(readyBatches);
    stats.currentPendingBatches = pendingBatchCount();
    stats.currentCompletedRecordQueueDepth = 0;
    const int completedBatches = completedBatchQueue.occupancy();
    stats.currentCopiedRecordQueueDepth = completedBatches < 0 ? 0 : static_cast<uint64_t>(completedBatches);
    stats.targetBatchQueueDepth = batchQueueDepth;
    stats.shardReadQueueDepth = shardReadQueueDepth;
    stats.shardRequestQueueDepth = shardRequestQueueDepth;
    stats.completedRecordQueueDepth = completedBatchQueueDepth;
    stats.recordCopyThreadCount = 0;
    stats.recordSizeBytes = recordSizeBytes;
    {
        std::lock_guard<std::mutex> guard(statsMutex);
        stats.resolvedIoBackend = resolvedIoBackend;
    }
    return stats;
}

void IndexedLocalNamedBatchAssembler::emitStatsIfEnabled(const char *event, uint64_t batchNum) {
    if (!statsLoggingEnabled()) {
        return;
    }

    const IndexedLocalNamedBatchAssemblerStats stats = getStatsSnapshot();
    std::fprintf(stderr,
                 "IndexedLocalNamedBatchLoader stats: event=%s split=%s batch=%lu "
                 "records_requested=%lu logical_bytes_requested=%lu read_calls_submitted=%lu "
                 "read_bytes_submitted=%lu read_calls_completed=%lu read_bytes_completed=%lu records_copied=%lu "
                 "copy_bytes=%lu copy_memcpy_calls=%lu copy_active_ns=%lu copy_wait_ns=%lu completed_push_wait_ns=%lu "
                 "completed_batch_push_wait_ns=%lu avg_copy_ns_per_record=%.1f avg_copy_calls_per_record=%.1f "
                 "read_amplification=%.6f planning_lead_records=%.0f batches_assembled=%lu batches_delivered=%lu "
                 "batch_buffers_returned=%lu ready_batches=%lu pending_batches=%lu completed_record_queue=%lu completed_batch_queue=%lu "
                 "record_buffer_pool=%lu/%lu queue_depth=%lu load_work_queue=%d/%lu load_workers=%lu copy_threads=%lu "
                 "resolved_io_backend=%s\n",
                 event,
                 stats.splitName.c_str(),
                 batchNum,
                 stats.recordsRequested,
                 stats.logicalRecordBytesRequested,
                 stats.readCallsSubmitted,
                 stats.readBytesSubmitted,
                 stats.readCallsCompleted,
                 stats.readBytesCompleted,
                 stats.recordsCopied,
                 stats.recordCopyBytes,
                 stats.recordCopyMemcpyCalls,
                 stats.recordCopyActiveNanoseconds,
                 stats.recordCopyPopWaitNanoseconds,
                 stats.completedRecordQueuePushWaitNanoseconds,
                 stats.copiedRecordQueuePushWaitNanoseconds,
                 stats.averageCopyNanosecondsPerRecord(),
                 stats.averageCopyMemcpyCallsPerRecord(),
                 stats.readAmplification(),
                 stats.planningLeadRecords(),
                 stats.batchesAssembled,
                 stats.batchesDelivered,
                 stats.batchBuffersReturned,
                 stats.currentReadyBatches,
                 stats.currentPendingBatches,
                 stats.currentCompletedRecordQueueDepth,
                 stats.currentCopiedRecordQueueDepth,
                 stats.currentRecordBufferPoolDepth,
                 stats.recordBufferPoolCapacity,
                 stats.targetBatchQueueDepth,
                 loadWorkQueue.occupancy(),
                 loadWorkQueueDepth,
                 loadWorkerThreadCount,
                 stats.recordCopyThreadCount,
                 stats.resolvedIoBackend.c_str());
    std::fflush(stderr);
}

void IndexedLocalNamedBatchAssembler::validateGlobalIndex(uint64_t index, const char *context) const {
    THOR_THROW_IF_FALSE(reader != nullptr);
    reader->validateGlobalIndex(index, context);
}

uint64_t IndexedLocalNamedBatchAssembler::nextLogicalSplitPosition() {
    if (randomized) {
        THOR_THROW_IF_FALSE(randomizer != nullptr);
        return randomizer->getRandomNumber();
    }

    const uint64_t logicalPosition = nextLogicalPosition;
    nextLogicalPosition = (nextLogicalPosition + 1) % indices.size();
    return logicalPosition;
}

void IndexedLocalNamedBatchAssembler::loadWorkerThread(uint64_t workerIndex) {
    try {
        loadWorkerThreadMain(workerIndex);
    } catch (...) {
        recordWorkerException(std::current_exception());
    }
}

void IndexedLocalNamedBatchAssembler::loadWorkerThreadMain(uint64_t workerIndex) {
    (void)workerIndex;

    auto session = reader->createSession(shardReadQueueDepth);

    auto flushReaderSessionStats = [&]() {
        IndexedLocalNamedExampleReaderSessionStats sessionStats = session->takeStats();
        if (sessionStats.readCallsSubmitted != 0) {
            statsReadCallsSubmitted.fetch_add(sessionStats.readCallsSubmitted, std::memory_order_relaxed);
            statsReadBytesSubmitted.fetch_add(sessionStats.readBytesSubmitted, std::memory_order_relaxed);
        }
        if (sessionStats.readCallsCompleted != 0) {
            statsReadCallsCompleted.fetch_add(sessionStats.readCallsCompleted, std::memory_order_relaxed);
            statsReadBytesCompleted.fetch_add(sessionStats.readBytesCompleted, std::memory_order_relaxed);
        }
        for (const std::string &backendName : sessionStats.resolvedIoBackends) {
            setResolvedIoBackend(backendName);
        }
    };

    auto markLoadChunkComplete = [&](const IndexedLocalNamedBatchLoadWork &work) -> bool {
        THOR_THROW_IF_FALSE(work.batchState != nullptr);
        const uint64_t completedChunks = work.batchState->completedLoadChunks.fetch_add(1, std::memory_order_acq_rel) + 1;
        THOR_THROW_IF_FALSE(completedChunks <= work.batchState->expectedLoadChunks);
        if (completedChunks != work.batchState->expectedLoadChunks) {
            return true;
        }

        IndexedLocalNamedCompletedBatch completed;
        completed.batchOrdinal = work.batchOrdinal;
        return completedBatchQueue.push(completed);
    };

    while (true) {
        IndexedLocalNamedBatchLoadWork work;
        if (!loadWorkQueue.pop(work)) {
            session->drain();
            flushReaderSessionStats();
            return;
        }
        THOR_THROW_IF_FALSE(work.batchState != nullptr);
        THOR_THROW_IF_FALSE(work.batchOrdinal == work.batchState->batchOrdinal);
        THOR_THROW_IF_FALSE(work.slotBegin < work.slotEnd);
        THOR_THROW_IF_FALSE(work.slotEnd <= work.batchState->expectedRecords);
        THOR_THROW_IF_FALSE(work.batchState->globalExampleIndices.size() == work.batchState->expectedRecords);
        THOR_THROW_IF_FALSE(work.batchState->tensorBasePointers.size() == reader->getTensorCount());

        for (uint64_t slot = work.slotBegin; slot < work.slotEnd; ++slot) {
            session->loadExampleInto(work.batchState->globalExampleIndices.at(slot), slot, work.batchState->tensorBasePointers);
        }

        session->drain();
        flushReaderSessionStats();
        if (!markLoadChunkComplete(work)) {
            return;
        }
    }
}

void IndexedLocalNamedBatchAssembler::batchAssemblerThread() {
    try {
        batchAssemblerThreadMain();
    } catch (...) {
        recordWorkerException(std::current_exception());
    }
}

void IndexedLocalNamedBatchAssembler::batchAssemblerThreadMain() {
    while (true) {
        while (pendingBatchCount() < batchQueueDepth) {
            // If there is already in-flight work, never block trying to acquire another
            // output tensor set. Ready batches and in-flight batches share the same
            // tensor queues. Blocking here can prevent the coordinator from publishing
            // completed batches and releasing pressure on the tensor queues.
            //
            // When there is no in-flight work, startNextBatch() is allowed to block;
            // that is the normal backpressure path when all ready batches are held by
            // the consumer and no returned buffers are currently available.
            if (pendingBatchCount() != 0 && !canStartNextBatchWithoutBlocking()) {
                break;
            }
            if (!startNextBatch()) {
                return;
            }
            publishCompletedBatches();
        }

        if (publishCompletedBatches()) {
            continue;
        }

        if (pendingBatchCount() == 0) {
            continue;
        }

        if (!waitForCompletedBatch()) {
            return;
        }
    }
}

bool IndexedLocalNamedBatchAssembler::canStartNextBatchWithoutBlocking() {
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        if (batchTensorQueues.at(spec.name)->isFull()) {
            return false;
        }
    }
    return true;
}

bool IndexedLocalNamedBatchAssembler::startNextBatch() {
    auto batchState = std::make_shared<IndexedLocalNamedBatchState>();
    batchState->batchOrdinal = nextBatchOrdinal++;
    batchState->batchNum = nextBatchNum;
    batchState->expectedRecords = batchSize;
    batchState->expectedLoadChunks = 1;
    batchState->completedLoadChunks.store(0, std::memory_order_relaxed);
    batchState->loadComplete = false;
    batchState->tensorBasePointers.assign(reader->getTensorCount(), nullptr);
    batchState->globalExampleIndices.reserve(batchSize);
    nextBatchNum = (nextBatchNum + 1) % batchesPerEpoch;

    for (uint64_t specIndex = 0; specIndex < layout.tensors().size(); ++specIndex) {
        const LocalNamedExampleLayout::TensorSpec &spec = layout.tensors().at(specIndex);
        Tensor tensor;
        if (!batchTensorQueues.at(spec.name)->getBufferToLoad(tensor)) {
            return false;
        }
        const uint64_t readerOrdinal = layoutTensorOrdinals.at(specIndex);
        THOR_THROW_IF_FALSE(readerOrdinal < batchState->tensorBasePointers.size());
        THOR_THROW_IF_FALSE(batchState->tensorBasePointers.at(readerOrdinal) == nullptr);
        batchState->tensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        batchState->tensors.emplace(spec.name, tensor);
    }
    for (uint8_t *basePointer : batchState->tensorBasePointers) {
        if (basePointer == nullptr) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler failed to bind every reader tensor ordinal to a batch tensor.");
        }
    }

    uint64_t localRecordsRequested = 0;
    uint64_t localLogicalRecordBytesRequested = 0;
    auto flushLocalRequestStats = [&]() {
        if (localRecordsRequested != 0) {
            statsRecordsRequested.fetch_add(localRecordsRequested, std::memory_order_relaxed);
            statsLogicalRecordBytesRequested.fetch_add(localLogicalRecordBytesRequested, std::memory_order_relaxed);
            localRecordsRequested = 0;
            localLogicalRecordBytesRequested = 0;
        }
    };

    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        const uint64_t logicalPosition = nextLogicalSplitPosition();
        const uint64_t globalExampleIndex = indices.at(logicalPosition);
        batchState->globalExampleIndices.push_back(globalExampleIndex);
        localRecordsRequested += 1;
        localLogicalRecordBytesRequested += recordSizeBytes;
    }
    flushLocalRequestStats();

    const uint64_t batchOrdinal = batchState->batchOrdinal;
    {
        std::lock_guard<std::mutex> guard(pendingBatchesMutex);
        auto [insertIt, inserted] = pendingBatches.emplace(batchOrdinal, batchState);
        THOR_THROW_IF_FALSE(inserted);
        (void)insertIt;
    }

    IndexedLocalNamedBatchLoadWork work;
    work.batchState = batchState.get();
    work.batchOrdinal = batchOrdinal;
    work.slotBegin = 0;
    work.slotEnd = batchSize;
    if (!pushLoadWorkWithDrain(work)) {
        return false;
    }
    return true;
}

bool IndexedLocalNamedBatchAssembler::pushLoadWorkWithDrain(const IndexedLocalNamedBatchLoadWork &work) {
    THOR_THROW_IF_FALSE(work.batchState != nullptr);
    THOR_THROW_IF_FALSE(work.slotBegin < work.slotEnd);

    while (!loadWorkQueue.tryPush(work)) {
        if (!loadWorkQueue.isOpen()) {
            return false;
        }
        if (publishCompletedBatches()) {
            continue;
        }

        // Load work is intentionally coarse grained: one queue item covers a
        // contiguous slot range, and the loader-owned worker does all direct
        // reads for that range.  If all workers are busy, yield rather than
        // blocking the assembler away from completed-batch publication.
        std::this_thread::yield();
    }
    return true;
}

bool IndexedLocalNamedBatchAssembler::waitForCompletedBatch() {
    IndexedLocalNamedCompletedBatch completed;
    if (!completedBatchQueue.pop(completed)) {
        return false;
    }
    if (!markBatchLoaded(completed.batchOrdinal)) {
        return false;
    }
    publishCompletedBatches();
    return true;
}

void IndexedLocalNamedBatchAssembler::markAvailableCompletedBatches() {
    IndexedLocalNamedCompletedBatch completed;
    while (completedBatchQueue.tryPop(completed)) {
        THOR_THROW_IF_FALSE(markBatchLoaded(completed.batchOrdinal));
    }
}

bool IndexedLocalNamedBatchAssembler::markBatchLoaded(uint64_t batchOrdinal) {
    std::lock_guard<std::mutex> guard(pendingBatchesMutex);
    auto batchIt = pendingBatches.find(batchOrdinal);
    if (batchIt == pendingBatches.end()) {
        return false;
    }
    THOR_THROW_IF_FALSE(batchIt->second != nullptr);
    batchIt->second->loadComplete = true;
    return true;
}

uint64_t IndexedLocalNamedBatchAssembler::pendingBatchCount() const {
    std::lock_guard<std::mutex> guard(pendingBatchesMutex);
    return static_cast<uint64_t>(pendingBatches.size());
}

bool IndexedLocalNamedBatchAssembler::publishCompletedBatches() {
    markAvailableCompletedBatches();
    bool publishedAny = false;
    while (true) {
        std::shared_ptr<IndexedLocalNamedBatchState> batchState;
        {
            std::lock_guard<std::mutex> mapGuard(pendingBatchesMutex);
            auto batchIt = pendingBatches.find(nextPublishOrdinal);
            if (batchIt == pendingBatches.end()) {
                return publishedAny;
            }
            batchState = batchIt->second;
            THOR_THROW_IF_FALSE(batchState != nullptr);

            if (!batchState->loadComplete) {
                return publishedAny;
            }
            pendingBatches.erase(batchIt);
        }

        for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
            const bool queueOpen = batchTensorQueues.at(spec.name)->bufferLoaded(batchState->tensors.at(spec.name));
            if (!queueOpen) {
                return publishedAny;
            }
        }
        if (!batchNumQueue.push(batchState->batchNum)) {
            return publishedAny;
        }
        const uint64_t assembled = statsBatchesAssembled.fetch_add(1, std::memory_order_relaxed) + 1;
        if (shouldEmitStats(assembled)) {
            emitStatsIfEnabled("assembled_batch", batchState->batchNum);
        }
        nextPublishOrdinal += 1;
        publishedAny = true;
    }
}

void IndexedLocalNamedBatchAssembler::getBatch(std::map<std::string, Tensor> &tensors, uint64_t &batchNum) {
    if (batchesPerEpoch == 0) {
        throw std::runtime_error("IndexedLocalNamedBatchAssembler cannot get a batch from an empty split.");
    }

    std::lock_guard<std::mutex> deliveryGuard(batchDeliveryMutex);
    throwIfWorkerFailed();
    tensors.clear();

    const bool batchNumQueueOpen = batchNumQueue.pop(batchNum);
    if (!batchNumQueueOpen) {
        throwIfWorkerFailed();
        THOR_THROW_IF_FALSE(batchNumQueueOpen);
    }

    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        Tensor tensor;
        const bool tensorQueueOpen = batchTensorQueues.at(spec.name)->getBufferToUnload(tensor);
        if (!tensorQueueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(tensorQueueOpen);
        }
        tensors.emplace(spec.name, tensor);
    }
    const uint64_t delivered = statsBatchesDelivered.fetch_add(1, std::memory_order_relaxed) + 1;
    if (shouldEmitStats(delivered)) {
        emitStatsIfEnabled("get_batch", batchNum);
    }
}

uint64_t IndexedLocalNamedBatchAssembler::getNextBatchNum() {
    std::lock_guard<std::mutex> deliveryGuard(batchDeliveryMutex);
    throwIfWorkerFailed();
    uint64_t batchNum = 0;
    const bool queueOpen = batchNumQueue.peek(batchNum);
    if (!queueOpen) {
        throwIfWorkerFailed();
        THOR_THROW_IF_FALSE(queueOpen);
    }
    return batchNum;
}

void IndexedLocalNamedBatchAssembler::validateReturnedTensorMapExact(const std::map<std::string, Tensor> &tensors) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("IndexedLocalNamedBatchAssembler returned tensor count does not match layout tensor count.");
    }

    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler missing returned tensor: " + spec.name);
        }
        THOR_THROW_IF_FALSE(it->second.isInitialized());
        if (it->second.getDescriptor() != batchTensorDescriptors.at(spec.name)) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler returned tensor has wrong descriptor for: " + spec.name);
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
}

void IndexedLocalNamedBatchAssembler::returnBuffers(const std::map<std::string, Tensor> &tensors) {
    std::lock_guard<std::mutex> returnGuard(returnBuffersMutex);
    throwIfWorkerFailed();
    validateReturnedTensorMapExact(tensors);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const bool queueOpen = batchTensorQueues.at(spec.name)->bufferUnloaded(tensors.at(spec.name));
        if (!queueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(queueOpen);
        }
    }
    const uint64_t returned = statsBatchBuffersReturned.fetch_add(1, std::memory_order_relaxed) + 1;
    if (shouldEmitStats(returned)) {
        emitStatsIfEnabled("return_batch", 0);
    }
}

uint64_t IndexedLocalNamedBatchAssembler::getNumBatchesPerEpoch() const { return batchesPerEpoch; }

uint64_t IndexedLocalNamedBatchAssembler::getNumExamples() const { return static_cast<uint64_t>(indices.size()); }
