#include "Utilities/Loaders/IndexedLocalNamedBatchAssembler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include "DeepLearning/Implementation/ThorError.h"

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

uint64_t clampUint64(uint64_t value, uint64_t low, uint64_t high) { return std::max(low, std::min(value, high)); }

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

uint64_t computeDefaultShardReadQueueDepth(uint64_t exampleSizeInBytes, uint64_t batchSize) {
    constexpr uint64_t MIN_READS = 32;
    constexpr uint64_t MAX_READS = 4096;
    constexpr uint64_t LEGACY_MAX_READS = 1024;
    constexpr uint64_t TARGET_READ_BYTES = 8ull * 1024ull * 1024ull;
    const uint64_t safeExampleSize = std::max<uint64_t>(exampleSizeInBytes, 1);
    const uint64_t byteTargetDepth = clampUint64(TARGET_READ_BYTES / safeExampleSize, MIN_READS, LEGACY_MAX_READS);

    // The indexed named reader uses one assembler worker per batch and the async
    // readv session owns a bounded pool of reusable iovec arrays per shard.
    // If this queue is smaller than the batch, the worker can exhaust the pool
    // while still in the submit loop and spend most of the batch load time
    // waiting for slot recycling.  Default to at least one batch worth of read
    // slots so the intended shape is submit whole batch first, then drain.
    return clampUint64(std::max<uint64_t>(byteTargetDepth, batchSize + 10), MIN_READS, MAX_READS);
}

uint64_t computeShardReadQueueDepth(uint64_t exampleSizeInBytes, uint64_t batchSize) {
    const uint64_t defaultDepth = computeDefaultShardReadQueueDepth(exampleSizeInBytes, batchSize);
    return parsePositiveUint64Env(
        "THOR_INDEXED_LOCAL_NAMED_LOADER_SHARD_READ_QUEUE_DEPTH", "THOR_INDEXED_LOCAL_NAMED_READER_SHARD_READ_QUEUE_DEPTH", defaultDepth);
}

uint64_t computeCompletedBatchQueueDepth(uint64_t batchQueueDepth) {
    constexpr uint64_t MIN_BATCHES = 1;
    constexpr uint64_t MAX_BATCHES = 4096;
    return clampUint64(batchQueueDepth, MIN_BATCHES, MAX_BATCHES);
}

uint64_t computeLoadWorkerThreadCount(uint64_t batchSize) {
    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const uint64_t conservativeHardwareDefault =
        hardwareThreads == 0 ? uint64_t{4} : std::max<uint64_t>(1, static_cast<uint64_t>(hardwareThreads) / 3);
    const uint64_t defaultWorkers = clampUint64(conservativeHardwareDefault, 1, std::min<uint64_t>(batchSize, 4));
    const uint64_t requestedWorkers =
        parsePositiveUint64Env("THOR_INDEXED_LOCAL_NAMED_LOADER_LOAD_WORKERS", "THOR_INDEXED_LOCAL_NAMED_LOADER_WORKERS", defaultWorkers);
    return clampUint64(requestedWorkers, 1, batchSize);
}

uint64_t computeLoadWorkQueueDepth(uint64_t batchQueueDepth, uint64_t loadWorkerThreadCount) {
    constexpr uint64_t MIN_WORK_ITEMS = 1;
    constexpr uint64_t MAX_WORK_ITEMS = 4096;
    return clampUint64(
        saturatedMultiplyUint64(batchQueueDepth, saturatedMultiplyUint64(loadWorkerThreadCount, 2)), MIN_WORK_ITEMS, MAX_WORK_ITEMS);
}

uint32_t checkedQueueDepth(uint64_t depth, const char *context) {
    if (depth == 0 || depth > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(context) + " queue depth is outside uint32_t range.");
    }
    return static_cast<uint32_t>(depth);
}

using SteadyClock = std::chrono::steady_clock;

bool diagnosticsTimingEnabled() {
    static const bool enabled = [] {
        const char *specific = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_DIAGNOSTICS");
        if (specific != nullptr && specific[0] != '\0') {
            return !(specific[0] == '0' && specific[1] == '\0');
        }
        const char *shared = std::getenv("THOR_TRAINING_QUEUE_DIAGNOSTICS");
        return shared != nullptr && shared[0] != '\0' && !(shared[0] == '0' && shared[1] == '\0');
    }();
    return enabled;
}

SteadyClock::time_point diagnosticNow() { return diagnosticsTimingEnabled() ? SteadyClock::now() : SteadyClock::time_point{}; }

uint64_t diagnosticElapsedNanoseconds(SteadyClock::time_point start) {
    if (!diagnosticsTimingEnabled()) {
        return 0;
    }
    const auto elapsed = SteadyClock::now() - start;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

uint64_t elapsedNanoseconds(SteadyClock::time_point start, SteadyClock::time_point end) {
    if (start == SteadyClock::time_point{} || end == SteadyClock::time_point{} || end < start) {
        return 0;
    }
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

bool statsLoggingEnabled() {
    static const bool enabled = [] {
        const char *specific = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_STATS");
        if (specific != nullptr && specific[0] != '\0') {
            return !(specific[0] == '0' && specific[1] == '\0');
        }
        const char *diagnostics = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_DIAGNOSTICS");
        if (diagnostics != nullptr && diagnostics[0] != '\0') {
            return !(diagnostics[0] == '0' && diagnostics[1] == '\0');
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
                                                                 DatasetLayout layout,
                                                                 std::shared_ptr<const Thor::ExampleIndexSet> indices,
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
      batchesPerEpoch(batchesFor(this->indices == nullptr ? 0 : this->indices->size(), batchSize)),
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
    THOR_THROW_IF_FALSE(this->indices != nullptr);
    THOR_THROW_IF_FALSE(!this->indices->empty());
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(batchQueueDepth > 0);
    this->layout.validate();
    THOR_THROW_IF_FALSE(recordSizeBytes == this->layout.recordSizeBytes());
    THOR_THROW_IF_FALSE(this->reader->getTensorCount() == this->layout.tensors().size());
    THOR_THROW_IF_FALSE(this->reader->getWindowedTensorCount() == this->layout.windowedTensors().size());

    shardReadQueueDepth = computeShardReadQueueDepth(recordSizeBytes, batchSize);
    loadWorkerThreadCount = computeLoadWorkerThreadCount(batchSize);
    loadWorkQueueDepth = computeLoadWorkQueueDepth(batchQueueDepth, loadWorkerThreadCount);
    shardRequestQueueDepth = loadWorkQueueDepth;
    completedBatchQueueDepth = computeCompletedBatchQueueDepth(batchQueueDepth);

    if (this->indices->isRangeBacked()) {
        for (const Thor::ExampleIndexRange &range : this->indices->getRanges()) {
            validateGlobalIndex(range.last(), this->splitName.c_str());
        }
    } else {
        for (uint64_t position = 0; position < this->indices->size(); ++position) {
            validateGlobalIndex(this->indices->at(position), this->splitName.c_str());
        }
    }

    if (randomized) {
        randomizer = std::make_unique<FullPeriodRandom>(this->indices->size(), false);
        if (seed.has_value()) {
            randomizer->reseed(seed.value());
        }
    }

    layoutTensorOrdinals.reserve(this->layout.tensors().size());
    for (const DatasetLayout::TensorSpec &spec : this->layout.tensors()) {
        std::vector<uint64_t> dimensions;
        dimensions.reserve(spec.dimensions.size() + 1);
        dimensions.push_back(batchSize);
        dimensions.insert(dimensions.end(), spec.dimensions.begin(), spec.dimensions.end());
        batchTensorDescriptors.emplace(spec.name, TensorDescriptor(spec.dataType, dimensions));
        layoutTensorOrdinals.push_back(this->reader->getLayoutTensorOrdinal(spec.name));
    }

    layoutWindowedTensorOrdinals.reserve(this->layout.windowedTensors().size());
    for (const DatasetLayout::WindowedTensorSpec &spec : this->layout.windowedTensors()) {
        std::vector<uint64_t> dimensions;
        dimensions.reserve(spec.dimensions.size() + 1);
        dimensions.push_back(batchSize);
        dimensions.insert(dimensions.end(), spec.dimensions.begin(), spec.dimensions.end());
        batchTensorDescriptors.emplace(spec.name, TensorDescriptor(spec.dataType, dimensions));
        layoutWindowedTensorOrdinals.push_back(this->reader->getLayoutWindowedTensorOrdinal(spec.name));
        if (spec.maskName.has_value()) {
            std::vector<uint64_t> maskDimensions{batchSize, spec.windowLength()};
            batchTensorDescriptors.emplace(spec.maskName.value(), TensorDescriptor(ThorImplementation::DataType::UINT8, maskDimensions));
        }
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

        completedBatchQueue.resize(checkedQueueDepth(completedBatchQueueDepth, "IndexedLocalNamedBatchAssembler completed batch"));
        completedBatchQueue.open();

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        for (const auto &entry : batchTensorDescriptors) {
            auto queue = std::make_unique<AsyncTensorQueue>();
            queue->resize(batchQueueDepth, entry.second, cpuPlacement);
            queue->open();
            batchTensorQueues.emplace(entry.first, std::move(queue));
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
    stats.windowedSourceReadCalls = statsWindowedSourceReadCalls.load(std::memory_order_relaxed);
    stats.windowedSourceReadBytes = statsWindowedSourceReadBytes.load(std::memory_order_relaxed);
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
    stats.loadWorkPopWaitNanoseconds = statsLoadWorkPopWaitNanoseconds.load(std::memory_order_relaxed);
    stats.loadWorkPopCalls = statsLoadWorkPopCalls.load(std::memory_order_relaxed);
    stats.loadWorkerBatches = statsLoadWorkerBatches.load(std::memory_order_relaxed);
    stats.loadWorkerActiveNanoseconds = statsLoadWorkerActiveNanoseconds.load(std::memory_order_relaxed);
    stats.loadWorkerReadSubmitNanoseconds = statsLoadWorkerReadSubmitNanoseconds.load(std::memory_order_relaxed);
    stats.loadWorkerReadDrainNanoseconds = statsLoadWorkerReadDrainNanoseconds.load(std::memory_order_relaxed);
    stats.loadWorkerCompletedBatchPushWaitNanoseconds = statsLoadWorkerCompletedBatchPushWaitNanoseconds.load(std::memory_order_relaxed);
    stats.readvSubmitNanoseconds = statsReadvSubmitNanoseconds.load(std::memory_order_relaxed);
    stats.readvSubmitBackpressureCount = statsReadvSubmitBackpressureCount.load(std::memory_order_relaxed);
    stats.readvSubmitBackpressureNanoseconds = statsReadvSubmitBackpressureNanoseconds.load(std::memory_order_relaxed);
    stats.readvCompletionWaitCalls = statsReadvCompletionWaitCalls.load(std::memory_order_relaxed);
    stats.readvCompletionWaitNanoseconds = statsReadvCompletionWaitNanoseconds.load(std::memory_order_relaxed);
    stats.readerDrainCalls = statsReaderDrainCalls.load(std::memory_order_relaxed);
    stats.readerDrainNanoseconds = statsReaderDrainNanoseconds.load(std::memory_order_relaxed);
    stats.readerDrainContextVisits = statsReaderDrainContextVisits.load(std::memory_order_relaxed);
    stats.readerDrainSubmitCalls = statsReaderDrainSubmitCalls.load(std::memory_order_relaxed);
    stats.readerDrainSubmitNanoseconds = statsReaderDrainSubmitNanoseconds.load(std::memory_order_relaxed);
    stats.readerDrainWaitLoopNanoseconds = statsReaderDrainWaitLoopNanoseconds.load(std::memory_order_relaxed);
    stats.readerDrainCompletionProcessNanoseconds =
        statsReaderDrainCompletionProcessNanoseconds.load(std::memory_order_relaxed);
    stats.readerDrainCompletions = statsReaderDrainCompletions.load(std::memory_order_relaxed);
    stats.readerDrainMaxInflightReads = statsReaderDrainMaxInflightReads.load(std::memory_order_relaxed);
    stats.readerShardContextOpenCount = statsReaderShardContextOpenCount.load(std::memory_order_relaxed);
    stats.readerMaxOpenShardContexts = statsReaderMaxOpenShardContexts.load(std::memory_order_relaxed);
    stats.readerLoadExampleCalls = statsReaderLoadExampleCalls.load(std::memory_order_relaxed);
    stats.readerLoadExampleNanoseconds = statsReaderLoadExampleNanoseconds.load(std::memory_order_relaxed);
    stats.readerResolveShardNanoseconds = statsReaderResolveShardNanoseconds.load(std::memory_order_relaxed);
    stats.readerShardContextLookupCalls = statsReaderShardContextLookupCalls.load(std::memory_order_relaxed);
    stats.readerShardContextCacheHits = statsReaderShardContextCacheHits.load(std::memory_order_relaxed);
    stats.readerShardContextCacheMisses = statsReaderShardContextCacheMisses.load(std::memory_order_relaxed);
    stats.readerShardContextLookupNanoseconds = statsReaderShardContextLookupNanoseconds.load(std::memory_order_relaxed);
    stats.readerShardReadRequestNanoseconds = statsReaderShardReadRequestNanoseconds.load(std::memory_order_relaxed);
    stats.readerIovecSlotAcquireNanoseconds = statsReaderIovecSlotAcquireNanoseconds.load(std::memory_order_relaxed);
    stats.readerIovecFillNanoseconds = statsReaderIovecFillNanoseconds.load(std::memory_order_relaxed);
    stats.readerReadvSubmitCallNanoseconds = statsReaderReadvSubmitCallNanoseconds.load(std::memory_order_relaxed);
    stats.getBatchCalls = statsGetBatchCalls.load(std::memory_order_relaxed);
    stats.getBatchReadyQueueEmptyCount = statsGetBatchReadyQueueEmptyCount.load(std::memory_order_relaxed);
    stats.getBatchImmediateCount = statsGetBatchImmediateCount.load(std::memory_order_relaxed);
    stats.getBatchWaitNanoseconds = statsGetBatchWaitNanoseconds.load(std::memory_order_relaxed);
    stats.getBatchTensorUnloadWaitNanoseconds = statsGetBatchTensorUnloadWaitNanoseconds.load(std::memory_order_relaxed);
    stats.returnBufferCalls = statsReturnBufferCalls.load(std::memory_order_relaxed);
    stats.returnBufferWaitNanoseconds = statsReturnBufferWaitNanoseconds.load(std::memory_order_relaxed);
    stats.startBatchCalls = statsStartBatchCalls.load(std::memory_order_relaxed);
    stats.startBatchTensorAcquireNanoseconds = statsStartBatchTensorAcquireNanoseconds.load(std::memory_order_relaxed);
    stats.startBatchPlanningNanoseconds = statsStartBatchPlanningNanoseconds.load(std::memory_order_relaxed);
    stats.pushLoadWorkWaitNanoseconds = statsPushLoadWorkWaitNanoseconds.load(std::memory_order_relaxed);
    stats.waitForCompletedBatchCalls = statsWaitForCompletedBatchCalls.load(std::memory_order_relaxed);
    stats.waitForCompletedBatchNanoseconds = statsWaitForCompletedBatchNanoseconds.load(std::memory_order_relaxed);
    stats.publishCompletedBatchCalls = statsPublishCompletedBatchCalls.load(std::memory_order_relaxed);
    stats.publishCompletedBatchNanoseconds = statsPublishCompletedBatchNanoseconds.load(std::memory_order_relaxed);
    fillPendingBatchAgeStats(stats);
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
    std::fprintf(
        stderr,
        "IndexedNamedBatchSession stats: event=%s split=%s batch=%lu "
        "records_requested=%lu logical_bytes_requested=%lu read_calls_submitted=%lu "
        "read_bytes_submitted=%lu read_calls_completed=%lu read_bytes_completed=%lu records_copied=%lu "
        "copy_bytes=%lu copy_memcpy_calls=%lu copy_active_ns=%lu copy_wait_ns=%lu completed_push_wait_ns=%lu "
        "completed_batch_push_wait_ns=%lu avg_copy_ns_per_record=%.1f avg_copy_calls_per_record=%.1f "
        "read_amplification=%.6f planning_lead_records=%.0f batches_assembled=%lu batches_delivered=%lu "
        "batch_buffers_returned=%lu ready_batches=%lu pending_batches=%lu completed_record_queue=%lu completed_batch_queue=%lu "
        "record_buffer_pool=%lu/%lu queue_depth=%lu shard_read_queue_depth=%lu load_work_queue=%d/%lu load_workers=%lu copy_threads=%lu "
        "get_batch_calls=%lu get_batch_empty=%lu get_batch_immediate=%lu get_batch_wait_ns=%lu "
        "get_batch_tensor_wait_ns=%lu return_calls=%lu return_wait_ns=%lu "
        "load_work_pop_calls=%lu load_work_pop_wait_ns=%lu load_worker_batches=%lu "
        "load_worker_active_ns=%lu load_worker_submit_ns=%lu load_worker_drain_ns=%lu "
        "load_worker_complete_push_wait_ns=%lu readv_submit_ns=%lu readv_backpressure=%lu "
        "readv_backpressure_ns=%lu readv_completion_wait_calls=%lu readv_completion_wait_ns=%lu "
        "reader_drain_calls=%lu reader_drain_ns=%lu reader_drain_contexts=%lu "
        "reader_drain_submit_calls=%lu reader_drain_submit_ns=%lu reader_drain_wait_loop_ns=%lu "
        "reader_drain_completion_process_ns=%lu reader_drain_completions=%lu reader_drain_max_inflight=%lu "
        "reader_shard_opens=%lu reader_max_open_shards=%lu "
        "reader_load_example_calls=%lu reader_load_example_ns=%lu reader_resolve_ns=%lu "
        "reader_context_lookup_calls=%lu reader_context_hits=%lu reader_context_misses=%lu "
        "reader_context_lookup_ns=%lu reader_shard_request_ns=%lu reader_iovec_acquire_ns=%lu "
        "reader_iovec_fill_ns=%lu reader_submit_call_ns=%lu "
        "start_batch_calls=%lu start_batch_tensor_acquire_ns=%lu start_batch_planning_ns=%lu "
        "push_load_work_wait_ns=%lu wait_completed_calls=%lu wait_completed_ns=%lu "
        "publish_calls=%lu publish_ns=%lu pending_loaded=%lu pending_unloaded=%lu "
        "oldest_pending_age_ns=%lu avg_pending_age_ns=%lu "
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
        stats.shardReadQueueDepth,
        loadWorkQueue.occupancy(),
        loadWorkQueueDepth,
        loadWorkerThreadCount,
        stats.recordCopyThreadCount,
        stats.getBatchCalls,
        stats.getBatchReadyQueueEmptyCount,
        stats.getBatchImmediateCount,
        stats.getBatchWaitNanoseconds,
        stats.getBatchTensorUnloadWaitNanoseconds,
        stats.returnBufferCalls,
        stats.returnBufferWaitNanoseconds,
        stats.loadWorkPopCalls,
        stats.loadWorkPopWaitNanoseconds,
        stats.loadWorkerBatches,
        stats.loadWorkerActiveNanoseconds,
        stats.loadWorkerReadSubmitNanoseconds,
        stats.loadWorkerReadDrainNanoseconds,
        stats.loadWorkerCompletedBatchPushWaitNanoseconds,
        stats.readvSubmitNanoseconds,
        stats.readvSubmitBackpressureCount,
        stats.readvSubmitBackpressureNanoseconds,
        stats.readvCompletionWaitCalls,
        stats.readvCompletionWaitNanoseconds,
        stats.readerDrainCalls,
        stats.readerDrainNanoseconds,
        stats.readerDrainContextVisits,
        stats.readerDrainSubmitCalls,
        stats.readerDrainSubmitNanoseconds,
        stats.readerDrainWaitLoopNanoseconds,
        stats.readerDrainCompletionProcessNanoseconds,
        stats.readerDrainCompletions,
        stats.readerDrainMaxInflightReads,
        stats.readerShardContextOpenCount,
        stats.readerMaxOpenShardContexts,
        stats.readerLoadExampleCalls,
        stats.readerLoadExampleNanoseconds,
        stats.readerResolveShardNanoseconds,
        stats.readerShardContextLookupCalls,
        stats.readerShardContextCacheHits,
        stats.readerShardContextCacheMisses,
        stats.readerShardContextLookupNanoseconds,
        stats.readerShardReadRequestNanoseconds,
        stats.readerIovecSlotAcquireNanoseconds,
        stats.readerIovecFillNanoseconds,
        stats.readerReadvSubmitCallNanoseconds,
        stats.startBatchCalls,
        stats.startBatchTensorAcquireNanoseconds,
        stats.startBatchPlanningNanoseconds,
        stats.pushLoadWorkWaitNanoseconds,
        stats.waitForCompletedBatchCalls,
        stats.waitForCompletedBatchNanoseconds,
        stats.publishCompletedBatchCalls,
        stats.publishCompletedBatchNanoseconds,
        stats.currentPendingLoadedBatches,
        stats.currentPendingUnloadedBatches,
        stats.oldestPendingBatchAgeNanoseconds,
        stats.averagePendingBatchAgeNanoseconds,
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
    nextLogicalPosition = (nextLogicalPosition + 1) % indices->size();
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
        if (sessionStats.windowedSourceReadCalls != 0) {
            statsWindowedSourceReadCalls.fetch_add(sessionStats.windowedSourceReadCalls, std::memory_order_relaxed);
            statsWindowedSourceReadBytes.fetch_add(sessionStats.windowedSourceReadBytes, std::memory_order_relaxed);
        }
        if (sessionStats.readvSubmitNanoseconds != 0) {
            statsReadvSubmitNanoseconds.fetch_add(sessionStats.readvSubmitNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.readvSubmitBackpressureCount != 0) {
            statsReadvSubmitBackpressureCount.fetch_add(sessionStats.readvSubmitBackpressureCount, std::memory_order_relaxed);
        }
        if (sessionStats.readvSubmitBackpressureNanoseconds != 0) {
            statsReadvSubmitBackpressureNanoseconds.fetch_add(sessionStats.readvSubmitBackpressureNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.readvCompletionWaitCalls != 0) {
            statsReadvCompletionWaitCalls.fetch_add(sessionStats.readvCompletionWaitCalls, std::memory_order_relaxed);
        }
        if (sessionStats.readvCompletionWaitNanoseconds != 0) {
            statsReadvCompletionWaitNanoseconds.fetch_add(sessionStats.readvCompletionWaitNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.drainCalls != 0) {
            statsReaderDrainCalls.fetch_add(sessionStats.drainCalls, std::memory_order_relaxed);
        }
        if (sessionStats.drainNanoseconds != 0) {
            statsReaderDrainNanoseconds.fetch_add(sessionStats.drainNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.drainContextVisits != 0) {
            statsReaderDrainContextVisits.fetch_add(sessionStats.drainContextVisits, std::memory_order_relaxed);
        }
        if (sessionStats.drainSubmitCalls != 0) {
            statsReaderDrainSubmitCalls.fetch_add(sessionStats.drainSubmitCalls, std::memory_order_relaxed);
        }
        if (sessionStats.drainSubmitNanoseconds != 0) {
            statsReaderDrainSubmitNanoseconds.fetch_add(sessionStats.drainSubmitNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.drainWaitLoopNanoseconds != 0) {
            statsReaderDrainWaitLoopNanoseconds.fetch_add(sessionStats.drainWaitLoopNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.drainCompletionProcessNanoseconds != 0) {
            statsReaderDrainCompletionProcessNanoseconds.fetch_add(sessionStats.drainCompletionProcessNanoseconds,
                                                                  std::memory_order_relaxed);
        }
        if (sessionStats.drainCompletions != 0) {
            statsReaderDrainCompletions.fetch_add(sessionStats.drainCompletions, std::memory_order_relaxed);
        }
        uint64_t observedMaxInflightReads = statsReaderDrainMaxInflightReads.load(std::memory_order_relaxed);
        while (observedMaxInflightReads < sessionStats.drainMaxInflightReads &&
               !statsReaderDrainMaxInflightReads.compare_exchange_weak(observedMaxInflightReads,
                                                                       sessionStats.drainMaxInflightReads,
                                                                       std::memory_order_relaxed,
                                                                       std::memory_order_relaxed)) {
        }
        if (sessionStats.shardContextOpenCount != 0) {
            statsReaderShardContextOpenCount.fetch_add(sessionStats.shardContextOpenCount, std::memory_order_relaxed);
        }
        uint64_t observedMaxOpenContexts = statsReaderMaxOpenShardContexts.load(std::memory_order_relaxed);
        while (observedMaxOpenContexts < sessionStats.maxOpenShardContexts &&
               !statsReaderMaxOpenShardContexts.compare_exchange_weak(
                   observedMaxOpenContexts, sessionStats.maxOpenShardContexts, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
        if (sessionStats.loadExampleCalls != 0) {
            statsReaderLoadExampleCalls.fetch_add(sessionStats.loadExampleCalls, std::memory_order_relaxed);
        }
        if (sessionStats.loadExampleNanoseconds != 0) {
            statsReaderLoadExampleNanoseconds.fetch_add(sessionStats.loadExampleNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.resolveShardNanoseconds != 0) {
            statsReaderResolveShardNanoseconds.fetch_add(sessionStats.resolveShardNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.shardContextLookupCalls != 0) {
            statsReaderShardContextLookupCalls.fetch_add(sessionStats.shardContextLookupCalls, std::memory_order_relaxed);
        }
        if (sessionStats.shardContextCacheHits != 0) {
            statsReaderShardContextCacheHits.fetch_add(sessionStats.shardContextCacheHits, std::memory_order_relaxed);
        }
        if (sessionStats.shardContextCacheMisses != 0) {
            statsReaderShardContextCacheMisses.fetch_add(sessionStats.shardContextCacheMisses, std::memory_order_relaxed);
        }
        if (sessionStats.shardContextLookupNanoseconds != 0) {
            statsReaderShardContextLookupNanoseconds.fetch_add(sessionStats.shardContextLookupNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.shardReadRequestNanoseconds != 0) {
            statsReaderShardReadRequestNanoseconds.fetch_add(sessionStats.shardReadRequestNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.iovecSlotAcquireNanoseconds != 0) {
            statsReaderIovecSlotAcquireNanoseconds.fetch_add(sessionStats.iovecSlotAcquireNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.iovecFillNanoseconds != 0) {
            statsReaderIovecFillNanoseconds.fetch_add(sessionStats.iovecFillNanoseconds, std::memory_order_relaxed);
        }
        if (sessionStats.readvSubmitCallNanoseconds != 0) {
            statsReaderReadvSubmitCallNanoseconds.fetch_add(sessionStats.readvSubmitCallNanoseconds, std::memory_order_relaxed);
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
        const SteadyClock::time_point popStart = diagnosticNow();
        if (!loadWorkQueue.pop(work)) {
            statsLoadWorkPopWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(popStart), std::memory_order_relaxed);
            statsLoadWorkPopCalls.fetch_add(1, std::memory_order_relaxed);
            session->drain();
            flushReaderSessionStats();
            return;
        }
        statsLoadWorkPopWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(popStart), std::memory_order_relaxed);
        statsLoadWorkPopCalls.fetch_add(1, std::memory_order_relaxed);
        const SteadyClock::time_point batchStart = diagnosticNow();
        THOR_THROW_IF_FALSE(work.batchState != nullptr);
        THOR_THROW_IF_FALSE(work.batchOrdinal == work.batchState->batchOrdinal);
        THOR_THROW_IF_FALSE(work.slotBegin < work.slotEnd);
        THOR_THROW_IF_FALSE(work.slotEnd <= work.batchState->expectedRecords);
        THOR_THROW_IF_FALSE(work.batchState->globalExampleIndices.size() == work.batchState->expectedRecords);
        THOR_THROW_IF_FALSE(work.batchState->tensorBasePointers.size() == reader->getTensorCount());
        THOR_THROW_IF_FALSE(work.batchState->windowedTensorBasePointers.size() == reader->getWindowedTensorCount());
        THOR_THROW_IF_FALSE(work.batchState->windowedMaskBasePointers.size() == reader->getWindowedTensorCount());

        const SteadyClock::time_point submitStart = diagnosticNow();
        for (uint64_t slot = work.slotBegin; slot < work.slotEnd; ++slot) {
            session->loadExampleInto(work.batchState->globalExampleIndices.at(slot),
                                     slot,
                                     work.batchState->tensorBasePointers,
                                     work.batchState->windowedTensorBasePointers,
                                     work.batchState->windowedMaskBasePointers);
        }
        statsLoadWorkerReadSubmitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(submitStart), std::memory_order_relaxed);

        const SteadyClock::time_point drainStart = diagnosticNow();
        session->drain();
        statsLoadWorkerReadDrainNanoseconds.fetch_add(diagnosticElapsedNanoseconds(drainStart), std::memory_order_relaxed);
        flushReaderSessionStats();
        const SteadyClock::time_point completePushStart = diagnosticNow();
        const bool completed = markLoadChunkComplete(work);
        statsLoadWorkerCompletedBatchPushWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(completePushStart),
                                                                   std::memory_order_relaxed);
        statsLoadWorkerBatches.fetch_add(1, std::memory_order_relaxed);
        statsLoadWorkerActiveNanoseconds.fetch_add(diagnosticElapsedNanoseconds(batchStart), std::memory_order_relaxed);
        if (!completed) {
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
    for (const auto &entry : batchTensorQueues) {
        if (entry.second->isFull()) {
            return false;
        }
    }
    return true;
}

bool IndexedLocalNamedBatchAssembler::startNextBatch() {
    statsStartBatchCalls.fetch_add(1, std::memory_order_relaxed);
    auto batchState = std::make_shared<IndexedLocalNamedBatchState>();
    batchState->batchOrdinal = nextBatchOrdinal++;
    batchState->batchNum = nextBatchNum;
    batchState->expectedRecords = batchSize;
    batchState->expectedLoadChunks = 1;
    batchState->completedLoadChunks.store(0, std::memory_order_relaxed);
    batchState->loadComplete = false;
    batchState->tensorBasePointers.assign(reader->getTensorCount(), nullptr);
    batchState->windowedTensorBasePointers.assign(reader->getWindowedTensorCount(), nullptr);
    batchState->windowedMaskBasePointers.assign(reader->getWindowedTensorCount(), nullptr);
    batchState->globalExampleIndices.reserve(batchSize);
    batchState->pendingSince = SteadyClock::now();
    nextBatchNum = (nextBatchNum + 1) % batchesPerEpoch;

    const SteadyClock::time_point acquireStart = diagnosticNow();
    for (uint64_t specIndex = 0; specIndex < layout.tensors().size(); ++specIndex) {
        const DatasetLayout::TensorSpec &spec = layout.tensors().at(specIndex);
        Tensor tensor;
        if (!batchTensorQueues.at(spec.name)->getBufferToLoad(tensor)) {
            statsStartBatchTensorAcquireNanoseconds.fetch_add(diagnosticElapsedNanoseconds(acquireStart), std::memory_order_relaxed);
            return false;
        }
        const uint64_t readerOrdinal = layoutTensorOrdinals.at(specIndex);
        THOR_THROW_IF_FALSE(readerOrdinal < batchState->tensorBasePointers.size());
        THOR_THROW_IF_FALSE(batchState->tensorBasePointers.at(readerOrdinal) == nullptr);
        batchState->tensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        batchState->tensors.emplace(spec.name, tensor);
    }
    for (uint64_t specIndex = 0; specIndex < layout.windowedTensors().size(); ++specIndex) {
        const DatasetLayout::WindowedTensorSpec &spec = layout.windowedTensors().at(specIndex);
        Tensor tensor;
        if (!batchTensorQueues.at(spec.name)->getBufferToLoad(tensor)) {
            statsStartBatchTensorAcquireNanoseconds.fetch_add(diagnosticElapsedNanoseconds(acquireStart), std::memory_order_relaxed);
            return false;
        }
        const uint64_t readerOrdinal = layoutWindowedTensorOrdinals.at(specIndex);
        THOR_THROW_IF_FALSE(readerOrdinal < batchState->windowedTensorBasePointers.size());
        THOR_THROW_IF_FALSE(batchState->windowedTensorBasePointers.at(readerOrdinal) == nullptr);
        batchState->windowedTensorBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(tensor.getMemPtr());
        batchState->tensors.emplace(spec.name, tensor);

        if (spec.maskName.has_value()) {
            Tensor maskTensor;
            if (!batchTensorQueues.at(spec.maskName.value())->getBufferToLoad(maskTensor)) {
                statsStartBatchTensorAcquireNanoseconds.fetch_add(diagnosticElapsedNanoseconds(acquireStart), std::memory_order_relaxed);
                return false;
            }
            THOR_THROW_IF_FALSE(batchState->windowedMaskBasePointers.at(readerOrdinal) == nullptr);
            batchState->windowedMaskBasePointers.at(readerOrdinal) = static_cast<uint8_t *>(maskTensor.getMemPtr());
            batchState->tensors.emplace(spec.maskName.value(), maskTensor);
        }
    }
    for (uint8_t *basePointer : batchState->tensorBasePointers) {
        if (basePointer == nullptr) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler failed to bind every reader tensor ordinal to a batch tensor.");
        }
    }
    for (uint64_t specIndex = 0; specIndex < layout.windowedTensors().size(); ++specIndex) {
        const DatasetLayout::WindowedTensorSpec &spec = layout.windowedTensors().at(specIndex);
        const uint64_t readerOrdinal = layoutWindowedTensorOrdinals.at(specIndex);
        if (batchState->windowedTensorBasePointers.at(readerOrdinal) == nullptr) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler failed to bind every reader windowed tensor ordinal to a batch tensor.");
        }
        if (spec.maskName.has_value() && batchState->windowedMaskBasePointers.at(readerOrdinal) == nullptr) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler failed to bind every reader windowed mask ordinal to a batch tensor.");
        }
    }
    statsStartBatchTensorAcquireNanoseconds.fetch_add(diagnosticElapsedNanoseconds(acquireStart), std::memory_order_relaxed);

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

    const SteadyClock::time_point planningStart = diagnosticNow();
    for (uint64_t slot = 0; slot < batchSize; ++slot) {
        const uint64_t logicalPosition = nextLogicalSplitPosition();
        const uint64_t globalExampleIndex = indices->at(logicalPosition);
        batchState->globalExampleIndices.push_back(globalExampleIndex);
        localRecordsRequested += 1;
        localLogicalRecordBytesRequested += recordSizeBytes;
    }
    statsStartBatchPlanningNanoseconds.fetch_add(diagnosticElapsedNanoseconds(planningStart), std::memory_order_relaxed);
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

    const SteadyClock::time_point pushStart = diagnosticNow();
    while (!loadWorkQueue.tryPush(work)) {
        if (!loadWorkQueue.isOpen()) {
            statsPushLoadWorkWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(pushStart), std::memory_order_relaxed);
            return false;
        }
        if (publishCompletedBatches()) {
            continue;
        }

        // Load work is intentionally coarse grained: one queue item covers a
        // contiguous slot range, and the assembler-owned worker does all direct
        // reads for that range.  If all workers are busy, yield rather than
        // blocking the assembler away from completed-batch publication.
        std::this_thread::yield();
    }
    statsPushLoadWorkWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(pushStart), std::memory_order_relaxed);
    return true;
}

bool IndexedLocalNamedBatchAssembler::waitForCompletedBatch() {
    IndexedLocalNamedCompletedBatch completed;
    const SteadyClock::time_point waitStart = diagnosticNow();
    statsWaitForCompletedBatchCalls.fetch_add(1, std::memory_order_relaxed);
    if (!completedBatchQueue.pop(completed)) {
        statsWaitForCompletedBatchNanoseconds.fetch_add(diagnosticElapsedNanoseconds(waitStart), std::memory_order_relaxed);
        return false;
    }
    statsWaitForCompletedBatchNanoseconds.fetch_add(diagnosticElapsedNanoseconds(waitStart), std::memory_order_relaxed);
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
    batchIt->second->loadedAt = SteadyClock::now();
    return true;
}

uint64_t IndexedLocalNamedBatchAssembler::pendingBatchCount() const {
    std::lock_guard<std::mutex> guard(pendingBatchesMutex);
    return static_cast<uint64_t>(pendingBatches.size());
}

void IndexedLocalNamedBatchAssembler::fillPendingBatchAgeStats(IndexedLocalNamedBatchAssemblerStats &stats) const {
    const SteadyClock::time_point now = SteadyClock::now();
    uint64_t pendingLoaded = 0;
    uint64_t pendingUnloaded = 0;
    uint64_t oldestAge = 0;
    uint64_t totalAge = 0;
    uint64_t ageCount = 0;

    std::lock_guard<std::mutex> guard(pendingBatchesMutex);
    for (const auto &entry : pendingBatches) {
        THOR_THROW_IF_FALSE(entry.second != nullptr);
        const IndexedLocalNamedBatchState &batch = *entry.second;
        if (batch.loadComplete) {
            pendingLoaded += 1;
        } else {
            pendingUnloaded += 1;
        }
        const uint64_t age = elapsedNanoseconds(batch.pendingSince, now);
        oldestAge = std::max(oldestAge, age);
        totalAge += age;
        ageCount += 1;
    }

    stats.currentPendingLoadedBatches = pendingLoaded;
    stats.currentPendingUnloadedBatches = pendingUnloaded;
    stats.oldestPendingBatchAgeNanoseconds = oldestAge;
    stats.averagePendingBatchAgeNanoseconds = ageCount == 0 ? 0 : totalAge / ageCount;
}

bool IndexedLocalNamedBatchAssembler::publishCompletedBatches() {
    const SteadyClock::time_point publishStart = diagnosticNow();
    statsPublishCompletedBatchCalls.fetch_add(1, std::memory_order_relaxed);
    auto finishPublishTiming = [&]() {
        statsPublishCompletedBatchNanoseconds.fetch_add(diagnosticElapsedNanoseconds(publishStart), std::memory_order_relaxed);
    };
    markAvailableCompletedBatches();
    bool publishedAny = false;
    while (true) {
        std::shared_ptr<IndexedLocalNamedBatchState> batchState;
        {
            std::lock_guard<std::mutex> mapGuard(pendingBatchesMutex);
            auto batchIt = pendingBatches.find(nextPublishOrdinal);
            if (batchIt == pendingBatches.end()) {
                finishPublishTiming();
                return publishedAny;
            }
            batchState = batchIt->second;
            THOR_THROW_IF_FALSE(batchState != nullptr);

            if (!batchState->loadComplete) {
                finishPublishTiming();
                return publishedAny;
            }
            pendingBatches.erase(batchIt);
        }

        for (const auto &entry : batchTensorQueues) {
            const bool queueOpen = entry.second->bufferLoaded(batchState->tensors.at(entry.first));
            if (!queueOpen) {
                finishPublishTiming();
                return publishedAny;
            }
        }
        if (!batchNumQueue.push(batchState->batchNum)) {
            finishPublishTiming();
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

void IndexedLocalNamedBatchAssembler::acquireBatch(std::map<std::string, Tensor> &tensors, uint64_t &batchNum) {
    if (batchesPerEpoch == 0) {
        throw std::runtime_error("IndexedLocalNamedBatchAssembler cannot get a batch from an empty split.");
    }

    std::lock_guard<std::mutex> deliveryGuard(batchDeliveryMutex);
    throwIfWorkerFailed();
    tensors.clear();

    const int readyBeforePop = batchNumQueue.occupancy();
    const bool hadReadyBatch = readyBeforePop > 0;
    const SteadyClock::time_point batchWaitStart = diagnosticNow();
    const bool batchNumQueueOpen = batchNumQueue.pop(batchNum);
    const uint64_t batchWaitNs = diagnosticElapsedNanoseconds(batchWaitStart);
    statsGetBatchCalls.fetch_add(1, std::memory_order_relaxed);
    statsGetBatchWaitNanoseconds.fetch_add(batchWaitNs, std::memory_order_relaxed);
    if (!hadReadyBatch) {
        statsGetBatchReadyQueueEmptyCount.fetch_add(1, std::memory_order_relaxed);
    } else {
        statsGetBatchImmediateCount.fetch_add(1, std::memory_order_relaxed);
    }
    if (!batchNumQueueOpen) {
        throwIfWorkerFailed();
        THOR_THROW_IF_FALSE(batchNumQueueOpen);
    }

    const SteadyClock::time_point tensorUnloadStart = diagnosticNow();
    for (const auto &entry : batchTensorQueues) {
        Tensor tensor;
        const bool tensorQueueOpen = entry.second->getBufferToUnload(tensor);
        if (!tensorQueueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(tensorQueueOpen);
        }
        tensors.emplace(entry.first, tensor);
    }
    statsGetBatchTensorUnloadWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(tensorUnloadStart), std::memory_order_relaxed);
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
    if (tensors.size() != batchTensorDescriptors.size()) {
        throw std::runtime_error("IndexedLocalNamedBatchAssembler returned tensor count does not match output tensor count.");
    }

    for (const auto &entry : batchTensorDescriptors) {
        const auto it = tensors.find(entry.first);
        if (it == tensors.end()) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler missing returned tensor: " + entry.first);
        }
        THOR_THROW_IF_FALSE(it->second.isInitialized());
        if (it->second.getDescriptor() != entry.second) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler returned tensor has wrong descriptor for: " + entry.first);
        }
    }

    for (const auto &entry : tensors) {
        if (batchTensorDescriptors.find(entry.first) == batchTensorDescriptors.end()) {
            throw std::runtime_error("IndexedLocalNamedBatchAssembler returned unexpected tensor: " + entry.first);
        }
    }
}

void IndexedLocalNamedBatchAssembler::returnBuffers(const std::map<std::string, Tensor> &tensors) {
    std::lock_guard<std::mutex> returnGuard(returnBuffersMutex);
    throwIfWorkerFailed();
    validateReturnedTensorMapExact(tensors);
    statsReturnBufferCalls.fetch_add(1, std::memory_order_relaxed);
    const SteadyClock::time_point returnStart = diagnosticNow();
    for (const auto &entry : batchTensorQueues) {
        const bool queueOpen = entry.second->bufferUnloaded(tensors.at(entry.first));
        if (!queueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(queueOpen);
        }
    }
    statsReturnBufferWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(returnStart), std::memory_order_relaxed);
    const uint64_t returned = statsBatchBuffersReturned.fetch_add(1, std::memory_order_relaxed) + 1;
    if (shouldEmitStats(returned)) {
        emitStatsIfEnabled("return_batch", 0);
    }
}

uint64_t IndexedLocalNamedBatchAssembler::getNumBatchesPerEpoch() const { return batchesPerEpoch; }

uint64_t IndexedLocalNamedBatchAssembler::getNumExamples() const { return static_cast<uint64_t>(indices->size()); }
