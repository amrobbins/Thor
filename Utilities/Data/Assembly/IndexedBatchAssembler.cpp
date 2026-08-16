#include "Utilities/Data/Assembly/IndexedBatchAssembler.h"
#include "DeepLearning/Implementation/Data/BatchCardinality.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include "DeepLearning/Implementation/ThorError.h"

using ThorImplementation::RaggedTensor;
using ThorImplementation::RaggedTensorDescriptor;
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

    // The indexed dataset reader uses one assembler worker per batch and the async
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
        "THOR_INDEXED_DATASET_READER_QUEUE_DEPTH", nullptr, defaultDepth);
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
        parsePositiveUint64Env("THOR_INDEXED_BATCH_ASSEMBLER_LOAD_WORKERS", nullptr, defaultWorkers);
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

uint64_t checkedAddUint64(uint64_t left, uint64_t right, const std::string &context) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error(context + " overflows uint64_t.");
    }
    return left + right;
}

uint64_t checkedMulUint64(uint64_t left, uint64_t right, const std::string &context) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(context + " overflows uint64_t.");
    }
    return left * right;
}

void writeRaggedOffset(void *offsets, ThorImplementation::DataType dataType, uint64_t index, uint64_t value) {
    THOR_THROW_IF_FALSE(offsets != nullptr);
    if (dataType == ThorImplementation::DataType::UINT32) {
        if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("IndexedBatchAssembler ragged UINT32 offset is outside uint32_t range.");
        }
        static_cast<uint32_t *>(offsets)[index] = static_cast<uint32_t>(value);
        return;
    }
    if (dataType == ThorImplementation::DataType::UINT64) {
        static_cast<uint64_t *>(offsets)[index] = value;
        return;
    }
    throw std::runtime_error("IndexedBatchAssembler ragged offsets dtype must be UINT32 or UINT64.");
}

using SteadyClock = std::chrono::steady_clock;

bool diagnosticsTimingEnabled() {
    static const bool enabled = [] {
        const char *specific = std::getenv("THOR_INDEXED_BATCH_ASSEMBLER_DIAGNOSTICS");
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
        const char *specific = std::getenv("THOR_INDEXED_BATCH_ASSEMBLER_STATS");
        if (specific != nullptr && specific[0] != '\0') {
            return !(specific[0] == '0' && specific[1] == '\0');
        }
        const char *diagnostics = std::getenv("THOR_INDEXED_BATCH_ASSEMBLER_DIAGNOSTICS");
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
        const char *value = std::getenv("THOR_INDEXED_BATCH_ASSEMBLER_STATS_EVERY");
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

IndexedBatchAssembler::IndexedBatchAssembler(
    std::shared_ptr<IndexedDatasetReader> reader,
    DatasetLayout layout,
    std::shared_ptr<const Thor::ExampleIndexSet> indices,
    std::string splitName,
    uint64_t batchSize,
    uint64_t batchQueueDepth,
    bool randomized,
    std::optional<uint64_t> seed,
    bool wrapTail,
    std::map<std::string, RaggedTensorDescriptor> raggedTensorDescriptors)
    : reader(std::move(reader)),
      layout(std::move(layout)),
      indices(std::move(indices)),
      splitName(std::move(splitName)),
      raggedTensorDescriptors(std::move(raggedTensorDescriptors)),
      batchSize(batchSize),
      batchQueueDepth(batchQueueDepth),
      shardReadQueueDepth(0),
      shardRequestQueueDepth(0),
      completedBatchQueueDepth(0),
      recordSizeBytes(this->reader == nullptr ? 0 : this->reader->getRecordSizeBytes()),
      batchesPerEpoch(batchesFor(this->indices == nullptr ? 0 : this->indices->size(), batchSize)),
      numDatasetExamples(this->reader == nullptr ? 0 : this->reader->getNumExamples()),
      nextBatchToSchedule(0),
      nextBatchToDeliver(0),
      nextLogicalPosition(0),
      nextBatchOrdinal(0),
      nextPublishOrdinal(0),
      randomized(randomized),
      wrapTail(wrapTail),
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
    THOR_THROW_IF_FALSE(this->reader->getRaggedTensorCount() == this->layout.raggedTensors().size());

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

    layoutRaggedTensorOrdinals.reserve(this->layout.raggedTensors().size());
    for (const DatasetLayout::RaggedTensorSpec &spec : this->layout.raggedTensors()) {
        const uint64_t readerOrdinal = this->reader->getLayoutRaggedTensorOrdinal(spec.name);
        layoutRaggedTensorOrdinals.push_back(readerOrdinal);
        const auto descriptorIt = this->raggedTensorDescriptors.find(spec.name);
        if (descriptorIt == this->raggedTensorDescriptors.end()) {
            continue;
        }
        const RaggedTensorDescriptor &descriptor = descriptorIt->second;
        if (descriptor.getValuesDataType() != spec.dataType ||
            descriptor.getTrailingDimensions() != spec.valueDimensions ||
            descriptor.getBatchSize() != batchSize) {
            throw std::runtime_error(
                "IndexedBatchAssembler ragged materialization descriptor does not match dataset field '" +
                spec.name + "'.");
        }
    }
    for (const auto &[name, descriptor] : this->raggedTensorDescriptors) {
        (void)descriptor;
        if (std::none_of(this->layout.raggedTensors().begin(),
                         this->layout.raggedTensors().end(),
                         [&name](const DatasetLayout::RaggedTensorSpec &spec) { return spec.name == name; })) {
            throw std::runtime_error(
                "IndexedBatchAssembler received a ragged materialization descriptor for unknown field '" + name + "'.");
        }
    }

    open();
}

IndexedBatchAssembler::~IndexedBatchAssembler() { close(); }

void IndexedBatchAssembler::open() {
    try {
        THOR_THROW_IF_FALSE(loadWorkerThreads.empty());
        THOR_THROW_IF_FALSE(batchTensorQueues.empty());
        THOR_THROW_IF_FALSE(raggedValuesQueues.empty());
        THOR_THROW_IF_FALSE(raggedOffsetsQueues.empty());

        loadWorkQueue.resize(checkedQueueDepth(loadWorkQueueDepth, "IndexedBatchAssembler load work"));
        loadWorkQueue.open();

        completedBatchQueue.resize(checkedQueueDepth(completedBatchQueueDepth, "IndexedBatchAssembler completed batch"));
        completedBatchQueue.open();

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        for (const auto &entry : batchTensorDescriptors) {
            auto queue = std::make_unique<AsyncTensorQueue>();
            queue->resize(batchQueueDepth, entry.second, cpuPlacement);
            queue->open();
            batchTensorQueues.emplace(entry.first, std::move(queue));
        }
        for (const auto &[name, descriptor] : raggedTensorDescriptors) {
            auto valuesQueue = std::make_unique<AsyncTensorQueue>();
            valuesQueue->resize(batchQueueDepth, descriptor.getValuesDescriptor(), cpuPlacement);
            valuesQueue->open();
            raggedValuesQueues.emplace(name, std::move(valuesQueue));

            auto offsetsQueue = std::make_unique<AsyncTensorQueue>();
            offsetsQueue->resize(batchQueueDepth, descriptor.getOffsetsDescriptor(), cpuPlacement);
            offsetsQueue->open();
            raggedOffsetsQueues.emplace(name, std::move(offsetsQueue));
        }

        readyBatchQueue.resize(checkedQueueDepth(batchQueueDepth, "IndexedBatchAssembler ready batch"));
        readyBatchQueue.open();

        for (uint64_t i = 0; i < loadWorkerThreadCount; ++i) {
            loadWorkerThreads.emplace_back(&IndexedBatchAssembler::loadWorkerThread, this, i);
        }
        assemblerThread = std::thread(&IndexedBatchAssembler::batchAssemblerThread, this);
    } catch (...) {
        close();
        throw;
    }
}

void IndexedBatchAssembler::close() {
    loadWorkQueue.close();
    completedBatchQueue.close();
    for (auto &entry : batchTensorQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    for (auto &entry : raggedValuesQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    for (auto &entry : raggedOffsetsQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    readyBatchQueue.close();

    for (std::thread &thread : loadWorkerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (assemblerThread.joinable()) {
        assemblerThread.join();
    }

    batchTensorQueues.clear();
    raggedValuesQueues.clear();
    raggedOffsetsQueues.clear();
    loadWorkerThreads.clear();
    {
        std::lock_guard<std::mutex> guard(pendingBatchesMutex);
        pendingBatches.clear();
    }
}

void IndexedBatchAssembler::recordWorkerException(std::exception_ptr exception) {
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
    for (auto &entry : raggedValuesQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    for (auto &entry : raggedOffsetsQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    readyBatchQueue.close();
}

void IndexedBatchAssembler::throwIfWorkerFailed() const {
    std::exception_ptr exception;
    {
        std::lock_guard<std::mutex> guard(workerExceptionMutex);
        exception = workerException;
    }
    if (exception != nullptr) {
        std::rethrow_exception(exception);
    }
}

void IndexedBatchAssembler::setResolvedIoBackend(const std::string &backendName) {
    std::lock_guard<std::mutex> guard(statsMutex);
    if (resolvedIoBackend == "unresolved" || resolvedIoBackend == backendName) {
        resolvedIoBackend = backendName;
    } else if (resolvedIoBackend.find(backendName) == std::string::npos) {
        resolvedIoBackend += "," + backendName;
    }
}

IndexedBatchAssemblerStats IndexedBatchAssembler::getStatsSnapshot() {
    IndexedBatchAssemblerStats stats;
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
    const int readyBatches = readyBatchQueue.occupancy();
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

void IndexedBatchAssembler::emitStatsIfEnabled(const char *event, uint64_t batchNum) {
    if (!statsLoggingEnabled()) {
        return;
    }

    const IndexedBatchAssemblerStats stats = getStatsSnapshot();
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

void IndexedBatchAssembler::validateGlobalIndex(uint64_t index, const char *context) const {
    THOR_THROW_IF_FALSE(reader != nullptr);
    reader->validateGlobalIndex(index, context);
}

uint64_t IndexedBatchAssembler::nextLogicalSplitPosition() {
    if (randomized) {
        THOR_THROW_IF_FALSE(randomizer != nullptr);
        return randomizer->getRandomNumber();
    }

    const uint64_t logicalPosition = nextLogicalPosition;
    nextLogicalPosition = (nextLogicalPosition + 1) % indices->size();
    return logicalPosition;
}

void IndexedBatchAssembler::loadWorkerThread(uint64_t workerIndex) {
    try {
        loadWorkerThreadMain(workerIndex);
    } catch (...) {
        recordWorkerException(std::current_exception());
    }
}

void IndexedBatchAssembler::loadWorkerThreadMain(uint64_t workerIndex) {
    (void)workerIndex;

    auto session = reader->createSession(shardReadQueueDepth);

    auto flushReaderSessionStats = [&]() {
        IndexedDatasetReaderSessionStats sessionStats = session->takeStats();
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

    auto markLoadChunkComplete = [&](const IndexedBatchLoadWork &work) -> bool {
        THOR_THROW_IF_FALSE(work.batchState != nullptr);
        const uint64_t completedChunks = work.batchState->completedLoadChunks.fetch_add(1, std::memory_order_acq_rel) + 1;
        THOR_THROW_IF_FALSE(completedChunks <= work.batchState->expectedLoadChunks);
        if (completedChunks != work.batchState->expectedLoadChunks) {
            return true;
        }

        IndexedCompletedBatch completed;
        completed.batchOrdinal = work.batchOrdinal;
        return completedBatchQueue.push(completed);
    };

    while (true) {
        IndexedBatchLoadWork work;
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
        THOR_THROW_IF_FALSE(work.batchState->raggedReferenceBasePointers.size() == reader->getRaggedTensorCount());

        const SteadyClock::time_point submitStart = diagnosticNow();
        for (uint64_t slot = work.slotBegin; slot < work.slotEnd; ++slot) {
            session->loadExampleInto(work.batchState->globalExampleIndices.at(slot),
                                     slot,
                                     work.batchState->tensorBasePointers,
                                     work.batchState->windowedTensorBasePointers,
                                     work.batchState->windowedMaskBasePointers,
                                     work.batchState->raggedReferenceBasePointers);
        }
        statsLoadWorkerReadSubmitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(submitStart), std::memory_order_relaxed);

        const SteadyClock::time_point drainStart = diagnosticNow();
        session->drain();
        statsLoadWorkerReadDrainNanoseconds.fetch_add(diagnosticElapsedNanoseconds(drainStart), std::memory_order_relaxed);
        flushReaderSessionStats();
        materializeRaggedBatch(*session, *work.batchState);
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

void IndexedBatchAssembler::materializeRaggedBatch(IndexedDatasetReader::Session &session,
                                                   IndexedBatchState &batchState) const {
    if (raggedTensorDescriptors.empty()) {
        return;
    }
    THOR_THROW_IF_FALSE(batchState.raggedReferences.size() == reader->getRaggedTensorCount());
    THOR_THROW_IF_FALSE(batchState.raggedTensors.size() == raggedTensorDescriptors.size());

    const uint64_t logicalRows = wrapTail ? batchSize : batchState.validExampleCount;
    THOR_THROW_IF_FALSE(logicalRows <= batchSize);

    for (uint64_t layoutOrdinal = 0; layoutOrdinal < layout.raggedTensors().size(); ++layoutOrdinal) {
        const DatasetLayout::RaggedTensorSpec &spec = layout.raggedTensors().at(static_cast<size_t>(layoutOrdinal));
        const auto descriptorIt = raggedTensorDescriptors.find(spec.name);
        if (descriptorIt == raggedTensorDescriptors.end()) {
            continue;
        }
        const uint64_t readerOrdinal = layoutRaggedTensorOrdinals.at(static_cast<size_t>(layoutOrdinal));
        const RaggedTensorDescriptor &descriptor = descriptorIt->second;
        const auto raggedIt = batchState.raggedTensors.find(spec.name);
        THOR_THROW_IF_FALSE(raggedIt != batchState.raggedTensors.end());
        RaggedTensor ragged = raggedIt->second;
        Tensor values = ragged.getValues();
        Tensor offsets = ragged.getOffsets();
        THOR_THROW_IF_FALSE(values.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        THOR_THROW_IF_FALSE(offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);

        const auto &references = batchState.raggedReferences.at(static_cast<size_t>(readerOrdinal));
        THOR_THROW_IF_FALSE(references.size() == batchSize);
        const uint64_t storedValueCount = spec.storedValueCount();
        const uint64_t maxTotalValues = descriptor.getMaxTotalValues();
        uint64_t activeValueCount = 0;
        writeRaggedOffset(offsets.getMemPtr(), descriptor.getOffsetsDataType(), 0, 0);

        for (uint64_t row = 0; row < logicalRows; ++row) {
            const IndexedRaggedTensorReference &reference = references.at(static_cast<size_t>(row));
            if (reference.startValue > storedValueCount ||
                reference.valueCount > storedValueCount - reference.startValue) {
                throw std::runtime_error(
                    "IndexedBatchAssembler ragged field '" + spec.name + "' row " + std::to_string(row) +
                    " references values outside the sidecar.");
            }
            const uint64_t nextActiveValueCount = checkedAddUint64(
                activeValueCount, reference.valueCount, "IndexedBatchAssembler ragged active value count");
            if (nextActiveValueCount > maxTotalValues) {
                throw std::runtime_error(
                    "IndexedBatchAssembler ragged field '" + spec.name + "' requires " +
                    std::to_string(nextActiveValueCount) + " active values at row " + std::to_string(row) +
                    ", exceeding maxTotalValues=" + std::to_string(maxTotalValues) + ".");
            }
            activeValueCount = nextActiveValueCount;
            writeRaggedOffset(offsets.getMemPtr(), descriptor.getOffsetsDataType(), row + 1, activeValueCount);
        }
        for (uint64_t row = logicalRows; row < batchSize; ++row) {
            writeRaggedOffset(offsets.getMemPtr(), descriptor.getOffsetsDataType(), row + 1, activeValueCount);
        }

        uint8_t *const valuesBase = static_cast<uint8_t *>(values.getMemPtr());
        const uint64_t valueNumBytes = spec.valueNumBytes();
        uint64_t destinationValue = 0;
        uint64_t row = 0;
        while (row < logicalRows) {
            const IndexedRaggedTensorReference &first = references.at(static_cast<size_t>(row));
            if (first.valueCount == 0) {
                ++row;
                continue;
            }
            uint64_t sourceStart = first.startValue;
            uint64_t runValues = first.valueCount;
            const uint64_t destinationStart = destinationValue;
            destinationValue = checkedAddUint64(destinationValue, first.valueCount,
                                                "IndexedBatchAssembler ragged destination value count");
            uint64_t nextRow = row + 1;
            while (nextRow < logicalRows) {
                const IndexedRaggedTensorReference &next = references.at(static_cast<size_t>(nextRow));
                if (next.valueCount == 0) {
                    ++nextRow;
                    continue;
                }
                const uint64_t expectedSourceStart = checkedAddUint64(
                    sourceStart, runValues, "IndexedBatchAssembler ragged contiguous source range");
                if (next.startValue != expectedSourceStart) {
                    break;
                }
                runValues = checkedAddUint64(
                    runValues, next.valueCount, "IndexedBatchAssembler ragged contiguous read size");
                destinationValue = checkedAddUint64(
                    destinationValue, next.valueCount, "IndexedBatchAssembler ragged destination value count");
                ++nextRow;
            }
            const uint64_t destinationByteOffset = checkedMulUint64(
                destinationStart, valueNumBytes, "IndexedBatchAssembler ragged destination byte offset");
            session.loadRaggedValuesInto(
                readerOrdinal, sourceStart, runValues, valuesBase + destinationByteOffset);
            row = nextRow;
        }
        THOR_THROW_IF_FALSE(destinationValue == activeValueCount);

        // This is the row-partition production boundary for indexed CPU batches.
        // Offsets remain the semantic source of truth; publish offsets[B] into the
        // shared runtime cache so downstream host-dispatched ragged operations do
        // not need to rediscover or attach this state to the packed values tensor.
        ragged.getRowPartitionRuntime().setHostActiveValueCount(activeValueCount);
    }
}

void IndexedBatchAssembler::batchAssemblerThread() {
    try {
        batchAssemblerThreadMain();
    } catch (...) {
        recordWorkerException(std::current_exception());
    }
}

void IndexedBatchAssembler::batchAssemblerThreadMain() {
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

bool IndexedBatchAssembler::canStartNextBatchWithoutBlocking() {
    for (const auto &entry : batchTensorQueues) {
        if (entry.second->isFull()) {
            return false;
        }
    }
    for (const auto &entry : raggedValuesQueues) {
        if (entry.second->isFull()) {
            return false;
        }
    }
    for (const auto &entry : raggedOffsetsQueues) {
        if (entry.second->isFull()) {
            return false;
        }
    }
    return true;
}

bool IndexedBatchAssembler::startNextBatch() {
    statsStartBatchCalls.fetch_add(1, std::memory_order_relaxed);
    auto batchState = std::make_shared<IndexedBatchState>();
    batchState->batchOrdinal = nextBatchOrdinal++;
    batchState->batchNum = nextBatchToSchedule;
    batchState->validExampleCount = wrapTail
        ? ThorImplementation::fullBatchValidExampleCount(batchSize)
        : ThorImplementation::validExamplesForBatch(
              batchState->batchNum,
              indices->size(),
              batchSize);
    batchState->expectedRecords = batchSize;
    batchState->expectedLoadChunks = 1;
    batchState->completedLoadChunks.store(0, std::memory_order_relaxed);
    batchState->loadComplete = false;
    batchState->tensorBasePointers.assign(reader->getTensorCount(), nullptr);
    batchState->windowedTensorBasePointers.assign(reader->getWindowedTensorCount(), nullptr);
    batchState->windowedMaskBasePointers.assign(reader->getWindowedTensorCount(), nullptr);
    batchState->raggedReferences.resize(reader->getRaggedTensorCount());
    batchState->raggedReferenceBasePointers.assign(reader->getRaggedTensorCount(), nullptr);
    for (uint64_t ordinal = 0; ordinal < reader->getRaggedTensorCount(); ++ordinal) {
        auto &references = batchState->raggedReferences.at(static_cast<size_t>(ordinal));
        references.resize(static_cast<size_t>(batchSize));
        batchState->raggedReferenceBasePointers.at(static_cast<size_t>(ordinal)) = references.data();
    }
    batchState->globalExampleIndices.reserve(batchSize);
    batchState->pendingSince = SteadyClock::now();
    nextBatchToSchedule = (nextBatchToSchedule + 1) % batchesPerEpoch;

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
    for (const auto &[name, descriptor] : raggedTensorDescriptors) {
        Tensor values;
        if (!raggedValuesQueues.at(name)->getBufferToLoad(values)) {
            statsStartBatchTensorAcquireNanoseconds.fetch_add(
                diagnosticElapsedNanoseconds(acquireStart), std::memory_order_relaxed);
            return false;
        }
        Tensor offsets;
        if (!raggedOffsetsQueues.at(name)->getBufferToLoad(offsets)) {
            statsStartBatchTensorAcquireNanoseconds.fetch_add(
                diagnosticElapsedNanoseconds(acquireStart), std::memory_order_relaxed);
            return false;
        }
        THOR_THROW_IF_FALSE(values.getDescriptor() == descriptor.getValuesDescriptor());
        THOR_THROW_IF_FALSE(offsets.getDescriptor() == descriptor.getOffsetsDescriptor());
        batchState->raggedTensors.emplace(name, RaggedTensor(values, offsets));
    }
    for (uint8_t *basePointer : batchState->tensorBasePointers) {
        if (basePointer == nullptr) {
            throw std::runtime_error("IndexedBatchAssembler failed to bind every reader tensor ordinal to a batch tensor.");
        }
    }
    for (uint64_t specIndex = 0; specIndex < layout.windowedTensors().size(); ++specIndex) {
        const DatasetLayout::WindowedTensorSpec &spec = layout.windowedTensors().at(specIndex);
        const uint64_t readerOrdinal = layoutWindowedTensorOrdinals.at(specIndex);
        if (batchState->windowedTensorBasePointers.at(readerOrdinal) == nullptr) {
            throw std::runtime_error("IndexedBatchAssembler failed to bind every reader windowed tensor ordinal to a batch tensor.");
        }
        if (spec.maskName.has_value() && batchState->windowedMaskBasePointers.at(readerOrdinal) == nullptr) {
            throw std::runtime_error("IndexedBatchAssembler failed to bind every reader windowed mask ordinal to a batch tensor.");
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
    for (uint64_t slot = 0; slot < batchState->validExampleCount; ++slot) {
        const uint64_t logicalPosition = nextLogicalSplitPosition();
        const uint64_t globalExampleIndex = indices->at(logicalPosition);
        batchState->globalExampleIndices.push_back(globalExampleIndex);
        localRecordsRequested += 1;
        localLogicalRecordBytesRequested += recordSizeBytes;
    }
    THOR_THROW_IF_FALSE(!batchState->globalExampleIndices.empty());
    if (!wrapTail) {
        const uint64_t paddingExampleIndex = batchState->globalExampleIndices.back();
        for (uint64_t slot = batchState->validExampleCount; slot < batchSize; ++slot) {
            batchState->globalExampleIndices.push_back(paddingExampleIndex);
            localRecordsRequested += 1;
            localLogicalRecordBytesRequested += recordSizeBytes;
        }
    }
    THOR_THROW_IF_FALSE(batchState->globalExampleIndices.size() == batchSize);
    statsStartBatchPlanningNanoseconds.fetch_add(diagnosticElapsedNanoseconds(planningStart), std::memory_order_relaxed);
    flushLocalRequestStats();

    const uint64_t batchOrdinal = batchState->batchOrdinal;
    {
        std::lock_guard<std::mutex> guard(pendingBatchesMutex);
        auto [insertIt, inserted] = pendingBatches.emplace(batchOrdinal, batchState);
        THOR_THROW_IF_FALSE(inserted);
        (void)insertIt;
    }

    IndexedBatchLoadWork work;
    work.batchState = batchState.get();
    work.batchOrdinal = batchOrdinal;
    work.slotBegin = 0;
    work.slotEnd = batchSize;
    if (!pushLoadWorkWithDrain(work)) {
        return false;
    }
    return true;
}

bool IndexedBatchAssembler::pushLoadWorkWithDrain(const IndexedBatchLoadWork &work) {
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

bool IndexedBatchAssembler::waitForCompletedBatch() {
    IndexedCompletedBatch completed;
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

void IndexedBatchAssembler::markAvailableCompletedBatches() {
    IndexedCompletedBatch completed;
    while (completedBatchQueue.tryPop(completed)) {
        THOR_THROW_IF_FALSE(markBatchLoaded(completed.batchOrdinal));
    }
}

bool IndexedBatchAssembler::markBatchLoaded(uint64_t batchOrdinal) {
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

uint64_t IndexedBatchAssembler::pendingBatchCount() const {
    std::lock_guard<std::mutex> guard(pendingBatchesMutex);
    return static_cast<uint64_t>(pendingBatches.size());
}

void IndexedBatchAssembler::fillPendingBatchAgeStats(IndexedBatchAssemblerStats &stats) const {
    const SteadyClock::time_point now = SteadyClock::now();
    uint64_t pendingLoaded = 0;
    uint64_t pendingUnloaded = 0;
    uint64_t oldestAge = 0;
    uint64_t totalAge = 0;
    uint64_t ageCount = 0;

    std::lock_guard<std::mutex> guard(pendingBatchesMutex);
    for (const auto &entry : pendingBatches) {
        THOR_THROW_IF_FALSE(entry.second != nullptr);
        const IndexedBatchState &batch = *entry.second;
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

bool IndexedBatchAssembler::publishCompletedBatches() {
    const SteadyClock::time_point publishStart = diagnosticNow();
    statsPublishCompletedBatchCalls.fetch_add(1, std::memory_order_relaxed);
    auto finishPublishTiming = [&]() {
        statsPublishCompletedBatchNanoseconds.fetch_add(diagnosticElapsedNanoseconds(publishStart), std::memory_order_relaxed);
    };
    markAvailableCompletedBatches();
    bool publishedAny = false;
    while (true) {
        std::shared_ptr<IndexedBatchState> batchState;
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
        for (const auto &[name, ragged] : batchState->raggedTensors) {
            const bool valuesQueueOpen = raggedValuesQueues.at(name)->bufferLoaded(ragged.getValues());
            if (!valuesQueueOpen) {
                finishPublishTiming();
                return publishedAny;
            }
            const bool offsetsQueueOpen = raggedOffsetsQueues.at(name)->bufferLoaded(ragged.getOffsets());
            if (!offsetsQueueOpen) {
                finishPublishTiming();
                return publishedAny;
            }
        }
        if (!readyBatchQueue.push(IndexedReadyBatch{
                .batchNum = batchState->batchNum,
                .validExampleCount = batchState->validExampleCount})) {
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

void IndexedBatchAssembler::acquireBatch(std::map<std::string, Tensor> &tensors,
                                         std::map<std::string, RaggedTensor> &raggedTensors,
                                         IndexedReadyBatch &readyBatch) {
    if (batchesPerEpoch == 0) {
        throw std::runtime_error("IndexedBatchAssembler cannot get a batch from an empty split.");
    }

    std::lock_guard<std::mutex> deliveryGuard(batchDeliveryMutex);
    throwIfWorkerFailed();
    tensors.clear();
    raggedTensors.clear();

    const int readyBeforePop = readyBatchQueue.occupancy();
    const bool hadReadyBatch = readyBeforePop > 0;
    const SteadyClock::time_point batchWaitStart = diagnosticNow();
    const bool readyBatchQueueOpen = readyBatchQueue.pop(readyBatch);
    const uint64_t batchWaitNs = diagnosticElapsedNanoseconds(batchWaitStart);
    statsGetBatchCalls.fetch_add(1, std::memory_order_relaxed);
    statsGetBatchWaitNanoseconds.fetch_add(batchWaitNs, std::memory_order_relaxed);
    if (!hadReadyBatch) {
        statsGetBatchReadyQueueEmptyCount.fetch_add(1, std::memory_order_relaxed);
    } else {
        statsGetBatchImmediateCount.fetch_add(1, std::memory_order_relaxed);
    }
    if (!readyBatchQueueOpen) {
        throwIfWorkerFailed();
        THOR_THROW_IF_FALSE(readyBatchQueueOpen);
    }
    THOR_THROW_IF_FALSE(readyBatch.validExampleCount >= 1);
    THOR_THROW_IF_FALSE(readyBatch.validExampleCount <= batchSize);

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
    for (const auto &[name, descriptor] : raggedTensorDescriptors) {
        Tensor values;
        const bool valuesQueueOpen = raggedValuesQueues.at(name)->getBufferToUnload(values);
        if (!valuesQueueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(valuesQueueOpen);
        }
        Tensor offsets;
        const bool offsetsQueueOpen = raggedOffsetsQueues.at(name)->getBufferToUnload(offsets);
        if (!offsetsQueueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(offsetsQueueOpen);
        }
        THOR_THROW_IF_FALSE(values.getDescriptor() == descriptor.getValuesDescriptor());
        THOR_THROW_IF_FALSE(offsets.getDescriptor() == descriptor.getOffsetsDescriptor());
        raggedTensors.emplace(name, RaggedTensor(values, offsets));
    }
    statsGetBatchTensorUnloadWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(tensorUnloadStart), std::memory_order_relaxed);
    THOR_THROW_IF_FALSE(readyBatch.batchNum == nextBatchToDeliver);
    nextBatchToDeliver = (nextBatchToDeliver + 1) % batchesPerEpoch;

    const uint64_t delivered = statsBatchesDelivered.fetch_add(1, std::memory_order_relaxed) + 1;
    if (shouldEmitStats(delivered)) {
        emitStatsIfEnabled("get_batch", readyBatch.batchNum);
    }
}

uint64_t IndexedBatchAssembler::getNextBatchNum() {
    std::lock_guard<std::mutex> deliveryGuard(batchDeliveryMutex);
    throwIfWorkerFailed();
    return nextBatchToDeliver;
}

void IndexedBatchAssembler::validateReturnedTensorMapExact(const std::map<std::string, Tensor> &tensors) const {
    if (tensors.size() != batchTensorDescriptors.size()) {
        throw std::runtime_error("IndexedBatchAssembler returned tensor count does not match output tensor count.");
    }

    for (const auto &entry : batchTensorDescriptors) {
        const auto it = tensors.find(entry.first);
        if (it == tensors.end()) {
            throw std::runtime_error("IndexedBatchAssembler missing returned tensor: " + entry.first);
        }
        THOR_THROW_IF_FALSE(it->second.isInitialized());
        if (it->second.getDescriptor() != entry.second) {
            throw std::runtime_error("IndexedBatchAssembler returned tensor has wrong descriptor for: " + entry.first);
        }
    }

    for (const auto &entry : tensors) {
        if (batchTensorDescriptors.find(entry.first) == batchTensorDescriptors.end()) {
            throw std::runtime_error("IndexedBatchAssembler returned unexpected tensor: " + entry.first);
        }
    }
}

void IndexedBatchAssembler::validateReturnedRaggedTensorMapExact(
    const std::map<std::string, RaggedTensor> &raggedTensors) const {
    if (raggedTensors.size() != raggedTensorDescriptors.size()) {
        throw std::runtime_error(
            "IndexedBatchAssembler returned ragged tensor count does not match output ragged tensor count.");
    }
    for (const auto &[name, descriptor] : raggedTensorDescriptors) {
        const auto it = raggedTensors.find(name);
        if (it == raggedTensors.end()) {
            throw std::runtime_error("IndexedBatchAssembler missing returned ragged tensor: " + name);
        }
        THOR_THROW_IF_FALSE(it->second.isInitialized());
        if (it->second.getDescriptor() != descriptor) {
            throw std::runtime_error("IndexedBatchAssembler returned ragged tensor has wrong descriptor for: " + name);
        }
    }
    for (const auto &[name, ragged] : raggedTensors) {
        (void)ragged;
        if (raggedTensorDescriptors.find(name) == raggedTensorDescriptors.end()) {
            throw std::runtime_error("IndexedBatchAssembler returned unexpected ragged tensor: " + name);
        }
    }
}

void IndexedBatchAssembler::returnBuffers(
    const std::map<std::string, Tensor> &tensors,
    const std::map<std::string, RaggedTensor> &raggedTensors) {
    std::lock_guard<std::mutex> returnGuard(returnBuffersMutex);
    throwIfWorkerFailed();
    validateReturnedTensorMapExact(tensors);
    validateReturnedRaggedTensorMapExact(raggedTensors);
    statsReturnBufferCalls.fetch_add(1, std::memory_order_relaxed);
    const SteadyClock::time_point returnStart = diagnosticNow();
    for (const auto &entry : batchTensorQueues) {
        const bool queueOpen = entry.second->bufferUnloaded(tensors.at(entry.first));
        if (!queueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(queueOpen);
        }
    }
    for (const auto &[name, ragged] : raggedTensors) {
        const bool valuesQueueOpen = raggedValuesQueues.at(name)->bufferUnloaded(ragged.getValues());
        if (!valuesQueueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(valuesQueueOpen);
        }
        const bool offsetsQueueOpen = raggedOffsetsQueues.at(name)->bufferUnloaded(ragged.getOffsets());
        if (!offsetsQueueOpen) {
            throwIfWorkerFailed();
            THOR_THROW_IF_FALSE(offsetsQueueOpen);
        }
    }
    statsReturnBufferWaitNanoseconds.fetch_add(diagnosticElapsedNanoseconds(returnStart), std::memory_order_relaxed);
    const uint64_t returned = statsBatchBuffersReturned.fetch_add(1, std::memory_order_relaxed) + 1;
    if (shouldEmitStats(returned)) {
        emitStatsIfEnabled("return_batch", 0);
    }
}

uint64_t IndexedBatchAssembler::getNumBatchesPerEpoch() const { return batchesPerEpoch; }

uint64_t IndexedBatchAssembler::getNumExamples() const { return static_cast<uint64_t>(indices->size()); }
