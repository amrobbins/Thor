#include "Utilities/Loaders/LocalNamedBatchAssembler.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TarFile/UringDirect.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

namespace {

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

uint64_t computeShardExampleQueueDepth(uint64_t batchSize, uint64_t shardReadQueueDepth) {
    constexpr uint64_t MIN_EXAMPLES = 32;
    constexpr uint64_t MAX_EXAMPLES = 4096;
    return clampUint64(std::max(batchSize, shardReadQueueDepth), MIN_EXAMPLES, MAX_EXAMPLES);
}

uint64_t mixSeed(uint64_t seed, uint64_t salt) {
    uint64_t value = seed + 0x9e3779b97f4a7c15ull + (salt << 6) + (salt >> 2);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

uint64_t entropySeed() {
    std::random_device randomDevice;
    const uint64_t hi = static_cast<uint64_t>(randomDevice()) << 32;
    const uint64_t lo = static_cast<uint64_t>(randomDevice());
    return hi ^ lo ^ static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

}  // namespace

LocalNamedBatchAssembler::LocalNamedBatchAssembler(std::vector<std::shared_ptr<Shard>> shards,
                                                   ExampleType exampleType,
                                                   LocalNamedExampleLayout layout,
                                                   uint64_t batchSize,
                                                   uint64_t batchQueueDepth,
                                                   bool randomizeExamples,
                                                   std::optional<uint64_t> seed)
    : exampleType(exampleType),
      layout(std::move(layout)),
      batchSize(batchSize),
      batchQueueDepth(batchQueueDepth),
      shardReadQueueDepth(0),
      shardExampleQueueDepth(0),
      recordSizeBytes(this->layout.recordSizeBytes()),
      numExamples(0),
      batchesPerEpoch(0),
      randomizeExamples(randomizeExamples),
      seed(seed),
      shardSelectionGenerator(seed.has_value() ? mixSeed(seed.value(), 0xfeedbeefull) : entropySeed()) {
    THOR_THROW_IF_FALSE(!shards.empty());
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(batchQueueDepth > 0);
    this->layout.validate();
    THOR_THROW_IF_FALSE(recordSizeBytes > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

    shardReadQueueDepth = computeShardReadQueueDepth(recordSizeBytes);
    shardExampleQueueDepth = computeShardExampleQueueDepth(batchSize, shardReadQueueDepth);

    for (std::shared_ptr<Shard> &shard : shards) {
        THOR_THROW_IF_FALSE(shard);
        THOR_THROW_IF_FALSE(shard->isOpen());
        THOR_THROW_IF_FALSE(shard->getExampleSizeInBytes() == recordSizeBytes);
        THOR_THROW_IF_FALSE(shard->getDataType() == this->layout.dataType());

        const uint64_t shardExamples = shard->getNumExamples(exampleType);
        if (shardExamples == 0) {
            continue;
        }
        this->shards.push_back(shard);
        numExamplesPerShard.push_back(shardExamples);
        numExamples += shardExamples;
        randomizers.emplace_back(std::make_unique<FullPeriodRandom>(shardExamples, false));
        if (randomizeExamples && seed.has_value()) {
            randomizers.back()->reseed(mixSeed(seed.value(), this->shards.size()));
        }
    }
    THOR_THROW_IF_FALSE(!this->shards.empty());
    THOR_THROW_IF_FALSE(numExamples > 0);

    batchesPerEpoch = (numExamples + (batchSize - 1)) / batchSize;

    for (const LocalNamedExampleLayout::TensorSpec &spec : this->layout.tensors()) {
        std::vector<uint64_t> dimensions;
        dimensions.reserve(spec.dimensions.size() + 1);
        dimensions.push_back(batchSize);
        dimensions.insert(dimensions.end(), spec.dimensions.begin(), spec.dimensions.end());
        batchTensorDescriptors.emplace(spec.name, TensorDescriptor(spec.dataType, dimensions));
    }

    open();
}

LocalNamedBatchAssembler::~LocalNamedBatchAssembler() { close(); }

void LocalNamedBatchAssembler::open() {
    THOR_THROW_IF_FALSE(shardQueues.empty());
    THOR_THROW_IF_FALSE(batchTensorQueues.empty());

    for (uint64_t i = 0; i < shards.size(); ++i) {
        shardQueues.emplace_back(std::make_unique<AsyncQueue<LocalNamedExampleRecord>>(static_cast<uint32_t>(shardExampleQueueDepth)));
        shardQueues.back()->open();
        shardThreads.emplace_back(&LocalNamedBatchAssembler::shardReaderThread, this, i);
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        auto queue = std::make_unique<AsyncTensorQueue>();
        queue->resize(batchQueueDepth, batchTensorDescriptors.at(spec.name), cpuPlacement);
        queue->open();
        batchTensorQueues.emplace(spec.name, std::move(queue));
    }

    batchNumQueue.resize(static_cast<uint32_t>(batchQueueDepth));
    batchNumQueue.open();

    assemblerThread = std::thread(&LocalNamedBatchAssembler::batchAssemblerThread, this);
}

void LocalNamedBatchAssembler::close() {
    for (std::unique_ptr<AsyncQueue<LocalNamedExampleRecord>> &queue : shardQueues) {
        if (queue) {
            queue->close();
        }
    }
    for (auto &entry : batchTensorQueues) {
        if (entry.second) {
            entry.second->close();
        }
    }
    batchNumQueue.close();

    for (std::thread &thread : shardThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (assemblerThread.joinable()) {
        assemblerThread.join();
    }

    shardQueues.clear();
    batchTensorQueues.clear();
    shardThreads.clear();
}

void LocalNamedBatchAssembler::shardReaderThread(uint64_t shardIndex) {
    struct PendingRead {
        LocalNamedExampleRecord record;
        uint64_t expectedBytes = 0;
    };

    const uint64_t maxInFlightReads = shardReadQueueDepth;
    const uint64_t shardExamples = numExamplesPerShard.at(shardIndex);
    uint64_t sequentialExampleIndex = 0;

    UringDirect cachedReader(static_cast<unsigned>(maxInFlightReads));
    cachedReader.registerCachedLoadFile(shards.at(shardIndex)->getFilename());

    std::deque<PendingRead> pendingReads;

    auto nextExampleIndex = [&]() -> uint64_t {
        if (randomizeExamples) {
            return randomizers.at(shardIndex)->getRandomNumber();
        }
        const uint64_t index = sequentialExampleIndex;
        sequentialExampleIndex = (sequentialExampleIndex + 1) % shardExamples;
        return index;
    };

    auto submitOneRead = [&]() {
        pendingReads.emplace_back();
        PendingRead &pendingRead = pendingReads.back();
        pendingRead.record.data.resize(recordSizeBytes);

        ShardExampleReadRequest request = shards.at(shardIndex)->getExampleReadRequest(exampleType, nextExampleIndex());
        THOR_THROW_IF_FALSE(request.numBytes == recordSizeBytes);
        pendingRead.expectedBytes = request.numBytes;

        while (!cachedReader.submitReadCached(pendingRead.record.data.data(),
                                              request.fileOffsetBytes,
                                              static_cast<uint32_t>(request.numBytes))) {
            cachedReader.submit();
        }
    };

    auto waitOneRead = [&]() -> LocalNamedExampleRecord {
        THOR_THROW_IF_FALSE(!pendingReads.empty());
        UringDirect::Completion completion = cachedReader.waitCompletionInOrder();
        PendingRead pendingRead = std::move(pendingReads.front());
        pendingReads.pop_front();

        if (completion.responseCode < 0) {
            throw std::runtime_error("cached io_uring local named batch-loader read failed for shard '" +
                                     shards.at(shardIndex)->getFilename() + "': " + std::strerror(-completion.responseCode));
        }
        THOR_THROW_IF_FALSE(static_cast<uint64_t>(completion.responseCode) == pendingRead.expectedBytes);
        return std::move(pendingRead.record);
    };

    auto drainPendingReads = [&]() {
        cachedReader.submit();
        while (!pendingReads.empty()) {
            (void)waitOneRead();
        }
    };

    while (true) {
        while (pendingReads.size() < maxInFlightReads) {
            if (!shardQueues.at(shardIndex)->isOpen()) {
                drainPendingReads();
                return;
            }
            submitOneRead();
        }

        cachedReader.submit();
        LocalNamedExampleRecord record = waitOneRead();
        if (!shardQueues.at(shardIndex)->push(record)) {
            drainPendingReads();
            return;
        }
    }
}

uint64_t LocalNamedBatchAssembler::nextShardSelection(uint64_t numExamplesLeftInEpoch,
                                                      const std::vector<uint64_t> &numExamplesLeftPerShard) {
    THOR_THROW_IF_FALSE(numExamplesLeftInEpoch > 0);
    THOR_THROW_IF_FALSE(numExamplesLeftPerShard.size() == numExamplesPerShard.size());

    uint64_t selectedExample = 0;
    if (randomizeExamples) {
        selectedExample = shardSelectionGenerator() % numExamplesLeftInEpoch;
    }

    uint64_t priorExamples = 0;
    for (uint64_t i = 0; i < numExamplesLeftPerShard.size(); ++i) {
        priorExamples += numExamplesLeftPerShard[i];
        if (priorExamples > selectedExample) {
            return i;
        }
    }
    THOR_UNREACHABLE();
}

void LocalNamedBatchAssembler::batchAssemblerThread() {
    const uint64_t numExamplesInEpoch = numExamples;
    uint64_t numExamplesLeftInEpoch = 0;
    std::vector<uint64_t> numExamplesLeftPerShard = numExamplesPerShard;

    std::map<std::string, Tensor> batchTensors;
    LocalNamedExampleRecord record;
    uint64_t batchSlot = 0;
    uint64_t currentBatchNum = 0;
    uint64_t currentExampleNum = 0;

    while (true) {
        if (numExamplesLeftInEpoch == 0) {
            numExamplesLeftInEpoch = numExamplesInEpoch;
            numExamplesLeftPerShard = numExamplesPerShard;
        }

        if (batchSlot == 0) {
            batchTensors.clear();
            for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
                Tensor tensor;
                if (!batchTensorQueues.at(spec.name)->getBufferToLoad(tensor)) {
                    return;
                }
                batchTensors.emplace(spec.name, tensor);
            }
        }

        const uint64_t chosenShard = nextShardSelection(numExamplesLeftInEpoch, numExamplesLeftPerShard);
        THOR_THROW_IF_FALSE(chosenShard < numExamplesLeftPerShard.size());
        THOR_THROW_IF_FALSE(numExamplesLeftPerShard[chosenShard] > 0);
        numExamplesLeftPerShard[chosenShard] -= 1;
        numExamplesLeftInEpoch -= 1;

        if (!shardQueues.at(chosenShard)->pop(record)) {
            for (const auto &entry : batchTensors) {
                batchTensorQueues.at(entry.first)->bufferLoaded(entry.second);
            }
            return;
        }
        THOR_THROW_IF_FALSE(record.data.size() == recordSizeBytes);

        for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
            Tensor &tensor = batchTensors.at(spec.name);
            std::memcpy(static_cast<uint8_t *>(tensor.getMemPtr()) + spec.numBytes * batchSlot,
                        record.data.data() + spec.offsetBytes,
                        spec.numBytes);
        }

        currentExampleNum += 1;
        batchSlot += 1;
        if (batchSlot == batchSize) {
            batchSlot = 0;

            for (const auto &entry : batchTensors) {
                batchTensorQueues.at(entry.first)->bufferLoaded(entry.second);
            }
            if (!batchNumQueue.push(currentBatchNum)) {
                return;
            }

            currentBatchNum += 1;
            if (currentBatchNum == batchesPerEpoch) {
                const uint64_t batchProgress = currentExampleNum % numExamples;
                const uint64_t batchesLeftInEpoch = ((numExamples - batchProgress) + (batchSize - 1)) / batchSize;
                currentBatchNum = batchesPerEpoch - batchesLeftInEpoch;
            }
        }
    }
}

void LocalNamedBatchAssembler::getBatch(std::map<std::string, Tensor> &tensors, uint64_t &batchNum) {
    tensors.clear();
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        Tensor tensor;
        const bool queueOpen = batchTensorQueues.at(spec.name)->getBufferToUnload(tensor);
        THOR_THROW_IF_FALSE(queueOpen);
        tensors.emplace(spec.name, tensor);
    }
    const bool queueOpen = batchNumQueue.pop(batchNum);
    THOR_THROW_IF_FALSE(queueOpen);
}

uint64_t LocalNamedBatchAssembler::getNextBatchNum() {
    uint64_t nextBatchNum = 0;
    const bool queueOpen = batchNumQueue.peek(nextBatchNum);
    THOR_THROW_IF_FALSE(queueOpen);
    return nextBatchNum;
}

void LocalNamedBatchAssembler::validateReturnedTensorMapExact(const std::map<std::string, Tensor> &tensors) const {
    if (tensors.size() != layout.tensors().size()) {
        throw std::runtime_error("LocalNamedBatchAssembler returned tensor count does not match layout tensor count.");
    }

    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const auto it = tensors.find(spec.name);
        if (it == tensors.end()) {
            throw std::runtime_error("LocalNamedBatchAssembler missing returned tensor: " + spec.name);
        }
        THOR_THROW_IF_FALSE(it->second.isInitialized());
        if (it->second.getDescriptor() != batchTensorDescriptors.at(spec.name)) {
            throw std::runtime_error("LocalNamedBatchAssembler returned tensor has wrong descriptor for: " + spec.name);
        }
    }

    for (const auto &entry : tensors) {
        (void)layout.tensor(entry.first);
    }
}

void LocalNamedBatchAssembler::returnBuffers(const std::map<std::string, Tensor> &tensors) {
    validateReturnedTensorMapExact(tensors);
    for (const LocalNamedExampleLayout::TensorSpec &spec : layout.tensors()) {
        const bool queueOpen = batchTensorQueues.at(spec.name)->bufferUnloaded(tensors.at(spec.name));
        THOR_THROW_IF_FALSE(queueOpen);
    }
}

uint64_t LocalNamedBatchAssembler::getNumBatchesPerEpoch() const { return batchesPerEpoch; }

uint64_t LocalNamedBatchAssembler::getNumExamples() const { return numExamples; }
