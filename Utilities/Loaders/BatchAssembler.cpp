#include "Utilities/Loaders/BatchAssembler.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TarFile/UringDirect.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

using namespace std;

namespace {

uint64_t clampUint64(uint64_t value, uint64_t low, uint64_t high) {
    return std::max(low, std::min(value, high));
}


bool queueDiagnosticsEnabled() {
    const char* enabled = std::getenv("THOR_TRAINING_QUEUE_DIAGNOSTICS");
    return enabled != nullptr && enabled[0] != '\0' && !(enabled[0] == '0' && enabled[1] == '\0');
}

uint64_t queueDiagnosticsEvery() {
    const char* value = std::getenv("THOR_TRAINING_QUEUE_DIAGNOSTICS_EVERY");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
        return 1;
    }
    return static_cast<uint64_t>(parsed);
}

bool shouldEmitQueueDiagnostic(uint64_t index, uint64_t waitMicros = 0) {
    const uint64_t every = queueDiagnosticsEvery();
    return waitMicros > 0 || index <= 3 || (every != 0 && (index % every) == 0);
}

const char* exampleTypeName(ExampleType exampleType) {
    switch (exampleType) {
        case ExampleType::TRAIN:
            return "train";
        case ExampleType::VALIDATE:
            return "validate";
        case ExampleType::TEST:
            return "test";
        default:
            return "unknown";
    }
}

uint64_t microsSince(std::chrono::high_resolution_clock::time_point start,
                     std::chrono::high_resolution_clock::time_point finish) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
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

}  // namespace

const half BatchAssembler::HALF_ONE = (half)1.0f;

BatchAssembler::BatchAssembler(vector<std::shared_ptr<Shard>> shards,
                               ExampleType exampleType,
                               TensorDescriptor exampleDescriptor,
                               TensorDescriptor labelDescriptor,
                               uint64_t batchSize,
                               uint64_t batchQueueDepth) {
    THOR_THROW_IF_FALSE(!shards.empty());
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(batchQueueDepth > 0);

    this->exampleType = exampleType;
    this->batchSize = batchSize;
    this->batchQueueDepth = batchQueueDepth;
    this->exampleDescriptor = exampleDescriptor;

    vector<uint64_t> batchDimensions;
    batchDimensions.push_back(batchSize);
    for (uint64_t i = 0; i < exampleDescriptor.getDimensions().size(); ++i)
        batchDimensions.push_back(exampleDescriptor.getDimensions()[i]);
    batchDataTensorDescriptor = TensorDescriptor(exampleDescriptor.getDataType(), batchDimensions);
    this->exampleDescriptor = exampleDescriptor;

    examplePayloadSizeInBytes = exampleDescriptor.getArraySizeInBytes();
    labelPayloadSizeInBytes = labelDescriptor.getArraySizeInBytes();
    shardRecordSizeInBytes = shards[0]->getExampleSizeInBytes();
    inlinePayloadLabels = exampleDescriptor.getDataType() == labelDescriptor.getDataType() &&
                          shardRecordSizeInBytes == examplePayloadSizeInBytes + labelPayloadSizeInBytes;
    shiftedInlinePayloadLabels = exampleDescriptor.getDataType() == DataType::UINT8 &&
                                 labelDescriptor.getDataType() == DataType::UINT8 &&
                                 examplePayloadSizeInBytes == labelPayloadSizeInBytes &&
                                 shardRecordSizeInBytes == examplePayloadSizeInBytes + 1;
    if (!inlinePayloadLabels && !shiftedInlinePayloadLabels) {
        THOR_THROW_IF_FALSE(shardRecordSizeInBytes == examplePayloadSizeInBytes);
    }
    shardReadQueueDepth = computeShardReadQueueDepth(shardRecordSizeInBytes);
    shardExampleQueueDepth = computeShardExampleQueueDepth(batchSize, shardReadQueueDepth);

    numExamples = 0;
    for (uint64_t i = 0; i < shards.size(); ++i) {
        THOR_THROW_IF_FALSE(shards[i]->isOpen());
        THOR_THROW_IF_FALSE(shards[i]->getExampleSizeInBytes() == shardRecordSizeInBytes);

        const uint64_t shardExamples = shards[i]->getNumExamples(exampleType);
        if (shardExamples == 0) {
            continue;
        }
        this->shards.push_back(shards[i]);
        numExamples += shardExamples;
        numExamplesPerShard.push_back(shardExamples);
        randomizers.emplace_back(new FullPeriodRandom(shardExamples, false));
    }
    THOR_THROW_IF_FALSE(!this->shards.empty());

    const std::vector<std::string> &allClasses = shards[0]->getAllClasses();
    for (uint64_t c = 0; c < allClasses.size(); ++c) {
        const string &className = allClasses[c];
        if (classIndexes.count(className) == 0) {
            uint64_t curNumClasses = classIndexes.size();
            classIndexes[className] = curNumClasses;
        }
    }

    vector<uint64_t> labelDimensions = labelDescriptor.getDimensions();
    DataType labelsDataType = labelDescriptor.getDataType();
    perClassLabels = false;
    classIndexLabels = false;
    if (!inlinePayloadLabels && !shiftedInlinePayloadLabels) {
        THOR_THROW_IF_FALSE(labelDimensions.size() == 1);
        perClassLabels = labelDimensions[0] == classIndexes.size() &&
                         (labelsDataType == DataType::UINT8 || labelsDataType == DataType::FP16 ||
                          labelsDataType == DataType::FP32);
        classIndexLabels = labelDimensions[0] == 1 &&
                           (labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                            labelsDataType == DataType::UINT32);
        THOR_THROW_IF_FALSE(perClassLabels ^ classIndexLabels);
    }

    vector<uint64_t> batchedLabelDimensions;
    batchedLabelDimensions.push_back(batchSize);
    batchedLabelDimensions.insert(batchedLabelDimensions.end(), labelDimensions.begin(), labelDimensions.end());
    batchLabelTensorDescriptor = ThorImplementation::TensorDescriptor(labelDescriptor.getDataType(), batchedLabelDimensions);

    batchesPerEpoch = (numExamples + (batchSize - 1)) / batchSize;

    if (queueDiagnosticsEnabled()) {
        std::fprintf(stderr,
                     "THOR_TRAINING_QUEUE_DIAGNOSTIC loader_config phase=%s shards=%zu examples=%lu batches_per_epoch=%lu batch_size=%lu batch_queue_depth=%lu example_bytes=%lu shard_example_queue_depth=%lu shard_read_queue_depth=%lu\n",
                     exampleTypeName(exampleType),
                     shards.size(),
                     numExamples,
                     batchesPerEpoch,
                     batchSize,
                     batchQueueDepth,
                     shardRecordSizeInBytes,
                     shardExampleQueueDepth,
                     shardReadQueueDepth);
        std::fflush(stderr);
    }

    open();
}

BatchAssembler::~BatchAssembler() { close(); }

void BatchAssembler::open() {
    THOR_THROW_IF_FALSE(!batchDataQueue.isOpen());
    THOR_THROW_IF_FALSE(!batchLabelQueue.isOpen());
    THOR_THROW_IF_FALSE(shardQueues.empty());

    for (uint64_t i = 0; i < shards.size(); ++i) {
        shardQueues.emplace_back(new AsyncQueue<LabeledExample>(static_cast<uint32_t>(shardExampleQueueDepth)));
        shardQueues.back()->open();
        randomizers[i]->reseed();

        shardThreads.emplace_back(&BatchAssembler::shardReaderThread, this, i);
    }
    currentBatchNum = 0;
    currentExampleNum = 0;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    batchDataQueue.resize(batchQueueDepth, batchDataTensorDescriptor, cpuPlacement);
    batchDataQueue.open();
    batchLabelQueue.resize(batchQueueDepth, batchLabelTensorDescriptor, cpuPlacement);
    batchLabelQueue.open();
    batchNumQueue.resize(batchQueueDepth);
    batchNumQueue.open();

    assemblerThread = thread(&BatchAssembler::batchAssemblerThread, this);
}

void BatchAssembler::close() {
    for (uint64_t i = 0; i < shardQueues.size(); ++i)
        shardQueues[i]->close();
    batchDataQueue.close();
    batchLabelQueue.close();
    batchNumQueue.close();

    for (uint64_t i = 0; i < shardThreads.size(); ++i)
        shardThreads[i].join();
    assemblerThread.join();

    shardQueues.clear();
}

void BatchAssembler::shardReaderThread(uint64_t shard) {
    struct PendingRead {
        LabeledExample labeledExample;
        uint64_t expectedBytes = 0;
    };

    const uint64_t maxInFlightReads = shardReadQueueDepth;

    const uint64_t exampleSizeInBytes = shards[shard]->getExampleSizeInBytes();
    THOR_THROW_IF_FALSE(exampleSizeInBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

    UringDirect cachedReader(static_cast<unsigned>(maxInFlightReads));
    cachedReader.registerCachedLoadFile(shards[shard]->getFilename());

    std::deque<PendingRead> pendingReads;

    auto submitOneRead = [&]() {
        pendingReads.emplace_back();
        PendingRead &pendingRead = pendingReads.back();
        pendingRead.labeledExample.data.resize(exampleSizeInBytes);

        ShardExampleReadRequest request = shards[shard]->getExampleReadRequest(exampleType, randomizers[shard]->getRandomNumber());
        THOR_THROW_IF_FALSE(request.numBytes == exampleSizeInBytes);
        pendingRead.expectedBytes = request.numBytes;
        pendingRead.labeledExample.label = std::move(request.label);
        pendingRead.labeledExample.filename = std::move(request.filename);

        while (!cachedReader.submitReadCached(pendingRead.labeledExample.data.data(),
                                              request.fileOffsetBytes,
                                              static_cast<uint32_t>(request.numBytes))) {
            cachedReader.submit();
        }
        diagnosticReadsSubmitted.fetch_add(1, std::memory_order_relaxed);
    };

    auto waitOneRead = [&]() {
        THOR_THROW_IF_FALSE(!pendingReads.empty());
        UringDirect::Completion completion = cachedReader.waitCompletionInOrder();
        PendingRead pendingRead = std::move(pendingReads.front());
        pendingReads.pop_front();

        if (completion.responseCode < 0) {
            throw std::runtime_error("cached io_uring batch-loader read failed for shard '" + shards[shard]->getFilename() +
                                     "': " + std::strerror(-completion.responseCode));
        }
        THOR_THROW_IF_FALSE(static_cast<uint64_t>(completion.responseCode) == pendingRead.expectedBytes);
        diagnosticReadsCompleted.fetch_add(1, std::memory_order_relaxed);
        return pendingRead.labeledExample;
    };

    auto drainPendingReads = [&]() {
        cachedReader.submit();
        while (!pendingReads.empty()) {
            (void)waitOneRead();
        }
    };

    while (true) {
        while (pendingReads.size() < maxInFlightReads) {
            if (!shardQueues[shard]->isOpen()) {
                drainPendingReads();
                return;
            }
            submitOneRead();
        }

        cachedReader.submit();
        LabeledExample labeledExample = waitOneRead();

        const auto pushStart = std::chrono::high_resolution_clock::now();
        bool queueOpen = shardQueues[shard]->push(labeledExample);
        const auto pushFinish = std::chrono::high_resolution_clock::now();
        diagnosticShardQueuePushWaitMicros.fetch_add(microsSince(pushStart, pushFinish), std::memory_order_relaxed);
        if (queueOpen) {
            diagnosticExamplesPushed.fetch_add(1, std::memory_order_relaxed);
        }
        if (!queueOpen) {
            drainPendingReads();
            return;
        }
    }
}

// There can be only 1 batchAssemblerThread, it is designed expecting that there is just the one.
void BatchAssembler::batchAssemblerThread() {
    const uint64_t shardExampleSizeInBytes = shards[0]->getExampleSizeInBytes();
    THOR_THROW_IF_FALSE(shardExampleSizeInBytes == shardRecordSizeInBytes);

    uint64_t numExamplesInEpoch = 0;
    for (uint64_t i = 0; i < numExamplesPerShard.size(); ++i)
        numExamplesInEpoch += numExamplesPerShard[i];

    uint64_t numExamplesLeftInEpoch = 0;
    vector<uint64_t> numExamplesLeftPerShard = numExamplesPerShard;

    bool queueOpen;
    LabeledExample labeledExample;
    Tensor batchDataBuffer;
    Tensor batchLabelsBuffer;
    uint64_t batchSlotOffset;

    uint64_t batchSlot = 0;
    while (1) {
        if (numExamplesLeftInEpoch == 0) {
            numExamplesLeftInEpoch = numExamplesInEpoch;
            for (uint64_t i = 0; i < numExamplesPerShard.size(); ++i) {
                numExamplesLeftPerShard[i] = numExamplesPerShard[i];
            }
        }

        if (batchSlot == 0) {
            const auto bufferWaitStart = std::chrono::high_resolution_clock::now();
            queueOpen = batchDataQueue.getBufferToLoad(batchDataBuffer);
            if (!queueOpen)
                return;
            queueOpen = batchLabelQueue.getBufferToLoad(batchLabelsBuffer);
            const auto bufferWaitFinish = std::chrono::high_resolution_clock::now();
            diagnosticBatchBufferWaitMicros.fetch_add(microsSince(bufferWaitStart, bufferWaitFinish), std::memory_order_relaxed);
            if (!queueOpen) {
                batchDataQueue.bufferLoaded(batchDataBuffer);
                return;
            }
        }

        uint64_t rand0 = rand() & 0xFFFF;
        uint64_t rand1 = rand() & 0xFFFF;
        uint64_t rand2 = rand() & 0xFFFF;
        uint64_t rand3 = rand() & 0xFFFF;
        uint64_t randomNumber = (rand0 << 48) | (rand1 << 32) | (rand2 << 16) | rand3;
        randomNumber %= numExamplesLeftInEpoch;

        uint64_t priorExamples = 0;
        uint64_t chosenShard = numExamplesPerShard.size();
        for (uint64_t i = 0; i < numExamplesPerShard.size(); ++i) {
            priorExamples += numExamplesLeftPerShard[i];
            if (priorExamples > randomNumber) {
                numExamplesLeftPerShard[i] -= 1;
                numExamplesLeftInEpoch -= 1;
                chosenShard = i;
                break;
            }
        }
        THOR_THROW_IF_FALSE(chosenShard < numExamplesPerShard.size());

        const auto popStart = std::chrono::high_resolution_clock::now();
        queueOpen = shardQueues[chosenShard]->pop(labeledExample);
        const auto popFinish = std::chrono::high_resolution_clock::now();
        diagnosticShardQueuePopWaitMicros.fetch_add(microsSince(popStart, popFinish), std::memory_order_relaxed);
        if (!queueOpen) {
            batchDataQueue.bufferLoaded(batchDataBuffer);
            batchLabelQueue.bufferLoaded(batchLabelsBuffer);
            return;
        }

        // Load data to pinned memory buffer.  Shards may either carry only the example payload and use the
        // directory/class metadata as labels, or carry an inline [example][label] payload.  The inline path is
        // used by sequence datasets such as byte-level language modeling, where each record has many target ids.
        batchSlotOffset = examplePayloadSizeInBytes * batchSlot;
        memcpy((uint8_t *)batchDataBuffer.getMemPtr() + batchSlotOffset, labeledExample.data.data(), examplePayloadSizeInBytes);

        // Load labels to pinned memory buffer.
        DataType labelsDataType = batchLabelTensorDescriptor.getDataType();
        if (inlinePayloadLabels) {
            batchSlotOffset = labelPayloadSizeInBytes * batchSlot;
            memcpy((uint8_t *)batchLabelsBuffer.getMemPtr() + batchSlotOffset,
                   labeledExample.data.data() + examplePayloadSizeInBytes,
                   labelPayloadSizeInBytes);
        } else if (shiftedInlinePayloadLabels) {
            batchSlotOffset = labelPayloadSizeInBytes * batchSlot;
            memcpy((uint8_t *)batchLabelsBuffer.getMemPtr() + batchSlotOffset,
                   labeledExample.data.data() + 1,
                   labelPayloadSizeInBytes);
        } else if (perClassLabels) {
            THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::FP16 ||
                   labelsDataType == DataType::FP32);

            batchSlotOffset = classIndexes.size() * batchSlot;
            if (labelsDataType == DataType::UINT8) {
                uint8_t *batchLabels = (uint8_t *)batchLabelsBuffer.getMemPtr() + batchSlotOffset;
                memset(batchLabels, 0, sizeof(uint8_t) * classIndexes.size());
                batchLabels[classIndexes[labeledExample.label]] = (uint8_t)1;
            } else if (labelsDataType == DataType::FP16) {
                half *batchLabels = (half *)batchLabelsBuffer.getMemPtr() + batchSlotOffset;
                memset(batchLabels, 0, sizeof(half) * classIndexes.size());
                batchLabels[classIndexes[labeledExample.label]] = HALF_ONE;
            } else if (labelsDataType == DataType::FP32) {
                float *batchLabels = (float *)batchLabelsBuffer.getMemPtr() + batchSlotOffset;
                memset(batchLabels, 0, sizeof(float) * classIndexes.size());
                batchLabels[classIndexes[labeledExample.label]] = 1.0f;
                // printf("\n\rwrote %f to %ld for label %s\n", batchLabels[classIndexes[labeledExample.label]],
                // classIndexes[labeledExample.label], labeledExample.label.c_str()); for(auto it = classIndexes.begin(); it !=
                // classIndexes.end(); ++it)
                //    printf("\r%s:%ld\n", it->first.c_str(), it->second);
            } else {
                THOR_UNREACHABLE();
            }
        } else {  // class index labels
            THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                   labelsDataType == DataType::UINT32);
            if (labelsDataType == DataType::UINT8) {
                uint8_t *batchLabels = (uint8_t *)batchLabelsBuffer.getMemPtr();
                batchLabels[batchSlot] = (uint8_t)classIndexes[labeledExample.label];
            } else if (labelsDataType == DataType::UINT16) {
                uint16_t *batchLabels = (uint16_t *)batchLabelsBuffer.getMemPtr();
                batchLabels[batchSlot] = (uint16_t)classIndexes[labeledExample.label];
            } else if (labelsDataType == DataType::UINT32) {
                uint32_t *batchLabels = (uint32_t *)batchLabelsBuffer.getMemPtr();
                batchLabels[batchSlot] = (uint32_t)classIndexes[labeledExample.label];
            } else {
                THOR_UNREACHABLE();
            }
        }

        currentExampleNum += 1;

        batchSlot += 1;
        if (batchSlot == batchSize) {
            batchSlot = 0;

            batchDataQueue.bufferLoaded(batchDataBuffer);
            batchLabelQueue.bufferLoaded(batchLabelsBuffer);

            batchNumQueue.push(currentBatchNum);
            const uint64_t assembled = diagnosticBatchesAssembled.fetch_add(1, std::memory_order_relaxed) + 1;
            if (queueDiagnosticsEnabled() && shouldEmitQueueDiagnostic(assembled)) {
                emitQueueDiagnostics("assembled_batch", currentBatchNum);
            }
            currentBatchNum += 1;
            if (currentBatchNum == batchesPerEpoch) {
                // An epoch may not be exactly divisible by an integer number of batches, make sure this does not cause drift.
                uint64_t batchProgress = currentExampleNum % numExamples;
                uint64_t batchesLeftInEpoch = ((numExamples - batchProgress) + (batchSize - 1)) / batchSize;
                currentBatchNum = batchesPerEpoch - batchesLeftInEpoch;
            }
        }
    }
}


void BatchAssembler::emitQueueDiagnostics(const char* event, uint64_t batchNum, uint64_t waitMicros) {
    if (!queueDiagnosticsEnabled()) {
        return;
    }

    AsyncTensorQueueSnapshot dataState = batchDataQueue.snapshot();
    AsyncTensorQueueSnapshot labelState = batchLabelQueue.snapshot();
    const int batchNumOccupancy = batchNumQueue.occupancy();
    const int batchNumCapacity = batchNumQueue.capacity();

    int shardMin = 0;
    int shardMax = 0;
    uint64_t shardSum = 0;
    int shardCapacity = 0;
    for (size_t i = 0; i < shardQueues.size(); ++i) {
        const int occupancy = shardQueues[i]->occupancy();
        const int capacity = shardQueues[i]->capacity();
        if (i == 0 || occupancy < shardMin) {
            shardMin = occupancy;
        }
        if (i == 0 || occupancy > shardMax) {
            shardMax = occupancy;
        }
        shardSum += static_cast<uint64_t>(occupancy);
        shardCapacity += capacity;
    }

    std::fprintf(stderr,
                 "THOR_TRAINING_QUEUE_DIAGNOSTIC loader event=%s phase=%s batch=%lu wait_us=%lu assembled=%lu loaded=%lu returned=%lu "
                 "reads(submitted=%lu completed=%lu pushed=%lu) wait_totals_us(batch_buffer=%lu shard_pop=%lu shard_push=%lu) "
                 "data(empty=%d loading=%d loaded=%d unloading=%d cap=%d) "
                 "labels(empty=%d loading=%d loaded=%d unloading=%d cap=%d) batch_nums=%d/%d shard_examples(sum=%lu min=%d max=%d cap=%d)\n",
                 event,
                 exampleTypeName(exampleType),
                 batchNum,
                 waitMicros,
                 diagnosticBatchesAssembled.load(std::memory_order_relaxed),
                 diagnosticBatchesLoaded.load(std::memory_order_relaxed),
                 diagnosticBuffersReturned.load(std::memory_order_relaxed),
                 diagnosticReadsSubmitted.load(std::memory_order_relaxed),
                 diagnosticReadsCompleted.load(std::memory_order_relaxed),
                 diagnosticExamplesPushed.load(std::memory_order_relaxed),
                 diagnosticBatchBufferWaitMicros.load(std::memory_order_relaxed),
                 diagnosticShardQueuePopWaitMicros.load(std::memory_order_relaxed),
                 diagnosticShardQueuePushWaitMicros.load(std::memory_order_relaxed),
                 dataState.empty,
                 dataState.loading,
                 dataState.loaded,
                 dataState.unloading,
                 dataState.capacity,
                 labelState.empty,
                 labelState.loading,
                 labelState.loaded,
                 labelState.unloading,
                 labelState.capacity,
                 batchNumOccupancy,
                 batchNumCapacity,
                 shardSum,
                 shardMin,
                 shardMax,
                 shardCapacity);
    std::fflush(stderr);
}

void BatchAssembler::getBatch(Tensor &batchTensor, Tensor &labelTensor, uint64_t &batchNum) {
    const auto start = std::chrono::high_resolution_clock::now();
    bool queueOpen;
    queueOpen = batchDataQueue.getBufferToUnload(batchTensor);
    THOR_THROW_IF_FALSE(queueOpen);
    queueOpen = batchLabelQueue.getBufferToUnload(labelTensor);
    THOR_THROW_IF_FALSE(queueOpen);
    queueOpen = batchNumQueue.pop(batchNum);
    THOR_THROW_IF_FALSE(queueOpen);
    const auto finish = std::chrono::high_resolution_clock::now();

    const uint64_t loaded = diagnosticBatchesLoaded.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t waitMicros = microsSince(start, finish);
    if (queueDiagnosticsEnabled() && shouldEmitQueueDiagnostic(loaded, waitMicros)) {
        emitQueueDiagnostics("get_batch", batchNum, waitMicros);
    }
}

uint64_t BatchAssembler::getNextBatchNum() {
    uint64_t nextBatchNum;
    bool queueOpen = batchNumQueue.peek(nextBatchNum);
    THOR_THROW_IF_FALSE(queueOpen);
    return nextBatchNum;
}

void BatchAssembler::returnBuffer(Tensor &batchTensor, Tensor &labelTensor) {
    bool queueOpen;
    queueOpen = batchDataQueue.bufferUnloaded(batchTensor);
    THOR_THROW_IF_FALSE(queueOpen);
    queueOpen = batchLabelQueue.bufferUnloaded(labelTensor);
    THOR_THROW_IF_FALSE(queueOpen);

    const uint64_t returned = diagnosticBuffersReturned.fetch_add(1, std::memory_order_relaxed) + 1;
    if (queueDiagnosticsEnabled() && shouldEmitQueueDiagnostic(returned)) {
        emitQueueDiagnostics("return_batch", 0);
    }
}

uint64_t BatchAssembler::getNumBatchesPerEpoch() { return batchesPerEpoch; }
