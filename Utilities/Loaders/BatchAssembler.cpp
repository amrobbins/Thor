#include "Utilities/Loaders/BatchAssembler.h"

using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;

using std::shared_ptr;
using std::thread;
using std::unique_ptr;

BatchAssembler::BatchAssembler(vector<std::shared_ptr<Shard>> shards,
                               ExampleType exampleType,
                               TensorDescriptor exampleDescriptor,
                               uint64_t batchSize) {
    assert(!shards.empty());
    assert(batchSize > 0);

    this->shards = shards;
    this->exampleType = exampleType;
    this->batchSize = batchSize;
    this->exampleDescriptor = exampleDescriptor;

    vector<uint64_t> batchDimensions;
    batchDimensions.push_back(batchSize);
    for (uint64_t i = 0; i < exampleDescriptor.getDimensions().size(); ++i)
        batchDimensions.push_back(exampleDescriptor.getDimensions()[i]);
    batchDataTensorDescriptor = TensorDescriptor(exampleDescriptor.getDataType(), batchDimensions);
    this->exampleDescriptor = exampleDescriptor;

    numExamples = 0;
    for (uint64_t i = 0; i < shards.size(); ++i) {
        assert(shards[i]->isOpen());
        assert(shards[i]->getExampleSizeInBytes() == exampleDescriptor.getArraySizeInBytes());

        numExamples += shards[i]->getNumExamples(exampleType);
        numExamplesPerShard.push_back(shards[i]->getNumExamples(exampleType));
        randomizers.emplace_back(new FullPeriodRandom(numExamplesPerShard[i], false));
    }

    file_string_vector_t *allClasses = shards[0]->getAllClasses();
    for (uint64_t c = 0; c < allClasses->size(); ++c) {
        string className = (*allClasses)[c].c_str();
        if (classIndexes.count(className) == 0) {
            classIndexes[className] = classIndexes.size();
        }
    }

    // FIXME: datatype
    batchLabelTensorDescriptor = ThorImplementation::TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, classIndexes.size()});

    batchesPerEpoch = (numExamples + (batchSize - 1)) / batchSize;

    open();
}

BatchAssembler::~BatchAssembler() { close(); }

void BatchAssembler::open() {
    assert(!batchDataQueue.isOpen());
    assert(!batchLabelQueue.isOpen());
    assert(shardQueues.empty());

    for (uint64_t i = 0; i < shards.size(); ++i) {
        shardQueues.emplace_back(new AsyncQueue<LabeledExample>(32));
        shardQueues.back()->open();
        randomizers[i]->reseed();

        shardThreads.emplace_back(&BatchAssembler::shardReaderThread, this, i);
    }
    currentBatchNum = 0;
    currentExampleNum = 0;

    batchDataQueue.resize(32, batchDataTensorDescriptor, TensorPlacement::MemDevices::CPU);
    batchDataQueue.open();
    batchLabelQueue.resize(32, batchLabelTensorDescriptor, TensorPlacement::MemDevices::CPU);
    batchLabelQueue.open();
    batchNumQueue.resize(32);
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
    bool queueOpen;
    LabeledExample labeledExample;
    labeledExample.data.resize(shards[shard]->getExampleSizeInBytes());

    while (1) {
        shards[shard]->loadExample(
            labeledExample.data.data(), labeledExample.label, labeledExample.filename, exampleType, randomizers[shard]->getRandomNumber());

        queueOpen = shardQueues[shard]->push(labeledExample);
        if (!queueOpen)
            return;
    }
}

// There can be only 1 batchAssemblerThread, it is designed expecting that there is just the one.
void BatchAssembler::batchAssemblerThread() {
    uint64_t exampleSizeInBytes = shards[0]->getExampleSizeInBytes();

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
            queueOpen = batchDataQueue.getBufferToLoad(batchDataBuffer);
            if (!queueOpen)
                return;
            queueOpen = batchLabelQueue.getBufferToLoad(batchLabelsBuffer);
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
        assert(chosenShard < numExamplesPerShard.size());

        queueOpen = shardQueues[chosenShard]->pop(labeledExample);
        if (!queueOpen) {
            batchDataQueue.bufferLoaded(batchDataBuffer);
            batchLabelQueue.bufferLoaded(batchLabelsBuffer);
            return;
        }

        // Load data to pinned memory buffer
        batchSlotOffset = exampleSizeInBytes * batchSlot;
        memcpy((uint8_t *)batchDataBuffer.getMemPtr() + batchSlotOffset, labeledExample.data.data(), exampleSizeInBytes);

        // Load one-hot labels to pinned memory buffer
        batchSlotOffset = classIndexes.size() * batchSlot;
        float *batchLabels = (float *)batchLabelsBuffer.getMemPtr() + batchSlotOffset;
        memset(batchLabels, 0, sizeof(float) * classIndexes.size());
        batchLabels[classIndexes[labeledExample.label]] = 1.0f;

        currentExampleNum += 1;

        batchSlot += 1;
        if (batchSlot == batchSize) {
            batchSlot = 0;

            batchDataQueue.bufferLoaded(batchDataBuffer);
            batchLabelQueue.bufferLoaded(batchLabelsBuffer);

            batchNumQueue.push(currentBatchNum);
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

void BatchAssembler::getBatch(Tensor &batchTensor, Tensor &labelTensor, uint64_t &batchNum) {
    bool queueOpen;
    queueOpen = batchDataQueue.getBufferToUnload(batchTensor);
    assert(queueOpen);
    queueOpen = batchLabelQueue.getBufferToUnload(labelTensor);
    assert(queueOpen);
    queueOpen = batchNumQueue.pop(batchNum);
    assert(queueOpen);
}

uint64_t BatchAssembler::getNextBatchNum() {
    uint64_t nextBatchNum;
    bool queueOpen = batchNumQueue.peek(nextBatchNum);
    assert(queueOpen);
    return nextBatchNum;
}

void BatchAssembler::returnBuffer(Tensor &batchTensor, Tensor &labelTensor) {
    bool queueOpen;
    queueOpen = batchDataQueue.bufferUnloaded(batchTensor);
    assert(queueOpen);
    queueOpen = batchLabelQueue.bufferUnloaded(labelTensor);
    assert(queueOpen);
}

uint64_t BatchAssembler::getNumBatchesPerEpoch() { return batchesPerEpoch; }
