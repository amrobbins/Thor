#include "Utilities/Loaders/BatchAssembler.h"

using std::thread;
using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;

BatchAssembler::BatchAssembler(vector<Shard *> shards, ExampleType exampleType, TensorDescriptor exampleDescriptor, uint64_t batchSize) {
    assert(!shards.empty());
    assert(batchSize > 0);

    this->shards = shards;
    this->exampleType = exampleType;
    this->batchSize = batchSize;
    this->exampleDescriptor = exampleDescriptor;
    vector<uint64_t> batchDimensions;
    batchDimensions.push_back(batchSize);
    batchDimensions.insert(batchDimensions.end(), exampleDescriptor.getDimensions().begin(), exampleDescriptor.getDimensions().end());
    batchDataTensorDescriptor = TensorDescriptor(exampleDescriptor.getDataType(), batchDimensions);
    this->exampleDescriptor = exampleDescriptor;

    for (uint64_t i = 0; i < shards.size(); ++i) {
        assert(shards[i]->isOpen());
        assert(shards[i]->getExampleSizeInBytes() == exampleDescriptor.getArraySizeInBytes());

        numExamples += shards[i]->getNumExamples(exampleType);
        numExamplesPerShard.push_back(shards[i]->getNumExamples(exampleType));
        randomizers.emplace_back(numExamplesPerShard[i]);

        set<string> classesInShard = shards[i]->getAllClassesInShard();
        for (auto it = classesInShard.begin(); it != classesInShard.end(); ++it) {
            string className = *it;
            if (classIndexes.count(className) == 0) {
                uint64_t index = classIndexes.size();
                classIndexes[className] = index;
            }
        }
    }
    batchLabelTensorDescriptor = ThorImplementation::TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, classIndexes.size()});

    batchesPerEpoch = (numExamples + (batchSize - 1)) / batchSize;

    open();
}

BatchAssembler::~BatchAssembler() { close(); }

void BatchAssembler::open() {
    assert(!batchDataQueue.isOpen());
    assert(!batchLabelQueue.isOpen());
    assert(shardQueues.empty());

    for (uint64_t i = 0; i < shards.size(); ++i) {
        shardQueues.emplace_back(32);
        shardQueues.back().open();
        randomizers[i].reseed();

        shardThreads.emplace_back(&BatchAssembler::shardReaderThread, this, i);
    }
    currentBatchNum = 0;

    batchDataQueue.resize(32, batchDataTensorDescriptor, TensorPlacement::MemDevices::CPU);
    batchDataQueue.open();
    batchLabelQueue.resize(32, batchLabelTensorDescriptor, TensorPlacement::MemDevices::CPU);
    batchLabelQueue.open();

    assemblerThread = thread(&BatchAssembler::batchAssemblerThread, this);
}

void BatchAssembler::close() {
    for (uint64_t i = 0; i < shards.size(); ++i)
        shardQueues[i].close();
    batchDataQueue.close();
    batchLabelQueue.close();

    for (uint64_t i = 0; i < shards.size(); ++i)
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
            labeledExample.data.data(), labeledExample.label, labeledExample.filename, exampleType, randomizers[shard].getRandomNumber());

        queueOpen = shardQueues[shard].push(labeledExample);
        if (!queueOpen)
            return;
    }
}

void BatchAssembler::batchAssemblerThread() {
    uint64_t exampleSizeInBytes = shards[0]->getExampleSizeInBytes();

    uint64_t numExamplesInEpoch = 0;
    for (uint64_t i = 0; i < numExamplesPerShard.size(); ++i)
        numExamplesInEpoch += numExamplesPerShard[i];

    uint64_t numExamplesLeftInEpoch = 0;
    vector<uint64_t> numExamplesLeftPerShard = numExamplesPerShard;

    bool queueOpen;
    LabeledExample labeledExample;
    uint64_t batchSlot = 0;

    while (1) {
        if (numExamplesLeftInEpoch == 0) {
            numExamplesLeftInEpoch = numExamplesInEpoch;
            for (uint64_t i = 0; i < numExamplesPerShard.size(); ++i) {
                numExamplesLeftPerShard[i] = numExamplesPerShard[i];
            }
        }

        uint64_t rand0 = rand() & 0xFFFF;
        uint64_t rand1 = rand() & 0xFFFF;
        uint64_t rand2 = rand() & 0xFFFF;
        uint64_t rand3 = rand() & 0xFFFF;
        uint64_t randomNumber = (rand0 << 48) | (rand1 << 32) | (rand2 << 16) | rand3;
        randomNumber %= numExamplesLeftInEpoch;

        uint64_t priorExamples = 0;
        uint64_t chosenShard;
        for (uint64_t i = 0; i < numExamplesPerShard.size(); ++i) {
            priorExamples += numExamplesLeftPerShard[i];
            if (priorExamples > randomNumber) {
                numExamplesLeftPerShard[i] -= 1;
                numExamplesLeftInEpoch -= 1;
                chosenShard = i;
                break;
            }
        }

        queueOpen = shardQueues[chosenShard].pop(labeledExample);
        if (!queueOpen)
            return;

        Tensor batchDataBuffer;
        uint64_t batchSlotOffset;
        queueOpen = batchDataQueue.getBufferToLoad(batchDataBuffer);
        if (!queueOpen)
            return;
        batchSlotOffset = exampleSizeInBytes * batchSlot;
        memcpy((uint8_t *)batchDataBuffer.getMemPtr() + batchSlotOffset, labeledExample.data.data(), exampleSizeInBytes);

        Tensor batchLabelsBuffer;
        queueOpen = batchLabelQueue.getBufferToLoad(batchLabelsBuffer);
        if (!queueOpen)
            return;
        batchSlotOffset = classIndexes.size() * batchSlot;
        float *batchLabels = (float *)batchLabelsBuffer.getMemPtr() + batchSlotOffset;
        memset(batchLabels, 0, sizeof(float) * classIndexes.size());
        batchLabels[classIndexes[labeledExample.label]] = 1.0f;

        batchSlot += 1;
        if (batchSlot == batchSize) {
            batchSlot = 0;

            queueOpen = batchDataQueue.bufferLoaded(batchDataBuffer);
            if (!queueOpen)
                return;

            queueOpen = batchLabelQueue.bufferLoaded(batchLabelsBuffer);
            if (!queueOpen)
                return;

            batchNumQueue.push_back(currentBatchNum);
            currentBatchNum += 1;
            if (currentBatchNum == batchesPerEpoch)
                currentBatchNum = 0;
        }
    }
}

void BatchAssembler::getBatch(Tensor &batchTensor, Tensor &labelTensor, uint64_t &batchNum, uint64_t &numBatchesInEpoch) {
    bool queueOpen;
    queueOpen = batchDataQueue.getBufferToUnload(batchTensor);
    assert(queueOpen);
    queueOpen = batchLabelQueue.getBufferToUnload(labelTensor);
    assert(queueOpen);
    batchNum = batchNumQueue.front() + 1;
    batchNumQueue.pop_front();
    numBatchesInEpoch = batchesPerEpoch;
}

void BatchAssembler::returnBuffer(Tensor &batchTensor, Tensor &labelTensor) {
    bool queueOpen;
    queueOpen = batchDataQueue.bufferUnloaded(batchTensor);
    assert(queueOpen);
    queueOpen = batchLabelQueue.bufferUnloaded(labelTensor);
    assert(queueOpen);
}

uint64_t BatchAssembler::getNumBatchesPerEpoch() { return batchesPerEpoch; }
