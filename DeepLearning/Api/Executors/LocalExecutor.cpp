#include "DeepLearning/Api/Executors/LocalExecutor.h"

using std::condition_variable;
using std::map;
using std::string;
using std::thread;

using namespace Thor;

shared_ptr<LocalExecutor> LocalExecutor::Builder::build() {
    assert(_network.isPresent());
    assert(_loader);
    assert(_optimizer);
    // FIXME: add hyperparameter controller
    // assert(_hyperparameterController.isPresent());

    if (_visualizers.isEmpty()) {
        _visualizers = vector<Visualizer *>();
        _visualizers.get().push_back(&ConsoleVisualizer::instance());
    }

    shared_ptr<LocalExecutor> localExecutor = make_shared<LocalExecutor>();
    localExecutor->network = _network;
    localExecutor->loader = _loader;
    localExecutor->optimizer = _optimizer;
    localExecutor->optimizer->setNetwork(&(localExecutor->network));
    // localExecutor->hyperparameterController = _hyperparameterController;
    localExecutor->visualizers = _visualizers;

    for (uint64_t i = 0; i < localExecutor->visualizers.size(); ++i) {
        localExecutor->visualizerExecutionState.emplace_back(new AsyncQueue<ExecutionState>(32));
        localExecutor->visualizers[i]->startUI();
    }

    uint64_t batchSize = localExecutor->loader->getBatchSize();

    // Stamp the network
    // FIXME: stamp N networks per GPU, currently just stamping 1 network on gpu 0.
    // FIXME: save known optimal kernels on disk
    localExecutor->network.preOptimize(0, batchSize);
    localExecutor->stampedNetworks.emplace_back();
    Thor::Network::StatusCode statusCode = localExecutor->network.stampNetwork(0, batchSize, localExecutor->stampedNetworks.back());
    assert(statusCode == Thor::Network::StatusCode::SUCCESS);
    localExecutor->stampedNetworks.back().initialize();

    // FIXME: temp
    for (uint64_t i = 0; i < localExecutor->stampedNetworks.size(); ++i) {
        for (uint64_t t = 0; t < localExecutor->stampedNetworks[i].trainableLayers.size(); ++t)
            localExecutor->stampedNetworks[i].trainableLayers[t]->setLearningRate(0.001);
    }

    localExecutor->batchDataReady = make_shared<map<uint64_t, bool>>();
    localExecutor->batchData = make_shared<unordered_map<uint64_t, unordered_map<string, vector<uint8_t>>>>();
    localExecutor->batchFinished = make_shared<condition_variable>();

    localExecutor->epochMutex = make_shared<mutex>();
    localExecutor->currentEpoch = make_shared<uint64_t>(0);
    localExecutor->numBatchesDoneInEpoch = make_shared<uint64_t>(0);

    localExecutor->initialized = true;

    return localExecutor;
}

void CUDART_CB LocalExecutor::bufferStampTensors(void *data) {
    BufferStampTensorsParams *params = (BufferStampTensorsParams *)data;

    unordered_map<string, vector<uint8_t>> bufferMap;
    for (auto it = params->tensorsToReturn.begin(); it != params->tensorsToReturn.end(); ++it) {
        string tensorName = *it;
        ThorImplementation::Tensor copyFromTensor;
        if (params->batchletInput.count(tensorName) == 1) {
            assert(params->batchletOutput.count(tensorName) == 0);
            copyFromTensor = params->batchletInput[tensorName];
        } else {
            assert(params->batchletOutput.count(tensorName) == 1);
            copyFromTensor = params->batchletOutput[tensorName];
        }
        assert(copyFromTensor.getPlacement() == TensorPlacement::MemDevices::CPU);
        uint64_t numTensorBytes = copyFromTensor.getDescriptor().getArraySizeInBytes();
        bufferMap[tensorName] = vector<uint8_t>(numTensorBytes);
        memcpy(&(bufferMap[tensorName][0]), copyFromTensor.getMemPtr(), numTensorBytes);
    }

    params->batchMutex->lock();
    params->batchletData->push_back(bufferMap);

    // When all batchlets are available for a batch, then concatenate the data into the batch data array
    if (params->batchletData->size() == params->numBatchletsInBatch) {
        params->epochMutex->lock();
        for (auto it = bufferMap.begin(); it != bufferMap.end(); ++it) {
            string tensorName = it->first;
            uint64_t bytesPerBatchlet = it->second.size();
            (*(params->batchData))[params->epochBatchNum][tensorName] = vector<uint8_t>(bytesPerBatchlet * params->numBatchletsInBatch);
        }

        // batchletData:
        // batchNumber ->   batchlet0                              batchlet1
        //               [[input0 -> buffer, output0 -> buffer], [input0 -> buffer, output0 -> buffer]]
        for (uint64_t batchletIndex = 0; batchletIndex < params->numBatchletsInBatch; ++batchletIndex) {
            for (auto it = bufferMap.begin(); it != bufferMap.end(); ++it) {
                string tensorName = it->first;
                uint64_t bytesPerBatchlet = it->second.size();
                unordered_map<string, vector<uint8_t>> &currentIterationBatchData = (*(params->batchletData))[batchletIndex];

                memcpy(&((*(params->batchData))[params->epochBatchNum][tensorName][bytesPerBatchlet * batchletIndex]),
                       &(currentIterationBatchData[tensorName][0]),
                       bytesPerBatchlet);
            }
        }

        (*(params->batchDataReady))[params->epochBatchNum] = true;
        (*(params->numBatchesDoneInEpoch)) += 1;
        params->batchFinished->notify_all();
        params->epochMutex->unlock();

        params->loader->returnBatchBuffers(params->exampleType, params->batchletInput);
        params->batchMutex->unlock();
        delete params;
    } else {
        params->loader->returnBatchBuffers(params->exampleType, params->batchletInput);
        params->batchMutex->unlock();
    }
}

void LocalExecutor::trainBatches(
    uint64_t initialEpochBatchNum, uint64_t batches, uint64_t batchesPerEpoch, ExampleType exampleType, set<string> tensorsToReturn) {
    assert(batches > 0);
    assert(initialEpochBatchNum + batches <= batchesPerEpoch);

    // FIXME: this should be based on first expected to be done. Also there should be a GPU side input and output queue.
    uint64_t nextStampToProcess = 0;

    vector<map<string, Event>> outputReadyEvents(stampedNetworks.size());
    map<uint64_t, Event> processingFinishedEvents;
    cudaError_t cudaStatus;

    // FIXME:
    uint64_t batchletsPerBatch = 1;

    shared_ptr<mutex> batchMutex = make_shared<mutex>();

    *numBatchesDoneInEpoch = initialEpochBatchNum;

    // Scheduling in the following loop schedules far enough ahead untill all input batch buffers are exhausted.
    // Once that happens scheduling does not get any further ahead.
    for (uint64_t batch = 0; batch < batches; ++batch) {
        uint64_t epochBatchNum = initialEpochBatchNum + batch;

        optimizer->updateParameters(*currentEpoch, epochBatchNum, batchesPerEpoch);

        // batchNumber ->   batchlet0                              batchlet1
        //               [[input0 -> buffer, output0 -> buffer], [input0 -> buffer, output0 -> buffer]]
        shared_ptr<vector<unordered_map<string, vector<uint8_t>>>> batchletData =
            make_shared<vector<unordered_map<string, vector<uint8_t>>>>();

        {
            unique_lock<mutex> lck(*epochMutex);
            (*(batchDataReady))[epochBatchNum] = false;
            while (batchDataReady->size() > 32) {
                batchDataPopped.wait(lck);
            }
        }

        for (uint64_t batchlet = 0; batchlet < batchletsPerBatch; ++batchlet) {
            BufferStampTensorsParams *bufferStampTensorsParams = new BufferStampTensorsParams();
            bufferStampTensorsParams->batchletData = batchletData;
            bufferStampTensorsParams->batchData = batchData;
            bufferStampTensorsParams->batchMutex = batchMutex;
            bufferStampTensorsParams->epochMutex = epochMutex;
            bufferStampTensorsParams->numBatchesDoneInEpoch = numBatchesDoneInEpoch;
            bufferStampTensorsParams->batchFinished = batchFinished;
            bufferStampTensorsParams->batchDataReady = batchDataReady;
            bufferStampTensorsParams->loader = loader;
            bufferStampTensorsParams->exampleType = exampleType;
            bufferStampTensorsParams->epochBatchNum = epochBatchNum;
            bufferStampTensorsParams->numBatchletsInBatch = batchletsPerBatch;
            bufferStampTensorsParams->tensorsToReturn = tensorsToReturn;

            // Note: blocks when batch input buffers are exhausted (all of them loaned out)
            bufferStampTensorsParams->batchletInput = loader->getBatch(exampleType, epochBatchNum);

            // Note that all work is done for a stamp at the end of any input stream belonging to the stamp
            assert(!stampedNetworks[nextStampToProcess].inputs.empty());
            Stream stream = stampedNetworks[nextStampToProcess].inputs[0]->getStream();

            // Execute the stamp, noting the time taken using events.
            Event startBatchletEvent = stream.putEvent(true, false);
            stampedNetworks[nextStampToProcess].sendBatch(
                bufferStampTensorsParams->batchletInput, bufferStampTensorsParams->batchletOutput, outputReadyEvents[nextStampToProcess]);
            Event doneBatchletEvent = stream.putEvent(true, false);
            batchletTimingEvents[nextStampToProcess].emplace_back(startBatchletEvent, doneBatchletEvent);

            // Copy all data to buffers at the end of the work stream
            cudaStatus = cudaLaunchHostFunc(stream, bufferStampTensors, bufferStampTensorsParams);
            assert(cudaStatus == cudaSuccess);
            processingFinishedEvents[nextStampToProcess] = stream.putEvent();

            nextStampToProcess += 1;
            nextStampToProcess %= stampedNetworks.size();
        }

        // FIXME: Reduce gradients and broadcast updated weights

        // Now the input stream from each stamp waits till all gradient update work is done for the stamp, before it will begin processing
        // the next set of inputs passed to it.
        // FIXME: wait for weight update to happen
    }

    /*
    // FIXME: I should push each batch data to an async queue or something like that
    for (auto it = processingFinishedEvents.begin(); it != processingFinishedEvents.end(); ++it) {
        Event processingFinishedEvent = it->second;
        processingFinishedEvent.synchronize();  // FIXME: temp
    }
    */
}

void LocalExecutor::trainEpoch(ExampleType exampleType, set<string> tensorsToReturn) {
    uint64_t nextBatchNum = loader->getNextBatchNum(exampleType);
    uint64_t batchesPerEpoch = loader->getNumBatchesPerEpoch(exampleType);
    uint64_t batchesToTrain = loader->getNumBatchesPerEpoch(exampleType) - nextBatchNum;

    thread trainingThread(&LocalExecutor::trainBatches, this, nextBatchNum, batchesToTrain, batchesPerEpoch, exampleType, tensorsToReturn);
    unordered_map<string, std::vector<uint8_t>> batchData;
    while (true) {
        waitForBatchData();
        if (isBatchDataReady()) {
            batchData = popBatchData();
            // FIXME: fill in executionState
            ExecutionState executionState;
            executionState.batchesPerEpoch = batchesPerEpoch;
            for (uint32_t i = 0; i < visualizers.size(); ++i) {
                visualizers[i]->updateState(executionState);
            }
        } else {
            break;
        }
    }
    trainingThread.join();

    (*currentEpoch) += 1;
}

bool LocalExecutor::isBatchDataReady() {
    unique_lock<mutex> lck(*epochMutex);
    return isBatchDataReadyUnlocked();
}

bool LocalExecutor::isBatchDataReadyUnlocked() {
    if (batchDataReady->empty()) {
        return false;
    } else {
        auto it = batchDataReady->begin();
        return it->second;
    }
}

// Blocking
// Waits until a batch is finished
// --or--
// there is no batch finished and there are no more batches being processed
void LocalExecutor::waitForBatchData() {
    unique_lock<mutex> lck(*epochMutex);
    waitForBatchDataUnlocked(lck);
}

void LocalExecutor::waitForBatchDataUnlocked(unique_lock<mutex> &lck) {
    uint64_t numBatchesInEpoch = loader->getNumBatchesPerEpoch(ExampleType::TRAIN);  // FIXME exampleType
    // uint64_t numBatchesInEpoch = 5;  // FIXME temp
    bool allBatchesDoneForEpoch = (*numBatchesDoneInEpoch == numBatchesInEpoch);
    while (!allBatchesDoneForEpoch && !isBatchDataReadyUnlocked()) {
        batchFinished->wait(lck);
    }
}

// Blocking
// Returns a map of tensorName -> vector<uint8_t>, which holds the raw tensor value.
// If there is no more data that is being processed, returns an empty map.
unordered_map<string, std::vector<uint8_t>> LocalExecutor::popBatchData() {
    unique_lock<mutex> lck(*epochMutex);

    waitForBatchDataUnlocked(lck);

    if (!isBatchDataReadyUnlocked())
        return unordered_map<string, std::vector<uint8_t>>();

    auto it = batchData->begin();
    unordered_map<string, std::vector<uint8_t>> currentBatchData = it->second;
    batchData->erase(it);
    batchDataReady->erase(batchDataReady->begin());

    batchDataPopped.notify_all();

    return currentBatchData;
}
