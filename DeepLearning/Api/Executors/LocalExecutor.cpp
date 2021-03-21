#include "DeepLearning/Api/Executors/LocalExecutor.h"

using std::map;
using std::string;

using namespace Thor;

shared_ptr<LocalExecutor> LocalExecutor::Builder::build() {
    assert(_network.isPresent());
    assert(_loader);
    // FIXME: add hyperparameter controller
    // assert(_hyperparameterController.isPresent());

    if (_visualizers.isEmpty()) {
        _visualizers = vector<Visualizer *>();
        _visualizers.get().push_back(&ConsoleVisualizer::instance());
    }

    shared_ptr<LocalExecutor> localExecutor = make_shared<LocalExecutor>();
    localExecutor->network = _network;
    localExecutor->loader = _loader;
    // localExecutor->hyperparameterController = _hyperparameterController;
    localExecutor->visualizers = _visualizers;

    localExecutor->hyperparameterControllerExecutionState.reset(new AsyncQueue<ExecutionState>(32));
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

    params->mtx->lock();

    // FIXME: what about collision by training more than one epoch at a time and not clearing the entry before writing the new one?

    vector<unordered_map<string, vector<uint8_t>>> &currentBatchesBatchletData = (*(params->batchletData))[params->epochBatchNum];
    currentBatchesBatchletData.push_back(bufferMap);

    // When all batchlets are available for a batch, then concatenate the data into the batch data array
    if (currentBatchesBatchletData.size() == params->numBatchletsInBatch) {
        unordered_map<string, vector<uint8_t>> &currentBatchData = (*params->batchData)[params->epochBatchNum];
        for (auto it = bufferMap.begin(); it != bufferMap.end(); ++it) {
            string tensorName = it->first;
            uint64_t bytesPerBatchlet = it->second.size();
            currentBatchData[tensorName] = vector<uint8_t>(bytesPerBatchlet * params->numBatchletsInBatch);
        }

        // batchletData:
        // batchNumber ->   batchlet0                              batchlet1
        //               [[input0 -> buffer, output0 -> buffer], [input0 -> buffer, output0 -> buffer]]
        for (uint64_t batchletIndex = 0; batchletIndex < params->numBatchletsInBatch; ++batchletIndex) {
            for (auto it = bufferMap.begin(); it != bufferMap.end(); ++it) {
                string tensorName = it->first;
                uint64_t bytesPerBatchlet = it->second.size();
                unordered_map<string, vector<uint8_t>> &currentIterationBatchData =
                    (*(params->batchletData))[params->epochBatchNum][batchletIndex];

                memcpy(&(currentBatchData[tensorName][bytesPerBatchlet * batchletIndex]),
                       &(currentIterationBatchData[tensorName][0]),
                       bytesPerBatchlet);
            }
        }

        params->batchletData->erase(params->epochBatchNum);
    }

    // FIXME: could push batchdata to an async queue here

    params->mtx->unlock();

    params->loader->returnBatchBuffers(params->exampleType, params->batchletInput);

    delete params;
}

uint64_t LocalExecutor::trainBatches(uint64_t batches, ExampleType exampleType, set<string> tensorsToReturn) {
    assert(batches > 0);

    // FIXME: this should be based on first expected to be done. Also there should be a GPU side input and output queue.
    uint64_t nextStampToProcess = 0;

    vector<map<string, Event>> outputReadyEvents(stampedNetworks.size());
    map<uint64_t, Event> processingFinishedEvents;
    cudaError_t cudaStatus;

    // FIXME:
    uint64_t batchletsPerBatch = 1;

    // batchNumber ->   batchlet0                              batchlet1
    //               [[input0 -> buffer, output0 -> buffer], [input0 -> buffer, output0 -> buffer]]
    shared_ptr<unordered_map<uint64_t, vector<unordered_map<string, vector<uint8_t>>>>> batchletData =
        make_shared<unordered_map<uint64_t, vector<unordered_map<string, vector<uint8_t>>>>>();
    // The concatenation of all batchlet datas for the batch:
    shared_ptr<unordered_map<uint64_t, unordered_map<string, vector<uint8_t>>>> batchData =
        make_shared<unordered_map<uint64_t, unordered_map<string, vector<uint8_t>>>>();
    shared_ptr<mutex> mtx = make_shared<mutex>();

    uint64_t epochBatchNum;
    for (uint64_t batch = 0; batch < batches; ++batch) {
        // FIXME: wait until there are 10 or less batches scheduled -> check the contents of batchletData, use mutex.

        for (uint64_t batchlet = 0; batchlet < batchletsPerBatch; ++batchlet) {
            BufferStampTensorsParams *bufferStampTensorsParams = new BufferStampTensorsParams();
            bufferStampTensorsParams->batchletData = batchletData;
            bufferStampTensorsParams->batchData = batchData;
            bufferStampTensorsParams->mtx = mtx;
            bufferStampTensorsParams->loader = loader;
            bufferStampTensorsParams->exampleType = exampleType;
            bufferStampTensorsParams->epochBatchNum = epochBatchNum;
            bufferStampTensorsParams->numBatchletsInBatch = batchletsPerBatch;
            bufferStampTensorsParams->tensorsToReturn = tensorsToReturn;

            bufferStampTensorsParams->batchletInput = loader->getBatch(exampleType, epochBatchNum);

            // Note that all work is done for a stamp at the end of any input stream belonging to the stamp
            assert(!stampedNetworks[nextStampToProcess].inputs.empty());
            Stream stream = stampedNetworks[nextStampToProcess].inputs[0]->getStream();

            // Execute the stamp, noting the time taken using events.
            Event startBatchletEvent = stream.putEvent(true, false);
            processingFinishedEvents[nextStampToProcess] = stampedNetworks[nextStampToProcess].sendBatch(
                bufferStampTensorsParams->batchletInput, bufferStampTensorsParams->batchletOutput, outputReadyEvents[nextStampToProcess]);
            Event doneBatchletEvent = stream.putEvent(true, false);
            batchletTimingEvents[nextStampToProcess].emplace_back(startBatchletEvent, doneBatchletEvent);

            // Copy all data to buffers at the end of the work stream
            cudaStatus = cudaLaunchHostFunc(stream, bufferStampTensors, bufferStampTensorsParams);
            assert(cudaStatus == cudaSuccess);

            nextStampToProcess += 1;
            nextStampToProcess %= stampedNetworks.size();
        }

        // Now the input stream from each stamp waits till all gradient update work is done for the stamp, before it will begin processing
        // the next set of inputs passed to it.
    }

    // FIXME: I should push each batch data to an async queue or something like that
    for (auto it = processingFinishedEvents.begin(); it != processingFinishedEvents.end(); ++it) {
        Event processingFinishedEvent = it->second;
        processingFinishedEvent.synchronize();
    }

    // FIXME need to fill batchData

    return epochBatchNum;
}

void LocalExecutor::trainTillEpochIsFinished(ExampleType exampleType, set<string> tensorsToReturn) {
    uint64_t nextBatchNum = loader->getNextBatchNum(exampleType);
    uint64_t batchesToTrain = loader->getNumBatchesPerEpoch(exampleType) - nextBatchNum;
    trainBatches(batchesToTrain, exampleType, tensorsToReturn);
}
