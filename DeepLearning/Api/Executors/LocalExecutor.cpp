#include "DeepLearning/Api/Executors/LocalExecutor.h"

using std::condition_variable;
using std::map;
using std::string;
using std::thread;

using namespace Thor;
using namespace std;

#define LOCAL_EXECUTOR_PROFILE false
#define LOCAL_EXECUTOR_PROFILE_NUM_BATCHES 10

shared_ptr<LocalExecutor> LocalExecutor::Builder::build() {
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
    // localExecutor->hyperparameterController = _hyperparameterController;
    localExecutor->visualizers = _visualizers;

    if (_outputDirectory.isEmpty()) {
        boost::filesystem::path outputPath = boost::filesystem::absolute(boost::filesystem::path("./")).string();
        localExecutor->outputDirectory = boost::filesystem::canonical(outputPath).string();
    } else {
        localExecutor->outputDirectory = _outputDirectory;
    }

    for (uint64_t i = 0; i < localExecutor->visualizers.size(); ++i) {
        localExecutor->visualizerExecutionState.push_back(std::make_shared<AsyncQueue<ExecutionState>>(1024));
        localExecutor->visualizerExecutionState.back()->open();
        localExecutor->visualizers[i]->connectStateUpdateQueue(localExecutor->visualizerExecutionState.back());
        localExecutor->visualizers[i]->startUI();
    }

    uint64_t batchSize = localExecutor->loader->getBatchSize();

    // Stamp the network
    // FIXME: stamp N networks per GPU, currently just stamping 1 network on gpu 0.
    // FIXME: save known optimal kernels on disk
    Network::StatusCode statusCode;
    statusCode = localExecutor->network->place(batchSize);
    assert(statusCode == Network::StatusCode::SUCCESS);
    assert(!localExecutor->network->getStampedNetworks().empty());
    localExecutor->stampedNetworks = localExecutor->network->getStampedNetworks();
    for (uint32_t i = 0; i < localExecutor->stampedNetworks.size(); ++i) {
        localExecutor->stampedNetworks.back().initialize();
    }

    localExecutor->batchDataReady = make_shared<map<uint64_t, bool>>();
    localExecutor->batchData = make_shared<unordered_map<uint64_t, unordered_map<string, vector<uint8_t>>>>();
    localExecutor->batchFinished = make_shared<condition_variable>();

    localExecutor->epochMutex = make_shared<mutex>();
    localExecutor->currentEpoch = make_shared<uint64_t>(0);
    localExecutor->numBatchesDoneInEpoch = make_shared<uint64_t>(0);
    localExecutor->numBatchesInEpoch = make_shared<uint64_t>(0);

    localExecutor->initialized = true;

    return localExecutor;
}

LocalExecutor::~LocalExecutor() {
    for (uint64_t i = 0; i < visualizerExecutionState.size(); ++i) {
        visualizerExecutionState[i]->close();
    }
    visualizerExecutionState.clear();
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
        assert(copyFromTensor.getPlacement() == ThorImplementation::TensorPlacement::MemDevices::CPU);
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

void LocalExecutor::trainBatches(uint64_t initialEpochBatchNum, uint64_t batches, ExampleType exampleType, set<string> tensorsToReturn) {
    assert(batches > 0);
    assert(initialEpochBatchNum + batches <= *numBatchesInEpoch);

    bool validationPass = (exampleType != ExampleType::TRAIN);

    // FIXME: this should be based on first expected to be done. Also there should be a GPU side input and output queue.
    uint64_t nextStampToProcess = 0;

    vector<map<string, Event>> outputReadyEvents(stampedNetworks.size());
    map<uint64_t, Event> processingFinishedEvents;
    cudaError_t cudaStatus;

    // FIXME:
    uint64_t batchletsPerBatch = 1;

    shared_ptr<mutex> batchMutex = make_shared<mutex>();

    // Scheduling in the following loop schedules far enough ahead untill all input batch buffers are exhausted.
    // Once that happens scheduling does not get any further ahead.
#if LOCAL_EXECUTOR_PROFILE
    cudaStatus = cudaProfilerStart();
    assert(cudaStatus == cudaSuccess);
#endif
    for (uint64_t batch = 0; batch < batches; ++batch) {
#if LOCAL_EXECUTOR_PROFILE
        if (batch == LOCAL_EXECUTOR_PROFILE_NUM_BATCHES) {
            cudaStatus = cudaProfilerStop();
            assert(cudaStatus == cudaSuccess);
            exit(0);
        }
#endif

        uint64_t epochBatchNum = initialEpochBatchNum + batch;

        optimizer->updateHyperParameters(*currentEpoch, epochBatchNum, *numBatchesInEpoch);

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
            Event doneBatchletEvent = stampedNetworks[nextStampToProcess].sendBatch(bufferStampTensorsParams->batchletInput,
                                                                                    bufferStampTensorsParams->batchletOutput,
                                                                                    outputReadyEvents[nextStampToProcess],
                                                                                    validationPass);
            // FIXME: use to distribute batchlets:
            // batchletTimingEvents[nextStampToProcess].emplace_back(startBatchletEvent, doneBatchletEvent);

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

void LocalExecutor::trainEpochs(uint32_t numEpochs, set<string> tensorsToReturn) {
    double averageTrainingBatchTime = -1;
    double averageValidationBatchTime = -1;

    for (uint32_t i = 0; i < numEpochs; ++i) {
        // Training phase
        uint64_t batchNum = loader->getNextBatchNum(ExampleType::TRAIN);
        uint64_t batchesPerEpoch = loader->getNumBatchesPerEpoch(ExampleType::TRAIN);
        uint64_t batchesToTrain = loader->getNumBatchesPerEpoch(ExampleType::TRAIN) - batchNum;
        uint64_t batchSize = loader->getBatchSize();

        *numBatchesDoneInEpoch = batchNum;
        *numBatchesInEpoch = batchesPerEpoch;

        thread trainingThread(&LocalExecutor::trainBatches, this, batchNum, batchesToTrain, ExampleType::TRAIN, tensorsToReturn);
        unordered_map<string, std::vector<uint8_t>> batchData;

        ExecutionState executionState;
        executionState.outputDirectory = outputDirectory;
        executionState.epochsToTrain = numEpochs;
        executionState.networkName = network->getNetworkName();
        executionState.datasetName = loader->getDatasetName();
        executionState.executionMode = ExampleType::TRAIN;
        executionState.epochNum = *currentEpoch;
        executionState.batchSize = batchSize;
        executionState.batchesPerEpoch = batchesPerEpoch;
        executionState.numTrainingExamples = loader->getNumExamples(ExampleType::TRAIN);
        executionState.numValidationExamples = loader->getNumExamples(ExampleType::VALIDATE);
        executionState.numTestExamples = loader->getNumExamples(ExampleType::TEST);
        executionState.batchesPerEpoch = batchesPerEpoch;
        executionState.flopsPerExample =
            stampedNetworks[0].floatingPointOperationsPerExampleForward + stampedNetworks[0].floatingPointOperationsPerExampleBackward;

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        while (true) {
            waitForBatchData();
            if (isBatchDataReady()) {
                std::chrono::high_resolution_clock::time_point done = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(done - start);
                start = done;
                if (averageTrainingBatchTime < 0.0)
                    averageTrainingBatchTime = elapsed.count();
                else
                    averageTrainingBatchTime = 0.05 * elapsed.count() + 0.95 * averageTrainingBatchTime;

                unordered_map<string, float> optimizerParameters =
                    optimizer->getAllHyperParameters(*currentEpoch, batchNum, batchesPerEpoch);
                executionState.learningRate = optimizerParameters["currentLearningRate"];
                executionState.momentum = optimizerParameters["momentum"];

                batchData = popBatchData();
                executionState.batchNum = batchNum + 1;
                executionState.runningAverageTimePerTrainingBatch = averageTrainingBatchTime;
                float *batchLoss = (float *)(batchData["loss"].data());
                executionState.batchLoss = *batchLoss;
                float *batchAccuracy = (float *)(batchData["accuracy"].data());
                executionState.batchAccuracy = *batchAccuracy;
                for (uint32_t i = 0; i < visualizers.size(); ++i) {
                    visualizerExecutionState[i]->push(executionState);
                }
                batchNum += 1;
            } else {
                break;
            }
        }
        trainingThread.join();

        // Validation phase
        batchNum = loader->getNextBatchNum(ExampleType::VALIDATE);
        batchesPerEpoch = loader->getNumBatchesPerEpoch(ExampleType::VALIDATE);
        uint64_t batchesToValidate = loader->getNumBatchesPerEpoch(ExampleType::VALIDATE) - batchNum;
        batchSize = loader->getBatchSize();

        *numBatchesDoneInEpoch = batchNum;
        *numBatchesInEpoch = batchesPerEpoch;

        // FIXME: I am currently training using the validation data. A validation step needs to be built.
        thread validationThread(&LocalExecutor::trainBatches, this, batchNum, batchesToValidate, ExampleType::VALIDATE, tensorsToReturn);

        executionState.outputDirectory = outputDirectory;
        executionState.epochsToTrain = numEpochs;
        executionState.networkName = network->getNetworkName();
        executionState.datasetName = loader->getDatasetName();
        executionState.executionMode = ExampleType::VALIDATE;
        executionState.epochNum = *currentEpoch;
        executionState.batchSize = batchSize;
        executionState.batchesPerEpoch = batchesPerEpoch;
        executionState.numTrainingExamples = loader->getNumExamples(ExampleType::TRAIN);
        executionState.numValidationExamples = loader->getNumExamples(ExampleType::VALIDATE);
        executionState.numTestExamples = loader->getNumExamples(ExampleType::TEST);
        executionState.batchesPerEpoch = batchesPerEpoch;
        executionState.flopsPerExample = stampedNetworks[0].floatingPointOperationsPerExampleForward;

        start = std::chrono::high_resolution_clock::now();

        while (true) {
            waitForBatchData();
            if (isBatchDataReady()) {
                std::chrono::high_resolution_clock::time_point done = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(done - start);
                start = done;
                if (averageValidationBatchTime < 0.0)
                    averageValidationBatchTime = elapsed.count();
                else
                    averageValidationBatchTime = 0.05 * elapsed.count() + 0.95 * averageValidationBatchTime;

                unordered_map<string, float> optimizerParameters =
                    optimizer->getAllHyperParameters(*currentEpoch, batchNum, batchesPerEpoch);
                executionState.learningRate = optimizerParameters["currentLearningRate"];
                executionState.momentum = optimizerParameters["momentum"];

                batchData = popBatchData();
                executionState.batchNum = batchNum + 1;
                executionState.runningAverageTimePerValidationBatch = averageValidationBatchTime;
                float *batchLoss = (float *)(batchData["loss"].data());
                executionState.batchLoss = *batchLoss;
                float *batchAccuracy = (float *)(batchData["accuracy"].data());
                executionState.batchAccuracy = *batchAccuracy;
                for (uint32_t i = 0; i < visualizers.size(); ++i) {
                    visualizerExecutionState[i]->push(executionState);
                }
                batchNum += 1;
            } else {
                break;
            }
        }
        validationThread.join();

        (*currentEpoch) += 1;
    }
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
    bool allBatchesDoneForEpoch = (*numBatchesDoneInEpoch == *numBatchesInEpoch);
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
