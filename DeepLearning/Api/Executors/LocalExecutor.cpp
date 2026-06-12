#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Executors/LocalExecutor.h"

#include <chrono>
#include <cstring>
#include <optional>
#include <utility>

using std::condition_variable;
using std::map;
using std::string;
using std::thread;

using namespace Thor;
using namespace std;

#define LOCAL_EXECUTOR_PROFILE false
#define LOCAL_EXECUTOR_PROFILE_NUM_BATCHES 10

shared_ptr<LocalExecutor> LocalExecutor::Builder::build() {
    THOR_THROW_IF_FALSE(_loader);
    // FIXME: add hyperparameter controller
    // THOR_THROW_IF_FALSE(_hyperparameterController.has_value());

    if (!observers_.has_value()) {
        observers_ = vector<shared_ptr<TrainingObserver>>();
        if (statsEnabled_) {
            observers_.value().push_back(make_shared<LineStatsReporter>(statsIntervalSeconds_, true));
        } else {
            observers_.value().push_back(make_shared<NullTrainingObserver>());
        }
    }

    shared_ptr<LocalExecutor> localExecutor = make_shared<LocalExecutor>();
    localExecutor->loader = _loader;
    localExecutor->optimizer = _optimizer;
    // localExecutor->hyperparameterController = _hyperparameterController;
    localExecutor->observers = observers_.value();
    localExecutor->statsEnabled = statsEnabled_;
    localExecutor->statsIntervalSeconds = statsIntervalSeconds_;
    localExecutor->maxInFlightBatches = maxInFlightBatches_;
    localExecutor->synchronizeAfterEveryBatch = synchronizeAfterEveryBatch_;

    if (!_outputDirectory.has_value()) {
        std::filesystem::path outputPath = std::filesystem::absolute(std::filesystem::path("./")).string();
        localExecutor->outputDirectory = std::filesystem::canonical(outputPath).string();
    } else {
        localExecutor->outputDirectory = _outputDirectory.value();
    }

    uint64_t batchSize = localExecutor->loader->getBatchSize();

    if (_network->getDefaultOptimizer() == nullptr && !_network->allTrainingEnabledParametersHaveOptimizers()) {
        std::shared_ptr<Optimizer> fallbackOptimizer = _optimizer;
        if (fallbackOptimizer == nullptr && trainingProgram_.has_value() && trainingProgram_->getNumSteps() == 1) {
            fallbackOptimizer = trainingProgram_->getStep(0).getOptimizer();
        }
        if (fallbackOptimizer != nullptr) {
            _network->setDefaultOptimizer(fallbackOptimizer);
        }
    }

    // Place the network
    // FIXME: stamp N networks per GPU, currently just stamping 1 network on gpu 0.
    vector<Event> initDoneEvents;
    localExecutor->placedNetwork = _network->place(batchSize, initDoneEvents);

    THOR_THROW_IF_FALSE(localExecutor->placedNetwork->getNumStamps() >= 1);

    if (trainingProgram_.has_value()) {
        localExecutor->executableTrainingPlan = ExecutableTrainingPlan::compile(trainingProgram_.value(), *localExecutor->placedNetwork);
        localExecutor->executableTrainingPlan->assertLegacyLocalExecutorCompatible();
        std::shared_ptr<Optimizer> stepOptimizer = localExecutor->executableTrainingPlan->getStep(0).getOptimizer();
        if (stepOptimizer != nullptr) {
            localExecutor->optimizer = stepOptimizer;
        }
    }

    localExecutor->batchDataReady = make_shared<map<uint64_t, bool>>();
    localExecutor->batchData = make_shared<unordered_map<uint64_t, unordered_map<string, vector<uint8_t>>>>();
    localExecutor->batchFinished = make_shared<condition_variable>();

    localExecutor->epochMutex = make_shared<mutex>();
    localExecutor->currentEpoch = make_shared<uint64_t>(0);
    localExecutor->numBatchesDoneInEpoch = make_shared<uint64_t>(0);
    localExecutor->numBatchesInEpoch = make_shared<uint64_t>(0);

    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        // Host sync with init is done.
        initDoneEvents[i].synchronize();
    }
    localExecutor->initialized = true;

    return localExecutor;
}

LocalExecutor::~LocalExecutor() = default;

void LocalExecutor::emitTrainingEvent(const TrainingEvent &event) {
    if (!statsEnabled && event.type == TrainingEventType::STATS) {
        return;
    }
    for (const shared_ptr<TrainingObserver> &observer : observers) {
        if (observer) {
            observer->onTrainingEvent(event);
        }
    }
}

void CUDART_CB LocalExecutor::bufferStampTensors(void *data) {
    BufferStampTensorsParams *params = (BufferStampTensorsParams *)data;

    unordered_map<string, vector<uint8_t>> bufferMap;
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    for (auto it = params->tensorsToReturn.begin(); it != params->tensorsToReturn.end(); ++it) {
        string tensorName = *it;
        ThorImplementation::Tensor copyFromTensor;
        if (params->batchletInput.count(tensorName) == 1) {
            THOR_THROW_IF_FALSE(params->batchletOutput.count(tensorName) == 0);
            copyFromTensor = params->batchletInput[tensorName];
        } else {
            THOR_THROW_IF_FALSE(params->batchletOutput.count(tensorName) == 1);
            copyFromTensor = params->batchletOutput[tensorName];
        }
        THOR_THROW_IF_FALSE(copyFromTensor.getPlacement() == cpuPlacement);
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

        params->loader->returnBatchBuffers(params->exampleType, std::move(params->batchletInput));
        params->batchMutex->unlock();
        delete params;
    } else {
        params->loader->returnBatchBuffers(params->exampleType, std::move(params->batchletInput));
        params->batchMutex->unlock();
    }
}

void LocalExecutor::trainBatches(uint64_t initialEpochBatchNum, uint64_t batches, ExampleType exampleType, set<string> tensorsToReturn) {
    THOR_THROW_IF_FALSE(batches > 0);
    THOR_THROW_IF_FALSE(initialEpochBatchNum + batches <= *numBatchesInEpoch);

    bool validationPass = (exampleType != ExampleType::TRAIN);

    // FIXME: this should be based on first expected to be done. Also there should be a GPU side input and output queue.
    uint64_t nextStampToProcess = 0;

    vector<map<string, Event>> outputReadyEvents(placedNetwork->getNumStamps());
    map<uint64_t, Event> processingFinishedEvents;
    cudaError_t cudaStatus;

    // FIXME:
    uint64_t batchletsPerBatch = 1;

    shared_ptr<mutex> batchMutex = make_shared<mutex>();

    // Scheduling in the following loop schedules far enough ahead untill all input batch buffers are exhausted.
    // Once that happens scheduling does not get any further ahead.
#if LOCAL_EXECUTOR_PROFILE
    cudaStatus = cudaProfilerStart();
    THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
#endif
    for (uint64_t batch = 0; batch < batches; ++batch) {
#if LOCAL_EXECUTOR_PROFILE
        if (batch == LOCAL_EXECUTOR_PROFILE_NUM_BATCHES) {
            cudaStatus = cudaProfilerStop();
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            exit(0);
        }
#endif

        uint64_t epochBatchNum = initialEpochBatchNum + batch;

        Optimizer::updateHyperParameters(placedNetwork.get(), *currentEpoch, epochBatchNum, *numBatchesInEpoch);

        // batchNumber ->   batchlet0                              batchlet1
        //               [[input0 -> buffer, output0 -> buffer], [input0 -> buffer, output0 -> buffer]]
        shared_ptr<vector<unordered_map<string, vector<uint8_t>>>> batchletData =
            make_shared<vector<unordered_map<string, vector<uint8_t>>>>();

        {
            unique_lock<mutex> lck(*epochMutex);
            while (batchDataReady->size() >= maxInFlightBatches) {
                batchDataPopped.wait(lck);
            }
            (*(batchDataReady))[epochBatchNum] = false;
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
            ThorImplementation::StampedNetwork &stampedNetwork = placedNetwork->getStampedNetwork(nextStampToProcess);
            THOR_THROW_IF_FALSE(!stampedNetwork.inputs.empty());
            Stream stream = stampedNetwork.inputs[0]->getStream();

            // Execute the stamp, noting the time taken using events.
            Event startBatchletEvent = stream.putEvent(true, false);
            Event doneBatchletEvent = stampedNetwork.sendBatch(bufferStampTensorsParams->batchletInput,
                                                               bufferStampTensorsParams->batchletOutput,
                                                               outputReadyEvents[nextStampToProcess],
                                                               validationPass);
            // FIXME: TEMP FOR DEBUG. GET RID OF SYNCHRONIZE!!!
            // Stream::deviceSynchronize(0);

            // FIXME: use to distribute batchlets:
            // batchletTimingEvents[nextStampToProcess].emplace_back(startBatchletEvent, doneBatchletEvent);

            // Copy all data to buffers at the end of the work stream
            cudaStatus = cudaLaunchHostFunc(stream, bufferStampTensors, bufferStampTensorsParams);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            processingFinishedEvents[nextStampToProcess] = stream.putEvent();
            if (synchronizeAfterEveryBatch) {
                processingFinishedEvents[nextStampToProcess].synchronize();
            }

            nextStampToProcess += 1;
            nextStampToProcess %= placedNetwork->getNumStamps();
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

uint64_t LocalExecutor::getOutstandingBatchCount() const {
    unique_lock<mutex> lck(*epochMutex);
    return batchDataReady->size();
}

void LocalExecutor::trainEpochs(uint32_t numEpochs, set<string> tensorsToReturn) {
    THOR_THROW_IF_FALSE(numEpochs > 0);

    double averageTrainingBatchTime = -1;
    double averageValidationBatchTime = -1;
    ThorImplementation::StampedNetwork& statsStampedNetwork = placedNetwork->getStampedNetwork(0);
    const uint64_t batchSizeForFlops = loader->getBatchSize();
    const uint64_t forwardFlopsPerBatch = statsStampedNetwork.getFloatingPointOperationsPerExampleForward() * batchSizeForFlops;
    const uint64_t trainingFlopsPerBatch = statsStampedNetwork.getFloatingPointOperationsPerExampleTraining() * batchSizeForFlops;
    const auto runStart = std::chrono::high_resolution_clock::now();

    auto elapsedSinceRunStart = [&]() {
        const auto now = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - runStart);
        return elapsed.count();
    };

    auto optionalFloatFromBatch = [](const unordered_map<string, vector<uint8_t>>& batchData, const string& tensorName) -> optional<double> {
        auto it = batchData.find(tensorName);
        if (it == batchData.end() || it->second.size() < sizeof(float)) {
            return nullopt;
        }
        float value = 0.0f;
        memcpy(&value, it->second.data(), sizeof(float));
        return static_cast<double>(value);
    };

    auto makeBaseSnapshot = [&](TrainingPhase phase, uint64_t epoch, uint64_t batchSize, uint64_t batchesPerEpoch) {
        TrainingStatsSnapshot snapshot;
        snapshot.networkName = placedNetwork->getNetworkName();
        snapshot.datasetName = loader->getDatasetName();
        snapshot.phase = phase;
        snapshot.epoch = epoch;
        snapshot.epochs = numEpochs;
        snapshot.batchSize = batchSize;
        snapshot.stepsPerEpoch = batchesPerEpoch;
        snapshot.elapsedSeconds = elapsedSinceRunStart();
        snapshot.inFlightBatches = getOutstandingBatchCount();
        return snapshot;
    };

    emitTrainingEvent(TrainingEvent::runStarted(makeBaseSnapshot(TrainingPhase::UNKNOWN, 0, loader->getBatchSize(), 0)));

    for (uint32_t epochOffset = 0; epochOffset < numEpochs; ++epochOffset) {
        const uint64_t humanEpoch = *currentEpoch + 1;

        // Training phase
        uint64_t batchNum = loader->getNextBatchNum(ExampleType::TRAIN);
        uint64_t batchesPerEpoch = loader->getNumBatchesPerEpoch(ExampleType::TRAIN);
        uint64_t batchesToTrain = loader->getNumBatchesPerEpoch(ExampleType::TRAIN) - batchNum;
        uint64_t batchSize = loader->getBatchSize();

        *numBatchesDoneInEpoch = batchNum;
        *numBatchesInEpoch = batchesPerEpoch;

        emitTrainingEvent(TrainingEvent::epochStarted(makeBaseSnapshot(TrainingPhase::TRAIN, humanEpoch, batchSize, batchesPerEpoch)));

        thread trainingThread(&LocalExecutor::trainBatches, this, batchNum, batchesToTrain, ExampleType::TRAIN, tensorsToReturn);
        unordered_map<string, vector<uint8_t>> batchData;

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

                unordered_map<string, float> optimizerParameters;
                if (optimizer != nullptr) {
                    optimizerParameters = optimizer->getAllHyperParameters(placedNetwork.get());
                }

                batchData = popBatchData();

                TrainingStatsSnapshot snapshot = makeBaseSnapshot(TrainingPhase::TRAIN, humanEpoch, batchSize, batchesPerEpoch);
                snapshot.stepInEpoch = batchNum + 1;
                snapshot.step = (*currentEpoch * batchesPerEpoch) + snapshot.stepInEpoch;
                snapshot.samplesProcessed = snapshot.step * batchSize;
                if (averageTrainingBatchTime > 0.0) {
                    snapshot.samplesPerSecond = static_cast<double>(batchSize) / averageTrainingBatchTime;
                    snapshot.batchesPerSecond = 1.0 / averageTrainingBatchTime;
                    snapshot.floatingPointOperationsPerBatch = trainingFlopsPerBatch;
                    snapshot.floatingPointOperationsPerSecond = static_cast<double>(trainingFlopsPerBatch) / averageTrainingBatchTime;
                }
                if (optimizerParameters.count("currentLearningRate") > 0) {
                    snapshot.learningRate = optimizerParameters["currentLearningRate"];
                }
                if (optimizerParameters.count("momentum") > 0) {
                    snapshot.momentum = optimizerParameters["momentum"];
                }
                snapshot.loss = optionalFloatFromBatch(batchData, "loss");
                snapshot.accuracy = optionalFloatFromBatch(batchData, "accuracy");
                emitTrainingEvent(TrainingEvent::statsUpdated(std::move(snapshot)));

                batchNum += 1;
            } else {
                break;
            }
        }
        trainingThread.join();
        emitTrainingEvent(TrainingEvent::epochFinished(makeBaseSnapshot(TrainingPhase::TRAIN, humanEpoch, batchSize, batchesPerEpoch)));

        // Validation phase
        batchNum = loader->getNextBatchNum(ExampleType::VALIDATE);
        batchesPerEpoch = loader->getNumBatchesPerEpoch(ExampleType::VALIDATE);
        uint64_t batchesToValidate = loader->getNumBatchesPerEpoch(ExampleType::VALIDATE) - batchNum;
        batchSize = loader->getBatchSize();

        *numBatchesDoneInEpoch = batchNum;
        *numBatchesInEpoch = batchesPerEpoch;

        emitTrainingEvent(TrainingEvent::epochStarted(makeBaseSnapshot(TrainingPhase::VALIDATE, humanEpoch, batchSize, batchesPerEpoch)));

        // FIXME: I am currently training using the validation data. A validation step needs to be built.
        thread validationThread(&LocalExecutor::trainBatches, this, batchNum, batchesToValidate, ExampleType::VALIDATE, tensorsToReturn);

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

                unordered_map<string, float> optimizerParameters;
                if (optimizer != nullptr) {
                    optimizerParameters = optimizer->getAllHyperParameters(placedNetwork.get());
                }

                batchData = popBatchData();

                TrainingStatsSnapshot snapshot = makeBaseSnapshot(TrainingPhase::VALIDATE, humanEpoch, batchSize, batchesPerEpoch);
                snapshot.stepInEpoch = batchNum + 1;
                snapshot.step = (*currentEpoch * batchesPerEpoch) + snapshot.stepInEpoch;
                snapshot.samplesProcessed = snapshot.step * batchSize;
                if (averageValidationBatchTime > 0.0) {
                    snapshot.samplesPerSecond = static_cast<double>(batchSize) / averageValidationBatchTime;
                    snapshot.batchesPerSecond = 1.0 / averageValidationBatchTime;
                    snapshot.floatingPointOperationsPerBatch = forwardFlopsPerBatch;
                    snapshot.floatingPointOperationsPerSecond = static_cast<double>(forwardFlopsPerBatch) / averageValidationBatchTime;
                }
                if (optimizerParameters.count("currentLearningRate") > 0) {
                    snapshot.learningRate = optimizerParameters["currentLearningRate"];
                }
                if (optimizerParameters.count("momentum") > 0) {
                    snapshot.momentum = optimizerParameters["momentum"];
                }
                snapshot.loss = optionalFloatFromBatch(batchData, "loss");
                snapshot.accuracy = optionalFloatFromBatch(batchData, "accuracy");
                emitTrainingEvent(TrainingEvent::statsUpdated(std::move(snapshot)));

                batchNum += 1;
            } else {
                break;
            }
        }
        validationThread.join();
        emitTrainingEvent(TrainingEvent::epochFinished(makeBaseSnapshot(TrainingPhase::VALIDATE, humanEpoch, batchSize, batchesPerEpoch)));

        (*currentEpoch) += 1;
    }

    emitTrainingEvent(TrainingEvent::runFinished(makeBaseSnapshot(TrainingPhase::UNKNOWN, *currentEpoch, loader->getBatchSize(), 0)));
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
