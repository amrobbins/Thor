#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Training/ExecutableTrainingPlan.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <exception>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

namespace {

struct QueuedTrainingState;

struct NativeBatchCompletionParams {
    std::shared_ptr<QueuedTrainingState> state;
    std::shared_ptr<Loader> loader;
    ExampleType exampleType = ExampleType::TRAIN;
    uint64_t epochBatchNum = 0;
    std::set<std::string> tensorsToReturn;
    std::map<std::string, ThorImplementation::Tensor> batchInput;
    std::map<std::string, ThorImplementation::Tensor> batchOutput;
};

struct QueuedTrainingState {
    std::mutex mutex;
    std::condition_variable batchFinished;
    std::condition_variable batchPopped;

    std::map<uint64_t, bool> batchReady;
    std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<uint8_t>>> batchData;
    std::unordered_map<uint64_t, std::unique_ptr<NativeBatchCompletionParams>> completedBatchParams;
    std::exception_ptr failure;
    bool cancelRequested = false;

    uint64_t numBatchesDoneInEpoch = 0;
    uint64_t numBatchesInEpoch = 0;
};

std::optional<double> optionalFloatFromBatch(const std::unordered_map<std::string, std::vector<uint8_t>>& batchData,
                                             const std::string& tensorName) {
    auto it = batchData.find(tensorName);
    if (it == batchData.end() || it->second.size() < sizeof(float)) {
        return std::nullopt;
    }
    float value = 0.0f;
    std::memcpy(&value, it->second.data(), sizeof(float));
    return static_cast<double>(value);
}

std::string phaseName(TrainingPhase phase) {
    switch (phase) {
        case TrainingPhase::TRAIN:
            return "train";
        case TrainingPhase::VALIDATE:
            return "validate";
        case TrainingPhase::TEST:
            return "test";
        case TrainingPhase::UNKNOWN:
        default:
            return "unknown";
    }
}

void ensureNativeQueuedPlanCompatible(const ExecutableTrainingPlan& plan, const Network& network) {
    plan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences(/*trainingEnabledOnly=*/true));
}

TrainingProgram defaultTrainingProgramForRequest(const TrainingRunRequest& request) {
    if (request.trainingProgram.has_value()) {
        if (!request.trainingProgram->isInitialized()) {
            throw std::runtime_error("Trainer execution received an uninitialized TrainingProgram.");
        }
        return request.trainingProgram.value();
    }

    std::vector<Tensor> lossRoots = request.network->getLossRootTensors();
    if (lossRoots.empty()) {
        throw std::runtime_error("Trainer could not synthesize a default TrainingProgram because the Network has no loss roots.");
    }

    std::vector<ParameterReference> parameters = request.network->getTrainableParameterReferences(/*trainingEnabledOnly=*/true);
    if (parameters.empty()) {
        throw std::runtime_error("Trainer could not synthesize a default TrainingProgram because the Network has no trainable parameters.");
    }

    std::shared_ptr<Optimizer> optimizer = request.optimizer;
    if (optimizer == nullptr && request.network != nullptr) {
        optimizer = request.network->getDefaultOptimizer();
    }

    return TrainingProgram({TrainingStep("default", std::move(lossRoots), optimizer, std::move(parameters))});
}

std::shared_ptr<Optimizer> placementFallbackOptimizerForRequest(const TrainingRunRequest& request, const TrainingProgram& program) {
    if (request.optimizer != nullptr) {
        return request.optimizer;
    }
    if (program.getNumSteps() == 1) {
        return program.getStep(0).getOptimizer();
    }
    return nullptr;
}

void attachPlacementFallbackOptimizerIfNeeded(const TrainingRunRequest& request, const TrainingProgram& program) {
    if (request.network == nullptr || request.network->getDefaultOptimizer() != nullptr ||
        request.network->allTrainingEnabledParametersHaveOptimizers()) {
        return;
    }

    std::shared_ptr<Optimizer> fallbackOptimizer = placementFallbackOptimizerForRequest(request, program);
    if (fallbackOptimizer != nullptr) {
        // This preserves layer/parameter overrides because Network::connect attaches the default only to
        // trainable parameters that do not already have an optimizer.
        request.network->setDefaultOptimizer(fallbackOptimizer);
    }
}

std::shared_ptr<Optimizer> singleResolvedUpdateOptimizer(const StepExecutable& step) {
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::optional<uint64_t> optimizerOriginalId;

    for (const BoundParameter& parameter : step.getResolvedUpdateParameters()) {
        if (!parameter.hasOptimizer()) {
            return nullptr;
        }
        std::shared_ptr<Optimizer> parameterOptimizer = parameter.getOptimizer();
        if (parameterOptimizer == nullptr) {
            return nullptr;
        }

        if (!optimizerOriginalId.has_value()) {
            optimizer = parameterOptimizer;
            optimizerOriginalId = parameterOptimizer->getOriginalId();
            continue;
        }
        if (optimizerOriginalId.value() != parameterOptimizer->getOriginalId()) {
            return nullptr;
        }
    }

    return optimizer;
}

std::map<std::string, ThorImplementation::Tensor> bindBatchInputs(
    const StepExecutable& step,
    const std::map<std::string, ThorImplementation::Tensor>& batchInput) {
    std::map<std::string, ThorImplementation::Tensor> bound;
    for (const TrainingInputBinding& binding : step.getResolvedInputBindings()) {
        auto it = batchInput.find(binding.getBatchInputName());
        if (it == batchInput.end()) {
            throw std::runtime_error("Training batch is missing input '" + binding.getBatchInputName() +
                                     "' required for network input '" + binding.getNetworkInputName() + "'.");
        }
        bound.emplace(binding.getNetworkInputName(), it->second);
    }
    return bound;
}

void CUDART_CB completeNativeQueuedBatch(void* data) {
    NativeBatchCompletionParams* params = static_cast<NativeBatchCompletionParams*>(data);
    std::shared_ptr<QueuedTrainingState> state = params->state;
    const uint64_t epochBatchNum = params->epochBatchNum;

    try {
        std::unordered_map<std::string, std::vector<uint8_t>> bufferMap;
        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        for (const std::string& tensorName : params->tensorsToReturn) {
            ThorImplementation::Tensor copyFromTensor;
            auto inputIt = params->batchInput.find(tensorName);
            auto outputIt = params->batchOutput.find(tensorName);
            if (inputIt != params->batchInput.end()) {
                copyFromTensor = inputIt->second;
            } else if (outputIt != params->batchOutput.end()) {
                copyFromTensor = outputIt->second;
            } else {
                throw std::runtime_error("Requested training stat tensor '" + tensorName + "' was not present in batch inputs or outputs.");
            }

            THOR_THROW_IF_FALSE(copyFromTensor.getPlacement() == cpuPlacement);
            const uint64_t numTensorBytes = copyFromTensor.getDescriptor().getArraySizeInBytes();
            bufferMap[tensorName] = std::vector<uint8_t>(numTensorBytes);
            std::memcpy(bufferMap[tensorName].data(), copyFromTensor.getMemPtr(), numTensorBytes);
        }

        // CUDA host callbacks must not call CUDA APIs. Tensor destruction can call cudaFree/cudaFreeHost,
        // and some loaders may return CUDA-backed buffers. Defer ownership cleanup and loader buffer return
        // to the consumer thread by storing the completion parameters in QueuedTrainingState.
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->batchData[epochBatchNum] = std::move(bufferMap);
            state->completedBatchParams[epochBatchNum] = std::unique_ptr<NativeBatchCompletionParams>(params);
            state->batchReady[epochBatchNum] = true;
            state->numBatchesDoneInEpoch += 1;
        }
    } catch (...) {
        // Never let exceptions escape a CUDA host callback: doing so terminates the process.
        // Store the failure and the completion params so the normal consumer thread can return
        // loader-owned tensors and rethrow the error through Trainer.fit(...).
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->failure == nullptr) {
            state->failure = std::current_exception();
        }
        state->completedBatchParams[epochBatchNum] = std::unique_ptr<NativeBatchCompletionParams>(params);
        state->batchReady[epochBatchNum] = true;
        state->numBatchesDoneInEpoch += 1;
    }
    state->batchFinished.notify_all();
}

bool isBatchDataReadyUnlocked(const QueuedTrainingState& state) {
    if (state.batchReady.empty()) {
        return false;
    }
    return state.batchReady.begin()->second;
}

void waitForBatchDataUnlocked(QueuedTrainingState& state, std::unique_lock<std::mutex>& lock) {
    while (state.failure == nullptr && state.numBatchesDoneInEpoch != state.numBatchesInEpoch && !isBatchDataReadyUnlocked(state)) {
        state.batchFinished.wait(lock);
    }
}

struct BatchPopResult {
    bool hasBatch = false;
    std::unordered_map<std::string, std::vector<uint8_t>> data;
    std::unique_ptr<NativeBatchCompletionParams> completionParams;
};

BatchPopResult popBatchData(const std::shared_ptr<QueuedTrainingState>& state) {
    std::unique_lock<std::mutex> lock(state->mutex);
    waitForBatchDataUnlocked(*state, lock);

    if (!isBatchDataReadyUnlocked(*state)) {
        if (state->failure != nullptr) {
            std::rethrow_exception(state->failure);
        }
        return {};
    }

    const uint64_t epochBatchNum = state->batchReady.begin()->first;
    BatchPopResult result;
    result.hasBatch = true;
    result.data = std::move(state->batchData[epochBatchNum]);
    state->batchData.erase(epochBatchNum);

    auto completedParamsIt = state->completedBatchParams.find(epochBatchNum);
    THOR_THROW_IF_FALSE(completedParamsIt != state->completedBatchParams.end());
    result.completionParams = std::move(completedParamsIt->second);
    state->completedBatchParams.erase(completedParamsIt);

    state->batchReady.erase(state->batchReady.begin());

    lock.unlock();
    state->batchPopped.notify_all();
    return result;
}

uint64_t outstandingBatchCount(const std::shared_ptr<QueuedTrainingState>& state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->batchReady.size();
}

void throwIfQueuedTrainingStateFailed(const std::shared_ptr<QueuedTrainingState>& state) {
    std::exception_ptr failure;
    bool cancelRequested = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        failure = state->failure;
        cancelRequested = state->cancelRequested;
    }

    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    if (cancelRequested) {
        throw std::runtime_error("Native queued trainer was cancelled without a recorded failure.");
    }
}

void emitTrainingEvent(TrainingObserver& observer, bool statsEnabled, const TrainingEvent& event) {
    if (!statsEnabled && event.type == TrainingEventType::STATS) {
        return;
    }
    observer.onTrainingEvent(event);
}

class NativeQueuedEpochScheduler {
   public:
    NativeQueuedEpochScheduler(std::shared_ptr<PlacedNetwork> placedNetwork,
                               std::shared_ptr<Loader> loader,
                               const ExecutableTrainingPlan& plan,
                               const NativeQueuedTrainingOptions& options,
                               std::set<std::string> tensorsToReturn,
                               std::shared_ptr<QueuedTrainingState> state,
                               uint64_t currentEpoch,
                               uint64_t batchesPerEpoch,
                               ExampleType exampleType)
        : placedNetwork(std::move(placedNetwork)),
          loader(std::move(loader)),
          plan(plan),
          options(options),
          tensorsToReturn(std::move(tensorsToReturn)),
          state(std::move(state)),
          currentEpoch(currentEpoch),
          batchesPerEpoch(batchesPerEpoch),
          exampleType(exampleType) {}

    void operator()(uint64_t initialEpochBatchNum, uint64_t batches) {
        if (batches == 0) {
            return;
        }

        uint64_t nextStampToProcess = 0;
        std::vector<std::map<std::string, Event>> outputReadyEvents(placedNetwork->getNumStamps());
        const bool validationPass = exampleType != ExampleType::TRAIN;

        for (uint64_t batch = 0; batch < batches; ++batch) {
            uint64_t epochBatchNum = initialEpochBatchNum + batch;
            Optimizer::updateHyperParameters(placedNetwork.get(), currentEpoch, epochBatchNum, batchesPerEpoch);

            {
                std::unique_lock<std::mutex> lock(state->mutex);
                while (state->failure == nullptr && !state->cancelRequested && state->batchReady.size() >= options.maxInFlightBatches) {
                    state->batchPopped.wait(lock);
                }
                if (state->failure != nullptr || state->cancelRequested) {
                    return;
                }
                state->batchReady[epochBatchNum] = false;
            }

            auto params = std::make_unique<NativeBatchCompletionParams>();
            params->state = state;
            params->loader = loader;
            params->exampleType = exampleType;
            params->epochBatchNum = epochBatchNum;
            params->tensorsToReturn = tensorsToReturn;

            params->batchInput = loader->getBatch(exampleType, epochBatchNum);

            ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(nextStampToProcess);
            std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputs = stampedNetwork.getInputs();
            THOR_THROW_IF_FALSE(!inputs.empty());
            Stream completionStream = inputs[0]->getStream();

            for (const StepExecutable& step : plan.getSteps()) {
                for (uint32_t repeat = 0; repeat < step.getRepeatCount(); ++repeat) {
                    std::map<std::string, ThorImplementation::Tensor> boundBatchInput = bindBatchInputs(step, params->batchInput);
                    params->batchOutput.clear();
                    placedNetwork->submitBatch(nextStampToProcess,
                                               boundBatchInput,
                                               params->batchOutput,
                                               outputReadyEvents[nextStampToProcess],
                                               validationPass);
                }
            }

            NativeBatchCompletionParams* rawParams = params.get();
            cudaError_t cudaStatus = cudaLaunchHostFunc(completionStream, completeNativeQueuedBatch, rawParams);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            params.release();

            Event processingFinishedEvent = completionStream.putEvent();
            if (options.synchronizeAfterEveryBatch) {
                processingFinishedEvent.synchronize();
            }

            nextStampToProcess += 1;
            nextStampToProcess %= placedNetwork->getNumStamps();
        }
    }

   private:
    std::shared_ptr<PlacedNetwork> placedNetwork;
    std::shared_ptr<Loader> loader;
    const ExecutableTrainingPlan& plan;
    NativeQueuedTrainingOptions options;
    std::set<std::string> tensorsToReturn;
    std::shared_ptr<QueuedTrainingState> state;
    uint64_t currentEpoch;
    uint64_t batchesPerEpoch;
    ExampleType exampleType;
};

}  // namespace

void runNativeQueuedTraining(const TrainingRunRequest& request,
                             TrainingObserver& observer,
                             const NativeQueuedTrainingOptions& options) {
    THOR_THROW_IF_FALSE(request.network != nullptr);
    THOR_THROW_IF_FALSE(request.loader != nullptr);
    THOR_THROW_IF_FALSE(request.epochs > 0);
    THOR_THROW_IF_FALSE(options.maxInFlightBatches >= 1);

    TrainingProgram trainingProgram = defaultTrainingProgramForRequest(request);
    attachPlacementFallbackOptimizerIfNeeded(request, trainingProgram);

    const uint64_t batchSize = request.loader->getBatchSize();
    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placedNetwork = request.network->place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    THOR_THROW_IF_FALSE(placedNetwork->getNumStamps() == 1);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    ExecutableTrainingPlan plan = ExecutableTrainingPlan::compile(trainingProgram, *placedNetwork);
    ensureNativeQueuedPlanCompatible(plan, *request.network);

    std::shared_ptr<Optimizer> statsOptimizer = singleResolvedUpdateOptimizer(plan.getStep(0));
    ThorImplementation::StampedNetwork& statsStampedNetwork = placedNetwork->getStampedNetwork(0);
    const uint64_t forwardFlopsPerBatch = statsStampedNetwork.getFloatingPointOperationsPerExampleForward() * batchSize;
    const uint64_t trainingFlopsPerBatch = statsStampedNetwork.getFloatingPointOperationsPerExampleTraining() * batchSize;

    const auto runStart = std::chrono::high_resolution_clock::now();
    uint64_t currentEpoch = 0;
    double averageTrainingBatchTime = -1.0;
    double averageValidationBatchTime = -1.0;

    auto elapsedSinceRunStart = [&]() {
        const auto now = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - runStart);
        return elapsed.count();
    };

    auto makeBaseSnapshot = [&](TrainingPhase phase, uint64_t epoch, uint64_t batchSize, uint64_t batchesPerEpoch,
                                const std::shared_ptr<QueuedTrainingState>& state) {
        TrainingStatsSnapshot snapshot;
        snapshot.networkName = placedNetwork->getNetworkName();
        snapshot.datasetName = request.loader->getDatasetName();
        snapshot.phase = phase;
        snapshot.epoch = epoch;
        snapshot.epochs = request.epochs;
        snapshot.batchSize = batchSize;
        snapshot.stepsPerEpoch = batchesPerEpoch;
        snapshot.elapsedSeconds = elapsedSinceRunStart();
        snapshot.inFlightBatches = state ? outstandingBatchCount(state) : 0;
        return snapshot;
    };

    emitTrainingEvent(observer,
                      request.runtime.statsEnabled,
                      TrainingEvent::runStarted(makeBaseSnapshot(TrainingPhase::UNKNOWN, 0, batchSize, 0, nullptr)));

    for (uint32_t epochOffset = 0; epochOffset < request.epochs; ++epochOffset) {
        const uint64_t humanEpoch = currentEpoch + 1;

        for (const auto& phaseSpec : {std::pair<ExampleType, TrainingPhase>{ExampleType::TRAIN, TrainingPhase::TRAIN},
                                      std::pair<ExampleType, TrainingPhase>{ExampleType::VALIDATE, TrainingPhase::VALIDATE}}) {
            const ExampleType exampleType = phaseSpec.first;
            const TrainingPhase phase = phaseSpec.second;
            uint64_t batchNum = request.loader->getNextBatchNum(exampleType);
            const uint64_t batchesPerEpoch = request.loader->getNumBatchesPerEpoch(exampleType);
            if (batchNum > batchesPerEpoch) {
                throw std::runtime_error("Loader returned next batch number beyond batches per epoch for " + phaseName(phase) + ".");
            }
            const uint64_t batchesToRun = batchesPerEpoch - batchNum;

            auto state = std::make_shared<QueuedTrainingState>();
            state->numBatchesDoneInEpoch = batchNum;
            state->numBatchesInEpoch = batchesPerEpoch;

            emitTrainingEvent(observer,
                              request.runtime.statsEnabled,
                              TrainingEvent::epochStarted(makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state)));

            NativeQueuedEpochScheduler scheduler(placedNetwork,
                                                 request.loader,
                                                 plan,
                                                 options,
                                                 request.runtime.statsEnabled ? request.runtime.scalarTensorsToReport : std::set<std::string>{},
                                                 state,
                                                 currentEpoch,
                                                 batchesPerEpoch,
                                                 exampleType);
            std::thread schedulingThread([scheduler = std::move(scheduler), state, batchNum, batchesToRun]() mutable {
                try {
                    scheduler(batchNum, batchesToRun);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        if (state->failure == nullptr) {
                            state->failure = std::current_exception();
                        }
                        state->cancelRequested = true;
                        state->numBatchesDoneInEpoch = state->numBatchesInEpoch;
                    }
                    state->batchFinished.notify_all();
                    state->batchPopped.notify_all();
                }
            });

            auto cancelAndJoinScheduler = [&](std::exception_ptr failure) {
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    if (state->failure == nullptr) {
                        state->failure = failure;
                    }
                    state->cancelRequested = true;
                    state->numBatchesDoneInEpoch = state->numBatchesInEpoch;
                }
                state->batchFinished.notify_all();
                state->batchPopped.notify_all();
                if (schedulingThread.joinable()) {
                    schedulingThread.join();
                }
            };

            auto previousBatchDone = std::chrono::high_resolution_clock::now();
            try {
                while (true) {
                    BatchPopResult completedBatch = popBatchData(state);
                    if (!completedBatch.hasBatch) {
                        break;
                    }

                    if (completedBatch.completionParams != nullptr) {
                        completedBatch.completionParams->loader->returnBatchBuffers(completedBatch.completionParams->exampleType,
                                                                                     completedBatch.completionParams->batchInput);
                        completedBatch.completionParams.reset();
                    }

                    {
                        std::exception_ptr failure;
                        {
                            std::lock_guard<std::mutex> lock(state->mutex);
                            failure = state->failure;
                        }
                        if (failure != nullptr) {
                            cancelAndJoinScheduler(failure);
                            std::rethrow_exception(failure);
                        }
                    }

                    const auto now = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - previousBatchDone);
                    previousBatchDone = now;

                    double& averageBatchTime = (phase == TrainingPhase::TRAIN) ? averageTrainingBatchTime : averageValidationBatchTime;
                    if (averageBatchTime < 0.0) {
                        averageBatchTime = elapsed.count();
                    } else {
                        averageBatchTime = 0.05 * elapsed.count() + 0.95 * averageBatchTime;
                    }

                    TrainingStatsSnapshot snapshot = makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state);
                    snapshot.stepInEpoch = batchNum + 1;
                    snapshot.step = (currentEpoch * batchesPerEpoch) + snapshot.stepInEpoch;
                    snapshot.samplesProcessed = snapshot.step * batchSize;
                    if (averageBatchTime > 0.0) {
                        snapshot.samplesPerSecond = static_cast<double>(batchSize) / averageBatchTime;
                        snapshot.batchesPerSecond = 1.0 / averageBatchTime;
                        snapshot.floatingPointOperationsPerBatch =
                            (phase == TrainingPhase::TRAIN) ? trainingFlopsPerBatch : forwardFlopsPerBatch;
                        snapshot.floatingPointOperationsPerSecond =
                            static_cast<double>(snapshot.floatingPointOperationsPerBatch) / averageBatchTime;
                    }

                    if (statsOptimizer != nullptr) {
                        std::unordered_map<std::string, float> optimizerParameters = statsOptimizer->getAllHyperParameters(placedNetwork.get());
                        if (optimizerParameters.count("currentLearningRate") > 0) {
                            snapshot.learningRate = optimizerParameters["currentLearningRate"];
                        }
                        if (optimizerParameters.count("momentum") > 0) {
                            snapshot.momentum = optimizerParameters["momentum"];
                        }
                    }

                    snapshot.loss = optionalFloatFromBatch(completedBatch.data, "loss");
                    snapshot.accuracy = optionalFloatFromBatch(completedBatch.data, "accuracy");
                    emitTrainingEvent(observer, request.runtime.statsEnabled, TrainingEvent::statsUpdated(std::move(snapshot)));

                    batchNum += 1;
                }
            } catch (...) {
                cancelAndJoinScheduler(std::current_exception());
                throw;
            }

            schedulingThread.join();
            throwIfQueuedTrainingStateFailed(state);
            emitTrainingEvent(observer,
                              request.runtime.statsEnabled,
                              TrainingEvent::epochFinished(makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state)));
        }

        currentEpoch += 1;
    }

    emitTrainingEvent(observer,
                      request.runtime.statsEnabled,
                      TrainingEvent::runFinished(makeBaseSnapshot(TrainingPhase::UNKNOWN, currentEpoch, batchSize, 0, nullptr)));
}

}  // namespace Thor
