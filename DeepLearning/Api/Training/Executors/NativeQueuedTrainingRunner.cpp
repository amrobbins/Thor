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
#include <csignal>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace Thor {

namespace {

struct QueuedTrainingState;

struct ScalarStatSlot {
    bool present = false;
    float value = 0.0f;
};

struct NativeBatchCompletionParams {
    std::shared_ptr<QueuedTrainingState> state;
    std::shared_ptr<Loader> loader;
    ExampleType exampleType = ExampleType::TRAIN;
    uint64_t currentEpoch = 0;
    uint64_t epochBatchNum = 0;
    uint64_t slotIndex = 0;
    std::map<std::string, ThorImplementation::Tensor> batchInput;
    std::map<std::string, ThorImplementation::Tensor> batchOutput;
    std::vector<ScalarStatSlot> scalarStats;
};

struct QueuedBatchSlot {
    bool occupied = false;
    bool ready = false;
    uint64_t epochBatchNum = 0;
    uint64_t paramsIndex = 0;
    std::vector<ScalarStatSlot> scalarStats;
};

struct QueuedTrainingState {
    QueuedTrainingState(uint64_t maxInFlightBatches, std::vector<std::string> scalarTensorNames)
        : scalarTensorNames(std::move(scalarTensorNames)), slots(maxInFlightBatches), completionParams(maxInFlightBatches) {
        THOR_THROW_IF_FALSE(maxInFlightBatches >= 1);
        for (QueuedBatchSlot& slot : slots) {
            slot.scalarStats.resize(this->scalarTensorNames.size());
        }
        for (NativeBatchCompletionParams& params : completionParams) {
            params.scalarStats.resize(this->scalarTensorNames.size());
        }
    }

    std::mutex mutex;
    std::condition_variable batchFinished;
    std::condition_variable batchPopped;

    std::vector<std::string> scalarTensorNames;
    std::vector<QueuedBatchSlot> slots;
    std::vector<NativeBatchCompletionParams> completionParams;
    uint64_t headSlot = 0;
    uint64_t tailSlot = 0;
    uint64_t inFlightBatches = 0;

    std::exception_ptr failure;
    bool cancelRequested = false;
    bool interruptRequested = false;

    uint64_t numBatchesDoneInEpoch = 0;
    uint64_t numBatchesInEpoch = 0;
};

float copyScalarStatTensor(const std::map<std::string, ThorImplementation::Tensor>& batchInput,
                           const std::map<std::string, ThorImplementation::Tensor>& batchOutput,
                           const std::string& tensorName) {
    ThorImplementation::Tensor copyFromTensor;
    auto inputIt = batchInput.find(tensorName);
    auto outputIt = batchOutput.find(tensorName);
    if (inputIt != batchInput.end()) {
        copyFromTensor = inputIt->second;
    } else if (outputIt != batchOutput.end()) {
        copyFromTensor = outputIt->second;
    } else {
        throw std::runtime_error("Requested training stat tensor '" + tensorName + "' was not present in batch inputs or outputs.");
    }

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    THOR_THROW_IF_FALSE(copyFromTensor.getPlacement() == cpuPlacement);
    THOR_THROW_IF_FALSE(copyFromTensor.getDescriptor().getArraySizeInBytes() >= sizeof(float));
    float value = 0.0f;
    std::memcpy(&value, copyFromTensor.getMemPtr(), sizeof(float));
    return value;
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

uint64_t elapsedMicros(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point finish) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
}

void emitNativeQueueDiagnostic(const char* event,
                               TrainingPhase phase,
                               uint64_t epoch,
                               uint64_t batch,
                               uint64_t slot,
                               uint64_t inFlight,
                               uint64_t done,
                               uint64_t total,
                               uint64_t waitMicros = 0) {
    if (!queueDiagnosticsEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "THOR_TRAINING_QUEUE_DIAGNOSTIC native event=%s phase=%s epoch=%lu batch=%lu slot=%lu in_flight=%lu done=%lu/%lu wait_us=%lu\n",
                 event,
                 phaseName(phase).c_str(),
                 epoch + 1,
                 batch + 1,
                 slot,
                 inFlight,
                 done,
                 total,
                 waitMicros);
    std::fflush(stderr);
}

void emitNativeQueueScheduleTimingDiagnostic(TrainingPhase phase,
                                             uint64_t epoch,
                                             uint64_t batch,
                                             uint64_t slot,
                                             uint64_t inFlight,
                                             uint64_t done,
                                             uint64_t total,
                                             uint64_t optimizerMicros,
                                             uint64_t reserveMicros,
                                             uint64_t getBatchMicros,
                                             uint64_t submitMicros,
                                             uint64_t completionMicros,
                                             uint64_t totalMicros) {
    if (!queueDiagnosticsEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "THOR_TRAINING_QUEUE_DIAGNOSTIC native event=schedule_timing phase=%s epoch=%lu batch=%lu slot=%lu "
                 "in_flight=%lu done=%lu/%lu optimizer_us=%lu reserve_us=%lu get_batch_us=%lu submit_us=%lu completion_us=%lu total_us=%lu\n",
                 phaseName(phase).c_str(),
                 epoch + 1,
                 batch + 1,
                 slot,
                 inFlight,
                 done,
                 total,
                 optimizerMicros,
                 reserveMicros,
                 getBatchMicros,
                 submitMicros,
                 completionMicros,
                 totalMicros);
    std::fflush(stderr);
}

void emitNativeQueueCompletionTimingDiagnostic(TrainingPhase phase,
                                               uint64_t epoch,
                                               uint64_t batch,
                                               uint64_t slot,
                                               uint64_t inFlight,
                                               uint64_t done,
                                               uint64_t total,
                                               uint64_t outputWaitCount,
                                               uint64_t waitProcessingMicros,
                                               uint64_t waitOutputsMicros,
                                               uint64_t hostFuncMicros,
                                               uint64_t putEventMicros,
                                               uint64_t extendOutputsMicros,
                                               uint64_t totalMicros) {
    if (!queueDiagnosticsEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "THOR_TRAINING_QUEUE_DIAGNOSTIC native event=completion_timing phase=%s epoch=%lu batch=%lu slot=%lu "
                 "in_flight=%lu done=%lu/%lu output_waits=%lu wait_processing_us=%lu wait_outputs_us=%lu "
                 "host_func_us=%lu put_event_us=%lu extend_outputs_us=%lu total_us=%lu\n",
                 phaseName(phase).c_str(),
                 epoch + 1,
                 batch + 1,
                 slot,
                 inFlight,
                 done,
                 total,
                 outputWaitCount,
                 waitProcessingMicros,
                 waitOutputsMicros,
                 hostFuncMicros,
                 putEventMicros,
                 extendOutputsMicros,
                 totalMicros);
    std::fflush(stderr);
}

volatile std::sig_atomic_t gNativeQueuedSigintRequested = 0;

void handleNativeQueuedSigint(int) { gNativeQueuedSigintRequested = 1; }

class NativeQueuedSigintScope {
   public:
    NativeQueuedSigintScope() {
        gNativeQueuedSigintRequested = 0;
        previousHandler = std::signal(SIGINT, handleNativeQueuedSigint);
    }

    ~NativeQueuedSigintScope() { std::signal(SIGINT, previousHandler); }

    bool interrupted() const { return gNativeQueuedSigintRequested != 0; }

   private:
    using SignalHandler = void (*)(int);
    SignalHandler previousHandler = SIG_DFL;
};

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

std::map<std::string, ThorImplementation::Tensor> bindBatchInputs(const StepExecutable& step,
                                                                  const std::map<std::string, ThorImplementation::Tensor>& batchInput) {
    std::map<std::string, ThorImplementation::Tensor> bound;
    for (const TrainingInputBinding& binding : step.getResolvedInputBindings()) {
        auto it = batchInput.find(binding.getBatchInputName());
        if (it == batchInput.end()) {
            throw std::runtime_error("Training batch is missing input '" + binding.getBatchInputName() + "' required for network input '" +
                                     binding.getNetworkInputName() + "'.");
        }
        bound.emplace(binding.getNetworkInputName(), it->second);
    }
    return bound;
}

void CUDART_CB completeNativeQueuedBatch(void* data) {
    NativeBatchCompletionParams* params = static_cast<NativeBatchCompletionParams*>(data);
    std::shared_ptr<QueuedTrainingState> state = params->state;
    const uint64_t epochBatchNum = params->epochBatchNum;
    const uint64_t slotIndex = params->slotIndex;

    try {
        THOR_THROW_IF_FALSE(params->scalarStats.size() == state->scalarTensorNames.size());
        for (size_t i = 0; i < state->scalarTensorNames.size(); ++i) {
            params->scalarStats[i].value = copyScalarStatTensor(params->batchInput, params->batchOutput, state->scalarTensorNames[i]);
            params->scalarStats[i].present = true;
        }

        uint64_t inFlightAtComplete = 0;
        uint64_t doneAtComplete = 0;
        uint64_t totalAtComplete = 0;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            THOR_THROW_IF_FALSE(slotIndex < state->slots.size());
            QueuedBatchSlot& slot = state->slots[slotIndex];
            THOR_THROW_IF_FALSE(slot.occupied);
            THOR_THROW_IF_FALSE(slot.epochBatchNum == epochBatchNum);
            THOR_THROW_IF_FALSE(slot.scalarStats.size() == params->scalarStats.size());
            slot.scalarStats = params->scalarStats;
            slot.ready = true;
            state->numBatchesDoneInEpoch += 1;
            inFlightAtComplete = state->inFlightBatches;
            doneAtComplete = state->numBatchesDoneInEpoch;
            totalAtComplete = state->numBatchesInEpoch;
        }
        if (shouldEmitQueueDiagnostic(doneAtComplete)) {
            emitNativeQueueDiagnostic("complete",
                                      params->exampleType == ExampleType::TRAIN ? TrainingPhase::TRAIN
                                                                                : params->exampleType == ExampleType::VALIDATE
                                                                                      ? TrainingPhase::VALIDATE
                                                                                      : TrainingPhase::TEST,
                                      params->currentEpoch,
                                      epochBatchNum,
                                      slotIndex,
                                      inFlightAtComplete,
                                      doneAtComplete,
                                      totalAtComplete);
        }
    } catch (...) {
        // Never let exceptions escape a CUDA host callback: doing so terminates the process.
        // Store the failure and mark the slot ready so the consumer thread can return
        // loader-owned tensors and rethrow the error through Trainer.fit(...).
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->failure == nullptr) {
            state->failure = std::current_exception();
        }
        if (slotIndex < state->slots.size()) {
            QueuedBatchSlot& slot = state->slots[slotIndex];
            if (slot.occupied && slot.epochBatchNum == epochBatchNum) {
                slot.ready = true;
            }
        }
        state->numBatchesDoneInEpoch += 1;
    }
    state->batchFinished.notify_all();
}

bool isBatchDataReadyUnlocked(const QueuedTrainingState& state) {
    if (state.inFlightBatches == 0) {
        return false;
    }
    const QueuedBatchSlot& slot = state.slots[state.headSlot];
    return slot.occupied && slot.ready;
}

void waitForBatchDataUnlocked(QueuedTrainingState& state, std::unique_lock<std::mutex>& lock) {
    while (state.failure == nullptr && !isBatchDataReadyUnlocked(state)) {
        if (state.cancelRequested && state.inFlightBatches == 0) {
            return;
        }
        if (state.numBatchesDoneInEpoch == state.numBatchesInEpoch && state.inFlightBatches == 0) {
            return;
        }
        state.batchFinished.wait_for(lock, std::chrono::milliseconds(50));
    }
}

struct BatchPopResult {
    bool hasBatch = false;
    std::shared_ptr<Loader> loader;
    ExampleType exampleType = ExampleType::TRAIN;
    uint64_t currentEpoch = 0;
    uint64_t epochBatchNum = 0;
    uint64_t slotIndex = 0;
    uint64_t inFlightAfterPop = 0;
    uint64_t doneInEpoch = 0;
    uint64_t batchesInEpoch = 0;
    std::map<std::string, ThorImplementation::Tensor> batchInput;
    std::vector<ScalarStatSlot> scalarStats;
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

    QueuedBatchSlot& slot = state->slots[state->headSlot];
    NativeBatchCompletionParams& params = state->completionParams[slot.paramsIndex];

    BatchPopResult result;
    result.hasBatch = true;
    result.loader = params.loader;
    result.exampleType = params.exampleType;
    result.epochBatchNum = slot.epochBatchNum;
    result.slotIndex = state->headSlot;
    result.doneInEpoch = state->numBatchesDoneInEpoch;
    result.batchesInEpoch = state->numBatchesInEpoch;
    result.batchInput = std::move(params.batchInput);
    result.scalarStats = slot.scalarStats;

    params.batchOutput.clear();
    params.loader.reset();
    params.state.reset();
    for (ScalarStatSlot& scalarStat : slot.scalarStats) {
        scalarStat.present = false;
        scalarStat.value = 0.0f;
    }
    for (ScalarStatSlot& scalarStat : params.scalarStats) {
        scalarStat.present = false;
        scalarStat.value = 0.0f;
    }
    slot.ready = false;
    slot.occupied = false;
    state->headSlot = (state->headSlot + 1) % state->slots.size();
    state->inFlightBatches -= 1;
    result.inFlightAfterPop = state->inFlightBatches;

    lock.unlock();
    state->batchPopped.notify_all();
    return result;
}

uint64_t outstandingBatchCount(const std::shared_ptr<QueuedTrainingState>& state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->inFlightBatches;
}

void throwIfQueuedTrainingStateFailed(const std::shared_ptr<QueuedTrainingState>& state) {
    std::exception_ptr failure;
    bool cancelRequested = false;
    bool interruptRequested = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        failure = state->failure;
        cancelRequested = state->cancelRequested;
        interruptRequested = state->interruptRequested;
    }

    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    if (interruptRequested) {
        throw std::runtime_error("Native queued trainer interrupted by SIGINT.");
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

void assignScalarStatsToSnapshot(TrainingStatsSnapshot& snapshot,
                                 const std::vector<std::string>& scalarTensorNames,
                                 const std::vector<ScalarStatSlot>& scalarStats) {
    THOR_THROW_IF_FALSE(scalarTensorNames.size() == scalarStats.size());
    for (size_t i = 0; i < scalarTensorNames.size(); ++i) {
        if (!scalarStats[i].present) {
            continue;
        }

        const double value = static_cast<double>(scalarStats[i].value);
        const std::string& name = scalarTensorNames[i];
        if (name == "loss") {
            snapshot.loss = value;
        } else if (name == "accuracy") {
            snapshot.accuracy = value;
        } else if (name == "learning_rate" || name == "learningRate" || name == "lr") {
            snapshot.learningRate = value;
        } else if (name == "momentum") {
            snapshot.momentum = value;
        } else {
            snapshot.metrics[name] = value;
        }
    }
}

class NativeQueuedEpochScheduler {
   public:
    NativeQueuedEpochScheduler(std::shared_ptr<PlacedNetwork> placedNetwork,
                               std::shared_ptr<Loader> loader,
                               const ExecutableTrainingPlan& plan,
                               const NativeQueuedTrainingOptions& options,
                               std::shared_ptr<QueuedTrainingState> state,
                               uint64_t currentEpoch,
                               uint64_t batchesPerEpoch,
                               ExampleType exampleType)
        : placedNetwork(std::move(placedNetwork)),
          loader(std::move(loader)),
          plan(plan),
          options(options),
          state(std::move(state)),
          currentEpoch(currentEpoch),
          batchesPerEpoch(batchesPerEpoch),
          exampleType(exampleType) {}

    void operator()(uint64_t initialEpochBatchNum, uint64_t batches) {
        if (batches == 0) {
            return;
        }

        const TrainingPhase diagnosticPhase = exampleType == ExampleType::TRAIN
                                                  ? TrainingPhase::TRAIN
                                                  : exampleType == ExampleType::VALIDATE ? TrainingPhase::VALIDATE : TrainingPhase::TEST;
        emitNativeQueueDiagnostic("phase_schedule_start",
                                  diagnosticPhase,
                                  currentEpoch,
                                  initialEpochBatchNum,
                                  0,
                                  0,
                                  initialEpochBatchNum,
                                  batchesPerEpoch);

        uint64_t nextStampToProcess = 0;
        std::vector<std::map<std::string, Event>> outputReadyEvents(placedNetwork->getNumStamps());
        std::vector<Event> processingFinishedEvents(options.maxInFlightBatches);
        std::vector<Event> completionFinishedEvents(options.maxInFlightBatches);
        std::vector<Stream> completionStreams;
        completionStreams.reserve(placedNetwork->getNumStamps());
        for (uint64_t stamp = 0; stamp < placedNetwork->getNumStamps(); ++stamp) {
            ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(stamp);
            std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputs = stampedNetwork.getInputs();
            THOR_THROW_IF_FALSE(!inputs.empty());
            const int gpuNum = inputs[0]->getStream().getGpuNum();
            completionStreams.push_back(Stream::getNextDownloadStream(gpuNum));
        }
        const bool validationPass = exampleType != ExampleType::TRAIN;
        const std::vector<StepExecutable>& steps = plan.getSteps();

        for (uint64_t batch = 0; batch < batches; ++batch) {
            const auto scheduleIterationStart = std::chrono::high_resolution_clock::now();
            uint64_t epochBatchNum = initialEpochBatchNum + batch;
            const auto optimizerStart = std::chrono::high_resolution_clock::now();
            Optimizer::updateHyperParameters(placedNetwork.get(), currentEpoch, epochBatchNum, batchesPerEpoch);
            const auto optimizerFinish = std::chrono::high_resolution_clock::now();

            uint64_t slotIndex = 0;
            uint64_t inFlightAfterReserve = 0;
            const auto reserveStart = std::chrono::high_resolution_clock::now();
            {
                std::unique_lock<std::mutex> lock(state->mutex);
                while (state->failure == nullptr && !state->cancelRequested && state->inFlightBatches >= options.maxInFlightBatches) {
                    state->batchPopped.wait(lock);
                }
                if (state->failure != nullptr || state->cancelRequested) {
                    return;
                }

                slotIndex = state->tailSlot;
                QueuedBatchSlot& slot = state->slots[slotIndex];
                THOR_THROW_IF_FALSE(!slot.occupied);
                slot.occupied = true;
                slot.ready = false;
                slot.epochBatchNum = epochBatchNum;
                slot.paramsIndex = slotIndex;
                for (ScalarStatSlot& scalarStat : slot.scalarStats) {
                    scalarStat.present = false;
                    scalarStat.value = 0.0f;
                }
                state->tailSlot = (state->tailSlot + 1) % state->slots.size();
                state->inFlightBatches += 1;
                inFlightAfterReserve = state->inFlightBatches;
            }
            const auto reserveFinish = std::chrono::high_resolution_clock::now();
            if (shouldEmitQueueDiagnostic(batch + 1)) {
                emitNativeQueueDiagnostic("reserve",
                                          diagnosticPhase,
                                          currentEpoch,
                                          epochBatchNum,
                                          slotIndex,
                                          inFlightAfterReserve,
                                          initialEpochBatchNum + batch,
                                          batchesPerEpoch);
            }

            NativeBatchCompletionParams* params = &state->completionParams[slotIndex];
            params->state = state;
            params->loader = loader;
            params->exampleType = exampleType;
            params->currentEpoch = currentEpoch;
            params->epochBatchNum = epochBatchNum;
            params->slotIndex = slotIndex;
            params->batchInput.clear();
            params->batchOutput.clear();
            for (ScalarStatSlot& scalarStat : params->scalarStats) {
                scalarStat.present = false;
                scalarStat.value = 0.0f;
            }

            const auto getBatchStart = std::chrono::high_resolution_clock::now();
            params->batchInput = loader->getBatch(exampleType, epochBatchNum);
            const auto getBatchFinish = std::chrono::high_resolution_clock::now();
            const uint64_t getBatchWaitMicros = elapsedMicros(getBatchStart, getBatchFinish);
            if (shouldEmitQueueDiagnostic(batch + 1, getBatchWaitMicros)) {
                emitNativeQueueDiagnostic("get_batch_done",
                                          diagnosticPhase,
                                          currentEpoch,
                                          epochBatchNum,
                                          slotIndex,
                                          inFlightAfterReserve,
                                          initialEpochBatchNum + batch,
                                          batchesPerEpoch,
                                          getBatchWaitMicros);
            }

            const auto submitStart = std::chrono::high_resolution_clock::now();
            for (const StepExecutable& step : steps) {
                for (uint32_t repeat = 0; repeat < step.getRepeatCount(); ++repeat) {
                    std::map<std::string, ThorImplementation::Tensor> boundBatchInput = bindBatchInputs(step, params->batchInput);
                    params->batchOutput.clear();
                    placedNetwork->submitBatch(nextStampToProcess,
                                               std::move(boundBatchInput),
                                               params->batchOutput,
                                               outputReadyEvents[nextStampToProcess],
                                               validationPass,
                                               &processingFinishedEvents[slotIndex],
                                               /*waitForOutputsOnProcessingStream=*/false);
                }
            }
            const auto submitFinish = std::chrono::high_resolution_clock::now();

            const auto completionSetupStart = std::chrono::high_resolution_clock::now();
            // Keep CPU stats/output completion off the stamp's input stream.  The input stream
            // event is the point where the GPU training work is done enough for the next batch
            // to be queued on this single stamp.  Output tensors that are copied through
            // NetworkOutput-owned download streams are waited on here, and the host callback
            // snapshots the shared CPU output tensors into per-slot scalarStats before those
            // public output tensors may be reused by a later batch.
            Stream completionStream = completionStreams[nextStampToProcess];
            const auto waitProcessingStart = std::chrono::high_resolution_clock::now();
            completionStream.waitEvent(processingFinishedEvents[slotIndex]);
            const auto waitProcessingFinish = std::chrono::high_resolution_clock::now();

            const auto waitOutputsStart = std::chrono::high_resolution_clock::now();
            uint64_t outputWaitCount = 0;
            for (const auto& [outputName, outputReadyEvent] : outputReadyEvents[nextStampToProcess]) {
                (void)outputName;
                completionStream.waitEvent(outputReadyEvent);
                outputWaitCount += 1;
            }
            const auto waitOutputsFinish = std::chrono::high_resolution_clock::now();

            const auto hostFuncStart = std::chrono::high_resolution_clock::now();
            cudaError_t cudaStatus = cudaLaunchHostFunc(completionStream, completeNativeQueuedBatch, params);
            const auto hostFuncFinish = std::chrono::high_resolution_clock::now();
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            if (shouldEmitQueueDiagnostic(batch + 1)) {
                emitNativeQueueDiagnostic("submit",
                                          diagnosticPhase,
                                          currentEpoch,
                                          epochBatchNum,
                                          slotIndex,
                                          inFlightAfterReserve,
                                          initialEpochBatchNum + batch,
                                          batchesPerEpoch);
            }
            const auto putEventStart = std::chrono::high_resolution_clock::now();
            completionStream.putEvent(completionFinishedEvents[slotIndex], false, true);
            const auto putEventFinish = std::chrono::high_resolution_clock::now();

            const auto extendOutputsStart = std::chrono::high_resolution_clock::now();
            placedNetwork->extendOutputWritableEvents(nextStampToProcess, completionFinishedEvents[slotIndex]);
            const auto extendOutputsFinish = std::chrono::high_resolution_clock::now();
            const auto completionSetupFinish = extendOutputsFinish;

            if (shouldEmitQueueDiagnostic(batch + 1)) {
                emitNativeQueueCompletionTimingDiagnostic(diagnosticPhase,
                                                          currentEpoch,
                                                          epochBatchNum,
                                                          slotIndex,
                                                          inFlightAfterReserve,
                                                          initialEpochBatchNum + batch,
                                                          batchesPerEpoch,
                                                          outputWaitCount,
                                                          elapsedMicros(waitProcessingStart, waitProcessingFinish),
                                                          elapsedMicros(waitOutputsStart, waitOutputsFinish),
                                                          elapsedMicros(hostFuncStart, hostFuncFinish),
                                                          elapsedMicros(putEventStart, putEventFinish),
                                                          elapsedMicros(extendOutputsStart, extendOutputsFinish),
                                                          elapsedMicros(completionSetupStart, completionSetupFinish));
            }

            if (shouldEmitQueueDiagnostic(batch + 1)) {
                emitNativeQueueScheduleTimingDiagnostic(diagnosticPhase,
                                                        currentEpoch,
                                                        epochBatchNum,
                                                        slotIndex,
                                                        inFlightAfterReserve,
                                                        initialEpochBatchNum + batch,
                                                        batchesPerEpoch,
                                                        elapsedMicros(optimizerStart, optimizerFinish),
                                                        elapsedMicros(reserveStart, reserveFinish),
                                                        getBatchWaitMicros,
                                                        elapsedMicros(submitStart, submitFinish),
                                                        elapsedMicros(completionSetupStart, completionSetupFinish),
                                                        elapsedMicros(scheduleIterationStart, completionSetupFinish));
            }

            if (options.synchronizeAfterEveryBatch) {
                completionStream.synchronize();
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
    std::shared_ptr<QueuedTrainingState> state;
    uint64_t currentEpoch;
    uint64_t batchesPerEpoch;
    ExampleType exampleType;
};

}  // namespace

void runNativeQueuedTraining(const TrainingRunRequest& request, TrainingObserver& observer, const NativeQueuedTrainingOptions& options) {
    NativeQueuedSigintScope sigintScope;

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
    for (size_t i = 0; i < initDoneEvents.size(); ++i) {
        initDoneEvents[i].synchronize();
    }

    ExecutableTrainingPlan plan = ExecutableTrainingPlan::compile(trainingProgram, *placedNetwork);
    ensureNativeQueuedPlanCompatible(plan, *request.network);

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

    auto makeBaseSnapshot = [&](TrainingPhase phase,
                                uint64_t epoch,
                                uint64_t batchSize,
                                uint64_t batchesPerEpoch,
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

    if (request.runtime.statsEnabled) {
        emitTrainingEvent(observer,
                          request.runtime.statsEnabled,
                          TrainingEvent::runStarted(makeBaseSnapshot(TrainingPhase::UNKNOWN, 0, batchSize, 0, nullptr)));
    }

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

            std::vector<std::string> scalarTensorNames;
            if (request.runtime.statsEnabled) {
                scalarTensorNames.assign(request.runtime.scalarTensorsToReport.begin(), request.runtime.scalarTensorsToReport.end());
            }
            auto state = std::make_shared<QueuedTrainingState>(options.maxInFlightBatches, std::move(scalarTensorNames));
            state->numBatchesDoneInEpoch = batchNum;
            state->numBatchesInEpoch = batchesPerEpoch;

            if (request.runtime.statsEnabled) {
                emitTrainingEvent(observer,
                                  request.runtime.statsEnabled,
                                  TrainingEvent::epochStarted(makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state)));
            }

            NativeQueuedEpochScheduler scheduler(
                placedNetwork, request.loader, plan, options, state, currentEpoch, batchesPerEpoch, exampleType);
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

            auto requestSigintCancel = [&]() {
                if (!sigintScope.interrupted()) {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->cancelRequested = true;
                    state->interruptRequested = true;
                }
                state->batchFinished.notify_all();
                state->batchPopped.notify_all();
            };

            auto previousBatchDone = std::chrono::high_resolution_clock::now();
            bool haveCompletedBatchTimingInPhase = false;
            try {
                while (true) {
                    requestSigintCancel();
                    BatchPopResult completedBatch = popBatchData(state);
                    if (!completedBatch.hasBatch) {
                        break;
                    }

                    if (completedBatch.loader != nullptr) {
                        completedBatch.loader->returnBatchBuffers(completedBatch.exampleType, std::move(completedBatch.batchInput));
                    }
                    if (shouldEmitQueueDiagnostic(completedBatch.doneInEpoch)) {
                        emitNativeQueueDiagnostic("pop_return",
                                                  phase,
                                                  currentEpoch,
                                                  completedBatch.epochBatchNum,
                                                  completedBatch.slotIndex,
                                                  completedBatch.inFlightAfterPop,
                                                  completedBatch.doneInEpoch,
                                                  completedBatch.batchesInEpoch);
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

                    if (request.runtime.statsEnabled) {
                        const auto now = std::chrono::high_resolution_clock::now();
                        const std::chrono::duration<double> elapsed =
                            std::chrono::duration_cast<std::chrono::duration<double>>(now - previousBatchDone);
                        previousBatchDone = now;

                        double& averageBatchTime = (phase == TrainingPhase::TRAIN) ? averageTrainingBatchTime : averageValidationBatchTime;
                        const double completedBatchTime = elapsed.count();
                        double batchTimeForStats = completedBatchTime;
                        if (haveCompletedBatchTimingInPhase) {
                            if (averageBatchTime < 0.0) {
                                averageBatchTime = completedBatchTime;
                            } else {
                                averageBatchTime = 0.25 * completedBatchTime + 0.75 * averageBatchTime;
                            }
                            batchTimeForStats = averageBatchTime;
                        } else if (averageBatchTime > 0.0) {
                            // The first completion in a phase includes queue fill and any loader/startup
                            // latency before the pipeline reaches steady state.  Use it for the current
                            // line only when no prior phase average exists; do not let it poison the EMA.
                            batchTimeForStats = averageBatchTime;
                        }
                        haveCompletedBatchTimingInPhase = true;

                        TrainingStatsSnapshot snapshot = makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state);
                        snapshot.stepInEpoch = batchNum + 1;
                        snapshot.step = (currentEpoch * batchesPerEpoch) + snapshot.stepInEpoch;
                        snapshot.samplesProcessed = snapshot.step * batchSize;
                        if (batchTimeForStats > 0.0) {
                            snapshot.samplesPerSecond = static_cast<double>(batchSize) / batchTimeForStats;
                            snapshot.batchesPerSecond = 1.0 / batchTimeForStats;
                            snapshot.floatingPointOperationsPerBatch =
                                (phase == TrainingPhase::TRAIN) ? trainingFlopsPerBatch : forwardFlopsPerBatch;
                            snapshot.floatingPointOperationsPerSecond =
                                static_cast<double>(snapshot.floatingPointOperationsPerBatch) / batchTimeForStats;
                        }

                        assignScalarStatsToSnapshot(snapshot, state->scalarTensorNames, completedBatch.scalarStats);
                        emitTrainingEvent(observer, request.runtime.statsEnabled, TrainingEvent::statsUpdated(std::move(snapshot)));
                    }

                    batchNum += 1;
                }
            } catch (...) {
                cancelAndJoinScheduler(std::current_exception());
                throw;
            }

            schedulingThread.join();
            throwIfQueuedTrainingStateFailed(state);
            if (request.runtime.statsEnabled) {
                emitTrainingEvent(observer,
                                  request.runtime.statsEnabled,
                                  TrainingEvent::epochFinished(makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state)));
            }
        }

        currentEpoch += 1;
    }

    if (request.runtime.statsEnabled) {
        emitTrainingEvent(observer,
                          request.runtime.statsEnabled,
                          TrainingEvent::runFinished(makeBaseSnapshot(TrainingPhase::UNKNOWN, currentEpoch, batchSize, 0, nullptr)));
    }
}

}  // namespace Thor
