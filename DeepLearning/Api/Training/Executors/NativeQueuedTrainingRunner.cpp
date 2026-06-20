#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Loaders/Batch.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Training/ExecutableTrainingPlan.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"

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
#include <variant>

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
    Batch batchInput;
    std::map<std::string, ThorImplementation::Tensor> batchOutput;
    std::vector<ScalarStatSlot> scalarStats;
};

struct QueuedBatchSlot {
    bool occupied = false;
    bool ready = false;
    uint64_t epochBatchNum = 0;
    uint64_t paramsIndex = 0;
    std::chrono::high_resolution_clock::time_point completionTime{};
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

void requestQueuedTrainingCancellation(const std::shared_ptr<QueuedTrainingState>& state) {
    if (state == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->cancelRequested = true;
    }
    state->batchFinished.notify_all();
    state->batchPopped.notify_all();
}

float copyScalarStatTensor(const Batch& batchInput,
                           const std::map<std::string, ThorImplementation::Tensor>& batchOutput,
                           const std::string& tensorName) {
    ThorImplementation::Tensor copyFromTensor;
    auto outputIt = batchOutput.find(tensorName);
    if (batchInput.contains(tensorName)) {
        copyFromTensor = batchInput.getTensor(tensorName);
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

std::string phaseName(TrainingEventPhase phase) {
    switch (phase) {
        case TrainingEventPhase::TRAIN:
            return "train";
        case TrainingEventPhase::VALIDATE:
            return "validate";
        case TrainingEventPhase::TEST:
            return "test";
        case TrainingEventPhase::UNKNOWN:
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
                               TrainingEventPhase phase,
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

void emitNativeQueueScheduleTimingDiagnostic(TrainingEventPhase phase,
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

void emitNativeQueueCompletionTimingDiagnostic(TrainingEventPhase phase,
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

void emitNativeQueueSubmitTimingDiagnostic(TrainingEventPhase phase,
                                           uint64_t epoch,
                                           uint64_t batch,
                                           uint64_t slot,
                                           uint64_t inFlight,
                                           uint64_t done,
                                           uint64_t total,
                                           uint64_t submitCalls,
                                           uint64_t bindMicros,
                                           uint64_t submitBatchMicros,
                                           const ThorImplementation::BatchSubmissionTiming& timing) {
    if (!queueDiagnosticsEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "THOR_TRAINING_QUEUE_DIAGNOSTIC native event=submit_timing phase=%s epoch=%lu batch=%lu slot=%lu "
                 "in_flight=%lu done=%lu/%lu submit_calls=%lu bind_us=%lu submit_batch_us=%lu "
                 "active_loss_roots_us=%lu set_active_loss_roots_us=%lu send_batch_us=%lu batch_unwrap_us=%lu "
                 "physical_total_us=%lu input_forward_us=%lu output_collect_us=%lu output_wait_processing_us=%lu "
                 "processing_event_us=%lu input_fanout_us=%lu total_us=%lu inputs=%lu outputs=%lu active_loss_roots=%lu\n",
                 phaseName(phase).c_str(),
                 epoch + 1,
                 batch + 1,
                 slot,
                 inFlight,
                 done,
                 total,
                 submitCalls,
                 bindMicros,
                 submitBatchMicros,
                 timing.activeLossRootsMicros,
                 timing.setActiveLossRootsMicros,
                 timing.sendBatchMicros,
                 timing.batchUnwrapMicros,
                 timing.physicalTotalMicros,
                 timing.inputForwardMicros,
                 timing.outputCollectMicros,
                 timing.outputWaitOnProcessingMicros,
                 timing.processingEventMicros,
                 timing.inputFanoutMicros,
                 timing.totalMicros,
                 timing.numInputs,
                 timing.numOutputs,
                 timing.activeLossRootCount);
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

void ensureNativeQueuedPlanCompatible(const ExecutableTrainingPlan& plan, const Network& network, bool evaluateOnly) {
    if (evaluateOnly) {
        plan.validateNativeQueuedExecutorCompatible({});
        return;
    }
    plan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences(/*trainingEnabledOnly=*/true));
}

std::shared_ptr<TrainingProgram> evaluationOnlyProgramForRequest(const TrainingRunRequest& request) {
    std::vector<std::shared_ptr<TrainingStep>> evaluationSteps;

    if (request.trainingProgram != nullptr) {
        if (!request.trainingProgram->isInitialized()) {
            throw std::runtime_error("Trainer execution received an uninitialized TrainingProgram.");
        }
        evaluationSteps.reserve(request.trainingProgram->getNumSteps());
        for (uint64_t i = 0; i < request.trainingProgram->getNumSteps(); ++i) {
            const TrainingStep& step = request.trainingProgram->getStep(i);
            evaluationSteps.push_back(std::make_shared<TrainingStep>(step.getName(),
                                                                     step.getPhases(),
                                                                     /*optimizer=*/nullptr,
                                                                     std::vector<ParameterReference>{},
                                                                     step.getRepeatCount(),
                                                                     step.getGradientClearPolicy(),
                                                                     step.getInputBindings(),
                                                                     step.isEnabled()));
        }
    } else {
        std::vector<Tensor> lossRoots = request.network->getLossRootTensors();
        if (lossRoots.empty()) {
            throw std::runtime_error("Trainer could not synthesize an evaluation TrainingProgram because the Network has no loss roots.");
        }
        evaluationSteps.push_back(
            std::make_shared<TrainingStep>("default", std::move(lossRoots), /*optimizer=*/nullptr, std::vector<ParameterReference>{}));
    }

    return std::make_shared<TrainingProgram>(std::move(evaluationSteps));
}

std::shared_ptr<TrainingProgram> defaultTrainingProgramForRequest(const TrainingRunRequest& request) {
    if (request.executionMode == TrainingRunExecutionMode::EVALUATE) {
        return evaluationOnlyProgramForRequest(request);
    }

    if (request.trainingProgram != nullptr) {
        if (!request.trainingProgram->isInitialized()) {
            throw std::runtime_error("Trainer execution received an uninitialized TrainingProgram.");
        }
        return request.trainingProgram;
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

    auto defaultStep = std::make_shared<TrainingStep>("default", std::move(lossRoots), optimizer, std::move(parameters));
    return std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{defaultStep});
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

Batch bindBatchInputs(const StepExecutable& step, const Batch& batchInput) {
    Batch bound;
    for (const TrainingInputBinding& binding : step.getResolvedInputBindings()) {
        if (!batchInput.contains(binding.getBatchInputName())) {
            throw std::runtime_error("Training batch is missing input '" + binding.getBatchInputName() + "' required for network input '" +
                                     binding.getNetworkInputName() + "'.");
        }
        const BatchValue& value = batchInput.at(binding.getBatchInputName());
        if (std::holds_alternative<ThorImplementation::Tensor>(value)) {
            bound.insert(binding.getNetworkInputName(), std::get<ThorImplementation::Tensor>(value));
        } else if (std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
            bound.insert(binding.getNetworkInputName(), std::get<ThorImplementation::RaggedTensor>(value));
        } else {
            THOR_UNREACHABLE();
        }
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
            // Timestamp the batch when the completion callback has actually observed the
            // GPU work and required output copies as complete. Throughput must be based
            // on completion times, not on when the consumer thread later pops an already
            // ready slot; otherwise draining a backlog of completed slots can create
            // impossible end-of-epoch rate spikes.
            slot.completionTime = std::chrono::high_resolution_clock::now();
            slot.ready = true;
            state->numBatchesDoneInEpoch += 1;
            inFlightAtComplete = state->inFlightBatches;
            doneAtComplete = state->numBatchesDoneInEpoch;
            totalAtComplete = state->numBatchesInEpoch;
        }
        if (shouldEmitQueueDiagnostic(doneAtComplete)) {
            emitNativeQueueDiagnostic("complete",
                                      params->exampleType == ExampleType::TRAIN ? TrainingEventPhase::TRAIN
                                                                                : params->exampleType == ExampleType::VALIDATE
                                                                                      ? TrainingEventPhase::VALIDATE
                                                                                      : TrainingEventPhase::TEST,
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
                slot.completionTime = std::chrono::high_resolution_clock::now();
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
    std::chrono::high_resolution_clock::time_point completionTime{};
    Batch batchInput;
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
    result.completionTime = slot.completionTime;
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
        throw TrainingInterrupted("Native queued trainer interrupted by SIGINT.");
    }
    if (cancelRequested) {
        throw TrainingCancelled("Native queued trainer was cancelled.");
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
                               ExampleType exampleType,
                               TrainingCancellationToken cancellationToken)
        : placedNetwork(std::move(placedNetwork)),
          loader(std::move(loader)),
          plan(plan),
          options(options),
          state(std::move(state)),
          currentEpoch(currentEpoch),
          batchesPerEpoch(batchesPerEpoch),
          exampleType(exampleType),
          cancellationToken(std::move(cancellationToken)) {}

    void operator()(uint64_t initialEpochBatchNum, uint64_t batches) {
        if (batches == 0) {
            return;
        }

        const TrainingEventPhase diagnosticPhase = exampleType == ExampleType::TRAIN
                                                  ? TrainingEventPhase::TRAIN
                                                  : exampleType == ExampleType::VALIDATE ? TrainingEventPhase::VALIDATE : TrainingEventPhase::TEST;
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
        std::vector<int> stampGpuNums;
        stampGpuNums.reserve(placedNetwork->getNumStamps());
        for (uint64_t stamp = 0; stamp < placedNetwork->getNumStamps(); ++stamp) {
            ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(stamp);
            std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputs = stampedNetwork.getInputs();
            THOR_THROW_IF_FALSE(!inputs.empty());
            stampGpuNums.push_back(inputs[0]->getStream().getGpuNum());
        }

        // Completion must not serialize all queued slots through one stream: the
        // wait/event/host-callback tail can otherwise cap the native queue below
        // max_in_flight.  Assign each queue slot a completion stream from the
        // existing download-stream pool.  getNextDownloadStream already limits
        // stream fanout and round-robins per device, so slots distribute across
        // the same streams used for ordinary output-download work instead of
        // creating private ad-hoc streams here.
        std::vector<Stream> completionStreams;
        completionStreams.reserve(options.maxInFlightBatches);
        for (uint64_t slotIndex = 0; slotIndex < options.maxInFlightBatches; ++slotIndex) {
            const uint64_t stamp = slotIndex % placedNetwork->getNumStamps();
            completionStreams.push_back(Stream::getNextDownloadStream(stampGpuNums[stamp]));
        }
        const bool validationPass = exampleType != ExampleType::TRAIN;
        const std::vector<StepExecutable>& steps = plan.getSteps();

        for (uint64_t batch = 0; batch < batches; ++batch) {
            if (cancellationToken.isCancellationRequested()) {
                requestQueuedTrainingCancellation(state);
                return;
            }
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
                slot.completionTime = {};
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
            uint64_t bindMicros = 0;
            uint64_t submitBatchMicros = 0;
            uint64_t submitCalls = 0;
            ThorImplementation::BatchSubmissionTiming submitTiming;
            for (const StepExecutable& step : steps) {
                for (uint32_t repeat = 0; repeat < step.getRepeatCount(); ++repeat) {
                    const auto bindStart = std::chrono::high_resolution_clock::now();
                    Batch boundBatchInput = bindBatchInputs(step, params->batchInput);
                    const auto bindFinish = std::chrono::high_resolution_clock::now();
                    bindMicros += elapsedMicros(bindStart, bindFinish);
                    params->batchOutput.clear();
                    ThorImplementation::BatchSubmissionTiming singleSubmitTiming;
                    const auto submitBatchStart = std::chrono::high_resolution_clock::now();
                    const bool emitLayerSubmitDiagnostics =
                        ThorImplementation::layerSubmitDiagnosticsEnabled() && shouldEmitQueueDiagnostic(batch + 1);
                    ThorImplementation::ScopedLayerSubmitDiagnosticContext layerSubmitContext(phaseName(diagnosticPhase),
                                                                                              currentEpoch,
                                                                                              epochBatchNum,
                                                                                              slotIndex,
                                                                                              inFlightAfterReserve,
                                                                                              initialEpochBatchNum + batch,
                                                                                              batchesPerEpoch,
                                                                                              validationPass,
                                                                                              emitLayerSubmitDiagnostics);
                    placedNetwork->submitBatch(nextStampToProcess,
                                               boundBatchInput,
                                               params->batchOutput,
                                               outputReadyEvents[nextStampToProcess],
                                               validationPass,
                                               step.getLossRoots(),
                                               &processingFinishedEvents[slotIndex],
                                               /*waitForOutputsOnProcessingStream=*/false,
                                               &singleSubmitTiming);
                    const auto submitBatchFinish = std::chrono::high_resolution_clock::now();
                    submitBatchMicros += elapsedMicros(submitBatchStart, submitBatchFinish);
                    ThorImplementation::accumulateBatchSubmissionTiming(submitTiming, singleSubmitTiming);
                    submitCalls += 1;
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
            Stream completionStream = completionStreams[slotIndex];
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
                emitNativeQueueSubmitTimingDiagnostic(diagnosticPhase,
                                                      currentEpoch,
                                                      epochBatchNum,
                                                      slotIndex,
                                                      inFlightAfterReserve,
                                                      initialEpochBatchNum + batch,
                                                      batchesPerEpoch,
                                                      submitCalls,
                                                      bindMicros,
                                                      submitBatchMicros,
                                                      submitTiming);
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
    TrainingCancellationToken cancellationToken;
};

}  // namespace

void runNativeQueuedTraining(const TrainingRunRequest& request, TrainingObserver& observer, const NativeQueuedTrainingOptions& options) {
    NativeQueuedSigintScope sigintScope;

    THOR_THROW_IF_FALSE(request.network != nullptr);
    THOR_THROW_IF_FALSE(request.loader != nullptr);
    THOR_THROW_IF_FALSE(request.epochs > 0);
    THOR_THROW_IF_FALSE(options.maxInFlightBatches >= 1);
    THOR_THROW_IF_FALSE(request.executionMode == TrainingRunExecutionMode::FIT ||
                        request.executionMode == TrainingRunExecutionMode::EVALUATE);
    request.cancellationToken.throwIfCancellationRequested();

    const bool evaluateOnly = request.executionMode == TrainingRunExecutionMode::EVALUATE;
    std::shared_ptr<TrainingProgram> trainingProgram = defaultTrainingProgramForRequest(request);
    if (!evaluateOnly) {
        attachPlacementFallbackOptimizerIfNeeded(request, *trainingProgram);
    }

    if (evaluateOnly && request.evaluationPhase == TrainingEventPhase::UNKNOWN) {
        throw std::runtime_error("Trainer evaluation requires a concrete evaluation phase.");
    }

    const uint64_t batchSize = request.loader->getBatchSize();
    std::vector<Event> initDoneEvents;
    request.cancellationToken.throwIfCancellationRequested();
    std::shared_ptr<PlacedNetwork> placedNetwork = request.network->place(batchSize, initDoneEvents, /*inferenceOnly=*/evaluateOnly);
    THOR_THROW_IF_FALSE(placedNetwork->getNumStamps() == 1);
    for (size_t i = 0; i < initDoneEvents.size(); ++i) {
        request.cancellationToken.throwIfCancellationRequested();
        initDoneEvents[i].synchronize();
    }

    request.cancellationToken.throwIfCancellationRequested();
    ExecutableTrainingPlan plan = ExecutableTrainingPlan::compile(*trainingProgram, *placedNetwork, /*resolveEmptyUpdateParametersAsAllTrainable=*/!evaluateOnly);
    ensureNativeQueuedPlanCompatible(plan, *request.network, evaluateOnly);

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

    auto makeBaseSnapshot = [&](TrainingEventPhase phase,
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
                          TrainingEvent::runStarted(makeBaseSnapshot(TrainingEventPhase::UNKNOWN, 0, batchSize, 0, nullptr)));
    }

    for (uint32_t epochOffset = 0; epochOffset < request.epochs; ++epochOffset) {
        request.cancellationToken.throwIfCancellationRequested();
        const uint64_t humanEpoch = currentEpoch + 1;

        std::vector<std::pair<ExampleType, TrainingEventPhase>> phaseSpecs;
        if (evaluateOnly) {
            phaseSpecs.emplace_back(request.evaluationExampleType, request.evaluationPhase);
        } else {
            phaseSpecs.emplace_back(ExampleType::TRAIN, TrainingEventPhase::TRAIN);
            phaseSpecs.emplace_back(ExampleType::VALIDATE, TrainingEventPhase::VALIDATE);
        }

        for (const auto& phaseSpec : phaseSpecs) {
            request.cancellationToken.throwIfCancellationRequested();
            const ExampleType exampleType = phaseSpec.first;
            const TrainingEventPhase phase = phaseSpec.second;
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
                placedNetwork, request.loader, plan, options, state, currentEpoch, batchesPerEpoch, exampleType, request.cancellationToken);
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

            auto requestExternalCancel = [&]() {
                if (request.cancellationToken.isCancellationRequested()) {
                    requestQueuedTrainingCancellation(state);
                }
                if (sigintScope.interrupted()) {
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        state->cancelRequested = true;
                        state->interruptRequested = true;
                    }
                    state->batchFinished.notify_all();
                    state->batchPopped.notify_all();
                }
            };

            std::chrono::high_resolution_clock::time_point previousBatchCompletionTime{};
            bool havePreviousBatchCompletionTime = false;
            bool haveCompletedBatchTimingInPhase = false;
            try {
                while (true) {
                    requestExternalCancel();
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
                        double& averageBatchTime = (phase == TrainingEventPhase::TRAIN) ? averageTrainingBatchTime : averageValidationBatchTime;
                        double completedBatchTime = -1.0;
                        if (havePreviousBatchCompletionTime && completedBatch.completionTime > previousBatchCompletionTime) {
                            const std::chrono::duration<double> elapsed =
                                std::chrono::duration_cast<std::chrono::duration<double>>(completedBatch.completionTime - previousBatchCompletionTime);
                            completedBatchTime = elapsed.count();
                        }
                        previousBatchCompletionTime = completedBatch.completionTime;
                        havePreviousBatchCompletionTime = true;

                        double batchTimeForStats = completedBatchTime;
                        if (completedBatchTime > 0.0) {
                            if (haveCompletedBatchTimingInPhase) {
                                if (averageBatchTime < 0.0) {
                                    averageBatchTime = completedBatchTime;
                                } else {
                                    averageBatchTime = 0.25 * completedBatchTime + 0.75 * averageBatchTime;
                                }
                                batchTimeForStats = averageBatchTime;
                            } else if (averageBatchTime > 0.0) {
                                // The first completion in a phase includes queue fill and any loader/startup
                                // latency before the pipeline reaches steady state. Use the previous phase EMA
                                // for the current line, but do not let the first interval poison the EMA.
                                batchTimeForStats = averageBatchTime;
                            }
                            haveCompletedBatchTimingInPhase = true;
                        } else if (averageBatchTime > 0.0) {
                            batchTimeForStats = averageBatchTime;
                        }

                        TrainingStatsSnapshot snapshot = makeBaseSnapshot(phase, humanEpoch, batchSize, batchesPerEpoch, state);
                        snapshot.stepInEpoch = batchNum + 1;
                        snapshot.step = (currentEpoch * batchesPerEpoch) + snapshot.stepInEpoch;
                        snapshot.samplesProcessed = snapshot.step * batchSize;
                        if (batchTimeForStats > 0.0) {
                            snapshot.samplesPerSecond = static_cast<double>(batchSize) / batchTimeForStats;
                            snapshot.batchesPerSecond = 1.0 / batchTimeForStats;
                            snapshot.floatingPointOperationsPerBatch =
                                (phase == TrainingEventPhase::TRAIN) ? trainingFlopsPerBatch : forwardFlopsPerBatch;
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

    request.cancellationToken.throwIfCancellationRequested();
    if (!evaluateOnly && request.saveModelDirectory.has_value()) {
        placedNetwork->save(*request.saveModelDirectory, request.saveModelOverwrite, request.saveOptimizerState);
    }

    if (request.runtime.statsEnabled) {
        emitTrainingEvent(observer,
                          request.runtime.statsEnabled,
                          TrainingEvent::runFinished(makeBaseSnapshot(TrainingEventPhase::UNKNOWN, currentEpoch, batchSize, 0, nullptr)));
    }
}

}  // namespace Thor
