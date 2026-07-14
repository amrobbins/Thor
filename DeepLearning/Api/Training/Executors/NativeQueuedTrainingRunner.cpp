#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"

#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Api/Training/ExecutableTrainingPlan.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetStorageSelection.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace Thor {

namespace {

struct QueuedTrainingState;

struct ScalarStatSlot {
    bool present = false;
    float value = 0.0f;
};

enum class ScalarStatSource {
    UNRESOLVED,
    INPUT,
    OUTPUT
};

struct NativeBatchCompletionParams {
    std::shared_ptr<QueuedTrainingState> state;
    bool completionCallbackLaunched = false;
    bool completionCallbackFinished = false;
    ExampleType exampleType = ExampleType::TRAIN;
    TrainingEventPhase phase = TrainingEventPhase::TRAIN;
    uint64_t currentEpoch = 0;
    uint64_t epochBatchNum = 0;
    uint64_t batchesInEpoch = 0;
    uint64_t slotIndex = 0;
    BatchLease batchLease;
    std::map<std::string, ThorImplementation::Tensor> batchOutput;
    std::vector<ScalarStatSlot> scalarStats;
    std::vector<ScalarStatSource> scalarStatSources;
};

struct QueuedBatchSlot {
    bool occupied = false;
    bool ready = false;
    TrainingEventPhase phase = TrainingEventPhase::TRAIN;
    uint64_t epochBatchNum = 0;
    uint64_t batchesInEpoch = 0;
    uint64_t doneInEpochAtComplete = 0;
    uint64_t paramsIndex = 0;
    std::chrono::high_resolution_clock::time_point completionTime{};
    std::vector<ScalarStatSlot> scalarStats;
};

struct QueuedPhaseProgress {
    uint64_t completedBatches = 0;
    uint64_t poppedBatches = 0;
};

struct QueuedTrainingState {
    QueuedTrainingState(uint64_t maxInFlightBatches,
                        std::vector<std::string> scalarTensorNames,
                        std::vector<std::string> aggregateLossTensorNames)
        : scalarTensorNames(std::move(scalarTensorNames)),
          aggregateLossTensorNames(std::move(aggregateLossTensorNames)),
          slots(maxInFlightBatches),
          completionParams(maxInFlightBatches) {
        THOR_THROW_IF_FALSE(maxInFlightBatches >= 1);
        for (QueuedBatchSlot& slot : slots) {
            slot.scalarStats.resize(this->scalarTensorNames.size());
        }
        for (NativeBatchCompletionParams& params : completionParams) {
            params.scalarStats.resize(this->scalarTensorNames.size());
            params.scalarStatSources.resize(
                this->scalarTensorNames.size(),
                ScalarStatSource::UNRESOLVED);
        }
    }

    std::mutex mutex;
    std::condition_variable batchFinished;
    std::condition_variable batchPopped;

    std::vector<std::string> scalarTensorNames;
    std::vector<std::string> aggregateLossTensorNames;
    std::array<QueuedPhaseProgress, 4> phaseProgress{};
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
struct WallThroughputEmaState {
    bool initialized = false;
    double lastElapsedSeconds = 0.0;
    uint64_t lastCompletedBatches = 0;
    double samplesPerSecond = 0.0;
    double batchesPerSecond = 0.0;
    double floatingPointOperationsPerSecond = 0.0;
};

constexpr double WALL_THROUGHPUT_EMA_ALPHA = 0.25;
constexpr double WALL_THROUGHPUT_EMA_MIN_INTERVAL_SECONDS = 0.25;

void updateWallThroughputRates(TrainingStatsSnapshot& snapshot,
                               WallThroughputEmaState& state,
                               uint64_t completedBatches,
                               uint64_t batchSize,
                               uint64_t floatingPointOperationsPerBatch) {
    snapshot.floatingPointOperationsPerBatch = floatingPointOperationsPerBatch;
    if (snapshot.elapsedSeconds <= 0.0 || completedBatches == 0 || batchSize == 0) {
        return;
    }

    auto assignRates = [&](double batchesPerSecond,
                           double samplesPerSecond,
                           double floatingPointOperationsPerSecond) {
        snapshot.batchesPerSecond = batchesPerSecond;
        snapshot.samplesPerSecond = samplesPerSecond;
        snapshot.floatingPointOperationsPerSecond = floatingPointOperationsPerSecond;
    };

    if (!state.initialized) {
        const double batchesPerSecond = static_cast<double>(completedBatches) / snapshot.elapsedSeconds;
        const double samplesPerSecond = (static_cast<double>(completedBatches) * static_cast<double>(batchSize)) /
                                        snapshot.elapsedSeconds;
        const double floatingPointOperationsPerSecond =
            (static_cast<double>(completedBatches) * static_cast<double>(floatingPointOperationsPerBatch)) /
            snapshot.elapsedSeconds;
        state.initialized = true;
        state.lastElapsedSeconds = snapshot.elapsedSeconds;
        state.lastCompletedBatches = completedBatches;
        state.batchesPerSecond = batchesPerSecond;
        state.samplesPerSecond = samplesPerSecond;
        state.floatingPointOperationsPerSecond = floatingPointOperationsPerSecond;
        assignRates(batchesPerSecond, samplesPerSecond, floatingPointOperationsPerSecond);
        return;
    }

    if (snapshot.elapsedSeconds <= state.lastElapsedSeconds || completedBatches <= state.lastCompletedBatches) {
        assignRates(state.batchesPerSecond, state.samplesPerSecond, state.floatingPointOperationsPerSecond);
        return;
    }

    const double intervalSeconds = snapshot.elapsedSeconds - state.lastElapsedSeconds;
    if (intervalSeconds < WALL_THROUGHPUT_EMA_MIN_INTERVAL_SECONDS) {
        // Completion callbacks can leave several finished batches queued for the
        // host to pop back-to-back.  Updating the wall-clock EMA on those tiny
        // dequeue intervals turns a short drain burst into a visible throughput
        // spike.  Keep accumulating progress until there is a meaningful wall
        // interval; the eventual update uses the full elapsed/progress delta.
        assignRates(state.batchesPerSecond, state.samplesPerSecond, state.floatingPointOperationsPerSecond);
        return;
    }

    const double intervalBatches = static_cast<double>(completedBatches - state.lastCompletedBatches);
    const double intervalBatchesPerSecond = intervalBatches / intervalSeconds;
    const double intervalSamplesPerSecond = (intervalBatches * static_cast<double>(batchSize)) / intervalSeconds;
    const double intervalFloatingPointOperationsPerSecond =
        (intervalBatches * static_cast<double>(floatingPointOperationsPerBatch)) / intervalSeconds;

    state.batchesPerSecond = (WALL_THROUGHPUT_EMA_ALPHA * intervalBatchesPerSecond) +
                             ((1.0 - WALL_THROUGHPUT_EMA_ALPHA) * state.batchesPerSecond);
    state.samplesPerSecond = (WALL_THROUGHPUT_EMA_ALPHA * intervalSamplesPerSecond) +
                             ((1.0 - WALL_THROUGHPUT_EMA_ALPHA) * state.samplesPerSecond);
    state.floatingPointOperationsPerSecond =
        (WALL_THROUGHPUT_EMA_ALPHA * intervalFloatingPointOperationsPerSecond) +
        ((1.0 - WALL_THROUGHPUT_EMA_ALPHA) * state.floatingPointOperationsPerSecond);

    state.lastElapsedSeconds = snapshot.elapsedSeconds;
    state.lastCompletedBatches = completedBatches;
    assignRates(state.batchesPerSecond, state.samplesPerSecond, state.floatingPointOperationsPerSecond);
}


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

std::set<std::string> networkOutputNames(Network& network) {
    std::set<std::string> names;
    const uint32_t numLayers = network.getNumLayers();
    for (uint32_t i = 0; i < numLayers; ++i) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(network.getLayer(i));
        if (output != nullptr) {
            names.insert(output->getName());
        }
    }
    return names;
}

std::vector<std::string> outputBackedReportableLossNames(Network& network) {
    const std::set<std::string> outputs = networkOutputNames(network);
    std::set<std::string> lossNames;
    for (const NetworkLossReference& reference : network.getReportableLosses()) {
        if (outputs.count(reference.lossName) != 0) {
            lossNames.insert(reference.lossName);
        }
    }
    return std::vector<std::string>(lossNames.begin(), lossNames.end());
}

std::set<std::string> setFromVector(const std::vector<std::string>& values) {
    return std::set<std::string>(values.begin(), values.end());
}

std::vector<std::string> plainTrainingProgramAggregateLossNames(Network& network) {
    // Explicit and implicit TrainingStep programs are resolved to a regular active graph before placement.
    // All output-backed graph losses in that graph remain reportable for aggregate loss/stat purposes.
    return outputBackedReportableLossNames(network);
}

bool isRuntimeScalarName(const std::string& name) {
    return name == "loss" || name == "learning_rate" || name == "learningRate" || name == "lr" || name == "momentum";
}

void filterRuntimeScalarsToActiveTrainingProgramOutputs(TrainingRuntimeConfig& runtime,
                                                        Network& network,
                                                        const std::vector<std::string>& activeAggregateLossTensorNames) {
    const std::set<std::string> allOutputBackedLossNames = setFromVector(outputBackedReportableLossNames(network));
    const std::set<std::string> activeOutputBackedLossNames = setFromVector(activeAggregateLossTensorNames);

    for (auto it = runtime.scalarTensorsToReport.begin(); it != runtime.scalarTensorsToReport.end();) {
        const std::string& name = *it;
        const bool isReportableLoss = allOutputBackedLossNames.count(name) != 0;
        if (isReportableLoss && activeOutputBackedLossNames.count(name) == 0) {
            it = runtime.scalarTensorsToReport.erase(it);
            continue;
        }

        ++it;
    }
}

void filterRuntimeScalarsToExistingExecutionOutputs(TrainingRuntimeConfig& runtime, Network& network) {
    const std::set<std::string> outputs = networkOutputNames(network);
    for (auto it = runtime.scalarTensorsToReport.begin(); it != runtime.scalarTensorsToReport.end();) {
        const std::string& name = *it;
        if (isRuntimeScalarName(name) || outputs.count(name) != 0) {
            ++it;
            continue;
        }
        it = runtime.scalarTensorsToReport.erase(it);
    }
}

bool outputNameExists(Network& network, const std::string& name) {
    return networkOutputNames(network).count(name) != 0;
}

float copyAggregateLossStatTensor(const std::map<std::string, ThorImplementation::Tensor>& batchOutput,
                                  const std::vector<std::string>& aggregateLossTensorNames) {
    if (aggregateLossTensorNames.empty()) {
        throw std::runtime_error("Requested aggregate training stat tensor 'loss', but the graph has no output-backed reportable losses.");
    }

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    double sum = 0.0;
    for (const std::string& lossTensorName : aggregateLossTensorNames) {
        auto outputIt = batchOutput.find(lossTensorName);
        if (outputIt == batchOutput.end()) {
            throw std::runtime_error("Requested aggregate training stat tensor 'loss', but reportable graph loss '" + lossTensorName +
                                     "' was not present in batch outputs.");
        }
        const ThorImplementation::Tensor& copyFromTensor = outputIt->second;
        THOR_THROW_IF_FALSE(copyFromTensor.getPlacement() == cpuPlacement);
        THOR_THROW_IF_FALSE(copyFromTensor.getDescriptor().getArraySizeInBytes() >= sizeof(float));
        float value = 0.0f;
        std::memcpy(&value, copyFromTensor.getMemPtr(), sizeof(float));
        sum += static_cast<double>(value);
    }
    return static_cast<float>(sum);
}

float copyCpuScalarTensor(const ThorImplementation::Tensor& copyFromTensor) {
    ThorImplementation::TensorPlacement cpuPlacement(
        ThorImplementation::TensorPlacement::MemDevices::CPU);
    THOR_THROW_IF_FALSE(copyFromTensor.getPlacement() == cpuPlacement);
    THOR_THROW_IF_FALSE(
        copyFromTensor.getDescriptor().getArraySizeInBytes() >= sizeof(float));
    float value = 0.0f;
    std::memcpy(&value, copyFromTensor.getMemPtr(), sizeof(float));
    return value;
}

float copyInputScalarStatTensor(
    const Batch& batchInput,
    const std::string& tensorName) {
    if (!batchInput.contains(tensorName)) {
        throw std::runtime_error(
            "Requested input training stat tensor '" + tensorName +
            "' was not present in batch inputs.");
    }
    if (!batchInput.isTensor(tensorName)) {
        throw std::runtime_error(
            "Requested input training stat tensor '" + tensorName +
            "' is not a materialized dense tensor.");
    }
    return copyCpuScalarTensor(batchInput.getTensor(tensorName));
}

float copyOutputScalarStatTensor(
    const std::map<std::string, ThorImplementation::Tensor>& batchOutput,
    const std::string& tensorName,
    const std::vector<std::string>& aggregateLossTensorNames) {
    auto outputIt = batchOutput.find(tensorName);
    if (tensorName == "loss" && !aggregateLossTensorNames.empty()) {
        return copyAggregateLossStatTensor(batchOutput, aggregateLossTensorNames);
    }
    if (outputIt != batchOutput.end()) {
        return copyCpuScalarTensor(outputIt->second);
    }
    if (tensorName == "loss") {
        return copyAggregateLossStatTensor(batchOutput, aggregateLossTensorNames);
    }
    throw std::runtime_error(
        "Requested output training stat tensor '" + tensorName +
        "' was not present in batch outputs.");
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

size_t queuedPhaseIndex(TrainingEventPhase phase) {
    const size_t index = static_cast<size_t>(phase);
    THOR_THROW_IF_FALSE(index < 4);
    return index;
}

QueuedPhaseProgress& phaseProgress(QueuedTrainingState& state, TrainingEventPhase phase) {
    return state.phaseProgress[queuedPhaseIndex(phase)];
}

const QueuedPhaseProgress& phaseProgress(const QueuedTrainingState& state, TrainingEventPhase phase) {
    return state.phaseProgress[queuedPhaseIndex(phase)];
}

#if THOR_ENABLE_TRAINING_QUEUE_DIAGNOSTICS
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

using DiagnosticTimePoint = std::chrono::high_resolution_clock::time_point;

uint64_t elapsedMicros(DiagnosticTimePoint start, DiagnosticTimePoint finish) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
}

DiagnosticTimePoint diagnosticNow(bool enabled) {
    return enabled ? std::chrono::high_resolution_clock::now() : DiagnosticTimePoint{};
}
#else
constexpr bool queueDiagnosticsEnabled() { return false; }
constexpr bool shouldEmitQueueDiagnostic(uint64_t, uint64_t = 0) { return false; }
struct DiagnosticTimePoint {};
constexpr uint64_t elapsedMicros(DiagnosticTimePoint, DiagnosticTimePoint) { return 0; }
constexpr DiagnosticTimePoint diagnosticNow(bool) { return {}; }
#endif

bool gpuSubmitCoordinatorEnabled() {
    const char* enabled = std::getenv("THOR_TRAINING_GPU_SUBMIT_COORDINATOR");

    // Enabled by default. Disable explicitly with:
    //   THOR_TRAINING_GPU_SUBMIT_COORDINATOR=0
    return enabled == nullptr || enabled[0] == '\0' || !(enabled[0] == '0' && enabled[1] == '\0');
}

struct GpuSubmitCoordinatorTiming {
    uint64_t queueWaitMicros = 0;
    uint64_t setGpuMicros = 0;
    uint64_t execMicros = 0;
};

class GpuSubmitCoordinator {
   public:
    explicit GpuSubmitCoordinator(int gpuNum) : gpuNum(gpuNum), worker(&GpuSubmitCoordinator::workerLoop, this) {}

    ~GpuSubmitCoordinator() { stop(); }

    GpuSubmitCoordinator(const GpuSubmitCoordinator&) = delete;
    GpuSubmitCoordinator& operator=(const GpuSubmitCoordinator&) = delete;

    template <typename Fn>
    auto submit(Fn&& fn, GpuSubmitCoordinatorTiming* timing = nullptr) -> std::future<std::invoke_result_t<Fn>> {
        using Result = std::invoke_result_t<Fn>;

        const auto enqueuedAt = diagnosticNow(timing != nullptr);
        auto task =
            std::make_shared<std::packaged_task<Result()>>([this, fn = std::forward<Fn>(fn), timing, enqueuedAt]() mutable -> Result {
                const auto startedAt = diagnosticNow(timing != nullptr);
                if (timing != nullptr) {
                    timing->queueWaitMicros = elapsedMicros(enqueuedAt, startedAt);
                }
                ScopedGpu scopedGpu(gpuNum);
                const auto execStartedAt = diagnosticNow(timing != nullptr);
                if (timing != nullptr) {
                    timing->setGpuMicros = elapsedMicros(startedAt, execStartedAt);
                }

                if constexpr (std::is_void_v<Result>) {
                    try {
                        std::invoke(fn);
                    } catch (...) {
                        const auto finishedAt = diagnosticNow(timing != nullptr);
                        if (timing != nullptr) {
                            timing->execMicros = elapsedMicros(execStartedAt, finishedAt);
                        }
                        throw;
                    }

                    const auto finishedAt = diagnosticNow(timing != nullptr);
                    if (timing != nullptr) {
                        timing->execMicros = elapsedMicros(execStartedAt, finishedAt);
                    }
                } else {
                    try {
                        Result result = std::invoke(fn);
                        const auto finishedAt = diagnosticNow(timing != nullptr);
                        if (timing != nullptr) {
                            timing->execMicros = elapsedMicros(execStartedAt, finishedAt);
                        }
                        return result;
                    } catch (...) {
                        const auto finishedAt = diagnosticNow(timing != nullptr);
                        if (timing != nullptr) {
                            timing->execMicros = elapsedMicros(execStartedAt, finishedAt);
                        }
                        throw;
                    }
                }
            });

        std::future<Result> future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                throw std::runtime_error("GpuSubmitCoordinator is stopping.");
            }
            queue.emplace_back([task]() { (*task)(); });
        }
        cv.notify_one();
        return future;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                return;
            }
            stopping = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

   private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this]() { return stopping || !queue.empty(); });
                if (queue.empty() && stopping) {
                    break;
                }
                task = std::move(queue.front());
                queue.pop_front();
            }
            task();
        }
    }

    int gpuNum;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> queue;
    bool stopping = false;
};

class GpuSubmitCoordinatorRegistry {
   public:
    static GpuSubmitCoordinator& get(int gpuNum) {
        static GpuSubmitCoordinatorRegistry registry;
        return registry.getCoordinator(gpuNum);
    }

   private:
    GpuSubmitCoordinator& getCoordinator(int gpuNum) {
        std::lock_guard<std::mutex> lock(mutex);
        auto& coordinator = coordinators[gpuNum];
        if (coordinator == nullptr) {
            coordinator = std::make_unique<GpuSubmitCoordinator>(gpuNum);
        }
        return *coordinator;
    }

    std::mutex mutex;
    std::unordered_map<int, std::unique_ptr<GpuSubmitCoordinator>> coordinators;
};

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
    std::fprintf(
        stderr,
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
    std::fprintf(
        stderr,
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
                                           const ThorImplementation::BatchSubmissionTiming& timing,
                                           bool usedGpuSubmitCoordinator = false,
                                           uint64_t coordinatorQueueWaitMicros = 0,
                                           uint64_t coordinatorSetGpuMicros = 0,
                                           uint64_t coordinatorExecMicros = 0,
                                           uint64_t coordinatorRoundtripMicros = 0) {
    if (!queueDiagnosticsEnabled()) {
        return;
    }

    if (!usedGpuSubmitCoordinator) {
        std::fprintf(stderr,
                     "THOR_TRAINING_QUEUE_DIAGNOSTIC native event=submit_timing phase=%s epoch=%lu batch=%lu slot=%lu "
                     "in_flight=%lu done=%lu/%lu submit_calls=%lu bind_us=%lu submit_batch_us=%lu "
                     "active_objective_roots_us=%lu set_active_objective_roots_us=%lu send_batch_us=%lu batch_unwrap_us=%lu "
                     "physical_total_us=%lu input_forward_us=%lu output_collect_us=%lu output_wait_processing_us=%lu "
                     "processing_event_us=%lu input_fanout_us=%lu total_us=%lu inputs=%lu outputs=%lu active_objective_roots=%lu\n",
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
                     timing.activeObjectiveRootsMicros,
                     timing.setActiveObjectiveRootsMicros,
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
                     timing.activeObjectiveRootCount);
    } else {
        std::fprintf(stderr,
                     "THOR_TRAINING_QUEUE_DIAGNOSTIC native event=submit_timing phase=%s epoch=%lu batch=%lu slot=%lu "
                     "in_flight=%lu done=%lu/%lu submit_calls=%lu bind_us=%lu submit_batch_us=%lu "
                     "active_objective_roots_us=%lu set_active_objective_roots_us=%lu send_batch_us=%lu batch_unwrap_us=%lu "
                     "physical_total_us=%lu input_forward_us=%lu output_collect_us=%lu output_wait_processing_us=%lu "
                     "processing_event_us=%lu input_fanout_us=%lu total_us=%lu inputs=%lu outputs=%lu active_objective_roots=%lu "
                     "gpu_submit_coord=1 coord_queue_wait_us=%lu coord_set_gpu_us=%lu coord_exec_us=%lu "
                     "coord_roundtrip_us=%lu\n",
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
                     timing.activeObjectiveRootsMicros,
                     timing.setActiveObjectiveRootsMicros,
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
                     timing.activeObjectiveRootCount,
                     coordinatorQueueWaitMicros,
                     coordinatorSetGpuMicros,
                     coordinatorExecMicros,
                     coordinatorRoundtripMicros);
    }
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

class EpochLossAccumulator {
   public:
    void update(const TrainingStatsSnapshot& snapshot) {
        if (snapshot.phase == TrainingEventPhase::TRAIN) {
            train.update(snapshot);
        } else if (snapshot.phase == TrainingEventPhase::VALIDATE) {
            validate.update(snapshot);
        } else if (snapshot.phase == TrainingEventPhase::TEST) {
            test.update(snapshot);
        }
    }

    [[nodiscard]] std::optional<double> trainLoss() const { return train.snapshot().loss; }
    [[nodiscard]] std::optional<double> validationLoss() const { return validate.snapshot().loss; }

    [[nodiscard]] TrainingModelSelectionContext modelSelectionContext(uint64_t epoch) const {
        TrainingModelSelectionContext context;
        context.epoch = epoch;
        context.train = train.snapshot();
        context.validate = validate.snapshot();
        context.test = test.snapshot();
        return context;
    }

   private:
    struct RunningMean {
        double sum = 0.0;
        uint64_t count = 0;

        void add(double value) {
            sum += value;
            count += 1;
        }

        [[nodiscard]] std::optional<double> mean() const {
            if (count == 0) {
                return std::nullopt;
            }
            return sum / static_cast<double>(count);
        }
    };

    struct PhaseAccumulator {
        RunningMean lossMean{};
        std::unordered_map<std::string, RunningMean> lossMeans{};
        std::unordered_map<std::string, RunningMean> metricMeans{};

        void update(const TrainingStatsSnapshot& snapshot) {
            if (snapshot.loss.has_value()) {
                lossMean.add(snapshot.loss.value());
            }
            for (const auto& [name, value] : snapshot.losses) {
                lossMeans[name].add(value);
            }
            for (const auto& [name, value] : snapshot.metrics) {
                metricMeans[name].add(value);
            }
        }

        [[nodiscard]] TrainingModelSelectionPhaseStats snapshot() const {
            TrainingModelSelectionPhaseStats out;
            out.loss = lossMean.mean();
            for (const auto& [name, mean] : lossMeans) {
                const std::optional<double> value = mean.mean();
                if (value.has_value()) {
                    out.losses[name] = value.value();
                }
            }
            for (const auto& [name, mean] : metricMeans) {
                const std::optional<double> value = mean.mean();
                if (value.has_value()) {
                    out.metrics[name] = value.value();
                }
            }
            return out;
        }
    };

    PhaseAccumulator train{};
    PhaseAccumulator validate{};
    PhaseAccumulator test{};
};

struct TrainingSelectionMetadata {
    std::optional<uint64_t> bestEpoch{};
    std::optional<double> bestScore{};
    uint64_t latestEpoch = 0;
    std::optional<double> latestScore{};
    std::optional<double> latestTrainingLoss{};
    std::optional<double> latestValidationLoss{};
    uint64_t completedEpoch = 0;
    std::string completionReason = "completed";
    uint32_t checkBestModelEveryEpochs = 0;
    uint64_t firstModelSelectionEpoch = 0;
};

class TrainingArtifactManager {
   public:
    TrainingArtifactManager(std::optional<std::string> saveModelDirectory, bool overwrite)
        : saveModelDirectory(std::move(saveModelDirectory)), overwrite(overwrite) {}

    ~TrainingArtifactManager() {
        if (bestCandidateDirectory.has_value()) {
            std::error_code errorCode;
            std::filesystem::remove_all(bestCandidateDirectory.value(), errorCode);
        }
    }

    [[nodiscard]] bool enabled() const { return saveModelDirectory.has_value(); }

    void maybeSnapshotBestCandidate(PlacedNetwork& placedNetwork, uint64_t epoch, std::optional<double> score) {
        if (!score.has_value() || !std::isfinite(score.value())) {
            return;
        }
        if (bestScore.has_value() && score.value() >= bestScore.value()) {
            return;
        }

        bestScore = score.value();
        bestEpoch = epoch;
        if (!enabled()) {
            return;
        }

        const std::filesystem::path newCandidate = uniqueCandidateDirectory(epoch);
        const std::filesystem::path tmpCandidate = uniqueTemporaryDirectory(epoch);
        removePathIfExists(tmpCandidate);
        removePathIfExists(newCandidate);

        try {
            placedNetwork.save(tmpCandidate.string(), /*overwrite=*/true, /*saveOptimizerState=*/true);
            std::filesystem::rename(tmpCandidate, newCandidate);
        } catch (...) {
            removePathIfExists(tmpCandidate);
            removePathIfExists(newCandidate);
            throw;
        }

        if (bestCandidateDirectory.has_value()) {
            removePathIfExists(bestCandidateDirectory.value());
        }
        bestCandidateDirectory = newCandidate;
    }

    void finalize(PlacedNetwork& placedNetwork,
                  const TrainingSelectionMetadata& metadata,
                  bool persistLatestArtifact) {
        if (!enabled()) {
            return;
        }
        if (!persistLatestArtifact && !bestCandidateDirectory.has_value()) {
            throw std::runtime_error(
                "Training artifact finalization requires either a latest artifact or a persisted best candidate.");
        }

        const std::filesystem::path artifactRoot = std::filesystem::path(saveModelDirectory.value());
        if (std::filesystem::exists(artifactRoot) && !overwrite) {
            throw std::runtime_error("Training artifact cannot replace existing save_model_dir '" + artifactRoot.string() +
                                     "' because save_model_overwrite is false.");
        }

        const std::filesystem::path finalTemporaryDirectory = uniqueFinalTemporaryDirectory();
        const std::filesystem::path replacementBackupDirectory = uniqueReplacementBackupDirectory();
        removePathIfExists(finalTemporaryDirectory);
        removePathIfExists(replacementBackupDirectory);

        try {
            std::filesystem::create_directories(finalTemporaryDirectory);

            if (persistLatestArtifact) {
                const std::filesystem::path latestDirectory = finalTemporaryDirectory / "latest";
                const std::filesystem::path latestTemporaryDirectory = finalTemporaryDirectory / ".latest.tmp";
                removePathIfExists(latestTemporaryDirectory);
                placedNetwork.save(latestTemporaryDirectory.string(), /*overwrite=*/true, /*saveOptimizerState=*/true);
                std::filesystem::rename(latestTemporaryDirectory, latestDirectory);
            }

            if (bestCandidateDirectory.has_value()) {
                const std::filesystem::path bestDirectory = finalTemporaryDirectory / "best";
                removePathIfExists(bestDirectory);
                std::filesystem::rename(bestCandidateDirectory.value(), bestDirectory);
                bestCandidateDirectory.reset();
            }

            writeSelectionMetadata(finalTemporaryDirectory, metadata);
            replaceArtifactRoot(finalTemporaryDirectory, artifactRoot, replacementBackupDirectory);
        } catch (...) {
            restoreArtifactRootIfNeeded(artifactRoot, replacementBackupDirectory);
            removePathIfExists(finalTemporaryDirectory);
            removePathIfExists(replacementBackupDirectory);
            throw;
        }
    }

    [[nodiscard]] std::optional<double> getBestScore() const { return bestScore; }
    [[nodiscard]] std::optional<uint64_t> getBestEpoch() const { return bestEpoch; }
    [[nodiscard]] bool hasBestCandidateArtifact() const { return bestCandidateDirectory.has_value(); }

   private:
    static void removePathIfExists(const std::filesystem::path& path) {
        std::error_code errorCode;
        if (!std::filesystem::exists(path, errorCode) && !errorCode) {
            return;
        }
        errorCode.clear();
        std::filesystem::remove_all(path, errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to remove path '" + path.string() + "': " + errorCode.message());
        }
    }

    [[nodiscard]] std::filesystem::path baseCandidatePrefix() const {
        std::filesystem::path finalDirectory(saveModelDirectory.value());
        std::filesystem::path parent = finalDirectory.parent_path();
        std::string filename = finalDirectory.filename().string();
        if (filename.empty()) {
            filename = "model";
        }
        std::ostringstream out;
        out << "." << filename << ".best_candidate." << reinterpret_cast<uintptr_t>(this);
        return parent / out.str();
    }

    [[nodiscard]] std::filesystem::path uniqueCandidateDirectory(uint64_t epoch) const {
        std::ostringstream out;
        out << baseCandidatePrefix().string() << ".epoch_" << epoch;
        return std::filesystem::path(out.str());
    }

    [[nodiscard]] std::filesystem::path uniqueTemporaryDirectory(uint64_t epoch) const {
        std::ostringstream out;
        out << baseCandidatePrefix().string() << ".epoch_" << epoch << ".tmp";
        return std::filesystem::path(out.str());
    }

    [[nodiscard]] std::filesystem::path uniqueFinalTemporaryDirectory() const {
        std::ostringstream out;
        out << baseCandidatePrefix().string() << ".final.tmp";
        return std::filesystem::path(out.str());
    }

    [[nodiscard]] std::filesystem::path uniqueReplacementBackupDirectory() const {
        std::ostringstream out;
        out << baseCandidatePrefix().string() << ".previous";
        return std::filesystem::path(out.str());
    }

    static void replaceArtifactRoot(const std::filesystem::path& finalTemporaryDirectory,
                                    const std::filesystem::path& artifactRoot,
                                    const std::filesystem::path& replacementBackupDirectory) {
        std::error_code errorCode;
        const bool hadPreviousRoot = std::filesystem::exists(artifactRoot, errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to inspect training artifact root '" + artifactRoot.string() + "': " + errorCode.message());
        }

        if (hadPreviousRoot) {
            std::filesystem::rename(artifactRoot, replacementBackupDirectory, errorCode);
            if (errorCode) {
                throw std::runtime_error("Failed to stage replacement of training artifact root '" + artifactRoot.string() +
                                         "': " + errorCode.message());
            }
        }

        errorCode.clear();
        std::filesystem::rename(finalTemporaryDirectory, artifactRoot, errorCode);
        if (errorCode) {
            restoreArtifactRootIfNeeded(artifactRoot, replacementBackupDirectory);
            throw std::runtime_error("Failed to finalize training artifact root '" + artifactRoot.string() + "': " + errorCode.message());
        }

        if (hadPreviousRoot) {
            removePathIfExists(replacementBackupDirectory);
        }
    }

    static void restoreArtifactRootIfNeeded(const std::filesystem::path& artifactRoot,
                                            const std::filesystem::path& replacementBackupDirectory) {
        std::error_code errorCode;
        const bool backupExists = std::filesystem::exists(replacementBackupDirectory, errorCode);
        if (errorCode || !backupExists) {
            return;
        }

        errorCode.clear();
        const bool artifactRootExists = std::filesystem::exists(artifactRoot, errorCode);
        if (errorCode || artifactRootExists) {
            return;
        }

        errorCode.clear();
        std::filesystem::rename(replacementBackupDirectory, artifactRoot, errorCode);
    }

    static void writeOptionalUint64(std::ostream& out, std::optional<uint64_t> value) {
        if (value.has_value()) {
            out << value.value();
        } else {
            out << "null";
        }
    }

    static void writeOptionalDouble(std::ostream& out, std::optional<double> value) {
        if (value.has_value() && std::isfinite(value.value())) {
            out << std::setprecision(17) << value.value();
        } else {
            out << "null";
        }
    }

    static void writeSelectionMetadata(const std::filesystem::path& artifactRoot, const TrainingSelectionMetadata& metadata) {
        std::filesystem::create_directories(artifactRoot);
        const std::filesystem::path metadataPath = artifactRoot / "training_selection_metadata.json";
        const std::filesystem::path tmpPath = artifactRoot / ".training_selection_metadata.json.tmp";

        {
            std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("Unable to open training selection metadata file for writing: " + tmpPath.string());
            }
            out << "{\n";
            out << "  \"schema_version\": 2,\n";
            out << "  \"latest_epoch\": " << metadata.latestEpoch << ",\n";
            out << "  \"latest_score\": ";
            writeOptionalDouble(out, metadata.latestScore);
            out << ",\n";
            out << "  \"latest_training_loss\": ";
            writeOptionalDouble(out, metadata.latestTrainingLoss);
            out << ",\n";
            out << "  \"latest_validation_loss\": ";
            writeOptionalDouble(out, metadata.latestValidationLoss);
            out << ",\n";
            out << "  \"has_best_candidate\": " << (metadata.bestEpoch.has_value() ? "true" : "false") << ",\n";
            out << "  \"best_epoch\": ";
            writeOptionalUint64(out, metadata.bestEpoch);
            out << ",\n";
            out << "  \"best_score\": ";
            writeOptionalDouble(out, metadata.bestScore);
            out << ",\n";
            out << "  \"completed_epoch\": " << metadata.completedEpoch << ",\n";
            out << "  \"completion_reason\": \"" << metadata.completionReason << "\",\n";
            out << "  \"check_best_model_every_epochs\": " << metadata.checkBestModelEveryEpochs << ",\n";
            out << "  \"first_model_selection_epoch\": " << metadata.firstModelSelectionEpoch << "\n";
            out << "}\n";
            if (!out) {
                throw std::runtime_error("Failed while writing training selection metadata file: " + tmpPath.string());
            }
        }

        std::error_code errorCode;
        std::filesystem::rename(tmpPath, metadataPath, errorCode);
        if (errorCode) {
            removePathIfExists(metadataPath);
            errorCode.clear();
            std::filesystem::rename(tmpPath, metadataPath, errorCode);
            if (errorCode) {
                removePathIfExists(tmpPath);
                throw std::runtime_error("Failed to finalize training selection metadata file '" + metadataPath.string() +
                                         "': " + errorCode.message());
            }
        }
    }

    std::optional<std::string> saveModelDirectory{};
    bool overwrite = false;
    std::optional<double> bestScore{};
    std::optional<uint64_t> bestEpoch{};
    std::optional<std::filesystem::path> bestCandidateDirectory{};
};

void ensureNativeQueuedPlanCompatible(const ExecutableTrainingPlan& plan, const Network& network, bool evaluateOnly) {
    if (evaluateOnly) {
        plan.validateNativeQueuedExecutorCompatible({});
        return;
    }
    plan.validateNativeQueuedExecutorCompatible(network.getTrainableParameterReferences(/*trainingEnabledOnly=*/true));
}

std::shared_ptr<TrainingStep> makeSingleNetworkTrainingStep(const std::string& stepName,
                                                              std::shared_ptr<Network> network,
                                                              std::shared_ptr<Optimizer> optimizer,
                                                              std::vector<ParameterReference> updateParameters = {},
                                                              uint32_t repeatCount = 1,
                                                              TrainingStep::GradientClearPolicy gradientClearPolicy =
                                                                  TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
                                                              std::vector<TrainingInputBinding> inputBindings = {},
                                                              bool enabled = true) {
    if (network == nullptr) {
        throw std::runtime_error("TrainingStep default phase requires a Network.");
    }
    if (network->getLossRootTensors().empty()) {
        throw std::runtime_error("TrainingStep default phase requires a Network with at least one graph loss.");
    }
    auto phase = std::make_shared<TrainingPhase>(stepName + "_phase", std::move(network), true);
    return std::make_shared<TrainingStep>(stepName,
                                          std::vector<std::shared_ptr<TrainingPhase>>{phase},
                                          std::move(optimizer),
                                          std::move(updateParameters),
                                          repeatCount,
                                          gradientClearPolicy,
                                          std::move(inputBindings),
                                          enabled);
}

std::vector<TrainingInputBinding> mergeDatasetInputBindings(
    const std::vector<TrainingInputBinding>& stepBindings,
    const std::vector<TrainingInputBinding>& datasetBindings) {
    std::vector<TrainingInputBinding> merged = stepBindings;
    std::map<std::string, std::string> batchNameByNetworkInput;
    for (const TrainingInputBinding& binding : stepBindings) {
        auto [it, inserted] = batchNameByNetworkInput.emplace(
            binding.getNetworkInputName(), binding.getBatchInputName());
        if (!inserted && it->second != binding.getBatchInputName()) {
            throw std::runtime_error("TrainingStep contains conflicting bindings for NetworkInput '" +
                                     binding.getNetworkInputName() + "'.");
        }
    }
    for (const TrainingInputBinding& binding : datasetBindings) {
        auto existing = batchNameByNetworkInput.find(binding.getNetworkInputName());
        if (existing != batchNameByNetworkInput.end()) {
            if (existing->second != binding.getBatchInputName()) {
                throw std::runtime_error(
                    "TrainingProgram input binding for NetworkInput '" + binding.getNetworkInputName() +
                    "' conflicts with the Trainer's dataset-resolved binding.");
            }
            continue;
        }
        batchNameByNetworkInput.emplace(binding.getNetworkInputName(), binding.getBatchInputName());
        merged.push_back(binding);
    }
    return merged;
}


std::vector<TrainingInputBinding> inputBindingsForNetwork(
    const std::vector<TrainingInputBinding>& bindings,
    const Network& network) {
    std::set<std::string> externalInputNames;
    for (const std::shared_ptr<NetworkInput>& input : network.getExternalNetworkInputs()) {
        externalInputNames.insert(input->getName());
    }

    std::vector<TrainingInputBinding> filtered;
    filtered.reserve(bindings.size());
    for (const TrainingInputBinding& binding : bindings) {
        if (externalInputNames.contains(binding.getNetworkInputName())) {
            filtered.push_back(binding);
        }
    }
    return filtered;
}

std::shared_ptr<TrainingProgram> programWithDatasetInputBindings(
    const TrainingProgram& program,
    const std::vector<TrainingInputBinding>& datasetBindings,
    bool evaluateOnly) {
    std::vector<std::shared_ptr<TrainingStep>> steps;
    steps.reserve(program.getNumSteps());
    for (uint64_t i = 0; i < program.getNumSteps(); ++i) {
        const TrainingStep& step = program.getStep(i);
        steps.push_back(std::make_shared<TrainingStep>(
            step.getName(),
            step.getPhases(),
            evaluateOnly ? nullptr : step.getOptimizer(),
            evaluateOnly ? std::vector<ParameterReference>{} : step.getUpdateParameters(),
            step.getRepeatCount(),
            step.getGradientClearPolicy(),
            mergeDatasetInputBindings(step.getInputBindings(), datasetBindings),
            step.isEnabled()));
    }
    return std::make_shared<TrainingProgram>(std::move(steps));
}

std::shared_ptr<TrainingProgram> evaluationOnlyProgramForRequest(const TrainingRunRequest& request) {
    std::vector<std::shared_ptr<TrainingStep>> evaluationSteps;

    if (request.trainingProgram != nullptr) {
        if (!request.trainingProgram->isInitialized()) {
            throw std::runtime_error("Trainer execution received an uninitialized TrainingProgram.");
        }
        return programWithDatasetInputBindings(
            *request.trainingProgram, request.datasetInputBindings, /*evaluateOnly=*/true);
    }

    evaluationSteps.push_back(makeSingleNetworkTrainingStep(
        "default",
        request.network,
        /*optimizer=*/nullptr,
        std::vector<ParameterReference>{},
        1,
        TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        request.datasetInputBindings));
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
        return programWithDatasetInputBindings(
            *request.trainingProgram, request.datasetInputBindings, /*evaluateOnly=*/false);
    }

    if (request.network->getLossRootTensors().empty()) {
        throw std::runtime_error("Trainer could not synthesize a default TrainingProgram because the Network has no graph losses.");
    }

    std::vector<ParameterReference> parameters = request.network->getTrainableParameterReferences(/*trainingEnabledOnly=*/true);
    if (parameters.empty()) {
        throw std::runtime_error("Trainer could not synthesize a default TrainingProgram because the Network has no trainable parameters.");
    }

    std::shared_ptr<Optimizer> optimizer = request.optimizer;
    if (optimizer == nullptr && request.network != nullptr) {
        optimizer = request.network->getDefaultOptimizer();
    }

    // Leave update_parameters empty on the implicit phase-backed step.  After active phase
    // composition, an empty update set resolves to all trainable parameters in the composed graph.
    auto defaultStep = makeSingleNetworkTrainingStep(
        "default",
        request.network,
        optimizer,
        std::vector<ParameterReference>{},
        1,
        TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        request.datasetInputBindings);
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

struct NativeQueuedExecutionGraph {
    std::shared_ptr<Network> network;
    std::shared_ptr<TrainingProgram> trainingProgram;
    bool composedFromTrainingPhases = false;
};

bool trainingProgramHasAnyPhase(const TrainingProgram& program) {
    for (const std::shared_ptr<TrainingStep>& step : program.getSteps()) {
        if (step == nullptr || !step->isInitialized()) {
            continue;
        }
        for (const std::shared_ptr<TrainingPhase>& phase : step->getPhases()) {
            if (phase != nullptr && phase->isInitialized()) {
                return true;
            }
        }
    }
    return false;
}

bool isImplicitDefaultSingleNetworkProgram(const TrainingProgram& program, const std::shared_ptr<Network>& network) {
    if (network == nullptr || program.getNumSteps() != 1) {
        return false;
    }
    const TrainingStep& step = program.getStep(0);
    if (step.getName() != "default") {
        return false;
    }
    const std::vector<std::shared_ptr<TrainingPhase>>& phases = step.getPhases();
    return phases.size() == 1 && phases[0] != nullptr && phases[0]->isInitialized() &&
           phases[0]->getName() == "default_phase" && phases[0]->getNetwork() == network;
}

void validateTrainingPhaseNativeQueuedProgramShape(const TrainingProgram& program) {
    if (program.getNumSteps() != 1) {
        throw std::runtime_error(
            "TrainingPhase native queued execution currently supports exactly one TrainingStep.");
    }

    const TrainingStep& step = program.getStep(0);
    if (!step.isEnabled()) {
        throw std::runtime_error("TrainingPhase native queued execution requires the TrainingStep to be enabled.");
    }
    if (step.getRepeatCount() != 1) {
        throw std::runtime_error(
            "TrainingPhase native queued execution currently supports only repeat_count=1.");
    }
    if (step.getGradientClearPolicy() != TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP) {
        throw std::runtime_error(
            "TrainingPhase native queued execution currently supports only clear_before_step gradient policy.");
    }
    if (!step.getUpdateParameters().empty()) {
        throw std::runtime_error(
            "TrainingPhase native queued execution currently requires empty update_parameters so the composed active graph can resolve all trainable parameters.");
    }

    for (const std::shared_ptr<TrainingPhase>& phase : step.getPhases()) {
        if (phase == nullptr || !phase->isInitialized()) {
            throw std::runtime_error("TrainingPhase native queued execution received an uninitialized phase.");
        }
    }
}

std::shared_ptr<Optimizer> optimizerForComposedPhaseGraph(const TrainingRunRequest& request, const TrainingStep& step) {
    if (step.getOptimizer() != nullptr) {
        return step.getOptimizer();
    }
    if (request.optimizer != nullptr) {
        return request.optimizer;
    }
    if (request.network != nullptr) {
        return request.network->getDefaultOptimizer();
    }
    return nullptr;
}

NativeQueuedExecutionGraph resolveNativeQueuedExecutionGraph(const TrainingRunRequest& request,
                                                            const std::shared_ptr<TrainingProgram>& requestedProgram,
                                                            bool evaluateOnly) {
    THOR_THROW_IF_FALSE(requestedProgram != nullptr);

    NativeQueuedExecutionGraph result;
    result.network = request.network;
    result.trainingProgram = requestedProgram;

    if (!trainingProgramHasAnyPhase(*requestedProgram) || isImplicitDefaultSingleNetworkProgram(*requestedProgram, request.network)) {
        if (request.network == nullptr) {
            throw std::runtime_error("Single-network Trainer execution requires network.");
        }
        return result;
    }

    validateTrainingPhaseNativeQueuedProgramShape(*requestedProgram);
    const TrainingStep& sourceStep = requestedProgram->getStep(0);

    const std::vector<std::shared_ptr<TrainingPhase>>& sourcePhases = sourceStep.getPhases();
    if (sourcePhases.empty() || sourcePhases.front() == nullptr || sourcePhases.front()->getNetwork() == nullptr) {
        throw std::runtime_error("TrainingPhase composition requires at least one phase Network.");
    }

    PhaseGraphComposeOptions composeOptions;
    composeOptions.networkName = sourcePhases.front()->getNetwork()->getNetworkName() + "_joined_" + sourceStep.getName();
    composeOptions.inferenceOnly = evaluateOnly;
    composeOptions.exposePhaseOutputsAsNetworkOutputs = true;

    ComposedPhaseGraph composedGraph = buildComposedPhaseGraphByName(sourceStep.getActivePhaseNetworkSpecs(), composeOptions);
    if (composedGraph.network == nullptr) {
        throw std::runtime_error("TrainingPhase composition produced a null active graph Network.");
    }

    if (composedGraph.network->getLossRootTensors().empty()) {
        throw std::runtime_error("TrainingPhase composition produced an active graph with no graph losses.");
    }

    std::shared_ptr<Optimizer> composedOptimizer = optimizerForComposedPhaseGraph(request, sourceStep);
    if (!evaluateOnly && composedOptimizer != nullptr && !composedGraph.network->allTrainingEnabledParametersHaveOptimizers()) {
        composedGraph.network->setDefaultOptimizer(composedOptimizer);
    }

    auto executionPhase = std::make_shared<TrainingPhase>(sourceStep.getName() + "_active_graph", composedGraph.network, true);
    auto executionStep = std::make_shared<TrainingStep>(
        sourceStep.getName(),
        std::vector<std::shared_ptr<TrainingPhase>>{executionPhase},
        sourceStep.getOptimizer(),
        std::vector<ParameterReference>{},
        sourceStep.getRepeatCount(),
        sourceStep.getGradientClearPolicy(),
        inputBindingsForNetwork(sourceStep.getInputBindings(), *composedGraph.network),
        sourceStep.isEnabled());

    result.network = composedGraph.network;
    result.trainingProgram = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{executionStep});
    result.composedFromTrainingPhases = true;
    return result;
}

std::map<std::string, BatchFieldSourceDescription> resolveNetworkInputBatchSources(
    const std::shared_ptr<BatchSession>& batchSession, const ExecutableTrainingPlan& plan) {
    const std::shared_ptr<BatchSession>& session = batchSession;
    std::map<std::string, BatchFieldSourceDescription> sources;
    for (const StepExecutable& step : plan.getSteps()) {
        for (const TrainingInputBinding& binding : step.getResolvedInputBindings()) {
            const BatchFieldSourceDescription source =
                session == nullptr
                    ? BatchFieldSourceDescription::materialized()
                    : session->getBatchFieldSourceDescription(binding.getBatchInputName());
            auto [existing, inserted] = sources.emplace(binding.getNetworkInputName(), source);
            if (inserted) {
                continue;
            }

            if (existing->second.kind != source.kind) {
                throw std::runtime_error(
                    "NetworkInput '" + binding.getNetworkInputName() +
                    "' is bound to both materialized-tensor and device-reference batch fields across training steps.");
            }

            if (source.kind == BatchFieldSourceKind::MATERIALIZED_TENSOR) {
                if (!existing->second.placement.has_value() || !source.placement.has_value() ||
                    existing->second.placement.value() != source.placement.value()) {
                    // Differing or unknown materialized placements use the
                    // conservative host-staged path for every step.
                    existing->second.placement = std::nullopt;
                }
            } else {
                if (!existing->second.placement.has_value() || !source.placement.has_value() ||
                    existing->second.placement.value() != source.placement.value()) {
                    throw std::runtime_error(
                        "NetworkInput '" + binding.getNetworkInputName() +
                        "' is bound to device-reference fields with differing destination placements.");
                }
            }
        }
    }
    return sources;
}

void cancelBatchSession(const std::shared_ptr<BatchSession>& batchSession) {
    if (batchSession != nullptr) {
        batchSession->cancel();
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
        } else if (std::holds_alternative<DeviceBatchReference>(value)) {
            bound.insert(binding.getNetworkInputName(), std::get<DeviceBatchReference>(value));
        } else {
            THOR_UNREACHABLE();
        }
        const std::optional<BatchSourceReference> sourceReference =
            batchInput.getSourceReference(binding.getBatchInputName());
        if (sourceReference.has_value()) {
            bound.setSourceReference(
                binding.getNetworkInputName(),
                sourceReference.value());
        }
    }
    return bound;
}

void CUDART_CB completeNativeQueuedBatch(void* data) {
    NativeBatchCompletionParams* params = static_cast<NativeBatchCompletionParams*>(data);
    std::shared_ptr<QueuedTrainingState> state = params->state;
    if (state == nullptr) {
        return;
    }
    const TrainingEventPhase phase = params->phase;
    const uint64_t epochBatchNum = params->epochBatchNum;
    const uint64_t slotIndex = params->slotIndex;

    try {
        THOR_THROW_IF_FALSE(params->scalarStats.size() == state->scalarTensorNames.size());
        THOR_THROW_IF_FALSE(
            params->scalarStatSources.size() == state->scalarTensorNames.size());
        for (size_t i = 0; i < state->scalarTensorNames.size(); ++i) {
            if (params->scalarStats[i].present) {
                continue;
            }
            const std::string& tensorName = state->scalarTensorNames[i];
            if (params->scalarStatSources[i] == ScalarStatSource::INPUT) {
                THOR_THROW_IF_FALSE(!params->batchLease.empty());
                params->scalarStats[i].value = copyInputScalarStatTensor(
                    params->batchLease.get(),
                    tensorName);
            } else if (params->scalarStatSources[i] == ScalarStatSource::OUTPUT) {
                params->scalarStats[i].value = copyOutputScalarStatTensor(
                    params->batchOutput,
                    tensorName,
                    state->aggregateLossTensorNames);
            } else {
                throw std::runtime_error(
                    "Training stat tensor '" + tensorName +
                    "' did not resolve to an input or output source.");
            }
            params->scalarStats[i].present = true;
        }

        // A retained device input statistic was the only reason a fully
        // source-tracked BatchLease could survive until this callback. Once
        // the statistic has been copied, release the source owner and session
        // lease without waiting for the consumer thread to pop the completed
        // network slot.
        if (!params->batchLease.empty() &&
            params->batchLease.get().allFieldsHaveSourceReferences()) {
            params->batchLease.releaseSourceResourcesExcept();
            params->batchLease.reset();
        }

        uint64_t inFlightAtComplete = 0;
        uint64_t doneAtComplete = 0;
        uint64_t totalAtComplete = 0;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            THOR_THROW_IF_FALSE(slotIndex < state->slots.size());
            QueuedBatchSlot& slot = state->slots[slotIndex];
            THOR_THROW_IF_FALSE(slot.occupied);
            THOR_THROW_IF_FALSE(slot.phase == phase);
            THOR_THROW_IF_FALSE(slot.epochBatchNum == epochBatchNum);
            THOR_THROW_IF_FALSE(slot.batchesInEpoch == params->batchesInEpoch);
            THOR_THROW_IF_FALSE(slot.scalarStats.size() == params->scalarStats.size());
            QueuedPhaseProgress& progress = phaseProgress(*state, phase);
            progress.completedBatches += 1;
            slot.doneInEpochAtComplete = progress.completedBatches;
            slot.scalarStats = params->scalarStats;
            // Timestamp the batch when the completion callback has actually observed the
            // GPU work and required output copies as complete. Throughput must be based
            // on completion times, not on when the consumer thread later pops an already
            // ready slot; otherwise draining a backlog of completed slots can create
            // impossible end-of-epoch rate spikes.
            slot.completionTime = std::chrono::high_resolution_clock::now();
            slot.ready = true;
            state->numBatchesDoneInEpoch += 1;
            params->completionCallbackFinished = true;
            inFlightAtComplete = state->inFlightBatches;
            doneAtComplete = slot.doneInEpochAtComplete;
            totalAtComplete = slot.batchesInEpoch;
        }
        if (shouldEmitQueueDiagnostic(doneAtComplete)) {
            emitNativeQueueDiagnostic("complete",
                                      phase,
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
        // batchSession-owned tensors and rethrow the error through Trainer.fit(...).
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->failure == nullptr) {
            state->failure = std::current_exception();
        }
        if (slotIndex < state->slots.size()) {
            QueuedBatchSlot& slot = state->slots[slotIndex];
            if (slot.occupied && slot.phase == phase && slot.epochBatchNum == epochBatchNum) {
                QueuedPhaseProgress& progress = phaseProgress(*state, phase);
                progress.completedBatches += 1;
                slot.doneInEpochAtComplete = progress.completedBatches;
                slot.completionTime = std::chrono::high_resolution_clock::now();
                slot.ready = true;
            }
        }
        state->numBatchesDoneInEpoch += 1;
        params->completionCallbackFinished = true;
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
    BatchLease batchLease;
    ExampleType exampleType = ExampleType::TRAIN;
    TrainingEventPhase phase = TrainingEventPhase::TRAIN;
    uint64_t currentEpoch = 0;
    uint64_t epochBatchNum = 0;
    uint64_t slotIndex = 0;
    uint64_t inFlightAfterPop = 0;
    uint64_t doneInEpoch = 0;
    uint64_t poppedInEpoch = 0;
    uint64_t batchesInEpoch = 0;
    std::chrono::high_resolution_clock::time_point completionTime{};
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
    result.exampleType = params.exampleType;
    result.phase = slot.phase;
    result.currentEpoch = params.currentEpoch;
    result.epochBatchNum = slot.epochBatchNum;
    result.slotIndex = state->headSlot;
    result.doneInEpoch = slot.doneInEpochAtComplete;
    result.batchesInEpoch = slot.batchesInEpoch;
    result.completionTime = slot.completionTime;
    result.batchLease = std::move(params.batchLease);
    result.scalarStats = slot.scalarStats;

    params.batchOutput.clear();
    params.completionCallbackLaunched = false;
    params.completionCallbackFinished = false;
    params.state.reset();
    for (ScalarStatSlot& scalarStat : slot.scalarStats) {
        scalarStat.present = false;
        scalarStat.value = 0.0f;
    }
    for (ScalarStatSlot& scalarStat : params.scalarStats) {
        scalarStat.present = false;
        scalarStat.value = 0.0f;
    }
    for (ScalarStatSource& scalarStatSource : params.scalarStatSources) {
        scalarStatSource = ScalarStatSource::UNRESOLVED;
    }
    QueuedPhaseProgress& progress = phaseProgress(*state, slot.phase);
    progress.poppedBatches += 1;
    result.poppedInEpoch = progress.poppedBatches;

    slot.ready = false;
    slot.occupied = false;
    slot.phase = TrainingEventPhase::TRAIN;
    slot.epochBatchNum = 0;
    slot.batchesInEpoch = 0;
    slot.doneInEpochAtComplete = 0;
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

bool queuedCompletionCallbacksPendingUnlocked(const QueuedTrainingState& state) {
    for (const NativeBatchCompletionParams& params : state.completionParams) {
        if (params.completionCallbackLaunched && !params.completionCallbackFinished) {
            return true;
        }
    }
    return false;
}

void releaseQueuedTrainingStateReferencesAfterAbort(const std::shared_ptr<QueuedTrainingState>& state, bool submittedWorkDrained) {
    if (state == nullptr) {
        return;
    }

    std::vector<BatchLease> leasesToRelease;

    std::unique_lock<std::mutex> lock(state->mutex);
    if (!submittedWorkDrained && queuedCompletionCallbacksPendingUnlocked(*state)) {
        // If CUDA synchronization failed, a launched host callback may still hold
        // a raw pointer into state->completionParams. Do not break those references
        // here; preserving the old leak behavior is safer than risking a UAF in an
        // already-failing CUDA context. Normal trainer failures reach this path with
        // submittedWorkDrained=true and are cleaned up below.
        return;
    }
    while (queuedCompletionCallbacksPendingUnlocked(*state)) {
        state->batchFinished.wait_for(lock, std::chrono::milliseconds(50));
    }

    for (NativeBatchCompletionParams& params : state->completionParams) {
        if (!params.batchLease.empty()) {
            leasesToRelease.push_back(std::move(params.batchLease));
        }
        params.batchOutput.clear();
        for (ScalarStatSlot& scalarStat : params.scalarStats) {
            scalarStat.present = false;
            scalarStat.value = 0.0f;
        }
        for (ScalarStatSource& scalarStatSource : params.scalarStatSources) {
            scalarStatSource = ScalarStatSource::UNRESOLVED;
        }
        params.completionCallbackLaunched = false;
        params.completionCallbackFinished = false;
        params.state.reset();
    }

    for (QueuedBatchSlot& slot : state->slots) {
        slot.occupied = false;
        slot.ready = false;
        slot.phase = TrainingEventPhase::TRAIN;
        slot.epochBatchNum = 0;
        slot.batchesInEpoch = 0;
        slot.doneInEpochAtComplete = 0;
        slot.paramsIndex = 0;
        slot.completionTime = {};
        for (ScalarStatSlot& scalarStat : slot.scalarStats) {
            scalarStat.present = false;
            scalarStat.value = 0.0f;
        }
    }
    state->headSlot = 0;
    state->tailSlot = 0;
    state->inFlightBatches = 0;
    lock.unlock();

    leasesToRelease.clear();
    state->batchPopped.notify_all();
}

void emitTrainingEvent(TrainingObserver& observer, const TrainingEvent& event) {
    observer.onTrainingEvent(event);
}

void assignScalarStatsToSnapshot(TrainingStatsSnapshot& snapshot,
                                 const std::vector<std::string>& scalarTensorNames,
                                 const std::vector<ScalarStatSlot>& scalarStats,
                                 const std::vector<std::string>& aggregateLossTensorNames) {
    THOR_THROW_IF_FALSE(scalarTensorNames.size() == scalarStats.size());
    std::map<std::string, double> scalarValuesByName;
    const std::set<std::string> aggregateLossTensorNameSet = setFromVector(aggregateLossTensorNames);
    for (size_t i = 0; i < scalarTensorNames.size(); ++i) {
        if (!scalarStats[i].present) {
            continue;
        }

        const double value = static_cast<double>(scalarStats[i].value);
        const std::string& name = scalarTensorNames[i];
        scalarValuesByName[name] = value;
        if (name == "loss") {
            snapshot.loss = value;
        } else if (name == "learning_rate" || name == "learningRate" || name == "lr") {
            snapshot.learningRate = value;
        } else if (name == "momentum") {
            snapshot.momentum = value;
        } else {
            if (aggregateLossTensorNameSet.count(name) != 0) {
                snapshot.losses[name] = value;
            }
            snapshot.metrics[name] = value;
        }
    }

    if (!snapshot.loss.has_value() && !aggregateLossTensorNames.empty()) {
        double aggregateLoss = 0.0;
        bool missingAggregateLossScalar = false;
        for (const std::string& lossTensorName : aggregateLossTensorNames) {
            auto valueIt = scalarValuesByName.find(lossTensorName);
            if (valueIt == scalarValuesByName.end()) {
                missingAggregateLossScalar = true;
                break;
            }
            aggregateLoss += valueIt->second;
        }
        // Preserve non-finite aggregate losses in the stats snapshot.  The
        // Trainer owns the policy decision that a non-finite TRAIN/VALIDATE
        // loss fails the attempt; dropping NaN/Inf here makes that failure
        // indistinguishable from an absent loss.
        if (!missingAggregateLossScalar) {
            snapshot.loss = aggregateLoss;
        }
    }
}

struct QueuedEpochPhaseWork {
    ExampleType exampleType = ExampleType::TRAIN;
    TrainingEventPhase phase = TrainingEventPhase::TRAIN;
    uint64_t initialBatchNum = 0;
    uint64_t batchesToRunCount = 0;
    uint64_t batchesPerEpoch = 0;

    [[nodiscard]] uint64_t batchesToRun() const { return batchesToRunCount; }
};

class NativeQueuedEpochScheduler {
   public:
    NativeQueuedEpochScheduler(std::shared_ptr<PlacedNetwork> placedNetwork,
                               std::shared_ptr<BatchSession> batchSession,
                               std::shared_ptr<const ExecutableTrainingPlan> plan,
                               const NativeQueuedTrainingOptions& options,
                               std::shared_ptr<QueuedTrainingState> state,
                               uint64_t currentEpoch,
                               TrainingCancellationToken cancellationToken)
        : placedNetwork(std::move(placedNetwork)),
          batchSession(std::move(batchSession)),
          plan(std::move(plan)),
          options(options),
          state(std::move(state)),
          currentEpoch(currentEpoch),
          cancellationToken(std::move(cancellationToken)),
          outputReadyEvents(this->placedNetwork->getNumStamps()),
          processingFinishedEvents(options.maxInFlightBatches),
          completionFinishedEvents(options.maxInFlightBatches) {
        stampGpuNums.reserve(this->placedNetwork->getNumStamps());
        for (uint64_t stamp = 0; stamp < this->placedNetwork->getNumStamps(); ++stamp) {
            ThorImplementation::StampedNetwork& stampedNetwork = this->placedNetwork->getStampedNetwork(stamp);
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
        completionStreams.reserve(options.maxInFlightBatches);
        for (uint64_t slotIndex = 0; slotIndex < options.maxInFlightBatches; ++slotIndex) {
            const uint64_t stamp = slotIndex % this->placedNetwork->getNumStamps();
            completionStreams.push_back(Stream::getNextDownloadStream(stampGpuNums[stamp]));
        }
    }

    void operator()(uint64_t initialEpochBatchNum,
                    uint64_t batches,
                    uint64_t batchesPerEpoch,
                    ExampleType exampleType,
                    TrainingEventPhase diagnosticPhase) {
        if (batches == 0) {
            return;
        }

        emitNativeQueueDiagnostic(
            "phase_schedule_start", diagnosticPhase, currentEpoch, initialEpochBatchNum, 0, 0, initialEpochBatchNum, batchesPerEpoch);

        const bool validationPass = exampleType != ExampleType::TRAIN;
        const bool useGpuSubmitCoordinator = gpuSubmitCoordinatorEnabled();
#if THOR_ENABLE_TRAINING_QUEUE_DIAGNOSTICS
        const bool collectQueueDiagnostics = queueDiagnosticsEnabled();
#else
        constexpr bool collectQueueDiagnostics = false;
#endif
        THOR_THROW_IF_FALSE(plan != nullptr);
        const std::vector<StepExecutable>& steps = plan->getSteps();

        for (uint64_t batch = 0; batch < batches; ++batch) {
            if (cancellationToken.isCancellationRequested()) {
                requestQueuedTrainingCancellation(state);
                return;
            }
            const auto scheduleIterationStart = diagnosticNow(collectQueueDiagnostics);
            const uint64_t epochBatchNum = initialEpochBatchNum + batch;
            const auto optimizerStart = diagnosticNow(collectQueueDiagnostics);
            Optimizer::updateHyperParameters(placedNetwork.get(), currentEpoch, epochBatchNum, batchesPerEpoch);
            const auto optimizerFinish = diagnosticNow(collectQueueDiagnostics);

            uint64_t slotIndex = 0;
            uint64_t inFlightAfterReserve = 0;
            const auto reserveStart = diagnosticNow(collectQueueDiagnostics);
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
                slot.phase = diagnosticPhase;
                slot.epochBatchNum = epochBatchNum;
                slot.batchesInEpoch = batchesPerEpoch;
                slot.doneInEpochAtComplete = 0;
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
            const auto reserveFinish = diagnosticNow(collectQueueDiagnostics);
            if (collectQueueDiagnostics && shouldEmitQueueDiagnostic(batch + 1)) {
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
            params->exampleType = exampleType;
            params->phase = diagnosticPhase;
            params->completionCallbackLaunched = false;
            params->completionCallbackFinished = false;
            params->currentEpoch = currentEpoch;
            params->epochBatchNum = epochBatchNum;
            params->batchesInEpoch = batchesPerEpoch;
            params->slotIndex = slotIndex;
            params->batchLease.reset();
            params->batchOutput.clear();
            for (ScalarStatSlot& scalarStat : params->scalarStats) {
                scalarStat.present = false;
                scalarStat.value = 0.0f;
            }
            for (ScalarStatSource& scalarStatSource : params->scalarStatSources) {
                scalarStatSource = ScalarStatSource::UNRESOLVED;
            }

            const auto acquireBatchStart = diagnosticNow(collectQueueDiagnostics);
            uint64_t sessionBatchNum = epochBatchNum;
            params->batchLease = batchSession->leaseBatch(exampleType, sessionBatchNum);
            const auto acquireBatchFinish = diagnosticNow(collectQueueDiagnostics);
            const uint64_t acquireBatchWaitMicros = collectQueueDiagnostics ? elapsedMicros(acquireBatchStart, acquireBatchFinish) : 0;
            if (collectQueueDiagnostics && shouldEmitQueueDiagnostic(batch + 1, acquireBatchWaitMicros)) {
                emitNativeQueueDiagnostic("acquire_batch_done",
                                          diagnosticPhase,
                                          currentEpoch,
                                          epochBatchNum,
                                          slotIndex,
                                          inFlightAfterReserve,
                                          initialEpochBatchNum + batch,
                                          batchesPerEpoch,
                                          acquireBatchWaitMicros);
            }

            const auto submitStart = diagnosticNow(collectQueueDiagnostics);
            uint64_t bindMicros = 0;
            uint64_t submitBatchMicros = 0;
            uint64_t submitCalls = 0;
            uint64_t coordinatorQueueWaitMicros = 0;
            uint64_t coordinatorSetGpuMicros = 0;
            uint64_t coordinatorExecMicros = 0;
            uint64_t coordinatorRoundtripMicros = 0;
            ThorImplementation::BatchSubmissionTiming submitTiming;
            for (const StepExecutable& step : steps) {
                for (uint32_t repeat = 0; repeat < step.getRepeatCount(); ++repeat) {
                    const auto bindStart = diagnosticNow(collectQueueDiagnostics);
                    Batch boundBatchInput = bindBatchInputs(step, params->batchLease.get());
                    const auto bindFinish = diagnosticNow(collectQueueDiagnostics);
                    if (collectQueueDiagnostics) {
                        bindMicros += elapsedMicros(bindStart, bindFinish);
                    }
                    params->batchOutput.clear();
                    ThorImplementation::BatchSubmissionTiming singleSubmitTiming;
                    auto submitWork = [&]() {
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
                        return placedNetwork->submitBatch(nextStampToProcess,
                                                          boundBatchInput,
                                                          params->batchOutput,
                                                          outputReadyEvents[nextStampToProcess],
                                                          validationPass,
                                                          step.getObjectiveRoots(),
                                                          &processingFinishedEvents[slotIndex],
                                                          /*waitForOutputsOnProcessingStream=*/false,
                                                          collectQueueDiagnostics ? &singleSubmitTiming : nullptr,
                                                          slotIndex);
                    };

                    const auto submitBatchStart = diagnosticNow(collectQueueDiagnostics);
                    if (useGpuSubmitCoordinator) {
                        GpuSubmitCoordinatorTiming coordinatorTiming;
                        auto& coordinator = GpuSubmitCoordinatorRegistry::get(stampGpuNums[nextStampToProcess]);
                        auto submitFuture = coordinator.submit(submitWork, collectQueueDiagnostics ? &coordinatorTiming : nullptr);
                        submitFuture.get();
                        if (collectQueueDiagnostics) {
                            coordinatorQueueWaitMicros += coordinatorTiming.queueWaitMicros;
                            coordinatorSetGpuMicros += coordinatorTiming.setGpuMicros;
                            coordinatorExecMicros += coordinatorTiming.execMicros;
                        }
                    } else {
                        submitWork();
                    }
                    const auto submitBatchFinish = diagnosticNow(collectQueueDiagnostics);
                    if (collectQueueDiagnostics) {
                        const uint64_t submitBatchElapsedMicros = elapsedMicros(submitBatchStart, submitBatchFinish);
                        submitBatchMicros += submitBatchElapsedMicros;
                        if (useGpuSubmitCoordinator) {
                            coordinatorRoundtripMicros += submitBatchElapsedMicros;
                        }
                    }
                    if (collectQueueDiagnostics) {
                        ThorImplementation::accumulateBatchSubmissionTiming(submitTiming, singleSubmitTiming);
                    }
                    submitCalls += 1;
                }
            }
            const auto submitFinish = diagnosticNow(collectQueueDiagnostics);

            // Every NetworkInput has now recorded the stream point after which
            // it no longer reads its session-owned source. Snapshot CPU input
            // statistics now, before those source slots are released; output
            // statistics remain deferred until the normal completion callback.
            THOR_THROW_IF_FALSE(params->scalarStats.size() == state->scalarTensorNames.size());
            THOR_THROW_IF_FALSE(
                params->scalarStatSources.size() == state->scalarTensorNames.size());
            const bool fullySourceTracked =
                params->batchLease.get().allFieldsHaveSourceReferences();
            std::set<std::string> retainedSourceFields;
            for (size_t i = 0; i < state->scalarTensorNames.size(); ++i) {
                const std::string& scalarTensorName = state->scalarTensorNames[i];
                if (!params->batchLease.get().contains(scalarTensorName)) {
                    params->scalarStatSources[i] = ScalarStatSource::OUTPUT;
                    continue;
                }

                params->scalarStatSources[i] = ScalarStatSource::INPUT;
                if (!params->batchLease.get().isTensor(scalarTensorName)) {
                    throw std::runtime_error(
                        "Requested input training stat tensor '" + scalarTensorName +
                        "' is not a materialized dense tensor.");
                }
                const ThorImplementation::Tensor& inputStatTensor =
                    params->batchLease.get().getTensor(scalarTensorName);
                if (inputStatTensor.getPlacement().getMemDevice() ==
                    ThorImplementation::TensorPlacement::MemDevices::CPU) {
                    params->scalarStats[i].value = copyInputScalarStatTensor(
                        params->batchLease.get(),
                        scalarTensorName);
                    params->scalarStats[i].present = true;
                } else {
                    // Device-backed materialized input statistics are copied in
                    // the normal completion callback. Retain only the source
                    // resource containing that field until then.
                    retainedSourceFields.insert(scalarTensorName);
                }
            }
            params->batchLease.releaseSourceResourcesExcept(retainedSourceFields);
            if (fullySourceTracked && retainedSourceFields.empty()) {
                // NetworkInput rings now own any queued tensor/reference values,
                // and source owners hold the reusable buffers until their actual
                // read events complete. Forward/backward and output completion no
                // longer require the dataset Batch or its session lease.
                params->batchLease.reset();
            }

            const auto completionSetupStart = diagnosticNow(collectQueueDiagnostics);
            // Keep CPU stats/output completion off the stamp's input stream.  The input stream
            // event is the point where the GPU training work is done enough for the next batch
            // to be queued on this single stamp.  Output tensors that are copied through
            // NetworkOutput-owned download streams are waited on here, and the host callback
            // snapshots the shared CPU output tensors into per-slot scalarStats before those
            // public output tensors may be reused by a later batch.
            Stream completionStream = completionStreams[slotIndex];
            const auto waitProcessingStart = diagnosticNow(collectQueueDiagnostics);
            completionStream.waitEvent(processingFinishedEvents[slotIndex]);
            const auto waitProcessingFinish = diagnosticNow(collectQueueDiagnostics);

            const auto waitOutputsStart = diagnosticNow(collectQueueDiagnostics);
            uint64_t outputWaitCount = 0;
            for (const auto& [outputName, outputReadyEvent] : outputReadyEvents[nextStampToProcess]) {
                (void)outputName;
                completionStream.waitEvent(outputReadyEvent);
                outputWaitCount += 1;
            }
            const auto waitOutputsFinish = diagnosticNow(collectQueueDiagnostics);

            const auto hostFuncStart = diagnosticNow(collectQueueDiagnostics);
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                params->completionCallbackLaunched = true;
                params->completionCallbackFinished = false;
            }
            cudaError_t cudaStatus = cudaLaunchHostFunc(completionStream, completeNativeQueuedBatch, params);
            const auto hostFuncFinish = diagnosticNow(collectQueueDiagnostics);
            if (cudaStatus != cudaSuccess) {
                std::lock_guard<std::mutex> lock(state->mutex);
                params->completionCallbackLaunched = false;
                params->completionCallbackFinished = false;
                THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            }
            if (collectQueueDiagnostics && shouldEmitQueueDiagnostic(batch + 1)) {
                emitNativeQueueDiagnostic("submit",
                                          diagnosticPhase,
                                          currentEpoch,
                                          epochBatchNum,
                                          slotIndex,
                                          inFlightAfterReserve,
                                          initialEpochBatchNum + batch,
                                          batchesPerEpoch);
            }
            const auto putEventStart = diagnosticNow(collectQueueDiagnostics);
            completionStream.putEvent(completionFinishedEvents[slotIndex], false, true);
            const auto putEventFinish = diagnosticNow(collectQueueDiagnostics);

            const auto extendOutputsStart = diagnosticNow(collectQueueDiagnostics);
            placedNetwork->extendOutputWritableEvents(nextStampToProcess, completionFinishedEvents[slotIndex], slotIndex);
            const auto extendOutputsFinish = diagnosticNow(collectQueueDiagnostics);
            const auto completionSetupFinish = extendOutputsFinish;

            if (collectQueueDiagnostics && shouldEmitQueueDiagnostic(batch + 1)) {
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

            if (collectQueueDiagnostics && shouldEmitQueueDiagnostic(batch + 1)) {
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
                                                      submitTiming,
                                                      useGpuSubmitCoordinator,
                                                      coordinatorQueueWaitMicros,
                                                      coordinatorSetGpuMicros,
                                                      coordinatorExecMicros,
                                                      coordinatorRoundtripMicros);
            }

            if (collectQueueDiagnostics && shouldEmitQueueDiagnostic(batch + 1)) {
                emitNativeQueueScheduleTimingDiagnostic(diagnosticPhase,
                                                        currentEpoch,
                                                        epochBatchNum,
                                                        slotIndex,
                                                        inFlightAfterReserve,
                                                        initialEpochBatchNum + batch,
                                                        batchesPerEpoch,
                                                        elapsedMicros(optimizerStart, optimizerFinish),
                                                        elapsedMicros(reserveStart, reserveFinish),
                                                        acquireBatchWaitMicros,
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
    std::shared_ptr<BatchSession> batchSession;
    std::shared_ptr<const ExecutableTrainingPlan> plan;
    NativeQueuedTrainingOptions options;
    std::shared_ptr<QueuedTrainingState> state;
    uint64_t currentEpoch;
    TrainingCancellationToken cancellationToken;
    uint64_t nextStampToProcess = 0;
    std::vector<std::map<std::string, Event>> outputReadyEvents;
    std::vector<Event> processingFinishedEvents;
    std::vector<Event> completionFinishedEvents;
    std::vector<int> stampGpuNums;
    std::vector<Stream> completionStreams;
};

struct NativeQueuedEpochExecution {
    std::shared_ptr<QueuedTrainingState> state;
    std::vector<QueuedEpochPhaseWork> phaseWorks;
    uint64_t initialCompletedBatches = 0;
    uint64_t initialWarmupCompletionTarget = 0;
    std::chrono::high_resolution_clock::time_point startedAt{};
    std::thread schedulingThread;
};

struct QueuedEpochWorkPlan {
    std::vector<QueuedEpochPhaseWork> phaseWorks;
    uint64_t initiallyCompletedBatches = 0;
    uint64_t totalBatchesAcrossPhases = 0;
    uint64_t remainingBatchesAcrossPhases = 0;
};

QueuedEpochWorkPlan buildQueuedEpochWorkPlan(
    const TrainingRunRequest& request,
    const std::shared_ptr<BatchSession>& effectiveSession,
    bool evaluateOnly) {
    QueuedEpochWorkPlan workPlan;

    std::vector<std::pair<ExampleType, TrainingEventPhase>> phaseSpecs;
    if (evaluateOnly) {
        phaseSpecs.emplace_back(
            request.evaluationExampleType, request.evaluationPhase);
    } else {
        phaseSpecs.emplace_back(
            ExampleType::TRAIN, TrainingEventPhase::TRAIN);
        phaseSpecs.emplace_back(
            ExampleType::VALIDATE, TrainingEventPhase::VALIDATE);
    }

    workPlan.phaseWorks.reserve(phaseSpecs.size());
    for (const auto& phaseSpec : phaseSpecs) {
        request.cancellationToken.throwIfCancellationRequested();
        const ExampleType exampleType = phaseSpec.first;
        const TrainingEventPhase phase = phaseSpec.second;
        const uint64_t sessionBatchNum =
            effectiveSession->getNextBatchNum(exampleType);
        const uint64_t sessionBatchesPerEpoch =
            effectiveSession->getNumBatchesPerEpoch(exampleType);
        if (sessionBatchNum > sessionBatchesPerEpoch) {
            throw std::runtime_error(
                "BatchSession returned next batch number beyond batches per "
                "epoch for " +
                phaseName(phase) + ".");
        }

        uint64_t publicInitialBatchNum = sessionBatchNum;
        uint64_t publicBatchesPerEpoch = sessionBatchesPerEpoch;
        uint64_t batchesToRun = sessionBatchesPerEpoch - sessionBatchNum;
        if (!evaluateOnly && phase == TrainingEventPhase::TRAIN &&
            request.maxTrainingBatchesPerEpoch.has_value() &&
            sessionBatchesPerEpoch >
                request.maxTrainingBatchesPerEpoch.value()) {
            // A capped public training epoch is a fixed-size work quantum. It
            // must not end early merely because the underlying full-dataset
            // epoch boundary is reached; sessions wrap and continue streaming.
            batchesToRun = request.maxTrainingBatchesPerEpoch.value();
            publicInitialBatchNum = 0;
            publicBatchesPerEpoch = batchesToRun;
        }

        workPlan.phaseWorks.push_back(QueuedEpochPhaseWork{
            exampleType,
            phase,
            publicInitialBatchNum,
            batchesToRun,
            publicBatchesPerEpoch});
        workPlan.initiallyCompletedBatches += publicInitialBatchNum;
        workPlan.totalBatchesAcrossPhases += publicBatchesPerEpoch;
        workPlan.remainingBatchesAcrossPhases += batchesToRun;
    }

    return workPlan;
}

NativeQueuedEpochExecution launchNativeQueuedEpochExecution(
    const TrainingRunRequest& request,
    const std::shared_ptr<PlacedNetwork>& placedNetwork,
    const std::shared_ptr<BatchSession>& effectiveSession,
    std::shared_ptr<const ExecutableTrainingPlan> plan,
    const NativeQueuedTrainingOptions& options,
    const std::vector<std::string>& scalarTensorNames,
    const std::vector<std::string>& aggregateLossTensorNames,
    uint64_t currentEpoch,
    bool evaluateOnly) {
    QueuedEpochWorkPlan workPlan =
        buildQueuedEpochWorkPlan(request, effectiveSession, evaluateOnly);

    NativeQueuedEpochExecution execution;
    execution.phaseWorks = std::move(workPlan.phaseWorks);
    execution.initialCompletedBatches =
        workPlan.initiallyCompletedBatches;
    execution.initialWarmupCompletionTarget =
        workPlan.initiallyCompletedBatches +
        std::min<uint64_t>(
            workPlan.remainingBatchesAcrossPhases,
            uint64_t{1});
    execution.startedAt = std::chrono::high_resolution_clock::now();
    execution.state = std::make_shared<QueuedTrainingState>(
        options.maxInFlightBatches,
        scalarTensorNames,
        aggregateLossTensorNames);
    execution.state->numBatchesDoneInEpoch =
        workPlan.initiallyCompletedBatches;
    execution.state->numBatchesInEpoch =
        workPlan.totalBatchesAcrossPhases;
    for (const QueuedEpochPhaseWork& work : execution.phaseWorks) {
        QueuedPhaseProgress& progress =
            phaseProgress(*execution.state, work.phase);
        progress.completedBatches = work.initialBatchNum;
        progress.poppedBatches = work.initialBatchNum;
    }

    NativeQueuedEpochScheduler scheduler(
        placedNetwork,
        effectiveSession,
        std::move(plan),
        options,
        execution.state,
        currentEpoch,
        request.cancellationToken);
    execution.schedulingThread = std::thread(
        [scheduler = std::move(scheduler),
         state = execution.state,
         phaseWorks = execution.phaseWorks]() mutable {
            try {
                for (const QueuedEpochPhaseWork& work : phaseWorks) {
                    scheduler(
                        work.initialBatchNum,
                        work.batchesToRun(),
                        work.batchesPerEpoch,
                        work.exampleType,
                        work.phase);
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    if (state->failure == nullptr) {
                        state->failure = std::current_exception();
                    }
                    state->cancelRequested = true;
                    state->numBatchesDoneInEpoch =
                        state->numBatchesInEpoch;
                }
                state->batchFinished.notify_all();
                state->batchPopped.notify_all();
            }
        });

    return execution;
}

void waitForInitialQueueCompletion(
    const NativeQueuedEpochExecution& execution) {
    std::unique_lock<std::mutex> lock(execution.state->mutex);
    while (execution.state->failure == nullptr &&
           !execution.state->cancelRequested &&
           execution.state->numBatchesDoneInEpoch <
               execution.initialWarmupCompletionTarget) {
        execution.state->batchFinished.wait_for(
            lock, std::chrono::milliseconds(50));
    }

    const std::exception_ptr failure = execution.state->failure;
    const bool cancelRequested = execution.state->cancelRequested;
    const bool interruptRequested = execution.state->interruptRequested;
    lock.unlock();

    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    if (interruptRequested) {
        throw TrainingInterrupted(
            "Native queued trainer interrupted by SIGINT.");
    }
    if (cancelRequested) {
        throw TrainingCancelled(
            "Native queued trainer was cancelled.");
    }
}

bool drainQueuedTrainingWorkAfterFailure(
    const std::shared_ptr<PlacedNetwork>& placedNetwork,
    int deviceNum) noexcept {
    if (placedNetwork == nullptr) {
        return true;
    }

    try {
        // Drain only streams owned by the attempted placement. Once these
        // producer streams are complete, the queued completion streams can run
        // their already-enqueued waits and host callbacks; the state cleanup
        // below waits for those callbacks before dropping tensor references.
        placedNetwork->synchronize();
        return true;
    } catch (...) {
        const std::exception_ptr firstFailure = std::current_exception();
        if (!ThorImplementation::isDeviceStartupMemoryFailure(firstFailure)) {
            return false;
        }
    }

    // A failed allocation/launch can leave a recoverable OOM in this startup
    // thread's last-error slot. Consume it and retry the placement-local drain;
    // a device-wide barrier would unnecessarily stop healthy sibling models.
    ThorImplementation::clearDeviceStartupCudaErrorState(deviceNum);
    try {
        placedNetwork->synchronize();
        return true;
    } catch (...) {
        return false;
    }
}

bool abortNativeQueuedEpochExecution(
    NativeQueuedEpochExecution& execution,
    const std::shared_ptr<BatchSession>& effectiveSession,
    const std::shared_ptr<PlacedNetwork>& placedNetwork,
    std::exception_ptr failure,
    int deviceNum) noexcept {
    if (execution.state == nullptr) {
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(execution.state->mutex);
        if (execution.state->failure == nullptr) {
            execution.state->failure = failure;
        }
        execution.state->cancelRequested = true;
        execution.state->numBatchesDoneInEpoch =
            execution.state->numBatchesInEpoch;
    }
    execution.state->batchFinished.notify_all();
    execution.state->batchPopped.notify_all();

    try {
        cancelBatchSession(effectiveSession);
    } catch (...) {
        return false;
    }
    try {
        if (execution.schedulingThread.joinable()) {
            execution.schedulingThread.join();
        }
    } catch (...) {
        return false;
    }

    const bool submittedWorkDrained =
        drainQueuedTrainingWorkAfterFailure(placedNetwork, deviceNum);
    releaseQueuedTrainingStateReferencesAfterAbort(
        execution.state, submittedWorkDrained);
    return submittedWorkDrained;
}

class PendingNativeQueuedEpochExecutionGuard {
   public:
    PendingNativeQueuedEpochExecutionGuard(
        std::optional<NativeQueuedEpochExecution>& execution,
        const std::shared_ptr<BatchSession>& effectiveSession,
        const std::shared_ptr<PlacedNetwork>& placedNetwork,
        int deviceNum)
        : execution(execution),
          effectiveSession(effectiveSession),
          placedNetwork(placedNetwork),
          deviceNum(deviceNum) {}

    ~PendingNativeQueuedEpochExecutionGuard() {
        if (!execution.has_value() ||
            !execution->schedulingThread.joinable()) {
            return;
        }
        (void)abortNativeQueuedEpochExecution(
            execution.value(),
            effectiveSession,
            placedNetwork,
            std::make_exception_ptr(std::runtime_error(
                "Native queued training exited before consuming its admitted "
                "first-epoch queue.")),
            deviceNum);
    }

   private:
    std::optional<NativeQueuedEpochExecution>& execution;
    const std::shared_ptr<BatchSession>& effectiveSession;
    const std::shared_ptr<PlacedNetwork>& placedNetwork;
    int deviceNum;
};

struct NativeQueuedStartupState {
    std::shared_ptr<PlacedNetwork> placedNetwork;
    std::shared_ptr<const ExecutableTrainingPlan> plan;
    std::shared_ptr<BatchSession> sourceSession;
    std::shared_ptr<BatchSession> effectiveSession;
    DeviceDatasetStorageReport deviceDatasetStorageReport;
    std::optional<NativeQueuedEpochExecution> firstEpochExecution;
};

void releaseFailedNativeQueuedStartupAttempt(
    NativeQueuedStartupState& attempt) noexcept {
    attempt.firstEpochExecution.reset();

    // Session leases may own resident dataset and per-session device tensors.
    // Release them before the placed graph, then destroy the executable plan
    // before its physical tensor references become invalid.
    attempt.effectiveSession.reset();
    attempt.sourceSession.reset();
    attempt.plan.reset();

    if (attempt.placedNetwork != nullptr) {
        try {
            // State restoration can enqueue model-specific copies. Drain those
            // before destroying the destination placement; do not synchronize
            // unrelated models sharing the GPU.
            attempt.placedNetwork->synchronize();
        } catch (...) {
            // Preserve the original startup exception. PlacedNetwork teardown is
            // still RAII-safe and occurs while this startup retains the FIFO turn.
        }
        attempt.placedNetwork.reset();
    }
}

std::shared_ptr<BatchSession> reopenNativeQueuedBatchSessionForRetry(
    const TrainingRunRequest& request,
    uint64_t expectedBatchSize) {
    if (!request.batchSessionFactory) {
        throw std::runtime_error(
            "Native queued startup consumed the BatchSession before a retryable "
            "GPU out-of-memory failure, but no batchSessionFactory was supplied "
            "to open a fresh session.");
    }

    std::shared_ptr<BatchSession> session = request.batchSessionFactory();
    if (session == nullptr) {
        throw std::runtime_error(
            "TrainingRunRequest batchSessionFactory returned null while retrying "
            "GPU startup after an out-of-memory failure.");
    }
    if (session->getBatchSize() != expectedBatchSize) {
        throw std::runtime_error(
            "TrainingRunRequest batchSessionFactory returned a session with a "
            "different batch size while retrying GPU startup.");
    }
    return session;
}

NativeQueuedStartupState startNativeQueuedTrainingWithMemoryAdmissionRetry(
    const TrainingRunRequest& request,
    const NativeQueuedExecutionGraph& executionGraph,
    bool evaluateOnly,
    uint64_t batchSize,
    const NativeQueuedTrainingOptions& options,
    const std::vector<std::string>& scalarTensorNames,
    const std::vector<std::string>& aggregateLossTensorNames,
    uint64_t currentEpoch) {
    constexpr int startupDeviceNum = 0;
    std::optional<ThorImplementation::DeviceStartupReservation>
        reservedStartupTurn;
    auto reserveStartupTurn = [&]() {
        if (reservedStartupTurn.has_value()) {
            throw std::logic_error(
                "Initial device startup sequencer invoked its reservation "
                "callback more than once.");
        }
        reservedStartupTurn.emplace(
            ThorImplementation::reserveDeviceStartupTurn(startupDeviceNum));
    };
    if (request.initialDeviceStartupSequencer) {
        request.initialDeviceStartupSequencer(reserveStartupTurn);
    } else {
        reserveStartupTurn();
    }
    if (!reservedStartupTurn.has_value()) {
        throw std::logic_error(
            "Initial device startup sequencer did not invoke its reservation "
            "callback.");
    }

    // The potentially indefinite FIFO/memory wait must happen only after the
    // TrainingRuns declaration-order sequencer has released its mutex.
    ThorImplementation::DeviceStartupGuard startupGuard =
        reservedStartupTurn->acquire();
    bool emptyDeviceRetryAlreadyUsed = false;
    bool forceSourceSession = false;
    bool deviceDatasetFallbackAlreadyUsed = false;
    std::shared_ptr<BatchSession> nextSourceSession = request.batchSession;

    for (;;) {
        request.cancellationToken.throwIfCancellationRequested();
        ThorImplementation::clearDeviceStartupCudaErrorState(startupDeviceNum);

        NativeQueuedStartupState attempt;
        attempt.sourceSession = nextSourceSession;
        attempt.effectiveSession = attempt.sourceSession;
        attempt.deviceDatasetStorageReport = request.deviceDatasetStorageReport;

        std::exception_ptr startupFailure;
        bool sessionWasConsumed = false;
        bool cleanupDrained = true;
        bool retryWithoutDeviceDataset = false;
        try {
            std::vector<Event> initDoneEvents;
            attempt.placedNetwork = executionGraph.network->place(
                batchSize,
                initDoneEvents,
                /*inferenceOnly=*/evaluateOnly);
            THOR_THROW_IF_FALSE(attempt.placedNetwork->getNumStamps() == 1);
            THOR_THROW_IF_FALSE(
                attempt.placedNetwork->getStampedNetwork(0).getGpuNum() ==
                startupDeviceNum);
            for (Event& event : initDoneEvents) {
                request.cancellationToken.throwIfCancellationRequested();
                event.synchronize();
            }

            if (!evaluateOnly && request.previousPlacedNetwork != nullptr) {
                request.cancellationToken.throwIfCancellationRequested();
                // Copying state from a previously trained placement is a
                // phase/replacement boundary. The retained source placement is
                // excluded from retryable-sibling accounting below because this
                // blocked startup itself owns that reference and cannot cause it
                // to be released while waiting.
                request.previousPlacedNetwork->synchronize();
                if (executionGraph.composedFromTrainingPhases) {
                    attempt.placedNetwork->copyMatchingTrainingStateFrom(
                        *request.previousPlacedNetwork);
                } else {
                    attempt.placedNetwork->copyTrainingStateFrom(
                        *request.previousPlacedNetwork);
                }
            }

            if (!evaluateOnly &&
                request.previousModelArtifactDirectory.has_value()) {
                request.cancellationToken.throwIfCancellationRequested();
                if (!request.previousModelNetworkName.has_value() ||
                    request.previousModelNetworkName->empty()) {
                    throw std::runtime_error(
                        "Trainer artifact handoff requires "
                        "previousModelNetworkName.");
                }

                if (executionGraph.composedFromTrainingPhases) {
                    attempt.placedNetwork->loadMatchingTrainingStateFromArtifact(
                        request.previousModelArtifactDirectory.value(),
                        request.previousModelNetworkName.value());
                } else {
                    attempt.placedNetwork
                        ->loadTrainingStateFromSameNetworkArtifact(
                            request.previousModelArtifactDirectory.value(),
                            request.previousModelNetworkName.value());
                }
            }

            request.cancellationToken.throwIfCancellationRequested();
            attempt.placedNetwork->preallocateOutputSlots(
                static_cast<uint32_t>(options.maxInFlightBatches));

            request.cancellationToken.throwIfCancellationRequested();
            attempt.plan = std::make_shared<ExecutableTrainingPlan>(
                ExecutableTrainingPlan::compile(
                    *executionGraph.trainingProgram,
                    *attempt.placedNetwork,
                    /*resolveEmptyUpdateParametersAsAllTrainable=*/!evaluateOnly));
            ensureNativeQueuedPlanCompatible(
                *attempt.plan, *executionGraph.network, evaluateOnly);

            if (!forceSourceSession && !evaluateOnly &&
                request.trainingData != nullptr) {
                const uint64_t deviceDatasetBatchQueueDepth =
                    std::max<uint64_t>(
                        uint64_t{1}, options.maxInFlightBatches);
                DeviceDatasetStorageSelection deviceDatasetSelection =
                    selectDeviceDatasetStorageSession(
                        attempt.sourceSession,
                        *request.trainingData,
                        ThorImplementation::TensorPlacement(
                            ThorImplementation::TensorPlacement::MemDevices::GPU,
                            attempt.placedNetwork
                                ->getStampedNetwork(0)
                                .getGpuNum()),
                        deviceDatasetBatchQueueDepth);
                attempt.effectiveSession =
                    std::move(deviceDatasetSelection.session);
                attempt.deviceDatasetStorageReport =
                    std::move(deviceDatasetSelection.report);
            } else if (forceSourceSession) {
                attempt.deviceDatasetStorageReport.used = false;
                attempt.deviceDatasetStorageReport.reason =
                    "startup_memory_fallback";
            }

            request.cancellationToken.throwIfCancellationRequested();
            attempt.placedNetwork->configureBatchInputSources(
                resolveNetworkInputBatchSources(
                    attempt.effectiveSession, *attempt.plan));
            attempt.placedNetwork->preallocateInputSlots(
                static_cast<uint32_t>(options.maxInFlightBatches));
            attempt.placedNetwork->synchronize();
            ThorImplementation::requireCleanDeviceStartupCudaErrorState(
                startupDeviceNum);

            // Admission is not complete merely because placement and slot
            // preallocation succeeded. Start the actual first epoch while this
            // FIFO startup turn is still held and wait for one real batch to
            // complete successfully. That proves the forward/backward/update
            // path and lazy allocations work. The scheduler continues filling
            // the remaining queue depth concurrently after admission.
            attempt.firstEpochExecution.emplace(
                launchNativeQueuedEpochExecution(
                    request,
                    attempt.placedNetwork,
                    attempt.effectiveSession,
                    attempt.plan,
                    options,
                    scalarTensorNames,
                    aggregateLossTensorNames,
                    currentEpoch,
                    evaluateOnly));
            sessionWasConsumed = true;
            waitForInitialQueueCompletion(
                attempt.firstEpochExecution.value());
            ThorImplementation::requireCleanDeviceStartupCudaErrorState(
                startupDeviceNum);

            startupGuard.complete(*attempt.placedNetwork);
            return attempt;
        } catch (...) {
            startupFailure = std::current_exception();

            if (attempt.firstEpochExecution.has_value()) {
                cleanupDrained = abortNativeQueuedEpochExecution(
                    attempt.firstEpochExecution.value(),
                    attempt.effectiveSession,
                    attempt.placedNetwork,
                    startupFailure,
                    startupDeviceNum);
            } else if (ThorImplementation::isDeviceStartupMemoryFailure(
                           startupFailure)) {
                // Placement/compile/autotune can fail before a scheduler exists.
                // Drain and clear that attempt's recoverable CUDA OOM state too,
                // so the retry begins from the same clean boundary as a warmup
                // failure.
                cleanupDrained = drainQueuedTrainingWorkAfterFailure(
                    attempt.placedNetwork, startupDeviceNum);
            }

            const bool canFallbackFromDeviceDataset =
                !deviceDatasetFallbackAlreadyUsed &&
                !evaluateOnly &&
                request.trainingData != nullptr &&
                request.trainingData->getAccessPolicy().deviceStorage ==
                    DeviceDatasetStorage::BEST_EFFORT &&
                attempt.deviceDatasetStorageReport.used &&
                attempt.effectiveSession != attempt.sourceSession &&
                ThorImplementation::isDeviceStartupMemoryFailure(
                    startupFailure);
            retryWithoutDeviceDataset = canFallbackFromDeviceDataset;
        }

        // Destroy every allocation from this failed attempt while retaining the
        // FIFO startup turn. A resident sibling cannot be released concurrently
        // until waitForModelRelease() atomically releases the coordinator mutex.
        releaseFailedNativeQueuedStartupAttempt(attempt);

        if (!ThorImplementation::isDeviceStartupMemoryFailure(startupFailure)) {
            std::rethrow_exception(startupFailure);
        }
        if (!cleanupDrained) {
            throw std::runtime_error(
                "Native queued startup encountered GPU out-of-memory and could "
                "not drain its submitted work safely; refusing to retry with "
                "callback-owned device tensors still live.");
        }

        auto reopenSessionIfNeeded = [&]() {
            if (sessionWasConsumed) {
                nextSourceSession = reopenNativeQueuedBatchSessionForRetry(
                    request, batchSize);
            }
        };

        if (retryWithoutDeviceDataset) {
            deviceDatasetFallbackAlreadyUsed = true;
            forceSourceSession = true;
            reopenSessionIfNeeded();
            request.cancellationToken.throwIfCancellationRequested();
            continue;
        }

        const Thor::PlacedNetwork* retainedPlacement =
            request.previousPlacedNetwork.get();
        const uint64_t loadedModels = startupGuard.getLoadedModelCount();
        const uint64_t retryableLoadedModels =
            startupGuard.getRetryableLoadedModelCount(retainedPlacement);
        const auto disposition =
            ThorImplementation::decideDeviceStartupMemoryFailureDisposition(
                loadedModels,
                retryableLoadedModels,
                emptyDeviceRetryAlreadyUsed);

        if (disposition == ThorImplementation::
                               DeviceStartupMemoryFailureDisposition::
                                   WAIT_FOR_MODEL_RELEASE) {
            startupGuard.waitForModelRelease(
                [&]() {
                    request.cancellationToken.throwIfCancellationRequested();
                },
                retainedPlacement);
            reopenSessionIfNeeded();
            // Keep the same FIFO turn and retry from a completely fresh
            // placement and, once batch execution began, a fresh session.
            continue;
        }

        if (disposition == ThorImplementation::
                               DeviceStartupMemoryFailureDisposition::
                                   RETRY_EMPTY_DEVICE_ONCE) {
            emptyDeviceRetryAlreadyUsed = true;
            ThorImplementation::prepareDeviceForEmptyStartupRetry(
                startupDeviceNum);
            reopenSessionIfNeeded();
            request.cancellationToken.throwIfCancellationRequested();
            continue;
        }

        // The model failed twice in a clean state with no independently running
        // placement able to release more memory. It genuinely does not fit.
        std::rethrow_exception(startupFailure);
    }
}

}  // namespace

void runNativeQueuedTraining(const TrainingRunRequest& request, TrainingObserver& observer, const NativeQueuedTrainingOptions& options) {
    NativeQueuedSigintScope sigintScope;

    THOR_THROW_IF_FALSE(request.network != nullptr || request.trainingProgram != nullptr);
    THOR_THROW_IF_FALSE(request.batchSession != nullptr);
    THOR_THROW_IF_FALSE(request.epochs > 0);
    THOR_THROW_IF_FALSE(options.maxInFlightBatches >= 1);
    THOR_THROW_IF_FALSE(request.executionMode == TrainingRunExecutionMode::FIT ||
                        request.executionMode == TrainingRunExecutionMode::EVALUATE);
    request.cancellationToken.throwIfCancellationRequested();

    const bool evaluateOnly = request.executionMode == TrainingRunExecutionMode::EVALUATE;
    if (!evaluateOnly && request.checkBestModelEveryEpochs == 0 && !request.earlyCompletionPolicies.empty()) {
        throw std::runtime_error("Trainer early_completion_policies require check_best_model_every_epochs > 0.");
    }
    if (request.maxTrainingBatchesPerEpoch.has_value() && request.maxTrainingBatchesPerEpoch.value() == 0) {
        throw std::runtime_error("Trainer max_training_batches_per_epoch must be >= 1 or None.");
    }

    TrainingRuntimeConfig runtime = request.runtime;

    std::shared_ptr<TrainingProgram> requestedTrainingProgram = defaultTrainingProgramForRequest(request);
    const bool requestedProgramUsesPhases = trainingProgramHasAnyPhase(*requestedTrainingProgram);
    const bool requestedProgramIsImplicitDefault =
        isImplicitDefaultSingleNetworkProgram(*requestedTrainingProgram, request.network);
    if (!evaluateOnly && (!requestedProgramUsesPhases || requestedProgramIsImplicitDefault)) {
        attachPlacementFallbackOptimizerIfNeeded(request, *requestedTrainingProgram);
    }

    NativeQueuedExecutionGraph executionGraph = resolveNativeQueuedExecutionGraph(request, requestedTrainingProgram, evaluateOnly);
    std::shared_ptr<Network> executionNetwork = executionGraph.network;
    if (evaluateOnly && request.evaluationPhase == TrainingEventPhase::UNKNOWN) {
        throw std::runtime_error("Trainer evaluation requires a concrete evaluation phase.");
    }

    const uint64_t batchSize = request.batchSession->getBatchSize();
    const bool modelSelectionEnabled =
        !evaluateOnly && request.checkBestModelEveryEpochs > 0;
    const std::vector<std::string> aggregateLossTensorNames =
        executionGraph.composedFromTrainingPhases
            ? outputBackedReportableLossNames(*executionNetwork)
            : plainTrainingProgramAggregateLossNames(*executionNetwork);
    const bool hasConcreteLossOutput =
        outputNameExists(*executionNetwork, "loss");
    if (executionGraph.composedFromTrainingPhases) {
        filterRuntimeScalarsToExistingExecutionOutputs(
            runtime, *executionNetwork);
    } else {
        filterRuntimeScalarsToActiveTrainingProgramOutputs(
            runtime, *executionNetwork, aggregateLossTensorNames);
    }
    if (!evaluateOnly &&
        (request.saveModelDirectory.has_value() ||
         !request.earlyCompletionPolicies.empty() ||
         modelSelectionEnabled)) {
        const bool concreteLossOutputIsInactiveReportableLoss =
            setFromVector(outputBackedReportableLossNames(*executionNetwork))
                    .count("loss") != 0 &&
            setFromVector(aggregateLossTensorNames).count("loss") == 0;
        if (!aggregateLossTensorNames.empty() ||
            (hasConcreteLossOutput &&
             !concreteLossOutputIsInactiveReportableLoss)) {
            runtime.scalarTensorsToReport.insert("loss");
        }
        // Model selection can use named losses even when they are not part of
        // the human report list. Always collect active graph-loss scalars while
        // best-candidate/early-completion scoring is enabled.
        if (modelSelectionEnabled) {
            runtime.scalarTensorsToReport.insert(
                aggregateLossTensorNames.begin(),
                aggregateLossTensorNames.end());
        }
    }
    if (runtime.scalarTensorsToReport.count("loss") != 0 &&
        !hasConcreteLossOutput && aggregateLossTensorNames.empty()) {
        // The default runtime historically asks for a scalar named "loss". In
        // loss-centric graphs there may be graph losses without a concrete
        // NetworkOutput named "loss"; do not fail merely because the default
        // reporter requested an inactive name.
        runtime.scalarTensorsToReport.erase("loss");
    }

    const std::vector<std::string> scalarTensorNames(
        runtime.scalarTensorsToReport.begin(),
        runtime.scalarTensorsToReport.end());
    uint64_t currentEpoch =
        evaluateOnly ? 0 : request.initialCompletedEpochs;

    NativeQueuedStartupState startup =
        startNativeQueuedTrainingWithMemoryAdmissionRetry(
            request,
            executionGraph,
            evaluateOnly,
            batchSize,
            options,
            scalarTensorNames,
            aggregateLossTensorNames,
            currentEpoch);
    std::shared_ptr<PlacedNetwork> placedNetwork =
        std::move(startup.placedNetwork);
    std::shared_ptr<const ExecutableTrainingPlan> plan =
        std::move(startup.plan);
    THOR_THROW_IF_FALSE(plan != nullptr);
    std::shared_ptr<BatchSession> effectiveSession =
        std::move(startup.effectiveSession);
    DeviceDatasetStorageReport deviceDatasetStorageReport =
        std::move(startup.deviceDatasetStorageReport);
    std::optional<NativeQueuedEpochExecution> firstEpochExecution =
        std::move(startup.firstEpochExecution);
    PendingNativeQueuedEpochExecutionGuard pendingFirstEpochGuard(
        firstEpochExecution,
        effectiveSession,
        placedNetwork,
        placedNetwork->getStampedNetwork(0).getGpuNum());

    ThorImplementation::StampedNetwork& statsStampedNetwork =
        placedNetwork->getStampedNetwork(0);
    const uint64_t forwardFlopsPerBatch =
        statsStampedNetwork.getFloatingPointOperationsPerExampleForward() *
        batchSize;
    const uint64_t trainingFlopsPerBatch =
        statsStampedNetwork.getFloatingPointOperationsPerExampleTraining() *
        batchSize;

    THOR_THROW_IF_FALSE(firstEpochExecution.has_value());
    const auto runStart = firstEpochExecution->startedAt;
    const double initialElapsedSeconds =
        evaluateOnly ? 0.0 : std::max(0.0, request.initialElapsedSeconds);
    std::map<TrainingEventPhase, WallThroughputEmaState> throughputByPhase;
    std::array<uint64_t, 4> cappedReportedStepsByPhase{};
    const bool trainingBatchCapEnabled = !evaluateOnly && request.maxTrainingBatchesPerEpoch.has_value();
    const uint64_t totalRequestedEpochs = currentEpoch + request.epochs;
    auto elapsedSinceRunStart = [&]() {
        const auto now = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - runStart);
        return initialElapsedSeconds + elapsed.count();
    };

    auto makeBaseSnapshot = [&](TrainingEventPhase phase,
                                uint64_t epoch,
                                uint64_t batchSize,
                                uint64_t batchesPerEpoch,
                                const std::shared_ptr<QueuedTrainingState>& state) {
        TrainingStatsSnapshot snapshot;
        snapshot.networkName = placedNetwork->getNetworkName();
        snapshot.datasetName = effectiveSession->getDatasetName();
        snapshot.phase = phase;
        snapshot.epoch = epoch;
        snapshot.epochs = totalRequestedEpochs;
        snapshot.batchSize = batchSize;
        snapshot.stepsPerEpoch = batchesPerEpoch;
        snapshot.elapsedSeconds = elapsedSinceRunStart();
        snapshot.inFlightBatches = state ? outstandingBatchCount(state) : 0;
        snapshot.deviceDatasetStorage = deviceDatasetStorageReport;
        return snapshot;
    };

    TrainingArtifactManager trainingArtifacts(request.saveModelDirectory, request.saveModelOverwrite);
    const uint64_t firstModelSelectionEpoch =
        request.firstModelSelectionEpoch == 0 ? request.checkBestModelEveryEpochs : request.firstModelSelectionEpoch;
    bool runEarlyCompleted = false;
    std::optional<uint64_t> completedEpoch{};
    std::optional<double> latestModelSelectionScore{};
    std::optional<double> latestTrainingLoss{};
    std::optional<double> latestValidationLoss{};
    TrainingModelSelectionContext latestEpochSelectionContext{};
    bool latestEpochSelectionContextValid = false;

    try {
        emitTrainingEvent(
            observer,
            TrainingEvent::runStarted(makeBaseSnapshot(
                TrainingEventPhase::UNKNOWN, 0, batchSize, 0, nullptr)));
    } catch (...) {
        if (firstEpochExecution.has_value()) {
            (void)abortNativeQueuedEpochExecution(
                firstEpochExecution.value(),
                effectiveSession,
                placedNetwork,
                std::current_exception(),
                placedNetwork->getStampedNetwork(0).getGpuNum());
        }
        throw;
    }

    for (uint32_t epochOffset = 0; epochOffset < request.epochs; ++epochOffset) {
        const uint64_t cumulativeEpoch = currentEpoch + 1;
        EpochLossAccumulator epochLosses;

        NativeQueuedEpochExecution epochExecution;
        if (epochOffset == 0) {
            THOR_THROW_IF_FALSE(firstEpochExecution.has_value());
            epochExecution = std::move(firstEpochExecution.value());
            firstEpochExecution.reset();
        } else {
            request.cancellationToken.throwIfCancellationRequested();
            epochExecution = launchNativeQueuedEpochExecution(
                request,
                placedNetwork,
                effectiveSession,
                plan,
                options,
                scalarTensorNames,
                aggregateLossTensorNames,
                currentEpoch,
                evaluateOnly);
        }

        std::shared_ptr<QueuedTrainingState> state = epochExecution.state;
        const std::vector<QueuedEpochPhaseWork>& phaseWorks =
            epochExecution.phaseWorks;

        std::array<bool, 4> phaseStarted{};
        std::array<bool, 4> phaseFinished{};
        size_t lifecyclePhaseIndex = 0;
        auto emitReadyPhaseLifecycleEvents = [&]() {
            while (lifecyclePhaseIndex < phaseWorks.size()) {
                const QueuedEpochPhaseWork& work = phaseWorks[lifecyclePhaseIndex];
                const size_t index = queuedPhaseIndex(work.phase);
                if (!phaseStarted[index]) {
                    phaseStarted[index] = true;
                    emitTrainingEvent(observer,
                                      TrainingEvent::epochStarted(
                                          makeBaseSnapshot(work.phase, cumulativeEpoch, batchSize, work.batchesPerEpoch, state)));
                }

                const QueuedPhaseProgress& progress = phaseProgress(*state, work.phase);
                if (progress.poppedBatches < work.batchesPerEpoch) {
                    break;
                }

                if (!phaseFinished[index]) {
                    phaseFinished[index] = true;
                    emitTrainingEvent(observer,
                                      TrainingEvent::epochFinished(
                                          makeBaseSnapshot(work.phase, cumulativeEpoch, batchSize, work.batchesPerEpoch, state)));
                }
                lifecyclePhaseIndex += 1;
            }
        };

        auto cancelAndJoinScheduler = [&](std::exception_ptr failure) {
            (void)abortNativeQueuedEpochExecution(
                epochExecution,
                effectiveSession,
                placedNetwork,
                failure,
                placedNetwork->getStampedNetwork(0).getGpuNum());
        };

        auto requestExternalCancel = [&]() {
            if (request.cancellationToken.isCancellationRequested()) {
                requestQueuedTrainingCancellation(state);
                cancelBatchSession(effectiveSession);
            }
            if (sigintScope.interrupted()) {
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->cancelRequested = true;
                    state->interruptRequested = true;
                }
                state->batchFinished.notify_all();
                state->batchPopped.notify_all();
                cancelBatchSession(effectiveSession);
            }
        };

        try {
            request.cancellationToken.throwIfCancellationRequested();
            emitReadyPhaseLifecycleEvents();

            while (true) {
                requestExternalCancel();
                BatchPopResult completedBatch = popBatchData(state);
                if (!completedBatch.hasBatch) {
                    break;
                }

                if (shouldEmitQueueDiagnostic(completedBatch.poppedInEpoch)) {
                    emitNativeQueueDiagnostic("pop_return",
                                              completedBatch.phase,
                                              currentEpoch,
                                              completedBatch.epochBatchNum,
                                              completedBatch.slotIndex,
                                              completedBatch.inFlightAfterPop,
                                              completedBatch.poppedInEpoch,
                                              completedBatch.batchesInEpoch);
                }

                {
                    std::exception_ptr failure;
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        failure = state->failure;
                    }
                    if (failure != nullptr) {
                        std::rethrow_exception(failure);
                    }
                }

                {
                    TrainingStatsSnapshot snapshot =
                        makeBaseSnapshot(completedBatch.phase, cumulativeEpoch, batchSize, completedBatch.batchesInEpoch, state);
                    snapshot.inFlightBatches = completedBatch.inFlightAfterPop;
                    snapshot.stepInEpoch = completedBatch.epochBatchNum + 1;
                    if (trainingBatchCapEnabled && completedBatch.phase == TrainingEventPhase::TRAIN) {
                        const size_t phaseIndex = queuedPhaseIndex(completedBatch.phase);
                        snapshot.step = cappedReportedStepsByPhase[phaseIndex] + 1;
                        cappedReportedStepsByPhase[phaseIndex] += 1;
                    } else {
                        snapshot.step = (currentEpoch * completedBatch.batchesInEpoch) + snapshot.stepInEpoch;
                    }
                    snapshot.samplesProcessed = snapshot.step * batchSize;

                    // Public throughput stats use wall-clock time, not the CUDA
                    // completion-callback interval. Smooth the exact wall-clock
                    // interval rate between same-phase stats snapshots so the visible
                    // numbers respond to changes without reverting to active-kernel
                    // micro-throughput. The progress basis matches snapshot.step, so
                    // follow-up fit phases use the same cumulative timer/progress basis
                    // as the reported elapsed time.
                    const uint64_t completedBatchesForThroughput = snapshot.step;
                    const uint64_t phaseFlopsPerBatch =
                        (completedBatch.phase == TrainingEventPhase::TRAIN) ? trainingFlopsPerBatch : forwardFlopsPerBatch;
                    updateWallThroughputRates(snapshot,
                                              throughputByPhase[completedBatch.phase],
                                              completedBatchesForThroughput,
                                              batchSize,
                                              phaseFlopsPerBatch);

                    assignScalarStatsToSnapshot(snapshot,
                                                state->scalarTensorNames,
                                                completedBatch.scalarStats,
                                                state->aggregateLossTensorNames);
                    epochLosses.update(snapshot);
                    emitTrainingEvent(observer, TrainingEvent::statsUpdated(std::move(snapshot)));
                }

                emitReadyPhaseLifecycleEvents();
            }

            if (epochExecution.schedulingThread.joinable()) {
                epochExecution.schedulingThread.join();
            }
            throwIfQueuedTrainingStateFailed(state);
            emitReadyPhaseLifecycleEvents();
        } catch (...) {
            cancelAndJoinScheduler(std::current_exception());
            throw;
        }

        bool earlyCompletionRequested = false;
        const uint64_t phaseLocalEpoch = cumulativeEpoch - request.initialCompletedEpochs;
        const bool modelSelectionEligible = modelSelectionEnabled && phaseLocalEpoch >= firstModelSelectionEpoch &&
                                            ((phaseLocalEpoch - firstModelSelectionEpoch) % request.checkBestModelEveryEpochs == 0);
        if (modelSelectionEligible) {
            const TrainingModelSelectionContext currentSelectionContext = epochLosses.modelSelectionContext(cumulativeEpoch);
            const std::optional<double> currentScore = request.modelSelectionScore.evaluate(currentSelectionContext);
            latestModelSelectionScore = currentScore;
            trainingArtifacts.maybeSnapshotBestCandidate(*placedNetwork, cumulativeEpoch, currentScore);
            const std::optional<double> bestScore = trainingArtifacts.getBestScore();
            const std::optional<uint64_t> bestCumulativeEpoch = trainingArtifacts.getBestEpoch();
            if (currentScore.has_value() && std::isfinite(currentScore.value()) && bestScore.has_value() && bestCumulativeEpoch.has_value()) {
                for (const TrainingEarlyCompletionPolicy& policy : request.earlyCompletionPolicies) {
                    if (policy.shouldComplete(currentScore.value(), bestScore.value(), cumulativeEpoch, bestCumulativeEpoch.value())) {
                        earlyCompletionRequested = true;
                        break;
                    }
                }
            }
        }

        latestTrainingLoss = epochLosses.trainLoss();
        latestValidationLoss = epochLosses.validationLoss();
        latestEpochSelectionContext = epochLosses.modelSelectionContext(cumulativeEpoch);
        latestEpochSelectionContextValid = true;
        currentEpoch += 1;
        if (earlyCompletionRequested) {
            runEarlyCompleted = true;
            completedEpoch = cumulativeEpoch;
            break;
        }
    }

    request.cancellationToken.throwIfCancellationRequested();
    const uint64_t finalCompletedEpoch = completedEpoch.value_or(currentEpoch);
    const char* finalCompletionReason = runEarlyCompleted ? "early_completed" : "completed";
    const uint64_t finalCompletedPhaseEpoch = finalCompletedEpoch - request.initialCompletedEpochs;
    const bool finalModelSelectionEligible = modelSelectionEnabled && finalCompletedPhaseEpoch >= firstModelSelectionEpoch;
    if (finalModelSelectionEligible) {
        // The final/latest state is the handoff and deployment boundary. If best
        // candidate tracking is enabled and the fit has reached the model-selection
        // eligibility threshold, always consider the final state for best even when
        // the final epoch does not fall on the periodic cadence.
        TrainingModelSelectionContext finalSelectionContext;
        if (latestEpochSelectionContextValid) {
            finalSelectionContext = latestEpochSelectionContext;
            finalSelectionContext.epoch = finalCompletedEpoch;
        } else {
            finalSelectionContext.epoch = finalCompletedEpoch;
            finalSelectionContext.train.loss = latestTrainingLoss;
            finalSelectionContext.validate.loss = latestValidationLoss;
        }
        const std::optional<double> finalScore = request.modelSelectionScore.evaluate(finalSelectionContext);
        latestModelSelectionScore = finalScore;
        trainingArtifacts.maybeSnapshotBestCandidate(*placedNetwork, finalCompletedEpoch, finalScore);
    }
    if (!evaluateOnly) {
        TrainingSelectionMetadata selectionMetadata;
        selectionMetadata.bestEpoch = trainingArtifacts.getBestEpoch();
        selectionMetadata.bestScore = trainingArtifacts.getBestScore();
        selectionMetadata.latestEpoch = finalCompletedEpoch;
        selectionMetadata.latestScore = latestModelSelectionScore;
        selectionMetadata.latestTrainingLoss = latestTrainingLoss;
        selectionMetadata.latestValidationLoss = latestValidationLoss;
        selectionMetadata.completedEpoch = finalCompletedEpoch;
        selectionMetadata.completionReason = finalCompletionReason;
        selectionMetadata.checkBestModelEveryEpochs = request.checkBestModelEveryEpochs;
        selectionMetadata.firstModelSelectionEpoch = request.firstModelSelectionEpoch;

        // The epoch counter used by later fit() calls must describe the state
        // that will actually be handed off.  With a saved best candidate, Trainer
        // will reload artifactRoot/best for the next phase, so resume from the
        // selected best epoch rather than from the later epoch where early
        // completion stopped.  Without a saved best artifact, the only reusable
        // state is the latest in-memory placement/latest artifact, so keep the
        // full completed epoch.
        const std::optional<uint64_t> selectedArtifactEpoch =
            trainingArtifacts.hasBestCandidateArtifact() ? trainingArtifacts.getBestEpoch() : std::nullopt;

        const bool persistLatestArtifact =
            request.earlyCompletionPolicies.empty() || !trainingArtifacts.hasBestCandidateArtifact();
        // With early completion enabled, a persisted best candidate is the only
        // state that can be selected for phase handoff or ensemble composition.
        // Serializing the later in-memory state to latest keeps the completed
        // placement resident for the full archive write even though latest will
        // never be consumed. Keep latest only as the fallback when no best
        // candidate exists. Fixed-length training still persists latest.
        trainingArtifacts.finalize(*placedNetwork, selectionMetadata, persistLatestArtifact);
        // fit() completion is a semantic boundary: even when no save_model_dir is
        // configured, callers may immediately reuse, inspect, save, or pass the
        // completed PlacedNetwork into a follow-up phase.  Preserve the pipelined
        // batch path, but drain device work before publishing the final trained
        // state outside this request. This is scoped to the completed model so
        // unrelated models sharing its CUDA device continue running.
        placedNetwork->synchronize();
        if (request.completedPlacedNetwork != nullptr) {
            *request.completedPlacedNetwork = placedNetwork;
        }
        if (request.completedTrainingEpochs != nullptr) {
            *request.completedTrainingEpochs = selectedArtifactEpoch.value_or(finalCompletedEpoch);
        }
        if (request.completedTrainingElapsedSeconds != nullptr) {
            *request.completedTrainingElapsedSeconds = elapsedSinceRunStart();
        }
    }

    TrainingStatsSnapshot finishedStats = makeBaseSnapshot(TrainingEventPhase::UNKNOWN, currentEpoch, batchSize, 0, nullptr);
    finishedStats.metrics["completed_epoch"] = static_cast<double>(finalCompletedEpoch);
    finishedStats.metrics["first_model_selection_epoch"] = static_cast<double>(request.firstModelSelectionEpoch);
    if (trainingArtifacts.getBestEpoch().has_value()) {
        finishedStats.metrics["best_epoch"] = static_cast<double>(trainingArtifacts.getBestEpoch().value());
    }
    if (trainingArtifacts.getBestScore().has_value()) {
        finishedStats.metrics["best_score"] = trainingArtifacts.getBestScore().value();
    }
    emitTrainingEvent(observer,
                      TrainingEvent::runFinished(std::move(finishedStats), finalCompletionReason));
}

}  // namespace Thor
