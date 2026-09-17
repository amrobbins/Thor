#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"
#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunnerTestHooks.h"
#include "DeepLearning/Implementation/Data/Sessions/BatchSessionRuntimeAccess.h"

#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Api/Training/ExecutableTrainingPlan.h"
#include "DeepLearning/Api/Training/MetricEpochAccumulator.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetStorageSelection.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "DeepLearning/Implementation/Training/PhaseWallThroughputTracker.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <atomic>
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
#include <iostream>
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

constexpr const char* kThorNsightControlDirectoryEnvironment = "THOR_NSYS_CONTROL_DIR";
constexpr const char* kThorNsightProfileOutputRequestPrefix = "profile_output_path.";
constexpr const char* kThorNsightProfileReportOutputPrefix = "report_output_path.";
constexpr std::chrono::seconds kThorNsightReportRelocationTimeout{60};
constexpr std::chrono::milliseconds kThorNsightReportRelocationPollInterval{20};

std::atomic<bool>& nsightSystemsCaptureClaimedForProcess() {
    static std::atomic<bool> claimed{false};
    return claimed;
}

std::atomic<uint64_t>& nsightSystemsCaptureSequenceForProcess() {
    static std::atomic<uint64_t> sequence{0};
    return sequence;
}

class NsightSystemsEpochCapture {
   public:
    NsightSystemsEpochCapture(const TrainingRunRequest& request, bool evaluateOnly) {
        if (evaluateOnly || !request.runtime.nsightSystemsProfile.has_value()) {
            return;
        }

        initialCompletedEpochs = request.initialCompletedEpochs;
        for (const NsightSystemsProfileCaptureConfig& config : request.runtime.nsightSystemsProfile->captures) {
            if (config.startEpoch == 0 || config.epochCount == 0) {
                throw std::runtime_error(
                    "Trainer Nsight profile capture startEpoch and epochCount must both be >= 1.");
            }
            if (config.outputPath.empty()) {
                throw std::runtime_error("Trainer Nsight profile capture outputPath must not be empty.");
            }
            if (std::filesystem::path(config.outputPath).extension() != ".nsys-rep") {
                throw std::runtime_error("Trainer Nsight profile capture outputPath must end in '.nsys-rep'.");
            }
            if ((config.epochCount - 1) > std::numeric_limits<uint64_t>::max() - config.startEpoch) {
                throw std::runtime_error("Trainer Nsight profile capture epoch range overflows uint64_t.");
            }
            const uint64_t endPhaseEpoch = config.startEpoch + config.epochCount - 1;
            if (endPhaseEpoch > request.epochs) {
                throw std::runtime_error(
                    "Trainer Nsight profile epoch range is relative to the current training phase and must be "
                    "fully contained in this fit; requested phase epochs " +
                    std::to_string(config.startEpoch) + ".." + std::to_string(endPhaseEpoch) +
                    ", current phase has " + std::to_string(request.epochs) + " epochs.");
            }
            CaptureState state;
            state.config = config;
            state.endPhaseEpoch = endPhaseEpoch;
            state.outputPath = std::filesystem::absolute(std::filesystem::path(config.outputPath)).lexically_normal();
            if (std::filesystem::exists(state.outputPath)) {
                throw std::runtime_error(
                    "Trainer Nsight profile output already exists: '" + state.outputPath.string() + "'.");
            }
            captures.push_back(std::move(state));
        }
        std::sort(captures.begin(), captures.end(), [](const CaptureState& lhs, const CaptureState& rhs) {
            return lhs.config.startEpoch < rhs.config.startEpoch;
        });
        for (size_t i = 1; i < captures.size(); ++i) {
            if (captures[i].config.startEpoch <= captures[i - 1].endPhaseEpoch) {
                throw std::runtime_error(
                    "Trainer Nsight profile capture windows for one training phase must not overlap.");
            }
        }

        const char* controlDirectory = std::getenv(kThorNsightControlDirectoryEnvironment);
        if (controlDirectory == nullptr || controlDirectory[0] == '\0') {
            throw std::runtime_error(
                "Trainer Nsight profiling requires launching the process with `thor-nsys-profile`; "
                "that launcher prepares Nsight Systems before CUDA work begins.");
        }
        controlDirectoryPath = std::filesystem::path(controlDirectory);
        if (!std::filesystem::exists(controlDirectoryPath) ||
            !std::filesystem::is_directory(controlDirectoryPath)) {
            throw std::runtime_error(
                "Trainer Nsight profiling received an invalid THOR_NSYS_CONTROL_DIR from `thor-nsys-profile`.");
        }
    }

    NsightSystemsEpochCapture(const NsightSystemsEpochCapture&) = delete;
    NsightSystemsEpochCapture& operator=(const NsightSystemsEpochCapture&) = delete;

    ~NsightSystemsEpochCapture() { stopNoThrow(); }

    void beginEpoch(uint64_t cumulativeEpoch) {
        const std::optional<uint64_t> phaseEpoch = phaseEpochFor(cumulativeEpoch);
        if (!phaseEpoch.has_value() || activeCaptureIndex.has_value()) {
            return;
        }
        for (size_t captureIndex = 0; captureIndex < captures.size(); ++captureIndex) {
            CaptureState& capture = captures[captureIndex];
            if (capture.attempted || capture.config.startEpoch != phaseEpoch.value()) {
                continue;
            }
            capture.attempted = true;

            bool expected = false;
            if (!nsightSystemsCaptureClaimedForProcess().compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                std::fprintf(
                    stderr,
                    "Thor: Nsight profile request for training-phase epoch %lu ignored because another Trainer "
                    "is currently using the process-wide profiling session.\n",
                    static_cast<unsigned long>(phaseEpoch.value()));
                return;
            }

            try {
                const uint64_t sequence =
                    nsightSystemsCaptureSequenceForProcess().fetch_add(1, std::memory_order_relaxed);
                std::ostringstream requestFileName;
                requestFileName << kThorNsightProfileOutputRequestPrefix << std::setfill('0') << std::setw(20)
                                << sequence;
                capture.requestFile = controlDirectoryPath / requestFileName.str();
                std::ostringstream reportMarkerFileName;
                reportMarkerFileName << kThorNsightProfileReportOutputPrefix << std::setfill('0') << std::setw(20)
                                     << sequence;
                capture.reportMarkerFile = controlDirectoryPath / reportMarkerFileName.str();
                std::ofstream outputRequest(capture.requestFile, std::ios::out | std::ios::trunc);
                if (!outputRequest.is_open()) {
                    throw std::runtime_error(
                        "could not create Nsight output request file '" + capture.requestFile.string() + "'");
                }
                outputRequest << capture.outputPath.string() << '\n';
                outputRequest.flush();
                if (!outputRequest.good()) {
                    throw std::runtime_error(
                        "could not write Nsight output request file '" + capture.requestFile.string() + "'");
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr,
                             "Thor: Nsight profiling disabled before training-phase epoch %lu: %s. Training will "
                             "continue.\n",
                             static_cast<unsigned long>(phaseEpoch.value()),
                             error.what());
                nsightSystemsCaptureClaimedForProcess().store(false, std::memory_order_release);
                return;
            }

            const cudaError_t status = cudaProfilerStart();
            if (status != cudaSuccess) {
                std::fprintf(stderr,
                             "Thor: cudaProfilerStart() failed before training-phase epoch %lu: %s. Training will "
                             "continue without this profile capture.\n",
                             static_cast<unsigned long>(phaseEpoch.value()),
                             cudaGetErrorString(status));
                std::error_code ignored;
                std::filesystem::remove(capture.requestFile, ignored);
                nsightSystemsCaptureClaimedForProcess().store(false, std::memory_order_release);
                return;
            }
            capture.started = true;
            activeCaptureIndex = captureIndex;
            std::fprintf(stderr,
                         "Thor: Nsight Systems capture started at training-phase epoch %lu (cumulative epoch %lu); "
                         "report will be written to '%s'.\n",
                         static_cast<unsigned long>(phaseEpoch.value()),
                         static_cast<unsigned long>(cumulativeEpoch),
                         capture.outputPath.string().c_str());
            return;
        }
    }

    void endEpoch(uint64_t cumulativeEpoch) {
        const std::optional<uint64_t> phaseEpoch = phaseEpochFor(cumulativeEpoch);
        if (!activeCaptureIndex.has_value() || !phaseEpoch.has_value()) {
            return;
        }
        const CaptureState& capture = captures[activeCaptureIndex.value()];
        if (phaseEpoch.value() == capture.endPhaseEpoch) {
            stopNoThrow();
        }
    }

   private:
    struct CaptureState {
        NsightSystemsProfileCaptureConfig config{};
        std::filesystem::path outputPath{};
        std::filesystem::path requestFile{};
        std::filesystem::path reportMarkerFile{};
        uint64_t endPhaseEpoch = 0;
        bool attempted = false;
        bool started = false;
    };

    [[nodiscard]] std::optional<uint64_t> phaseEpochFor(uint64_t cumulativeEpoch) const {
        if (cumulativeEpoch <= initialCompletedEpochs) {
            return std::nullopt;
        }
        return cumulativeEpoch - initialCompletedEpochs;
    }

    void stopNoThrow() noexcept {
        if (!activeCaptureIndex.has_value()) {
            return;
        }
        const size_t captureIndex = activeCaptureIndex.value();
        CaptureState& capture = captures[captureIndex];
        activeCaptureIndex.reset();
        if (!capture.started) {
            nsightSystemsCaptureClaimedForProcess().store(false, std::memory_order_release);
            return;
        }
        capture.started = false;
        const cudaError_t status = cudaProfilerStop();
        if (status != cudaSuccess) {
            nsightSystemsCaptureClaimedForProcess().store(false, std::memory_order_release);
            std::fprintf(stderr,
                         "Thor: cudaProfilerStop() failed: %s. Training will continue; the Nsight report may be "
                         "incomplete.\n",
                         cudaGetErrorString(status));
            return;
        }

        // thor-nsys-profile launches Nsight with --capture-range-end=repeat:sync,
        // so cudaProfilerStop() does not return until Nsight has finalized the
        // current .nsys-rep. The report-ready callback relocates that finalized
        // report to the Trainer-requested destination and publishes this marker.
        // Wait for that final relocation too, so the next training epoch/phase
        // cannot begin while report processing is still outstanding.
        const auto relocationDeadline = std::chrono::steady_clock::now() + kThorNsightReportRelocationTimeout;
        while (!capture.reportMarkerFile.empty() &&
               !std::filesystem::exists(capture.reportMarkerFile) &&
               std::chrono::steady_clock::now() < relocationDeadline) {
            std::this_thread::sleep_for(kThorNsightReportRelocationPollInterval);
        }

        const bool relocationCompleted =
            !capture.reportMarkerFile.empty() && std::filesystem::exists(capture.reportMarkerFile);
        nsightSystemsCaptureClaimedForProcess().store(false, std::memory_order_release);
        if (!relocationCompleted) {
            std::fprintf(
                stderr,
                "Thor: Nsight Systems finalized the capture after training-phase epoch %lu, but the report "
                "relocation callback did not complete within %lld seconds. Training will continue; inspect '%s' "
                "for the finalized report and pending output request.\n",
                static_cast<unsigned long>(capture.endPhaseEpoch),
                static_cast<long long>(kThorNsightReportRelocationTimeout.count()),
                controlDirectoryPath.string().c_str());
            return;
        }

        std::fprintf(stderr,
                     "Thor: Nsight Systems capture stopped after training-phase epoch %lu and the finalized report "
                     "is ready at '%s'. Training continues normally.\n",
                     static_cast<unsigned long>(capture.endPhaseEpoch),
                     capture.outputPath.string().c_str());
    }

    std::vector<CaptureState> captures{};
    std::filesystem::path controlDirectoryPath{};
    uint64_t initialCompletedEpochs = 0;
    std::optional<size_t> activeCaptureIndex{};
};

struct NativeQueuedSchedulerResourceDiagnosticsState {
    std::mutex mutex;
    std::atomic<bool> enabled{false};
    detail::NativeQueuedSchedulerResourceDiagnosticsForTests snapshot;
    const void* firstSchedulingWindowResourceIdentity = nullptr;
    bool multipleSchedulingWindowResourceIdentities = false;
    const void* firstSchedulingWindowRunStateIdentity = nullptr;
    bool multipleSchedulingWindowRunStateIdentities = false;
    const void* firstSchedulingWindowSlotStorageIdentity = nullptr;
    bool multipleSchedulingWindowSlotStorageIdentities = false;
    std::optional<std::thread::id> firstWorkerThreadId;
    bool multipleWorkerThreadIds = false;
};

NativeQueuedSchedulerResourceDiagnosticsState&
nativeQueuedSchedulerResourceDiagnosticsState() {
    static NativeQueuedSchedulerResourceDiagnosticsState state;
    return state;
}

void recordNativeQueuedSchedulerResourceConstructionForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot.resourceConstructionCount += 1;
}

void recordNativeQueuedRunStateConstructionForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot.runStateConstructionCount += 1;
}

void recordNativeQueuedSchedulingWindowLaunchForTests(
    const void* resourceIdentity,
    const void* runStateIdentity,
    const void* slotStorageIdentity) {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot.schedulingWindowCount += 1;
    if (diagnostics.firstSchedulingWindowResourceIdentity == nullptr) {
        diagnostics.firstSchedulingWindowResourceIdentity = resourceIdentity;
    } else if (diagnostics.firstSchedulingWindowResourceIdentity !=
               resourceIdentity) {
        diagnostics.multipleSchedulingWindowResourceIdentities = true;
    }
    diagnostics.snapshot.distinctResourceInstancesObserved =
        diagnostics.multipleSchedulingWindowResourceIdentities ? 2 : 1;

    if (diagnostics.firstSchedulingWindowRunStateIdentity == nullptr) {
        diagnostics.firstSchedulingWindowRunStateIdentity = runStateIdentity;
    } else if (diagnostics.firstSchedulingWindowRunStateIdentity !=
               runStateIdentity) {
        diagnostics.multipleSchedulingWindowRunStateIdentities = true;
    }
    diagnostics.snapshot.distinctRunStateInstancesObserved =
        diagnostics.multipleSchedulingWindowRunStateIdentities ? 2 : 1;

    if (diagnostics.firstSchedulingWindowSlotStorageIdentity == nullptr) {
        diagnostics.firstSchedulingWindowSlotStorageIdentity = slotStorageIdentity;
    } else if (diagnostics.firstSchedulingWindowSlotStorageIdentity !=
               slotStorageIdentity) {
        diagnostics.multipleSchedulingWindowSlotStorageIdentities = true;
    }
    diagnostics.snapshot.slotStorageStableAcrossSchedulingWindows =
        !diagnostics.multipleSchedulingWindowSlotStorageIdentities;
}

void recordNativeQueuedHostDecisionBarrierForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot.hostDecisionBarrierCount += 1;
}

void recordNativeQueuedBatchSubmissionForTests(uint64_t optimizerEpoch) {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot.submittedBatchCount += 1;
    if (!diagnostics.snapshot.hasSubmittedBatch ||
        optimizerEpoch > diagnostics.snapshot.maxOptimizerEpochSubmitted) {
        diagnostics.snapshot.maxOptimizerEpochSubmitted = optimizerEpoch;
    }
    diagnostics.snapshot.hasSubmittedBatch = true;
}

void recordNativeQueuedSchedulerWorkerThreadStartForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot.workerThreadStartCount += 1;
    const std::thread::id workerThreadId = std::this_thread::get_id();
    if (!diagnostics.firstWorkerThreadId.has_value()) {
        diagnostics.firstWorkerThreadId = workerThreadId;
    } else if (diagnostics.firstWorkerThreadId.value() != workerThreadId) {
        diagnostics.multipleWorkerThreadIds = true;
    }
    diagnostics.snapshot.distinctWorkerThreadsObserved =
        diagnostics.multipleWorkerThreadIds ? 2 : 1;
}

void recordNativeQueuedSchedulerEventReuseForTests(
    uint64_t processingFinishedEventId,
    uint64_t completionFinishedEventId) {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    if (!diagnostics.enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(diagnostics.mutex);

    if (processingFinishedEventId != 0) {
        if (diagnostics.snapshot.firstProcessingFinishedEventId == 0) {
            diagnostics.snapshot.firstProcessingFinishedEventId =
                processingFinishedEventId;
        } else if (diagnostics.snapshot.firstProcessingFinishedEventId !=
                   processingFinishedEventId) {
            diagnostics.snapshot
                .processingFinishedEventIdStableAcrossSchedulingWindows = false;
        }
    }
    if (completionFinishedEventId != 0) {
        if (diagnostics.snapshot.firstCompletionFinishedEventId == 0) {
            diagnostics.snapshot.firstCompletionFinishedEventId =
                completionFinishedEventId;
        } else if (diagnostics.snapshot.firstCompletionFinishedEventId !=
                   completionFinishedEventId) {
            diagnostics.snapshot
                .completionFinishedEventIdStableAcrossSchedulingWindows = false;
        }
    }
}

struct NativeQueuedSchedulingWindowState;
struct QueuedWorkSegmentState;

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
    std::shared_ptr<NativeQueuedSchedulingWindowState> state;
    std::shared_ptr<QueuedWorkSegmentState> segment;
    bool completionCallbackLaunched = false;
    bool completionCallbackFinished = false;
    uint64_t epochBatchNum = 0;
    uint64_t validExampleCount = 0;
    std::optional<ThorImplementation::LogicalWorkCount> logicalWork =
        ThorImplementation::LogicalWorkCount{};
    uint64_t slotIndex = 0;
    BatchLease batchLease;
    std::map<std::string, ThorImplementation::Tensor> batchOutput;
    std::vector<ScalarStatSlot> scalarStats;
    std::vector<ScalarStatSource> scalarStatSources;
    std::map<std::string, ThorImplementation::MetricBatchStatisticTensors> metricStatisticTensors;
    std::unordered_map<std::string, MetricBatchStat> metricBatchStats;
};

struct QueuedBatchSlot {
    bool occupied = false;
    bool ready = false;
    std::shared_ptr<QueuedWorkSegmentState> segment;
    uint64_t epochBatchNum = 0;
    uint64_t validExampleCount = 0;
    std::optional<ThorImplementation::LogicalWorkCount> logicalWork =
        ThorImplementation::LogicalWorkCount{};
    uint64_t doneInEpochAtComplete = 0;
    uint64_t validExamplesThroughBatch = 0;
    uint64_t paramsIndex = 0;
    std::chrono::high_resolution_clock::time_point phaseStartedAt{};
    std::chrono::high_resolution_clock::time_point completionTime{};
    std::vector<ScalarStatSlot> scalarStats;
    std::unordered_map<std::string, MetricBatchStat> metricBatchStats;
};

// One ordered logical unit of native queued work.  Segment identity and
// progress travel with queued batches instead of being recovered from a
// global phase lookup. A scheduling window may contain segments from multiple
// epochs while each slot still updates exactly one segment.
struct QueuedWorkSegmentState {
    std::shared_ptr<BatchSession> batchSession;
    ExampleType exampleType = ExampleType::TRAIN;
    TrainingEventPhase phase = TrainingEventPhase::TRAIN;
    uint64_t optimizerEpoch = 0;
    uint64_t reportedEpoch = 0;
    std::optional<std::string> validationPopulation;
    bool isDefaultValidationPopulation = false;
    std::optional<uint64_t> maxBatchesToRun;

    // Mutable BatchSession cursor-derived fields are intentionally resolved by
    // the scheduler worker immediately before this segment is scheduled. Do not
    // freeze them while a future scheduling window is built on the control thread.
    bool prepared = false;
    uint64_t initialBatchNum = 0;
    uint64_t sessionInitialBatchNum = 0;
    uint64_t batchesToRunCount = 0;
    uint64_t batchesPerEpoch = 0;
    uint64_t initialValidExamples = 0;
    uint64_t validExamplesPerEpoch = 0;
    bool requiresEpochBoundaryValidation = true;

    uint64_t completedBatches = 0;
    uint64_t poppedBatches = 0;
    uint64_t completedValidExamples = 0;
    uint64_t poppedValidExamples = 0;
    std::chrono::high_resolution_clock::time_point schedulingStartedAt{};
};

// Queue-slot storage is a placed-run resource. Scheduling windows borrow this
// ring. Commands remain non-overlapping, but a single command may span ordinary
// epoch boundaries without draining the ring between them.
struct NativeQueuedRunState {
    NativeQueuedRunState(uint64_t maxInFlightBatches,
                         std::vector<std::string> scalarTensorNames,
                         std::vector<std::string> aggregateLossTensorNames)
        : scalarTensorNames(std::move(scalarTensorNames)),
          aggregateLossTensorNames(std::move(aggregateLossTensorNames)),
          slots(maxInFlightBatches),
          completionParams(maxInFlightBatches) {
        THOR_THROW_IF_FALSE(maxInFlightBatches >= 1);
        for (size_t i = 0; i < this->scalarTensorNames.size(); ++i) {
            THOR_THROW_IF_FALSE(
                scalarTensorIndexByName.emplace(this->scalarTensorNames[i], i).second);
        }
        for (QueuedBatchSlot& slot : slots) {
            slot.scalarStats.resize(this->scalarTensorNames.size());
        }
        for (NativeBatchCompletionParams& params : completionParams) {
            params.scalarStats.resize(this->scalarTensorNames.size());
            params.scalarStatSources.resize(
                this->scalarTensorNames.size(),
                ScalarStatSource::UNRESOLVED);
        }
        recordNativeQueuedRunStateConstructionForTests();
    }

    std::mutex mutex;
    std::condition_variable batchFinished;
    std::condition_variable batchPopped;

    std::vector<std::string> scalarTensorNames;
    std::unordered_map<std::string, size_t> scalarTensorIndexByName;
    std::vector<std::string> aggregateLossTensorNames;
    std::vector<QueuedBatchSlot> slots;
    std::vector<NativeBatchCompletionParams> completionParams;
    uint64_t headSlot = 0;
    uint64_t tailSlot = 0;
    uint64_t inFlightBatches = 0;

    // Failure/cancellation terminate the placed run, so they belong to the
    // same run-scoped state as the queue rather than to one logical epoch.
    std::exception_ptr failure;
    bool cancelRequested = false;
    bool interruptRequested = false;
};

struct NativeQueuedSchedulingWindowState {
    explicit NativeQueuedSchedulingWindowState(
        std::shared_ptr<NativeQueuedRunState> runState)
        : runState(std::move(runState)) {
        THOR_THROW_IF_FALSE(this->runState != nullptr);
    }

    std::shared_ptr<NativeQueuedRunState> runState;
    std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments;
    uint64_t completedBatchCallbacks = 0;
    bool schedulingFinished = false;
};

NativeQueuedRunState& queuedRunState(NativeQueuedSchedulingWindowState& state) {
    THOR_THROW_IF_FALSE(state.runState != nullptr);
    return *state.runState;
}

const NativeQueuedRunState& queuedRunState(
    const NativeQueuedSchedulingWindowState& state) {
    THOR_THROW_IF_FALSE(state.runState != nullptr);
    return *state.runState;
}
std::optional<ThorImplementation::LogicalWorkCount> bestEffortCurrentBatchLogicalWork(
    ThorImplementation::StampedNetwork& stampedNetwork,
    TrainingEventPhase phase,
    uint64_t validExampleCount) noexcept {
    try {
        return phase == TrainingEventPhase::TRAIN
            ? stampedNetwork.getLogicalWorkCurrentBatchTraining(validExampleCount)
            : stampedNetwork.getLogicalWorkCurrentBatchForward(validExampleCount);
    } catch (...) {
        // Logical-work accounting is telemetry. A counting overflow or any other
        // diagnostic-only failure must never turn a successfully submitted batch
        // into a failed training job.
        return std::nullopt;
    }
}


void requestQueuedTrainingCancellation(const std::shared_ptr<NativeQueuedSchedulingWindowState>& state) {
    if (state == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state->runState->mutex);
        state->runState->cancelRequested = true;
    }
    state->runState->batchFinished.notify_all();
    state->runState->batchPopped.notify_all();
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
    explicit GpuSubmitCoordinator(int gpuNum) : gpuNum(gpuNum) {
        // Do not start the worker from a member initializer. Member construction
        // follows declaration order, and the worker may begin executing
        // immediately. It must not observe mutex/cv/queue/stopping before those
        // members have finished construction.
        worker = std::thread(&GpuSubmitCoordinator::workerLoop, this);
    }

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
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> queue;
    bool stopping = false;
    // Keep the worker last as an additional construction-order safeguard. It is
    // started only from the constructor body after all state above exists.
    std::thread worker;
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
            const std::string population = snapshot.validationPopulation.empty()
                                               ? std::string("validate")
                                               : snapshot.validationPopulation;
            validations[population].update(snapshot);
        } else if (snapshot.phase == TrainingEventPhase::TEST) {
            test.update(snapshot);
        }
    }

    void ensureValidationPopulation(const std::string& population) {
        validations.try_emplace(population);
    }

    [[nodiscard]] std::optional<double> trainLoss() const { return train.snapshot().loss; }
    [[nodiscard]] std::optional<double> validationLoss(const std::string& defaultPopulation) const {
        const auto it = validations.find(defaultPopulation);
        return it == validations.end() ? std::optional<double>{} : it->second.snapshot().loss;
    }

    [[nodiscard]] TrainingModelSelectionContext modelSelectionContext(
        uint64_t epoch,
        const std::string& defaultPopulation) const {
        TrainingModelSelectionContext context;
        context.epoch = epoch;
        context.train = train.snapshot();
        context.defaultValidationPopulation = defaultPopulation;
        for (const auto& [name, accumulator] : validations) {
            context.validations.emplace(name, accumulator.snapshot());
        }
        const auto defaultIt = context.validations.find(defaultPopulation);
        if (defaultIt != context.validations.end()) {
            context.validate = defaultIt->second;
        }
        context.test = test.snapshot();
        return context;
    }

   private:
    struct RunningMean {
        double weightedSum = 0.0;
        uint64_t totalWeight = 0;

        void add(double value, uint64_t weight) {
            THOR_THROW_IF_FALSE(weight > 0);
            weightedSum += value * static_cast<double>(weight);
            totalWeight += weight;
        }

        [[nodiscard]] std::optional<double> mean() const {
            if (totalWeight == 0) {
                return std::nullopt;
            }
            return weightedSum / static_cast<double>(totalWeight);
        }
    };

    struct PhaseAccumulator {
        RunningMean lossMean{};
        std::unordered_map<std::string, RunningMean> lossMeans{};
        MetricEpochAccumulatorMap metricAccumulators{};

        void update(const TrainingStatsSnapshot& snapshot) {
            THOR_THROW_IF_FALSE(snapshot.validExamplesInBatch > 0);
            const uint64_t weight = snapshot.validExamplesInBatch;
            if (snapshot.loss.has_value()) {
                lossMean.add(snapshot.loss.value(), weight);
            }
            for (const auto& [name, value] : snapshot.losses) {
                lossMeans[name].add(value, weight);
            }
            for (const auto& [name, value] : snapshot.metrics) {
                metricAccumulators.add(
                    name, resolveMetricBatchStat(snapshot, name, value));
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
            out.metrics = metricAccumulators.values();
            return out;
        }
    };

    PhaseAccumulator train{};
    std::map<std::string, PhaseAccumulator> validations{};
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
        if (bestCandidate.has_value() && bestCandidate->directory.has_value()) {
            std::error_code errorCode;
            std::filesystem::remove_all(bestCandidate->directory.value(), errorCode);
        }
    }

    [[nodiscard]] bool enabled() const { return saveModelDirectory.has_value(); }

    void clearBestCandidate() {
        if (bestCandidate.has_value() && bestCandidate->directory.has_value()) {
            removePathIfExists(bestCandidate->directory.value());
        }
        bestCandidate.reset();
    }

    void maybeSnapshotBestCandidate(PlacedNetwork& placedNetwork,
                                    const TrainingModelSelectionContext& context,
                                    std::optional<double> score) {
        if (!score.has_value() || !std::isfinite(score.value())) {
            return;
        }
        if (bestCandidate.has_value() && score.value() >= bestCandidate->score) {
            return;
        }

        BestCandidate nextCandidate;
        nextCandidate.epoch = context.epoch;
        nextCandidate.score = score.value();
        nextCandidate.context = context;
        if (!enabled()) {
            bestCandidate = std::move(nextCandidate);
            return;
        }

        const std::filesystem::path newCandidate = uniqueCandidateDirectory(context.epoch);
        const std::filesystem::path tmpCandidate = uniqueTemporaryDirectory(context.epoch);
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

        if (bestCandidate.has_value() && bestCandidate->directory.has_value()) {
            removePathIfExists(bestCandidate->directory.value());
        }
        nextCandidate.directory = newCandidate;
        bestCandidate = std::move(nextCandidate);
    }

    void finalize(PlacedNetwork& placedNetwork,
                  const TrainingSelectionMetadata& metadata,
                  bool persistLatestArtifact) {
        if (!enabled()) {
            return;
        }
        if (!persistLatestArtifact && !hasBestCandidateArtifact()) {
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

            if (hasBestCandidateArtifact()) {
                const std::filesystem::path bestDirectory = finalTemporaryDirectory / "best";
                removePathIfExists(bestDirectory);
                std::filesystem::rename(bestCandidate->directory.value(), bestDirectory);
                bestCandidate->directory.reset();
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

    [[nodiscard]] std::optional<double> getBestScore() const {
        return bestCandidate.has_value() ? std::optional<double>(bestCandidate->score) : std::nullopt;
    }
    [[nodiscard]] std::optional<uint64_t> getBestEpoch() const {
        return bestCandidate.has_value() ? std::optional<uint64_t>(bestCandidate->epoch) : std::nullopt;
    }
    [[nodiscard]] std::optional<TrainingModelSelectionContext> getBestModelSelectionContext() const {
        return bestCandidate.has_value() ? std::optional<TrainingModelSelectionContext>(bestCandidate->context) : std::nullopt;
    }
    [[nodiscard]] bool hasBestCandidateArtifact() const {
        return bestCandidate.has_value() && bestCandidate->directory.has_value();
    }

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

    struct BestCandidate {
        uint64_t epoch = 0;
        double score = 0.0;
        TrainingModelSelectionContext context{};
        std::optional<std::filesystem::path> directory{};
    };

    std::optional<std::string> saveModelDirectory{};
    bool overwrite = false;
    std::optional<BestCandidate> bestCandidate{};
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
    const std::vector<std::string> logicalInputNames = network.getExternalNetworkInputNames();
    const std::set<std::string> externalInputNames(logicalInputNames.begin(), logicalInputNames.end());

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

void configureWrappedTailFallback(
    const std::shared_ptr<PlacedNetwork>& placedNetwork,
    const std::shared_ptr<BatchSession>& sourceSession,
    std::vector<NamedValidationSession>& validationSessions,
    bool& warningEmitted) {
    THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    THOR_THROW_IF_FALSE(sourceSession != nullptr);
    const std::vector<ThorImplementation::PartialBatchIncompatibility> incompatibilities =
        placedNetwork->getPartialBatchIncompatibilities();
    if (incompatibilities.empty()) {
        return;
    }

    auto configure = [](const std::shared_ptr<BatchSession>& session,
                        const std::string& context) {
        THOR_THROW_IF_FALSE(session != nullptr);
        try {
            ThorImplementation::BatchSessionRuntimeAccess::setTailMode(
                *session, ThorImplementation::BatchTailMode::WRAP);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Thor cannot fall back to legacy wrapped-tail batching for " +
                context + ": " + e.what());
        }
    };
    configure(sourceSession, "dataset '" + sourceSession->getDatasetName() + "'");
    for (NamedValidationSession& validation : validationSessions) {
        configure(
            validation.batchSession,
            "validation population '" + validation.name + "'");
    }

    if (warningEmitted) {
        return;
    }
    warningEmitted = true;
    std::cerr
        << "Thor warning: exact partial tail batches are not compatible with this network.\n"
        << "The following layers/metrics require full batches:\n";
    for (const ThorImplementation::PartialBatchIncompatibility& incompatibility :
         incompatibilities) {
        std::cerr << "  - layer " << incompatibility.layerId << " '"
                  << (incompatibility.layerName.empty()
                          ? std::string("<unnamed>")
                          : incompatibility.layerName)
                  << "' (" << incompatibility.layerType << ")\n";
    }
    std::cerr
        << "Thor is reverting this run to legacy wrapped full-batch epochs. "
        << "The final batch may consume examples from the beginning of the next "
        << "dataset traversal; the next epoch continues from there. Those wrapped "
        << "examples participate fully in losses, metrics, and optimizer updates. "
        << "Make every listed layer "
        << "partial-batch compatible to restore exact epoch semantics.\n";
}

void cancelBatchSession(const std::shared_ptr<BatchSession>& batchSession) {
    if (batchSession != nullptr) {
        batchSession->cancel();
    }
}

Batch bindBatchInputs(const StepExecutable& step, const Batch& batchInput) {
    Batch bound;
    const std::optional<uint32_t> validExampleCount = batchInput.getValidExampleCount();
    if (validExampleCount.has_value()) {
        bound.setValidExampleCount(validExampleCount.value());
    }
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
    std::shared_ptr<NativeQueuedSchedulingWindowState> state = params->state;
    if (state == nullptr) {
        return;
    }
    const std::shared_ptr<QueuedWorkSegmentState> segment = params->segment;
    if (segment == nullptr) {
        return;
    }
    const TrainingEventPhase phase = segment->phase;
    const uint64_t epochBatchNum = params->epochBatchNum;
    const uint64_t slotIndex = params->slotIndex;

    try {
        THOR_THROW_IF_FALSE(params->scalarStats.size() == state->runState->scalarTensorNames.size());
        THOR_THROW_IF_FALSE(
            params->scalarStatSources.size() == state->runState->scalarTensorNames.size());
        for (size_t i = 0; i < state->runState->scalarTensorNames.size(); ++i) {
            if (params->scalarStats[i].present) {
                continue;
            }
            const std::string& tensorName = state->runState->scalarTensorNames[i];
            if (params->scalarStatSources[i] == ScalarStatSource::INPUT) {
                THOR_THROW_IF_FALSE(!params->batchLease.empty());
                params->scalarStats[i].value = copyInputScalarStatTensor(
                    params->batchLease.get(),
                    tensorName);
            } else if (params->scalarStatSources[i] == ScalarStatSource::OUTPUT) {
                params->scalarStats[i].value = copyOutputScalarStatTensor(
                    params->batchOutput,
                    tensorName,
                    state->runState->aggregateLossTensorNames);
            } else {
                throw std::runtime_error(
                    "Training stat tensor '" + tensorName +
                    "' did not resolve to an input or output source.");
            }
            params->scalarStats[i].present = true;
        }

        params->metricBatchStats.clear();
        for (const auto& [metricName, tensors] : params->metricStatisticTensors) {
            const auto scalarIndex = state->runState->scalarTensorIndexByName.find(metricName);
            THOR_THROW_IF_FALSE(scalarIndex != state->runState->scalarTensorIndexByName.end());
            THOR_THROW_IF_FALSE(params->scalarStats[scalarIndex->second].present);

            MetricBatchStat statistic;
            statistic.aggregation = tensors.aggregation;
            statistic.value = static_cast<double>(
                params->scalarStats[scalarIndex->second].value);
            statistic.validExamples = params->validExampleCount;
            if (tensors.contributionCount.has_value()) {
                THOR_THROW_IF_FALSE(tensors.aggregation == MetricAggregation::MIN ||
                                    tensors.aggregation == MetricAggregation::MAX);
                const float contributionCount = copyCpuScalarTensor(tensors.contributionCount.value());
                THOR_THROW_IF_FALSE(std::isfinite(contributionCount));
                THOR_THROW_IF_FALSE(contributionCount >= 0.0f);
                statistic.hasContribution = contributionCount > 0.0f;
            }
            if (tensors.aggregation == MetricAggregation::RATIO) {
                THOR_THROW_IF_FALSE(!tensors.contributionCount.has_value());
                THOR_THROW_IF_FALSE(tensors.numerator.has_value());
                THOR_THROW_IF_FALSE(tensors.denominator.has_value());
                statistic.numerator = static_cast<double>(
                    copyCpuScalarTensor(tensors.numerator.value()));
                statistic.denominator = static_cast<double>(
                    copyCpuScalarTensor(tensors.denominator.value()));
                statistic.zeroDenominatorMeansNoContribution =
                    tensors.zeroDenominatorMeansNoContribution;
                if (statistic.zeroDenominatorMeansNoContribution)
                    statistic.hasContribution = statistic.denominator.value() != 0.0;
            } else {
                THOR_THROW_IF_FALSE(!tensors.zeroDenominatorMeansNoContribution);
                THOR_THROW_IF_FALSE(!tensors.numerator.has_value());
                THOR_THROW_IF_FALSE(!tensors.denominator.has_value());
            }
            THOR_THROW_IF_FALSE(
                params->metricBatchStats.emplace(metricName, std::move(statistic)).second);
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
            std::lock_guard<std::mutex> lock(state->runState->mutex);
            THOR_THROW_IF_FALSE(slotIndex < state->runState->slots.size());
            QueuedBatchSlot& slot = state->runState->slots[slotIndex];
            THOR_THROW_IF_FALSE(slot.occupied);
            THOR_THROW_IF_FALSE(slot.segment.get() == segment.get());
            THOR_THROW_IF_FALSE(slot.epochBatchNum == epochBatchNum);
            THOR_THROW_IF_FALSE(slot.validExampleCount == params->validExampleCount);
            THOR_THROW_IF_FALSE(slot.validExampleCount > 0);
            THOR_THROW_IF_FALSE(slot.scalarStats.size() == params->scalarStats.size());
            segment->completedBatches += 1;
            segment->completedValidExamples += slot.validExampleCount;
            slot.doneInEpochAtComplete = segment->completedBatches;
            slot.scalarStats = params->scalarStats;
            slot.metricBatchStats = params->metricBatchStats;
            // Timestamp the batch when the completion callback has actually observed the
            // GPU work and required output copies as complete. Throughput must be based
            // on completion times, not on when the consumer thread later pops an already
            // ready slot; otherwise draining a backlog of completed slots can create
            // impossible end-of-epoch rate spikes.
            slot.completionTime = std::chrono::high_resolution_clock::now();
            slot.ready = true;
            state->completedBatchCallbacks += 1;
            params->completionCallbackFinished = true;
            inFlightAtComplete = state->runState->inFlightBatches;
            doneAtComplete = slot.doneInEpochAtComplete;
            totalAtComplete = segment->batchesPerEpoch;
        }
        if (shouldEmitQueueDiagnostic(doneAtComplete)) {
            emitNativeQueueDiagnostic("complete",
                                      phase,
                                      segment->optimizerEpoch,
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
        std::lock_guard<std::mutex> lock(state->runState->mutex);
        if (state->runState->failure == nullptr) {
            state->runState->failure = std::current_exception();
        }
        if (slotIndex < state->runState->slots.size()) {
            QueuedBatchSlot& slot = state->runState->slots[slotIndex];
            if (slot.occupied && slot.segment.get() == segment.get() &&
                slot.epochBatchNum == epochBatchNum) {
                segment->completedBatches += 1;
                segment->completedValidExamples += slot.validExampleCount;
                slot.doneInEpochAtComplete = segment->completedBatches;
                slot.completionTime = std::chrono::high_resolution_clock::now();
                slot.ready = true;
            }
        }
        state->completedBatchCallbacks += 1;
        params->completionCallbackFinished = true;
    }
    state->runState->batchFinished.notify_all();
}

bool isBatchDataReadyUnlocked(const NativeQueuedSchedulingWindowState& state) {
    const NativeQueuedRunState& runState = queuedRunState(state);
    if (runState.inFlightBatches == 0) {
        return false;
    }
    const QueuedBatchSlot& slot = runState.slots[runState.headSlot];
    return slot.occupied && slot.ready;
}

void waitForBatchDataUnlocked(NativeQueuedSchedulingWindowState& state, std::unique_lock<std::mutex>& lock) {
    NativeQueuedRunState& runState = queuedRunState(state);
    while (runState.failure == nullptr && !isBatchDataReadyUnlocked(state)) {
        if (runState.cancelRequested && runState.inFlightBatches == 0) {
            return;
        }
        if (state.schedulingFinished && runState.inFlightBatches == 0) {
            return;
        }
        runState.batchFinished.wait_for(lock, std::chrono::milliseconds(50));
    }
}

struct BatchPopResult {
    bool hasBatch = false;
    BatchLease batchLease;
    std::shared_ptr<QueuedWorkSegmentState> segment;
    ExampleType exampleType = ExampleType::TRAIN;
    TrainingEventPhase phase = TrainingEventPhase::TRAIN;
    uint64_t currentEpoch = 0;
    uint64_t epochBatchNum = 0;
    uint64_t slotIndex = 0;
    uint64_t inFlightAfterPop = 0;
    uint64_t doneInEpoch = 0;
    uint64_t poppedInEpoch = 0;
    uint64_t batchesInEpoch = 0;
    uint64_t validExampleCount = 0;
    std::optional<ThorImplementation::LogicalWorkCount> logicalWork{};
    uint64_t validExamplesInEpoch = 0;
    std::chrono::high_resolution_clock::time_point phaseStartedAt{};
    std::chrono::high_resolution_clock::time_point completionTime{};
    std::vector<ScalarStatSlot> scalarStats;
    std::unordered_map<std::string, MetricBatchStat> metricBatchStats;
};

BatchPopResult popBatchData(
    const std::shared_ptr<NativeQueuedSchedulingWindowState>& state,
    std::optional<uint64_t> maximumReportedEpoch = std::nullopt) {
    std::unique_lock<std::mutex> lock(state->runState->mutex);
    waitForBatchDataUnlocked(*state, lock);

    if (!isBatchDataReadyUnlocked(*state)) {
        if (state->runState->failure != nullptr) {
            std::rethrow_exception(state->runState->failure);
        }
        return {};
    }

    QueuedBatchSlot& slot = state->runState->slots[state->runState->headSlot];
    const std::shared_ptr<QueuedWorkSegmentState> segment = slot.segment;
    THOR_THROW_IF_FALSE(segment != nullptr);
    if (maximumReportedEpoch.has_value() &&
        segment->reportedEpoch > maximumReportedEpoch.value()) {
        // The FIFO head already belongs to a later logical epoch. Leave it in
        // the persistent ring so the consumer can finalize the current epoch
        // while the scheduler/GPU continue ahead inside the same window.
        return {};
    }
    NativeBatchCompletionParams& params = state->runState->completionParams[slot.paramsIndex];
    THOR_THROW_IF_FALSE(params.segment.get() == segment.get());

    BatchPopResult result;
    result.hasBatch = true;
    result.segment = segment;
    result.exampleType = segment->exampleType;
    result.phase = segment->phase;
    result.currentEpoch = segment->reportedEpoch;
    result.epochBatchNum = slot.epochBatchNum;
    result.slotIndex = state->runState->headSlot;
    result.doneInEpoch = slot.doneInEpochAtComplete;
    result.batchesInEpoch = segment->batchesPerEpoch;
    result.validExampleCount = slot.validExampleCount;
    result.logicalWork = slot.logicalWork;
    result.validExamplesInEpoch =
        slot.validExamplesThroughBatch;
    result.phaseStartedAt = slot.phaseStartedAt;
    result.completionTime = slot.completionTime;
    result.batchLease = std::move(params.batchLease);
    result.scalarStats = slot.scalarStats;
    result.metricBatchStats = std::move(slot.metricBatchStats);

    params.batchOutput.clear();
    params.metricStatisticTensors.clear();
    params.metricBatchStats.clear();
    params.completionCallbackLaunched = false;
    params.completionCallbackFinished = false;
    params.validExampleCount = 0;
    params.logicalWork = ThorImplementation::LogicalWorkCount{};
    params.segment.reset();
    params.state.reset();
    for (ScalarStatSlot& scalarStat : slot.scalarStats) {
        scalarStat.present = false;
        scalarStat.value = 0.0f;
    }
    slot.metricBatchStats.clear();
    for (ScalarStatSlot& scalarStat : params.scalarStats) {
        scalarStat.present = false;
        scalarStat.value = 0.0f;
    }
    for (ScalarStatSource& scalarStatSource : params.scalarStatSources) {
        scalarStatSource = ScalarStatSource::UNRESOLVED;
    }
    segment->poppedBatches += 1;
    segment->poppedValidExamples += slot.validExampleCount;
    result.poppedInEpoch = segment->poppedBatches;
    THOR_THROW_IF_FALSE(
        result.validExamplesInEpoch == segment->poppedValidExamples);

    slot.ready = false;
    slot.occupied = false;
    slot.segment.reset();
    slot.epochBatchNum = 0;
    slot.validExampleCount = 0;
    slot.logicalWork = ThorImplementation::LogicalWorkCount{};
    slot.doneInEpochAtComplete = 0;
    slot.validExamplesThroughBatch = 0;
    slot.phaseStartedAt = {};
    slot.completionTime = {};
    state->runState->headSlot = (state->runState->headSlot + 1) % state->runState->slots.size();
    state->runState->inFlightBatches -= 1;
    result.inFlightAfterPop = state->runState->inFlightBatches;

    lock.unlock();
    state->runState->batchPopped.notify_all();
    return result;
}

uint64_t outstandingBatchCount(const std::shared_ptr<NativeQueuedSchedulingWindowState>& state) {
    std::lock_guard<std::mutex> lock(state->runState->mutex);
    return state->runState->inFlightBatches;
}

void throwIfSchedulingWindowStateFailed(
    const std::shared_ptr<NativeQueuedSchedulingWindowState>& state) {
    std::exception_ptr failure;
    bool cancelRequested = false;
    bool interruptRequested = false;
    {
        std::lock_guard<std::mutex> lock(state->runState->mutex);
        failure = state->runState->failure;
        cancelRequested = state->runState->cancelRequested;
        interruptRequested = state->runState->interruptRequested;
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

bool queuedCompletionCallbacksPendingUnlocked(
    const NativeQueuedSchedulingWindowState& state) {
    for (const NativeBatchCompletionParams& params :
         queuedRunState(state).completionParams) {
        if (params.completionCallbackLaunched && !params.completionCallbackFinished) {
            return true;
        }
    }
    return false;
}

void releaseSchedulingWindowStateReferencesAfterAbort(
    const std::shared_ptr<NativeQueuedSchedulingWindowState>& state,
    bool submittedWorkDrained) {
    if (state == nullptr) {
        return;
    }

    std::vector<BatchLease> leasesToRelease;

    std::unique_lock<std::mutex> lock(state->runState->mutex);
    if (!submittedWorkDrained && queuedCompletionCallbacksPendingUnlocked(*state)) {
        // If CUDA synchronization failed, a launched host callback may still hold
        // a raw pointer into the run-scoped completionParams storage. Do not break
        // those references here; preserving them is safer than risking a UAF in an
        // already-failing CUDA context. Normal trainer failures reach this path with
        // submittedWorkDrained=true and are cleaned up below.
        return;
    }
    while (queuedCompletionCallbacksPendingUnlocked(*state)) {
        state->runState->batchFinished.wait_for(lock, std::chrono::milliseconds(50));
    }

    for (NativeBatchCompletionParams& params : state->runState->completionParams) {
        if (!params.batchLease.empty()) {
            leasesToRelease.push_back(std::move(params.batchLease));
        }
        params.batchOutput.clear();
        params.metricStatisticTensors.clear();
        params.metricBatchStats.clear();
        for (ScalarStatSlot& scalarStat : params.scalarStats) {
            scalarStat.present = false;
            scalarStat.value = 0.0f;
        }
        for (ScalarStatSource& scalarStatSource : params.scalarStatSources) {
            scalarStatSource = ScalarStatSource::UNRESOLVED;
        }
        params.completionCallbackLaunched = false;
        params.completionCallbackFinished = false;
        params.validExampleCount = 0;
        params.logicalWork = ThorImplementation::LogicalWorkCount{};
        params.segment.reset();
        params.state.reset();
    }

    for (QueuedBatchSlot& slot : state->runState->slots) {
        slot.occupied = false;
        slot.ready = false;
        slot.segment.reset();
        slot.epochBatchNum = 0;
        slot.validExampleCount = 0;
        slot.logicalWork = ThorImplementation::LogicalWorkCount{};
        slot.doneInEpochAtComplete = 0;
        slot.validExamplesThroughBatch = 0;
        slot.paramsIndex = 0;
        slot.completionTime = {};
        for (ScalarStatSlot& scalarStat : slot.scalarStats) {
            scalarStat.present = false;
            scalarStat.value = 0.0f;
        }
        slot.metricBatchStats.clear();
    }
    state->runState->headSlot = 0;
    state->runState->tailSlot = 0;
    state->runState->inFlightBatches = 0;
    lock.unlock();

    leasesToRelease.clear();
    state->runState->batchPopped.notify_all();
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

void prepareQueuedWorkSegmentForScheduling(
    const std::shared_ptr<NativeQueuedSchedulingWindowState>& state,
    const std::shared_ptr<QueuedWorkSegmentState>& segment);

struct NativeQueuedSchedulerResources {
    NativeQueuedSchedulerResources(
        std::shared_ptr<PlacedNetwork> placedNetwork,
        std::shared_ptr<const ExecutableTrainingPlan> plan,
        const NativeQueuedTrainingOptions& options)
        : placedNetwork(std::move(placedNetwork)),
          plan(std::move(plan)),
          options(options),
          outputReadyEvents(this->placedNetwork->getNumStamps()),
          processingFinishedEvents(options.maxInFlightBatches),
          completionFinishedEvents(options.maxInFlightBatches) {
        THOR_THROW_IF_FALSE(this->placedNetwork != nullptr);
        THOR_THROW_IF_FALSE(this->plan != nullptr);

        stampGpuNums.reserve(this->placedNetwork->getNumStamps());
        for (uint64_t stamp = 0; stamp < this->placedNetwork->getNumStamps(); ++stamp) {
            ThorImplementation::StampedNetwork& stampedNetwork =
                this->placedNetwork->getStampedNetwork(stamp);
            std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputs =
                stampedNetwork.getInputs();
            THOR_THROW_IF_FALSE(!inputs.empty());
            stampGpuNums.push_back(inputs[0]->getStream().getGpuNum());
        }

        // Completion must not serialize all queued slots through one stream: the
        // wait/event/host-callback tail can otherwise cap the native queue below
        // max_in_flight. Assign each queue slot a completion stream from the
        // existing download-stream pool. These streams and the recurring event
        // banks belong to the placed training run, not to one logical epoch.
        completionStreams.reserve(options.maxInFlightBatches);
        for (uint64_t slotIndex = 0; slotIndex < options.maxInFlightBatches;
             ++slotIndex) {
            const uint64_t stamp = slotIndex % this->placedNetwork->getNumStamps();
            completionStreams.push_back(
                Stream::getNextDownloadStream(stampGpuNums[stamp]));
        }

        recordNativeQueuedSchedulerResourceConstructionForTests();
    }

    void recordEventReuseForTests() const {
        const uint64_t processingFinishedEventId =
            processingFinishedEvents.empty()
                ? 0
                : processingFinishedEvents.front().getId();
        const uint64_t completionFinishedEventId =
            completionFinishedEvents.empty()
                ? 0
                : completionFinishedEvents.front().getId();
        recordNativeQueuedSchedulerEventReuseForTests(
            processingFinishedEventId,
            completionFinishedEventId);
    }

    std::shared_ptr<PlacedNetwork> placedNetwork;
    std::shared_ptr<const ExecutableTrainingPlan> plan;
    NativeQueuedTrainingOptions options;
    uint64_t nextStampToProcess = 0;
    std::vector<std::map<std::string, Event>> outputReadyEvents;
    std::vector<Event> processingFinishedEvents;
    std::vector<Event> completionFinishedEvents;
    std::vector<int> stampGpuNums;
    std::vector<Stream> completionStreams;
};

class NativeQueuedSegmentScheduler {
   public:
    NativeQueuedSegmentScheduler(
        std::shared_ptr<NativeQueuedSchedulerResources> resources,
        std::shared_ptr<NativeQueuedSchedulingWindowState> state,
        TrainingCancellationToken cancellationToken)
        : resources(std::move(resources)),
          state(std::move(state)),
          cancellationToken(std::move(cancellationToken)) {
        THOR_THROW_IF_FALSE(this->resources != nullptr);
    }

    void operator()(const std::shared_ptr<QueuedWorkSegmentState>& segment) {
        THOR_THROW_IF_FALSE(segment != nullptr);
        if (cancellationToken.isCancellationRequested()) {
            requestQueuedTrainingCancellation(state);
            return;
        }
        prepareQueuedWorkSegmentForScheduling(state, segment);

        const uint64_t initialEpochBatchNum = segment->initialBatchNum;
        const uint64_t initialSessionBatchNum = segment->sessionInitialBatchNum;
        const uint64_t initialValidExamples = segment->initialValidExamples;
        const uint64_t batches = segment->batchesToRunCount;
        const uint64_t batchesPerEpoch = segment->batchesPerEpoch;
        const ExampleType exampleType = segment->exampleType;
        const TrainingEventPhase diagnosticPhase = segment->phase;
        const uint64_t currentEpoch = segment->optimizerEpoch;
        const std::shared_ptr<BatchSession>& batchSession = segment->batchSession;

        if (batches == 0) {
            return;
        }

        const std::shared_ptr<PlacedNetwork>& placedNetwork =
            resources->placedNetwork;
        const std::shared_ptr<const ExecutableTrainingPlan>& plan =
            resources->plan;
        const NativeQueuedTrainingOptions& options = resources->options;
        uint64_t& nextStampToProcess = resources->nextStampToProcess;
        std::vector<std::map<std::string, Event>>& outputReadyEvents =
            resources->outputReadyEvents;
        std::vector<Event>& processingFinishedEvents =
            resources->processingFinishedEvents;
        std::vector<Event>& completionFinishedEvents =
            resources->completionFinishedEvents;
        const std::vector<int>& stampGpuNums = resources->stampGpuNums;
        const std::vector<Stream>& completionStreams =
            resources->completionStreams;

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
        uint64_t validExamplesScheduled = initialValidExamples;

        for (uint64_t batch = 0; batch < batches; ++batch) {
            if (cancellationToken.isCancellationRequested()) {
                requestQueuedTrainingCancellation(state);
                return;
            }
            const auto scheduleIterationStart = diagnosticNow(collectQueueDiagnostics);
            const uint64_t epochBatchNum = initialEpochBatchNum + batch;
            const auto optimizerStart = diagnosticNow(collectQueueDiagnostics);
            // Hyper-parameter schedules advance with optimizer updates. Validation is
            // forward-only, so it must neither traverse nor mutate optimizer state.
            if (!validationPass) {
                Optimizer::updateHyperParameters(placedNetwork.get(), currentEpoch, epochBatchNum, batchesPerEpoch);
            }
            const auto optimizerFinish = diagnosticNow(collectQueueDiagnostics);

            uint64_t slotIndex = 0;
            uint64_t inFlightAfterReserve = 0;
            const auto reserveStart = diagnosticNow(collectQueueDiagnostics);
            {
                std::unique_lock<std::mutex> lock(state->runState->mutex);
                while (state->runState->failure == nullptr && !state->runState->cancelRequested && state->runState->inFlightBatches >= options.maxInFlightBatches) {
                    state->runState->batchPopped.wait(lock);
                }
                if (state->runState->failure != nullptr || state->runState->cancelRequested) {
                    return;
                }

                slotIndex = state->runState->tailSlot;
                QueuedBatchSlot& slot = state->runState->slots[slotIndex];
                THOR_THROW_IF_FALSE(!slot.occupied);
                slot.occupied = true;
                slot.ready = false;
                slot.segment = segment;
                slot.epochBatchNum = epochBatchNum;
                slot.validExampleCount = 0;
                slot.logicalWork = ThorImplementation::LogicalWorkCount{};
                slot.doneInEpochAtComplete = 0;
                slot.validExamplesThroughBatch = 0;
                slot.paramsIndex = slotIndex;
                slot.phaseStartedAt = segment->schedulingStartedAt;
                slot.completionTime = {};
                for (ScalarStatSlot& scalarStat : slot.scalarStats) {
                    scalarStat.present = false;
                    scalarStat.value = 0.0f;
                }
                slot.metricBatchStats.clear();
                state->runState->tailSlot = (state->runState->tailSlot + 1) % state->runState->slots.size();
                state->runState->inFlightBatches += 1;
                inFlightAfterReserve = state->runState->inFlightBatches;
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

            NativeBatchCompletionParams* params = &state->runState->completionParams[slotIndex];
            params->state = state;
            params->segment = segment;
            params->completionCallbackLaunched = false;
            params->completionCallbackFinished = false;
            params->epochBatchNum = epochBatchNum;
            params->validExampleCount = 0;
            params->logicalWork = ThorImplementation::LogicalWorkCount{};
            params->slotIndex = slotIndex;
            params->batchLease.reset();
            params->batchOutput.clear();
            params->metricStatisticTensors.clear();
            params->metricBatchStats.clear();
            for (ScalarStatSlot& scalarStat : params->scalarStats) {
                scalarStat.present = false;
                scalarStat.value = 0.0f;
            }
            for (ScalarStatSource& scalarStatSource : params->scalarStatSources) {
                scalarStatSource = ScalarStatSource::UNRESOLVED;
            }

            const auto acquireBatchStart = diagnosticNow(collectQueueDiagnostics);
            uint64_t sessionBatchNum =
                initialSessionBatchNum + batch;
            params->batchLease = batchSession->leaseBatch(exampleType, sessionBatchNum);
            params->validExampleCount =
                params->batchLease.get().getValidExampleCount().value_or(
                    static_cast<uint32_t>(batchSession->getBatchSize()));
            if (params->validExampleCount == 0 ||
                params->validExampleCount > batchSession->getBatchSize()) {
                throw std::runtime_error(
                    "BatchSession returned an invalid valid-example count.");
            }
            {
                std::lock_guard<std::mutex> lock(state->runState->mutex);
                QueuedBatchSlot& slot = state->runState->slots[slotIndex];
                THOR_THROW_IF_FALSE(slot.occupied);
                slot.validExampleCount = params->validExampleCount;
                validExamplesScheduled += params->validExampleCount;
                slot.validExamplesThroughBatch =
                    validExamplesScheduled;
            }
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
                    if (params->logicalWork.has_value()) {
                        ThorImplementation::StampedNetwork& submittedStamp =
                            placedNetwork->getStampedNetwork(nextStampToProcess);
                        const std::optional<ThorImplementation::LogicalWorkCount> submitLogicalWork =
                            bestEffortCurrentBatchLogicalWork(
                                submittedStamp, diagnosticPhase, params->validExampleCount);
                        const std::optional<ThorImplementation::LogicalWorkCount> accumulatedLogicalWork =
                            submitLogicalWork.has_value()
                                ? ThorImplementation::LogicalWorkCount::tryAdd(
                                      params->logicalWork.value(), submitLogicalWork.value())
                                : std::nullopt;
                        if (!accumulatedLogicalWork.has_value()) {
                            // The batch has already been submitted successfully.
                            // Drop logical-work telemetry for this batch instead of making
                            // an accounting limitation fatal to model training.
                            params->logicalWork.reset();
                        } else {
                            params->logicalWork = accumulatedLogicalWork;
                        }
                    }
                    submitCalls += 1;
                }
            }

            {
                std::lock_guard<std::mutex> lock(state->runState->mutex);
                QueuedBatchSlot& slot = state->runState->slots[slotIndex];
                THOR_THROW_IF_FALSE(slot.occupied);
                slot.logicalWork = params->logicalWork;
            }
            const auto submitFinish = diagnosticNow(collectQueueDiagnostics);

            // Metric aggregation metadata follows the public metric output. Ratio
            // metrics additionally expose slot-local host numerator/denominator
            // tensors whose ready event must join this batch's completion stream.
            const std::map<std::string, ThorImplementation::MetricBatchStatisticTensors>
                availableMetricStatistics =
                    placedNetwork->getMetricBatchStatisticTensorsForSlot(
                        nextStampToProcess,
                        static_cast<uint32_t>(slotIndex));
            for (const auto& [metricName, tensors] : availableMetricStatistics) {
                if (state->runState->scalarTensorIndexByName.count(metricName) == 0) {
                    continue;
                }
                THOR_THROW_IF_FALSE(
                    params->metricStatisticTensors.emplace(metricName, tensors).second);
            }

            // Every NetworkInput has now recorded the stream point after which
            // it no longer reads its session-owned source. Snapshot CPU input
            // statistics now, before those source slots are released; output
            // statistics remain deferred until the normal completion callback.
            THOR_THROW_IF_FALSE(params->scalarStats.size() == state->runState->scalarTensorNames.size());
            THOR_THROW_IF_FALSE(
                params->scalarStatSources.size() == state->runState->scalarTensorNames.size());
            const bool fullySourceTracked =
                params->batchLease.get().allFieldsHaveSourceReferences();
            std::set<std::string> retainedSourceFields;
            for (size_t i = 0; i < state->runState->scalarTensorNames.size(); ++i) {
                const std::string& scalarTensorName = state->runState->scalarTensorNames[i];
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
            for (const auto& [metricName, tensors] : params->metricStatisticTensors) {
                (void)metricName;
                if (tensors.aggregation == MetricAggregation::RATIO) {
                    THOR_THROW_IF_FALSE(tensors.readyEvent.isInitialized());
                    completionStream.waitEvent(tensors.readyEvent);
                    outputWaitCount += 1;
                } else {
                    THOR_THROW_IF_FALSE(!tensors.readyEvent.isInitialized());
                }
            }
            const auto waitOutputsFinish = diagnosticNow(collectQueueDiagnostics);

            const auto hostFuncStart = diagnosticNow(collectQueueDiagnostics);
            {
                std::lock_guard<std::mutex> lock(state->runState->mutex);
                params->completionCallbackLaunched = true;
                params->completionCallbackFinished = false;
            }
            try {
                CUDA_CHECK(cudaLaunchHostFunc(completionStream, completeNativeQueuedBatch, params));
            } catch (...) {
                std::lock_guard<std::mutex> lock(state->runState->mutex);
                params->completionCallbackLaunched = false;
                params->completionCallbackFinished = false;
                throw;
            }
            const auto hostFuncFinish = diagnosticNow(collectQueueDiagnostics);
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
            placedNetwork->extendMetricStatisticWritableEvents(
                nextStampToProcess,
                completionFinishedEvents[slotIndex],
                static_cast<uint32_t>(slotIndex));
            const auto extendOutputsFinish = diagnosticNow(collectQueueDiagnostics);
            const auto completionSetupFinish = extendOutputsFinish;
            recordNativeQueuedBatchSubmissionForTests(currentEpoch);

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

        if (segment->requiresEpochBoundaryValidation &&
            batchSession->getNextBatchNum(exampleType) != 0) {
            throw std::runtime_error(
                "Native queued " + phaseName(segment->phase) +
                " segment did not finish at its batch-session epoch boundary.");
        }
    }

   private:
    std::shared_ptr<NativeQueuedSchedulerResources> resources;
    std::shared_ptr<NativeQueuedSchedulingWindowState> state;
    TrainingCancellationToken cancellationToken;
};

struct NativeQueuedSchedulerCommandCompletion {
    void markFinished() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            finished = true;
        }
        finishedCondition.notify_all();
    }

    void wait() const {
        std::unique_lock<std::mutex> lock(mutex);
        while (!finished) {
            finishedCondition.wait(lock);
        }
    }

    mutable std::mutex mutex;
    mutable std::condition_variable finishedCondition;
    bool finished = false;
};

struct NativeQueuedSchedulerCommand {
    std::shared_ptr<NativeQueuedSchedulingWindowState> state;
    TrainingCancellationToken cancellationToken;
    std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments;
    std::shared_ptr<NativeQueuedSchedulerCommandCompletion> completion;
};

// Keep one producer worker alive for the placed run and feed it one
// non-overlapping scheduling-window command at a time. Ordinary epoch/phase
// boundaries may live inside a command; a new command is reserved for a true
// host-side decision boundary or an explicitly isolated scheduling operation.
class NativeQueuedSchedulerWorker {
   public:
    NativeQueuedSchedulerWorker(
        std::shared_ptr<NativeQueuedSchedulerResources> resources,
        std::shared_ptr<NativeQueuedRunState> runState)
        : resources(std::move(resources)), runState(std::move(runState)) {
        THOR_THROW_IF_FALSE(this->resources != nullptr);
        THOR_THROW_IF_FALSE(this->runState != nullptr);
        worker = std::thread(&NativeQueuedSchedulerWorker::workerLoop, this);
    }

    ~NativeQueuedSchedulerWorker() { shutdown(); }

    NativeQueuedSchedulerWorker(const NativeQueuedSchedulerWorker&) = delete;
    NativeQueuedSchedulerWorker& operator=(const NativeQueuedSchedulerWorker&) = delete;

    [[nodiscard]] const std::shared_ptr<NativeQueuedSchedulerResources>&
    getResources() const {
        return resources;
    }

    [[nodiscard]] const std::shared_ptr<NativeQueuedRunState>&
    getRunState() const {
        return runState;
    }

    std::shared_ptr<NativeQueuedSchedulerCommandCompletion> submit(
        NativeQueuedSchedulerCommand command) {
        THOR_THROW_IF_FALSE(command.state != nullptr);
        THOR_THROW_IF_FALSE(command.state->runState.get() == runState.get());
        auto completion =
            std::make_shared<NativeQueuedSchedulerCommandCompletion>();
        command.completion = completion;

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                throw std::runtime_error(
                    "Native queued scheduler worker is stopping.");
            }
            if (commandActive || pendingCommand.has_value()) {
                throw std::runtime_error(
                    "Native queued scheduler worker received overlapping scheduling-window "
                    "commands.");
            }
            commandActive = true;
            pendingCommand = std::move(command);
        }
        workAvailable.notify_one();
        return completion;
    }

    void shutdown() noexcept {
        std::shared_ptr<NativeQueuedSchedulingWindowState> activeState;
        std::vector<std::shared_ptr<BatchSession>> activeSessions;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                // Another shutdown path already owns the join below or the
                // worker has already been joined.
            } else {
                stopping = true;
            }
            if (pendingCommand.has_value()) {
                activeState = pendingCommand->state;
                for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
                     pendingCommand->segments) {
                    if (segment != nullptr && segment->batchSession != nullptr) {
                        activeSessions.push_back(segment->batchSession);
                    }
                }
            } else {
                activeState = activeCommandState;
                activeSessions = activeCommandSessions;
            }
        }

        // Normal shutdown happens while the worker is idle. These two calls are
        // only a last-resort unwind path for an exception that escapes while a
        // command is still active; they prevent a producer blocked on queue
        // capacity or BatchSession acquisition from making destruction hang.
        if (activeState != nullptr) {
            requestQueuedTrainingCancellation(activeState);
        }
        std::set<BatchSession*> cancelledSessions;
        for (const std::shared_ptr<BatchSession>& activeSession : activeSessions) {
            if (activeSession == nullptr) {
                continue;
            }
            if (!cancelledSessions.insert(activeSession.get()).second) {
                continue;
            }
            try {
                cancelBatchSession(activeSession);
            } catch (...) {
            }
        }

        workAvailable.notify_all();
        try {
            if (worker.joinable()) {
                worker.join();
            }
        } catch (...) {
        }
    }

   private:
    void workerLoop() noexcept {
        recordNativeQueuedSchedulerWorkerThreadStartForTests();
        while (true) {
            std::optional<NativeQueuedSchedulerCommand> command;
            {
                std::unique_lock<std::mutex> lock(mutex);
                workAvailable.wait(lock, [this]() {
                    return stopping || pendingCommand.has_value();
                });
                if (!pendingCommand.has_value()) {
                    return;
                }

                command.emplace(std::move(pendingCommand.value()));
                pendingCommand.reset();
                activeCommandState = command->state;
                activeCommandSessions.clear();
                for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
                     command->segments) {
                    if (segment != nullptr && segment->batchSession != nullptr) {
                        activeCommandSessions.push_back(segment->batchSession);
                    }
                }
            }

            executeCommand(command.value());

            {
                std::lock_guard<std::mutex> lock(mutex);
                activeCommandState.reset();
                activeCommandSessions.clear();
                commandActive = false;
            }
            command->completion->markFinished();
        }
    }

    void executeCommand(NativeQueuedSchedulerCommand& command) noexcept {
        try {
            {
                std::lock_guard<std::mutex> lock(command.state->runState->mutex);
                if (command.state->runState->failure != nullptr ||
                    command.state->runState->cancelRequested) {
                    command.state->schedulingFinished = true;
                    command.state->runState->batchFinished.notify_all();
                    return;
                }
            }

            NativeQueuedSegmentScheduler scheduler(
                resources,
                command.state,
                command.cancellationToken);
            for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
                 command.segments) {
                scheduler(segment);
            }
            {
                std::lock_guard<std::mutex> lock(command.state->runState->mutex);
                command.state->schedulingFinished = true;
            }
            command.state->runState->batchFinished.notify_all();
            resources->recordEventReuseForTests();
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(command.state->runState->mutex);
                if (command.state->runState->failure == nullptr) {
                    command.state->runState->failure = std::current_exception();
                }
                command.state->runState->cancelRequested = true;
                command.state->schedulingFinished = true;
            }
            command.state->runState->batchFinished.notify_all();
            command.state->runState->batchPopped.notify_all();
        }
    }

    std::shared_ptr<NativeQueuedSchedulerResources> resources;
    std::shared_ptr<NativeQueuedRunState> runState;
    std::mutex mutex;
    std::condition_variable workAvailable;
    bool stopping = false;
    bool commandActive = false;
    std::optional<NativeQueuedSchedulerCommand> pendingCommand;
    std::shared_ptr<NativeQueuedSchedulingWindowState> activeCommandState;
    std::vector<std::shared_ptr<BatchSession>> activeCommandSessions;
    // Start this last from the constructor body so the persistent worker can
    // never observe partially constructed synchronization/command state.
    std::thread worker;
};

struct NativeQueuedSchedulingWindowExecution {
    std::shared_ptr<NativeQueuedSchedulingWindowState> state;
    std::shared_ptr<NativeQueuedSchedulerCommandCompletion>
        schedulerCommandCompletion;
    std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments;
    std::chrono::high_resolution_clock::time_point startedAt{};
};

struct QueuedValidationPopulationMetadata {
    std::string name;
    bool isDefault = false;
};

uint64_t validExamplesBeforeBatch(
    uint64_t batchNum,
    uint64_t populationSize,
    uint64_t physicalBatchSize) {
    THOR_THROW_IF_FALSE(physicalBatchSize > 0);
    if (batchNum > populationSize / physicalBatchSize) {
        return populationSize;
    }
    return std::min(populationSize, batchNum * physicalBatchSize);
}

void prepareQueuedWorkSegmentForScheduling(
    const std::shared_ptr<NativeQueuedSchedulingWindowState>& state,
    const std::shared_ptr<QueuedWorkSegmentState>& segment) {
    THOR_THROW_IF_FALSE(state != nullptr);
    THOR_THROW_IF_FALSE(segment != nullptr);
    THOR_THROW_IF_FALSE(segment->batchSession != nullptr);

    const std::shared_ptr<BatchSession>& batchSession = segment->batchSession;
    const ExampleType exampleType = segment->exampleType;
    const TrainingEventPhase phase = segment->phase;
    const uint64_t sessionBatchNum = batchSession->getNextBatchNum(exampleType);
    const uint64_t sessionBatchesPerEpoch =
        batchSession->getNumBatchesPerEpoch(exampleType);
    if (sessionBatchNum > sessionBatchesPerEpoch) {
        throw std::runtime_error(
            "BatchSession returned next batch number beyond batches per epoch for " +
            phaseName(phase) + ".");
    }

    const uint64_t populationSize = batchSession->getNumExamples(exampleType);
    const uint64_t physicalBatchSize = batchSession->getBatchSize();
    const bool wrapsTail =
        ThorImplementation::BatchSessionRuntimeAccess::getTailMode(*batchSession) ==
        ThorImplementation::BatchTailMode::WRAP;
    const uint64_t examplesProcessedPerEpoch =
        ThorImplementation::BatchSessionRuntimeAccess::examplesProcessedPerEpoch(
            *batchSession, exampleType);
    const uint64_t initialValidExamples = wrapsTail
        ? sessionBatchNum * physicalBatchSize
        : validExamplesBeforeBatch(
              sessionBatchNum, populationSize, physicalBatchSize);

    uint64_t publicInitialBatchNum = sessionBatchNum;
    uint64_t publicBatchesPerEpoch = sessionBatchesPerEpoch;
    uint64_t batchesToRun = sessionBatchesPerEpoch - sessionBatchNum;
    uint64_t publicInitialValidExamples = initialValidExamples;
    uint64_t publicValidExamplesPerEpoch = examplesProcessedPerEpoch;
    bool requiresEpochBoundaryValidation = true;
    if (segment->maxBatchesToRun.has_value() &&
        sessionBatchesPerEpoch > segment->maxBatchesToRun.value()) {
        // A capped public training epoch is a fixed-size work quantum. It may
        // end before the population boundary and therefore deliberately does
        // not claim exact-population epoch semantics.
        batchesToRun = segment->maxBatchesToRun.value();
        publicInitialBatchNum = 0;
        publicBatchesPerEpoch = batchesToRun;
        publicInitialValidExamples = 0;
        publicValidExamplesPerEpoch = 0;
        requiresEpochBoundaryValidation = false;
    }

    {
        std::lock_guard<std::mutex> lock(state->runState->mutex);
        THOR_THROW_IF_FALSE(!segment->prepared);
        segment->initialBatchNum = publicInitialBatchNum;
        segment->sessionInitialBatchNum = sessionBatchNum;
        segment->batchesToRunCount = batchesToRun;
        segment->batchesPerEpoch = publicBatchesPerEpoch;
        segment->initialValidExamples = publicInitialValidExamples;
        segment->validExamplesPerEpoch = publicValidExamplesPerEpoch;
        segment->requiresEpochBoundaryValidation =
            requiresEpochBoundaryValidation;
        segment->completedBatches = publicInitialBatchNum;
        segment->poppedBatches = publicInitialBatchNum;
        segment->completedValidExamples = publicInitialValidExamples;
        segment->poppedValidExamples = publicInitialValidExamples;
        segment->schedulingStartedAt =
            std::chrono::high_resolution_clock::now();
        segment->prepared = true;
    }
    state->runState->batchFinished.notify_all();
}

std::vector<std::shared_ptr<QueuedWorkSegmentState>>
buildQueuedSchedulingWindowSegments(
    const TrainingRunRequest& request,
    const std::shared_ptr<BatchSession>& effectiveSession,
    const std::vector<NamedValidationSession>& additionalValidationSessions,
    uint64_t firstOptimizerEpoch,
    uint64_t firstReportedEpoch,
    bool evaluateOnly,
    uint32_t epochsToSchedule,
    const std::optional<QueuedValidationPopulationMetadata>&
        validationPopulationMetadata) {
    THOR_THROW_IF_FALSE(epochsToSchedule >= 1);
    if (evaluateOnly) {
        THOR_THROW_IF_FALSE(epochsToSchedule == 1);
    }

    std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments;
    const size_t baseSegmentsPerEpoch = evaluateOnly ? 1 : 2;
    segments.reserve(
        (baseSegmentsPerEpoch + additionalValidationSessions.size()) *
        epochsToSchedule);

    auto appendSegment = [&](const std::shared_ptr<BatchSession>& batchSession,
                             ExampleType exampleType,
                             TrainingEventPhase phase,
                             uint64_t optimizerEpoch,
                             uint64_t reportedEpoch,
                             const std::optional<QueuedValidationPopulationMetadata>&
                                 populationMetadata) {
        THOR_THROW_IF_FALSE(batchSession != nullptr);
        request.cancellationToken.throwIfCancellationRequested();
        auto segment = std::make_shared<QueuedWorkSegmentState>();
        segment->batchSession = batchSession;
        segment->exampleType = exampleType;
        segment->phase = phase;
        segment->optimizerEpoch = optimizerEpoch;
        segment->reportedEpoch = reportedEpoch;
        if (!evaluateOnly && phase == TrainingEventPhase::TRAIN &&
            request.maxTrainingBatchesPerEpoch.has_value()) {
            segment->maxBatchesToRun = request.maxTrainingBatchesPerEpoch;
        }
        if (phase == TrainingEventPhase::VALIDATE) {
            if (populationMetadata.has_value()) {
                segment->validationPopulation = populationMetadata->name;
                segment->isDefaultValidationPopulation =
                    populationMetadata->isDefault;
            } else {
                segment->validationPopulation = request.defaultValidationPopulation;
                segment->isDefaultValidationPopulation = true;
            }
        }
        segments.push_back(std::move(segment));
    };

    for (uint32_t epochOffset = 0; epochOffset < epochsToSchedule; ++epochOffset) {
        const uint64_t optimizerEpoch = firstOptimizerEpoch + epochOffset;
        const uint64_t reportedEpoch = firstReportedEpoch + epochOffset;
        if (evaluateOnly) {
            appendSegment(effectiveSession,
                          request.evaluationExampleType,
                          request.evaluationPhase,
                          optimizerEpoch,
                          reportedEpoch,
                          validationPopulationMetadata);
        } else {
            appendSegment(effectiveSession,
                          ExampleType::TRAIN,
                          TrainingEventPhase::TRAIN,
                          optimizerEpoch,
                          reportedEpoch,
                          std::nullopt);
            appendSegment(effectiveSession,
                          ExampleType::VALIDATE,
                          TrainingEventPhase::VALIDATE,
                          optimizerEpoch,
                          reportedEpoch,
                          QueuedValidationPopulationMetadata{
                              request.defaultValidationPopulation,
                              /*isDefault=*/true});
        }

        // Additional validation populations are just more ordered validation
        // segments over different example sets. Keep them in the same stream so
        // ordinary batch ordering guarantees they all observe the same checkpoint
        // without a host-side queue drain between populations.
        for (const NamedValidationSession& validation :
             additionalValidationSessions) {
            appendSegment(validation.batchSession,
                          ExampleType::VALIDATE,
                          TrainingEventPhase::VALIDATE,
                          optimizerEpoch,
                          reportedEpoch,
                          QueuedValidationPopulationMetadata{
                              validation.name,
                              /*isDefault=*/false});
        }
    }

    return segments;
}

std::vector<std::shared_ptr<QueuedWorkSegmentState>>
queuedSegmentsForReportedEpoch(
    const NativeQueuedSchedulingWindowExecution& execution,
    uint64_t reportedEpoch) {
    std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments;
    for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
         execution.segments) {
        THOR_THROW_IF_FALSE(segment != nullptr);
        if (segment->reportedEpoch == reportedEpoch) {
            segments.push_back(segment);
        }
    }
    return segments;
}

bool queuedSegmentsFullyPopped(
    const std::shared_ptr<NativeQueuedSchedulingWindowState>& state,
    const std::vector<std::shared_ptr<QueuedWorkSegmentState>>& segments) {
    THOR_THROW_IF_FALSE(state != nullptr);
    std::lock_guard<std::mutex> lock(state->runState->mutex);
    for (const std::shared_ptr<QueuedWorkSegmentState>& segment : segments) {
        THOR_THROW_IF_FALSE(segment != nullptr);
        if (!segment->prepared ||
            segment->poppedBatches < segment->batchesPerEpoch) {
            return false;
        }
    }
    return true;
}

bool schedulingWindowHasLaterReportedEpoch(
    const NativeQueuedSchedulingWindowExecution& execution,
    uint64_t reportedEpoch) {
    for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
         execution.segments) {
        THOR_THROW_IF_FALSE(segment != nullptr);
        if (segment->reportedEpoch > reportedEpoch) {
            return true;
        }
    }
    return false;
}

uint64_t firstTrainedModelSelectionEpochForRequest(
    const TrainingRunRequest& request) {
    THOR_THROW_IF_FALSE(request.checkBestModelEveryEpochs > 0);
    // Epoch zero is the phase-entry incumbent when requested. Trained
    // candidates retain the historical cadence: firstModelSelectionEpoch=0
    // means the first post-update candidate is checked at the normal cadence.
    return request.firstModelSelectionEpoch == 0
        ? request.checkBestModelEveryEpochs
        : request.firstModelSelectionEpoch;
}

bool isTrainedModelSelectionDecisionEpoch(
    const TrainingRunRequest& request,
    uint64_t cumulativeEpoch) {
    if (request.checkBestModelEveryEpochs == 0) {
        return false;
    }
    THOR_THROW_IF_FALSE(cumulativeEpoch >= request.initialCompletedEpochs);
    const uint64_t phaseLocalEpoch =
        cumulativeEpoch - request.initialCompletedEpochs;
    const uint64_t firstTrainedModelSelectionEpoch =
        firstTrainedModelSelectionEpochForRequest(request);
    return phaseLocalEpoch >= firstTrainedModelSelectionEpoch &&
           ((phaseLocalEpoch - firstTrainedModelSelectionEpoch) %
                request.checkBestModelEveryEpochs ==
            0);
}

uint32_t schedulingWindowEpochCountThroughNextHostDecision(
    const TrainingRunRequest& request,
    bool evaluateOnly,
    uint64_t currentCompletedEpoch,
    uint32_t remainingEpochs) {
    THOR_THROW_IF_FALSE(remainingEpochs >= 1);
    if (evaluateOnly) {
        return 1;
    }

    // Epoch/phase transitions are not barriers. Keep submitting through the
    // persistent ring until validation results are actually needed by the host
    // to decide whether future optimizer work is allowed to proceed. Today that
    // semantic barrier is a trained model-selection / early-completion decision.
    if (request.checkBestModelEveryEpochs == 0) {
        return remainingEpochs;
    }

    for (uint32_t epochOffset = 1; epochOffset <= remainingEpochs; ++epochOffset) {
        if (isTrainedModelSelectionDecisionEpoch(
                request, currentCompletedEpoch + epochOffset)) {
            return epochOffset;
        }
    }
    return remainingEpochs;
}

NativeQueuedSchedulingWindowExecution launchNativeQueuedSchedulingWindow(
    const TrainingRunRequest& request,
    const std::shared_ptr<NativeQueuedSchedulerWorker>& schedulerWorker,
    const std::shared_ptr<BatchSession>& effectiveSession,
    uint64_t currentEpoch,
    bool evaluateOnly,
    std::optional<QueuedValidationPopulationMetadata>
        validationPopulationMetadata = std::nullopt,
    std::optional<uint64_t> reportedEpochOverride = std::nullopt,
    uint32_t epochsToSchedule = 1,
    const std::vector<NamedValidationSession>* additionalValidationSessions =
        nullptr) {
    THOR_THROW_IF_FALSE(schedulerWorker != nullptr);
    THOR_THROW_IF_FALSE(epochsToSchedule >= 1);
    if (evaluateOnly) {
        THOR_THROW_IF_FALSE(epochsToSchedule == 1);
    }
    const std::shared_ptr<NativeQueuedSchedulerResources>& schedulerResources =
        schedulerWorker->getResources();
    const std::shared_ptr<NativeQueuedRunState>& runState =
        schedulerWorker->getRunState();
    THOR_THROW_IF_FALSE(schedulerResources != nullptr);
    THOR_THROW_IF_FALSE(runState != nullptr);
    {
        std::lock_guard<std::mutex> lock(runState->mutex);
        // Commands remain non-overlapping, but one command may now span many
        // ordinary epochs. Cross-epoch flow happens inside this scheduling
        // window while the one persistent ring provides backpressure.
        THOR_THROW_IF_FALSE(runState->inFlightBatches == 0);
        THOR_THROW_IF_FALSE(runState->failure == nullptr);
        THOR_THROW_IF_FALSE(!runState->cancelRequested);
    }

    const uint64_t firstReportedEpoch =
        reportedEpochOverride.value_or(currentEpoch + 1);
    static const std::vector<NamedValidationSession> noAdditionalValidationSessions;
    const std::vector<NamedValidationSession>& namedValidationSessions =
        additionalValidationSessions != nullptr
            ? *additionalValidationSessions
            : noAdditionalValidationSessions;
    std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments =
        buildQueuedSchedulingWindowSegments(
        request,
        effectiveSession,
        namedValidationSessions,
        currentEpoch,
        firstReportedEpoch,
        evaluateOnly,
        epochsToSchedule,
        validationPopulationMetadata);

    NativeQueuedSchedulingWindowExecution execution;
    execution.segments = std::move(segments);
    execution.startedAt = std::chrono::high_resolution_clock::now();
    execution.state = std::make_shared<NativeQueuedSchedulingWindowState>(runState);
    execution.state->segments = execution.segments;

    execution.schedulerCommandCompletion = schedulerWorker->submit(
        NativeQueuedSchedulerCommand{
            execution.state,
            request.cancellationToken,
            execution.segments,
            nullptr});
    recordNativeQueuedSchedulingWindowLaunchForTests(
        schedulerResources.get(),
        runState.get(),
        runState->slots.data());

    return execution;
}

void waitForSchedulerCommandCompletion(
    const NativeQueuedSchedulingWindowExecution& execution) {
    THOR_THROW_IF_FALSE(execution.schedulerCommandCompletion != nullptr);
    execution.schedulerCommandCompletion->wait();
}

void waitForInitialQueueCompletion(
    const NativeQueuedSchedulingWindowExecution& execution) {
    std::unique_lock<std::mutex> lock(execution.state->runState->mutex);
    while (execution.state->runState->failure == nullptr &&
           !execution.state->runState->cancelRequested &&
           execution.state->completedBatchCallbacks == 0 &&
           !execution.state->schedulingFinished) {
        execution.state->runState->batchFinished.wait_for(
            lock, std::chrono::milliseconds(50));
    }

    const std::exception_ptr failure = execution.state->runState->failure;
    const bool cancelRequested = execution.state->runState->cancelRequested;
    const bool interruptRequested = execution.state->runState->interruptRequested;
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

bool abortNativeQueuedSchedulingWindowExecution(
    NativeQueuedSchedulingWindowExecution& execution,
    const std::shared_ptr<PlacedNetwork>& placedNetwork,
    std::exception_ptr failure,
    int deviceNum) noexcept {
    if (execution.state == nullptr) {
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(execution.state->runState->mutex);
        if (execution.state->runState->failure == nullptr) {
            execution.state->runState->failure = failure;
        }
        execution.state->runState->cancelRequested = true;
        execution.state->schedulingFinished = true;
    }
    execution.state->runState->batchFinished.notify_all();
    execution.state->runState->batchPopped.notify_all();

    std::set<BatchSession*> cancelledSessions;
    for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
         execution.segments) {
        if (segment == nullptr || segment->batchSession == nullptr) {
            continue;
        }
        if (!cancelledSessions.insert(segment->batchSession.get()).second) {
            continue;
        }
        try {
            cancelBatchSession(segment->batchSession);
        } catch (...) {
            return false;
        }
    }
    try {
        if (execution.schedulerCommandCompletion != nullptr) {
            waitForSchedulerCommandCompletion(execution);
        }
    } catch (...) {
        return false;
    }
    execution.schedulerCommandCompletion.reset();

    const bool submittedWorkDrained =
        drainQueuedTrainingWorkAfterFailure(placedNetwork, deviceNum);
    releaseSchedulingWindowStateReferencesAfterAbort(
        execution.state, submittedWorkDrained);
    return submittedWorkDrained;
}

class PendingNativeQueuedSchedulingWindowGuard {
   public:
    PendingNativeQueuedSchedulingWindowGuard(
        std::optional<NativeQueuedSchedulingWindowExecution>& execution,
        const std::shared_ptr<PlacedNetwork>& placedNetwork,
        int deviceNum)
        : execution(execution),
          placedNetwork(placedNetwork),
          deviceNum(deviceNum) {}

    ~PendingNativeQueuedSchedulingWindowGuard() {
        if (!execution.has_value() ||
            execution->schedulerCommandCompletion == nullptr) {
            return;
        }
        (void)abortNativeQueuedSchedulingWindowExecution(
            execution.value(),
            placedNetwork,
            std::make_exception_ptr(std::runtime_error(
                "Native queued training exited before consuming its active "
                "scheduling window.")),
            deviceNum);
    }

   private:
    std::optional<NativeQueuedSchedulingWindowExecution>& execution;
    const std::shared_ptr<PlacedNetwork>& placedNetwork;
    int deviceNum;
};

struct NativeQueuedStartupState {
    std::shared_ptr<PlacedNetwork> placedNetwork;
    std::shared_ptr<const ExecutableTrainingPlan> plan;
    std::shared_ptr<NativeQueuedSchedulerResources> schedulerResources;
    std::shared_ptr<NativeQueuedRunState> runState;
    std::shared_ptr<NativeQueuedSchedulerWorker> schedulerWorker;
    std::shared_ptr<BatchSession> sourceSession;
    std::shared_ptr<BatchSession> effectiveSession;
    std::vector<NamedValidationSession> additionalValidationSessions;
    DeviceDatasetStorageReport deviceDatasetStorageReport;
    std::optional<NativeQueuedSchedulingWindowExecution> firstSchedulingWindowExecution;
    std::optional<double> initialModelSelectionScore{};
};

void releaseFailedNativeQueuedStartupAttempt(
    NativeQueuedStartupState& attempt) noexcept {
    attempt.firstSchedulingWindowExecution.reset();
    attempt.schedulerWorker.reset();
    attempt.runState.reset();
    attempt.schedulerResources.reset();

    // Session leases may own resident dataset and per-session device tensors.
    // Release them before the placed graph, then destroy the executable plan
    // before its physical tensor references become invalid.
    for (NamedValidationSession& validation : attempt.additionalValidationSessions) {
        validation.batchSession.reset();
    }
    attempt.additionalValidationSessions.clear();
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

void validateFullEpochPhaseCompletion(
    const std::shared_ptr<QueuedWorkSegmentState>& segment);

TrainingModelSelectionContext evaluateInitialModelSelectionState(
    const TrainingRunRequest& request,
    const std::shared_ptr<NativeQueuedSchedulerWorker>& schedulerWorker,
    const std::shared_ptr<BatchSession>& defaultValidationSession,
    const std::vector<NamedValidationSession>& additionalValidationSessions,
    uint64_t cumulativeEpoch) {
    THOR_THROW_IF_FALSE(schedulerWorker != nullptr);
    const std::shared_ptr<NativeQueuedSchedulerResources>& schedulerResources =
        schedulerWorker->getResources();
    THOR_THROW_IF_FALSE(schedulerResources != nullptr);
    const std::shared_ptr<PlacedNetwork>& placedNetwork =
        schedulerResources->placedNetwork;
    EpochLossAccumulator losses;
    losses.ensureValidationPopulation(request.defaultValidationPopulation);
    for (const NamedValidationSession& validation : additionalValidationSessions) {
        losses.ensureValidationPopulation(validation.name);
    }

    TrainingRunRequest validationRequest = request;
    validationRequest.evaluationExampleType = ExampleType::VALIDATE;
    validationRequest.evaluationPhase = TrainingEventPhase::VALIDATE;

    auto evaluatePopulation = [&](const std::string& population,
                                  const std::shared_ptr<BatchSession>& session) {
        if (session == nullptr) {
            throw std::runtime_error(
                "Initial model-selection validation population '" + population +
                "' has a null BatchSession.");
        }

        NativeQueuedSchedulingWindowExecution execution = launchNativeQueuedSchedulingWindow(
            validationRequest,
            schedulerWorker,
            session,
            cumulativeEpoch,
            /*evaluateOnly=*/true,
            QueuedValidationPopulationMetadata{
                population,
                population == request.defaultValidationPopulation},
            cumulativeEpoch);
        const std::shared_ptr<NativeQueuedSchedulingWindowState> state = execution.state;

        try {
            while (true) {
                if (request.cancellationToken.isCancellationRequested()) {
                    requestQueuedTrainingCancellation(state);
                    cancelBatchSession(session);
                }

                BatchPopResult completedBatch = popBatchData(state);
                if (!completedBatch.hasBatch) {
                    break;
                }

                TrainingStatsSnapshot snapshot;
                snapshot.phase = TrainingEventPhase::VALIDATE;
                snapshot.epoch = cumulativeEpoch;
                snapshot.validationPopulation = population;
                snapshot.isDefaultValidationPopulation =
                    population == request.defaultValidationPopulation;
                snapshot.validExamplesInBatch = completedBatch.validExampleCount;
                assignScalarStatsToSnapshot(
                    snapshot,
                    state->runState->scalarTensorNames,
                    completedBatch.scalarStats,
                    state->runState->aggregateLossTensorNames);
                snapshot.metricBatchStats =
                    std::move(completedBatch.metricBatchStats);
                losses.update(snapshot);
            }

            waitForSchedulerCommandCompletion(execution);
            throwIfSchedulingWindowStateFailed(state);
            for (const std::shared_ptr<QueuedWorkSegmentState>& segment :
                 execution.segments) {
                validateFullEpochPhaseCompletion(segment);
            }
        } catch (...) {
            (void)abortNativeQueuedSchedulingWindowExecution(
                execution,
                placedNetwork,
                std::current_exception(),
                placedNetwork->getStampedNetwork(0).getGpuNum());
            throw;
        }
    };

    evaluatePopulation(
        request.defaultValidationPopulation, defaultValidationSession);
    for (const NamedValidationSession& validation : additionalValidationSessions) {
        evaluatePopulation(validation.name, validation.batchSession);
    }

    return losses.modelSelectionContext(
        cumulativeEpoch, request.defaultValidationPopulation);
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

std::vector<NamedValidationSession>
reopenNativeQueuedNamedValidationSessionsForRetry(
    const TrainingRunRequest& request,
    uint64_t expectedBatchSize) {
    std::vector<NamedValidationSession> reopened;
    reopened.reserve(request.additionalValidationSessions.size());
    for (const NamedValidationSession& configured :
         request.additionalValidationSessions) {
        if (!configured.batchSessionFactory) {
            throw std::runtime_error(
                "Native queued startup consumed named validation population '" +
                configured.name +
                "' before a retryable GPU out-of-memory failure, but no "
                "batchSessionFactory was supplied for that population.");
        }
        NamedValidationSession validation = configured;
        validation.batchSession = configured.batchSessionFactory();
        if (validation.batchSession == nullptr) {
            throw std::runtime_error(
                "Named validation batchSessionFactory returned null for population '" +
                configured.name + "' while retrying GPU startup.");
        }
        if (validation.batchSession->getBatchSize() != expectedBatchSize) {
            throw std::runtime_error(
                "Named validation batchSessionFactory returned a different batch size "
                "while retrying GPU startup for population '" + configured.name +
                "'.");
        }
        reopened.push_back(std::move(validation));
    }
    return reopened;
}

NativeQueuedStartupState startNativeQueuedTrainingWithMemoryAdmissionRetry(
    const TrainingRunRequest& request,
    const NativeQueuedExecutionGraph& executionGraph,
    bool evaluateOnly,
    uint64_t batchSize,
    const NativeQueuedTrainingOptions& options,
    const std::vector<std::string>& scalarTensorNames,
    const std::vector<std::string>& aggregateLossTensorNames,
    uint64_t currentEpoch,
    TrainingArtifactManager* initialModelSelectionArtifacts) {
    constexpr int startupDeviceNum = 0;
    auto notifyStatus = [&](TrainingRunStatus status) {
        if (request.statusCallback) {
            request.statusCallback(status);
        }
    };
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
    notifyStatus(TrainingRunStatus::WAITING_TO_START);
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
    notifyStatus(TrainingRunStatus::STARTING);
    bool emptyDeviceRetryAlreadyUsed = false;
    bool forceSourceSession = false;
    bool deviceDatasetFallbackAlreadyUsed = false;
    std::shared_ptr<BatchSession> nextSourceSession = request.batchSession;
    std::vector<NamedValidationSession> nextAdditionalValidationSessions =
        request.additionalValidationSessions;
    bool wrappedTailFallbackWarningEmitted = false;

    for (;;) {
        request.cancellationToken.throwIfCancellationRequested();
        ThorImplementation::clearDeviceStartupCudaErrorState(startupDeviceNum);
        if (initialModelSelectionArtifacts != nullptr) {
            // A retry creates a fresh placement and may randomly initialize newly
            // enabled phase parameters differently.  Never retain the epoch-0
            // candidate from a failed startup attempt.
            initialModelSelectionArtifacts->clearBestCandidate();
        }

        NativeQueuedStartupState attempt;
        attempt.sourceSession = nextSourceSession;
        attempt.effectiveSession = attempt.sourceSession;
        attempt.additionalValidationSessions = nextAdditionalValidationSessions;
        const std::vector<NamedValidationSession>
            sourceAdditionalValidationSessions =
                attempt.additionalValidationSessions;
        attempt.deviceDatasetStorageReport = request.deviceDatasetStorageReport;

        std::exception_ptr startupFailure;
        bool sessionWasConsumed = false;
        bool cleanupDrained = true;
        bool retryWithoutDeviceDataset = false;
        bool initialModelSelectionEvaluationFailed = false;
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

            configureWrappedTailFallback(
                attempt.placedNetwork,
                attempt.sourceSession,
                attempt.additionalValidationSessions,
                wrappedTailFallbackWarningEmitted);

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

                // Named validation sessions share the same immutable dataset
                // allocation. Select equivalent resident sessions so every
                // validation population uses the same input-source contract as
                // the default train/validate session.
                if (attempt.effectiveSession != attempt.sourceSession) {
                    bool namedValidationResidencyFallback = false;
                    std::string namedValidationFallbackReason;
                    for (NamedValidationSession& validation : attempt.additionalValidationSessions) {
                        const std::shared_ptr<BatchSession> sourceValidationSession =
                            validation.batchSession;
                        DeviceDatasetStorageSelection validationSelection =
                            selectDeviceDatasetStorageSession(
                                sourceValidationSession,
                                *request.trainingData,
                                request.trainingData->getSplits().withDefaultValidation(
                                    validation.name),
                                ThorImplementation::TensorPlacement(
                                    ThorImplementation::TensorPlacement::MemDevices::GPU,
                                    attempt.placedNetwork->getStampedNetwork(0).getGpuNum()),
                                deviceDatasetBatchQueueDepth);
                        if (validationSelection.session == sourceValidationSession) {
                            namedValidationResidencyFallback = true;
                            namedValidationFallbackReason =
                                validationSelection.report.reason.empty()
                                    ? std::string("named_validation_residency_unavailable")
                                    : validationSelection.report.reason;
                            break;
                        }
                        validation.batchSession =
                            std::move(validationSelection.session);
                    }
                    if (namedValidationResidencyFallback) {
                        // A placed network has one batch-input source contract.
                        // BEST_EFFORT therefore falls the entire run back to the
                        // source backend if any named validation population cannot
                        // obtain the same resident-session class as the default.
                        attempt.effectiveSession = attempt.sourceSession;
                        attempt.additionalValidationSessions =
                            sourceAdditionalValidationSessions;
                        attempt.deviceDatasetStorageReport.used = false;
                        attempt.deviceDatasetStorageReport.reason =
                            "named_validation_population_fallback:" +
                            namedValidationFallbackReason;
                        if (attempt.deviceDatasetStorageReport.windowedDeviceCache.attempted) {
                            attempt.deviceDatasetStorageReport.windowedDeviceCache.used = false;
                            attempt.deviceDatasetStorageReport.windowedDeviceCache.activeSources = 0;
                            attempt.deviceDatasetStorageReport.windowedDeviceCache.hitRatio = 0.0f;
                            attempt.deviceDatasetStorageReport.windowedDeviceCache.reason =
                                "device_dataset_fallback";
                        }
                    }
                }
            } else if (forceSourceSession) {
                attempt.deviceDatasetStorageReport.used = false;
                attempt.deviceDatasetStorageReport.reason =
                    "startup_memory_fallback";
                if (attempt.deviceDatasetStorageReport.windowedDeviceCache.attempted) {
                    attempt.deviceDatasetStorageReport.windowedDeviceCache.used = false;
                    attempt.deviceDatasetStorageReport.windowedDeviceCache.activeSources = 0;
                    attempt.deviceDatasetStorageReport.windowedDeviceCache.hitRatio = 0.0f;
                    attempt.deviceDatasetStorageReport.windowedDeviceCache.reason =
                        "device_dataset_fallback";
                }
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

            attempt.schedulerResources =
                std::make_shared<NativeQueuedSchedulerResources>(
                    attempt.placedNetwork, attempt.plan, options);
            attempt.runState = std::make_shared<NativeQueuedRunState>(
                options.maxInFlightBatches,
                scalarTensorNames,
                aggregateLossTensorNames);
            attempt.schedulerWorker =
                std::make_shared<NativeQueuedSchedulerWorker>(
                    attempt.schedulerResources, attempt.runState);

            if (initialModelSelectionArtifacts != nullptr) {
                // firstModelSelectionEpoch=0 means the exact model entering this
                // fit/phase is a candidate.  Evaluate and snapshot it before the
                // startup warmup can execute the first optimizer update.
                try {
                    const TrainingModelSelectionContext initialContext =
                        evaluateInitialModelSelectionState(
                            request,
                            attempt.schedulerWorker,
                            attempt.effectiveSession,
                            attempt.additionalValidationSessions,
                            currentEpoch);
                    recordNativeQueuedHostDecisionBarrierForTests();
                    const std::optional<double> initialScore =
                        request.modelSelectionScore.evaluate(initialContext);
                    attempt.initialModelSelectionScore = initialScore;
                    initialModelSelectionArtifacts->maybeSnapshotBestCandidate(
                        *attempt.placedNetwork, initialContext, initialScore);
                } catch (...) {
                    // Validation execution owns BatchSession cancellation on
                    // failure.  Do not feed those cancelled sessions into the
                    // normal GPU-memory retry path.
                    initialModelSelectionEvaluationFailed = true;
                    throw;
                }
            }

            // Admission is not complete merely because placement and slot
            // preallocation succeeded. Start the first scheduling window while
            // this FIFO startup turn is still held and wait for one real batch
            // to complete successfully. That proves the forward/backward/update
            // path and lazy allocations work. The scheduler continues filling
            // the remaining queue depth concurrently after admission.
            const uint32_t startupSchedulingWindowEpochs =
                schedulingWindowEpochCountThroughNextHostDecision(
                    request,
                    evaluateOnly,
                    currentEpoch,
                    request.epochs);
            attempt.firstSchedulingWindowExecution.emplace(
                launchNativeQueuedSchedulingWindow(
                    request,
                    attempt.schedulerWorker,
                    attempt.effectiveSession,
                    currentEpoch,
                    evaluateOnly,
                    std::nullopt,
                    std::nullopt,
                    startupSchedulingWindowEpochs,
                    &attempt.additionalValidationSessions));
            sessionWasConsumed = true;
            waitForInitialQueueCompletion(
                attempt.firstSchedulingWindowExecution.value());
            ThorImplementation::requireCleanDeviceStartupCudaErrorState(
                startupDeviceNum);

            startupGuard.complete(*attempt.placedNetwork);
            notifyStatus(TrainingRunStatus::RUNNING);
            return attempt;
        } catch (...) {
            startupFailure = std::current_exception();

            if (attempt.firstSchedulingWindowExecution.has_value()) {
                cleanupDrained = abortNativeQueuedSchedulingWindowExecution(
                    attempt.firstSchedulingWindowExecution.value(),
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

        if (initialModelSelectionEvaluationFailed) {
            std::rethrow_exception(startupFailure);
        }
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
                nextAdditionalValidationSessions =
                    reopenNativeQueuedNamedValidationSessionsForRetry(
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
            notifyStatus(TrainingRunStatus::WAITING_FOR_MEMORY);
            startupGuard.waitForModelRelease(
                [&]() {
                    request.cancellationToken.throwIfCancellationRequested();
                },
                retainedPlacement);
            notifyStatus(TrainingRunStatus::STARTING);
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

void validateFullEpochPhaseCompletion(
    const std::shared_ptr<QueuedWorkSegmentState>& segment) {
    THOR_THROW_IF_FALSE(segment != nullptr);
    THOR_THROW_IF_FALSE(segment->batchSession != nullptr);
    if (!segment->requiresEpochBoundaryValidation) {
        return;
    }

    if (segment->completedValidExamples != segment->validExamplesPerEpoch ||
        segment->poppedValidExamples != segment->validExamplesPerEpoch) {
        throw std::runtime_error(
            "Native queued " + phaseName(segment->phase) +
            " epoch completed/popped " +
            std::to_string(segment->completedValidExamples) + "/" +
            std::to_string(segment->poppedValidExamples) +
            " valid examples, but the segment expected " +
            std::to_string(segment->validExamplesPerEpoch) + ".");
    }
}

}  // namespace

namespace detail {

void resetNativeQueuedSchedulerResourceDiagnosticsForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.snapshot = NativeQueuedSchedulerResourceDiagnosticsForTests{};
    diagnostics.firstSchedulingWindowResourceIdentity = nullptr;
    diagnostics.multipleSchedulingWindowResourceIdentities = false;
    diagnostics.firstSchedulingWindowRunStateIdentity = nullptr;
    diagnostics.multipleSchedulingWindowRunStateIdentities = false;
    diagnostics.firstSchedulingWindowSlotStorageIdentity = nullptr;
    diagnostics.multipleSchedulingWindowSlotStorageIdentities = false;
    diagnostics.firstWorkerThreadId.reset();
    diagnostics.multipleWorkerThreadIds = false;
    diagnostics.enabled.store(true, std::memory_order_relaxed);
}

NativeQueuedSchedulerResourceDiagnosticsForTests
peekNativeQueuedSchedulerResourceDiagnosticsForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    return diagnostics.snapshot;
}

NativeQueuedSchedulerResourceDiagnosticsForTests
nativeQueuedSchedulerResourceDiagnosticsForTests() {
    NativeQueuedSchedulerResourceDiagnosticsState& diagnostics =
        nativeQueuedSchedulerResourceDiagnosticsState();
    std::lock_guard<std::mutex> lock(diagnostics.mutex);
    diagnostics.enabled.store(false, std::memory_order_relaxed);
    return diagnostics.snapshot;
}

}  // namespace detail

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
    NsightSystemsEpochCapture nsightSystemsEpochCapture(request, evaluateOnly);
    if (!evaluateOnly && request.checkBestModelEveryEpochs == 0 && !request.earlyCompletionPolicies.empty()) {
        throw std::runtime_error("Trainer early_completion_policies require check_best_model_every_epochs > 0.");
    }
    if (request.maxTrainingBatchesPerEpoch.has_value() && request.maxTrainingBatchesPerEpoch.value() == 0) {
        throw std::runtime_error("Trainer max_training_batches_per_epoch must be >= 1 or None.");
    }
    if (request.defaultValidationPopulation.empty()) {
        throw std::runtime_error("TrainingRunRequest default validation population must not be empty.");
    }
    std::set<std::string> validationPopulationNames{request.defaultValidationPopulation};
    for (const NamedValidationSession& validation : request.additionalValidationSessions) {
        if (validation.name.empty()) {
            throw std::runtime_error("TrainingRunRequest validation population names must not be empty.");
        }
        if (!validationPopulationNames.insert(validation.name).second) {
            throw std::runtime_error("TrainingRunRequest contains duplicate validation population '" +
                                     validation.name + "'.");
        }
        if (validation.batchSession == nullptr) {
            throw std::runtime_error("TrainingRunRequest validation population '" + validation.name +
                                     "' has a null BatchSession.");
        }
        if (validation.batchSession->getBatchSize() != request.batchSession->getBatchSize()) {
            throw std::runtime_error("TrainingRunRequest validation population '" + validation.name +
                                     "' uses a different batch size.");
        }
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

    TrainingArtifactManager trainingArtifacts(
        request.saveModelDirectory, request.saveModelOverwrite);
    TrainingArtifactManager* initialModelSelectionArtifacts =
        modelSelectionEnabled && request.firstModelSelectionEpoch == 0
            ? &trainingArtifacts
            : nullptr;

    NativeQueuedStartupState startup =
        startNativeQueuedTrainingWithMemoryAdmissionRetry(
            request,
            executionGraph,
            evaluateOnly,
            batchSize,
            options,
            scalarTensorNames,
            aggregateLossTensorNames,
            currentEpoch,
            initialModelSelectionArtifacts);
    std::shared_ptr<PlacedNetwork> placedNetwork =
        std::move(startup.placedNetwork);
    std::shared_ptr<NativeQueuedSchedulerResources> schedulerResources =
        std::move(startup.schedulerResources);
    std::shared_ptr<NativeQueuedRunState> runState =
        std::move(startup.runState);
    std::shared_ptr<NativeQueuedSchedulerWorker> schedulerWorker =
        std::move(startup.schedulerWorker);
    THOR_THROW_IF_FALSE(schedulerResources != nullptr);
    THOR_THROW_IF_FALSE(runState != nullptr);
    THOR_THROW_IF_FALSE(schedulerWorker != nullptr);
    THOR_THROW_IF_FALSE(
        schedulerWorker->getResources().get() == schedulerResources.get());
    THOR_THROW_IF_FALSE(
        schedulerWorker->getRunState().get() == runState.get());
    THOR_THROW_IF_FALSE(schedulerResources->plan != nullptr);
    // The scheduler resource owner now carries the executable plan for the
    // entire placed run. Drop the startup state's duplicate reference.
    startup.plan.reset();
    std::shared_ptr<BatchSession> effectiveSession =
        std::move(startup.effectiveSession);
    std::vector<NamedValidationSession> additionalValidationSessions =
        std::move(startup.additionalValidationSessions);
    DeviceDatasetStorageReport deviceDatasetStorageReport =
        std::move(startup.deviceDatasetStorageReport);
    const std::optional<double> initialModelSelectionScore =
        startup.initialModelSelectionScore;
    std::optional<NativeQueuedSchedulingWindowExecution> firstSchedulingWindowExecution =
        std::move(startup.firstSchedulingWindowExecution);
    PendingNativeQueuedSchedulingWindowGuard pendingFirstSchedulingWindowGuard(
        firstSchedulingWindowExecution,
        placedNetwork,
        placedNetwork->getStampedNetwork(0).getGpuNum());

    THOR_THROW_IF_FALSE(firstSchedulingWindowExecution.has_value());
    const auto runStart = firstSchedulingWindowExecution->startedAt;
    const double initialElapsedSeconds =
        evaluateOnly ? 0.0 : std::max(0.0, request.initialElapsedSeconds);
    std::map<TrainingEventPhase, ThorImplementation::PhaseWallThroughputTracker> throughputByPhase;
    std::map<std::string, ThorImplementation::PhaseWallThroughputTracker>
        validationThroughputByPopulation;
    std::array<uint64_t, 4> cappedReportedStepsByPhase{};
    std::array<uint64_t, 4> cappedReportedSamplesByPhase{};
    auto cancelAdditionalValidationSessions = [&]() {
        for (NamedValidationSession& validation : additionalValidationSessions) {
            cancelBatchSession(validation.batchSession);
        }
    };
    const bool namedValidationPopulationMetadataEnabled =
        request.defaultValidationPopulation != "validate" ||
        !additionalValidationSessions.empty();
    const bool trainingBatchCapEnabled =
        !evaluateOnly &&
        request.maxTrainingBatchesPerEpoch.has_value() &&
        effectiveSession->getNumBatchesPerEpoch(ExampleType::TRAIN) >
            request.maxTrainingBatchesPerEpoch.value();
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
                                const std::shared_ptr<NativeQueuedSchedulingWindowState>& state) {
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

    auto makeSegmentSnapshot = [&](
                                   const std::shared_ptr<QueuedWorkSegmentState>& segment,
                                   const std::shared_ptr<NativeQueuedSchedulingWindowState>& state) {
        THOR_THROW_IF_FALSE(segment != nullptr);
        THOR_THROW_IF_FALSE(segment->batchSession != nullptr);
        TrainingStatsSnapshot snapshot = makeBaseSnapshot(
            segment->phase,
            segment->reportedEpoch,
            batchSize,
            segment->batchesPerEpoch,
            state);
        snapshot.datasetName = segment->batchSession->getDatasetName();
        if (segment->phase == TrainingEventPhase::VALIDATE &&
            namedValidationPopulationMetadataEnabled &&
            segment->validationPopulation.has_value()) {
            snapshot.validationPopulation = segment->validationPopulation.value();
            snapshot.isDefaultValidationPopulation =
                segment->isDefaultValidationPopulation;
        }
        return snapshot;
    };

    auto throughputTrackerForSegment = [&](
                                            const std::shared_ptr<QueuedWorkSegmentState>& segment)
        -> ThorImplementation::PhaseWallThroughputTracker& {
        THOR_THROW_IF_FALSE(segment != nullptr);
        if (segment->phase == TrainingEventPhase::VALIDATE &&
            segment->validationPopulation.has_value()) {
            return validationThroughputByPopulation[
                segment->validationPopulation.value()];
        }
        return throughputByPhase[segment->phase];
    };

    bool runEarlyCompleted = false;
    std::optional<uint64_t> completedEpoch{};
    std::optional<double> latestModelSelectionScore = initialModelSelectionScore;
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
        if (firstSchedulingWindowExecution.has_value()) {
            (void)abortNativeQueuedSchedulingWindowExecution(
                firstSchedulingWindowExecution.value(),
                placedNetwork,
                std::current_exception(),
                placedNetwork->getStampedNetwork(0).getGpuNum());
        }
        throw;
    }

    std::optional<NativeQueuedSchedulingWindowExecution>
        activeSchedulingWindowExecution;
    PendingNativeQueuedSchedulingWindowGuard activeSchedulingWindowGuard(
        activeSchedulingWindowExecution,
        placedNetwork,
        placedNetwork->getStampedNetwork(0).getGpuNum());

    for (uint32_t epochOffset = 0; epochOffset < request.epochs; ++epochOffset) {
        const uint64_t cumulativeEpoch = currentEpoch + 1;
        nsightSystemsEpochCapture.beginEpoch(cumulativeEpoch);
        EpochLossAccumulator epochLosses;
        epochLosses.ensureValidationPopulation(request.defaultValidationPopulation);
        for (const NamedValidationSession& validation : additionalValidationSessions) {
            epochLosses.ensureValidationPopulation(validation.name);
        }

        if (!activeSchedulingWindowExecution.has_value()) {
            if (epochOffset == 0) {
                THOR_THROW_IF_FALSE(firstSchedulingWindowExecution.has_value());
                activeSchedulingWindowExecution.emplace(
                    std::move(firstSchedulingWindowExecution.value()));
                firstSchedulingWindowExecution.reset();
            } else {
                request.cancellationToken.throwIfCancellationRequested();
                const uint32_t remainingEpochs =
                    request.epochs - epochOffset;
                const uint32_t epochsToSchedule =
                    schedulingWindowEpochCountThroughNextHostDecision(
                        request,
                        evaluateOnly,
                        currentEpoch,
                        remainingEpochs);
                activeSchedulingWindowExecution.emplace(
                    launchNativeQueuedSchedulingWindow(
                        request,
                        schedulerWorker,
                        effectiveSession,
                        currentEpoch,
                        evaluateOnly,
                        std::nullopt,
                        std::nullopt,
                        epochsToSchedule,
                        &additionalValidationSessions));
            }
        }

        NativeQueuedSchedulingWindowExecution& schedulingWindowExecution =
            activeSchedulingWindowExecution.value();
        std::shared_ptr<NativeQueuedSchedulingWindowState> state =
            schedulingWindowExecution.state;
        std::vector<std::shared_ptr<QueuedWorkSegmentState>> segments =
            queuedSegmentsForReportedEpoch(
                schedulingWindowExecution, cumulativeEpoch);
        if (segments.empty()) {
            throw std::runtime_error(
                "Native queued scheduling window does not contain requested epoch " +
                std::to_string(cumulativeEpoch) + ".");
        }
        const bool schedulingWindowContinuesPastEpoch =
            schedulingWindowHasLaterReportedEpoch(
                schedulingWindowExecution, cumulativeEpoch);

        std::vector<bool> segmentStarted(segments.size(), false);
        std::vector<bool> segmentFinished(segments.size(), false);
        size_t lifecycleSegmentIndex = 0;
        auto emitReadyPhaseLifecycleEvents = [&]() {
            while (lifecycleSegmentIndex < segments.size()) {
                const std::shared_ptr<QueuedWorkSegmentState>& segment =
                    segments[lifecycleSegmentIndex];
                THOR_THROW_IF_FALSE(segment != nullptr);
                {
                    std::unique_lock<std::mutex> lock(state->runState->mutex);
                    while (state->runState->failure == nullptr &&
                           !state->runState->cancelRequested &&
                           !segment->prepared &&
                           !state->schedulingFinished) {
                        state->runState->batchFinished.wait_for(
                            lock, std::chrono::milliseconds(50));
                    }
                    if (!segment->prepared) {
                        return;
                    }
                }

                if (!segmentStarted[lifecycleSegmentIndex]) {
                    segmentStarted[lifecycleSegmentIndex] = true;
                    TrainingStatsSnapshot startedStats =
                        makeSegmentSnapshot(segment, state);
                    // Model-selection scoring happens after a decision epoch's
                    // validation phases complete. Carry that latest completed score
                    // on the next TRAIN lifecycle event without making lifecycle
                    // delivery a scheduler synchronization point.
                    if (segment->phase == TrainingEventPhase::TRAIN &&
                        latestModelSelectionScore.has_value() &&
                        std::isfinite(latestModelSelectionScore.value()) &&
                        trainingArtifacts.getBestEpoch().has_value() &&
                        trainingArtifacts.getBestScore().has_value()) {
                        startedStats.metrics["latest_score"] = latestModelSelectionScore.value();
                        startedStats.metrics["best_epoch"] =
                            static_cast<double>(trainingArtifacts.getBestEpoch().value());
                        startedStats.metrics["best_score"] = trainingArtifacts.getBestScore().value();
                    }
                    emitTrainingEvent(observer, TrainingEvent::epochStarted(std::move(startedStats)));
                }

                if (segment->poppedBatches < segment->batchesPerEpoch) {
                    break;
                }

                if (!segmentFinished[lifecycleSegmentIndex]) {
                    segmentFinished[lifecycleSegmentIndex] = true;
                    emitTrainingEvent(
                        observer,
                        TrainingEvent::epochFinished(
                            makeSegmentSnapshot(segment, state)));
                }
                lifecycleSegmentIndex += 1;
            }
        };

        auto cancelSchedulingWindow = [&](std::exception_ptr failure) {
            (void)abortNativeQueuedSchedulingWindowExecution(
                schedulingWindowExecution,
                placedNetwork,
                failure,
                placedNetwork->getStampedNetwork(0).getGpuNum());
        };

        auto requestExternalCancel = [&]() {
            if (request.cancellationToken.isCancellationRequested()) {
                requestQueuedTrainingCancellation(state);
                cancelBatchSession(effectiveSession);
                cancelAdditionalValidationSessions();
            }
            if (sigintScope.interrupted()) {
                {
                    std::lock_guard<std::mutex> lock(state->runState->mutex);
                    state->runState->cancelRequested = true;
                    state->runState->interruptRequested = true;
                }
                state->runState->batchFinished.notify_all();
                state->runState->batchPopped.notify_all();
                cancelBatchSession(effectiveSession);
                cancelAdditionalValidationSessions();
            }
        };

        try {
            request.cancellationToken.throwIfCancellationRequested();
            emitReadyPhaseLifecycleEvents();

            while (!queuedSegmentsFullyPopped(state, segments)) {
                requestExternalCancel();
                BatchPopResult completedBatch =
                    popBatchData(state, cumulativeEpoch);
                if (!completedBatch.hasBatch) {
                    throwIfSchedulingWindowStateFailed(state);
                    if (queuedSegmentsFullyPopped(state, segments)) {
                        break;
                    }
                    throw std::runtime_error(
                        "Native queued scheduling window reached a later epoch "
                        "before the current epoch was fully consumed.");
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
                        std::lock_guard<std::mutex> lock(state->runState->mutex);
                        failure = state->runState->failure;
                    }
                    if (failure != nullptr) {
                        std::rethrow_exception(failure);
                    }
                }

                {
                    THOR_THROW_IF_FALSE(completedBatch.segment != nullptr);
                    const std::shared_ptr<QueuedWorkSegmentState>& completedSegment =
                        completedBatch.segment;
                    TrainingStatsSnapshot snapshot =
                        makeSegmentSnapshot(completedSegment, state);
                    snapshot.inFlightBatches = completedBatch.inFlightAfterPop;
                    snapshot.stepInEpoch = completedBatch.epochBatchNum + 1;
                    const size_t phaseIndex =
                        queuedPhaseIndex(completedBatch.phase);
                    if (trainingBatchCapEnabled &&
                        completedBatch.phase == TrainingEventPhase::TRAIN) {
                        snapshot.step = cappedReportedStepsByPhase[phaseIndex] + 1;
                        cappedReportedStepsByPhase[phaseIndex] += 1;
                    } else {
                        snapshot.step =
                            (currentEpoch * completedBatch.batchesInEpoch) +
                            snapshot.stepInEpoch;
                    }
                    snapshot.validExamplesInBatch =
                        completedBatch.validExampleCount;
                    snapshot.samplesProcessedInEpoch =
                        completedBatch.validExamplesInEpoch;
                    if (trainingBatchCapEnabled &&
                        completedBatch.phase == TrainingEventPhase::TRAIN) {
                        cappedReportedSamplesByPhase[phaseIndex] +=
                            completedBatch.validExampleCount;
                        snapshot.samplesProcessed =
                            cappedReportedSamplesByPhase[phaseIndex];
                    } else {
                        const uint64_t examplesProcessedPerEpoch =
                            ThorImplementation::BatchSessionRuntimeAccess::examplesProcessedPerEpoch(
                                *completedSegment->batchSession,
                                completedBatch.exampleType);
                        snapshot.samplesProcessed =
                            (currentEpoch * examplesProcessedPerEpoch) +
                            completedBatch.validExamplesInEpoch;
                    }

                    // Samples use the exact valid-example count. Logical FLOPs
                    // and bytes use the paired work captured for this exact
                    // submitted batch, so both rates share one active-phase wall
                    // interval and one EMA update. In particular, ragged
                    // Attention counts only score pairs from the published row
                    // partitions rather than packed capacity.
                    throughputTrackerForSegment(completedSegment).observeCompletedBatch(
                        snapshot,
                        completedBatch.phaseStartedAt,
                        completedBatch.completionTime,
                        completedBatch.validExampleCount,
                        completedBatch.logicalWork,
                        completedBatch.poppedInEpoch >= completedBatch.batchesInEpoch);

                    assignScalarStatsToSnapshot(snapshot,
                                                state->runState->scalarTensorNames,
                                                completedBatch.scalarStats,
                                                state->runState->aggregateLossTensorNames);
                    snapshot.metricBatchStats =
                        std::move(completedBatch.metricBatchStats);
                    epochLosses.update(snapshot);
                    emitTrainingEvent(observer, TrainingEvent::statsUpdated(std::move(snapshot)));
                }

                emitReadyPhaseLifecycleEvents();
            }

            // A scheduling window may already be submitting later epochs. Do
            // not wait for the command or drain the persistent ring merely to
            // finalize this logical epoch. Only the last epoch in a window owns
            // the command-completion wait.
            if (!schedulingWindowContinuesPastEpoch) {
                waitForSchedulerCommandCompletion(schedulingWindowExecution);
                throwIfSchedulingWindowStateFailed(state);
            }
            for (const std::shared_ptr<QueuedWorkSegmentState>& segment : segments) {
                validateFullEpochPhaseCompletion(segment);
            }
            emitReadyPhaseLifecycleEvents();
        } catch (...) {
            cancelSchedulingWindow(std::current_exception());
            throw;
        }

        // Default and named validation populations for this logical epoch are
        // already part of the scheduling window and have been consumed above in
        // submission order. No host-side queue drain or standalone named
        // validation command is required here.
        //
        // Deliberately do not drain the cross-epoch queue for profiling either.
        // A few boundary batches may already be in flight, while the interior of
        // the requested epoch range remains representative steady state.
        nsightSystemsEpochCapture.endEpoch(cumulativeEpoch);

        bool earlyCompletionRequested = false;
        const bool modelSelectionEligible =
            modelSelectionEnabled &&
            isTrainedModelSelectionDecisionEpoch(request, cumulativeEpoch);
        if (modelSelectionEligible) {
            if (epochOffset + 1 < request.epochs) {
                recordNativeQueuedHostDecisionBarrierForTests();
            }
            const TrainingModelSelectionContext currentSelectionContext =
                epochLosses.modelSelectionContext(
                    cumulativeEpoch, request.defaultValidationPopulation);
            const std::optional<double> currentScore = request.modelSelectionScore.evaluate(currentSelectionContext);
            latestModelSelectionScore = currentScore;
            trainingArtifacts.maybeSnapshotBestCandidate(*placedNetwork, currentSelectionContext, currentScore);
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
        latestValidationLoss = epochLosses.validationLoss(request.defaultValidationPopulation);
        latestEpochSelectionContext = epochLosses.modelSelectionContext(
            cumulativeEpoch, request.defaultValidationPopulation);
        latestEpochSelectionContextValid = true;
        currentEpoch += 1;
        if (!schedulingWindowContinuesPastEpoch) {
            activeSchedulingWindowExecution.reset();
        }
        if (earlyCompletionRequested) {
            runEarlyCompleted = true;
            completedEpoch = cumulativeEpoch;
            break;
        }
    }

    // All scheduling windows have crossed their semantic completion barriers.
    // The scheduler worker is no longer needed for this
    // run, so stop and join it once while keeping the reusable scheduler
    // resources alive for final checkpoint/save handoff work.
    schedulerWorker->shutdown();

    request.cancellationToken.throwIfCancellationRequested();
    const uint64_t finalCompletedEpoch = completedEpoch.value_or(currentEpoch);
    const char* finalCompletionReason = runEarlyCompleted ? "early_completed" : "completed";
    const uint64_t finalCompletedPhaseEpoch = finalCompletedEpoch - request.initialCompletedEpochs;
    const bool finalModelSelectionEligible =
        modelSelectionEnabled &&
        finalCompletedPhaseEpoch >=
            firstTrainedModelSelectionEpochForRequest(request);
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
        trainingArtifacts.maybeSnapshotBestCandidate(*placedNetwork, finalSelectionContext, finalScore);
    }
    std::optional<uint64_t> selectedModelEpoch{};
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
        selectedModelEpoch = selectedArtifactEpoch.value_or(finalCompletedEpoch);

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
    if (selectedModelEpoch.has_value()) {
        finishedStats.metrics["selected_epoch"] = static_cast<double>(selectedModelEpoch.value());
    }
    if (latestModelSelectionScore.has_value() && std::isfinite(latestModelSelectionScore.value())) {
        finishedStats.metrics["latest_score"] = latestModelSelectionScore.value();
    }
    if (trainingArtifacts.getBestEpoch().has_value()) {
        finishedStats.metrics["best_epoch"] = static_cast<double>(trainingArtifacts.getBestEpoch().value());
    }
    if (trainingArtifacts.getBestScore().has_value()) {
        finishedStats.metrics["best_score"] = trainingArtifacts.getBestScore().value();
    }
    emitTrainingEvent(observer,
                      TrainingEvent::runFinished(std::move(finishedStats),
                                                 finalCompletionReason,
                                                 trainingArtifacts.getBestModelSelectionContext()));
}

}  // namespace Thor
