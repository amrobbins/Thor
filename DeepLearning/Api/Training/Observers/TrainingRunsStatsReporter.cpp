#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace Thor {
namespace {

std::FILE* requireTrainingRunsOutput(std::FILE* output) {
    if (output == nullptr) {
        throw std::runtime_error("TrainingRunsStatsReporter requires a non-null output file.");
    }
    return output;
}

const char* terminalStatusName(const TrainingRunResult& result) { return trainingRunStatusName(result.status); }

std::optional<double> lossFromEvent(const std::optional<TrainingStatsEvent>& event) {
    if (!event.has_value()) {
        return std::nullopt;
    }
    return event->stats.loss;
}

// Final-report colors live here so they are easy to tune without touching the
// single-trainer LineStatsReporter palette.  These are ANSI SGR sequences.
namespace FinalReportAnsi {
constexpr const char* reset = "\x1b[0m";
constexpr const char* border = "\x1b[1;38;5;33m";
constexpr const char* completed = "\x1b[1;38;5;28m";   // matches LineStatsReporter phase=train green
constexpr const char* failed = "\x1b[1;38;5;196m";
constexpr const char* cancelled = "\x1b[1;38;5;208m";
constexpr const char* interrupted = "\x1b[1;38;5;129m";
constexpr const char* outOfMemory = "\x1b[1;38;5;196m";
constexpr const char* notStarted = "\x1b[38;5;244m";
constexpr const char* running = "\x1b[1;38;5;21m";
constexpr const char* trainLoss = "\x1b[1;38;5;208m";
constexpr const char* validateLoss = "\x1b[1;38;5;33m";
constexpr const char* testLoss = "\x1b[1;38;5;129m";
}  // namespace FinalReportAnsi

}  // namespace

TrainingRunsStatsReporter::TrainingRunsStatsReporter(std::FILE* output, LineStatsColorMode colorMode, double maxSummaryLogsPerSecond)
    : output(requireTrainingRunsOutput(output)), colorMode(colorMode), maxSummaryLogsPerSecond(maxSummaryLogsPerSecond) {
    if (!std::isfinite(maxSummaryLogsPerSecond) || maxSummaryLogsPerSecond < 0.0) {
        throw std::runtime_error("TrainingRunsStatsReporter maxSummaryLogsPerSecond must be finite and >= 0.");
    }
    worker = std::thread([this]() { workerLoop(); });
}

TrainingRunsStatsReporter::~TrainingRunsStatsReporter() { close(); }

void TrainingRunsStatsReporter::configureRun(std::string runName, RunConfig config) {
    if (runName.empty()) {
        throw std::runtime_error("TrainingRunsStatsReporter run name must not be empty.");
    }
    if (!std::isfinite(config.intervalSeconds) || config.intervalSeconds < 0.0) {
        throw std::runtime_error("TrainingRunsStatsReporter intervalSeconds must be finite and >= 0.");
    }
    if (config.ensembleGroup.has_value() && config.ensembleGroup->empty()) {
        throw std::runtime_error("TrainingRunsStatsReporter ensembleGroup must not be empty when specified.");
    }
    if (!std::isfinite(config.ensembleWeight) || config.ensembleWeight <= 0.0) {
        throw std::runtime_error("TrainingRunsStatsReporter ensembleWeight must be finite and > 0.");
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (closeRequested) {
        throw std::runtime_error("TrainingRunsStatsReporter cannot configure runs after close.");
    }
    if (runStateIndices.find(runName) != runStateIndices.end()) {
        throw std::runtime_error("TrainingRunsStatsReporter run '" + runName + "' has already been configured.");
    }

    RunState state;
    state.runName = std::move(runName);
    state.config = config;
    state.status = DisplayStatus::NOT_STARTED;
    runStateIndices.emplace(state.runName, runStates.size());
    runStates.push_back(std::move(state));
}

void TrainingRunsStatsReporter::markRunStarting(const std::string& runName) {
    ReporterEvent event;
    event.type = ReporterEventType::RUN_STARTING;
    event.runName = runName;
    enqueueEvent(std::move(event));
}

void TrainingRunsStatsReporter::markRunFinished(const TrainingRunResult& result) {
    ReporterEvent event;
    event.type = ReporterEventType::RUN_FINISHED;
    event.runName = result.runName;
    event.result = result;
    enqueueEvent(std::move(event));
}

void TrainingRunsStatsReporter::emitFinalReport(const std::vector<TrainingRunResult>& results) {
    std::lock_guard<std::mutex> lock(mutex);
    size_t runPrefixWidth = runPrefixWidthLocked();
    for (const TrainingRunResult& result : results) {
        runPrefixWidth = std::max(runPrefixWidth, std::string("INFO runs[").size() + result.runName.size() + std::string("]:").size());
    }

    size_t completed = 0;
    size_t failed = 0;
    size_t cancelled = 0;
    size_t interrupted = 0;
    size_t oom = 0;
    size_t notStarted = 0;
    size_t running = 0;
    for (const TrainingRunResult& result : results) {
        switch (result.status) {
            case TrainingRunStatus::NOT_STARTED:
                ++notStarted;
                break;
            case TrainingRunStatus::RUNNING:
                ++running;
                break;
            case TrainingRunStatus::COMPLETED:
                ++completed;
                break;
            case TrainingRunStatus::FAILED:
                ++failed;
                break;
            case TrainingRunStatus::CANCELLED:
                ++cancelled;
                break;
            case TrainingRunStatus::INTERRUPTED:
                ++interrupted;
                break;
            case TrainingRunStatus::OUT_OF_MEMORY:
                ++oom;
                break;
        }
    }

    const bool useColor = shouldUseColorLocked();

    emitLineLocked("");
    emitLineLocked(styled("INFO runs final: ==================== final results ====================", FinalReportAnsi::border, useColor));

    std::string line = "INFO runs final:";
    line += " total=" + std::to_string(results.size());
    line += " not_started=" + std::to_string(notStarted);
    line += " running=" + std::to_string(running);
    line += " completed=" + std::to_string(completed);
    line += " failed=" + std::to_string(failed);
    line += " cancelled=" + std::to_string(cancelled);
    line += " interrupted=" + std::to_string(interrupted);
    line += " oom=" + std::to_string(oom);
    emitLineLocked(line);

    for (const TrainingRunResult& result : results) {
        writeResultLineLocked(result, runPrefixWidth);
    }
    emitLineLocked(styled("INFO runs final: =====================================================", FinalReportAnsi::border, useColor));
    emitLineLocked("");
    flushOutputLocked();
}

void TrainingRunsStatsReporter::emitEnsembleReport(const std::vector<TrainingEnsembleResult>& ensembles) {
    if (ensembles.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    size_t ensemblePrefixWidth = 0;
    for (const TrainingEnsembleResult& ensemble : ensembles) {
        ensemblePrefixWidth =
            std::max(ensemblePrefixWidth, std::string("INFO runs ensemble[").size() + ensemble.ensembleGroup.size() + std::string("]:").size());
    }

    const bool useColor = shouldUseColorLocked();
    emitLineLocked("");
    emitLineLocked(styled("INFO runs ensemble: ================= ensemble results =================", FinalReportAnsi::border, useColor));
    emitLineLocked("INFO runs ensemble: total=" + std::to_string(ensembles.size()));
    for (const TrainingEnsembleResult& ensemble : ensembles) {
        writeEnsembleLineLocked(ensemble, ensemblePrefixWidth);
    }
    emitLineLocked(styled("INFO runs ensemble: ===================================================", FinalReportAnsi::border, useColor));
    emitLineLocked("");
    flushOutputLocked();
}

void TrainingRunsStatsReporter::onStatsEvent(const TrainingStatsEvent& event) {
    ReporterEvent reporterEvent;
    reporterEvent.type = ReporterEventType::RUN_STATS;
    reporterEvent.runName = event.runName;
    reporterEvent.stats = event;
    enqueueEvent(std::move(reporterEvent));
}

void TrainingRunsStatsReporter::enqueueEvent(ReporterEvent event) {
    std::lock_guard<std::mutex> lock(mutex);
    if (closeRequested) {
        return;
    }
    queue.push_back(std::move(event));
    workAvailable.notify_one();
}

void TrainingRunsStatsReporter::flush() {
    std::unique_lock<std::mutex> lock(mutex);
    drained.wait(lock, [this]() { return queue.empty() && !processingEvent; });
    maybeEmitSummary(std::chrono::steady_clock::now(), true);
    flushOutputLocked();
}

void TrainingRunsStatsReporter::close() {
    std::thread workerToJoin;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closeRequested && !worker.joinable()) {
            return;
        }
        closeRequested = true;
    }
    workAvailable.notify_one();
    if (worker.joinable()) {
        workerToJoin = std::move(worker);
    }
    if (workerToJoin.joinable()) {
        workerToJoin.join();
    }

    std::lock_guard<std::mutex> lock(mutex);
    maybeEmitSummary(std::chrono::steady_clock::now(), true);
    flushOutputLocked();
}

void TrainingRunsStatsReporter::workerLoop() noexcept {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto nextDeadline = emittedAnySummary ? lastSummaryTime + summaryInterval() : std::chrono::steady_clock::time_point::min();
        if (!closeRequested && queue.empty()) {
            if (dirty && emittedAnySummary && maxSummaryLogsPerSecond > 0.0) {
                workAvailable.wait_until(lock, nextDeadline, [this]() { return closeRequested || !queue.empty(); });
            } else {
                workAvailable.wait(lock, [this]() { return closeRequested || !queue.empty() || dirty; });
            }
        }

        while (!queue.empty()) {
            ReporterEvent event = std::move(queue.front());
            queue.pop_front();
            processingEvent = true;
            try {
                processEvent(event);
            } catch (...) {
                // Logging must never take down training. Drop malformed reporter events and keep
                // draining so TrainingRuns can still join workers and return structured results.
            }
            processingEvent = false;
        }

        const auto now = std::chrono::steady_clock::now();
        maybeEmitSummary(now, closeRequested);

        if (queue.empty() && !processingEvent) {
            drained.notify_all();
        }
        if (closeRequested && queue.empty()) {
            break;
        }
    }
}

void TrainingRunsStatsReporter::processEvent(const ReporterEvent& event) {
    if (event.runName.empty()) {
        return;
    }

    RunState& state = stateForRun(event.runName);

    switch (event.type) {
        case ReporterEventType::RUN_STARTING:
            if (state.status == DisplayStatus::NOT_STARTED) {
                state.status = DisplayStatus::STARTING;
            }
            state.dirty = true;
            dirty = true;
            break;
        case ReporterEventType::RUN_STATS:
            if (!state.config.enabled || event.stats.type != TrainingEventType::STATS) {
                break;
            }
            state.latestStats = event.stats;
            switch (event.stats.stats.phase) {
                case TrainingEventPhase::TRAIN:
                    state.latestTrainingStats = event.stats;
                    break;
                case TrainingEventPhase::VALIDATE:
                    state.latestValidationStats = event.stats;
                    break;
                case TrainingEventPhase::TEST:
                    state.latestTestStats = event.stats;
                    break;
                case TrainingEventPhase::UNKNOWN:
                default:
                    break;
            }
            if (state.status == DisplayStatus::NOT_STARTED || state.status == DisplayStatus::STARTING) {
                state.status = DisplayStatus::RUNNING;
            }
            state.dirty = true;
            dirty = true;
            break;
        case ReporterEventType::RUN_FINISHED:
            state.terminalResult = event.result;
            state.status = displayStatusFromRunStatus(event.result.status);
            state.dirty = true;
            dirty = true;
            break;
    }
}

TrainingRunsStatsReporter::RunState& TrainingRunsStatsReporter::stateForRun(const std::string& runName) {
    const auto existing = runStateIndices.find(runName);
    if (existing != runStateIndices.end()) {
        return runStates[existing->second];
    }

    RunState state;
    state.runName = runName;
    state.status = DisplayStatus::NOT_STARTED;
    runStateIndices.emplace(state.runName, runStates.size());
    runStates.push_back(std::move(state));
    dirty = true;
    return runStates.back();
}

void TrainingRunsStatsReporter::maybeEmitSummary(std::chrono::steady_clock::time_point now, bool force) {
    if (runStates.empty() || !anyDirtyLocked()) {
        return;
    }
    if (!force && emittedAnySummary && maxSummaryLogsPerSecond > 0.0 && now - lastSummaryTime < summaryInterval()) {
        return;
    }
    emitSummaryLocked(now);
}

void TrainingRunsStatsReporter::emitSummaryLocked(std::chrono::steady_clock::time_point now) {
    const size_t runPrefixWidth = runPrefixWidthLocked();
    writeSummaryHeaderLocked("summary");
    for (const RunState& state : runStates) {
        writeRunLineLocked(state, runPrefixWidth);
    }
    emittedAnySummary = true;
    lastSummaryTime = now;
    clearDirtyLocked();
    flushOutputLocked();
}

void TrainingRunsStatsReporter::writeSummaryHeaderLocked(std::string_view label) {
    size_t notStarted = 0;
    size_t starting = 0;
    size_t running = 0;
    size_t completed = 0;
    size_t failed = 0;
    size_t cancelled = 0;
    size_t interrupted = 0;
    size_t oom = 0;
    size_t enabledRuns = 0;

    for (const RunState& state : runStates) {
        ++enabledRuns;
        switch (state.status) {
            case DisplayStatus::NOT_STARTED:
                ++notStarted;
                break;
            case DisplayStatus::STARTING:
                ++starting;
                break;
            case DisplayStatus::RUNNING:
                ++running;
                break;
            case DisplayStatus::COMPLETED:
                ++completed;
                break;
            case DisplayStatus::FAILED:
                ++failed;
                break;
            case DisplayStatus::CANCELLED:
                ++cancelled;
                break;
            case DisplayStatus::INTERRUPTED:
                ++interrupted;
                break;
            case DisplayStatus::OUT_OF_MEMORY:
                ++oom;
                break;
        }
    }

    std::string line = "INFO runs " + std::string(label) + ":";
    line += " total=" + std::to_string(enabledRuns);
    line += " not_started=" + std::to_string(notStarted);
    line += " starting=" + std::to_string(starting);
    line += " running=" + std::to_string(running);
    line += " completed=" + std::to_string(completed);
    line += " failed=" + std::to_string(failed);
    line += " cancelled=" + std::to_string(cancelled);
    line += " interrupted=" + std::to_string(interrupted);
    line += " oom=" + std::to_string(oom);
    emitLineLocked(line);
}

void TrainingRunsStatsReporter::writeRunLineLocked(const RunState& state, size_t runPrefixWidth) {
    if (state.status == DisplayStatus::RUNNING && state.latestStats.has_value()) {
        std::string line = LineStatsReporter::formatStatsLine(state.latestStats->stats, state.runName, runPrefixWidth, colorMode, output);
        appendRunMetadataLocked(line, state);
        appendPhaseLossesLocked(line, state);
        emitLineLocked(line);
        return;
    }

    if (state.terminalResult.has_value()) {
        writeResultLineLocked(state.terminalResult.value(), runPrefixWidth);
        return;
    }

    std::string line = formatRunPrefix(state.runName, runPrefixWidth) + " status=" + displayStatusName(state.status);
    appendRunMetadataLocked(line, state);
    appendPhaseLossesLocked(line, state);
    emitLineLocked(line);
}

void TrainingRunsStatsReporter::appendRunMetadataLocked(std::string& line, const RunState& state) {
    if (!state.config.ensembleGroup.has_value()) {
        return;
    }
    line += " ensemble_group=";
    line += *state.config.ensembleGroup;
    if (state.config.ensembleWeight != 1.0) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), " ensemble_weight=%.6f", state.config.ensembleWeight);
        line += buffer;
    }
}

void TrainingRunsStatsReporter::appendPhaseLossesLocked(std::string& line, const RunState& state) {
    const bool useColor = shouldUseColorLocked();
    const std::optional<double> trainLoss = lossFromEvent(state.latestTrainingStats);
    if (trainLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "train_loss=%.6f", trainLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::trainLoss, useColor);
    }
    const std::optional<double> validateLoss = lossFromEvent(state.latestValidationStats);
    if (validateLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "validate_loss=%.6f", validateLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::validateLoss, useColor);
    }
    const std::optional<double> testLoss = lossFromEvent(state.latestTestStats);
    if (testLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "test_loss=%.6f", testLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::testLoss, useColor);
    }
}

void TrainingRunsStatsReporter::writeResultLineLocked(const TrainingRunResult& result, size_t runPrefixWidth) {
    const bool useColor = shouldUseColorLocked();
    const char* statusStyle = statusColorStyle(result.status);

    std::string line = formatRunPrefix(result.runName, runPrefixWidth);
    line += " ";
    line += styled(std::string("status=") + displayStatusName(displayStatusFromRunStatus(result.status)), statusStyle, useColor);
    line += " ";
    line += styled(std::string("result=") + terminalStatusName(result), statusStyle, useColor);
    if (result.ensembleGroup.has_value()) {
        line += " ensemble_group=";
        line += *result.ensembleGroup;
        if (result.ensembleWeight != 1.0) {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), " ensemble_weight=%.6f", result.ensembleWeight);
            line += buffer;
        }
    }
    if (result.finalTrainingStats.has_value() && result.finalTrainingStats->loss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "final_train_loss=%.6f", result.finalTrainingStats->loss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::trainLoss, useColor);
    }
    if (result.finalValidationStats.has_value() && result.finalValidationStats->loss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "final_validate_loss=%.6f", result.finalValidationStats->loss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::validateLoss, useColor);
    }
    if (result.finalTestStats.has_value() && result.finalTestStats->loss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "final_test_loss=%.6f", result.finalTestStats->loss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::testLoss, useColor);
    }
    if (!result.exception.message.empty()) {
        line += " message=\"" + result.exception.message + "\"";
    }
    emitLineLocked(line);
}

void TrainingRunsStatsReporter::writeEnsembleLineLocked(const TrainingEnsembleResult& result, size_t ensemblePrefixWidth) {
    const bool useColor = shouldUseColorLocked();
    const TrainingRunStatus aggregateStatus = result.allCompleted() ? TrainingRunStatus::COMPLETED
                                                                    : (result.anyFailed() ? TrainingRunStatus::FAILED : TrainingRunStatus::RUNNING);
    const char* statusStyle = statusColorStyle(aggregateStatus);

    const std::map<std::string, size_t> counts = result.statusCounts();
    auto count = [&counts](const std::string& name) -> size_t {
        const auto it = counts.find(name);
        return it == counts.end() ? 0u : it->second;
    };

    char weightBuffer[64];
    std::snprintf(weightBuffer, sizeof(weightBuffer), "total_weight=%.6f", result.totalWeight());

    std::string line = formatEnsemblePrefix(result.ensembleGroup, ensemblePrefixWidth);
    line += " ";
    line += styled(std::string("status=") + trainingRunStatusName(aggregateStatus), statusStyle, useColor);
    line += " aggregation=member_weighted";
    line += " members=" + std::to_string(result.size());
    line += " ";
    line += weightBuffer;
    line += " completed=" + std::to_string(count("completed"));
    line += " failed=" + std::to_string(count("failed"));
    line += " cancelled=" + std::to_string(count("cancelled"));
    line += " interrupted=" + std::to_string(count("interrupted"));
    line += " oom=" + std::to_string(count("oom"));

    const std::optional<double> trainLoss = result.weightedFinalTrainingLoss();
    if (trainLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "weighted_train_loss=%.6f", trainLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::trainLoss, useColor);
    }
    const std::optional<double> validateLoss = result.weightedFinalValidationLoss();
    if (validateLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "weighted_validate_loss=%.6f", validateLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::validateLoss, useColor);
    }
    const std::optional<double> testLoss = result.weightedFinalTestLoss();
    if (testLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "weighted_test_loss=%.6f", testLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::testLoss, useColor);
    }
    emitLineLocked(line);
}

bool TrainingRunsStatsReporter::shouldUseColorLocked() const {
    if (colorMode == LineStatsColorMode::NEVER) {
        return false;
    }
    if (colorMode == LineStatsColorMode::ALWAYS) {
        return true;
    }
    return LineStatsReporter::isAnsiColorSupported(output);
}

const char* TrainingRunsStatsReporter::statusColorStyle(TrainingRunStatus status) {
    switch (status) {
        case TrainingRunStatus::COMPLETED:
            return FinalReportAnsi::completed;
        case TrainingRunStatus::FAILED:
            return FinalReportAnsi::failed;
        case TrainingRunStatus::CANCELLED:
            return FinalReportAnsi::cancelled;
        case TrainingRunStatus::INTERRUPTED:
            return FinalReportAnsi::interrupted;
        case TrainingRunStatus::OUT_OF_MEMORY:
            return FinalReportAnsi::outOfMemory;
        case TrainingRunStatus::RUNNING:
            return FinalReportAnsi::running;
        case TrainingRunStatus::NOT_STARTED:
        default:
            return FinalReportAnsi::notStarted;
    }
}

std::string TrainingRunsStatsReporter::styled(std::string_view text, const char* style, bool useColor) {
    if (!useColor || style == nullptr || style[0] == '\0') {
        return std::string(text);
    }
    std::string out;
    out.reserve(std::char_traits<char>::length(style) + text.size() + std::char_traits<char>::length(FinalReportAnsi::reset));
    out += style;
    out += text;
    out += FinalReportAnsi::reset;
    return out;
}

size_t TrainingRunsStatsReporter::runPrefixWidthLocked() const {
    size_t width = 0;
    for (const RunState& state : runStates) {
        width = std::max(width, std::string("INFO runs[").size() + state.runName.size() + std::string("]:").size());
    }
    return width;
}

std::string TrainingRunsStatsReporter::formatRunPrefix(std::string_view runName, size_t runPrefixWidth) {
    std::string prefix = "INFO runs[" + std::string(runName) + "]:";
    if (prefix.size() < runPrefixWidth) {
        prefix.append(runPrefixWidth - prefix.size(), ' ');
    }
    return prefix;
}

std::string TrainingRunsStatsReporter::formatEnsemblePrefix(std::string_view ensembleGroup, size_t ensemblePrefixWidth) {
    std::string prefix = "INFO runs ensemble[" + std::string(ensembleGroup) + "]:";
    if (prefix.size() < ensemblePrefixWidth) {
        prefix.append(ensemblePrefixWidth - prefix.size(), ' ');
    }
    return prefix;
}

void TrainingRunsStatsReporter::emitLineLocked(const std::string& line) {
    if (output != nullptr) {
        std::fprintf(output, "%s\n", line.c_str());
    }
}

void TrainingRunsStatsReporter::flushOutputLocked() {
    if (output != nullptr) {
        std::fflush(output);
    }
}

bool TrainingRunsStatsReporter::anyDirtyLocked() const {
    if (dirty) {
        return true;
    }
    return std::any_of(runStates.begin(), runStates.end(), [](const RunState& state) { return state.dirty; });
}

void TrainingRunsStatsReporter::clearDirtyLocked() {
    dirty = false;
    for (RunState& state : runStates) {
        state.dirty = false;
    }
}

bool TrainingRunsStatsReporter::anyRunEnabledLocked() const {
    return !runStates.empty();
}

std::chrono::steady_clock::duration TrainingRunsStatsReporter::summaryInterval() const {
    if (maxSummaryLogsPerSecond <= 0.0) {
        return std::chrono::steady_clock::duration::zero();
    }
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / maxSummaryLogsPerSecond));
}

TrainingRunsStatsReporter::DisplayStatus TrainingRunsStatsReporter::displayStatusFromRunStatus(TrainingRunStatus status) {
    switch (status) {
        case TrainingRunStatus::NOT_STARTED:
            return DisplayStatus::NOT_STARTED;
        case TrainingRunStatus::RUNNING:
            return DisplayStatus::RUNNING;
        case TrainingRunStatus::COMPLETED:
            return DisplayStatus::COMPLETED;
        case TrainingRunStatus::FAILED:
            return DisplayStatus::FAILED;
        case TrainingRunStatus::CANCELLED:
            return DisplayStatus::CANCELLED;
        case TrainingRunStatus::INTERRUPTED:
            return DisplayStatus::INTERRUPTED;
        case TrainingRunStatus::OUT_OF_MEMORY:
            return DisplayStatus::OUT_OF_MEMORY;
        default:
            return DisplayStatus::FAILED;
    }
}

const char* TrainingRunsStatsReporter::displayStatusName(DisplayStatus status) {
    switch (status) {
        case DisplayStatus::NOT_STARTED:
            return "not_started";
        case DisplayStatus::STARTING:
            return "starting";
        case DisplayStatus::RUNNING:
            return "running";
        case DisplayStatus::COMPLETED:
            return "completed";
        case DisplayStatus::FAILED:
            return "failed";
        case DisplayStatus::CANCELLED:
            return "cancelled";
        case DisplayStatus::INTERRUPTED:
            return "interrupted";
        case DisplayStatus::OUT_OF_MEMORY:
            return "oom";
        default:
            return "unknown";
    }
}

}  // namespace Thor
