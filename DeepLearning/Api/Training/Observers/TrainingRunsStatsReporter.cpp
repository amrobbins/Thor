#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <stdexcept>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {
namespace {

std::FILE* requireTrainingRunsOutput(std::FILE* output) {
    if (output == nullptr) {
        throw std::runtime_error("TrainingRunsStatsReporter requires a non-null output file.");
    }
    return output;
}

const char* terminalStatusName(const TrainingRunResult& result) { return result.resultName(); }

enum class PadAlignment { LEFT, RIGHT };

std::string formatFixedString(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return std::string(buffer);
}

std::string formatScientificString(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*e", precision, value);
    return std::string(buffer);
}

std::string formatCompactRateString(double value, bool integral = false) {
    const double absValue = std::abs(value);
    const char* suffix = "";
    double scaled = value;
    if (absValue >= 1.0e24) {
        scaled = value / 1.0e24;
        suffix = "Y";
    } else if (absValue >= 1.0e21) {
        scaled = value / 1.0e21;
        suffix = "Z";
    } else if (absValue >= 1.0e18) {
        scaled = value / 1.0e18;
        suffix = "E";
    } else if (absValue >= 1.0e15) {
        scaled = value / 1.0e15;
        suffix = "P";
    } else if (absValue >= 1.0e12) {
        scaled = value / 1.0e12;
        suffix = "T";
    } else if (absValue >= 1.0e9) {
        scaled = value / 1.0e9;
        suffix = "G";
    } else if (absValue >= 1.0e6) {
        scaled = value / 1.0e6;
        suffix = "M";
    } else if (absValue >= 1.0e3) {
        scaled = value / 1.0e3;
        suffix = "K";
    }

    const double absScaled = std::abs(scaled);
    const int precision = integral ? 0 : (absScaled >= 100.0 ? 0 : (absScaled >= 10.0 ? 1 : 2));
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f%s", precision, scaled, suffix);
    return std::string(buffer);
}

std::string formatCompactFlopsRateString(double value) {
    static constexpr const char* suffixes[] = {"", "K", "M", "G", "T", "P", "E", "Z", "Y"};
    constexpr int maxSuffixIndex = static_cast<int>(sizeof(suffixes) / sizeof(suffixes[0])) - 1;

    double scaled = value;
    int suffixIndex = 0;
    while (std::abs(scaled) >= 1000.0 && suffixIndex < maxSuffixIndex) {
        scaled /= 1000.0;
        ++suffixIndex;
    }

    while (true) {
        const double absScaled = std::abs(scaled);
        const int precision = absScaled >= 100.0 ? 1 : (absScaled >= 10.0 ? 2 : 3);
        char number[64];
        std::snprintf(number, sizeof(number), "%.*f", precision, scaled);
        if (std::char_traits<char>::length(number) <= 5 || suffixIndex >= maxSuffixIndex) {
            return std::string(number) + suffixes[suffixIndex];
        }

        scaled /= 1000.0;
        ++suffixIndex;
    }
}

std::string formatUnsigned(uint64_t value) { return std::to_string(value); }

std::string formatRatio(uint64_t numerator, uint64_t denominator) { return std::to_string(numerator) + "/" + std::to_string(denominator); }

void appendPadded(std::string& line, const std::string& value, size_t width, PadAlignment alignment = PadAlignment::RIGHT) {
    if (value.size() < width && alignment == PadAlignment::RIGHT) {
        line.append(width - value.size(), ' ');
    }
    line += value;
    if (value.size() < width && alignment == PadAlignment::LEFT) {
        line.append(width - value.size(), ' ');
    }
}

constexpr size_t RATE_FIELD_WIDTH = 5;
constexpr size_t FLOPS_RATE_FIELD_WIDTH = 6;
constexpr size_t TRAIN_LOSS_FIELD_WIDTH = sizeof(" train_loss=0.000000") - 1;
constexpr size_t VALIDATE_LOSS_FIELD_WIDTH = sizeof(" validate_loss=0.000000") - 1;
constexpr size_t RUN_PROGRESS_FIELDS_WIDTH = sizeof(" epoch=     20/20 batch=        24/24 step=       480") - 1;

std::string formatRunLabel(std::string_view runName, const std::optional<std::string>& ensembleGroup) {
    std::string label(runName);
    if (ensembleGroup.has_value()) {
        label += "|";
        label += *ensembleGroup;
    }
    return label;
}

std::string formatRunPrefixForLabel(std::string_view runLabel, size_t runPrefixWidth) {
    std::string prefix = "INFO runs[" + std::string(runLabel) + "]:";
    if (prefix.size() < runPrefixWidth) {
        prefix.append(runPrefixWidth - prefix.size(), ' ');
    }
    return prefix;
}

// Multi-run progress colors intentionally mirror the single-trainer
// LineStatsReporter palette so TrainingRuns summaries do not look like a
// partially colorized downgrade of the normal trainer output.
namespace SummaryAnsi {
constexpr const char* reset = "\x1b[0m";
constexpr const char* bold = "\x1b[1m";
constexpr const char* key = "\x1b[38;5;235m";
constexpr const char* label = "\x1b[38;5;235m";
constexpr const char* progress = "\x1b[38;5;21m";
constexpr const char* loss = "\x1b[1;38;5;0m";
constexpr const char* accuracy = "\x1b[38;5;22m";
constexpr const char* learningRate = "\x1b[38;5;53m";
constexpr const char* throughput = "\x1b[1;38;2;140;84;0m";
constexpr const char* elapsed = "\x1b[38;5;0m";
}  // namespace SummaryAnsi

std::string styledText(std::string_view text, const char* style, bool useColor) {
    if (!useColor || style == nullptr || style[0] == '\0') {
        return std::string(text);
    }
    std::string out;
    out.reserve(std::char_traits<char>::length(style) + text.size() + std::char_traits<char>::length("\x1b[0m"));
    out += style;
    out += text;
    out += "\x1b[0m";
    return out;
}

std::string formatRunPrefixForLabel(std::string_view runLabel, size_t runPrefixWidth, bool useColor) {
    if (!useColor) {
        return formatRunPrefixForLabel(runLabel, runPrefixWidth);
    }

    const std::string info = "INFO ";
    const std::string suffix = "runs[" + std::string(runLabel) + "]:";
    const size_t visibleWidth = info.size() + suffix.size();

    std::string prefix = styledText(info, SummaryAnsi::label, true);
    prefix += styledText(suffix, SummaryAnsi::bold, true);
    if (visibleWidth < runPrefixWidth) {
        prefix.append(runPrefixWidth - visibleWidth, ' ');
    }
    return prefix;
}

void appendStyledPadded(std::string& line,
                        const std::string& value,
                        size_t width,
                        PadAlignment alignment,
                        const char* style,
                        bool useColor) {
    if (useColor && style != nullptr && style[0] != '\0') {
        line += style;
    }
    appendPadded(line, value, width, alignment);
    if (useColor && style != nullptr && style[0] != '\0') {
        line += SummaryAnsi::reset;
    }
}

void appendStyledPadded(std::string& line, const std::string& value, size_t width, const char* style, bool useColor) {
    appendStyledPadded(line, value, width, PadAlignment::RIGHT, style, useColor);
}

void appendSummaryDimKey(std::string& line, const char* key, bool useColor) {
    line += " ";
    if (useColor) {
        line += SummaryAnsi::key;
    }
    line += key;
    line += "=";
    if (useColor) {
        line += SummaryAnsi::reset;
    }
}

void appendPhaseLossColumns(std::string& line,
                            std::optional<double> trainLoss,
                            std::optional<double> validateLoss,
                            bool useColor,
                            const char* trainLossStyle,
                            const char* validateLossStyle);

bool shouldSuppressMetricColumnName(const std::string& name) {
    return name.empty() || name == "loss" || name == "learning_rate" || name == "learningRate" || name == "lr" ||
           name == "momentum" || name == "completed_epoch" || name == "best_epoch" || name == "best_score" ||
           name == "min_early_completion_epochs";
}

std::vector<std::pair<std::string, double>> orderedMetricColumns(
    const std::unordered_map<std::string, double>& metrics,
    const std::vector<std::string>& reportOrder) {
    std::vector<std::pair<std::string, double>> ordered;
    ordered.reserve(metrics.size());
    std::set<std::string> emitted;

    for (const std::string& name : reportOrder) {
        if (shouldSuppressMetricColumnName(name)) {
            continue;
        }
        const auto it = metrics.find(name);
        if (it == metrics.end()) {
            continue;
        }
        ordered.emplace_back(it->first, it->second);
        emitted.insert(it->first);
    }

    std::vector<std::pair<std::string, double>> remaining;
    remaining.reserve(metrics.size());
    for (const auto& [name, value] : metrics) {
        if (shouldSuppressMetricColumnName(name) || emitted.count(name) != 0) {
            continue;
        }
        remaining.emplace_back(name, value);
    }
    std::sort(remaining.begin(), remaining.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    ordered.insert(ordered.end(), remaining.begin(), remaining.end());
    return ordered;
}

std::string formatRunsStatsLineBase(const TrainingStatsSnapshot& stats,
                                    std::string_view runLabel,
                                    size_t runPrefixWidth,
                                    std::optional<double> trainLoss,
                                    std::optional<double> validateLoss,
                                    bool useColor,
                                    const std::vector<std::string>& reportOrder,
                                    const char* trainLossStyle,
                                    const char* validateLossStyle) {
    std::string line = formatRunPrefixForLabel(runLabel, runPrefixWidth, useColor);

    if (stats.epochs > 0) {
        appendSummaryDimKey(line, "epoch", useColor);
        appendStyledPadded(line, formatRatio(stats.epoch, stats.epochs), 10, SummaryAnsi::progress, useColor);
    } else if (stats.epoch > 0) {
        appendSummaryDimKey(line, "epoch", useColor);
        appendStyledPadded(line, formatUnsigned(stats.epoch), 10, SummaryAnsi::progress, useColor);
    }

    if (stats.stepsPerEpoch > 0) {
        appendSummaryDimKey(line, "batch", useColor);
        appendStyledPadded(line, formatRatio(stats.stepInEpoch, stats.stepsPerEpoch), 13, SummaryAnsi::progress, useColor);
    } else if (stats.stepInEpoch > 0) {
        appendSummaryDimKey(line, "batch", useColor);
        appendStyledPadded(line, formatUnsigned(stats.stepInEpoch), 13, SummaryAnsi::progress, useColor);
    }

    if (stats.step > 0) {
        appendSummaryDimKey(line, "step", useColor);
        appendStyledPadded(line, formatUnsigned(stats.step), 10, SummaryAnsi::progress, useColor);
    }

    appendPhaseLossColumns(line, trainLoss, validateLoss, useColor, trainLossStyle, validateLossStyle);

    if (stats.learningRate.has_value()) {
        appendSummaryDimKey(line, "lr", useColor);
        appendStyledPadded(line, formatScientificString(stats.learningRate.value(), 3), 9, SummaryAnsi::learningRate, useColor);
    }
    for (const auto& metric : orderedMetricColumns(stats.metrics, reportOrder)) {
        appendSummaryDimKey(line, metric.first.c_str(), useColor);
        appendStyledPadded(line, formatFixedString(metric.second, 6), 9, SummaryAnsi::loss, useColor);
    }

    if (stats.samplesPerSecond > 0.0) {
        appendSummaryDimKey(line, "samples/s", useColor);
        appendStyledPadded(line, formatCompactRateString(stats.samplesPerSecond), RATE_FIELD_WIDTH, SummaryAnsi::throughput, useColor);
    }
    if (stats.batchesPerSecond > 0.0) {
        appendSummaryDimKey(line, "batches/s", useColor);
        appendStyledPadded(line, formatCompactRateString(stats.batchesPerSecond), RATE_FIELD_WIDTH, SummaryAnsi::throughput, useColor);
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        appendSummaryDimKey(line, "flops/s", useColor);
        appendStyledPadded(line, formatCompactFlopsRateString(stats.floatingPointOperationsPerSecond), FLOPS_RATE_FIELD_WIDTH, SummaryAnsi::throughput, useColor);
    }
    if (stats.inFlightBatches > 0) {
        appendSummaryDimKey(line, "in_flight", useColor);
        appendStyledPadded(line, formatCompactRateString(stats.inFlightBatches, true), RATE_FIELD_WIDTH, SummaryAnsi::throughput, useColor);
    } else {
        line.append(16, ' ');
    }

    appendSummaryDimKey(line, "elapsed", useColor);
    appendStyledPadded(line, LineStatsReporter::formatElapsedSeconds(stats.elapsedSeconds), 9, SummaryAnsi::elapsed, useColor);
    return line;
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
constexpr const char* accuracy = "\x1b[38;5;22m";
}  // namespace FinalReportAnsi

void appendPhaseLossColumns(std::string& line,
                            std::optional<double> trainLoss,
                            std::optional<double> validateLoss,
                            bool useColor,
                            const char* trainLossStyle,
                            const char* validateLossStyle) {
    if (trainLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "train_loss=%.6f", trainLoss.value());
        line += " ";
        line += styledText(buffer, trainLossStyle, useColor);
    } else {
        line.append(TRAIN_LOSS_FIELD_WIDTH, ' ');
    }

    if (validateLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "validate_loss=%.6f", validateLoss.value());
        line += " ";
        line += styledText(buffer, validateLossStyle, useColor);
    } else {
        line.append(VALIDATE_LOSS_FIELD_WIDTH, ' ');
    }
}

void appendFinalPhaseMetricColumns(std::string& line,
                                   const std::optional<TrainingStatsSnapshot>& stats,
                                   const char* phasePrefix,
                                   bool useColor,
                                   const std::vector<std::string>& reportOrder) {
    if (!stats.has_value()) {
        return;
    }


    for (const auto& [name, value] : orderedMetricColumns(stats->metrics, reportOrder)) {
        char buffer[160];
        const bool looksLikeAccuracy = name.find("accuracy") != std::string::npos;
        std::snprintf(buffer, sizeof(buffer), "%s_%s=%.*f", phasePrefix, name.c_str(), looksLikeAccuracy ? 4 : 6, value);
        line += " ";
        line += styledText(buffer, looksLikeAccuracy ? FinalReportAnsi::accuracy : FinalReportAnsi::testLoss, useColor);
    }
}

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
        const std::string runLabel = formatRunLabel(result.runName, result.ensembleGroup);
        runPrefixWidth = std::max(runPrefixWidth, std::string("INFO runs[").size() + runLabel.size() + std::string("]:").size());
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
        const auto stateIt = runStateIndices.find(result.runName);
        const std::vector<std::string> emptyReportOrder;
        const std::vector<std::string>& reportOrder = stateIt == runStateIndices.end()
            ? emptyReportOrder
            : runStates[stateIt->second].config.reportOrder;
        writeResultLineLocked(result, runPrefixWidth, reportOrder);
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
            if (event.stats.type != TrainingEventType::STATS) {
                break;
            }
            state.latestStats = event.stats;
            switch (event.stats.stats.phase) {
                case TrainingEventPhase::TRAIN:
                    state.latestTrainingStats = event.stats;
                    updateSmoothedLossState(state.trainingLoss, event.stats.stats);
                    break;
                case TrainingEventPhase::VALIDATE:
                    state.latestValidationStats = event.stats;
                    updateSmoothedLossState(state.validationLoss, event.stats.stats);
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
        case ReporterEventType::RUN_FINISHED: {
            TrainingRunResult result = event.result;
            if (!result.ensembleGroup.has_value() && state.config.ensembleGroup.has_value()) {
                result.ensembleGroup = state.config.ensembleGroup;
                result.ensembleWeight = state.config.ensembleWeight;
            }
            state.terminalResult = std::move(result);
            state.status = displayStatusFromRunStatus(state.terminalResult->status);
            state.dirty = true;
            dirty = true;
            break;
        }
    }
}

void TrainingRunsStatsReporter::updateSmoothedLossState(PhaseLossState& lossState, const TrainingStatsSnapshot& stats) {
    if (!stats.loss.has_value()) {
        return;
    }

    if (lossState.currentEpoch != stats.epoch) {
        if (lossState.currentEpoch != 0 && lossState.currentEpochLossCount > 0) {
            lossState.previousEpochLoss = lossState.currentEpochLossSum / static_cast<double>(lossState.currentEpochLossCount);
        }
        lossState.currentEpoch = stats.epoch;
        lossState.currentEpochLossSum = 0.0;
        lossState.currentEpochLossCount = 0;
    }

    lossState.currentEpochLossSum += stats.loss.value();
    lossState.currentEpochLossCount += 1;

    const double currentEpochLoss = lossState.currentEpochLossSum / static_cast<double>(lossState.currentEpochLossCount);
    if (!lossState.previousEpochLoss.has_value()) {
        // In the first observed epoch for this phase there is no previous epoch to stabilize against,
        // so report the running average of the current epoch.
        lossState.displayedLoss = currentEpochLoss;
        return;
    }

    double progress = 1.0;
    if (stats.stepsPerEpoch > 0) {
        const uint64_t effectiveStepInEpoch = stats.stepInEpoch > 0 ? stats.stepInEpoch : lossState.currentEpochLossCount;
        progress = static_cast<double>(effectiveStepInEpoch) / static_cast<double>(stats.stepsPerEpoch);
        progress = std::clamp(progress, 0.0, 1.0);
    }

    lossState.displayedLoss = (lossState.previousEpochLoss.value() + currentEpochLoss * progress) / (1.0 + progress);
}

std::optional<double> TrainingRunsStatsReporter::displayedLossFromState(const PhaseLossState& lossState) {
    return lossState.displayedLoss;
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

    emitLineLocked("");

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
        // A multi-run progress row should describe training progress. Validation/test
        // stats can arrive after the latest train stat for an epoch and are still used
        // to update validate_loss/test_loss, but using them as the row base makes the
        // line appear to jump from e.g. batch=24/24 to batch=6/6 and shows
        // forward-only validation throughput as if it were training throughput.
        const TrainingStatsEvent& progressEvent = state.latestTrainingStats.has_value() ? state.latestTrainingStats.value()
                                                                                         : state.latestStats.value();
        std::string line = formatRunsStatsLineBase(progressEvent.stats,
                                                  formatRunLabel(state.runName, state.config.ensembleGroup),
                                                  runPrefixWidth,
                                                  displayedLossFromState(state.trainingLoss),
                                                  displayedLossFromState(state.validationLoss),
                                                  shouldUseColorLocked(),
                                                  state.config.reportOrder,
                                                  FinalReportAnsi::trainLoss,
                                                  FinalReportAnsi::validateLoss);
        emitLineLocked(line);
        return;
    }

    if (state.terminalResult.has_value()) {
        writeResultLineLocked(state.terminalResult.value(), runPrefixWidth, state.config.reportOrder);
        return;
    }

    std::string line = formatRunPrefix(formatRunLabel(state.runName, state.config.ensembleGroup), runPrefixWidth) + " status=" + displayStatusName(state.status);
    appendPhaseLossColumnsLocked(line, state);
    emitLineLocked(line);
}

void TrainingRunsStatsReporter::appendPhaseLossColumnsLocked(std::string& line, const RunState& state) {
    const bool useColor = shouldUseColorLocked();

    const std::optional<double> trainLoss = displayedLossFromState(state.trainingLoss);
    if (trainLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "train_loss=%.6f", trainLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::trainLoss, useColor);
    } else {
        line.append(TRAIN_LOSS_FIELD_WIDTH, ' ');
    }

    const std::optional<double> validateLoss = displayedLossFromState(state.validationLoss);
    if (validateLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "validate_loss=%.6f", validateLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::validateLoss, useColor);
    } else {
        line.append(VALIDATE_LOSS_FIELD_WIDTH, ' ');
    }

}

void TrainingRunsStatsReporter::writeResultLineLocked(const TrainingRunResult& result,
                                                      size_t runPrefixWidth,
                                                      const std::vector<std::string>& reportOrder) {
    const bool useColor = shouldUseColorLocked();
    const char* statusStyle = statusColorStyle(result.status);

    std::string line = formatRunPrefix(formatRunLabel(result.runName, result.ensembleGroup), runPrefixWidth);
    const std::string statusText = std::string("status=") + displayStatusName(displayStatusFromRunStatus(result.status));
    const std::string resultText = std::string("result=") + terminalStatusName(result);
    const size_t terminalStatusWidth = 1 + statusText.size() + 1 + resultText.size();
    line += " ";
    line += styled(statusText, statusStyle, useColor);
    line += " ";
    line += styled(resultText, statusStyle, useColor);
    if (terminalStatusWidth < RUN_PROGRESS_FIELDS_WIDTH) {
        line.append(RUN_PROGRESS_FIELDS_WIDTH - terminalStatusWidth, ' ');
    }

    appendPhaseLossColumns(line,
                           result.finalTrainingStats.has_value() ? result.finalTrainingStats->loss : std::optional<double>{},
                           result.finalValidationStats.has_value() ? result.finalValidationStats->loss : std::optional<double>{},
                           useColor,
                           FinalReportAnsi::trainLoss,
                           FinalReportAnsi::validateLoss);
    if (result.finalTestStats.has_value() && result.finalTestStats->loss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "test_loss=%.6f", result.finalTestStats->loss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::testLoss, useColor);
    }
    appendFinalPhaseMetricColumns(line, result.finalTrainingStats, "train", useColor, reportOrder);
    appendFinalPhaseMetricColumns(line, result.finalValidationStats, "validate", useColor, reportOrder);
    appendFinalPhaseMetricColumns(line, result.finalTestStats, "test", useColor, reportOrder);
    if (result.earlyCompleted()) {
        if (result.completedEpoch.has_value()) {
            line += " completed_epoch=" + std::to_string(result.completedEpoch.value());
        }
        if (result.bestEpoch.has_value()) {
            line += " best_epoch=" + std::to_string(result.bestEpoch.value());
        }
        if (result.bestScore.has_value()) {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "best_score=%.6f", result.bestScore.value());
            line += " ";
            line += styled(buffer, FinalReportAnsi::validateLoss, useColor);
        }
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

    std::string line = formatEnsemblePrefix(result.ensembleGroup, ensemblePrefixWidth);
    line += " ";
    line += styled(std::string("status=") + trainingRunStatusName(aggregateStatus), statusStyle, useColor);
    if (result.hasEnsembleEvaluationMetrics()) {
        line += " aggregation=ensemble_eval";
    }
    line += " members=" + std::to_string(result.size());
    if (!result.allCompleted()) {
        line += " completed=" + std::to_string(count("completed"));
        line += " failed=" + std::to_string(count("failed"));
        line += " cancelled=" + std::to_string(count("cancelled"));
        line += " interrupted=" + std::to_string(count("interrupted"));
        line += " oom=" + std::to_string(count("oom"));
    }

    if (result.ensembleTrainingLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "ensemble_train_loss=%.6f", result.ensembleTrainingLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::trainLoss, useColor);
    }
    if (result.ensembleTestLoss.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "ensemble_test_loss=%.6f", result.ensembleTestLoss.value());
        line += " ";
        line += styled(buffer, FinalReportAnsi::testLoss, useColor);
    }
    auto appendNamedLossMetric = [&](const TrainingNamedMetricResult& namedMetric) {
        // The aggregate fields above are the public overall graph-loss columns.
        // A single graph loss is commonly named "loss", which would format to the
        // exact same ensemble_train_loss / ensemble_test_loss keys here. Avoid
        // emitting duplicate columns while still reporting distinct named losses
        // such as daily_loss or aggregate_loss.
        if (namedMetric.name == "loss") {
            return;
        }
        if (namedMetric.trainValue.has_value()) {
            char buffer[128];
            std::snprintf(buffer, sizeof(buffer), "ensemble_train_%s=%.6f", namedMetric.name.c_str(), namedMetric.trainValue.value());
            line += " ";
            line += styled(buffer, FinalReportAnsi::trainLoss, useColor);
        }
        if (namedMetric.testValue.has_value()) {
            char buffer[128];
            std::snprintf(buffer, sizeof(buffer), "ensemble_test_%s=%.6f", namedMetric.name.c_str(), namedMetric.testValue.value());
            line += " ";
            line += styled(buffer, FinalReportAnsi::testLoss, useColor);
        }
    };

    auto appendNamedGraphMetric = [&](const TrainingNamedMetricResult& namedMetric) {
        const bool looksLikeAccuracy = namedMetric.name.find("accuracy") != std::string::npos;
        const char* style = looksLikeAccuracy ? FinalReportAnsi::accuracy : FinalReportAnsi::testLoss;
        if (namedMetric.trainValue.has_value()) {
            char buffer[128];
            std::snprintf(buffer, sizeof(buffer), "ensemble_train_%s=%.*f", namedMetric.name.c_str(), looksLikeAccuracy ? 4 : 6,
                          namedMetric.trainValue.value());
            line += " ";
            line += styled(buffer, style, useColor);
        }
        if (namedMetric.testValue.has_value()) {
            char buffer[128];
            std::snprintf(buffer, sizeof(buffer), "ensemble_test_%s=%.*f", namedMetric.name.c_str(), looksLikeAccuracy ? 4 : 6,
                          namedMetric.testValue.value());
            line += " ";
            line += styled(buffer, style, useColor);
        }
    };

    if (!result.reportOrder.empty()) {
        std::map<std::string, const TrainingNamedMetricResult*> lossByName;
        std::map<std::string, const TrainingNamedMetricResult*> graphMetricByName;
        for (const TrainingNamedMetricResult& namedMetric : result.namedMetrics) {
            lossByName.emplace(namedMetric.name, &namedMetric);
        }
        for (const TrainingNamedMetricResult& namedMetric : result.namedGraphMetrics) {
            graphMetricByName.emplace(namedMetric.name, &namedMetric);
        }
        std::set<std::string> emittedLosses;
        std::set<std::string> emittedGraphMetrics;
        for (const std::string& reportName : result.reportOrder) {
            const auto lossIt = lossByName.find(reportName);
            if (lossIt != lossByName.end()) {
                appendNamedLossMetric(*lossIt->second);
                emittedLosses.insert(reportName);
                continue;
            }
            const auto graphMetricIt = graphMetricByName.find(reportName);
            if (graphMetricIt != graphMetricByName.end()) {
                appendNamedGraphMetric(*graphMetricIt->second);
                emittedGraphMetrics.insert(reportName);
            }
        }
        for (const TrainingNamedMetricResult& namedMetric : result.namedMetrics) {
            if (emittedLosses.count(namedMetric.name) == 0) {
                appendNamedLossMetric(namedMetric);
            }
        }
        for (const TrainingNamedMetricResult& namedMetric : result.namedGraphMetrics) {
            if (emittedGraphMetrics.count(namedMetric.name) == 0) {
                appendNamedGraphMetric(namedMetric);
            }
        }
    } else {
        for (const TrainingNamedMetricResult& namedMetric : result.namedMetrics) {
            appendNamedLossMetric(namedMetric);
        }
        for (const TrainingNamedMetricResult& namedMetric : result.namedGraphMetrics) {
            appendNamedGraphMetric(namedMetric);
        }
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
        const std::string runLabel = formatRunLabel(state.runName, state.config.ensembleGroup);
        width = std::max(width, std::string("INFO runs[").size() + runLabel.size() + std::string("]:").size());
    }
    return width;
}

std::string TrainingRunsStatsReporter::formatRunPrefix(std::string_view runName, size_t runPrefixWidth) {
    return formatRunPrefixForLabel(runName, runPrefixWidth);
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
