#pragma once

#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"
#include "DeepLearning/Api/Training/Results/TrainingRunResult.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Thor {

class TrainingRunsStatsReporter : public TrainingStatsSink {
   public:
    struct RunConfig {
        double intervalSeconds = 10.0;
        std::optional<std::string> ensembleGroup{};
        double ensembleWeight = 1.0;
    };

    explicit TrainingRunsStatsReporter(std::FILE* output = stdout,
                                       LineStatsColorMode colorMode = LineStatsColorMode::AUTO,
                                       double maxSummaryLogsPerSecond = 2.0);
    ~TrainingRunsStatsReporter() override;

    TrainingRunsStatsReporter(const TrainingRunsStatsReporter&) = delete;
    TrainingRunsStatsReporter& operator=(const TrainingRunsStatsReporter&) = delete;

    void configureRun(std::string runName, RunConfig config);
    void markRunStarting(const std::string& runName);
    void markRunFinished(const TrainingRunResult& result);
    void emitFinalReport(const std::vector<TrainingRunResult>& results);
    void emitEnsembleReport(const std::vector<TrainingEnsembleResult>& ensembles);
    void onStatsEvent(const TrainingStatsEvent& event) override;
    void flush() override;
    void close() override;

   private:
    enum class DisplayStatus { NOT_STARTED, STARTING, RUNNING, COMPLETED, FAILED, CANCELLED, INTERRUPTED, OUT_OF_MEMORY };
    enum class ReporterEventType { RUN_STARTING, RUN_STATS, RUN_FINISHED };

    struct ReporterEvent {
        ReporterEventType type = ReporterEventType::RUN_STATS;
        std::string runName{};
        TrainingStatsEvent stats{};
        TrainingRunResult result{};
    };

    struct PhaseLossState {
        uint64_t currentEpoch = 0;
        double currentEpochLossSum = 0.0;
        uint64_t currentEpochLossCount = 0;
        std::optional<double> previousEpochLoss{};
        std::optional<double> displayedLoss{};
    };

    struct RunState {
        std::string runName{};
        RunConfig config{};
        DisplayStatus status = DisplayStatus::NOT_STARTED;
        std::optional<TrainingStatsEvent> latestStats{};
        std::optional<TrainingStatsEvent> latestTrainingStats{};
        std::optional<TrainingStatsEvent> latestValidationStats{};
        std::optional<TrainingStatsEvent> latestTestStats{};
        PhaseLossState trainingLoss{};
        PhaseLossState validationLoss{};
        std::optional<TrainingRunResult> terminalResult{};
        bool dirty = false;
    };

    void enqueueEvent(ReporterEvent event);
    void workerLoop() noexcept;
    void processEvent(const ReporterEvent& event);
    static void updateSmoothedLossState(PhaseLossState& lossState, const TrainingStatsSnapshot& stats);
    static std::optional<double> displayedLossFromState(const PhaseLossState& lossState);
    RunState& stateForRun(const std::string& runName);
    void maybeEmitSummary(std::chrono::steady_clock::time_point now, bool force = false);
    void emitSummaryLocked(std::chrono::steady_clock::time_point now);
    void writeSummaryHeaderLocked(std::string_view label);
    void writeRunLineLocked(const RunState& state, size_t runPrefixWidth);
    void appendPhaseLossColumnsLocked(std::string& line, const RunState& state);
    void writeResultLineLocked(const TrainingRunResult& result, size_t runPrefixWidth);
    void writeEnsembleLineLocked(const TrainingEnsembleResult& result, size_t ensemblePrefixWidth);
    [[nodiscard]] bool shouldUseColorLocked() const;
    [[nodiscard]] static const char* statusColorStyle(TrainingRunStatus status);
    [[nodiscard]] static std::string styled(std::string_view text, const char* style, bool useColor);
    [[nodiscard]] size_t runPrefixWidthLocked() const;
    [[nodiscard]] static std::string formatRunPrefix(std::string_view runName, size_t runPrefixWidth);
    [[nodiscard]] static std::string formatEnsemblePrefix(std::string_view ensembleGroup, size_t ensemblePrefixWidth);
    void emitLineLocked(const std::string& line);
    void flushOutputLocked();
    [[nodiscard]] bool anyDirtyLocked() const;
    void clearDirtyLocked();
    [[nodiscard]] bool anyRunEnabledLocked() const;
    [[nodiscard]] std::chrono::steady_clock::duration summaryInterval() const;
    [[nodiscard]] static DisplayStatus displayStatusFromRunStatus(TrainingRunStatus status);
    [[nodiscard]] static const char* displayStatusName(DisplayStatus status);

    std::FILE* output = nullptr;
    LineStatsColorMode colorMode = LineStatsColorMode::AUTO;
    double maxSummaryLogsPerSecond = 2.0;

    std::mutex mutex;
    std::condition_variable workAvailable;
    std::condition_variable drained;
    std::deque<ReporterEvent> queue;
    bool closeRequested = false;
    bool processingEvent = false;
    std::thread worker;

    std::vector<RunState> runStates;
    std::unordered_map<std::string, size_t> runStateIndices;
    bool dirty = false;
    bool emittedAnySummary = false;
    std::chrono::steady_clock::time_point lastSummaryTime{};
};

}  // namespace Thor
