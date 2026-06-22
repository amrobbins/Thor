#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace Thor;

namespace {

TrainingStatsSnapshot makeStats(TrainingEventPhase phase = TrainingEventPhase::TRAIN, double loss = 0.25) {
    TrainingStatsSnapshot stats;
    stats.networkName = "net";
    stats.datasetName = "data";
    stats.phase = phase;
    stats.epoch = 2;
    stats.epochs = 5;
    stats.step = 17;
    stats.stepInEpoch = 7;
    stats.stepsPerEpoch = 100;
    stats.loss = loss;
    stats.samplesPerSecond = 1024.0;
    stats.batchesPerSecond = 8.0;
    stats.inFlightBatches = 4;
    stats.elapsedSeconds = 12.0;
    return stats;
}

class CapturingStatsSink : public TrainingStatsSink {
   public:
    void onStatsEvent(const TrainingStatsEvent& event) override { events.push_back(event); }
    void flush() override { flushed = true; }
    void close() override { closed = true; }

    std::vector<TrainingStatsEvent> events{};
    bool flushed = false;
    bool closed = false;
};

std::string readAndCloseFile(std::FILE* file) {
    if (file == nullptr) {
        throw std::runtime_error("failed to create temporary output file");
    }
    std::fflush(file);
    std::rewind(file);

    std::string output;
    char buffer[4096];
    while (true) {
        const std::size_t bytesRead = std::fread(buffer, 1, sizeof(buffer), file);
        output.append(buffer, bytesRead);
        if (bytesRead < sizeof(buffer)) {
            break;
        }
    }

    std::fclose(file);
    return output;
}

std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t newline = text.find('\n', start);
        if (newline == std::string::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        lines.push_back(text.substr(start, newline - start));
        start = newline + 1;
    }
    return lines;
}

bool containsAll(const std::string& line, std::initializer_list<const char*> tokens) {
    for (const char* token : tokens) {
        if (line.find(token) == std::string::npos) {
            return false;
        }
    }
    return true;
}

std::string findLineWithAll(const std::string& output, std::initializer_list<const char*> tokens) {
    for (const std::string& line : splitLines(output)) {
        if (containsAll(line, tokens)) {
            return line;
        }
    }
    return {};
}

bool hasLineWithAll(const std::string& output, std::initializer_list<const char*> tokens) {
    return !findLineWithAll(output, tokens).empty();
}

bool hasTokenWithValue(const std::string& line, const std::string& key, const std::string& expectedValue) {
    const std::string prefix = key + "=";
    const size_t keyPos = line.find(prefix);
    if (keyPos == std::string::npos) {
        return false;
    }

    size_t valuePos = keyPos + prefix.size();
    while (valuePos < line.size() && line[valuePos] == ' ') {
        ++valuePos;
    }

    return line.compare(valuePos, expectedValue.size(), expectedValue) == 0;
}

}  // namespace

TEST(TrainingStatsEvent, PreservesTrainingEventPayloadAndRunName) {
    TrainingEvent event = TrainingEvent::statsUpdated(makeStats(), "updated");

    TrainingStatsEvent statsEvent = TrainingStatsEvent::fromTrainingEvent(std::move(event), "fold_0");

    EXPECT_EQ(statsEvent.runName, "fold_0");
    EXPECT_EQ(statsEvent.type, TrainingEventType::STATS);
    EXPECT_EQ(statsEvent.message, "updated");
    EXPECT_EQ(statsEvent.stats.networkName, "net");
    EXPECT_EQ(statsEvent.stats.datasetName, "data");
    EXPECT_EQ(statsEvent.stats.phase, TrainingEventPhase::TRAIN);
    EXPECT_EQ(statsEvent.stats.epoch, 2u);
    ASSERT_TRUE(statsEvent.stats.loss.has_value());
    EXPECT_EQ(statsEvent.stats.loss.value(), 0.25);
}

TEST(TrainingStatsSinkObserver, ForwardsTrainingEventsAsStructuredStatsEvents) {
    auto sink = std::make_shared<CapturingStatsSink>();
    TrainingStatsSinkObserver observer(sink, "fold_1");

    observer.onTrainingEvent(TrainingEvent::epochStarted(makeStats(), "begin"));
    observer.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(), "stats"));
    observer.flush();
    observer.close();

    ASSERT_EQ(sink->events.size(), 2u);
    EXPECT_EQ(sink->events[0].runName, "fold_1");
    EXPECT_EQ(sink->events[0].type, TrainingEventType::EPOCH_STARTED);
    EXPECT_EQ(sink->events[0].message, "begin");
    EXPECT_EQ(sink->events[0].stats.step, 17u);
    EXPECT_EQ(sink->events[1].runName, "fold_1");
    EXPECT_EQ(sink->events[1].type, TrainingEventType::STATS);
    EXPECT_EQ(sink->events[1].message, "stats");
    EXPECT_TRUE(sink->flushed);
    EXPECT_TRUE(sink->closed);
}

TEST(TrainingStatsSinkObserver, IgnoresNullSink) {
    TrainingStatsSinkObserver observer(nullptr, "fold_2");

    EXPECT_NO_THROW(observer.onTrainingEvent(TrainingEvent::statsUpdated(makeStats())));
    EXPECT_NO_THROW(observer.flush());
    EXPECT_NO_THROW(observer.close());
}

TEST(TrainingRunsStatsReporter, EmitsConfiguredRunSummaryWithoutDependingOnColumnWidths) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0});
    reporter.configureRun("fold_1", TrainingRunsStatsReporter::RunConfig{0.0});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs summary:", "total=2", "running=1", "not_started=1"})) << output;
    const std::string runningLine = findLineWithAll(output, {"INFO runs[fold_0]:", "epoch=", "batch=", "step=", "train_loss="});
    ASSERT_FALSE(runningLine.empty()) << output;
    EXPECT_TRUE(hasTokenWithValue(runningLine, "epoch", "2/5")) << runningLine;
    EXPECT_TRUE(hasTokenWithValue(runningLine, "batch", "7/100")) << runningLine;
    EXPECT_TRUE(hasTokenWithValue(runningLine, "step", "17")) << runningLine;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[fold_1]:", "status=not_started"})) << output;
}

TEST(TrainingRunsStatsReporter, ValidationStatsUpdateValidationLossWithoutReplacingTrainingProgress) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, std::string("digits_dense_cv5"), 1.0});

    TrainingStatsSnapshot trainStats = makeStats(TrainingEventPhase::TRAIN, 0.30);
    trainStats.epoch = 20;
    trainStats.epochs = 20;
    trainStats.step = 480;
    trainStats.stepInEpoch = 24;
    trainStats.stepsPerEpoch = 24;
    trainStats.samplesPerSecond = 300000.0;
    trainStats.batchesPerSecond = 147.0;
    trainStats.floatingPointOperationsPerSecond = 91.32e12;

    TrainingStatsSnapshot validateStats = makeStats(TrainingEventPhase::VALIDATE, 0.20);
    validateStats.epoch = 20;
    validateStats.epochs = 20;
    validateStats.step = 115;
    validateStats.stepInEpoch = 1;
    validateStats.stepsPerEpoch = 6;
    validateStats.samplesPerSecond = 1710000.0;
    validateStats.batchesPerSecond = 834.0;
    validateStats.floatingPointOperationsPerSecond = 105.8e12;

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(trainStats), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(validateStats), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string line = findLineWithAll(output, {"INFO runs[fold_0|digits_dense_cv5]:", "train_loss=", "validate_loss="});
    ASSERT_FALSE(line.empty()) << output;
    EXPECT_TRUE(hasTokenWithValue(line, "batch", "24/24")) << line;
    EXPECT_TRUE(hasTokenWithValue(line, "step", "480")) << line;
    EXPECT_FALSE(hasTokenWithValue(line, "batch", "1/6")) << line;
    EXPECT_FALSE(hasTokenWithValue(line, "step", "115")) << line;
}

TEST(TrainingRunsStatsReporter, IgnoresNonStatsTrainingEventsWhenTrackingLatestRunningStats) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::runFinished(makeStats()), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[fold_0]:", "epoch=", "batch=", "step=", "train_loss="})) << output;
    EXPECT_FALSE(hasLineWithAll(output, {"INFO runs[fold_0]:", "status=completed"})) << output;
}

TEST(TrainingRunsStatsReporter, TerminalRunResultsReportStatusAndPhaseLosses) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("completed_fold", TrainingRunsStatsReporter::RunConfig{0.0, std::string("digits_dense_cv5"), 2.0});
    reporter.configureRun("failed_fold", TrainingRunsStatsReporter::RunConfig{0.0});

    TrainingRunResult completed = TrainingRunResult::completedResult(
        "completed_fold", makeStats(TrainingEventPhase::TRAIN, 0.50), makeStats(TrainingEventPhase::VALIDATE, 0.40));
    reporter.markRunStarting("completed_fold");
    reporter.markRunFinished(completed);

    TrainingRunResult failed;
    failed.runName = "failed_fold";
    failed.status = TrainingRunStatus::FAILED;
    failed.exception = TrainingRunExceptionSummary{"FakeError", "boom"};
    reporter.markRunStarting("failed_fold");
    reporter.markRunFinished(failed);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs summary:", "completed=1", "failed=1"})) << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[completed_fold|digits_dense_cv5]:", "status=completed", "result=completed", "train_loss=", "validate_loss="}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[failed_fold]:", "status=failed", "result=failed", "message=\"boom\""})) << output;
}


TEST(TrainingRunsStatsReporter, EarlyCompletedRunResultReportsCompletionMetadata) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("early_fold", TrainingRunsStatsReporter::RunConfig{0.0});

    TrainingRunResult result = TrainingRunResult::completedResult(
        "early_fold",
        makeStats(TrainingEventPhase::TRAIN, 0.50),
        makeStats(TrainingEventPhase::VALIDATE, 0.40),
        {},
        TrainingRunCompletionReason::EARLY_COMPLETED,
        2,
        1,
        0.125);

    reporter.markRunStarting("early_fold");
    reporter.markRunFinished(result);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[early_fold]:",
                                "status=completed",
                                "result=early_completed",
                                "completed_epoch=2",
                                "best_epoch=1",
                                "best_score=0.125000"}))
        << output;
}

TEST(TrainingRunsStatsReporter, FinalReportIncludesStatusCountsAndAvailablePhaseMetrics) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);

    TrainingStatsSnapshot testStats = makeStats(TrainingEventPhase::TEST, 0.35);
    testStats.accuracy = 0.875;
    TrainingRunResult completed = TrainingRunResult::completedResult(
        "completed_fold", makeStats(TrainingEventPhase::TRAIN, 0.50), makeStats(TrainingEventPhase::VALIDATE, 0.40), testStats);
    completed.ensembleGroup = "digits_dense_cv5";

    TrainingRunResult failed;
    failed.runName = "failed_fold";
    failed.status = TrainingRunStatus::FAILED;
    failed.exception = TrainingRunExceptionSummary{"FakeError", "boom"};

    reporter.emitFinalReport(std::vector<TrainingRunResult>{completed, failed});
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs final:", "total=2", "completed=1", "failed=1"})) << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[completed_fold|digits_dense_cv5]:",
                                "status=completed",
                                "train_loss=",
                                "validate_loss=",
                                "test_loss=",
                                "test_accuracy="}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[failed_fold]:", "status=failed", "message=\"boom\""})) << output;
}

TEST(TrainingRunsStatsReporter, EnsembleReportShowsEvaluationMetricsAndIncompleteStatusCounts) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);

    TrainingEnsembleResult completedEnsemble;
    completedEnsemble.ensembleGroup = "digits_dense_cv5";
    TrainingEnsembleMemberResult member0;
    member0.runName = "fold_0";
    member0.status = TrainingRunStatus::COMPLETED;
    TrainingEnsembleMemberResult member1 = member0;
    member1.runName = "fold_1";
    completedEnsemble.members = {member0, member1};
    completedEnsemble.ensembleTrainingLoss = 0.123;
    completedEnsemble.ensembleTestLoss = 0.456;

    TrainingEnsembleResult incompleteEnsemble;
    incompleteEnsemble.ensembleGroup = "mixed_group";
    TrainingEnsembleMemberResult completedMember;
    completedMember.runName = "fold_2";
    completedMember.status = TrainingRunStatus::COMPLETED;
    TrainingEnsembleMemberResult failedMember;
    failedMember.runName = "fold_3";
    failedMember.status = TrainingRunStatus::FAILED;
    incompleteEnsemble.members = {completedMember, failedMember};

    reporter.emitEnsembleReport(std::vector<TrainingEnsembleResult>{completedEnsemble, incompleteEnsemble});
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string completedLine = findLineWithAll(
        output, {"INFO runs ensemble[digits_dense_cv5]:", "status=completed", "aggregation=ensemble_eval", "members=2", "ensemble_train_loss=", "ensemble_test_loss="});
    ASSERT_FALSE(completedLine.empty()) << output;
    EXPECT_EQ(completedLine.find(" completed="), std::string::npos) << completedLine;
    EXPECT_EQ(completedLine.find(" failed="), std::string::npos) << completedLine;

    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs ensemble[mixed_group]:",
                                "status=failed",
                                "members=2",
                                "completed=1",
                                "failed=1",
                                "cancelled=0",
                                "interrupted=0",
                                "oom=0"}))
        << output;
}

