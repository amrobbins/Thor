#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"
#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"

#include "gtest/gtest.h"

#include <cstdio>
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

namespace {

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

}  // namespace

TEST(TrainingRunsStatsReporter, PrefixesQueuedStatsLinesWithRunName) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});

    TrainingStatsEvent event = TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0");
    reporter.onStatsEvent(event);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("INFO runs[fold_0]: phase=train"), std::string::npos);
    EXPECT_EQ(output.find("INFO trainer:"), std::string::npos);
    EXPECT_NE(output.find("loss= 0.250000"), std::string::npos);
}

TEST(TrainingRunsStatsReporter, DrainsQueuedStatsForMultipleRunsBeforeClose) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});
    reporter.configureRun("fold_1", TrainingRunsStatsReporter::RunConfig{0.0, true});

    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_1"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("INFO runs[fold_0]: phase=train"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[fold_1]: phase=train"), std::string::npos);
}

TEST(TrainingRunsStatsReporter, RunningLineIncludesEnsembleGroupAndLatestPhaseLosses) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true, std::string("digits_dense_cv5"), 2.0});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats(TrainingEventPhase::TRAIN, 0.30)), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(makeStats(TrainingEventPhase::VALIDATE, 0.20)), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("INFO runs[fold_0]: phase=validate"), std::string::npos);
    EXPECT_NE(output.find("ensemble_group=digits_dense_cv5"), std::string::npos);
    EXPECT_NE(output.find("ensemble_weight=2.000000"), std::string::npos);
    EXPECT_NE(output.find("train_loss=0.300000"), std::string::npos);
    EXPECT_NE(output.find("validate_loss=0.200000"), std::string::npos);
}

TEST(TrainingRunsStatsReporter, EmitsWholeGroupSummaryIncludingRunsWithoutStats) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});
    reporter.configureRun("fold_1", TrainingRunsStatsReporter::RunConfig{0.0, true});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("INFO runs summary:"), std::string::npos);
    EXPECT_NE(output.find("total=2"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[fold_0]: phase=train"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[fold_1]: status=not_started"), std::string::npos);
}


TEST(TrainingRunsStatsReporter, FirstSummaryIncludesAllConfiguredRuns) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});
    reporter.configureRun("fold_1", TrainingRunsStatsReporter::RunConfig{0.0, true});
    reporter.configureRun("fold_2", TrainingRunsStatsReporter::RunConfig{0.0, true});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("INFO runs summary: total=3"), std::string::npos);
    EXPECT_EQ(output.find("INFO runs summary: total=1"), std::string::npos);
    EXPECT_EQ(output.find("INFO runs summary: total=2"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[fold_1]: status=not_started"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[fold_2]: status=not_started"), std::string::npos);
}


TEST(TrainingRunsStatsReporter, IgnoresNonStatsTrainingEventsWhenTrackingLatestRunningStats) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::runFinished(), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("INFO runs[fold_0]: phase=train"), std::string::npos);
    EXPECT_EQ(output.find("phase=unknown"), std::string::npos);
}

TEST(TrainingRunsStatsReporter, EmitsTerminalStatusInGroupSummary) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "fold_0"));
    reporter.markRunFinished(TrainingRunResult::completedResult("fold_0", makeStats(), makeStats()));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("completed=1"), std::string::npos);
    EXPECT_NE(output.find("status=completed"), std::string::npos);
    EXPECT_NE(output.find("final_train_loss=0.250000"), std::string::npos);
    EXPECT_NE(output.find("final_validate_loss=0.250000"), std::string::npos);
}


TEST(TrainingRunsStatsReporter, IncludesStatsDisabledRunsInLifecycleSummaries) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("stats_disabled", TrainingRunsStatsReporter::RunConfig{0.0, false});
    reporter.configureRun("stats_enabled", TrainingRunsStatsReporter::RunConfig{0.0, true});

    reporter.markRunStarting("stats_disabled");
    reporter.markRunStarting("stats_enabled");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "stats_disabled"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(makeStats()), "stats_enabled"));

    TrainingRunResult failedResult;
    failedResult.runName = "stats_disabled";
    failedResult.status = TrainingRunStatus::FAILED;
    failedResult.exception = TrainingRunExceptionSummary{"FakeError", "boom"};
    reporter.markRunFinished(failedResult);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("total=2"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[stats_disabled]: status=failed"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[stats_enabled]:  phase=train"), std::string::npos);
    EXPECT_EQ(output.find("INFO runs[stats_disabled]: phase=train"), std::string::npos);
}

TEST(TrainingRunsStatsReporter, EmitsFailedTerminalStatusInGroupSummary) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});

    TrainingRunResult result;
    result.runName = "fold_0";
    result.status = TrainingRunStatus::FAILED;
    result.exception = TrainingRunExceptionSummary{"FakeError", "boom"};
    reporter.markRunStarting("fold_0");
    reporter.markRunFinished(result);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("failed=1"), std::string::npos);
    EXPECT_NE(output.find("INFO runs[fold_0]: status=failed"), std::string::npos);
    EXPECT_NE(output.find("message=\"boom\""), std::string::npos);
}

TEST(TrainingRunsStatsReporter, SeparatesFinalReportAndIncludesTestLoss) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, true});

    TrainingRunResult result = TrainingRunResult::completedResult(
        "fold_0",
        makeStats(TrainingEventPhase::TRAIN, 0.50),
        makeStats(TrainingEventPhase::VALIDATE, 0.40),
        makeStats(TrainingEventPhase::TEST, 0.35));
    reporter.emitFinalReport(std::vector<TrainingRunResult>{result});
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("\nINFO runs final: ==================== final results"), std::string::npos);
    EXPECT_NE(output.find("INFO runs final: total=1"), std::string::npos);
    EXPECT_NE(output.find("final_train_loss=0.500000"), std::string::npos);
    EXPECT_NE(output.find("final_validate_loss=0.400000"), std::string::npos);
    EXPECT_NE(output.find("final_test_loss=0.350000"), std::string::npos);
    EXPECT_NE(output.find("INFO runs final: ====================================================="), std::string::npos);
}

TEST(TrainingRunsStatsReporter, EnsembleReportLabelsMemberWeightedLosses) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);

    TrainingEnsembleResult ensemble;
    ensemble.ensembleGroup = "digits_dense_cv5";
    TrainingEnsembleMemberResult member0;
    member0.runName = "fold_0";
    member0.status = TrainingRunStatus::COMPLETED;
    member0.finalTrainingLoss = 0.2;
    member0.finalValidationLoss = 0.4;
    TrainingEnsembleMemberResult member1 = member0;
    member1.runName = "fold_1";
    member1.finalTrainingLoss = 0.4;
    member1.finalValidationLoss = 0.8;
    ensemble.members = {member0, member1};

    reporter.emitEnsembleReport(std::vector<TrainingEnsembleResult>{ensemble});
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("aggregation=member_weighted"), std::string::npos);
    EXPECT_NE(output.find("weighted_train_loss=0.300000"), std::string::npos);
    EXPECT_NE(output.find("weighted_validate_loss=0.600000"), std::string::npos);
    EXPECT_EQ(output.find("final_train_loss=0.300000"), std::string::npos);
}

TEST(TrainingRunsStatsReporter, ColorCodesFinalReportStatusBorderAndLossColumns) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::ALWAYS, 0.0);
    reporter.configureRun("completed_fold", TrainingRunsStatsReporter::RunConfig{0.0, true});
    reporter.configureRun("failed_fold", TrainingRunsStatsReporter::RunConfig{0.0, true});

    TrainingRunResult completed = TrainingRunResult::completedResult(
        "completed_fold",
        makeStats(TrainingEventPhase::TRAIN, 0.50),
        makeStats(TrainingEventPhase::VALIDATE, 0.40),
        makeStats(TrainingEventPhase::TEST, 0.35));

    TrainingRunResult failed;
    failed.runName = "failed_fold";
    failed.status = TrainingRunStatus::FAILED;
    failed.exception = TrainingRunExceptionSummary{"FakeError", "boom"};

    reporter.emitFinalReport(std::vector<TrainingRunResult>{completed, failed});
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("\x1b[1;38;5;33mINFO runs final: ==================== final results"), std::string::npos);
    EXPECT_NE(output.find("\x1b[1;38;5;28mstatus=completed\x1b[0m"), std::string::npos);
    EXPECT_NE(output.find("\x1b[1;38;5;196mstatus=failed\x1b[0m"), std::string::npos);
    EXPECT_NE(output.find("\x1b[1;38;5;208mfinal_train_loss=0.500000\x1b[0m"), std::string::npos);
    EXPECT_NE(output.find("\x1b[1;38;5;33mfinal_validate_loss=0.400000\x1b[0m"), std::string::npos);
    EXPECT_NE(output.find("\x1b[1;38;5;129mfinal_test_loss=0.350000\x1b[0m"), std::string::npos);
}
