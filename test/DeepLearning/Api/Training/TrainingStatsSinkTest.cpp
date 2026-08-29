#include "DeepLearning/Api/Training/Observers/TrainingRunsStatsReporter.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <cstring>
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
    stats.batchSize = 4;
    stats.validExamplesInBatch = 4;
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

bool tokensAppearInOrder(const std::string& line, std::initializer_list<const char*> tokens) {
    size_t searchFrom = 0;
    for (const char* token : tokens) {
        const size_t pos = line.find(token, searchFrom);
        if (pos == std::string::npos) {
            return false;
        }
        searchFrom = pos + std::strlen(token);
    }
    return true;
}

bool hasTokenWithValue(const std::string& line, const std::string& key, const std::string& expectedValue) {
    const std::string prefix = " " + key + "=";
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


TEST(TrainingStatsSinkObserver, CanSuppressSharedSinkFlushForConcurrentRunObservers) {
    auto sink = std::make_shared<CapturingStatsSink>();
    TrainingStatsSinkObserver observer(sink, "fold_0", /*flushSinkOnFlush=*/false);

    observer.onTrainingEvent(TrainingEvent::statsUpdated(makeStats()));
    observer.flush();

    ASSERT_EQ(sink->events.size(), 1u);
    EXPECT_FALSE(sink->flushed);

    // The owner of the shared sink still performs the aggregate flush once all
    // concurrent producers have stopped.
    sink->flush();
    EXPECT_TRUE(sink->flushed);
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
    EXPECT_EQ(runningLine.find(" score="), std::string::npos) << runningLine;
    EXPECT_EQ(runningLine.find(" best_epoch="), std::string::npos) << runningLine;
    EXPECT_EQ(runningLine.find(" best_score="), std::string::npos) << runningLine;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[fold_1]:", "status=not_started"})) << output;
}

TEST(TrainingRunsStatsReporter, RunningSummaryShowsLatestModelSelectionScoreFromPreviousEpoch) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0});

    TrainingStatsSnapshot epoch1Stats = makeStats(TrainingEventPhase::TRAIN, 0.30);
    epoch1Stats.epoch = 1;
    epoch1Stats.stepInEpoch = 100;
    epoch1Stats.stepsPerEpoch = 100;

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(epoch1Stats), "fold_0"));
    reporter.flush();

    TrainingStatsSnapshot epoch2Started = makeStats(TrainingEventPhase::TRAIN, 0.0);
    epoch2Started.epoch = 2;
    epoch2Started.step = 0;
    epoch2Started.stepInEpoch = 0;
    epoch2Started.metrics["latest_score"] = -5.31;
    epoch2Started.metrics["best_epoch"] = 1.0;
    epoch2Started.metrics["best_score"] = -5.31;
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::epochStarted(epoch2Started), "fold_0"));

    TrainingStatsSnapshot epoch2Stats = makeStats(TrainingEventPhase::TRAIN, 0.25);
    epoch2Stats.epoch = 2;
    epoch2Stats.stepInEpoch = 7;
    epoch2Stats.stepsPerEpoch = 100;
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(epoch2Stats), "fold_0"));
    reporter.flush();

    TrainingStatsSnapshot epoch3Started = epoch2Started;
    epoch3Started.epoch = 3;
    epoch3Started.metrics["latest_score"] = -5.20;
    epoch3Started.metrics["best_epoch"] = 1.0;
    epoch3Started.metrics["best_score"] = -5.31;
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::epochStarted(epoch3Started), "fold_0"));

    TrainingStatsSnapshot epoch3Stats = makeStats(TrainingEventPhase::TRAIN, 0.20);
    epoch3Stats.epoch = 3;
    epoch3Stats.stepInEpoch = 8;
    epoch3Stats.stepsPerEpoch = 100;
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(epoch3Stats), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    std::string epoch2Line;
    std::string epoch3Line;
    for (const std::string& line : splitLines(output)) {
        if (line.find("INFO runs[fold_0]:") == std::string::npos) {
            continue;
        }
        if (hasTokenWithValue(line, "epoch", "2/5") &&
            hasTokenWithValue(line, "score", "-5.310000")) {
            epoch2Line = line;
        }
        if (hasTokenWithValue(line, "epoch", "3/5") &&
            hasTokenWithValue(line, "score", "-5.200000")) {
            epoch3Line = line;
        }
    }

    ASSERT_FALSE(epoch2Line.empty()) << output;
    EXPECT_TRUE(hasTokenWithValue(epoch2Line, "best_epoch", "1")) << epoch2Line;
    EXPECT_TRUE(hasTokenWithValue(epoch2Line, "best_score", "-5.310000")) << epoch2Line;
    EXPECT_TRUE(tokensAppearInOrder(
        epoch2Line,
        {"epoch=", "score=", "best_epoch=", "best_score=", "batch="}))
        << epoch2Line;

    ASSERT_FALSE(epoch3Line.empty()) << output;
    EXPECT_TRUE(hasTokenWithValue(epoch3Line, "best_epoch", "1")) << epoch3Line;
    EXPECT_TRUE(hasTokenWithValue(epoch3Line, "best_score", "-5.310000")) << epoch3Line;
}

TEST(TrainingRunsStatsReporter, ReportsStartupWaitStates) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0});

    reporter.markRunStarting("fold_0");
    reporter.flush();
    reporter.markRunStatus("fold_0", TrainingRunStatus::WAITING_TO_START);
    reporter.flush();
    reporter.markRunStatus("fold_0", TrainingRunStatus::STARTING);
    reporter.flush();
    reporter.markRunStatus("fold_0", TrainingRunStatus::WAITING_FOR_MEMORY);
    reporter.flush();
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[fold_0]:", "status=starting"})) << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[fold_0]:", "status=waiting_to_start"})) << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[fold_0]:", "status=waiting_for_memory"})) << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs summary:", "waiting_to_start=1"})) << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs summary:", "waiting_for_memory=1"})) << output;
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

TEST(TrainingRunsStatsReporter, ReportsAdditionalNamedValidationPopulationsWithoutDuplicatingDefault) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun(
        "fold_0",
        TrainingRunsStatsReporter::RunConfig{
            0.0,
            std::string("sku_demand_cv5"),
            1.0,
            {"daily_central_loss"}});

    TrainingStatsSnapshot trainStats = makeStats(TrainingEventPhase::TRAIN, 3.0);
    trainStats.metrics["daily_central_loss"] = 30.0;

    TrainingStatsSnapshot unseenStats = makeStats(TrainingEventPhase::VALIDATE, 5.0);
    unseenStats.validationPopulation = "unseen_sku";
    unseenStats.isDefaultValidationPopulation = true;
    unseenStats.metrics["daily_central_loss"] = 50.0;

    TrainingStatsSnapshot seenStats = makeStats(TrainingEventPhase::VALIDATE, 2.0);
    seenStats.validationPopulation = "seen_sku";
    seenStats.metrics["daily_central_loss"] = 20.0;

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(trainStats), "fold_0"));
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(unseenStats), "fold_0"));
    reporter.onStatsEvent(
        TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(seenStats), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string line = findLineWithAll(
        output,
        {"INFO runs[fold_0|sku_demand_cv5]:",
         "validate_loss=",
         "validate_seen_sku_loss=",
         "validate_seen_sku_daily_central_loss="});
    ASSERT_FALSE(line.empty()) << output;
    EXPECT_EQ(line.find("validate_unseen_sku_loss="), std::string::npos) << line;
}

TEST(TrainingRunsStatsReporter, TerminalEventDoesNotCoalesceAwayPendingValidationSummary) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 1.0);
    reporter.configureRun("fold_0", TrainingRunsStatsReporter::RunConfig{0.0, std::string("tiny_ensemble"), 1.0});

    TrainingStatsSnapshot trainStats = makeStats(TrainingEventPhase::TRAIN, 2.0);
    trainStats.epoch = 1;
    trainStats.epochs = 1;
    trainStats.step = 1;
    trainStats.stepInEpoch = 1;
    trainStats.stepsPerEpoch = 1;
    trainStats.inFlightBatches = 1;

    TrainingStatsSnapshot validateStats = makeStats(TrainingEventPhase::VALIDATE, 1.9208);
    validateStats.epoch = 1;
    validateStats.epochs = 1;
    validateStats.step = 1;
    validateStats.stepInEpoch = 1;
    validateStats.stepsPerEpoch = 1;
    validateStats.inFlightBatches = 0;

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(trainStats), "fold_0"));
    reporter.flush();

    // These events intentionally arrive inside the reporter's one-second rate-limit
    // window. The validation snapshot must still be emitted as a running row before
    // RUN_FINISHED changes the row to terminal status.
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(validateStats), "fold_0"));
    TrainingRunResult result = TrainingRunResult::completedResult("fold_0", trainStats, validateStats);
    reporter.markRunFinished(result);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string runningValidationLine =
        findLineWithAll(output, {"INFO runs[fold_0|tiny_ensemble]:", "epoch=", "train_loss=", "validate_loss="});
    ASSERT_FALSE(runningValidationLine.empty()) << output;
    EXPECT_TRUE(hasTokenWithValue(runningValidationLine, "epoch", "1/1")) << runningValidationLine;
    EXPECT_TRUE(hasTokenWithValue(runningValidationLine, "batch", "1/1")) << runningValidationLine;
    EXPECT_TRUE(hasTokenWithValue(runningValidationLine, "step", "1")) << runningValidationLine;
    EXPECT_EQ(runningValidationLine.find("status=completed"), std::string::npos) << runningValidationLine;
}

TEST(TrainingRunsStatsReporter, RunningSummaryReportsTrainAndValidateMetricsInConfiguredPairs) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("fold_0",
                          TrainingRunsStatsReporter::RunConfig{0.0,
                                                               std::string("sku_demand_cv5"),
                                                               1.0,
                                                               {"daily_central_loss",
                                                                "daily_true",
                                                                "daily_pred",
                                                                "daily_lower",
                                                                "daily_upper"}});

    TrainingStatsSnapshot trainStats = makeStats(TrainingEventPhase::TRAIN, 0.50);
    trainStats.metrics["daily_central_loss"] = 12.50;
    trainStats.metrics["daily_true"] = 5.57;
    trainStats.metrics["daily_pred"] = 4.96;
    trainStats.metrics["daily_lower"] = 3.20;
    trainStats.metrics["daily_upper"] = 7.60;
    trainStats.metrics["daily_quantile_low_loss"] = 99.0;

    TrainingStatsSnapshot validateStats = makeStats(TrainingEventPhase::VALIDATE, 0.40);
    validateStats.metrics["daily_central_loss"] = 13.75;
    validateStats.metrics["daily_true"] = 5.80;
    validateStats.metrics["daily_pred"] = 5.10;
    validateStats.metrics["daily_lower"] = 3.40;
    validateStats.metrics["daily_upper"] = 7.90;
    validateStats.metrics["daily_quantile_low_loss"] = 101.0;

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(trainStats), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(TrainingEvent::statsUpdated(validateStats), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string line = findLineWithAll(output,
                                             {"INFO runs[fold_0|sku_demand_cv5]:",
                                              "train_daily_central_loss=",
                                              "validate_daily_upper="});
    ASSERT_FALSE(line.empty()) << output;
    EXPECT_TRUE(tokensAppearInOrder(line,
                                    {"train_daily_central_loss=",
                                     "validate_daily_central_loss=",
                                     "train_daily_true=",
                                     "validate_daily_true=",
                                     "train_daily_pred=",
                                     "validate_daily_pred=",
                                     "train_daily_lower=",
                                     "validate_daily_lower=",
                                     "train_daily_upper=",
                                     "validate_daily_upper="}))
        << line;
    EXPECT_EQ(line.find("daily_quantile_low_loss="), std::string::npos) << line;
}

TEST(TrainingRunsStatsReporter, RunningSummarySmoothsPartialEpochsButPublishesCompletedEpochsExactly) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun(
        "fold_0",
        TrainingRunsStatsReporter::RunConfig{
            0.0,
            std::string("sku_demand_cv5"),
            1.0,
            {"daily_central_loss"}});

    auto makeEpochStats = [](TrainingEventPhase phase,
                             uint64_t epoch,
                             uint64_t stepInEpoch,
                             uint64_t stepsPerEpoch,
                             double loss,
                             double metric) {
        TrainingStatsSnapshot stats = makeStats(phase, loss);
        stats.epoch = epoch;
        stats.stepInEpoch = stepInEpoch;
        stats.stepsPerEpoch = stepsPerEpoch;
        stats.metrics["daily_central_loss"] = metric;
        return stats;
    };

    TrainingStatsSnapshot trainEpoch1Batch1 =
        makeEpochStats(TrainingEventPhase::TRAIN, 1, 1, 2, 10.0, 100.0);
    TrainingStatsSnapshot trainEpoch1Batch2 =
        makeEpochStats(TrainingEventPhase::TRAIN, 1, 2, 2, 14.0, 140.0);
    TrainingStatsSnapshot validateEpoch1 =
        makeEpochStats(TrainingEventPhase::VALIDATE, 1, 1, 1, 20.0, 200.0);
    TrainingStatsSnapshot namedValidateEpoch1 =
        makeEpochStats(TrainingEventPhase::VALIDATE, 1, 1, 1, 30.0, 300.0);
    namedValidateEpoch1.validationPopulation = "seen_sku";

    TrainingStatsSnapshot trainEpoch2Batch1 =
        makeEpochStats(TrainingEventPhase::TRAIN, 2, 1, 2, 100.0, 1000.0);
    TrainingStatsSnapshot validateEpoch2 =
        makeEpochStats(TrainingEventPhase::VALIDATE, 2, 1, 1, 80.0, 800.0);
    TrainingStatsSnapshot namedValidateEpoch2 =
        makeEpochStats(TrainingEventPhase::VALIDATE, 2, 1, 1, 90.0, 900.0);
    namedValidateEpoch2.validationPopulation = "seen_sku";

    reporter.markRunStarting("fold_0");
    for (const TrainingStatsSnapshot* stats : {&trainEpoch1Batch1,
                                               &trainEpoch1Batch2,
                                               &validateEpoch1,
                                               &namedValidateEpoch1,
                                               &trainEpoch2Batch1,
                                               &validateEpoch2,
                                               &namedValidateEpoch2}) {
        reporter.onStatsEvent(
            TrainingStatsEvent::fromTrainingEvent(
                TrainingEvent::statsUpdated(*stats), "fold_0"));
    }
    reporter.close();

    const std::string output = readAndCloseFile(out);
    // Epoch-two training is only halfway complete, so it remains stabilized
    // against epoch one. Both validation populations are complete (1/1), so
    // their exact epoch-two aggregates must replace any previous-epoch blend.
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|sku_demand_cv5]:",
                                "train_loss=56.000000",
                                "validate_loss=80.000000",
                                "train_daily_central_loss=560.000000",
                                "validate_daily_central_loss=800.000000",
                                "validate_seen_sku_loss=90.000000",
                                "validate_seen_sku_daily_central_loss=900.000000"}))
        << output;
}

TEST(TrainingRunsStatsReporter, RunningSummaryWeightsPreviousEpochByRemainingFraction) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun(
        "fold_0",
        TrainingRunsStatsReporter::RunConfig{
            0.0,
            std::string("progress_weighting"),
            1.0,
            {"tracked_metric"}});

    auto makeTrainStats = [](uint64_t epoch,
                             uint64_t stepInEpoch,
                             uint64_t stepsPerEpoch,
                             double value) {
        TrainingStatsSnapshot stats = makeStats(TrainingEventPhase::TRAIN, value);
        stats.epoch = epoch;
        stats.stepInEpoch = stepInEpoch;
        stats.stepsPerEpoch = stepsPerEpoch;
        stats.metrics["tracked_metric"] = value * 10.0;
        return stats;
    };

    reporter.markRunStarting("fold_0");

    // Establish an exact completed previous epoch value of 10 (metric 100).
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(makeTrainStats(1, 10, 10, 10.0)), "fold_0"));

    // Keep the current epoch running aggregate fixed at 100 (metric 1000) so
    // the expected display values isolate only the progress weighting:
    //   10%:  0.90*10 + 0.10*100 = 19
    //   50%:  0.50*10 + 0.50*100 = 55
    //   90%:  0.10*10 + 0.90*100 = 91
    //  100%:  exact current epoch value = 100.
    for (uint64_t stepInEpoch : {1U, 5U, 9U, 10U}) {
        reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
            TrainingEvent::statsUpdated(makeTrainStats(2, stepInEpoch, 10, 100.0)), "fold_0"));
    }
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|progress_weighting]:",
                                "batch=         1/10",
                                "train_loss=19.000000",
                                "train_tracked_metric=190.000000"}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|progress_weighting]:",
                                "batch=         5/10",
                                "train_loss=55.000000",
                                "train_tracked_metric=550.000000"}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|progress_weighting]:",
                                "batch=         9/10",
                                "train_loss=91.000000",
                                "train_tracked_metric=910.000000"}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|progress_weighting]:",
                                "batch=        10/10",
                                "train_loss=100.000000",
                                "train_tracked_metric=1000.000000"}))
        << output;
}

TEST(TrainingRunsStatsReporter, CompletedValidationMetricMatchesScoreShownDuringNextEpoch) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun(
        "fold_0",
        TrainingRunsStatsReporter::RunConfig{
            0.0,
            std::string("product_transformer_cv"),
            1.0,
            {"transformer_loss"}});

    auto validationStats = [](uint64_t epoch, double transformerLoss) {
        TrainingStatsSnapshot stats = makeStats(TrainingEventPhase::VALIDATE, transformerLoss);
        stats.epoch = epoch;
        stats.stepInEpoch = 1;
        stats.stepsPerEpoch = 1;
        stats.metrics["transformer_loss"] = transformerLoss;
        return stats;
    };

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(validationStats(406, -4.707376)), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(validationStats(407, -4.871078)), "fold_0"));

    TrainingStatsSnapshot epoch408Started = makeStats(TrainingEventPhase::TRAIN, 0.0);
    epoch408Started.epoch = 408;
    epoch408Started.metrics["latest_score"] = -4.871078;
    epoch408Started.metrics["best_epoch"] = 407.0;
    epoch408Started.metrics["best_score"] = -4.871078;
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::epochStarted(epoch408Started), "fold_0"));

    TrainingStatsSnapshot epoch408Train = makeStats(TrainingEventPhase::TRAIN, -8.450872);
    epoch408Train.epoch = 408;
    epoch408Train.epochs = 2905;
    epoch408Train.step = 181022;
    epoch408Train.stepInEpoch = 314;
    epoch408Train.stepsPerEpoch = 444;
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(epoch408Train), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string line = findLineWithAll(
        output,
        {"INFO runs[fold_0|product_transformer_cv]:",
         "epoch=",
         "score=-4.871078",
         "best_epoch=407",
         "best_score=-4.871078",
         "validate_transformer_loss=-4.871078"});
    ASSERT_FALSE(line.empty()) << output;
    EXPECT_TRUE(hasTokenWithValue(line, "epoch", "408/2905")) << line;
}

TEST(TrainingRunsStatsReporter, RunningSummaryUsesDeclaredMetricAggregationContracts) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun(
        "fold_0",
        TrainingRunsStatsReporter::RunConfig{
            0.0,
            std::string("exact_metrics"),
            1.0,
            {"mean", "sum", "min", "max", "ratio"}});

    auto addMetric = [](TrainingStatsSnapshot& stats,
                        const std::string& name,
                        MetricAggregation aggregation,
                        double value,
                        uint64_t validExamples,
                        std::optional<double> numerator = std::nullopt,
                        std::optional<double> denominator = std::nullopt) {
        stats.metrics[name] = value;
        stats.metricBatchStats[name] = MetricBatchStat{
            .aggregation = aggregation,
            .value = value,
            .validExamples = validExamples,
            .numerator = numerator,
            .denominator = denominator,
        };
    };

    TrainingStatsSnapshot batch0 = makeStats(TrainingEventPhase::TRAIN, 1.0);
    batch0.epoch = 1;
    batch0.stepInEpoch = 1;
    batch0.stepsPerEpoch = 2;
    batch0.validExamplesInBatch = 4;
    addMetric(batch0, "mean", MetricAggregation::MEAN_BY_EXAMPLE, 2.5, 4);
    addMetric(batch0, "sum", MetricAggregation::SUM, 10.0, 4);
    addMetric(batch0, "min", MetricAggregation::MIN, 1.0, 4);
    addMetric(batch0, "max", MetricAggregation::MAX, 4.0, 4);
    addMetric(batch0, "ratio", MetricAggregation::RATIO, 5.0, 4, 10.0, 2.0);

    TrainingStatsSnapshot batch1 = makeStats(TrainingEventPhase::TRAIN, 1.0);
    batch1.epoch = 1;
    batch1.stepInEpoch = 2;
    batch1.stepsPerEpoch = 2;
    batch1.validExamplesInBatch = 2;
    addMetric(batch1, "mean", MetricAggregation::MEAN_BY_EXAMPLE, 10.0, 2);
    addMetric(batch1, "sum", MetricAggregation::SUM, 20.0, 2);
    addMetric(batch1, "min", MetricAggregation::MIN, -3.0, 2);
    addMetric(batch1, "max", MetricAggregation::MAX, 11.0, 2);
    addMetric(batch1, "ratio", MetricAggregation::RATIO, 10.0, 2, 90.0, 9.0);

    reporter.markRunStarting("fold_0");
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(batch0), "fold_0"));
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(batch1), "fold_0"));

    // The reporter drains queued events before emitting a summary, so without
    // an explicit boundary the epoch-one-final row is scheduling-dependent: the
    // worker may consume epoch2Batch0 in the same drain and only emit afterward.
    // This test asserts both the exact completed epoch-one aggregate and the
    // epoch-two smoothed aggregate, so make the observation point deterministic.
    reporter.flush();

    TrainingStatsSnapshot epoch2Batch0 = makeStats(TrainingEventPhase::TRAIN, 1.0);
    epoch2Batch0.epoch = 2;
    epoch2Batch0.stepInEpoch = 1;
    epoch2Batch0.stepsPerEpoch = 2;
    epoch2Batch0.validExamplesInBatch = 4;
    addMetric(epoch2Batch0, "mean", MetricAggregation::MEAN_BY_EXAMPLE, 20.0, 4);
    addMetric(epoch2Batch0, "sum", MetricAggregation::SUM, 40.0, 4);
    addMetric(epoch2Batch0, "min", MetricAggregation::MIN, 2.0, 4);
    addMetric(epoch2Batch0, "max", MetricAggregation::MAX, 40.0, 4);
    addMetric(epoch2Batch0, "ratio", MetricAggregation::RATIO, 15.0, 4, 30.0, 2.0);
    reporter.onStatsEvent(TrainingStatsEvent::fromTrainingEvent(
        TrainingEvent::statsUpdated(epoch2Batch0), "fold_0"));
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|exact_metrics]:",
                                "train_mean=5.000000",
                                "train_sum=30.000000",
                                "train_min=-3.000000",
                                "train_max=11.000000",
                                "train_ratio=9.090909"}))
        << output;
    // Epoch-two smoothing blends the previous exact epoch aggregate with the
    // current exact running aggregate at 50% progress. It must not blend raw
    // batch scalars or arithmetic means of batch-level sums/extrema/ratios.
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0|exact_metrics]:",
                                "train_mean=12.500000",
                                "train_sum=35.000000",
                                "train_min=-0.500000",
                                "train_max=25.500000",
                                "train_ratio=12.045455"}))
        << output;
}

TEST(TrainingRunsStatsReporter, FinalReportKeepsTrainAndValidateMetricsPairedByConfiguredOrder) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);
    reporter.configureRun("completed_fold",
                          TrainingRunsStatsReporter::RunConfig{0.0,
                                                               std::string("sku_demand_cv5"),
                                                               1.0,
                                                               {"agg_central_loss",
                                                                "agg_true",
                                                                "agg_pred",
                                                                "agg_lower",
                                                                "agg_upper"}});

    TrainingStatsSnapshot trainStats = makeStats(TrainingEventPhase::TRAIN, 0.50);
    trainStats.metrics["agg_central_loss"] = 42.0;
    trainStats.metrics["agg_true"] = 280.07;
    trainStats.metrics["agg_pred"] = 259.65;
    trainStats.metrics["agg_lower"] = 220.10;
    trainStats.metrics["agg_upper"] = 297.60;

    TrainingStatsSnapshot validateStats = makeStats(TrainingEventPhase::VALIDATE, 0.40);
    validateStats.metrics["agg_central_loss"] = 45.0;
    validateStats.metrics["agg_true"] = 281.00;
    validateStats.metrics["agg_pred"] = 260.00;
    validateStats.metrics["agg_lower"] = 221.00;
    validateStats.metrics["agg_upper"] = 299.00;

    TrainingRunResult completed = TrainingRunResult::completedResult(
        "completed_fold", trainStats, validateStats, std::nullopt);
    completed.ensembleGroup = "sku_demand_cv5";

    reporter.emitFinalReport(std::vector<TrainingRunResult>{completed});
    reporter.close();

    const std::string output = readAndCloseFile(out);
    const std::string line = findLineWithAll(output,
                                             {"INFO runs[completed_fold|sku_demand_cv5]:",
                                              "train_agg_central_loss=",
                                              "validate_agg_upper="});
    ASSERT_FALSE(line.empty()) << output;
    EXPECT_TRUE(tokensAppearInOrder(line,
                                    {"train_agg_central_loss=",
                                     "validate_agg_central_loss=",
                                     "train_agg_true=",
                                     "validate_agg_true=",
                                     "train_agg_pred=",
                                     "validate_agg_pred=",
                                     "train_agg_lower=",
                                     "validate_agg_lower=",
                                     "train_agg_upper=",
                                     "validate_agg_upper="}))
        << line;
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

    TrainingStatsSnapshot finalTrainingStats = makeStats(TrainingEventPhase::TRAIN, 0.50);
    finalTrainingStats.metrics["daily_central_loss"] = 0.55;
    TrainingStatsSnapshot finalValidationStats = makeStats(TrainingEventPhase::VALIDATE, 0.40);
    finalValidationStats.metrics["daily_central_loss"] = 0.45;
    TrainingRunResult result = TrainingRunResult::completedResult(
        "early_fold",
        finalTrainingStats,
        finalValidationStats,
        {},
        TrainingRunCompletionReason::EARLY_COMPLETED,
        2,
        1,
        0.125);
    result.selectedEpoch = 1;
    result.latestScore = 0.45;
    TrainingModelSelectionContext bestContext;
    bestContext.epoch = 1;
    bestContext.train.loss = 0.25;
    bestContext.train.losses["daily_central_loss"] = 0.30;
    bestContext.validate.loss = 0.20;
    bestContext.validate.losses["daily_central_loss"] = 0.125;
    bestContext.validations["validate"] = bestContext.validate;
    result.bestModelSelectionContext = bestContext;

    reporter.markRunStarting("early_fold");
    reporter.markRunFinished(result);
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[early_fold]:",
                                "status=completed",
                                "result=early_completed",
                                "train_loss=0.250000",
                                "validate_loss=0.200000",
                                "train_daily_central_loss=0.300000",
                                "validate_daily_central_loss=0.125000",
                                "latest_train_loss=0.500000",
                                "latest_validate_loss=0.400000",
                                "metrics_epoch=1",
                                "completed_epoch=2",
                                "selected_epoch=1",
                                "best_epoch=1",
                                "best_score=0.125000",
                                "latest_score=0.450000"}))
        << output;
}

TEST(TrainingRunsStatsReporter, FinalReportIncludesStatusCountsAndAvailablePhaseMetrics) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);

    TrainingStatsSnapshot trainStats = makeStats(TrainingEventPhase::TRAIN, 0.50);
    trainStats.metrics["top1_accuracy"] = 0.90;
    trainStats.metrics["top5_accuracy"] = 0.98;
    TrainingStatsSnapshot validateStats = makeStats(TrainingEventPhase::VALIDATE, 0.40);
    validateStats.metrics["f1_score"] = 0.72;
    TrainingStatsSnapshot testStats = makeStats(TrainingEventPhase::TEST, 0.35);
    testStats.metrics["top1_accuracy"] = 0.875;
    testStats.metrics["top5_accuracy"] = 0.975;
    TrainingRunResult completed = TrainingRunResult::completedResult(
        "completed_fold", trainStats, validateStats, testStats);
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
                                "train_top1_accuracy=",
                                "train_top5_accuracy=",
                                "validate_f1_score=",
                                "test_top1_accuracy=",
                                "test_top5_accuracy="}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output, {"INFO runs[failed_fold]:", "status=failed", "message=\"boom\""})) << output;
}

TEST(TrainingRunsStatsReporter, TrainingHistoryReportPrintsEveryCompletedPhaseWithItsOwnReportOrder) {
    std::FILE* out = std::tmpfile();
    TrainingRunsStatsReporter reporter(out, LineStatsColorMode::NEVER, 0.0);

    TrainingStatsSnapshot baseTrain = makeStats(TrainingEventPhase::TRAIN, 0.80);
    baseTrain.metrics["base_metric"] = 1.25;
    TrainingStatsSnapshot baseValidate = makeStats(TrainingEventPhase::VALIDATE, 0.70);
    baseValidate.metrics["base_metric"] = 1.10;
    TrainingRunResult baseResult = TrainingRunResult::completedResult("fold_0", baseTrain, baseValidate);

    TrainingStatsSnapshot transformerTrain = makeStats(TrainingEventPhase::TRAIN, 0.50);
    transformerTrain.metrics["transformer_metric"] = 2.25;
    TrainingStatsSnapshot transformerValidate = makeStats(TrainingEventPhase::VALIDATE, 0.40);
    transformerValidate.metrics["transformer_metric"] = 2.10;
    TrainingRunResult transformerResult =
        TrainingRunResult::completedResult("fold_0", transformerTrain, transformerValidate);

    reporter.emitTrainingHistoryReport({
        TrainingRunsStatsReporter::TrainingPhaseReport{
            "poisson_glm_pretrain",
            {TrainingRunsStatsReporter::HistoricalRunResult{baseResult, {"base_metric"}}}},
        TrainingRunsStatsReporter::TrainingPhaseReport{
            "transformer_residual",
            {TrainingRunsStatsReporter::HistoricalRunResult{transformerResult, {"transformer_metric"}}}},
    });
    reporter.close();

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(output.find("INFO runs final: ================== training history ==================") != std::string::npos) << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs phase[poisson_glm_pretrain]:", "total=1", "completed=1"}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs phase[transformer_residual]:", "total=1", "completed=1"}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0]:", "train_base_metric=1.250000", "validate_base_metric=1.100000"}))
        << output;
    EXPECT_TRUE(hasLineWithAll(output,
                               {"INFO runs[fold_0]:", "train_transformer_metric=2.250000", "validate_transformer_metric=2.100000"}))
        << output;

    const size_t basePhasePosition = output.find("INFO runs phase[poisson_glm_pretrain]:");
    const size_t transformerPhasePosition = output.find("INFO runs phase[transformer_residual]:");
    ASSERT_NE(basePhasePosition, std::string::npos);
    ASSERT_NE(transformerPhasePosition, std::string::npos);
    EXPECT_LT(basePhasePosition, transformerPhasePosition) << output;
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
    TrainingNamedMetricResult overallNamedLoss;
    overallNamedLoss.name = "loss";
    overallNamedLoss.trainValue = 0.123;
    overallNamedLoss.testValue = 0.456;
    TrainingNamedMetricResult dailyLoss;
    dailyLoss.name = "daily_loss";
    dailyLoss.trainValue = 0.111;
    dailyLoss.testValue = 0.222;
    TrainingNamedMetricResult aggregateLoss;
    aggregateLoss.name = "aggregate_loss";
    aggregateLoss.testValue = 0.333;
    completedEnsemble.namedMetrics = {overallNamedLoss, dailyLoss, aggregateLoss};
    TrainingNamedMetricResult accuracyMetric;
    accuracyMetric.name = "top1_accuracy";
    accuracyMetric.trainValue = 0.889;
    accuracyMetric.testValue = 0.901;
    TrainingNamedMetricResult top5Metric;
    top5Metric.name = "top5_accuracy";
    top5Metric.trainValue = 0.970;
    top5Metric.testValue = 0.981;
    completedEnsemble.namedGraphMetrics = {accuracyMetric, top5Metric};

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
        output, {"INFO runs ensemble[digits_dense_cv5]:",
                 "status=completed",
                 "aggregation=ensemble_eval",
                 "members=2",
                 "ensemble_train_loss=",
                 "ensemble_test_loss=",
                 "ensemble_train_top1_accuracy=",
                 "ensemble_test_top1_accuracy=",
                 "ensemble_train_daily_loss=",
                 "ensemble_test_daily_loss=",
                 "ensemble_test_aggregate_loss=",
                 "ensemble_train_top5_accuracy=",
                 "ensemble_test_top5_accuracy="});
    ASSERT_FALSE(completedLine.empty()) << output;
    EXPECT_EQ(completedLine.find(" completed="), std::string::npos) << completedLine;
    EXPECT_EQ(completedLine.find(" failed="), std::string::npos) << completedLine;
    const std::string trainLossKey = "ensemble_train_loss=";
    const std::string testLossKey = "ensemble_test_loss=";
    EXPECT_NE(completedLine.find(trainLossKey), std::string::npos) << completedLine;
    EXPECT_EQ(completedLine.find(trainLossKey, completedLine.find(trainLossKey) + trainLossKey.size()), std::string::npos) << completedLine;
    EXPECT_NE(completedLine.find(testLossKey), std::string::npos) << completedLine;
    EXPECT_EQ(completedLine.find(testLossKey, completedLine.find(testLossKey) + testLossKey.size()), std::string::npos) << completedLine;

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

