#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include "gtest/gtest.h"

#include <sstream>
#include <string>

using namespace Thor;

namespace {

TrainingStatsSnapshot makeStats(double elapsedSeconds) {
    TrainingStatsSnapshot stats;
    stats.phase = TrainingPhase::TRAIN;
    stats.epoch = 1;
    stats.epochs = 3;
    stats.step = 17;
    stats.stepInEpoch = 7;
    stats.stepsPerEpoch = 100;
    stats.loss = 1.25;
    stats.accuracy = 0.75;
    stats.learningRate = 3.0e-4;
    stats.samplesPerSecond = 1024.0;
    stats.batchesPerSecond = 8.0;
    stats.inFlightBatches = 32;
    stats.elapsedSeconds = elapsedSeconds;
    return stats;
}

}  // namespace

TEST(LineStatsReporter, FormatsVllmStyleStatsLine) {
    const std::string line = LineStatsReporter::formatStatsLine(makeStats(65.0));

    EXPECT_EQ(line,
              "INFO trainer: phase=train epoch=1/3 step=17 batch=7/100 loss=1.250000 accuracy=0.7500 "
              "lr=3.000e-04 samples/s=1024.0 batches/s=8.00 in_flight=32 elapsed=00:01:05");
}

TEST(LineStatsReporter, DisabledReporterEmitsNothing) {
    std::ostringstream out;
    LineStatsReporter reporter(out, 0.0, false);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0)));

    EXPECT_TRUE(out.str().empty());
}

TEST(LineStatsReporter, RateLimitsStatsByElapsedSeconds) {
    std::ostringstream out;
    LineStatsReporter reporter(out, 5.0, true);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(2.0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(6.0)));

    const std::string output = out.str();
    size_t lines = 0;
    for (char ch : output) {
        if (ch == '\n') {
            ++lines;
        }
    }
    EXPECT_EQ(lines, 2u);
}

TEST(LineStatsReporter, ColorModeNeverEmitsPlainStatsLine) {
    std::ostringstream out;
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::NEVER);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    EXPECT_EQ(out.str(), LineStatsReporter::formatStatsLine(makeStats(65.0)) + "\n");
}

TEST(LineStatsReporter, ColorModeAlwaysAddsAnsi) {
    std::ostringstream out;
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::ALWAYS);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    const std::string output = out.str();
    EXPECT_NE(output.find("\x1b["), std::string::npos);
    EXPECT_NE(output.find("INFO"), std::string::npos);
    EXPECT_NE(output.find("loss"), std::string::npos);
    EXPECT_NE(output.find("1.250000"), std::string::npos);
}

TEST(LineStatsReporter, ColorModeAutoDoesNotColorNonTerminalStreams) {
    std::ostringstream out;
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::AUTO);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    const std::string output = out.str();
    EXPECT_EQ(output.find("\x1b["), std::string::npos);
    EXPECT_EQ(output, LineStatsReporter::formatStatsLine(makeStats(65.0)) + "\n");
}
