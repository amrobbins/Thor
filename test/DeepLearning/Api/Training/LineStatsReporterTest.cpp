#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdio>
#include <stdexcept>
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
    stats.floatingPointOperationsPerBatch = 1250000000;
    stats.floatingPointOperationsPerSecond = 2500000000000.0;
    stats.inFlightBatches = 32;
    stats.elapsedSeconds = elapsedSeconds;
    return stats;
}

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

TEST(LineStatsReporter, FormatsVllmStyleStatsLine) {
    const std::string line = LineStatsReporter::formatStatsLine(makeStats(65.0));

    EXPECT_EQ(line,
              "INFO trainer: phase=train epoch=1/3 step=17 batch=7/100 loss=1.250000 accuracy=0.7500 "
              "lr=3.000e-04 samples/s=1024.0 batches/s=8.00 flops/s=2.50T in_flight=32 elapsed=00:01:05");
}

TEST(LineStatsReporter, DisabledReporterEmitsNothing) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, false);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0)));

    EXPECT_TRUE(readAndCloseFile(out).empty());
}

TEST(LineStatsReporter, RateLimitsStatsByElapsedSeconds) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 5.0, true);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(2.0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(6.0)));

    const std::string output = readAndCloseFile(out);
    std::size_t lines = 0;
    for (char ch : output) {
        if (ch == '\n') {
            ++lines;
        }
    }
    EXPECT_EQ(lines, 2u);
}

TEST(LineStatsReporter, ColorModeNeverEmitsPlainStatsLine) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::NEVER);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    EXPECT_EQ(readAndCloseFile(out), LineStatsReporter::formatStatsLine(makeStats(65.0)) + "\n");
}

TEST(LineStatsReporter, ColorModeAlwaysAddsAnsi) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::ALWAYS);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("\x1b["), std::string::npos);
    EXPECT_NE(output.find("INFO"), std::string::npos);
    EXPECT_NE(output.find("loss"), std::string::npos);
    EXPECT_NE(output.find("1.250000"), std::string::npos);
    EXPECT_NE(output.find("flops/s"), std::string::npos);
    EXPECT_NE(output.find("2.50T"), std::string::npos);
    EXPECT_NE(output.find("\x1b[1;90m2.50T"), std::string::npos);
    EXPECT_NE(output.find("\x1b[34m00:01:05"), std::string::npos);
}

TEST(LineStatsReporter, ColorModeAutoDoesNotColorNonTerminalFiles) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::AUTO);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    const std::string output = readAndCloseFile(out);
    EXPECT_EQ(output.find("\x1b["), std::string::npos);
    EXPECT_EQ(output, LineStatsReporter::formatStatsLine(makeStats(65.0)) + "\n");
}
