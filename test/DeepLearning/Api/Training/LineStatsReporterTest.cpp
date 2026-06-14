#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace Thor;

namespace {

TrainingStatsSnapshot makeStats(double elapsedSeconds,
                                TrainingPhase phase = TrainingPhase::TRAIN,
                                uint64_t epoch = 1,
                                uint64_t step = 17,
                                uint64_t stepInEpoch = 7,
                                uint64_t stepsPerEpoch = 100) {
    TrainingStatsSnapshot stats;
    stats.phase = phase;
    stats.epoch = epoch;
    stats.epochs = 3;
    stats.step = step;
    stats.stepInEpoch = stepInEpoch;
    stats.stepsPerEpoch = stepsPerEpoch;
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


std::size_t countLines(const std::string& output) {
    std::size_t lines = 0;
    for (char ch : output) {
        if (ch == '\n') {
            ++lines;
        }
    }
    return lines;
}

std::string stripAnsiSequences(const std::string& output) {
    std::string stripped;
    for (std::size_t i = 0; i < output.size();) {
        if (output[i] == '\x1b' && i + 1 < output.size() && output[i + 1] == '[') {
            i += 2;
            while (i < output.size() && output[i] != 'm') {
                ++i;
            }
            if (i < output.size()) {
                ++i;
            }
            continue;
        }
        stripped.push_back(output[i]);
        ++i;
    }
    return stripped;
}

const char* alignedColorStatsLineWithoutAnsi() {
    return "INFO trainer: phase=train    epoch=    1/3 step=      17 batch=  7/100 loss= 1.250000 accuracy=0.7500 "
           "lr=3.000e-04 samples/s=  1024.0 batches/s=   8.00 flops/s=  2.50T in_flight= 32 elapsed=00:01:05";
}

bool tokenHasAnsiStyle(const std::string& output, const std::string& token, bool requireBold) {
    std::size_t pos = 0;
    while ((pos = output.find(token, pos)) != std::string::npos) {
        const std::size_t esc = output.rfind("\x1b[", pos);
        if (esc != std::string::npos) {
            const std::size_t styleEnd = output.find('m', esc);
            if (styleEnd != std::string::npos && styleEnd < pos) {
                bool onlyPaddingBetweenStyleAndToken = true;
                for (std::size_t i = styleEnd + 1; i < pos; ++i) {
                    if (output[i] != ' ') {
                        onlyPaddingBetweenStyleAndToken = false;
                        break;
                    }
                }

                const std::string style = output.substr(esc, styleEnd - esc + 1);
                const bool isBoldStyle = style.find("\x1b[1") != std::string::npos || style.find(";1") != std::string::npos;
                if (onlyPaddingBetweenStyleAndToken && (!requireBold || isBoldStyle)) {
                    return true;
                }
            }
        }
        pos += token.size();
    }
    return false;
}

}  // namespace

TEST(LineStatsReporter, FormatsAlignedStatsLineWithoutAnsi) {
    const std::string line = LineStatsReporter::formatStatsLine(makeStats(65.0));

    EXPECT_EQ(line, alignedColorStatsLineWithoutAnsi());
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
    EXPECT_EQ(countLines(output), 2u);
}

TEST(LineStatsReporter, PrintsFirstStatsForEachPhaseOccurrenceDespiteInterval) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 100.0, true, LineStatsColorMode::NEVER);

    reporter.onTrainingEvent(TrainingEvent::epochStarted(makeStats(0.0, TrainingPhase::TRAIN, 1, 0, 0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0, TrainingPhase::TRAIN, 1, 1, 1)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(1.0, TrainingPhase::TRAIN, 1, 2, 2)));
    reporter.onTrainingEvent(TrainingEvent::epochFinished(makeStats(1.0, TrainingPhase::TRAIN, 1, 0, 0)));

    reporter.onTrainingEvent(TrainingEvent::epochStarted(makeStats(2.0, TrainingPhase::VALIDATE, 1, 0, 0, 5)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(2.0, TrainingPhase::VALIDATE, 1, 1, 1, 5)));

    const std::string output = readAndCloseFile(out);
    EXPECT_NE(output.find("phase=train    epoch=    1/3 step=       1 batch=  1/100"), std::string::npos);
    EXPECT_NE(output.find("phase=validate epoch=    1/3 step=       1 batch=    1/5"), std::string::npos);
}

TEST(LineStatsReporter, PrintsLastStatsAtPhaseFinishDespiteInterval) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 100.0, true, LineStatsColorMode::NEVER);

    reporter.onTrainingEvent(TrainingEvent::epochStarted(makeStats(0.0, TrainingPhase::TRAIN, 1, 0, 0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0, TrainingPhase::TRAIN, 1, 1, 1)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(1.0, TrainingPhase::TRAIN, 1, 2, 2)));
    reporter.onTrainingEvent(TrainingEvent::epochFinished(makeStats(1.0, TrainingPhase::TRAIN, 1, 0, 0)));

    const std::string output = readAndCloseFile(out);
    EXPECT_EQ(countLines(output), 2u);
    EXPECT_NE(output.find("phase=train    epoch=    1/3 step=       1 batch=  1/100"), std::string::npos);
    EXPECT_NE(output.find("phase=train    epoch=    1/3 step=       2 batch=  2/100"), std::string::npos);
}

TEST(LineStatsReporter, DoesNotDuplicatePhaseFinishWhenLastStatsAlreadyPrinted) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::NEVER);

    reporter.onTrainingEvent(TrainingEvent::epochStarted(makeStats(0.0, TrainingPhase::TRAIN, 1, 0, 0)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(0.0, TrainingPhase::TRAIN, 1, 1, 1)));
    reporter.onTrainingEvent(TrainingEvent::epochFinished(makeStats(0.0, TrainingPhase::TRAIN, 1, 0, 0)));

    const std::string output = readAndCloseFile(out);
    EXPECT_EQ(countLines(output), 1u);
    EXPECT_NE(output.find("phase=train    epoch=    1/3 step=       1 batch=  1/100"), std::string::npos);
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
    EXPECT_TRUE(tokenHasAnsiStyle(output, "train", true));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "1024.0", true));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "8.00", true));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "2.50T", true));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "1.250000", false));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "00:01:05", false));
    EXPECT_EQ(stripAnsiSequences(output), LineStatsReporter::formatStatsLine(makeStats(65.0)) + "\n");
}

TEST(LineStatsReporter, ColorModeAlwaysBoldColorsPhaseByPhaseType) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::ALWAYS);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(1.0, TrainingPhase::TRAIN, 1, 1, 1)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(2.0, TrainingPhase::VALIDATE, 1, 2, 1, 5)));
    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(3.0, TrainingPhase::TEST, 1, 3, 1, 5)));

    const std::string output = readAndCloseFile(out);
    EXPECT_TRUE(tokenHasAnsiStyle(output, "train", true));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "validate", true));
    EXPECT_TRUE(tokenHasAnsiStyle(output, "test", true));
}

TEST(LineStatsReporter, ColorModeAutoDoesNotColorNonTerminalFiles) {
    std::FILE* out = std::tmpfile();
    LineStatsReporter reporter(out, 0.0, true, LineStatsColorMode::AUTO);

    reporter.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(65.0)));

    const std::string output = readAndCloseFile(out);
    EXPECT_EQ(output.find("\x1b["), std::string::npos);
    EXPECT_EQ(output, LineStatsReporter::formatStatsLine(makeStats(65.0)) + "\n");
}
