#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace Thor {
namespace {

namespace Ansi {
constexpr const char* reset = "\x1b[0m";
constexpr const char* bold = "\x1b[1m";
constexpr const char* dim = "\x1b[2m";
constexpr const char* green = "\x1b[32m";
constexpr const char* yellow = "\x1b[33m";
constexpr const char* blue = "\x1b[34m";
constexpr const char* magenta = "\x1b[35m";
constexpr const char* cyan = "\x1b[36m";
constexpr const char* brightBlack = "\x1b[90m";
constexpr const char* boldBrightBlack = "\x1b[1;90m";
constexpr const char* dimGreen = "\x1b[2;32m";
constexpr const char* dimBlue = "\x1b[2;34m";
constexpr const char* dimMagenta = "\x1b[2;35m";
}  // namespace Ansi

void appendFixed(std::ostringstream& out, double value, int precision) {
    const std::ios::fmtflags oldFlags = out.flags();
    const std::streamsize oldPrecision = out.precision();
    out << std::fixed << std::setprecision(precision) << value;
    out.flags(oldFlags);
    out.precision(oldPrecision);
}

void appendScientific(std::ostringstream& out, double value, int precision) {
    const std::ios::fmtflags oldFlags = out.flags();
    const std::streamsize oldPrecision = out.precision();
    out << std::scientific << std::setprecision(precision) << value;
    out.flags(oldFlags);
    out.precision(oldPrecision);
}

void appendCompactRate(std::ostringstream& out, double value) {
    const double absValue = std::abs(value);
    const char* suffix = "";
    double scaled = value;
    if (absValue >= 1.0e12) {
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
    const int precision = absScaled >= 100.0 ? 0 : (absScaled >= 10.0 ? 1 : 2);
    appendFixed(out, scaled, precision);
    out << suffix;
}

bool envFlagIsSet(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

bool envFlagIsZero(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '0' && value[1] == '\0';
}

bool terminalEnvDisablesColor() {
    if (envFlagIsSet("NO_COLOR")) {
        return true;
    }
    return envFlagIsZero("CLICOLOR");
}

bool terminalEnvForcesColor() {
    return envFlagIsSet("CLICOLOR_FORCE") && !envFlagIsZero("CLICOLOR_FORCE");
}

bool terminalKindAllowsColor() {
    const char* term = std::getenv("TERM");
    return term == nullptr || std::string(term) != "dumb";
}

bool fileIsTerminal(std::FILE* output) {
#if defined(__unix__) || defined(__APPLE__)
    return output != nullptr && ::isatty(::fileno(output)) != 0;
#else
    (void)output;
    return false;
#endif
}

bool fileSupportsAnsiColor(std::FILE* output) {
    if (envFlagIsSet("NO_COLOR")) {
        return false;
    }
    if (terminalEnvForcesColor()) {
        return true;
    }
    if (terminalEnvDisablesColor() || !terminalKindAllowsColor()) {
        return false;
    }
    return fileIsTerminal(output);
}

void appendPhaseValue(std::ostringstream& out, TrainingPhase phase) {
    switch (phase) {
        case TrainingPhase::TRAIN:
            out << Ansi::dimGreen;
            break;
        case TrainingPhase::VALIDATE:
            out << Ansi::dimBlue;
            break;
        case TrainingPhase::TEST:
            out << Ansi::dimMagenta;
            break;
        case TrainingPhase::UNKNOWN:
        default:
            out << Ansi::brightBlack;
            break;
    }
    out << trainingPhaseName(phase) << Ansi::reset;
}

void appendDimKey(std::ostringstream& out, const char* key) {
    out << ' ' << Ansi::dim << key << '=' << Ansi::reset;
}

}  // namespace

LineStatsReporter::LineStatsReporter(std::FILE* output,
                                     double intervalSeconds,
                                     bool enabled,
                                     LineStatsColorMode colorMode)
    : outputFile(output), intervalSeconds(intervalSeconds), enabled(enabled), colorMode(colorMode) {
    setIntervalSeconds(intervalSeconds);
}

LineStatsReporter::LineStatsReporter(double intervalSeconds, bool enabled, LineStatsColorMode colorMode)
    : LineStatsReporter(stdout, intervalSeconds, enabled, colorMode) {}

void LineStatsReporter::setIntervalSeconds(double intervalSeconds) {
    if (!std::isfinite(intervalSeconds) || intervalSeconds < 0.0) {
        throw std::runtime_error("LineStatsReporter intervalSeconds must be finite and >= 0.");
    }
    this->intervalSeconds = intervalSeconds;
}

void LineStatsReporter::onTrainingEvent(const TrainingEvent& event) {
    if (!enabled || outputFile == nullptr) {
        return;
    }

    if (event.type != TrainingEventType::STATS) {
        return;
    }

    if (!shouldPrintStats(event.stats)) {
        return;
    }

    writeStatsLine(event.stats);
}

bool LineStatsReporter::shouldPrintStats(const TrainingStatsSnapshot& stats) {
    if (!printedAnyStats) {
        printedAnyStats = true;
        lastPrintedElapsedSeconds = stats.elapsedSeconds;
        return true;
    }

    if (intervalSeconds == 0.0) {
        lastPrintedElapsedSeconds = stats.elapsedSeconds;
        return true;
    }

    const double lastElapsed = lastPrintedElapsedSeconds.value_or(0.0);
    if (stats.elapsedSeconds - lastElapsed >= intervalSeconds) {
        lastPrintedElapsedSeconds = stats.elapsedSeconds;
        return true;
    }
    return false;
}

bool LineStatsReporter::isAnsiColorSupported(std::FILE* output) {
    return fileSupportsAnsiColor(output);
}

bool LineStatsReporter::shouldUseColor() const {
    if (colorMode == LineStatsColorMode::NEVER) {
        return false;
    }
    if (colorMode == LineStatsColorMode::ALWAYS) {
        return true;
    }
    return fileSupportsAnsiColor(outputFile);
}

namespace {

std::string formatColorStatsLine(const TrainingStatsSnapshot& stats) {
    std::ostringstream out;
    out << Ansi::dim << "INFO " << Ansi::reset << Ansi::bold << "trainer:" << Ansi::reset;

    appendDimKey(out, "phase");
    appendPhaseValue(out, stats.phase);

    if (stats.epochs > 0) {
        appendDimKey(out, "epoch");
        out << Ansi::blue << stats.epoch << '/' << stats.epochs << Ansi::reset;
    } else if (stats.epoch > 0) {
        appendDimKey(out, "epoch");
        out << Ansi::blue << stats.epoch << Ansi::reset;
    }

    if (stats.step > 0) {
        appendDimKey(out, "step");
        out << Ansi::blue << stats.step << Ansi::reset;
    }

    if (stats.stepsPerEpoch > 0) {
        appendDimKey(out, "batch");
        out << Ansi::blue << stats.stepInEpoch << '/' << stats.stepsPerEpoch << Ansi::reset;
    } else if (stats.stepInEpoch > 0) {
        appendDimKey(out, "batch");
        out << Ansi::blue << stats.stepInEpoch << Ansi::reset;
    }

    if (stats.loss.has_value()) {
        appendDimKey(out, "loss");
        out << Ansi::yellow;
        appendFixed(out, stats.loss.value(), 6);
        out << Ansi::reset;
    }
    if (stats.accuracy.has_value()) {
        appendDimKey(out, "accuracy");
        out << Ansi::green;
        appendFixed(out, stats.accuracy.value(), 4);
        out << Ansi::reset;
    }
    if (stats.learningRate.has_value()) {
        appendDimKey(out, "lr");
        out << Ansi::magenta;
        appendScientific(out, stats.learningRate.value(), 3);
        out << Ansi::reset;
    }

    if (stats.samplesPerSecond > 0.0) {
        appendDimKey(out, "samples/s");
        out << Ansi::cyan;
        appendFixed(out, stats.samplesPerSecond, 1);
        out << Ansi::reset;
    }
    if (stats.batchesPerSecond > 0.0) {
        appendDimKey(out, "batches/s");
        out << Ansi::cyan;
        appendFixed(out, stats.batchesPerSecond, 2);
        out << Ansi::reset;
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        appendDimKey(out, "flops/s");
        out << Ansi::boldBrightBlack;
        appendCompactRate(out, stats.floatingPointOperationsPerSecond);
        out << Ansi::reset;
    }
    if (stats.inFlightBatches > 0) {
        appendDimKey(out, "in_flight");
        out << Ansi::blue << stats.inFlightBatches << Ansi::reset;
    }

    appendDimKey(out, "elapsed");
    out << Ansi::blue << LineStatsReporter::formatElapsedSeconds(stats.elapsedSeconds) << Ansi::reset;
    return out.str();
}

}  // namespace

void LineStatsReporter::writeStatsLine(const TrainingStatsSnapshot& stats) {
    emitLine(shouldUseColor() ? formatColorStatsLine(stats) : formatStatsLine(stats));
}

void LineStatsReporter::emitLine(const std::string& line) {
    if (outputFile != nullptr) {
        std::fprintf(outputFile, "%s\n", line.c_str());
    }
}

std::string LineStatsReporter::formatStatsLine(const TrainingStatsSnapshot& stats) {
    std::ostringstream out;
    out << "INFO trainer: phase=" << trainingPhaseName(stats.phase);

    if (stats.epochs > 0) {
        out << " epoch=" << stats.epoch << '/' << stats.epochs;
    } else if (stats.epoch > 0) {
        out << " epoch=" << stats.epoch;
    }

    if (stats.step > 0) {
        out << " step=" << stats.step;
    }

    if (stats.stepsPerEpoch > 0) {
        out << " batch=" << stats.stepInEpoch << '/' << stats.stepsPerEpoch;
    } else if (stats.stepInEpoch > 0) {
        out << " batch=" << stats.stepInEpoch;
    }

    if (stats.loss.has_value()) {
        out << " loss=";
        appendFixed(out, stats.loss.value(), 6);
    }
    if (stats.accuracy.has_value()) {
        out << " accuracy=";
        appendFixed(out, stats.accuracy.value(), 4);
    }
    if (stats.learningRate.has_value()) {
        out << " lr=";
        appendScientific(out, stats.learningRate.value(), 3);
    }

    if (stats.samplesPerSecond > 0.0) {
        out << " samples/s=";
        appendFixed(out, stats.samplesPerSecond, 1);
    }
    if (stats.batchesPerSecond > 0.0) {
        out << " batches/s=";
        appendFixed(out, stats.batchesPerSecond, 2);
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        out << " flops/s=";
        appendCompactRate(out, stats.floatingPointOperationsPerSecond);
    }
    if (stats.inFlightBatches > 0) {
        out << " in_flight=" << stats.inFlightBatches;
    }

    out << " elapsed=" << formatElapsedSeconds(stats.elapsedSeconds);

    return out.str();
}

std::string LineStatsReporter::formatElapsedSeconds(double elapsedSeconds) {
    const uint64_t roundedSeconds = static_cast<uint64_t>(std::max(0.0, elapsedSeconds));
    const uint64_t hours = roundedSeconds / 3600;
    const uint64_t minutes = (roundedSeconds / 60) % 60;
    const uint64_t seconds = roundedSeconds % 60;

    std::ostringstream out;
    out << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
    return out.str();
}

}  // namespace Thor
