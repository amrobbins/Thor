#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <cstdio>
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
constexpr const char* white = "\x1b[37m";
}  // namespace Ansi

void appendFixed(std::ostream& out, double value, int precision) {
    const std::ios::fmtflags oldFlags = out.flags();
    const std::streamsize oldPrecision = out.precision();
    out << std::fixed << std::setprecision(precision) << value;
    out.flags(oldFlags);
    out.precision(oldPrecision);
}

void appendScientific(std::ostream& out, double value, int precision) {
    const std::ios::fmtflags oldFlags = out.flags();
    const std::streamsize oldPrecision = out.precision();
    out << std::scientific << std::setprecision(precision) << value;
    out.flags(oldFlags);
    out.precision(oldPrecision);
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

bool streamIsTerminal(std::ostream* output) {
#if defined(__unix__) || defined(__APPLE__)
    if (output == &std::cout) {
        return ::isatty(::fileno(stdout)) != 0;
    }
    if (output == &std::cerr || output == &std::clog) {
        return ::isatty(::fileno(stderr)) != 0;
    }
#else
    (void)output;
#endif
    return false;
}

bool streamSupportsAnsiColor(std::ostream* output) {
    if (envFlagIsSet("NO_COLOR")) {
        return false;
    }
    if (terminalEnvForcesColor()) {
        return true;
    }
    if (terminalEnvDisablesColor() || !terminalKindAllowsColor()) {
        return false;
    }
    return streamIsTerminal(output);
}

void appendPhaseValue(std::ostream& out, TrainingPhase phase) {
    switch (phase) {
        case TrainingPhase::TRAIN:
            out << Ansi::green;
            break;
        case TrainingPhase::VALIDATE:
            out << Ansi::cyan;
            break;
        case TrainingPhase::TEST:
            out << Ansi::magenta;
            break;
        case TrainingPhase::UNKNOWN:
        default:
            out << Ansi::white;
            break;
    }
    out << trainingPhaseName(phase) << Ansi::reset;
}

void appendDimKey(std::ostream& out, const char* key) {
    out << ' ' << Ansi::dim << key << '=' << Ansi::reset;
}

}  // namespace

LineStatsReporter::LineStatsReporter(std::ostream& output,
                                     double intervalSeconds,
                                     bool enabled,
                                     LineStatsColorMode colorMode)
    : output(&output), intervalSeconds(intervalSeconds), enabled(enabled), colorMode(colorMode) {
    setIntervalSeconds(intervalSeconds);
}

LineStatsReporter::LineStatsReporter(double intervalSeconds, bool enabled, LineStatsColorMode colorMode)
    : LineStatsReporter(std::cout, intervalSeconds, enabled, colorMode) {}

void LineStatsReporter::setIntervalSeconds(double intervalSeconds) {
    if (!std::isfinite(intervalSeconds) || intervalSeconds < 0.0) {
        throw std::runtime_error("LineStatsReporter intervalSeconds must be finite and >= 0.");
    }
    this->intervalSeconds = intervalSeconds;
}

void LineStatsReporter::onTrainingEvent(const TrainingEvent& event) {
    if (!enabled || output == nullptr) {
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

bool LineStatsReporter::isAnsiColorSupported(std::ostream& output) {
    return streamSupportsAnsiColor(&output);
}

bool LineStatsReporter::shouldUseColor() const {
    if (colorMode == LineStatsColorMode::NEVER) {
        return false;
    }
    if (colorMode == LineStatsColorMode::ALWAYS) {
        return true;
    }
    return streamSupportsAnsiColor(output);
}

void LineStatsReporter::writeStatsLine(const TrainingStatsSnapshot& stats) {
    if (shouldUseColor()) {
        writeColorStatsLine(stats);
    } else {
        (*output) << formatStatsLine(stats) << '\n';
    }
}

void LineStatsReporter::writeColorStatsLine(const TrainingStatsSnapshot& stats) {
    (*output) << Ansi::dim << "INFO " << Ansi::reset << Ansi::bold << "trainer:" << Ansi::reset;

    appendDimKey(*output, "phase");
    appendPhaseValue(*output, stats.phase);

    if (stats.epochs > 0) {
        appendDimKey(*output, "epoch");
        (*output) << Ansi::blue << stats.epoch << '/' << stats.epochs << Ansi::reset;
    } else if (stats.epoch > 0) {
        appendDimKey(*output, "epoch");
        (*output) << Ansi::blue << stats.epoch << Ansi::reset;
    }

    if (stats.step > 0) {
        appendDimKey(*output, "step");
        (*output) << Ansi::blue << stats.step << Ansi::reset;
    }

    if (stats.stepsPerEpoch > 0) {
        appendDimKey(*output, "batch");
        (*output) << Ansi::blue << stats.stepInEpoch << '/' << stats.stepsPerEpoch << Ansi::reset;
    } else if (stats.stepInEpoch > 0) {
        appendDimKey(*output, "batch");
        (*output) << Ansi::blue << stats.stepInEpoch << Ansi::reset;
    }

    if (stats.loss.has_value()) {
        appendDimKey(*output, "loss");
        (*output) << Ansi::yellow;
        appendFixed(*output, stats.loss.value(), 6);
        (*output) << Ansi::reset;
    }
    if (stats.accuracy.has_value()) {
        appendDimKey(*output, "accuracy");
        (*output) << Ansi::green;
        appendFixed(*output, stats.accuracy.value(), 4);
        (*output) << Ansi::reset;
    }
    if (stats.learningRate.has_value()) {
        appendDimKey(*output, "lr");
        (*output) << Ansi::magenta;
        appendScientific(*output, stats.learningRate.value(), 3);
        (*output) << Ansi::reset;
    }

    if (stats.samplesPerSecond > 0.0) {
        appendDimKey(*output, "samples/s");
        (*output) << Ansi::cyan;
        appendFixed(*output, stats.samplesPerSecond, 1);
        (*output) << Ansi::reset;
    }
    if (stats.batchesPerSecond > 0.0) {
        appendDimKey(*output, "batches/s");
        (*output) << Ansi::cyan;
        appendFixed(*output, stats.batchesPerSecond, 2);
        (*output) << Ansi::reset;
    }
    if (stats.inFlightBatches > 0) {
        appendDimKey(*output, "in_flight");
        (*output) << Ansi::blue << stats.inFlightBatches << Ansi::reset;
    }

    appendDimKey(*output, "elapsed");
    (*output) << Ansi::white << formatElapsedSeconds(stats.elapsedSeconds) << Ansi::reset << '\n';
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
