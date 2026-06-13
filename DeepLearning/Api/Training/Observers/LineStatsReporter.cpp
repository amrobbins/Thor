#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

class LineBuffer {
   public:
    const char* c_str() const { return buffer.data(); }

    void append(const char* text) {
        if (text == nullptr) {
            return;
        }
        appendBytes(text, std::strlen(text));
    }

    void append(char value) {
        if (length + 1 < buffer.size()) {
            buffer[length++] = value;
            buffer[length] = '\0';
        }
    }

    template <typename... Args>
    void appendFormat(const char* fmt, Args... args) {
        if (length >= buffer.size()) {
            return;
        }
        const int written = std::snprintf(buffer.data() + length, buffer.size() - length, fmt, args...);
        if (written <= 0) {
            return;
        }
        const size_t available = buffer.size() - length;
        const size_t advanced = std::min(static_cast<size_t>(written), available - 1);
        length += advanced;
        buffer[length] = '\0';
    }

   private:
    void appendBytes(const char* bytes, size_t count) {
        if (length >= buffer.size()) {
            return;
        }
        const size_t available = buffer.size() - length;
        const size_t copied = std::min(count, available - 1);
        if (copied > 0) {
            std::memcpy(buffer.data() + length, bytes, copied);
            length += copied;
            buffer[length] = '\0';
        }
    }

    std::array<char, 1024> buffer{};
    size_t length = 0;
};

void appendFixed(LineBuffer& out, double value, int precision) {
    out.appendFormat("%.*f", precision, value);
}

void appendScientific(LineBuffer& out, double value, int precision) {
    out.appendFormat("%.*e", precision, value);
}

void appendCompactRate(LineBuffer& out, double value) {
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
    out.append(suffix);
}

void appendElapsed(LineBuffer& out, double elapsedSeconds) {
    const uint64_t roundedSeconds = static_cast<uint64_t>(std::max(0.0, elapsedSeconds));
    const uint64_t hours = roundedSeconds / 3600;
    const uint64_t minutes = (roundedSeconds / 60) % 60;
    const uint64_t seconds = roundedSeconds % 60;
    out.appendFormat("%02llu:%02llu:%02llu",
                     static_cast<unsigned long long>(hours),
                     static_cast<unsigned long long>(minutes),
                     static_cast<unsigned long long>(seconds));
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
    return term == nullptr || std::strcmp(term, "dumb") != 0;
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

void appendPhaseValue(LineBuffer& out, TrainingPhase phase) {
    switch (phase) {
        case TrainingPhase::TRAIN:
            out.append(Ansi::dimGreen);
            break;
        case TrainingPhase::VALIDATE:
            out.append(Ansi::dimBlue);
            break;
        case TrainingPhase::TEST:
            out.append(Ansi::dimMagenta);
            break;
        case TrainingPhase::UNKNOWN:
        default:
            out.append(Ansi::brightBlack);
            break;
    }
    out.append(trainingPhaseName(phase));
    out.append(Ansi::reset);
}

void appendDimKey(LineBuffer& out, const char* key) {
    out.append(' ');
    out.append(Ansi::dim);
    out.append(key);
    out.append('=');
    out.append(Ansi::reset);
}

void appendPlainStatsLine(LineBuffer& out, const TrainingStatsSnapshot& stats) {
    out.append("INFO trainer: phase=");
    out.append(trainingPhaseName(stats.phase));

    if (stats.epochs > 0) {
        out.appendFormat(" epoch=%llu/%llu",
                         static_cast<unsigned long long>(stats.epoch),
                         static_cast<unsigned long long>(stats.epochs));
    } else if (stats.epoch > 0) {
        out.appendFormat(" epoch=%llu", static_cast<unsigned long long>(stats.epoch));
    }

    if (stats.step > 0) {
        out.appendFormat(" step=%llu", static_cast<unsigned long long>(stats.step));
    }

    if (stats.stepsPerEpoch > 0) {
        out.appendFormat(" batch=%llu/%llu",
                         static_cast<unsigned long long>(stats.stepInEpoch),
                         static_cast<unsigned long long>(stats.stepsPerEpoch));
    } else if (stats.stepInEpoch > 0) {
        out.appendFormat(" batch=%llu", static_cast<unsigned long long>(stats.stepInEpoch));
    }

    if (stats.loss.has_value()) {
        out.append(" loss=");
        appendFixed(out, stats.loss.value(), 6);
    }
    if (stats.accuracy.has_value()) {
        out.append(" accuracy=");
        appendFixed(out, stats.accuracy.value(), 4);
    }
    if (stats.learningRate.has_value()) {
        out.append(" lr=");
        appendScientific(out, stats.learningRate.value(), 3);
    }
    for (const auto& metric : stats.metrics) {
        out.append(" ");
        out.append(metric.first.c_str());
        out.append("=");
        appendFixed(out, metric.second, 6);
    }

    if (stats.samplesPerSecond > 0.0) {
        out.append(" samples/s=");
        appendFixed(out, stats.samplesPerSecond, 1);
    }
    if (stats.batchesPerSecond > 0.0) {
        out.append(" batches/s=");
        appendFixed(out, stats.batchesPerSecond, 2);
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        out.append(" flops/s=");
        appendCompactRate(out, stats.floatingPointOperationsPerSecond);
    }
    if (stats.inFlightBatches > 0) {
        out.appendFormat(" in_flight=%llu", static_cast<unsigned long long>(stats.inFlightBatches));
    }

    out.append(" elapsed=");
    appendElapsed(out, stats.elapsedSeconds);
}

void appendColorStatsLine(LineBuffer& out, const TrainingStatsSnapshot& stats) {
    out.append(Ansi::dim);
    out.append("INFO ");
    out.append(Ansi::reset);
    out.append(Ansi::bold);
    out.append("trainer:");
    out.append(Ansi::reset);

    appendDimKey(out, "phase");
    appendPhaseValue(out, stats.phase);

    if (stats.epochs > 0) {
        appendDimKey(out, "epoch");
        out.append(Ansi::blue);
        out.appendFormat("%llu/%llu", static_cast<unsigned long long>(stats.epoch), static_cast<unsigned long long>(stats.epochs));
        out.append(Ansi::reset);
    } else if (stats.epoch > 0) {
        appendDimKey(out, "epoch");
        out.append(Ansi::blue);
        out.appendFormat("%llu", static_cast<unsigned long long>(stats.epoch));
        out.append(Ansi::reset);
    }

    if (stats.step > 0) {
        appendDimKey(out, "step");
        out.append(Ansi::blue);
        out.appendFormat("%llu", static_cast<unsigned long long>(stats.step));
        out.append(Ansi::reset);
    }

    if (stats.stepsPerEpoch > 0) {
        appendDimKey(out, "batch");
        out.append(Ansi::blue);
        out.appendFormat("%llu/%llu",
                         static_cast<unsigned long long>(stats.stepInEpoch),
                         static_cast<unsigned long long>(stats.stepsPerEpoch));
        out.append(Ansi::reset);
    } else if (stats.stepInEpoch > 0) {
        appendDimKey(out, "batch");
        out.append(Ansi::blue);
        out.appendFormat("%llu", static_cast<unsigned long long>(stats.stepInEpoch));
        out.append(Ansi::reset);
    }

    if (stats.loss.has_value()) {
        appendDimKey(out, "loss");
        out.append(Ansi::yellow);
        appendFixed(out, stats.loss.value(), 6);
        out.append(Ansi::reset);
    }
    if (stats.accuracy.has_value()) {
        appendDimKey(out, "accuracy");
        out.append(Ansi::green);
        appendFixed(out, stats.accuracy.value(), 4);
        out.append(Ansi::reset);
    }
    if (stats.learningRate.has_value()) {
        appendDimKey(out, "lr");
        out.append(Ansi::magenta);
        appendScientific(out, stats.learningRate.value(), 3);
        out.append(Ansi::reset);
    }
    for (const auto& metric : stats.metrics) {
        appendDimKey(out, metric.first.c_str());
        out.append(Ansi::yellow);
        appendFixed(out, metric.second, 6);
        out.append(Ansi::reset);
    }

    if (stats.samplesPerSecond > 0.0) {
        appendDimKey(out, "samples/s");
        out.append(Ansi::cyan);
        appendFixed(out, stats.samplesPerSecond, 1);
        out.append(Ansi::reset);
    }
    if (stats.batchesPerSecond > 0.0) {
        appendDimKey(out, "batches/s");
        out.append(Ansi::cyan);
        appendFixed(out, stats.batchesPerSecond, 2);
        out.append(Ansi::reset);
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        appendDimKey(out, "flops/s");
        out.append(Ansi::boldBrightBlack);
        appendCompactRate(out, stats.floatingPointOperationsPerSecond);
        out.append(Ansi::reset);
    }
    if (stats.inFlightBatches > 0) {
        appendDimKey(out, "in_flight");
        out.append(Ansi::blue);
        out.appendFormat("%llu", static_cast<unsigned long long>(stats.inFlightBatches));
        out.append(Ansi::reset);
    }

    appendDimKey(out, "elapsed");
    out.append(Ansi::blue);
    appendElapsed(out, stats.elapsedSeconds);
    out.append(Ansi::reset);
}

}  // namespace

LineStatsReporter::LineStatsReporter(std::FILE* output,
                                     double intervalSeconds,
                                     bool enabled,
                                     LineStatsColorMode colorMode)
    : outputFile(output), intervalSeconds(intervalSeconds), enabled(enabled), colorMode(colorMode) {
    setIntervalSeconds(intervalSeconds);
}

LineStatsReporter::LineStatsReporter(double intervalSeconds,
                                     bool enabled,
                                     LineStatsColorMode colorMode,
                                     LineStatsOutputMode outputMode)
    : printer(std::make_shared<AsyncBufferedPrinter>()),
      intervalSeconds(intervalSeconds),
      enabled(enabled),
      colorMode(colorMode),
      outputMode(outputMode) {
    setIntervalSeconds(intervalSeconds);
}

LineStatsReporter::~LineStatsReporter() { close(); }

void LineStatsReporter::setIntervalSeconds(double intervalSeconds) {
    if (!std::isfinite(intervalSeconds) || intervalSeconds < 0.0) {
        throw std::runtime_error("LineStatsReporter intervalSeconds must be finite and >= 0.");
    }
    this->intervalSeconds = intervalSeconds;
}

void LineStatsReporter::onTrainingEvent(const TrainingEvent& event) {
    if (!enabled || (outputFile == nullptr && printer == nullptr)) {
        return;
    }

    if (event.type == TrainingEventType::EPOCH_STARTED) {
        beginPhase(event.stats);
        return;
    }

    if (event.type == TrainingEventType::EPOCH_FINISHED) {
        finishPhase(event.stats);
        return;
    }

    if (event.type != TrainingEventType::STATS) {
        return;
    }

    if (!samePhaseOccurrence(event.stats)) {
        beginPhase(event.stats);
    }
    lastSeenStats = event.stats;

    if (!shouldPrintStats(event.stats)) {
        return;
    }

    writeStatsLine(event.stats);
}

bool LineStatsReporter::shouldPrintStats(const TrainingStatsSnapshot& stats) {
    if (!printedAnyStats) {
        printedAnyStats = true;
        printedStatsForActivePhase = true;
        lastPrintedElapsedSeconds = stats.elapsedSeconds;
        return true;
    }

    if (!printedStatsForActivePhase) {
        printedStatsForActivePhase = true;
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

void LineStatsReporter::beginPhase(const TrainingStatsSnapshot& stats) {
    hasActivePhase = true;
    activePhase = stats.phase;
    activeEpoch = stats.epoch;
    printedStatsForActivePhase = false;
    lastSeenStats.reset();
}

void LineStatsReporter::finishPhase(const TrainingStatsSnapshot& stats) {
    if (!lastSeenStats.has_value()) {
        return;
    }
    if (!samePhaseOccurrence(stats) || !samePhaseOccurrence(lastSeenStats.value())) {
        return;
    }
    if (lastPrintedStats.has_value() && sameStatsIdentity(lastSeenStats.value(), lastPrintedStats.value())) {
        return;
    }

    printedAnyStats = true;
    printedStatsForActivePhase = true;
    lastPrintedElapsedSeconds = lastSeenStats->elapsedSeconds;
    writeStatsLine(lastSeenStats.value());
}

bool LineStatsReporter::samePhaseOccurrence(const TrainingStatsSnapshot& stats) const {
    return hasActivePhase && stats.phase == activePhase && stats.epoch == activeEpoch;
}

bool LineStatsReporter::sameStatsIdentity(const TrainingStatsSnapshot& lhs, const TrainingStatsSnapshot& rhs) const {
    return lhs.phase == rhs.phase && lhs.epoch == rhs.epoch && lhs.step == rhs.step &&
           lhs.stepInEpoch == rhs.stepInEpoch && lhs.stepsPerEpoch == rhs.stepsPerEpoch;
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
    return fileSupportsAnsiColor(outputFile != nullptr ? outputFile : stdout);
}

void LineStatsReporter::writeStatsLine(const TrainingStatsSnapshot& stats) {
    LineBuffer line;
    if (shouldUseColor()) {
        appendColorStatsLine(line, stats);
    } else {
        appendPlainStatsLine(line, stats);
    }
    emitLine(line.c_str());
    lastPrintedStats = stats;
}

void LineStatsReporter::emitLine(const char* line) {
    if (line == nullptr) {
        return;
    }
    if (printer != nullptr) {
        const AsyncBufferedPrinterDestination destination =
            (outputMode == LineStatsOutputMode::STDOUT_AND_STDERR) ? AsyncBufferedPrinterDestination::STDOUT_AND_STDERR
                                                                    : AsyncBufferedPrinterDestination::STDOUT;
        printer->writeLine(line, destination);
        return;
    }
    if (outputFile != nullptr) {
        std::fprintf(outputFile, "%s\n", line);
    }
}

void LineStatsReporter::flush() {
    if (printer != nullptr) {
        printer->flush();
    } else if (outputFile != nullptr) {
        std::fflush(outputFile);
    }
}

void LineStatsReporter::close() {
    if (printer != nullptr) {
        printer->close();
        printer.reset();
    } else if (outputFile != nullptr) {
        std::fflush(outputFile);
    }
}

std::string LineStatsReporter::formatStatsLine(const TrainingStatsSnapshot& stats) {
    LineBuffer line;
    appendPlainStatsLine(line, stats);
    return std::string(line.c_str());
}

std::string LineStatsReporter::formatElapsedSeconds(double elapsedSeconds) {
    LineBuffer line;
    appendElapsed(line, elapsedSeconds);
    return std::string(line.c_str());
}

}  // namespace Thor
