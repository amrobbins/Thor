#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace Thor {
namespace {

namespace Ansi {
constexpr const char* reset = "\x1b[0m";
constexpr const char* bold = "\x1b[1m";
constexpr const char* key = "\x1b[38;5;235m";
constexpr const char* label = "\x1b[38;5;235m";
constexpr const char* phaseTrain = "\x1b[1;38;5;28m";
constexpr const char* phaseValidate = "\x1b[1;38;5;18m";
constexpr const char* phaseTest = "\x1b[1;38;5;53m";
constexpr const char* phaseUnknown = "\x1b[1;38;5;235m";
constexpr const char* progress = "\x1b[38;5;21m";
constexpr const char* loss = "\x1b[1;38;5;0m";
constexpr const char* accuracy = "\x1b[38;5;22m";
constexpr const char* learningRate = "\x1b[38;5;53m";
constexpr const char* throughput = "\x1b[1;38;2;140;84;0m";  // "\x1b[38;5;202m";
constexpr const char* elapsed = "\x1b[38;5;0m";
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

[[maybe_unused]] void appendFixed(LineBuffer& out, double value, int precision) { out.appendFormat("%.*f", precision, value); }

[[maybe_unused]] void appendScientific(LineBuffer& out, double value, int precision) { out.appendFormat("%.*e", precision, value); }

[[maybe_unused]] void appendCompactRate(LineBuffer& out, double value) {
    const double absValue = std::abs(value);
    const char* suffix = "";
    double scaled = value;
    if (absValue >= 1.0e24) {
        scaled = value / 1.0e24;
        suffix = "Y";
    } else if (absValue >= 1.0e21) {
        scaled = value / 1.0e21;
        suffix = "Z";
    } else if (absValue >= 1.0e18) {
        scaled = value / 1.0e18;
        suffix = "E";
    } else if (absValue >= 1.0e15) {
        scaled = value / 1.0e15;
        suffix = "P";
    } else if (absValue >= 1.0e12) {
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

std::string formatFixedString(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return std::string(buffer);
}

std::string formatScientificString(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*e", precision, value);
    return std::string(buffer);
}

std::string formatCompactRateString(double value, bool integral = false) {
    const double absValue = std::abs(value);
    const char* suffix = "";
    double scaled = value;
    if (absValue >= 1.0e24) {
        scaled = value / 1.0e24;
        suffix = "Y";
    } else if (absValue >= 1.0e21) {
        scaled = value / 1.0e21;
        suffix = "Z";
    } else if (absValue >= 1.0e18) {
        scaled = value / 1.0e18;
        suffix = "E";
    } else if (absValue >= 1.0e15) {
        scaled = value / 1.0e15;
        suffix = "P";
    } else if (absValue >= 1.0e12) {
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
    const int precision = integral ? 0 : (absScaled >= 100.0 ? 0 : (absScaled >= 10.0 ? 1 : 2));
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f%s", precision, scaled, suffix);
    return std::string(buffer);
}

std::string formatCompactFlopsRateString(double value) {
    static constexpr const char* suffixes[] = {"", "K", "M", "G", "T", "P", "E", "Z", "Y"};
    constexpr int maxSuffixIndex = static_cast<int>(sizeof(suffixes) / sizeof(suffixes[0])) - 1;

    double scaled = value;
    int suffixIndex = 0;
    while (std::abs(scaled) >= 1000.0 && suffixIndex < maxSuffixIndex) {
        scaled /= 1000.0;
        ++suffixIndex;
    }

    while (true) {
        const double absScaled = std::abs(scaled);
        const int precision = absScaled >= 100.0 ? 1 : (absScaled >= 10.0 ? 2 : 3);
        char number[64];
        std::snprintf(number, sizeof(number), "%.*f", precision, scaled);
        if (std::char_traits<char>::length(number) <= 5 || suffixIndex >= maxSuffixIndex) {
            return std::string(number) + suffixes[suffixIndex];
        }

        scaled /= 1000.0;
        ++suffixIndex;
    }
}

std::string formatElapsedString(double elapsedSeconds) {
    const uint64_t roundedSeconds = static_cast<uint64_t>(std::max(0.0, elapsedSeconds));
    const uint64_t hours = roundedSeconds / 3600;
    const uint64_t minutes = (roundedSeconds / 60) % 60;
    const uint64_t seconds = roundedSeconds % 60;
    char buffer[32];
    if (hours >= 100) {
        std::snprintf(
            buffer, sizeof(buffer), "%llu:%02llu", static_cast<unsigned long long>(hours), static_cast<unsigned long long>(minutes));
    } else {
        std::snprintf(buffer,
                      sizeof(buffer),
                      "%02llu:%02llu:%02llu",
                      static_cast<unsigned long long>(hours),
                      static_cast<unsigned long long>(minutes),
                      static_cast<unsigned long long>(seconds));
    }
    return std::string(buffer);
}

enum class PadAlignment { LEFT, RIGHT };

void appendPadded(LineBuffer& out, const std::string& value, size_t width, PadAlignment alignment = PadAlignment::RIGHT) {
    if (value.size() < width && alignment == PadAlignment::RIGHT) {
        for (size_t i = value.size(); i < width; ++i) {
            out.append(' ');
        }
    }
    out.append(value.c_str());
    if (value.size() < width && alignment == PadAlignment::LEFT) {
        for (size_t i = value.size(); i < width; ++i) {
            out.append(' ');
        }
    }
}

void appendStyledPadded(
    LineBuffer& out, const char* style, const std::string& value, size_t width, PadAlignment alignment = PadAlignment::RIGHT) {
    out.append(style);
    appendPadded(out, value, width, alignment);
    out.append(Ansi::reset);
}

std::string formatUnsigned(uint64_t value) { return std::to_string(value); }

std::string formatRatio(uint64_t numerator, uint64_t denominator) { return std::to_string(numerator) + "/" + std::to_string(denominator); }

void appendElapsed(LineBuffer& out, double elapsedSeconds) { out.append(formatElapsedString(elapsedSeconds).c_str()); }

bool envFlagIsSet(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

bool envFlagIsZero(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '0' && value[1] == '\0';
}

bool terminalEnvDisablesColor() { return envFlagIsSet("NO_COLOR"); }

bool terminalEnvForcesColor() {
    return (envFlagIsSet("CLICOLOR_FORCE") && !envFlagIsZero("CLICOLOR_FORCE")) ||
           (envFlagIsSet("FORCE_COLOR") && !envFlagIsZero("FORCE_COLOR"));
}

bool terminalKindAllowsColor() {
    const char* term = std::getenv("TERM");
    return term == nullptr || std::strcmp(term, "dumb") != 0;
}

bool fileDescriptorIsTerminal(int fd) {
#if defined(__unix__) || defined(__APPLE__)
    return fd >= 0 && ::isatty(fd) != 0;
#else
    (void)fd;
    return false;
#endif
}

bool fileDescriptorSupportsAnsiColor(int fd) {
    if (terminalEnvDisablesColor()) {
        return false;
    }
    if (terminalEnvForcesColor()) {
        return true;
    }
    if (!terminalKindAllowsColor()) {
        return false;
    }
    return fileDescriptorIsTerminal(fd);
}

bool fileSupportsAnsiColor(std::FILE* output) {
#if defined(__unix__) || defined(__APPLE__)
    return output != nullptr && fileDescriptorSupportsAnsiColor(::fileno(output));
#else
    (void)output;
    return false;
#endif
}

bool shouldUseColorForOutput(std::FILE* output, LineStatsColorMode colorMode) {
    if (colorMode == LineStatsColorMode::NEVER) {
        return false;
    }
    if (colorMode == LineStatsColorMode::ALWAYS) {
        return true;
    }
#if defined(__unix__) || defined(__APPLE__)
    if (output != nullptr) {
        return fileSupportsAnsiColor(output);
    }
    return fileDescriptorSupportsAnsiColor(STDOUT_FILENO);
#else
    return fileSupportsAnsiColor(output != nullptr ? output : stdout);
#endif
}

const char* phaseStyle(TrainingEventPhase phase) {
    switch (phase) {
        case TrainingEventPhase::TRAIN:
            return Ansi::phaseTrain;
        case TrainingEventPhase::VALIDATE:
            return Ansi::phaseValidate;
        case TrainingEventPhase::TEST:
            return Ansi::phaseTest;
        case TrainingEventPhase::UNKNOWN:
        default:
            return Ansi::phaseUnknown;
    }
}

void appendPhaseValue(LineBuffer& out, TrainingEventPhase phase) {
    appendStyledPadded(out, phaseStyle(phase), trainingPhaseName(phase), 8, PadAlignment::LEFT);
}

void appendDimKey(LineBuffer& out, const char* key) {
    out.append(' ');
    out.append(Ansi::key);
    out.append(key);
    out.append('=');
    out.append(Ansi::reset);
}

void appendPlainDimKey(LineBuffer& out, const char* key) {
    out.append(' ');
    out.append(key);
    out.append('=');
}

bool shouldReportDeviceDatasetStorage(const DeviceDatasetStorageReport& report) {
    return report.attempted || report.used || !report.reason.empty() || report.examples > 0 || report.requiredBytes > 0 ||
           report.availableBytesAfterPlacement > 0 || report.residentBytes > 0 || report.residentCacheHit ||
           report.residentConstructionJoined || report.residentConstructionStarted || report.materializationSeconds > 0.0;
}

void appendPlainDeviceDatasetStorage(LineBuffer& out, const DeviceDatasetStorageReport& report) {
    if (!shouldReportDeviceDatasetStorage(report)) {
        return;
    }
    appendPlainDimKey(out, "device_dataset_storage");
    out.append("requested=");
    out.append(deviceDatasetStorageName(report.requested));
    out.append(",used=");
    out.append(report.used ? "true" : "false");
    if (!report.reason.empty()) {
        out.append(",reason=");
        out.append(report.reason.c_str());
    }
    if (report.examples > 0) {
        out.append(",examples=");
        out.append(formatUnsigned(report.examples).c_str());
    }
    if (report.requiredBytes > 0) {
        out.append(",required_bytes=");
        out.append(formatUnsigned(report.requiredBytes).c_str());
    }
    if (report.availableBytesAfterPlacement > 0) {
        out.append(",available_bytes_after_model_placement=");
        out.append(formatUnsigned(report.availableBytesAfterPlacement).c_str());
    }
    if (report.residentBytes > 0) {
        out.append(",resident_bytes=");
        out.append(formatUnsigned(report.residentBytes).c_str());
    }
    if (report.residentCacheHit) {
        out.append(",residency=cache_hit");
    } else if (report.residentConstructionJoined) {
        out.append(",residency=joined_construction");
    } else if (report.residentConstructionStarted) {
        out.append(",residency=constructed");
    }
    if (report.materializationSeconds > 0.0) {
        out.append(",materialization_s=");
        out.append(formatFixedString(report.materializationSeconds, 3).c_str());
    }
}

void appendColorDeviceDatasetStorage(LineBuffer& out, const DeviceDatasetStorageReport& report) {
    if (!shouldReportDeviceDatasetStorage(report)) {
        return;
    }
    appendDimKey(out, "device_dataset_storage");
    out.append(Ansi::throughput);
    out.append("requested=");
    out.append(deviceDatasetStorageName(report.requested));
    out.append(",used=");
    out.append(report.used ? "true" : "false");
    if (!report.reason.empty()) {
        out.append(",reason=");
        out.append(report.reason.c_str());
    }
    if (report.examples > 0) {
        out.append(",examples=");
        out.append(formatUnsigned(report.examples).c_str());
    }
    if (report.requiredBytes > 0) {
        out.append(",required_bytes=");
        out.append(formatUnsigned(report.requiredBytes).c_str());
    }
    if (report.availableBytesAfterPlacement > 0) {
        out.append(",available_bytes_after_model_placement=");
        out.append(formatUnsigned(report.availableBytesAfterPlacement).c_str());
    }
    if (report.residentBytes > 0) {
        out.append(",resident_bytes=");
        out.append(formatUnsigned(report.residentBytes).c_str());
    }
    if (report.residentCacheHit) {
        out.append(",residency=cache_hit");
    } else if (report.residentConstructionJoined) {
        out.append(",residency=joined_construction");
    } else if (report.residentConstructionStarted) {
        out.append(",residency=constructed");
    }
    if (report.materializationSeconds > 0.0) {
        out.append(",materialization_s=");
        out.append(formatFixedString(report.materializationSeconds, 3).c_str());
    }
    out.append(Ansi::reset);
}

constexpr size_t RATE_FIELD_WIDTH = 5;
constexpr size_t FLOPS_RATE_FIELD_WIDTH = 6;

void appendPlainStatsLine(LineBuffer& out,
                          const TrainingStatsSnapshot& stats,
                          std::string_view runName = {},
                          size_t runPrefixWidth = 0) {
    if (runName.empty()) {
        out.append("INFO trainer:");
    } else {
        std::string prefix = "INFO runs[" + std::string(runName) + "]:";
        out.append(prefix.c_str());
        if (prefix.size() < runPrefixWidth) {
            for (size_t i = prefix.size(); i < runPrefixWidth; ++i) {
                out.append(' ');
            }
        }
    }

    appendPlainDimKey(out, "phase");
    appendPadded(out, trainingPhaseName(stats.phase), 8, PadAlignment::LEFT);

    if (stats.epochs > 0) {
        appendPlainDimKey(out, "epoch");
        appendPadded(out, formatRatio(stats.epoch, stats.epochs), 9);
    } else if (stats.epoch > 0) {
        appendPlainDimKey(out, "epoch");
        appendPadded(out, formatUnsigned(stats.epoch), 9);
    }

    if (stats.stepsPerEpoch > 0) {
        appendPlainDimKey(out, "batch");
        appendPadded(out, formatRatio(stats.stepInEpoch, stats.stepsPerEpoch), 13);
    } else if (stats.stepInEpoch > 0) {
        appendPlainDimKey(out, "batch");
        appendPadded(out, formatUnsigned(stats.stepInEpoch), 13);
    }

    if (stats.step > 0) {
        appendPlainDimKey(out, "step");
        appendPadded(out, formatUnsigned(stats.step), 10);
    }

    if (stats.loss.has_value()) {
        appendPlainDimKey(out, "loss");
        appendPadded(out, formatFixedString(stats.loss.value(), 6), 9);
    }
    if (stats.accuracy.has_value()) {
        appendPlainDimKey(out, "accuracy");
        appendPadded(out, formatFixedString(stats.accuracy.value(), 4), 6);
    }
    if (stats.learningRate.has_value()) {
        appendPlainDimKey(out, "lr");
        appendPadded(out, formatScientificString(stats.learningRate.value(), 3), 9);
    }
    for (const auto& metric : stats.metrics) {
        appendPlainDimKey(out, metric.first.c_str());
        appendPadded(out, formatFixedString(metric.second, 6), 9);
    }

    if (stats.samplesPerSecond > 0.0) {
        appendPlainDimKey(out, "samples/s");
        appendPadded(out, formatCompactRateString(stats.samplesPerSecond), RATE_FIELD_WIDTH);
    }
    if (stats.batchesPerSecond > 0.0) {
        appendPlainDimKey(out, "batches/s");
        appendPadded(out, formatCompactRateString(stats.batchesPerSecond), RATE_FIELD_WIDTH);
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        appendPlainDimKey(out, "flops/s");
        appendPadded(out, formatCompactFlopsRateString(stats.floatingPointOperationsPerSecond), FLOPS_RATE_FIELD_WIDTH);
    }
    if (stats.inFlightBatches > 0) {
        appendPlainDimKey(out, "in_flight");
        appendPadded(out, formatCompactRateString(stats.inFlightBatches, true), RATE_FIELD_WIDTH);
    } else {
        out.appendFormat("%16s", "");
    }

    appendPlainDimKey(out, "elapsed");
    appendPadded(out, formatElapsedString(stats.elapsedSeconds), 9);
    appendPlainDeviceDatasetStorage(out, stats.deviceDatasetStorage);
}

void appendColorStatsLine(LineBuffer& out,
                          const TrainingStatsSnapshot& stats,
                          std::string_view runName = {},
                          size_t runPrefixWidth = 0) {
    out.append(Ansi::label);
    out.append("INFO ");
    out.append(Ansi::reset);
    out.append(Ansi::bold);
    if (runName.empty()) {
        out.append("trainer:");
    } else {
        std::string prefix = "runs[" + std::string(runName) + "]:";
        out.append(prefix.c_str());
        const size_t visiblePrefixWidth = runPrefixWidth > std::string("INFO ").size() ? runPrefixWidth - std::string("INFO ").size() : 0;
        if (prefix.size() < visiblePrefixWidth) {
            for (size_t i = prefix.size(); i < visiblePrefixWidth; ++i) {
                out.append(' ');
            }
        }
    }
    out.append(Ansi::reset);

    appendDimKey(out, "phase");
    appendPhaseValue(out, stats.phase);

    if (stats.epochs > 0) {
        appendDimKey(out, "epoch");
        appendStyledPadded(out, Ansi::progress, formatRatio(stats.epoch, stats.epochs), 9);
    } else if (stats.epoch > 0) {
        appendDimKey(out, "epoch");
        appendStyledPadded(out, Ansi::progress, formatUnsigned(stats.epoch), 9);
    }

    if (stats.stepsPerEpoch > 0) {
        appendDimKey(out, "batch");
        appendStyledPadded(out, Ansi::progress, formatRatio(stats.stepInEpoch, stats.stepsPerEpoch), 13);
    } else if (stats.stepInEpoch > 0) {
        appendDimKey(out, "batch");
        appendStyledPadded(out, Ansi::progress, formatUnsigned(stats.stepInEpoch), 13);
    }

    if (stats.step > 0) {
        appendDimKey(out, "step");
        appendStyledPadded(out, Ansi::progress, formatUnsigned(stats.step), 10);
    }

    if (stats.loss.has_value()) {
        appendDimKey(out, "loss");
        appendStyledPadded(out, Ansi::loss, formatFixedString(stats.loss.value(), 6), 9);
    }
    if (stats.accuracy.has_value()) {
        appendDimKey(out, "accuracy");
        appendStyledPadded(out, Ansi::accuracy, formatFixedString(stats.accuracy.value(), 4), 6);
    }
    if (stats.learningRate.has_value()) {
        appendDimKey(out, "lr");
        appendStyledPadded(out, Ansi::learningRate, formatScientificString(stats.learningRate.value(), 3), 9);
    }
    for (const auto& metric : stats.metrics) {
        appendDimKey(out, metric.first.c_str());
        appendStyledPadded(out, Ansi::loss, formatFixedString(metric.second, 6), 9);
    }

    if (stats.samplesPerSecond > 0.0) {
        appendDimKey(out, "samples/s");
        appendStyledPadded(out, Ansi::throughput, formatCompactRateString(stats.samplesPerSecond), RATE_FIELD_WIDTH);
    }
    if (stats.batchesPerSecond > 0.0) {
        appendDimKey(out, "batches/s");
        appendStyledPadded(out, Ansi::throughput, formatCompactRateString(stats.batchesPerSecond), RATE_FIELD_WIDTH);
    }
    if (stats.floatingPointOperationsPerSecond > 0.0) {
        appendDimKey(out, "flops/s");
        appendStyledPadded(out, Ansi::throughput, formatCompactFlopsRateString(stats.floatingPointOperationsPerSecond), FLOPS_RATE_FIELD_WIDTH);
    }
    if (stats.inFlightBatches > 0) {
        appendDimKey(out, "in_flight");
        appendStyledPadded(out, Ansi::throughput, formatCompactRateString(stats.inFlightBatches, true), RATE_FIELD_WIDTH);
    } else {
        out.appendFormat("%16s", "");
    }

    appendDimKey(out, "elapsed");
    appendStyledPadded(out, Ansi::elapsed, formatElapsedString(stats.elapsedSeconds), 9);
    appendColorDeviceDatasetStorage(out, stats.deviceDatasetStorage);
}

}  // namespace

LineStatsReporter::LineStatsReporter(std::FILE* output, double intervalSeconds, bool enabled, LineStatsColorMode colorMode)
    : outputFile(output), intervalSeconds(intervalSeconds), enabled(enabled), colorMode(colorMode) {
    setIntervalSeconds(intervalSeconds);
}

LineStatsReporter::LineStatsReporter(double intervalSeconds, bool enabled, LineStatsColorMode colorMode, LineStatsOutputMode outputMode)
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
    onStatsEvent(TrainingStatsEvent::fromTrainingEvent(event));
}

void LineStatsReporter::onStatsEvent(const TrainingStatsEvent& event) {
    if (!enabled || (outputFile == nullptr && printer == nullptr)) {
        return;
    }

    if (event.type == TrainingEventType::EPOCH_STARTED) {
        beginPhase(event.stats);
        return;
    }

    if (event.type == TrainingEventType::EPOCH_FINISHED) {
        finishPhase(event.stats, event.runName);
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

    writeStatsLine(event.stats, event.runName);
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

void LineStatsReporter::finishPhase(const TrainingStatsSnapshot& stats, std::string_view runName) {
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
    writeStatsLine(lastSeenStats.value(), runName);
}

bool LineStatsReporter::samePhaseOccurrence(const TrainingStatsSnapshot& stats) const {
    return hasActivePhase && stats.phase == activePhase && stats.epoch == activeEpoch;
}

bool LineStatsReporter::sameStatsIdentity(const TrainingStatsSnapshot& lhs, const TrainingStatsSnapshot& rhs) const {
    return lhs.phase == rhs.phase && lhs.epoch == rhs.epoch && lhs.step == rhs.step && lhs.stepInEpoch == rhs.stepInEpoch &&
           lhs.stepsPerEpoch == rhs.stepsPerEpoch;
}

bool LineStatsReporter::isAnsiColorSupported(std::FILE* output) { return fileSupportsAnsiColor(output); }

bool LineStatsReporter::shouldUseColor() const { return shouldUseColorForOutput(outputFile, colorMode); }

void LineStatsReporter::writeStatsLine(const TrainingStatsSnapshot& stats, std::string_view runName) {
    const std::string line = formatStatsLine(stats, runName, 0, colorMode, outputFile);
    emitLine(line.c_str());
    lastPrintedStats = stats;
}

void LineStatsReporter::emitLine(const char* line) {
    if (line == nullptr) {
        return;
    }
    if (printer != nullptr) {
        const AsyncBufferedPrinterDestination destination = (outputMode == LineStatsOutputMode::STDOUT_AND_STDERR)
                                                                ? AsyncBufferedPrinterDestination::STDOUT_AND_STDERR
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
    return formatStatsLine(stats, std::string_view{});
}

std::string LineStatsReporter::formatStatsLine(const TrainingStatsSnapshot& stats, std::string_view runName) {
    return formatStatsLine(stats, runName, 0);
}

std::string LineStatsReporter::formatStatsLine(const TrainingStatsSnapshot& stats, std::string_view runName, size_t runPrefixWidth) {
    return formatStatsLine(stats, runName, runPrefixWidth, LineStatsColorMode::NEVER, stdout);
}

std::string LineStatsReporter::formatStatsLine(const TrainingStatsSnapshot& stats,
                                               std::string_view runName,
                                               size_t runPrefixWidth,
                                               LineStatsColorMode colorMode,
                                               std::FILE* output) {
    LineBuffer line;
    if (shouldUseColorForOutput(output, colorMode)) {
        appendColorStatsLine(line, stats, runName, runPrefixWidth);
    } else {
        appendPlainStatsLine(line, stats, runName, runPrefixWidth);
    }
    return std::string(line.c_str());
}

std::string LineStatsReporter::formatElapsedSeconds(double elapsedSeconds) {
    LineBuffer line;
    appendElapsed(line, elapsedSeconds);
    return std::string(line.c_str());
}

}  // namespace Thor
