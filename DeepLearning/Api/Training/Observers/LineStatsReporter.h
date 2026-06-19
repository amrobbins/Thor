#pragma once

#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"
#include "DeepLearning/Api/Training/Observers/TrainingStatsSink.h"
#include "Utilities/Common/AsyncBufferedPrinter.h"

#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace Thor {

enum class LineStatsColorMode { AUTO, NEVER, ALWAYS };
enum class LineStatsOutputMode { STDOUT, STDOUT_AND_STDERR };

class LineStatsReporter : public TrainingObserver, public TrainingStatsSink {
   public:
    explicit LineStatsReporter(std::FILE* output,
                               double intervalSeconds = 10.0,
                               bool enabled = true,
                               LineStatsColorMode colorMode = LineStatsColorMode::AUTO);
    explicit LineStatsReporter(double intervalSeconds = 10.0,
                               bool enabled = true,
                               LineStatsColorMode colorMode = LineStatsColorMode::AUTO,
                               LineStatsOutputMode outputMode = LineStatsOutputMode::STDOUT);
    ~LineStatsReporter() override;

    void onTrainingEvent(const TrainingEvent& event) override;
    void onStatsEvent(const TrainingStatsEvent& event) override;
    void flush() override;
    void close() override;

    void setEnabled(bool enabled) { this->enabled = enabled; }
    [[nodiscard]] bool isEnabled() const { return enabled; }

    void setIntervalSeconds(double intervalSeconds);
    [[nodiscard]] double getIntervalSeconds() const { return intervalSeconds; }

    void setColorMode(LineStatsColorMode colorMode) { this->colorMode = colorMode; }
    [[nodiscard]] LineStatsColorMode getColorMode() const { return colorMode; }

    void setOutputMode(LineStatsOutputMode outputMode) { this->outputMode = outputMode; }
    [[nodiscard]] LineStatsOutputMode getOutputMode() const { return outputMode; }

    [[nodiscard]] static bool isAnsiColorSupported(std::FILE* output);
    [[nodiscard]] static std::string formatStatsLine(const TrainingStatsSnapshot& stats);
    [[nodiscard]] static std::string formatStatsLine(const TrainingStatsSnapshot& stats, std::string_view runName);
    [[nodiscard]] static std::string formatElapsedSeconds(double elapsedSeconds);

   private:
    bool shouldPrintStats(const TrainingStatsSnapshot& stats);
    void beginPhase(const TrainingStatsSnapshot& stats);
    void finishPhase(const TrainingStatsSnapshot& stats, std::string_view runName = {});
    [[nodiscard]] bool samePhaseOccurrence(const TrainingStatsSnapshot& stats) const;
    [[nodiscard]] bool sameStatsIdentity(const TrainingStatsSnapshot& lhs, const TrainingStatsSnapshot& rhs) const;
    [[nodiscard]] bool shouldUseColor() const;
    void writeStatsLine(const TrainingStatsSnapshot& stats, std::string_view runName = {});
    void emitLine(const char* line);

    std::FILE* outputFile = nullptr;
    std::shared_ptr<AsyncBufferedPrinter> printer = nullptr;
    double intervalSeconds;
    bool enabled;
    LineStatsColorMode colorMode;
    LineStatsOutputMode outputMode = LineStatsOutputMode::STDOUT;
    bool printedAnyStats = false;
    std::optional<double> lastPrintedElapsedSeconds{};
    bool hasActivePhase = false;
    TrainingEventPhase activePhase = TrainingEventPhase::UNKNOWN;
    uint64_t activeEpoch = 0;
    bool printedStatsForActivePhase = false;
    std::optional<TrainingStatsSnapshot> lastSeenStats{};
    std::optional<TrainingStatsSnapshot> lastPrintedStats{};
};

}  // namespace Thor
