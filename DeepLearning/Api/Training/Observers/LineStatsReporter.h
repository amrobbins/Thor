#pragma once

#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

#include <iosfwd>
#include <optional>
#include <string>

namespace Thor {

enum class LineStatsColorMode { AUTO, NEVER, ALWAYS };

class LineStatsReporter : public TrainingObserver {
   public:
    explicit LineStatsReporter(std::ostream& output,
                               double intervalSeconds = 10.0,
                               bool enabled = true,
                               LineStatsColorMode colorMode = LineStatsColorMode::AUTO);
    explicit LineStatsReporter(double intervalSeconds = 10.0,
                               bool enabled = true,
                               LineStatsColorMode colorMode = LineStatsColorMode::AUTO);

    void onTrainingEvent(const TrainingEvent& event) override;

    void setEnabled(bool enabled) { this->enabled = enabled; }
    [[nodiscard]] bool isEnabled() const { return enabled; }

    void setIntervalSeconds(double intervalSeconds);
    [[nodiscard]] double getIntervalSeconds() const { return intervalSeconds; }

    void setColorMode(LineStatsColorMode colorMode) { this->colorMode = colorMode; }
    [[nodiscard]] LineStatsColorMode getColorMode() const { return colorMode; }

    [[nodiscard]] static bool isAnsiColorSupported(std::ostream& output);
    [[nodiscard]] static std::string formatStatsLine(const TrainingStatsSnapshot& stats);
    [[nodiscard]] static std::string formatElapsedSeconds(double elapsedSeconds);

   private:
    bool shouldPrintStats(const TrainingStatsSnapshot& stats);
    [[nodiscard]] bool shouldUseColor() const;
    void writeStatsLine(const TrainingStatsSnapshot& stats);
    void writeColorStatsLine(const TrainingStatsSnapshot& stats);

    std::ostream* output;
    double intervalSeconds;
    bool enabled;
    LineStatsColorMode colorMode;
    bool printedAnyStats = false;
    std::optional<double> lastPrintedElapsedSeconds{};
};

}  // namespace Thor
