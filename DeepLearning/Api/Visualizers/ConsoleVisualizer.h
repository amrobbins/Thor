#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <boost/algorithm/string.hpp>

#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <csignal>

namespace Thor {

class Visualizer;

struct ProgressRow {
    ExampleType executionMode;
    uint64_t epochNum;
    std::string label;
    uint64_t curBatch;
    uint64_t numBatches;
    double batchLoss;
    double accuracy;

    ProgressRow(ExampleType executionMode, uint64_t epochNum, uint64_t curBatch, uint64_t numBatches, double batchLoss, double accuracy) {
        this->executionMode = executionMode;
        this->epochNum = epochNum;
        label = executionMode == ExampleType::TRAIN ? "Train " : "Validate ";
        label += std::to_string(epochNum);
        this->curBatch = curBatch;
        this->numBatches = numBatches;
        this->batchLoss = batchLoss;
        this->accuracy = accuracy;
    }
};

class ConsoleVisualizer : public Visualizer {
   public:
    static ConsoleVisualizer &instance() {
        static ConsoleVisualizer singletonInstance;  // Guaranteed to be destroyed. Instantiated on first use.
        return singletonInstance;
    }

    // Forbid copying since ConsoleVisualizer is a singleton
    ConsoleVisualizer(const ConsoleVisualizer &) = delete;
    ConsoleVisualizer &operator=(const ConsoleVisualizer &) = delete;

    virtual ~ConsoleVisualizer();

    virtual void startUI();
    virtual void stopUI();

    static void cursorForward(int32_t spaces) { printf("\033[%dC", spaces); }
    static void cursorBackward(int32_t spaces) { printf("\033[%dD", spaces); }

   private:
    ConsoleVisualizer();

    static const int MIN_WIDTH;
    static const int HEIGHT_W0;
    static const int MIN_HEIGHT_W1;
    static const int HEIGHT_W2;

    static int terminalRows;
    static int terminalCols;
    static int windowWidth;
    static int heightW0;
    static int heightW1;
    static int heightW2;

    static std::shared_ptr<std::thread> uiThread;
    static std::recursive_mutex mtx;
    static bool uiRunning;

    static bool scrollVisible;
    static int scrollTop;
    static int scrollBottom;
    static int scrollLeft;
    static int scrollRight;
    static int scrollTopElement;
    static bool scrollBarSelected;
    static int scrollClickFromTopOffset;
    static int scrollBarDesiredTop;

    static int thorDashLeft;
    static int thorDashRight;
    static int thorDashY;
    static std::string thorDashUrl;

    static ExecutionState mostRecentExecutionState;
    static std::vector<ProgressRow> rows;
    static std::chrono::high_resolution_clock::time_point start;
    static double totalEpochLoss;
    static double totalEpochAccuracy;

    static std::string cudaDevicesString;

    // Optional<ExecutionState> previousExecutionState;

    static void *win0;
    static void *win1;
    static void *win2;

    static void initializeWindows();
    static void createWindows();
    static void deleteWindows();

    static void resizeHandler(int sig);
    static void (*originalResizeHandler)(int);

    static void interruptHandler(int sig);
    static void (*originalInterruptHandler)(int);

    static void abortHandler(int sig);
    static void (*originalAbortHandler)(int);

    static void noOpHandler(int sig);

    void inputHandler();

    static void redrawWindows();

    static void drawHeader();
    static void drawProgressRows();
    static void drawFooter();
    static void drawOverallStatusBar();

    static void display();

    static int openUrl(std::string URL);

    static void drawStatusBar(
        void *win, int y, int xStart, int xEnd, double progress, std::string leftLabel, std::string rightLabel, bool boldLabels = false);

    static std::string popUpPrompt(std::string message);
    static void popUpAcknowledge(std::string message);

    static void drawBox(void *win, int top, int bottom, int left, int right);
    static void drawBlock(void *win, int top, int bottom, int left, int right);

    static void updateLog();
    static void dumpSummaryToTerminal();
};

}  // namespace Thor
