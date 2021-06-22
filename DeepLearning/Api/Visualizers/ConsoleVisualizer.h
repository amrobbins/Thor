#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"

#include <boost/algorithm/string.hpp>

#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <csignal>

namespace Thor {

class Visualizer;

struct ProgressRow {
    string label;
    int curBatch;
    int numBatches;
    double batchLoss;
    double accuracy;

    ProgressRow(string label, int curBatch, int numBatches, double batchLoss, double accuracy) {
        this->label = label;
        this->curBatch = curBatch;
        this->numBatches = numBatches;
        this->batchLoss = batchLoss;
        this->accuracy = accuracy;
    }
};

// FIXME: should be a singleton

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

    virtual void updateState(ExecutionState executionState);

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

    static std::thread *uiThread;
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
    static string thorDashUrl;

    Optional<ExecutionState> previousExecutionState;

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

    static void inputHandler();

    static void redrawWindows();

    static void drawHeader();
    static void drawProgressRows();
    static void drawFooter();
    static void drawOverallStatusBar();

    static void display();

    static int openUrl(string URL);

    static void drawStatusBar(
        void *win, int y, int xStart, int xEnd, double progress, string leftLabel, string rightLabel, bool boldLabels = false);

    static string popUpPrompt(string message);
    static void popUpAcknowledge(string message);

    static void printLine(ExecutionState executionState);

    static void drawBox(void *win, int top, int bottom, int left, int right);
    static void drawBlock(void *win, int top, int bottom, int left, int right);
};

}  // namespace Thor
