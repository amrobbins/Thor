#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"

#include <boost/algorithm/string.hpp>

#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <mutex>
#include <utility>
#include <vector>

#include <csignal>

namespace Thor {

class Visualizer;

class ConsoleVisualizer : public Visualizer {
   public:
    class Builder;
    ConsoleVisualizer();

    virtual ~ConsoleVisualizer();

    void updateState(ExecutionState executionState, HyperparameterController hyperparameterController);

    // temp public:
    static void display();

   private:
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

    bool initialized;

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

    static void drawHeader();
    static void drawProgressRows();
    static void drawFooter();
    static void drawOverallStatusBar();

    static void drawStatusBar(void *win, int y, int xStart, int xEnd, double progress, string leftLabel, string rightLabel);

    static string popUpPrompt(string message);
    static void popUpAcknowledge(string message);

    static void printLine(ExecutionState executionState, HyperparameterController hyperparameterController);
};

class ConsoleVisualizer::Builder {
   public:
    virtual shared_ptr<Visualizer> build();
};

}  // namespace Thor
