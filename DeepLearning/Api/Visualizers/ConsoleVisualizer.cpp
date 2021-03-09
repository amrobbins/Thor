#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"

// curses.h creates macros that clash with GraphicsMagick, so the two cannot be used together
// so curses.h is included in the source file rather than the header file.
#include <ncurses.h>

using namespace Thor;

using std::make_shared;
using std::pair;
using std::vector;

const int ConsoleVisualizer::MIN_WIDTH = 140;
const int ConsoleVisualizer::HEIGHT_W0 = 5;
const int ConsoleVisualizer::MIN_HEIGHT_W1 = 30;
const int ConsoleVisualizer::HEIGHT_W2 = 5;

void *ConsoleVisualizer::win0 = nullptr;
void *ConsoleVisualizer::win1 = nullptr;
void *ConsoleVisualizer::win2 = nullptr;
int ConsoleVisualizer::terminalRows;
int ConsoleVisualizer::terminalCols;
int ConsoleVisualizer::heightW0;
int ConsoleVisualizer::heightW1;
int ConsoleVisualizer::heightW2;

void (*ConsoleVisualizer::originalResizeHandler)(int);

shared_ptr<Visualizer> ConsoleVisualizer::Builder::build() {
    shared_ptr<ConsoleVisualizer> consoleVisualizer = make_shared<ConsoleVisualizer>();
    consoleVisualizer->initialized = true;
    return consoleVisualizer;
}

void ConsoleVisualizer::updateState(ExecutionState executionState, HyperparameterController hyperparameterController) {
    if (previousExecutionState.isEmpty()) {
        printHeader(executionState, hyperparameterController);
        printLine(executionState, hyperparameterController);
    } else {
        if (previousExecutionState.get().executionMode != executionState.executionMode) {
            // start a new line
        } else {
            // continue with line
        }
    }

    previousExecutionState = executionState;
}

void ConsoleVisualizer::resizeHandler(int sig) {
    originalResizeHandler(sig);

    deleteWindows();
    createWindows();
    display();
}

ConsoleVisualizer::ConsoleVisualizer() {
    assert(win0 == nullptr);
    assert(win1 == nullptr);
    assert(win2 == nullptr);

    initialized = false;
    initializeWindows();
    createWindows();
    originalResizeHandler = signal(SIGWINCH, resizeHandler);
}

ConsoleVisualizer::~ConsoleVisualizer() {
    signal(SIGWINCH, SIG_DFL);
    deleteWindows();
    endwin();
}

void ConsoleVisualizer::initializeWindows() {
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
}

void ConsoleVisualizer::createWindows() {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    terminalRows = w.ws_row;
    terminalCols = w.ws_col;

    int windowWidth = terminalCols;
    if (windowWidth < MIN_WIDTH)
        windowWidth = MIN_WIDTH;
    heightW0 = HEIGHT_W0;
    if (heightW0 > terminalRows)
        heightW0 = terminalRows;
    assert(windowWidth > 0);
    assert(heightW0 > 0);

    heightW1 = terminalRows - (heightW0 + HEIGHT_W2);
    if (heightW1 < 0)
        heightW1 = 0;

    heightW2 = terminalRows - (heightW0 + heightW1);

    win0 = newwin(heightW0, windowWidth, 0, 0);
    if (heightW1 > 0)
        win1 = newwin(heightW1, windowWidth, heightW0, 0);
    if (heightW2 > 0)
        win2 = newwin(HEIGHT_W2, windowWidth, heightW0 + heightW1, 0);
}

void ConsoleVisualizer::deleteWindows() {
    if (win0 != nullptr) {
        wrefresh((WINDOW *)win0);
        delwin((WINDOW *)win0);
        win0 = nullptr;
    }
    if (win1 != nullptr) {
        wrefresh((WINDOW *)win1);
        delwin((WINDOW *)win1);
        win1 = nullptr;
    }
    if (win2 != nullptr) {
        wrefresh((WINDOW *)win2);
        delwin((WINDOW *)win2);
        win2 = nullptr;
    }
}

void ConsoleVisualizer::display() {
    wrefresh((WINDOW *)win0);
    wmove((WINDOW *)win0, 0, 0);
    wprintw((WINDOW *)win0,
            "0: [%d, %d] height %d, width %d  (%d, %d)---------------------",
            0,
            heightW0 - 1,
            terminalRows,
            terminalCols,
            LINES,
            COLS);
    wrefresh((WINDOW *)win0);

    if (win1 != nullptr) {
        wrefresh((WINDOW *)win1);
        wmove((WINDOW *)win1, 0, 0);
        wprintw((WINDOW *)win1, "1: [%d, %d] height %d, width %d", heightW0, (heightW0 + heightW1) - 1, terminalRows, terminalCols);
        wrefresh((WINDOW *)win1);
    }

    if (win2 != nullptr) {
        wrefresh((WINDOW *)win2);
        wmove((WINDOW *)win2, 0, 0);
        wprintw((WINDOW *)win2,
                "2: [%d, %d] height %d, width %d",
                heightW0 + heightW1,
                (heightW0 + heightW1 + HEIGHT_W2) - 1,
                terminalRows,
                terminalCols);
        wrefresh((WINDOW *)win2);
    }
}

void ConsoleVisualizer::printHeader(ExecutionState executionState, HyperparameterController hyperparameterController) {
    // FIXME: use ncurses
    // initscr();
    // cbreak();
    // noecho();

    // int h, w;
    // getmaxyx(stdscr, h, w);
    // WINDOW *win = newwin(h, w, 0, 0);

    // wrefresh(win);
    // wmove(win, h-1, 0);
    // wprintw(win, "height %d, width %d\n", h, w);
    // addstr("a string");
    /*
        wmove((WINDOW*)win0, 0, 0);
        wprintw((WINDOW*)win0, "height %d, width %d", terminalRows, terminalCols);
        wrefresh((WINDOW*)win0);

        wmove((WINDOW*)win1, 0, 0);
        wprintw((WINDOW*)win1, "height %d, width %d", terminalRows, terminalCols);
        wrefresh((WINDOW*)win1);

        wmove((WINDOW*)win2, 0, 0);
        wprintw((WINDOW*)win2, "height %d, width %d", terminalRows, terminalCols);
        wrefresh((WINDOW*)win2);


        wgetch((WINDOW*)win0);
    */

    /*
        printf("Thor console visualizer\n");
        printf("Training network FIXME_NEED_NETWORK_NAME\n");
        printf("\nData set stats\n");
        printf("Example Classes:\n");
        printf("Training Examples:\n");
        printf("Validation Examples:\n");
        printf("Test Examples:\n");
        printf("\nTrain session stats\n");
        printf("Epochs FIXME_N\n");
        printf("Batch size FIXME_N\n");
        printf("\n");
        vector<pair<string, string>> hyperparameterDisplayInfo = hyperparameterController.getHeaderDisplayInfo();
        for (uint32_t i = 0; i < hyperparameterDisplayInfo.size(); ++i) {
            printf("%s %s\n", hyperparameterDisplayInfo[i].first.c_str(), hyperparameterDisplayInfo[i].second.c_str());
        }
    */

    // delwin(win);
    // endwin();
}

void ConsoleVisualizer::printLine(ExecutionState executionState, HyperparameterController hyperparameterController) {
    vector<pair<string, string>> hyperparameterDisplayInfo = hyperparameterController.getCurrentEpochInfo(executionState);
    for (uint32_t i = 0; i < hyperparameterDisplayInfo.size(); ++i) {
        printf("%s %s\n", hyperparameterDisplayInfo[i].first.c_str(), hyperparameterDisplayInfo[i].second.c_str());
    }
    string epochType;
    if (executionState.executionMode == ExampleType::TRAIN)
        epochType = "Trainining";
    if (executionState.executionMode == ExampleType::TRAIN)
        epochType = "Validating";
    else
        epochType = "Testing";
    printf("%s Epoch %ld, batch %ld of %ld\n",
           epochType.c_str(),
           executionState.epochNum,
           executionState.batchNum + 1,
           executionState.batchesPerEpoch);
    double percentComplete = (executionState.batchNum + 1) / executionState.batchesPerEpoch;
    uint32_t numStars = percentComplete * 100;
    for (uint32_t i = 0; i < numStars; ++i) {
        printf("%s%s\n", std::string(numStars, '*').c_str(), std::string(100 - numStars, 'o').c_str());
    }
}
