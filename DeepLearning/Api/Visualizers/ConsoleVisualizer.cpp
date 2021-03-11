#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"

// curses.h creates macros that clash with GraphicsMagick, so the two cannot be used together
// so curses.h is included in the source file rather than the header file.
#include <ncurses.h>

using namespace Thor;

using std::make_shared;
using std::pair;
using std::vector;

const int ConsoleVisualizer::MIN_WIDTH = 140;
const int ConsoleVisualizer::HEIGHT_W0 = 8;
const int ConsoleVisualizer::MIN_HEIGHT_W1 = 10;
const int ConsoleVisualizer::HEIGHT_W2 = 15;

void *ConsoleVisualizer::win0 = nullptr;
void *ConsoleVisualizer::win1 = nullptr;
void *ConsoleVisualizer::win2 = nullptr;
int ConsoleVisualizer::terminalRows;
int ConsoleVisualizer::terminalCols;
int ConsoleVisualizer::windowWidth;
int ConsoleVisualizer::heightW0;
int ConsoleVisualizer::heightW1;
int ConsoleVisualizer::heightW2;

void (*ConsoleVisualizer::originalResizeHandler)(int);
void (*ConsoleVisualizer::originalInterruptHandler)(int);

shared_ptr<Visualizer> ConsoleVisualizer::Builder::build() {
    shared_ptr<ConsoleVisualizer> consoleVisualizer = make_shared<ConsoleVisualizer>();
    consoleVisualizer->initialized = true;
    return consoleVisualizer;
}

void ConsoleVisualizer::updateState(ExecutionState executionState, HyperparameterController hyperparameterController) {
    if (previousExecutionState.isEmpty()) {
        drawHeader();
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

string ConsoleVisualizer::popUpPrompt(string message) {
    // FIXME: implement
    return "";
}

void ConsoleVisualizer::popUpAcknowledge(string message) {
    // FIXME: implement
    wgetch((WINDOW *)win0);
}

void ConsoleVisualizer::interruptHandler(int sig) {
    originalInterruptHandler(sig);

    string response = "yes";
    response = popUpPrompt("Do you really want to quit? Type yes to quit: ");
    response = "yes";  // FIXME

    if (boost::iequals(response, "yes")) {
        signal(SIGWINCH, originalResizeHandler);
        signal(SIGWINCH, originalInterruptHandler);
        originalInterruptHandler(sig);
    } else {
        popUpAcknowledge("Response was not yes - press any key to continue");
    }
}

ConsoleVisualizer::ConsoleVisualizer() {
    assert(win0 == nullptr);
    assert(win1 == nullptr);
    assert(win2 == nullptr);

    initialized = false;
    initializeWindows();
    createWindows();
    originalResizeHandler = signal(SIGWINCH, resizeHandler);
    originalInterruptHandler = signal(SIGINT, interruptHandler);
}

ConsoleVisualizer::~ConsoleVisualizer() {
    signal(SIGWINCH, originalResizeHandler);
    signal(SIGWINCH, originalInterruptHandler);
    deleteWindows();
    endwin();
}

void ConsoleVisualizer::initializeWindows() {
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    start_color();
    init_color(COLOR_BLACK, 999, 999, 999);   // black is now white
    init_color(COLOR_WHITE, 0, 0, 0);         // white is now black
    init_color(COLOR_YELLOW, 333, 333, 333);  // YELLOW is now gray
    init_color(COLOR_GREEN, 86, 539, 141);
    init_pair(2, COLOR_GREEN, COLOR_BLACK);
    init_pair(3, COLOR_YELLOW, COLOR_BLACK);
    init_pair(4, COLOR_BLUE, COLOR_BLACK);
}

void ConsoleVisualizer::createWindows() {
    getmaxyx(stdscr, terminalRows, terminalCols);

    windowWidth = terminalCols;
    if (windowWidth < MIN_WIDTH)
        windowWidth = MIN_WIDTH;

    heightW0 = HEIGHT_W0;

    heightW1 = terminalRows - (heightW0 + HEIGHT_W2);
    if (heightW1 < MIN_HEIGHT_W1)
        heightW1 = MIN_HEIGHT_W1;

    heightW2 = HEIGHT_W2;

    win0 = newwin(heightW0, windowWidth, 0, 0);
    win1 = newwin(heightW1, windowWidth, heightW0, 0);
    scrollok((WINDOW *)win1, TRUE);
    win2 = newwin(heightW2, windowWidth, heightW0 + heightW1, 0);
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

    drawHeader();
    drawProgressRows();
    drawFooter();
    drawOverallStatusBar();

    wrefresh((WINDOW *)win0);
    wrefresh((WINDOW *)win1);
    wrefresh((WINDOW *)win2);
}

void ConsoleVisualizer::drawHeader() {
    wattron((WINDOW *)win0, A_BOLD);

    string title = "Thor - Console Visualizer";
    int titleStart = (terminalCols - title.length()) / 2;
    if (titleStart < 0)
        titleStart = 0;

    wmove((WINDOW *)win0, 0, titleStart);
    waddstr((WINDOW *)win0, title.c_str());

    wmove((WINDOW *)win0, 3, 0);
    waddstr((WINDOW *)win0, "Network Name: ");
    wattroff((WINDOW *)win0, A_BOLD);
    waddstr((WINDOW *)win0, "EyeNet");
    wattron((WINDOW *)win0, A_BOLD);
    wmove((WINDOW *)win0, 4, 0);
    waddstr((WINDOW *)win0, "Dataset Name: ");
    wattroff((WINDOW *)win0, A_BOLD);
    waddstr((WINDOW *)win0, "ImageNet 2012");
    wattron((WINDOW *)win0, A_BOLD);

    wattron((WINDOW *)win0, A_UNDERLINE);

    wmove((WINDOW *)win0, 7, 0);
    waddstr((WINDOW *)win0, "Epoch");

    wmove((WINDOW *)win0, 7, 15);
    waddstr((WINDOW *)win0, "Progress");

    wmove((WINDOW *)win0, 7, 70);
    waddstr((WINDOW *)win0, "Batch");

    wmove((WINDOW *)win0, 7, 95);
    waddstr((WINDOW *)win0, "Batch Loss");

    wmove((WINDOW *)win0, 7, 110);
    waddstr((WINDOW *)win0, "Accuracy");

    wattroff((WINDOW *)win0, A_UNDERLINE);

    wattroff((WINDOW *)win0, A_BOLD);
}

void ConsoleVisualizer::drawProgressRows() {
    wmove((WINDOW *)win1, 0, 0);
    waddstr((WINDOW *)win1, "Train 0");
    drawStatusBar(win1, 0, 15, 65, 100.0, "", "");
    wmove((WINDOW *)win1, 0, 70);
    waddstr((WINDOW *)win1, "7500 of 7500");
    wmove((WINDOW *)win1, 0, 95);
    waddstr((WINDOW *)win1, "0.753");
    wmove((WINDOW *)win1, 0, 110);
    waddstr((WINDOW *)win1, "0.174");

    wmove((WINDOW *)win1, 1, 0);
    waddstr((WINDOW *)win1, "Validate 0");
    drawStatusBar(win1, 1, 15, 65, 100.0, "", "");
    wmove((WINDOW *)win1, 1, 70);
    waddstr((WINDOW *)win1, "1000 of 1000");
    wmove((WINDOW *)win1, 1, 95);
    waddstr((WINDOW *)win1, "0.620");
    wmove((WINDOW *)win1, 1, 110);
    waddstr((WINDOW *)win1, "0.196");

    wmove((WINDOW *)win1, 2, 0);
    waddstr((WINDOW *)win1, "Train 1");
    drawStatusBar(win1, 2, 15, 65, 100.0, "", "");
    wmove((WINDOW *)win1, 2, 70);
    waddstr((WINDOW *)win1, "7500 of 7500");
    wmove((WINDOW *)win1, 2, 95);
    waddstr((WINDOW *)win1, "0.572");
    wmove((WINDOW *)win1, 2, 110);
    waddstr((WINDOW *)win1, "0.212");

    wmove((WINDOW *)win1, 3, 0);
    waddstr((WINDOW *)win1, "Validate 1");
    drawStatusBar(win1, 3, 15, 65, 100.0, "", "");
    wmove((WINDOW *)win1, 3, 70);
    waddstr((WINDOW *)win1, "1000 of 1000");
    wmove((WINDOW *)win1, 3, 95);
    waddstr((WINDOW *)win1, "0.565");
    wmove((WINDOW *)win1, 3, 110);
    waddstr((WINDOW *)win1, "0.243");

    double progress = (rand() % 101) / 100.0;
    wmove((WINDOW *)win1, 4, 0);
    waddstr((WINDOW *)win1, "Train 2");
    drawStatusBar(win1, 4, 15, 65, progress, "", "");
    wmove((WINDOW *)win1, 4, 70);
    wprintw((WINDOW *)win1, "%d of 7500", (int)(7500 * progress));
    wmove((WINDOW *)win1, 4, 95);
    waddstr((WINDOW *)win1, "0.510");
    wmove((WINDOW *)win1, 4, 110);
    waddstr((WINDOW *)win1, "0.271");
}

void ConsoleVisualizer::drawFooter() {
    wattron((WINDOW *)win2, A_BOLD);
    wattron((WINDOW *)win2, A_UNDERLINE);

    wmove((WINDOW *)win2, 0, 0);
    waddstr((WINDOW *)win2, "Training Info");

    wmove((WINDOW *)win2, 0, 70);
    waddstr((WINDOW *)win2, "Job Info");

    wattroff((WINDOW *)win2, A_UNDERLINE);
    wattroff((WINDOW *)win2, A_BOLD);

    wmove((WINDOW *)win2, 1, 0);
    waddstr((WINDOW *)win2, "Training Algorithm:");
    wmove((WINDOW *)win2, 1, 35);
    waddstr((WINDOW *)win2, "Minibatch SGD");
    wmove((WINDOW *)win2, 2, 0);
    waddstr((WINDOW *)win2, "Current Learning Rate:");
    wmove((WINDOW *)win2, 2, 35);
    waddstr((WINDOW *)win2, "0.05");
    wmove((WINDOW *)win2, 3, 0);
    waddstr((WINDOW *)win2, "Momentum:");
    wmove((WINDOW *)win2, 3, 35);
    waddstr((WINDOW *)win2, "0.75");
    wmove((WINDOW *)win2, 4, 0);
    waddstr((WINDOW *)win2, "Number of epochs to train:");
    wmove((WINDOW *)win2, 4, 35);
    waddstr((WINDOW *)win2, "50");

    wmove((WINDOW *)win2, 1, 70);
    waddstr((WINDOW *)win2, "Training examples per hour:");
    wmove((WINDOW *)win2, 1, 105);
    waddstr((WINDOW *)win2, "7,562,149");
    wmove((WINDOW *)win2, 2, 70);
    waddstr((WINDOW *)win2, "Gpu's being used:");
    wmove((WINDOW *)win2, 2, 105);
    waddstr((WINDOW *)win2, "4 Nvidia 2080 Ti's");
    wmove((WINDOW *)win2, 3, 70);
    waddstr((WINDOW *)win2, "Parallelization strategy:");
    wmove((WINDOW *)win2, 3, 105);
    waddstr((WINDOW *)win2, "Replicate and reduce");
    wmove((WINDOW *)win2, 4, 70);
    waddstr((WINDOW *)win2, "Output directory:");
    wmove((WINDOW *)win2, 4, 105);
    waddstr((WINDOW *)win2, "/home/andrew/EyeNet/train7/");
    wmove((WINDOW *)win2, 5, 70);
    waddstr((WINDOW *)win2, "ThorDash:");
    wmove((WINDOW *)win2, 5, 105);
    waddstr((WINDOW *)win2, "file:///home/andrew/EyeNet/train7/ThorDash/index.html");
}

void ConsoleVisualizer::drawOverallStatusBar() {
    int statusBarEnd = terminalCols - 5;
    if (terminalCols < 75)
        statusBarEnd = 70;
    drawStatusBar(win2, heightW2 - 1, 5, statusBarEnd, (rand() % 101) / 100.0, "Elapsed: 5h 23m 7s", "Remaining: 2h 12m 32s");
}

void ConsoleVisualizer::drawStatusBar(void *win, int y, int xStart, int xEnd, double progress, string leftLabel, string rightLabel) {
    int labelLength = leftLabel.length() + rightLabel.length();
    if (!leftLabel.empty())
        labelLength += 1;  // space
    if (!rightLabel.empty())
        labelLength += 1;  // space

    if (xEnd - xStart < 6 + labelLength)
        return;
    if (progress < 0.0)
        progress = 0.0;
    if (progress > 1.0)
        progress = 1.0;

    wmove((WINDOW *)win, y, xStart);

    if (!leftLabel.empty())
        wprintw((WINDOW *)win, "%s ", leftLabel.c_str());

    waddch((WINDOW *)win, '[' | A_BOLD);
    int rangeSize = ((xEnd - xStart) - 1) - labelLength;
    for (int i = 0; i < rangeSize; ++i) {
        if (progress > i / (double)rangeSize)
            waddch((WINDOW *)win, '=' | A_BOLD | COLOR_PAIR(2));
        else
            waddch((WINDOW *)win, ' ');
    }
    waddch((WINDOW *)win, ']' | A_BOLD);

    if (!rightLabel.empty())
        wprintw((WINDOW *)win, " %s", rightLabel.c_str());

    int percentLocation = xStart + (rangeSize / 2);
    if (!leftLabel.empty())
        percentLocation += leftLabel.length() + 1;
    wmove((WINDOW *)win, y, percentLocation);
    wprintw((WINDOW *)win, "%0.0lf%%", progress * 100.0);
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
