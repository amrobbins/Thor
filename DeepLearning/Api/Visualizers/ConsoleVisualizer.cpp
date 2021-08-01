#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"

// curses.h creates macros that clash with GraphicsMagick, so the two cannot be used together
// so curses.h is included in the source file rather than the header file.
#include <curses.h>

using namespace Thor;

using std::make_shared;
using std::map;
using std::mutex;
using std::pair;
using std::thread;
using std::to_string;
using std::unique_lock;
using std::vector;

const int ConsoleVisualizer::MIN_WIDTH = 140;
const int ConsoleVisualizer::HEIGHT_W0 = 9;
const int ConsoleVisualizer::MIN_HEIGHT_W1 = 10;
const int ConsoleVisualizer::HEIGHT_W2 = 10;

void *ConsoleVisualizer::win0 = nullptr;
void *ConsoleVisualizer::win1 = nullptr;
void *ConsoleVisualizer::win2 = nullptr;
int ConsoleVisualizer::terminalRows;
int ConsoleVisualizer::terminalCols;
int ConsoleVisualizer::windowWidth;
int ConsoleVisualizer::heightW0;
int ConsoleVisualizer::heightW1;
int ConsoleVisualizer::heightW2;
std::thread *ConsoleVisualizer::uiThread = nullptr;
bool ConsoleVisualizer::uiRunning = false;
std::recursive_mutex ConsoleVisualizer::mtx;

bool ConsoleVisualizer::scrollVisible = false;
int ConsoleVisualizer::scrollTop;
int ConsoleVisualizer::scrollBottom;
int ConsoleVisualizer::scrollLeft;
int ConsoleVisualizer::scrollRight;
int ConsoleVisualizer::scrollTopElement = -1;
bool ConsoleVisualizer::scrollBarSelected = false;
int ConsoleVisualizer::scrollClickFromTopOffset;
int ConsoleVisualizer::scrollBarDesiredTop;

int ConsoleVisualizer::thorDashLeft;
int ConsoleVisualizer::thorDashRight;
int ConsoleVisualizer::thorDashY;
string ConsoleVisualizer::thorDashUrl;

ExecutionState ConsoleVisualizer::mostRecentExecutionState;
string ConsoleVisualizer::cudaDevicesString;
vector<ProgressRow> ConsoleVisualizer::rows;
std::chrono::high_resolution_clock::time_point ConsoleVisualizer::start;
double ConsoleVisualizer::totalEpochLoss = 0;

void (*ConsoleVisualizer::originalResizeHandler)(int) = nullptr;
void (*ConsoleVisualizer::originalInterruptHandler)(int) = nullptr;
void (*ConsoleVisualizer::originalAbortHandler)(int) = nullptr;

void ConsoleVisualizer::resizeHandler(int sig) {
    unique_lock<recursive_mutex> lck(mtx);
    originalResizeHandler(SIGWINCH);
    display();
}

void ConsoleVisualizer::abortHandler(int sig) {
    unique_lock<recursive_mutex> lck(mtx);

    signal(SIGWINCH, originalResizeHandler);
    signal(SIGINT, originalInterruptHandler);
    signal(SIGABRT, originalAbortHandler);
    printf("\033[?1003l\n");  // Disable mouse movement events, as l = low
    endwin();

    if (originalAbortHandler != nullptr)
        originalAbortHandler(sig);
    else
        interruptHandler(SIGINT);
}

void ConsoleVisualizer::interruptHandler(int sig) {
    unique_lock<recursive_mutex> lck(mtx);
    signal(SIGINT, noOpHandler);

    wprintw((WINDOW *)win1, "interrupt");

    string response = popUpPrompt("Do you really want to quit? Type yes to quit:");

    if (boost::iequals(response, "yes")) {
        signal(SIGWINCH, originalResizeHandler);
        signal(SIGINT, originalInterruptHandler);
        signal(SIGABRT, originalAbortHandler);
        printf("\033[?1003l\n");  // Disable mouse movement events, as l = low
        endwin();
        originalInterruptHandler(sig);
    } else {
        popUpAcknowledge("Response was not yes");
    }

    signal(SIGINT, interruptHandler);

    originalResizeHandler(SIGWINCH);
    display();
}

void ConsoleVisualizer::noOpHandler(int sig) {}

void ConsoleVisualizer::inputHandler() {
    assert(executionStateQueue != nullptr);

    if (originalResizeHandler == nullptr)
        originalResizeHandler = signal(SIGWINCH, resizeHandler);
    assert(originalResizeHandler != nullptr);
    if (originalInterruptHandler == nullptr)
        originalInterruptHandler = signal(SIGINT, interruptHandler);
    assert(originalInterruptHandler != nullptr);
    if (originalAbortHandler == nullptr)
        originalAbortHandler = signal(SIGABRT, abortHandler);

    uint32_t delayMicroseconds = 10000;

    while (uiRunning) {
        mtx.lock();

        int ch = wgetch((WINDOW *)win1);
        if (ch == KEY_MOUSE) {
            // Pass mouse events to scroll bar
            MEVENT event;
            while (getmouse(&event) == OK) {
                if (event.bstate & BUTTON1_PRESSED) {
                    if (event.x >= thorDashLeft && event.x <= thorDashRight && event.y == thorDashY) {
                        openUrl(thorDashUrl);
                    } else if (event.x >= scrollLeft && event.x <= scrollRight && (event.y - heightW0) >= scrollTop &&
                               (event.y - heightW0) <= scrollBottom) {
                        scrollBarSelected = true;
                        scrollClickFromTopOffset = scrollTop - (event.y - heightW0);
                    } else if (event.x >= scrollLeft && event.x <= scrollRight && (event.y - heightW0) >= 1 &&
                               (event.y - heightW0) <= heightW1 - 1) {
                        scrollBarSelected = true;
                        scrollBarDesiredTop = (event.y - heightW0) + scrollClickFromTopOffset;
                        display();
                        scrollBarSelected = false;
                    } else {
                        scrollBarSelected = false;
                    }
                } else if (event.bstate & BUTTON1_RELEASED) {
                    scrollBarSelected = false;
                } else {
                    if (scrollBarSelected) {
                        scrollBarDesiredTop = (event.y - heightW0) + scrollClickFromTopOffset;
                        display();
                    }
                }
            }
        }

        ExecutionState executionState;
        bool newStateArrived = false;
        while (executionStateQueue->tryPop(executionState)) {
            newStateArrived = true;
            mostRecentExecutionState = executionState;

            if (mostRecentExecutionState.batchSize <= 0)
                totalEpochLoss = 0;
            else if (rows.empty() || rows.back().epochNum != mostRecentExecutionState.epochNum ||
                     rows.back().executionMode != mostRecentExecutionState.executionMode) {
                totalEpochLoss = mostRecentExecutionState.batchLoss;
            } else {
                totalEpochLoss += mostRecentExecutionState.batchLoss;
            }

            updateLog();
        }
        if (newStateArrived) {
            display();
        }

        mtx.unlock();
        if (ch == ERR && delayMicroseconds > 0)
            usleep(delayMicroseconds);

        if (executionStateQueue->occupancy() > 8)
            delayMicroseconds /= 10;
    }
}

string ConsoleVisualizer::popUpPrompt(string message) {
    // The resize handler needs to be removed while this message is active,
    // or else resizing the window will display the usual contents and hide this message.
    void (*initialResizeHandler)(int);
    initialResizeHandler = signal(SIGWINCH, noOpHandler);

    deleteWindows();

    WINDOW *win = newwin(terminalRows, windowWidth, 0, 0);
    nodelay(win, TRUE);
    keypad(win, TRUE);
    waddstr(win, message.c_str());
    waddch(win, ' ');
    int ch;
    string response;
    flushinp();
    while (true) {
        ch = wgetch(win);

        if (ch == ERR) {
            continue;
        } else if (ch == KEY_BACKSPACE) {
            if (!response.empty()) {
                response = response.substr(0, response.length() - 1);
                mvwaddch(win, 0, message.length() + 1 + response.length(), ' ');
                wmove(win, 0, message.length() + 1 + response.length());
            }
            continue;
        } else if (ch == '\n' || ch == '\r' || ch == EOF || ch == KEY_ENTER) {
            break;
        } else if ((ch < 'a' || ch > 'z') && (ch < 'A' || ch > 'Z') && (ch < '0' || ch > '9')) {
            continue;
        }

        waddch(win, (char)ch);
        response += (char)ch;
    }

    wrefresh(win);
    delwin(win);

    signal(SIGWINCH, initialResizeHandler);
    return response;
}

void ConsoleVisualizer::popUpAcknowledge(string message) {
    // The resize handler needs to be removed while this message is active,
    // or else resizing the window will display the usual contents and hide this message.
    void (*initialResizeHandler)(int);
    initialResizeHandler = signal(SIGWINCH, noOpHandler);

    deleteWindows();

    WINDOW *win = newwin(terminalRows, windowWidth, 0, 0);
    waddstr(win, message.c_str());
    waddstr(win, " - press any key to continue");
    flushinp();
    wgetch(win);

    wrefresh(win);
    delwin(win);

    signal(SIGWINCH, initialResizeHandler);
}

ConsoleVisualizer::ConsoleVisualizer() {
    assert(win0 == nullptr);
    assert(win1 == nullptr);
    assert(win2 == nullptr);
}

ConsoleVisualizer::~ConsoleVisualizer() {
    if (originalResizeHandler != nullptr)
        signal(SIGWINCH, originalResizeHandler);
    signal(SIGINT, originalInterruptHandler);
    signal(SIGABRT, originalAbortHandler);
    deleteWindows();

    stopUI();

    endwin();
}

void ConsoleVisualizer::initializeWindows() {
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);
    curs_set(0);
    if (has_colors()) {
        start_color();
        init_color(COLOR_BLACK, 999, 999, 999);  // black is now white
        init_color(COLOR_WHITE, 0, 0, 0);        // white is now black
        init_color(COLOR_BLUE, 0, 0, 999);
        init_color(COLOR_GREEN, 86, 539, 141);
        init_pair(2, COLOR_GREEN, COLOR_BLACK);
        init_pair(3, COLOR_YELLOW, COLOR_BLACK);
        init_pair(4, COLOR_BLUE, COLOR_BLACK);
    }

    printf("\033[?1003h\n");  // Makes the terminal report mouse movement events

    mouseinterval(0);
    mmask_t newMouseMask;
    newMouseMask = BUTTON1_PRESSED | BUTTON1_RELEASED | REPORT_MOUSE_POSITION;
    mousemask(newMouseMask, nullptr);
}

int ConsoleVisualizer::openUrl(string URL) {
    string command;
#if __APPLE__
    command = "open " + URL;
#else
    command = "xdg-open " + URL;
#endif
    FILE *proc = popen(command.c_str(), "r");
    return pclose(proc);
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
    wtimeout((WINDOW *)win1, 3);
    nodelay((WINDOW *)win1, TRUE);
    keypad((WINDOW *)win1, TRUE);
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
    deleteWindows();
    createWindows();
    redrawWindows();
}

void ConsoleVisualizer::redrawWindows() {
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
    wmove((WINDOW *)win0, 3, 20);
    if (mostRecentExecutionState.batchSize <= 0)
        waddstr((WINDOW *)win0, "Loading...");
    else
        waddstr((WINDOW *)win0, mostRecentExecutionState.networkName.c_str());
    wattron((WINDOW *)win0, A_BOLD);
    wmove((WINDOW *)win0, 4, 0);
    waddstr((WINDOW *)win0, "Dataset Name: ");
    wattroff((WINDOW *)win0, A_BOLD);
    wmove((WINDOW *)win0, 4, 20);
    waddstr((WINDOW *)win0, mostRecentExecutionState.datasetName.c_str());
    wattron((WINDOW *)win0, A_BOLD);
    wmove((WINDOW *)win0, 5, 0);
    waddstr((WINDOW *)win0, "ThorDash: ");
    wattroff((WINDOW *)win0, A_BOLD);
    wattron((WINDOW *)win0, COLOR_PAIR(4));
    wmove((WINDOW *)win0, 5, 20);
    thorDashUrl = "file:///home/andrew/EyeNet/train7/ThorDash/index.html";
    waddstr((WINDOW *)win0, thorDashUrl.c_str());
    thorDashLeft = 20;
    thorDashRight = 20 + thorDashUrl.length() - 1;
    thorDashY = 5;
    wattroff((WINDOW *)win0, COLOR_PAIR(4));
    wattron((WINDOW *)win0, A_BOLD);

    wattron((WINDOW *)win0, A_UNDERLINE);

    wmove((WINDOW *)win0, 8, 0);
    waddstr((WINDOW *)win0, "Epoch");

    wmove((WINDOW *)win0, 8, 15);
    waddstr((WINDOW *)win0, "Progress");

    wmove((WINDOW *)win0, 8, 70);
    waddstr((WINDOW *)win0, "Batch");

    wmove((WINDOW *)win0, 8, 95);
    waddstr((WINDOW *)win0, "Batch Loss");

    wmove((WINDOW *)win0, 8, 110);
    waddstr((WINDOW *)win0, "Accuracy");

    wattroff((WINDOW *)win0, A_UNDERLINE);

    wattroff((WINDOW *)win0, A_BOLD);
}

void ConsoleVisualizer::drawProgressRows() {
    if (mostRecentExecutionState.batchSize <= 0) {
        return;
    }

    if (rows.empty() || rows.back().epochNum != mostRecentExecutionState.epochNum ||
        rows.back().executionMode != mostRecentExecutionState.executionMode) {
        rows.emplace_back(mostRecentExecutionState.executionMode,
                          mostRecentExecutionState.epochNum,
                          mostRecentExecutionState.batchNum,
                          mostRecentExecutionState.batchesPerEpoch,
                          totalEpochLoss / mostRecentExecutionState.batchNum,
                          mostRecentExecutionState.epochAccuracy);
    } else {
        rows.back().curBatch = mostRecentExecutionState.batchNum;
        rows.back().batchLoss = totalEpochLoss / mostRecentExecutionState.batchNum;
    }

    if ((int)rows.size() <= heightW1) {
        scrollVisible = false;
        scrollTopElement = -1;
    } else {
        scrollVisible = true;

        double numRowsHidden = rows.size() - heightW1;
        double elementsPerScrollSlot = numRowsHidden / (heightW1 - 7);

        if (scrollBarSelected) {
            if (elementsPerScrollSlot >= 1.0) {
                scrollTop = scrollBarDesiredTop;
            } else {
                // Snap to element
                if (scrollBarDesiredTop <= 1 || scrollBarDesiredTop >= heightW1 - 6)
                    scrollTop = scrollBarDesiredTop;
                else if ((int)(scrollBarDesiredTop * elementsPerScrollSlot) != (int)(scrollTop * elementsPerScrollSlot))
                    scrollTop = scrollBarDesiredTop;
            }
        }

        if (scrollTop < 1)
            scrollTop = 1;
        if (scrollTop > heightW1 - 6)
            scrollTop = heightW1 - 6;

        if (scrollTop == 1) {
            scrollTopElement = 0;
        } else if (scrollTop == heightW1 - 6) {
            scrollTopElement = -1;
        } else {
            if (scrollBarSelected) {
                scrollTopElement = elementsPerScrollSlot * (scrollTop - 1);
            } else {
                if (scrollTopElement == -1)
                    scrollTop = heightW1 - 6;
                else
                    scrollTop = (scrollTopElement / elementsPerScrollSlot) + 1;
            }

            if (scrollTopElement + heightW1 >= (int)rows.size()) {
                scrollTopElement = -1;
                scrollTop = heightW1 - 6;
            }
        }

        drawBox(win1, 0, heightW1 - 1, 125, 129);
        scrollBottom = scrollTop + 4;
        scrollLeft = 126;
        scrollRight = 128;
        drawBlock(win1, scrollTop, scrollBottom, scrollLeft, scrollRight);
    }

    int firstToDisplay = scrollTopElement;
    if (firstToDisplay == -1)
        firstToDisplay = rows.size() - heightW1;
    if (firstToDisplay < 0)
        firstToDisplay = 0;

    for (int i = 0; i < heightW1 && i + firstToDisplay < (int)rows.size(); ++i) {
        wmove((WINDOW *)win1, i, 0);
        waddstr((WINDOW *)win1, rows[firstToDisplay + i].label.c_str());
        drawStatusBar(win1, i, 15, 65, double(rows[firstToDisplay + i].curBatch) / rows[firstToDisplay + i].numBatches, "", "");
        wmove((WINDOW *)win1, i, 70);
        waddstr((WINDOW *)win1,
                (std::to_string(rows[firstToDisplay + i].curBatch) + " of " + std::to_string(rows[firstToDisplay + i].numBatches)).c_str());
        wmove((WINDOW *)win1, i, 95);
        waddstr((WINDOW *)win1, std::to_string(rows[firstToDisplay + i].batchLoss).c_str());
        wmove((WINDOW *)win1, i, 110);
        waddstr((WINDOW *)win1, std::to_string(rows[firstToDisplay + i].accuracy).c_str());
    }
}

void ConsoleVisualizer::drawFooter() {
    wattron((WINDOW *)win2, A_BOLD);
    wattron((WINDOW *)win2, A_UNDERLINE);

    wmove((WINDOW *)win2, 1, 0);
    waddstr((WINDOW *)win2, "Training Info");

    wmove((WINDOW *)win2, 1, 70);
    waddstr((WINDOW *)win2, "Job Info");

    wattroff((WINDOW *)win2, A_UNDERLINE);
    wattroff((WINDOW *)win2, A_BOLD);

    char learningRateString[10];
    snprintf(learningRateString, 10, "%0.5f", mostRecentExecutionState.learningRate);
    char momentumString[10];
    snprintf(momentumString, 10, "%0.2f", mostRecentExecutionState.momentum);

    wmove((WINDOW *)win2, 2, 0);
    waddstr((WINDOW *)win2, "Training Algorithm:");
    wmove((WINDOW *)win2, 2, 35);
    waddstr((WINDOW *)win2, "Minibatch SGD");
    wmove((WINDOW *)win2, 3, 0);
    waddstr((WINDOW *)win2, "Current Learning Rate:");
    wmove((WINDOW *)win2, 3, 35);
    waddstr((WINDOW *)win2, learningRateString);
    wmove((WINDOW *)win2, 4, 0);
    waddstr((WINDOW *)win2, "Momentum:");
    wmove((WINDOW *)win2, 4, 35);
    waddstr((WINDOW *)win2, momentumString);
    wmove((WINDOW *)win2, 5, 0);
    waddstr((WINDOW *)win2, "Number of epochs to train:");
    wmove((WINDOW *)win2, 5, 35);
    waddstr((WINDOW *)win2, to_string(mostRecentExecutionState.epochsToTrain).c_str());

    char examplesPerHourString[31];
    double examplesPerHour = 0;
    string exampleUnits;
    if (mostRecentExecutionState.batchSize == 0) {
        examplesPerHourString[0] = '0';
        examplesPerHourString[1] = 0;
    } else {
        examplesPerHour = (1.0 / mostRecentExecutionState.runningAverageTimePerTrainingBatch) * mostRecentExecutionState.batchSize * 3600;
        if (examplesPerHour > 1.0e15) {
            examplesPerHour /= 1.0e15;
            exampleUnits = "Quadrillion";
        } else if (examplesPerHour > 1.0e12) {
            examplesPerHour /= 1.0e12;
            exampleUnits = "Trillion";
        } else if (examplesPerHour > 1.0e9) {
            examplesPerHour /= 1.0e9;
            exampleUnits = "Billion";
        } else if (examplesPerHour > 1.0e6) {
            examplesPerHour /= 1.0e6;
            exampleUnits = "Million";
        } else if (examplesPerHour > 1.0e3) {
            examplesPerHour /= 1.0e3;
            exampleUnits = "Thousand";
        }
        snprintf(examplesPerHourString, 31, "%0.1lf %s", examplesPerHour, exampleUnits.c_str());
    }

    char flopsString[31];
    double flops = 0;
    string flopsUnits;
    if (mostRecentExecutionState.batchSize == 0) {
        flopsString[0] = '0';
        flopsString[1] = 0;
    } else {
        double examplesPerSecond = (1.0 / mostRecentExecutionState.runningAverageTimePerTrainingBatch) * mostRecentExecutionState.batchSize;
        flops = examplesPerSecond * mostRecentExecutionState.flopsPerExample;
        if (flops > 1.0e24) {
            flops /= 1.0e24;
            flopsUnits = "YFLOPS";
        } else if (flops > 1.0e21) {
            flops /= 1.0e21;
            flopsUnits = "ZFLOPS";
        } else if (flops > 1.0e18) {
            flops /= 1.0e18;
            flopsUnits = "EFLOPS";
        } else if (flops > 1.0e15) {
            flops /= 1.0e15;
            flopsUnits = "PFLOPS";
        } else if (flops > 1.0e12) {
            flops /= 1.0e12;
            flopsUnits = "TFLOPS";
        } else if (flops > 1.0e9) {
            flops /= 1.0e9;
            flopsUnits = "GFLOPS";
        } else if (flops > 1.0e6) {
            flops /= 1.0e6;
            flopsUnits = "MFLOPS";
        } else if (flops > 1.0e3) {
            flops /= 1.0e3;
            flopsUnits = "KFLOPS";
        } else {
            flopsUnits = "FLOPS";
        }
        snprintf(flopsString, 31, "%0.1lf %s", flops, flopsUnits.c_str());
    }

    wmove((WINDOW *)win2, 2, 70);
    waddstr((WINDOW *)win2, "Training examples per hour:");
    wmove((WINDOW *)win2, 2, 105);
    waddstr((WINDOW *)win2, examplesPerHourString);
    wmove((WINDOW *)win2, 3, 70);
    waddstr((WINDOW *)win2, "Effective FLOPS:");
    wmove((WINDOW *)win2, 3, 105);
    waddstr((WINDOW *)win2, flopsString);
    wmove((WINDOW *)win2, 4, 70);
    waddstr((WINDOW *)win2, "GPUs:");
    wmove((WINDOW *)win2, 4, 105);
    waddstr((WINDOW *)win2, cudaDevicesString.c_str());
    wmove((WINDOW *)win2, 5, 70);
    waddstr((WINDOW *)win2, "Parallelization strategy:");
    wmove((WINDOW *)win2, 5, 105);
    waddstr((WINDOW *)win2, "Replicate and reduce");
    wmove((WINDOW *)win2, 6, 70);
    waddstr((WINDOW *)win2, "Output directory:");
    wmove((WINDOW *)win2, 6, 105);
    waddstr((WINDOW *)win2, mostRecentExecutionState.outputDirectory.c_str());
}

void ConsoleVisualizer::drawOverallStatusBar() {
    if (mostRecentExecutionState.batchSize <= 0) {
        start = std::chrono::high_resolution_clock::now();
        return;
    }
    int statusBarEnd = terminalCols - 5;
    if (terminalCols < 75)
        statusBarEnd = 70;

    double timeElapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
    string timeElapsedString = string("Elapsed: ");
    bool includeAllSmaller = false;
    if (timeElapsed >= 60 * 60 * 24) {
        includeAllSmaller = true;
        uint32_t timeElapsedDays = timeElapsed / (60 * 60 * 24);
        timeElapsed -= timeElapsedDays * (60 * 60 * 24);
        timeElapsedString += to_string(timeElapsedDays) + " day" + (timeElapsedDays != 1 ? "s " : "  ");
    }
    if (timeElapsed >= 60 * 60 || includeAllSmaller) {
        includeAllSmaller = true;
        uint32_t timeElapsedHours = timeElapsed / (60 * 60);
        timeElapsed -= timeElapsedHours * (60 * 60);
        string hours = to_string(timeElapsedHours);
        if (hours.length() == 1)
            hours = " " + hours;
        timeElapsedString += hours + " hour" + (timeElapsedHours != 1 ? "s " : "  ");
    }
    if (timeElapsed >= 60 || includeAllSmaller) {
        includeAllSmaller = true;
        uint32_t timeElapsedMinutes = timeElapsed / 60;
        timeElapsed -= timeElapsedMinutes * 60;
        string minutes = to_string(timeElapsedMinutes);
        if (minutes.length() == 1)
            minutes = " " + minutes;
        timeElapsedString += minutes + " minute" + (timeElapsedMinutes != 1 ? "s " : "  ");
    }
    uint32_t timeElapsedSeconds = timeElapsed;
    string seconds = to_string(timeElapsedSeconds);
    if (seconds.length() == 1)
        seconds = " " + seconds;
    timeElapsedString += seconds + " second" + (timeElapsedSeconds != 1 ? "s " : "  ");

    // FIXME: need to correct estimates to consider training and validation
    uint64_t totalBatchesToTrain;
    uint64_t batchesTrained;
    uint64_t totalBatchesToValidate;
    uint64_t batchesValidated;
    double timePerValidationBatch;
    if (mostRecentExecutionState.executionMode == ExampleType::TRAIN) {
        totalBatchesToTrain =
            mostRecentExecutionState.epochsToTrain * (mostRecentExecutionState.numTrainingExamples / mostRecentExecutionState.batchSize);
        batchesTrained =
            mostRecentExecutionState.epochNum * (mostRecentExecutionState.numTrainingExamples / mostRecentExecutionState.batchSize) +
            mostRecentExecutionState.batchNum;
        totalBatchesToValidate =
            mostRecentExecutionState.epochsToTrain * (mostRecentExecutionState.numValidationExamples / mostRecentExecutionState.batchSize);
        batchesValidated =
            mostRecentExecutionState.epochNum * (mostRecentExecutionState.numValidationExamples / mostRecentExecutionState.batchSize);

        // guess before have a measurement
        if (mostRecentExecutionState.epochNum == 0)
            timePerValidationBatch = mostRecentExecutionState.runningAverageTimePerTrainingBatch / 4;
        else
            timePerValidationBatch = mostRecentExecutionState.runningAverageTimePerValidationBatch;
    } else if (mostRecentExecutionState.executionMode == ExampleType::VALIDATE) {
        totalBatchesToTrain =
            mostRecentExecutionState.epochsToTrain * (mostRecentExecutionState.numTrainingExamples / mostRecentExecutionState.batchSize);
        batchesTrained =
            (mostRecentExecutionState.epochNum + 1) * (mostRecentExecutionState.numTrainingExamples / mostRecentExecutionState.batchSize);
        totalBatchesToValidate =
            mostRecentExecutionState.epochsToTrain * (mostRecentExecutionState.numValidationExamples / mostRecentExecutionState.batchSize);
        batchesValidated =
            mostRecentExecutionState.epochNum * (mostRecentExecutionState.numValidationExamples / mostRecentExecutionState.batchSize) +
            mostRecentExecutionState.batchNum;

        timePerValidationBatch = mostRecentExecutionState.runningAverageTimePerValidationBatch;
    } else {
        // FIXME: test mode
        assert(false);
    }

    uint64_t trainingBatchesRemaining = totalBatchesToTrain - batchesTrained;
    uint64_t validationBatchesRemaining = totalBatchesToValidate - batchesValidated;
    double trainingTimeRemaining = trainingBatchesRemaining * mostRecentExecutionState.runningAverageTimePerTrainingBatch;
    double validationTimeRemaining = validationBatchesRemaining * timePerValidationBatch;
    double timeRemaining = trainingTimeRemaining + validationTimeRemaining;

    timeElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
    double progress = timeElapsed / (timeElapsed + timeRemaining);

    // double progress = batchesTrained / (double)totalBatchesToTrain;
    // uint64_t batchesRemaining = totalBatchesToTrain - batchesTrained;
    // double timeRemaining = batchesRemaining * mostRecentExecutionState.runningAverageTimePerTrainingBatch;

    string timeRemainingString = string("Remaining: ");
    includeAllSmaller = false;
    if (timeRemaining >= 60 * 60 * 24) {
        includeAllSmaller = true;
        uint32_t timeRemainingDays = timeRemaining / (60 * 60 * 24);
        timeRemaining -= timeRemainingDays * (60 * 60 * 24);
        timeRemainingString += to_string(timeRemainingDays) + " day" + (timeRemainingDays != 1 ? "s " : "  ");
    }
    if (timeRemaining >= 60 * 60 || includeAllSmaller) {
        includeAllSmaller = true;
        uint32_t timeRemainingHours = timeRemaining / (60 * 60);
        timeRemaining -= timeRemainingHours * (60 * 60);
        string hours = to_string(timeRemainingHours);
        if (hours.length() == 1)
            hours = " " + hours;
        timeRemainingString += hours + " hour" + (timeRemainingHours != 1 ? "s " : "  ");
    }
    uint32_t timeRemainingMinutes = timeRemaining / 60;
    timeRemaining -= timeRemainingMinutes * 60;
    string minutes = to_string(timeRemainingMinutes);
    if (timeRemainingMinutes > 0 || includeAllSmaller) {
        if (minutes.length() == 1)
            minutes = " " + minutes;
        timeRemainingString += minutes + " minute" + (timeRemainingMinutes != 1 ? "s " : "  ");
    } else {
        uint32_t timeRemainingSeconds = timeRemainingSeconds;
        string seconds = to_string(timeRemainingSeconds);
        if (seconds.length() == 1)
            seconds = " " + seconds;
        timeRemainingString += seconds + " second" + (timeRemainingMinutes != 1 ? "s " : "  ");
    }

    drawStatusBar(win2, heightW2 - 1, 5, statusBarEnd, progress, timeElapsedString.c_str(), timeRemainingString.c_str(), true);
}

void ConsoleVisualizer::drawStatusBar(
    void *win, int y, int xStart, int xEnd, double progress, string leftLabel, string rightLabel, bool boldLabels) {
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

    if (!leftLabel.empty()) {
        if (boldLabels)
            wattron((WINDOW *)win2, A_BOLD);
        wprintw((WINDOW *)win, "%s ", leftLabel.c_str());
        if (boldLabels)
            wattroff((WINDOW *)win2, A_BOLD);
    }

    waddch((WINDOW *)win, '[' | A_BOLD);
    int rangeSize = ((xEnd - xStart) - 1) - labelLength;
    for (int i = 0; i < rangeSize; ++i) {
        if (progress > i / (double)rangeSize) {
            if (has_colors())
                waddch((WINDOW *)win, '=' | A_BOLD | COLOR_PAIR(2));
            else
                waddch((WINDOW *)win, '=' | A_BOLD);
        } else {
            waddch((WINDOW *)win, ' ');
        }
    }
    waddch((WINDOW *)win, ']' | A_BOLD);

    if (!rightLabel.empty()) {
        if (boldLabels)
            wattron((WINDOW *)win2, A_BOLD);
        wprintw((WINDOW *)win, " %s", rightLabel.c_str());
        if (boldLabels)
            wattroff((WINDOW *)win2, A_BOLD);
    }

    int percentLocation = xStart + (rangeSize / 2);
    if (!leftLabel.empty())
        percentLocation += leftLabel.length() + 1;
    wmove((WINDOW *)win, y, percentLocation);
    if (progress < 1.0)
        percentLocation += 1;
    wprintw((WINDOW *)win, "%0.0lf%%", floor(progress * 100.0));
}

void ConsoleVisualizer::drawBox(void *win, int top, int bottom, int left, int right) {
    wmove((WINDOW *)win, top, left);
    waddch((WINDOW *)win, ACS_ULCORNER);
    wmove((WINDOW *)win, top, right);
    waddch((WINDOW *)win, ACS_URCORNER);
    wmove((WINDOW *)win, bottom, left);
    waddch((WINDOW *)win, ACS_LLCORNER);
    wmove((WINDOW *)win, bottom, right);
    waddch((WINDOW *)win, ACS_LRCORNER);

    for (int i = top + 1; i <= bottom - 1; ++i) {
        wmove((WINDOW *)win, i, left);
        waddch((WINDOW *)win, ACS_VLINE);
        wmove((WINDOW *)win, i, right);
        waddch((WINDOW *)win, ACS_VLINE);
    }

    for (int i = left + 1; i <= right - 1; ++i) {
        wmove((WINDOW *)win, top, i);
        waddch((WINDOW *)win, ACS_HLINE);
        wmove((WINDOW *)win, bottom, i);
        waddch((WINDOW *)win, ACS_HLINE);
    }
}

void ConsoleVisualizer::drawBlock(void *win, int top, int bottom, int left, int right) {
    for (int i = top; i <= bottom; ++i) {
        wmove((WINDOW *)win, i, left);
        for (int j = left; j <= right; ++j) {
            waddch((WINDOW *)win, ACS_CKBOARD);
        }
    }
}

void ConsoleVisualizer::dumpSummaryToTerminal() {
    // FIXME: implement
}

void ConsoleVisualizer::updateLog() {
    // FIXME: implement
}

void ConsoleVisualizer::startUI() {
    unique_lock<recursive_mutex> lck(mtx);

    if (uiRunning)
        return;

    initializeWindows();

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    map<string, uint32_t> cudaDevices;
    for (uint32_t i = 0; i < numGpus; ++i) {
        cudaDevices[MachineEvaluator::instance().getGpuType(i)] += 1;
    }
    cudaDevicesString = "";
    for (auto it = cudaDevices.begin(); it != cudaDevices.end(); ++it) {
        string deviceType = it->first;
        uint32_t deviceCount = it->second;
        if (cudaDevicesString.empty())
            cudaDevicesString = to_string(deviceCount) + "x " + deviceType;
        else
            cudaDevicesString += ", " + to_string(deviceCount) + "x " + deviceType;
    }

    mostRecentExecutionState.batchSize = 0;

    printf("\033[?1003l\n");  // Disable mouse movement events, as l = low
    uiRunning = true;
    uiThread = new thread(&ConsoleVisualizer::inputHandler, this);

    display();
}

void ConsoleVisualizer::stopUI() {
    {
        unique_lock<recursive_mutex> lck(mtx);

        if (!uiRunning)
            return;

        printf("\033[?1003l\n");  // Disable mouse movement events, as l = low

        uiRunning = false;
    }

    uiThread->join();
    delete uiThread;
    uiThread = nullptr;

    dumpSummaryToTerminal();
}
