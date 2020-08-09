#pragma once

/**
 * This queue opens when open() is called and creates a threadpool of the number of threads specified.
 * This queue runs the executor (that is passed in) on the input that are passed in and stored in the input queue.
 * After the executor processes an input it returns an output and places it out the output queue.
 * Once finished, the executor takes another input from the input queue and repeats as long as the input queue is not
 * empty. Each of the N threads performs an executor process in parallel with the other threads, but using a different
 * input. If the input queue empties then all executor threads wait until the queue is nonEmpty and then resume
 * executing once input becomes available again. All resources are released (input/output queues and executor) when
 * close is called. The queue cannot be reused again until open is called again.
 *
 * This is an unordered queue, so the order that the output queue is filled depends on which executors finish first.
 * This can lead to efficiencies above that of the ordered queue in some cases, at the cost of unordered output.
 * The user would need to pass information from the input to the output so that reading the output can give the proper
 * order (or just identify what it is). This would be implemented in the executor function.
 */

#include "WorkQueueExecutorBase.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <chrono>
#include <memory>

#include <deque>
#include <vector>

#include <assert.h>
#include <stdio.h>

template <class InputType, class OutputType>
class WorkQueueUnordered {
   public:
    WorkQueueUnordered();
    WorkQueueUnordered(bool hasOutput);
    virtual ~WorkQueueUnordered();
    WorkQueueUnordered(const WorkQueueUnordered &) = delete;
    WorkQueueUnordered &operator=(const WorkQueueUnordered &) = delete;

    // Blocking.
    // Takes full ownership of the constructed executor that is passed in. It will be deleted when close() is called. If
    // an lvalue pointer is used as the argument to executor, it will be set to NULL.
    void open(std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> &executor,
              int numThreads = -1,
              unsigned int inputBufferingMultiple = 8,
              unsigned int outputBufferingMultiple = 8);
    void open(std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> &&executor,
              int numThreads = -1,
              unsigned int inputBufferingMultiple = 8,
              unsigned int outputBufferingMultiple = 8);
    void close();

    // Time monitoring.
    // Can be used for input-queue-clearing-latency matching across multiple queues
    void beginExecutorPerformanceTiming(double decayFactor = 0.05);  // running average latency is calculated as: avgLatency = decayFactor *
                                                                     // mostRecentLatency + decayFactor * avgLatency.
    void stopExecutorPerformanceTiming();
    double getRunningAverageExecutorLatency();
    double getTimeTillEmptyIfInputStops();
    double getPercentageOfTimeInputsAreNotProcessedFastEnoughToKeepUp();
    bool wouldPipelineLikelyBenefitFromMoreOfThisResource();

    // Blocking API.
    // Returns true on success.
    // Returns false if the queue has been closed before the request has been serviced.
    bool push(const InputType &input);
    bool pop(OutputType &output);
    bool peek(OutputType &output);

    // Nonblocking API. Returns true on success.
    bool tryPush(const InputType &input);
    bool tryPop(OutputType &output);
    bool tryPeek(OutputType &output);

    // Nonblocking
    bool isFull();         // true when tryPush will fail
    bool isEmpty();        // true when there is no work left to be done and no outputs left to give
    bool isOutputReady();  // true when tryPop will succeed
    bool isOpen();

    int inputQueueOccupancy();
    int inputQueueSize();
    int outputQueueOccupancy();
    int outputQueueSize();

   private:
    unsigned int inputBufferSize;
    bool hasOutput;
    unsigned int outputBufferSize;

    std::mutex mtx;
    std::vector<std::thread *> threads;
    std::condition_variable inputNotEmpty;
    std::condition_variable outputNotEmpty;
    std::condition_variable notFull;

    std::deque<InputType> inputStorage;
    std::deque<OutputType> outputStorage;

    bool queueOpen = false;
    unsigned int threadsRunning = 0;

    bool isEmptyUnlocked();
    int totalOccupancyUnlocked();

    bool monitorExecutorTiming = false;
    double executorRunningAverageLatency;
    double decayFactor;
    bool firstPushRecorded = false;
    std::chrono::high_resolution_clock::time_point t_firstPush;
    std::chrono::high_resolution_clock::time_point t_previousChangeInOccupancy;
    std::chrono::high_resolution_clock::time_point t_finalPop;
    double elapsedTimeInputQueueFull;
    std::chrono::high_resolution_clock::time_point mostRecentOutputObserved;

    unsigned int numThreads = 0;
    std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> executor;

    void execute();

    void openImpl(int numThreads, unsigned int inputBufferingMultiple, unsigned int outputBufferingMultiple);

    inline bool inputQueueFullAtOccupancy(unsigned int occ) { return occ > 0.75 * inputBufferSize && occ > numThreads; }
    inline bool outputQueueFullAtOccupancy(unsigned int threadsRunning, unsigned int occ) {
        unsigned int totalOcc = threadsRunning + occ;
        return totalOcc > 0.75 * outputBufferSize && totalOcc > numThreads;
    }
};

template <class InputType, class OutputType>
WorkQueueUnordered<InputType, OutputType>::WorkQueueUnordered() {
    this->hasOutput = true;
}

template <class InputType, class OutputType>
WorkQueueUnordered<InputType, OutputType>::WorkQueueUnordered(bool hasOutput) {
    this->hasOutput = hasOutput;
}

template <class InputType, class OutputType>
WorkQueueUnordered<InputType, OutputType>::~WorkQueueUnordered() {
    close();
}

// By default the queue opens with num hardware threads - 1, and 4x this number of output buffers
template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::openImpl(int numThreads,
                                                         unsigned int inputBufferingMultiple,
                                                         unsigned int outputBufferingMultiple) {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    assert(numThreads < 10000);
    assert(inputBufferingMultiple > 0);
    assert(outputBufferingMultiple > 0);

    if (numThreads < 1) {
        int numCpuThreads = std::thread::hardware_concurrency();
        // One work queue will run using almost all of the CPU power by default, leaving 1 thread available for system
        // stability
        numThreads = numCpuThreads - 1;
        if (numThreads < 3)
            numThreads = 3;
    }
    this->numThreads = numThreads;

    this->inputBufferSize = inputBufferingMultiple * this->numThreads;
    this->outputBufferSize = outputBufferingMultiple * this->numThreads;

    queueOpen = true;
    for (int i = 0; i < numThreads; ++i) {
        threads.push_back(new std::thread(&WorkQueueUnordered<InputType, OutputType>::execute, this));
    }
}

template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::open(std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> &&executor,
                                                     int numThreads,
                                                     unsigned int inputBufferingMultiple,
                                                     unsigned int outputBufferingMultiple) {
    openImpl(numThreads, inputBufferingMultiple, outputBufferingMultiple);
    this->executor.reset(executor.release());
}

template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::open(std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> &executor,
                                                     int numThreads,
                                                     unsigned int inputBufferingMultiple,
                                                     unsigned int outputBufferingMultiple) {
    openImpl(numThreads, inputBufferingMultiple, outputBufferingMultiple);
    this->executor.reset(executor.release());
}

template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::close() {
    {
        std::unique_lock<std::mutex> lck(mtx);
        queueOpen = false;
        notFull.notify_all();
        inputNotEmpty.notify_all();
        outputNotEmpty.notify_all();
    }

    for (unsigned int i = 0; i < threads.size(); ++i)
        threads[i]->join();
    threads.clear();

    if (executor.get() != NULL)
        delete executor.release();
}

template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::execute() {
    InputType argument;

    while (1) {
        {  // locked section
            std::unique_lock<std::mutex> lck(mtx);
            while (inputStorage.empty()) {
                if (!queueOpen)
                    break;
                inputNotEmpty.wait(lck);
            }
            if (!queueOpen) {
                return;
            }
            threadsRunning += 1;

            argument = inputStorage.front();
            inputStorage.pop_front();

            if (monitorExecutorTiming) {
                std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
                if (threadsRunning == 1)  // previously none were running
                    mostRecentOutputObserved = now;
                t_finalPop = now;

                if (inputQueueFullAtOccupancy(inputStorage.size() - 1) && (!outputQueueFullAtOccupancy(threadsRunning,
                                                                                                       outputStorage.size()) ||
                                                                           !hasOutput)) {  // if input queue was full before this pop,
                                                                                           // and is not not backed up on the output side
                    std::chrono::duration<double> elapsedTimeBetweenChangeInOccupancy =
                        std::chrono::duration_cast<std::chrono::duration<double>>(now - t_previousChangeInOccupancy);
                    elapsedTimeInputQueueFull += elapsedTimeBetweenChangeInOccupancy.count();
                }
                t_previousChangeInOccupancy = now;
            }
        }

        const OutputType result = (*executor)(argument);

        {  // locked section
            std::unique_lock<std::mutex> lck(mtx);

            if (monitorExecutorTiming) {
                std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - mostRecentOutputObserved);
                mostRecentOutputObserved = now;
                executorRunningAverageLatency = decayFactor * elapsed.count() + (1.0 - decayFactor) * executorRunningAverageLatency;
            }

            if (hasOutput)
                outputStorage.push_back(result);
            threadsRunning -= 1;
            outputNotEmpty.notify_one();
        }
    }
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::push(const InputType &input) {
    std::unique_lock<std::mutex> lck(mtx);

    while (inputStorage.size() == inputBufferSize || (outputStorage.size() + threadsRunning == outputBufferSize && hasOutput)) {
        if (!queueOpen)
            break;
        notFull.wait(lck);
    }
    if (!queueOpen)
        return false;

    inputStorage.push_back(input);
    if (threadsRunning < numThreads)
        inputNotEmpty.notify_one();

    if (monitorExecutorTiming) {
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        t_previousChangeInOccupancy = now;
        if (!firstPushRecorded) {
            t_firstPush = now;
            firstPushRecorded = true;
        }
    }

    return true;
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::pop(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    while (outputStorage.empty()) {
        if (!queueOpen)
            break;
        outputNotEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    output = outputStorage.front();
    outputStorage.pop_front();

    notFull.notify_one();

    return true;
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::peek(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    while (outputStorage.empty()) {
        if (!queueOpen)
            break;
        outputNotEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    output = outputStorage.front();

    return true;
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::tryPush(const InputType &input) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (inputStorage.size() < inputBufferSize && (outputStorage.size() + threadsRunning < outputBufferSize || !hasOutput)) {
        inputStorage.push_back(input);
        if (threadsRunning < numThreads)
            inputNotEmpty.notify_one();

        if (monitorExecutorTiming) {
            std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            t_previousChangeInOccupancy = now;
            if (!firstPushRecorded) {
                t_firstPush = now;
                firstPushRecorded = true;
            }
        }

        return true;
    }
    return false;
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::tryPop(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!outputStorage.empty()) {
        output = outputStorage.front();
        outputStorage.pop_front();
        notFull.notify_one();
        return true;
    }
    return false;
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::tryPeek(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!outputStorage.empty()) {
        output = outputStorage.front();
        return true;
    }

    return false;
}

template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::beginExecutorPerformanceTiming(double decayFactor) {
    std::unique_lock<std::mutex> lck(mtx);
    this->decayFactor = decayFactor;
    executorRunningAverageLatency = 0.0;
    monitorExecutorTiming = true;

    firstPushRecorded = false;
    elapsedTimeInputQueueFull = 0.0;
}

template <class InputType, class OutputType>
void WorkQueueUnordered<InputType, OutputType>::stopExecutorPerformanceTiming() {
    std::unique_lock<std::mutex> lck(mtx);
    monitorExecutorTiming = false;
}

template <class InputType, class OutputType>
double WorkQueueUnordered<InputType, OutputType>::getRunningAverageExecutorLatency() {
    std::unique_lock<std::mutex> lck(mtx);
    return executorRunningAverageLatency;
}

template <class InputType, class OutputType>
double WorkQueueUnordered<InputType, OutputType>::getTimeTillEmptyIfInputStops() {
    std::unique_lock<std::mutex> lck(mtx);

    if (isEmptyUnlocked())
        return 0;

    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeSpentOnNextElement =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - t_previousChangeInOccupancy);
    double timeLeftOnNextElement = executorRunningAverageLatency - timeSpentOnNextElement.count();
    if (timeLeftOnNextElement < 0.0)
        timeLeftOnNextElement = 0.0;
    return timeLeftOnNextElement + (totalOccupancyUnlocked() - 1) * executorRunningAverageLatency;
}

template <class InputType, class OutputType>
double WorkQueueUnordered<InputType, OutputType>::getPercentageOfTimeInputsAreNotProcessedFastEnoughToKeepUp() {
    std::chrono::duration<double> durationOfOperation = std::chrono::duration_cast<std::chrono::duration<double>>(t_finalPop - t_firstPush);
    return elapsedTimeInputQueueFull / durationOfOperation.count();
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::wouldPipelineLikelyBenefitFromMoreOfThisResource() {
    return getPercentageOfTimeInputsAreNotProcessedFastEnoughToKeepUp() > 0.5;
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::isFull() {
    std::unique_lock<std::mutex> lck(mtx);
    return inputStorage.size() == inputBufferSize || (outputStorage.size() + threadsRunning == outputBufferSize && hasOutput);
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::isEmpty() {
    std::unique_lock<std::mutex> lck(mtx);
    // printf("emptyCheck: inputStorageEmpty %d threadsRunning %d outputStorageEmpty %d\n", inputStorage.empty(),
    // threadsRunning, outputStorage.empty()); fflush(stdout);
    return isEmptyUnlocked();
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::isEmptyUnlocked() {
    return inputStorage.empty() && threadsRunning == 0 && outputStorage.empty();
}

template <class InputType, class OutputType>
int WorkQueueUnordered<InputType, OutputType>::totalOccupancyUnlocked() {
    return inputStorage.size() + threadsRunning + outputStorage.size();
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::isOutputReady() {
    std::unique_lock<std::mutex> lck(mtx);
    return !outputStorage.empty();
}

template <class InputType, class OutputType>
bool WorkQueueUnordered<InputType, OutputType>::isOpen() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueOpen;
}

template <class InputType, class OutputType>
int WorkQueueUnordered<InputType, OutputType>::inputQueueOccupancy() {
    std::unique_lock<std::mutex> lck(mtx);
    return inputStorage.size();
}

template <class InputType, class OutputType>
int WorkQueueUnordered<InputType, OutputType>::inputQueueSize() {
    std::unique_lock<std::mutex> lck(mtx);
    return inputBufferSize;
}

template <class InputType, class OutputType>
int WorkQueueUnordered<InputType, OutputType>::outputQueueOccupancy() {
    std::unique_lock<std::mutex> lck(mtx);
    return outputStorage.size();
}

template <class InputType, class OutputType>
int WorkQueueUnordered<InputType, OutputType>::outputQueueSize() {
    std::unique_lock<std::mutex> lck(mtx);
    return outputBufferSize;
}
