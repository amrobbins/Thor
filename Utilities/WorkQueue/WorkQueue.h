#pragma once

#include "WorkQueueExecutorBase.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <deque>
#include <vector>

#include <assert.h>
#include <stdio.h>

// FIXME: use a unique pointer like the unordered queue does

template <class InputType, class OutputType>
class WorkQueue {
   public:
    WorkQueue();
    WorkQueue(const WorkQueue &) = delete;
    WorkQueue &operator=(const WorkQueue &) = delete;

    // Blocking
    void open(WorkQueueExecutorBase<InputType, OutputType> *executor,
              int numThreads = -1,
              unsigned int inputBufferingMultiple = 4,
              unsigned int outputBufferingMultiple = 4);
    void close();

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

   private:
    unsigned int inputBufferSize;
    unsigned int outputBufferSize;

    std::mutex mtx;
    std::vector<std::thread *> threads;
    std::condition_variable inputNotEmpty;
    std::condition_variable outputNotEmpty;
    std::condition_variable notFull;

    unsigned short nextInputOrder = 0;
    std::deque<InputType> inputStorage;
    std::deque<OutputType> outputStorage;
    std::deque<unsigned short> outputOrder;
    std::deque<bool> outputReady;

    bool queueOpen = false;
    unsigned int threadsRunning = 0;

    unsigned int numThreads = 0;
    WorkQueueExecutorBase<InputType, OutputType> *executor = NULL;

    void execute();
    void sendToOutput(const OutputType &output, unsigned short argumentOutputOrder);

    unsigned int computeOutputSlot(unsigned short argumentOutputOrder);
};

template <class InputType, class OutputType>
WorkQueue<InputType, OutputType>::WorkQueue() {}

// By default the queue opens with num hardware threads - 1, and 4x this number of output buffers
template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::open(WorkQueueExecutorBase<InputType, OutputType> *executor,
                                            int numThreads,
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
    this->executor = executor;

    this->inputBufferSize = inputBufferingMultiple * this->numThreads;
    this->outputBufferSize = outputBufferingMultiple * this->numThreads;

    queueOpen = true;
    for (int i = 0; i < numThreads; ++i) {
        threads.push_back(new std::thread(&WorkQueue<InputType, OutputType>::execute, this));
    }
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::close() {
    {
        std::unique_lock<std::mutex> lck(mtx);
        queueOpen = false;
        notFull.notify_all();
        inputNotEmpty.notify_all();
        outputNotEmpty.notify_all();
    }

    for (unsigned int i = 0; i < threads.size(); ++i)
        threads[i]->join();
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::execute() {
    InputType argument;
    unsigned short argumentOutputOrder;

    while (1) {
        {  // locked section
            std::unique_lock<std::mutex> lck(mtx);
            while (inputStorage.empty()) {
                if (!queueOpen)
                    break;
                // printf("execute outputReady.size() %ld, outputReady[0] %d\n", outputReady.size(),
                // !outputReady.empty() && outputReady[0]);
                inputNotEmpty.wait(lck);  // HERE
            }
            if (!queueOpen) {
                return;
            }
            threadsRunning += 1;

            argument = inputStorage.front();
            inputStorage.pop_front();
            argumentOutputOrder = nextInputOrder;
            nextInputOrder += 1;
            outputOrder.push_back(argumentOutputOrder);
        }

        const OutputType result = (*executor)(argument);
        sendToOutput(result, argumentOutputOrder);
    }
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::sendToOutput(const OutputType &output, unsigned short argumentOutputOrder) {
    std::unique_lock<std::mutex> lck(mtx);
    assert(!outputOrder.empty());

    // printf("send to output 0\n"); fflush(stdout);
    unsigned int outputSlot = computeOutputSlot(argumentOutputOrder);
    while (outputStorage.size() <= outputSlot) {
        outputStorage.push_back(OutputType());
        outputReady.push_back(false);
    }
    outputStorage[outputSlot] = output;
    outputReady[outputSlot] = true;
    threadsRunning -= 1;
    // more than one output can become available at a time due to ordering requirement.
    if (outputSlot == 0)
        outputNotEmpty.notify_all();
    // printf("send output just notified outputReady.size() %ld, outputReady[0] %d\n", outputReady.size(),
    // !outputReady.empty() && outputReady[0]);

    // printf("send to output end %d:%d\n", outputSlot, outputReady[0]); fflush(stdout);
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::push(const InputType &input) {
    // printf("push 0\n"); fflush(stdout);
    std::unique_lock<std::mutex> lck(mtx);

    while (inputStorage.size() == inputBufferSize || computeOutputSlot(nextInputOrder) == outputBufferSize) {
        if (!queueOpen)
            break;
        notFull.wait(lck);
    }
    // printf("push 1\n"); fflush(stdout);
    if (!queueOpen)
        return false;
    // printf("push 2\n"); fflush(stdout);

    inputStorage.push_back(input);
    if (threadsRunning < numThreads)
        inputNotEmpty.notify_one();

    // printf("push 3\n"); fflush(stdout);
    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::pop(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    while (outputReady.empty() || !outputReady[0]) {
        if (!queueOpen)
            break;
        // printf("pop sleeping outputReady.size() %ld, outputReady[0] %d\n", outputReady.size(), !outputReady.empty()
        // && outputReady[0]);
        outputNotEmpty.wait(lck);  // HERE
        // printf("pop waking outputReady.size() %ld, outputReady[0] %d\n", outputReady.size(), !outputReady.empty() &&
        // outputReady[0]);
    }
    if (!queueOpen)
        return false;

    output = outputStorage.front();
    outputStorage.pop_front();
    outputReady.pop_front();
    outputOrder.pop_front();

    notFull.notify_one();

    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::peek(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    while (outputReady.empty() || !outputReady[0]) {
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
bool WorkQueue<InputType, OutputType>::tryPush(const InputType &input) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    // printf("inputBuffersInUse %d   %d <= %d && %d < %d && %d\n", inputBufferSize, threadsRunning, numThreads,
    // computeOutputSlot(nextInputOrder), numThreads, queueOpen);
    if (inputStorage.size() < inputBufferSize && computeOutputSlot(nextInputOrder) < outputBufferSize) {
        inputStorage.push_back(input);
        if (threadsRunning < numThreads)
            inputNotEmpty.notify_one();
        return true;
    }
    return false;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPop(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!outputReady.empty() && outputReady[0]) {
        output = outputStorage.front();
        outputStorage.pop_front();
        outputReady.pop_front();
        outputOrder.pop_front();
        notFull.notify_one();
        return true;
    }
    return false;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPeek(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!outputReady.empty() && outputReady[0]) {
        output = outputStorage.front();
        return true;
    }

    return false;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isFull() {
    std::unique_lock<std::mutex> lck(mtx);
    return inputStorage.size() == inputBufferSize || computeOutputSlot(nextInputOrder) >= outputBufferSize;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isEmpty() {
    std::unique_lock<std::mutex> lck(mtx);
    // printf("emptyCheck: inputStorageEmpty %d threadsRunning %d outputReadyEmpty %d\n", inputStorage.empty(),
    // threadsRunning, outputReady.empty()); fflush(stdout);
    return inputStorage.empty() && threadsRunning == 0 && outputReady.empty();
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isOutputReady() {
    std::unique_lock<std::mutex> lck(mtx);
    return !outputReady.empty() && outputReady[0];
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isOpen() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueOpen;
}

template <class InputType, class OutputType>
unsigned int WorkQueue<InputType, OutputType>::computeOutputSlot(unsigned short argumentOutputOrder) {
    int outputSlot;
    if (outputOrder.empty())
        outputSlot = argumentOutputOrder;
    else
        outputSlot = argumentOutputOrder - outputOrder[0];
    // need to handle rollover case
    if (outputSlot < 0)
        outputSlot += std::numeric_limits<unsigned short>::max() + 1;
    return outputSlot;
}
