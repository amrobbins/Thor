#pragma once

#include "WorkQueueExecutorBase.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <deque>
#include <vector>

#include <assert.h>
#include <stdio.h>

template <class InputType, class OutputType>
class WorkQueue {
   public:
    WorkQueue();
    WorkQueue(const WorkQueue &) = delete;
    WorkQueue &operator=(const WorkQueue &) = delete;
    virtual ~WorkQueue() { close(); }

    // Blocking
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

    // Blocking API.
    // Returns true on success.
    // Returns false if the queue has been closed before the request has been serviced.
    bool push(const InputType input);
    bool pop(OutputType &output);
    bool peek(OutputType &output);

    // Nonblocking API. Returns true on success.
    bool tryPush(const InputType &input);
    bool tryPop(OutputType &output);
    bool tryPeek(OutputType &output);
    bool tryPushAll(std::deque<InputType> &input);
    bool tryPopAll(std::vector<OutputType> &output, uint32_t maxToPop = 0);

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
    std::condition_variable inputNotFull;
    std::condition_variable outputNotFull;

    uint32_t inputHead;
    uint32_t inputTail;
    bool inputEmpty;
    bool inputFull;
    uint32_t outputHead;
    uint32_t outputTail;
    bool outputEmpty;
    bool outputFull;
    InputType *inputStorage = nullptr;
    OutputType *outputStorage = nullptr;
    bool *outputReady = nullptr;

    uint32_t numThreadsWaitingForInputQueueNonEmpty;
    uint32_t numThreadsWaitingForOutputQueueNonFull;

    bool queueOpen = false;

    unsigned int numThreads = 0;
    std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> executor;

    void execute();

    void openImpl(int numThreads, unsigned int inputBufferingMultiple, unsigned int outputBufferingMultiple);

    uint32_t computeNextInputQueueSlot(uint32_t currentSlot) {
        uint32_t nextSlot = currentSlot + 1;
        if (nextSlot == inputBufferSize)
            nextSlot = 0;
        return nextSlot;
    }

    void popInputQueue() {
        assert(!inputEmpty);
        inputHead = computeNextInputQueueSlot(inputHead);
        if (inputHead == inputTail)
            inputEmpty = true;
        inputFull = false;
    }

    void pushInputQueue() {
        assert(!inputFull);
        inputTail = computeNextInputQueueSlot(inputTail);
        if (inputHead == inputTail)
            inputFull = true;
        inputEmpty = false;
    }

    uint32_t computeNextOutputQueueSlot(uint32_t currentSlot) {
        uint32_t nextSlot = currentSlot + 1;
        if (nextSlot == outputBufferSize)
            nextSlot = 0;
        return nextSlot;
    }

    void popOutputQueue() {
        assert(!outputEmpty);
        outputHead = computeNextOutputQueueSlot(outputHead);
        if (outputHead == outputTail)
            outputEmpty = true;
        outputFull = false;
    }

    void pushOutputQueue() {
        assert(!outputFull);
        outputTail = computeNextOutputQueueSlot(outputTail);
        if (outputHead == outputTail)
            outputFull = true;
        outputEmpty = false;
    }
};

template <class InputType, class OutputType>
WorkQueue<InputType, OutputType>::WorkQueue() {}

// By default the queue opens with num hardware threads - 1, and 4x this number of output buffers
template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::openImpl(int numThreads, unsigned int inputBufferingMultiple, unsigned int outputBufferingMultiple) {
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

    inputBufferSize = inputBufferingMultiple * this->numThreads;
    outputBufferSize = outputBufferingMultiple * this->numThreads;

    inputHead = 0;
    inputTail = 0;
    inputEmpty = true;
    inputFull = false;
    outputHead = 0;
    outputTail = 0;
    outputEmpty = true;
    outputFull = false;

    inputStorage = new InputType[inputBufferSize];
    outputStorage = new OutputType[outputBufferSize];
    outputReady = new bool[outputBufferSize];

    numThreadsWaitingForInputQueueNonEmpty = 0;
    numThreadsWaitingForOutputQueueNonFull = 0;
    queueOpen = true;
    // Ensure all state is visible to any newly created threads:
    for (int i = 0; i < numThreads; ++i) {
        threads.push_back(new std::thread(&WorkQueue<InputType, OutputType>::execute, this));
    }
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::open(std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> &&executor,
                                            int numThreads,
                                            unsigned int inputBufferingMultiple,
                                            unsigned int outputBufferingMultiple) {
    assert(!queueOpen);
    openImpl(numThreads, inputBufferingMultiple, outputBufferingMultiple);
    this->executor.reset(executor.release());
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::open(std::unique_ptr<WorkQueueExecutorBase<InputType, OutputType>> &executor,
                                            int numThreads,
                                            unsigned int inputBufferingMultiple,
                                            unsigned int outputBufferingMultiple) {
    assert(!queueOpen);
    openImpl(numThreads, inputBufferingMultiple, outputBufferingMultiple);
    this->executor.reset(executor.release());
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::close() {
    if (!queueOpen)
        return;

    {
        std::unique_lock<std::mutex> lck(mtx);
        queueOpen = false;
        inputNotFull.notify_all();
        outputNotFull.notify_all();
        inputNotEmpty.notify_all();
        outputNotEmpty.notify_all();
    }

    for (unsigned int i = 0; i < threads.size(); ++i)
        threads[i]->join();
    threads.clear();

    if (executor.get() != nullptr)
        delete executor.release();

    delete[] inputStorage;
    delete[] outputStorage;
    delete[] outputReady;
    inputStorage = nullptr;
    outputStorage = nullptr;
    outputReady = nullptr;
}

template <class InputType, class OutputType>
void WorkQueue<InputType, OutputType>::execute() {
    InputType argument;

    uint32_t inputQueueSlot;
    uint32_t outputQueueSlot;

    while (1) {
        {  // locked section
            // read from input queue, note output queue location
            std::unique_lock<std::mutex> lck(mtx);
            // Must acquire both an input entry and an output slot atomically:
            while (inputEmpty || outputFull) {
                if (!queueOpen)
                    break;
                if (inputEmpty) {
                    numThreadsWaitingForInputQueueNonEmpty += 1;
                    inputNotEmpty.wait(lck);
                    numThreadsWaitingForInputQueueNonEmpty -= 1;
                } else {
                    numThreadsWaitingForOutputQueueNonFull += 1;
                    outputNotFull.wait(lck);
                    numThreadsWaitingForOutputQueueNonFull -= 1;
                }
            }
            if (!queueOpen) {
                return;
            }

            bool inputQueueWasFull = inputFull;
            inputQueueSlot = inputHead;
            outputQueueSlot = outputTail;
            popInputQueue();
            pushOutputQueue();

            argument = inputStorage[inputQueueSlot];
            outputReady[outputQueueSlot] = false;

            // Inform 1 external thread (not a thread pool thread) waiting for input queue to be non-full that it is not
            // This is valid because the only reason that an external thread can be waiting on inputNotFull is for an
            // already invoked push to complete.
            if (inputQueueWasFull)
                inputNotFull.notify_one();
        }

        const OutputType result = (*executor)(argument);

        {  // locked section
            // write to output queue
            // notify anyone waiting on empty output queue that a job has finished
            std::unique_lock<std::mutex> lck(mtx);
            outputStorage[outputQueueSlot] = result;
            outputReady[outputQueueSlot] = true;
            // Inform all external threads (not thread pool threads) waiting for output that output is ready
            // Due to ordering requirement, multiple outputs can become ready at once
            // Also any number of threads can be waiting for peek, so this needs to be notify_all
            outputNotEmpty.notify_all();
        }
    }
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::push(const InputType input) {
    std::unique_lock<std::mutex> lck(mtx);

    while (inputFull) {
        if (!queueOpen)
            break;
        inputNotFull.wait(lck);
    }
    if (!queueOpen)
        return false;

    inputStorage[inputHead] = input;
    pushInputQueue();
    if (numThreadsWaitingForInputQueueNonEmpty > 0)
        inputNotEmpty.notify_one();

    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::pop(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    while (outputEmpty || !outputReady[outputHead]) {
        if (!queueOpen)
            break;
        outputNotEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    output = outputStorage[outputHead];
    popOutputQueue();
    if (numThreadsWaitingForOutputQueueNonFull > 0)
        outputNotFull.notify_one();

    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::peek(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    while (outputEmpty || !outputReady[outputHead]) {
        if (!queueOpen)
            break;
        outputNotEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    output = outputStorage[outputHead];

    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPush(const InputType &input) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!inputFull) {
        inputStorage[inputTail] = input;
        pushInputQueue();
        if (numThreadsWaitingForInputQueueNonEmpty > 0)
            inputNotEmpty.notify_one();
        return true;
    }
    return false;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPushAll(std::deque<InputType> &input) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    uint32_t numNotifications = 0;
    while (!input.empty()) {
        if (!inputFull) {
            inputStorage[inputTail] = input[0];
            input.pop_front();
            pushInputQueue();
            if (numThreadsWaitingForInputQueueNonEmpty > numNotifications) {
                inputNotEmpty.notify_one();
                numNotifications += 1;
            }
        } else {
            break;
        }
    }
    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPop(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!outputEmpty && outputReady[outputHead]) {
        output = outputStorage[outputHead];
        popOutputQueue();
        if (numThreadsWaitingForOutputQueueNonFull > 0)
            outputNotFull.notify_one();
        return true;
    }
    return false;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPopAll(std::vector<OutputType> &output, uint32_t maxToPop) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    uint32_t numNotifications = 0;
    uint32_t numPops = 0;
    while (!outputEmpty && outputReady[outputHead]) {
        output.push_back(outputStorage[outputHead]);
        popOutputQueue();
        if (numThreadsWaitingForOutputQueueNonFull > numNotifications) {
            outputNotFull.notify_one();
            numNotifications += 1;
        }
        if (maxToPop != 0) {
            numPops += 1;
            if (numPops == maxToPop)
                break;
        }
    }
    return true;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::tryPeek(OutputType &output) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!outputEmpty && outputReady[outputHead]) {
        output = outputStorage[outputHead];
        return true;
    }

    return false;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isFull() {
    std::unique_lock<std::mutex> lck(mtx);
    return inputFull && outputFull;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isEmpty() {
    std::unique_lock<std::mutex> lck(mtx);
    return inputEmpty && outputEmpty;
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isOutputReady() {
    std::unique_lock<std::mutex> lck(mtx);
    return !outputEmpty && outputReady[outputHead];
}

template <class InputType, class OutputType>
bool WorkQueue<InputType, OutputType>::isOpen() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueOpen;
}