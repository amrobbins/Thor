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

#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

template <class DataType>
class AsyncQueue {
   public:
    AsyncQueue(uint32_t queueSize);
    virtual ~AsyncQueue();
    AsyncQueue(const AsyncQueue &) = delete;
    AsyncQueue &operator=(const AsyncQueue &) = delete;

    // Blocking.
    void open();
    void close();

    // Blocking API.
    // Returns true on success.
    // Returns false if the queue has been closed before the request has been serviced.
    bool push(const DataType &element);
    bool pop(DataType &element);
    bool peek(DataType &element);

    // Nonblocking API. Returns true on success.
    bool tryPush(const DataType &element);
    bool tryPop(DataType &element);
    bool tryPeek(DataType &element);

    // Nonblocking
    bool isOpen();
    bool isFull();   // true when tryPush will fail
    bool isEmpty();  // true when tryPop will fail
    int occupancy();
    int capacity();

   private:
    bool queueOpen;
    uint32_t queueSize;

    std::deque<DataType> storage;

    std::mutex mtx;
    std::condition_variable notEmpty;
    std::condition_variable notFull;
};

template <class DataType>
AsyncQueue<DataType>::AsyncQueue(uint32_t queueSize) {
    queueOpen = false;
    this->queueSize = queueSize;
}

template <class DataType>
AsyncQueue<DataType>::~AsyncQueue() {
    close();
}

// By default the queue opens with num hardware threads - 1, and 4x this number of output buffers
template <class DataType>
void AsyncQueue<DataType>::open() {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    assert(queueSize > 0);

    storage.reserve(queueSize);

    queueOpen = true;
}

template <class DataType>
void AsyncQueue<DataType>::close() {
    std::unique_lock<std::mutex> lck(mtx);

    queueOpen = false;
    storage.reserve(0);
    storage.clear();

    notFull.notify_all();
    notEmpty.notify_all();
}

template <class DataType>
bool AsyncQueue<DataType>::push(const DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    while (storage.size() == queueSize) {
        if (!queueOpen)
            break;
        notFull.wait(lck);
    }
    if (!queueOpen)
        return false;

    storage.push_back(element);
    notEmpty.notify_one();

    return true;
}

template <class DataType>
bool AsyncQueue<DataType>::pop(DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    while (storage.empty()) {
        if (!queueOpen)
            break;
        notEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    element = storage.front();
    storage.pop_front();

    notFull.notify_one();

    return true;
}

template <class DataType>
bool AsyncQueue<DataType>::peek(DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    while (storage.empty()) {
        if (!queueOpen)
            break;
        notEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    element = storage.front();

    return true;
}

template <class DataType>
bool AsyncQueue<DataType>::tryPush(const DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (storage.size() < queueSize) {
        storage.push_back(element);
        notEmpty.notify_one();
        return true;
    }
    return false;
}

template <class DataType>
bool AsyncQueue<DataType>::tryPop(DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!storage.empty()) {
        element = storage.front();
        storage.pop_front();
        notFull.notify_one();
        return true;
    }
    return false;
}

template <class DataType>
bool AsyncQueue<DataType>::tryPeek(DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!storage.empty()) {
        element = storage.front();
        return true;
    }

    return false;
}

template <class DataType>
bool AsyncQueue<DataType>::isFull() {
    std::unique_lock<std::mutex> lck(mtx);
    return storage.size() == queueSize;
}

template <class DataType>
bool AsyncQueue<DataType>::isEmpty() {
    std::unique_lock<std::mutex> lck(mtx);
    // printf("emptyCheck: storageEmpty %d threadsRunning %d storageEmpty %d\n", storage.empty(),
    // threadsRunning, storage.empty()); fflush(stdout);
    return storage.empty();
}

template <class DataType>
bool AsyncQueue<DataType>::isOpen() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueOpen;
}

template <class DataType>
int AsyncQueue<DataType>::occupancy() {
    std::unique_lock<std::mutex> lck(mtx);
    return storage.size();
}

template <class DataType>
int AsyncQueue<DataType>::capacity() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueSize;
}
