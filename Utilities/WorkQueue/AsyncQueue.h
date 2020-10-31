#pragma once

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
    AsyncQueue();
    AsyncQueue(uint32_t queueSize);
    virtual ~AsyncQueue();
    AsyncQueue(const AsyncQueue &) = delete;
    AsyncQueue &operator=(const AsyncQueue &) = delete;

    void resize(uint32_t resize);

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
AsyncQueue<DataType>::AsyncQueue() {
    queueOpen = false;
    this->queueSize = 0;
}

template <class DataType>
AsyncQueue<DataType>::AsyncQueue(uint32_t queueSize) {
    queueOpen = false;
    this->queueSize = queueSize;
}

template <class DataType>
AsyncQueue<DataType>::~AsyncQueue() {
    close();
}

template <class DataType>
void AsyncQueue<DataType>::resize(uint32_t queueSize) {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    this->queueSize = queueSize;
}

// By default the queue opens with num hardware threads - 1, and 4x this number of output buffers
template <class DataType>
void AsyncQueue<DataType>::open() {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    assert(queueSize > 0);

    queueOpen = true;
}

template <class DataType>
void AsyncQueue<DataType>::close() {
    std::unique_lock<std::mutex> lck(mtx);

    queueOpen = false;
    storage.resize(0);

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
