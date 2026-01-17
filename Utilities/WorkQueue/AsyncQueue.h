#pragma once

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

template <class DataType>
class AsyncQueue {
   public:
    AsyncQueue();
    explicit AsyncQueue(uint32_t queueSize);
    explicit AsyncQueue(std::deque<DataType> &&initialStorage);
    virtual ~AsyncQueue();
    AsyncQueue(const AsyncQueue &) = delete;
    AsyncQueue &operator=(const AsyncQueue &) = delete;

    void resize(uint32_t queueSize);

    // Blocking.
    void open();
    void close();
    void waitForEmpty();

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
    bool queueOpen = false;
    uint32_t queueSize = 0;

    std::deque<DataType> storage;

    std::mutex mtx;
    std::condition_variable notEmpty;
    std::condition_variable notFull;
    std::condition_variable becameEmpty;
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
AsyncQueue<DataType>::AsyncQueue(std::deque<DataType> &&initialStorage) : storage(std::move(initialStorage)) {
    queueOpen = false;
    this->queueSize = storage.size();
}

template <class DataType>
AsyncQueue<DataType>::~AsyncQueue() {
    bool errorQueueNotClosed = false;
    {
        std::unique_lock<std::mutex> lck(mtx);
        if (queueOpen) {
            errorQueueNotClosed = true;
        }
    }
    if (errorQueueNotClosed) {
        // Close queue, wake all, terminate with an error message -> fix the code so you don't hit it.
        close();
        sleep(2);
        std::fprintf(stderr, "FATAL: AsyncQueue destroyed while still open. Close queue and join worker threads before destruction.\n");
        std::fflush(stderr);
        std::terminate();
    }
}

template <class DataType>
void AsyncQueue<DataType>::resize(uint32_t queueSize) {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    this->queueSize = queueSize;
}

template <class DataType>
void AsyncQueue<DataType>::open() {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    assert(queueSize > 0);

    queueOpen = true;
}

// No more pushes
template <class DataType>
void AsyncQueue<DataType>::close() {
    std::unique_lock<std::mutex> lck(mtx);

    queueOpen = false;
    storage.resize(0);

    notFull.notify_all();
    notEmpty.notify_all();
    becameEmpty.notify_all();
}

template <class DataType>
void AsyncQueue<DataType>::waitForEmpty() {
    std::unique_lock<std::mutex> lck(mtx);
    while (!storage.empty()) {
        becameEmpty.wait_for(lck, std::chrono::seconds(5), [&] { return storage.empty(); });
    }
}

template <class DataType>
bool AsyncQueue<DataType>::push(const DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    while (queueOpen && storage.size() == queueSize) {
        notFull.wait_for(lck, std::chrono::seconds(5), [&] { return !queueOpen || storage.size() != queueSize; });
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

    while (queueOpen && storage.empty()) {
        notEmpty.wait_for(lck, std::chrono::seconds(5), [&] { return !queueOpen || !storage.empty(); });
    }
    if (!queueOpen)
        return false;

    element = storage.front();
    storage.pop_front();

    notFull.notify_one();

    if (storage.empty())
        becameEmpty.notify_all();

    return true;
}

template <class DataType>
bool AsyncQueue<DataType>::peek(DataType &element) {
    std::unique_lock<std::mutex> lck(mtx);

    while (queueOpen && storage.empty()) {
        notEmpty.wait_for(lck, std::chrono::seconds(5), [&] { return !queueOpen || !storage.empty(); });
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

        if (storage.empty())
            becameEmpty.notify_all();

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
