#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

class AsyncTensorQueue {
   public:
    AsyncTensorQueue();
    AsyncTensorQueue(uint64_t queueSize,
                     ThorImplementation::TensorDescriptor bufferDescriptor,
                     ThorImplementation::TensorPlacement bufferPlacement);
    virtual ~AsyncTensorQueue();
    AsyncTensorQueue(const AsyncTensorQueue &) = delete;
    AsyncTensorQueue &operator=(const AsyncTensorQueue &) = delete;

    void resize(uint64_t queueSize,
                ThorImplementation::TensorDescriptor bufferDescriptor,
                ThorImplementation::TensorPlacement bufferPlacement);

    // Blocking.
    void open();
    void close();

    // Blocking API.
    // Returns true on success.
    // Returns false if the queue has been closed before the request has been serviced.
    bool getBufferToLoad(ThorImplementation::Tensor &bufferToLoad);
    bool getBufferToUnload(ThorImplementation::Tensor &bufferToUnload);

    // Nonblocking API. Returns true on success.
    bool tryGetBufferToLoad(ThorImplementation::Tensor &bufferToLoad);
    bool tryGetBufferToUnload(ThorImplementation::Tensor &bufferToUnload);

    // Nonblocking
    bool bufferLoaded(ThorImplementation::Tensor loadedBuffer);
    bool bufferUnloaded(ThorImplementation::Tensor unloadedBuffer);

    // Nonblocking
    bool isOpen();
    bool isFull();   // true when tryGetBufferToLoad will fail
    bool isEmpty();  // true when tryPop will fail
    int occupancy();
    int capacity();

   private:
    bool queueOpen;
    uint64_t queueSize;
    ThorImplementation::TensorDescriptor bufferDescriptor;
    ThorImplementation::TensorPlacement bufferPlacement;

    std::vector<ThorImplementation::Tensor> emptyBuffers;
    std::set<ThorImplementation::Tensor> loadingBuffers;
    std::deque<ThorImplementation::Tensor> loadedBuffers;
    std::set<ThorImplementation::Tensor> unloadingBuffers;

    std::mutex mtx;
    std::condition_variable notEmpty;
    std::condition_variable notFull;

    void allocateBuffers();
    void deallocateBuffers();
};
