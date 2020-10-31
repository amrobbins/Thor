#include "Utilities/WorkQueue/AsyncTensorQueue.h"

using namespace ThorImplementation;

AsyncTensorQueue::AsyncTensorQueue() {
    queueOpen = false;
    this->queueSize = 0;
}

AsyncTensorQueue::AsyncTensorQueue(uint64_t queueSize, TensorDescriptor bufferDescriptor, TensorPlacement bufferPlacement) {
    std::unique_lock<std::mutex> lck(mtx);

    assert(queueSize > 0);

    queueOpen = false;
    this->queueSize = queueSize;
    this->bufferDescriptor = bufferDescriptor;
    this->bufferPlacement = bufferPlacement;

    allocateBuffers();
}

AsyncTensorQueue::~AsyncTensorQueue() { close(); }

void AsyncTensorQueue::resize(uint64_t queueSize, TensorDescriptor bufferDescriptor, TensorPlacement bufferPlacement) {
    std::unique_lock<std::mutex> lck(mtx);

    assert(queueOpen == false);
    assert(queueSize > 0);

    this->queueSize = queueSize;
    this->bufferDescriptor = bufferDescriptor;
    this->bufferPlacement = bufferPlacement;

    allocateBuffers();
}

// By default the queue opens with num hardware threads - 1, and 4x this number of output buffers

void AsyncTensorQueue::open() {
    std::unique_lock<std::mutex> lck(mtx);
    assert(queueOpen == false);
    assert(queueSize > 0);

    queueOpen = true;
}

void AsyncTensorQueue::close() {
    std::unique_lock<std::mutex> lck(mtx);

    queueOpen = false;
    deallocateBuffers();

    notFull.notify_all();
    notEmpty.notify_all();
}

bool AsyncTensorQueue::getBufferToLoad(Tensor &bufferToLoad) {
    std::unique_lock<std::mutex> lck(mtx);

    while (emptyBuffers.empty()) {
        if (!queueOpen)
            break;
        notFull.wait(lck);
    }
    if (!queueOpen)
        return false;

    bufferToLoad = emptyBuffers.back();
    emptyBuffers.pop_back();
    loadingBuffers.insert(bufferToLoad);

    return true;
}

bool AsyncTensorQueue::bufferLoaded(Tensor loadedBuffer) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    assert(loadingBuffers.count(loadedBuffer) == 1);
    loadingBuffers.erase(loadedBuffer);
    loadedBuffers.push_back(loadedBuffer);

    notEmpty.notify_one();

    return true;
}

bool AsyncTensorQueue::getBufferToUnload(Tensor &bufferToUnload) {
    std::unique_lock<std::mutex> lck(mtx);

    while (loadedBuffers.empty()) {
        if (!queueOpen)
            break;
        notEmpty.wait(lck);
    }
    if (!queueOpen)
        return false;

    bufferToUnload = loadedBuffers.front();
    loadedBuffers.pop_front();
    unloadingBuffers.insert(bufferToUnload);

    notFull.notify_one();

    return true;
}

bool AsyncTensorQueue::bufferUnloaded(Tensor unloadedBuffer) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    assert(unloadingBuffers.count(unloadedBuffer) == 1);
    unloadingBuffers.erase(unloadedBuffer);
    emptyBuffers.push_back(unloadedBuffer);

    notFull.notify_one();

    return true;
}

bool AsyncTensorQueue::tryGetBufferToLoad(Tensor &bufferToLoad) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!emptyBuffers.empty()) {
        bufferToLoad = emptyBuffers.back();
        emptyBuffers.pop_back();
        loadingBuffers.insert(bufferToLoad);
        return true;
    }
    return false;
}

bool AsyncTensorQueue::tryGetBufferToUnload(Tensor &bufferToUnload) {
    std::unique_lock<std::mutex> lck(mtx);

    if (!queueOpen)
        return false;

    if (!loadedBuffers.empty()) {
        bufferToUnload = loadedBuffers.front();
        loadedBuffers.pop_front();
        unloadingBuffers.insert(bufferToUnload);
        notFull.notify_one();
        return true;
    }
    return false;
}

bool AsyncTensorQueue::isFull() {
    std::unique_lock<std::mutex> lck(mtx);
    return emptyBuffers.empty();
}

bool AsyncTensorQueue::isEmpty() {
    std::unique_lock<std::mutex> lck(mtx);
    return loadedBuffers.empty();
}

bool AsyncTensorQueue::isOpen() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueOpen;
}

int AsyncTensorQueue::occupancy() {
    std::unique_lock<std::mutex> lck(mtx);
    return loadedBuffers.size();
}

int AsyncTensorQueue::capacity() {
    std::unique_lock<std::mutex> lck(mtx);
    return queueSize;
}

// called only from locked methods
void AsyncTensorQueue::allocateBuffers() {
    assert(queueSize > 0);

    for (uint64_t i = 0; i < queueSize; ++i) {
        emptyBuffers.emplace_back(bufferPlacement, bufferDescriptor);
    }
}

// called only from locked methods
void AsyncTensorQueue::deallocateBuffers() {
    assert(queueOpen == false);
    assert(emptyBuffers.size() + loadingBuffers.size() + loadedBuffers.size() + unloadingBuffers.size() == queueSize);

    emptyBuffers.clear();
    emptyBuffers.shrink_to_fit();
    loadingBuffers.clear();
    loadedBuffers.clear();
    loadedBuffers.shrink_to_fit();
    unloadingBuffers.clear();
}
