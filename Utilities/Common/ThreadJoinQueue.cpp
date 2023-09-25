#include "Utilities/Common/ThreadJoinQueue.h"

using namespace std;

queue<thread> ThreadJoinQueue::threadQueue;
future<void> ThreadJoinQueue::joiningThread;
mutex ThreadJoinQueue::mtx;

ThreadJoinQueue& ThreadJoinQueue::instance() {
    // Guaranteed to be destroyed. Instantiated on first use.
    static ThreadJoinQueue singletonInstance;
    return singletonInstance;
}

ThreadJoinQueue::~ThreadJoinQueue() {
    // Wait for the queue to empty:
    unique_lock<mutex> lock(mtx);
}

void ThreadJoinQueue::push(thread&& t) {
    unique_lock<mutex> lock(mtx);
    atomic_thread_fence(memory_order_release);
    threadQueue.push(move(t));
    if (joiningThread.valid()) {
        // If it had previously finished, then re-launch it. Otherwise don't need to do anything.
        if (joiningThread.wait_for(chrono::seconds(0)) == future_status::ready) {
            joiningThread.get();
            joiningThread = async(launch::async, ThreadJoinQueue::joinAllThreads);
        }
    } else {
        // First time so launch it.
        joiningThread = async(launch::async, ThreadJoinQueue::joinAllThreads);
    }
}

void ThreadJoinQueue::joinAllThreads() {
    unique_lock<mutex> lock(mtx);
    while (!threadQueue.empty()) {
        thread& t = threadQueue.front();
        lock.unlock();
        t.join();
        lock.lock();
        threadQueue.pop();
    }
}