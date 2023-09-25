#pragma once

#include <atomic>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

// ThreadJoinQueue is not instantiable
class ThreadJoinQueue {
   public:
    static ThreadJoinQueue& instance();
    void push(std::thread&& t);

    virtual ~ThreadJoinQueue();

   private:
    static void joinAllThreads();

    static std::queue<std::thread> threadQueue;
    static std::future<void> joiningThread;
    static std::mutex mtx;

   private:
    // Private constructor to prevent direct instantiation
    ThreadJoinQueue() {}
    ThreadJoinQueue(const ThreadJoinQueue&) = delete;
    ThreadJoinQueue& operator=(const ThreadJoinQueue&) = delete;

    friend class Stream;
};