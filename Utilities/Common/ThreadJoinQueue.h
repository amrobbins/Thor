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

//// ThreadJoinQueue is not instantiable
// class ThreadJoinQueue {
//    public:
//     static void push(std::thread&& t);
//
//    private:
//     static void joinAllThreads();
//
//     static std::queue<std::thread> threadQueue;
//     static std::future<void> joiningThread;
//     static std::mutex mtx;
//
//    private:
//     // Private constructor to prevent direct instantiation
//     ThreadJoinQueue();
//     ThreadJoinQueue(const ThreadJoinQueue&) = delete;
//     ThreadJoinQueue& operator=(const ThreadJoinQueue&) = delete;
//
//     friend class Stream;
// };

//
//// ThreadJoinQueue is a singleton
// class ThreadJoinQueue {
//    private:
//     static std::queue<std::thread> threadQueue;
//
//     ThreadJoinQueue();  // Private constructor to prevent direct instantiation
//
//    public:
//     static ThreadJoinQueue& instance();
//
//     // Forbid copying since ThreadJoinQueue is a singleton
//     ThreadJoinQueue(const ThreadJoinQueue&) = delete;
//     ThreadJoinQueue& operator=(const ThreadJoinQueue&) = delete;
//
//     //~ThreadJoinQueue();
//
//     static void push(std::thread&& t);
//
//    private:
//     static void joinAllThreads();
//
//     static std::thread joiningThread;
//     static std::mutex mtx;
//
//     static ThreadJoinQueue *staticInstance;
// };