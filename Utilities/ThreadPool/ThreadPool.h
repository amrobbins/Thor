#pragma once
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

template <class Processor>
class ThreadPool {
   public:
    ThreadPool(Processor processor, uint32_t numThreads) : processor(std::move(processor)) {
        if (numThreads == 0)
            numThreads = 1;
        workers.reserve(numThreads);
        for (uint32_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this](std::stop_token st) { workerLoop(st); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() { wait(); }

    void wait() {
        for (auto& t : workers) {
            if (t.joinable())
                t.join();
        }
        workers.clear();
    }

    void stop() {
        for (auto& t : workers) {
            t.request_stop();
        }
        wait();
    }

   private:
    void workerLoop(std::stop_token st) {
        // Give each worker its own processor copy
        Processor threadProcessor = processor;

        while (!st.stop_requested()) {
            bool workToDo = threadProcessor.process();
            if (!workToDo)
                break;
        }
    }

    Processor processor;
    std::vector<std::jthread> workers;
};
