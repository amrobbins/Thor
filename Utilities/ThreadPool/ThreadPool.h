#pragma once
#include <atomic>
#include <cstdint>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

template <class Processor, class WorkItem, class Context>
class ThreadPool {
   public:
    ThreadPool(std::vector<WorkItem> workItems, Context& context, uint32_t numThreads)
        : workItems(std::move(workItems)), context(context), nextIndex(0) {
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
        for (auto& t : workers)
            t.request_stop();
        wait();
    }

   private:
    void workerLoop(std::stop_token st) {
        // Give each worker its own processor copy
        Processor threadProcessor(context);

        while (!st.stop_requested()) {
            const size_t idx = nextIndex.fetch_add(1);
            if (idx >= workItems.size())
                break;

            // Each item is processed by exactly one thread.
            threadProcessor.process(workItems[idx]);
        }
    }

    std::vector<WorkItem> workItems;
    Context& context;
    std::atomic<size_t> nextIndex;
    std::vector<std::jthread> workers;
};
