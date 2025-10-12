#include "Thor.h"

#include <math.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "gtest/gtest.h"

struct OutputStruct {
    std::thread::id id;
    std::string s;
    int num;
};

static OutputStruct doWork(int size, std::string input) {
    OutputStruct output;
    output.id = std::this_thread::get_id();
    output.num = atoi(input.c_str());
    double r = 0.0;
    for (int i = 1; i < size; ++i)
        r += exp(3.2) * pow(output.num, 1.5) / pow(output.num, 1.25);
    output.s = std::to_string(r) + " executed by thread ";

    return output;
}

class Executor : public WorkQueueExecutorBase<std::string, OutputStruct> {
   public:
    Executor(int workSize) { this->workSize = workSize; }

    OutputStruct operator()(std::string &input) { return doWork(workSize, input); }

   private:
    int workSize;
};

void tryPushThread(WorkQueueUnordered<std::string, OutputStruct> *workQueue, int numItems, int workSize) {
    bool pushSuccess = false;
    for (int i = 0; i < numItems; i += pushSuccess ? 1 : 0) {
        pushSuccess = workQueue->tryPush(std::to_string(i));
        if (pushSuccess)
            doWork(workSize, std::to_string(i));  // simulate previous pipeline stage delay
    }
}

void tryPopThread(WorkQueueUnordered<std::string, OutputStruct> *workQueue, int numItems, int workSize) {
    std::vector<OutputStruct> out;
    bool popSuccess = false;
    for (int i = 0; i < numItems; i += popSuccess ? 1 : 0) {
        OutputStruct thisOut;
        popSuccess = workQueue->tryPop(thisOut);
        if (popSuccess) {
            out.push_back(thisOut);
            doWork(workSize, std::to_string(i));
        }
    }

    std::unordered_set<int> jobs;
    for (int i = 0; i < numItems; ++i)
        jobs.insert(i);
    for (unsigned int i = 0; i < out.size(); ++i) {
        jobs.erase(out[i].num);
    }
    assert(jobs.empty());
}

void pushThread(WorkQueueUnordered<std::string, OutputStruct> *workQueue, int numItems, int workSize) {
    for (int i = 0; i < numItems; ++i) {
        bool success = workQueue->push(std::to_string(i));
        assert(success);
        doWork(workSize, std::to_string(i));  // simulate previous pipeline stage delay
    }
}

void popThread(WorkQueueUnordered<std::string, OutputStruct> *workQueue, int numItems, int workSize) {
    std::vector<OutputStruct> out;
    for (int i = 0; i < numItems; ++i) {
        OutputStruct thisOut;
        bool success = workQueue->pop(thisOut);
        assert(success);
        out.push_back(thisOut);
        doWork(workSize, std::to_string(i));
    }

    std::unordered_set<int> jobs;
    for (int i = 0; i < numItems; ++i)
        jobs.insert(i);
    for (unsigned int i = 0; i < out.size(); ++i) {
        jobs.erase(out[i].num);
    }
    ASSERT_TRUE(jobs.empty());
}

TEST(WorkQueueUnorderedTest, OutputIsCorrect) {
    constexpr int NUM_THREADS = 5;
    constexpr int NUM_ITERATIONS = 4000;

    constexpr int INPUT_WORK_SIZE = 2200;
    constexpr int EXECUTOR_WORK_SIZE = 10000;
    constexpr int OUTPUT_WORK_SIZE = 1600;

    WorkQueueUnordered<std::string, OutputStruct> workQueue;

    std::unique_ptr<WorkQueueExecutorBase<std::string, OutputStruct>> executor(new Executor(EXECUTOR_WORK_SIZE));
    // printf("before executor %p\n", executor.get());
    workQueue.open(executor, NUM_THREADS);
    // printf("after executor %p\n", executor.get());
    assert(executor.get() == NULL);
    workQueue.beginExecutorPerformanceTiming();

    // std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    threads.emplace_back(popThread, &workQueue, NUM_ITERATIONS, OUTPUT_WORK_SIZE);
    threads.emplace_back(pushThread, &workQueue, NUM_ITERATIONS, INPUT_WORK_SIZE);
    while (!threads.empty()) {
        threads.back().join();
        threads.pop_back();
    }

    /*
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
        printf("Execution time %lf seconds\n", elapsed.count());

        double avgLatency = workQueue.getRunningAverageExecutorLatency();
        printf("Average executor latency: %lf seconds. Total executor latency based on running average: %lf seconds\n",
       avgLatency, avgLatency * NUM_ITERATIONS);

        printf("Num Threads %d, Input Work %d, Executor Work %d (%d per thread), Output Work %d.\nPercent of time inputs
       are not being processed fast enough %lf. Would pipeline benefit from more threads, at this (loading, processing,
       unloading) rate? %c\n", NUM_THREADS, INPUT_WORK_SIZE, EXECUTOR_WORK_SIZE, EXECUTOR_WORK_SIZE / NUM_THREADS,
       OUTPUT_WORK_SIZE, workQueue.getPercentageOfTimeInputsAreNotProcessedFastEnoughToKeepUp(),
       workQueue.wouldPipelineLikelyBenefitFromMoreOfThisResource() ? 'Y' : 'N');
    */

    workQueue.close();
}
