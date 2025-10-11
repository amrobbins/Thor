#include "Thor.h"

#include <math.h>
#include <chrono>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

struct OutputStruct {
    std::thread::id id;
    std::string s;
    unsigned int num;
};

OutputStruct doWork(int size, std::string input) {
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

TEST(WorkQueueTest, OutputIsCorrect) {
    constexpr int EXECUTOR_WORK_SIZE = 10000;

    WorkQueue<std::string, OutputStruct> workQueue;

    std::unique_ptr<WorkQueueExecutorBase<std::string, OutputStruct>> executor(new Executor(EXECUTOR_WORK_SIZE));
    workQueue.open(executor, 11);

    // std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    std::vector<OutputStruct> out;
    bool pushSuccess = false;
    for (int i = 0; i < 10000; i += pushSuccess ? 1 : 0) {
        pushSuccess = workQueue.tryPush(std::to_string(i));
        OutputStruct thisOut;
        bool popSuccess = workQueue.tryPop(thisOut);
        if (popSuccess)
            out.push_back(thisOut);
    }
    while (!workQueue.isEmpty()) {
        OutputStruct thisOut;
        bool popSuccess = workQueue.pop(thisOut);
        if (popSuccess)
            out.push_back(thisOut);
    }

    // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    // printf("Execution time %lf seconds\n", elapsed.count());

    for (unsigned int i = 0; i < out.size(); ++i) {
        EXPECT_EQ(out[i].num, i);
    }

    workQueue.close();
}

TEST(WorkQueueTest, tryPushAllTryPopAll) {
    constexpr int EXECUTOR_WORK_SIZE = 10000;

    WorkQueue<std::string, OutputStruct> workQueue;

    std::unique_ptr<WorkQueueExecutorBase<std::string, OutputStruct>> executor(new Executor(EXECUTOR_WORK_SIZE));
    workQueue.open(executor, 11, 10, 10);

    std::vector<OutputStruct> out;
    bool pushSuccess;
    bool popSuccess;
    std::deque<std::string> inputDeque;
    for (int i = 0; i < 200; ++i)
        inputDeque.push_back(std::to_string(i));

    pushSuccess = workQueue.tryPushAll(inputDeque);
    EXPECT_EQ(pushSuccess, true);
    EXPECT_EQ(inputDeque.size(), 90u);
    while (out.size() < 110) {
        popSuccess = workQueue.tryPopAll(out);
        EXPECT_EQ(popSuccess, true);
    }

    pushSuccess = workQueue.tryPushAll(inputDeque);
    EXPECT_EQ(pushSuccess, true);
    EXPECT_EQ(inputDeque.size(), 0u);
    while (out.size() < 200u) {
        popSuccess = workQueue.tryPopAll(out);
        EXPECT_EQ(popSuccess, true);
    }

    EXPECT_EQ(out.size(), 200u);
    for (unsigned int i = 0; i < out.size(); ++i) {
        EXPECT_EQ(out[i].num, i);
    }

    workQueue.close();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
