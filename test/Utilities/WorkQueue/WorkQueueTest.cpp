#include "Thor.h"

#include <math.h>
#include <chrono>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

struct OutputStruct {
    std::thread::id id;
    std::string s;
    unsigned int num;
};

class Executor : public WorkQueueExecutorBase<std::string, OutputStruct> {
   public:
    OutputStruct operator()(std::string &input) {
        OutputStruct output;
        output.id = std::this_thread::get_id();
        output.num = atoi(input.c_str());
        double r = 0.0;
        for (int i = 1; i < 10000; ++i)
            r += exp(3.2) * pow(output.num, 1.5) / pow(output.num, 1.25);
        output.s = std::to_string(r) + " executed by thread ";

        return output;
    }
};

TEST(WorkQueueTest, OutputIsCorrect) {
    WorkQueue<std::string, OutputStruct> workQueue;

    Executor *executor = new Executor();
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
        ASSERT_EQ(out[i].num, i);
    }

    workQueue.close();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
