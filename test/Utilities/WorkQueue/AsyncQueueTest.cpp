#include "Utilities/WorkQueue/AsyncQueue.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "omp.h"

using std::mutex;
using std::pair;
using std::set;
using std::string;
using std::vector;

TEST(AsyncQueue, blockingApiWorks) {
    uint32_t numProcs = omp_get_num_procs() > 12 ? 12 : omp_get_num_procs();
    omp_set_num_threads(numProcs - 1);
    srand(time(nullptr));

    mutex mtx;

    for (int test = 0; test < 1; ++test) {
        AsyncQueue<uint32_t> queue(100);
        set<uint32_t> observedNumbers;

        const uint32_t numIters = 100000;

        queue.open();
#pragma omp parallel for schedule(static, 1)
        for (uint32_t i = 0; i < numIters; ++i) {
            bool status;
            status = queue.push(i);
            assert(status == true);

            if (queue.occupancy() > 95) {
                uint32_t val;
                if (i % 10 == 0) {
                    status = queue.peek(val);
                    assert(status == true);
                    assert(val < numIters);
                }
                status = queue.pop(val);
                assert(status == true);
                assert(val < numIters);
                mtx.lock();
                observedNumbers.insert(val);
                mtx.unlock();
            }
        }

        while (!queue.isEmpty()) {
            uint32_t val;
            bool status = queue.pop(val);
            ASSERT_EQ(status, true);
            ASSERT_LT(val, numIters);
            observedNumbers.insert(val);
        }

        ASSERT_EQ(observedNumbers.size(), numIters);
        ASSERT_EQ(queue.capacity(), 100);

        queue.close();
    }
}

TEST(AsyncQueue, nonBlockingApiWorks) {
    uint32_t numProcs = omp_get_num_procs() > 12 ? 12 : omp_get_num_procs();
    omp_set_num_threads(numProcs - 1);
    srand(time(nullptr));

    mutex mtx;

    for (int test = 0; test < 1; ++test) {
        AsyncQueue<uint32_t> queue(100);
        set<uint32_t> observedNumbers;

        const uint32_t numIters = 100000;

        queue.open();
#pragma omp parallel for schedule(static, 1)
        for (uint32_t i = 0; i < numIters; ++i) {
            bool status;
            do {
                status = queue.tryPush(i);
            } while (status != true);

            if (queue.occupancy() > 95) {
                uint32_t val;
                if (i % 10 == 0) {
                    status = queue.tryPeek(val);
                    if (status == true)
                        assert(val < numIters);
                }

                status = queue.tryPop(val);
                if (status == true) {
                    assert(status == true);
                    assert(val < numIters);
                    mtx.lock();
                    observedNumbers.insert(val);
                    mtx.unlock();
                }
            }
        }

        while (!queue.isEmpty()) {
            uint32_t val;
            bool status = queue.tryPop(val);
            ASSERT_EQ(status, true);
            ASSERT_LT(val, numIters);
            observedNumbers.insert(val);
        }

        ASSERT_EQ(observedNumbers.size(), numIters);
        ASSERT_EQ(queue.capacity(), 100);

        queue.close();
    }
}
