#include "Thor.h"

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

using ThorImplementation::Tensor;
using ThorImplementation::TensorDescriptor;
using ThorImplementation::TensorPlacement;

bool PRINT = false;

TEST(AsyncTensorQueue, blockingApiWorks) {
    uint32_t numProcs = omp_get_num_procs() > 12 ? 12 : omp_get_num_procs();
    omp_set_num_threads(numProcs - 1);
    srand(time(nullptr));

    mutex mtx;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    for (int test = 0; test < 1; ++test) {
        TensorDescriptor tensorDescriptor(TensorDescriptor::DataType::FP32, {2});
        AsyncTensorQueue queue(100, tensorDescriptor, cpuPlacement);
        set<uint32_t> observedNumbers;

        const uint32_t numIters = 100000;

        queue.open();
#pragma omp parallel for schedule(static, 1)
        for (uint32_t i = 0; i < numIters; ++i) {
            Tensor buffer;
            bool status;
            status = queue.getBufferToLoad(buffer);
            assert(status == true);
            float *mem = (float *)buffer.getMemPtr();
            mem[0] = i;
            mem[1] = i;
            status = queue.bufferLoaded(buffer);
            assert(status == true);

            if (queue.occupancy() > 95) {
                status = queue.getBufferToUnload(buffer);
                assert(status == true);
                mem = (float *)buffer.getMemPtr();
                assert(mem[0] == mem[1]);
                assert(mem[0] < numIters);
                mtx.lock();
                observedNumbers.insert(mem[0]);
                mtx.unlock();
                status = queue.bufferUnloaded(buffer);
                assert(status == true);
            }
        }

        while (!queue.isEmpty()) {
            Tensor buffer;
            bool status = queue.getBufferToUnload(buffer);
            ASSERT_EQ(status, true);
            float *mem = (float *)buffer.getMemPtr();
            ASSERT_EQ(mem[0], mem[1]);
            ASSERT_LT(mem[0], numIters);
            mtx.lock();
            observedNumbers.insert(mem[0]);
            mtx.unlock();
            status = queue.bufferUnloaded(buffer);
            ASSERT_EQ(status, true);
        }

        ASSERT_EQ(observedNumbers.size(), numIters);
        ASSERT_EQ(queue.capacity(), 100);
    }
}

TEST(AsyncTensorQueue, nonblockingApiWorks) {
    uint32_t numProcs = omp_get_num_procs() > 12 ? 12 : omp_get_num_procs();
    omp_set_num_threads(numProcs - 1);
    srand(time(nullptr));

    mutex mtx;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    for (int test = 0; test < 1; ++test) {
        TensorDescriptor tensorDescriptor(TensorDescriptor::DataType::FP32, {2});
        AsyncTensorQueue queue(100, tensorDescriptor, cpuPlacement);
        set<uint32_t> observedNumbers;

        const uint32_t numIters = 100000;

        queue.open();
#pragma omp parallel for schedule(static, 1)
        for (uint32_t i = 0; i < numIters; ++i) {
            Tensor buffer;
            bool status;
            do {
                status = queue.getBufferToLoad(buffer);
            } while (status != true);
            float *mem = (float *)buffer.getMemPtr();
            mem[0] = i;
            mem[1] = i;
            status = queue.bufferLoaded(buffer);
            assert(status == true);

            if (queue.occupancy() > 95) {
                status = queue.tryGetBufferToUnload(buffer);
                if (status == true) {
                    assert(status == true);
                    mem = (float *)buffer.getMemPtr();
                    assert(mem[0] == mem[1]);
                    assert(mem[0] < numIters);
                    mtx.lock();
                    observedNumbers.insert(mem[0]);
                    mtx.unlock();
                    status = queue.bufferUnloaded(buffer);
                    assert(status == true);
                }
            }
        }

        while (!queue.isEmpty()) {
            Tensor buffer;
            bool status = queue.tryGetBufferToUnload(buffer);
            ASSERT_EQ(status, true);
            float *mem = (float *)buffer.getMemPtr();
            ASSERT_EQ(mem[0], mem[1]);
            ASSERT_LT(mem[0], numIters);
            mtx.lock();
            observedNumbers.insert(mem[0]);
            mtx.unlock();
            status = queue.bufferUnloaded(buffer);
            ASSERT_EQ(status, true);
        }

        ASSERT_EQ(observedNumbers.size(), numIters);
        ASSERT_EQ(queue.capacity(), 100);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
