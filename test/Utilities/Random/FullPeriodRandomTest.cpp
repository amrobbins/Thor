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

using std::pair;
using std::set;
using std::string;
using std::vector;

bool PRINT = false;

TEST(FullPeriodRandom, createsAFullPeriodCycle) {
    srand(time(nullptr));

    for (int test = 0; test < 50; ++test) {
        uint32_t period = rand() % 10000 + 1;

        FullPeriodRandom fullPeriodRandom(period);

        set<uint64_t> observedNumbers;
        for (uint32_t i = 0; i < period; ++i) {
            uint64_t number = fullPeriodRandom.getRandomNumber();
            ASSERT_LT(number, period);
            observedNumbers.insert(number);
        }
        ASSERT_EQ(observedNumbers.size(), period);

        if (PRINT) {
            for (uint32_t j = 0; j < 2; ++j) {
                for (uint32_t i = 0; i < period; ++i) {
                    printf("%ld\n", fullPeriodRandom.getRandomNumber());
                }
                printf("\n");
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
