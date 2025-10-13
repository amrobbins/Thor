#include "Utilities/Random/FullPeriodRandom.h"

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

TEST(FullPeriodRandom, createsAFullPeriodCycle) {
    srand(time(nullptr));

    bool PRINT = false;

    for (int test = 0; test < 100; ++test) {
        uint32_t period = rand() % 10000 + 1;

        FullPeriodRandom fullPeriodRandom(period, rand() % 2 ? true : false);

        for (uint32_t j = 0; j < 3; ++j) {
            set<uint64_t> observedNumbers;
            for (uint32_t i = 0; i < period; ++i) {
                uint64_t number = fullPeriodRandom.getRandomNumber();
                ASSERT_LT(number, period);
                observedNumbers.insert(number);
            }
            ASSERT_EQ(observedNumbers.size(), period);
        }

        if (PRINT) {
            for (uint32_t j = 0; j < 3; ++j) {
                for (uint32_t i = 0; i < period; ++i) {
                    printf("%ld\n", fullPeriodRandom.getRandomNumber());
                }
                printf("\n\n");
            }
            printf("------------------------\n\n");
        }
    }
}
