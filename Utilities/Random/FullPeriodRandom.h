#pragma once

#include "Utilities/WorkQueue/AsyncQueue.h"

#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

using std::thread;

/**
 *  This is a Linear congruential generator.
 *
 *  Gets each number from 0 to period - 1 in a pseudo random sequence, where the sequence order depends on the seed value derived from the
 * clock. After sending all numbers from 0 to period - 1 exactly once, it reseeds itself using a new seed derived from the clock.
 *
 *  If a FullPeriodRandom object will be accessed by multiple threads, its constructor parameter synchronized must be set to true.
 */

class FullPeriodRandom {
   public:
    FullPeriodRandom(uint64_t period, bool synchronized = false) : period(period), periodCount(0), synchronized(synchronized) {
        assert(period != 0);
        if (period == 1) {
            return;
        }

        implementationPeriod = period;
        getFactors(period, periodFactors, periodNonFactors);
        createBaseAMinusOne();
        ensureThereIsRoomToRandomizeC();

        reseed();
    }

    FullPeriodRandom(const FullPeriodRandom &) = delete;
    FullPeriodRandom &operator=(const FullPeriodRandom &) = delete;

    uint64_t getRandomNumber() {
        if (period == 1)
            return 0;

        if (synchronized)
            std::unique_lock<std::mutex> lck(mtx);

        if (periodCount == period) {
            reseed();
        }

        do {
            X = (a * X + c) % implementationPeriod;
        } while (X >= period);

        periodCount += 1;
        return X;
    }

    void reseed(Optional<uint64_t> seed = Optional<uint64_t>::empty()) {
        periodCount = 0;

        if (period == 1)
            return;

        // a - 1 is divisible by all factors of period
        // a - 1 is divisible by 4 if period is divisible by 4
        // a < period
        // I may multiply aMinusOne by any positive integer so long as aMinusOne + 1 < a
        a = (baseAMinusOne * ((rand() % maxBaseAMinusOneMultiple) + 1)) + 1;
        assert(a < implementationPeriod);

        // period and c share no prime factors
        // c < period
        c = periodNonFactors[rand() % periodNonFactors.size()];
        for (uint64_t i = 0; i < 3; ++i) {
            uint64_t selectdNonFactor = periodNonFactors[rand() % periodNonFactors.size()];
            if (c * selectdNonFactor < implementationPeriod)
                c *= selectdNonFactor;
        }
        while (rand() % 5 != 0) {
            uint64_t selectdNonFactor = periodNonFactors[rand() % periodNonFactors.size()];
            if (c * selectdNonFactor < implementationPeriod)
                c *= selectdNonFactor;
        }
        assert(c < implementationPeriod);

        // Use a different seed for X compared to the seed used for c.
        if (seed.isEmpty())
            seed = getClockSeed();

        X = seed;
    }

   private:
    const uint64_t period;
    uint64_t implementationPeriod;

    uint64_t X;
    uint64_t a;
    uint64_t baseAMinusOne;
    uint64_t maxBaseAMinusOneMultiple;
    uint64_t c;

    uint64_t periodCount;

    std::vector<uint64_t> periodFactors;
    std::vector<uint64_t> periodNonFactors;

    const bool synchronized;
    mutex mtx;

    uint64_t getClockSeed() {
        std::hash<uint64_t> hashFunctor;
        std::chrono::high_resolution_clock::time_point timePoint = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::duration duration = timePoint.time_since_epoch();
        uint64_t ticks = duration.count();

        // Interleave the bits of the tick count since the lower bits change faster.
        uint64_t seed = 0;
        for (uint32_t i = 0; i < 32; ++i) {
            setBit(seed, 2 * i, getBit(2 * ticks, i));
            setBit(seed, 2 * i + 1, getBit(ticks, 63 - 2 * i));
        }

        seed = hashFunctor(seed);
        return seed;
    }

    void getFactors(uint64_t period, std::vector<uint64_t> &factors, std::vector<uint64_t> &nonFactors) {
        uint64_t maxPrime = period;
        std::vector<uint32_t> mem((maxPrime + 63) / 64, 0);

        factors.clear();
        nonFactors.clear();
        uint64_t remainingPeriod = period;
        // 4 acts sort of like a prime for this algorithm
        if (remainingPeriod % 4 == 0) {
            factors.push_back(4);
            while (remainingPeriod % 2 == 0) {
                remainingPeriod /= 2;
            }
        } else if (remainingPeriod % 2 == 0) {
            factors.push_back(2);
            while (remainingPeriod % 2 == 0) {
                remainingPeriod /= 2;
            }
        } else {
            nonFactors.push_back(2);
        }
        for (uint64_t i = 3; i <= maxPrime; i += 2) {
            // bool bit = (mem[i / 64] >> ((i/2) % 32)) & 0x1;
            uint8_t bit = (mem[i >> 6] >> ((i >> 1) & 0x1F)) & 0x1;
            if (bit == 0) {
                if (remainingPeriod % i == 0) {
                    factors.push_back(i);
                    while (remainingPeriod % i == 0) {
                        remainingPeriod /= i;
                    }
                } else {
                    assert(i < period);
                    nonFactors.push_back(i);
                }

                for (uint64_t j = i * i; j <= maxPrime; j += 2 * i) {
                    mem[j >> 6] |= (0x1 << ((j >> 1) & 0x1F));
                }
            }
        }
    }

    inline bool getBit(uint64_t word, uint32_t bit) {
        assert(bit < 64);
        return (word >> bit) & 1UL;
    }

    inline void setBit(uint64_t &word, uint32_t bit, bool value) {
        assert(bit < 64);
        if (value)
            word |= 1UL << bit;
        else
            word &= ~(1UL << bit);
    }

    void createBaseAMinusOne() {
        constexpr uint64_t MIN_A_MULTIPLE = 5;

        // a - 1 is divisible by all factors of period
        // a - 1 is divisible by 4 if period is divisible by 4
        // a < period
        baseAMinusOne = 1;
        for (uint64_t i = 0; i < periodFactors.size(); ++i) {
            baseAMinusOne *= periodFactors[i];
        }
        uint64_t a = baseAMinusOne + 1;
        maxBaseAMinusOneMultiple = implementationPeriod / a;
        if (implementationPeriod % a == 0) {
            // Don't think this case is possible, but easier to do than to prove it is not possible.
            maxBaseAMinusOneMultiple -= 1;
        }

        while (maxBaseAMinusOneMultiple < MIN_A_MULTIPLE) {
            implementationPeriod += 1;
            getFactors(implementationPeriod, periodFactors, periodNonFactors);
            baseAMinusOne = 1;
            for (uint64_t i = 0; i < periodFactors.size(); ++i) {
                baseAMinusOne *= periodFactors[i];
            }
            a = baseAMinusOne + 1;
            maxBaseAMinusOneMultiple = implementationPeriod / a;
            if (implementationPeriod % a == 0) {
                // Don't think this case is possible, but easier to do than to prove it is not possible.
                maxBaseAMinusOneMultiple -= 1;
            }
        }

        // Now a = baseAMinusOne + 1 is guaranteed to satisfy all constraints,
        // this is possible because implemenationPeriod is guaranteed to have a repeated prime factor
        assert(a < implementationPeriod);
        assert(maxBaseAMinusOneMultiple * a < implementationPeriod);
        assert(maxBaseAMinusOneMultiple >= MIN_A_MULTIPLE);
    }

    void ensureThereIsRoomToRandomizeC() {
        while (periodNonFactors.size() < 10) {
            implementationPeriod *= 2;
            getFactors(implementationPeriod, periodFactors, periodNonFactors);
            if (implementationPeriod % 4 == 0 && baseAMinusOne % 4 != 0)
                baseAMinusOne *= 2;
            else if (implementationPeriod % 2 == 0 && baseAMinusOne % 2 != 0)
                baseAMinusOne *= 2;
            maxBaseAMinusOneMultiple = implementationPeriod / (baseAMinusOne + 1);
            assert(maxBaseAMinusOneMultiple >= 1);
        }
    }
};
