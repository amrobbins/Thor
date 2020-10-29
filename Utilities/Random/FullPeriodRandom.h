#pragma once

#include "Utilities/WorkQueue/AsyncQueue.h"

#include <chrono>
#include <cmath>
#include <thread>

using std::thread;

/**
 *  This is a Linear congruential generator.
 *
 *  Gets each number from 0 to period - 1 in a pseudo random sequence, where the sequence order depends on the seed value derived from the
 * clock. After sending all numbers from 0 to period - 1 exactly once, it reseeds itself using a new seed derived from the clock.
 *
 *  FullPeriodRandom supports multiple threads and handles synchronization internally.
 */

class FullPeriodRandom {
   public:
    FullPeriodRandom(uint64_t period) : period(period), periodCount(0), randomNumberQueue(period < 1000000 ? 128 : 4096) {
        assert(period != 0);
        if (period == 1) {
            return;
        }

        getFactors(period, periodFactors, periodNonFactors);
        createBaseAMinusOne();
        ensureThereIsRoomToRandomizeC();

        reseed();

        startReadAheadThread();
    }

    FullPeriodRandom(const FullPeriodRandom &) = delete;
    FullPeriodRandom &operator=(const FullPeriodRandom &) = delete;

    virtual ~FullPeriodRandom() { killReadAheadThread(); }

    uint64_t getRandomNumber() {
        uint64_t randomNumber = 0;
        bool succeeded = randomNumberQueue.pop(randomNumber);
        assert(succeeded);

        return randomNumber;
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

    vector<uint64_t> periodFactors;
    vector<uint64_t> periodNonFactors;

    thread readAheadThread;
    AsyncQueue<uint64_t> randomNumberQueue;

    void reseed(Optional<uint64_t> seed = Optional<uint64_t>::empty()) {
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

    void getFactors(uint64_t period, vector<uint64_t> &factors, vector<uint64_t> &nonFactors) {
        uint64_t maxPrime = period;
        vector<uint32_t> mem((maxPrime + 63) / 64, 0);

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
        // a - 1 is divisible by all factors of period
        // a - 1 is divisible by 4 if period is divisible by 4
        // a < period
        baseAMinusOne = 1;
        for (uint64_t i = 0; i < periodFactors.size(); ++i) {
            baseAMinusOne *= periodFactors[i];
        }

        // If period has no repeated prime factors, then the 'a' constraints cannot be satisfied
        // in that case increase the period such that the constraints can be satisfied
        implementationPeriod = period;
        if (baseAMinusOne + 1 >= period / 5) {
            bool hasFactor5 = false;
            bool hasFactor7 = false;
            for (uint64_t i = 0; i < 4 && i < periodFactors.size(); ++i) {
                if (periodFactors[i] == 5)
                    hasFactor5 = true;
                if (periodFactors[i] == 7)
                    hasFactor7 = true;
            }
            if (hasFactor5) {
                implementationPeriod *= 5;
            } else if (hasFactor7) {
                implementationPeriod *= 7;
            } else {
                baseAMinusOne *= 3;
                implementationPeriod *= 9;
            }

            getFactors(implementationPeriod, periodFactors, periodNonFactors);
        }
        // Now a = baseAMinusOne + 1 is guaranteed to satisfy all constraints,
        // this is possible because implemenationPeriod is guaranteed to have a repeated prime factor
        uint64_t a = baseAMinusOne + 1;
        assert(a < implementationPeriod);
        maxBaseAMinusOneMultiple = implementationPeriod / a;
        if (implementationPeriod % a == 0) {
            // Don't think this case is possible, but easier to do than to prove it is not possible.
            maxBaseAMinusOneMultiple -= 1;
        }
        assert(maxBaseAMinusOneMultiple * a < implementationPeriod);
        assert(maxBaseAMinusOneMultiple >= 2);
    }

    void ensureThereIsRoomToRandomizeC() {
        while (periodNonFactors.size() < 5) {
            implementationPeriod *= 2;
            getFactors(implementationPeriod, periodFactors, periodNonFactors);
            if (implementationPeriod % 4 == 0 && baseAMinusOne % 4 != 0)
                baseAMinusOne *= 2;
            else if (implementationPeriod % 2 == 0 && baseAMinusOne % 2 != 0)
                baseAMinusOne *= 2;
            maxBaseAMinusOneMultiple = implementationPeriod / (baseAMinusOne + 1);
            assert(maxBaseAMinusOneMultiple >= 2);
        }
    }

    void startReadAheadThread() {
        randomNumberQueue.open();
        readAheadThread = thread(&FullPeriodRandom::fetchNumbers, this);
    }

    void fetchNumbers() {
        while (randomNumberQueue.isOpen()) {
            randomNumberQueue.push(getNextRandomNumber());
        }
    }

    void killReadAheadThread() {
        randomNumberQueue.close();
        readAheadThread.join();
    }

    uint64_t getNextRandomNumber() {
        if (period == 1)
            return 0;

        if (periodCount == period) {
            periodCount = 0;
            reseed();
        }

        do {
            X = (a * X + c) % implementationPeriod;
        } while (X >= period);

        periodCount += 1;
        return X;
    }
};
