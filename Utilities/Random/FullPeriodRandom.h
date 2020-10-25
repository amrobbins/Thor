#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <cmath>

using boost::multiprecision::uint128_t;

/**
 *  Gets each number from 0 to period - 1 in a pseudo random sequence, where the sequence order depends on the seed value.
 *  To iterate over the same numbers another time in another order, first call reseedUsingClock, and then get period random numbers.
 *
 *  getRandomNumber() will need to be synchronized externally if more than one thread will be getting random numbers.
 *
 *  none of the functions have internal synchronization and all would need to be synchronized externally for the multithreaded case.
 */
class FullPeriodRandom {
   public:
    FullPeriodRandom(uint64_t period, uint64_t *seed = nullptr) : period(period) {
        if (seed == nullptr)
            reseedUsingClock();
        else
            X = *seed;

        // m and c share no prime factors
        vector<uint64_t> periodNonFactors;
        getPrimeNonFactors(period, periodNonFactors, 50);
        uint64_t numDivisorsInC = 3;
        c = 1;
        uint8_t randomNumbers[4] = {
            (uint8_t)(X & 0x255), (uint8_t)((X >> 8) & 0x255), (uint8_t)((X >> 16) & 0x255), (uint8_t)((X >> 24) & 0x255)};
        for (uint64_t i = 0; i < numDivisorsInC; ++i) {
            c *= periodNonFactors[randomNumbers[i] % periodNonFactors.size()];
        }

        // a - 1 is divisible by all factors of period
        // a - 1 is divisible by 4 if period is divisible by 4
        // Just taking a - 1 to equal period.
        a = period + 1;

        if (seed == nullptr)
            reseedUsingClock();
    }

    void reseedUsingClock() {
        std::chrono::high_resolution_clock::time_point timePoint = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::duration duration = timePoint.time_since_epoch();
        uint64_t ticks = duration.count();

        // Interleave the bits of the tick count since the lower bits change faster.
        uint64_t seed = 0;
        for (uint32_t i = 0; i < 32; ++i) {
            setBit(seed, 2 * i, getBit(2 * ticks, i));
            setBit(seed, 2 * i + 1, getBit(ticks, 63 - 2 * i));
        }

        std::hash<uint64_t> hashFunctor;
        X = hashFunctor(seed);
    }

    void setSeed(uint64_t seed) { X = seed; }

    uint64_t getRandomNumber() {
        X = (a * X + c) % period;
        return X;
    }

   private:
    const uint64_t period;

    uint64_t X;
    uint64_t a;
    uint64_t c;

    void getPrimeNonFactors(uint64_t period, vector<uint64_t> &nonFactors, uint64_t numFactors) {
        uint64_t maxPrime = 2000;
        vector<uint32_t> mem((maxPrime + 63) / 64, 0);

        nonFactors.clear();
        uint64_t remainingPeriod = period;
        if (remainingPeriod % 2 == 0) {
            while (remainingPeriod % 2 == 0) {
                remainingPeriod /= 2;
            }
        } else {
            nonFactors.push_back(2);
        }
        for (uint64_t i = 3; i < maxPrime; i += 2) {
            // bool bit = (mem[i / 64] >> ((i/2) % 32)) & 0x1;
            uint8_t bit = (mem[i >> 6] >> ((i >> 1) & 0x1F)) & 0x1;
            if (bit == 0) {
                if (remainingPeriod % i == 0) {
                    while (remainingPeriod % i == 0) {
                        remainingPeriod /= i;
                    }
                } else {
                    nonFactors.push_back(i);
                }

                if (nonFactors.size() == numFactors)
                    return;

                for (uint64_t j = i * i; j < maxPrime; j += 2 * i) {
                    mem[j >> 6] |= (0x1 << ((j >> 1) & 0x1F));
                }
            }
        }
        assert(nonFactors.size() == numFactors);
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
};
