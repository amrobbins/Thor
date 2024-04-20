#pragma once

#include "Utilities/WorkQueue/AsyncQueue.h"

#include <assert.h>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <vector>

/**
 *  This is a hybrid random number generator that uses a linear congruential generator along with system entropy randomness.
 *
 *  Gets each number from 0 to period - 1 in a pseudo random sequence, however the particular pseudo random sequence that is selected
 *  to begin each period is truly random. After sending all numbers from 0 to period - 1 exactly once, FullPeriodRandom reseeds itself
 *  using a combination of system entropy, high precision system clock time and thread id.
 *
 *  This means that your dataset will be exactly evenly sampled on a per-epoch basis, if you want to oversample some training examples
 *  that should be handled by adding more than one copy of those to the dataset, then you will get oversampling resulting from that ratio.
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

        std::random_device randomDevice;
        std::hash<std::thread::id> hasher;
        uint32_t seed =
            randomDevice() + std::chrono::system_clock::now().time_since_epoch().count() * 10000000 + hasher(std::this_thread::get_id());
        generator = std::mt19937(seed);
        distribution = std::uniform_real_distribution<double>(0, pow(2.0, 60));

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

    // The actual period of the LCG can be bigger than the period. When a number larger than period is returned by the LCG,
    // then it is not returned, instead the LCG is asked for a new number until the number given is within the period.
    void reseed() {
        uint32_t maxOvershoot;
        if (period > 10000)
            maxOvershoot = 1000 + (period / 1000);
        else if (period > 100)
            maxOvershoot = period / 10;
        else
            maxOvershoot = 10;

        if (maxOvershoot > 50000)
            maxOvershoot = 50000;

        implementationPeriod = period + (entropyRand() % maxOvershoot);

        getFactors(implementationPeriod, periodFactors, periodNonFactors);
        createBaseAMinusOne();
        ensureThereIsRoomToRandomizeC();

        seedParameters();
    }

    void seedParameters(Optional<uint64_t> seedValue = Optional<uint64_t>::empty()) {
        periodCount = 0;

        if (period == 1)
            return;

        // a - 1 is divisible by all factors of period
        // a - 1 is divisible by 4 if period is divisible by 4
        // a < period
        // I may multiply aMinusOne by any positive integer so long as aMinusOne + 1 < a
        a = (baseAMinusOne * ((entropyRand() % maxBaseAMinusOneMultiple) + 1)) + 1;
        assert(a < implementationPeriod);

        // period and c share no prime factors
        // c < period
        c = periodNonFactors[entropyRand() % periodNonFactors.size()];
        for (uint64_t i = 0; i < 3; ++i) {
            uint64_t selectdNonFactor = periodNonFactors[entropyRand() % periodNonFactors.size()];
            if (c * selectdNonFactor < implementationPeriod)
                c *= selectdNonFactor;
        }
        while (entropyRand() % 5 != 0) {
            uint64_t selectdNonFactor = periodNonFactors[entropyRand() % periodNonFactors.size()];
            if (c * selectdNonFactor < implementationPeriod)
                c *= selectdNonFactor;
        }
        assert(c < implementationPeriod);

        // Use a different seed for X compared to the seed used for c.
        if (seedValue.isEmpty())
            seedValue = entropyRand();

        X = seedValue;
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
    std::mutex mtx;

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

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution;

    uint64_t entropyRand() { return uint64_t(distribution(generator)); }
};
