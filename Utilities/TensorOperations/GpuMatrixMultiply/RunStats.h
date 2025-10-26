#pragma once

namespace ThorImplementation {

struct RunStats {
    RunStats() {
        errorFlag = false;
        runCount = 0;
        totalExecutionTimeMilliseconds = 0.0;
        stashedRunCount = 0;
        stashedExecutionTimeMilliseconds = 0.0;
    }

    RunStats(const RunStats &other) {
        // implemented using operator=
        *this = other;
    }

    RunStats &operator=(const RunStats &other) {
        errorFlag = other.errorFlag;
        runCount = other.runCount;
        totalExecutionTimeMilliseconds = other.totalExecutionTimeMilliseconds;
        stashedRunCount = other.stashedRunCount;
        stashedExecutionTimeMilliseconds = other.stashedExecutionTimeMilliseconds;
        return *this;
    }

    bool errorFlag;
    int runCount;
    double totalExecutionTimeMilliseconds;

    int stashedRunCount;
    double stashedExecutionTimeMilliseconds;

    void recordRun(double executionTimeOfRun) {
        mtx.lock();
        runCount += 1;
        totalExecutionTimeMilliseconds += executionTimeOfRun;
        mtx.unlock();
    }

    inline double getAverageRunTimeMilliseconds() {
        // Updates should not be concurrently ongoing when running this function.
        assert(runCount > 0);
        return totalExecutionTimeMilliseconds / runCount;
    }

    void stashRunStats() {
        mtx.lock();
        stashedRunCount += runCount;
        stashedExecutionTimeMilliseconds += totalExecutionTimeMilliseconds;

        runCount = 0;
        totalExecutionTimeMilliseconds = 0.0;
        mtx.unlock();
    }

    void unstashRunStats() {
        mtx.lock();
        runCount += stashedRunCount;
        totalExecutionTimeMilliseconds += stashedExecutionTimeMilliseconds;

        stashedRunCount = 0;
        stashedExecutionTimeMilliseconds = 0.0;
        mtx.unlock();
    }

    inline bool operator<(RunStats &rhs) {
        if (errorFlag)
            return false;
        else if (rhs.errorFlag)
            return true;
        return getAverageRunTimeMilliseconds() < rhs.getAverageRunTimeMilliseconds();
    }

   private:
    std::mutex mtx;
};

}  // namespace ThorImplementation
