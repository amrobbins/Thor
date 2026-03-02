#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <cublasLt.h>
#include "cuda.h"
#include "cuda_runtime.h"

class Stream;

struct GpuConnectionRanking {
    int peerGpuNum;
    bool isPeerToPeerSupported;
    int peerToPeerSpeedRanking;  // Lower postive number is better, 0 is best. This connection speed is only valid if
                                 // isPeerToPeerSupported.

    inline bool operator<(const GpuConnectionRanking& rhs) const { return peerToPeerSpeedRanking < rhs.peerToPeerSpeedRanking; }
};

// Singleton Object: MachineEvaluator
//
// MachineEvaluator evaluates the machine (gpu's/connection topology/...) and enables performance features (peerToPeer
// access, ...) when it is instantiated, which occurs the first time instance() is called.
// MachineEvaluator also determines the fastest kernels for each device type for some important kernels such as matrix
// multiply and convolutional network layers
class MachineEvaluator {
   public:
    static MachineEvaluator& instance() {
        static MachineEvaluator singletonInstance;  // Guaranteed to be destroyed. Instantiated on first use.
        return singletonInstance;
    }

    // Forbid copying since MachineEvaluator is a singleton
    MachineEvaluator(const MachineEvaluator&) = delete;
    MachineEvaluator& operator=(const MachineEvaluator&) = delete;

    virtual ~MachineEvaluator();

   private:
    MachineEvaluator();

    void getDeviceProps();

   public:
    int getCurrentGpuNum();

    std::vector<GpuConnectionRanking> getConnectionSpeedRankings(int sourceGpuNum);
    // Returns false if not known:
    bool isPeerToPeerAvailable(int sourceGpuNum, int destGpuNum);
    std::string getGpuType(int gpuNum);
    std::string getGpuType();
    int getGpuPciBusId(int gpuNum);
    int getGpuNumFromBusId(int gpuBusId);
    int getAdjacentHigherGpu(int gpuNum);
    int getAdjacentLowerGpu(int gpuNum);
    std::vector<int> getOrderedGpus() { return orderedGpus; }

    unsigned int getNumGpus() { return numGpus; }
    unsigned int getNumMultiProcessors(int gpuNum);
    unsigned int getNumMultiProcessors();

    unsigned long getTotalGlobalMemBytes(int gpuNum);
    unsigned long getFreeMemBytes(int gpuNum);

    // returns the previously active device
    static int swapActiveDevice(int newGpuNum);

    cublasLtHandle_t getCublasLtHandle(int gpuNum);

    static const int NONE;
    static const int CPU_DEVICE_NUM;

   private:
    std::map<int, std::vector<GpuConnectionRanking>> connectionRankings;
    std::map<int, std::map<int, bool>> peerToPeerEnabled;
    std::vector<std::string> gpuType;
    std::vector<int> gpuPciBusId;  // index is gpuNum, value is gpuPciBusId
    std::vector<int> orderedGpus;  // index is order (adjacent gpus are adjacent in this vector), value is gpuNum
    std::map<int, int> gpuNumFromBusId;
    std::vector<cudaDeviceProp> deviceProps;

    unsigned int numGpus;

    std::mutex mtx;

    void evaluateConnectionSpeeds();
    void getGpuTypes();
    void getGpuPciBusIds();

    std::vector<cublasLtHandle_t> cublasLtHandlePerDevice;
};
