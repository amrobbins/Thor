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

using std::map;
using std::mutex;
using std::set;
using std::string;
using std::vector;

struct GpuConnectionRanking {
    int peerGpuNum;
    bool isPeerToPeerSupported;
    int peerToPeerSpeedRanking;  // Lower postive number is better, 0 is best. This connection speed is only valid if
                                 // isPeerToPeerSupported.

    inline bool operator<(const GpuConnectionRanking& rhs) { return peerToPeerSpeedRanking < rhs.peerToPeerSpeedRanking; }
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

    vector<GpuConnectionRanking> getConnectionSpeedRankings(int sourceGpuNum);
    // Returns -1 if not known:
    int getConnectionSpeedRanking(int sourceGpuNum, int destGpuNum);
    // Returns false if not known:
    bool isPeerToPeerAvailable(int sourceGpuNum, int destGpuNum);
    string getGpuType(int gpuNum);
    string getGpuType();
    int getGpuPciBusId(int gpuNum);
    int getGpuNumFromBusId(int gpuBusId);
    int getAdjacentHigherGpu(int gpuNum);
    int getAdjacentLowerGpu(int gpuNum);
    vector<int> getOrderedGpus() { return orderedGpus; }

    unsigned int getNumGpus() { return numGpus; }
    unsigned int getNumMultiProcessors(int gpuNum);
    unsigned int getNumMultiProcessors();

    unsigned long getTotalGlobalMemBytes(int gpuNum);

    // returns the previously active device
    static int swapActiveDevice(int newGpuNum);

    // There are 5 mem copy streams created per gpu, to make optimal use of the hardware:
    //
    //                    [    gpu 0      ]
    //                           |
    //                        fromLower
    //                           |
    //                           v
    // [cpu] --fromCpu--> [ gpu 1 (Local) ]
    // [cpu] <--toCpu---- [               ]
    //                           ^
    //                           |
    //                        fromHigher
    //                           |
    //                    [    gpu 2      ]
    //
    // Also, CPU has (only) a local copy stream that is (arbitrarily) associated with gpu 0, since all streams are
    // assocaited with a gpu.
    // FIXME: this is going to require more thought, currently one model instance will have to wait for the processing
    // of another one unneccesarily.
    //        I think that I will need one set of these per model instance.
    //        Also, I think that I should have just a toOtherGpu and fromOtherGpu, I think that is a more accurate representation of the
    //        hardware.
    Stream getCopyStreamFromLower(int gpuNum);
    Stream getCopyStreamFromHigher(int gpuNum);
    Stream getCopyStreamFromCpu(int gpuNum);
    Stream getCopyStreamToCpu(int gpuNum);
    Stream getCopyStreamLocal(int gpuNum);

    cublasLtHandle_t getCublasLtHandle(int gpuNum);

    // FIXME: add kernel evaluation per gpu type. spread evaluations across all devices of a type. evaluate once per type and lock/block if
    // still evaluating when the kernel type is requested.

    static const int NONE;
    static const int CPU_DEVICE_NUM;

   private:
    map<int, vector<GpuConnectionRanking>> connectionRankings;
    map<int, map<int, bool>> peerToPeerEnabled;
    map<int, map<int, int>> peerConnectionRankings;
    vector<string> gpuType;
    vector<int> gpuPciBusId;  // index is gpuNum, value is gpuPciBusId
    vector<int> orderedGpus;  // index is order (adjacent gpus are adjacent in this vector), value is gpuNum
    map<int, int> gpuNumFromBusId;
    vector<cudaDeviceProp> deviceProps;

    map<int, Stream> copyStreamFromLower;
    map<int, Stream> copyStreamFromHigher;
    map<int, Stream> copyStreamFromCpu;
    map<int, Stream> copyStreamToCpu;
    map<int, Stream> copyStreamLocal;

    unsigned int numGpus;

    mutex mtx;

    void evaluateConnectionSpeeds();
    void getGpuTypes();
    void getGpuPciBusIds();
    void createCopyStreams();

    vector<cublasLtHandle_t> cublasLtHandlePerDevice;
};
