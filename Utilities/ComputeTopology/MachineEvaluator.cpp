#include "MachineEvaluator.h"
#include "Utilities/Expression/CudaHelpers.h"

using namespace std;

const int MachineEvaluator::NONE = -10;
const int MachineEvaluator::CPU_DEVICE_NUM = -1;

MachineEvaluator::MachineEvaluator() {
    int iNumGpus;
    CUDA_CHECK(cudaGetDeviceCount(&iNumGpus));
    numGpus = (unsigned int)iNumGpus;

    getDeviceProps();

    getGpuTypes();
    getGpuPciBusIds();
    evaluateConnectionSpeeds();

    for (unsigned int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        ScopedGpu scopedGpu(gpuNum);

        cublasStatus_t cublasStatus;
        cublasLtHandle_t cublasLtHandle;
        cublasStatus = cublasLtCreate(&cublasLtHandle);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        cublasLtHandlePerDevice.push_back(cublasLtHandle);
    }
}

MachineEvaluator::~MachineEvaluator() {
    for (unsigned int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        ScopedGpu scopedGpu(gpuNum);
        cublasStatus_t cublasStatus;
        cublasStatus = cublasLtDestroy(cublasLtHandlePerDevice[gpuNum]);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }
}

int MachineEvaluator::getCurrentGpuNum() {
    int curGpuNum;
    CUDA_CHECK(cudaGetDevice(&curGpuNum));
    return curGpuNum;
}

vector<GpuConnectionRanking> MachineEvaluator::getConnectionSpeedRankings(int sourceGpuNum) {
    if (connectionRankings.find(sourceGpuNum) == connectionRankings.end())
        return vector<GpuConnectionRanking>();
    return connectionRankings[sourceGpuNum];
}

bool MachineEvaluator::isPeerToPeerAvailable(int sourceGpuNum, int destGpuNum) {
    if (peerToPeerEnabled.find(sourceGpuNum) == peerToPeerEnabled.end())
        return false;
    if (peerToPeerEnabled[sourceGpuNum].find(destGpuNum) == peerToPeerEnabled[sourceGpuNum].end())
        return false;

    return peerToPeerEnabled[sourceGpuNum][destGpuNum];
}

void MachineEvaluator::evaluateConnectionSpeeds() {
    int previousGpu;
    CUDA_CHECK(cudaGetDevice(&previousGpu));

    for (unsigned int i = 0; i < numGpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));

        for (unsigned int j = 0; j < numGpus; ++j) {
            if (i == j)
                continue;

            int perfRank = 0;
            int accessSupported = 0;
            CUDA_CHECK(cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, i, j));
            if (accessSupported) {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
            }
            CUDA_CHECK(cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, i, j));

            GpuConnectionRanking connectionRanking;
            connectionRanking.peerGpuNum = j;
            connectionRanking.isPeerToPeerSupported = accessSupported;
            connectionRanking.peerToPeerSpeedRanking = accessSupported ? perfRank : 1000;

            connectionRankings[i].push_back(connectionRanking);
        }
        std::sort(connectionRankings[i].begin(), connectionRankings[i].end());
    }

    CUDA_CHECK(cudaSetDevice(previousGpu));
}

void MachineEvaluator::getGpuTypes() {
    for (unsigned int i = 0; i < numGpus; ++i) {
        gpuType.push_back(deviceProps[i].name);
    }
}

void MachineEvaluator::getDeviceProps() {
    int previousGpu;
    CUDA_CHECK(cudaGetDevice(&previousGpu));

    for (unsigned int i = 0; i < numGpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        deviceProps.push_back(deviceProp);
    }

    CUDA_CHECK(cudaSetDevice(previousGpu));
}

void MachineEvaluator::getGpuPciBusIds() {
    set<int> orderedGpuPciBusIds;

    for (unsigned int i = 0; i < numGpus; ++i) {
        gpuPciBusId.push_back(deviceProps[i].pciBusID);
        gpuNumFromBusId[deviceProps[i].pciBusID] = i;
        orderedGpuPciBusIds.insert(deviceProps[i].pciBusID);
        // FIXME: consider deviceProp.pciDomainID, deviceProp.pciDeviceID also. For now assuming there is only one
        // domain and host device
    }

    for (int pciBusId : orderedGpuPciBusIds) {
        orderedGpus.push_back(gpuNumFromBusId[pciBusId]);
    }
}

string MachineEvaluator::getGpuType(int gpuNum) {
    assert((unsigned int)gpuNum < gpuType.size());
    return gpuType[gpuNum];
}

string MachineEvaluator::getGpuType() { return getGpuType(getCurrentGpuNum()); }

int MachineEvaluator::getGpuPciBusId(int gpuNum) {
    assert((unsigned int)gpuNum < gpuPciBusId.size());
    return gpuPciBusId[gpuNum];
}

int MachineEvaluator::getGpuNumFromBusId(int gpuBusId) {
    assert(gpuNumFromBusId.count(gpuBusId) == 1);
    return gpuNumFromBusId[gpuBusId];
}

int MachineEvaluator::getAdjacentHigherGpu(int gpuNum) {
    assert((unsigned int)gpuNum < orderedGpus.size());

    if ((unsigned int)gpuNum == orderedGpus.size() - 1) {
        return NONE;
    }

    return orderedGpus[gpuNum + 1];
}

int MachineEvaluator::getAdjacentLowerGpu(int gpuNum) {
    assert((unsigned int)gpuNum < orderedGpus.size());

    if (gpuNum == 0) {
        return NONE;
    }

    return orderedGpus[gpuNum - 1];
}

int MachineEvaluator::swapActiveDevice(int newGpuNum) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(newGpuNum < deviceCount);

    int previousGpuNum;
    CUDA_CHECK(cudaGetDevice(&previousGpuNum));
    CUDA_CHECK(cudaSetDevice(newGpuNum));

    return previousGpuNum;
}

unsigned int MachineEvaluator::getNumMultiProcessors(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    return deviceProps[gpuNum].multiProcessorCount;
}

unsigned int MachineEvaluator::getNumMultiProcessors() { return getNumMultiProcessors(getCurrentGpuNum()); }

unsigned long MachineEvaluator::getTotalGlobalMemBytes(int gpuNum) {
    assert(gpuNum < (int)deviceProps.size());
    return deviceProps[gpuNum].totalGlobalMem;
}

unsigned long MachineEvaluator::getFreeMemBytes(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    ScopedGpu scopedGpu(gpuNum);
    size_t freeMemBytes;
    size_t totalMemBytes;
    CUDA_CHECK(cudaMemGetInfo(&freeMemBytes, &totalMemBytes));
    return freeMemBytes;
}

cublasLtHandle_t MachineEvaluator::getCublasLtHandle(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);
    return cublasLtHandlePerDevice[gpuNum];
}
