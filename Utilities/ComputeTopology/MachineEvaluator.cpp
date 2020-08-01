#include "MachineEvaluator.h"

const int MachineEvaluator::NONE = -10;
const int MachineEvaluator::CPU_DEVICE_NUM = -1;

MachineEvaluator::MachineEvaluator() {
    cudaError_t cudaStatus;

    int iNumGpus = (int)numGpus;
    cudaStatus = cudaGetDeviceCount(&iNumGpus);
    numGpus = (unsigned int)iNumGpus;
    assert(cudaStatus == cudaSuccess);

    getDeviceProps();

    getGpuTypes();
    getGpuPciBusIds();
    evaluateConnectionSpeeds();
    createCopyStreams();

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
    cudaError_t cudaStatus = cudaGetDevice(&curGpuNum);
    assert(cudaStatus == cudaSuccess);
    return curGpuNum;
}

vector<GpuConnectionRanking> MachineEvaluator::getConnectionSpeedRankings(int sourceGpuNum) {
    if (connectionRankings.find(sourceGpuNum) == connectionRankings.end())
        return vector<GpuConnectionRanking>();
    return connectionRankings[sourceGpuNum];
}

int MachineEvaluator::getConnectionSpeedRanking(int sourceGpuNum, int destGpuNum) {
    if (peerConnectionRankings.find(sourceGpuNum) == peerConnectionRankings.end())
        return -1;
    if (peerConnectionRankings[sourceGpuNum].find(destGpuNum) == peerConnectionRankings[sourceGpuNum].end())
        return -1;

    return peerConnectionRankings[sourceGpuNum][destGpuNum];
}

bool MachineEvaluator::isPeerToPeerAvailable(int sourceGpuNum, int destGpuNum) {
    if (peerToPeerEnabled.find(sourceGpuNum) == peerToPeerEnabled.end())
        return false;
    if (peerToPeerEnabled[sourceGpuNum].find(destGpuNum) == peerToPeerEnabled[sourceGpuNum].end())
        return false;

    return peerToPeerEnabled[sourceGpuNum][destGpuNum];
}

void MachineEvaluator::evaluateConnectionSpeeds() {
    cudaError_t cudaStatus;

    int previousGpu;
    cudaStatus = cudaGetDevice(&previousGpu);
    assert(cudaStatus == cudaSuccess);

    for (unsigned int i = 0; i < numGpus; ++i) {
        cudaStatus = cudaSetDevice(i);
        assert(cudaStatus == cudaSuccess);

        for (unsigned int j = 0; j < numGpus; ++j) {
            if (i == j)
                continue;

            int perfRank = 0;
            int accessSupported = 0;
            cudaStatus = cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, i, j);
            assert(cudaStatus == cudaSuccess);
            if (accessSupported) {
                cudaStatus = cudaDeviceEnablePeerAccess(j, 0);
                assert(cudaStatus == cudaSuccess);
            }
            cudaStatus = cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, i, j);
            assert(cudaStatus == cudaSuccess);

            GpuConnectionRanking connectionRanking;
            connectionRanking.peerGpuNum = j;
            connectionRanking.isPeerToPeerSupported = accessSupported;
            connectionRanking.peerToPeerSpeedRanking = accessSupported ? perfRank : 1000;

            connectionRankings[i].push_back(connectionRanking);
        }
        std::sort(connectionRankings[i].begin(), connectionRankings[i].end());
    }

    cudaStatus = cudaSetDevice(previousGpu);
    assert(cudaStatus == cudaSuccess);
}

void MachineEvaluator::getGpuTypes() {
    for (unsigned int i = 0; i < numGpus; ++i) {
        gpuType.push_back(deviceProps[i].name);
    }
}

void MachineEvaluator::getDeviceProps() {
    cudaError_t cudaStatus;

    int previousGpu;
    cudaStatus = cudaGetDevice(&previousGpu);
    assert(cudaStatus == cudaSuccess);

    for (unsigned int i = 0; i < numGpus; ++i) {
        cudaStatus = cudaSetDevice(i);
        assert(cudaStatus == cudaSuccess);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        deviceProps.push_back(deviceProp);
    }

    cudaStatus = cudaSetDevice(previousGpu);
    assert(cudaStatus == cudaSuccess);
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
    cudaError_t cudaStatus;
    int deviceCount;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    assert(cudaStatus == cudaSuccess);
    assert(newGpuNum < deviceCount);

    int previousGpuNum;
    cudaStatus = cudaGetDevice(&previousGpuNum);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaSetDevice(newGpuNum);
    assert(cudaStatus == cudaSuccess);

    return previousGpuNum;
}

void MachineEvaluator::createCopyStreams() {
    for (unsigned int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        ScopedGpu scopedGpu(gpuNum);

        copyStreamFromLower.emplace(gpuNum, gpuNum);
        copyStreamFromHigher.emplace(gpuNum, gpuNum);
        copyStreamFromCpu.emplace(gpuNum, gpuNum);
        copyStreamToCpu.emplace(gpuNum, gpuNum);
        copyStreamLocal.emplace(gpuNum, gpuNum);
    }

    // CPU local stream. (Associated with GPU 0, since it needs to be associated with a GPU)
    ScopedGpu scopedGpu(0);
    copyStreamLocal.emplace(CPU_DEVICE_NUM, 0);
}

unsigned int MachineEvaluator::getNumMultiProcessors(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    return deviceProps[gpuNum].multiProcessorCount;
}

unsigned int MachineEvaluator::getNumMultiProcessors() { return getNumMultiProcessors(getCurrentGpuNum()); }

Stream MachineEvaluator::getCopyStreamFromLower(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    return copyStreamFromLower[gpuNum];
}

Stream MachineEvaluator::getCopyStreamFromHigher(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    return copyStreamFromHigher[gpuNum];
}

Stream MachineEvaluator::getCopyStreamFromCpu(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    return copyStreamFromCpu[gpuNum];
}

Stream MachineEvaluator::getCopyStreamToCpu(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);

    return copyStreamToCpu[gpuNum];
}

Stream MachineEvaluator::getCopyStreamLocal(int gpuNum) {
    assert(gpuNum < (int)numGpus);

    return copyStreamLocal[gpuNum];
}

unsigned long MachineEvaluator::getTotalGlobalMemBytes(int gpuNum) {
    assert(gpuNum < (int)deviceProps.size());
    return deviceProps[gpuNum].totalGlobalMem;
}

cublasLtHandle_t MachineEvaluator::getCublasLtHandle(int gpuNum) {
    assert((unsigned int)gpuNum < numGpus);
    return cublasLtHandlePerDevice[gpuNum];
}
