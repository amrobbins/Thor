#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

struct WorkspaceHelperInfo {
    WorkspaceHelperInfo() {}

    WorkspaceHelperInfo(void *mem, Stream stream, int gpuNum) {
        this->mem = mem;
        this->stream = stream;
        this->gpuNum = gpuNum;
    }

    void *mem;
    Stream stream;
    int gpuNum;
};

class MonitorStreamAndFreeMemExecutor : public WorkQueueExecutorBase<WorkspaceHelperInfo, cudaError_t> {
    cudaError_t operator()(WorkspaceHelperInfo &input) {
        ScopedGpu(input.gpuNum);
        cudaStreamSynchronize(input.stream.getStream());
        cudaError_t cudaStatus = cudaFree(input.mem);
        if (cudaStatus != cudaSuccess) {
            printf("cudaStatus %d\n", cudaStatus);
            fflush(stdout);
        }
        assert(cudaStatus == cudaSuccess);
        // Return value is ignored because work queue is told there is no output.
        return cudaStatus;
    }
};

class WorkspaceHelper {
   public:
    WorkspaceHelper() : activeStreamQueue(false), isRunning(false) {}

    void start() {
        assert(isRunning == false);
        activeStreamQueue.open(std::unique_ptr<MonitorStreamAndFreeMemExecutor>(new MonitorStreamAndFreeMemExecutor), 2, 128);
        isRunning = true;
    }

    void stop() {
        assert(isRunning == true);
        activeStreamQueue.close();
        isRunning = false;
    }

    bool isEmpty() { return isRunning == false || activeStreamQueue.isEmpty(); }

    void deleteWorkspaceAfterUse(void *workspace_d, Stream stream, int gpuNum) {
        activeStreamQueue.push(WorkspaceHelperInfo(workspace_d, stream, gpuNum));
    }

   private:
    WorkQueueUnordered<WorkspaceHelperInfo, cudaError_t> activeStreamQueue;

    bool isRunning;
};
