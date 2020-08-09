#ifndef NEARESTNEIGHBOREXECUTOR_H_
#define NEARESTNEIGHBOREXECUTOR_H_

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <deque>
#include <list>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>
#include "GpuRTreeKernels.h"
#include "GpuRTreeNode.h"
#include "WorkQueueUnordered.h"

struct GpuContext;

struct PointData {
    unsigned long firstPointIndex;
    int numPoints;
    float *mem = NULL;
};

// FIXME: I only need to allocate some buffers when the tree is 3 levels deep or more
class NearestNeighborGpuQueryHandle {
   public:
    inline int getGpuNum() { return gpuNum; }
    inline cudaStream_t getStream() { return stream; }

    inline int2 *getMinMinAndMinMaxBoxIndexesPerPoint() { return minMinAndMinMaxBoxIndexesPerPoint; }
    inline float *getBoxSpanOrLeafPoints() { return boxSpanOrLeafPoints; }
    inline uint *getFlaggedBoxes() { return flaggedBoxes; }
    inline int *getIndexOfNearestPoint() { return indexOfNearestPoint; }
    inline float *getSquaredDistanceToNearestPoint() { return squaredDistanceToNearestPoint; }

    inline float *getQueryPoint_d() { return queryPoint_d; }
    inline float2 *getMinAndMaxDistanceToAllBoxesPerPoint_d() { return minAndMaxDistanceToAllBoxesPerPoint_d; }
    inline int2 *getMinMinAndMinMaxBoxIndexesPerPoint_d() { return minMinAndMinMaxBoxIndexesPerPoint_d; }
    inline float2 *getBoxSpanOrLeafPoints_d() { return boxSpanOrLeafPoints_d; }
    inline uint *getFlaggedBoxes_d() { return flaggedBoxes_d; }
    inline int *getIndexOfNearestPoint_d() { return indexOfNearestPoint_d; }
    inline float *getSquaredDistanceToNearestPoint_d() { return squaredDistanceToNearestPoint_d; }

    NearestNeighborGpuQueryHandle() = delete;
    NearestNeighborGpuQueryHandle(const NearestNeighborGpuQueryHandle &) = delete;
    NearestNeighborGpuQueryHandle &operator=(const NearestNeighborGpuQueryHandle &) = delete;

    NearestNeighborGpuQueryHandle(int gpuNum, int numDimensions, int maxNumPointsPerQuery);
    virtual ~NearestNeighborGpuQueryHandle();

   private:
    int gpuNum;
    cudaStream_t stream;

    int numDimensions;
    int bufferWidth;

    int maxNumPointsPerQuery;

    int2 *minMinAndMinMaxBoxIndexesPerPoint;
    float *boxSpanOrLeafPoints;
    uint *flaggedBoxes;
    int *indexOfNearestPoint;
    float *squaredDistanceToNearestPoint;

    float *queryPoint_d;
    float2 *minAndMaxDistanceToAllBoxesPerPoint_d;
    int2 *minMinAndMinMaxBoxIndexesPerPoint_d;
    float2 *boxSpanOrLeafPoints_d;
    uint *flaggedBoxes_d;
    int *indexOfNearestPoint_d;
    float *squaredDistanceToNearestPoint_d;
};

struct GpuPointData {
    PointData cpuPointData;
    long *nearestNeighborIndexes;
};

class PackedArrayPointLoaderExecutor : public ExecutorBase<PointData, PointData> {
   public:
    PackedArrayPointLoaderExecutor(float *packedQueryPoints, int numDimensions) {
        assert(numDimensions > 0);

        this->packedQueryPoints = packedQueryPoints;
        this->numDimensions = numDimensions;
    }

    PointData operator()(const PointData &pointData) {
        // memcpy(pointData.mem, &(packedQueryPoints[pointData.firstPointIndex * numDimensions]), pointData.numPoints *
        // numDimensions * sizeof(float));
        cudaMemcpy(pointData.mem,
                   &(packedQueryPoints[pointData.firstPointIndex * numDimensions]),
                   pointData.numPoints * numDimensions * sizeof(float),
                   cudaMemcpyHostToHost);
        return pointData;
    }

   private:
    float *packedQueryPoints;
    int numDimensions;
};

class NearestNeighborExecutor : public ExecutorBase<GpuPointData, GpuPointData> {
   public:
    NearestNeighborExecutor(int gpuNum,
                            int numDimensions,
                            int numHandlesToCreate,
                            int maxNumQueryPointsPerBuffer,
                            GpuContext *globalGpuContext,
                            GpuRtreeNode *root);
    virtual ~NearestNeighborExecutor();

    GpuPointData operator()(const GpuPointData &gpuPointData);

   private:
    int gpuNum;
    int numDimensions;
    int maxNumQueryPointsPerBuffer;
    std::vector<NearestNeighborGpuQueryHandle *> gpuHandles;
    std::mutex gpuHandleQueueMutex;
    GpuContext *globalGpuContext;
    GpuRtreeNode *root;

    void getClosePoint(const GpuPointData &gpuPointData,
                       GpuPointData &result,
                       std::vector<std::set<std::vector<int>>> &leafNodesSearchedPerPoint,
                       NearestNeighborGpuQueryHandle *queryHandle,
                       std::vector<float> &distanceToCurrentNearestNeigbor);
    void getClosestPoint(const GpuPointData &gpuPointData,
                         GpuPointData &result,
                         std::vector<std::set<std::vector<int>>> &leafNodesSearchedPerPoint,
                         NearestNeighborGpuQueryHandle *queryHandle,
                         std::vector<float> &distanceToCurrentNearestNeigbor);

    inline GpuRtreeNode *getNode(const std::vector<int> &nodePath);
    static inline void getBoxNumbersFromFlagsAndAddToPath(unsigned int flags[32],
                                                          std::vector<int> pathToHere,
                                                          std::deque<std::vector<int>> &boxNumbers);
};

#endif /* NEARESTNEIGHBOREXECUTOR_H_ */
