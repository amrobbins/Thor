#ifndef GPURTREE_H_
#define GPURTREE_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

#include "GpuRTreeKernels.h"
#include "GpuRTreeNode.h"
#include "NearestNeighborExecutor.h"
#include "WorkQueueUnordered.h"

class GpuRtree {
   public:
    GpuRtree(Point points[], long numPoints, bool buildMultiThread, int numDimensions);
    ~GpuRtree();

    GpuRtree() = delete;
    GpuRtree(const GpuRtree &) = delete;
    GpuRtree &operator=(const GpuRtree &) = delete;

    long floatsNeededToPackRtree();
    bool isPacked() { return packed; }
    void pack(float *allocatedSpace, long allocatedSpaceNumBytes);
    void unpack();

    bool isNearestNeighborPipelineSetUp();
    void setUpNearestNeighborPipeline(float *packedQueryPointArray);
    void setUpNearestNeighborPipeline(std::vector<float *> packedQueryPointArrays);
    void setUpNearestNeighborPipeline(std::vector<float *> packedQueryPointArrays, const std::vector<int> &gpus);
    void tearDownNearestNeighborPipeline();
    void getNearestNeighborForPoints(unsigned long firstPointIndex, unsigned long numPoints, long *nearestNeighborIndexes);

    // For testing
    void verifyRtree(Point points[], long numPoints);
    void verifyRtreePacking();

   private:
    Point *points;
    int numDimensions;

    GpuRtreeNode *root;

    bool packed;
    int numGpus;

    bool nearestNeighborPipelineSetUp = false;

    unsigned int numPointsPerBuffer;
    std::vector<PointData> pointReaderQueueBuffers;
    std::vector<WorkQueueUnordered<PointData, PointData> *> pointReaderQueues;
    std::vector<std::vector<GpuPointData>> nearestNeighborQueueBuffers;
    std::vector<WorkQueueUnordered<GpuPointData, GpuPointData> *> nearestNeighborQueues;
    std::vector<int> gpus;

    GpuContext **globalContextPerGpu;

    long floatsNeededToPackSubtree(long level, GpuRtreeNode *subtree);
    void packSubtree(long level, GpuRtreeNode *subtree, float *allocatedSpace, float *endOfAllocatedSpace);
    void unpackSubtree(GpuRtreeNode *subtree);

    void deleteTree(GpuRtreeNode *subtreeRoot);

    void setUpGlobalContext(const std::vector<int> &gpus);
    void tearDownGlobalContext();

    // For testing
    void traverseSubtreeNotingPoints(GpuRtreeNode *node,
                                     Point points[],
                                     std::unordered_set<long> &pointsStillToFindInTree,
                                     std::unordered_map<long, long> &pointIndexToArrayIndex,
                                     float *subtreeDimensionMins,
                                     float *subtreeDimensionMaxes);
    void traverseSubtreeVerifyingPackedData(long level, GpuRtreeNode *subtree);
};

inline float randRange(float min, float max) { return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min))); }

#endif /* GPURTREE_H_ */
