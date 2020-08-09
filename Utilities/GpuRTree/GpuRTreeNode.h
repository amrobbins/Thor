#ifndef GPURTREENODE_H_
#define GPURTREENODE_H_

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class GpuRtree;
class GpuRtreeQueryHandle;
class NearestNeighborExecutor;

constexpr long MIN_POINTS_TO_SPLIT_NODES = 25;

struct Point {
    float *mem;
    long pointIndex;

    Point &operator=(const Point &other) {
        this->mem = other.mem;
        this->pointIndex = other.pointIndex;
        return *this;
    }
};

struct LessThanInDimension {
    long dimension;
    LessThanInDimension(long dimension) { this->dimension = dimension; }

    bool operator()(Point &lhs, Point &rhs) const { return lhs.mem[dimension] < rhs.mem[dimension]; }
};

struct PointCount {
    long boxIndex;
    long pointCount;

    PointCount(long index, long count) {
        boxIndex = index;
        pointCount = count;
    }
};

struct LessThanByCount {
    bool operator()(const PointCount &lhs, const PointCount &rhs) const { return lhs.pointCount < rhs.pointCount; }
};

struct DimensionSpan {
    float span;
    long dimensionNumber;

    DimensionSpan(float span, long dimensionNumber) {
        this->span = span;
        this->dimensionNumber = dimensionNumber;
    }
};

struct LessThanBySpan {
    bool operator()(const DimensionSpan &lhs, const DimensionSpan &rhs) const { return lhs.span < rhs.span; }
};

struct GpuContext {
    // The first 2 levels are pre-loaded into GPU:
    float2 *boxSpanL0_d = NULL;
    float *leafPointsL0_d = NULL;

    std::vector<float2 *> boxSpanL1_d;
    std::vector<float *> leafPointsL1_d;

    float2 *boxSpanL1_d_mem = NULL;
    float *leafPointsL1_d_mem = NULL;

    GpuContext() {}
    virtual ~GpuContext() { tearDown(); }

    GpuContext(const GpuContext &) = delete;
    GpuContext &operator=(const GpuContext &) = delete;

    void tearDown() {
        cudaError_t cudaStatus;

        if (boxSpanL0_d != NULL) {
            cudaStatus = cudaFree(boxSpanL0_d);
            assert(cudaStatus == cudaSuccess);
            boxSpanL0_d = NULL;
        }
        if (leafPointsL0_d != NULL) {
            cudaStatus = cudaFree(leafPointsL0_d);
            assert(cudaStatus == cudaSuccess);
            leafPointsL0_d = NULL;
        }
        if (boxSpanL1_d_mem != NULL) {
            cudaStatus = cudaFree(boxSpanL1_d_mem);
            assert(cudaStatus == cudaSuccess);
            boxSpanL1_d_mem = NULL;
        }
        boxSpanL1_d = std::vector<float2 *>();
        if (leafPointsL1_d_mem != NULL) {
            cudaStatus = cudaFree(leafPointsL1_d_mem);
            assert(cudaStatus == cudaSuccess);
            leafPointsL1_d_mem = NULL;
        }
        leafPointsL1_d = std::vector<float *>();
    }
};

class GpuRtreeNode {
    GpuRtreeNode() = delete;
    GpuRtreeNode(const GpuRtreeNode &) = delete;
    GpuRtreeNode &operator=(const GpuRtreeNode &) = delete;

   private:
    long nodeSize = 1024;
    int numDimensions;

    float *dimensionMins = NULL;
    float *dimensionMaxes = NULL;

    float *packedNodeData = NULL;  // contains the spans for a non-leaf. contains the points for a leaf.

    bool isLeaf() { return children.empty(); }

    long numSubtreePoints;  // for a leaf this is just the number of points at this leaf node. For a non-leaf it is the
                            // number of points off all leaf nodes in the subtree.
    std::vector<GpuRtreeNode *> children;

    Point *leafPoints;

    GpuRtreeNode(long numPoints, Point *points, int numDimensions);
    void treeify(long level, bool multithread);

    // returns the element number that is the pivot
    void partitionPoints(Point *points, long dimension, long numPoints, long pivot, GpuRtreeNode *newChild, float &newChildDimensionMin) {
        assert(numPoints > 1);

        std::nth_element(points, points + pivot, points + numPoints, LessThanInDimension(dimension));
        newChildDimensionMin = newChild->getDimensionMin(dimension);

        // for(long i = pivot - 3; i <= pivot + 3; i++)
        //    printf("%f ", points[i].mem[dimension]);
        // printf("   : %ld\n", dimension);
    }

    void splitNode(bool threadSafe,
                   std::mutex *mtx,
                   long chosenBox,
                   std::priority_queue<PointCount, std::vector<PointCount>, LessThanByCount> *boxPointCounts,
                   std::vector<std::priority_queue<DimensionSpan, std::vector<DimensionSpan>, LessThanBySpan>> *childDimensionSpans);

    void setMinAndMaxPerDimension();
    float getDimensionMin(int dimension);

    friend class GpuRtree;
    friend class GpuRtreeQueryHandle;
    friend class NearestNeighborExecutor;
};

#endif /* GPURTREENODE_H_ */
