#ifndef GPURTREEQUERYHANDLE_H_
#define GPURTREEQUERYHANDLE_H_

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>

#include "GpuRTreeNode.h"

template <long numDimensions>
class GpuRtreeQueryHandle {
   private:
    int gpuNum;
    cudaStream_t stream[8];

    float2 *boxSpan[8];
    int *subtreePointCount[8];
    bool *flaggedBoxes[8];
    float *boxPoints[8];

    float *queryPoint_d;
    float2 *boxSpan_d[8];
    int *subtreePointCount_d[8];
    bool *flaggedBoxes_d[8];
    float *boxPoints_d[8];

    GpuRtreeQueryHandle() {
        gpuNum = -1;
        queryPoint_d = NULL;

        for (int i = 0; i < 8; i++) {
            boxSpan[i] = NULL;
            subtreePointCount[i] = NULL;
            flaggedBoxes[i] = NULL;
            boxPoints[i] = NULL;

            boxSpan_d[i] = NULL;
            subtreePointCount_d[i] = NULL;
            flaggedBoxes_d[i] = NULL;
            boxPoints_d[i] = NULL;
        }
    }

    GpuRtreeQueryHandle(const GpuRtreeQueryHandle &other) {
        gpuNum = other.gpuNum;
        queryPoint_d = other.queryPoint_d;
        for (int i = 0; i < 8; ++i) {
            stream[i] = other.stream[i];
            boxSpan[i] = other.boxSpan[i];
            subtreePointCount[i] = other.subtreePointCount[i];
            flaggedBoxes[i] = other.flaggedBoxes[i];
            boxPoints[i] = other.boxPoints[i];
            boxSpan_d[i] = other.boxSpan_d[i];
            subtreePointCount_d[i] = other.subtreePointCount_d[i];
            flaggedBoxes_d[i] = other.flaggedBoxes_d[i];
            boxPoints_d[i] = other.boxPoints_d[i];
        }
    }

    GpuRtreeQueryHandle &operator=(const GpuRtreeQueryHandle &other) {
        *this = GpuRtreeQueryHandle(other);
        return *this;
    }

    bool isHandleInitialized() { return queryPoint_d != NULL; }

    // Allocates the memory, creates the streams etc that will be used for this handle
    void initializeHandle(int gpuNum) {
        if (isHandleInitialized()) {
            printf(
                "Library Internal ERROR: trying to initialize a handle that is already initialized (this one is for "
                "gpu %d\n",
                gpuNum);
            fflush(stdout);
        }
        assert(!isHandleInitialized());  // Make sure that it is not already initialized

        cudaError_t cudaStatus;

        int numDevices;
        cudaStatus = cudaGetDeviceCount(&numDevices);
        assert(cudaStatus == cudaSuccess);
        assert(gpuNum >= 0);
        assert(gpuNum < numDevices);
        cudaStatus = cudaSetDevice(gpuNum);
        assert(cudaStatus == cudaSuccess);

        this->gpuNum = gpuNum;

        cudaStatus = cudaMalloc(&queryPoint_d, numDimensions * sizeof(float));
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < 8; i++) {
            cudaStatus = cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaHostAlloc(&(boxSpan[i]), 1024 * numDimensions * sizeof(float2), cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&(subtreePointCount[i]), 1024 * sizeof(int), cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&(flaggedBoxes[i]), 1024 * sizeof(bool), cudaHostAllocDefault);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&(boxPoints[i]), 1024 * numDimensions * sizeof(float), cudaHostAllocDefault);
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaMalloc(&(boxSpan_d[i]), 1024 * numDimensions * sizeof(float2));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMalloc(&(subtreePointCount_d[i]), 1024 * sizeof(int));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMalloc(&(flaggedBoxes_d[i]), 1024 * sizeof(bool));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMalloc(&(boxPoints_d[i]), 1024 * numDimensions * sizeof(float));
            assert(cudaStatus == cudaSuccess);
        }
    }

    void destroyHandle() {
        if (isHandleInitialized()) {
            if (gpuNum == -1)  // never initialized
                printf("ERROR: attempt to destroy a handle that has never been initialized (this one is for gpu %d)\n", gpuNum);
            else
                printf(
                    "ERROR: attempt to destroy a handle that has already been destroyed and not reinitialized (this "
                    "one is for gpu %d)\n",
                    gpuNum);
            fflush(stdout);
        }
        assert(!isHandleInitialized());  // Make sure that it is currently initialized

        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(gpuNum);
        assert(cudaStatus == cudaSuccess);

        cudaStatus = cudaFree(queryPoint_d);
        assert(cudaStatus == cudaSuccess);
        queryPoint_d = NULL;

        for (int i = 0; i < 8; i++) {
            cudaStatus = cudaStreamSynchronize(stream[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaStreamDestroy(stream[i]);
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaFreeHost(boxSpan[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(subtreePointCount[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(flaggedBoxes[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(boxPoints[i]);
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaFree(boxSpan_d[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFree(subtreePointCount_d[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFree(flaggedBoxes_d[i]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFree(boxPoints_d[i]);
            assert(cudaStatus == cudaSuccess);

            boxSpan[i] = NULL;
            subtreePointCount[i] = NULL;
            flaggedBoxes[i] = NULL;

            boxSpan_d[i] = NULL;
            subtreePointCount_d[i] = NULL;
            flaggedBoxes_d[i] = NULL;
        }
    }

    //    void findNearestNeighbor(float *queryPoint, float *nearestNeighbor) {
    //
    //        assert(point_d != NULL);   // Make sure that it is currently initialized
    //        assert(overallNumPoints > 0);
    //
    //        cudaError_t cudaStatus;
    //        cudaStatus = cudaSetDevice(gpuContext.gpuNum);
    //        assert(cudaStatus == cudaSuccess);
    //
    //        // level 0 and level 1 of the rtree are stored in gpu memory
    //        // Get the level 0 boxes that contain the point
    //        // In a bulk loaded rtree, this will often just be one box, but the search is written to work in general.
    //        cudaStatus = cudaMemcpyAsync(point_d, queryPoint, numDimensions * sizeof(float), cudaMemcpyHostToDevice,
    //        stream[0]); assert(cudaStatus == cudaSuccess);
    //
    //        dim3 gridSize = dim3(4);
    //        dim3 blockSize = dim3(256);
    //        getMinAndMaxDistanceToAllBoxes<<<gridSize, blockSize, 0, stream[0]>>>(point_d, numDimensions, boxSpanL0_d,
    //        boxValidL0_d, boxSquaredDistanceExtremes_d[0]); cudaStatus =
    //        cudaMemcpyAsync(boxSquaredDistanceExtremes[0], boxSquaredDistanceExtremes_d[0], 1024 * sizeof(float2));
    //        assert(cudaStatus == cudaSuccess);
    //        cudaStatus = cudaStreamSynchronize(stream[0]);
    //        assert(cudaStatus == cudaSuccess);
    //
    //        float minMin = -1.0f, minMax = -1.0f;
    //        for(int i = 0; i < 1024; i++) {
    //            if(boxSquaredDistanceExtremes.x >= 0.0f && boxSquaredDistanceExtremes.x < minMin) {
    //                minMin = boxSquaredDistanceExtremes.x;
    //            }
    //            if(boxSquaredDistanceExtremes.y >= 0.0f && boxSquaredDistanceExtremes.y < minMax) {
    //                minMax = boxSquaredDistanceExtremes.y;
    //            }
    //        }
    //
    //        std::vector<std::unordered_set<int>> boxesToSearchIn;
    //        boxesToSearchIn.emplace_back();
    //        for(int i = 0; i < 1024; i++) {
    //            if(boxSquaredDistanceExtremes.x == minMin || boxSquaredDistanceExtremes.x == minMax) {
    //                boxesToSearchIn.back().insert(i);
    //            }
    //        }
    //
    //        std::vector<std::unordered_set<int>> leavesSearched;
    //        float squaredRadiusBound;
    //        rtreeLevelSearchForSmallRadiusBoundNeighbor(boxesToSearchIn, squaredRadiusBound, leavesSearched);
    //        rtreeLevelSearchForNearestNeighbor(squaredRadiusBound, leavesSearched, nearestNeighbor);
    //    }
    //
    //    void rtreeLevelSearchForSmallRadiusBound(std::vector<std::unordered_set<int>> &boxesToSearchIn, float
    //    &squaredRadiusBound, std::vector<std::unordered_set<int>> &leavesSearched) {
    //        int leafLevel = boxesToSearchIn.size()-1;
    //        int boxLevel = leafLevel + 1;
    //
    //        if(leafLevel == 0) {
    //            dim3 gridSize = dim3(4);
    //            dim3 blockSize = dim3(128);
    //            for(auto boxNumberIter = boxesToSearchIn.back().begin(); boxNumberIter !=
    //            boxesToSearchIn.back().end(); ++boxNumberIter) {
    //                FIXME launch up to 8 of these at a time
    //                if(boxIsLeafL0[]) {
    //                    getNearestPointInLeafNode<<<gridSize, blockSize, 0, fixmeStream>>>(point_d, numDimensions,
    //                    float* boxesPoints, bool *pointValidFlags, int *indexOfNearestPoint, float
    //                    *squaredDistanceToNearestPoint); fixme save off distance and index of nearest point also use
    //                    this info to prune away boxes whose nearest distance is less than the currently known nearest
    //                    point
    //                }
    //            }
    //        } else if leafLevel == 1) {
    //        } else {
    //        }
    //
    //        if(boxLevel == 1) {
    //        } else {
    //        }
    //
    //    }
    //
    //
    //
    //
    //    void findNearestNeighbor_v1(float *queryPoint, float *nearestNeighbor) {
    //
    //        assert(point_d != NULL);   // Make sure that it is currently initialized
    //        assert(overallNumPoints > 0);
    //
    //        cudaError_t cudaStatus;
    //        cudaStatus = cudaSetDevice(gpuContext.gpuNum);
    //        assert(cudaStatus == cudaSuccess);
    //
    //        // level 0 and level 1 of the rtree are stored in gpu memory
    //        // Get the level 0 boxes that contain the point
    //        // In a bulk loaded rtree, this will often just be one box, but the search is written to work in general.
    //        cudaStatus = cudaMemcpyAsync(point_d, queryPoint, numDimensions * sizeof(float), cudaMemcpyHostToDevice,
    //        stream[0]); assert(cudaStatus == cudaSuccess);
    //
    //        dim3 gridSize = dim3(4);
    //        dim3 blockSize = dim3(256);
    //        findBoxesThatContainPoint<<<gridSize, blockSize, 0, stream[0]>>>(point_d, numDimensions, boxSpanL0_d,
    //        boxValidL0_d, flaggedBoxes_d[0]); cudaStatus = cudaMemcpyAsync(flaggedBoxes[0], flaggedBoxes_d[0], 1024 *
    //        sizeof(bool)); assert(cudaStatus == cudaSuccess);
    //
    //        // While the first box search is occurring, check if point is in the overall box and compute the distance
    //        for the case that it is not
    //        // This overlap is to speed up the common case where the point is inside the box
    //        float squaredDistanceToRtree = 0.0f;
    //        for(int d = 0; d < numDimensions; d++) {
    //            if(queryPoint[d] < overallBottomLeftCorner[d]) {
    //                float diff = overallBottomLeftCorner[d] - queryPoint[d];
    //                squaredDistanceToRtree += diff * diff;
    //            } else if(queryPoint[d] > overallTopRightCorner[d]) {
    //                float diff = overallTopRightCorner[d] - queryPoint[d];
    //                squaredDistanceToRtree += diff * diff;
    //            }
    //        }
    //
    //        cudaStatus = cudaStreamSynchronize(stream[0]);
    //        assert(cudaStatus == cudaSuccess);
    //
    //        std::vector<long> boxesToSearchIn;
    //        for(int i = 0; i < 1024; i++) {
    //            if(flaggedBoxes[0][i]) {
    //                boxesToSearchIn.push_back(i);
    //            }
    //        }
    //
    //        // If no boxes contain the query point, then resort to a radius expansion search for a box that contains a
    //        nearest neighbor candidate if(boxesToSearchIn.empty()) {
    //            FIXME: Since the point is not inside a inner box in this outer box, it will never be inside an a
    //            deeper level inner box, since they are all completely inside this levels boxes
    //                   In this case, maybe I should create std::unordered_set<long>
    //                   boxesToSearchDeeperWithRadiusExpansion Also, since radius expansion is slower, I should do this
    //                   after drill search and abort early if I search at a distance greater than the current best
    //                   candidate Maybe I should only do the radius expansion searchs if drilling all the way down does
    //                   not produce a candidate
    //            int successfulStream = radiusExpansionSearch(point_d, squaredDistanceToRtree, boxSpanL0_d,
    //            boxValidL0_d, -1.0f); assert(successfulStream >= 0); for(int i = 0; i < 1024; i++) {
    //                if(flaggedBoxes[successfulStream][i]) {
    //                    fixme set so-far-nearest-neighbor-candidate to be max distance inside box from point, if this
    //                    distance is less than the current dist to NN candidate boxesToSearchIn.push_back(i);
    //                }
    //            }
    //        }
    //        assert(!boxesToSearchIn.empty());
    //
    //        std::unordered_set<long> leavesSearched;
    //        std::unordered_set<long> boxesToDrillDeeper;
    //        float nearestPointSquaredDistance = -1.0f;
    //        for(auto boxNumberIter = boxesToSearchIn.begin(); boxNumberIter != boxesToSearchIn.end(); ++boxNumberIter)
    //        {
    //            if(boxIsLeafL0[*boxNumberIter]) {
    //                leavesSearched.insert(*boxNumberIter);
    //                dim3 gridSize = dim3(4);
    //                dim3 blockSize = dim3(128);
    //                FIXME get the point, and keep if it is the nearest so far. looks like I will need to allocate
    //                squaredDistanceToNearestPoint(_d)[8][4] as part of the handle
    //                getNearestPointInLeafNode<<<gridSize, blockSize, 0, fixmeStream>>>(point_d, numDimensions, float*
    //                boxesPoints, bool *pointValidFlags, int *indexOfNearestPoint, float
    //                *squaredDistanceToNearestPoint);
    //            } else {
    //                boxesToDrillDeeper.insert(*boxNumberIter);
    //            }
    //        }
    //
    //        // Now I need to iterate over all boxes to drill deeper,
    //        // I could call a function at this point, but there is the L1 memory, that function could know about that.
    //        if(!boxesToDrillDeeper.empty()) {
    //            stepDeeperForNearestNeighborCandidate(1, boxesToDrillDeeper, nearestNeighborCandidate,
    //            nearestNeighborCandidateDistanceSquared);
    //        }
    //
    //        // Now we have our best nearest neighbor candidate, so do an exhaustive search over the radius to this
    //        nearest neighbor candidate
    //        // No need to search in **leaf nodes** that have already been searched.
    //        // I do need to search in boxes since the candidate search is not an exhaustive search, there may be
    //        nearer neighbors than it returns.
    //    }
    //
    //    void stepDeeperForNearestNeighborCandidate(int level, set<Long> &incomingBoxesToDrillDeeper, float
    //    *nearestNeighborCandidate, float nearestNeighborCandidateDistanceSquared) {
    //
    //
    //
    //
    //
    //
    //        int kernelsInProgress = 0;
    //        int oldestStream = 0;
    //        int kernelsLaunched = 0;
    //        boxesToSearchIn.emplace_back();
    //        for(auto iter = boxesToSearchIn.begin(); iter != boxesToSearchIn.end(); ++iter) {
    //            int boxNumber = *iter;
    //
    //            int nextStream = (oldestStream + kernelsInProgress) % 8;
    //            findBoxesThatContainPoint<<<gridSize, blockSize, 0, stream[nextStream]>>>(point_d, numDimensions,
    //            boxSpanL1_d[1024 * boxNum], boxValidL1_d[1024 * boxNum], flaggedBoxes_d[nextStream]); cudaStatus =
    //            cudaMemcpyAsync(flaggedBoxes[nextStream], flaggedBoxes_d[nextStream], 1024 * sizeof(bool));
    //            kernelsInProgress++;
    //            kernelsLaunched++;
    //            if(kernelsInProgress == 8) {
    //                cudaStatus = cudaStreamSynchronize(stream[oldestStream]);
    //                assert(cudaStatus == cudaSuccess);
    //
    //                // now get the list of flagged boxes
    //                for(int i = 0; i < 1024; i++)
    //                    boxesToSearchIn[1].insert(boxNumber * 1024 + i);
    //
    //                oldestStream = (oldestStream + 1) % 8;
    //                kernelsInProgress--;
    //            }
    //        }
    //        while(kernelsInProgress != 0) {
    //            cudaStatus = cudaStreamSynchronize(stream[oldestStream]);
    //            assert(cudaStatus == cudaSuccess);
    //            oldestStream = (oldestStream + 1) % 8;
    //            kernelsInProgress--;
    //        }
    //        if(kernelsLaunched == 0) {
    //            // FIXME: case where point is not in the box. Can't test this by checking overall box alone, because
    //            that assumes other boxes are dense.
    //            // However if I do have the overall dimensions, I can catch some cases where the point is not in a
    //            box. Wonder if its worth making the extra check? Maybe depends on implementation.
    //        }
    //    }
    //
    //    int radiusExpansionSearch(float *queryPoint_d, float *squaredDistanceToRtree, float2 *boxSpan_d, bool
    //    *boxValid_d, int maxSquaredSearchRadius) {
    //        cudaError_t cudaStatus;
    //        float squaredSearchRadius = defaultSquaredSearchRadius;
    //        int nextStream = 0;
    //        int oldestStream = 0;
    //        int kernelsInProgress = 0;
    //        cudaEvent_t completedEvent[8];
    //        int streamThatFoundABox = -1;
    //        constexpr int MAX_KERNELS_IN_PROGRESS = 8;
    //
    //        while(squaredSearchRadius < maxSquaredSearchRadius || maxSquaredSearchRadius < 0.0f) {
    //            dim3 gridSize = dim3(4);
    //            dim3 blockSize = dim3(256);
    //            findBoxesWithinRadius<<<gridSize, blockSize, 0, stream[nextStream]>>>(queryPoint_d,
    //            squaredSearchRadius + squaredDistanceToRtree, numDimensions, boxSpan_d, boxValid_d,
    //            flaggedBoxes_d[nextStream]); cudaMemcpyAsync(flaggedBoxes[nextStream], flaggedBoxes_d[nextStream],
    //            1024 * sizeof(bool), cudaMemcpyDeviceToHost, streams[nextStream]);
    //            cudaEventCreateWithFlags(completedEvent[nextStream], cudaEventDisableTiming);
    //            cudaEventRecord(completedEvent[nextStream], stream[nextStream]);
    //            squaredSearchRadius *= radiusIncreaseFactor;
    //            kernelsInProgress++;
    //            nextStream++;
    //            if(nextStream == 8)
    //                nextStream = 0;
    //
    //            do {
    //                cudaStatus = cudaEventQuery(completedEvent[oldestStream]);
    //                if(cudaStatus == cudaSuccess) {   // if the event has completed
    //                    cudaStatus = cudaEventDestroy(completedEvent[oldestStream]);
    //                    assert(cudaStatus == cudaSuccess);
    //
    //                    for(int i = 0; i < 1024; i++) {
    //                        if(flaggedBoxes[oldestStream][i]) {
    //                            streamThatFoundABox = oldestStream;
    //                            break;
    //                        }
    //                    }
    //
    //                    oldestStream++;
    //                    if(oldestStream == 8)
    //                        oldestStream = 0;
    //                    kernelsInProgress--;
    //
    //                    if(streamThatFoundABox != -1)
    //                        break;
    //                } else {
    //                    assert(cudaStatus == cudaErrorNotReady);    // event is still in progress
    //                }
    //            } while(kernelsInProgress == MAX_KERNELS_IN_PROGRESS);
    //
    //            if(streamThatFoundABox != -1)
    //                break;
    //            // TODO: end the search with an error when it is beyond all outer boundaries of the box?
    //        }
    //
    //        // destroy the events, they may be destroyed before they are completed. The only outputs to this kernel
    //        are flaggedBoxes_d[stream] and this is only to be used in its stream,
    //        // so it may be reused right away on that stream and it will not be interfered with by these kernels.
    //        while(kernelsInProgress > 0) {
    //            cudaStatus = cudaEventDestroy(completedEvent[oldestStream]);
    //            assert(cudaStatus == cudaSuccess);
    //            oldestStream++;
    //            if(oldestStream == 8)
    //                oldestStream = 0;
    //            kernelsInProgress--;
    //        }
    //
    //        return streamThatFoundABox;
    //    }
};

#endif /* GPURTREEQUERYHANDLE_H_ */
