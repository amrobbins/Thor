#include "NearestNeighborExecutor.h"

NearestNeighborExecutor::NearestNeighborExecutor(int gpuNum,
                                                 int numDimensions,
                                                 int numHandlesToCreate,
                                                 int maxNumQueryPointsPerBuffer,
                                                 GpuContext *globalGpuContext,
                                                 GpuRtreeNode *root) {
    this->gpuNum = gpuNum;
    this->numDimensions = numDimensions;

    for (int b = 0; b < numHandlesToCreate; ++b) {
        gpuHandles.push_back(new NearestNeighborGpuQueryHandle(gpuNum, numDimensions, maxNumQueryPointsPerBuffer));
    }
    this->maxNumQueryPointsPerBuffer = maxNumQueryPointsPerBuffer;
    this->globalGpuContext = globalGpuContext;
    this->root = root;
}

NearestNeighborExecutor::~NearestNeighborExecutor() {
    while (!gpuHandles.empty()) {
        delete gpuHandles.back();
        gpuHandles.pop_back();
    }
}

// FIXME: optimization: batch first iteration of launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint after getting it
// working, only batch first iteration
// FIXME: optimization: levels 2+ could be size 32. Still use 256 threads though. Then batch up a bunch of searches and
// send the offset into the data pointer.
// FIXME: optimization: I could make all levels size 32, and batch up points that belong to the same box at each level.
// FIXME: optimization: multiple streams per thread
// FIXME: optimization: first level could be size 256 for economy mode.
// FIXME: optimization: if the tree is less than N nodes (maybe 8 or 16) just do an exhaustive search in all leafs using
// launchGetNearestPointInLeafNode. Check actual speed to choose N.
GpuPointData NearestNeighborExecutor::operator()(const GpuPointData &gpuPointData) {
    GpuPointData result = gpuPointData;
    std::vector<std::set<std::vector<int>>> leafNodesSearchedPerPoint;
    std::vector<float> distanceToCurrentNearestNeigbor;

    gpuHandleQueueMutex.lock();
    assert(!gpuHandles.empty());
    NearestNeighborGpuQueryHandle *queryHandle = gpuHandles.back();
    gpuHandles.pop_back();
    gpuHandleQueueMutex.unlock();

    getClosePoint(gpuPointData, result, leafNodesSearchedPerPoint, queryHandle, distanceToCurrentNearestNeigbor);
    if (root->isLeaf())
        return result;

    getClosestPoint(gpuPointData, result, leafNodesSearchedPerPoint, queryHandle, distanceToCurrentNearestNeigbor);

    gpuHandleQueueMutex.lock();
    gpuHandles.push_back(queryHandle);
    gpuHandleQueueMutex.unlock();

    return result;
}

// Uses a heurisitic to get a point that is close to the query point quickly
// when the root is a leaf, getClosePoint will always yield the closest point
void NearestNeighborExecutor::getClosePoint(const GpuPointData &gpuPointData,
                                            GpuPointData &result,
                                            std::vector<std::set<std::vector<int>>> &leafNodesSearchedPerPoint,
                                            NearestNeighborGpuQueryHandle *queryHandle,
                                            std::vector<float> &distanceToCurrentNearestNeigbor) {
    cudaError_t cudaStatus;
    assert(gpuPointData.cpuPointData.numPoints <= maxNumQueryPointsPerBuffer);
    cudaStatus = cudaSetDevice(gpuNum);
    assert(cudaStatus == cudaSuccess);
    cudaStream_t stream = queryHandle->getStream();
    int numPoints = gpuPointData.cpuPointData.numPoints;
    cudaStatus = cudaMemcpyAsync(queryHandle->getQueryPoint_d(),
                                 gpuPointData.cpuPointData.mem,
                                 numPoints * numDimensions * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 stream);
    assert(cudaStatus == cudaSuccess);

    leafNodesSearchedPerPoint.clear();
    leafNodesSearchedPerPoint.reserve(numPoints);
    distanceToCurrentNearestNeigbor.clear();
    distanceToCurrentNearestNeigbor.reserve(numPoints);
    for (int queryPointNum = 0; queryPointNum < numPoints; ++queryPointNum) {
        distanceToCurrentNearestNeigbor.push_back(-1.0f);
        leafNodesSearchedPerPoint.emplace_back();
    }

    std::unordered_map<int, std::deque<std::vector<int>>> boxesToSearchPerPoint;
    boxesToSearchPerPoint.reserve(numPoints);
    for (int queryPointNumber = 0; queryPointNumber < numPoints; ++queryPointNumber) {
        boxesToSearchPerPoint[queryPointNumber].emplace_back();
        boxesToSearchPerPoint[queryPointNumber].back().emplace_back();  // For each point, I need to search the root.
    }

    while (!boxesToSearchPerPoint.empty()) {
        // Launch one kernel per point and then synchronize until there are no more kernels to launch
        std::unordered_set<int> isLeaf;
        std::unordered_set<int> isNonLeaf;
        std::unordered_map<int, std::vector<int>> initialPathPerPoint;
        for (auto pointBoxesIter = boxesToSearchPerPoint.begin(); pointBoxesIter != boxesToSearchPerPoint.end();
             /*NOP*/) {
            int queryPointNumber = pointBoxesIter->first;
            std::deque<std::vector<int>> &boxesToSearchForThisPoint = pointBoxesIter->second;

            // Find a box to search
            // If there isn't one that has not already been searched then delete the entry for this point and continue
            std::vector<int> boxToSearch;
            if (boxesToSearchForThisPoint.empty()) {
                auto oldIter = pointBoxesIter;
                ++pointBoxesIter;
                boxesToSearchPerPoint.erase(oldIter);
                continue;
            }
            do {
                boxToSearch = boxesToSearchForThisPoint.front();
                boxesToSearchForThisPoint.pop_front();
            } while (leafNodesSearchedPerPoint[queryPointNumber].find(boxToSearch) != leafNodesSearchedPerPoint[queryPointNumber].end() &&
                     !boxesToSearchForThisPoint.empty());
            if (leafNodesSearchedPerPoint[queryPointNumber].find(boxToSearch) != leafNodesSearchedPerPoint[queryPointNumber].end()) {
                auto oldIter = pointBoxesIter;
                ++pointBoxesIter;
                boxesToSearchPerPoint.erase(oldIter);
                continue;
            }
            initialPathPerPoint[queryPointNumber] = boxToSearch;

            GpuRtreeNode *node = getNode(boxToSearch);
            if (boxToSearch.size() == 0) {
                if (node->isLeaf()) {
                    launchGetNearestPointInLeafNode(stream,
                                                    queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                                                    numDimensions,
                                                    node->numSubtreePoints,
                                                    globalGpuContext->leafPointsL0_d,
                                                    queryHandle->getIndexOfNearestPoint_d(),
                                                    queryHandle->getSquaredDistanceToNearestPoint_d());
                    leafNodesSearchedPerPoint[queryPointNumber].insert(boxToSearch);
                    isLeaf.insert(queryPointNumber);
                } else {
                    launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(
                        queryHandle->getStream(),
                        queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                        numDimensions,
                        globalGpuContext->boxSpanL0_d,
                        node->children.size(),
                        queryHandle->getMinAndMaxDistanceToAllBoxesPerPoint_d() + queryPointNumber * 1024,
                        queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d() + queryPointNumber);
                    isNonLeaf.insert(queryPointNumber);
                }
            } else if (boxToSearch.size() == 1) {
                if (node->isLeaf()) {
                    launchGetNearestPointInLeafNode(stream,
                                                    queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                                                    numDimensions,
                                                    node->numSubtreePoints,
                                                    globalGpuContext->leafPointsL1_d[boxToSearch.back()],
                                                    queryHandle->getIndexOfNearestPoint_d(),
                                                    queryHandle->getSquaredDistanceToNearestPoint_d());
                    leafNodesSearchedPerPoint[queryPointNumber].insert(boxToSearch);
                    isLeaf.insert(queryPointNumber);
                } else {
                    launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(
                        queryHandle->getStream(),
                        queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                        numDimensions,
                        globalGpuContext->boxSpanL1_d[boxToSearch.back()],
                        node->children.size(),
                        queryHandle->getMinAndMaxDistanceToAllBoxesPerPoint_d() + queryPointNumber * 1024,
                        queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d() + queryPointNumber);
                    isNonLeaf.insert(queryPointNumber);
                }
            } else {
                assert(false);  // implement later once smaller trees are working
            }

            ++pointBoxesIter;
        }

        if (!isLeaf.empty()) {
            cudaStatus = cudaMemcpyAsync(queryHandle->getIndexOfNearestPoint(),
                                         queryHandle->getIndexOfNearestPoint_d(),
                                         numPoints * sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMemcpyAsync(queryHandle->getSquaredDistanceToNearestPoint(),
                                         queryHandle->getSquaredDistanceToNearestPoint_d(),
                                         numPoints * sizeof(float),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            assert(cudaStatus == cudaSuccess);
        }
        if (!isNonLeaf.empty()) {
            cudaStatus = cudaMemcpyAsync(queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint(),
                                         queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d(),
                                         numPoints * sizeof(int2),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            assert(cudaStatus == cudaSuccess);
        }
        if (!(isLeaf.empty() && isNonLeaf.empty())) {
            cudaStatus = cudaStreamSynchronize(stream);
            assert(cudaStatus == cudaSuccess);
        }

        for (auto leafIter = isLeaf.begin(); leafIter != isLeaf.end(); ++leafIter) {
            int queryPointNumber = *leafIter;
            GpuRtreeNode *node = getNode(initialPathPerPoint[queryPointNumber]);
            if (result.nearestNeighborIndexes[queryPointNumber] < 0.0f ||
                queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber] < result.nearestNeighborIndexes[queryPointNumber]) {
                result.nearestNeighborIndexes[queryPointNumber] =
                    node->leafPoints[queryHandle->getIndexOfNearestPoint()[queryPointNumber]].pointIndex;
                distanceToCurrentNearestNeigbor[queryPointNumber] = queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber];
            }
        }
        for (auto nonLeafIter = isNonLeaf.begin(); nonLeafIter != isNonLeaf.end(); ++nonLeafIter) {
            int queryPointNumber = *nonLeafIter;
            int2 pointExtremeBoxes = queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint()[queryPointNumber];
            std::vector<int> path = initialPathPerPoint[queryPointNumber];
            path.push_back(pointExtremeBoxes.x);
            boxesToSearchPerPoint[queryPointNumber].push_back(path);
            if (pointExtremeBoxes.y != pointExtremeBoxes.x) {
                path.pop_back();
                path.push_back(pointExtremeBoxes.y);
                boxesToSearchPerPoint[queryPointNumber].push_back(path);
            }
        }
    }
}

// Uses a heurisitic to get a point that is close to the query point quickly
// when the root is a leaf, getClosePoint will always yield the closest point
// void NearestNeighborExecutor::getClosePoint_old(const GpuPointData &gpuPointData, GpuPointData &result,
// std::vector<std::set<std::vector<int>>> &leafNodesSearchedPerPoint,
//        NearestNeighborGpuQueryHandle *queryHandle, std::vector<float> &distanceToCurrentNearestNeigbor) {
//    cudaError_t cudaStatus;
//    assert(gpuPointData.cpuPointData.numPoints <= maxNumQueryPointsPerBuffer);
//    cudaStatus = cudaSetDevice(gpuNum);
//    assert(cudaStatus == cudaSuccess);
//    cudaStream_t stream = queryHandle->getStream();
//    int numPoints = gpuPointData.cpuPointData.numPoints;
//    cudaStatus = cudaMemcpyAsync(queryHandle->getQueryPoint_d(), gpuPointData.cpuPointData.mem, numPoints *
//    numDimensions * sizeof(float), cudaMemcpyHostToDevice, stream); assert(cudaStatus == cudaSuccess);
//
//    // Drill through level 0 (pre-loaded on gpu)
//    if(root->isLeaf()) {
//        for(int queryPointNumber = 0; queryPointNumber < gpuPointData.cpuPointData.numPoints; ++queryPointNumber) {
//            launchGetNearestPointInLeafNode(stream, queryHandle->getQueryPoint_d() + (numDimensions*queryPointNumber),
//            numDimensions, root->numSubtreePoints,
//                    globalGpuContext->leafPointsL0_d,queryHandle->getIndexOfNearestPoint_d(),
//                    queryHandle->getSquaredDistanceToNearestPoint_d());
//        }
//    } else {
//        for(int queryPointNumber = 0; queryPointNumber < gpuPointData.cpuPointData.numPoints; ++queryPointNumber) {
//            launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(queryHandle->getStream(),
//            queryHandle->getQueryPoint_d() + (numDimensions*queryPointNumber), numDimensions,
//            globalGpuContext->boxSpanL0_d,
//                    root->children.size(), queryHandle->getMinAndMaxDistanceToAllBoxesPerPoint_d() +
//                    queryPointNumber*1024, queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d() + queryPointNumber);
//        }
//    }
//    if(root->isLeaf()) {
//        cudaStatus = cudaMemcpyAsync(queryHandle->getIndexOfNearestPoint(), queryHandle->getIndexOfNearestPoint_d(),
//        numPoints * sizeof(int), cudaMemcpyDeviceToHost, stream); assert(cudaStatus == cudaSuccess); cudaStatus =
//        cudaStreamSynchronize(stream); assert(cudaStatus == cudaSuccess); for(int queryPointNumber = 0;
//        queryPointNumber < gpuPointData.cpuPointData.numPoints; ++queryPointNumber) {
//            // The cuda kernel returns the index of the nearest point with respect the the owning box, i.e. the
//            indexes range from 0 to 1023
//            // I need to use this to get the point's index in the rtree
//            result.nearestNeighborIndexes[queryPointNumber] =
//            root->leafPoints[queryHandle->getIndexOfNearestPoint()[queryPointNumber]].pointIndex;
//        }
//        return;
//    } else {
//        cudaStatus = cudaMemcpyAsync(queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint(),
//        queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d(), numPoints * sizeof(int2), cudaMemcpyDeviceToHost,
//        stream); assert(cudaStatus == cudaSuccess);
//    }
//
//    // Past here the root is not a leaf
//    // Now initialize some data structures that I will need below
//    // note: leafNodesSearchedPerPoint[pointIndex].find(pathVector)
//    std::unordered_map<int, std::set<std::vector<int>>> boxesThatStillNeedToBeSearchedPerPoint;
//    std::set<int> pointsLeftToService;
//    std::set<int> pointsLeftToServiceInitial;
//    leafNodesSearchedPerPoint.clear();
//    leafNodesSearchedPerPoint.reserve(numPoints);
//    distanceToCurrentNearestNeigbor.clear();
//    distanceToCurrentNearestNeigbor.reserve(numPoints);
//    for(int queryPointNum = 0; queryPointNum < numPoints; ++queryPointNum) {
//        distanceToCurrentNearestNeigbor.push_back(-1.0f);
//        leafNodesSearchedPerPoint.emplace_back();
//    }
//
//    cudaStatus = cudaStreamSynchronize(stream);
//    assert(cudaStatus == cudaSuccess);
//    for(int queryPointNumber = 0; queryPointNumber < numPoints; ++queryPointNumber) {
//        int2 pointExtremeBoxes = queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint()[queryPointNumber];
//        std::vector<int> path;
//        path.push_back(pointExtremeBoxes.x);
//        boxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].insert(path);
//        if(pointExtremeBoxes.y != pointExtremeBoxes.x) {
//            path.clear();
//            path.push_back(pointExtremeBoxes.y);
//            boxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].insert(path);
//        }
//    }
//
//    // Drill through level 1 (pre-loaded on gpu)
//    // At this level, each point has exactly 1 or 2 boxes that need to be searched
//    std::unordered_map<int, std::set<std::vector<int>>> initialBoxesThatStillNeedToBeSearchedPerPoint =
//    std::move(boxesThatStillNeedToBeSearchedPerPoint); boxesThatStillNeedToBeSearchedPerPoint.clear();
//
//    std::unordered_map<int, std::vector<int>> initialPathPerPoint;
//
//    // Box 1
//    std::unordered_set<int> isLeaf;
//    std::unordered_set<int> isNonLeaf;
//    for(int queryPointNumber = 0; queryPointNumber < numPoints; ++queryPointNumber) {
//        std::vector<int> path = *(initialBoxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].begin());
//        initialPathPerPoint[queryPointNumber] = path;
//        GpuRtreeNode *node = getNode(path);
//        int boxIndex = path.back();
//        if(node->isLeaf()) {
//            launchGetNearestPointInLeafNode(stream, queryHandle->getQueryPoint_d() + (numDimensions*queryPointNumber),
//            numDimensions, node->numSubtreePoints, globalGpuContext->leafPointsL1_d[boxIndex],
//                    queryHandle->getIndexOfNearestPoint_d(), queryHandle->getSquaredDistanceToNearestPoint_d());
//            isLeaf.insert(queryPointNumber);
//            leafNodesSearchedPerPoint[queryPointNumber].insert(path);
//        } else {
//            launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(queryHandle->getStream(),
//            queryHandle->getQueryPoint_d() + (numDimensions*queryPointNumber), numDimensions,
//            globalGpuContext->boxSpanL1_d[boxIndex],
//                    node->children.size(), queryHandle->getMinAndMaxDistanceToAllBoxesPerPoint_d() +
//                    queryPointNumber*1024, queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d() + queryPointNumber);
//            isNonLeaf.insert(queryPointNumber);
//            pointsLeftToService.insert(queryPointNumber);
//        }
//    }
//    if(!isLeaf.empty()) {
//        cudaStatus = cudaMemcpyAsync(queryHandle->getIndexOfNearestPoint(), queryHandle->getIndexOfNearestPoint_d(),
//        numPoints * sizeof(int), cudaMemcpyDeviceToHost, stream); assert(cudaStatus == cudaSuccess); cudaStatus =
//        cudaMemcpyAsync(queryHandle->getSquaredDistanceToNearestPoint(),
//        queryHandle->getSquaredDistanceToNearestPoint_d(), numPoints * sizeof(float), cudaMemcpyDeviceToHost, stream);
//        assert(cudaStatus == cudaSuccess);
//    }
//    if(!isNonLeaf.empty()) {
//        cudaStatus = cudaMemcpyAsync(queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint(),
//        queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d(), numPoints * sizeof(int2), cudaMemcpyDeviceToHost,
//        stream); assert(cudaStatus == cudaSuccess);
//    }
//    cudaStatus = cudaStreamSynchronize(stream);
//    assert(cudaStatus == cudaSuccess);
//
//    for(auto leafIter = isLeaf.begin(); leafIter != isLeaf.end(); ++leafIter) {
//        int queryPointNumber = *leafIter;
//        std::vector<int> path = *(initialBoxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].begin());
//        GpuRtreeNode *node = getNode(path);
//        result.nearestNeighborIndexes[queryPointNumber] =
//        node->leafPoints[queryHandle->getIndexOfNearestPoint()[queryPointNumber]].pointIndex;
//        distanceToCurrentNearestNeigbor[queryPointNumber] =
//        queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber];
//    }
//    for(auto nonLeafIter = isNonLeaf.begin(); nonLeafIter != isNonLeaf.end(); ++nonLeafIter) {
//        int queryPointNumber = *nonLeafIter;
//        int2 pointExtremeBoxes = queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint()[queryPointNumber];
//        std::vector<int> path = initialPathPerPoint[queryPointNumber];
//        path.push_back(pointExtremeBoxes.x);
//        boxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].insert(path);
//        if(pointExtremeBoxes.y != pointExtremeBoxes.x) {
//            path.pop_back();
//            path.push_back(pointExtremeBoxes.y);
//            boxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].insert(path);
//        }
//    }
//
//    // Box 2
//    isLeaf.clear();
//    isNonLeaf.clear();
//    initialPathPerPoint.clear();
//    for(int queryPointNumber = 0; queryPointNumber < numPoints; ++queryPointNumber) {
//        if(initialBoxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].size() == 2) {
//            auto boxIter = initialBoxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].begin();
//            ++boxIter;
//            std::vector<int> path = *boxIter;
//            GpuRtreeNode *node = getNode(path);
//            int boxIndex = path.back();
//            if(node->isLeaf()) {
//                launchGetNearestPointInLeafNode(stream, queryHandle->getQueryPoint_d() +
//                (numDimensions*queryPointNumber), numDimensions, node->numSubtreePoints,
//                globalGpuContext->leafPointsL1_d[boxIndex],
//                        queryHandle->getIndexOfNearestPoint_d(), queryHandle->getSquaredDistanceToNearestPoint_d());
//                isLeaf.insert(queryPointNumber);
//                leafNodesSearchedPerPoint[queryPointNumber].insert(path);
//            } else {
//                launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(queryHandle->getStream(),
//                queryHandle->getQueryPoint_d() + (numDimensions*queryPointNumber), numDimensions,
//                globalGpuContext->boxSpanL1_d[boxIndex],
//                        node->children.size(), queryHandle->getMinAndMaxDistanceToAllBoxesPerPoint_d() +
//                        queryPointNumber*1024, queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d() +
//                        queryPointNumber);
//                isNonLeaf.insert(queryPointNumber);
//                pointsLeftToService.insert(queryPointNumber);
//            }
//        }
//    }
//    if(!isLeaf.empty()) {
//        cudaStatus = cudaMemcpyAsync(queryHandle->getIndexOfNearestPoint(), queryHandle->getIndexOfNearestPoint_d(),
//        numPoints * sizeof(int), cudaMemcpyDeviceToHost, stream); assert(cudaStatus == cudaSuccess); cudaStatus =
//        cudaMemcpyAsync(queryHandle->getSquaredDistanceToNearestPoint(),
//        queryHandle->getSquaredDistanceToNearestPoint_d(), numPoints * sizeof(float), cudaMemcpyDeviceToHost, stream);
//        assert(cudaStatus == cudaSuccess);
//    }
//    if(!isNonLeaf.empty()) {
//        cudaStatus = cudaMemcpyAsync(queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint(),
//        queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint_d(), numPoints * sizeof(int2), cudaMemcpyDeviceToHost,
//        stream); assert(cudaStatus == cudaSuccess);
//    }
//    cudaStatus = cudaStreamSynchronize(stream);
//    assert(cudaStatus == cudaSuccess);
//
//    for(auto leafIter = isLeaf.begin(); leafIter != isLeaf.end(); ++leafIter) {
//        int queryPointNumber = *leafIter;
//        auto boxIter = initialBoxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].begin();
//        ++boxIter;
//        std::vector<int> path = *boxIter;
//        GpuRtreeNode *node = getNode(path);
//        if(result.nearestNeighborIndexes[queryPointNumber] < 0.0f ||
//        queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber] <
//        result.nearestNeighborIndexes[queryPointNumber]) {
//            result.nearestNeighborIndexes[queryPointNumber] =
//            node->leafPoints[queryHandle->getIndexOfNearestPoint()[queryPointNumber]].pointIndex;
//            distanceToCurrentNearestNeigbor[queryPointNumber] =
//            queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber];
//        }
//    }
//    for(auto nonLeafIter = isNonLeaf.begin(); nonLeafIter != isNonLeaf.end(); ++nonLeafIter) {
//        int queryPointNumber = *nonLeafIter;
//        int2 pointExtremeBoxes = queryHandle->getMinMinAndMinMaxBoxIndexesPerPoint()[queryPointNumber];
//        std::vector<int> path = initialPathPerPoint[queryPointNumber];
//        path.push_back(pointExtremeBoxes.x);
//        boxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].insert(path);
//        if(pointExtremeBoxes.y != pointExtremeBoxes.x) {
//            path.pop_back();
//            path.push_back(pointExtremeBoxes.y);
//            boxesThatStillNeedToBeSearchedPerPoint[queryPointNumber].insert(path);
//        }
//    }
//
//    // FIXME: solve the case of 3+ levels later. It is only needed for trees of a million or more points
//    assert(pointsLeftToService.empty());    // This function will fail for now when the code mentioned above is
//    actually needed.
//}

// Searches exhaustively for the closest point to the query point given an initial close point.
// The closer the initial point is, the smaller and quicker the exhaustive search will be.
// The function assumes the proper gpu was activated by getClosePoints. getClosePoints needs to be called before
// getClosestPoint
void NearestNeighborExecutor::getClosestPoint(const GpuPointData &gpuPointData,
                                              GpuPointData &result,
                                              std::vector<std::set<std::vector<int>>> &leafNodesSearchedPerPoint,
                                              NearestNeighborGpuQueryHandle *queryHandle,
                                              std::vector<float> &distanceToCurrentNearestNeigbor) {
    cudaError_t cudaStatus;
    cudaStream_t stream = queryHandle->getStream();
    int numPoints = gpuPointData.cpuPointData.numPoints;

    std::unordered_map<int, std::deque<std::vector<int>>> boxesToSearchPerPoint;
    boxesToSearchPerPoint.reserve(numPoints);
    for (int queryPointNumber = 0; queryPointNumber < numPoints; ++queryPointNumber) {
        boxesToSearchPerPoint[queryPointNumber].emplace_back();
        boxesToSearchPerPoint[queryPointNumber].back().emplace_back();  // For each point, I need to search the root.
    }

    while (!boxesToSearchPerPoint.empty()) {
        // Launch one kernel per point and then synchronize until there are no more kernels to launch
        std::unordered_set<int> isLeaf;
        std::unordered_set<int> isNonLeaf;
        std::unordered_map<int, std::vector<int>> initialPathPerPoint;
        for (auto pointBoxesIter = boxesToSearchPerPoint.begin(); pointBoxesIter != boxesToSearchPerPoint.end();
             /*NOP*/) {
            int queryPointNumber = pointBoxesIter->first;
            std::deque<std::vector<int>> &boxesToSearchForThisPoint = pointBoxesIter->second;

            // Find a box to search
            // If there isn't one that has not already been searched then delete the entry for this point and continue
            std::vector<int> boxToSearch;
            if (boxesToSearchForThisPoint.empty()) {
                auto oldIter = pointBoxesIter;
                ++pointBoxesIter;
                boxesToSearchPerPoint.erase(oldIter);
                continue;
            }
            do {
                boxToSearch = boxesToSearchForThisPoint.front();
                boxesToSearchForThisPoint.pop_front();
            } while (leafNodesSearchedPerPoint[queryPointNumber].find(boxToSearch) != leafNodesSearchedPerPoint[queryPointNumber].end() &&
                     !boxesToSearchForThisPoint.empty());
            if (leafNodesSearchedPerPoint[queryPointNumber].find(boxToSearch) != leafNodesSearchedPerPoint[queryPointNumber].end()) {
                auto oldIter = pointBoxesIter;
                ++pointBoxesIter;
                boxesToSearchPerPoint.erase(oldIter);
                continue;
            }
            initialPathPerPoint[queryPointNumber] = boxToSearch;

            GpuRtreeNode *node = getNode(boxToSearch);
            if (boxToSearch.size() == 0) {
                if (node->isLeaf()) {
                    launchGetNearestPointInLeafNode(stream,
                                                    queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                                                    numDimensions,
                                                    node->numSubtreePoints,
                                                    globalGpuContext->leafPointsL0_d,
                                                    queryHandle->getIndexOfNearestPoint_d(),
                                                    queryHandle->getSquaredDistanceToNearestPoint_d());
                    leafNodesSearchedPerPoint[queryPointNumber].insert(boxToSearch);
                    isLeaf.insert(queryPointNumber);
                } else {
                    launchFindBoxesWithinRadius(stream,
                                                queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                                                numDimensions,
                                                distanceToCurrentNearestNeigbor[queryPointNumber],
                                                globalGpuContext->boxSpanL0_d,
                                                node->children.size(),
                                                queryHandle->getFlaggedBoxes_d() + 32 * queryPointNumber);
                    isNonLeaf.insert(queryPointNumber);
                }
            } else if (boxToSearch.size() == 1) {
                if (node->isLeaf()) {
                    launchGetNearestPointInLeafNode(stream,
                                                    queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                                                    numDimensions,
                                                    node->numSubtreePoints,
                                                    globalGpuContext->leafPointsL1_d[boxToSearch.back()],
                                                    queryHandle->getIndexOfNearestPoint_d(),
                                                    queryHandle->getSquaredDistanceToNearestPoint_d());
                    leafNodesSearchedPerPoint[queryPointNumber].insert(boxToSearch);
                    isLeaf.insert(queryPointNumber);
                } else {
                    launchFindBoxesWithinRadius(stream,
                                                queryHandle->getQueryPoint_d() + (numDimensions * queryPointNumber),
                                                numDimensions,
                                                distanceToCurrentNearestNeigbor[queryPointNumber],
                                                globalGpuContext->boxSpanL1_d[boxToSearch.back()],
                                                node->children.size(),
                                                queryHandle->getFlaggedBoxes_d() + 32 * queryPointNumber);
                    isNonLeaf.insert(queryPointNumber);
                }
            } else {
                assert(false);  // implement later once smaller trees are working
            }

            ++pointBoxesIter;
        }

        if (!isLeaf.empty()) {
            cudaStatus = cudaMemcpyAsync(queryHandle->getIndexOfNearestPoint(),
                                         queryHandle->getIndexOfNearestPoint_d(),
                                         numPoints * sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMemcpyAsync(queryHandle->getSquaredDistanceToNearestPoint(),
                                         queryHandle->getSquaredDistanceToNearestPoint_d(),
                                         numPoints * sizeof(float),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            assert(cudaStatus == cudaSuccess);
        }
        if (!isNonLeaf.empty()) {
            cudaStatus = cudaMemcpyAsync(queryHandle->getFlaggedBoxes(),
                                         queryHandle->getFlaggedBoxes_d(),
                                         numPoints * 32 * sizeof(uint),
                                         cudaMemcpyDeviceToHost,
                                         stream);
            assert(cudaStatus == cudaSuccess);
        }
        if (!(isLeaf.empty() && isNonLeaf.empty())) {
            cudaStatus = cudaStreamSynchronize(stream);
            assert(cudaStatus == cudaSuccess);
        }

        for (auto leafIter = isLeaf.begin(); leafIter != isLeaf.end(); ++leafIter) {
            int queryPointNumber = *leafIter;
            GpuRtreeNode *node = getNode(initialPathPerPoint[queryPointNumber]);
            if (result.nearestNeighborIndexes[queryPointNumber] < 0.0f ||
                queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber] < result.nearestNeighborIndexes[queryPointNumber]) {
                result.nearestNeighborIndexes[queryPointNumber] =
                    node->leafPoints[queryHandle->getIndexOfNearestPoint()[queryPointNumber]].pointIndex;
                distanceToCurrentNearestNeigbor[queryPointNumber] = queryHandle->getSquaredDistanceToNearestPoint()[queryPointNumber];
            }
        }
        for (auto nonLeafIter = isNonLeaf.begin(); nonLeafIter != isNonLeaf.end(); ++nonLeafIter) {
            int queryPointNumber = *nonLeafIter;
            getBoxNumbersFromFlagsAndAddToPath(queryHandle->getFlaggedBoxes() + 32 * queryPointNumber,
                                               initialPathPerPoint[queryPointNumber],
                                               boxesToSearchPerPoint[queryPointNumber]);
        }
    }
}

inline GpuRtreeNode *NearestNeighborExecutor::getNode(const std::vector<int> &nodePath) {
    GpuRtreeNode *node = root;
    for (auto it = nodePath.begin(); it != nodePath.end(); ++it)
        node = node->children[*it];
    return node;
}

inline void NearestNeighborExecutor::getBoxNumbersFromFlagsAndAddToPath(unsigned int flags[32],
                                                                        std::vector<int> pathToHere,
                                                                        std::deque<std::vector<int>> &boxNumbers) {
    boxNumbers.clear();

    unsigned int indicator = 0x01;

    for (int i = 0; i < 32; ++i) {
        if (flags[i] != 0) {
            int start = 32 * i;
            if (flags[i] & 0x0000FFFF) {
                if (flags[i] & 0x000000FF) {
                    if (flags[i] & 0x0000000F) {
                        for (int b = 0; b < 4; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                    if (flags[i] & 0x000000F0) {
                        for (int b = 4; b < 8; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                }
                if (flags[i] & 0x0000FF00) {
                    if (flags[i] & 0x00000F00) {
                        for (int b = 8; b < 12; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                    if (flags[i] & 0x00000F000) {
                        for (int b = 12; b < 16; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                }
            }
            if (flags[i] & 0xFFFF0000) {
                if (flags[i] & 0x00FF0000) {
                    if (flags[i] & 0x000F0000) {
                        for (int b = 16; b < 20; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                    if (flags[i] & 0x00F00000) {
                        for (int b = 20; b < 24; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                }
                if (flags[i] & 0xFF000000) {
                    if (flags[i] & 0x0F000000) {
                        for (int b = 24; b < 28; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                    if (flags[i] & 0xF00000000) {
                        for (int b = 28; b < 32; ++b) {
                            if (flags[i] & (indicator << b)) {
                                boxNumbers.push_back(pathToHere);
                                boxNumbers.back().push_back(start + b);
                            }
                        }
                    }
                }
            }
        }
    }
}

NearestNeighborGpuQueryHandle::NearestNeighborGpuQueryHandle(int gpuNum, int numDimensions, int maxNumPointsPerQuery) {
    int numDevices;
    cudaError_t cudaStatus;
    cudaStatus = cudaGetDeviceCount(&numDevices);
    assert(cudaStatus == cudaSuccess);
    assert(gpuNum >= 0);
    assert(gpuNum < numDevices);
    int previousGpuNum;
    cudaStatus = cudaGetDevice(&previousGpuNum);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaSetDevice(gpuNum);
    assert(cudaStatus == cudaSuccess);

    assert(maxNumPointsPerQuery > 0);
    bufferWidth = maxNumPointsPerQuery < 6 ? maxNumPointsPerQuery : 6;

    this->numDimensions = numDimensions;

    this->gpuNum = gpuNum;
    this->maxNumPointsPerQuery = maxNumPointsPerQuery;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaHostAlloc(&minMinAndMinMaxBoxIndexesPerPoint, maxNumPointsPerQuery * sizeof(float2), cudaHostAllocDefault);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&boxSpanOrLeafPoints, bufferWidth * 1024 * numDimensions * sizeof(float2), cudaHostAllocWriteCombined);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&flaggedBoxes, bufferWidth * 32 * sizeof(uint), cudaHostAllocDefault);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&indexOfNearestPoint, maxNumPointsPerQuery * sizeof(int), cudaHostAllocDefault);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&squaredDistanceToNearestPoint, maxNumPointsPerQuery * sizeof(float), cudaHostAllocDefault);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMalloc(&queryPoint_d, maxNumPointsPerQuery * numDimensions * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&minAndMaxDistanceToAllBoxesPerPoint_d, maxNumPointsPerQuery * 1024 * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&minMinAndMinMaxBoxIndexesPerPoint_d, maxNumPointsPerQuery * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&boxSpanOrLeafPoints_d, bufferWidth * 1024 * numDimensions * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&flaggedBoxes_d, bufferWidth * 32 * sizeof(uint));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&indexOfNearestPoint_d, maxNumPointsPerQuery * sizeof(int));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&squaredDistanceToNearestPoint_d, maxNumPointsPerQuery * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaSetDevice(previousGpuNum);
    assert(cudaStatus == cudaSuccess);
}

NearestNeighborGpuQueryHandle::~NearestNeighborGpuQueryHandle() {
    cudaError_t cudaStatus;

    assert(boxSpanOrLeafPoints != NULL);

    int previousGpuNum;
    cudaStatus = cudaGetDevice(&previousGpuNum);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaSetDevice(gpuNum);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaFreeHost(minMinAndMinMaxBoxIndexesPerPoint);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFreeHost(boxSpanOrLeafPoints);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFreeHost(flaggedBoxes);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaFree(queryPoint_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(minMinAndMinMaxBoxIndexesPerPoint_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(boxSpanOrLeafPoints_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(flaggedBoxes_d);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaStreamDestroy(stream);
    assert(cudaStatus == cudaSuccess);

    gpuNum = -1;
    maxNumPointsPerQuery = -1;

    boxSpanOrLeafPoints = NULL;
    flaggedBoxes = NULL;
    queryPoint_d = NULL;
    boxSpanOrLeafPoints_d = NULL;
    flaggedBoxes_d = NULL;

    cudaStatus = cudaSetDevice(previousGpuNum);
    assert(cudaStatus == cudaSuccess);
}
