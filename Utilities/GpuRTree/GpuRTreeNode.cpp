#include "GpuRTreeNode.h"

void GpuRtreeNode::setMinAndMaxPerDimension() {
    dimensionMins = new float[numDimensions];
    dimensionMaxes = new float[numDimensions];
    for (long d = 0; d < numDimensions; ++d) {
        dimensionMins[d] = leafPoints[0].mem[d];
        dimensionMaxes[d] = dimensionMins[d];
    }

    for (long p = 0; p < numSubtreePoints; ++p) {
        for (long d = 0; d < numDimensions; ++d) {
            float dimVal = leafPoints[p].mem[d];
            if (dimVal < dimensionMins[d])
                dimensionMins[d] = dimVal;
            if (dimVal > dimensionMaxes[d])
                dimensionMaxes[d] = dimVal;
        }
    }
}

float GpuRtreeNode::getDimensionMin(int dimension) {
    assert(dimension <= numDimensions);
    assert(numSubtreePoints > 0);

    float dimensionMin = leafPoints[0].mem[dimension];
    for (long p = 1; p < numSubtreePoints; ++p) {
        if (leafPoints[p].mem[dimension] < dimensionMin) {
            dimensionMin = leafPoints[p].mem[dimension];
        }
    }
    return dimensionMin;
}

GpuRtreeNode::GpuRtreeNode(long numPoints, Point *points, int numDimensions) {
    this->numSubtreePoints = numPoints;
    leafPoints = points;
    this->numDimensions = numDimensions;
}

void GpuRtreeNode::treeify(long level, bool multithread) {
    // First 2 levels are loaded into GPU memory, so prefer large leaves in first 2 levels.
    // After level 2, nodes must be copied to the gpu at search time, so having 32 leaf nodes with 32 points is better
    // for tx bandwidth than one leaf of 1024 points, because need to tx 32 (nodes) + 32 (points i.e 1 leaf) so about 64
    // vs having to transfer 1024
    if (numSubtreePoints <= nodeSize && level <= 2)
        return;
    else if (numSubtreePoints < MIN_POINTS_TO_SPLIT_NODES)
        return;
    assert(children.empty());

    children.push_back(new GpuRtreeNode(numSubtreePoints, leafPoints, numDimensions));
    children.back()->setMinAndMaxPerDimension();
    std::priority_queue<PointCount, std::vector<PointCount>, LessThanByCount> boxPointCounts;
    boxPointCounts.emplace(0, numSubtreePoints);

    std::vector<std::priority_queue<DimensionSpan, std::vector<DimensionSpan>, LessThanBySpan>> childDimensionSpans;
    childDimensionSpans.emplace_back();
    for (long d = 0; d < numDimensions; d++) {
        childDimensionSpans.back().emplace(dimensionMaxes[d] - dimensionMins[d], d);
    }

    std::mutex mtx;
    std::vector<std::thread *> workers;
    workers.reserve(1124);

    bool continueLoop = true;
    while (continueLoop) {
        if (level == 1 && multithread)
            mtx.lock();

        if (!boxPointCounts.empty() && boxPointCounts.top().pointCount >= MIN_POINTS_TO_SPLIT_NODES) {
            long chosenBox = boxPointCounts.top().boxIndex;
            boxPointCounts.pop();

            if (level == 1 && multithread)
                workers.push_back(
                    new std::thread(&GpuRtreeNode::splitNode, this, true, &mtx, chosenBox, &boxPointCounts, &childDimensionSpans));
            else
                splitNode(false, &mtx, chosenBox, &boxPointCounts, &childDimensionSpans);
        }

        // each worker creates 1 child, so to know how many children are currently being built, check num workers
        if (workers.size() == nodeSize - 1 || children.size() == nodeSize)
            continueLoop = false;
        else if ((boxPointCounts.size() == workers.size() || boxPointCounts.size() == children.size()) &&
                 boxPointCounts.top().pointCount < MIN_POINTS_TO_SPLIT_NODES)
            continueLoop = false;

        if (level == 1 && multithread)
            mtx.unlock();
    }
    while (!workers.empty()) {
        workers.back()->join();
        delete workers.back();
        workers.pop_back();
    }

    while (!boxPointCounts.empty() && boxPointCounts.top().pointCount > nodeSize) {
        long childIndex = boxPointCounts.top().boxIndex;
        boxPointCounts.pop();
        if (level == 1 && multithread)
            workers.push_back(new std::thread(&GpuRtreeNode::treeify, children[childIndex], level + 1, multithread));
        else
            children[childIndex]->treeify(level + 1, multithread);
    }
    while (!workers.empty()) {
        workers.back()->join();
        delete workers.back();
        workers.pop_back();
    }
}

void GpuRtreeNode::splitNode(
    bool threadSafe,
    std::mutex *mtx,
    long chosenBox,
    std::priority_queue<PointCount, std::vector<PointCount>, LessThanByCount> *boxPointCounts,
    std::vector<std::priority_queue<DimensionSpan, std::vector<DimensionSpan>, LessThanBySpan>> *childDimensionSpans) {
    if (threadSafe)
        mtx->lock();
    DimensionSpan dimensionOfMaxSpan = (*childDimensionSpans)[chosenBox].top();
    (*childDimensionSpans)[chosenBox].pop();
    childDimensionSpans->emplace_back();

    //    printf("Splitting child %ld (of %ld), dimension %ld, numPoints %ld, maxSpan %f\n", chosenBox, children.size(),
    //    dimensionOfMaxSpan.dimensionNumber, children[chosenBox]->numSubtreePoints, dimensionOfMaxSpan.span);

    long numBoxPoints = children[chosenBox]->numSubtreePoints;
    Point *boxPoints = children[chosenBox]->leafPoints;
    long splitPoint = ((numBoxPoints + 1) / 2) - 1;
    long numOriginalBoxPoints = splitPoint + 1;
    long numChildPoints = numBoxPoints - numOriginalBoxPoints;
    GpuRtreeNode *newChild = new GpuRtreeNode(numChildPoints, boxPoints + numOriginalBoxPoints, numDimensions);
    long newChildIndex = children.size();
    children.push_back(newChild);
    float newChildDimensionMin;
    if (threadSafe)
        mtx->unlock();

    // Next line does all of the work of setting up the rtree:
    partitionPoints(boxPoints, dimensionOfMaxSpan.dimensionNumber, numBoxPoints, splitPoint, newChild, newChildDimensionMin);

    if (threadSafe)
        mtx->lock();
    children[newChildIndex]->dimensionMins = new float[numDimensions];
    children[newChildIndex]->dimensionMaxes = new float[numDimensions];
    for (int d = 0; d < numDimensions; ++d) {
        children[newChildIndex]->dimensionMins[d] = children[chosenBox]->dimensionMins[d];
        children[newChildIndex]->dimensionMaxes[d] = children[chosenBox]->dimensionMaxes[d];
    }
    // only children[chosenBox]->leafPoints[children[chosenBox]->numSubtreePoints-1] is in its correct sorted order, and
    // is the max for that point, in that dimension. so I need to find the min for the new box.
    children[newChildIndex]->dimensionMins[dimensionOfMaxSpan.dimensionNumber] = newChildDimensionMin;
    children[chosenBox]->numSubtreePoints = numOriginalBoxPoints;
    children[chosenBox]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber] =
        children[chosenBox]->leafPoints[children[chosenBox]->numSubtreePoints - 1].mem[dimensionOfMaxSpan.dimensionNumber];
    boxPointCounts->emplace(chosenBox, splitPoint + 1);
    boxPointCounts->emplace(newChildIndex, numBoxPoints - (splitPoint + 1));
    (*childDimensionSpans)[newChildIndex] = (*childDimensionSpans)[chosenBox];
    (*childDimensionSpans)[chosenBox].emplace(children[chosenBox]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber] -
                                                  children[chosenBox]->dimensionMins[dimensionOfMaxSpan.dimensionNumber],
                                              dimensionOfMaxSpan.dimensionNumber);
    (*childDimensionSpans)[newChildIndex].emplace(children[newChildIndex]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber] -
                                                      children[newChildIndex]->dimensionMins[dimensionOfMaxSpan.dimensionNumber],
                                                  dimensionOfMaxSpan.dimensionNumber);
    //    printf("original box dimension %ld span updated from  %f:%f  to  %f:%f\n", dimensionOfMaxSpan.dimensionNumber,
    //            children[chosenBox]->dimensionMins[dimensionOfMaxSpan.dimensionNumber],
    //            children[newChildIndex]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber],
    //            children[chosenBox]->dimensionMins[dimensionOfMaxSpan.dimensionNumber],
    //            children[chosenBox]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber]);
    //    printf("new box dimension %ld span updated from  %f:%f  to  %f:%f\n", dimensionOfMaxSpan.dimensionNumber,
    //            children[chosenBox]->dimensionMins[dimensionOfMaxSpan.dimensionNumber],
    //            children[newChildIndex]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber],
    //            children[newChildIndex]->dimensionMins[dimensionOfMaxSpan.dimensionNumber],
    //            children[newChildIndex]->dimensionMaxes[dimensionOfMaxSpan.dimensionNumber]);
    //
    //    printf("children.size() %ld, largestPointCount %ld is of child %ld splitPoint %ld\n", children.size(),
    //    boxPointCounts->top().pointCount, boxPointCounts->top().boxIndex, splitPoint);

    if (threadSafe)
        mtx->unlock();
}
