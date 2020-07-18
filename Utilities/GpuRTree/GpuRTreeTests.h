#ifndef GPURTREETESTS_H_
#define GPURTREETESTS_H_

#include "GpuRTree.h"

#include <unistd.h>

void testRTreeConstruction(long numPoints, long numBuilds, bool pack) {
    constexpr long numDimensions = 512;

    float *pointsMem;
    pointsMem = new float[numPoints * numDimensions];

    printf("Randomizing test points\n");

    srand(time(NULL));
    for (long i = 0; i < numPoints * numDimensions; ++i) {
        pointsMem[i] = randRange(-10.0f, 10.0f);
    }
    Point *points = new Point[numPoints];
    for (long p = 0; p < numPoints; ++p) {
        points[p].mem = pointsMem + p * numDimensions;
        points[p].pointIndex = p;
    }

    for (long i = 0; i < numBuilds; i++) {
        printf("starting build points: %ld\n", numPoints);
        fflush(stdout);
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        GpuRtree rtree(points, numPoints, true, numDimensions);

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);

        printf("RTree construction time %lf seconds\n", elapsed.count());

        rtree.verifyRtree(points, numPoints);

        if (pack) {
            long numFloats = rtree.floatsNeededToPackRtree();
            float *mem = new float[numFloats];

            printf("starting packing using %ld floats\n", numFloats);

            std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

            rtree.pack(mem, numFloats);

            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
            printf("RTree packing time %lf seconds\n", elapsed.count());

            rtree.verifyRtreePacking();

            rtree.setUpNearestNeighborPipeline(points[0].mem);
            rtree.tearDownNearestNeighborPipeline();

            delete[] mem;
        }
    }

    printf("\n");

    delete[] pointsMem;
}

void GpuRTreeConstructionTest() {
    srand(time(NULL));

    for (long i = 0; i < 4; ++i)
        testRTreeConstruction(16 + (rand() % 30), 1, true);
    for (long i = 0; i < 4; ++i)
        testRTreeConstruction(250000 + (rand() % 30), 1, true);
    for (long i = 0; i < 1; ++i)
        testRTreeConstruction(3000001, 1, true);
    for (long i = 0; i < 1; ++i)
        testRTreeConstruction(10000000 + (rand() % 30), 1, true);
}

void KernelTest() {
    srand(time(NULL));

    int testDimensions;

    testDimensions = 3;
    for (int i = 0; i < 10; ++i)
        testFindBoxThatContainsPoint(testDimensions);
    for (int i = 0; i < 10; ++i)
        testFindBoxesWithinRadius(testDimensions);
    for (int i = 0; i < 10; ++i)
        testGetNearestPointInLeafNode(testDimensions);
    for (int i = 0; i < 10; ++i)
        testMinAndMaxDistanceToAllBoxes(testDimensions);

    testDimensions = 33;
    for (int i = 0; i < 10; ++i)
        testFindBoxThatContainsPoint(testDimensions);
    for (int i = 0; i < 10; ++i)
        testFindBoxesWithinRadius(testDimensions);
    for (int i = 0; i < 10; ++i)
        testGetNearestPointInLeafNode(testDimensions);
    for (int i = 0; i < 10; ++i)
        testMinAndMaxDistanceToAllBoxes(testDimensions);

    testDimensions = 65;
    for (int i = 0; i < 10; ++i)
        testFindBoxThatContainsPoint(testDimensions);
    for (int i = 0; i < 10; ++i)
        testFindBoxesWithinRadius(testDimensions);
    for (int i = 0; i < 10; ++i)
        testGetNearestPointInLeafNode(testDimensions);
    for (int i = 0; i < 10; ++i)
        testMinAndMaxDistanceToAllBoxes(testDimensions);

    testDimensions = 512;
    for (int i = 0; i < 10; ++i)
        testFindBoxThatContainsPoint(testDimensions);
    for (int i = 0; i < 10; ++i)
        testFindBoxesWithinRadius(testDimensions);
    for (int i = 0; i < 10; ++i)
        testGetNearestPointInLeafNode(testDimensions);
    for (int i = 0; i < 10; ++i)
        testMinAndMaxDistanceToAllBoxes(testDimensions);
}

#endif
