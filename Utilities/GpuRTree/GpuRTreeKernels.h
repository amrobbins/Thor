/*
 * GpuRTreeKernels.h
 *
 *  Created on: Jun 1, 2019
 *      Author: andrew
 */

#ifndef GPURTREEKERNELS_H_
#define GPURTREEKERNELS_H_

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <set>

void launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(cudaStream_t stream,
                                                        float *queryPoint_d,
                                                        int numDimensions,
                                                        float2 *boxSpan_d,
                                                        int numBoxes,
                                                        float2 *minMaxBoxDistance_d,
                                                        int2 *minMinAndMinMaxBoxIndexesPerPoint_d);
void launchGetNearestPointInLeafNode(cudaStream_t stream,
                                     float *queryPoint_d,
                                     int numDimensions,
                                     int numPointsInBox,
                                     float *boxesPoints_d,
                                     int *indexOfNearestPoint_d,
                                     float *squaredDistanceToNearestPoint_d);
void launchFindBoxesWithinRadius(cudaStream_t stream,
                                 float *queryPoint_d,
                                 int numDimensions,
                                 float squaredRadius,
                                 float2 *boxSpan_d,
                                 int numBoxes,
                                 uint *radiusIntersectsBoxFlags_d);

void testFindBoxThatContainsPoint(int numDimensions);
void testFindBoxesWithinRadius(int numDimensions);
void testFindBoxesThatIntersectRange(int numDimensions);
void testGetNearestPointInLeafNode(int numDimensions);
void testMinAndMaxDistanceToAllBoxes(int numDimensions);

inline float kernelsRandRange(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

#endif /* GPURTREEKERNELS_H_ */
