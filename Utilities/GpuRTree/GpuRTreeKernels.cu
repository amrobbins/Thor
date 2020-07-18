#include "GpuRTreeKernels.h"

#define FULL_MASK ((unsigned int)0xffffffff)

__global__ void findBoxesThatContainPoint(float *point, int numDimensions, float2 *boxSpan, int numBoxes, uint *boxContainsPointFlags);
__global__ void findBoxesWithinRadius(
    float *point, float squaredRadius, int numDimensions, float2 *boxSpan, int numBoxes, uint *radiusIntersectsBoxFlags);
__global__ void getMinAndMaxDistanceToAllBoxes(float *point, int numDimensions, float2 *boxSpan, int numBoxes, float2 *minMaxBoxDistance);
__global__ void getMinMinAndMinMax(float2 *minMaxBoxDistance, int numBoxes, int2 *extremeBoxIndexes);
__global__ void getNearestPointInLeafNode(
    float2 *point, int numDimensions, float *boxesPoints, int numPoints, int *indexOfNearestPoint, float *distanceToNearestPoint);

//__global__ void findBoxesThatContainPointBatched(float2 *points, int numPoints, int numDimensions, float2* boxSpan,
// int numBoxes, unsigned int *boxContainsPoint);
//__global__ void findBoxesWithinRadiusBatched(float *points, int numPoints, float squaredRadius, int numDimensions,
// float2* boxSpan, int numBoxes, bool *radiusIntersectsBox);
//__global__ void getMinAndMaxDistanceToAllBoxesBatched(float *points, int numDimensions, float2* boxSpan, int numBoxes,
// float2 *minMaxBoxDistance);
//__global__ void getMinMinAndMinMaxBatched(float2 *minMaxBoxDistance, int numBoxes, int2 *extremeBoxIndexes);
//__global__ void getNearestPointInLeafNodeBatched(float2 *points, int numDimensions, float* boxesPoints, int numPoints,
// int *indexOfNearestPoint, float *distanceToNearestPoint);

__device__ __forceinline__ void warpShuffleToFindMin(float &value, int &index, bool &valid);
__device__ __forceinline__ void shuffleCombineFlags(uint &flags);
__device__ __forceinline__ void swap(float &a, float &b);
__device__ __forceinline__ void warpTransposeUint(uint2 *row);

// 256 threads per block * 4 blocks = 1024 threads
__global__ void findBoxesThatContainPoint(float *point, int numDimensions, float2 *boxSpan, int numBoxes, uint *boxContainsPointFlags) {
    __shared__ float sharedPoint[64];
    __shared__ uint sharedFlags[8];

    bool dead = false;

    // boxContainsPoint[256*blockIdx.x + threadIdx.x] = false;
    bool myBoxValid = 256 * blockIdx.x + threadIdx.x < numBoxes;
    if (myBoxValid == false) {
        dead = true;
    }

    for (int dimension = 0; dimension < numDimensions; dimension++) {
        // When the point is within some box, then I need to read the whole point, doesn't matter which threads die.
        // If the point is not in any box, then it would be ok to not read the rest of the point, but it would be
        // difficult to know this without slowing down the operation and reading just 1 512 dimensional point is a small
        // part of the whole operation of reading 256 spans of 512 dimensions, i.e. it is just 1/512 of the memory
        // reads, so not worth trying to avoid finishing reading the point when I know early that it isn't contained in
        // any box
        if (dimension % 64 == 0) {
            __syncthreads();
            if (threadIdx.x < 32) {
                if (dimension + threadIdx.x < numDimensions)
                    ((float2 *)sharedPoint)[threadIdx.x] = ((float2 *)point)[dimension / 2 + threadIdx.x];
            }
            __syncthreads();
        }
        if (!dead) {
            float2 span = boxSpan[1024 * dimension + 256 * blockIdx.x + threadIdx.x];
            float pointDimension = sharedPoint[dimension % 64];
            if (span.x > span.y)
                swap(span.x, span.y);
            if (pointDimension < span.x || pointDimension > span.y) {
                dead = true;
            }
        }
    }

    // The box of every still-alive thread contains the point
    bool myBoxContainsPoint = !dead;
    int warpThread = threadIdx.x % 32;
    uint flags;
    if (myBoxContainsPoint)
        flags = 1 << warpThread;
    else
        flags = 0;

    shuffleCombineFlags(flags);
    if (warpThread == 0)
        sharedFlags[threadIdx.x / 32] = flags;
    __syncthreads();

    if (threadIdx.x < 8) {
        flags = sharedFlags[threadIdx.x];
        boxContainsPointFlags[8 * blockIdx.x + threadIdx.x] = flags;
    }
}

//__global__ void convertBoolsToFlags(int numBools, bool *bools, uint *flags) {
//    uint myFlags = 0;
//    bool myBools[32];
//    ((double4*)myBools)[0] = ((double4*)bools)[threadIdx.x];   // read 32 bools at once
//    for(int i = 0; i < 32; ++i) {
//        myFlags |= myBools[i] << i;
//    }
//    flags[threadIdx.x] = myFlags;
//}

/*
// 256 threads per block * 4 blocks = 1024 threads
// 1 point per blockIdx.y
// i.e.
// blockDim(256)
// gridDim(4, (numPoints+31)/32)
// point is in float2, so if odd number of dimensions there is one float of padding after each point.
__global__ void findBoxesThatContainPointBatched(float2 *points, int numPoints, int numDimensions, float2* boxSpan, int
numBoxes, unsigned int *boxContainsPointFlags) {
    __shared__ float sharedPoint[16*64];

    int myFirstPoint = blockIdx.y * 32;
    int myLastPoint = myFirstPoint + 31;
    if(myLastPoint >= numPoints)
        myLastPoint = numPoints-1;
    int myNumPoints = (myLastPoint - myFirstPoint) + 1;
    unsigned int pointInBoxFlags = 0xFFFFFFFF;
    pointInBoxFlags >>= 32 - myNumPoints;

    boxContainsPoint[256*blockIdx.x + threadIdx.x] = false;
    bool myBoxValid = 256*blockIdx.x + threadIdx.x < numBoxes;
    if(myBoxValid == false) {
        pointInBoxFlags = 0x0;
    }

    for(int dimension = 0; dimension < numDimensions; dimension++) {

        if(dimension % 64 == 0) {
            __syncthreads();
            for(int pointToLoad = 0; pointToLoad < myNumPoints; pointToLoad += 8) {
                int dimensionOffset = dimension + (threadIdx.x%32);
                if(dimensionOffset < numDimensions) {
                    int sharedPointOffset = (pointToLoad + (threadIdx.x/32)*32) * ((numDimensions+1)/2);
                    int globalPointOffset = sharedPointOffset + myFirstPoint * ((numDimensions+1)/2);
                    ((float2*)sharedPoint)[sharedPointOffset] = points[globalPointOffset + dimensionOffset];
                }
            }
            __syncthreads();
        }

        if(pointInBoxFlags != 0x0) {
            float2 span = boxSpan[1024*dimension + 256*blockIdx.x + threadIdx.x];
            if(span.x > span.y)
                swap(span.x, span.y);
            for(int p = 0; p < myNumPoints; ++p) {
                if(!(pointInBoxFlags & (1 << p)))
                    continue;
                float pointDimension = sharedPoint[p*64 + (dimension % 64)];
                if(pointDimension < span.x || pointDimension > span.y) {
                    pointInBoxFlags ^= (1 << p);
                }
            }
        }

    }

    boxContainsPointFlags[blockIdx.y*1024 + (blockIdx.x*256 + threadIdx.x)] = pointInBoxFlags;
}


// Convert flags from form: box0 -> 32 points in box0 flags, box1 -> 32 points in box1 flags
// To form: point0 -> point0 in first 32 boxes flags, point0 -> point0 in next 32 boxes flags
//
// 1 warp of 32 threads
// t0 loads b0 of p0
// t1 loads b1 of p1
// shuffle (custom shuffle using mod so wraps)
// t0 loads b1 of p0
// t1 loads b2 of p2
// after full cycle shuffles, each thread has first set of flags (lowest numbered boxes) for 1 point
// do this 32 times so that each thread contains all boxes for 1 point
// warp transpose, then each thread writes out the points via unsigned int write, and there are 32 writes.
__global__ void transposeFlags(unsigned int *flagsIn, unsigned int *transposedFlagsOut) {
    unsigned int sourceFlags = flagsIn[threadIdx.x];
    unsigned int outputFlags[32];

    // FIXME: One kernel per 32 points.
    for(int i = 0; i < 32; ++i) {
        outputFlags[i] = 0x0;
        int flagSlot = threadIdx.x;
        for(int f = 0; f < 32; ++f) {
            outputFlags[i] |= (sourceFlags & (1<<flagSlot)) << threadIdx.x;
            if(f < 31) {
                flagSlot = (flagSlot + 1) % 32;
                sourceFlags = shuffleRotateLeft(sourceFlags);
            }
        }
    }
    warpTransposeUint(outputFlags);

    // FIXME: This is half efficiency and it could be made 100% with a proper shuffling function to form 1 column of
uint2's from 2 columns of uint's
    // but that is hard so maybe do it later. But, it might not be worth it. This is only
    for(int i = 0; i < 32; ++i)
        transposedFlagsOut[i*32 + threadIdx.x] = sourceFlags[i];
}


// FIXME: I have not tested this kernel:
__device__ __forceinline__ float createStackedDoubleColumns(unsigned int *row) {
    for(int j = 0; j < 16; j += 2) {
        for(int i = 0; i < 16; ++i) {
            unsigned int otherValue;
            otherValue = __shfl_up_sync(0xffffffff, row[j+1], 1);
            if(threadIdx.x > i && threadIdx.x < 32 - i) {
                row[j] = otherValue;
                if(threadIdx.x < 31 - i) {
                    otherValue = row[j];
                    row[j] = row[j+1];
                    row[j+1] = otherValue;
                }
            }
        }
    }
}
*/

__device__ __forceinline__ float getMinimumDimensionSquaredDistance(float pointDimension, float rangeEnd0, float rangeEnd1) {
    if (rangeEnd0 > rangeEnd1)
        swap(rangeEnd0, rangeEnd1);

    float diff;
    if (rangeEnd1 < pointDimension)  // top and bottom are below point
        diff = rangeEnd1;
    else if (rangeEnd0 > pointDimension)  // top and bottom are above point
        diff = rangeEnd0;
    else  // else top is above point bottom is below point, so distance is 0 in this dimension
        return 0.0f;

    diff = pointDimension - diff;
    return diff * diff;
}

__global__ void findBoxesWithinRadius(
    float *point, float squaredRadius, int numDimensions, float2 *boxSpan, int numBoxes, uint *radiusIntersectsBoxFlags) {
    __shared__ float sharedPoint[64];
    __shared__ uint sharedFlags[8];

    bool dead = false;

    bool myBoxValid = 256 * blockIdx.x + threadIdx.x < numBoxes;
    if (myBoxValid == false) {
        dead = true;
    }

    float squaredDist = 0.0f;

    for (int dimension = 0; dimension < numDimensions; dimension++) {
        // When the point is within some box, then I need to read the whole point, doesn't matter which threads die.
        // If the point is not in any box, then it would be ok to not read the rest of the point, but it would be
        // difficult to know this without slowing down the operation and reading just 1 512 dimensional point is a small
        // part of the whole operation of reading 256 spans of 512 dimensions, i.e. it is just 1/512 of the memory
        // reads, so not worth trying to avoid finishing reading the point when I know early that it isn't contained in
        // any box
        if (dimension % 64 == 0) {
            __syncthreads();
            if (threadIdx.x < 32) {
                if (dimension + threadIdx.x < numDimensions)
                    ((float2 *)sharedPoint)[threadIdx.x] = ((float2 *)point)[dimension / 2 + threadIdx.x];
            }
            __syncthreads();
        }
        if (!dead) {
            float2 span = boxSpan[1024 * dimension + 256 * blockIdx.x + threadIdx.x];
            float pointDimension = sharedPoint[dimension % 64];
            squaredDist += getMinimumDimensionSquaredDistance(pointDimension, span.x, span.y);
            if (squaredDist > squaredRadius) {
                dead = true;
            }
        }
    }

    // The box of every still-alive thread contains the point
    bool myBoxIntersectsRadius = !dead;
    int warpThread = threadIdx.x % 32;
    uint flags;
    if (myBoxIntersectsRadius)
        flags = 1 << warpThread;
    else
        flags = 0;

    shuffleCombineFlags(flags);
    if (warpThread == 0)
        sharedFlags[threadIdx.x / 32] = flags;
    __syncthreads();

    if (threadIdx.x < 8) {
        flags = sharedFlags[threadIdx.x];
        radiusIntersectsBoxFlags[8 * blockIdx.x + threadIdx.x] = flags;
    }
}

/*
// invert to shared box
// each block checks all 1024 boxes for 16 points
// load 256 dimensions of spans for a single box and store it in shared (2048 bytes)
__global__ void findBoxesWithinRadiusBatched(float *points, int numPoints, float squaredRadius, int numDimensions,
float2* boxSpan, int numBoxes, unsigned int *radiusIntersectsBoxFlags) {
    __shared__ float2 sharedBox[64];

    unsigned int myRadiusIntersectsBoxFlags = 0xFFFFFFFF;

    int myFirstBox = threadIdx.x * 32;
    int myLastBox = myFirstBox + 31;
    if(myLastBox >= numBoxes)
        myLastBox = numBoxes - 1;

    dont kill any threads
    bool dead = false;

    radiusIntersectsBox[256*blockIdx.x + threadIdx.x] = false;
    bool myBoxValid = 256*blockIdx.x + threadIdx.x < numBoxes;
    if(myBoxValid == false) {
        if(threadIdx.x >= 32)
            return;
        else
            dead = true;
    }

    float squaredDist = 0.0f;

    for(int dimension = 0; dimension < numDimensions; dimension++) {

        // When the point is within some box, then I need to read the whole point, doesn't matter which threads die.
        // If the point is not in any box, then it would be ok to not read the rest of the point, but it would be
difficult to know this without slowing down the operation
        // and reading just 1 512 dimensional point is a small part of the whole operation of reading 256 spans of 512
dimensions, i.e. it is just 1/512 of the memory reads,
        // so not worth trying to avoid finishing reading the point when I know early that it isn't contained in any box
        if(dimension % 64 == 0) {
            __syncthreads();
            if(threadIdx.x < 32) {
                if(dimension + threadIdx.x < numDimensions)
                    ((float2*)sharedPoint)[threadIdx.x] = ((float2*)point)[dimension/2 + threadIdx.x];
            }
            __syncthreads();
        }
        if(threadIdx.x >= 32 || !dead) {
            float2 span = boxSpan[1024*dimension + 256*blockIdx.x + threadIdx.x];
            float pointDimension = sharedPoint[dimension % 64];
            squaredDist += getMinimumDimensionSquaredDistance(pointDimension, span.x, span.y);
            if(squaredDist > squaredRadius) {
                // The first 32 threads don't actually die because they read the point dimensions from main memory
                if(threadIdx.x >= 32)
                    return;
                else
                    dead = true;
            }
        }
    }

    // The box of every still-alive thread contains the point
    if(!dead)
        radiusIntersectsBox[256*blockIdx.x + threadIdx.x] = true;
}
*/

__device__ __forceinline__ float getMaximumDimensionSquaredDistance(float pointDimension, float rangeEnd0, float rangeEnd1) {
    float dist1, dist2;

    dist1 = pointDimension - rangeEnd0;
    dist1 *= dist1;
    dist2 = pointDimension - rangeEnd1;
    dist2 *= dist2;
    return dist1 > dist2 ? dist1 : dist2;
}

// Then keep the box with the minimum max distance and all those with the minimum min distance
// 256 threads per block * 4 blocks = 1024 threads
__global__ void getMinAndMaxDistanceToAllBoxes(float *point, int numDimensions, float2 *boxSpan, int numBoxes, float2 *minMaxBoxDistance) {
    __shared__ float sharedPoint[64];

    bool myBoxValid = 256 * blockIdx.x + threadIdx.x < numBoxes;
    bool dead = false;
    float squaredMinDist = 0.0f;
    float squaredMaxDist = 0.0f;
    if (myBoxValid == false) {
        dead = true;
        squaredMinDist = -1.0f;
        squaredMaxDist = -1.0f;
    }

    for (int dimension = 0; dimension < numDimensions; dimension++) {
        if (dimension % 64 == 0) {
            __syncthreads();
            if (threadIdx.x < 32) {
                if (dimension + threadIdx.x < numDimensions)
                    ((float2 *)sharedPoint)[threadIdx.x] = ((float2 *)point)[dimension / 2 + threadIdx.x];
            }
            __syncthreads();
        }
        if (!dead) {
            float2 span = boxSpan[1024 * dimension + 256 * blockIdx.x + threadIdx.x];
            float pointDimension = sharedPoint[dimension % 64];
            squaredMinDist += getMinimumDimensionSquaredDistance(pointDimension, span.x, span.y);
            squaredMaxDist += getMaximumDimensionSquaredDistance(pointDimension, span.x, span.y);
        }
    }

    // write the output
    float2 extremes;
    extremes.x = squaredMinDist;
    extremes.y = squaredMaxDist;
    minMaxBoxDistance[256 * blockIdx.x + threadIdx.x] = extremes;
}

/*
// Then keep the box with the minimum max distance and all those with the minimum min distance
// 256 threads per block * 4 blocks = 1024 threads
__global__ void getMinAndMaxDistanceToAllBoxesBatched(float *point, int numDimensions, float2* boxSpan, int numBoxes,
float2 *minMaxBoxDistance) {
    __shared__ float sharedPoint[64];

    bool myBoxValid = 256*blockIdx.x + threadIdx.x < numBoxes;
    bool dead = false;
    float squaredMinDist = 0.0f;
    float squaredMaxDist = 0.0f;
    if(myBoxValid == false) {
        dead = true;
        squaredMinDist = -1.0f;
        squaredMaxDist = -1.0f;
    }


    for(int dimension = 0; dimension < numDimensions; dimension++) {

        if(dimension % 64 == 0) {
            __syncthreads();
            if(threadIdx.x < 32) {
                if(dimension + threadIdx.x < numDimensions)
                    ((float2*)sharedPoint)[threadIdx.x] = ((float2*)point)[dimension/2 + threadIdx.x];
            }
            __syncthreads();
        }
        if(!dead) {
            float2 span = boxSpan[1024*dimension + 256*blockIdx.x + threadIdx.x];
            float pointDimension = sharedPoint[dimension % 64];
            squaredMinDist += getMinimumDimensionSquaredDistance(pointDimension, span.x, span.y);
            squaredMaxDist += getMaximumDimensionSquaredDistance(pointDimension, span.x, span.y);
        }
    }

    // write the output
    float2 extremes;
    extremes.x = squaredMinDist;
    extremes.y = squaredMaxDist;
    minMaxBoxDistance[256*blockIdx.x + threadIdx.x] = extremes;
}
*/

// float2[1024] minAndMaxDistance contains 1024 entries of (minDist, maxDist)
// 1 warp of 32 threads
__global__ void getMinMinAndMinMax(float2 *minMaxBoxDistance, int numBoxes, int2 *extremeBoxIndexes) {
    float minMin;
    int minMinIndex;
    float minMax;
    int minMaxIndex;

    float2 buffer;

    if (threadIdx.x < numBoxes) {
        buffer = minMaxBoxDistance[threadIdx.x];
        minMin = buffer.x;
        minMax = buffer.y;
        minMinIndex = threadIdx.x;
        minMaxIndex = threadIdx.x;
    }
    for (int i = 32; i < numBoxes; i += 32) {
        if (threadIdx.x + i < numBoxes) {
            buffer = minMaxBoxDistance[threadIdx.x + i];
            if (buffer.x < minMin) {
                minMin = buffer.x;
                minMinIndex = threadIdx.x + i;
            }
            if (buffer.y < minMax) {
                minMax = buffer.y;
                minMaxIndex = threadIdx.x + i;
            }
        }
    }

    bool valid = threadIdx.x < numBoxes;
    warpShuffleToFindMin(minMin, minMinIndex, valid);
    warpShuffleToFindMin(minMax, minMaxIndex, valid);

    if (threadIdx.x == 0) {
        int2 extremes;
        extremes.x = minMinIndex;
        extremes.y = minMaxIndex;
        extremeBoxIndexes[0] = extremes;
    }
}

// 128 threads, for mem read efficiency, per each block and there are 4 blocks
// each thread reads 2 points, all dimensions and computes the distance
// points are laid out in memory as [p0d0, p1d0, p0d1, p1d1, p0d2, p1d2] (2 points of 3 dimensions here) (i.e. all
// dimension 0's then all dimension 1's etc.) each thread keeps the smaller of its two, negative distance indicates no
// valid points for the thread then the warp does some shuffling to compare its 32 values and find the smallest warps 1
// - 3 place the smallest value in the shared memory warp0 reads these values and decides the final winner
//
// If there is no nearest point in the node, then the output is undefined.
__global__ void getNearestPointInLeafNode(
    float2 *point, int numDimensions, float *boxesPoints, int numPoints, int *indexOfNearestPoint, float *squaredDistanceToNearestPoint) {
    __shared__ float sharedPoint[64];
    __shared__ float nearestDist[3];
    __shared__ int pointIndex[3];
    __shared__ bool sharedPointValid[3];
    float dist[2] = {0.0f, 0.0f};

    bool pointValid[2];
    int firstPointNumber = blockIdx.x * 256 + 2 * threadIdx.x;
    pointValid[0] = firstPointNumber < numPoints;
    pointValid[1] = firstPointNumber + 1 < numPoints;

    float dimensionOfPoint[2];
    for (int dimension = 0; dimension < numDimensions; dimension++) {
        if (dimension % 64 == 0) {
            __syncthreads();
            if (threadIdx.x < 32) {
                if (dimension + threadIdx.x < numDimensions) {
                    float2 twoDimensionsOfPoint = point[dimension / 2 + threadIdx.x];
                    sharedPoint[2 * threadIdx.x] = twoDimensionsOfPoint.x;
                    sharedPoint[2 * threadIdx.x + 1] = twoDimensionsOfPoint.y;
                }
            }
            __syncthreads();
        }

        if (pointValid[0]) {  // if at least one point is valid
            ((float2 *)dimensionOfPoint)[0] = ((float2 *)boxesPoints)[512 * dimension + 128 * blockIdx.x + threadIdx.x];
            float diff[2];
            if (pointValid[0]) {
                diff[0] = sharedPoint[dimension % 64] - dimensionOfPoint[0];
                diff[0] = diff[0] * diff[0];
                dist[0] += diff[0];
            }
            if (pointValid[1]) {
                diff[1] = sharedPoint[dimension % 64] - dimensionOfPoint[1];
                diff[1] = diff[1] * diff[1];
                dist[1] += diff[1];
            }
        }
    }

    // Put the minimum of the two distances into dist[0]
    int myPointIndex = 256 * blockIdx.x + 2 * threadIdx.x;
    if (pointValid[1] && dist[1] < dist[0]) {
        dist[0] = dist[1];
        myPointIndex += 1;
        pointValid[0] = pointValid[1];
    }

    // Put the minimum of the warps 32 distances into dist[0] of thread 0
    float finalDist = dist[0];
    bool finalPointValid = pointValid[0];
    warpShuffleToFindMin(finalDist, myPointIndex, finalPointValid);
    if (threadIdx.x == 32 || threadIdx.x == 64 || threadIdx.x == 96) {
        nearestDist[threadIdx.x / 32 - 1] = finalDist;
        pointIndex[threadIdx.x / 32 - 1] = myPointIndex;
        sharedPointValid[threadIdx.x / 32 - 1] = finalPointValid;
    }
    __syncthreads();

    // Now the 4 smallest distances are in diff[0] of threadIdx.x == 0 and in the 3 elements of nearestDist
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++) {
            float otherDist = nearestDist[i];
            bool otherPointValid = sharedPointValid[i];
            if (!finalPointValid || (otherPointValid && otherDist < finalDist)) {
                finalDist = otherDist;
                myPointIndex = pointIndex[i];
                finalPointValid = otherPointValid;
            }
        }

        // Write output
        squaredDistanceToNearestPoint[blockIdx.x] = finalDist;
        indexOfNearestPoint[blockIdx.x] = finalPointValid ? myPointIndex : -1;
    }
}

__device__ __forceinline__ void swap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

__device__ __forceinline__ void shuffleCombineFlags(uint &flags) {
    flags |= __shfl_down_sync(0xffffffff, flags, 16);
    flags |= __shfl_down_sync(0x0000ffff, flags, 8);
    flags |= __shfl_down_sync(0x000000ff, flags, 4);
    flags |= __shfl_down_sync(0x0000000f, flags, 2);
    flags |= __shfl_down_sync(0x00000003, flags, 1);
}

// the minimum valid value in the warp (and its index and valid status) will go to thread 0
__device__ __forceinline__ void warpShuffleToFindMin(float &value, int &index, bool &valid) {
    float otherValue;
    int otherIndex;
    bool otherValid;

    otherValue = __shfl_down_sync(0xffffffff, value, 16);
    otherIndex = __shfl_down_sync(0xffffffff, index, 16);
    otherValid = __shfl_down_sync(0xffffffff, valid, 16);
    if (!valid || (otherValid && otherValue < value)) {
        value = otherValue;
        index = otherIndex;
        valid = otherValid;
    }
    otherValue = __shfl_down_sync(0x0000ffff, value, 8);
    otherIndex = __shfl_down_sync(0x0000ffff, index, 8);
    otherValid = __shfl_down_sync(0x0000ffff, valid, 8);
    if (!valid || (otherValid && otherValue < value)) {
        value = otherValue;
        index = otherIndex;
        valid = otherValid;
    }
    otherValue = __shfl_down_sync(0x000000ff, value, 4);
    otherIndex = __shfl_down_sync(0x000000ff, index, 4);
    otherValid = __shfl_down_sync(0x000000ff, valid, 4);
    if (!valid || (otherValid && otherValue < value)) {
        value = otherValue;
        index = otherIndex;
        valid = otherValid;
    }
    otherValue = __shfl_down_sync(0x0000000f, value, 2);
    otherIndex = __shfl_down_sync(0x0000000f, index, 2);
    otherValid = __shfl_down_sync(0x0000000f, valid, 2);
    if (!valid || (otherValid && otherValue < value)) {
        value = otherValue;
        index = otherIndex;
        valid = otherValid;
    }
    otherValue = __shfl_down_sync(0x00003, value, 1);
    otherIndex = __shfl_down_sync(0x00003, index, 1);
    otherValid = __shfl_down_sync(0x00003, valid, 1);
    if (!valid || (otherValid && otherValue < value)) {
        value = otherValue;
        index = otherIndex;
        valid = otherValid;
    }
}
//__device__ __forceinline__ float warpShuffleToFindMin(float val) {
//  float otherVal;
//
//  otherVal = __shfl_down_sync(0xffffffff, val, 16);
//  if(otherVal < val)
//      val = otherVal;
//  otherVal = __shfl_down_sync(0x0000ffff, val, 8);
//  if(otherVal < val)
//      val = otherVal;
//  otherVal = __shfl_down_sync(0x000000ff, val, 4);
//  if(otherVal < val)
//      val = otherVal;
//  otherVal = __shfl_down_sync(0x0000000f, val, 2);
//  if(otherVal < val)
//      val = otherVal;
//  otherVal = __shfl_down_sync(0x00003, val, 1);
//  if(otherVal < val)
//      val = otherVal;
//
//  return val;
//}

// 16 columns of unsigned int's
// e.g.
//
// Numbers indicate their target row. Each thread owns one row in registers, starting with thread0 on the top.
//
// | 0| 0| 0| 0| ... | 0| 0|
// | 1| 1| 1| 1| ... | 1| 1|
// | 2| 2| 2| 2| ... | 2| 2|
// | 3| 3| 3| 3| ... | 3| 3|
// .........................
// |30|30|30|30| ... |30|30|
// |31|31|31|31| ... |31|31|
//
//
// And the result is this:
//
// | 0| 1| 2| 3| ... |30|31|
// | 0| 1| 2| 3| ... |30|31|
// | 0| 1| 2| 3| ... |30|31|
// | 0| 1| 2| 3| ... |30|31|
// .........................
// | 0| 1| 2| 3| ... |30|31|
// | 0| 1| 2| 3| ... |30|31|
//
__device__ void __forceinline__ warpTransposeUnit(unsigned int *row) {
    int warpThreadId = (threadIdx.x % 32);
    for (int startThread = 1; startThread < 32; startThread++) {
        for (int col = 0; col < 32 - startThread; col++) {
            unsigned int swap = row[col + 1];
            unsigned int received;
            received = __shfl_down_sync(FULL_MASK, row[col], 1);
            if (warpThreadId < 31 && warpThreadId >= startThread - 1)
                row[col + 1] = received;
            received = __shfl_up_sync(FULL_MASK, swap, 1);
            if (warpThreadId >= startThread)
                row[col] = received;
        }
    }
}

inline void enforceMinMax(float &min, float &max) {
    if (min > max) {
        float t = max;
        max = min;
        min = t;
    }
}

void testFindBoxThatContainsPoint(int numDimensions) {
    cudaError_t cudaStatus;
    cudaStream_t stream;

    float *point;
    float2 *boxSpan;
    int numBoxes;
    uint *boxContainsPoint;
    cudaStatus = cudaHostAlloc(&point, numDimensions * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&boxSpan, 1024 * numDimensions * sizeof(float2), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&boxContainsPoint, 32 * sizeof(uint), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    float first[numDimensions];
    float last[numDimensions];
    for (int dimension = 0; dimension < numDimensions; dimension++) {
        first[dimension] = kernelsRandRange(-10, 3);
        last[dimension] = kernelsRandRange(4, 10);
        float addPerBox = (last[dimension] - first[dimension]) / 40.0f;
        for (int box = 0; box < 1024; box++) {
            boxSpan[1024 * dimension + box].x = kernelsRandRange(first[dimension] + addPerBox * box, last[dimension]);
            boxSpan[1024 * dimension + box].y = kernelsRandRange(first[dimension] + addPerBox * box, last[dimension]);
        }
    }

    numBoxes = (rand() % 1024) + 1;

    for (int dimension = 0; dimension < numDimensions; dimension++)
        point[dimension] = kernelsRandRange(boxSpan[1024 * dimension + 0].x, boxSpan[1024 * dimension + 1023].y);

    bool boxContainsPoint_cpu[1024];
    int boxCount = 0;
    for (int box = 0; box < 1024; box++) {
        if (box >= numBoxes) {
            boxContainsPoint_cpu[box] = false;
            continue;
        }

        int d;
        for (d = 0; d < numDimensions; d++) {
            enforceMinMax(boxSpan[1024 * d + box].x, boxSpan[1024 * d + box].y);
            if (point[d] < boxSpan[1024 * d + box].x || point[d] > boxSpan[1024 * d + box].y) {
                boxContainsPoint_cpu[box] = false;
                break;
            }
        }
        if (d == numDimensions) {
            boxContainsPoint_cpu[box] = true;
            boxCount++;
        }
    }

    printf("Point %f:%f is in %d box%s\n", point[0], point[1], boxCount, boxCount == 1 ? "" : "es");
    printf("CPU Boxes:");
    for (int box = 0; box < 1024; box++) {
        if (boxContainsPoint_cpu[box])
            printf(" %d", box);
    }
    printf("\n");

    cudaStatus = cudaSetDevice(0);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    float *point_d;
    float2 *boxSpan_d;
    uint *boxContainsPoint_d;

    cudaStatus = cudaMalloc(&point_d, numDimensions * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&boxSpan_d, 1024 * numDimensions * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&boxContainsPoint_d, 32 * sizeof(uint));
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(point_d, point, numDimensions * sizeof(float), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(boxSpan_d, boxSpan, 1024 * numDimensions * sizeof(float2), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(256);
    for (int i = 0; i < 1000; i++)
        findBoxesThatContainPoint<<<gridSize, blockSize, 0, stream>>>(point_d, numDimensions, boxSpan_d, numBoxes, boxContainsPoint_d);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(boxContainsPoint, boxContainsPoint_d, 32 * sizeof(uint), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    printf("GPU Boxes:");
    for (int box = 0; box < 1024; box++) {
        if ((boxContainsPoint[box / 32] >> (box % 32)) & 0x1)
            printf(" %d", box);
    }
    printf("\n");
    fflush(stdout);

    for (int box = 0; box < 1024; box++) {
        bool boxContainsPoint_gpu = (boxContainsPoint[box / 32] >> (box % 32)) & 0x1;
        assert(boxContainsPoint_cpu[box] == boxContainsPoint_gpu);
    }

    cudaFreeHost(point);
    cudaFreeHost(boxSpan);
    cudaFreeHost(boxContainsPoint);

    cudaFree(point_d);
    cudaFree(boxSpan_d);
    cudaFree(boxContainsPoint_d);

    cudaStatus = cudaStreamDestroy(stream);
    assert(cudaStatus == cudaSuccess);
}

void hostSwap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

float hostGetMinimumDimensionSquaredDistance(float pointDimension, float rangeEnd0, float rangeEnd1) {
    if (rangeEnd0 > rangeEnd1)
        hostSwap(rangeEnd0, rangeEnd1);

    if (rangeEnd0 <= pointDimension && rangeEnd1 >= pointDimension) {  // top is above point bottom is below point
        return 0.0f;
    } else if (rangeEnd1 < pointDimension) {  // top and bottom are below point
        float diff = pointDimension - rangeEnd1;
        return diff * diff;
    } else {  // top and bottom are above point
        float diff = pointDimension - rangeEnd0;
        return diff * diff;
    }
}

void testFindBoxesWithinRadius(int numDimensions) {
    cudaError_t cudaStatus;
    cudaStream_t stream;

    float *point;
    float2 *boxSpan;
    int numBoxes;
    uint *radiusIntersectsBox;
    cudaStatus = cudaHostAlloc(&point, numDimensions * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&boxSpan, 1024 * numDimensions * sizeof(float2), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&radiusIntersectsBox, 32 * sizeof(uint), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    float first[numDimensions];
    float last[numDimensions];
    for (int dimension = 0; dimension < numDimensions; dimension++) {
        first[dimension] = kernelsRandRange(-10, 3);
        last[dimension] = kernelsRandRange(4, 10);
        float addPerBox = (last[dimension] - first[dimension]) / 40.0f;
        for (int box = 0; box < 1024; box++) {
            boxSpan[1024 * dimension + box].x = kernelsRandRange(first[dimension] + addPerBox * box, last[dimension]);
            boxSpan[1024 * dimension + box].y = kernelsRandRange(first[dimension] + addPerBox * box, last[dimension]);
        }
    }

    for (int dimension = 0; dimension < numDimensions; dimension++)
        point[dimension] = kernelsRandRange(boxSpan[1024 * dimension + 0].x, boxSpan[1024 * dimension + 1023].y);

    numBoxes = (rand() % 1024) + 1;

    float squaredRadius = 1000000.0f;

    bool radiusIntersectsBox_cpu[1024];
    int boxCount = 0;
    for (int box = 0; box < 1024; box++) {
        if (box >= numBoxes) {
            radiusIntersectsBox_cpu[box] = false;
            continue;
        }
        float squaredDist = 0.0f;
        int d;
        for (d = 0; d < numDimensions; d++) {
            enforceMinMax(boxSpan[1024 * d + box].x, boxSpan[1024 * d + box].y);
            squaredDist += hostGetMinimumDimensionSquaredDistance(point[d], boxSpan[1024 * d + box].x, boxSpan[1024 * d + box].y);
            if (squaredDist > squaredRadius) {
                radiusIntersectsBox_cpu[box] = false;
                break;
            }
        }
        if (d == numDimensions) {
            radiusIntersectsBox_cpu[box] = true;
            boxCount++;
        }
    }

    printf("%d box%s are in radius %f of point\n", boxCount, boxCount == 1 ? "" : "es", sqrt(squaredRadius));
    printf("CPU Boxes:");
    for (int box = 0; box < 1024; box++) {
        if (radiusIntersectsBox_cpu[box])
            printf(" %d", box);
    }
    printf("\n");

    cudaStatus = cudaSetDevice(0);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    float *point_d;
    float2 *boxSpan_d;
    uint *radiusIntersectsBox_d;

    cudaStatus = cudaMalloc(&point_d, numDimensions * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&boxSpan_d, 1024 * numDimensions * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&radiusIntersectsBox_d, 32 * sizeof(uint));
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(point_d, point, numDimensions * sizeof(float), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(boxSpan_d, boxSpan, 1024 * numDimensions * sizeof(float2), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(256);
    for (int i = 0; i < 1000; i++)
        findBoxesWithinRadius<<<gridSize, blockSize, 0, stream>>>(
            point_d, squaredRadius, numDimensions, boxSpan_d, numBoxes, radiusIntersectsBox_d);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(radiusIntersectsBox, radiusIntersectsBox_d, 32 * sizeof(uint), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    printf("GPU Boxes:");
    for (int box = 0; box < 1024; box++) {
        if ((radiusIntersectsBox[box / 32] >> (box % 32)) & 0x1)
            printf(" %d", box);
    }
    printf("\n");
    fflush(stdout);

    for (int box = 0; box < 1024; box++) {
        bool radiusIntersectsBox_gpu = (radiusIntersectsBox[box / 32] >> (box % 32)) & 0x1;
        assert(radiusIntersectsBox_cpu[box] == radiusIntersectsBox_gpu);
    }

    cudaFreeHost(point);
    cudaFreeHost(boxSpan);
    cudaFreeHost(radiusIntersectsBox);

    cudaFree(point_d);
    cudaFree(boxSpan_d);
    cudaFree(radiusIntersectsBox_d);

    cudaStatus = cudaStreamDestroy(stream);
    assert(cudaStatus == cudaSuccess);
}

void minValidDist(int numPoints, double dist[], int &minIndex, float &minDist) {
    minIndex = -1;
    for (int i = 0; i < numPoints; i++) {
        if (minIndex == -1) {
            minDist = dist[i];
            minIndex = i;
        } else if (dist[i] < minDist) {  // this is split to ensure NaN cannot cause issues in the condition logic
            minDist = dist[i];
            minIndex = i;
        }
    }
    assert(minIndex != -1);
}

void testGetNearestPointInLeafNode(int numDimensions) {
    // put 231 points in a leaf node with 25 invalid points
    // some of the invalid points are the the nearest points
    // Check that the nearest valid point is returned

    cudaError_t cudaStatus;
    cudaStream_t stream;

    cudaStatus = cudaSetDevice(0);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    float *point;
    float *boxesPoints;
    int numPoints;
    int *indexOfNearestPoint;
    float *squaredDistanceToNearestPoint;
    cudaStatus = cudaHostAlloc(&point, (numDimensions + 1) / 2 * sizeof(float2), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&boxesPoints, 1024 * numDimensions * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&indexOfNearestPoint, 4 * sizeof(int), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&squaredDistanceToNearestPoint, 4 * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    double dist[1024];
    for (int d = 0; d < numDimensions; d++) {
        point[d] = kernelsRandRange(-1000.0f, 1000.0f);
        for (int i = 0; i < 1024; i++) {
            if (d == 0)
                dist[i] = 0.0;
            boxesPoints[d * 1024 + i] = kernelsRandRange(-1000.0f, 1000.0f);
            float diff = (point[d] - boxesPoints[d * 1024 + i]);
            dist[i] += diff * diff;
            if (d == numDimensions - 1)
                dist[i] = sqrt(dist[i]);
        }
    }

    numPoints = (rand() % 1024) + 1;

    // find the nearest neighbor
    int minIndex;
    float minDist;
    minValidDist(numPoints, dist, minIndex, minDist);

    float2 *point_d;
    float *boxesPoints_d;
    int *indexOfNearestPoint_d;
    float *squaredDistanceToNearestPoint_d;
    cudaStatus = cudaMalloc(&point_d, (numDimensions + 1) / 2 * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&boxesPoints_d, 1024 * numDimensions * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&indexOfNearestPoint_d, 4 * sizeof(int));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&squaredDistanceToNearestPoint_d, 4 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(point_d, point, (numDimensions + 1) / 2 * sizeof(float2), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(boxesPoints_d, boxesPoints, 1024 * numDimensions * sizeof(float), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(128);
    for (int i = 0; i < 1000; i++)
        getNearestPointInLeafNode<<<gridSize, blockSize, 0, stream>>>(
            (float2 *)point_d, numDimensions, boxesPoints_d, numPoints, indexOfNearestPoint_d, squaredDistanceToNearestPoint_d);

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        printf("cudaStatus: %d\n", cudaStatus);
        fflush(stdout);
    }
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(indexOfNearestPoint, indexOfNearestPoint_d, 4 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus =
        cudaMemcpyAsync(squaredDistanceToNearestPoint, squaredDistanceToNearestPoint_d, 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    for (int i = 1; i < 4; i++) {
        if (indexOfNearestPoint[0] < 0 ||
            (squaredDistanceToNearestPoint[i] < squaredDistanceToNearestPoint[0] && indexOfNearestPoint[i] >= 0)) {
            squaredDistanceToNearestPoint[0] = squaredDistanceToNearestPoint[i];
            indexOfNearestPoint[0] = indexOfNearestPoint[i];
        }
    }

    printf("CPU  nearestNeighborIndex:%d nearestNeigborDist:%f   |  nearestNeighborIndex:%d, nearestNeigborDist:%f  GPU\n",
           minIndex,
           minDist,
           indexOfNearestPoint[0],
           sqrt(squaredDistanceToNearestPoint[0]));
    fflush(stdout);

    assert(minIndex == indexOfNearestPoint[0]);
    assert(abs(minDist - sqrt(squaredDistanceToNearestPoint[0])) < 0.01);

    cudaFreeHost(point);
    cudaFreeHost(boxesPoints);
    cudaFreeHost(indexOfNearestPoint);
    cudaFreeHost(squaredDistanceToNearestPoint);

    cudaFree(point_d);
    cudaFree(boxesPoints_d);
    cudaFree(indexOfNearestPoint_d);
    cudaFree(squaredDistanceToNearestPoint_d);

    cudaStatus = cudaStreamDestroy(stream);
    assert(cudaStatus == cudaSuccess);
}

void testMinAndMaxDistanceToAllBoxes(int numDimensions) {
    cudaError_t cudaStatus;
    cudaStream_t stream;

    float *point;
    float2 *boxSpan;
    int numBoxes;
    float2 *minMaxBoxDistance;
    cudaStatus = cudaHostAlloc(&point, numDimensions * sizeof(float), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&boxSpan, 1024 * numDimensions * sizeof(float2), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaHostAlloc(&minMaxBoxDistance, 1024 * sizeof(float2), cudaHostAllocPortable);
    assert(cudaStatus == cudaSuccess);

    float first[numDimensions];
    float last[numDimensions];
    for (int dimension = 0; dimension < numDimensions; dimension++) {
        first[dimension] = kernelsRandRange(-10, 3);
        last[dimension] = kernelsRandRange(4, 10);
        float addPerBox = (last[dimension] - first[dimension]) / 40.0f;
        for (int box = 0; box < 1024; box++) {
            boxSpan[1024 * dimension + box].x = kernelsRandRange(first[dimension] + addPerBox * box, last[dimension]);
            boxSpan[1024 * dimension + box].y = kernelsRandRange(first[dimension] + addPerBox * box, last[dimension]);
        }
    }

    numBoxes = (rand() % 1024) + 1;

    for (int dimension = 0; dimension < numDimensions; dimension++)
        point[dimension] = kernelsRandRange(boxSpan[1024 * dimension + 0].x, boxSpan[1024 * dimension + 1023].y);

    double2 minMaxBoxDistance_cpu[1024];
    for (int box = 0; box < numBoxes; box++) {
        minMaxBoxDistance_cpu[box].x = 0.0;
        minMaxBoxDistance_cpu[box].y = 0.0;
        for (int d = 0; d < numDimensions; d++) {
            enforceMinMax(boxSpan[1024 * d + box].x, boxSpan[1024 * d + box].y);

            // min distance to box
            if (boxSpan[1024 * d + box].x > point[d]) {
                minMaxBoxDistance_cpu[box].x += (boxSpan[1024 * d + box].x - point[d]) * (boxSpan[1024 * d + box].x - point[d]);
            } else if (boxSpan[1024 * d + box].y < point[d]) {
                minMaxBoxDistance_cpu[box].x += (boxSpan[1024 * d + box].y - point[d]) * (boxSpan[1024 * d + box].y - point[d]);
            }

            // max distance within box
            float diff0 = (boxSpan[1024 * d + box].x - point[d]) * (boxSpan[1024 * d + box].x - point[d]);
            float diff1 = (boxSpan[1024 * d + box].y - point[d]) * (boxSpan[1024 * d + box].y - point[d]);
            minMaxBoxDistance_cpu[box].y += diff0 > diff1 ? diff0 : diff1;
        }
    }

    cudaStatus = cudaSetDevice(0);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaStatus == cudaSuccess);

    float *point_d;
    float2 *boxSpan_d;
    float2 *minMaxBoxDistance_d;

    cudaStatus = cudaMalloc(&point_d, numDimensions * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&boxSpan_d, 1024 * numDimensions * sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&minMaxBoxDistance_d, 1024 * sizeof(float2));
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(point_d, point, numDimensions * sizeof(float), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpyAsync(boxSpan_d, boxSpan, 1024 * numDimensions * sizeof(float2), cudaMemcpyHostToDevice, stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(256);
    for (int i = 0; i < 1000; i++)
        getMinAndMaxDistanceToAllBoxes<<<gridSize, blockSize, 0, stream>>>(
            point_d, numDimensions, boxSpan_d, numBoxes, minMaxBoxDistance_d);

    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpyAsync(minMaxBoxDistance, minMaxBoxDistance_d, 1024 * sizeof(float2), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    for (int box = 0; box < numBoxes; box++) {
        // printf("box:%d min: %lf %f\n", box, minMaxBoxDistance_cpu[box].x, minMaxBoxDistance[box].x);
        // printf("box:%d max: %lf %f\n", box, minMaxBoxDistance_cpu[box].y, minMaxBoxDistance[box].y);
        assert(abs(minMaxBoxDistance_cpu[box].x - minMaxBoxDistance[box].x) < 0.04 * numDimensions &&
               std::isfinite(minMaxBoxDistance[box].x));
        assert(abs(minMaxBoxDistance_cpu[box].y - minMaxBoxDistance[box].y) < 0.04 * numDimensions &&
               std::isfinite(minMaxBoxDistance[box].y));
    }
    printf("Min max box distance test passed.\n");

    // Now test minMin and minMax
    int2 extremeIndexes;
    int2 *extremeIndexes_d;
    cudaStatus = cudaMalloc(&extremeIndexes_d, sizeof(float2));
    assert(cudaStatus == cudaSuccess);
    gridSize = dim3(1);
    blockSize = dim3(32);
    for (int i = 0; i < 1000; i++)
        getMinMinAndMinMax<<<gridSize, blockSize, 0, stream>>>(minMaxBoxDistance_d, numBoxes, extremeIndexes_d);
    cudaStatus = cudaMemcpyAsync(&extremeIndexes, extremeIndexes_d, sizeof(float2), cudaMemcpyDeviceToHost, stream);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaStreamSynchronize(stream);
    assert(cudaStatus == cudaSuccess);

    float2 extremes_cpu;
    std::set<int> minIndexes_cpu;
    std::set<int> maxIndexes_cpu;
    extremes_cpu.x = minMaxBoxDistance[0].x;
    extremes_cpu.y = minMaxBoxDistance[0].y;
    minIndexes_cpu.insert(0);
    maxIndexes_cpu.insert(0);
    for (int i = 1; i < numBoxes; ++i) {
        if (minMaxBoxDistance[i].x == extremes_cpu.x) {
            minIndexes_cpu.insert(i);
        } else if (minMaxBoxDistance[i].x < extremes_cpu.x) {
            extremes_cpu.x = minMaxBoxDistance[i].x;
            minIndexes_cpu = std::set<int>();
            minIndexes_cpu.insert(i);
        }
        if (minMaxBoxDistance[i].y == extremes_cpu.y) {
            maxIndexes_cpu.insert(i);
        } else if (minMaxBoxDistance[i].y < extremes_cpu.y) {
            extremes_cpu.y = minMaxBoxDistance[i].y;
            maxIndexes_cpu = std::set<int>();
            maxIndexes_cpu.insert(i);
        }
    }
    printf("GPU : minMinIndex %d, minMaxIndex %d, minMin %f, minMax %f, cpuMinThatBox %f, cpuMaxThatBox %f",
           extremeIndexes.x,
           extremeIndexes.y,
           minMaxBoxDistance[extremeIndexes.x].x,
           minMaxBoxDistance[extremeIndexes.y].y,
           minMaxBoxDistance_cpu[extremeIndexes.x].x,
           minMaxBoxDistance_cpu[extremeIndexes.y].y);
    fflush(stdout);
    if (minIndexes_cpu.find(extremeIndexes.x) == minIndexes_cpu.end() || maxIndexes_cpu.find(extremeIndexes.y) == maxIndexes_cpu.end()) {
        printf("\ncpu mins: ");
        for (auto it = minIndexes_cpu.begin(); it != minIndexes_cpu.end(); ++it)
            printf("%d_cpu:%f_gpu:%f ", *it, minMaxBoxDistance_cpu[*it].x, minMaxBoxDistance[*it].x);
        printf("\n");

        printf("\ncpu maxes: ");
        for (auto it = maxIndexes_cpu.begin(); it != maxIndexes_cpu.end(); ++it)
            printf("%d_cpu:%f_gpu:%f ", *it, minMaxBoxDistance_cpu[*it].y, minMaxBoxDistance[*it].y);
        printf("\n");

        fflush(stdout);
    }
    assert(minIndexes_cpu.find(extremeIndexes.x) != minIndexes_cpu.end());
    assert(maxIndexes_cpu.find(extremeIndexes.y) != maxIndexes_cpu.end());
    cudaFree(extremeIndexes_d);
    printf("  ...  test passed\n");
    fflush(stdout);

    cudaFreeHost(point);
    cudaFreeHost(boxSpan);
    cudaFreeHost(minMaxBoxDistance);

    cudaFree(point_d);
    cudaFree(boxSpan_d);
    cudaFree(minMaxBoxDistance_d);

    cudaStatus = cudaStreamDestroy(stream);
    assert(cudaStatus == cudaSuccess);
}

// cudaSetDevice must have been called previous to set the device associate with the stream as the current device
void launchGetMinMinAndMinMaxDistanceBoxIndexesForPoint(cudaStream_t stream,
                                                        float *queryPoint_d,
                                                        int numDimensions,
                                                        float2 *boxSpan_d,
                                                        int numBoxes,
                                                        float2 *minMaxBoxDistance_d,
                                                        int2 *minMinAndMinMaxBoxIndexesPerPoint_d) {
    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(256);
    getMinAndMaxDistanceToAllBoxes<<<gridSize, blockSize, 0, stream>>>(
        queryPoint_d, numDimensions, boxSpan_d, numBoxes, minMaxBoxDistance_d);
    gridSize = dim3(1);
    blockSize = dim3(32);
    getMinMinAndMinMax<<<gridSize, blockSize, 0, stream>>>(minMaxBoxDistance_d, numBoxes, minMinAndMinMaxBoxIndexesPerPoint_d);
}

// cudaSetDevice must have been called previous to set the device associate with the stream as the current device
void launchGetNearestPointInLeafNode(cudaStream_t stream,
                                     float *queryPoint_d,
                                     int numDimensions,
                                     int numPointsInBox,
                                     float *boxesPoints_d,
                                     int *indexOfNearestPoint_d,
                                     float *squaredDistanceToNearestPoint_d) {
    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(128);
    getNearestPointInLeafNode<<<gridSize, blockSize, 0, stream>>>(
        (float2 *)queryPoint_d, numDimensions, boxesPoints_d, numPointsInBox, indexOfNearestPoint_d, squaredDistanceToNearestPoint_d);
}

void launchFindBoxesWithinRadius(cudaStream_t stream,
                                 float *queryPoint_d,
                                 int numDimensions,
                                 float squaredRadius,
                                 float2 *boxSpan_d,
                                 int numBoxes,
                                 uint *radiusIntersectsBoxFlags_d) {
    dim3 gridSize = dim3(4);
    dim3 blockSize = dim3(256);
    findBoxesWithinRadius<<<gridSize, blockSize, 0, stream>>>(
        queryPoint_d, squaredRadius, numDimensions, boxSpan_d, numBoxes, radiusIntersectsBoxFlags_d);
}
