#include "GpuRTree.h"
#include "GpuRTreeTests.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>

GpuRtree::GpuRtree(Point points[], long numPoints, bool buildMultiThread, int numDimensions) {
    assert(numPoints > 0);
    this->points = new Point[numPoints];
    for (long p = 0; p < numPoints; p++) {
        this->points[p] = points[p];
    }
    this->numDimensions = numDimensions;

    root = new GpuRtreeNode(numPoints, this->points, numDimensions);
    root->setMinAndMaxPerDimension();
    root->treeify(1, buildMultiThread);

    packed = false;

    cudaError_t cudaStatus = cudaGetDeviceCount(&numGpus);
    assert(cudaStatus == cudaSuccess);
    assert(numGpus > 0);
    globalContextPerGpu = new GpuContext *[numGpus];
    for (int g = 0; g < numGpus; ++g)
        globalContextPerGpu[g] = new GpuContext();
}

GpuRtree::~GpuRtree() {
    delete[] points;
    deleteTree(root);

    for (int g = 0; g < numGpus; ++g) {
        globalContextPerGpu[g]->tearDown();
        delete globalContextPerGpu[g];
    }
    delete[] globalContextPerGpu;
}

void GpuRtree::deleteTree(GpuRtreeNode *subtreeRoot) {
    // Delete all children then self
    for (auto iter = subtreeRoot->children.begin(); iter != subtreeRoot->children.end(); ++iter) {
        deleteTree(*iter);
    }
    if (subtreeRoot->dimensionMins != NULL)
        delete[] subtreeRoot->dimensionMins;
    if (subtreeRoot->dimensionMaxes != NULL)
        delete[] subtreeRoot->dimensionMaxes;
    delete subtreeRoot;
}

void GpuRtree::verifyRtree(Point points[], long numPoints) {
    printf("verifing rtree\n");

    // Verify that every point is in the tree exactly once
    std::unordered_map<long, long> pointIndexToArrayIndex;
    std::unordered_set<long> pointsStillToFindInTree;
    for (int i = 0; i < numPoints; ++i) {
        pointIndexToArrayIndex[points[i].pointIndex] = i;
        pointsStillToFindInTree.insert(points[i].pointIndex);
    }
    // Traverse the tree and find all expected points
    traverseSubtreeNotingPoints(root, points, pointsStillToFindInTree, pointIndexToArrayIndex, root->dimensionMins, root->dimensionMaxes);
    assert(pointsStillToFindInTree.empty());

    printf("rtree verified\n");
}

void GpuRtree::traverseSubtreeNotingPoints(GpuRtreeNode *node,
                                           Point points[],
                                           std::unordered_set<long> &pointsStillToFindInTree,
                                           std::unordered_map<long, long> &pointIndexToArrayIndex,
                                           float *subtreeDimensionMins,
                                           float *subtreeDimensionMaxes) {
    // Ensure the announced subtree bounds do not extend beyond the bounds of its parent
    for (int i = 0; i < numDimensions; ++i) {
        assert(node->dimensionMins[i] >= subtreeDimensionMins[i]);
        assert(node->dimensionMaxes[i] <= subtreeDimensionMaxes[i]);
    }

    if (node->isLeaf()) {
        assert(node->numSubtreePoints > 0);
        for (int p = 0; p < node->numSubtreePoints; ++p) {
            if (pointsStillToFindInTree.find(node->leafPoints[p].pointIndex) == pointsStillToFindInTree.end()) {
                printf("Error, found unexpected point with pointIndex %ld\n", node->leafPoints[p].pointIndex);
                fflush(stdout);
                assert(false);
            }
            pointsStillToFindInTree.erase(node->leafPoints[p].pointIndex);

            for (int d = 0; d < numDimensions; ++d) {
                assert(&(node->leafPoints[p].mem[d]) == &(points[pointIndexToArrayIndex[node->leafPoints[p].pointIndex]].mem[d]));
                assert(node->leafPoints[p].mem[d] >= node->dimensionMins[d]);
                assert(node->leafPoints[p].mem[d] <= node->dimensionMaxes[d]);
            }
        }
    } else {
        for (int i = 0; i < node->children.size(); ++i)
            traverseSubtreeNotingPoints(
                node->children[i], points, pointsStillToFindInTree, pointIndexToArrayIndex, node->dimensionMins, node->dimensionMaxes);
    }
}

void GpuRtree::verifyRtreePacking() {
    printf("verifing packed rtree\n");

    assert(isPacked());
    traverseSubtreeVerifyingPackedData(1, root);

    printf("packed rtree verified\n");
}

void GpuRtree::traverseSubtreeVerifyingPackedData(long level, GpuRtreeNode *subtree) {
    if (level <= 2) {
        assert(subtree->packedNodeData == NULL);
        for (long c = 0; c < subtree->children.size(); ++c)
            traverseSubtreeVerifyingPackedData(level + 1, subtree->children[c]);
    } else {
        assert(subtree->packedNodeData != NULL);

        if (subtree->isLeaf()) {
            for (long p = 0; p < subtree->numSubtreePoints; ++p) {
                for (long d = 0; d < numDimensions; ++d) {
                    assert(subtree->packedNodeData[d * subtree->numSubtreePoints + p] == subtree->leafPoints[p].mem[d]);
                }
            }
        } else {
            for (long d = 0; d < numDimensions; ++d) {
                for (long c = 0; c < subtree->children.size(); ++c) {
                    assert(subtree->packedNodeData[2 * (d * subtree->children.size() + c)] == subtree->children[c]->dimensionMins[d]);
                    assert(subtree->packedNodeData[2 * (d * subtree->children.size() + c) + 1] == subtree->children[c]->dimensionMaxes[d]);
                }
            }
        }

        for (long c = 0; c < subtree->children.size(); ++c)
            traverseSubtreeVerifyingPackedData(level + 1, subtree->children[c]);
    }
}

long GpuRtree::floatsNeededToPackRtree() { return floatsNeededToPackSubtree(1, root); }

long GpuRtree::floatsNeededToPackSubtree(long level, GpuRtreeNode *subtree) {
    if (level > 2) {
        if (subtree->isLeaf())
            return subtree->numSubtreePoints * numDimensions;

        // box spans for this level
        long totalFloats = subtree->children.size() + numDimensions * 2;
        // bytes used by subtrees rooted at each child
        for (long c = 0; c < subtree->children.size(); ++c) {
            totalFloats += floatsNeededToPackSubtree(level + 1, subtree->children[c]);
        }
        return totalFloats;
    } else {
        long totalFloats = 0;
        // bytes used by subtrees rooted at each child
        for (long c = 0; c < subtree->children.size(); ++c) {
            totalFloats += floatsNeededToPackSubtree(level + 1, subtree->children[c]);
        }
        return totalFloats;
    }
}

void GpuRtree::pack(float *allocatedSpace, long allocatedSpaceNumFloats) {
    long minFloats = floatsNeededToPackRtree();
    if (allocatedSpaceNumFloats < minFloats) {
        printf("Error: the rtree requires at least %ld floats of allocated space, but it was only given %ld\n",
               minFloats,
               allocatedSpaceNumFloats);
        assert(allocatedSpaceNumFloats >= minFloats);
    }

    packSubtree(1, root, allocatedSpace, allocatedSpace + allocatedSpaceNumFloats);
    packed = true;
}

void GpuRtree::unpack() {
    unpackSubtree(root);
    packed = false;
}

void GpuRtree::unpackSubtree(GpuRtreeNode *subtree) {
    subtree->packedNodeData = NULL;
    for (long c = 0; c < subtree->children.size(); ++c) {
        unpackSubtree(subtree->children[c]);
    }
}

void GpuRtree::packSubtree(long level, GpuRtreeNode *subtree, float *allocatedSpace, float *endOfAllocatedSpace) {
    std::vector<std::thread *> workers;

    if (level == 1) {
        workers.reserve(1100);
        subtree->packedNodeData = NULL;
        long numChildren = subtree->children.size();
        for (long c = 0; c < numChildren; ++c) {
            workers.push_back(
                new std::thread(&GpuRtree::packSubtree, this, level + 1, subtree->children[c], allocatedSpace, endOfAllocatedSpace));
            allocatedSpace += floatsNeededToPackSubtree(level + 1, subtree->children[c]);
        }
    } else if (level == 2) {
        subtree->packedNodeData = NULL;
        long numChildren = subtree->children.size();
        for (long c = 0; c < numChildren; ++c) {
            packSubtree(level + 1, subtree->children[c], allocatedSpace, endOfAllocatedSpace);
            allocatedSpace += floatsNeededToPackSubtree(level + 1, subtree->children[c]);
        }
    } else if (level > 2) {
        subtree->packedNodeData = allocatedSpace;
        if (subtree->isLeaf()) {
            assert(allocatedSpace + subtree->numSubtreePoints * numDimensions <= endOfAllocatedSpace);
            for (long d = 0; d < numDimensions; ++d) {
                for (long p = 0; p < subtree->numSubtreePoints; ++p) {
                    subtree->packedNodeData[d * subtree->numSubtreePoints + p] = subtree->leafPoints[p].mem[d];
                }
            }
            allocatedSpace += subtree->numSubtreePoints * numDimensions;
        } else {
            long numChildren = subtree->children.size();
            long numElementsInMinData = numChildren * numDimensions;
            assert(allocatedSpace + 2 * numElementsInMinData <= endOfAllocatedSpace);
            for (long d = 0; d < numDimensions; ++d) {
                for (long c = 0; c < numChildren; ++c) {
                    int spanOffset = 2 * (d * numChildren + c);
                    subtree->packedNodeData[spanOffset] = subtree->children[c]->dimensionMins[d];
                    subtree->packedNodeData[spanOffset + 1] = subtree->children[c]->dimensionMaxes[d];
                }
            }
            allocatedSpace += 2 * numElementsInMinData;

            for (long c = 0; c < numChildren; ++c) {
                packSubtree(level + 1, subtree->children[c], allocatedSpace, endOfAllocatedSpace);
                allocatedSpace += floatsNeededToPackSubtree(level + 1, subtree->children[c]);
            }
        }
    }

    if (level == 1) {
        while (!workers.empty()) {
            workers.back()->join();
            delete workers.back();
            workers.pop_back();
        }
    }
}

void GpuRtree::tearDownGlobalContext() {
    for (int i = 0; i < numGpus; ++i)
        globalContextPerGpu[i]->tearDown();
}

void GpuRtree::setUpGlobalContext(const std::vector<int> &gpus) {
    assert(!gpus.empty());

    cudaError_t cudaStatus;

    // Set up global context for every gpu that will be used
    int numLeafPointsL0 = 0;
    float *leafPointsL0 = NULL;
    int numBoxSpansL0 = 0;
    float2 *boxSpanL0 = NULL;

    int numLeafPointsL1 = 0;
    float *leafPointsL1 = NULL;
    int numBoxSpansL1 = 0;
    float2 *boxSpanL1 = NULL;

    std::vector<cudaStream_t> perGpuStreams;

    if (root->isLeaf()) {
        numLeafPointsL0 = root->numSubtreePoints;
        cudaStatus = cudaHostAlloc(
            &leafPointsL0, numLeafPointsL0 * numDimensions * sizeof(float), cudaHostAllocPortable | cudaHostAllocWriteCombined);
        assert(cudaStatus == cudaSuccess);
        float *nextLeafStart = leafPointsL0;
        for (int p = 0; p < root->numSubtreePoints; ++p) {
            memcpy(nextLeafStart, root->leafPoints[p].mem, numDimensions * sizeof(float));
            nextLeafStart += numDimensions;
        }

        for (std::vector<int>::const_iterator it = gpus.begin(); it != gpus.end(); ++it) {
            int gpuNum = *it;
            cudaStatus = cudaSetDevice(gpuNum);
            assert(cudaStatus == cudaSuccess);
            perGpuStreams.emplace_back();
            cudaStatus = cudaStreamCreateWithFlags(&(perGpuStreams.back()), cudaStreamNonBlocking);
            assert(cudaStatus == cudaSuccess);

            printf("copying %d leaf points for %ld MB for L0 on gpu %d\n",
                   numLeafPointsL0,
                   (numLeafPointsL0 * numDimensions * sizeof(float2)) / 1000000,
                   gpuNum);
            assert(numLeafPointsL0 > 0);
            cudaStatus = cudaMalloc(&(globalContextPerGpu[gpuNum]->leafPointsL0_d), numLeafPointsL0 * numDimensions * sizeof(float));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMemcpyAsync(globalContextPerGpu[gpuNum]->leafPointsL0_d,
                                         leafPointsL0,
                                         numLeafPointsL0 * numDimensions * sizeof(float),
                                         cudaMemcpyHostToDevice,
                                         perGpuStreams.back());
            assert(cudaStatus == cudaSuccess);
        }
    } else {
        numBoxSpansL0 = root->children.size();
        cudaStatus =
            cudaHostAlloc(&boxSpanL0, numBoxSpansL0 * numDimensions * sizeof(float2), cudaHostAllocPortable | cudaHostAllocWriteCombined);
        assert(cudaStatus == cudaSuccess);
        float2 *nextBoxSpanStart = boxSpanL0;
        for (int d = 0; d < numDimensions; ++d) {
            for (int i = 0; i < root->children.size(); ++i) {
                nextBoxSpanStart->x = root->children[i]->dimensionMins[d];
                nextBoxSpanStart->y = root->children[i]->dimensionMaxes[d];
                nextBoxSpanStart += 1;
            }
        }

        numLeafPointsL1 = 0;
        numBoxSpansL1 = 0;
        for (auto nodeIt : root->children) {
            if (nodeIt->isLeaf()) {
                numLeafPointsL1 += nodeIt->numSubtreePoints;
            } else {
                numBoxSpansL1 += nodeIt->children.size();
            }
        }
        if (numLeafPointsL1 != 0) {
            cudaStatus = cudaHostAlloc(
                &leafPointsL1, numLeafPointsL1 * numDimensions * sizeof(float), cudaHostAllocPortable | cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
        }
        cudaStatus =
            cudaHostAlloc(&boxSpanL1, numBoxSpansL1 * numDimensions * sizeof(float2), cudaHostAllocPortable | cudaHostAllocWriteCombined);
        assert(cudaStatus == cudaSuccess);
        for (auto nodeIt : root->children) {
            if (nodeIt->isLeaf()) {
                float *nextLeafStart = leafPointsL1;
                for (int p = 0; p < nodeIt->numSubtreePoints; ++p) {
                    memcpy(nextLeafStart, nodeIt->leafPoints[p].mem, numDimensions * sizeof(float));
                    nextLeafStart += numDimensions;
                }
            } else {
                nextBoxSpanStart = boxSpanL1;
                for (int d = 0; d < numDimensions; ++d) {
                    for (int i = 0; i < nodeIt->children.size(); ++i) {
                        nextBoxSpanStart->x = nodeIt->children[i]->dimensionMins[d];
                        nextBoxSpanStart->y = nodeIt->children[i]->dimensionMaxes[d];
                        nextBoxSpanStart += 1;
                    }
                }
            }
        }

        for (std::vector<int>::const_iterator it = gpus.begin(); it != gpus.end(); ++it) {
            int gpuNum = *it;
            cudaStatus = cudaSetDevice(gpuNum);
            assert(cudaStatus == cudaSuccess);
            perGpuStreams.emplace_back();
            cudaStatus = cudaStreamCreateWithFlags(&(perGpuStreams.back()), cudaStreamNonBlocking);
            assert(cudaStatus == cudaSuccess);

            assert(numBoxSpansL0 > 0);
            printf("copying %d box spans for %ld MB for L0 on gpu %d\n",
                   numBoxSpansL0,
                   (numBoxSpansL0 * numDimensions * sizeof(float2)) / 1000000,
                   gpuNum);
            cudaStatus = cudaMalloc(&(globalContextPerGpu[gpuNum]->boxSpanL0_d), numBoxSpansL0 * numDimensions * sizeof(float2));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMemcpyAsync(globalContextPerGpu[gpuNum]->boxSpanL0_d,
                                         boxSpanL0,
                                         numBoxSpansL0 * numDimensions * sizeof(float2),
                                         cudaMemcpyHostToDevice,
                                         perGpuStreams.back());
            assert(cudaStatus == cudaSuccess);

            if (numLeafPointsL1 > 0) {
                printf("copying %d leaf points for %ld MB for L1 on gpu %d\n",
                       numLeafPointsL1,
                       (numLeafPointsL1 * numDimensions * sizeof(float2)) / 1000000,
                       gpuNum);
                cudaStatus =
                    cudaMalloc(&(globalContextPerGpu[gpuNum]->leafPointsL1_d_mem), numLeafPointsL1 * numDimensions * sizeof(float));
                assert(cudaStatus == cudaSuccess);
                cudaStatus = cudaMemcpyAsync(globalContextPerGpu[gpuNum]->leafPointsL1_d_mem,
                                             leafPointsL1,
                                             numLeafPointsL1 * numDimensions * sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             perGpuStreams.back());
                assert(cudaStatus == cudaSuccess);
            }

            printf("copying %d box spans for %ld MB for L1 on gpu %d\n",
                   numBoxSpansL1,
                   (numBoxSpansL1 * numDimensions * sizeof(float2)) / 1000000,
                   gpuNum);
            if (numBoxSpansL1 > 0) {
                cudaStatus = cudaMalloc(&(globalContextPerGpu[gpuNum]->boxSpanL1_d_mem), numBoxSpansL1 * numDimensions * sizeof(float2));
                assert(cudaStatus == cudaSuccess);
                cudaStatus = cudaMemcpyAsync(globalContextPerGpu[gpuNum]->boxSpanL1_d_mem,
                                             boxSpanL1,
                                             numBoxSpansL1 * numDimensions * sizeof(float2),
                                             cudaMemcpyHostToDevice,
                                             perGpuStreams.back());
                assert(cudaStatus == cudaSuccess);
            }
        }

        for (std::vector<int>::const_iterator it = gpus.begin(); it != gpus.end(); ++it) {
            int gpuNum = *it;
            float *nextLeafStart = globalContextPerGpu[gpuNum]->leafPointsL1_d_mem;
            float2 *nextBoxSpanStart = globalContextPerGpu[gpuNum]->boxSpanL1_d_mem;
            for (int c = 0; c < root->children.size(); ++c) {
                if (root->children[c]->isLeaf()) {
                    globalContextPerGpu[gpuNum]->leafPointsL1_d.push_back(nextLeafStart);
                    nextLeafStart += numDimensions * root->children[c]->numSubtreePoints;
                } else {
                    globalContextPerGpu[gpuNum]->boxSpanL1_d.push_back(nextBoxSpanStart);
                    nextBoxSpanStart += numDimensions * root->children[c]->children.size();
                }
            }
        }
    }

    // Wait for all memory transfers to finish and then destroy the streams
    for (std::vector<cudaStream_t>::iterator it = perGpuStreams.begin(); it != perGpuStreams.end(); ++it) {
        cudaStatus = cudaStreamSynchronize(*it);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaStreamDestroy(*it);
        assert(cudaStatus == cudaSuccess);
    }
}

bool GpuRtree::isNearestNeighborPipelineSetUp() { return nearestNeighborPipelineSetUp; }

// 1 point array, all gpus

void GpuRtree::setUpNearestNeighborPipeline(float *packedQueryPointArray) {
    int numDevices;
    cudaError_t cudaStatus = cudaGetDeviceCount(&numDevices);
    std::vector<int> allGpus;
    for (int i = 0; i < numDevices; ++i)
        allGpus.push_back(i);
    std::vector<float *> singleArray;
    singleArray.push_back(packedQueryPointArray);
    setUpNearestNeighborPipeline(singleArray, allGpus);
}

// N point arrays, all gpus

void GpuRtree::setUpNearestNeighborPipeline(std::vector<float *> packedQueryPointArrays) {
    int numDevices;
    cudaError_t cudaStatus = cudaGetDeviceCount(&numDevices);
    std::vector<int> allGpus;
    for (int i = 0; i < numDevices; ++i)
        allGpus.push_back(i);
    setUpNearestNeighborPipeline(packedQueryPointArrays, allGpus);
}

// N arrays, any set of the gpus
// The idea about using N arrays is that each array is a memory mapped file (see BOOST library) and each array is on a
// different SSD, multiplying the read bandwidth. When memory is being used to store the point array, then it is
// probably not useful or necessary to use more than one point array. Don't use a spinning disk. Use memory or 1 or more
// SSD's. This is too fast for a spinning disk, the disk will probably break.

void GpuRtree::setUpNearestNeighborPipeline(std::vector<float *> packedQueryPointArrays, const std::vector<int> &gpus) {
    constexpr int TX_SIZE_BYTES = 1024 * 1024;
    constexpr int THREADS_PER_NEAREST_NEIGHBOR_QUEUE = 16;
    constexpr int INPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE = 3;
    constexpr int OUTPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE = 2;
    constexpr int THREADS_PER_POINT_READER_QUEUE = 4;
    constexpr int INPUT_BUFFER_MULTIPLE_FOR_POINT_READER_QUEUE = 10;
    constexpr int OUTPUT_BUFFER_MULTIPLE_FOR_POINT_READER_QUEUE = 10;

    assert(nearestNeighborPipelineSetUp == false);

    assert(pointReaderQueues.empty());
    assert(nearestNeighborQueues.empty());
    assert(pointReaderQueueBuffers.empty());
    assert(nearestNeighborQueueBuffers.empty());

    assert(!packedQueryPointArrays.empty());
    assert(!gpus.empty());
    this->gpus = gpus;

    cudaError_t cudaStatus;

    unsigned int numBytesPerPoint = numDimensions * sizeof(float);
    numPointsPerBuffer = (TX_SIZE_BYTES + (numBytesPerPoint - 1)) / numBytesPerPoint;
    if (numPointsPerBuffer < 4)
        numPointsPerBuffer = 4;
    unsigned int numBytesPerTransfer = numPointsPerBuffer * numBytesPerPoint;

    int numDevices;
    cudaStatus = cudaGetDeviceCount(&numDevices);
    assert(numDevices > 0);
    int numNearestNeighborBuffersPerQueue = THREADS_PER_NEAREST_NEIGHBOR_QUEUE * (INPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE +
                                                                                  OUTPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE);
    for (int i = 0; i < gpus.size(); ++i) {
        int gpuNum = gpus[i];
        assert(gpuNum < numDevices);

        nearestNeighborQueueBuffers.emplace_back();  // an empty vector of buffers per gpu
        for (int b = 0; b < numNearestNeighborBuffersPerQueue; ++b) {
            nearestNeighborQueueBuffers.back().emplace_back();
            nearestNeighborQueueBuffers.back().back().nearestNeighborIndexes = new long[numPointsPerBuffer];
        }

        // FIXME: I think these template parameters are causing problems, remove them
        nearestNeighborQueues.push_back(new WorkQueueUnordered<GpuPointData, GpuPointData>());
        int numHandlesNeeded = THREADS_PER_NEAREST_NEIGHBOR_QUEUE * OUTPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE;
        nearestNeighborQueues.back()->open(
            std::unique_ptr<ExecutorBase<GpuPointData, GpuPointData>>(new NearestNeighborExecutor(
                gpuNum, numDimensions, numHandlesNeeded, numPointsPerBuffer, globalContextPerGpu[gpuNum], root)),
            THREADS_PER_NEAREST_NEIGHBOR_QUEUE,
            INPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE,
            OUTPUT_BUFFER_MULTIPLE_FOR_NEAREST_NEIGHBOR_QUEUE);
        nearestNeighborQueues.back()->beginExecutorPerformanceTiming(0.1);
    }
    int numNearestNeighborBuffers = numNearestNeighborBuffersPerQueue * nearestNeighborQueues.size();

    for (int i = 0; i < packedQueryPointArrays.size(); ++i) {
        pointReaderQueues.push_back(new WorkQueueUnordered<PointData, PointData>());
        pointReaderQueues.back()->open(std::unique_ptr<ExecutorBase<PointData, PointData>>(
                                           new PackedArrayPointLoaderExecutor(packedQueryPointArrays[i], numDimensions)),
                                       THREADS_PER_POINT_READER_QUEUE,
                                       INPUT_BUFFER_MULTIPLE_FOR_POINT_READER_QUEUE,
                                       OUTPUT_BUFFER_MULTIPLE_FOR_POINT_READER_QUEUE);
        pointReaderQueues.back()->beginExecutorPerformanceTiming(0.1);
    }
    // Num buffers calculation: numThreads * (inputBufferMultiple + outputBufferMultiple) * numQueues, per type of queue
    // Note: The point reader buffers are in use until the nearest neighbors lookup finishes, so every nearest neighbor
    // buffer carries a point reader buffer
    int numPointReaderBuffers = THREADS_PER_POINT_READER_QUEUE *
                                (INPUT_BUFFER_MULTIPLE_FOR_POINT_READER_QUEUE + OUTPUT_BUFFER_MULTIPLE_FOR_POINT_READER_QUEUE) *
                                pointReaderQueues.size();
    for (int b = 0; b < numPointReaderBuffers + numNearestNeighborBuffers; ++b) {
        pointReaderQueueBuffers.emplace_back();
        cudaStatus =
            cudaHostAlloc(&(pointReaderQueueBuffers.back().mem), numBytesPerTransfer, cudaHostAllocPortable | cudaHostAllocWriteCombined);
        assert(cudaStatus == cudaSuccess);
    }

    setUpGlobalContext(gpus);

    nearestNeighborPipelineSetUp = true;
}

void GpuRtree::tearDownNearestNeighborPipeline() {
    while (!pointReaderQueues.empty()) {
        pointReaderQueues.back()->close();
        delete pointReaderQueues.back();
        pointReaderQueues.pop_back();
    }

    while (!nearestNeighborQueues.empty()) {
        nearestNeighborQueues.back()->close();
        delete nearestNeighborQueues.back();
        nearestNeighborQueues.pop_back();
    }

    for (int g = 0; g < numGpus; ++g)
        globalContextPerGpu[g]->tearDown();

    while (!pointReaderQueueBuffers.empty()) {
        cudaError_t cudaStatus;
        cudaStatus = cudaFreeHost(pointReaderQueueBuffers.back().mem);
        assert(cudaStatus == cudaSuccess);
        pointReaderQueueBuffers.pop_back();
    }

    while (!nearestNeighborQueueBuffers.empty()) {
        while (!nearestNeighborQueueBuffers.back().empty()) {
            delete[] nearestNeighborQueueBuffers.back().back().nearestNeighborIndexes;
            nearestNeighborQueueBuffers.back().pop_back();
        }
        nearestNeighborQueueBuffers.pop_back();
    }

    nearestNeighborPipelineSetUp = false;
}

// nearestNeigborIndexes is a pre-allocated array of long[numPoints]
void GpuRtree::getNearestNeighborForPoints(unsigned long firstPointIndex, unsigned long numPoints, long *nearestNeighborIndexes) {
    int numPointReaderQueues = pointReaderQueues.size();
    int numNearestNeighborQueues = nearestNeighborQueues.size();

    int nextPointReaderQueueToPush = 0;
    int nextPointReaderQueueToPop = 0;
    int nextNearestNeighborQueueToPush = 0;
    int nextNearestNeighborQueueToPop = 0;

    cudaError_t cudaStatus;

    unsigned long p = 0;
    unsigned long numNeighborsFilled = 0;
    unsigned long numIterations = (numPoints + numPointsPerBuffer - 1) / numPointsPerBuffer;
    unsigned long curIteration = 0;
    while (numNeighborsFilled < numPoints) {
        curIteration += 1;
        bool equalizeQueues = (float)curIteration / numIterations > 0.6667;

        if (p < numPoints && !pointReaderQueueBuffers.empty()) {
            PointData emptyPointDataBuffer = pointReaderQueueBuffers.back();
            emptyPointDataBuffer.firstPointIndex = p + firstPointIndex;
            emptyPointDataBuffer.numPoints = numPointsPerBuffer;
            if (emptyPointDataBuffer.firstPointIndex + emptyPointDataBuffer.numPoints > firstPointIndex + numPoints)
                emptyPointDataBuffer.numPoints = firstPointIndex + numPoints - emptyPointDataBuffer.firstPointIndex;

            std::vector<double> bufferLatency;
            double minLatency;
            std::vector<int> targetInputQueueOccupancy;
            if (equalizeQueues) {
                for (int i = 0; i < pointReaderQueues.size(); ++i) {
                    bufferLatency.push_back(pointReaderQueues[i]->getRunningAverageExecutorLatency());
                    if (i == 0 || bufferLatency.back() < minLatency)
                        minLatency = bufferLatency.back();
                }
                for (int i = 0; i < pointReaderQueues.size(); ++i)
                    targetInputQueueOccupancy.push_back(round(minLatency / bufferLatency[i] * pointReaderQueues[i]->inputQueueSize()));
            }

            for (int q = 0; q < numPointReaderQueues; ++q) {
                if (equalizeQueues && pointReaderQueues[nextPointReaderQueueToPush]->inputQueueOccupancy() >=
                                          targetInputQueueOccupancy[nextPointReaderQueueToPush]) {
                    nextPointReaderQueueToPush = (nextPointReaderQueueToPush + 1) % numPointReaderQueues;
                    continue;
                }
                bool pushSucceeded = pointReaderQueues[nextPointReaderQueueToPush]->tryPush(emptyPointDataBuffer);
                nextPointReaderQueueToPush = (nextPointReaderQueueToPush + 1) % numPointReaderQueues;
                if (pushSucceeded) {
                    pointReaderQueueBuffers.pop_back();
                    p += emptyPointDataBuffer.numPoints;
                    break;
                }
            }
        }

        // Advance the data from the output of the point reader queue to the input of the nearestNeighbor queue when
        // output is ready and next input is not full
        bool pointReaderPopWillSucceed;
        for (int i = 0; i < numPointReaderQueues; ++i) {
            pointReaderPopWillSucceed = pointReaderQueues[nextPointReaderQueueToPop]->isOutputReady();
            if (pointReaderPopWillSucceed)
                break;
            nextPointReaderQueueToPop = (nextPointReaderQueueToPop + 1) % numPointReaderQueues;
        }
        if (pointReaderPopWillSucceed) {
            std::vector<double> bufferLatency;
            double minLatency;
            std::vector<int> targetInputQueueOccupancy;
            if (equalizeQueues) {
                for (int i = 0; i < nearestNeighborQueues.size(); ++i) {
                    bufferLatency.push_back(nearestNeighborQueues[i]->getRunningAverageExecutorLatency());
                    if (i == 0 || bufferLatency.back() < minLatency)
                        minLatency = bufferLatency.back();
                }
                for (int i = 0; i < nearestNeighborQueues.size(); ++i)
                    // FIXME: this can go to zero. also shouldn't I consider the total latency of each queue after I
                    // enqueue 1 more tx? I probably should just enqueue to the queue that will finish currentQueueSize
                    // + 1 tx's first.
                    targetInputQueueOccupancy.push_back(round(minLatency / bufferLatency[i] * nearestNeighborQueues[i]->inputQueueSize()));
            }

            bool nearestNeighborPushWillSucceed = false;
            for (int i = 0; i < numNearestNeighborQueues; ++i) {
                if (equalizeQueues && nearestNeighborQueues[nextNearestNeighborQueueToPush]->inputQueueOccupancy() >=
                                          targetInputQueueOccupancy[nextNearestNeighborQueueToPush]) {
                    nextNearestNeighborQueueToPush = (nextNearestNeighborQueueToPush + 1) % numNearestNeighborQueues;
                    continue;
                }
                nearestNeighborPushWillSucceed = !nearestNeighborQueueBuffers[nextNearestNeighborQueueToPush].empty() &&
                                                 !nearestNeighborQueues[nextNearestNeighborQueueToPush]->isFull();
                if (nearestNeighborPushWillSucceed)
                    break;
                nextNearestNeighborQueueToPush = (nextNearestNeighborQueueToPush + 1) % numNearestNeighborQueues;
            }
            if (nearestNeighborPushWillSucceed) {
                PointData filledPointDataBuffer;
                bool pointReaderPopSucceeded = pointReaderQueues[nextPointReaderQueueToPop]->tryPop(filledPointDataBuffer);
                assert(pointReaderPopSucceeded);
                nextPointReaderQueueToPop = (nextPointReaderQueueToPop + 1) % numPointReaderQueues;

                bool nearestNeighborPushSucceeded;
                GpuPointData gpuPointDataBuffer = nearestNeighborQueueBuffers[nextNearestNeighborQueueToPush].back();
                nearestNeighborQueueBuffers[nextNearestNeighborQueueToPush].pop_back();
                gpuPointDataBuffer.cpuPointData = filledPointDataBuffer;
                nearestNeighborPushSucceeded = nearestNeighborQueues[nextNearestNeighborQueueToPush]->tryPush(gpuPointDataBuffer);
                assert(nearestNeighborPushSucceeded == true);
                nextNearestNeighborQueueToPush = (nextNearestNeighborQueueToPush + 1) % numNearestNeighborQueues;
            }
        }

        // Try to pop from any of the nearestNeighbor queues, and if successful then place all point indexes into their
        // proper spot in the output vector
        GpuPointData filledGpuPointDataBuffer;
        bool nearestNeighborPopSucceeded;
        int nearestNeighborQueueThatWasPopped = -1;
        for (int i = 0; i < numNearestNeighborQueues; ++i) {
            nearestNeighborPopSucceeded = nearestNeighborQueues[nextNearestNeighborQueueToPop]->tryPop(filledGpuPointDataBuffer);
            nearestNeighborQueueThatWasPopped = nextNearestNeighborQueueToPop;
            nextNearestNeighborQueueToPop = (nextNearestNeighborQueueToPop + 1) % numNearestNeighborQueues;
            if (nearestNeighborPopSucceeded) {
                break;
            }
        }
        if (nearestNeighborPopSucceeded) {
            memcpy(&(nearestNeighborIndexes[filledGpuPointDataBuffer.cpuPointData.firstPointIndex]),
                   filledGpuPointDataBuffer.nearestNeighborIndexes,
                   filledGpuPointDataBuffer.cpuPointData.numPoints * sizeof(long));
            numNeighborsFilled += filledGpuPointDataBuffer.cpuPointData.numPoints;
            pointReaderQueueBuffers.push_back(filledGpuPointDataBuffer.cpuPointData);
            nearestNeighborQueueBuffers[nearestNeighborQueueThatWasPopped].push_back(filledGpuPointDataBuffer);
        }
    }  // while(numNeighborsFilled < numPoints)
}

int main() {
    KernelTest();
    GpuRTreeConstructionTest();

    return 0;
}
