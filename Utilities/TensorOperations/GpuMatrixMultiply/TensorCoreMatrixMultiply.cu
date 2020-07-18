#include "TensorCoreMatrixMultiply.h"

using namespace nvcuda;

//-----------------------------------------------
//
// A = | WFIn0FOut0  WFIn1FOut0  WFIn2FOut0 ... |
//     | WFIn0FOut1  WFIn1FOut1  WFIn2FOut1 ... |
//     | WFIn0FOut2  WFIn1FOut2  WFIn2FOut2 ... |
//
// B = | FIn0B0      FIn0B1      FIn0B2     ... |
//     | FIn1B0      FIn1B1      FIn1B2     ... |
//     | FIn2B0      FIn2B1      FIn2B2     ... |
//
// i.e. WeightFor_feature0_output0, feature0_fromBatch0
//
// C = AB
//
// Dimensions:
// A: A_rows x A_cols
// B: A_cols x B_cols
// C: A_rows x B_cols
//
// All data is regular C++ row major
//
// ld_A means leading dimension of matrix A,
// i.e. the number of elements that separate the start
// of each row of A in memory, usually ld_A = A_cols.
//
// A_rows (M): number of outputs
// A_cols (K): number of input features
// B_cols (N): batch size
//-----------------------------------------------
void TensorCoreMatrixMultiply::multiply(const half *A,
                                        const half *B,
                                        half *C,
                                        half *workspace,
                                        const int32_t A_rows,
                                        const int32_t A_cols,
                                        const int32_t B_cols,
                                        const int32_t ld_A,
                                        const int32_t ld_B,
                                        const int32_t ld_C,
                                        const Stream stream) {
    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_cols > 0);
    assert(ld_A >= A_cols);
    assert(ld_B >= B_cols);
    assert(ld_C >= B_cols);

    // Don't call multiply(...) between calls to startingMultiThreadedKernelOptimization() and finishedMultiThreadedKernelOptimization()
    // only call chooseOptimalKernel(...) between those calls
    assert(TensorCoreMatrixMultiply::instance().useLocks == false);

    KernelRequirement kernelRequirement;
    int gpuNum = stream.getGpuNum();
    ScopedGpu scopedGpu(gpuNum);
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    kernelRequirement.rowsA = A_rows;
    kernelRequirement.colsA = A_cols;
    kernelRequirement.colsB = B_cols;
    kernelRequirement.ldA = A_cols;  // Note: for optimization purposes, all matrices are evaluated as being packed.
    kernelRequirement.ldB = B_cols;
    kernelRequirement.ldC = B_cols;
    kernelRequirement.allowWorkspace = true;

    KernelWithSpec kernelWithSpec;
    auto it = TensorCoreMatrixMultiply::instance().optimalKernels.find(kernelRequirement);
    assert(it != TensorCoreMatrixMultiply::instance().optimalKernels.end());
    kernelWithSpec = it->second;

    // printf("Using kernelIndex=%d   kernelHeight %d kernelWidth %d AMod %d BMod %d\n",
    //       kernelWithSpec.id,
    //       kernelWithSpec.aRowsPerBlock,
    //       kernelWithSpec.bColsPerBlock,
    //       kernelWithSpec.aRowSizeModulusRequirement,
    //       kernelWithSpec.bRowSizeModulusRequirement);

    kernelWithSpec.executeKernel(A, B, C, workspace, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C, stream);
}

void TensorCoreMatrixMultiply::multiply(const half *A,
                                        const half *B,
                                        half *C,
                                        const int32_t A_rows,
                                        const int32_t A_cols,
                                        const int32_t B_cols,
                                        const int32_t ld_A,
                                        const int32_t ld_B,
                                        const int32_t ld_C,
                                        const Stream stream) {
    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_cols > 0);
    assert(ld_A >= A_cols);
    assert(ld_B >= B_cols);
    assert(ld_C >= B_cols);

    // Don't call multiply(...) between calls to startingMultiThreadedKernelOptimization() and finishedMultiThreadedKernelOptimization()
    // only call chooseOptimalKernel(...) between those calls
    assert(TensorCoreMatrixMultiply::instance().useLocks == false);

    KernelRequirement kernelRequirement;
    int gpuNum = stream.getGpuNum();
    ScopedGpu scopedGpu(gpuNum);
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    kernelRequirement.rowsA = A_rows;
    kernelRequirement.colsA = A_cols;
    kernelRequirement.colsB = B_cols;
    kernelRequirement.ldA = A_cols;  // Note: for optimization purposes, all matrices are evaluated as being packed.
    kernelRequirement.ldB = B_cols;
    kernelRequirement.ldC = B_cols;
    kernelRequirement.allowWorkspace = false;

    KernelWithSpec kernelWithSpec;
    auto it = TensorCoreMatrixMultiply::instance().optimalKernels.find(kernelRequirement);
    assert(it != TensorCoreMatrixMultiply::instance().optimalKernels.end());
    kernelWithSpec = it->second;

    /*
    printf("Using kernelHeight %d kernelWidth %d AMod %d BMod %d kernelIndex %d\n",
           kernelWithSpec.aRowsPerBlock,
           kernelWithSpec.bColsPerBlock,
           kernelWithSpec.aRowSizeModulusRequirement,
           kernelWithSpec.bRowSizeModulusRequirement,
           (int)kernelWithSpec.id);
    */

    kernelWithSpec.executeKernel(A, B, C, nullptr, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C, stream);
}

void TensorCoreMatrixMultiply::multiplyUsingHeuristicKernelChoice(const half *A,
                                                                  const half *B,
                                                                  half *C,
                                                                  const int32_t A_rows,
                                                                  const int32_t A_cols,
                                                                  const int32_t B_cols,
                                                                  const int32_t ld_A,
                                                                  const int32_t ld_B,
                                                                  const int32_t ld_C,
                                                                  const Stream stream) {
    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_cols > 0);
    assert(ld_A >= A_cols);
    assert(ld_B >= B_cols);
    assert(ld_C >= B_cols);

    // Don't call multiply(...) between calls to startingMultiThreadedKernelOptimization() and finishedMultiThreadedKernelOptimization()
    // only call chooseOptimalKernel(...) between those calls
    assert(TensorCoreMatrixMultiply::instance().useLocks == false);

    KernelRequirement kernelRequirement;
    int gpuNum = stream.getGpuNum();
    ScopedGpu scopedGpu(gpuNum);
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    kernelRequirement.rowsA = A_rows;
    kernelRequirement.colsA = A_cols;
    kernelRequirement.colsB = B_cols;
    kernelRequirement.ldA = ld_A;
    kernelRequirement.ldB = ld_B;
    kernelRequirement.ldC = ld_C;
    kernelRequirement.allowWorkspace = false;

    auto it = TensorCoreMatrixMultiply::instance().optimalKernels.find(kernelRequirement);

    KernelWithSpec kernelWithSpec;
    if (it == TensorCoreMatrixMultiply::instance().optimalKernels.end())
        kernelWithSpec = getHeuristicKernel(kernelRequirement);
    else
        kernelWithSpec = it->second;

    /*
    printf("Using kernelHeight %d kernelWidth %d AMod %d BMod %d\n",
           kernelWithSpec.aRowsPerBlock,
           kernelWithSpec.bColsPerBlock,
           kernelWithSpec.aRowSizeModulusRequirement,
           kernelWithSpec.bRowSizeModulusRequirement);
    */

    kernelWithSpec.executeKernel(A, B, C, nullptr, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C, stream);
}

vector<KernelWithSpec> TensorCoreMatrixMultiply::getEligibleKernels(KernelRequirement kernelRequirement) {
    vector<KernelWithSpec> eligibleKernels;

    const int numKernels = TensorCoreMatrixMultiply::instance().kernels.size();

    for (int i = 0; i < numKernels; ++i) {
        if (TensorCoreMatrixMultiply::instance().kernels[i].getWorkspaceSize(kernelRequirement) > 0 &&
            kernelRequirement.allowWorkspace == false)
            continue;
        if (TensorCoreMatrixMultiply::instance().kernels[i].aRowSizeModulusRequirement > 1 &&
            kernelRequirement.rowsA % TensorCoreMatrixMultiply::instance().kernels[i].aRowSizeModulusRequirement != 0)
            continue;
        if (TensorCoreMatrixMultiply::instance().kernels[i].aColSizeModulusRequirement > 1 &&
            kernelRequirement.colsA % TensorCoreMatrixMultiply::instance().kernels[i].aColSizeModulusRequirement != 0)
            continue;
        if (TensorCoreMatrixMultiply::instance().kernels[i].aColSizeModulusRequirement > 1 &&
            kernelRequirement.ldA % TensorCoreMatrixMultiply::instance().kernels[i].aColSizeModulusRequirement != 0)
            continue;
        if (TensorCoreMatrixMultiply::instance().kernels[i].bRowSizeModulusRequirement > 1 &&
            kernelRequirement.colsA % TensorCoreMatrixMultiply::instance().kernels[i].bRowSizeModulusRequirement != 0)
            continue;
        if (TensorCoreMatrixMultiply::instance().kernels[i].bColSizeModulusRequirement > 1 &&
            kernelRequirement.colsB % TensorCoreMatrixMultiply::instance().kernels[i].bColSizeModulusRequirement != 0)
            continue;
        if (TensorCoreMatrixMultiply::instance().kernels[i].bColSizeModulusRequirement > 1 &&
            kernelRequirement.ldB % TensorCoreMatrixMultiply::instance().kernels[i].bColSizeModulusRequirement != 0)
            continue;

        // Don't waste time with BCol8 kernels if the matrix is big enough that a BCol16 kernel will be fine
        if (kernelRequirement.colsB >= 32 && TensorCoreMatrixMultiply::instance().kernels[i].bColsPerBlock == 8)
            continue;

        eligibleKernels.push_back(TensorCoreMatrixMultiply::instance().kernels[i]);
    }

    return eligibleKernels;
}

KernelWithSpec TensorCoreMatrixMultiply::getHeuristicKernel(KernelRequirement kernelRequirement) {
    vector<KernelWithSpec> eligibleKernels = getEligibleKernels(kernelRequirement);
    assert(eligibleKernels.size() > 0);
    const int numEligibleKernels = eligibleKernels.size();

    // Find a gpu of the proper type or fail
    int gpuNum = -1;
    for (int i = 0; i < MachineEvaluator::instance().getNumGpus(); ++i) {
        if (MachineEvaluator::instance().getGpuType(i) == kernelRequirement.gpuType) {
            gpuNum = i;
            break;
        }
    }
    assert(gpuNum >= 0);

    constexpr float WAVE_MIN = 0.9;

    // Choose the kernel with the largest A rows per block * B cols per block that is
    // greater than WAVE_MIN of a wave.
    int selectedKernelIndex = 0;
    int largestKernelSize = eligibleKernels[0].aRowsPerBlock * eligibleKernels[0].bColsPerBlock;
    float selectedKernelWaves = computeWaves(kernelRequirement, eligibleKernels[0], gpuNum);
    /*
    printf("kernelHeight %d kernelWidth %d waves %f\n",
           eligibleKernels[0].aRowsPerBlock,
           eligibleKernels[0].bColsPerBlock,
           selectedKernelWaves);
    */
    for (int i = 1; i < numEligibleKernels; ++i) {
        /*
        printf("kernelHeight %d kernelWidth %d waves %f\n",
               eligibleKernels[i].aRowsPerBlock,
               eligibleKernels[i].bColsPerBlock,
               (kernelRequirement.rowsA / eligibleKernels[i].aRowsPerBlock + kernelRequirement.colsB / eligibleKernels[i].bColsPerBlock) /
                   blocksPerWave);
        */

        if (eligibleKernels[i].aRowsPerBlock >= 2 * kernelRequirement.rowsA && eligibleKernels[i].aRowsPerBlock > 16)
            continue;
        if (eligibleKernels[i].bColsPerBlock >= kernelRequirement.colsB && eligibleKernels[i].bColsPerBlock > 16)
            continue;

        float thisKernelWaves = computeWaves(kernelRequirement, eligibleKernels[i], gpuNum);

        float thisRatio = eligibleKernels[i].aRowsPerBlock / eligibleKernels[i].bColsPerBlock;
        if (thisRatio < 1.0f)
            thisRatio = 1.0f / thisRatio;
        float selectedRatio = eligibleKernels[selectedKernelIndex].aRowsPerBlock / eligibleKernels[selectedKernelIndex].bColsPerBlock;
        if (selectedRatio < 1.0f)
            selectedRatio = 1.0f / selectedRatio;

        if (selectedKernelWaves < WAVE_MIN && thisKernelWaves > selectedKernelWaves) {
            selectedKernelIndex = i;
            largestKernelSize = eligibleKernels[i].aRowsPerBlock * eligibleKernels[i].bColsPerBlock;
            selectedKernelWaves = thisKernelWaves;
        } else if (thisKernelWaves >= WAVE_MIN) {
            if (eligibleKernels[i].aRowsPerBlock * eligibleKernels[i].bColsPerBlock > largestKernelSize) {
                selectedKernelIndex = i;
                largestKernelSize = eligibleKernels[i].aRowsPerBlock * eligibleKernels[i].bColsPerBlock;
                selectedKernelWaves = thisKernelWaves;
            } else if (thisRatio < selectedRatio) {
                selectedKernelIndex = i;
                largestKernelSize = eligibleKernels[i].aRowsPerBlock * eligibleKernels[i].bColsPerBlock;
                selectedKernelWaves = thisKernelWaves;
            } else if (eligibleKernels[i].aRowsPerBlock * eligibleKernels[i].bColsPerBlock == largestKernelSize) {
                int selectedModReq = eligibleKernels[selectedKernelIndex].aRowSizeModulusRequirement +
                                     eligibleKernels[selectedKernelIndex].aColSizeModulusRequirement *
                                         eligibleKernels[selectedKernelIndex].bRowSizeModulusRequirement +
                                     eligibleKernels[selectedKernelIndex].bColSizeModulusRequirement;
                int thisKernelModReq = eligibleKernels[i].aRowSizeModulusRequirement +
                                       eligibleKernels[i].aColSizeModulusRequirement * eligibleKernels[i].bRowSizeModulusRequirement +
                                       eligibleKernels[i].bColSizeModulusRequirement;
                // Kernels with tighter requirements tend to be faster, otherwise there isn't a need to have the kernel since one is
                // available with looser requirements
                if (thisKernelModReq > selectedModReq) {
                    selectedKernelIndex = i;
                    largestKernelSize = eligibleKernels[i].aRowsPerBlock * eligibleKernels[i].bColsPerBlock;
                    selectedKernelWaves = thisKernelWaves;
                }
            }
        }
    }

    /*
    printf("kernelHeight %d kernelWidth %d waves %f aRows %d bCols %d\n",
           eligibleKernels[selectedKernelIndex].aRowsPerBlock,
           eligibleKernels[selectedKernelIndex].bColsPerBlock,
           selectedKernelWaves,
           kernelRequirement.rowsA,
           kernelRequirement.colsB);
    */

    return eligibleKernels[selectedKernelIndex];
}

float TensorCoreMatrixMultiply::computeWaves(KernelRequirement kernelRequirement, KernelWithSpec kernel, int gpuNum) {
    // Assuming here that two blocks can reside in an SM at a time
    int numMultiProcessors = MachineEvaluator::instance().getNumMultiProcessors(gpuNum);
    float blocksPerWave = numMultiProcessors * 2.0f;

    int aBlocks = (kernelRequirement.rowsA + (kernel.aRowsPerBlock - 1)) / kernel.aRowsPerBlock;
    int bBlocks = (kernelRequirement.colsB + (kernel.bColsPerBlock - 1)) / kernel.bColsPerBlock;

    return (aBlocks * bBlocks * kernel.blocksKSplitInto) / blocksPerWave;
}

void TensorCoreMatrixMultiply::chooseOptimalKernel(int gpuNum, int rowsA, int colsA, int colsB) {
    assert(gpuNum >= 0);
    assert(gpuNum < MachineEvaluator::instance().getNumGpus());

    string gpuType = MachineEvaluator::instance().getGpuType(gpuNum);

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = gpuType;
    kernelRequirement.rowsA = rowsA;
    kernelRequirement.colsA = colsA;
    kernelRequirement.colsB = colsB;
    kernelRequirement.ldA = colsA;  // Note: for optimization purposes, all matrices are evaluated as being packed.
    kernelRequirement.ldB = colsB;
    kernelRequirement.ldC = colsB;
    kernelRequirement.allowWorkspace = true;

    KernelRequirement kernelRequirementNoWorkspace = kernelRequirement;
    kernelRequirementNoWorkspace.allowWorkspace = false;

    if (TensorCoreMatrixMultiply::instance().useLocks)
        TensorCoreMatrixMultiply::instance().mtx.lock();

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (TensorCoreMatrixMultiply::instance().optimalKernels.count(kernelRequirement) == 1) {
        if (TensorCoreMatrixMultiply::instance().useLocks)
            TensorCoreMatrixMultiply::instance().mtx.unlock();
        return;
    }

    if (optimalKernelListing != nullptr) {
        try {
            auto it = TensorCoreMatrixMultiply::instance().optimalKernelListing->find(kernelRequirement);
            if (it != TensorCoreMatrixMultiply::instance().optimalKernelListing->end()) {
                pair<int, float> optimalKernelIdAndTime = it->second;
                int optimalKernelId = optimalKernelIdAndTime.first;
                float optimalKernelTime = optimalKernelIdAndTime.second;
                vector<KernelWithSpec> eligibleKernels = getEligibleKernels(kernelRequirement);
                int numKernels = eligibleKernels.size();
                int i;
                for (i = 0; i < numKernels; ++i) {
                    if ((int)eligibleKernels[i].id == optimalKernelId) {
                        TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirement] = eligibleKernels[i];
                        TensorCoreMatrixMultiply::instance().optimalKernelMeasuredTime[kernelRequirement] = optimalKernelTime;
                        TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirementNoWorkspace] = eligibleKernels[i];
                        TensorCoreMatrixMultiply::instance().optimalKernelMeasuredTime[kernelRequirementNoWorkspace] = optimalKernelTime;
                        break;
                    }
                }
                // If I found the desired kernel in the eligible kernels then return.
                // Otherwise continue on and measure for the optimal kernel.
                if (i < numKernels) {
                    if (TensorCoreMatrixMultiply::instance().useLocks)
                        TensorCoreMatrixMultiply::instance().mtx.unlock();
                    return;
                }
            }
        } catch (interprocess_exception ex) {
            removeDiskIndex();
            createDiskIndex();
            // Continue on and measure for the optimal kernel.
        }
    }

    // Put in a dummy kernel so in the multi-threaded case, another thread cannot try
    // to run a duplicate optimization in parallel. It will be replaced with the real
    // thing once the optimal kernel has been found.
    TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirement] = KernelWithSpec();

    if (TensorCoreMatrixMultiply::instance().useLocks)
        TensorCoreMatrixMultiply::instance().mtx.unlock();

    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    constexpr int REPEAT = 40;

    vector<KernelWithSpec> eligibleKernels = getEligibleKernels(kernelRequirement);
    assert(eligibleKernels.size() > 0);
    const int numEligibleKernels = eligibleKernels.size();

    // Allocate a lot of memory, to ensure subsequent calls are not benefitting from cache hits
    long FIVE_HUNDRED_MEGS = 536870912;
    long memPerInstance = (rowsA * colsA + colsA * colsB + rowsA * colsB) * sizeof(half);
    long numInstances = (FIVE_HUNDRED_MEGS + (memPerInstance - 1)) / memPerInstance;
    if (numInstances > (REPEAT + 1) * numEligibleKernels)
        numInstances = (REPEAT + 1) * numEligibleKernels;

    unsigned int maxWorkspaceSize = eligibleKernels[0].getWorkspaceSize(kernelRequirement);
    for (int i = 1; i < numEligibleKernels; ++i) {
        if (eligibleKernels[i].getWorkspaceSize(kernelRequirement) > maxWorkspaceSize)
            maxWorkspaceSize = eligibleKernels[i].getWorkspaceSize(kernelRequirement);
    }
    long TWO_FIFTY_MEGS = FIVE_HUNDRED_MEGS / 2;
    long numWorkspaceInstances = (TWO_FIFTY_MEGS + (maxWorkspaceSize - 1)) / maxWorkspaceSize;
    if (numWorkspaceInstances > (REPEAT + 1) * numEligibleKernels)
        numWorkspaceInstances = (REPEAT + 1) * numEligibleKernels;

    half **A;
    half **B;
    half **C;
    half **workspace;
    A = new half *[numInstances];
    B = new half *[numInstances];
    C = new half *[numInstances];
    workspace = new half *[numWorkspaceInstances];
    cudaError_t cudaStatus;

    for (int i = 0; i < numInstances; ++i) {
        cudaStatus = cudaMalloc(&(A[i]), rowsA * colsA * sizeof(half));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMalloc(&(B[i]), colsA * colsB * sizeof(half));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMalloc(&(C[i]), rowsA * colsB * sizeof(half));
        assert(cudaStatus == cudaSuccess);
    }
    for (int i = 0; i < numWorkspaceInstances; ++i) {
        cudaStatus = cudaMalloc(&(workspace[i]), maxWorkspaceSize);
        assert(cudaStatus == cudaSuccess);
    }

    cudaEvent_t *startEvents[REPEAT];
    cudaEvent_t *stopEvents[REPEAT];
    for (int i = 0; i < REPEAT; ++i) {
        startEvents[i] = new cudaEvent_t[numEligibleKernels];
        stopEvents[i] = new cudaEvent_t[numEligibleKernels];
        for (int j = 0; j < numEligibleKernels; ++j) {
            cudaStatus = cudaEventCreate(&(startEvents[i][j]));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaEventCreate(&(stopEvents[i][j]));
            assert(cudaStatus == cudaSuccess);
        }
    }

    int instance = 0;
    int workspaceInstance = 0;

    // warm up
    for (int i = 0; i < numEligibleKernels; ++i) {
        eligibleKernels[i].executeKernel(
            A[instance], B[instance], C[instance], workspace[workspaceInstance], rowsA, colsA, colsB, colsA, colsB, colsB, stream);

        ++instance;
        if (instance == numInstances)
            instance = 0;
        ++workspaceInstance;
        if (workspaceInstance == numWorkspaceInstances)
            workspaceInstance = 0;
    }

    // measure latency of every eligible kernel
    for (int i = 0; i < REPEAT; ++i) {
        for (int j = 0; j < numEligibleKernels; ++j) {
            cudaStatus = cudaEventRecord(startEvents[i][j], stream.getStream());
            eligibleKernels[j].executeKernel(
                A[instance], B[instance], C[instance], workspace[workspaceInstance], rowsA, colsA, colsB, colsA, colsB, colsB, stream);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaEventRecord(stopEvents[i][j], stream.getStream());
            assert(cudaStatus == cudaSuccess);

            ++instance;
            if (instance == numInstances)
                instance = 0;
            ++workspaceInstance;
            if (workspaceInstance == numWorkspaceInstances)
                workspaceInstance = 0;
        }
    }

    cudaEventSynchronize(stopEvents[REPEAT - 1][numEligibleKernels - 1]);

    float milliseconds = 0.0f;
    float *kernelTime = new float[numEligibleKernels];
    for (int i = 0; i < numEligibleKernels; ++i)
        kernelTime[i] = 0.0f;

    for (int i = 0; i < REPEAT; ++i) {
        for (int j = 0; j < numEligibleKernels; ++j) {
            cudaStatus = cudaEventElapsedTime(&milliseconds, startEvents[i][j], stopEvents[i][j]);
            assert(cudaStatus == cudaSuccess);
            kernelTime[j] += milliseconds;
        }
    }

    // Set optimal kernel
    int optimalKernelIndexWorkspace = 0;
    float optimalKernelTimeWorkspace = kernelTime[0];
    for (int i = 0; i < numEligibleKernels; ++i) {
        if (kernelTime[i] < optimalKernelTimeWorkspace) {
            optimalKernelTimeWorkspace = kernelTime[i];
            optimalKernelIndexWorkspace = i;
        }
    }
    optimalKernelTimeWorkspace /= REPEAT;

    int optimalKernelIndexNoWorkspace = -1;
    float optimalKernelTimeNoWorkspace;
    for (int i = 0; i < numEligibleKernels; ++i) {
        if ((optimalKernelIndexNoWorkspace == -1 || kernelTime[i] < optimalKernelTimeNoWorkspace) &&
            eligibleKernels[i].getWorkspaceSize(kernelRequirement) == 0) {
            optimalKernelTimeNoWorkspace = kernelTime[i];
            optimalKernelIndexNoWorkspace = i;
        }
    }
    optimalKernelTimeNoWorkspace /= REPEAT;

    if (TensorCoreMatrixMultiply::instance().useLocks)
        TensorCoreMatrixMultiply::instance().mtx.lock();

    // Store in memory
    TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirement] = eligibleKernels[optimalKernelIndexWorkspace];
    TensorCoreMatrixMultiply::instance().optimalKernelMeasuredTime[kernelRequirement] = optimalKernelTimeWorkspace;
    TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirementNoWorkspace] = eligibleKernels[optimalKernelIndexNoWorkspace];
    TensorCoreMatrixMultiply::instance().optimalKernelMeasuredTime[kernelRequirementNoWorkspace] = optimalKernelTimeNoWorkspace;

    // Store on disk
    if (optimalKernelListing != nullptr) {
        try {
            (*(TensorCoreMatrixMultiply::instance().optimalKernelListing))[kernelRequirement] =
                make_pair((int)eligibleKernels[optimalKernelIndexWorkspace].id, optimalKernelTimeWorkspace);
            (*(TensorCoreMatrixMultiply::instance().optimalKernelListing))[kernelRequirementNoWorkspace] =
                make_pair((int)eligibleKernels[optimalKernelIndexNoWorkspace].id, optimalKernelTimeNoWorkspace);
        } catch (interprocess_exception ex) {
            removeDiskIndex();
            createDiskIndex();
            // If disk is full or some other issue, then we just won't save the optimal kernel specs to disk.
        }
    }

    if (TensorCoreMatrixMultiply::instance().useLocks)
        TensorCoreMatrixMultiply::instance().mtx.unlock();

    /*
    for (int i = 0; i < numEligibleKernels; ++i) {
        printf("Matrix: ARows %d ACols %d BCols %d   |    Kernel: ARows %d BCols %d time %f\n",
               rowsA,
               colsA,
               colsB,
               eligibleKernels[i].aRowsPerBlock,
               eligibleKernels[i].bColsPerBlock,
               kernelTime[i] / REPEAT);
    }
    printf("minTime %f kernelHeight %d kernelWidth %d   TFLOPS %f workspace, %f no workspace\n",
           optimalKernelTimeWorkspace,
           TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirement].aRowsPerBlock,
           TensorCoreMatrixMultiply::instance().optimalKernels[kernelRequirement].bColsPerBlock,
           ((2.0 * colsA - 1.0) * rowsA * colsB) / (optimalKernelTimeWorkspace * 1.0e9),
           ((2.0 * colsA - 1.0) * rowsA * colsB) / (optimalKernelTimeNoWorkspace * 1.0e9));
    */

    // Clean up
    delete[] kernelTime;

    for (int i = 0; i < REPEAT; ++i) {
        for (int j = 0; j < numEligibleKernels; ++j) {
            cudaStatus = cudaEventDestroy(startEvents[i][j]);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaEventDestroy(stopEvents[i][j]);
            assert(cudaStatus == cudaSuccess);
        }
        delete[] startEvents[i];
        delete[] stopEvents[i];
    }

    for (int i = 0; i < numInstances; ++i) {
        cudaStatus = cudaFree((void *)(A[i]));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree((void *)(B[i]));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree((void *)(C[i]));
        assert(cudaStatus == cudaSuccess);
    }
    for (int i = 0; i < numWorkspaceInstances; ++i) {
        cudaStatus = cudaFree((void *)(workspace[i]));
        assert(cudaStatus == cudaSuccess);
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] workspace;
}

void TensorCoreMatrixMultiply::chooseOptimalKernel(string gpuType, int rowsA, int colsA, int colsB) {
    // Find a gpu of the proper type or fail, switch current gpu to that one while in this scope
    int gpuNum = -1;
    for (int i = 0; i < MachineEvaluator::instance().getNumGpus(); ++i) {
        if (MachineEvaluator::instance().getGpuType(i) == gpuType) {
            gpuNum = i;
            break;
        }
    }
    assert(gpuNum >= 0);

    chooseOptimalKernel(gpuNum, rowsA, colsA, colsB);
}

unsigned int TensorCoreMatrixMultiply::getWorkspaceSizeInBytes(int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC) {
    // in case it's not already known (will not measure when it's already known):
    chooseOptimalKernel(gpuNum, rowsA, colsA, colsB);

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    kernelRequirement.rowsA = rowsA;
    kernelRequirement.colsA = colsA;
    kernelRequirement.colsB = colsB;
    kernelRequirement.ldA = ldA;
    kernelRequirement.ldB = ldB;
    kernelRequirement.ldC = ldC;
    kernelRequirement.allowWorkspace = true;

    // Note: for optimization purposes, all matrices are evaluated as being packed.
    KernelRequirement optimizedKernelRequirement = kernelRequirement;
    optimizedKernelRequirement.ldA = colsA;
    optimizedKernelRequirement.ldB = colsB;
    optimizedKernelRequirement.ldC = colsB;

    return TensorCoreMatrixMultiply::instance().optimalKernels[optimizedKernelRequirement].getWorkspaceSize(kernelRequirement);
}

unsigned int TensorCoreMatrixMultiply::getWorkspaceSizeInBytes(string gpuType, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC) {
    // Find a gpu of the proper type or fail, switch current gpu to that one while in this scope
    int gpuNum = -1;
    for (int i = 0; i < MachineEvaluator::instance().getNumGpus(); ++i) {
        if (MachineEvaluator::instance().getGpuType(i) == gpuType) {
            gpuNum = i;
            break;
        }
    }
    assert(gpuNum >= 0);

    return getWorkspaceSizeInBytes(gpuNum, rowsA, colsA, colsB);
}

/**
 * Return the measured time of the fastest kernel for these matrix multiply dimensions on this type of gpu.
 * If chooseOptimalKernel(...) was not previously called for this kernel on this type of gpu, throws an out_of_range exception
 */
float TensorCoreMatrixMultiply::getOptimalKernelTime(string gpuType, int rowsA, int colsA, int colsB, bool workspaceAllowed) {
    // Don't call getOptimalKernelTime(...) between calls to startingMultiThreadedKernelOptimization() and
    // finishedMultiThreadedKernelOptimization() only call chooseOptimalKernel(...) between those calls
    assert(TensorCoreMatrixMultiply::instance().useLocks == false);

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = gpuType;
    kernelRequirement.rowsA = rowsA;
    kernelRequirement.colsA = colsA;
    kernelRequirement.colsB = colsB;
    kernelRequirement.ldA = colsA;
    kernelRequirement.ldB = colsB;
    kernelRequirement.ldC = colsB;
    kernelRequirement.allowWorkspace = workspaceAllowed;

    auto it = TensorCoreMatrixMultiply::instance().optimalKernelMeasuredTime.find(kernelRequirement);
    if (it == TensorCoreMatrixMultiply::instance().optimalKernelMeasuredTime.end()) {
        string message =
            "TensorCoreMatrixMultiply::getOptimalKernelTime() : Kernel time is not known because kernel time has not been measured for "
            "gpuType " +
            gpuType + " rowsA " + std::to_string(rowsA) + " colsA " + std::to_string(colsA) + " colsB " + std::to_string(colsB);
        throw(TensorCoreMatrixMultiply::Youreusingitwrong(message));
    }
    return it->second;
}

float TensorCoreMatrixMultiply::getOptimalKernelTime(int gpuNum, int rowsA, int colsA, int colsB, bool workspaceAllowed) {
    return getOptimalKernelTime(MachineEvaluator::instance().getGpuType(gpuNum), rowsA, colsA, colsB, workspaceAllowed);
}

void TensorCoreMatrixMultiply::startingMultiThreadedKernelOptimization() {
    TensorCoreMatrixMultiply::instance().mtx.lock();
    assert(TensorCoreMatrixMultiply::instance().useLocks == false);
    TensorCoreMatrixMultiply::instance().useLocks = true;
    TensorCoreMatrixMultiply::instance().mtx.unlock();
}

void TensorCoreMatrixMultiply::finishedMultiThreadedKernelOptimization() {
    TensorCoreMatrixMultiply::instance().mtx.lock();
    assert(TensorCoreMatrixMultiply::instance().useLocks == true);
    TensorCoreMatrixMultiply::instance().useLocks = false;
    TensorCoreMatrixMultiply::instance().mtx.unlock();
}

TensorCoreMatrixMultiply::TensorCoreMatrixMultiply() : CURRENT_KERNEL_VERSION(2) {
    useLocks = false;

    vector<KernelWithSpec> someKernels;
    someKernels = getBCol8Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol16Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol32Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol48Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol64Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol80Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol96Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());
    someKernels = getBCol112Kernels();
    kernels.insert(kernels.end(), someKernels.begin(), someKernels.end());

    // The optimalKernelListing map is nice to have but not necessary.
    // If it doesn't work on the host machine for any reason (permissions, no local storage, ...)
    // then it simply will not be used. This feature must never block the framework from executing.
    diskIndexFileName = std::getenv("HOME");
    if (!diskIndexFileName.empty())
        diskIndexFileName += "/.config/MLDev/MatrixMultiplyOptimalKernelListing.bin";
    optimalKernelListing = createDiskIndex();
}

TensorCoreMatrixMultiply::kernel_listing_map_t *TensorCoreMatrixMultiply::createDiskIndex() {
    if (diskIndexFileName.empty())
        return nullptr;

    optimalKernelListing = nullptr;
    /*
    try {
        optimalKernelListingFile = bi::managed_mapped_file(bi::open_or_create, diskIndexFileName.c_str(), 10000000);

        int *kernelVersion = optimalKernelListingFile.find_or_construct<int>("KERNEL_VERSION")(CURRENT_KERNEL_VERSION);

        if(*kernelVersion != CURRENT_KERNEL_VERSION) {
            removeDiskIndex();
            optimalKernelListingFile = bi::managed_mapped_file(bi::open_or_create, diskIndexFileName.c_str(), 10000000);
            int *kernelVersion = optimalKernelListingFile.find_or_construct<int>("KERNEL_VERSION")(CURRENT_KERNEL_VERSION);
            *kernelVersion = CURRENT_KERNEL_VERSION;
        }

        kernel_listing_map_allocator_t kernelListingMapAllocator(optimalKernelListingFile.get_segment_manager());
        optimalKernelListing =
            optimalKernelListingFile.find_or_construct<kernel_listing_map_t>("optimal_kernel_index_map")(kernelListingMapAllocator);

    } catch (interprocess_exception ex) {
        removeDiskIndex();
        optimalKernelListing = nullptr;
    }
    */

    return optimalKernelListing;
}

void TensorCoreMatrixMultiply::removeDiskIndex() {
    if (diskIndexFileName.empty())
        return;

    bi::file_mapping::remove(diskIndexFileName.c_str());
}
