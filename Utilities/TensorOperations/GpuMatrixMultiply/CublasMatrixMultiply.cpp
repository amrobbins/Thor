#include "CublasMatrixMultiply.h"

//-----------------------------------------------
//
// A = | B0Fin0   B0Fin1   B0Fin2  ... |
//     | B1Fin0   B1Fin1   B1Fin2  ... |
//     | B2Fin0   B2Fin1   B2Fin2  ... |
//
// B = | W0Fin0   W1Fin0   W2Fin0  ... |
//     | W0Fin1   W1Fin1   W2Fin1  ... |
//     | W0Fin2   W1Fin2   W2Fin2  ... |
//
// C = | B0Fout0  B0Fout1  B0Fout2 ... |
//     | B1Fout0  B1Fout1  B1Fout2 ... |
//     | B2Fout0  B2Fout1  B2Fout2 ... |
//
// Where C = AB
//
// Dimensions:
// A: A_rows x A_cols
// B: A_cols x B_cols
// C: A_rows x B_cols
//
// All data is regular C++ row major
//
// B is |  w  w  w  ... |
//      |  e  e  e  ... |
//      |  i  i  i  ... |
//      |  g  g  g  ... |
//      |  h  h  h  ... |
//      |  t  t  t  ... |
//      |  s  s  s  ... |
//      |  F  F  F  ... |
//      |  o  o  o  ... |
//      |  u  u  u  ... |
//      |  t  t  t  ... |
//      |  0  1  2  ... |
//
// i.e. one column per output feature, whose height is the number of input features
//
// The other 2 matrices A_n_x_fin and C_n_x_fout are matrices with batchSize rows and numFeatures columns
//
// ld_A means leading dimension of matrix A,
// i.e. the number of elements that separate the start
// of each row of A in memory, usually ld_A = A_cols.
//
// A_rows (M): batch size
// A_cols (K): number of input features
// B_cols (N): number of output features
//-----------------------------------------------

// This variant allows non-packed matrices and uses a workspace
void CublasMatrixMultiply::multiply(Tensor A,
                                    Tensor B,
                                    Tensor C,
                                    Tensor workspace,
                                    const int32_t A_rows,
                                    const int32_t A_cols,
                                    const int32_t B_cols,
                                    // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two
                                    // adjacent rows in memory. Some slots at the end of a row may be unused.
                                    const int32_t ld_A,
                                    const int32_t ld_B,
                                    const int32_t ld_C,
                                    bool accumulate,
                                    TensorDescriptor::DataType ABCDataType,
                                    const Stream stream) {
    multiply(A, B, C, workspace, A_rows, A_cols, B_cols, ld_A, ld_B, ld_C, accumulate, ABCDataType, stream);
}

// This variant allows non-packed matrices, does not use a workspace
void CublasMatrixMultiply::multiply(Tensor A,
                                    Tensor B,
                                    Tensor C,
                                    const int32_t A_rows,
                                    const int32_t A_cols,
                                    const int32_t B_cols,
                                    // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two
                                    // adjacent rows in memory. Some slots at the end of a row may be unused.
                                    const int32_t ld_A,
                                    const int32_t ld_B,
                                    const int32_t ld_C,
                                    bool accumulate,
                                    TensorDescriptor::DataType ABCDataType,
                                    const Stream stream) {
    multiply(A, B, C, Optional<Tensor>::empty(), A_rows, A_cols, B_cols, ld_A, ld_B, ld_C, accumulate, ABCDataType, stream);
}

// This variant allows non-packed matrices
void CublasMatrixMultiply::multiply(Tensor A,
                                    Tensor B,
                                    Tensor C,
                                    Optional<Tensor> workspace,
                                    const int32_t A_rows,
                                    const int32_t A_cols,
                                    const int32_t B_cols,
                                    // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two
                                    // adjacent rows in memory. Some slots at the end of a row may be unused.
                                    const int32_t ld_A,
                                    const int32_t ld_B,
                                    const int32_t ld_C,
                                    bool accumulate,
                                    TensorDescriptor::DataType ABCDataType,
                                    const Stream stream) {
    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_cols > 0);
    assert(ld_A >= A_cols);
    assert(ld_B >= B_cols);
    assert(ld_C >= B_cols);
    // Check dataType of tensors
    assert(A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 ||
           A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(A.getDescriptor().getDataType() == B.getDescriptor().getDataType());
    assert(A.getDescriptor().getDataType() == C.getDescriptor().getDataType());
    // Check dimensions of tensors
    vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
    vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
    vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
    assert(ADimensions.size() == 2);
    assert(ADimensions[0] == (uint32_t)A_rows);
    assert(ADimensions[1] == (uint32_t)ld_A);
    assert(BDimensions[0] == (uint32_t)A_cols);
    assert(BDimensions[1] == (uint32_t)ld_B);
    assert(BDimensions[0] == (uint32_t)A_rows);
    assert(BDimensions[1] == (uint32_t)ld_C);

    // Don't call multiply(...) between calls to startingMultiThreadedKernelOptimization() and finishedMultiThreadedKernelOptimization()
    // only call chooseOptimalKernel(...) between those calls
    assert(CublasMatrixMultiply::instance().useLocks == false);

    int gpuNum = stream.getGpuNum();
    ScopedGpu scopedGpu(gpuNum);

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    kernelRequirement.rowsA = A_rows;
    kernelRequirement.colsA = A_cols;
    kernelRequirement.colsB = B_cols;
    kernelRequirement.ldA = ld_A;
    kernelRequirement.ldB = ld_B;
    kernelRequirement.ldC = ld_C;
    kernelRequirement.allowWorkspace = workspace.isPresent();

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    auto it = optimalKernels.find(cublasKernelRequirement);
    assert(it != optimalKernels.end());
    CublasKernel cublasKernel = it->second;

    // printf("Using kernelIndex=%d   kernelHeight %d kernelWidth %d AMod %d BMod %d\n",
    //       kernelWithSpec.id,
    //       kernelWithSpec.aRowsPerBlock,
    //       kernelWithSpec.bColsPerBlock,
    //       kernelWithSpec.aRowSizeModulusRequirement,
    //       kernelWithSpec.bRowSizeModulusRequirement);

    // Check byte size of workspace
    if (workspace.isPresent())
        assert(cublasKernel.getWorkspaceSizeInBytes(gpuNum) <= workspace.get().getDescriptor().getArraySizeInBytes());

    cublasKernel.executeKernel(A, B, C, C, workspace, accumulate, stream);
}

cudaDataType_t CublasMatrixMultiply::mapToCublasDataType(TensorDescriptor::DataType dataType) {
    if (dataType == TensorDescriptor::DataType::FP32)
        return CUDA_R_32F;
    else if (dataType == TensorDescriptor::DataType::FP16)
        return CUDA_R_16F;
    else
        assert(false);
}

// This variant allows non-packed matrices
void CublasMatrixMultiply::multiplyUsingHeuristicKernelChoice(
    Tensor A,
    Tensor B,
    Tensor C,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_cols,
    // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of
    // two adjacent rows in memory. Some slots at the end of a row may be unused.
    const int32_t ld_A,
    const int32_t ld_B,
    const int32_t ld_C,
    bool accumulate,
    TensorDescriptor::DataType ABCDataType,
    const Stream stream) {}

long minl(long a, long b) { return a < b ? a : b; }

long maxl(long a, long b) { return a > b ? a : b; }

void CublasMatrixMultiply::chooseOptimalKernel(
    int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC, TensorDescriptor::DataType ABCDataType, bool printResults) {
    bool bestKernelHasWorkspace = chooseOptimalKernel(gpuNum, rowsA, colsA, colsB, ldA, ldB, ldC, ABCDataType, true, printResults);

    // If the best kernel did not have a workspace, then it will be used for the no workspace version of the computation also
    if (!bestKernelHasWorkspace)
        chooseOptimalKernel(gpuNum, rowsA, colsA, colsB, ldA, ldB, ldC, ABCDataType, false, printResults);
}

bool CublasMatrixMultiply::chooseOptimalKernel(int gpuNum,
                                               int rowsA,
                                               int colsA,
                                               int colsB,
                                               int ldA,
                                               int ldB,
                                               int ldC,
                                               TensorDescriptor::DataType ABCDataType,
                                               bool allowWorkspaces,
                                               bool printResults) {
    assert(gpuNum >= 0);
    assert(gpuNum < (int)MachineEvaluator::instance().getNumGpus());
    assert(ABCDataType == TensorDescriptor::DataType::FP32 || ABCDataType == TensorDescriptor::DataType::FP16);

    constexpr long FIVE_HUNDRED_MEGS = 536870912;
    const int ELEMENT_SIZE = (ABCDataType == TensorDescriptor::DataType::FP32 ? sizeof(float) : sizeof(half));

    string gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    ScopedGpu scopedGpu(gpuNum);
    Stream stream(gpuNum);

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = gpuType;
    kernelRequirement.rowsA = rowsA;
    kernelRequirement.colsA = colsA;
    kernelRequirement.colsB = colsB;
    kernelRequirement.ldA = ldA;
    kernelRequirement.ldB = ldB;
    kernelRequirement.ldC = ldC;
    kernelRequirement.allowWorkspace = allowWorkspaces;

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    if (CublasMatrixMultiply::instance().useLocks)
        CublasMatrixMultiply::instance().mtx.lock();

    // Will only evaluate kernel once per gpu type
    // First check in memory.
    // If not in memory, check in file-based map
    // If not in file based map, measure latency to determine optimal kernel
    if (optimalKernels.count(cublasKernelRequirement) == 1) {
        if (CublasMatrixMultiply::instance().useLocks)
            CublasMatrixMultiply::instance().mtx.unlock();
        return false;  // To be safe, do not assume anything about whether the best kernel has a workspace or not.
    }

    // Put in a dummy kernel so in the multi-threaded case, another thread cannot try
    // to run a duplicate optimization in parallel. It will be replaced with the real
    // thing once the optimal kernel has been found.
    optimalKernels[cublasKernelRequirement] = CublasKernel();

    if (CublasMatrixMultiply::instance().useLocks)
        CublasMatrixMultiply::instance().mtx.unlock();

    vector<CublasKernel> kernels;
    const vector<int> splitKSequence{0, 2, 3, 4, 5, 6, 8, 12, 16, 32};
    unsigned long maxWorkspaceSizeInBytes = 0;

    vector<cublasLtMatmulAlgo_t> supportedAlgorithms;
    vector<int> supportedAlgorithmIds;
    getSupportedCublasAlgorithms(operationType, supportedAlgorithms, supportedAlgorithmIds, cublasKernelRequirement, gpuNum);
    assert(!supportedAlgorithms.empty());

    for (unsigned int algorithmIndex = 0; algorithmIndex < supportedAlgorithms.size(); ++algorithmIndex) {
        // Get the options that are supported for this algorithm
        vector<cublasLtMatmulTile_t> supportedTileSizes = getSupportedTileSizes(supportedAlgorithms[algorithmIndex]);
        int splitKSupported = isSplitKSupported(supportedAlgorithms[algorithmIndex]);
        int reductionSupportMask = getReductionSupportMask(supportedAlgorithms[algorithmIndex]);
        int swizzleMax = getSwizzleMaxValue(supportedAlgorithms[algorithmIndex]);
        int customKernelOptionMaxValue = getCustomKernelOptionMaxValue(supportedAlgorithms[algorithmIndex]);

        // Probably can use computed waves to choose the right ones, experiment by seeing the actual data.
        for (int tileIndex = 0; tileIndex < (int)supportedTileSizes.size(); ++tileIndex) {
            for (int splitKIndex = -1; splitKIndex == -1 || (splitKSupported && splitKIndex < (int)splitKSequence.size()); ++splitKIndex) {
                for (int reductionBitIndex = -1; reductionBitIndex < 32; ++reductionBitIndex) {
                    for (int swizzleType = 0; swizzleType <= swizzleMax; ++swizzleType) {
                        for (int customOptionValue = 0; customOptionValue < customKernelOptionMaxValue; ++customOptionValue) {
                            cublasLtMatmulTile_t tileSize = supportedTileSizes[tileIndex];
                            int splitK = splitKIndex == -1 ? 0 : splitKSequence[splitKIndex];
                            uint32_t reductionFlag = 0L;
                            if (reductionBitIndex == 0) {
                                // Avoid left shift by 0 not sure if it is always defined.
                                reductionFlag = 1 & reductionSupportMask;
                                if (reductionFlag == 0)
                                    continue;
                            } else if (reductionBitIndex > 0) {
                                reductionFlag = ((uint32_t)(1L << reductionBitIndex)) & reductionSupportMask;
                                if (reductionFlag == 0)
                                    continue;
                            }

                            int algorithmId = supportedAlgorithmIds[algorithmIndex];

                            CublasKernelOptions cublasKernelOptions(
                                algorithmId, tileSize, splitK, reductionFlag, swizzleType, customOptionValue);
                            CublasKernel cublasKernel(cublasKernelRequirement, cublasKernelOptions, gpuType);

                            unsigned long workspaceSizeInBytes = cublasKernel.getWorkspaceSizeInBytes(gpuNum);
                            if (workspaceSizeInBytes > 0 && !allowWorkspaces)
                                continue;
                            if (workspaceSizeInBytes > maxWorkspaceSizeInBytes)
                                maxWorkspaceSizeInBytes = workspaceSizeInBytes;

                            kernels.push_back(cublasKernel);
                        }
                    }
                }
            }
        }
    }
    assert(!kernels.empty());

    kernels = pruneBadFitKernels(kernels);
    assert(!kernels.empty());

    /**
     * All fully specified kernels that will be tried are now contained in the kernels vector.
     *
     * The following sequence will be performed to find the fastest kernel.
     *     1. Warm up the gpu (boost clock, cooling, etc) by running kernels for max(10 kernel executions, about 10 ms).
     *     2. Run all kernels 10 times, measure performance of each kernel.
     *          a. It is important to run kernel x once follow by kernel x+1 once followed by kernel x+2 once, ...,
     *             and run this whole sequence 10 times so that each kernel gets approximately the same clock frequencies on average.
     *     3. Discard all but the top 20 performing kernels.
     *     4. Run the top 20 kernels 40 times more each, add to the frequency measurement of these kernels.
     *     5. Choose the fastest one.
     *
     *     An option for the future is add a parameter to prefer a wave count less than 1.0, say 0.5, because the GPU will
     *     be running say 3 parallel streams, so that left over 50% of the GPU will be used for other kernels in parallel,
     *     the effect being that these lower wave kernels are more efficent and execution overlap provides full GPU utilization,
     *     which increases throughput, compared to using the fastest kernels as measured in isolation.
     *     To do this I would need to add a target waves parameter and have some way of computing the performance cost of choosing
     *     a kernel with a higher than requested wave count.
     */

    cublasStatus_t cublasStatus;

    // Allocate a lot of memory, to ensure subsequent calls are not benefitting from cache hits
    long totalGpuMem = MachineEvaluator::instance().getTotalGlobalMemBytes(gpuNum);
    long memPerInstance = (rowsA * ldA + colsA * ldB + rowsA * ldC) * ELEMENT_SIZE;
    long totalMatrixMemory = minl(totalGpuMem * 0.4, maxl(FIVE_HUNDRED_MEGS, 10 * memPerInstance));
    long numInstances = totalMatrixMemory / memPerInstance;
    assert(numInstances > 0);

    long numWorkspaceInstances;
    if (maxWorkspaceSizeInBytes == 0) {
        maxWorkspaceSizeInBytes = 1;
        numWorkspaceInstances = 1;
    } else {
        long totalWorkspaceMem = minl(totalGpuMem * 0.4, maxl(FIVE_HUNDRED_MEGS, 10 * maxWorkspaceSizeInBytes));
        numWorkspaceInstances = totalWorkspaceMem / maxWorkspaceSizeInBytes;
        assert(numWorkspaceInstances > 0);
    }

    vector<Tensor> A(numInstances);
    vector<Tensor> B(numInstances);
    vector<Tensor> C(numInstances);
    vector<Tensor> workspace(numWorkspaceInstances);
    for (int i = 0; i < numInstances; ++i) {
        A.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum), TensorDescriptor(ABCDataType, rowsA, ldA));
        B.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum), TensorDescriptor(ABCDataType, colsA, ldB));
        C.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum), TensorDescriptor(ABCDataType, rowsA, ldC));
    }
    for (int i = 0; i < numWorkspaceInstances; ++i) {
        workspace.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                               TensorDescriptor(TensorDescriptor::DataType::UINT8, maxWorkspaceSizeInBytes));
    }

    int tensorInstance = 0;
    int workspaceInstance = 0;

    // Warm up
    double elapsedTime = 0.0;
    Event startEvent = stream.putEvent();
    for (int i = 0; i < 5; ++i) {
        kernels[rand() % kernels.size()].runWithoutChecks(
            A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, stream);
        tensorInstance += 1;
        if (tensorInstance >= numInstances)
            tensorInstance = 0;
        workspaceInstance += 1;
        if (workspaceInstance >= numWorkspaceInstances)
            workspaceInstance = 0;
    }

    Event stopEvent = stream.putEvent();
    elapsedTime = stopEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
    double kernelExecutionTimeMilliseconds = elapsedTime / 5.0;
    int kernelsToExecute = maxl(5, 12.0 / kernelExecutionTimeMilliseconds);

    elapsedTime = 0.0;
    while (elapsedTime < 9.0) {
        Event startEvent = stream.putEvent();

        for (int i = 0; i < kernelsToExecute; ++i) {
            kernels[rand() % kernels.size()].runWithoutChecks(
                A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, stream);
            tensorInstance += 1;
            if (tensorInstance >= numInstances)
                tensorInstance = 0;
            workspaceInstance += 1;
            if (workspaceInstance >= numWorkspaceInstances)
                workspaceInstance = 0;
        }

        Event stopEvent = stream.putEvent();
        elapsedTime += stopEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
    }

    // Run all kernels 10 times and measure performance of each
    // But first put some initial work in the stream so that all timed work will be queued before running
    for (int i = 0; i < 5; ++i) {
        kernels[rand() % kernels.size()].runWithoutChecks(
            A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, stream);
        tensorInstance += 1;
        if (tensorInstance >= numInstances)
            tensorInstance = 0;
        workspaceInstance += 1;
        if (workspaceInstance >= numWorkspaceInstances)
            workspaceInstance = 0;
    }

    vector<vector<Event>> startEvents(kernels.size());
    vector<vector<Event>> stopEvents(kernels.size());
    for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
        startEvents[kernelIndex].reserve(100);
        stopEvents[kernelIndex].reserve(100);
    }

    for (int run = 0; run < 10; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            startEvents[kernelIndex].push_back(stream.putEvent());
            cublasStatus = kernels[kernelIndex].runWithoutChecks(
                A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, stream);
            tensorInstance += 1;
            if (tensorInstance >= numInstances)
                tensorInstance = 0;
            workspaceInstance += 1;
            if (workspaceInstance >= numWorkspaceInstances)
                workspaceInstance = 0;
            stopEvents[kernelIndex].push_back(stream.putEvent());
            if (cublasStatus != CUBLAS_STATUS_SUCCESS)
                kernels[kernelIndex].setErrorFlag();
        }
    }

    for (int run = 0; run < 10; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            kernels[kernelIndex].recordRun(
                stopEvents[kernelIndex][run].synchronizeAndReportElapsedTimeInMilliseconds(startEvents[kernelIndex][run]));
        }
    }

    // Discard any kernels that were not successful for any reason
    vector<CublasKernel> errorFreeKernels;
    errorFreeKernels.reserve(kernels.size());
    for (unsigned int i = 0; i < kernels.size(); ++i) {
        if (!kernels[i].getErrorFlag()) {
            errorFreeKernels.push_back(kernels[i]);
        } else if (printResults) {
            printf("discarded kernel to due error: %s\n", kernels[i].toString(gpuNum).c_str());
        }
    }
    if (printResults)
        printf("\n");

    assert(!errorFreeKernels.empty());
    kernels = errorFreeKernels;

    if (printResults) {
        printf("Initial kernel measurments:\n\n");
        std::sort(kernels.begin(), kernels.end(), CublasKernel::executionTimeComparison);
        for (unsigned int i = 0; i < kernels.size(); ++i) {
            printf("%s\n", kernels[i].toString(gpuNum).c_str());
        }
        printf("\n\n");
    }

    // Keep the best 20, discard the rest
    int kernelsToKeep = minl(20, kernels.size());
    std::partial_sort(kernels.begin(), kernels.begin() + kernelsToKeep, kernels.end(), CublasKernel::executionTimeComparison);
    kernels.erase(kernels.begin() + kernelsToKeep + 1, kernels.end());

    for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
        startEvents[kernelIndex].clear();
        stopEvents[kernelIndex].clear();
    }

    // Run the best 20 kernels 20 times more each and update performance measurements for each of them
    for (int run = 0; run < 20; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            startEvents[kernelIndex].push_back(stream.putEvent());
            cublasStatus = kernels[kernelIndex].runWithoutChecks(
                A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, stream);
            tensorInstance += 1;
            if (tensorInstance >= numInstances)
                tensorInstance = 0;
            workspaceInstance += 1;
            if (workspaceInstance >= numWorkspaceInstances)
                workspaceInstance = 0;
            stopEvents[kernelIndex].push_back(stream.putEvent());
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        }
    }

    for (int run = 0; run < 20; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            kernels[kernelIndex].recordRun(
                stopEvents[kernelIndex][run].synchronizeAndReportElapsedTimeInMilliseconds(startEvents[kernelIndex][run]));
        }
    }

    if (printResults) {
        printf("Kernel optimization results:\n\n");
        for (int i = 0; i < 20; ++i) {
            printf("%s\n", kernels[i].toString(gpuNum).c_str());
        }
        printf("\n");
    }

    CublasKernel bestKernel = *std::min_element(kernels.begin(), kernels.end(), CublasKernel::executionTimeComparison);
    bool bestKernelHasWorkspace = bestKernel.getWorkspaceSizeInBytes(gpuNum) > 0;

    // Save the result to be used later for this computation
    if (CublasMatrixMultiply::instance().useLocks)
        CublasMatrixMultiply::instance().mtx.lock();

    optimalKernels[cublasKernelRequirement] = bestKernel;

    // If the best one that may have a workspace has no workspace, then this is also the best one that may not have a workspace.
    if (allowWorkspaces && !bestKernelHasWorkspace) {
        CublasKernelRequirement noWorkspaceCublasKernelRequirement = bestKernel.getCublasKernelRequirement();
        noWorkspaceCublasKernelRequirement.kernelRequirement.allowWorkspace = false;
        optimalKernels[noWorkspaceCublasKernelRequirement] = bestKernel;
    }

    if (CublasMatrixMultiply::instance().useLocks)
        CublasMatrixMultiply::instance().mtx.unlock();

    return bestKernelHasWorkspace;
}

vector<CublasKernel> CublasMatrixMultiply::pruneBadFitKernels(vector<CublasKernel> &kernels) {
    // FIXME: implement
    return kernels;
}

void CublasMatrixMultiply::getSupportedCublasAlgorithms(const OperationType &operationType,
                                                        vector<cublasLtMatmulAlgo_t> &supportedAlgorithms,
                                                        vector<int> &supportedAlgorithmIds,
                                                        CublasKernelRequirement cublasKernelRequirement,
                                                        int gpuNum) {
    cublasStatus_t cublasStatus;

    vector<int> allSupportedAlgorithmIds;

    int numRequestedAlgos = 1000;
    int numReturnedAlgos = 1001;
    while (numReturnedAlgos >= numRequestedAlgos) {
        numRequestedAlgos *= 2;
        allSupportedAlgorithmIds = vector<int>(numRequestedAlgos);

        cublasStatus = cublasLtMatmulAlgoGetIds(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                                operationType.getComputeDataType(),
                                                operationType.getScaleDataType(),
                                                operationType.getADataType(),
                                                operationType.getBDataType(),
                                                operationType.getCDataType(),
                                                operationType.getDDataType(),
                                                numRequestedAlgos,
                                                allSupportedAlgorithmIds.data(),
                                                &numReturnedAlgos);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    for (unsigned int i = 0; i < allSupportedAlgorithmIds.size(); ++i) {
        cublasLtMatmulAlgo_t algo;
        cublasStatus = cublasLtMatmulAlgoInit(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                              operationType.getComputeDataType(),
                                              operationType.getScaleDataType(),
                                              operationType.getADataType(),
                                              operationType.getBDataType(),
                                              operationType.getCDataType(),
                                              operationType.getDDataType(),
                                              allSupportedAlgorithmIds[i],
                                              &algo);
        if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
            supportedAlgorithms.push_back(algo);
            supportedAlgorithmIds.push_back(allSupportedAlgorithmIds[i]);
        }
    }
}

vector<cublasLtMatmulTile_t> CublasMatrixMultiply::getSupportedTileSizes(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;

    // By sending sizeInBytes==0 below, the sizeWritten parameter is filled with the number bytes in an
    // array of cublasLtMatmulTile_t enums that would hold all of the supported tile configurations.
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
    unsigned int numSupportedTilesConfigurations = sizeWritten;
    vector<cublasLtMatmulTile_t> supportedTileSizeEnums;
    if (numSupportedTilesConfigurations == 0) {
        supportedTileSizeEnums.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    } else {
        supportedTileSizeEnums = vector<cublasLtMatmulTile_t>(numSupportedTilesConfigurations);
        cublasLtMatmulAlgoCapGetAttribute(&algo,
                                          CUBLASLT_ALGO_CAP_TILE_IDS,
                                          supportedTileSizeEnums.data(),
                                          numSupportedTilesConfigurations * sizeof(cublasLtMatmulTile_t),
                                          &sizeWritten);
        assert(sizeWritten == numSupportedTilesConfigurations);
    }

    return supportedTileSizeEnums;
}

int CublasMatrixMultiply::isSplitKSupported(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int splitKSupported;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitKSupported, sizeof(splitKSupported), &sizeWritten);
    assert(sizeWritten == sizeof(splitKSupported));

    return splitKSupported;
}

int CublasMatrixMultiply::getReductionSupportMask(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int reductionSupportMask;
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reductionSupportMask, sizeof(reductionSupportMask), &sizeWritten);
    assert(sizeWritten == sizeof(reductionSupportMask));

    return reductionSupportMask;
}

int CublasMatrixMultiply::getSwizzleMaxValue(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int swizzleMax;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzleMax, sizeof(swizzleMax), &sizeWritten);
    assert(sizeWritten == sizeof(swizzleMax));
    assert(swizzleMax == 0 || swizzleMax == 1);

    return swizzleMax;
}

int CublasMatrixMultiply::getCustomKernelOptionMaxValue(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int kernelOptionMaxValue;
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &kernelOptionMaxValue, sizeof(kernelOptionMaxValue), &sizeWritten);
    assert(sizeWritten == sizeof(kernelOptionMaxValue));

    return kernelOptionMaxValue;
}

unsigned int CublasMatrixMultiply::getWorkspaceSizeInBytes(
    int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC, TensorDescriptor::DataType ABCDataType) {
    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    kernelRequirement.rowsA = rowsA;
    kernelRequirement.colsA = colsA;
    kernelRequirement.colsB = colsB;
    kernelRequirement.ldA = ldA;
    kernelRequirement.ldB = ldB;
    kernelRequirement.ldC = ldC;
    kernelRequirement.allowWorkspace = true;

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    assert(optimalKernels.count(cublasKernelRequirement) == 1);

    return optimalKernels[cublasKernelRequirement].getWorkspaceSizeInBytes(gpuNum);
}

void CublasMatrixMultiply::startingMultiThreadedKernelOptimization() {
    CublasMatrixMultiply::instance().mtx.lock();
    assert(CublasMatrixMultiply::instance().useLocks == false);
    CublasMatrixMultiply::instance().useLocks = true;
    CublasMatrixMultiply::instance().mtx.unlock();
}

void CublasMatrixMultiply::finishedMultiThreadedKernelOptimization() {
    CublasMatrixMultiply::instance().mtx.lock();
    assert(CublasMatrixMultiply::instance().useLocks == true);
    CublasMatrixMultiply::instance().useLocks = false;
    CublasMatrixMultiply::instance().mtx.unlock();
}

double CublasMatrixMultiply::getOptimalKernelTime(string gpuType,
                                                  int rowsA,
                                                  int colsA,
                                                  int colsB,
                                                  int ldA,
                                                  int ldB,
                                                  int ldC,
                                                  bool workspaceAllowed,
                                                  TensorDescriptor::DataType ABCDataType) {
    // Don't call getOptimalKernelTime(...) between calls to startingMultiThreadedKernelOptimization() and
    // finishedMultiThreadedKernelOptimization() only call chooseOptimalKernel(...) between those calls
    assert(CublasMatrixMultiply::instance().useLocks == false);

    KernelRequirement kernelRequirement;
    kernelRequirement.gpuType = gpuType;
    kernelRequirement.rowsA = rowsA;
    kernelRequirement.colsA = colsA;
    kernelRequirement.colsB = colsB;
    kernelRequirement.ldA = ldA;
    kernelRequirement.ldB = ldB;
    kernelRequirement.ldC = ldC;
    kernelRequirement.allowWorkspace = workspaceAllowed;

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);
    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    string ABCDataTypeString = ABCDataType == TensorDescriptor::DataType::FP32 ? "FP32" : "FP16";

    auto it = optimalKernels.find(cublasKernelRequirement);
    if (it == optimalKernels.end()) {
        string message =
            "CublasMatrixMultiply::getOptimalKernelTime() : Kernel time is not known because kernel time has not been measured for "
            "gpuType " +
            gpuType + " rowsA " + std::to_string(rowsA) + " colsA " + std::to_string(colsA) + " colsB " + std::to_string(colsB) +
            "dataType " + ABCDataTypeString;
        throw(CublasMatrixMultiply::Youreusingitwrong(message));
    }
    return it->second.getAverageRunTimeMilliseconds();
}

double CublasMatrixMultiply::getOptimalKernelTime(
    int gpuNum, int rowsA, int colsA, int colsB, int ldA, int ldB, int ldC, bool workspaceAllowed, TensorDescriptor::DataType ABCDataType) {
    return getOptimalKernelTime(
        MachineEvaluator::instance().getGpuType(gpuNum), rowsA, colsA, colsB, ldA, ldB, ldC, workspaceAllowed, ABCDataType);
}
