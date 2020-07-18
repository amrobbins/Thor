#include "gpuMatrixMultiply.h"

cublasStatus_t matrixMultiply(cublasLtHandle_t cublasLtHandle,
                              cudaStream_t stream,
                              const void *A_d,
                              const void *B_d,
                              void *C_d,
                              void *workspace_d,
                              const MatrixMultiplyKernelInfo &kernelInfo,
                              void *transposeBuffer_d) {
    if (kernelInfo.CElementOrder == ElementOrder::CPP_ROW_MAJOR) {
        if (transposeBuffer_d == NULL) {
            printf(
                "ERROR: You specified C++ element order in the C matrix, so you must provide a transpose buffer as an "
                "argument to this function. Its size must be the same size as C.");
            assert(transposeBuffer_d != NULL);
        }
    }

    const float alpha = 1.0;
    const float beta = 0.0;
    cublasStatus_t cublasStatus;
    void *result_d;
    if (kernelInfo.CElementOrder == ElementOrder::CPP_ROW_MAJOR) {
        result_d = transposeBuffer_d;
    } else {
        result_d = C_d;
    }
    cublasStatus = cublasLtMatmul(cublasLtHandle,
                                  kernelInfo.operationDescriptor,
                                  &alpha,
                                  A_d,
                                  kernelInfo.ADescriptor,
                                  B_d,
                                  kernelInfo.BDescriptor,
                                  &beta,
                                  result_d,
                                  kernelInfo.CDescriptor,
                                  result_d,
                                  kernelInfo.CDescriptor,
                                  &kernelInfo.algorithm,
                                  workspace_d,
                                  kernelInfo.workspaceSizeInBytes,
                                  stream);
    if (kernelInfo.CElementOrder == ElementOrder::CPP_ROW_MAJOR) {
        if (kernelInfo.dataType == DataType::FP32) {
            matrixTranspose((float *)C_d,
                            (float *)transposeBuffer_d,
                            kernelInfo.CCols,
                            kernelInfo.CRows,
                            stream);  // The source matrix is the transpose of C, so rows and cols are switched
        } else {
            matrixTranspose((half *)C_d,
                            (half *)transposeBuffer_d,
                            kernelInfo.CCols,
                            kernelInfo.CRows,
                            stream);  // The source matrix is the transpose of C, so rows and cols are switched
        }
    }
    return cublasStatus;
}

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(const customMatmulPerf_t &perf) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;

    const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);

    printf(
        "algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d} status %d "
        "time %f workspace=%d mathMode=%d waves=%f\n",
        algoId,
        tile,
        matmulTileName[tile],
        numSplitsK,
        reductionScheme,
        swizzle,
        customOption,
        perf.status,
        perf.time,
        (int)perf.workspaceSize,
        (int)perf.mathMode,
        perf.wavesCount);
}

static inline bool time_compare(const customMatmulPerf_t &perf_a, const customMatmulPerf_t &perf_b) {
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static inline bool info_time_compare(const MatrixMultiplyKernelInfo &a, const MatrixMultiplyKernelInfo &b) {
    return ((a.status == CUBLAS_STATUS_SUCCESS) && (a.time < b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle,  // to get the capabilities (required a GPU)
                                      cublasLtMatmulDesc_t operationDesc,
                                      const void *alpha, /* host or device pointer */
                                      const void *A,
                                      cublasLtMatrixLayout_t Adesc,
                                      const void *B,
                                      cublasLtMatrixLayout_t Bdesc,
                                      const void *beta, /* host or device pointer */
                                      const void *C,
                                      cublasLtMatrixLayout_t Cdesc,
                                      void *D,
                                      cublasLtMatrixLayout_t Ddesc,
                                      const cublasLtMatmulAlgo_t &algo,
                                      int kernelRepeats,
                                      void *workSpace,
                                      size_t workSpaceSizeInBytes,
                                      customMatmulPerf_t &perfResults,
                                      cudaStream_t stream,
                                      cudaEvent_t &startEvent,
                                      cudaEvent_t &stopEvent) {
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int repeats = kernelRepeats;

    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            cudaError_t err, err1, err2, err3;
            err = cudaEventRecord(startEvent, stream);
            for (int loop = 0; loop < repeats; loop++) {
                cublasStatus_t oneRunStatus = cublasLtMatmul(ltHandle,
                                                             operationDesc,
                                                             alpha,
                                                             A,
                                                             Adesc,
                                                             B,
                                                             Bdesc,
                                                             beta,
                                                             C,
                                                             Cdesc,
                                                             D,
                                                             Ddesc,
                                                             &algo,
                                                             workSpace,
                                                             workSpaceSizeInBytes,
                                                             stream);
                // matrixTranspose(C, C, 256, 256, false, stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord(stopEvent, stream);
            err2 = cudaEventSynchronize(stopEvent);
            float time;
            err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
            if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                perfResults.algo = algo;
                perfResults.time = time;
                perfResults.workspaceSize = heurResult.workspaceSize;
                perfResults.wavesCount = heurResult.wavesCount;
            }
        } else {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
        }
    }

    return algoStatus;
}

MatrixMultiplyKernelInfo LtSgemmCustomFind(cublasLtHandle_t ltHandle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m,
                                           int n,
                                           int k,
                                           const float *alpha, /* host pointer */
                                           const float *A,
                                           int lda,
                                           const float *B,
                                           int ldb,
                                           const float *beta, /* host pointer */
                                           float *C,
                                           int ldc,
                                           void *workSpace,
                                           size_t workSpaceSize,
                                           DataType dataType,
                                           ElementOrder AElementOrder,
                                           ElementOrder BElementOrder,
                                           ElementOrder CElementOrder,
                                           bool printResults) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    // FIXME: don't constraint the user like this. Instead I will use my tensor core kernel for all fp16 and smaller
    // and this function will only be used to find the best MM kernel for fp32. I should disallow tensor cores here then,
    // since that would require down conversion and that case is covered by my tensor core kernel.
    //
    // NOTE: VERY IMPORTANT! YOU MUST USE ALL DIMENSIONS AS MULTIPLES OF 8 AND 0 PAD THE EDGES
    // This will use tensor cores which are many times faster than regular floating point units
    assert(m % 8 == 0);
    assert(n % 8 == 0);
    assert(k % 8 == 0);

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    cudaStream_t stream = NULL;
    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    // Let try a fixed number of combinations
#define ALGO_COMBINATIONS 5000
    int AlgoCombinations = ALGO_COMBINATIONS;
    int AlgoCount = 0;
    int kernelRepeats = 10;  // number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    MatrixMultiplyKernelInfo kernelInfo[ALGO_COMBINATIONS];
    int nbAlgoIds = 0;
#define ALGO_IDS 100
    int algoIdA[ALGO_IDS];
    cudaDataType_t precision = dataType == DataType::FP32 ? CUDA_R_32F : CUDA_R_16F;
    cublasComputeType_t computeType = dataType == DataType::FP32 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;
    cudaDataType_t scaleType = precision, Atype = precision, Btype = precision, Ctype = precision;
    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    status = cublasLtMatmulDescCreate(&operationDesc, computeType, precision);
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;
    status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;
    status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;

    // Create matrix descriptors. We are good with the details here so no need to set any extra attributes
    status = cublasLtMatrixLayoutCreate(&Adesc, precision, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;
    status = cublasLtMatrixLayoutCreate(&Bdesc, precision, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;

    status = cublasLtMatrixLayoutCreate(&Cdesc, precision, m, n, ldc);
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType = Atype = Btype = Ctype = Dtype =
    // precision)
    status = cublasLtMatmulAlgoGetIds(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS)
        goto CLEANUP;

    // Create CUDA event to time the execution time of each algo
    if (cudaEventCreate(&startEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }
    if (cudaEventCreate(&stopEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int nbTiles = int(sizeWritten / sizeof(int));
        int *tileA = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);

        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++) {
                cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                /* Loop over the CTAs swizzling support */
                for (int k = 0; k <= swizzlingMax; k++) {
                    int splitK_trial = 0;
                    if (splitkSupport) {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in addtion to the case where
                    // splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                        /* Setup attribute of the algo to run */
                        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                        int splitK_val = 0;
                        int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));

                        if (l > 0) {  // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1; redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations);
                                 redScheme = redScheme << 1) {
                                if (redScheme & redMask) {
                                    cublasLtMatmulAlgoConfigSetAttribute(
                                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme));

                                    status = customMatmulRun(ltHandle,
                                                             operationDesc,
                                                             alpha, /* host or device pointer */
                                                             A,
                                                             Adesc,
                                                             B,
                                                             Bdesc,
                                                             beta, /* host or device pointer */
                                                             C,
                                                             Cdesc,
                                                             C,
                                                             Cdesc,
                                                             algo,
                                                             kernelRepeats,
                                                             workSpace,
                                                             workSpaceSize,
                                                             perfResults[AlgoCount],
                                                             stream,
                                                             startEvent,
                                                             stopEvent);
                                    perfResults[AlgoCount].status = status;

                                    kernelInfo[AlgoCount].operationDescriptor = operationDesc;
                                    kernelInfo[AlgoCount].algorithm = algo;
                                    kernelInfo[AlgoCount].ADescriptor = Adesc;
                                    kernelInfo[AlgoCount].BDescriptor = Bdesc;
                                    kernelInfo[AlgoCount].CDescriptor = Cdesc;
                                    kernelInfo[AlgoCount].workspaceSizeInBytes = workSpaceSize;
                                    kernelInfo[AlgoCount].time = perfResults[AlgoCount].time;
                                    kernelInfo[AlgoCount].status = status;
                                    kernelInfo[AlgoCount].CRows = m;
                                    kernelInfo[AlgoCount].CCols = n;
                                    kernelInfo[AlgoCount].dataType = dataType;
                                    kernelInfo[AlgoCount].AElementOrder = AElementOrder;
                                    kernelInfo[AlgoCount].BElementOrder = BElementOrder;
                                    kernelInfo[AlgoCount].CElementOrder = CElementOrder;

                                    if (status == CUBLAS_STATUS_SUCCESS)
                                        AlgoCount++;

                                }  // end if
                            }      // end for
                        } else {   // Non-splitK case
                            /* if user preference is ok with workspace */
                            if (AlgoCount < AlgoCombinations) {
                                status = customMatmulRun(ltHandle,
                                                         operationDesc,
                                                         alpha, /* host or device pointer */
                                                         A,
                                                         Adesc,
                                                         B,
                                                         Bdesc,
                                                         beta, /* host or device pointer */
                                                         C,
                                                         Cdesc,
                                                         C,
                                                         Cdesc,
                                                         algo,
                                                         kernelRepeats,
                                                         workSpace,
                                                         workSpaceSize,
                                                         perfResults[AlgoCount],
                                                         stream,
                                                         startEvent,
                                                         stopEvent);
                                perfResults[AlgoCount].status = status;

                                kernelInfo[AlgoCount].operationDescriptor = operationDesc;
                                kernelInfo[AlgoCount].algorithm = algo;
                                kernelInfo[AlgoCount].ADescriptor = Adesc;
                                kernelInfo[AlgoCount].BDescriptor = Bdesc;
                                kernelInfo[AlgoCount].CDescriptor = Cdesc;
                                kernelInfo[AlgoCount].workspaceSizeInBytes = workSpaceSize;
                                kernelInfo[AlgoCount].time = perfResults[AlgoCount].time;
                                kernelInfo[AlgoCount].status = status;
                                kernelInfo[AlgoCount].CRows = m;
                                kernelInfo[AlgoCount].CCols = n;
                                kernelInfo[AlgoCount].dataType = dataType;
                                kernelInfo[AlgoCount].AElementOrder = AElementOrder;
                                kernelInfo[AlgoCount].BElementOrder = BElementOrder;
                                kernelInfo[AlgoCount].CElementOrder = CElementOrder;

                                if (status == CUBLAS_STATUS_SUCCESS)
                                    AlgoCount++;
                            }
                        }
                    }  // end l
                }      // end k
            }          // end customOption
        }              // end tileIdx
        delete[] tileA;
    }  // end idx

    std::sort(kernelInfo, kernelInfo + AlgoCount, info_time_compare);

    // Sort the results per run duration
    if (printResults) {
        std::sort(perfResults, perfResults + AlgoCount, time_compare);

        // Print timing and perf details
        for (int i = 0; i < AlgoCount; i++) {
            printf("result %03d : ", i);
            printPerfStructure(perfResults[i]);
            printf("%lf TFLOPS\n", ((2.0 * k - 1.0) * m * n * kernelRepeats) / (perfResults[i].time * 1.0e9));
            assert(kernelInfo[0].time == perfResults[0].time);
        }
    }

CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    // if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    // if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    // if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    // if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    if (startEvent)
        cudaEventDestroy(startEvent);
    if (stopEvent)
        cudaEventDestroy(stopEvent);
    assert(AlgoCount > 0);
    assert(kernelInfo[0].status == CUBLAS_STATUS_SUCCESS);
    return kernelInfo[0];
}

float randRange(float min, float max) { return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min))); }

// Find the fastest matrix multiple kernel for the following equation:
// C_mxn = A_mxk * B_kxn
MatrixMultiplyKernelInfo getBestGemmKernel(unsigned int m,
                                           unsigned int n,
                                           unsigned int k,
                                           int deviceNum,
                                           DataType dataType,
                                           ElementOrder AElementOrder,
                                           ElementOrder BElementOrder,
                                           ElementOrder CElementOrder,
                                           bool printResults) {
    assert(dataType != DataType::FP16);  // I have not gotten that to work.

    cublasStatus_t cublasStatus;
    cudaError_t cudaStatus;
    cublasLtHandle_t cublasLtHandle;
    cudaStatus = cudaSetDevice(deviceNum);
    assert(cudaStatus == cudaSuccess);
    cublasStatus = cublasLtCreate(&cublasLtHandle);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    float alpha = 1.0f;
    float beta = 0.0f;

    float *A_d;
    cudaStatus = cudaMalloc(&A_d, m * k * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    float *B_d;
    cudaStatus = cudaMalloc(&B_d, k * n * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    float *C_d;
    cudaStatus = cudaMalloc(&C_d, m * n * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    float *workspace_d;
    cudaStatus = cudaMalloc(&workspace_d, m * n * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    MatrixMultiplyKernelInfo bestKernelInfo = LtSgemmCustomFind(cublasLtHandle,
                                                                AElementOrder == ElementOrder::CPP_ROW_MAJOR ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                                BElementOrder == ElementOrder::CPP_ROW_MAJOR ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                                m,
                                                                n,
                                                                k,
                                                                &alpha,
                                                                A_d,
                                                                k,
                                                                B_d,
                                                                n,
                                                                &beta,
                                                                C_d,
                                                                m,
                                                                workspace_d,
                                                                m * k * (sizeof(float) / 4),
                                                                dataType,
                                                                AElementOrder,
                                                                BElementOrder,
                                                                CElementOrder,
                                                                printResults);

    // typedef enum{
    //    CUBLAS_STATUS_SUCCESS         =0,
    //    CUBLAS_STATUS_NOT_INITIALIZED =1,
    //    CUBLAS_STATUS_ALLOC_FAILED    =3,
    //    CUBLAS_STATUS_INVALID_VALUE   =7,
    //    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    //    CUBLAS_STATUS_MAPPING_ERROR   =11,
    //    CUBLAS_STATUS_EXECUTION_FAILED=13,
    //    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    //    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    //    CUBLAS_STATUS_LICENSE_ERROR   =16
    //} cublasStatus_t;

    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(A_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(B_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(C_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(workspace_d);
    assert(cudaStatus == cudaSuccess);

    return bestKernelInfo;
}
