#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

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

using namespace ThorImplementation;
using namespace std;

const float CublasMatrixMultiply::ALPHA_NO_SCALE = 1.0f;
const float CublasMatrixMultiply::ALPHA_NEGATE = -1.0f;
const float CublasMatrixMultiply::BETA_ACCUMULATE = 1.0f;
const float CublasMatrixMultiply::BETA_CLEAR = 0.0f;

// This variant allows non-packed matrices
void CublasMatrixMultiply::multiply(Tensor A,
                                    Tensor B,
                                    Tensor C,
                                    Optional<Tensor> workspace,
                                    const int32_t A_rows,
                                    const int32_t A_cols,
                                    const int32_t B_rows,
                                    const int32_t B_cols,
                                    bool transposeA,
                                    bool transposeB,
                                    const bool accumulate,
                                    const bool negate,
                                    const TensorDescriptor::DataType ABCDataType,
                                    Stream stream) {
    float alpha = negate ? ALPHA_NEGATE : ALPHA_NO_SCALE;
    float beta = accumulate ? BETA_ACCUMULATE : BETA_CLEAR;
    gemm(A, B, C, C, workspace, A_rows, A_cols, B_rows, B_cols, transposeA, transposeB, false, alpha, beta, ABCDataType, stream);
}

void CublasMatrixMultiply::gemm(Tensor A,
                                Tensor B,
                                Tensor C,
                                Tensor D,
                                Optional<Tensor> workspace,
                                const int32_t A_rows,
                                const int32_t A_cols,
                                const int32_t B_rows,
                                const int32_t B_cols,
                                bool transposeA,
                                bool transposeB,
                                bool transposeC,
                                float alpha,
                                float beta,
                                const TensorDescriptor::DataType ABCDDataType,
                                Stream stream) {
    assert(A.getDimensions().size() == 2);
    assert(B.getDimensions().size() == 2);
    assert(C.getDimensions().size() == 2);
    assert(D.getDimensions().size() == 2);
    const int32_t ld_A = A.getDimensions()[1];
    const int32_t ld_B = B.getDimensions()[1];
    const int32_t ld_C = C.getDimensions()[1];
    const int32_t ld_D = D.getDimensions()[1];

    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    int32_t D_rows = C_rows;
    int32_t D_cols = C_cols;

    assert(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    assert(!(C == D && transposeC));

    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_rows > 0);
    assert(B_cols > 0);
    assert(ld_A >= A_cols);
    assert(ld_B >= B_cols);
    assert(ld_C >= C_cols);
    assert(ld_D >= D_cols);
    // Check dataType of tensors
    assert(A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 ||
           A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(A.getDescriptor().getDataType() == B.getDescriptor().getDataType());
    assert(A.getDescriptor().getDataType() == C.getDescriptor().getDataType());
    assert(A.getDescriptor().getDataType() == D.getDescriptor().getDataType());
    // Check dimensions of tensors
    vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
    vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
    vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
    vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
    assert(ADimensions.size() == 2);
    assert(ADimensions[0] == (uint32_t)A_rows);
    assert(ADimensions[1] == (uint32_t)ld_A);
    assert(BDimensions[0] == (uint32_t)B_rows);
    assert(BDimensions[1] == (uint32_t)ld_B);
    assert(CDimensions[0] == (uint32_t)C_rows);
    assert(CDimensions[1] == (uint32_t)ld_C);
    assert(DDimensions[0] == (uint32_t)D_rows);
    assert(DDimensions[1] == (uint32_t)ld_D);

    int gpuNum = stream.getGpuNum();
    ScopedGpu scopedGpu(gpuNum);

    KernelRequirement kernelRequirement(MachineEvaluator::instance().getGpuType(gpuNum),
                                        A_rows,
                                        A_cols,
                                        B_rows,
                                        B_cols,
                                        transposeA,
                                        transposeB,
                                        transposeC,
                                        ld_A,
                                        ld_B,
                                        ld_C,
                                        ld_D,
                                        workspace.isPresent());

    cudaDataType_t ABCDDataTypeCuda = mapToCublasDataType(ABCDDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDDataTypeCuda, ABCDDataTypeCuda, ABCDDataTypeCuda, ABCDDataTypeCuda);

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    CublasMatrixMultiply::instance().mtx.lock();
    auto it = optimalKernels.find(cublasKernelRequirement);
    assert(it != optimalKernels.end());
    CublasKernel cublasKernel = it->second;
    CublasMatrixMultiply::instance().mtx.unlock();

    // Check byte size of workspace
    if (workspace.isPresent()) {
        bool kernelWillRunOnGpu;
        size_t workspaceSizeInBytes = cublasKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu);
        assert(kernelWillRunOnGpu);

        if (workspaceSizeInBytes > 0)
            assert(cublasKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu) <=
                   workspace.get().getDescriptor().getArraySizeInBytes());
    }

    cublasKernel.executeKernel(A, B, C, D, ld_A, ld_B, ld_C, ld_D, workspace, alpha, beta, stream);
}

cudaDataType_t CublasMatrixMultiply::mapToCublasDataType(TensorDescriptor::DataType dataType) {
    if (dataType == TensorDescriptor::DataType::FP32)
        return CUDA_R_32F;
    else if (dataType == TensorDescriptor::DataType::FP16)
        return CUDA_R_16F;
    else
        assert(false);
}

// FIXME: This one now just calls gemmUsingH...
void CublasMatrixMultiply::multiplyUsingHeuristicKernelChoice(Tensor A,
                                                              Tensor B,
                                                              Tensor C,
                                                              const int32_t A_rows,
                                                              const int32_t A_cols,
                                                              const int32_t B_rows,
                                                              const int32_t B_cols,
                                                              bool transposeA,
                                                              bool transposeB,
                                                              const bool accumulate,
                                                              const bool negate,
                                                              const TensorDescriptor::DataType ABCDataType,
                                                              Stream stream) {
    float alpha = negate ? ALPHA_NEGATE : ALPHA_NO_SCALE;
    float beta = accumulate ? BETA_ACCUMULATE : BETA_CLEAR;
    gemmUsingHeuristicKernelChoice(
        A, B, C, C, A_rows, A_cols, B_rows, B_cols, transposeA, transposeB, false, alpha, beta, ABCDataType, stream);
}

void CublasMatrixMultiply::gemmUsingHeuristicKernelChoice(
    Tensor A,
    Tensor B,
    Tensor C,
    Tensor D,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    // Leading dimension of A, i.e. number of elements (not bytes) that separate the beginning of two adjacent rows in
    // memory. Some slots at the end of a row may be unused.
    bool transposeA,
    bool transposeB,
    bool transposeC,
    float alpha,
    float beta,
    const TensorDescriptor::DataType ABCDDataType,
    Stream stream) {
    assert(A.getDimensions().size() == 2);
    assert(B.getDimensions().size() == 2);
    assert(C.getDimensions().size() == 2);
    assert(D.getDimensions().size() == 2);
    const int32_t ld_A = A.getDimensions()[1];
    const int32_t ld_B = B.getDimensions()[1];
    const int32_t ld_C = C.getDimensions()[1];
    const int32_t ld_D = D.getDimensions()[1];

    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    int32_t D_rows = C_rows;
    int32_t D_cols = C_cols;

    assert(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    assert(!(C == D && transposeC));

    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_rows > 0);
    assert(B_cols > 0);
    assert(ld_A >= A_cols);
    assert(ld_B >= B_cols);
    assert(ld_C >= C_cols);
    assert(ld_D >= D_cols);
    // Check dataType of tensors
    assert(A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 ||
           A.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);
    assert(A.getDescriptor().getDataType() == B.getDescriptor().getDataType());
    assert(A.getDescriptor().getDataType() == C.getDescriptor().getDataType());
    assert(A.getDescriptor().getDataType() == D.getDescriptor().getDataType());
    // Check dimensions of tensors
    vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
    vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
    vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
    vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
    assert(ADimensions.size() == 2);
    assert(ADimensions[0] == (uint32_t)A_rows);
    assert(ADimensions[1] == (uint32_t)ld_A);
    assert(BDimensions[0] == (uint32_t)B_rows);
    assert(BDimensions[1] == (uint32_t)ld_B);
    assert(CDimensions[0] == (uint32_t)C_rows);
    assert(CDimensions[1] == (uint32_t)ld_C);
    assert(DDimensions[0] == (uint32_t)D_rows);
    assert(DDimensions[1] == (uint32_t)ld_D);

    ScopedGpu scopedGpu(stream.getGpuNum());

    cublasStatus_t cublasStatus;

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ADesc;
    cublasLtMatrixLayout_t BDesc;
    cublasLtMatrixLayout_t CDesc;
    cublasLtMatrixLayout_t DDesc;

    cublasStatus = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
    const cublasLtPointerMode_t hostPointerMode = CUBLASLT_POINTER_MODE_HOST;
    cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, pointerModeAttribute, &hostPointerMode, sizeof(hostPointerMode));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    if (transposeA) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }
    if (transposeB) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }
    if (transposeC) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transpose, sizeof(transpose));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
    cudaDataType_t ABCDDataTypeCuda = mapToCublasDataType(ABCDDataType);

    cublasStatus = cublasLtMatrixLayoutCreate(&ADesc, ABCDDataTypeCuda, A_rows, A_cols, ld_A);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    int64_t ld = ld_A;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    cublasStatus = cublasLtMatrixLayoutCreate(&BDesc, ABCDDataTypeCuda, B_rows, B_cols, ld_B);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    ld = ld_B;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    cublasStatus = cublasLtMatrixLayoutCreate(&CDesc, ABCDDataTypeCuda, C_rows, C_cols, ld_C);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    ld = ld_C;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    cublasStatus = cublasLtMatrixLayoutCreate(&DDesc, ABCDDataTypeCuda, D_rows, D_cols, ld_D);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    ld = ld_D;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    KernelRequirement kernelRequirement(MachineEvaluator::instance().getGpuType(stream.getGpuNum()),
                                        A_rows,
                                        A_cols,
                                        B_rows,
                                        B_cols,
                                        transposeA,
                                        transposeB,
                                        transposeC,
                                        ld_A,
                                        ld_B,
                                        ld_C,
                                        ld_D,
                                        false);

    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDDataTypeCuda, ABCDDataTypeCuda, ABCDDataTypeCuda, ABCDDataTypeCuda);
    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    // If there is already a known kernel, use it. Otherwise a heuristic search will be performed and the kernel remembered.
    CublasMatrixMultiply::instance().mtx.lock();
    if (optimalKernels.count(cublasKernelRequirement) == 1) {
        cublasLtMatmulAlgo_t algorithm = optimalKernels[cublasKernelRequirement].getAlgorithm(stream.getGpuNum());
        CublasMatrixMultiply::instance().mtx.unlock();

        cublasStatus = cublasLtMatmul(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
                                      operationDesc,
                                      &alpha,
                                      A.getMemPtr(),
                                      ADesc,
                                      B.getMemPtr(),
                                      BDesc,
                                      &beta,
                                      C.getMemPtr(),
                                      CDesc,
                                      D.getMemPtr(),
                                      DDesc,
                                      &algorithm,
                                      nullptr,
                                      0,
                                      stream);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        return;
    } else if (knownHeuristicAlgorithms.count(cublasKernelRequirement) == 1) {
        cublasLtMatmulAlgo_t algorithm = knownHeuristicAlgorithms[cublasKernelRequirement];
        CublasMatrixMultiply::instance().mtx.unlock();

        cublasStatus = cublasLtMatmul(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
                                      operationDesc,
                                      &alpha,
                                      A.getMemPtr(),
                                      ADesc,
                                      B.getMemPtr(),
                                      BDesc,
                                      &beta,
                                      C.getMemPtr(),
                                      CDesc,
                                      D.getMemPtr(),
                                      DDesc,
                                      &algorithm,
                                      nullptr,
                                      0,
                                      stream);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        return;
    }
    CublasMatrixMultiply::instance().mtx.unlock();

    cublasLtMatmulPreference_t searchPreferences;
    cublasStatus = cublasLtMatmulPreferenceCreate(&searchPreferences);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatmulPreferenceInit(searchPreferences);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    // cublasLtMatmulPreferenceAttributes_t attribute = CUBLASLT_MATMUL_PREF_IMPL_MASK;
    // cublasLtNumericalImplFlags_t computeType = CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK |
    // CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F; cublasStatus = cublasLtMatmulPreferenceSetAttribute(searchPreferences, attribute,
    // &computeType, sizeof(computeType)); assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    int returnedAlgoCount;
    vector<cublasLtMatmulHeuristicResult_t> results(30);
    cublasStatus = cublasLtMatmulAlgoGetHeuristic(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
                                                  operationDesc,
                                                  ADesc,
                                                  BDesc,
                                                  CDesc,
                                                  DDesc,
                                                  searchPreferences,
                                                  30,
                                                  results.data(),
                                                  &returnedAlgoCount);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    results.resize(returnedAlgoCount);

    // Algorithms aren't guaranteed to run, so find the first one that does and then return.
    bool kernelLaunchedSuccessfully = false;
    for (int i = 0; i < returnedAlgoCount && !kernelLaunchedSuccessfully; ++i) {
        // have seen kernels that say wavesCount == 0 that sporadically fail.
        if (!(results[i].wavesCount > 0.0f))
            continue;
        cublasStatus = cublasLtMatmul(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
                                      operationDesc,
                                      &alpha,
                                      A.getMemPtr(),
                                      ADesc,
                                      B.getMemPtr(),
                                      BDesc,
                                      &beta,
                                      C.getMemPtr(),
                                      CDesc,
                                      D.getMemPtr(),
                                      DDesc,
                                      &(results[i].algo),
                                      nullptr,
                                      0,
                                      stream);
        if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
            kernelLaunchedSuccessfully = true;
            CublasMatrixMultiply::instance().mtx.lock();
            knownHeuristicAlgorithms[cublasKernelRequirement] = results[i].algo;
            CublasMatrixMultiply::instance().mtx.unlock();
        }
    }

    if (!kernelLaunchedSuccessfully) {
        results = vector<cublasLtMatmulHeuristicResult_t>(10000);
        cublasStatus = cublasLtMatmulAlgoGetHeuristic(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
                                                      operationDesc,
                                                      ADesc,
                                                      BDesc,
                                                      CDesc,
                                                      DDesc,
                                                      searchPreferences,
                                                      10000,
                                                      results.data(),
                                                      &returnedAlgoCount);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        results.resize(returnedAlgoCount);

        for (int i = 0; i < returnedAlgoCount && !kernelLaunchedSuccessfully; ++i) {
            cublasStatus = cublasLtMatmul(MachineEvaluator::instance().getCublasLtHandle(stream.getGpuNum()),
                                          operationDesc,
                                          &alpha,
                                          A.getMemPtr(),
                                          ADesc,
                                          B.getMemPtr(),
                                          BDesc,
                                          &beta,
                                          C.getMemPtr(),
                                          CDesc,
                                          D.getMemPtr(),
                                          DDesc,
                                          &(results[i].algo),
                                          nullptr,
                                          0,
                                          stream);
            if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
                kernelLaunchedSuccessfully = true;
                // FIXME: may need an algo per gpu
                // FIXME: add their top N heuristic kernels to the set of optimal candidates, will need to change CublasKernel to take
                // algorithm in constructor
                CublasMatrixMultiply::instance().mtx.lock();
                knownHeuristicAlgorithms[cublasKernelRequirement] = results[i].algo;
                CublasMatrixMultiply::instance().mtx.unlock();
            }
        }
    }

    assert(kernelLaunchedSuccessfully);

    cublasStatus = cublasLtMatmulPreferenceDestroy(searchPreferences);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(DDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(CDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(BDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(ADesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatmulDescDestroy(operationDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
}

vector<CublasKernel> CublasMatrixMultiply::getHeuristicGemmKernels(const int32_t numChoices,
                                                                   const int32_t gpuNum,
                                                                   const int32_t A_rows,
                                                                   const int32_t A_cols,
                                                                   const int32_t B_rows,
                                                                   const int32_t B_cols,
                                                                   const bool transposeA,
                                                                   const bool transposeB,
                                                                   const bool transposeC,
                                                                   const int32_t ld_A,
                                                                   const int32_t ld_B,
                                                                   const int32_t ld_C,
                                                                   const int32_t ld_D,
                                                                   const uint64_t maxWorkspaceSize,
                                                                   const float maxWaves,
                                                                   const TensorDescriptor::DataType ABCDDataType) {
    ScopedGpu scopedGpu(gpuNum);
    cublasStatus_t cublasStatus;

    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    int32_t D_rows = C_rows;
    int32_t D_cols = C_cols;

    // Remember leading dimension refers to the stride between a consecutive series in memory.
    // In c++, each column is laid out consecutively for a row and then afterward the next row begins,
    // so the leading dimension is the number of columns.
    //    const int32_t ld_A = A_cols;
    //    const int32_t ld_B = B_cols;
    //    const int32_t ld_C = C_cols;
    //    const int32_t ld_D = D_cols;

    assert(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    assert(A_rows > 0);
    assert(A_cols > 0);
    assert(B_rows > 0);
    assert(B_cols > 0);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ADesc;
    cublasLtMatrixLayout_t BDesc;
    cublasLtMatrixLayout_t CDesc;
    cublasLtMatrixLayout_t DDesc;

    cublasStatus = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
    const cublasLtPointerMode_t hostPointerMode = CUBLASLT_POINTER_MODE_HOST;
    cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, pointerModeAttribute, &hostPointerMode, sizeof(hostPointerMode));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    if (transposeA) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }
    if (transposeB) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }
    if (transposeC) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        cublasStatus = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transpose, sizeof(transpose));
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
    cudaDataType_t ABCDDataTypeCuda = mapToCublasDataType(ABCDDataType);

    cublasStatus = cublasLtMatrixLayoutCreate(&ADesc, ABCDDataTypeCuda, A_rows, A_cols, ld_A);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    int64_t ld = ld_A;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    cublasStatus = cublasLtMatrixLayoutCreate(&BDesc, ABCDDataTypeCuda, B_rows, B_cols, ld_B);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    ld = ld_B;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    cublasStatus = cublasLtMatrixLayoutCreate(&CDesc, ABCDDataTypeCuda, C_rows, C_cols, ld_C);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    ld = ld_C;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    cublasStatus = cublasLtMatrixLayoutCreate(&DDesc, ABCDDataTypeCuda, D_rows, D_cols, ld_D);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutSetAttribute(DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    ld = ld_D;
    cublasStatus = cublasLtMatrixLayoutSetAttribute(DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    KernelRequirement kernelRequirement(MachineEvaluator::instance().getGpuType(gpuNum),
                                        A_rows,
                                        A_cols,
                                        B_rows,
                                        B_cols,
                                        transposeA,
                                        transposeB,
                                        transposeC,
                                        ld_A,
                                        ld_B,
                                        ld_C,
                                        ld_D,
                                        false);

    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDDataTypeCuda, ABCDDataTypeCuda, ABCDDataTypeCuda, ABCDDataTypeCuda);
    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    cublasLtMatmulPreference_t searchPreferences;
    cublasStatus = cublasLtMatmulPreferenceCreate(&searchPreferences);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatmulPreferenceInit(searchPreferences);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatmulPreferenceSetAttribute(
        searchPreferences, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspaceSize, sizeof(maxWorkspaceSize));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus =
        cublasLtMatmulPreferenceSetAttribute(searchPreferences, CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT, &maxWaves, sizeof(maxWaves));
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    int returnedAlgoCount;
    vector<cublasLtMatmulHeuristicResult_t> rawResults(numChoices);
    cublasStatus = cublasLtMatmulAlgoGetHeuristic(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                                  operationDesc,
                                                  ADesc,
                                                  BDesc,
                                                  CDesc,
                                                  DDesc,
                                                  searchPreferences,
                                                  numChoices,
                                                  rawResults.data(),
                                                  &returnedAlgoCount);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    rawResults.resize(returnedAlgoCount);

    vector<CublasKernel> results;
    string gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    for (int32_t i = 0; i < returnedAlgoCount; ++i) {
        if (rawResults[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        // It would be nice if cublasLt only returned algos that are supported by the GPU, but that is not the case.
        cublasLtMatmulHeuristicResult_t result;
        cublasStatus = cublasLtMatmulAlgoCheck(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                               operationDesc,
                                               ADesc,
                                               BDesc,
                                               CDesc,
                                               DDesc,
                                               &rawResults[i].algo,
                                               &result);
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        size_t sizeWritten;
        int32_t algoId = 0;
        cublasStatus =
            cublasLtMatmulAlgoConfigGetAttribute(&rawResults[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint32_t tileId = 0;
        cublasStatus =
            cublasLtMatmulAlgoConfigGetAttribute(&rawResults[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint32_t splitK = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint32_t reductionFlag = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionFlag, sizeof(reductionFlag), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint32_t swizzleType = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzleType, sizeof(swizzleType), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint32_t customOptionValue = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOptionValue, sizeof(customOptionValue), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint32_t stagesId = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesId, sizeof(stagesId), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint16_t innerShapeId = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &innerShapeId, sizeof(innerShapeId), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        uint16_t clusterShapeId = 0;
        cublasStatus = cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &clusterShapeId, sizeof(clusterShapeId), &sizeWritten);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

        CublasKernelOptions cublasKernelOptions(rawResults[i].algo,
                                                algoId,
                                                static_cast<cublasLtMatmulTile_t>(tileId),
                                                splitK,
                                                reductionFlag,
                                                swizzleType,
                                                customOptionValue,
                                                stagesId,
                                                innerShapeId,
                                                clusterShapeId,
                                                rawResults[i].workspaceSize,
                                                rawResults[i].wavesCount);
        CublasKernel cublasKernel(cublasKernelRequirement, cublasKernelOptions, gpuType);
        results.push_back(cublasKernel);
    }

    cublasStatus = cublasLtMatmulPreferenceDestroy(searchPreferences);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(DDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(CDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(BDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatrixLayoutDestroy(ADesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    cublasStatus = cublasLtMatmulDescDestroy(operationDesc);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);

    return results;
}

long minl(long a, long b) { return a < b ? a : b; }

long maxl(long a, long b) { return a > b ? a : b; }

void CublasMatrixMultiply::chooseOptimalGemmKernel(int gpuNum,
                                                   int rowsA,
                                                   int colsA,
                                                   int rowsB,
                                                   int colsB,
                                                   int ldA,
                                                   int ldB,
                                                   int ldC,
                                                   int ldD,
                                                   bool transposeA,
                                                   bool transposeB,
                                                   bool transposeC,
                                                   TensorDescriptor::DataType ABCDataType,
                                                   bool printResults) {
    bool bestKernelHasWorkspace = chooseOptimalGemmKernel(
        gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, transposeC, ABCDataType, true, printResults);

    // If the best kernel did not have a workspace, then it will be used for the no workspace version of the computation also
    if (bestKernelHasWorkspace) {
        chooseOptimalGemmKernel(
            gpuNum, rowsA, colsA, rowsB, colsB, ldA, ldB, ldC, ldD, transposeA, transposeB, transposeC, ABCDataType, false, printResults);

        // The heuristic that is being used to choose kernels tries to get a kernel as close as possible to 1 wave,
        // for smaller operations this can be achieved by adding a workspace, but this doesn't create a faster kernel.
        // So in this case sometimes forcing there to be no workspace and choosing from the remaining kernels,
        // the kernels with the closest to 1 wave is a better heuristic. Because of this, check if the no workspace
        // kernel is faster than the workspace one, and if so replace the workspace version with the no workspace version.
        KernelRequirement kernelRequirementNoWorkspace(MachineEvaluator::instance().getGpuType(gpuNum),
                                                       rowsA,
                                                       colsA,
                                                       rowsB,
                                                       colsB,
                                                       transposeA,
                                                       transposeB,
                                                       transposeC,
                                                       ldA,
                                                       ldB,
                                                       ldC,
                                                       ldD,
                                                       false);
        cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
        OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);
        CublasKernelRequirement noWorkspaceCublasKernelRequirement(kernelRequirementNoWorkspace, operationType);
        KernelRequirement kernelRequirementWithWorkspace(MachineEvaluator::instance().getGpuType(gpuNum),
                                                         rowsA,
                                                         colsA,
                                                         rowsB,
                                                         colsB,
                                                         transposeA,
                                                         transposeB,
                                                         transposeC,
                                                         ldA,
                                                         ldB,
                                                         ldC,
                                                         ldD,
                                                         true);
        CublasKernelRequirement workspaceCublasKernelRequirement(kernelRequirementWithWorkspace, operationType);
        CublasMatrixMultiply::instance().mtx.lock();
        if (optimalKernels[noWorkspaceCublasKernelRequirement].getAverageRunTimeMilliseconds() <
            optimalKernels[workspaceCublasKernelRequirement].getAverageRunTimeMilliseconds())
            optimalKernels[workspaceCublasKernelRequirement] = optimalKernels[noWorkspaceCublasKernelRequirement];
        CublasMatrixMultiply::instance().mtx.unlock();
    }
}

bool CublasMatrixMultiply::chooseOptimalGemmKernel(const int gpuNum,
                                                   const int rowsA,
                                                   const int colsA,
                                                   const int rowsB,
                                                   const int colsB,
                                                   const int ldA,
                                                   const int ldB,
                                                   const int ldC,
                                                   const int ldD,
                                                   const bool transposeA,
                                                   const bool transposeB,
                                                   const bool transposeC,
                                                   const TensorDescriptor::DataType ABCDataType,
                                                   const bool allowWorkspaces,
                                                   const bool printResults) {
    lock_guard<mutex> lck(CublasMatrixMultiply::instance().mtx);

    assert(gpuNum >= 0);
    assert(gpuNum < (int)MachineEvaluator::instance().getNumGpus());
    assert(ABCDataType == TensorDescriptor::DataType::FP32 || ABCDataType == TensorDescriptor::DataType::FP16);

    // Ensure the operation is legal
    // The number of C and D columns is specified by the sizes of A and B, so verify A and B
    const int32_t finalRowsA = (transposeA ? colsA : rowsA);
    const int32_t finalColsA = (transposeA ? rowsA : colsA);
    const int32_t finalRowsB = (transposeB ? colsB : rowsB);
    const int32_t finalColsB = (transposeB ? rowsB : colsB);
    assert(finalColsA == finalRowsB);

    const int32_t initialRowsC = (transposeC ? finalColsB : finalRowsA);
    const int32_t initialColsC = (transposeC ? finalRowsA : finalColsB);
    assert(ldC >= initialColsC);

    // const int32_t finalRowsC = finalRowsA;
    const int32_t finalColsC = finalColsB;
    assert(ldD >= finalColsC);

    Stream stream(gpuNum);

    Event optimizationStartEvent;
    if (printResults)
        optimizationStartEvent = stream.putEvent(true);

    double opSize = (long)rowsA * (long)colsA * (long)finalColsC;
    double targetCount;
    if (opSize < pow(1024.0, 3)) {
        targetCount = 10000;
    } else {
        // Following equation from best fit line from https://www.socscistatistics.com/tests/regression/default.aspx
        targetCount = -0.000000052825 * opSize + 550;
    }
    if (printResults)
        printf("target initial contestants %lf\n", targetCount);
    // unsigned int initialContestantCount = maxl(100, targetCount);
    uint32_t initialContestantCount = 100;
    unsigned int finalContestantCount = 100;
    if (printResults)
        printf("finalContestantCount %d\n", finalContestantCount);

    constexpr int initialRun = 5;
    constexpr int finalRun = 20;

    constexpr long ONE_HUNDRED_MEGS = 134217728;
    const int ELEMENT_SIZE = TensorDescriptor::getElementSizeInBytes(ABCDataType);

    string gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    ScopedGpu scopedGpu(gpuNum);

    long totalGpuMem = MachineEvaluator::instance().getTotalGlobalMemBytes(gpuNum);

    KernelRequirement kernelRequirement(MachineEvaluator::instance().getGpuType(gpuNum),
                                        rowsA,
                                        colsA,
                                        rowsB,
                                        colsB,
                                        transposeA,
                                        transposeB,
                                        transposeC,
                                        ldA,
                                        ldB,
                                        ldC,
                                        ldD,
                                        allowWorkspaces);

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    // CublasMatrixMultiply::instance().mtx.lock();

    // Will only evaluate kernel once per gpu type
    if (optimalKernels.count(cublasKernelRequirement) == 1) {
        CublasKernel optimalKernel = optimalKernels[cublasKernelRequirement];
        bool kernelWillRunOnGpu;
        unsigned int workspaceSizeInBytes = optimalKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu);
        assert(kernelWillRunOnGpu);
        // CublasMatrixMultiply::instance().mtx.unlock();
        return workspaceSizeInBytes > 0 ? true : false;
    }

    assert(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    const int32_t rowsC = (transposeA == false ? rowsA : colsA);
    const int32_t colsC = (transposeB == false ? colsB : rowsB);
    int32_t rowsD = rowsC;
    int32_t colsD = colsC;

    // Get the expected best kernels
    uint64_t maxElements = max({rowsA * colsA, rowsB * colsB, rowsC * colsC, rowsD * colsD});
    const uint64_t maxAllowedWorkspaceSizeInBytes =
        allowWorkspaces ? maxElements * TensorDescriptor::getElementSizeInBytes(ABCDataType) : 0;
    float maxWaves = 0.0f;
    vector<CublasKernel> preCheckedKernels;
    preCheckedKernels = getHeuristicGemmKernels(initialContestantCount,
                                                gpuNum,
                                                rowsA,
                                                colsA,
                                                rowsB,
                                                colsB,
                                                transposeA,
                                                transposeB,
                                                transposeC,
                                                ldA,
                                                ldB,
                                                ldC,
                                                ldD,
                                                // When set to 0, no workspace allowed:
                                                maxAllowedWorkspaceSizeInBytes,
                                                // When set to 0.0f, any number of waves allowed:
                                                maxWaves,
                                                ABCDataType);

    vector<CublasKernel> kernels;
    uint64_t maxWorkspaceSizeInBytes = 0;
    for (uint32_t i = 0; i < preCheckedKernels.size(); ++i) {
        bool kernelWillRunOnGpu;
        CublasKernel cublasKernel = preCheckedKernels[i];

        unsigned long workspaceSizeInBytes = cublasKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu);
        if (!kernelWillRunOnGpu) {
            continue;
        }
        if (workspaceSizeInBytes > maxAllowedWorkspaceSizeInBytes) {
            printf("B allow workspaces %i workspaceSizeInBytes %ld maxWorkspaceSize %ld\n",
                   allowWorkspaces,
                   workspaceSizeInBytes,
                   maxAllowedWorkspaceSizeInBytes);
            continue;
        }
        if (workspaceSizeInBytes > maxWorkspaceSizeInBytes) {
            maxWorkspaceSizeInBytes = workspaceSizeInBytes;
        }

        // I have seen some kernel report 0 waves and causes sporadic runtime failures, so avoid that.
        if (!(cublasKernel.getWavesCount(gpuNum) > 0.0f)) {
            continue;
        }

        kernels.push_back(cublasKernel);
    }

    if (printResults)
        printf("%ld selected initial contestants\n", kernels.size());
    assert(!kernels.empty());

    /**
     * All fully specified kernels that will be tried are now contained in the kernels vector.
     *
     * The following sequence will be performed to find the fastest kernel.
     *     1. Warm up the gpu (boost clock, cooling, etc) by running kernels for max(10 kernel executions, about 10 ms).
     *     2. Run 100 kernels nearest to 1 wave 5 times each, measure performance of each kernel.
     *          a. It is important to run kernel x once follow by kernel x+1 once followed by kernel x+2 once, ...,
     *             and run this whole sequence 5 times so that each kernel gets approximately the same clock frequencies on average.
     *     3. Discard all but the top 10 performing kernels.
     *     4. Run the top 10 kernels 10 times more each, add to the frequency measurement of these kernels.
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

    // Ensure there are no unreported runtime errors
    stream.synchronize();

    // Allocate a lot of memory, to ensure subsequent calls are not benefiting from cache hits
    long memPerInstance = (rowsA * ldA + rowsB * ldB + initialRowsC * ldC) * ELEMENT_SIZE;
    long totalMatrixMemory = minl(totalGpuMem * 0.4, maxl(ONE_HUNDRED_MEGS, 10 * memPerInstance));
    long numInstances = minl(totalMatrixMemory / memPerInstance, 5000);
    assert(numInstances > 0);

    long numWorkspaceInstances;
    if (maxWorkspaceSizeInBytes == 0) {
        maxWorkspaceSizeInBytes = 1;
        numWorkspaceInstances = 1;
    } else {
        long totalWorkspaceMem = minl(totalGpuMem * 0.4, maxl(ONE_HUNDRED_MEGS, 10 * maxWorkspaceSizeInBytes));
        numWorkspaceInstances = minl(totalWorkspaceMem / maxWorkspaceSizeInBytes, 5000);
        assert(numWorkspaceInstances > 0);
    }

    vector<Tensor> A;
    vector<Tensor> B;
    vector<Tensor> C;
    vector<Tensor> workspace;
    A.reserve(numInstances);
    B.reserve(numInstances);
    C.reserve(numInstances);
    workspace.reserve(numWorkspaceInstances);
    for (int i = 0; i < numInstances; ++i) {
        A.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(ABCDataType, {(uint64_t)rowsA, (uint64_t)ldA}));
        B.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(ABCDataType, {(uint64_t)rowsB, (uint64_t)ldB}));
        C.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(ABCDataType, {(uint64_t)initialRowsC, (uint64_t)ldC}));
    }
    for (int i = 0; i < numWorkspaceInstances; ++i) {
        workspace.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                               TensorDescriptor(TensorDescriptor::DataType::UINT8, {maxWorkspaceSizeInBytes}));
    }

    vector<CublasKernel> prunedKernels;
    if (kernels.size() > initialContestantCount) {
        // Prune all but initialContestantCount provably working kernels
        prunedKernels.reserve(initialContestantCount * 2);

        constexpr float TARGET_WAVES = 1.0f;

        // Want the kernels nearest to 1 wave.
        std::sort(kernels.begin(), kernels.end(), [gpuNum](const CublasKernel &lhs, const CublasKernel &rhs) {
            return abs(TARGET_WAVES - lhs.getWavesCount(gpuNum)) < abs(TARGET_WAVES - rhs.getWavesCount(gpuNum));
        });

        // Keep all kernels that are as good as the kernelsToKeep'th best kernel.
        float maxWavesDiff = abs(TARGET_WAVES - kernels[initialContestantCount - 1].getWavesCount(gpuNum));
        for (unsigned int i = 0; i < kernels.size() && (abs(TARGET_WAVES - kernels[i].getWavesCount(gpuNum)) <= maxWavesDiff ||
                                                        prunedKernels.size() < initialContestantCount);
             ++i) {
            cublasStatus = kernels[i].runWithoutChecks(A[0], B[0], C[0], C[0], workspace[0], false, false, stream);
            if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
                stream.synchronize();
                prunedKernels.push_back(kernels[i]);
            }
        }
        kernels = prunedKernels;
    } else {
        prunedKernels.reserve(kernels.size());

        // Prune just the non-working kernels
        for (unsigned int i = 0; i < kernels.size(); ++i) {
            cublasStatus = kernels[i].runWithoutChecks(A[0], B[0], C[0], C[0], workspace[0], false, false, stream);
            if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
                stream.synchronize();
                prunedKernels.push_back(kernels[i]);
            }
        }
        kernels = prunedKernels;
    }

    if (printResults)
        printf("got %ld kernels\n", kernels.size());
    assert(!kernels.empty());

    int tensorInstance = 0;
    int workspaceInstance = 0;

    // Warm up
    double elapsedTime = 0.0;
    Event startEvent = stream.putEvent(true);
    for (int i = 0; i < 5; ++i) {
        kernels[rand() % kernels.size()].runWithoutChecks(
            A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, false, stream);
        tensorInstance += 1;
        if (tensorInstance >= numInstances)
            tensorInstance = 0;
        workspaceInstance += 1;
        if (workspaceInstance >= numWorkspaceInstances)
            workspaceInstance = 0;
    }

    Event stopEvent = stream.putEvent(true);
    elapsedTime = stopEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
    double kernelExecutionTimeMilliseconds = elapsedTime / 5.0;
    int kernelsToExecute = maxl(5, 12.0 / kernelExecutionTimeMilliseconds);

    elapsedTime = 0.0;
    while (elapsedTime < 9.0) {
        Event startEvent = stream.putEvent(true);

        for (int i = 0; i < kernelsToExecute; ++i) {
            kernels[rand() % kernels.size()].runWithoutChecks(A[tensorInstance],
                                                              B[tensorInstance],
                                                              C[tensorInstance],
                                                              C[tensorInstance],
                                                              workspace[workspaceInstance],
                                                              false,
                                                              false,
                                                              stream);
            tensorInstance += 1;
            if (tensorInstance >= numInstances)
                tensorInstance = 0;
            workspaceInstance += 1;
            if (workspaceInstance >= numWorkspaceInstances)
                workspaceInstance = 0;
        }

        Event stopEvent = stream.putEvent(true);
        elapsedTime += stopEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
    }

    // Run all kernels 10 times and measure performance of each
    // But first put some initial work in the stream so that all timed work will be queued before running
    for (int i = 0; i < 5; ++i) {
        kernels[rand() % kernels.size()].runWithoutChecks(
            A[tensorInstance], B[tensorInstance], C[tensorInstance], C[tensorInstance], workspace[workspaceInstance], false, false, stream);
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

    for (int run = 0; run < initialRun; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            startEvents[kernelIndex].push_back(stream.putEvent(true));
            cublasStatus = kernels[kernelIndex].runWithoutChecks(A[tensorInstance],
                                                                 B[tensorInstance],
                                                                 C[tensorInstance],
                                                                 C[tensorInstance],
                                                                 workspace[workspaceInstance],
                                                                 false,
                                                                 false,
                                                                 stream);
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
            tensorInstance += 1;
            if (tensorInstance >= numInstances)
                tensorInstance = 0;
            workspaceInstance += 1;
            if (workspaceInstance >= numWorkspaceInstances)
                workspaceInstance = 0;
            stopEvents[kernelIndex].push_back(stream.putEvent(true));
            if (cublasStatus != CUBLAS_STATUS_SUCCESS)
                kernels[kernelIndex].setErrorFlag();
        }
    }

    for (int run = 0; run < initialRun; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            kernels[kernelIndex].recordRun(
                stopEvents[kernelIndex][run].synchronizeAndReportElapsedTimeInMilliseconds(startEvents[kernelIndex][run]));
        }
    }

    if (printResults) {
        printf("Initial kernel measurments:\n\n");
        std::sort(kernels.begin(), kernels.end(), CublasKernel::executionTimeComparison);
        for (unsigned int i = 0; i < kernels.size(); ++i) {
            printf("%s\n", kernels[i].toString(gpuNum).c_str());
        }
        printf("\n\n");
    }

    // Keep the best finalContestantCount kernels, discard the rest
    int kernelsToKeep = minl(finalContestantCount, kernels.size());
    std::partial_sort(kernels.begin(), kernels.begin() + kernelsToKeep, kernels.end(), CublasKernel::executionTimeComparison);
    kernels.erase(kernels.begin() + kernelsToKeep, kernels.end());
    if (printResults)
        printf("kernels.size() %ld\n", kernels.size());

    for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
        startEvents[kernelIndex].clear();
        stopEvents[kernelIndex].clear();
    }

    // Run the best finalContestantCount kernels finalRun times more each and update performance measurements for each of them
    for (unsigned int i = 0; i < kernels.size(); ++i)
        kernels[i].stashRunStats();

    cublasStatus = CUBLAS_STATUS_SUCCESS;
    for (int run = 0; run < finalRun; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            startEvents[kernelIndex].push_back(stream.putEvent(true));
            if (!kernels[kernelIndex].getErrorFlag())
                cublasStatus = kernels[kernelIndex].runWithoutChecks(A[tensorInstance],
                                                                     B[tensorInstance],
                                                                     C[tensorInstance],
                                                                     C[tensorInstance],
                                                                     workspace[workspaceInstance],
                                                                     false,
                                                                     false,
                                                                     stream);
            tensorInstance += 1;
            if (tensorInstance >= numInstances)
                tensorInstance = 0;
            workspaceInstance += 1;
            if (workspaceInstance >= numWorkspaceInstances)
                workspaceInstance = 0;
            stopEvents[kernelIndex].push_back(stream.putEvent(true));

            if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
                int checkIndex = kernelIndex == 0 ? kernels.size() - 1 : kernelIndex - 1;
                kernels[checkIndex].setErrorFlag();
                printf("cublasStatus %d on kernelIndex %d run %d tensorInstance %d of %ld workspaceInstance %d of %ld workspaceSize %ld\n",
                       cublasStatus,
                       kernelIndex,
                       run,
                       tensorInstance,
                       numInstances,
                       workspaceInstance,
                       numWorkspaceInstances,
                       maxWorkspaceSizeInBytes);
                printf("%s\n", kernels[checkIndex].toString(gpuNum).c_str());
            }

            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        }
    }
    for (int run = 0; run < finalRun; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            kernels[kernelIndex].recordRun(
                stopEvents[kernelIndex][run].synchronizeAndReportElapsedTimeInMilliseconds(startEvents[kernelIndex][run]));
        }
    }

    if (printResults) {
        std::sort(kernels.begin(), kernels.end(), CublasKernel::executionTimeComparison);
        printf("Kernel optimization results:\n\n");
        for (unsigned int i = 0; i < kernels.size(); ++i) {
            printf("%s\n", kernels[i].toString(gpuNum).c_str());
        }
        printf("\n");
    }

    CublasKernel bestKernel = *std::min_element(kernels.begin(), kernels.end(), CublasKernel::executionTimeComparison);
    assert(!bestKernel.getErrorFlag());
    bestKernel.unstashRunStats();
    bool kernelWillRunOnGpu;
    bool bestKernelHasWorkspace = bestKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu) > 0;
    assert(kernelWillRunOnGpu);

    optimalKernels[cublasKernelRequirement] = bestKernel;

    // If the best one that may have a workspace has no workspace, then this is also the best one that may not have a workspace.
    if (allowWorkspaces && !bestKernelHasWorkspace) {
        KernelRequirement kernelRequirementWithoutWorkspace(MachineEvaluator::instance().getGpuType(gpuNum),
                                                            rowsA,
                                                            colsA,
                                                            rowsB,
                                                            colsB,
                                                            transposeA,
                                                            transposeB,
                                                            transposeC,
                                                            ldA,
                                                            ldB,
                                                            ldC,
                                                            ldD,
                                                            false);

        CublasKernelRequirement noWorkspaceCublasKernelRequirement(kernelRequirementWithoutWorkspace, operationType);
        optimalKernels[noWorkspaceCublasKernelRequirement] = bestKernel;
    }

    // CublasMatrixMultiply::instance().mtx.unlock();

    Event optimizationEndEvent;
    if (printResults) {
        optimizationEndEvent = stream.putEvent(true);
        float optimizationTimeMillis = optimizationEndEvent.synchronizeAndReportElapsedTimeInMilliseconds(optimizationStartEvent);
        printf("\nOverall optimization time %0.1fms\n", optimizationTimeMillis);
    }

    return bestKernelHasWorkspace;
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
                                                operationType.computeDataType,
                                                operationType.scaleDataType,
                                                operationType.ADataType,
                                                operationType.BDataType,
                                                operationType.CDataType,
                                                operationType.DDataType,
                                                numRequestedAlgos,
                                                allSupportedAlgorithmIds.data(),
                                                &numReturnedAlgos);
        assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    }

    for (int i = 0; i < numReturnedAlgos; ++i) {
        cublasLtMatmulAlgo_t algo;
        cublasStatus = cublasLtMatmulAlgoInit(MachineEvaluator::instance().getCublasLtHandle(gpuNum),
                                              operationType.computeDataType,
                                              operationType.scaleDataType,
                                              operationType.ADataType,
                                              operationType.BDataType,
                                              operationType.CDataType,
                                              operationType.DDataType,
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

bool CublasMatrixMultiply::isSplitKSupported(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int splitKSupported;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitKSupported, sizeof(splitKSupported), &sizeWritten);
    assert(sizeWritten == sizeof(splitKSupported));

    return splitKSupported ? true : false;
}

uint32_t CublasMatrixMultiply::getReductionSupportMask(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    uint32_t reductionSupportMask;
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

unsigned int CublasMatrixMultiply::getGemmWorkspaceSizeInBytes(int gpuNum,
                                                               int rowsA,
                                                               int colsA,
                                                               int rowsB,
                                                               int colsB,
                                                               int ldA,
                                                               int ldB,
                                                               int ldC,
                                                               int ldD,
                                                               bool transposeA,
                                                               bool transposeB,
                                                               bool transposeC,
                                                               TensorDescriptor::DataType ABCDataType,
                                                               bool &kernelWillRunOnGpu) {
    KernelRequirement kernelRequirement(MachineEvaluator::instance().getGpuType(gpuNum),
                                        rowsA,
                                        colsA,
                                        rowsB,
                                        colsB,
                                        transposeA,
                                        transposeB,
                                        transposeC,
                                        ldA,
                                        ldB,
                                        ldC,
                                        ldD,
                                        true);

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    CublasMatrixMultiply::instance().mtx.lock();
    assert(optimalKernels.count(cublasKernelRequirement) == 1);
    unsigned int workspaceSize = optimalKernels[cublasKernelRequirement].getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu);
    CublasMatrixMultiply::instance().mtx.unlock();

    return workspaceSize;
}

double CublasMatrixMultiply::getOptimalKernelTime(string gpuType,
                                                  int rowsA,
                                                  int colsA,
                                                  int rowsB,
                                                  int colsB,
                                                  int ldA,
                                                  int ldB,
                                                  int ldC,
                                                  int ldD,
                                                  bool transposeA,
                                                  bool transposeB,
                                                  bool transposeC,
                                                  TensorDescriptor::DataType ABCDataType,
                                                  bool workspaceAllowed) {
    KernelRequirement kernelRequirement(
        gpuType, rowsA, colsA, rowsB, colsB, transposeA, transposeB, transposeC, ldA, ldB, ldC, ldD, workspaceAllowed);

    cudaDataType_t ABCDataTypeCuda = mapToCublasDataType(ABCDataType);
    OperationType operationType(CUBLAS_COMPUTE_32F, CUDA_R_32F, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda, ABCDataTypeCuda);
    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    string ABCDataTypeString = ABCDataType == TensorDescriptor::DataType::FP32 ? "FP32" : "FP16";

    CublasMatrixMultiply::instance().mtx.lock();
    auto it = optimalKernels.find(cublasKernelRequirement);
    if (it == optimalKernels.end()) {
        string message =
            "CublasMatrixMultiply::getOptimalKernelTime() : Kernel time is not known because kernel time has not been measured for "
            "gpuType " +
            gpuType + " rowsA " + std::to_string(rowsA) + " colsA " + std::to_string(colsA) + " colsB " + std::to_string(colsB) +
            "dataType " + ABCDataTypeString;
        CublasMatrixMultiply::instance().mtx.unlock();
        throw(CublasMatrixMultiply::Youreusingitwrong(message));
    }
    double averageRunTimeMilliseconds = it->second.getAverageRunTimeMilliseconds();
    CublasMatrixMultiply::instance().mtx.unlock();

    return averageRunTimeMilliseconds;
}

double CublasMatrixMultiply::getOptimalKernelTime(int gpuNum,
                                                  int rowsA,
                                                  int colsA,
                                                  int rowsB,
                                                  int colsB,
                                                  int ldA,
                                                  int ldB,
                                                  int ldC,
                                                  int ldD,
                                                  bool transposeA,
                                                  bool transposeB,
                                                  bool transposeC,
                                                  TensorDescriptor::DataType ABCDataType,
                                                  bool workspaceAllowed) {
    return getOptimalKernelTime(MachineEvaluator::instance().getGpuType(gpuNum),
                                rowsA,
                                colsA,
                                rowsB,
                                colsB,
                                ldA,
                                ldB,
                                ldC,
                                ldD,
                                transposeA,
                                transposeB,
                                transposeC,
                                ABCDataType,
                                workspaceAllowed);
}
