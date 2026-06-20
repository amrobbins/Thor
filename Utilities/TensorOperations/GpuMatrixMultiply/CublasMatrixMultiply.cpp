#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include "Utilities/Expression/CudaHelpers.h"

#include <stdexcept>
#include "DeepLearning/Implementation/ThorError.h"

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

namespace {

void destroyLtMatrixLayoutNoThrow(cublasLtMatrixLayout_t& desc) noexcept {
    if (desc) {
        (void)cublasLtMatrixLayoutDestroy(desc);
        desc = nullptr;
    }
}

void destroyLtMatmulDescNoThrow(cublasLtMatmulDesc_t& desc) noexcept {
    if (desc) {
        (void)cublasLtMatmulDescDestroy(desc);
        desc = nullptr;
    }
}

}  // namespace

CublasMatrixMultiply::LtMatmulPlan::~LtMatmulPlan() {
    destroyLtMatrixLayoutNoThrow(d_desc);
    destroyLtMatrixLayoutNoThrow(c_desc);
    destroyLtMatrixLayoutNoThrow(b_desc);
    destroyLtMatrixLayoutNoThrow(a_desc);
    destroyLtMatmulDescNoThrow(operation_desc_device);
    destroyLtMatmulDescNoThrow(operation_desc_host);
}

void CublasMatrixMultiply::LtMatmulPlan::runGemmWithEpilogue(Tensor A,
                                                             Tensor B,
                                                             std::optional<Tensor> addend,
                                                             Tensor D,
                                                             const float *alpha,
                                                             const float *beta,
                                                             Stream stream,
                                                             CublasScalarPointerMode pointerMode,
                                                             std::optional<Tensor> workspace,
                                                             bool addendIsBiasVector) const {
    ScopedGpu scopedGpu(stream.getGpuNum());
    const uint64_t workspaceLimitBytes = workspace.has_value() ? workspace.value().getArraySizeInBytes() : 0;
    const uint64_t selectedWorkspaceBytes = algorithm.workspace_size_in_bytes;
    if (selectedWorkspaceBytes > workspaceLimitBytes) {
        throw std::runtime_error("Cached cuBLASLt epilogue GEMM plan requires more workspace than the stamped launch provided.");
    }

    const void *ltA = B.getMemPtr();
    const void *ltB = A.getMemPtr();
    const void *ltC = (addend.has_value() && !addendIsBiasVector) ? addend.value().getMemPtr() : D.getMemPtr();
    const float betaZero = 0.0f;
    const float betaOne = 1.0f;
    const float *effectiveBeta = addend.has_value() && !addendIsBiasVector ? beta : &betaZero;
    if (addendIsBiasVector) {
        effectiveBeta = &betaZero;
    } else if (addend.has_value() && effectiveBeta == nullptr) {
        effectiveBeta = &betaOne;
    }
    void *workspacePtr = selectedWorkspaceBytes > 0 ? workspace.value().getMemPtr() : nullptr;

    CHECK_CUBLAS(cublasLtMatmul(stream.getCublasLtHandleUnchecked(),
                                operationDesc(pointerMode),
                                alpha,
                                ltA,
                                a_desc,
                                ltB,
                                b_desc,
                                effectiveBeta,
                                ltC,
                                c_desc,
                                D.getMemPtr(),
                                d_desc,
                                &algorithm.algorithm,
                                workspacePtr,
                                static_cast<size_t>(selectedWorkspaceBytes),
                                stream));
}

void CublasMatrixMultiply::LtMatmulPlan::runGemmWithBackwardEpilogue(Tensor A,
                                                                     Tensor B,
                                                                     std::optional<Tensor> addend,
                                                                     Tensor D,
                                                                     const float *alpha,
                                                                     const float *beta,
                                                                     Stream stream,
                                                                     CublasScalarPointerMode pointerMode,
                                                                     std::optional<Tensor> workspace) const {
    ScopedGpu scopedGpu(stream.getGpuNum());
    const uint64_t workspaceLimitBytes = workspace.has_value() ? workspace.value().getArraySizeInBytes() : 0;
    const uint64_t selectedWorkspaceBytes = algorithm.workspace_size_in_bytes;
    if (selectedWorkspaceBytes > workspaceLimitBytes) {
        throw std::runtime_error("Cached cuBLASLt backward epilogue GEMM plan requires more workspace than the stamped launch provided.");
    }

    const void *ltA = B.getMemPtr();
    const void *ltB = A.getMemPtr();
    const void *ltC = addend.has_value() ? addend.value().getMemPtr() : D.getMemPtr();
    const float betaZero = 0.0f;
    const float betaOne = 1.0f;
    const float *effectiveBeta = addend.has_value() ? beta : &betaZero;
    if (addend.has_value() && effectiveBeta == nullptr) {
        effectiveBeta = &betaOne;
    }
    void *workspacePtr = selectedWorkspaceBytes > 0 ? workspace.value().getMemPtr() : nullptr;

    CHECK_CUBLAS(cublasLtMatmul(stream.getCublasLtHandleUnchecked(),
                                operationDesc(pointerMode),
                                alpha,
                                ltA,
                                a_desc,
                                ltB,
                                b_desc,
                                effectiveBeta,
                                ltC,
                                c_desc,
                                D.getMemPtr(),
                                d_desc,
                                &algorithm.algorithm,
                                workspacePtr,
                                static_cast<size_t>(selectedWorkspaceBytes),
                                stream));
}


namespace {

cublasLtHandle_t compileTimeCublasLtHandle(int gpuNum) {
    static std::mutex compileHandleMutex;
    static std::unordered_map<int, Stream> compileStreamsByDevice;

    std::lock_guard<std::mutex> lock(compileHandleMutex);
    auto [it, inserted] = compileStreamsByDevice.try_emplace(gpuNum, gpuNum);
    if (inserted) {
        it->second.informIsStatic();
    }
    return it->second.getCublasLtHandle();
}

void setTensorwideFp8ScaleMode(cublasLtMatmulDesc_t operationDesc, cublasLtMatmulDescAttributes_t attribute) {
    const cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, attribute, &scaleMode, sizeof(scaleMode)));
}

void setFp8ScalePointerIfPresent(cublasLtMatmulDesc_t operationDesc,
                                 cublasLtMatmulDescAttributes_t attribute,
                                 const float *scaleDevicePointer) {
    if (scaleDevicePointer != nullptr) {
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, attribute, &scaleDevicePointer, sizeof(scaleDevicePointer)));
    }
}

void setFp8AmaxPointerIfPresent(cublasLtMatmulDesc_t operationDesc, cublasLtMatmulDescAttributes_t attribute, float *amaxDevicePointer) {
    if (amaxDevicePointer != nullptr) {
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, attribute, &amaxDevicePointer, sizeof(amaxDevicePointer)));
    }
}

bool usesFp8ColumnMajorLtPath(const OperationType &operationType) { return isCublasLtFp8OperationType(operationType); }

uint64_t cudaDataTypeSizeInBytes(cudaDataType_t dataType) {
    switch (dataType) {
        case CUDA_R_32F:
            return 4;
        case CUDA_R_16BF:
        case CUDA_R_16F:
            return 2;
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
        case CUDA_R_8I:
            return 1;
        default:
            THOR_UNREACHABLE();
            return 1;
    }
}

cudaDataType_t getLtADescDataType(const OperationType &operationType) {
    return usesFp8ColumnMajorLtPath(operationType) ? operationType.BDataType : operationType.ADataType;
}

cudaDataType_t getLtBDescDataType(const OperationType &operationType) {
    return usesFp8ColumnMajorLtPath(operationType) ? operationType.ADataType : operationType.BDataType;
}

CublasMatrixMultiply::Fp8MatmulScales getLtFp8Scales(const OperationType &operationType, CublasMatrixMultiply::Fp8MatmulScales fp8Scales) {
    if (!usesFp8ColumnMajorLtPath(operationType)) {
        return fp8Scales;
    }
    return CublasMatrixMultiply::Fp8MatmulScales::tensorwide(fp8Scales.BScaleDevicePointer,
                                                             fp8Scales.AScaleDevicePointer,
                                                             fp8Scales.CScaleDevicePointer,
                                                             fp8Scales.DScaleDevicePointer,
                                                             fp8Scales.DAmaxDevicePointer);
}

bool fp8NeedsRowMajorTransposeWorkspace(const OperationType &operationType, bool transposeA, bool transposeB) {
    return usesFp8ColumnMajorLtPath(operationType) && (transposeA || !transposeB);
}

uint64_t fp8RowMajorTransposeWorkspaceUpperBoundBytes(
    const OperationType &operationType, int rowsA, int colsA, int rowsB, int colsB, bool transposeA, bool transposeB) {
    if (!fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        return 0;
    }

    uint64_t bytes = 0;
    if (transposeA) {
        bytes += static_cast<uint64_t>(rowsA) * static_cast<uint64_t>(colsA) * cudaDataTypeSizeInBytes(operationType.ADataType);
    }
    if (!transposeB) {
        bytes += static_cast<uint64_t>(rowsB) * static_cast<uint64_t>(colsB) * cudaDataTypeSizeInBytes(operationType.BDataType);
    }
    // Match CublasKernel's conservative workspace alignment between temporary regions.
    return bytes + 512;
}

void validateFp8RowMajorGemmShapeAndLayoutOrThrow(const OperationType &operationType,
                                                  int32_t rowsA,
                                                  int32_t colsA,
                                                  int32_t rowsB,
                                                  int32_t colsB,
                                                  int32_t ldA,
                                                  int32_t ldB,
                                                  int32_t ldC,
                                                  int32_t ldD,
                                                  bool transposeA,
                                                  bool transposeB,
                                                  const std::string &context) {
    if (!usesFp8ColumnMajorLtPath(operationType)) {
        return;
    }

    const int32_t k = transposeA ? rowsA : colsA;
    const int32_t n = transposeB ? rowsB : colsB;

    if ((n % 2) != 0) {
        throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires even N.");
    }
    if ((k % 2) != 0) {
        throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires even K.");
    }
    if (ldA != colsA) {
        throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed A: ldA must equal colsA.");
    }
    if (ldB != colsB) {
        throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed B: ldB must equal colsB.");
    }
    if (ldC != n) {
        throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed C: ldC must equal N.");
    }
    if (ldD != n) {
        throw std::runtime_error(context + " FP8 row-major cuBLASLt path requires packed D: ldD must equal N.");
    }
}

void configureTensorwideFp8Scales(cublasLtMatmulDesc_t operationDesc,
                                  const OperationType &operationType,
                                  const CublasMatrixMultiply::Fp8MatmulScales &fp8Scales) {
    const cudaDataType_t ltADataType = getLtADescDataType(operationType);
    const cudaDataType_t ltBDataType = getLtBDescDataType(operationType);
    const CublasMatrixMultiply::Fp8MatmulScales ltFp8Scales = getLtFp8Scales(operationType, fp8Scales);

    if (!ltFp8Scales.hasAnyScalePointer() && !isCublasLtFp8CudaType(ltADataType) && !isCublasLtFp8CudaType(ltBDataType) &&
        !isCublasLtFp8CudaType(operationType.CDataType) && !isCublasLtFp8CudaType(operationType.DDataType)) {
        return;
    }

    if (isCublasLtFp8CudaType(ltADataType)) {
        setTensorwideFp8ScaleMode(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE);
    }
    if (isCublasLtFp8CudaType(ltBDataType)) {
        setTensorwideFp8ScaleMode(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE);
    }
    if (isCublasLtFp8CudaType(operationType.CDataType) || ltFp8Scales.hasCScale()) {
        setTensorwideFp8ScaleMode(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE);
    }
    if (isCublasLtFp8CudaType(operationType.DDataType)) {
        setTensorwideFp8ScaleMode(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE);
    }

    setFp8ScalePointerIfPresent(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, ltFp8Scales.AScaleDevicePointer);
    setFp8ScalePointerIfPresent(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, ltFp8Scales.BScaleDevicePointer);
    setFp8ScalePointerIfPresent(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, ltFp8Scales.CScaleDevicePointer);

    if (isCublasLtFp8CudaType(operationType.DDataType)) {
        setFp8ScalePointerIfPresent(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, ltFp8Scales.DScaleDevicePointer);
        setFp8AmaxPointerIfPresent(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, ltFp8Scales.DAmaxDevicePointer);
    }
}

void configureLtOperationTransposes(
    cublasLtMatmulDesc_t operationDesc, const OperationType &operationType, bool transposeA, bool transposeB, bool transposeC) {
    if (usesFp8ColumnMajorLtPath(operationType)) {
        // cuBLASLt exposes the usable FP8 kernels as column-major TN.  With row-major Thor buffers,
        // this computes D^T = (op(B))^T * (op(A))^T while preserving the public D = op(A) * op(B) API.
        cublasOperation_t transpose = CUBLAS_OP_T;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        return;
    }

    if (transposeA) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    }
    if (transposeB) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose)));
    }
    if (transposeC) {
        cublasOperation_t transpose = CUBLAS_OP_T;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transpose, sizeof(transpose)));
    }
}

void createLtMatrixLayoutsForRowMajorGemm(cublasLtMatrixLayout_t *ADesc,
                                          cublasLtMatrixLayout_t *BDesc,
                                          cublasLtMatrixLayout_t *CDesc,
                                          cublasLtMatrixLayout_t *DDesc,
                                          const OperationType &operationType,
                                          const int32_t A_rows,
                                          const int32_t A_cols,
                                          const int32_t B_rows,
                                          const int32_t B_cols,
                                          const int32_t C_rows,
                                          const int32_t C_cols,
                                          const int32_t D_rows,
                                          const int32_t D_cols,
                                          const int32_t ld_A,
                                          const int32_t ld_B,
                                          const int32_t ld_C,
                                          const int32_t ld_D,
                                          bool transposeA,
                                          bool transposeB) {
    int64_t ld;

    if (usesFp8ColumnMajorLtPath(operationType)) {
        const cublasLtOrder_t columnMajorOrder = CUBLASLT_ORDER_COL;

        // Internal cuBLASLt A operand is X=(op(B))^T, presented as column-major X^T and used with TRANSA=T.
        const int32_t internalARowMajorRows = transposeB ? B_rows : B_cols;
        const int32_t internalARowMajorCols = transposeB ? B_cols : B_rows;
        const int32_t internalALd = transposeB ? ld_B : B_rows;

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
            ADesc, getLtADescDataType(operationType), internalARowMajorCols, internalARowMajorRows, internalALd));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
        ld = internalALd;
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

        // Internal cuBLASLt B operand is Y=op(A), presented as column-major Y^T and used with TRANSB=N.
        const int32_t internalBRowMajorRows = transposeA ? A_cols : A_rows;
        const int32_t internalBRowMajorCols = transposeA ? A_rows : A_cols;
        const int32_t internalBLd = transposeA ? A_rows : ld_A;

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
            BDesc, getLtBDescDataType(operationType), internalBRowMajorCols, internalBRowMajorRows, internalBLd));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
        ld = internalBLd;
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(CDesc, operationType.CDataType, C_cols, C_rows, ld_C));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
        ld = ld_C;
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(DDesc, operationType.DDataType, D_cols, D_rows, ld_D));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
        ld = ld_D;
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
        return;
    }

    const cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(ADesc, operationType.ADataType, A_rows, A_cols, ld_A));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));
    ld = ld_A;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(BDesc, operationType.BDataType, B_rows, B_cols, ld_B));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));
    ld = ld_B;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(CDesc, operationType.CDataType, C_rows, C_cols, ld_C));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));
    ld = ld_C;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(DDesc, operationType.DDataType, D_rows, D_cols, ld_D));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));
    ld = ld_D;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
}
}  // namespace

// This variant allows non-packed matrices
void CublasMatrixMultiply::multiply(Tensor A,
                                    Tensor B,
                                    Tensor C,
                                    std::optional<Tensor> workspace,
                                    const int32_t A_rows,
                                    const int32_t A_cols,
                                    const int32_t B_rows,
                                    const int32_t B_cols,
                                    bool transposeA,
                                    bool transposeB,
                                    const bool accumulate,
                                    const bool negate,
                                    const DataType ABCDataType,
                                    Stream stream) {
    multiply(A,
             B,
             C,
             workspace,
             A_rows,
             A_cols,
             B_rows,
             B_cols,
             transposeA,
             transposeB,
             accumulate,
             negate,
             MatmulDataTypes::same(ABCDataType),
             stream);
}

void CublasMatrixMultiply::multiply(Tensor A,
                                    Tensor B,
                                    Tensor C,
                                    std::optional<Tensor> workspace,
                                    const int32_t A_rows,
                                    const int32_t A_cols,
                                    const int32_t B_rows,
                                    const int32_t B_cols,
                                    bool transposeA,
                                    bool transposeB,
                                    const bool accumulate,
                                    const bool negate,
                                    const MatmulDataTypes dataTypes,
                                    Stream stream) {
    const float *alpha = negate ? &ALPHA_NEGATE : &ALPHA_NO_SCALE;
    const float *beta = accumulate ? &BETA_ACCUMULATE : &BETA_CLEAR;

    gemm(A,
         B,
         C,
         C,
         workspace,
         A_rows,
         A_cols,
         B_rows,
         B_cols,
         transposeA,
         transposeB,
         false,
         alpha,
         beta,
         dataTypes,
         stream,
         CublasScalarPointerMode::Host);
}

void CublasMatrixMultiply::gemm(Tensor A,
                                Tensor B,
                                Tensor C,
                                Tensor D,
                                std::optional<Tensor> workspace,
                                const int32_t A_rows,
                                const int32_t A_cols,
                                const int32_t B_rows,
                                const int32_t B_cols,
                                bool transposeA,
                                bool transposeB,
                                bool transposeC,
                                const float *alpha,
                                const float *beta,
                                const DataType ABCDDataType,
                                Stream stream,
                                CublasScalarPointerMode pointerMode) {
    gemm(A,
         B,
         C,
         D,
         workspace,
         A_rows,
         A_cols,
         B_rows,
         B_cols,
         transposeA,
         transposeB,
         transposeC,
         alpha,
         beta,
         MatmulDataTypes::same(ABCDDataType),
         stream,
         pointerMode);
}

void CublasMatrixMultiply::gemm(Tensor A,
                                Tensor B,
                                Tensor C,
                                Tensor D,
                                std::optional<Tensor> workspace,
                                const int32_t A_rows,
                                const int32_t A_cols,
                                const int32_t B_rows,
                                const int32_t B_cols,
                                bool transposeA,
                                bool transposeB,
                                bool transposeC,
                                const float *alpha,
                                const float *beta,
                                const MatmulDataTypes dataTypes,
                                Stream stream,
                                CublasScalarPointerMode pointerMode) {
    gemm(A,
         B,
         C,
         D,
         workspace,
         A_rows,
         A_cols,
         B_rows,
         B_cols,
         transposeA,
         transposeB,
         transposeC,
         alpha,
         beta,
         dataTypes,
         Fp8MatmulScales::none(),
         stream,
         pointerMode);
}

void CublasMatrixMultiply::gemm(Tensor A,
                                Tensor B,
                                Tensor C,
                                Tensor D,
                                std::optional<Tensor> workspace,
                                const int32_t A_rows,
                                const int32_t A_cols,
                                const int32_t B_rows,
                                const int32_t B_cols,
                                bool transposeA,
                                bool transposeB,
                                bool transposeC,
                                const float *alpha,
                                const float *beta,
                                const MatmulDataTypes dataTypes,
                                const Fp8MatmulScales fp8Scales,
                                Stream stream,
                                CublasScalarPointerMode pointerMode) {
    THOR_THROW_IF_FALSE(A.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(B.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(C.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(D.getDimensions().size() == 2);
    const int32_t ld_A = A.getDimensions()[1];
    const int32_t ld_B = B.getDimensions()[1];
    const int32_t ld_C = C.getDimensions()[1];
    const int32_t ld_D = D.getDimensions()[1];

    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    int32_t D_rows = C_rows;
    int32_t D_cols = C_cols;

    THOR_THROW_IF_FALSE(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    THOR_THROW_IF_FALSE(!(C == D && transposeC));
    THOR_THROW_IF_FALSE(!(C == D) || dataTypes.C == dataTypes.D);

    THOR_THROW_IF_FALSE(A_rows > 0);
    THOR_THROW_IF_FALSE(A_cols > 0);
    THOR_THROW_IF_FALSE(B_rows > 0);
    THOR_THROW_IF_FALSE(B_cols > 0);
    THOR_THROW_IF_FALSE(ld_A >= A_cols);
    THOR_THROW_IF_FALSE(ld_B >= B_cols);
    THOR_THROW_IF_FALSE(ld_C >= C_cols);
    THOR_THROW_IF_FALSE(ld_D >= D_cols);
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::gemm");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, fp8Scales, "CublasMatrixMultiply::gemm");
    THOR_THROW_IF_FALSE(A.getDescriptor().getDataType() == dataTypes.A);
    THOR_THROW_IF_FALSE(B.getDescriptor().getDataType() == dataTypes.B);
    THOR_THROW_IF_FALSE(C.getDescriptor().getDataType() == dataTypes.C);
    THOR_THROW_IF_FALSE(D.getDescriptor().getDataType() == dataTypes.D);
    // Check dimensions of tensors
    vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
    vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
    vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
    vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(ADimensions.size() == 2);
    THOR_THROW_IF_FALSE(ADimensions[0] == static_cast<uint32_t>(A_rows));
    THOR_THROW_IF_FALSE(ADimensions[1] == static_cast<uint32_t>(ld_A));
    THOR_THROW_IF_FALSE(BDimensions[0] == static_cast<uint32_t>(B_rows));
    THOR_THROW_IF_FALSE(BDimensions[1] == static_cast<uint32_t>(ld_B));
    THOR_THROW_IF_FALSE(CDimensions[0] == static_cast<uint32_t>(C_rows));
    THOR_THROW_IF_FALSE(CDimensions[1] == static_cast<uint32_t>(ld_C));
    THOR_THROW_IF_FALSE(DDimensions[0] == static_cast<uint32_t>(D_rows));
    THOR_THROW_IF_FALSE(DDimensions[1] == static_cast<uint32_t>(ld_D));

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
                                        workspace.has_value());

    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(
        operationType, A_rows, A_cols, B_rows, B_cols, ld_A, ld_B, ld_C, ld_D, transposeA, transposeB, "CublasMatrixMultiply::gemm");
    if (!workspace.has_value() && fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        throw std::runtime_error("CublasMatrixMultiply::gemm FP8 row-major path requires a workspace tensor for temporary A/B transposes.");
    }

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    auto maybeCublasKernel = CublasMatrixMultiply::instance().optimalKernels.get(cublasKernelRequirement);
    THOR_THROW_IF_FALSE(maybeCublasKernel.has_value());
    CublasKernel cublasKernel = maybeCublasKernel.value();

    // Check byte size of workspace
    if (workspace.has_value()) {
        bool kernelWillRunOnGpu;
        size_t workspaceSizeInBytes = cublasKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales);
        THOR_THROW_IF_FALSE(kernelWillRunOnGpu);

        if (workspaceSizeInBytes > 0)
            THOR_THROW_IF_FALSE(cublasKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales) <=
                                workspace.value().getDescriptor().getArraySizeInBytes());
    }

    cublasKernel.executeKernel(A, B, C, D, ld_A, ld_B, ld_C, ld_D, workspace, alpha, beta, stream, pointerMode, fp8Scales);
}

cudaDataType_t CublasMatrixMultiply::mapToCublasDataType(DataType dataType) {
    switch (dataType) {
        case DataType::FP32:
            return CUDA_R_32F;
        case DataType::BF16:
            return CUDA_R_16BF;
        case DataType::FP16:
            return CUDA_R_16F;
        case DataType::FP8_E4M3:
            return CUDA_R_8F_E4M3;
        case DataType::FP8_E5M2:
            return CUDA_R_8F_E5M2;
        case DataType::INT8:
            return CUDA_R_8I;
        default:
            THOR_UNREACHABLE();
            return CUDA_R_32F;
    }
}

std::optional<cublasComputeType_t> CublasMatrixMultiply::mapToCublasComputeType(DataType dataType) {
    switch (dataType) {
        case DataType::FP32:
            return CUBLAS_COMPUTE_32F;
        case DataType::FP16:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        case DataType::BF16:
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        case DataType::INT32:
            return CUBLAS_COMPUTE_32I;
        default:
            return std::nullopt;
    }
}

bool CublasMatrixMultiply::isSupportedMatmulDataTypes(MatmulDataTypes dataTypes) {
    switch (dataTypes.A) {
        case DataType::FP32:
        case DataType::BF16:
        case DataType::FP16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::INT8:
            break;
        default:
            return false;
    }

    switch (dataTypes.B) {
        case DataType::FP32:
        case DataType::BF16:
        case DataType::FP16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::INT8:
            break;
        default:
            return false;
    }

    switch (dataTypes.C) {
        case DataType::FP32:
        case DataType::BF16:
        case DataType::FP16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::INT8:
            break;
        default:
            return false;
    }

    switch (dataTypes.D) {
        case DataType::FP32:
        case DataType::BF16:
        case DataType::FP16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::INT8:
            break;
        default:
            return false;
    }

    const cudaDataType_t ADataType = mapToCublasDataType(dataTypes.A);
    const cudaDataType_t BDataType = mapToCublasDataType(dataTypes.B);
    const cudaDataType_t CDataType = mapToCublasDataType(dataTypes.C);
    const cudaDataType_t DDataType = mapToCublasDataType(dataTypes.D);

    const std::optional<cublasComputeType_t> computeType = mapToCublasComputeType(dataTypes.compute);
    if (!computeType.has_value()) {
        return false;
    }
    return isSupportedCublasLtOperationType(computeType.value(), CUDA_R_32F, ADataType, BDataType, CDataType, DDataType);
}

bool CublasMatrixMultiply::isSupportedSameDataTypeMatmul(DataType ABCDDataType) {
    return isSupportedMatmulDataTypes(MatmulDataTypes::same(ABCDDataType));
}

OperationType CublasMatrixMultiply::makeOperationType(DataType ABCDDataType) {
    return makeOperationType(MatmulDataTypes::same(ABCDDataType));
}

OperationType CublasMatrixMultiply::makeOperationType(MatmulDataTypes dataTypes) {
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::makeOperationType");

    const cudaDataType_t ADataType = mapToCublasDataType(dataTypes.A);
    const cudaDataType_t BDataType = mapToCublasDataType(dataTypes.B);
    const cudaDataType_t CDataType = mapToCublasDataType(dataTypes.C);
    const cudaDataType_t DDataType = mapToCublasDataType(dataTypes.D);

    const std::optional<cublasComputeType_t> computeType = mapToCublasComputeType(dataTypes.compute);
    if (!computeType.has_value()) {
        throw std::invalid_argument("Unsupported Thor compute dtype for cuBLASLt GEMM: " + dataTypeToString(dataTypes.compute));
    }

    // Thor's GEMM scale plumbing passes alpha/beta as float host/device pointers, so the cublasLt scale type is CUDA_R_32F.
    return OperationType(computeType.value(), CUDA_R_32F, ADataType, BDataType, CDataType, DDataType);
}

std::string CublasMatrixMultiply::dataTypeToString(DataType dataType) {
    switch (dataType) {
        case DataType::FP32:
            return "FP32";
        case DataType::BF16:
            return "BF16";
        case DataType::FP16:
            return "FP16";
        case DataType::FP8_E4M3:
            return "FP8_E4M3";
        case DataType::FP8_E5M2:
            return "FP8_E5M2";
        case DataType::INT8:
            return "INT8";
        case DataType::INT32:
            return "INT32";
        default:
            return "unsupported";
    }
}

std::string CublasMatrixMultiply::dataTypesToString(MatmulDataTypes dataTypes) {
    return std::string("{A=") + dataTypeToString(dataTypes.A) + ", B=" + dataTypeToString(dataTypes.B) +
           ", C=" + dataTypeToString(dataTypes.C) + ", D=" + dataTypeToString(dataTypes.D) +
           ", compute=" + dataTypeToString(dataTypes.compute) + "}";
}

bool CublasMatrixMultiply::isFp8DataType(DataType dataType) { return dataType == DataType::FP8_E4M3 || dataType == DataType::FP8_E5M2; }

bool CublasMatrixMultiply::isFp8Matmul(MatmulDataTypes dataTypes) {
    return isFp8DataType(dataTypes.A) || isFp8DataType(dataTypes.B) || isFp8DataType(dataTypes.C) || isFp8DataType(dataTypes.D);
}

bool CublasMatrixMultiply::isFp8InputsWithFp32Output(MatmulDataTypes dataTypes) {
    return (isFp8DataType(dataTypes.A) || isFp8DataType(dataTypes.B)) && dataTypes.C == DataType::FP32 && dataTypes.D == DataType::FP32;
}

bool CublasMatrixMultiply::hasRequiredExplicitFp8Scales(MatmulDataTypes dataTypes, Fp8MatmulScales fp8Scales) {
    if (!isFp8InputsWithFp32Output(dataTypes)) {
        return true;
    }

    return fp8Scales.hasAScale() && fp8Scales.hasBScale();
}

std::string CublasMatrixMultiply::unsupportedMatmulDataTypesMessage(MatmulDataTypes dataTypes, const std::string &context) {
    return context + ": unsupported cuBLASLt GEMM data type combination " + dataTypesToString(dataTypes) +
           ". Supported Thor cuBLASLt matmul/GEMM dtypes are FP32, FP16, BF16, selected INT8 paths, and selected FP8 paths. "
           "FP8 input GEMM with FP32 C/D output additionally requires explicit tensorwide FP8 scale pointers.";
}

std::string CublasMatrixMultiply::unsupportedFp8ScaleConfigurationMessage(MatmulDataTypes dataTypes,
                                                                          Fp8MatmulScales fp8Scales,
                                                                          const std::string &context) {
    std::string message =
        context + ": FP8 GEMM scale configuration is incomplete for data type combination " + dataTypesToString(dataTypes) + ".";

    if (isFp8InputsWithFp32Output(dataTypes)) {
        message +=
            " FP8 input GEMM with FP32 C/D output requires explicit tensorwide device scale pointers for A and B. "
            "Pass CublasMatrixMultiply::Fp8MatmulScales::tensorwide(aScaleDevicePtr, bScaleDevicePtr, ...) to the scale-aware "
            "heuristic GEMM/matmul API. Missing:";
        if (!fp8Scales.hasAScale()) {
            message += " A_SCALE_POINTER";
        }
        if (!fp8Scales.hasBScale()) {
            message += " B_SCALE_POINTER";
        }
        message += ".";
    }

    return message;
}

void CublasMatrixMultiply::validateMatmulDataTypesOrThrow(MatmulDataTypes dataTypes, const std::string &context) {
    if (!isSupportedMatmulDataTypes(dataTypes)) {
        throw std::invalid_argument(unsupportedMatmulDataTypesMessage(dataTypes, context));
    }
}

void CublasMatrixMultiply::validateFp8MatmulScaleConfigurationOrThrow(MatmulDataTypes dataTypes,
                                                                      Fp8MatmulScales fp8Scales,
                                                                      const std::string &context) {
    if (!hasRequiredExplicitFp8Scales(dataTypes, fp8Scales)) {
        throw std::invalid_argument(unsupportedFp8ScaleConfigurationMessage(dataTypes, fp8Scales, context));
    }
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
                                                              const DataType ABCDataType,
                                                              Stream stream) {
    multiplyUsingHeuristicKernelChoice(
        A, B, C, A_rows, A_cols, B_rows, B_cols, transposeA, transposeB, accumulate, negate, MatmulDataTypes::same(ABCDataType), stream);
}

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
                                                              const MatmulDataTypes dataTypes,
                                                              Stream stream) {
    multiplyUsingHeuristicKernelChoice(
        A, B, C, A_rows, A_cols, B_rows, B_cols, transposeA, transposeB, accumulate, negate, dataTypes, Fp8MatmulScales::none(), stream);
}

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
                                                              const MatmulDataTypes dataTypes,
                                                              const Fp8MatmulScales fp8Scales,
                                                              Stream stream) {
    const float *alpha = negate ? &ALPHA_NEGATE : &ALPHA_NO_SCALE;
    const float *beta = accumulate ? &BETA_ACCUMULATE : &BETA_CLEAR;

    gemmUsingHeuristicKernelChoice(A,
                                   B,
                                   C,
                                   C,
                                   A_rows,
                                   A_cols,
                                   B_rows,
                                   B_cols,
                                   transposeA,
                                   transposeB,
                                   false,
                                   alpha,
                                   beta,
                                   dataTypes,
                                   fp8Scales,
                                   stream,
                                   CublasScalarPointerMode::Host);
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
    const float *alpha,
    const float *beta,
    const DataType ABCDDataType,
    Stream stream,
    CublasScalarPointerMode pointerMode) {
    gemmUsingHeuristicKernelChoice(A,
                                   B,
                                   C,
                                   D,
                                   A_rows,
                                   A_cols,
                                   B_rows,
                                   B_cols,
                                   transposeA,
                                   transposeB,
                                   transposeC,
                                   alpha,
                                   beta,
                                   MatmulDataTypes::same(ABCDDataType),
                                   stream,
                                   pointerMode);
}

void CublasMatrixMultiply::gemmUsingHeuristicKernelChoice(Tensor A,
                                                          Tensor B,
                                                          Tensor C,
                                                          Tensor D,
                                                          const int32_t A_rows,
                                                          const int32_t A_cols,
                                                          const int32_t B_rows,
                                                          const int32_t B_cols,
                                                          bool transposeA,
                                                          bool transposeB,
                                                          bool transposeC,
                                                          const float *alpha,
                                                          const float *beta,
                                                          const MatmulDataTypes dataTypes,
                                                          Stream stream,
                                                          CublasScalarPointerMode pointerMode) {
    gemmUsingHeuristicKernelChoice(A,
                                   B,
                                   C,
                                   D,
                                   A_rows,
                                   A_cols,
                                   B_rows,
                                   B_cols,
                                   transposeA,
                                   transposeB,
                                   transposeC,
                                   alpha,
                                   beta,
                                   dataTypes,
                                   Fp8MatmulScales::none(),
                                   stream,
                                   pointerMode);
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
    const float *alpha,
    const float *beta,
    const MatmulDataTypes dataTypes,
    const Fp8MatmulScales fp8Scales,
    Stream stream,
    CublasScalarPointerMode pointerMode) {
    THOR_THROW_IF_FALSE(A.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(B.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(C.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(D.getDimensions().size() == 2);
    const int32_t ld_A = A.getDimensions()[1];
    const int32_t ld_B = B.getDimensions()[1];
    const int32_t ld_C = C.getDimensions()[1];
    const int32_t ld_D = D.getDimensions()[1];

    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    int32_t D_rows = C_rows;
    int32_t D_cols = C_cols;

    THOR_THROW_IF_FALSE(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    THOR_THROW_IF_FALSE(!(C == D && transposeC));

    THOR_THROW_IF_FALSE(A_rows > 0);
    THOR_THROW_IF_FALSE(A_cols > 0);
    THOR_THROW_IF_FALSE(B_rows > 0);
    THOR_THROW_IF_FALSE(B_cols > 0);
    THOR_THROW_IF_FALSE(ld_A >= A_cols);
    THOR_THROW_IF_FALSE(ld_B >= B_cols);
    THOR_THROW_IF_FALSE(ld_C >= C_cols);
    THOR_THROW_IF_FALSE(ld_D >= D_cols);
    THOR_THROW_IF_FALSE(!(C == D) || dataTypes.C == dataTypes.D);
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::gemmUsingHeuristicKernelChoice");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, fp8Scales, "CublasMatrixMultiply::gemmUsingHeuristicKernelChoice");
    THOR_THROW_IF_FALSE(A.getDescriptor().getDataType() == dataTypes.A);
    THOR_THROW_IF_FALSE(B.getDescriptor().getDataType() == dataTypes.B);
    THOR_THROW_IF_FALSE(C.getDescriptor().getDataType() == dataTypes.C);
    THOR_THROW_IF_FALSE(D.getDescriptor().getDataType() == dataTypes.D);
    // Check dimensions of tensors
    vector<unsigned long> ADimensions = A.getDescriptor().getDimensions();
    vector<unsigned long> BDimensions = B.getDescriptor().getDimensions();
    vector<unsigned long> CDimensions = C.getDescriptor().getDimensions();
    vector<unsigned long> DDimensions = D.getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(ADimensions.size() == 2);
    THOR_THROW_IF_FALSE(ADimensions[0] == static_cast<uint32_t>(A_rows));
    THOR_THROW_IF_FALSE(ADimensions[1] == static_cast<uint32_t>(ld_A));
    THOR_THROW_IF_FALSE(BDimensions[0] == static_cast<uint32_t>(B_rows));
    THOR_THROW_IF_FALSE(BDimensions[1] == static_cast<uint32_t>(ld_B));
    THOR_THROW_IF_FALSE(CDimensions[0] == static_cast<uint32_t>(C_rows));
    THOR_THROW_IF_FALSE(CDimensions[1] == static_cast<uint32_t>(ld_C));
    THOR_THROW_IF_FALSE(DDimensions[0] == static_cast<uint32_t>(D_rows));
    THOR_THROW_IF_FALSE(DDimensions[1] == static_cast<uint32_t>(ld_D));

    ScopedGpu scopedGpu(stream.getGpuNum());

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ADesc;
    cublasLtMatrixLayout_t BDesc;
    cublasLtMatrixLayout_t CDesc;
    cublasLtMatrixLayout_t DDesc;

    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
                                                 A_rows,
                                                 A_cols,
                                                 B_rows,
                                                 B_cols,
                                                 ld_A,
                                                 ld_B,
                                                 ld_C,
                                                 ld_D,
                                                 transposeA,
                                                 transposeB,
                                                 "CublasMatrixMultiply::gemmUsingHeuristicKernelChoice");
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, operationType.computeDataType, operationType.scaleDataType));
    const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
    const cublasLtPointerMode_t cublasPointerMode =
        (pointerMode == CublasScalarPointerMode::Device) ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, pointerModeAttribute, &cublasPointerMode, sizeof(cublasPointerMode)));
    configureTensorwideFp8Scales(operationDesc, operationType, fp8Scales);
    if (fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        throw std::runtime_error(
            "CublasMatrixMultiply::gemmUsingHeuristicKernelChoice FP8 row-major path requires chooseOptimalGemmKernel/gemm with workspace "
            "when temporary A/B transposes are needed.");
    }
    configureLtOperationTransposes(operationDesc, operationType, transposeA, transposeB, transposeC);
    createLtMatrixLayoutsForRowMajorGemm(&ADesc,
                                         &BDesc,
                                         &CDesc,
                                         &DDesc,
                                         operationType,
                                         A_rows,
                                         A_cols,
                                         B_rows,
                                         B_cols,
                                         C_rows,
                                         C_cols,
                                         D_rows,
                                         D_cols,
                                         ld_A,
                                         ld_B,
                                         ld_C,
                                         ld_D,
                                         transposeA,
                                         transposeB);

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

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    const void *ltA = usesFp8ColumnMajorLtPath(operationType) ? B.getMemPtr() : A.getMemPtr();
    const void *ltB = usesFp8ColumnMajorLtPath(operationType) ? A.getMemPtr() : B.getMemPtr();

    // If there is already a known kernel, use it. Otherwise, a heuristic search will be performed and the kernel remembered.
    if (auto optimalKernel = CublasMatrixMultiply::instance().optimalKernels.get(cublasKernelRequirement); optimalKernel.has_value()) {
        cublasLtMatmulAlgo_t algorithm = optimalKernel->getAlgorithm(stream.getGpuNum());

        CHECK_CUBLAS(cublasLtMatmul(stream.getCublasLtHandle(),
                                    operationDesc,
                                    alpha,
                                    ltA,
                                    ADesc,
                                    ltB,
                                    BDesc,
                                    beta,
                                    C.getMemPtr(),
                                    CDesc,
                                    D.getMemPtr(),
                                    DDesc,
                                    &algorithm,
                                    nullptr,
                                    0,
                                    stream));

        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(DDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(CDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(BDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(ADesc));
        CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));

        return;
    } else if (auto heuristicAlgorithm = CublasMatrixMultiply::instance().knownHeuristicAlgorithms.get(cublasKernelRequirement);
               heuristicAlgorithm.has_value()) {
        cublasLtMatmulAlgo_t algorithm = *heuristicAlgorithm;

        CHECK_CUBLAS(cublasLtMatmul(stream.getCublasLtHandle(),
                                    operationDesc,
                                    alpha,
                                    ltA,
                                    ADesc,
                                    ltB,
                                    BDesc,
                                    beta,
                                    C.getMemPtr(),
                                    CDesc,
                                    D.getMemPtr(),
                                    DDesc,
                                    &algorithm,
                                    nullptr,
                                    0,
                                    stream));

        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(DDesc));

        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(CDesc));

        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(BDesc));

        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(ADesc));

        CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));

        return;
    }

    cublasLtMatmulPreference_t searchPreferences;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&searchPreferences));

    CHECK_CUBLAS(cublasLtMatmulPreferenceInit(searchPreferences));

    // cublasLtMatmulPreferenceAttributes_t attribute = CUBLASLT_MATMUL_PREF_IMPL_MASK;
    // cublasLtNumericalImplFlags_t computeType = CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK |
    // CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F; CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(searchPreferences, attribute,

    int returnedAlgoCount;
    vector<cublasLtMatmulHeuristicResult_t> results(30);
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(stream.getCublasLtHandle(),
                                                operationDesc,
                                                ADesc,
                                                BDesc,
                                                CDesc,
                                                DDesc,
                                                searchPreferences,
                                                30,
                                                results.data(),
                                                &returnedAlgoCount));

    results.resize(returnedAlgoCount);

    // Algorithms aren't guaranteed to run, so find the first one that does and then return.
    bool kernelLaunchedSuccessfully = false;
    for (int i = 0; i < returnedAlgoCount && !kernelLaunchedSuccessfully; ++i) {
        // have seen kernels that say wavesCount == 0 that sporadically fail.
        if (!(results[i].wavesCount > 0.0f))
            continue;
        cublasStatus_t cublasStatus = cublasLtMatmul(stream.getCublasLtHandle(),
                                                     operationDesc,
                                                     alpha,
                                                     ltA,
                                                     ADesc,
                                                     ltB,
                                                     BDesc,
                                                     beta,
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
            CublasMatrixMultiply::instance().knownHeuristicAlgorithms.put(cublasKernelRequirement, results[i].algo);
        }
    }

    if (!kernelLaunchedSuccessfully) {
        results = vector<cublasLtMatmulHeuristicResult_t>(10000);
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(stream.getCublasLtHandle(),
                                                    operationDesc,
                                                    ADesc,
                                                    BDesc,
                                                    CDesc,
                                                    DDesc,
                                                    searchPreferences,
                                                    10000,
                                                    results.data(),
                                                    &returnedAlgoCount));

        results.resize(returnedAlgoCount);

        for (int i = 0; i < returnedAlgoCount && !kernelLaunchedSuccessfully; ++i) {
            cublasStatus_t cublasStatus = cublasLtMatmul(stream.getCublasLtHandle(),
                                                         operationDesc,
                                                         alpha,
                                                         ltA,
                                                         ADesc,
                                                         ltB,
                                                         BDesc,
                                                         beta,
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
                CublasMatrixMultiply::instance().knownHeuristicAlgorithms.put(cublasKernelRequirement, results[i].algo);
            }
        }
    }

    THOR_THROW_IF_FALSE(kernelLaunchedSuccessfully);

    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(searchPreferences));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(DDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(CDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(BDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(ADesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}

static cublasLtEpilogue_t toCublasLtBackwardEpilogue(CublasMatrixMultiply::BackwardEpilogueFusion epilogue, bool hasBgrad);

static cublasLtEpilogue_t toCublasLtEpilogue(CublasMatrixMultiply::EpilogueFusion epilogue, bool hasBias) {
    switch (epilogue) {
        case CublasMatrixMultiply::EpilogueFusion::Default:
            return hasBias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
        case CublasMatrixMultiply::EpilogueFusion::Relu:
            return hasBias ? static_cast<cublasLtEpilogue_t>(CUBLASLT_EPILOGUE_RELU | CUBLASLT_EPILOGUE_BIAS) : CUBLASLT_EPILOGUE_RELU;
        case CublasMatrixMultiply::EpilogueFusion::Gelu:
            return hasBias ? static_cast<cublasLtEpilogue_t>(CUBLASLT_EPILOGUE_GELU | CUBLASLT_EPILOGUE_BIAS) : CUBLASLT_EPILOGUE_GELU;
    }
    throw std::runtime_error("Unknown cuBLASLt GEMM epilogue fusion kind.");
}

std::shared_ptr<CublasMatrixMultiply::LtMatmulPlan> CublasMatrixMultiply::buildGemmWithEpiloguePlan(
    int gpuNum,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    const int32_t ld_A,
    const int32_t ld_B,
    const int32_t ld_C,
    const int32_t ld_D,
    bool transposeA,
    bool transposeB,
    const MatmulDataTypes dataTypes,
    EpilogueFusion epilogue,
    std::optional<Tensor> addend,
    bool addendIsBiasVector,
    std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> selectedAlgorithm,
    std::optional<uint64_t> maxWorkspaceSizeInBytes) {
    if (!addend.has_value() && epilogue == EpilogueFusion::Default) {
        throw std::runtime_error("CublasMatrixMultiply::buildGemmWithEpiloguePlan requires an addend or a non-default epilogue.");
    }
    if (transposeA || transposeB) {
        throw std::runtime_error("CublasMatrixMultiply::buildGemmWithEpiloguePlan currently supports only non-transposed row-major GEMM epilogues.");
    }

    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::buildGemmWithEpiloguePlan");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, Fp8MatmulScales::none(), "CublasMatrixMultiply::buildGemmWithEpiloguePlan");

    ScopedGpu scopedGpu(gpuNum);
    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
                                                 A_rows,
                                                 A_cols,
                                                 B_rows,
                                                 B_cols,
                                                 ld_A,
                                                 ld_B,
                                                 ld_C,
                                                 ld_D,
                                                 transposeA,
                                                 transposeB,
                                                 "CublasMatrixMultiply::buildGemmWithEpiloguePlan");
    if (fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        throw std::runtime_error("CublasMatrixMultiply::buildGemmWithEpiloguePlan FP8 row-major path requires non-epilogue workspace support.");
    }

    auto selection = selectedAlgorithm.has_value()
                         ? selectedAlgorithm
                         : selectGemmWithEpilogueAlgorithm(gpuNum,
                                                           A_rows,
                                                           A_cols,
                                                           B_rows,
                                                           B_cols,
                                                           ld_A,
                                                           ld_B,
                                                           ld_C,
                                                           ld_D,
                                                           transposeA,
                                                           transposeB,
                                                           dataTypes,
                                                           epilogue,
                                                           addend.has_value(),
                                                           addendIsBiasVector,
                                                           maxWorkspaceSizeInBytes);
    if (!selection.has_value()) {
        return nullptr;
    }

    const int32_t D_rows = A_rows;
    const int32_t D_cols = B_cols;
    const cublasLtEpilogue_t cublasEpilogue = toCublasLtEpilogue(epilogue, addendIsBiasVector);
    auto plan = std::make_shared<LtMatmulPlan>();
    plan->algorithm = selection.value();

    auto createOperationDesc = [&](cublasLtPointerMode_t pointerMode, cublasLtMatmulDesc_t *desc) {
        CHECK_CUBLAS(cublasLtMatmulDescCreate(desc, operationType.computeDataType, operationType.scaleDataType));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode)));
        configureTensorwideFp8Scales(*desc, operationType, Fp8MatmulScales::none());
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &cublasEpilogue, sizeof(cublasEpilogue)));
        if (addendIsBiasVector) {
            void *biasPtr = addend.value().getMemPtr();
            CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)));
        }
    };
    createOperationDesc(CUBLASLT_POINTER_MODE_HOST, &plan->operation_desc_host);
    createOperationDesc(CUBLASLT_POINTER_MODE_DEVICE, &plan->operation_desc_device);

    const cublasLtOrder_t columnMajorOrder = CUBLASLT_ORDER_COL;
    int64_t ld = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->a_desc, operationType.BDataType, B_cols, B_rows, ld_B));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_B;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->a_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->b_desc, operationType.ADataType, A_cols, A_rows, ld_A));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_A;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->b_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    const cudaDataType_t cDescType = addend.has_value() && !addendIsBiasVector ? operationType.CDataType : operationType.DDataType;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->c_desc, cDescType, D_cols, D_rows, ld_C));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_C;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->c_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->d_desc, operationType.DDataType, D_cols, D_rows, ld_D));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_D;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->d_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
    return plan;
}

void CublasMatrixMultiply::runGemmWithEpiloguePlan(Tensor A,
                                                   Tensor B,
                                                   std::optional<Tensor> addend,
                                                   Tensor D,
                                                   const float *alpha,
                                                   const float *beta,
                                                   Stream stream,
                                                   CublasScalarPointerMode pointerMode,
                                                   std::optional<Tensor> workspace,
                                                   const std::shared_ptr<LtMatmulPlan>& plan,
                                                   bool addendIsBiasVector) {
    if (!plan) {
        throw std::runtime_error("CublasMatrixMultiply::runGemmWithEpiloguePlan called with null plan.");
    }
    plan->runGemmWithEpilogue(A, B, addend, D, alpha, beta, stream, pointerMode, workspace, addendIsBiasVector);
}

std::shared_ptr<CublasMatrixMultiply::LtMatmulPlan> CublasMatrixMultiply::buildGemmWithBackwardEpiloguePlan(
    int gpuNum,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    const int32_t ld_A,
    const int32_t ld_B,
    const int32_t ld_C,
    const int32_t ld_D,
    bool transposeA,
    bool transposeB,
    const MatmulDataTypes dataTypes,
    BackwardEpilogueFusion epilogue,
    bool hasAddend,
    Tensor epilogueAux,
    std::optional<Tensor> biasGradient,
    std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> selectedAlgorithm,
    std::optional<uint64_t> maxWorkspaceSizeInBytes) {
    if (transposeA || transposeB) {
        throw std::runtime_error("CublasMatrixMultiply::buildGemmWithBackwardEpiloguePlan currently supports only non-transposed row-major GEMM epilogues.");
    }
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::buildGemmWithBackwardEpiloguePlan");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, Fp8MatmulScales::none(), "CublasMatrixMultiply::buildGemmWithBackwardEpiloguePlan");

    ScopedGpu scopedGpu(gpuNum);
    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
                                                 A_rows,
                                                 A_cols,
                                                 B_rows,
                                                 B_cols,
                                                 ld_A,
                                                 ld_B,
                                                 ld_C,
                                                 ld_D,
                                                 transposeA,
                                                 transposeB,
                                                 "CublasMatrixMultiply::buildGemmWithBackwardEpiloguePlan");
    if (fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        throw std::runtime_error("CublasMatrixMultiply::buildGemmWithBackwardEpiloguePlan FP8 row-major path requires non-epilogue workspace support.");
    }

    const int64_t epilogueAuxLd = static_cast<int64_t>(epilogueAux.getDimensions()[1]);
    auto selection = selectedAlgorithm.has_value()
                         ? selectedAlgorithm
                         : selectGemmWithBackwardEpilogueAlgorithm(gpuNum,
                                                                   A_rows,
                                                                   A_cols,
                                                                   B_rows,
                                                                   B_cols,
                                                                   ld_A,
                                                                   ld_B,
                                                                   ld_C,
                                                                   ld_D,
                                                                   transposeA,
                                                                   transposeB,
                                                                   dataTypes,
                                                                   epilogue,
                                                                   hasAddend,
                                                                   biasGradient.has_value(),
                                                                   epilogueAuxLd,
                                                                   maxWorkspaceSizeInBytes);
    if (!selection.has_value()) {
        return nullptr;
    }

    const int32_t D_rows = A_rows;
    const int32_t D_cols = B_cols;
    const cublasLtEpilogue_t cublasEpilogue = toCublasLtBackwardEpilogue(epilogue, biasGradient.has_value());
    auto plan = std::make_shared<LtMatmulPlan>();
    plan->algorithm = selection.value();

    auto createOperationDesc = [&](cublasLtPointerMode_t pointerMode, cublasLtMatmulDesc_t *desc) {
        CHECK_CUBLAS(cublasLtMatmulDescCreate(desc, operationType.computeDataType, operationType.scaleDataType));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode)));
        configureTensorwideFp8Scales(*desc, operationType, Fp8MatmulScales::none());
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &cublasEpilogue, sizeof(cublasEpilogue)));
        const void *auxPtr = epilogueAux.getMemPtr();
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &auxPtr, sizeof(auxPtr)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &epilogueAuxLd, sizeof(epilogueAuxLd)));
        if (biasGradient.has_value()) {
            void *biasPtr = biasGradient.value().getMemPtr();
            CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)));
        }
    };
    createOperationDesc(CUBLASLT_POINTER_MODE_HOST, &plan->operation_desc_host);
    createOperationDesc(CUBLASLT_POINTER_MODE_DEVICE, &plan->operation_desc_device);

    const cublasLtOrder_t columnMajorOrder = CUBLASLT_ORDER_COL;
    int64_t ld = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->a_desc, operationType.BDataType, B_cols, B_rows, ld_B));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_B;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->a_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->b_desc, operationType.ADataType, A_cols, A_rows, ld_A));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_A;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->b_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    const cudaDataType_t cDescType = hasAddend ? operationType.CDataType : operationType.DDataType;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->c_desc, cDescType, D_cols, D_rows, ld_C));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_C;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->c_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->d_desc, operationType.DDataType, D_cols, D_rows, ld_D));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_D;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(plan->d_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
    return plan;
}

void CublasMatrixMultiply::runGemmWithBackwardEpiloguePlan(Tensor A,
                                                           Tensor B,
                                                           std::optional<Tensor> addend,
                                                           Tensor epilogueAux,
                                                           Tensor D,
                                                           const float *alpha,
                                                           const float *beta,
                                                           Stream stream,
                                                           CublasScalarPointerMode pointerMode,
                                                           std::optional<Tensor> workspace,
                                                           const std::shared_ptr<LtMatmulPlan>& plan) {
    (void)epilogueAux;
    if (!plan) {
        throw std::runtime_error("CublasMatrixMultiply::runGemmWithBackwardEpiloguePlan called with null plan.");
    }
    plan->runGemmWithBackwardEpilogue(A, B, addend, D, alpha, beta, stream, pointerMode, workspace);
}

void CublasMatrixMultiply::gemmWithEpilogueUsingHeuristicKernelChoice(
    Tensor A,
    Tensor B,
    std::optional<Tensor> addend,
    Tensor D,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    bool transposeA,
    bool transposeB,
    const float *alpha,
    const float *beta,
    const MatmulDataTypes dataTypes,
    EpilogueFusion epilogue,
    bool addendIsBiasVector,
    Stream stream,
    CublasScalarPointerMode pointerMode,
    std::optional<Tensor> workspace,
    std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> selectedAlgorithm) {
    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    const int32_t D_rows = C_rows;
    const int32_t D_cols = C_cols;
    const int32_t ld_A = A.getDimensions()[1];
    const int32_t ld_B = B.getDimensions()[1];
    const int32_t ld_D = D.getDimensions()[1];
    const int32_t ld_C = (addend.has_value() && !addendIsBiasVector) ? addend.value().getDimensions()[1] : ld_D;

    THOR_THROW_IF_FALSE(A.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(B.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(D.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(A.getDimensions()[0] == static_cast<uint32_t>(A_rows));
    THOR_THROW_IF_FALSE(A.getDimensions()[1] == static_cast<uint32_t>(ld_A));
    THOR_THROW_IF_FALSE(B.getDimensions()[0] == static_cast<uint32_t>(B_rows));
    THOR_THROW_IF_FALSE(B.getDimensions()[1] == static_cast<uint32_t>(ld_B));
    THOR_THROW_IF_FALSE(D.getDimensions()[0] == static_cast<uint32_t>(D_rows));
    THOR_THROW_IF_FALSE(D.getDimensions()[1] == static_cast<uint32_t>(ld_D));

    if (addendIsBiasVector) {
        THOR_THROW_IF_FALSE(addend.has_value());
        THOR_THROW_IF_FALSE(addend.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(addend.value().getDimensions()[0] == static_cast<uint64_t>(D_cols));
        THOR_THROW_IF_FALSE(addend.value().getDescriptor().getDataType() == D.getDescriptor().getDataType());
    } else if (addend.has_value()) {
        THOR_THROW_IF_FALSE(addend.value().getDimensions().size() == 2);
        THOR_THROW_IF_FALSE(addend.value().getDimensions()[0] == static_cast<uint64_t>(D_rows));
        THOR_THROW_IF_FALSE(addend.value().getDimensions()[1] == static_cast<uint64_t>(ld_C));
    }

    const std::optional<uint64_t> workspaceLimitBytes = workspace.has_value()
                                                            ? std::optional<uint64_t>(workspace.value().getArraySizeInBytes())
                                                            : std::optional<uint64_t>(0);
    auto plan = buildGemmWithEpiloguePlan(stream.getGpuNum(),
                                          A_rows,
                                          A_cols,
                                          B_rows,
                                          B_cols,
                                          ld_A,
                                          ld_B,
                                          ld_C,
                                          ld_D,
                                          transposeA,
                                          transposeB,
                                          dataTypes,
                                          epilogue,
                                          addend,
                                          addendIsBiasVector,
                                          selectedAlgorithm,
                                          workspaceLimitBytes);
    if (!plan) {
        throw std::runtime_error("Unable to build cuBLASLt epilogue GEMM plan for legacy heuristic wrapper.");
    }
    plan->runGemmWithEpilogue(A, B, addend, D, alpha, beta, stream, pointerMode, workspace, addendIsBiasVector);
}

void CublasMatrixMultiply::gemmWithBiasUsingHeuristicKernelChoice(Tensor A,
                                                                  Tensor B,
                                                                  Tensor bias,
                                                                  Tensor D,
                                                                  const int32_t A_rows,
                                                                  const int32_t A_cols,
                                                                  const int32_t B_rows,
                                                                  const int32_t B_cols,
                                                                  bool transposeA,
                                                                  bool transposeB,
                                                                  const float *alpha,
                                                                  const MatmulDataTypes dataTypes,
                                                                  Stream stream,
                                                                  CublasScalarPointerMode pointerMode,
                                                                  std::optional<Tensor> workspace) {
    const float betaOne = 1.0f;
    gemmWithEpilogueUsingHeuristicKernelChoice(A,
                                               B,
                                               bias,
                                               D,
                                               A_rows,
                                               A_cols,
                                               B_rows,
                                               B_cols,
                                               transposeA,
                                               transposeB,
                                               alpha,
                                               &betaOne,
                                               dataTypes,
                                               EpilogueFusion::Default,
                                               true,
                                               stream,
                                               pointerMode,
                                               workspace);
}

static cublasLtEpilogue_t toCublasLtBackwardEpilogue(CublasMatrixMultiply::BackwardEpilogueFusion epilogue, bool hasBgrad) {
    switch (epilogue) {
        case CublasMatrixMultiply::BackwardEpilogueFusion::DRelu:
            return hasBgrad ? static_cast<cublasLtEpilogue_t>(CUBLASLT_EPILOGUE_DRELU | 16) : CUBLASLT_EPILOGUE_DRELU;
        case CublasMatrixMultiply::BackwardEpilogueFusion::DGelu:
            return hasBgrad ? static_cast<cublasLtEpilogue_t>(CUBLASLT_EPILOGUE_DGELU | 16) : CUBLASLT_EPILOGUE_DGELU;
    }
    throw std::runtime_error("Unknown cuBLASLt GEMM backward epilogue fusion kind.");
}

static uint64_t cublasLtEpilogueWorkspacePolicyLimitBytes(int gpuNum) {
    constexpr uint64_t EPILOGUE_WORKSPACE_ABSOLUTE_CAP_BYTES = 1ull * 1024ull * 1024ull * 1024ull;
    constexpr uint64_t EPILOGUE_WORKSPACE_TOTAL_MEM_DIVISOR = 20ull;  // At most 5% of device memory.
    constexpr uint64_t EPILOGUE_WORKSPACE_FREE_MEM_DIVISOR = 4ull;    // Leave at least 75% of currently-free memory alone.

    uint64_t maxWorkspaceBytes = EPILOGUE_WORKSPACE_ABSOLUTE_CAP_BYTES;
    const uint64_t totalMemBytes = static_cast<uint64_t>(MachineEvaluator::instance().getTotalGlobalMemBytes(gpuNum));
    const uint64_t freeMemBytes = static_cast<uint64_t>(MachineEvaluator::instance().getFreeMemBytes(gpuNum));
    if (totalMemBytes > 0) {
        maxWorkspaceBytes = std::min(maxWorkspaceBytes, totalMemBytes / EPILOGUE_WORKSPACE_TOTAL_MEM_DIVISOR);
    }
    if (freeMemBytes > 0) {
        maxWorkspaceBytes = std::min(maxWorkspaceBytes, freeMemBytes / EPILOGUE_WORKSPACE_FREE_MEM_DIVISOR);
    }
    return maxWorkspaceBytes;
}

static uint64_t checkedMatrixBytes(int64_t rows, int64_t cols, cudaDataType_t dataType, const char *what) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error(std::string(what) + " has non-positive dimensions.");
    }
    const unsigned __int128 bytes = static_cast<unsigned __int128>(static_cast<uint64_t>(rows)) *
                                    static_cast<unsigned __int128>(static_cast<uint64_t>(cols)) *
                                    static_cast<unsigned __int128>(cudaDataTypeSizeInBytes(dataType));
    if (bytes > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(std::string(what) + " byte size overflows uint64_t.");
    }
    return static_cast<uint64_t>(bytes);
}

static uint64_t checkedVectorBytes(int64_t elements, cudaDataType_t dataType, const char *what) {
    if (elements <= 0) {
        throw std::runtime_error(std::string(what) + " has non-positive dimensions.");
    }
    const unsigned __int128 bytes =
        static_cast<unsigned __int128>(static_cast<uint64_t>(elements)) * static_cast<unsigned __int128>(cudaDataTypeSizeInBytes(dataType));
    if (bytes > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(std::string(what) + " byte size overflows uint64_t.");
    }
    return static_cast<uint64_t>(bytes);
}

static uint64_t checkedAddBytes(uint64_t lhs, uint64_t rhs, const char *what) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        throw std::runtime_error(std::string(what) + " byte size overflows uint64_t.");
    }
    return lhs + rhs;
}

static uint64_t cublasLtForwardEpilogueOperationBytes(const OperationType &operationType,
                                                      int32_t A_rows,
                                                      int32_t A_cols,
                                                      int32_t B_rows,
                                                      int32_t B_cols,
                                                      int32_t D_rows,
                                                      int32_t D_cols,
                                                      bool hasAddend,
                                                      bool addendIsBiasVector) {
    uint64_t bytes = 0;
    bytes = checkedAddBytes(
        bytes, checkedMatrixBytes(A_rows, A_cols, operationType.ADataType, "forward epilogue A"), "forward epilogue operation");
    bytes = checkedAddBytes(
        bytes, checkedMatrixBytes(B_rows, B_cols, operationType.BDataType, "forward epilogue B"), "forward epilogue operation");
    bytes = checkedAddBytes(
        bytes, checkedMatrixBytes(D_rows, D_cols, operationType.DDataType, "forward epilogue D"), "forward epilogue operation");
    if (hasAddend) {
        bytes = checkedAddBytes(bytes,
                                addendIsBiasVector
                                    ? checkedVectorBytes(D_cols, operationType.DDataType, "forward epilogue bias addend")
                                    : checkedMatrixBytes(D_rows, D_cols, operationType.CDataType, "forward epilogue matrix addend"),
                                "forward epilogue operation");
    }
    return bytes;
}

static uint64_t cublasLtBackwardEpilogueOperationBytes(const OperationType &operationType,
                                                       int32_t A_rows,
                                                       int32_t A_cols,
                                                       int32_t B_rows,
                                                       int32_t B_cols,
                                                       int32_t D_rows,
                                                       int32_t D_cols,
                                                       int64_t epilogueAuxLd,
                                                       bool hasAddend,
                                                       bool hasBiasGradient) {
    uint64_t bytes = 0;
    bytes = checkedAddBytes(
        bytes, checkedMatrixBytes(A_rows, A_cols, operationType.ADataType, "backward epilogue A"), "backward epilogue operation");
    bytes = checkedAddBytes(
        bytes, checkedMatrixBytes(B_rows, B_cols, operationType.BDataType, "backward epilogue B"), "backward epilogue operation");
    bytes = checkedAddBytes(
        bytes, checkedMatrixBytes(D_rows, D_cols, operationType.DDataType, "backward epilogue D"), "backward epilogue operation");
    if (hasAddend) {
        bytes = checkedAddBytes(
            bytes, checkedMatrixBytes(D_rows, D_cols, operationType.CDataType, "backward epilogue addend"), "backward epilogue operation");
    }
    if (epilogueAuxLd > 0) {
        bytes = checkedAddBytes(bytes,
                                checkedMatrixBytes(D_rows, epilogueAuxLd, operationType.DDataType, "backward epilogue aux"),
                                "backward epilogue operation");
    }
    if (hasBiasGradient) {
        bytes = checkedAddBytes(
            bytes, checkedVectorBytes(D_cols, operationType.DDataType, "backward epilogue bias gradient"), "backward epilogue operation");
    }
    return bytes;
}

static uint64_t cublasLtWorkspaceInsignificantThresholdBytes(uint64_t operationBytes) {
    constexpr uint64_t MIN_SIGNIFICANT_WORKSPACE_BYTES = 16ull * 1024ull * 1024ull;
    constexpr uint64_t OPERATION_BYTES_DIVISOR = 16ull;  // Workspace <= 6.25% of matrix traffic is small enough.
    return std::max(MIN_SIGNIFICANT_WORKSPACE_BYTES, operationBytes / OPERATION_BYTES_DIVISOR);
}

static bool cublasLtWorkspaceIsSignificant(uint64_t workspaceBytes, uint64_t operationBytes) {
    return workspaceBytes > cublasLtWorkspaceInsignificantThresholdBytes(operationBytes);
}

static double cublasLtWorkspaceTradeSlowdownBudgetFraction(uint64_t preferredWorkspaceBytes,
                                                           uint64_t candidateWorkspaceBytes,
                                                           uint64_t operationBytes) {
    if (operationBytes == 0 || candidateWorkspaceBytes >= preferredWorkspaceBytes) {
        return 0.0;
    }

    // Treat workspace savings as valuable only in proportion to the size of
    // the operation being planned.  This intentionally caps the allowed
    // slowdown: a slower kernel may buy memory only when the memory saving is
    // large relative to the matrices involved, and even then it cannot give up
    // more than a small fraction of measured runtime.
    constexpr double MAX_WORKSPACE_TRADE_SLOWDOWN_FRACTION = 0.05;  // % max slowdown.
    constexpr double WORKSPACE_SAVINGS_TO_SLOWDOWN_DIVISOR = 4.0;
    const double savedWorkspaceBytes = static_cast<double>(preferredWorkspaceBytes - candidateWorkspaceBytes);
    const double savedVsOperation = savedWorkspaceBytes / static_cast<double>(operationBytes);
    return std::min(MAX_WORKSPACE_TRADE_SLOWDOWN_FRACTION, savedVsOperation / WORKSPACE_SAVINGS_TO_SLOWDOWN_DIVISOR);
}

static bool cublasLtMeasuredWorkspaceTradeIsWorthIt(double preferredTimeMilliseconds,
                                                    uint64_t preferredWorkspaceBytes,
                                                    double candidateTimeMilliseconds,
                                                    uint64_t candidateWorkspaceBytes,
                                                    uint64_t operationBytes) {
    constexpr double WORSPACE_SIZE_PROPORTION_BUYDOWN_MAX = 0.6667;

    if (candidateWorkspaceBytes >= preferredWorkspaceBytes) {
        return false;
    }

    // If timings are effectively tied, choose the lower-workspace candidate.
    // CUDA-event measurements have enough noise that forcing a strict runtime
    // ordering here would make the selector keep large scratch for no stable
    // speed benefit.
    constexpr double MEASURED_TIME_EQUIVALENT_FRACTION = 0.01;  // 1% tie band.
    if (candidateTimeMilliseconds <= preferredTimeMilliseconds * (1.0 + MEASURED_TIME_EQUIVALENT_FRACTION)) {
        return true;
    }

    if (!cublasLtWorkspaceIsSignificant(preferredWorkspaceBytes, operationBytes) ||
        candidateWorkspaceBytes >= preferredWorkspaceBytes * WORSPACE_SIZE_PROPORTION_BUYDOWN_MAX) {
        return false;
    }

    const double slowdownFraction = (candidateTimeMilliseconds - preferredTimeMilliseconds) / std::max(preferredTimeMilliseconds, 1.0e-9);
    return slowdownFraction <=
           cublasLtWorkspaceTradeSlowdownBudgetFraction(preferredWorkspaceBytes, candidateWorkspaceBytes, operationBytes);
}

static CublasKernel selectMeasuredKernelWithWorkspacePolicy(std::vector<CublasKernel> kernels,
                                                            int gpuNum,
                                                            uint64_t operationBytes,
                                                            CublasMatrixMultiply::Fp8MatmulScales fp8Scales) {
    THOR_THROW_IF_FALSE(!kernels.empty());

    std::sort(kernels.begin(), kernels.end(), CublasKernel::executionTimeComparison);

    std::optional<CublasKernel> preferred;
    uint64_t preferredWorkspaceBytes = 0;
    double preferredTimeMilliseconds = 0.0;
    for (CublasKernel &candidate : kernels) {
        if (candidate.getErrorFlag()) {
            continue;
        }

        if (preferred.has_value() && !cublasLtWorkspaceIsSignificant(preferredWorkspaceBytes, operationBytes)) {
            break;
        }

        bool kernelWillRunOnGpu = false;
        const uint64_t candidateWorkspaceBytes = candidate.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales);
        if (!kernelWillRunOnGpu) {
            continue;
        }
        const double candidateTimeMilliseconds = candidate.getAverageRunTimeMilliseconds();

        // Keep measured performance order as the primary ranking.  Move to a
        // lower-ranked kernel only when the workspace saving is worth the
        // measured runtime cost.  This prevents the selector from taking a
        // large slowdown just because a later candidate cuts scratch in half,
        // while still preferring lower-memory kernels when performance is tied
        // or the memory saving is large enough to justify a bounded slowdown.
        if (!preferred.has_value() ||
            cublasLtMeasuredWorkspaceTradeIsWorthIt(
                preferredTimeMilliseconds, preferredWorkspaceBytes, candidateTimeMilliseconds, candidateWorkspaceBytes, operationBytes)) {
            preferred = candidate;
            preferredWorkspaceBytes = candidateWorkspaceBytes;
            preferredTimeMilliseconds = candidateTimeMilliseconds;
        }
    }

    if (!preferred.has_value()) {
        throw std::runtime_error("CublasMatrixMultiply::chooseOptimalGemmKernel could not select any verified cuBLASLt kernel candidate.");
    }

    return preferred.value();
}

struct LtMeasuredMatmulAlgorithm {
    CublasMatrixMultiply::LtMatmulAlgorithmSelection selection;
    RunStats runStats;
    bool errorFlag = false;
    float wavesCount = 0.0f;
};

static bool ltMeasuredMatmulExecutionTimeComparison(LtMeasuredMatmulAlgorithm &lhs, LtMeasuredMatmulAlgorithm &rhs) {
    if (lhs.errorFlag)
        return false;
    if (rhs.errorFlag)
        return true;
    return lhs.runStats < rhs.runStats;
}

static CublasMatrixMultiply::LtMatmulAlgorithmSelection selectMeasuredLtMatmulAlgorithmWithWorkspacePolicy(
    std::vector<LtMeasuredMatmulAlgorithm> candidates, uint64_t operationBytes, const std::string &context) {
    THOR_THROW_IF_FALSE(!candidates.empty());

    std::sort(candidates.begin(), candidates.end(), ltMeasuredMatmulExecutionTimeComparison);

    std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> preferred;
    double preferredTimeMilliseconds = 0.0;
    for (LtMeasuredMatmulAlgorithm &candidate : candidates) {
        if (candidate.errorFlag) {
            continue;
        }

        if (preferred.has_value() && !cublasLtWorkspaceIsSignificant(preferred->workspace_size_in_bytes, operationBytes)) {
            break;
        }

        const double candidateTimeMilliseconds = candidate.runStats.getAverageRunTimeMilliseconds();

        // Same policy as the regular non-epilogue GEMM/matmul picker: measured
        // runtime is primary, but a lower-workspace candidate may replace the
        // current winner when the saving is worth the bounded measured slowdown.
        if (!preferred.has_value() || cublasLtMeasuredWorkspaceTradeIsWorthIt(preferredTimeMilliseconds,
                                                                              preferred->workspace_size_in_bytes,
                                                                              candidateTimeMilliseconds,
                                                                              candidate.selection.workspace_size_in_bytes,
                                                                              operationBytes)) {
            preferred = candidate.selection;
            preferredTimeMilliseconds = candidateTimeMilliseconds;
        }
    }

    if (!preferred.has_value()) {
        throw std::runtime_error(context + " could not select any measured cuBLASLt epilogue GEMM candidate.");
    }

    return preferred.value();
}

static std::vector<LtMeasuredMatmulAlgorithm> cublasLtGetVerifiedHeuristicContestants(int gpuNum,
                                                                                      cublasLtMatmulDesc_t operationDesc,
                                                                                      cublasLtMatrixLayout_t ADesc,
                                                                                      cublasLtMatrixLayout_t BDesc,
                                                                                      cublasLtMatrixLayout_t CDesc,
                                                                                      cublasLtMatrixLayout_t DDesc,
                                                                                      uint64_t workspaceLimitBytes,
                                                                                      int maxCandidateCount) {
    cublasLtMatmulPreference_t searchPreferences;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&searchPreferences));
    CHECK_CUBLAS(cublasLtMatmulPreferenceInit(searchPreferences));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        searchPreferences, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceLimitBytes, sizeof(workspaceLimitBytes)));

    int returnedAlgoCount = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> results(static_cast<size_t>(maxCandidateCount));
    const cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(compileTimeCublasLtHandle(gpuNum),
                                                                          operationDesc,
                                                                          ADesc,
                                                                          BDesc,
                                                                          CDesc,
                                                                          DDesc,
                                                                          searchPreferences,
                                                                          maxCandidateCount,
                                                                          results.data(),
                                                                          &returnedAlgoCount);
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(searchPreferences));
    if (heuristicStatus != CUBLAS_STATUS_SUCCESS) {
        return {};
    }

    std::vector<LtMeasuredMatmulAlgorithm> candidates;
    candidates.reserve(static_cast<size_t>(returnedAlgoCount));
    for (int i = 0; i < returnedAlgoCount; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS || !(results[i].wavesCount > 0.0f)) {
            continue;
        }
        const uint64_t candidateWorkspaceBytes = static_cast<uint64_t>(results[i].workspaceSize);
        if (candidateWorkspaceBytes > workspaceLimitBytes) {
            continue;
        }

        cublasLtMatmulHeuristicResult_t checked{};
        const cublasStatus_t checkStatus = cublasLtMatmulAlgoCheck(
            compileTimeCublasLtHandle(gpuNum), operationDesc, ADesc, BDesc, CDesc, DDesc, &results[i].algo, &checked);
        if (checkStatus != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        const uint64_t checkedWorkspaceBytes = static_cast<uint64_t>(checked.workspaceSize);
        const uint64_t requiredWorkspaceBytes = std::max(candidateWorkspaceBytes, checkedWorkspaceBytes);
        if (requiredWorkspaceBytes > workspaceLimitBytes) {
            continue;
        }

        LtMeasuredMatmulAlgorithm candidate;
        candidate.selection.algorithm = checked.algo;
        candidate.selection.workspace_size_in_bytes = requiredWorkspaceBytes;
        candidate.wavesCount = checked.wavesCount;
        candidates.push_back(candidate);
    }
    return candidates;
}

static cublasStatus_t runLtMatmulContestant(int gpuNum,
                                            cublasLtMatmulDesc_t operationDesc,
                                            const void *alpha,
                                            const void *ltA,
                                            cublasLtMatrixLayout_t ADesc,
                                            const void *ltB,
                                            cublasLtMatrixLayout_t BDesc,
                                            const void *beta,
                                            const void *ltC,
                                            cublasLtMatrixLayout_t CDesc,
                                            void *ltD,
                                            cublasLtMatrixLayout_t DDesc,
                                            LtMeasuredMatmulAlgorithm &candidate,
                                            void *workspacePtr,
                                            Stream stream) {
    return cublasLtMatmul(compileTimeCublasLtHandle(gpuNum),
                          operationDesc,
                          alpha,
                          ltA,
                          ADesc,
                          ltB,
                          BDesc,
                          beta,
                          ltC,
                          CDesc,
                          ltD,
                          DDesc,
                          &candidate.selection.algorithm,
                          workspacePtr,
                          static_cast<size_t>(candidate.selection.workspace_size_in_bytes),
                          stream);
}

static std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> cublasLtSelectMeasuredContestAlgorithm(
    int gpuNum,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t ADesc,
    cublasLtMatrixLayout_t BDesc,
    cublasLtMatrixLayout_t CDesc,
    cublasLtMatrixLayout_t DDesc,
    const void *alpha,
    const void *ltA,
    const void *ltB,
    const void *beta,
    const void *ltC,
    void *ltD,
    uint64_t workspaceLimitBytes,
    uint64_t operationBytes,
    const std::string &context) {
    constexpr int INITIAL_RUNS = 5;
    constexpr int FINAL_RUNS = 20;
    constexpr int INITIAL_CANDIDATE_COUNT = 100;
    constexpr int FALLBACK_CANDIDATE_COUNT = 10000;

    std::vector<LtMeasuredMatmulAlgorithm> candidates = cublasLtGetVerifiedHeuristicContestants(
        gpuNum, operationDesc, ADesc, BDesc, CDesc, DDesc, workspaceLimitBytes, INITIAL_CANDIDATE_COUNT);
    if (candidates.empty()) {
        candidates = cublasLtGetVerifiedHeuristicContestants(
            gpuNum, operationDesc, ADesc, BDesc, CDesc, DDesc, workspaceLimitBytes, FALLBACK_CANDIDATE_COUNT);
    }
    if (candidates.empty()) {
        return std::nullopt;
    }

    uint64_t contestWorkspaceBytes = 0;
    for (const LtMeasuredMatmulAlgorithm &candidate : candidates) {
        contestWorkspaceBytes = std::max(contestWorkspaceBytes, candidate.selection.workspace_size_in_bytes);
    }

    std::optional<Tensor> contestWorkspace;
    if (contestWorkspaceBytes > 0) {
        contestWorkspace =
            Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum), TensorDescriptor(DataType::UINT8, {contestWorkspaceBytes}));
    }
    void *workspacePtr = contestWorkspaceBytes > 0 ? contestWorkspace.value().getMemPtr() : nullptr;

    Stream stream(gpuNum);
    stream.synchronize();

    // First prove each checked algorithm actually launches with the contest
    // tensors and workspace.  cuBLASLt heuristic/check metadata is necessary
    // but not sufficient on every driver/toolkit combination.
    std::vector<LtMeasuredMatmulAlgorithm> runnable;
    runnable.reserve(candidates.size());
    for (LtMeasuredMatmulAlgorithm &candidate : candidates) {
        const cublasStatus_t status = runLtMatmulContestant(
            gpuNum, operationDesc, alpha, ltA, ADesc, ltB, BDesc, beta, ltC, CDesc, ltD, DDesc, candidate, workspacePtr, stream);
        if (status == CUBLAS_STATUS_SUCCESS) {
            stream.synchronize();
            runnable.push_back(candidate);
        }
    }
    candidates = std::move(runnable);
    if (candidates.empty()) {
        return std::nullopt;
    }

    std::vector<std::vector<Event>> startEvents(candidates.size());
    std::vector<std::vector<Event>> stopEvents(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        startEvents[i].reserve(static_cast<size_t>(INITIAL_RUNS + FINAL_RUNS));
        stopEvents[i].reserve(static_cast<size_t>(INITIAL_RUNS + FINAL_RUNS));
    }

    for (int run = 0; run < INITIAL_RUNS; ++run) {
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            startEvents[candidateIndex].push_back(stream.putEvent(true));
            const cublasStatus_t status = runLtMatmulContestant(gpuNum,
                                                                operationDesc,
                                                                alpha,
                                                                ltA,
                                                                ADesc,
                                                                ltB,
                                                                BDesc,
                                                                beta,
                                                                ltC,
                                                                CDesc,
                                                                ltD,
                                                                DDesc,
                                                                candidates[candidateIndex],
                                                                workspacePtr,
                                                                stream);
            stopEvents[candidateIndex].push_back(stream.putEvent(true));
            if (status != CUBLAS_STATUS_SUCCESS) {
                candidates[candidateIndex].errorFlag = true;
            }
        }
    }
    for (int run = 0; run < INITIAL_RUNS; ++run) {
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            if (!candidates[candidateIndex].errorFlag) {
                candidates[candidateIndex].runStats.recordRun(
                    stopEvents[candidateIndex][run].synchronizeAndReportElapsedTimeInMilliseconds(startEvents[candidateIndex][run]));
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), ltMeasuredMatmulExecutionTimeComparison);
    candidates.erase(
        std::remove_if(
            candidates.begin(), candidates.end(), [](const LtMeasuredMatmulAlgorithm &candidate) { return candidate.errorFlag; }),
        candidates.end());
    if (candidates.empty()) {
        return std::nullopt;
    }

    for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
        startEvents[candidateIndex].clear();
        stopEvents[candidateIndex].clear();
    }
    startEvents.resize(candidates.size());
    stopEvents.resize(candidates.size());

    for (int run = 0; run < FINAL_RUNS; ++run) {
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            startEvents[candidateIndex].push_back(stream.putEvent(true));
            const cublasStatus_t status = runLtMatmulContestant(gpuNum,
                                                                operationDesc,
                                                                alpha,
                                                                ltA,
                                                                ADesc,
                                                                ltB,
                                                                BDesc,
                                                                beta,
                                                                ltC,
                                                                CDesc,
                                                                ltD,
                                                                DDesc,
                                                                candidates[candidateIndex],
                                                                workspacePtr,
                                                                stream);
            stopEvents[candidateIndex].push_back(stream.putEvent(true));
            if (status != CUBLAS_STATUS_SUCCESS) {
                candidates[candidateIndex].errorFlag = true;
            }
        }
    }
    for (int run = 0; run < FINAL_RUNS; ++run) {
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            if (!candidates[candidateIndex].errorFlag) {
                candidates[candidateIndex].runStats.recordRun(
                    stopEvents[candidateIndex][run].synchronizeAndReportElapsedTimeInMilliseconds(startEvents[candidateIndex][run]));
            }
        }
    }

    return selectMeasuredLtMatmulAlgorithmWithWorkspacePolicy(candidates, operationBytes, context);
}

static void createColumnMajorPublicRowMajorEpilogueLayouts(cublasLtMatrixLayout_t *ADesc,
                                                           cublasLtMatrixLayout_t *BDesc,
                                                           cublasLtMatrixLayout_t *CDesc,
                                                           cublasLtMatrixLayout_t *DDesc,
                                                           const OperationType &operationType,
                                                           const int32_t A_rows,
                                                           const int32_t A_cols,
                                                           const int32_t B_rows,
                                                           const int32_t B_cols,
                                                           const int32_t D_rows,
                                                           const int32_t D_cols,
                                                           const int32_t ld_A,
                                                           const int32_t ld_B,
                                                           const int32_t ld_C,
                                                           const int32_t ld_D,
                                                           cudaDataType_t cDescType) {
    const cublasLtOrder_t columnMajorOrder = CUBLASLT_ORDER_COL;
    int64_t ld = 0;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(ADesc, operationType.BDataType, B_cols, B_rows, ld_B));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_B;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*ADesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(BDesc, operationType.ADataType, A_cols, A_rows, ld_A));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_A;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*BDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(CDesc, cDescType, D_cols, D_rows, ld_C));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_C;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*CDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(DDesc, operationType.DDataType, D_cols, D_rows, ld_D));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &columnMajorOrder, sizeof(columnMajorOrder)));
    ld = ld_D;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*DDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
}

std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> CublasMatrixMultiply::selectGemmWithEpilogueAlgorithm(
    int gpuNum,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    const int32_t ld_A,
    const int32_t ld_B,
    const int32_t ld_C,
    const int32_t ld_D,
    bool transposeA,
    bool transposeB,
    const MatmulDataTypes dataTypes,
    EpilogueFusion epilogue,
    bool hasAddend,
    bool addendIsBiasVector,
    std::optional<uint64_t> maxWorkspaceSizeInBytes) {
    if (transposeA || transposeB) {
        return std::nullopt;
    }
    if (!hasAddend && epilogue == EpilogueFusion::Default) {
        return std::nullopt;
    }

    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::selectGemmWithEpilogueAlgorithm");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, Fp8MatmulScales::none(), "CublasMatrixMultiply::selectGemmWithEpilogueAlgorithm");

    ScopedGpu scopedGpu(gpuNum);
    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
                                                 A_rows,
                                                 A_cols,
                                                 B_rows,
                                                 B_cols,
                                                 ld_A,
                                                 ld_B,
                                                 ld_C,
                                                 ld_D,
                                                 transposeA,
                                                 transposeB,
                                                 "CublasMatrixMultiply::selectGemmWithEpilogueAlgorithm");
    if (fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        return std::nullopt;
    }

    const int32_t D_rows = transposeA ? A_cols : A_rows;
    const int32_t D_cols = transposeB ? B_rows : B_cols;
    const cudaDataType_t cDescType = hasAddend && !addendIsBiasVector ? operationType.CDataType : operationType.DDataType;

    Tensor contestA(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                    TensorDescriptor(dataTypes.A, {static_cast<uint64_t>(A_rows), static_cast<uint64_t>(ld_A)}));
    Tensor contestB(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                    TensorDescriptor(dataTypes.B, {static_cast<uint64_t>(B_rows), static_cast<uint64_t>(ld_B)}));
    Tensor contestD(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                    TensorDescriptor(dataTypes.D, {static_cast<uint64_t>(D_rows), static_cast<uint64_t>(ld_D)}));
    std::optional<Tensor> contestAddend;
    if (hasAddend) {
        if (addendIsBiasVector) {
            contestAddend = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                   TensorDescriptor(dataTypes.D, {static_cast<uint64_t>(D_cols)}));
        } else {
            contestAddend = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                   TensorDescriptor(dataTypes.C, {static_cast<uint64_t>(D_rows), static_cast<uint64_t>(ld_C)}));
        }
    }

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ADesc;
    cublasLtMatrixLayout_t BDesc;
    cublasLtMatrixLayout_t CDesc;
    cublasLtMatrixLayout_t DDesc;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, operationType.computeDataType, operationType.scaleDataType));
    const cublasLtEpilogue_t cublasEpilogue = toCublasLtEpilogue(epilogue, addendIsBiasVector);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &cublasEpilogue, sizeof(cublasEpilogue)));
    if (addendIsBiasVector) {
        void *biasPtr = contestAddend.value().getMemPtr();
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)));
    }
    createColumnMajorPublicRowMajorEpilogueLayouts(
        &ADesc, &BDesc, &CDesc, &DDesc, operationType, A_rows, A_cols, B_rows, B_cols, D_rows, D_cols, ld_A, ld_B, ld_C, ld_D, cDescType);

    const uint64_t operationBytes =
        cublasLtForwardEpilogueOperationBytes(operationType, A_rows, A_cols, B_rows, B_cols, D_rows, D_cols, hasAddend, addendIsBiasVector);
    const void *ltA = contestB.getMemPtr();
    const void *ltB = contestA.getMemPtr();
    const void *ltC = (hasAddend && !addendIsBiasVector) ? contestAddend.value().getMemPtr() : contestD.getMemPtr();
    void *ltD = contestD.getMemPtr();
    const float alphaOne = 1.0f;
    const float betaZero = 0.0f;
    const float betaOne = 1.0f;
    const float *effectiveBeta = (hasAddend && !addendIsBiasVector) ? &betaOne : &betaZero;
    auto selection = cublasLtSelectMeasuredContestAlgorithm(gpuNum,
                                                            operationDesc,
                                                            ADesc,
                                                            BDesc,
                                                            CDesc,
                                                            DDesc,
                                                            &alphaOne,
                                                            ltA,
                                                            ltB,
                                                            effectiveBeta,
                                                            ltC,
                                                            ltD,
                                                            maxWorkspaceSizeInBytes.value_or(cublasLtEpilogueWorkspacePolicyLimitBytes(gpuNum)),
                                                            operationBytes,
                                                            "CublasMatrixMultiply::selectGemmWithEpilogueAlgorithm");

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(DDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(CDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(BDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(ADesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    return selection;
}

uint64_t CublasMatrixMultiply::getGemmWithEpilogueWorkspaceSizeInBytes(int gpuNum,
                                                                       const int32_t A_rows,
                                                                       const int32_t A_cols,
                                                                       const int32_t B_rows,
                                                                       const int32_t B_cols,
                                                                       const int32_t ld_A,
                                                                       const int32_t ld_B,
                                                                       const int32_t ld_C,
                                                                       const int32_t ld_D,
                                                                       bool transposeA,
                                                                       bool transposeB,
                                                                       const MatmulDataTypes dataTypes,
                                                                       EpilogueFusion epilogue,
                                                                       bool hasAddend,
                                                                       bool addendIsBiasVector,
                                                                       bool &kernelWillRunOnGpu) {
    auto selection = selectGemmWithEpilogueAlgorithm(gpuNum,
                                                     A_rows,
                                                     A_cols,
                                                     B_rows,
                                                     B_cols,
                                                     ld_A,
                                                     ld_B,
                                                     ld_C,
                                                     ld_D,
                                                     transposeA,
                                                     transposeB,
                                                     dataTypes,
                                                     epilogue,
                                                     hasAddend,
                                                     addendIsBiasVector);
    kernelWillRunOnGpu = selection.has_value();
    return selection.has_value() ? selection->workspace_size_in_bytes : 0;
}

std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> CublasMatrixMultiply::selectGemmWithBackwardEpilogueAlgorithm(
    int gpuNum,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    const int32_t ld_A,
    const int32_t ld_B,
    const int32_t ld_C,
    const int32_t ld_D,
    bool transposeA,
    bool transposeB,
    const MatmulDataTypes dataTypes,
    BackwardEpilogueFusion epilogue,
    bool hasAddend,
    bool hasBiasGradient,
    int64_t epilogueAuxLd,
    std::optional<uint64_t> maxWorkspaceSizeInBytes) {
    if (transposeA || transposeB) {
        return std::nullopt;
    }

    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::selectGemmWithBackwardEpilogueAlgorithm");
    validateFp8MatmulScaleConfigurationOrThrow(
        dataTypes, Fp8MatmulScales::none(), "CublasMatrixMultiply::selectGemmWithBackwardEpilogueAlgorithm");

    ScopedGpu scopedGpu(gpuNum);
    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
                                                 A_rows,
                                                 A_cols,
                                                 B_rows,
                                                 B_cols,
                                                 ld_A,
                                                 ld_B,
                                                 ld_C,
                                                 ld_D,
                                                 transposeA,
                                                 transposeB,
                                                 "CublasMatrixMultiply::selectGemmWithBackwardEpilogueAlgorithm");
    if (fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB)) {
        return std::nullopt;
    }

    const int32_t D_rows = transposeA ? A_cols : A_rows;
    const int32_t D_cols = transposeB ? B_rows : B_cols;
    const cudaDataType_t cDescType = hasAddend ? operationType.CDataType : operationType.DDataType;

    Tensor contestA(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                    TensorDescriptor(dataTypes.A, {static_cast<uint64_t>(A_rows), static_cast<uint64_t>(ld_A)}));
    Tensor contestB(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                    TensorDescriptor(dataTypes.B, {static_cast<uint64_t>(B_rows), static_cast<uint64_t>(ld_B)}));
    Tensor contestD(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                    TensorDescriptor(dataTypes.D, {static_cast<uint64_t>(D_rows), static_cast<uint64_t>(ld_D)}));
    const DataType epilogueAuxDataType = epilogue == BackwardEpilogueFusion::DGelu ? dataTypes.D : DataType::UINT8;
    Tensor contestEpilogueAux(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                              TensorDescriptor(epilogueAuxDataType, {static_cast<uint64_t>(D_rows), static_cast<uint64_t>(epilogueAuxLd)}));
    std::optional<Tensor> contestAddend;
    if (hasAddend) {
        contestAddend = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                               TensorDescriptor(dataTypes.C, {static_cast<uint64_t>(D_rows), static_cast<uint64_t>(ld_C)}));
    }
    std::optional<Tensor> contestBiasGradient;
    if (hasBiasGradient) {
        contestBiasGradient = Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                                     TensorDescriptor(dataTypes.D, {static_cast<uint64_t>(D_cols)}));
    }

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ADesc;
    cublasLtMatrixLayout_t BDesc;
    cublasLtMatrixLayout_t CDesc;
    cublasLtMatrixLayout_t DDesc;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, operationType.computeDataType, operationType.scaleDataType));
    const cublasLtEpilogue_t cublasEpilogue = toCublasLtBackwardEpilogue(epilogue, hasBiasGradient);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &cublasEpilogue, sizeof(cublasEpilogue)));
    const void *auxPtr = contestEpilogueAux.getMemPtr();
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &auxPtr, sizeof(auxPtr)));
    CHECK_CUBLAS(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &epilogueAuxLd, sizeof(epilogueAuxLd)));
    if (hasBiasGradient) {
        void *biasPtr = contestBiasGradient.value().getMemPtr();
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)));
    }
    createColumnMajorPublicRowMajorEpilogueLayouts(
        &ADesc, &BDesc, &CDesc, &DDesc, operationType, A_rows, A_cols, B_rows, B_cols, D_rows, D_cols, ld_A, ld_B, ld_C, ld_D, cDescType);

    const uint64_t operationBytes = cublasLtBackwardEpilogueOperationBytes(
        operationType, A_rows, A_cols, B_rows, B_cols, D_rows, D_cols, epilogueAuxLd, hasAddend, hasBiasGradient);
    const void *ltA = contestB.getMemPtr();
    const void *ltB = contestA.getMemPtr();
    const void *ltC = hasAddend ? contestAddend.value().getMemPtr() : contestD.getMemPtr();
    void *ltD = contestD.getMemPtr();
    const float alphaOne = 1.0f;
    const float betaZero = 0.0f;
    const float betaOne = 1.0f;
    const float *effectiveBeta = hasAddend ? &betaOne : &betaZero;
    auto selection = cublasLtSelectMeasuredContestAlgorithm(gpuNum,
                                                            operationDesc,
                                                            ADesc,
                                                            BDesc,
                                                            CDesc,
                                                            DDesc,
                                                            &alphaOne,
                                                            ltA,
                                                            ltB,
                                                            effectiveBeta,
                                                            ltC,
                                                            ltD,
                                                            maxWorkspaceSizeInBytes.value_or(cublasLtEpilogueWorkspacePolicyLimitBytes(gpuNum)),
                                                            operationBytes,
                                                            "CublasMatrixMultiply::selectGemmWithBackwardEpilogueAlgorithm");

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(DDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(CDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(BDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(ADesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    return selection;
}

uint64_t CublasMatrixMultiply::getGemmWithBackwardEpilogueWorkspaceSizeInBytes(int gpuNum,
                                                                               const int32_t A_rows,
                                                                               const int32_t A_cols,
                                                                               const int32_t B_rows,
                                                                               const int32_t B_cols,
                                                                               const int32_t ld_A,
                                                                               const int32_t ld_B,
                                                                               const int32_t ld_C,
                                                                               const int32_t ld_D,
                                                                               bool transposeA,
                                                                               bool transposeB,
                                                                               const MatmulDataTypes dataTypes,
                                                                               BackwardEpilogueFusion epilogue,
                                                                               bool hasAddend,
                                                                               bool hasBiasGradient,
                                                                               int64_t epilogueAuxLd,
                                                                               bool &kernelWillRunOnGpu) {
    auto selection = selectGemmWithBackwardEpilogueAlgorithm(gpuNum,
                                                             A_rows,
                                                             A_cols,
                                                             B_rows,
                                                             B_cols,
                                                             ld_A,
                                                             ld_B,
                                                             ld_C,
                                                             ld_D,
                                                             transposeA,
                                                             transposeB,
                                                             dataTypes,
                                                             epilogue,
                                                             hasAddend,
                                                             hasBiasGradient,
                                                             epilogueAuxLd);
    kernelWillRunOnGpu = selection.has_value();
    return selection.has_value() ? selection->workspace_size_in_bytes : 0;
}

void CublasMatrixMultiply::gemmWithBackwardEpilogueUsingHeuristicKernelChoice(
    Tensor A,
    Tensor B,
    std::optional<Tensor> addend,
    Tensor epilogueAux,
    Tensor D,
    std::optional<Tensor> biasGradient,
    const int32_t A_rows,
    const int32_t A_cols,
    const int32_t B_rows,
    const int32_t B_cols,
    bool transposeA,
    bool transposeB,
    const float *alpha,
    const float *beta,
    const MatmulDataTypes dataTypes,
    BackwardEpilogueFusion epilogue,
    Stream stream,
    CublasScalarPointerMode pointerMode,
    std::optional<Tensor> workspace,
    std::optional<CublasMatrixMultiply::LtMatmulAlgorithmSelection> selectedAlgorithm) {
    const int32_t C_rows = (transposeA == false ? A_rows : A_cols);
    const int32_t C_cols = (transposeB == false ? B_cols : B_rows);
    const int32_t D_rows = C_rows;
    const int32_t D_cols = C_cols;
    const int32_t ld_A = A.getDimensions()[1];
    const int32_t ld_B = B.getDimensions()[1];
    const int32_t ld_D = D.getDimensions()[1];
    const int32_t ld_C = addend.has_value() ? addend.value().getDimensions()[1] : ld_D;

    THOR_THROW_IF_FALSE(A.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(B.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(epilogueAux.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(D.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(A.getDimensions()[0] == static_cast<uint64_t>(A_rows));
    THOR_THROW_IF_FALSE(A.getDimensions()[1] == static_cast<uint64_t>(ld_A));
    THOR_THROW_IF_FALSE(B.getDimensions()[0] == static_cast<uint64_t>(B_rows));
    THOR_THROW_IF_FALSE(B.getDimensions()[1] == static_cast<uint64_t>(ld_B));
    THOR_THROW_IF_FALSE(D.getDimensions()[0] == static_cast<uint64_t>(D_rows));
    THOR_THROW_IF_FALSE(D.getDimensions()[1] == static_cast<uint64_t>(ld_D));
    THOR_THROW_IF_FALSE(epilogueAux.getDimensions()[0] == static_cast<uint64_t>(D_rows));
    THOR_THROW_IF_FALSE(epilogueAux.getDimensions()[1] >= static_cast<uint64_t>(D_cols));

    if (addend.has_value()) {
        THOR_THROW_IF_FALSE(addend.value().getDimensions().size() == 2);
        THOR_THROW_IF_FALSE(addend.value().getDimensions()[0] == static_cast<uint64_t>(D_rows));
        THOR_THROW_IF_FALSE(addend.value().getDimensions()[1] == static_cast<uint64_t>(ld_C));
    }
    if (biasGradient.has_value()) {
        THOR_THROW_IF_FALSE(biasGradient.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(biasGradient.value().getDimensions()[0] == static_cast<uint64_t>(D_cols));
        THOR_THROW_IF_FALSE(biasGradient.value().getDescriptor().getDataType() == D.getDescriptor().getDataType());
    }

    const std::optional<uint64_t> workspaceLimitBytes = workspace.has_value()
                                                            ? std::optional<uint64_t>(workspace.value().getArraySizeInBytes())
                                                            : std::optional<uint64_t>(0);
    auto plan = buildGemmWithBackwardEpiloguePlan(stream.getGpuNum(),
                                                  A_rows,
                                                  A_cols,
                                                  B_rows,
                                                  B_cols,
                                                  ld_A,
                                                  ld_B,
                                                  ld_C,
                                                  ld_D,
                                                  transposeA,
                                                  transposeB,
                                                  dataTypes,
                                                  epilogue,
                                                  addend.has_value(),
                                                  epilogueAux,
                                                  biasGradient,
                                                  selectedAlgorithm,
                                                  workspaceLimitBytes);
    if (!plan) {
        throw std::runtime_error("Unable to build cuBLASLt backward epilogue GEMM plan for legacy heuristic wrapper.");
    }
    plan->runGemmWithBackwardEpilogue(A, B, addend, D, alpha, beta, stream, pointerMode, workspace);
}

void CublasMatrixMultiply::gemmWithDReluUsingHeuristicKernelChoice(Tensor A,
                                                                   Tensor B,
                                                                   std::optional<Tensor> addend,
                                                                   Tensor reluAux,
                                                                   Tensor D,
                                                                   const int32_t A_rows,
                                                                   const int32_t A_cols,
                                                                   const int32_t B_rows,
                                                                   const int32_t B_cols,
                                                                   bool transposeA,
                                                                   bool transposeB,
                                                                   const float *alpha,
                                                                   const float *beta,
                                                                   const MatmulDataTypes dataTypes,
                                                                   Stream stream,
                                                                   CublasScalarPointerMode pointerMode,
                                                                   std::optional<Tensor> workspace) {
    gemmWithBackwardEpilogueUsingHeuristicKernelChoice(A,
                                                       B,
                                                       addend,
                                                       reluAux,
                                                       D,
                                                       std::nullopt,
                                                       A_rows,
                                                       A_cols,
                                                       B_rows,
                                                       B_cols,
                                                       transposeA,
                                                       transposeB,
                                                       alpha,
                                                       beta,
                                                       dataTypes,
                                                       BackwardEpilogueFusion::DRelu,
                                                       stream,
                                                       pointerMode,
                                                       workspace);
}

void CublasMatrixMultiply::gemmWithDGeluUsingHeuristicKernelChoice(Tensor A,
                                                                   Tensor B,
                                                                   std::optional<Tensor> addend,
                                                                   Tensor geluAux,
                                                                   Tensor D,
                                                                   const int32_t A_rows,
                                                                   const int32_t A_cols,
                                                                   const int32_t B_rows,
                                                                   const int32_t B_cols,
                                                                   bool transposeA,
                                                                   bool transposeB,
                                                                   const float *alpha,
                                                                   const float *beta,
                                                                   const MatmulDataTypes dataTypes,
                                                                   Stream stream,
                                                                   CublasScalarPointerMode pointerMode,
                                                                   std::optional<Tensor> workspace) {
    gemmWithBackwardEpilogueUsingHeuristicKernelChoice(A,
                                                       B,
                                                       addend,
                                                       geluAux,
                                                       D,
                                                       std::nullopt,
                                                       A_rows,
                                                       A_cols,
                                                       B_rows,
                                                       B_cols,
                                                       transposeA,
                                                       transposeB,
                                                       alpha,
                                                       beta,
                                                       dataTypes,
                                                       BackwardEpilogueFusion::DGelu,
                                                       stream,
                                                       pointerMode,
                                                       workspace);
}

void CublasMatrixMultiply::gemmWithDReluBgradUsingHeuristicKernelChoice(Tensor A,
                                                                        Tensor B,
                                                                        std::optional<Tensor> addend,
                                                                        Tensor reluAux,
                                                                        Tensor D,
                                                                        Tensor biasGradient,
                                                                        const int32_t A_rows,
                                                                        const int32_t A_cols,
                                                                        const int32_t B_rows,
                                                                        const int32_t B_cols,
                                                                        bool transposeA,
                                                                        bool transposeB,
                                                                        const float *alpha,
                                                                        const float *beta,
                                                                        const MatmulDataTypes dataTypes,
                                                                        Stream stream,
                                                                        CublasScalarPointerMode pointerMode,
                                                                        std::optional<Tensor> workspace) {
    gemmWithBackwardEpilogueUsingHeuristicKernelChoice(A,
                                                       B,
                                                       addend,
                                                       reluAux,
                                                       D,
                                                       biasGradient,
                                                       A_rows,
                                                       A_cols,
                                                       B_rows,
                                                       B_cols,
                                                       transposeA,
                                                       transposeB,
                                                       alpha,
                                                       beta,
                                                       dataTypes,
                                                       BackwardEpilogueFusion::DRelu,
                                                       stream,
                                                       pointerMode,
                                                       workspace);
}

void CublasMatrixMultiply::gemmWithDGeluBgradUsingHeuristicKernelChoice(Tensor A,
                                                                        Tensor B,
                                                                        std::optional<Tensor> addend,
                                                                        Tensor geluAux,
                                                                        Tensor D,
                                                                        Tensor biasGradient,
                                                                        const int32_t A_rows,
                                                                        const int32_t A_cols,
                                                                        const int32_t B_rows,
                                                                        const int32_t B_cols,
                                                                        bool transposeA,
                                                                        bool transposeB,
                                                                        const float *alpha,
                                                                        const float *beta,
                                                                        const MatmulDataTypes dataTypes,
                                                                        Stream stream,
                                                                        CublasScalarPointerMode pointerMode,
                                                                        std::optional<Tensor> workspace) {
    gemmWithBackwardEpilogueUsingHeuristicKernelChoice(A,
                                                       B,
                                                       addend,
                                                       geluAux,
                                                       D,
                                                       biasGradient,
                                                       A_rows,
                                                       A_cols,
                                                       B_rows,
                                                       B_cols,
                                                       transposeA,
                                                       transposeB,
                                                       alpha,
                                                       beta,
                                                       dataTypes,
                                                       BackwardEpilogueFusion::DGelu,
                                                       stream,
                                                       pointerMode,
                                                       workspace);
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
                                                                   const MatmulDataTypes dataTypes,
                                                                   const Fp8MatmulScales fp8Scales) {
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

    THOR_THROW_IF_FALSE(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    THOR_THROW_IF_FALSE(A_rows > 0);
    THOR_THROW_IF_FALSE(A_cols > 0);
    THOR_THROW_IF_FALSE(B_rows > 0);
    THOR_THROW_IF_FALSE(B_cols > 0);
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::getHeuristicGemmKernels");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, fp8Scales, "CublasMatrixMultiply::getHeuristicGemmKernels");

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ADesc;
    cublasLtMatrixLayout_t BDesc;
    cublasLtMatrixLayout_t CDesc;
    cublasLtMatrixLayout_t DDesc;

    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
                                                 A_rows,
                                                 A_cols,
                                                 B_rows,
                                                 B_cols,
                                                 ld_A,
                                                 ld_B,
                                                 ld_C,
                                                 ld_D,
                                                 transposeA,
                                                 transposeB,
                                                 "CublasMatrixMultiply::getHeuristicGemmKernels");
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, operationType.computeDataType, operationType.scaleDataType));

    const cublasLtMatmulDescAttributes_t pointerModeAttribute = CUBLASLT_MATMUL_DESC_POINTER_MODE;
    const cublasLtPointerMode_t hostPointerMode = CUBLASLT_POINTER_MODE_HOST;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, pointerModeAttribute, &hostPointerMode, sizeof(hostPointerMode)));
    configureTensorwideFp8Scales(operationDesc, operationType, fp8Scales);
    configureLtOperationTransposes(operationDesc, operationType, transposeA, transposeB, transposeC);
    createLtMatrixLayoutsForRowMajorGemm(&ADesc,
                                         &BDesc,
                                         &CDesc,
                                         &DDesc,
                                         operationType,
                                         A_rows,
                                         A_cols,
                                         B_rows,
                                         B_cols,
                                         C_rows,
                                         C_cols,
                                         D_rows,
                                         D_cols,
                                         ld_A,
                                         ld_B,
                                         ld_C,
                                         ld_D,
                                         transposeA,
                                         transposeB);

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

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    cublasLtMatmulPreference_t searchPreferences;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&searchPreferences));
    CHECK_CUBLAS(cublasLtMatmulPreferenceInit(searchPreferences));

    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        searchPreferences, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspaceSize, sizeof(maxWorkspaceSize)));
    CHECK_CUBLAS(
        cublasLtMatmulPreferenceSetAttribute(searchPreferences, CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT, &maxWaves, sizeof(maxWaves)));

    int returnedAlgoCount;
    vector<cublasLtMatmulHeuristicResult_t> rawResults(numChoices);
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(compileTimeCublasLtHandle(gpuNum),
                                                operationDesc,
                                                ADesc,
                                                BDesc,
                                                CDesc,
                                                DDesc,
                                                searchPreferences,
                                                numChoices,
                                                rawResults.data(),
                                                &returnedAlgoCount));

    rawResults.resize(returnedAlgoCount);

    vector<CublasKernel> results;
    string gpuType = MachineEvaluator::instance().getGpuType(gpuNum);
    for (int32_t i = 0; i < returnedAlgoCount; ++i) {
        if (rawResults[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        // It would be nice if cublasLt only returned algos that are supported by the GPU, but that is not the case.
        cublasLtMatmulHeuristicResult_t result;
        cublasStatus = cublasLtMatmulAlgoCheck(compileTimeCublasLtHandle(gpuNum),
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

        uint32_t tileId = 0;
        cublasStatus =
            cublasLtMatmulAlgoConfigGetAttribute(&rawResults[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId), &sizeWritten);

        uint32_t splitK = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), &sizeWritten));

        uint32_t reductionFlag = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionFlag, sizeof(reductionFlag), &sizeWritten));

        uint32_t swizzleType = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzleType, sizeof(swizzleType), &sizeWritten));

        uint32_t customOptionValue = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOptionValue, sizeof(customOptionValue), &sizeWritten));

        uint32_t stagesId = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesId, sizeof(stagesId), &sizeWritten));

        uint16_t innerShapeId = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &innerShapeId, sizeof(innerShapeId), &sizeWritten));

        uint16_t clusterShapeId = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
            &rawResults[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &clusterShapeId, sizeof(clusterShapeId), &sizeWritten));

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

    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(searchPreferences));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(DDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(CDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(BDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(ADesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));

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
                                                   DataType ABCDataType,
                                                   bool printResults) {
    chooseOptimalGemmKernel(gpuNum,
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
                            MatmulDataTypes::same(ABCDataType),
                            printResults);
}

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
                                                   MatmulDataTypes dataTypes,
                                                   bool printResults) {
    chooseOptimalGemmKernel(gpuNum,
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
                            dataTypes,
                            Fp8MatmulScales::none(),
                            printResults);
}

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
                                                   MatmulDataTypes dataTypes,
                                                   const Fp8MatmulScales fp8Scales,
                                                   bool printResults) {
    const OperationType operationType = makeOperationType(dataTypes);
    const bool fp8TemporaryWorkspaceRequired = fp8NeedsRowMajorTransposeWorkspace(operationType, transposeA, transposeB);

    bool bestKernelHasWorkspace = chooseOptimalGemmKernel(gpuNum,
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
                                                          dataTypes,
                                                          fp8Scales,
                                                          true,
                                                          printResults);

    // If the best kernel did not have a workspace, then it will be used for the no workspace version of the computation also
    if (bestKernelHasWorkspace && !fp8TemporaryWorkspaceRequired) {
        chooseOptimalGemmKernel(gpuNum,
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
                                dataTypes,
                                fp8Scales,
                                false,
                                printResults);

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
        std::lock_guard<std::mutex> lock(CublasMatrixMultiply::instance().mtx);
        auto noWorkspaceKernel = CublasMatrixMultiply::instance().optimalKernels.get(noWorkspaceCublasKernelRequirement);
        auto workspaceKernel = CublasMatrixMultiply::instance().optimalKernels.get(workspaceCublasKernelRequirement);
        THOR_THROW_IF_FALSE(noWorkspaceKernel.has_value());
        THOR_THROW_IF_FALSE(workspaceKernel.has_value());
        if (noWorkspaceKernel->getAverageRunTimeMilliseconds() < workspaceKernel->getAverageRunTimeMilliseconds()) {
            CublasMatrixMultiply::instance().optimalKernels.put(workspaceCublasKernelRequirement, *noWorkspaceKernel);
        }
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
                                                   const MatmulDataTypes dataTypes,
                                                   const Fp8MatmulScales fp8Scales,
                                                   const bool allowWorkspaces,
                                                   const bool printResults) {
    lock_guard<mutex> lck(CublasMatrixMultiply::instance().mtx);

    THOR_THROW_IF_FALSE(gpuNum >= 0);
    THOR_THROW_IF_FALSE(gpuNum < (int)MachineEvaluator::instance().getNumGpus());
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::chooseOptimalGemmKernel");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, fp8Scales, "CublasMatrixMultiply::chooseOptimalGemmKernel");

    // Ensure the operation is legal
    // The number of C and D columns is specified by the sizes of A and B, so verify A and B
    const int32_t finalRowsA = (transposeA ? colsA : rowsA);
    const int32_t finalColsA = (transposeA ? rowsA : colsA);
    const int32_t finalRowsB = (transposeB ? colsB : rowsB);
    const int32_t finalColsB = (transposeB ? rowsB : colsB);
    THOR_THROW_IF_FALSE(finalColsA == finalRowsB);

    const int32_t initialRowsC = (transposeC ? finalColsB : finalRowsA);
    const int32_t initialColsC = (transposeC ? finalRowsA : finalColsB);
    THOR_THROW_IF_FALSE(ldC >= initialColsC);

    // const int32_t finalRowsC = finalRowsA;
    const int32_t finalColsC = finalColsB;
    THOR_THROW_IF_FALSE(ldD >= finalColsC);

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
    const int A_ELEMENT_SIZE = TensorDescriptor::getElementSizeInBytes(dataTypes.A);
    const int B_ELEMENT_SIZE = TensorDescriptor::getElementSizeInBytes(dataTypes.B);
    const int C_ELEMENT_SIZE = TensorDescriptor::getElementSizeInBytes(dataTypes.C);
    const int D_ELEMENT_SIZE = TensorDescriptor::getElementSizeInBytes(dataTypes.D);

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

    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
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
                                                 "CublasMatrixMultiply::chooseOptimalGemmKernel");
    const uint64_t fp8TemporaryWorkspaceSizeInBytes =
        fp8RowMajorTransposeWorkspaceUpperBoundBytes(operationType, rowsA, colsA, rowsB, colsB, transposeA, transposeB);
    if (!allowWorkspaces && fp8TemporaryWorkspaceSizeInBytes > 0) {
        throw std::runtime_error(
            "CublasMatrixMultiply::chooseOptimalGemmKernel FP8 row-major path requires workspace for temporary A/B transposes.");
    }

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    // Will only evaluate kernel once per gpu type
    if (auto optimalKernel = optimalKernels.get(cublasKernelRequirement); optimalKernel.has_value()) {
        bool kernelWillRunOnGpu;
        unsigned int workspaceSizeInBytes = optimalKernel->getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales);
        THOR_THROW_IF_FALSE(kernelWillRunOnGpu);
        return workspaceSizeInBytes > 0 ? true : false;
    }

    THOR_THROW_IF_FALSE(transposeC == false);  // it seems cublas is not supporting this. You can use Tensor.transpose().

    const int32_t rowsC = (transposeA == false ? rowsA : colsA);
    int32_t rowsD = rowsC;

    // Get the expected best kernels
    uint64_t maxMatrixBytes = max({static_cast<uint64_t>(rowsA) * ldA * A_ELEMENT_SIZE,
                                   static_cast<uint64_t>(rowsB) * ldB * B_ELEMENT_SIZE,
                                   static_cast<uint64_t>(rowsC) * ldC * C_ELEMENT_SIZE,
                                   static_cast<uint64_t>(rowsD) * ldD * D_ELEMENT_SIZE});
    const uint64_t maxCublasWorkspaceSizeInBytes = allowWorkspaces ? maxMatrixBytes : 0;
    const uint64_t maxAllowedWorkspaceSizeInBytes = allowWorkspaces ? maxMatrixBytes + fp8TemporaryWorkspaceSizeInBytes : 0;
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
                                                maxCublasWorkspaceSizeInBytes,
                                                // When set to 0.0f, any number of waves allowed:
                                                maxWaves,
                                                dataTypes,
                                                fp8Scales);

    vector<CublasKernel> kernels;
    uint64_t maxWorkspaceSizeInBytes = 0;
    for (uint32_t i = 0; i < preCheckedKernels.size(); ++i) {
        bool kernelWillRunOnGpu;
        CublasKernel cublasKernel = preCheckedKernels[i];

        unsigned long workspaceSizeInBytes = cublasKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales);
        if (!kernelWillRunOnGpu) {
            continue;
        }
        if (workspaceSizeInBytes > maxAllowedWorkspaceSizeInBytes) {
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
    if (kernels.empty()) {
        throw std::runtime_error(
            "CublasMatrixMultiply::chooseOptimalGemmKernel could not find any cuBLASLt kernel candidates for this descriptor.");
    }

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

    cublasStatus_t cublasStatus = CUBLAS_STATUS_SUCCESS;

    // Ensure there are no unreported runtime errors
    stream.synchronize();

    // Allocate a lot of memory, to ensure subsequent calls are not benefiting from cache hits.
    // Keep enough distinct full matrix instances for this GEMM shape, and at least 2x L2 for small GEMMs.
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, gpuNum));
    long l2CacheBytes = deviceProperties.l2CacheSize;
    long memPerInstance =
        rowsA * ldA * A_ELEMENT_SIZE + rowsB * ldB * B_ELEMENT_SIZE + initialRowsC * ldC * C_ELEMENT_SIZE + rowsD * ldD * D_ELEMENT_SIZE;
    long targetMatrixMemory = maxl(10 * memPerInstance, 2.5 * l2CacheBytes);
    long numInstances = minl(targetMatrixMemory / memPerInstance, 5000);
    THOR_THROW_IF_FALSE(numInstances > 0);

    long numWorkspaceInstances;
    if (maxWorkspaceSizeInBytes == 0) {
        maxWorkspaceSizeInBytes = 1;
        numWorkspaceInstances = 1;
    } else {
        long totalWorkspaceMem = minl(totalGpuMem * 0.4, maxl(ONE_HUNDRED_MEGS, 10 * maxWorkspaceSizeInBytes));
        numWorkspaceInstances = minl(totalWorkspaceMem / maxWorkspaceSizeInBytes, 5000);
        THOR_THROW_IF_FALSE(numWorkspaceInstances > 0);
    }

    vector<Tensor> A;
    vector<Tensor> B;
    vector<Tensor> C;
    vector<Tensor> D;
    vector<Tensor> workspace;
    A.reserve(numInstances);
    B.reserve(numInstances);
    C.reserve(numInstances);
    D.reserve(numInstances);
    workspace.reserve(numWorkspaceInstances);
    for (int i = 0; i < numInstances; ++i) {
        A.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(dataTypes.A, {(uint64_t)rowsA, (uint64_t)ldA}));
        B.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(dataTypes.B, {(uint64_t)rowsB, (uint64_t)ldB}));
        C.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(dataTypes.C, {(uint64_t)initialRowsC, (uint64_t)ldC}));
        D.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                       TensorDescriptor(dataTypes.D, {(uint64_t)rowsD, (uint64_t)ldD}));
    }
    for (int i = 0; i < numWorkspaceInstances; ++i) {
        workspace.emplace_back(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum),
                               TensorDescriptor(DataType::UINT8, {maxWorkspaceSizeInBytes}));
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

        const float ONE = 1.0f;
        const float ZERO = 0.0f;
        // Keep all kernels that are as good as the kernelsToKeep'th best kernel.
        float maxWavesDiff = abs(TARGET_WAVES - kernels[initialContestantCount - 1].getWavesCount(gpuNum));
        for (unsigned int i = 0; i < kernels.size() && (abs(TARGET_WAVES - kernels[i].getWavesCount(gpuNum)) <= maxWavesDiff ||
                                                        prunedKernels.size() < initialContestantCount);
             ++i) {
            cublasStatus = kernels[i].runWithoutChecks(
                A[0], B[0], C[0], D[0], workspace[0], &ONE, &ZERO, stream, CublasScalarPointerMode::Host, fp8Scales);
            if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
                stream.synchronize();
                prunedKernels.push_back(kernels[i]);
            }
        }
        kernels = prunedKernels;
    } else {
        prunedKernels.reserve(kernels.size());

        // Prune just the non-working kernels
        const float ONE = 1.0f;
        const float ZERO = 0.0f;
        for (unsigned int i = 0; i < kernels.size(); ++i) {
            cublasStatus = kernels[i].runWithoutChecks(
                A[0], B[0], C[0], D[0], workspace[0], &ONE, &ZERO, stream, CublasScalarPointerMode::Host, fp8Scales);
            if (cublasStatus == CUBLAS_STATUS_SUCCESS) {
                stream.synchronize();
                prunedKernels.push_back(kernels[i]);
            }
        }
        kernels = prunedKernels;
    }

    if (printResults)
        printf("got %ld kernels\n", kernels.size());
    if (kernels.empty()) {
        throw std::runtime_error(
            "CublasMatrixMultiply::chooseOptimalGemmKernel could not find any cuBLASLt kernel candidates for this descriptor.");
    }

    int tensorInstance = 0;
    int workspaceInstance = 0;

    // Warm up
    double elapsedTime = 0.0;
    Event startEvent = stream.putEvent(true);
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    for (int i = 0; i < 5; ++i) {
        kernels[rand() % kernels.size()].runWithoutChecks(A[tensorInstance],
                                                          B[tensorInstance],
                                                          C[tensorInstance],
                                                          D[tensorInstance],
                                                          workspace[workspaceInstance],
                                                          &ONE,
                                                          &ZERO,
                                                          stream,
                                                          CublasScalarPointerMode::Host,
                                                          fp8Scales);
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
                                                              D[tensorInstance],
                                                              workspace[workspaceInstance],
                                                              &ONE,
                                                              &ZERO,
                                                              stream,
                                                              CublasScalarPointerMode::Host,
                                                              fp8Scales);
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
        kernels[rand() % kernels.size()].runWithoutChecks(A[tensorInstance],
                                                          B[tensorInstance],
                                                          C[tensorInstance],
                                                          D[tensorInstance],
                                                          workspace[workspaceInstance],
                                                          &ONE,
                                                          &ZERO,
                                                          stream,
                                                          CublasScalarPointerMode::Host,
                                                          fp8Scales);
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
                                                                 D[tensorInstance],
                                                                 workspace[workspaceInstance],
                                                                 &ONE,
                                                                 &ZERO,
                                                                 stream,
                                                                 CublasScalarPointerMode::Host,
                                                                 fp8Scales);

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

    THOR_THROW_IF_FALSE(cublasStatus == CUBLAS_STATUS_SUCCESS);
    for (int run = 0; run < finalRun; ++run) {
        for (unsigned int kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            startEvents[kernelIndex].push_back(stream.putEvent(true));
            cublasStatus = CUBLAS_STATUS_SUCCESS;
            if (!kernels[kernelIndex].getErrorFlag()) {
                cublasStatus = kernels[kernelIndex].runWithoutChecks(A[tensorInstance],
                                                                     B[tensorInstance],
                                                                     C[tensorInstance],
                                                                     D[tensorInstance],
                                                                     workspace[workspaceInstance],
                                                                     &ONE,
                                                                     &ZERO,
                                                                     stream,
                                                                     CublasScalarPointerMode::Host,
                                                                     fp8Scales);
            }
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

    const uint64_t operationBytes = static_cast<uint64_t>(memPerInstance);
    CublasKernel bestKernel = selectMeasuredKernelWithWorkspacePolicy(kernels, gpuNum, operationBytes, fp8Scales);
    THOR_THROW_IF_FALSE(!bestKernel.getErrorFlag());
    bestKernel.unstashRunStats();
    bool kernelWillRunOnGpu;
    bool bestKernelHasWorkspace = bestKernel.getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales) > 0;
    THOR_THROW_IF_FALSE(kernelWillRunOnGpu);

    optimalKernels.put(cublasKernelRequirement, bestKernel);

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
        optimalKernels.put(noWorkspaceCublasKernelRequirement, bestKernel);
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

        CHECK_CUBLAS(cublasLtMatmulAlgoGetIds(compileTimeCublasLtHandle(gpuNum),
                                              operationType.computeDataType,
                                              operationType.scaleDataType,
                                              operationType.ADataType,
                                              operationType.BDataType,
                                              operationType.CDataType,
                                              operationType.DDataType,
                                              numRequestedAlgos,
                                              allSupportedAlgorithmIds.data(),
                                              &numReturnedAlgos));
    }

    for (int i = 0; i < numReturnedAlgos; ++i) {
        cublasLtMatmulAlgo_t algo;
        cublasStatus = cublasLtMatmulAlgoInit(compileTimeCublasLtHandle(gpuNum),
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
        THOR_THROW_IF_FALSE(sizeWritten == numSupportedTilesConfigurations);
    }

    return supportedTileSizeEnums;
}

bool CublasMatrixMultiply::isSplitKSupported(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int splitKSupported;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitKSupported, sizeof(splitKSupported), &sizeWritten);
    THOR_THROW_IF_FALSE(sizeWritten == sizeof(splitKSupported));

    return splitKSupported ? true : false;
}

uint32_t CublasMatrixMultiply::getReductionSupportMask(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    uint32_t reductionSupportMask;
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reductionSupportMask, sizeof(reductionSupportMask), &sizeWritten);
    THOR_THROW_IF_FALSE(sizeWritten == sizeof(reductionSupportMask));

    return reductionSupportMask;
}

int CublasMatrixMultiply::getSwizzleMaxValue(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int swizzleMax;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzleMax, sizeof(swizzleMax), &sizeWritten);
    THOR_THROW_IF_FALSE(sizeWritten == sizeof(swizzleMax));
    THOR_THROW_IF_FALSE(swizzleMax == 0 || swizzleMax == 1);

    return swizzleMax;
}

int CublasMatrixMultiply::getCustomKernelOptionMaxValue(cublasLtMatmulAlgo_t algo) {
    size_t sizeWritten = 0;
    int kernelOptionMaxValue;
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &kernelOptionMaxValue, sizeof(kernelOptionMaxValue), &sizeWritten);
    THOR_THROW_IF_FALSE(sizeWritten == sizeof(kernelOptionMaxValue));

    return kernelOptionMaxValue;
}

CublasKernel CublasMatrixMultiply::getCachedGemmKernel(int gpuNum,
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
                                                     MatmulDataTypes dataTypes,
                                                     bool workspaceAllowed) {
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::getCachedGemmKernel");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, Fp8MatmulScales::none(), "CublasMatrixMultiply::getCachedGemmKernel");

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
                                        workspaceAllowed);
    OperationType operationType = makeOperationType(dataTypes);
    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    auto optimalKernel = optimalKernels.get(cublasKernelRequirement);
    if (!optimalKernel.has_value()) {
        throw std::runtime_error("CublasMatrixMultiply::getCachedGemmKernel called before chooseOptimalGemmKernel populated the cache.");
    }
    return optimalKernel.value();
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
                                                               DataType ABCDataType,
                                                               bool &kernelWillRunOnGpu) {
    return getGemmWorkspaceSizeInBytes(gpuNum,
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
                                       MatmulDataTypes::same(ABCDataType),
                                       kernelWillRunOnGpu);
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
                                                               MatmulDataTypes dataTypes,
                                                               bool &kernelWillRunOnGpu) {
    return getGemmWorkspaceSizeInBytes(gpuNum,
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
                                       dataTypes,
                                       Fp8MatmulScales::none(),
                                       kernelWillRunOnGpu);
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
                                                               MatmulDataTypes dataTypes,
                                                               Fp8MatmulScales fp8Scales,
                                                               bool &kernelWillRunOnGpu) {
    validateMatmulDataTypesOrThrow(dataTypes, "CublasMatrixMultiply::getGemmWorkspaceSizeInBytes");
    validateFp8MatmulScaleConfigurationOrThrow(dataTypes, fp8Scales, "CublasMatrixMultiply::getGemmWorkspaceSizeInBytes");

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

    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
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
                                                 "CublasMatrixMultiply::getGemmWorkspaceSizeInBytes");

    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    auto optimalKernel = CublasMatrixMultiply::instance().optimalKernels.get(cublasKernelRequirement);
    THOR_THROW_IF_FALSE(optimalKernel.has_value());
    unsigned int workspaceSize = optimalKernel->getWorkspaceSizeInBytes(gpuNum, kernelWillRunOnGpu, fp8Scales);

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
                                                  DataType ABCDataType,
                                                  bool workspaceAllowed) {
    return getOptimalKernelTime(gpuType,
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
                                MatmulDataTypes::same(ABCDataType),
                                workspaceAllowed);
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
                                                  MatmulDataTypes dataTypes,
                                                  bool workspaceAllowed) {
    KernelRequirement kernelRequirement(
        gpuType, rowsA, colsA, rowsB, colsB, transposeA, transposeB, transposeC, ldA, ldB, ldC, ldD, workspaceAllowed);

    OperationType operationType = makeOperationType(dataTypes);
    validateFp8RowMajorGemmShapeAndLayoutOrThrow(operationType,
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
                                                 "CublasMatrixMultiply::getOptimalKernelTime");
    CublasKernelRequirement cublasKernelRequirement(kernelRequirement, operationType);

    string dataTypesString = dataTypesToString(dataTypes);

    auto optimalKernel = CublasMatrixMultiply::instance().optimalKernels.get(cublasKernelRequirement);
    if (!optimalKernel.has_value()) {
        string message =
            "CublasMatrixMultiply::getOptimalKernelTime() : Kernel time is not known because kernel time has not been measured for "
            "gpuType " +
            gpuType + " rowsA " + std::to_string(rowsA) + " colsA " + std::to_string(colsA) + " colsB " + std::to_string(colsB) +
            " dataTypes " + dataTypesString;
        throw(runtime_error(message));
    }
    double averageRunTimeMilliseconds = optimalKernel->getAverageRunTimeMilliseconds();

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
                                                  DataType ABCDataType,
                                                  bool workspaceAllowed) {
    return getOptimalKernelTime(gpuNum,
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
                                MatmulDataTypes::same(ABCDataType),
                                workspaceAllowed);
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
                                                  MatmulDataTypes dataTypes,
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
                                dataTypes,
                                workspaceAllowed);
}
