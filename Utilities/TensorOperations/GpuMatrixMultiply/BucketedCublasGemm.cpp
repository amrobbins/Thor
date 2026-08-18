#include "Utilities/TensorOperations/GpuMatrixMultiply/BucketedCublasGemm.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ThorImplementation {
namespace {

bool bindsRowsA(BucketedCublasGemmRowBinding binding) {
    return binding == BucketedCublasGemmRowBinding::RowsA || binding == BucketedCublasGemmRowBinding::RowsAAndRowsB;
}

bool bindsRowsB(BucketedCublasGemmRowBinding binding) {
    return binding == BucketedCublasGemmRowBinding::RowsB || binding == BucketedCublasGemmRowBinding::RowsAAndRowsB;
}

BucketedCublasGemmShape shapeForCapacity(BucketedCublasGemmShape fullShape,
                                         BucketedCublasGemmRowBinding binding,
                                         uint64_t capacityRows) {
    if (capacityRows > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("BucketedCublasGemm capacity exceeds cuBLASLt int dimension range.");
    }

    const int rows = static_cast<int>(capacityRows);
    if (bindsRowsA(binding)) {
        fullShape.rowsA = rows;
    }
    if (bindsRowsB(binding)) {
        fullShape.rowsB = rows;
    }
    return fullShape;
}

void validateBuildArguments(uint64_t fullCapacityRows,
                            const BucketedCublasGemmShape &fullShape,
                            BucketedCublasGemmRowBinding rowBinding) {
    if (rowBinding != BucketedCublasGemmRowBinding::RowsA && rowBinding != BucketedCublasGemmRowBinding::RowsB &&
        rowBinding != BucketedCublasGemmRowBinding::RowsAAndRowsB) {
        throw std::invalid_argument("BucketedCublasGemm requires a valid row binding.");
    }
    if (fullCapacityRows == 0 || fullCapacityRows > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("BucketedCublasGemm requires fullCapacityRows in the cuBLASLt int dimension range.");
    }

    const int fullRows = static_cast<int>(fullCapacityRows);
    if (bindsRowsA(rowBinding) && fullShape.rowsA != fullRows) {
        throw std::invalid_argument("BucketedCublasGemm full shape rowsA must equal fullCapacityRows when rowsA is bucketed.");
    }
    if (bindsRowsB(rowBinding) && fullShape.rowsB != fullRows) {
        throw std::invalid_argument("BucketedCublasGemm full shape rowsB must equal fullCapacityRows when rowsB is bucketed.");
    }
}

}  // namespace

BucketedCublasGemm BucketedCublasGemm::build(int gpuNum,
                                              uint64_t fullCapacityRows,
                                              BucketedCublasGemmShape fullCapacityShape,
                                              BucketedCublasGemmRowBinding rowBinding,
                                              CublasMatrixMultiply::MatmulDataTypes dataTypes,
                                              bool printResults) {
    validateBuildArguments(fullCapacityRows, fullCapacityShape, rowBinding);

    BucketedCublasGemm built;
    built.fullCapacityRows = fullCapacityRows;
    built.capacityBuckets = makeRaggedMatmulCapacityBuckets(fullCapacityRows);
    built.buckets.reserve(built.capacityBuckets.size());

    CublasMatrixMultiply &cublas = CublasMatrixMultiply::instance();
    for (uint64_t capacityRows : built.capacityBuckets) {
        BucketedCublasGemmShape shape = shapeForCapacity(fullCapacityShape, rowBinding, capacityRows);

        cublas.chooseOptimalGemmKernel(gpuNum,
                                       shape.rowsA,
                                       shape.colsA,
                                       shape.rowsB,
                                       shape.colsB,
                                       shape.ldA,
                                       shape.ldB,
                                       shape.ldC,
                                       shape.ldD,
                                       shape.transposeA,
                                       shape.transposeB,
                                       shape.transposeC,
                                       dataTypes,
                                       printResults);

        CublasKernel kernel = cublas.getCachedGemmKernel(gpuNum,
                                                         shape.rowsA,
                                                         shape.colsA,
                                                         shape.rowsB,
                                                         shape.colsB,
                                                         shape.ldA,
                                                         shape.ldB,
                                                         shape.ldC,
                                                         shape.ldD,
                                                         shape.transposeA,
                                                         shape.transposeB,
                                                         shape.transposeC,
                                                         dataTypes,
                                                         true);

        built.workspaceSizeInBytes = std::max<uint64_t>(built.workspaceSizeInBytes, kernel.getWorkspaceSizeInBytes(gpuNum));
        built.buckets.push_back(Bucket{capacityRows, shape, std::move(kernel)});
    }

    return built;
}

BucketedCublasGemm::Bucket &BucketedCublasGemm::selectBucket(uint64_t activeRows) {
    return const_cast<Bucket &>(static_cast<const BucketedCublasGemm &>(*this).selectBucket(activeRows));
}

const BucketedCublasGemm::Bucket &BucketedCublasGemm::selectBucket(uint64_t activeRows) const {
    if (buckets.empty()) {
        throw std::logic_error("BucketedCublasGemm has not been built.");
    }

    // An all-empty ragged batch still needs a physical GEMM for dense outputs such
    // as parameter gradients, but it should use the smallest prebuilt capacity
    // rather than falling back to full packed capacity. The caller sanitizes that
    // smallest bucket to zero before launch, so the resulting dense reduction is
    // mathematically zero while inactive packed outputs remain undefined.
    const uint64_t selectedCapacity =
        activeRows == 0 ? capacityBuckets.front() : chooseRaggedMatmulCapacityBucket(activeRows, capacityBuckets);
    auto it = std::lower_bound(buckets.begin(), buckets.end(), selectedCapacity, [](const Bucket &bucket, uint64_t capacity) {
        return bucket.capacityRows < capacity;
    });
    if (it == buckets.end() || it->capacityRows != selectedCapacity) {
        throw std::logic_error("BucketedCublasGemm internal bucket/kernel mismatch.");
    }
    return *it;
}

uint64_t BucketedCublasGemm::getSelectedCapacityRows(uint64_t activeRows) const {
    return selectBucket(activeRows).capacityRows;
}

BucketedCublasGemmShape BucketedCublasGemm::getSelectedShape(uint64_t activeRows) const {
    return selectBucket(activeRows).shape;
}

CublasKernelRequirement BucketedCublasGemm::getSelectedKernelRequirement(uint64_t activeRows) const {
    return selectBucket(activeRows).kernel.getCublasKernelRequirement();
}

cublasStatus_t BucketedCublasGemm::launchUncheckedPrevalidated(uint64_t activeRows,
                                                                Tensor A,
                                                                Tensor B,
                                                                Tensor C,
                                                                Tensor D,
                                                                std::optional<Tensor> workspace,
                                                                const float *alpha,
                                                                const float *beta,
                                                                Stream stream,
                                                                CublasScalarPointerMode pointerMode,
                                                                CublasFp8MatmulScales fp8Scales) {
    Bucket &bucket = selectBucket(activeRows);
    return bucket.kernel.launchUncheckedPrevalidated(A, B, C, D, workspace, alpha, beta, stream, pointerMode, fp8Scales);
}

}  // namespace ThorImplementation
