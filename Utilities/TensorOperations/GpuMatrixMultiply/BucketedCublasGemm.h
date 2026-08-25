#pragma once

#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

/**
 * Identifies which raw row dimensions of a row-major GEMM are controlled by the
 * packed-row capacity bucket.
 *
 * RowsA is the common forward/input-gradient shape where only A/output rows vary.
 * RowsAAndRowsB is the common weight-gradient shape X^T*dY where the packed token
 * count appears as rowsA and rowsB before transposeA is applied.
 */
enum class BucketedCublasGemmRowBinding : uint8_t {
    RowsA = 1,
    RowsB = 2,
    RowsAAndRowsB = 3,
};

struct BucketedCublasGemmShape {
    int rowsA;
    int colsA;
    int rowsB;
    int colsB;
    int ldA;
    int ldB;
    int ldC;
    int ldD;
    bool transposeA;
    bool transposeB;
    bool transposeC;
};

/**
 * A prebuilt family of ordinary cuBLASLt GEMMs keyed by packed-row capacity.
 *
 * Build time chooses/caches one ordinary CublasKernel for every capacity returned
 * by makeRaggedMatmulCapacityBuckets(fullCapacityRows). Runtime selection only
 * performs a lower_bound over those already-built capacities; it never performs a
 * global kernel-cache lookup or heuristic search.
 *
 * The runtime tensors may be physically allocated for fullCapacityRows. Selected
 * kernels carry smaller cuBLASLt matrix descriptors and therefore operate only on
 * the prefix represented by the selected capacity bucket. An active row count of
 * zero selects the smallest cached bucket so empty ragged batches do not fall back
 * to full-capacity work.
 */
class BucketedCublasGemm {
   public:
    BucketedCublasGemm() = default;

    static BucketedCublasGemm build(int gpuNum,
                                    uint64_t fullCapacityRows,
                                    BucketedCublasGemmShape fullCapacityShape,
                                    BucketedCublasGemmRowBinding rowBinding,
                                    CublasMatrixMultiply::MatmulDataTypes dataTypes,
                                    bool printResults = false);

    [[nodiscard]] uint64_t getFullCapacityRows() const { return fullCapacityRows; }
    [[nodiscard]] const std::vector<uint64_t> &getCapacityBuckets() const { return capacityBuckets; }
    [[nodiscard]] uint64_t getSelectedCapacityRows(uint64_t activeRows) const;
    [[nodiscard]] BucketedCublasGemmShape getSelectedShape(uint64_t activeRows) const;
    [[nodiscard]] CublasKernelRequirement getSelectedKernelRequirement(uint64_t activeRows) const;
    [[nodiscard]] CublasKernelSelection getSelectedKernelSelectionForTests(uint64_t activeRows) const;
    [[nodiscard]] uintptr_t getSelectedExecutionStateIdForTests(uint64_t activeRows) const;
    [[nodiscard]] uint64_t getWorkspaceSizeInBytes() const { return workspaceSizeInBytes; }

    cublasStatus_t launchUncheckedPrevalidated(uint64_t activeRows,
                                               Tensor A,
                                               Tensor B,
                                               Tensor C,
                                               Tensor D,
                                               std::optional<Tensor> workspace,
                                               const float *alpha,
                                               const float *beta,
                                               Stream stream,
                                               CublasScalarPointerMode pointerMode = CublasScalarPointerMode::Host,
                                               CublasFp8MatmulScales fp8Scales = CublasFp8MatmulScales::none());

   private:
    struct Bucket {
        uint64_t capacityRows;
        BucketedCublasGemmShape shape;
        CublasKernel kernel;
    };

    [[nodiscard]] Bucket &selectBucket(uint64_t activeRows);
    [[nodiscard]] const Bucket &selectBucket(uint64_t activeRows) const;

    uint64_t fullCapacityRows = 0;
    std::vector<uint64_t> capacityBuckets;
    std::vector<Bucket> buckets;
    uint64_t workspaceSizeInBytes = 0;
};

}  // namespace ThorImplementation
