#pragma once

#include <cuda_runtime.h>

#include <cstdint>

/**
 * Gather rows of bytes on a device. For each of the first logicalRows rows r, the rowBytes at
 * source + rowIndices[r] * sourceStrideBytes are copied to destination + r * rowBytes.
 *
 * source is the first byte of source row 0, so a field of compact records is passed as records + fieldOffset with the
 * record size as the stride, and a tensor of rows as its base with the row size as the stride.
 * A row whose index is not below sourceRows is left untouched, as is everything past logicalRows.
 * Requires logicalRows >= 1, sourceRows >= 1 and 1 <= rowBytes <= sourceStrideBytes.
 *
 * This is the one row-gather kernel of the residency code; the named gather and the direct field materialization
 * validate their tensors and launch it.
 */
void launchDeviceResidentRowGatherKernel(const uint8_t *source,
                                         uint64_t sourceRows,
                                         uint64_t sourceStrideBytes,
                                         uint8_t *destination,
                                         const uint64_t *rowIndices,
                                         uint64_t logicalRows,
                                         uint64_t rowBytes,
                                         cudaStream_t stream);
