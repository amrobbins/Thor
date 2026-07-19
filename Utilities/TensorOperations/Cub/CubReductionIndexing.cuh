#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal {

inline __host__ __device__ uint64_t mapLogicalReductionIndex(const CubReductionIndexing& indexing,
                                                              uint64_t output_index,
                                                              uint64_t reduction_index) {
    uint64_t physical_index = 0;

    for (int32_t retained = static_cast<int32_t>(indexing.retained_axis_count) - 1; retained >= 0; --retained) {
        const uint64_t dimension = indexing.retained_dimensions[retained];
        const uint64_t coordinate = output_index % dimension;
        output_index /= dimension;
        physical_index += coordinate * indexing.input_strides[indexing.retained_axes[retained]];
    }

    for (int32_t reduced = static_cast<int32_t>(indexing.reduced_axis_count) - 1; reduced >= 0; --reduced) {
        const uint64_t dimension = indexing.reduced_dimensions[reduced];
        const uint64_t coordinate = reduction_index % dimension;
        reduction_index /= dimension;
        physical_index += coordinate * indexing.input_strides[indexing.reduced_axes[reduced]];
    }

    return physical_index;
}

}  // namespace ThorImplementation::CubReductionInternal
