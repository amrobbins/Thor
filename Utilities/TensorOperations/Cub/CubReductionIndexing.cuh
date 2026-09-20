#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal {

inline uint64_t mapLogicalReductionIndex(const CubReductionIndexing& indexing,
                                         uint64_t output_index,
                                         uint64_t reduction_index) {
    uint64_t physical_index = 0;

    for (size_t retained_index = indexing.retained_axes.size(); retained_index-- > 0;) {
        const uint64_t dimension = indexing.retained_dimensions[retained_index];
        const uint64_t coordinate = output_index % dimension;
        output_index /= dimension;
        physical_index += coordinate * indexing.input_strides[indexing.retained_axes[retained_index]];
    }

    for (size_t reduced_index = indexing.reduced_axes.size(); reduced_index-- > 0;) {
        const uint64_t dimension = indexing.reduced_dimensions[reduced_index];
        const uint64_t coordinate = reduction_index % dimension;
        reduction_index /= dimension;
        physical_index += coordinate * indexing.input_strides[indexing.reduced_axes[reduced_index]];
    }

    return physical_index;
}

template <typename IndexT>
inline __host__ __device__ IndexT mapLogicalReductionIndex(const CubReductionDeviceIndexingT<IndexT>& indexing,
                                                            IndexT output_index,
                                                            IndexT reduction_index) {
    IndexT physical_index = 0;

    for (uint32_t retained = indexing.retained_axis_count; retained-- > 0;) {
        const IndexT dimension = indexing.retained_dimensions[retained];
        const IndexT coordinate = output_index % dimension;
        output_index /= dimension;
        physical_index += coordinate * indexing.input_strides[indexing.retained_axes[retained]];
    }

    for (uint32_t reduced = indexing.reduced_axis_count; reduced-- > 0;) {
        const IndexT dimension = indexing.reduced_dimensions[reduced];
        const IndexT coordinate = reduction_index % dimension;
        reduction_index /= dimension;
        physical_index += coordinate * indexing.input_strides[indexing.reduced_axes[reduced]];
    }

    return physical_index;
}

}  // namespace ThorImplementation::CubReductionInternal
