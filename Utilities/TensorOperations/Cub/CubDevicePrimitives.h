#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

// The CUB dtype policy is shared across all supported CUB primitives:
// 64-bit data/index types are off by default to control template-instantiation
// and package size, while FP8 data/key types are on by default so Thor CI can
// validate the empirically supported CUB surface.
//
// The original radix-sort-specific macros are still accepted as aliases when
// they are supplied by a build configuration. Prefer the general THOR_CUB_*
// names for new code.
#ifndef THOR_CUB_ENABLE_64BIT_TYPES
#ifdef THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS
#define THOR_CUB_ENABLE_64BIT_TYPES THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS
#else
#define THOR_CUB_ENABLE_64BIT_TYPES 0
#endif
#endif

#ifndef THOR_CUB_ENABLE_FP8_TYPES
#ifdef THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS
#define THOR_CUB_ENABLE_FP8_TYPES THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS
#else
#define THOR_CUB_ENABLE_FP8_TYPES 1
#endif
#endif

#ifndef THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS
#define THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS THOR_CUB_ENABLE_64BIT_TYPES
#endif

#ifndef THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS
#define THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS THOR_CUB_ENABLE_FP8_TYPES
#endif

namespace ThorImplementation {

enum class CubSortOrder : uint8_t { Ascending = 0, Descending = 1 };

struct CubTemporaryStoragePlan {
    TensorPlacement placement;
    size_t bytes = 0;
};

struct CubDeviceRadixSortKeysPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    uint64_t num_items = 0;
    CubSortOrder order = CubSortOrder::Ascending;
    int begin_bit = 0;
    int end_bit = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceRadixSortPairsPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    DataType value_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    CubSortOrder order = CubSortOrder::Ascending;
    int begin_bit = 0;
    int end_bit = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceRunLengthEncodePlan {
    TensorPlacement placement;
    DataType input_dtype = DataType::UINT8;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceExclusiveSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedRadixSortKeysPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    CubSortOrder order = CubSortOrder::Ascending;
    int begin_bit = 0;
    int end_bit = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedRadixSortPairsPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    DataType value_dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    CubSortOrder order = CubSortOrder::Ascending;
    int begin_bit = 0;
    int end_bit = 0;
    size_t temp_storage_bytes = 0;
};

[[nodiscard]] bool isCubRadixSortKeyDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubRadixSortValueDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubRunLengthEncodeDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubExclusiveSumDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSegmentOffsetDTypeSupported(DataType dtype);

[[nodiscard]] CubTemporaryStoragePlan cubTemporaryStoragePlan(const TensorPlacement& placement, size_t bytes);
[[nodiscard]] CubTemporaryStoragePlan cubMaxTemporaryStoragePlan(std::initializer_list<CubTemporaryStoragePlan> plans);
[[nodiscard]] Tensor allocateCubTemporaryStorage(const TensorPlacement& placement, size_t bytes);
[[nodiscard]] Tensor allocateCubTemporaryStorage(const CubTemporaryStoragePlan& plan);

[[nodiscard]] CubDeviceRadixSortKeysPlan prepareCubDeviceRadixSortKeys(const Tensor& keys_in,
                                                                       const Tensor& keys_out,
                                                                       uint64_t num_items,
                                                                       CubSortOrder order = CubSortOrder::Ascending,
                                                                       int begin_bit = 0,
                                                                       int end_bit = 0);

[[nodiscard]] size_t cubDeviceRadixSortKeysTempBytes(const Tensor& keys_in,
                                                     const Tensor& keys_out,
                                                     uint64_t num_items,
                                                     CubSortOrder order = CubSortOrder::Ascending,
                                                     int begin_bit = 0,
                                                     int end_bit = 0);

void cubDeviceRadixSortKeys(const CubDeviceRadixSortKeysPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& keys_in,
                            Tensor& keys_out,
                            Stream& stream);

void cubDeviceRadixSortKeys(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& keys_in,
                            Tensor& keys_out,
                            uint64_t num_items,
                            Stream& stream,
                            CubSortOrder order = CubSortOrder::Ascending,
                            int begin_bit = 0,
                            int end_bit = 0);

[[nodiscard]] CubDeviceRadixSortPairsPlan prepareCubDeviceRadixSortPairs(const Tensor& keys_in,
                                                                         const Tensor& keys_out,
                                                                         const Tensor& values_in,
                                                                         const Tensor& values_out,
                                                                         uint64_t num_items,
                                                                         CubSortOrder order = CubSortOrder::Ascending,
                                                                         int begin_bit = 0,
                                                                         int end_bit = 0);

[[nodiscard]] size_t cubDeviceRadixSortPairsTempBytes(const Tensor& keys_in,
                                                      const Tensor& keys_out,
                                                      const Tensor& values_in,
                                                      const Tensor& values_out,
                                                      uint64_t num_items,
                                                      CubSortOrder order = CubSortOrder::Ascending,
                                                      int begin_bit = 0,
                                                      int end_bit = 0);

void cubDeviceRadixSortPairs(const CubDeviceRadixSortPairsPlan& plan,
                             const Tensor& temp_storage,
                             const Tensor& keys_in,
                             Tensor& keys_out,
                             const Tensor& values_in,
                             Tensor& values_out,
                             Stream& stream);

void cubDeviceRadixSortPairs(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& keys_in,
                             Tensor& keys_out,
                             const Tensor& values_in,
                             Tensor& values_out,
                             uint64_t num_items,
                             Stream& stream,
                             CubSortOrder order = CubSortOrder::Ascending,
                             int begin_bit = 0,
                             int end_bit = 0);

[[nodiscard]] CubDeviceSegmentedRadixSortKeysPlan prepareCubDeviceSegmentedRadixSortKeys(
    const Tensor& keys_in,
    const Tensor& keys_out,
    const Tensor& segment_offsets,
    uint64_t num_items,
    uint64_t num_segments,
    CubSortOrder order = CubSortOrder::Ascending,
    int begin_bit = 0,
    int end_bit = 0);

[[nodiscard]] size_t cubDeviceSegmentedRadixSortKeysTempBytes(const Tensor& keys_in,
                                                              const Tensor& keys_out,
                                                              const Tensor& segment_offsets,
                                                              uint64_t num_items,
                                                              uint64_t num_segments,
                                                              CubSortOrder order = CubSortOrder::Ascending,
                                                              int begin_bit = 0,
                                                              int end_bit = 0);

void cubDeviceSegmentedRadixSortKeys(const CubDeviceSegmentedRadixSortKeysPlan& plan,
                                     const Tensor& temp_storage,
                                     const Tensor& keys_in,
                                     Tensor& keys_out,
                                     const Tensor& segment_offsets,
                                     Stream& stream);

void cubDeviceSegmentedRadixSortKeys(const Tensor& temp_storage,
                                     size_t temp_storage_bytes,
                                     const Tensor& keys_in,
                                     Tensor& keys_out,
                                     const Tensor& segment_offsets,
                                     uint64_t num_items,
                                     uint64_t num_segments,
                                     Stream& stream,
                                     CubSortOrder order = CubSortOrder::Ascending,
                                     int begin_bit = 0,
                                     int end_bit = 0);

[[nodiscard]] CubDeviceSegmentedRadixSortPairsPlan prepareCubDeviceSegmentedRadixSortPairs(
    const Tensor& keys_in,
    const Tensor& keys_out,
    const Tensor& values_in,
    const Tensor& values_out,
    const Tensor& segment_offsets,
    uint64_t num_items,
    uint64_t num_segments,
    CubSortOrder order = CubSortOrder::Ascending,
    int begin_bit = 0,
    int end_bit = 0);

[[nodiscard]] size_t cubDeviceSegmentedRadixSortPairsTempBytes(const Tensor& keys_in,
                                                               const Tensor& keys_out,
                                                               const Tensor& values_in,
                                                               const Tensor& values_out,
                                                               const Tensor& segment_offsets,
                                                               uint64_t num_items,
                                                               uint64_t num_segments,
                                                               CubSortOrder order = CubSortOrder::Ascending,
                                                               int begin_bit = 0,
                                                               int end_bit = 0);

void cubDeviceSegmentedRadixSortPairs(const CubDeviceSegmentedRadixSortPairsPlan& plan,
                                      const Tensor& temp_storage,
                                      const Tensor& keys_in,
                                      Tensor& keys_out,
                                      const Tensor& values_in,
                                      Tensor& values_out,
                                      const Tensor& segment_offsets,
                                      Stream& stream);

void cubDeviceSegmentedRadixSortPairs(const Tensor& temp_storage,
                                      size_t temp_storage_bytes,
                                      const Tensor& keys_in,
                                      Tensor& keys_out,
                                      const Tensor& values_in,
                                      Tensor& values_out,
                                      const Tensor& segment_offsets,
                                      uint64_t num_items,
                                      uint64_t num_segments,
                                      Stream& stream,
                                      CubSortOrder order = CubSortOrder::Ascending,
                                      int begin_bit = 0,
                                      int end_bit = 0);

[[nodiscard]] CubDeviceRunLengthEncodePlan prepareCubDeviceRunLengthEncode(const Tensor& input,
                                                                           const Tensor& unique_out,
                                                                           const Tensor& counts_out,
                                                                           const Tensor& num_runs_out,
                                                                           uint64_t num_items);

[[nodiscard]] size_t cubDeviceRunLengthEncodeTempBytes(const Tensor& input,
                                                       const Tensor& unique_out,
                                                       const Tensor& counts_out,
                                                       const Tensor& num_runs_out,
                                                       uint64_t num_items);

void cubDeviceRunLengthEncode(const CubDeviceRunLengthEncodePlan& plan,
                              const Tensor& temp_storage,
                              const Tensor& input,
                              Tensor& unique_out,
                              Tensor& counts_out,
                              Tensor& num_runs_out,
                              Stream& stream);

void cubDeviceRunLengthEncode(const Tensor& temp_storage,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& unique_out,
                              Tensor& counts_out,
                              Tensor& num_runs_out,
                              uint64_t num_items,
                              Stream& stream);

[[nodiscard]] CubDeviceExclusiveSumPlan prepareCubDeviceExclusiveSum(const Tensor& input,
                                                                     const Tensor& output,
                                                                     uint64_t num_items);

[[nodiscard]] size_t cubDeviceExclusiveSumTempBytes(const Tensor& input,
                                                    const Tensor& output,
                                                    uint64_t num_items);

void cubDeviceExclusiveSum(const CubDeviceExclusiveSumPlan& plan,
                           const Tensor& temp_storage,
                           const Tensor& input,
                           Tensor& output,
                           Stream& stream);

void cubDeviceExclusiveSum(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           uint64_t num_items,
                           Stream& stream);

}  // namespace ThorImplementation
