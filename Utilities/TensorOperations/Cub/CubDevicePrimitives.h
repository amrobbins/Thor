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
enum class CubTopKOrder : uint8_t { Smallest = 0, Largest = 1 };
enum class CubScanOp : uint8_t { Sum = 0, Min = 1, Max = 2, Product = 3 };
enum class CubScanMode : uint8_t { Exclusive = 0, Inclusive = 1 };

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

struct CubDeviceTopKKeysPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    uint64_t num_items = 0;
    uint64_t k = 0;
    CubTopKOrder order = CubTopKOrder::Largest;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceTopKPairsPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    DataType value_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t k = 0;
    CubTopKOrder order = CubTopKOrder::Largest;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedTopKKeysPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    uint64_t num_segments = 0;
    uint64_t segment_size = 0;
    uint64_t k = 0;
    CubTopKOrder order = CubTopKOrder::Largest;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedTopKPairsPlan {
    TensorPlacement placement;
    DataType key_dtype = DataType::UINT8;
    DataType value_dtype = DataType::UINT32;
    uint64_t num_segments = 0;
    uint64_t segment_size = 0;
    uint64_t k = 0;
    CubTopKOrder order = CubTopKOrder::Largest;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSelectFlaggedPlan {
    TensorPlacement placement;
    DataType input_dtype = DataType::UINT8;
    DataType flag_dtype = DataType::BOOLEAN;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceLowerBoundPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT8;
    uint64_t range_num_items = 0;
    uint64_t values_num_items = 0;
    CubSortOrder order = CubSortOrder::Ascending;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceUpperBoundPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT8;
    uint64_t range_num_items = 0;
    uint64_t values_num_items = 0;
    CubSortOrder order = CubSortOrder::Ascending;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceFindIfFlaggedPlan {
    TensorPlacement placement;
    DataType flag_dtype = DataType::BOOLEAN;
    uint64_t num_items = 0;
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

struct CubDeviceInclusiveSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    CubScanOp op = CubScanOp::Sum;
    CubScanMode mode = CubScanMode::Exclusive;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceReduceSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceReduceMaxPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceReduceMinPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedExclusiveSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedInclusiveSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedUniformExclusiveSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    uint64_t segment_size = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedUniformInclusiveSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    uint64_t segment_size = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedUniformScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    uint64_t segment_size = 0;
    CubScanOp op = CubScanOp::Sum;
    CubScanMode mode = CubScanMode::Exclusive;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedReduceSumPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedReduceMaxPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
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
[[nodiscard]] bool isCubTopKKeyDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubTopKValueDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSelectValueDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSelectFlagDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubFindKeyDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubFindFlagDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubRunLengthEncodeDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubExclusiveSumDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubScanDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubReduceSumDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubReduceMaxDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubReduceMinDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSegmentedExclusiveSumDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSegmentedReduceSumDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSegmentedReduceMaxDTypeSupported(DataType dtype);
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

[[nodiscard]] CubDeviceTopKKeysPlan prepareCubDeviceTopKKeys(const Tensor& keys_in,
                                                               const Tensor& keys_out,
                                                               uint64_t num_items,
                                                               uint64_t k,
                                                               CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] size_t cubDeviceTopKKeysTempBytes(const Tensor& keys_in,
                                                const Tensor& keys_out,
                                                uint64_t num_items,
                                                uint64_t k,
                                                CubTopKOrder order = CubTopKOrder::Largest);

void cubDeviceTopKKeys(const CubDeviceTopKKeysPlan& plan,
                       const Tensor& temp_storage,
                       const Tensor& keys_in,
                       Tensor& keys_out,
                       Stream& stream);

void cubDeviceTopKKeys(const Tensor& temp_storage,
                       size_t temp_storage_bytes,
                       const Tensor& keys_in,
                       Tensor& keys_out,
                       uint64_t num_items,
                       uint64_t k,
                       Stream& stream,
                       CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] CubDeviceTopKPairsPlan prepareCubDeviceTopKPairs(const Tensor& keys_in,
                                                               const Tensor& keys_out,
                                                               const Tensor& values_in,
                                                               const Tensor& values_out,
                                                               uint64_t num_items,
                                                               uint64_t k,
                                                               CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] size_t cubDeviceTopKPairsTempBytes(const Tensor& keys_in,
                                                 const Tensor& keys_out,
                                                 const Tensor& values_in,
                                                 const Tensor& values_out,
                                                 uint64_t num_items,
                                                 uint64_t k,
                                                 CubTopKOrder order = CubTopKOrder::Largest);

void cubDeviceTopKPairs(const CubDeviceTopKPairsPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& keys_in,
                        Tensor& keys_out,
                        const Tensor& values_in,
                        Tensor& values_out,
                        Stream& stream);

void cubDeviceTopKPairs(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& keys_in,
                        Tensor& keys_out,
                        const Tensor& values_in,
                        Tensor& values_out,
                        uint64_t num_items,
                        uint64_t k,
                        Stream& stream,
                        CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] CubDeviceSegmentedTopKKeysPlan prepareCubDeviceSegmentedTopKKeys(
    const Tensor& keys_in,
    const Tensor& keys_out,
    uint64_t num_segments,
    uint64_t segment_size,
    uint64_t k,
    CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] size_t cubDeviceSegmentedTopKKeysTempBytes(const Tensor& keys_in,
                                                         const Tensor& keys_out,
                                                         uint64_t num_segments,
                                                         uint64_t segment_size,
                                                         uint64_t k,
                                                         CubTopKOrder order = CubTopKOrder::Largest);

void cubDeviceSegmentedTopKKeys(const CubDeviceSegmentedTopKKeysPlan& plan,
                                const Tensor& temp_storage,
                                const Tensor& keys_in,
                                Tensor& keys_out,
                                Stream& stream);

void cubDeviceSegmentedTopKKeys(const Tensor& temp_storage,
                                size_t temp_storage_bytes,
                                const Tensor& keys_in,
                                Tensor& keys_out,
                                uint64_t num_segments,
                                uint64_t segment_size,
                                uint64_t k,
                                Stream& stream,
                                CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] CubDeviceSegmentedTopKPairsPlan prepareCubDeviceSegmentedTopKPairs(
    const Tensor& keys_in,
    const Tensor& keys_out,
    const Tensor& values_in,
    const Tensor& values_out,
    uint64_t num_segments,
    uint64_t segment_size,
    uint64_t k,
    CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] size_t cubDeviceSegmentedTopKPairsTempBytes(const Tensor& keys_in,
                                                          const Tensor& keys_out,
                                                          const Tensor& values_in,
                                                          const Tensor& values_out,
                                                          uint64_t num_segments,
                                                          uint64_t segment_size,
                                                          uint64_t k,
                                                          CubTopKOrder order = CubTopKOrder::Largest);

void cubDeviceSegmentedTopKPairs(const CubDeviceSegmentedTopKPairsPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& keys_in,
                                 Tensor& keys_out,
                                 const Tensor& values_in,
                                 Tensor& values_out,
                                 Stream& stream);

void cubDeviceSegmentedTopKPairs(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& keys_in,
                                 Tensor& keys_out,
                                 const Tensor& values_in,
                                 Tensor& values_out,
                                 uint64_t num_segments,
                                 uint64_t segment_size,
                                 uint64_t k,
                                 Stream& stream,
                                 CubTopKOrder order = CubTopKOrder::Largest);

[[nodiscard]] CubDeviceSelectFlaggedPlan prepareCubDeviceSelectFlagged(const Tensor& input,
                                                                       const Tensor& flags,
                                                                       const Tensor& output,
                                                                       const Tensor& num_selected_out,
                                                                       uint64_t num_items);

[[nodiscard]] size_t cubDeviceSelectFlaggedTempBytes(const Tensor& input,
                                                     const Tensor& flags,
                                                     const Tensor& output,
                                                     const Tensor& num_selected_out,
                                                     uint64_t num_items);

void cubDeviceSelectFlagged(const CubDeviceSelectFlaggedPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& input,
                            const Tensor& flags,
                            Tensor& output,
                            Tensor& num_selected_out,
                            Stream& stream);

void cubDeviceSelectFlagged(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            const Tensor& flags,
                            Tensor& output,
                            Tensor& num_selected_out,
                            uint64_t num_items,
                            Stream& stream);

[[nodiscard]] CubDeviceLowerBoundPlan prepareCubDeviceLowerBound(const Tensor& range,
                                                                 const Tensor& values,
                                                                 const Tensor& output,
                                                                 uint64_t range_num_items,
                                                                 uint64_t values_num_items,
                                                                 CubSortOrder order = CubSortOrder::Ascending);

[[nodiscard]] size_t cubDeviceLowerBoundTempBytes(const Tensor& range,
                                                  const Tensor& values,
                                                  const Tensor& output,
                                                  uint64_t range_num_items,
                                                  uint64_t values_num_items,
                                                  CubSortOrder order = CubSortOrder::Ascending);

void cubDeviceLowerBound(const CubDeviceLowerBoundPlan& plan,
                         const Tensor& temp_storage,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         Stream& stream);

void cubDeviceLowerBound(const Tensor& temp_storage,
                         size_t temp_storage_bytes,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         uint64_t range_num_items,
                         uint64_t values_num_items,
                         Stream& stream,
                         CubSortOrder order = CubSortOrder::Ascending);

[[nodiscard]] CubDeviceUpperBoundPlan prepareCubDeviceUpperBound(const Tensor& range,
                                                                 const Tensor& values,
                                                                 const Tensor& output,
                                                                 uint64_t range_num_items,
                                                                 uint64_t values_num_items,
                                                                 CubSortOrder order = CubSortOrder::Ascending);

[[nodiscard]] size_t cubDeviceUpperBoundTempBytes(const Tensor& range,
                                                  const Tensor& values,
                                                  const Tensor& output,
                                                  uint64_t range_num_items,
                                                  uint64_t values_num_items,
                                                  CubSortOrder order = CubSortOrder::Ascending);

void cubDeviceUpperBound(const CubDeviceUpperBoundPlan& plan,
                         const Tensor& temp_storage,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         Stream& stream);

void cubDeviceUpperBound(const Tensor& temp_storage,
                         size_t temp_storage_bytes,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         uint64_t range_num_items,
                         uint64_t values_num_items,
                         Stream& stream,
                         CubSortOrder order = CubSortOrder::Ascending);

[[nodiscard]] CubDeviceFindIfFlaggedPlan prepareCubDeviceFindIfFlagged(const Tensor& flags,
                                                                       const Tensor& index_out,
                                                                       uint64_t num_items);

[[nodiscard]] size_t cubDeviceFindIfFlaggedTempBytes(const Tensor& flags,
                                                     const Tensor& index_out,
                                                     uint64_t num_items);

void cubDeviceFindIfFlagged(const CubDeviceFindIfFlaggedPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& flags,
                            Tensor& index_out,
                            Stream& stream);

void cubDeviceFindIfFlagged(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& flags,
                            Tensor& index_out,
                            uint64_t num_items,
                            Stream& stream);

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

[[nodiscard]] CubDeviceSegmentedExclusiveSumPlan prepareCubDeviceSegmentedExclusiveSum(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_items,
    uint64_t num_segments);

[[nodiscard]] size_t cubDeviceSegmentedExclusiveSumTempBytes(const Tensor& input,
                                                             const Tensor& output,
                                                             const Tensor& segment_offsets,
                                                             uint64_t num_items,
                                                             uint64_t num_segments);

void cubDeviceSegmentedExclusiveSum(const CubDeviceSegmentedExclusiveSumPlan& plan,
                                    const Tensor& temp_storage,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    Stream& stream);

void cubDeviceSegmentedExclusiveSum(const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    Stream& stream);

[[nodiscard]] CubDeviceSegmentedInclusiveSumPlan prepareCubDeviceSegmentedInclusiveSum(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_items,
    uint64_t num_segments);

[[nodiscard]] size_t cubDeviceSegmentedInclusiveSumTempBytes(const Tensor& input,
                                                             const Tensor& output,
                                                             const Tensor& segment_offsets,
                                                             uint64_t num_items,
                                                             uint64_t num_segments);

void cubDeviceSegmentedInclusiveSum(const CubDeviceSegmentedInclusiveSumPlan& plan,
                                    const Tensor& temp_storage,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    Stream& stream);

void cubDeviceSegmentedInclusiveSum(const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    Stream& stream);

[[nodiscard]] CubDeviceSegmentedUniformExclusiveSumPlan prepareCubDeviceSegmentedUniformExclusiveSum(
    const Tensor& input,
    const Tensor& output,
    uint64_t num_items,
    uint64_t num_segments,
    uint64_t segment_size);

[[nodiscard]] size_t cubDeviceSegmentedUniformExclusiveSumTempBytes(const Tensor& input,
                                                                    const Tensor& output,
                                                                    uint64_t num_items,
                                                                    uint64_t num_segments,
                                                                    uint64_t segment_size);

void cubDeviceSegmentedUniformExclusiveSum(const CubDeviceSegmentedUniformExclusiveSumPlan& plan,
                                           const Tensor& temp_storage,
                                           const Tensor& input,
                                           Tensor& output,
                                           Stream& stream);

[[nodiscard]] CubDeviceSegmentedUniformInclusiveSumPlan prepareCubDeviceSegmentedUniformInclusiveSum(
    const Tensor& input,
    const Tensor& output,
    uint64_t num_items,
    uint64_t num_segments,
    uint64_t segment_size);

[[nodiscard]] size_t cubDeviceSegmentedUniformInclusiveSumTempBytes(const Tensor& input,
                                                                    const Tensor& output,
                                                                    uint64_t num_items,
                                                                    uint64_t num_segments,
                                                                    uint64_t segment_size);

void cubDeviceSegmentedUniformInclusiveSum(const CubDeviceSegmentedUniformInclusiveSumPlan& plan,
                                           const Tensor& temp_storage,
                                           const Tensor& input,
                                           Tensor& output,
                                           Stream& stream);

[[nodiscard]] CubDeviceSegmentedUniformScanPlan prepareCubDeviceSegmentedUniformScan(const Tensor& input,
                                                                                     const Tensor& output,
                                                                                     uint64_t num_items,
                                                                                     uint64_t num_segments,
                                                                                     uint64_t segment_size,
                                                                                     CubScanOp op,
                                                                                     CubScanMode mode);

[[nodiscard]] size_t cubDeviceSegmentedUniformScanTempBytes(const Tensor& input,
                                                            const Tensor& output,
                                                            uint64_t num_items,
                                                            uint64_t num_segments,
                                                            uint64_t segment_size,
                                                            CubScanOp op,
                                                            CubScanMode mode);

void cubDeviceSegmentedUniformScan(const CubDeviceSegmentedUniformScanPlan& plan,
                                   const Tensor& temp_storage,
                                   const Tensor& input,
                                   Tensor& output,
                                   Stream& stream);

[[nodiscard]] CubDeviceSegmentedReduceSumPlan prepareCubDeviceSegmentedReduceSum(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_items,
    uint64_t num_segments);

[[nodiscard]] size_t cubDeviceSegmentedReduceSumTempBytes(const Tensor& input,
                                                          const Tensor& output,
                                                          const Tensor& segment_offsets,
                                                          uint64_t num_items,
                                                          uint64_t num_segments);

void cubDeviceSegmentedReduceSum(const CubDeviceSegmentedReduceSumPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 Stream& stream);

void cubDeviceSegmentedReduceSum(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 Stream& stream);

[[nodiscard]] CubDeviceSegmentedReduceMaxPlan prepareCubDeviceSegmentedReduceMax(
    const Tensor& input,
    const Tensor& output,
    const Tensor& segment_offsets,
    uint64_t num_items,
    uint64_t num_segments);

[[nodiscard]] size_t cubDeviceSegmentedReduceMaxTempBytes(const Tensor& input,
                                                          const Tensor& output,
                                                          const Tensor& segment_offsets,
                                                          uint64_t num_items,
                                                          uint64_t num_segments);

void cubDeviceSegmentedReduceMax(const CubDeviceSegmentedReduceMaxPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 Stream& stream);

void cubDeviceSegmentedReduceMax(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 Stream& stream);

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

[[nodiscard]] CubDeviceReduceSumPlan prepareCubDeviceReduceSum(const Tensor& input,
                                                               const Tensor& output,
                                                               uint64_t num_items);

[[nodiscard]] size_t cubDeviceReduceSumTempBytes(const Tensor& input,
                                                 const Tensor& output,
                                                 uint64_t num_items);

void cubDeviceReduceSum(const CubDeviceReduceSumPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& input,
                        Tensor& output,
                        Stream& stream);

void cubDeviceReduceSum(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        uint64_t num_items,
                        Stream& stream);

[[nodiscard]] CubDeviceReduceMaxPlan prepareCubDeviceReduceMax(const Tensor& input,
                                                               const Tensor& output,
                                                               uint64_t num_items);

[[nodiscard]] size_t cubDeviceReduceMaxTempBytes(const Tensor& input,
                                                 const Tensor& output,
                                                 uint64_t num_items);

void cubDeviceReduceMax(const CubDeviceReduceMaxPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& input,
                        Tensor& output,
                        Stream& stream);

void cubDeviceReduceMax(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        uint64_t num_items,
                        Stream& stream);

[[nodiscard]] CubDeviceReduceMinPlan prepareCubDeviceReduceMin(const Tensor& input,
                                                               const Tensor& output,
                                                               uint64_t num_items);

[[nodiscard]] size_t cubDeviceReduceMinTempBytes(const Tensor& input,
                                                 const Tensor& output,
                                                 uint64_t num_items);

void cubDeviceReduceMin(const CubDeviceReduceMinPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& input,
                        Tensor& output,
                        Stream& stream);

void cubDeviceReduceMin(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
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

[[nodiscard]] CubDeviceInclusiveSumPlan prepareCubDeviceInclusiveSum(const Tensor& input,
                                                                     const Tensor& output,
                                                                     uint64_t num_items);

[[nodiscard]] size_t cubDeviceInclusiveSumTempBytes(const Tensor& input,
                                                    const Tensor& output,
                                                    uint64_t num_items);

void cubDeviceInclusiveSum(const CubDeviceInclusiveSumPlan& plan,
                           const Tensor& temp_storage,
                           const Tensor& input,
                           Tensor& output,
                           Stream& stream);

void cubDeviceInclusiveSum(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           uint64_t num_items,
                           Stream& stream);

[[nodiscard]] CubDeviceScanPlan prepareCubDeviceScan(const Tensor& input,
                                                     const Tensor& output,
                                                     uint64_t num_items,
                                                     CubScanOp op,
                                                     CubScanMode mode);

[[nodiscard]] size_t cubDeviceScanTempBytes(const Tensor& input,
                                            const Tensor& output,
                                            uint64_t num_items,
                                            CubScanOp op,
                                            CubScanMode mode);

void cubDeviceScan(const CubDeviceScanPlan& plan,
                   const Tensor& temp_storage,
                   const Tensor& input,
                   Tensor& output,
                   Stream& stream);

void cubDeviceScan(const Tensor& temp_storage,
                   size_t temp_storage_bytes,
                   const Tensor& input,
                   Tensor& output,
                   uint64_t num_items,
                   Stream& stream,
                   CubScanOp op,
                   CubScanMode mode);

}  // namespace ThorImplementation
