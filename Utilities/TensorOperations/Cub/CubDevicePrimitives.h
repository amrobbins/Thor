#pragma once

#include <cstddef>
#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"

namespace ThorImplementation {

enum class CubScanOp : uint8_t { Sum = 0, Min = 1, Max = 2, Product = 3 };
enum class CubArgScanOp : uint8_t { ArgMin = 0, ArgMax = 1 };
enum class CubScanMode : uint8_t { Exclusive = 0, Inclusive = 1 };
enum class CubScanDirection : uint8_t { Forward = 0, Reverse = 1 };

struct CubDeviceScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    CubScanOp op = CubScanOp::Sum;
    CubScanMode mode = CubScanMode::Exclusive;
    CubScanDirection direction = CubScanDirection::Forward;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceArgScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    CubArgScanOp op = CubArgScanOp::ArgMin;
    CubScanMode mode = CubScanMode::Inclusive;
    CubScanDirection direction = CubScanDirection::Forward;
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
    CubScanDirection direction = CubScanDirection::Forward;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedUniformArgScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    uint64_t segment_size = 0;
    CubArgScanOp op = CubArgScanOp::ArgMin;
    CubScanMode mode = CubScanMode::Inclusive;
    CubScanDirection direction = CubScanDirection::Forward;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    CubScanOp op = CubScanOp::Sum;
    CubScanMode mode = CubScanMode::Exclusive;
    CubScanDirection direction = CubScanDirection::Forward;
    size_t temp_storage_bytes = 0;
};

struct CubDeviceSegmentedArgScanPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
    DataType offset_dtype = DataType::UINT32;
    uint64_t num_items = 0;
    uint64_t num_segments = 0;
    CubArgScanOp op = CubArgScanOp::ArgMin;
    CubScanMode mode = CubScanMode::Inclusive;
    CubScanDirection direction = CubScanDirection::Forward;
    size_t temp_storage_bytes = 0;
};

[[nodiscard]] bool isCubScanDTypeSupported(DataType dtype);
[[nodiscard]] bool isCubSegmentOffsetDTypeSupported(DataType dtype);

[[nodiscard]] Tensor allocateCubTemporaryStorage(const TensorPlacement& placement, size_t bytes);

[[nodiscard]] CubDeviceSegmentedUniformScanPlan prepareCubDeviceSegmentedUniformScan(const Tensor& input,
                                                                                     const Tensor& output,
                                                                                     uint64_t num_items,
                                                                                     uint64_t num_segments,
                                                                                     uint64_t segment_size,
                                                                                     CubScanOp op,
                                                                                     CubScanMode mode,
                                                                                     CubScanDirection direction = CubScanDirection::Forward);

void cubDeviceSegmentedUniformScan(const CubDeviceSegmentedUniformScanPlan& plan,
                                   const Tensor& temp_storage,
                                   const Tensor& input,
                                   Tensor& output,
                                   Stream& stream);

[[nodiscard]] CubDeviceSegmentedUniformArgScanPlan prepareCubDeviceSegmentedUniformArgScan(
    const Tensor& input,
    const Tensor& output,
    uint64_t num_items,
    uint64_t num_segments,
    uint64_t segment_size,
    CubArgScanOp op,
    CubScanMode mode,
    CubScanDirection direction = CubScanDirection::Forward);

void cubDeviceSegmentedUniformArgScan(const CubDeviceSegmentedUniformArgScanPlan& plan,
                                      const Tensor& temp_storage,
                                      const Tensor& input,
                                      Tensor& output,
                                      Stream& stream);

[[nodiscard]] CubDeviceSegmentedScanPlan prepareCubDeviceSegmentedScan(const Tensor& input,
                                                                       const Tensor& output,
                                                                       const Tensor& segment_offsets,
                                                                       uint64_t num_items,
                                                                       uint64_t num_segments,
                                                                       CubScanOp op,
                                                                       CubScanMode mode,
                                                                       CubScanDirection direction = CubScanDirection::Forward);

void cubDeviceSegmentedScan(const CubDeviceSegmentedScanPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& input,
                            Tensor& output,
                            const Tensor& segment_offsets,
                            Stream& stream);

[[nodiscard]] CubDeviceSegmentedArgScanPlan prepareCubDeviceSegmentedArgScan(const Tensor& input,
                                                                             const Tensor& output,
                                                                             const Tensor& segment_offsets,
                                                                             uint64_t num_items,
                                                                             uint64_t num_segments,
                                                                             CubArgScanOp op,
                                                                             CubScanMode mode,
                                                                             CubScanDirection direction = CubScanDirection::Forward);

void cubDeviceSegmentedArgScan(const CubDeviceSegmentedArgScanPlan& plan,
                               const Tensor& temp_storage,
                               const Tensor& input,
                               Tensor& output,
                               const Tensor& segment_offsets,
                               Stream& stream);

[[nodiscard]] CubDeviceScanPlan prepareCubDeviceScan(const Tensor& input,
                                                     const Tensor& output,
                                                     uint64_t num_items,
                                                     CubScanOp op,
                                                     CubScanMode mode,
                                                     CubScanDirection direction = CubScanDirection::Forward);

void cubDeviceScan(const CubDeviceScanPlan& plan,
                   const Tensor& temp_storage,
                   const Tensor& input,
                   Tensor& output,
                   Stream& stream);

[[nodiscard]] CubDeviceArgScanPlan prepareCubDeviceArgScan(const Tensor& input,
                                                           const Tensor& output,
                                                           uint64_t num_items,
                                                           CubArgScanOp op,
                                                           CubScanMode mode,
                                                           CubScanDirection direction = CubScanDirection::Forward);

void cubDeviceArgScan(const CubDeviceArgScanPlan& plan,
                      const Tensor& temp_storage,
                      const Tensor& input,
                      Tensor& output,
                      Stream& stream);

void cubDeviceArgScanValuesFromIndices(const Tensor& input,
                                       const Tensor& indices,
                                       Tensor& values,
                                       uint64_t num_items,
                                       CubArgScanOp op,
                                       Stream& stream);

}  // namespace ThorImplementation
