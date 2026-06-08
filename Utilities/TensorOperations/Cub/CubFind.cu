#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_find.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename T>
struct CubFindLess {
    __host__ __device__ bool operator()(const T& a, const T& b) const { return a < b; }
};

template <typename T>
struct CubFindGreater {
    __host__ __device__ bool operator()(const T& a, const T& b) const { return b < a; }
};

template <typename FlagT>
struct CubFindTruthy {
    __host__ __device__ bool operator()(const FlagT& value) const { return static_cast<bool>(value); }
};

template <typename T, typename Fn>
decltype(auto) dispatchFindComparator(CubSortOrder order, Fn&& fn) {
    switch (order) {
        case CubSortOrder::Ascending:
            return fn(CubFindLess<T>{});
        case CubSortOrder::Descending:
            return fn(CubFindGreater<T>{});
        default:
            throw std::invalid_argument("Unsupported CUB find order.");
    }
}

template <typename T, typename CompareT>
size_t queryLowerBoundBytes(const Tensor& range,
                            const Tensor& values,
                            const Tensor& output,
                            int range_items,
                            int values_items,
                            CompareT comp) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceFind::LowerBound(nullptr,
                                           queried_bytes,
                                           range.getMemPtr<T>(),
                                           range_items,
                                           values.getMemPtr<T>(),
                                           values_items,
                                           const_cast<uint32_t*>(output.getMemPtr<uint32_t>()),
                                           comp));
    return queried_bytes;
}

template <typename T, typename CompareT>
size_t queryUpperBoundBytes(const Tensor& range,
                            const Tensor& values,
                            const Tensor& output,
                            int range_items,
                            int values_items,
                            CompareT comp) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceFind::UpperBound(nullptr,
                                           queried_bytes,
                                           range.getMemPtr<T>(),
                                           range_items,
                                           values.getMemPtr<T>(),
                                           values_items,
                                           const_cast<uint32_t*>(output.getMemPtr<uint32_t>()),
                                           comp));
    return queried_bytes;
}

template <typename T, typename CompareT>
void launchLowerBound(void* temp_storage_ptr,
                      size_t temp_storage_bytes,
                      const Tensor& range,
                      const Tensor& values,
                      Tensor& output,
                      int range_items,
                      int values_items,
                      CompareT comp,
                      Stream& stream) {
    CUDA_CHECK(cub::DeviceFind::LowerBound(temp_storage_ptr,
                                           temp_storage_bytes,
                                           range.getMemPtr<T>(),
                                           range_items,
                                           values.getMemPtr<T>(),
                                           values_items,
                                           output.getMemPtr<uint32_t>(),
                                           comp,
                                           stream.getStream()));
}

template <typename T, typename CompareT>
void launchUpperBound(void* temp_storage_ptr,
                      size_t temp_storage_bytes,
                      const Tensor& range,
                      const Tensor& values,
                      Tensor& output,
                      int range_items,
                      int values_items,
                      CompareT comp,
                      Stream& stream) {
    CUDA_CHECK(cub::DeviceFind::UpperBound(temp_storage_ptr,
                                           temp_storage_bytes,
                                           range.getMemPtr<T>(),
                                           range_items,
                                           values.getMemPtr<T>(),
                                           values_items,
                                           output.getMemPtr<uint32_t>(),
                                           comp,
                                           stream.getStream()));
}

template <typename FlagT>
size_t queryFindIfFlaggedBytes(const Tensor& flags, const Tensor& index_out, int cub_items) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceFind::FindIf(nullptr,
                                       queried_bytes,
                                       flags.getMemPtr<FlagT>(),
                                       const_cast<uint32_t*>(index_out.getMemPtr<uint32_t>()),
                                       CubFindTruthy<FlagT>{},
                                       cub_items));
    return queried_bytes;
}

template <typename FlagT>
void launchFindIfFlagged(void* temp_storage_ptr,
                         size_t temp_storage_bytes,
                         const Tensor& flags,
                         Tensor& index_out,
                         int cub_items,
                         Stream& stream) {
    CUDA_CHECK(cub::DeviceFind::FindIf(temp_storage_ptr,
                                       temp_storage_bytes,
                                       flags.getMemPtr<FlagT>(),
                                       index_out.getMemPtr<uint32_t>(),
                                       CubFindTruthy<FlagT>{},
                                       cub_items,
                                       stream.getStream()));
}

}  // namespace

CubDeviceLowerBoundPlan prepareCubDeviceLowerBound(const Tensor& range,
                                                   const Tensor& values,
                                                   const Tensor& output,
                                                   uint64_t range_num_items,
                                                   uint64_t values_num_items,
                                                   CubSortOrder order) {
    validateFindBounds(range, values, output, range_num_items, values_num_items, "lower-bound");
    const int range_items = checkedCubNumItems(range_num_items);
    const int values_items = checkedCubNumItems(values_num_items);

    size_t bytes = 1;
    if (values_num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            return dispatchFindComparator<T>(order, [&](auto comp) -> size_t {
                return queryLowerBoundBytes<T>(range, values, output, range_items, values_items, comp);
            });
        };
        bytes = dispatchFindKeyDType(range.getDataType(), query);
    }

    CubDeviceLowerBoundPlan plan;
    plan.placement = range.getPlacement();
    plan.dtype = range.getDataType();
    plan.range_num_items = range_num_items;
    plan.values_num_items = values_num_items;
    plan.order = order;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceLowerBoundTempBytes(const Tensor& range,
                                    const Tensor& values,
                                    const Tensor& output,
                                    uint64_t range_num_items,
                                    uint64_t values_num_items,
                                    CubSortOrder order) {
    return prepareCubDeviceLowerBound(range, values, output, range_num_items, values_num_items, order).temp_storage_bytes;
}

void cubDeviceLowerBound(const CubDeviceLowerBoundPlan& plan,
                         const Tensor& temp_storage,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         Stream& stream) {
    validateFindBounds(range, values, output, plan.range_num_items, plan.values_num_items, "lower-bound");
    if (range.getPlacement() != plan.placement || range.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB lower-bound plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.values_num_items == 0) {
        return;
    }

    const int range_items = checkedCubNumItems(plan.range_num_items);
    const int values_items = checkedCubNumItems(plan.values_num_items);
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void {
        dispatchFindComparator<T>(plan.order, [&](auto comp) -> void {
            launchLowerBound<T>(temp_storage_ptr, temp_storage_bytes, range, values, output, range_items, values_items, comp, stream);
        });
    };
    dispatchFindKeyDType(plan.dtype, launch);
}

void cubDeviceLowerBound(const Tensor& temp_storage,
                         size_t temp_storage_bytes,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         uint64_t range_num_items,
                         uint64_t values_num_items,
                         Stream& stream,
                         CubSortOrder order) {
    CubDeviceLowerBoundPlan plan = prepareCubDeviceLowerBound(range, values, output, range_num_items, values_num_items, order);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB lower-bound requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceLowerBound(plan, temp_storage, range, values, output, stream);
}

CubDeviceUpperBoundPlan prepareCubDeviceUpperBound(const Tensor& range,
                                                   const Tensor& values,
                                                   const Tensor& output,
                                                   uint64_t range_num_items,
                                                   uint64_t values_num_items,
                                                   CubSortOrder order) {
    validateFindBounds(range, values, output, range_num_items, values_num_items, "upper-bound");
    const int range_items = checkedCubNumItems(range_num_items);
    const int values_items = checkedCubNumItems(values_num_items);

    size_t bytes = 1;
    if (values_num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            return dispatchFindComparator<T>(order, [&](auto comp) -> size_t {
                return queryUpperBoundBytes<T>(range, values, output, range_items, values_items, comp);
            });
        };
        bytes = dispatchFindKeyDType(range.getDataType(), query);
    }

    CubDeviceUpperBoundPlan plan;
    plan.placement = range.getPlacement();
    plan.dtype = range.getDataType();
    plan.range_num_items = range_num_items;
    plan.values_num_items = values_num_items;
    plan.order = order;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceUpperBoundTempBytes(const Tensor& range,
                                    const Tensor& values,
                                    const Tensor& output,
                                    uint64_t range_num_items,
                                    uint64_t values_num_items,
                                    CubSortOrder order) {
    return prepareCubDeviceUpperBound(range, values, output, range_num_items, values_num_items, order).temp_storage_bytes;
}

void cubDeviceUpperBound(const CubDeviceUpperBoundPlan& plan,
                         const Tensor& temp_storage,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         Stream& stream) {
    validateFindBounds(range, values, output, plan.range_num_items, plan.values_num_items, "upper-bound");
    if (range.getPlacement() != plan.placement || range.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB upper-bound plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.values_num_items == 0) {
        return;
    }

    const int range_items = checkedCubNumItems(plan.range_num_items);
    const int values_items = checkedCubNumItems(plan.values_num_items);
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void {
        dispatchFindComparator<T>(plan.order, [&](auto comp) -> void {
            launchUpperBound<T>(temp_storage_ptr, temp_storage_bytes, range, values, output, range_items, values_items, comp, stream);
        });
    };
    dispatchFindKeyDType(plan.dtype, launch);
}

void cubDeviceUpperBound(const Tensor& temp_storage,
                         size_t temp_storage_bytes,
                         const Tensor& range,
                         const Tensor& values,
                         Tensor& output,
                         uint64_t range_num_items,
                         uint64_t values_num_items,
                         Stream& stream,
                         CubSortOrder order) {
    CubDeviceUpperBoundPlan plan = prepareCubDeviceUpperBound(range, values, output, range_num_items, values_num_items, order);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB upper-bound requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceUpperBound(plan, temp_storage, range, values, output, stream);
}

CubDeviceFindIfFlaggedPlan prepareCubDeviceFindIfFlagged(const Tensor& flags, const Tensor& index_out, uint64_t num_items) {
    validateFindIfFlagged(flags, index_out, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename FlagT>() -> size_t { return queryFindIfFlaggedBytes<FlagT>(flags, index_out, cub_items); };
        bytes = dispatchFindFlagDType(flags.getDataType(), query);
    }

    CubDeviceFindIfFlaggedPlan plan;
    plan.placement = flags.getPlacement();
    plan.flag_dtype = flags.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceFindIfFlaggedTempBytes(const Tensor& flags, const Tensor& index_out, uint64_t num_items) {
    return prepareCubDeviceFindIfFlagged(flags, index_out, num_items).temp_storage_bytes;
}

void cubDeviceFindIfFlagged(const CubDeviceFindIfFlaggedPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& flags,
                            Tensor& index_out,
                            Stream& stream) {
    validateFindIfFlagged(flags, index_out, plan.num_items);
    if (flags.getPlacement() != plan.placement || flags.getDataType() != plan.flag_dtype) {
        throw std::invalid_argument("CUB find-if flagged plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0) {
        CUDA_CHECK(cudaMemsetAsync(index_out.getMemPtr<uint32_t>(), 0, sizeof(uint32_t), stream.getStream()));
        return;
    }

    const int cub_items = checkedCubNumItems(plan.num_items);
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename FlagT>() -> void {
        launchFindIfFlagged<FlagT>(temp_storage_ptr, temp_storage_bytes, flags, index_out, cub_items, stream);
    };
    dispatchFindFlagDType(plan.flag_dtype, launch);
}

void cubDeviceFindIfFlagged(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& flags,
                            Tensor& index_out,
                            uint64_t num_items,
                            Stream& stream) {
    CubDeviceFindIfFlaggedPlan plan = prepareCubDeviceFindIfFlagged(flags, index_out, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB find-if flagged requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceFindIfFlagged(plan, temp_storage, flags, index_out, stream);
}

}  // namespace ThorImplementation
