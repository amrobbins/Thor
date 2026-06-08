#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cuda/stream_ref>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

// CUB 3.3 fixed-size segmented top-k statically checks the maximum
// segment-size parameter against the selected worker tile size.  The
// conservative common bound for Thor's <=32-bit key surface is 8192.
constexpr int64_t kCubFixedSegmentedTopKMaxSegmentSize = 8192;
constexpr int64_t kCubFixedSegmentedTopKMaxK = 8192;

template <typename KeyT>
size_t queryTopKKeysBytes(const Tensor& keys_in,
                          const Tensor& keys_out,
                          int64_t cub_items,
                          int64_t cub_k,
                          CubTopKOrder order) {
    size_t queried_bytes = 0;
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto env = cuda::std::execution::env{requirements};
    if (order == CubTopKOrder::Largest) {
        CUDA_CHECK(cub::DeviceTopK::MaxKeys(nullptr,
                                           queried_bytes,
                                           keys_in.getMemPtr<KeyT>(),
                                           const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                           cub_items,
                                           cub_k,
                                           env));
    } else {
        CUDA_CHECK(cub::DeviceTopK::MinKeys(nullptr,
                                           queried_bytes,
                                           keys_in.getMemPtr<KeyT>(),
                                           const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                           cub_items,
                                           cub_k,
                                           env));
    }
    return queried_bytes;
}

template <typename KeyT, typename ValueT>
size_t queryTopKPairsBytes(const Tensor& keys_in,
                           const Tensor& keys_out,
                           const Tensor& values_in,
                           const Tensor& values_out,
                           int64_t cub_items,
                           int64_t cub_k,
                           CubTopKOrder order) {
    size_t queried_bytes = 0;
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto env = cuda::std::execution::env{requirements};
    if (order == CubTopKOrder::Largest) {
        CUDA_CHECK(cub::DeviceTopK::MaxPairs(nullptr,
                                            queried_bytes,
                                            keys_in.getMemPtr<KeyT>(),
                                            const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                            values_in.getMemPtr<ValueT>(),
                                            const_cast<ValueT*>(values_out.getMemPtr<ValueT>()),
                                            cub_items,
                                            cub_k,
                                            env));
    } else {
        CUDA_CHECK(cub::DeviceTopK::MinPairs(nullptr,
                                            queried_bytes,
                                            keys_in.getMemPtr<KeyT>(),
                                            const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                            values_in.getMemPtr<ValueT>(),
                                            const_cast<ValueT*>(values_out.getMemPtr<ValueT>()),
                                            cub_items,
                                            cub_k,
                                            env));
    }
    return queried_bytes;
}

template <typename KeyT>
void launchTopKKeys(void* temp_storage_ptr,
                    size_t temp_storage_bytes,
                    const Tensor& keys_in,
                    Tensor& keys_out,
                    int64_t cub_items,
                    int64_t cub_k,
                    CubTopKOrder order,
                    Stream& stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    cuda::stream_ref stream_ref{stream.getStream()};
    auto env = cuda::std::execution::env{stream_ref, requirements};
    if (order == CubTopKOrder::Largest) {
        CUDA_CHECK(cub::DeviceTopK::MaxKeys(temp_storage_ptr,
                                           temp_storage_bytes,
                                           keys_in.getMemPtr<KeyT>(),
                                           keys_out.getMemPtr<KeyT>(),
                                           cub_items,
                                           cub_k,
                                           env));
    } else {
        CUDA_CHECK(cub::DeviceTopK::MinKeys(temp_storage_ptr,
                                           temp_storage_bytes,
                                           keys_in.getMemPtr<KeyT>(),
                                           keys_out.getMemPtr<KeyT>(),
                                           cub_items,
                                           cub_k,
                                           env));
    }
}

template <typename KeyT, typename ValueT>
void launchTopKPairs(void* temp_storage_ptr,
                     size_t temp_storage_bytes,
                     const Tensor& keys_in,
                     Tensor& keys_out,
                     const Tensor& values_in,
                     Tensor& values_out,
                     int64_t cub_items,
                     int64_t cub_k,
                     CubTopKOrder order,
                     Stream& stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    cuda::stream_ref stream_ref{stream.getStream()};
    auto env = cuda::std::execution::env{stream_ref, requirements};
    if (order == CubTopKOrder::Largest) {
        CUDA_CHECK(cub::DeviceTopK::MaxPairs(temp_storage_ptr,
                                            temp_storage_bytes,
                                            keys_in.getMemPtr<KeyT>(),
                                            keys_out.getMemPtr<KeyT>(),
                                            values_in.getMemPtr<ValueT>(),
                                            values_out.getMemPtr<ValueT>(),
                                            cub_items,
                                            cub_k,
                                            env));
    } else {
        CUDA_CHECK(cub::DeviceTopK::MinPairs(temp_storage_ptr,
                                            temp_storage_bytes,
                                            keys_in.getMemPtr<KeyT>(),
                                            keys_out.getMemPtr<KeyT>(),
                                            values_in.getMemPtr<ValueT>(),
                                            values_out.getMemPtr<ValueT>(),
                                            cub_items,
                                            cub_k,
                                            env));
    }
}



template <typename T>
struct FixedSegmentPtrIterator {
    using value_type = T*;
    using difference_type = int64_t;
    using reference = T*;
    using pointer = T**;
    using iterator_category = std::random_access_iterator_tag;

    T* base = nullptr;
    int64_t stride = 0;

    __host__ __device__ T* operator[](difference_type index) const {
        return base + index * stride;
    }

    __host__ __device__ T* operator*() const {
        return base;
    }

    __host__ __device__ FixedSegmentPtrIterator operator+(difference_type offset) const {
        return FixedSegmentPtrIterator{base + offset * stride, stride};
    }

    __host__ __device__ FixedSegmentPtrIterator& operator+=(difference_type offset) {
        base += offset * stride;
        return *this;
    }
};

template <typename KeySegmentInputItItT,
          typename KeySegmentOutputItItT,
          typename ValueSegmentInputItItT,
          typename ValueSegmentOutputItItT>
cudaError_t dispatchFixedSizeSegmentedTopK(void* temp_storage_ptr,
                                           size_t& temp_storage_bytes,
                                           KeySegmentInputItItT key_segments_in,
                                           KeySegmentOutputItItT key_segments_out,
                                           ValueSegmentInputItItT value_segments_in,
                                           ValueSegmentOutputItItT value_segments_out,
                                           int64_t cub_segments,
                                           int64_t cub_segment_size,
                                           int64_t cub_k,
                                           int64_t cub_total_items,
                                           CubTopKOrder order,
                                           cudaStream_t stream = nullptr) {
    using namespace cub::detail::batched_topk;

    using SegmentSizeParamT = segment_size_uniform<1, kCubFixedSegmentedTopKMaxSegmentSize>;
    using KParamT = k_uniform<1, kCubFixedSegmentedTopKMaxK>;
    using SelectDirectionParamT = select_direction_uniform;
    using NumSegmentsParamT = num_segments_uniform<>;
    using TotalItemsParamT = total_num_items_guarantee<>;
    using Dispatcher = dispatch_batched_topk<KeySegmentInputItItT,
                                             KeySegmentOutputItItT,
                                             ValueSegmentInputItItT,
                                             ValueSegmentOutputItItT,
                                             SegmentSizeParamT,
                                             KParamT,
                                             SelectDirectionParamT,
                                             NumSegmentsParamT,
                                             TotalItemsParamT>;

    const auto select_direction = order == CubTopKOrder::Largest ? cub::detail::topk::select::max : cub::detail::topk::select::min;
    return Dispatcher::dispatch(temp_storage_ptr,
                                temp_storage_bytes,
                                key_segments_in,
                                key_segments_out,
                                value_segments_in,
                                value_segments_out,
                                SegmentSizeParamT{cub_segment_size},
                                KParamT{cub_k},
                                SelectDirectionParamT{select_direction},
                                NumSegmentsParamT{cub_segments},
                                TotalItemsParamT{cub_total_items},
                                stream);
}

template <typename KeyT>
size_t querySegmentedTopKKeysBytes(const Tensor& keys_in,
                                   const Tensor& keys_out,
                                   int64_t cub_segments,
                                   int64_t cub_segment_size,
                                   int64_t cub_k,
                                   int64_t cub_total_items,
                                   CubTopKOrder order) {
    size_t queried_bytes = 0;
    CUDA_CHECK(dispatchFixedSizeSegmentedTopK(nullptr,
                                             queried_bytes,
                                             FixedSegmentPtrIterator<const KeyT>{keys_in.getMemPtr<KeyT>(), cub_segment_size},
                                             FixedSegmentPtrIterator<KeyT>{const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()), cub_k},
                                             static_cast<cub::NullType**>(nullptr),
                                             static_cast<cub::NullType**>(nullptr),
                                             cub_segments,
                                             cub_segment_size,
                                             cub_k,
                                             cub_total_items,
                                             order));
    return queried_bytes;
}

template <typename KeyT, typename ValueT>
size_t querySegmentedTopKPairsBytes(const Tensor& keys_in,
                                    const Tensor& keys_out,
                                    const Tensor& values_in,
                                    const Tensor& values_out,
                                    int64_t cub_segments,
                                    int64_t cub_segment_size,
                                    int64_t cub_k,
                                    int64_t cub_total_items,
                                    CubTopKOrder order) {
    size_t queried_bytes = 0;
    CUDA_CHECK(dispatchFixedSizeSegmentedTopK(nullptr,
                                             queried_bytes,
                                             FixedSegmentPtrIterator<const KeyT>{keys_in.getMemPtr<KeyT>(), cub_segment_size},
                                             FixedSegmentPtrIterator<KeyT>{const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()), cub_k},
                                             FixedSegmentPtrIterator<const ValueT>{values_in.getMemPtr<ValueT>(), cub_segment_size},
                                             FixedSegmentPtrIterator<ValueT>{const_cast<ValueT*>(values_out.getMemPtr<ValueT>()), cub_k},
                                             cub_segments,
                                             cub_segment_size,
                                             cub_k,
                                             cub_total_items,
                                             order));
    return queried_bytes;
}

template <typename KeyT>
void launchSegmentedTopKKeys(void* temp_storage_ptr,
                             size_t temp_storage_bytes,
                             const Tensor& keys_in,
                             Tensor& keys_out,
                             int64_t cub_segments,
                             int64_t cub_segment_size,
                             int64_t cub_k,
                             int64_t cub_total_items,
                             CubTopKOrder order,
                             Stream& stream) {
    CUDA_CHECK(dispatchFixedSizeSegmentedTopK(temp_storage_ptr,
                                             temp_storage_bytes,
                                             FixedSegmentPtrIterator<const KeyT>{keys_in.getMemPtr<KeyT>(), cub_segment_size},
                                             FixedSegmentPtrIterator<KeyT>{keys_out.getMemPtr<KeyT>(), cub_k},
                                             static_cast<cub::NullType**>(nullptr),
                                             static_cast<cub::NullType**>(nullptr),
                                             cub_segments,
                                             cub_segment_size,
                                             cub_k,
                                             cub_total_items,
                                             order,
                                             stream.getStream()));
}

template <typename KeyT, typename ValueT>
void launchSegmentedTopKPairs(void* temp_storage_ptr,
                              size_t temp_storage_bytes,
                              const Tensor& keys_in,
                              Tensor& keys_out,
                              const Tensor& values_in,
                              Tensor& values_out,
                              int64_t cub_segments,
                              int64_t cub_segment_size,
                              int64_t cub_k,
                              int64_t cub_total_items,
                              CubTopKOrder order,
                              Stream& stream) {
    CUDA_CHECK(dispatchFixedSizeSegmentedTopK(temp_storage_ptr,
                                             temp_storage_bytes,
                                             FixedSegmentPtrIterator<const KeyT>{keys_in.getMemPtr<KeyT>(), cub_segment_size},
                                             FixedSegmentPtrIterator<KeyT>{keys_out.getMemPtr<KeyT>(), cub_k},
                                             FixedSegmentPtrIterator<const ValueT>{values_in.getMemPtr<ValueT>(), cub_segment_size},
                                             FixedSegmentPtrIterator<ValueT>{values_out.getMemPtr<ValueT>(), cub_k},
                                             cub_segments,
                                             cub_segment_size,
                                             cub_k,
                                             cub_total_items,
                                             order,
                                             stream.getStream()));
}

}  // namespace

CubDeviceTopKKeysPlan prepareCubDeviceTopKKeys(const Tensor& keys_in,
                                               const Tensor& keys_out,
                                               uint64_t num_items,
                                               uint64_t k,
                                               CubTopKOrder order) {
    validateTopKKeys(keys_in, keys_out, num_items, k);
    const int64_t cub_items = checkedCubInt64Count(num_items, "num_items");
    const int64_t cub_k = checkedCubInt64Count(k, "k");

    size_t bytes = 1;
    if (num_items != 0 && k != 0) {
        auto query_key = [&]<typename KeyT>() -> size_t {
            return queryTopKKeysBytes<KeyT>(keys_in, keys_out, cub_items, cub_k, order);
        };
        bytes = dispatchTopKKeyDType(keys_in.getDataType(), query_key);
    }

    CubDeviceTopKKeysPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.num_items = num_items;
    plan.k = k;
    plan.order = order;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceTopKKeysTempBytes(const Tensor& keys_in,
                                  const Tensor& keys_out,
                                  uint64_t num_items,
                                  uint64_t k,
                                  CubTopKOrder order) {
    return prepareCubDeviceTopKKeys(keys_in, keys_out, num_items, k, order).temp_storage_bytes;
}

void cubDeviceTopKKeys(const CubDeviceTopKKeysPlan& plan,
                       const Tensor& temp_storage,
                       const Tensor& keys_in,
                       Tensor& keys_out,
                       Stream& stream) {
    validateTopKKeys(keys_in, keys_out, plan.num_items, plan.k);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype) {
        throw std::invalid_argument("CUB top-k-keys plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int64_t cub_items = checkedCubInt64Count(plan.num_items, "num_items");
    const int64_t cub_k = checkedCubInt64Count(plan.k, "k");
    if (plan.num_items == 0 || plan.k == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_key = [&]<typename KeyT>() -> void {
        launchTopKKeys<KeyT>(temp_storage_ptr, temp_storage_bytes, keys_in, keys_out, cub_items, cub_k, plan.order, stream);
    };

    dispatchTopKKeyDType(plan.key_dtype, launch_key);
}

void cubDeviceTopKKeys(const Tensor& temp_storage,
                       size_t temp_storage_bytes,
                       const Tensor& keys_in,
                       Tensor& keys_out,
                       uint64_t num_items,
                       uint64_t k,
                       Stream& stream,
                       CubTopKOrder order) {
    CubDeviceTopKKeysPlan plan = prepareCubDeviceTopKKeys(keys_in, keys_out, num_items, k, order);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB top-k-keys requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceTopKKeys(plan, temp_storage, keys_in, keys_out, stream);
}

CubDeviceTopKPairsPlan prepareCubDeviceTopKPairs(const Tensor& keys_in,
                                                 const Tensor& keys_out,
                                                 const Tensor& values_in,
                                                 const Tensor& values_out,
                                                 uint64_t num_items,
                                                 uint64_t k,
                                                 CubTopKOrder order) {
    validateTopKPairs(keys_in, keys_out, values_in, values_out, num_items, k);
    const int64_t cub_items = checkedCubInt64Count(num_items, "num_items");
    const int64_t cub_k = checkedCubInt64Count(k, "k");

    size_t bytes = 1;
    if (num_items != 0 && k != 0) {
        auto query_key = [&]<typename KeyT>() -> size_t {
            auto query_value = [&]<typename ValueT>() -> size_t {
                return queryTopKPairsBytes<KeyT, ValueT>(keys_in, keys_out, values_in, values_out, cub_items, cub_k, order);
            };
            return dispatchTopKValueDType(values_in.getDataType(), query_value);
        };
        bytes = dispatchTopKKeyDType(keys_in.getDataType(), query_key);
    }

    CubDeviceTopKPairsPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.value_dtype = values_in.getDataType();
    plan.num_items = num_items;
    plan.k = k;
    plan.order = order;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceTopKPairsTempBytes(const Tensor& keys_in,
                                   const Tensor& keys_out,
                                   const Tensor& values_in,
                                   const Tensor& values_out,
                                   uint64_t num_items,
                                   uint64_t k,
                                   CubTopKOrder order) {
    return prepareCubDeviceTopKPairs(keys_in, keys_out, values_in, values_out, num_items, k, order).temp_storage_bytes;
}

void cubDeviceTopKPairs(const CubDeviceTopKPairsPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& keys_in,
                        Tensor& keys_out,
                        const Tensor& values_in,
                        Tensor& values_out,
                        Stream& stream) {
    validateTopKPairs(keys_in, keys_out, values_in, values_out, plan.num_items, plan.k);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype ||
        values_in.getDataType() != plan.value_dtype) {
        throw std::invalid_argument("CUB top-k-pairs plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int64_t cub_items = checkedCubInt64Count(plan.num_items, "num_items");
    const int64_t cub_k = checkedCubInt64Count(plan.k, "k");
    if (plan.num_items == 0 || plan.k == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_key = [&]<typename KeyT>() -> void {
        auto launch_value = [&]<typename ValueT>() -> void {
            launchTopKPairs<KeyT, ValueT>(
                temp_storage_ptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out, cub_items, cub_k, plan.order, stream);
        };
        dispatchTopKValueDType(plan.value_dtype, launch_value);
    };

    dispatchTopKKeyDType(plan.key_dtype, launch_key);
}

void cubDeviceTopKPairs(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& keys_in,
                        Tensor& keys_out,
                        const Tensor& values_in,
                        Tensor& values_out,
                        uint64_t num_items,
                        uint64_t k,
                        Stream& stream,
                        CubTopKOrder order) {
    CubDeviceTopKPairsPlan plan = prepareCubDeviceTopKPairs(keys_in, keys_out, values_in, values_out, num_items, k, order);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB top-k-pairs requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceTopKPairs(plan, temp_storage, keys_in, keys_out, values_in, values_out, stream);
}


CubDeviceSegmentedTopKKeysPlan prepareCubDeviceSegmentedTopKKeys(const Tensor& keys_in,
                                                                 const Tensor& keys_out,
                                                                 uint64_t num_segments,
                                                                 uint64_t segment_size,
                                                                 uint64_t k,
                                                                 CubTopKOrder order) {
    validateSegmentedTopKKeys(keys_in, keys_out, num_segments, segment_size, k);
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
    const int64_t cub_k = checkedCubInt64Count(k, "k");
    const uint64_t total_items = checkedSegmentedTopKTotalItems(num_segments, segment_size);
    const int64_t cub_total_items = checkedCubInt64Count(total_items, "num_segments * segment_size");

    size_t bytes = 1;
    if (num_segments != 0 && segment_size != 0 && k != 0) {
        auto query_key = [&]<typename KeyT>() -> size_t {
            return querySegmentedTopKKeysBytes<KeyT>(
                keys_in, keys_out, cub_segments, cub_segment_size, cub_k, cub_total_items, order);
        };
        bytes = dispatchTopKKeyDType(keys_in.getDataType(), query_key);
    }

    CubDeviceSegmentedTopKKeysPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.num_segments = num_segments;
    plan.segment_size = segment_size;
    plan.k = k;
    plan.order = order;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedTopKKeysTempBytes(const Tensor& keys_in,
                                           const Tensor& keys_out,
                                           uint64_t num_segments,
                                           uint64_t segment_size,
                                           uint64_t k,
                                           CubTopKOrder order) {
    return prepareCubDeviceSegmentedTopKKeys(keys_in, keys_out, num_segments, segment_size, k, order).temp_storage_bytes;
}

void cubDeviceSegmentedTopKKeys(const CubDeviceSegmentedTopKKeysPlan& plan,
                                const Tensor& temp_storage,
                                const Tensor& keys_in,
                                Tensor& keys_out,
                                Stream& stream) {
    validateSegmentedTopKKeys(keys_in, keys_out, plan.num_segments, plan.segment_size, plan.k);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype) {
        throw std::invalid_argument("CUB segmented top-k-keys plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(plan.segment_size, "segment_size");
    const int64_t cub_k = checkedCubInt64Count(plan.k, "k");
    const uint64_t total_items = checkedSegmentedTopKTotalItems(plan.num_segments, plan.segment_size);
    const int64_t cub_total_items = checkedCubInt64Count(total_items, "num_segments * segment_size");
    if (plan.num_segments == 0 || plan.segment_size == 0 || plan.k == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_key = [&]<typename KeyT>() -> void {
        launchSegmentedTopKKeys<KeyT>(temp_storage_ptr,
                                      temp_storage_bytes,
                                      keys_in,
                                      keys_out,
                                      cub_segments,
                                      cub_segment_size,
                                      cub_k,
                                      cub_total_items,
                                      plan.order,
                                      stream);
    };

    dispatchTopKKeyDType(plan.key_dtype, launch_key);
}

void cubDeviceSegmentedTopKKeys(const Tensor& temp_storage,
                                size_t temp_storage_bytes,
                                const Tensor& keys_in,
                                Tensor& keys_out,
                                uint64_t num_segments,
                                uint64_t segment_size,
                                uint64_t k,
                                Stream& stream,
                                CubTopKOrder order) {
    CubDeviceSegmentedTopKKeysPlan plan = prepareCubDeviceSegmentedTopKKeys(keys_in, keys_out, num_segments, segment_size, k, order);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented top-k-keys requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedTopKKeys(plan, temp_storage, keys_in, keys_out, stream);
}

CubDeviceSegmentedTopKPairsPlan prepareCubDeviceSegmentedTopKPairs(const Tensor& keys_in,
                                                                   const Tensor& keys_out,
                                                                   const Tensor& values_in,
                                                                   const Tensor& values_out,
                                                                   uint64_t num_segments,
                                                                   uint64_t segment_size,
                                                                   uint64_t k,
                                                                   CubTopKOrder order) {
    validateSegmentedTopKPairs(keys_in, keys_out, values_in, values_out, num_segments, segment_size, k);
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
    const int64_t cub_k = checkedCubInt64Count(k, "k");
    const uint64_t total_items = checkedSegmentedTopKTotalItems(num_segments, segment_size);
    const int64_t cub_total_items = checkedCubInt64Count(total_items, "num_segments * segment_size");

    size_t bytes = 1;
    if (num_segments != 0 && segment_size != 0 && k != 0) {
        auto query_key = [&]<typename KeyT>() -> size_t {
            auto query_value = [&]<typename ValueT>() -> size_t {
                return querySegmentedTopKPairsBytes<KeyT, ValueT>(
                    keys_in, keys_out, values_in, values_out, cub_segments, cub_segment_size, cub_k, cub_total_items, order);
            };
            return dispatchTopKValueDType(values_in.getDataType(), query_value);
        };
        bytes = dispatchTopKKeyDType(keys_in.getDataType(), query_key);
    }

    CubDeviceSegmentedTopKPairsPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.value_dtype = values_in.getDataType();
    plan.num_segments = num_segments;
    plan.segment_size = segment_size;
    plan.k = k;
    plan.order = order;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedTopKPairsTempBytes(const Tensor& keys_in,
                                            const Tensor& keys_out,
                                            const Tensor& values_in,
                                            const Tensor& values_out,
                                            uint64_t num_segments,
                                            uint64_t segment_size,
                                            uint64_t k,
                                            CubTopKOrder order) {
    return prepareCubDeviceSegmentedTopKPairs(keys_in, keys_out, values_in, values_out, num_segments, segment_size, k, order)
        .temp_storage_bytes;
}

void cubDeviceSegmentedTopKPairs(const CubDeviceSegmentedTopKPairsPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& keys_in,
                                 Tensor& keys_out,
                                 const Tensor& values_in,
                                 Tensor& values_out,
                                 Stream& stream) {
    validateSegmentedTopKPairs(keys_in, keys_out, values_in, values_out, plan.num_segments, plan.segment_size, plan.k);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype ||
        values_in.getDataType() != plan.value_dtype) {
        throw std::invalid_argument("CUB segmented top-k-pairs plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(plan.segment_size, "segment_size");
    const int64_t cub_k = checkedCubInt64Count(plan.k, "k");
    const uint64_t total_items = checkedSegmentedTopKTotalItems(plan.num_segments, plan.segment_size);
    const int64_t cub_total_items = checkedCubInt64Count(total_items, "num_segments * segment_size");
    if (plan.num_segments == 0 || plan.segment_size == 0 || plan.k == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_key = [&]<typename KeyT>() -> void {
        auto launch_value = [&]<typename ValueT>() -> void {
            launchSegmentedTopKPairs<KeyT, ValueT>(temp_storage_ptr,
                                                   temp_storage_bytes,
                                                   keys_in,
                                                   keys_out,
                                                   values_in,
                                                   values_out,
                                                   cub_segments,
                                                   cub_segment_size,
                                                   cub_k,
                                                   cub_total_items,
                                                   plan.order,
                                                   stream);
        };
        dispatchTopKValueDType(plan.value_dtype, launch_value);
    };

    dispatchTopKKeyDType(plan.key_dtype, launch_key);
}

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
                                 CubTopKOrder order) {
    CubDeviceSegmentedTopKPairsPlan plan =
        prepareCubDeviceSegmentedTopKPairs(keys_in, keys_out, values_in, values_out, num_segments, segment_size, k, order);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented top-k-pairs requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedTopKPairs(plan, temp_storage, keys_in, keys_out, values_in, values_out, stream);
}

}  // namespace ThorImplementation
