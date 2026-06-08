#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename T>
size_t queryReduceSumBytes(const Tensor& input, const Tensor& output, int cub_items) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,
                                      queried_bytes,
                                      input.getMemPtr<T>(),
                                      const_cast<T*>(output.getMemPtr<T>()),
                                      cub_items));
    return queried_bytes;
}

template <typename T>
size_t queryReduceMaxBytes(const Tensor& input, const Tensor& output, int cub_items) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(nullptr,
                                      queried_bytes,
                                      input.getMemPtr<T>(),
                                      const_cast<T*>(output.getMemPtr<T>()),
                                      cub_items));
    return queried_bytes;
}

template <typename T>
size_t queryReduceMinBytes(const Tensor& input, const Tensor& output, int cub_items) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Min(nullptr,
                                      queried_bytes,
                                      input.getMemPtr<T>(),
                                      const_cast<T*>(output.getMemPtr<T>()),
                                      cub_items));
    return queried_bytes;
}

template <typename T>
void launchReduceSum(void* temp_storage_ptr,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     int cub_items,
                     Stream& stream) {
    CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage_ptr,
                                      temp_storage_bytes,
                                      input.getMemPtr<T>(),
                                      output.getMemPtr<T>(),
                                      cub_items,
                                      stream.getStream()));
}

template <typename T>
void launchReduceMax(void* temp_storage_ptr,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     int cub_items,
                     Stream& stream) {
    CUDA_CHECK(cub::DeviceReduce::Max(temp_storage_ptr,
                                      temp_storage_bytes,
                                      input.getMemPtr<T>(),
                                      output.getMemPtr<T>(),
                                      cub_items,
                                      stream.getStream()));
}

template <typename T>
void launchReduceMin(void* temp_storage_ptr,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     int cub_items,
                     Stream& stream) {
    CUDA_CHECK(cub::DeviceReduce::Min(temp_storage_ptr,
                                      temp_storage_bytes,
                                      input.getMemPtr<T>(),
                                      output.getMemPtr<T>(),
                                      cub_items,
                                      stream.getStream()));
}

}  // namespace

CubDeviceReduceSumPlan prepareCubDeviceReduceSum(const Tensor& input, const Tensor& output, uint64_t num_items) {
    validateDeviceReduceSum(input, output, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    auto query = [&]<typename T>() -> size_t { return queryReduceSumBytes<T>(input, output, cub_items); };
    const size_t bytes = dispatchReduceDType(input.getDataType(), query);

    CubDeviceReduceSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceReduceSumTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items) {
    return prepareCubDeviceReduceSum(input, output, num_items).temp_storage_bytes;
}

void cubDeviceReduceSum(const CubDeviceReduceSumPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& input,
                        Tensor& output,
                        Stream& stream) {
    validateDeviceReduceSum(input, output, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB device reduce-sum plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void { launchReduceSum<T>(temp_storage_ptr, temp_storage_bytes, input, output, cub_items, stream); };
    dispatchReduceDType(plan.dtype, launch);
}

void cubDeviceReduceSum(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        uint64_t num_items,
                        Stream& stream) {
    CubDeviceReduceSumPlan plan = prepareCubDeviceReduceSum(input, output, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB device reduce-sum requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceReduceSum(plan, temp_storage, input, output, stream);
}

CubDeviceReduceMaxPlan prepareCubDeviceReduceMax(const Tensor& input, const Tensor& output, uint64_t num_items) {
    validateDeviceReduceMax(input, output, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    auto query = [&]<typename T>() -> size_t { return queryReduceMaxBytes<T>(input, output, cub_items); };
    const size_t bytes = dispatchReduceDType(input.getDataType(), query);

    CubDeviceReduceMaxPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceReduceMaxTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items) {
    return prepareCubDeviceReduceMax(input, output, num_items).temp_storage_bytes;
}

void cubDeviceReduceMax(const CubDeviceReduceMaxPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& input,
                        Tensor& output,
                        Stream& stream) {
    validateDeviceReduceMax(input, output, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB device reduce-max plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void { launchReduceMax<T>(temp_storage_ptr, temp_storage_bytes, input, output, cub_items, stream); };
    dispatchReduceDType(plan.dtype, launch);
}

void cubDeviceReduceMax(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        uint64_t num_items,
                        Stream& stream) {
    CubDeviceReduceMaxPlan plan = prepareCubDeviceReduceMax(input, output, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB device reduce-max requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceReduceMax(plan, temp_storage, input, output, stream);
}

CubDeviceReduceMinPlan prepareCubDeviceReduceMin(const Tensor& input, const Tensor& output, uint64_t num_items) {
    validateDeviceReduceMin(input, output, num_items);
    const int cub_items = checkedCubNumItems(num_items);

    auto query = [&]<typename T>() -> size_t { return queryReduceMinBytes<T>(input, output, cub_items); };
    const size_t bytes = dispatchReduceDType(input.getDataType(), query);

    CubDeviceReduceMinPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceReduceMinTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items) {
    return prepareCubDeviceReduceMin(input, output, num_items).temp_storage_bytes;
}

void cubDeviceReduceMin(const CubDeviceReduceMinPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& input,
                        Tensor& output,
                        Stream& stream) {
    validateDeviceReduceMin(input, output, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB device reduce-min plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename T>() -> void { launchReduceMin<T>(temp_storage_ptr, temp_storage_bytes, input, output, cub_items, stream); };
    dispatchReduceDType(plan.dtype, launch);
}

void cubDeviceReduceMin(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        uint64_t num_items,
                        Stream& stream) {
    CubDeviceReduceMinPlan plan = prepareCubDeviceReduceMin(input, output, num_items);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB device reduce-min requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceReduceMin(plan, temp_storage, input, output, stream);
}

}  // namespace ThorImplementation
