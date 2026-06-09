#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename T>
__host__ __device__ float scanToFloat(T value) {
    return static_cast<float>(value);
}

template <>
__host__ __device__ float scanToFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__host__ __device__ float scanToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__host__ __device__ T scanFromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ __half scanFromFloat<__half>(float value) {
    return __float2half(value);
}

template <>
__host__ __device__ __nv_bfloat16 scanFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
struct CubScanSumOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return scanFromFloat<T>(scanToFloat(a) + scanToFloat(b));
        } else {
            return a + b;
        }
    }
};

template <typename T>
struct CubScanProductOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return scanFromFloat<T>(scanToFloat(a) * scanToFloat(b));
        } else {
            return a * b;
        }
    }
};

template <typename T>
struct CubScanMinOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return scanToFloat(b) < scanToFloat(a) ? b : a;
        } else {
            return b < a ? b : a;
        }
    }
};

template <typename T>
struct CubScanMaxOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return scanToFloat(a) < scanToFloat(b) ? b : a;
        } else {
            return a < b ? b : a;
        }
    }
};

template <typename T>
T scanZero() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return scanFromFloat<T>(0.0f);
    } else {
        return T{0};
    }
}

template <typename T>
T scanOne() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return scanFromFloat<T>(1.0f);
    } else {
        return T{1};
    }
}

template <typename T>
T scanPositiveInfinityOrMax() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return scanFromFloat<T>(std::numeric_limits<float>::infinity());
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <typename T>
T scanNegativeInfinityOrLowest() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return scanFromFloat<T>(-std::numeric_limits<float>::infinity());
    } else if constexpr (std::is_floating_point_v<T>) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <typename T>
T scanIdentity(CubScanOp op) {
    switch (op) {
        case CubScanOp::Sum:
            return scanZero<T>();
        case CubScanOp::Product:
            return scanOne<T>();
        case CubScanOp::Min:
            return scanPositiveInfinityOrMax<T>();
        case CubScanOp::Max:
            return scanNegativeInfinityOrLowest<T>();
    }
    throw std::invalid_argument("Unsupported CUB scan op.");
}

template <typename T, typename ScanOpT>
size_t queryDeviceScan(const Tensor& input, const Tensor& output, uint64_t num_items, CubScanMode mode, ScanOpT scan_op, T init) {
    const int cub_items = checkedCubNumItems(num_items);
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveScan(nullptr,
                                                  queried_bytes,
                                                  input.getMemPtr<T>(),
                                                  const_cast<T*>(output.getMemPtr<T>()),
                                                  scan_op,
                                                  init,
                                                  cub_items));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr,
                                                  queried_bytes,
                                                  input.getMemPtr<T>(),
                                                  const_cast<T*>(output.getMemPtr<T>()),
                                                  scan_op,
                                                  cub_items));
    } else {
        throw std::invalid_argument("Unsupported CUB scan mode.");
    }
    return queried_bytes;
}

template <typename T, typename ScanOpT>
void launchDeviceScan(const CubDeviceScanPlan& plan,
                      const Tensor& temp_storage,
                      const Tensor& input,
                      Tensor& output,
                      Stream& stream,
                      ScanOpT scan_op,
                      T init) {
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;
    const int cub_items = checkedCubNumItems(plan.num_items);
    if (plan.mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveScan(temp_storage_ptr,
                                                  temp_storage_bytes,
                                                  input.getMemPtr<T>(),
                                                  output.getMemPtr<T>(),
                                                  scan_op,
                                                  init,
                                                  cub_items,
                                                  stream.getStream()));
    } else if (plan.mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(temp_storage_ptr,
                                                  temp_storage_bytes,
                                                  input.getMemPtr<T>(),
                                                  output.getMemPtr<T>(),
                                                  scan_op,
                                                  cub_items,
                                                  stream.getStream()));
    } else {
        throw std::invalid_argument("Unsupported CUB scan mode.");
    }
}

template <typename T, typename Fn>
decltype(auto) dispatchScanOperator(CubScanOp op, Fn&& fn) {
    switch (op) {
        case CubScanOp::Sum:
            return fn(CubScanSumOp<T>{}, scanIdentity<T>(op));
        case CubScanOp::Min:
            return fn(CubScanMinOp<T>{}, scanIdentity<T>(op));
        case CubScanOp::Max:
            return fn(CubScanMaxOp<T>{}, scanIdentity<T>(op));
        case CubScanOp::Product:
            return fn(CubScanProductOp<T>{}, scanIdentity<T>(op));
    }
    throw std::invalid_argument("Unsupported CUB scan op.");
}

}  // namespace

CubDeviceScanPlan prepareCubDeviceScan(const Tensor& input,
                                       const Tensor& output,
                                       uint64_t num_items,
                                       CubScanOp op,
                                       CubScanMode mode) {
    validateExclusiveSum(input, output, num_items);
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB scan dtype " + dtypeName(input.getDataType()) + ".");
    }

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            auto query_op = [&](auto scan_op, T init) -> size_t { return queryDeviceScan<T>(input, output, num_items, mode, scan_op, init); };
            return dispatchScanOperator<T>(op, query_op);
        };
        bytes = dispatchScanDType(input.getDataType(), query);
    }

    CubDeviceScanPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.op = op;
    plan.mode = mode;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceScanTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items, CubScanOp op, CubScanMode mode) {
    return prepareCubDeviceScan(input, output, num_items, op, mode).temp_storage_bytes;
}

void cubDeviceScan(const CubDeviceScanPlan& plan,
                   const Tensor& temp_storage,
                   const Tensor& input,
                   Tensor& output,
                   Stream& stream) {
    validateExclusiveSum(input, output, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB scan plan is not compatible with the provided tensors.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB scan dtype " + dtypeName(input.getDataType()) + ".");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0) {
        return;
    }

    auto launch = [&]<typename T>() -> void {
        auto launch_op = [&](auto scan_op, T init) -> void { launchDeviceScan<T>(plan, temp_storage, input, output, stream, scan_op, init); };
        dispatchScanOperator<T>(plan.op, launch_op);
    };

    dispatchScanDType(plan.dtype, launch);
}

void cubDeviceScan(const Tensor& temp_storage,
                   size_t temp_storage_bytes,
                   const Tensor& input,
                   Tensor& output,
                   uint64_t num_items,
                   Stream& stream,
                   CubScanOp op,
                   CubScanMode mode) {
    CubDeviceScanPlan plan = prepareCubDeviceScan(input, output, num_items, op, mode);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB scan requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceScan(plan, temp_storage, input, output, stream);
}

CubDeviceExclusiveSumPlan prepareCubDeviceExclusiveSum(const Tensor& input,
                                                       const Tensor& output,
                                                       uint64_t num_items) {
    CubDeviceScanPlan scan_plan = prepareCubDeviceScan(input, output, num_items, CubScanOp::Sum, CubScanMode::Exclusive);
    CubDeviceExclusiveSumPlan plan;
    plan.placement = scan_plan.placement;
    plan.dtype = scan_plan.dtype;
    plan.num_items = scan_plan.num_items;
    plan.temp_storage_bytes = scan_plan.temp_storage_bytes;
    return plan;
}

size_t cubDeviceExclusiveSumTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items) {
    return prepareCubDeviceExclusiveSum(input, output, num_items).temp_storage_bytes;
}

void cubDeviceExclusiveSum(const CubDeviceExclusiveSumPlan& plan,
                           const Tensor& temp_storage,
                           const Tensor& input,
                           Tensor& output,
                           Stream& stream) {
    CubDeviceScanPlan scan_plan;
    scan_plan.placement = plan.placement;
    scan_plan.dtype = plan.dtype;
    scan_plan.num_items = plan.num_items;
    scan_plan.op = CubScanOp::Sum;
    scan_plan.mode = CubScanMode::Exclusive;
    scan_plan.temp_storage_bytes = plan.temp_storage_bytes;
    cubDeviceScan(scan_plan, temp_storage, input, output, stream);
}

void cubDeviceExclusiveSum(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           uint64_t num_items,
                           Stream& stream) {
    cubDeviceScan(temp_storage, temp_storage_bytes, input, output, num_items, stream, CubScanOp::Sum, CubScanMode::Exclusive);
}

CubDeviceInclusiveSumPlan prepareCubDeviceInclusiveSum(const Tensor& input,
                                                       const Tensor& output,
                                                       uint64_t num_items) {
    CubDeviceScanPlan scan_plan = prepareCubDeviceScan(input, output, num_items, CubScanOp::Sum, CubScanMode::Inclusive);
    CubDeviceInclusiveSumPlan plan;
    plan.placement = scan_plan.placement;
    plan.dtype = scan_plan.dtype;
    plan.num_items = scan_plan.num_items;
    plan.temp_storage_bytes = scan_plan.temp_storage_bytes;
    return plan;
}

size_t cubDeviceInclusiveSumTempBytes(const Tensor& input, const Tensor& output, uint64_t num_items) {
    return prepareCubDeviceInclusiveSum(input, output, num_items).temp_storage_bytes;
}

void cubDeviceInclusiveSum(const CubDeviceInclusiveSumPlan& plan,
                           const Tensor& temp_storage,
                           const Tensor& input,
                           Tensor& output,
                           Stream& stream) {
    CubDeviceScanPlan scan_plan;
    scan_plan.placement = plan.placement;
    scan_plan.dtype = plan.dtype;
    scan_plan.num_items = plan.num_items;
    scan_plan.op = CubScanOp::Sum;
    scan_plan.mode = CubScanMode::Inclusive;
    scan_plan.temp_storage_bytes = plan.temp_storage_bytes;
    cubDeviceScan(scan_plan, temp_storage, input, output, stream);
}

void cubDeviceInclusiveSum(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           uint64_t num_items,
                           Stream& stream) {
    cubDeviceScan(temp_storage, temp_storage_bytes, input, output, num_items, stream, CubScanOp::Sum, CubScanMode::Inclusive);
}

}  // namespace ThorImplementation
