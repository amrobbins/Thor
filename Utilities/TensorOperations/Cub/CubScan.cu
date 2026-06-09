#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>
#include <math_constants.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

__host__ __device__ float scanPositiveInfinityFloat() {
#if defined(__CUDA_ARCH__)
    return __int_as_float(0x7f800000);
#else
    return std::numeric_limits<float>::infinity();
#endif
}

__host__ __device__ float scanNegativeInfinityFloat() {
#if defined(__CUDA_ARCH__)
    return __int_as_float(0xff800000);
#else
    return -std::numeric_limits<float>::infinity();
#endif
}

#if THOR_CUB_ENABLE_64BIT_TYPES
__host__ __device__ double scanPositiveInfinityDouble() {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(0x7ff0000000000000ULL);
#else
    return std::numeric_limits<double>::infinity();
#endif
}

__host__ __device__ double scanNegativeInfinityDouble() {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(0xfff0000000000000ULL);
#else
    return -std::numeric_limits<double>::infinity();
#endif
}
#endif

template <typename T>
__host__ __device__ T scanPositiveInfinityOrMax() {
    // These helpers are called from device code, and CUDA 13.3 rejects
    // std::numeric_limits::* from the device side of __host__ __device__
    // functions unless the build enables --expt-relaxed-constexpr. Use CUDA
    // intrinsics only in device compilation and keep the host path normal.
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return scanFromFloat<T>(scanPositiveInfinityFloat());
    } else if constexpr (std::is_same_v<T, float>) {
        return scanPositiveInfinityFloat();
#if THOR_CUB_ENABLE_64BIT_TYPES
    } else if constexpr (std::is_same_v<T, double>) {
        return scanPositiveInfinityDouble();
#endif
    } else if constexpr (std::is_unsigned_v<T>) {
        return ~T{0};
    } else {
        static_assert(std::is_unsigned_v<T>, "Unsupported scan identity dtype.");
    }
}

template <typename T>
__host__ __device__ T scanNegativeInfinityOrLowest() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return scanFromFloat<T>(scanNegativeInfinityFloat());
    } else if constexpr (std::is_same_v<T, float>) {
        return scanNegativeInfinityFloat();
#if THOR_CUB_ENABLE_64BIT_TYPES
    } else if constexpr (std::is_same_v<T, double>) {
        return scanNegativeInfinityDouble();
#endif
    } else if constexpr (std::is_unsigned_v<T>) {
        return T{0};
    } else {
        static_assert(std::is_unsigned_v<T>, "Unsupported scan identity dtype.");
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

template <typename T, typename ScanOpT, typename InputIt, typename OutputIt>
size_t queryDeviceScanIterator(InputIt input_begin, OutputIt output_begin, int cub_items, CubScanMode mode, ScanOpT scan_op, T init) {
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveScan(nullptr,
                                                  queried_bytes,
                                                  input_begin,
                                                  output_begin,
                                                  scan_op,
                                                  init,
                                                  cub_items));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, queried_bytes, input_begin, output_begin, scan_op, cub_items));
    } else {
        throw std::invalid_argument("Unsupported CUB scan mode.");
    }
    return queried_bytes;
}

template <typename T, typename ScanOpT>
size_t queryDeviceScan(const Tensor& input,
                       const Tensor& output,
                       uint64_t num_items,
                       CubScanMode mode,
                       CubScanDirection direction,
                       ScanOpT scan_op,
                       T init) {
    const int cub_items = checkedCubNumItems(num_items);
    if (direction == CubScanDirection::Forward) {
        return queryDeviceScanIterator<T>(
            input.getMemPtr<T>(), const_cast<T*>(output.getMemPtr<T>()), cub_items, mode, scan_op, init);
    }
    if (direction == CubScanDirection::Reverse) {
        auto input_begin = thrust::make_reverse_iterator(input.getMemPtr<T>() + cub_items);
        auto output_begin = thrust::make_reverse_iterator(const_cast<T*>(output.getMemPtr<T>()) + cub_items);
        return queryDeviceScanIterator<T>(input_begin, output_begin, cub_items, mode, scan_op, init);
    }
    throw std::invalid_argument("Unsupported CUB scan direction.");
}

template <typename T, typename ScanOpT, typename InputIt, typename OutputIt>
void launchDeviceScanIterator(void* temp_storage_ptr,
                              size_t temp_storage_bytes,
                              InputIt input_begin,
                              OutputIt output_begin,
                              int cub_items,
                              cudaStream_t stream,
                              CubScanMode mode,
                              ScanOpT scan_op,
                              T init) {
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveScan(
            temp_storage_ptr, temp_storage_bytes, input_begin, output_begin, scan_op, init, cub_items, stream));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(
            temp_storage_ptr, temp_storage_bytes, input_begin, output_begin, scan_op, cub_items, stream));
    } else {
        throw std::invalid_argument("Unsupported CUB scan mode.");
    }
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
    if (plan.direction == CubScanDirection::Forward) {
        launchDeviceScanIterator<T>(temp_storage_ptr,
                                    temp_storage_bytes,
                                    input.getMemPtr<T>(),
                                    output.getMemPtr<T>(),
                                    cub_items,
                                    stream.getStream(),
                                    plan.mode,
                                    scan_op,
                                    init);
        return;
    }
    if (plan.direction == CubScanDirection::Reverse) {
        auto input_begin = thrust::make_reverse_iterator(input.getMemPtr<T>() + cub_items);
        auto output_begin = thrust::make_reverse_iterator(output.getMemPtr<T>() + cub_items);
        launchDeviceScanIterator<T>(
            temp_storage_ptr, temp_storage_bytes, input_begin, output_begin, cub_items, stream.getStream(), plan.mode, scan_op, init);
        return;
    }
    throw std::invalid_argument("Unsupported CUB scan direction.");
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
                                       CubScanMode mode,
                                       CubScanDirection direction) {
    validateExclusiveSum(input, output, num_items);
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB scan dtype " + dtypeName(input.getDataType()) + ".");
    }

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            auto query_op = [&](auto scan_op, T init) -> size_t {
                return queryDeviceScan<T>(input, output, num_items, mode, direction, scan_op, init);
            };
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
    plan.direction = direction;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceScanTempBytes(
    const Tensor& input, const Tensor& output, uint64_t num_items, CubScanOp op, CubScanMode mode, CubScanDirection direction) {
    return prepareCubDeviceScan(input, output, num_items, op, mode, direction).temp_storage_bytes;
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
                   CubScanMode mode,
                   CubScanDirection direction) {
    CubDeviceScanPlan plan = prepareCubDeviceScan(input, output, num_items, op, mode, direction);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB scan requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceScan(plan, temp_storage, input, output, stream);
}

CubDeviceExclusiveSumPlan prepareCubDeviceExclusiveSum(const Tensor& input,
                                                       const Tensor& output,
                                                       uint64_t num_items) {
    CubDeviceScanPlan scan_plan =
        prepareCubDeviceScan(input, output, num_items, CubScanOp::Sum, CubScanMode::Exclusive, CubScanDirection::Forward);
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
    cubDeviceScan(
        temp_storage, temp_storage_bytes, input, output, num_items, stream, CubScanOp::Sum, CubScanMode::Exclusive, CubScanDirection::Forward);
}

CubDeviceInclusiveSumPlan prepareCubDeviceInclusiveSum(const Tensor& input,
                                                       const Tensor& output,
                                                       uint64_t num_items) {
    CubDeviceScanPlan scan_plan =
        prepareCubDeviceScan(input, output, num_items, CubScanOp::Sum, CubScanMode::Inclusive, CubScanDirection::Forward);
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
    cubDeviceScan(
        temp_storage, temp_storage_bytes, input, output, num_items, stream, CubScanOp::Sum, CubScanMode::Inclusive, CubScanDirection::Forward);
}

namespace {

template <typename T>
struct CubArgScanPair {
    T value;
    uint32_t index;
};

template <typename T>
struct CubArgScanInputOp {
    const T* input;
    uint32_t num_items;
    CubScanDirection direction;

    __host__ __device__ CubArgScanPair<T> operator()(uint32_t logical_index) const {
        const uint32_t physical = direction == CubScanDirection::Reverse ? (num_items - 1U - logical_index) : logical_index;
        return CubArgScanPair<T>{input[physical], physical};
    }
};

template <typename T>
struct CubArgScanMinOp {
    __host__ __device__ CubArgScanPair<T> operator()(const CubArgScanPair<T>& a, const CubArgScanPair<T>& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return scanToFloat(b.value) < scanToFloat(a.value) ? b : a;
        } else {
            return b.value < a.value ? b : a;
        }
    }
};

template <typename T>
struct CubArgScanMaxOp {
    __host__ __device__ CubArgScanPair<T> operator()(const CubArgScanPair<T>& a, const CubArgScanPair<T>& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return scanToFloat(a.value) < scanToFloat(b.value) ? b : a;
        } else {
            return a.value < b.value ? b : a;
        }
    }
};

template <typename T>
CubArgScanPair<T> argScanIdentity(CubArgScanOp op) {
    switch (op) {
        case CubArgScanOp::ArgMin:
            return CubArgScanPair<T>{scanPositiveInfinityOrMax<T>(), UINT32_MAX};
        case CubArgScanOp::ArgMax:
            return CubArgScanPair<T>{scanNegativeInfinityOrLowest<T>(), UINT32_MAX};
    }
    throw std::invalid_argument("Unsupported CUB arg scan op.");
}

template <typename T, typename Fn>
decltype(auto) dispatchArgScanOperator(CubArgScanOp op, Fn&& fn) {
    switch (op) {
        case CubArgScanOp::ArgMin:
            return fn(CubArgScanMinOp<T>{}, argScanIdentity<T>(op));
        case CubArgScanOp::ArgMax:
            return fn(CubArgScanMaxOp<T>{}, argScanIdentity<T>(op));
    }
    throw std::invalid_argument("Unsupported CUB arg scan op.");
}

inline size_t alignCubArgScanBytes(size_t bytes) { return (bytes + size_t{255}) & ~size_t{255}; }

template <typename T>
size_t argScanPairStorageBytes(uint64_t num_items) {
    if (num_items > std::numeric_limits<size_t>::max() / sizeof(CubArgScanPair<T>)) {
        throw std::invalid_argument("CUB arg scan pair workspace size overflow.");
    }
    return alignCubArgScanBytes(static_cast<size_t>(num_items) * sizeof(CubArgScanPair<T>));
}

template <typename T, typename ScanOpT>
size_t queryDeviceArgScan(const Tensor& input, uint64_t num_items, CubScanMode mode, CubScanDirection direction, ScanOpT scan_op, CubArgScanPair<T> init) {
    const int cub_items = checkedCubNumItems(num_items);
    auto input_begin = thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(0),
                                                       CubArgScanInputOp<T>{input.getMemPtr<T>(), static_cast<uint32_t>(cub_items), direction});
    CubArgScanPair<T>* output_begin = nullptr;
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveScan(nullptr, queried_bytes, input_begin, output_begin, scan_op, init, cub_items));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, queried_bytes, input_begin, output_begin, scan_op, cub_items));
    } else {
        throw std::invalid_argument("Unsupported CUB arg scan mode.");
    }
    return queried_bytes;
}

template <typename T>
__global__ void extractArgScanIndicesKernel(const CubArgScanPair<T>* pairs, uint32_t* output, uint32_t num_items, CubScanDirection direction) {
    const uint32_t logical = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical >= num_items) {
        return;
    }
    const uint32_t physical = direction == CubScanDirection::Reverse ? (num_items - 1U - logical) : logical;
    output[physical] = pairs[logical].index;
}

template <typename T, typename ScanOpT>
void launchDeviceArgScan(const CubDeviceArgScanPlan& plan,
                         const Tensor& temp_storage,
                         const Tensor& input,
                         Tensor& output,
                         Stream& stream,
                         ScanOpT scan_op,
                         CubArgScanPair<T> init) {
    const int cub_items = checkedCubNumItems(plan.num_items);
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    const size_t pair_bytes = argScanPairStorageBytes<T>(plan.num_items);
    auto* pair_output = reinterpret_cast<CubArgScanPair<T>*>(temp_storage_ptr);
    void* cub_temp = static_cast<void*>(static_cast<unsigned char*>(temp_storage_ptr) + pair_bytes);
    size_t cub_temp_bytes = plan.temp_storage_bytes - pair_bytes;
    auto input_begin = thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(0),
                                                       CubArgScanInputOp<T>{input.getMemPtr<T>(), static_cast<uint32_t>(cub_items), plan.direction});
    if (plan.mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveScan(cub_temp, cub_temp_bytes, input_begin, pair_output, scan_op, init, cub_items, stream.getStream()));
    } else if (plan.mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(cub_temp, cub_temp_bytes, input_begin, pair_output, scan_op, cub_items, stream.getStream()));
    } else {
        throw std::invalid_argument("Unsupported CUB arg scan mode.");
    }
    const uint32_t threads = 256;
    const uint32_t blocks = (static_cast<uint32_t>(cub_items) + threads - 1U) / threads;
    extractArgScanIndicesKernel<T><<<blocks, threads, 0, stream.getStream()>>>(pair_output, output.getMemPtr<uint32_t>(), static_cast<uint32_t>(cub_items), plan.direction);
    CUDA_CHECK(cudaGetLastError());
}

void validateArgScan(const Tensor& input, const Tensor& output, uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "arg scan input");
    requireDenseContiguousGpuTensor(output, "arg scan output");
    requireSameGpuPlacement(input, output, "arg scan input", "arg scan output");
    requireStorageForNumItems(input, "arg scan input", num_items);
    requireStorageForNumItems(output, "arg scan output", num_items);
    if (output.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB arg scan output dtype must be UINT32.");
    }
}

}  // namespace

CubDeviceArgScanPlan prepareCubDeviceArgScan(const Tensor& input,
                                             const Tensor& output,
                                             uint64_t num_items,
                                             CubArgScanOp op,
                                             CubScanMode mode,
                                             CubScanDirection direction) {
    validateArgScan(input, output, num_items);
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB arg scan dtype " + dtypeName(input.getDataType()) + ".");
    }

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename T>() -> size_t {
            auto query_op = [&](auto scan_op, CubArgScanPair<T> init) -> size_t {
                return argScanPairStorageBytes<T>(num_items) + queryDeviceArgScan<T>(input, num_items, mode, direction, scan_op, init);
            };
            return dispatchArgScanOperator<T>(op, query_op);
        };
        bytes = dispatchScanDType(input.getDataType(), query);
    }

    CubDeviceArgScanPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.op = op;
    plan.mode = mode;
    plan.direction = direction;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceArgScanTempBytes(const Tensor& input,
                                 const Tensor& output,
                                 uint64_t num_items,
                                 CubArgScanOp op,
                                 CubScanMode mode,
                                 CubScanDirection direction) {
    return prepareCubDeviceArgScan(input, output, num_items, op, mode, direction).temp_storage_bytes;
}

void cubDeviceArgScan(const CubDeviceArgScanPlan& plan,
                      const Tensor& temp_storage,
                      const Tensor& input,
                      Tensor& output,
                      Stream& stream) {
    validateArgScan(input, output, plan.num_items);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB arg scan plan is not compatible with the provided tensors.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB arg scan dtype " + dtypeName(input.getDataType()) + ".");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0) {
        return;
    }

    auto launch = [&]<typename T>() -> void {
        auto launch_op = [&](auto scan_op, CubArgScanPair<T> init) -> void {
            launchDeviceArgScan<T>(plan, temp_storage, input, output, stream, scan_op, init);
        };
        dispatchArgScanOperator<T>(plan.op, launch_op);
    };

    dispatchScanDType(plan.dtype, launch);
}


template <typename T>
__global__ void gatherArgScanValuesFromIndicesKernel(const T* input,
                                                     const uint32_t* indices,
                                                     T* values,
                                                     uint32_t num_items,
                                                     CubArgScanOp op) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_items) {
        return;
    }
    const uint32_t source = indices[i];
    if (source == UINT32_MAX) {
        values[i] = op == CubArgScanOp::ArgMin ? scanPositiveInfinityOrMax<T>() : scanNegativeInfinityOrLowest<T>();
    } else {
        values[i] = input[source];
    }
}

void cubDeviceArgScanValuesFromIndices(const Tensor& input,
                                       const Tensor& indices,
                                       Tensor& values,
                                       uint64_t num_items,
                                       CubArgScanOp op,
                                       Stream& stream) {
    requireDenseContiguousGpuTensor(input, "arg scan value input");
    requireDenseContiguousGpuTensor(indices, "arg scan index output");
    requireDenseContiguousGpuTensor(values, "arg scan value output");
    requireSameGpuPlacement(input, indices, "arg scan value input", "arg scan index output");
    requireSameGpuPlacement(input, values, "arg scan value input", "arg scan value output");
    requireStorageForNumItems(input, "arg scan value input", num_items);
    requireStorageForNumItems(indices, "arg scan index output", num_items);
    requireStorageForNumItems(values, "arg scan value output", num_items);
    if (indices.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB arg scan indices must be UINT32.");
    }
    if (input.getDataType() != values.getDataType()) {
        throw std::invalid_argument("CUB arg scan value output dtype must match input dtype.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB arg scan value dtype " + dtypeName(input.getDataType()) + ".");
    }
    if (num_items == 0) {
        return;
    }
    const uint32_t cub_items = static_cast<uint32_t>(checkedCubNumItems(num_items));
    const uint32_t threads = 256;
    const uint32_t blocks = (cub_items + threads - 1U) / threads;
    auto launch = [&]<typename T>() -> void {
        gatherArgScanValuesFromIndicesKernel<T><<<blocks, threads, 0, stream.getStream()>>>(
            input.getMemPtr<T>(), indices.getMemPtr<uint32_t>(), values.getMemPtr<T>(), cub_items, op);
    };
    dispatchScanDType(input.getDataType(), launch);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ThorImplementation
