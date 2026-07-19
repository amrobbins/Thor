#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation {
namespace {

using namespace CubDevicePrimitiveSupport;

[[nodiscard]] bool isSupportedFloatingStorageDType(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
#endif
            return true;
        default:
            return false;
    }
}

void requireSupportedFloatingStorageDType(DataType dtype, const char* role) {
    if (!isSupportedFloatingStorageDType(dtype)) {
        throw std::invalid_argument(std::string("CUB tensor reduction does not support ") + role + " dtype " + dtypeName(dtype) +
                                    ". Supported storage dtypes follow Thor's THOR_CUB_ENABLE_FP8_TYPES and "
                                    "THOR_CUB_ENABLE_64BIT_TYPES build policy; TF32 is not a storage dtype.");
    }
}

void requireCompatibleStream(const Tensor& input, const Stream& stream) {
    if (!stream.isInitialized()) {
        throw std::invalid_argument("CUB tensor reduction requires an initialized stream.");
    }
    if (input.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("CUB tensor reduction stream must belong to the input tensor's GPU.");
    }
}

void requireExpectedOutput(const Tensor& input,
                           const Tensor& output,
                           DataType output_dtype,
                           const CubReductionGeometry& geometry) {
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    if (output.getDataType() != output_dtype) {
        throw std::invalid_argument("CUB tensor reduction preallocated output dtype does not match the configured output dtype.");
    }
    if (output.getDimensions() != geometry.output_dimensions) {
        throw std::invalid_argument("CUB tensor reduction preallocated output dimensions do not match the single-axis reduction shape.");
    }

    const uintptr_t input_begin = reinterpret_cast<uintptr_t>(input.getMemPtr<void>());
    const uintptr_t output_begin = reinterpret_cast<uintptr_t>(output.getMemPtr<void>());
    const uintptr_t input_end = input_begin + input.getArraySizeInBytes();
    const uintptr_t output_end = output_begin + output.getArraySizeInBytes();
    if (input_begin < output_end && output_begin < input_end) {
        throw std::invalid_argument("CUB tensor reduction input and output storage must not overlap.");
    }
}

template <typename Fn>
decltype(auto) dispatchReductionStorageDType(DataType dtype, const char* role, Fn&& fn) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return fn.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return fn.template operator()<__nv_fp8_e5m2>();
#endif
        case DataType::FP16:
            return fn.template operator()<__half>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return fn.template operator()<float>();
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return fn.template operator()<double>();
#endif
        default:
            throw std::invalid_argument(std::string("Unsupported CUB tensor reduction ") + role + " dtype " + dtypeName(dtype) + ".");
    }
}

template <typename T>
struct ToFp32 {
    __host__ __device__ float operator()(T value) const { return static_cast<float>(value); }
};

template <>
struct ToFp32<__half> {
    __host__ __device__ float operator()(__half value) const { return __half2float(value); }
};

template <>
struct ToFp32<__nv_bfloat16> {
    __host__ __device__ float operator()(__nv_bfloat16 value) const { return __bfloat162float(value); }
};

template <typename OutputT>
struct Fp32ToOutput {
    __host__ __device__ OutputT operator()(float value) const { return static_cast<OutputT>(value); }
};

template <>
struct Fp32ToOutput<__half> {
    __host__ __device__ __half operator()(float value) const { return __float2half_rn(value); }
};

template <>
struct Fp32ToOutput<__nv_bfloat16> {
    __host__ __device__ __nv_bfloat16 operator()(float value) const { return __float2bfloat16_rn(value); }
};

#if THOR_CUB_ENABLE_FP8_TYPES
template <>
struct Fp32ToOutput<__nv_fp8_e4m3> {
    __host__ __device__ __nv_fp8_e4m3 operator()(float value) const {
        return ThorLowPrecision::toFp8E4M3Satfinite(value);
    }
};

template <>
struct Fp32ToOutput<__nv_fp8_e5m2> {
    __host__ __device__ __nv_fp8_e5m2 operator()(float value) const {
        __nv_fp8_e5m2 result;
        result.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
        return result;
    }
};
#endif

template <typename OutputT>
auto makeFp32OutputIterator(OutputT* output) {
    if constexpr (std::is_same_v<OutputT, float>) {
        return output;
    } else {
        return cuda::make_transform_output_iterator(output, Fp32ToOutput<OutputT>{});
    }
}

template <typename InputT>
struct StridedAxisToFp32 {
    const InputT* input;
    int64_t reduction_size;
    int64_t inner_size;

    __host__ __device__ float operator()(int64_t logical_index) const {
        const int64_t output_index = logical_index / reduction_size;
        const int64_t reduction_index = logical_index - output_index * reduction_size;
        const int64_t outer_index = output_index / inner_size;
        const int64_t inner_index = output_index - outer_index * inner_size;
        const int64_t physical_index = (outer_index * reduction_size + reduction_index) * inner_size + inner_index;
        return ToFp32<InputT>{}(input[physical_index]);
    }
};

template <typename InputT>
auto makeContiguousFp32Iterator(const InputT* input) {
    if constexpr (std::is_same_v<InputT, float>) {
        return input;
    } else {
        return thrust::make_transform_iterator(input, ToFp32<InputT>{});
    }
}

template <typename InputT>
auto makeStridedFp32Iterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        StridedAxisToFp32<InputT>{input.getMemPtr<InputT>(),
                                  static_cast<int64_t>(geometry.reduction_size),
                                  static_cast<int64_t>(geometry.inner_size)});
}

template <typename Fn>
decltype(auto) dispatchReductionOperator(CubReductionOp op, Fn&& fn) {
    switch (op) {
        case CubReductionOp::Sum:
            return fn(cuda::std::plus<float>{}, 0.0f);
        case CubReductionOp::Min:
            return fn(cuda::minimum<float>{}, std::numeric_limits<float>::infinity());
        case CubReductionOp::Max:
            return fn(cuda::maximum<float>{}, -std::numeric_limits<float>::infinity());
    }
    throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
}

template <typename InputT, typename OutputT, typename ReductionOpT>
size_t queryReductionBytes(CubReductionPath path,
                           const Tensor& input,
                           Tensor& output,
                           const CubReductionGeometry& geometry,
                           ReductionOpT reduction_op,
                           float init,
                           cudaStream_t stream) {
    using AccumulatorT =
        std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<float>(), std::declval<float>()))>;
    static_assert(std::is_same_v<AccumulatorT, float>, "CUB tensor reductions must accumulate in FP32.");

    size_t queried_bytes = 0;
    auto output_iterator = makeFp32OutputIterator(output.getMemPtr<OutputT>());

    switch (path) {
        case CubReductionPath::DeviceTransformReduce:
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(nullptr,
                                                          queried_bytes,
                                                          input.getMemPtr<InputT>(),
                                                          output_iterator,
                                                          static_cast<int64_t>(input.getTotalNumElements()),
                                                          reduction_op,
                                                          ToFp32<InputT>{},
                                                          init,
                                                          stream));
            break;
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>());
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedFp32Iterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
    }

    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename OutputT, typename ReductionOpT>
void launchReduction(CubReductionPath path,
                     const Tensor& temp_storage,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     const CubReductionGeometry& geometry,
                     ReductionOpT reduction_op,
                     float init,
                     cudaStream_t stream) {
    using AccumulatorT =
        std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<float>(), std::declval<float>()))>;
    static_assert(std::is_same_v<AccumulatorT, float>, "CUB tensor reductions must accumulate in FP32.");

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    auto output_iterator = makeFp32OutputIterator(output.getMemPtr<OutputT>());

    switch (path) {
        case CubReductionPath::DeviceTransformReduce:
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input.getMemPtr<InputT>(),
                                                          output_iterator,
                                                          static_cast<int64_t>(input.getTotalNumElements()),
                                                          reduction_op,
                                                          ToFp32<InputT>{},
                                                          init,
                                                          stream));
            break;
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>());
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedFp32Iterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
    }
}

size_t queryReductionBytes(CubReductionOp op,
                           const Tensor& input,
                           Tensor& output,
                           const CubReductionGeometry& geometry,
                           const Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        auto dispatch_output = [&]<typename OutputT>() -> size_t {
            return dispatchReductionOperator(op, [&](auto reduction_op, float init) {
                return queryReductionBytes<InputT, OutputT>(
                    geometry.path, input, output, geometry, reduction_op, init, stream.getStream());
            });
        };
        return dispatchReductionStorageDType(output.getDataType(), "output", dispatch_output);
    };
    return dispatchReductionStorageDType(input.getDataType(), "input", dispatch_input);
}

void launchReduction(CubReductionOp op,
                     const Tensor& temp_storage,
                     size_t temp_storage_bytes,
                     const Tensor& input,
                     Tensor& output,
                     const CubReductionGeometry& geometry,
                     Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> void {
        auto dispatch_output = [&]<typename OutputT>() -> void {
            dispatchReductionOperator(op, [&](auto reduction_op, float init) {
                launchReduction<InputT, OutputT>(geometry.path,
                                                 temp_storage,
                                                 temp_storage_bytes,
                                                 input,
                                                 output,
                                                 geometry,
                                                 reduction_op,
                                                 init,
                                                 stream.getStream());
            });
        };
        dispatchReductionStorageDType(output.getDataType(), "output", dispatch_output);
    };
    dispatchReductionStorageDType(input.getDataType(), "input", dispatch_input);
}


}  // namespace

CubReduction::CubReduction(CubReductionOp op, uint32_t axis, std::optional<DataType> output_dtype)
    : op(op), axis(axis), output_dtype(output_dtype) {
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Min:
        case CubReductionOp::Max:
            break;
        default:
            throw std::invalid_argument("Unsupported CUB tensor reduction operation.");
    }
    if (output_dtype.has_value()) {
        requireSupportedFloatingStorageDType(output_dtype.value(), "output");
    }
}

DataType CubReduction::resolveOutputDataType(DataType input_dtype) const {
    requireSupportedFloatingStorageDType(input_dtype, "input");
    const DataType resolved = output_dtype.value_or(input_dtype);
    requireSupportedFloatingStorageDType(resolved, "output");
    return resolved;
}

CubReductionGeometry CubReduction::analyzeGeometry(const std::vector<uint64_t>& input_dimensions, uint32_t axis) {
    if (input_dimensions.empty()) {
        throw std::invalid_argument("CUB tensor reduction requires at least one input dimension.");
    }
    if (axis >= input_dimensions.size()) {
        throw std::invalid_argument("CUB tensor reduction axis is outside the input rank.");
    }

    auto checkedMultiply = [](uint64_t a, uint64_t b, const char* quantity) {
        if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
            throw std::invalid_argument(std::string("CUB tensor reduction ") + quantity + " overflows uint64_t.");
        }
        return a * b;
    };

    for (uint64_t dimension : input_dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument("CUB tensor reduction does not support zero-sized dimensions.");
        }
    }

    CubReductionGeometry geometry;
    geometry.axis = axis;
    geometry.reduction_size = input_dimensions[axis];
    geometry.outer_size = 1;
    geometry.inner_size = 1;
    geometry.output_dimensions = input_dimensions;
    geometry.output_dimensions[axis] = 1;

    for (uint32_t dimension = 0; dimension < axis; ++dimension) {
        geometry.outer_size = checkedMultiply(geometry.outer_size, input_dimensions[dimension], "outer size");
    }
    for (uint32_t dimension = axis + 1; dimension < input_dimensions.size(); ++dimension) {
        geometry.inner_size = checkedMultiply(geometry.inner_size, input_dimensions[dimension], "inner size");
    }
    geometry.output_elements = checkedMultiply(geometry.outer_size, geometry.inner_size, "output element count");
    const uint64_t input_elements =
        checkedMultiply(geometry.output_elements, geometry.reduction_size, "input element count");

    if (input_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction input element count exceeds CUB's int64 item-count limit.");
    }
    if (geometry.output_elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("CUB tensor reduction output element count exceeds CUB's int64 segment-count limit.");
    }
    if (geometry.reduction_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUB tensor reduction axis length exceeds CUB's fixed segment-size int limit.");
    }

    if (geometry.output_elements == 1) {
        geometry.path = CubReductionPath::DeviceTransformReduce;
    } else if (geometry.inner_size == 1) {
        geometry.path = CubReductionPath::ContiguousFixedSegment;
    } else {
        geometry.path = CubReductionPath::StridedFixedSegment;
    }

    return geometry;
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const Tensor& input, const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry = analyzeGeometry(input.getDimensions(), axis);
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    Tensor output(input.getPlacement(), TensorDescriptor(resolved_output_dtype, geometry.output_dimensions));
    return stampValidated(input, output, geometry, stream);
}

std::shared_ptr<StampedCubReduction> CubReduction::stampValidated(const Tensor& input,
                                                                  const Tensor& output,
                                                                  const CubReductionGeometry& geometry,
                                                                  const Stream& stream) const {
    requireSupportedFloatingStorageDType(input.getDataType(), "input");
    requireSupportedFloatingStorageDType(output.getDataType(), "output");
    requireExpectedOutput(input, output, output.getDataType(), geometry);

    ScopedGpu scoped_gpu(stream.getGpuNum());
    Tensor mutable_output = output;
    const size_t temp_storage_bytes = queryReductionBytes(op, input, mutable_output, geometry, stream);
    Tensor temp_storage(input.getPlacement(), TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(temp_storage_bytes)}));

    return std::shared_ptr<StampedCubReduction>(
        new StampedCubReduction(op, geometry, input, output, temp_storage_bytes, temp_storage, stream));
}

std::shared_ptr<StampedCubReduction> CubReduction::stamp(const Tensor& input,
                                                         const Tensor& preallocated_output,
                                                         const Stream& stream) const {
    requireDenseContiguousGpuTensor(input, "input");
    requireCompatibleStream(input, stream);
    const CubReductionGeometry geometry = analyzeGeometry(input.getDimensions(), axis);
    const DataType resolved_output_dtype = resolveOutputDataType(input.getDataType());
    requireExpectedOutput(input, preallocated_output, resolved_output_dtype, geometry);
    return stampValidated(input, preallocated_output, geometry, stream);
}

StampedCubReduction::StampedCubReduction(CubReductionOp op,
                                         CubReductionGeometry geometry,
                                         const Tensor& input,
                                         const Tensor& output,
                                         size_t temp_storage_bytes,
                                         const Tensor& temp_storage,
                                         const Stream& stream)
    : op(op),
      geometry(std::move(geometry)),
      input(input),
      output(output),
      temp_storage_bytes(temp_storage_bytes),
      temp_storage(temp_storage),
      stream(stream) {
    requireTempStorage(this->temp_storage, input.getPlacement(), temp_storage_bytes);
}

void StampedCubReduction::run() { runOn(stream); }

void StampedCubReduction::runOn(Stream& run_stream) const {
    requireCompatibleStream(input, run_stream);
    ScopedGpu scoped_gpu(run_stream.getGpuNum());
    launchReduction(op, temp_storage, temp_storage_bytes, input, output, geometry, run_stream);
}

}  // namespace ThorImplementation
