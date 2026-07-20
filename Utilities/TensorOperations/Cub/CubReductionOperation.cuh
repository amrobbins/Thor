#pragma once

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::CubReductionInternal {

template <typename Fn>
decltype(auto) dispatchReductionInputDType(DataType dtype, Fn&& fn) {
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
            throw std::invalid_argument("Unsupported CUB tensor reduction input dtype value "
                                        + std::to_string(static_cast<int>(dtype)) + ".");
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

struct IdentityFp32 {
    __host__ __device__ float operator()(float value) const { return value; }
};

struct AbsoluteValueFp32 {
    __host__ __device__ float operator()(float value) const { return ::fabsf(value); }
};

struct SquareFp32 {
    __host__ __device__ float operator()(float value) const { return value * value; }
};

struct MeanFinalizeFp32 {
    float count;

    __host__ __device__ float operator()(float value) const { return count == 0.0f ? 0.0f : value / count; }
};

struct SquareRootFinalizeFp32 {
    __host__ __device__ float operator()(float value) const { return ::sqrtf(value); }
};

struct PropagatingMinimumFp32 {
    __host__ __device__ float operator()(float lhs, float rhs) const {
        if (lhs != lhs) {
            return lhs;
        }
        if (rhs != rhs) {
            return rhs;
        }
        return ::fminf(lhs, rhs);
    }
};

struct PropagatingMaximumFp32 {
    __host__ __device__ float operator()(float lhs, float rhs) const {
        if (lhs != lhs) {
            return lhs;
        }
        if (rhs != rhs) {
            return rhs;
        }
        return ::fmaxf(lhs, rhs);
    }
};

inline __host__ __device__ void storeFp32AsRuntimeDType(void* output,
                                                         DataType output_dtype,
                                                         uint64_t index,
                                                         float value) {
    switch (output_dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            static_cast<__nv_fp8_e4m3*>(output)[index] = ThorLowPrecision::toFp8E4M3Satfinite(value);
            return;
        case DataType::FP8_E5M2: {
            __nv_fp8_e5m2 converted;
            converted.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
            static_cast<__nv_fp8_e5m2*>(output)[index] = converted;
            return;
        }
#endif
        case DataType::FP16:
            static_cast<__half*>(output)[index] = __float2half_rn(value);
            return;
        case DataType::BF16:
            static_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(value);
            return;
        case DataType::FP32:
            static_cast<float*>(output)[index] = value;
            return;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            static_cast<double*>(output)[index] = static_cast<double>(value);
            return;
#endif
        default:
#if defined(__CUDA_ARCH__)
            asm("trap;");
#endif
            return;
    }
}

// Keep output storage dtype out of the CUB iterator type. The dtype is uniform for the launch and is selected only
// when each final aggregate is stored. This avoids instantiating every CUB kernel for InputT x OutputT x operation x
// path while preserving fused FP32 finalization and storage conversion.
template <typename OutputFinalizeT>
struct FinalizeAndStoreRuntimeFp32 {
    void* output;
    DataType output_dtype;
    OutputFinalizeT finalize;

    template <typename IndexT>
    __host__ __device__ void operator()(IndexT index, float value) const {
        storeFp32AsRuntimeDType(output, output_dtype, static_cast<uint64_t>(index), finalize(value));
    }
};

template <typename OutputFinalizeT>
auto makeRuntimeFp32OutputIterator(void* output, DataType output_dtype, OutputFinalizeT output_finalize) {
    return cuda::make_tabulate_output_iterator(
        FinalizeAndStoreRuntimeFp32<OutputFinalizeT>{output, output_dtype, output_finalize});
}

template <typename InputT, typename InputTransformT>
struct ConvertAndTransformInputToFp32 {
    InputTransformT transform;

    __host__ __device__ float operator()(InputT value) const { return transform(ToFp32<InputT>{}(value)); }
};

template <typename InputT, typename InputTransformT>
struct LogicalAxesToFp32 {
    const InputT* input;
    uint64_t reduction_size;
    CubReductionDeviceIndexing indexing;
    InputTransformT transform;

    __host__ __device__ float operator()(int64_t logical_index) const {
        const uint64_t unsigned_logical_index = static_cast<uint64_t>(logical_index);
        const uint64_t output_index = unsigned_logical_index / reduction_size;
        const uint64_t reduction_index = unsigned_logical_index - output_index * reduction_size;
        const uint64_t physical_index = mapLogicalReductionIndex(indexing, output_index, reduction_index);
        return transform(ToFp32<InputT>{}(input[physical_index]));
    }
};

template <typename InputT, typename InputTransformT>
auto makeContiguousFp32Iterator(const InputT* input, InputTransformT input_transform) {
    return thrust::make_transform_iterator(
        input, ConvertAndTransformInputToFp32<InputT, InputTransformT>{input_transform});
}

template <typename InputT, typename InputTransformT>
auto makeStridedFp32Iterator(const Tensor& input,
                             const CubReductionGeometry& geometry,
                             InputTransformT input_transform) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        LogicalAxesToFp32<InputT, InputTransformT>{
            input.getMemPtr<InputT>(), geometry.reduction_size, geometry.device_indexing, input_transform});
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
size_t queryReductionBytesForInput(const Tensor& input,
                                   Tensor& output,
                                   const CubReductionGeometry& geometry,
                                   ReductionOpT reduction_op,
                                   float init,
                                   InputTransformT input_transform,
                                   OutputFinalizeT output_finalize,
                                   cudaStream_t stream) {
    using AccumulatorT =
        std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<float>(), std::declval<float>()))>;
    static_assert(std::is_same_v<AccumulatorT, float>, "CUB tensor reductions must accumulate in FP32.");
    static_assert(std::is_same_v<decltype(std::declval<InputTransformT>()(std::declval<float>())), float>,
                  "CUB tensor reduction input transforms must produce FP32.");
    static_assert(std::is_same_v<decltype(std::declval<OutputFinalizeT>()(std::declval<float>())), float>,
                  "CUB tensor reduction output finalizers must produce FP32.");

    size_t queried_bytes = 0;
    auto output_iterator =
        makeRuntimeFp32OutputIterator(output.getMemPtr<void>(), output.getDataType(), output_finalize);
    const ConvertAndTransformInputToFp32<InputT, InputTransformT> device_input_transform{input_transform};

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce:
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(nullptr,
                                                          queried_bytes,
                                                          input.getMemPtr<InputT>(),
                                                          output_iterator,
                                                          static_cast<int64_t>(input.getTotalNumElements()),
                                                          reduction_op,
                                                          device_input_transform,
                                                          init,
                                                          stream));
            break;
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>(), input_transform);
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
            auto input_iterator = makeStridedFp32Iterator<InputT>(input, geometry, input_transform);
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
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB reduction received offset-segmented geometry.");
    }

    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchReductionForInput(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& input,
                             Tensor& output,
                             const CubReductionGeometry& geometry,
                             ReductionOpT reduction_op,
                             float init,
                             InputTransformT input_transform,
                             OutputFinalizeT output_finalize,
                             cudaStream_t stream) {
    using AccumulatorT =
        std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<float>(), std::declval<float>()))>;
    static_assert(std::is_same_v<AccumulatorT, float>, "CUB tensor reductions must accumulate in FP32.");

    void* temp_storage_ptr =
        const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto output_iterator =
        makeRuntimeFp32OutputIterator(output.getMemPtr<void>(), output.getDataType(), output_finalize);
    const ConvertAndTransformInputToFp32<InputT, InputTransformT> device_input_transform{input_transform};

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce:
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input.getMemPtr<InputT>(),
                                                          output_iterator,
                                                          static_cast<int64_t>(input.getTotalNumElements()),
                                                          reduction_op,
                                                          device_input_transform,
                                                          init,
                                                          stream));
            break;
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>(), input_transform);
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
            auto input_iterator = makeStridedFp32Iterator<InputT>(input, geometry, input_transform);
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
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB reduction received offset-segmented geometry.");
    }
}

template <typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
size_t queryOperationReductionBytes(const Tensor& input,
                                    Tensor& output,
                                    const CubReductionGeometry& geometry,
                                    ReductionOpT reduction_op,
                                    float init,
                                    InputTransformT input_transform,
                                    OutputFinalizeT output_finalize,
                                    const Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        return queryReductionBytesForInput<InputT>(input,
                                                   output,
                                                   geometry,
                                                   reduction_op,
                                                   init,
                                                   input_transform,
                                                   output_finalize,
                                                   stream.getStream());
    };
    return dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

template <typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchOperationReduction(const Tensor& temp_storage,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              ReductionOpT reduction_op,
                              float init,
                              InputTransformT input_transform,
                              OutputFinalizeT output_finalize,
                              Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> void {
        launchReductionForInput<InputT>(temp_storage,
                                        temp_storage_bytes,
                                        input,
                                        output,
                                        geometry,
                                        reduction_op,
                                        init,
                                        input_transform,
                                        output_finalize,
                                        stream.getStream());
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

}  // namespace ThorImplementation::CubReductionInternal
