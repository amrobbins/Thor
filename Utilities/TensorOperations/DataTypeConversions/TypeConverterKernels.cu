#include "TypeConverter.h"

#include <stdio.h>
#include <limits>
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/TensorOperations/DataTypeConversions/ElementConverter.h"

using namespace ThorImplementation;
using namespace std;

// Launch the out-of-place kernel:
template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);

// Launch the in-place widening kernel that converts the first chunk:
template <typename FROM_TYPE, typename TO_TYPE>
void launchReadConvertSyncWriteKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);

void TypeConverter::gpuConvertType(void *source_d,
                                   void *dest_d,
                                   DataType sourceDataType,
                                   DataType destDataType,
                                   long numElements,
                                   Stream stream) {
    switch (sourceDataType) {
        case DataType::FP8_E4M3:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, __nv_fp8_e4m3>(
                        (__nv_fp8_e4m3 *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, __nv_fp8_e5m2>(
                        (__nv_fp8_e4m3 *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, __nv_bfloat16>(
                        (__nv_fp8_e4m3 *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, half>((__nv_fp8_e4m3 *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, float>((__nv_fp8_e4m3 *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, double>((__nv_fp8_e4m3 *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, int8_t>((__nv_fp8_e4m3 *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, int16_t>((__nv_fp8_e4m3 *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, int32_t>((__nv_fp8_e4m3 *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, int64_t>((__nv_fp8_e4m3 *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, uint8_t>((__nv_fp8_e4m3 *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, uint16_t>((__nv_fp8_e4m3 *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, uint32_t>((__nv_fp8_e4m3 *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, uint64_t>((__nv_fp8_e4m3 *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<__nv_fp8_e4m3, bool>((__nv_fp8_e4m3 *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::FP8_E5M2:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, __nv_fp8_e4m3>(
                        (__nv_fp8_e5m2 *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, __nv_fp8_e5m2>(
                        (__nv_fp8_e5m2 *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, __nv_bfloat16>(
                        (__nv_fp8_e5m2 *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, half>((__nv_fp8_e5m2 *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, float>((__nv_fp8_e5m2 *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, double>((__nv_fp8_e5m2 *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, int8_t>((__nv_fp8_e5m2 *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, int16_t>((__nv_fp8_e5m2 *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, int32_t>((__nv_fp8_e5m2 *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, int64_t>((__nv_fp8_e5m2 *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, uint8_t>((__nv_fp8_e5m2 *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, uint16_t>((__nv_fp8_e5m2 *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, uint32_t>((__nv_fp8_e5m2 *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, uint64_t>((__nv_fp8_e5m2 *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<__nv_fp8_e5m2, bool>((__nv_fp8_e5m2 *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::BF16:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<__nv_bfloat16, __nv_fp8_e4m3>(
                        (__nv_bfloat16 *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<__nv_bfloat16, __nv_fp8_e5m2>(
                        (__nv_bfloat16 *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<__nv_bfloat16, __nv_bfloat16>(
                        (__nv_bfloat16 *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<__nv_bfloat16, half>((__nv_bfloat16 *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<__nv_bfloat16, float>((__nv_bfloat16 *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<__nv_bfloat16, double>((__nv_bfloat16 *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<__nv_bfloat16, int8_t>((__nv_bfloat16 *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<__nv_bfloat16, int16_t>((__nv_bfloat16 *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<__nv_bfloat16, int32_t>((__nv_bfloat16 *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<__nv_bfloat16, int64_t>((__nv_bfloat16 *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<__nv_bfloat16, uint8_t>((__nv_bfloat16 *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<__nv_bfloat16, uint16_t>((__nv_bfloat16 *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<__nv_bfloat16, uint32_t>((__nv_bfloat16 *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<__nv_bfloat16, uint64_t>((__nv_bfloat16 *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<__nv_bfloat16, bool>((__nv_bfloat16 *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::FP16:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<half, __nv_fp8_e4m3>((half *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<half, __nv_fp8_e5m2>((half *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<half, __nv_bfloat16>((half *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<half, half>((half *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<half, float>((half *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<half, double>((half *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<half, int8_t>((half *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<half, int16_t>((half *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<half, int32_t>((half *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<half, int64_t>((half *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<half, uint8_t>((half *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<half, uint16_t>((half *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<half, uint32_t>((half *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<half, uint64_t>((half *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<half, bool>((half *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::FP32:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<float, __nv_fp8_e4m3>((float *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<float, __nv_fp8_e5m2>((float *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<float, __nv_bfloat16>((float *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<float, half>((float *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<float, float>((float *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<float, double>((float *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<float, int8_t>((float *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<float, int16_t>((float *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<float, int32_t>((float *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<float, int64_t>((float *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<float, uint8_t>((float *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<float, uint16_t>((float *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<float, uint32_t>((float *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<float, uint64_t>((float *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<float, bool>((float *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::FP64:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<double, __nv_fp8_e4m3>((double *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<double, __nv_fp8_e5m2>((double *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<double, __nv_bfloat16>((double *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<double, half>((double *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<double, float>((double *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<double, double>((double *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<double, int8_t>((double *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<double, int16_t>((double *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<double, int32_t>((double *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<double, int64_t>((double *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<double, uint8_t>((double *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<double, uint16_t>((double *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<double, uint32_t>((double *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<double, uint64_t>((double *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<double, bool>((double *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::INT8:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<int8_t, __nv_fp8_e4m3>((int8_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<int8_t, __nv_fp8_e5m2>((int8_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<int8_t, __nv_bfloat16>((int8_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<int8_t, half>((int8_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<int8_t, float>((int8_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<int8_t, double>((int8_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<int8_t, int8_t>((int8_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<int8_t, int16_t>((int8_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<int8_t, int32_t>((int8_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<int8_t, int64_t>((int8_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<int8_t, uint8_t>((int8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<int8_t, uint16_t>((int8_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<int8_t, uint32_t>((int8_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<int8_t, uint64_t>((int8_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<int8_t, bool>((int8_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::INT16:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<int16_t, __nv_fp8_e4m3>((int16_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<int16_t, __nv_fp8_e5m2>((int16_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<int16_t, __nv_bfloat16>((int16_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<int16_t, half>((int16_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<int16_t, float>((int16_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<int16_t, double>((int16_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<int16_t, int8_t>((int16_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<int16_t, int16_t>((int16_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<int16_t, int32_t>((int16_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<int16_t, int64_t>((int16_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<int16_t, uint8_t>((int16_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<int16_t, uint16_t>((int16_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<int16_t, uint32_t>((int16_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<int16_t, uint64_t>((int16_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<int16_t, bool>((int16_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::INT32:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<int32_t, __nv_fp8_e4m3>((int32_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<int32_t, __nv_fp8_e5m2>((int32_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<int32_t, __nv_bfloat16>((int32_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<int32_t, half>((int32_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<int32_t, float>((int32_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<int32_t, double>((int32_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<int32_t, int8_t>((int32_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<int32_t, int16_t>((int32_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<int32_t, int32_t>((int32_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<int32_t, int64_t>((int32_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<int32_t, uint8_t>((int32_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<int32_t, uint16_t>((int32_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<int32_t, uint32_t>((int32_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<int32_t, uint64_t>((int32_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<int32_t, bool>((int32_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::INT64:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<int64_t, __nv_fp8_e4m3>((int64_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<int64_t, __nv_fp8_e5m2>((int64_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<int64_t, __nv_bfloat16>((int64_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<int64_t, half>((int64_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<int64_t, float>((int64_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<int64_t, double>((int64_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<int64_t, int8_t>((int64_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<int64_t, int16_t>((int64_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<int64_t, int32_t>((int64_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<int64_t, int64_t>((int64_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<int64_t, uint8_t>((int64_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<int64_t, uint16_t>((int64_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<int64_t, uint32_t>((int64_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<int64_t, uint64_t>((int64_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<int64_t, bool>((int64_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::UINT8:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<uint8_t, __nv_fp8_e4m3>((uint8_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<uint8_t, __nv_fp8_e5m2>((uint8_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<uint8_t, __nv_bfloat16>((uint8_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<uint8_t, half>((uint8_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<uint8_t, float>((uint8_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<uint8_t, double>((uint8_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<uint8_t, int8_t>((uint8_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<uint8_t, int16_t>((uint8_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<uint8_t, int32_t>((uint8_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<uint8_t, int64_t>((uint8_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<uint8_t, uint8_t>((uint8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<uint8_t, uint16_t>((uint8_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<uint8_t, uint32_t>((uint8_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<uint8_t, uint64_t>((uint8_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint8_t, bool>((uint8_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::UINT16:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<uint16_t, __nv_fp8_e4m3>((uint16_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<uint16_t, __nv_fp8_e5m2>((uint16_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<uint16_t, __nv_bfloat16>((uint16_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<uint16_t, half>((uint16_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<uint16_t, float>((uint16_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<uint16_t, double>((uint16_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<uint16_t, int8_t>((uint16_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<uint16_t, int16_t>((uint16_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<uint16_t, int32_t>((uint16_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<uint16_t, int64_t>((uint16_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<uint16_t, uint8_t>((uint16_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<uint16_t, uint16_t>((uint16_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<uint16_t, uint32_t>((uint16_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<uint16_t, uint64_t>((uint16_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint16_t, bool>((uint16_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::UINT32:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<uint32_t, __nv_fp8_e4m3>((uint32_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<uint32_t, __nv_fp8_e5m2>((uint32_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<uint32_t, __nv_bfloat16>((uint32_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<uint32_t, half>((uint32_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<uint32_t, float>((uint32_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<uint32_t, double>((uint32_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<uint32_t, int8_t>((uint32_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<uint32_t, int16_t>((uint32_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<uint32_t, int32_t>((uint32_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<uint32_t, int64_t>((uint32_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<uint32_t, uint8_t>((uint32_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<uint32_t, uint16_t>((uint32_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<uint32_t, uint32_t>((uint32_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<uint32_t, uint64_t>((uint32_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint32_t, bool>((uint32_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::UINT64:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<uint64_t, __nv_fp8_e4m3>((uint64_t *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<uint64_t, __nv_fp8_e5m2>((uint64_t *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<uint64_t, __nv_bfloat16>((uint64_t *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<uint64_t, half>((uint64_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<uint64_t, float>((uint64_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<uint64_t, double>((uint64_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<uint64_t, int8_t>((uint64_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<uint64_t, int16_t>((uint64_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<uint64_t, int32_t>((uint64_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<uint64_t, int64_t>((uint64_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<uint64_t, uint8_t>((uint64_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<uint64_t, uint16_t>((uint64_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<uint64_t, uint32_t>((uint64_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<uint64_t, uint64_t>((uint64_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint64_t, bool>((uint64_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        case DataType::BOOLEAN:
            switch (destDataType) {
                case DataType::FP8_E4M3:
                    gpuConvertTypeImpl<bool, __nv_fp8_e4m3>((bool *)source_d, (__nv_fp8_e4m3 *)dest_d, numElements, stream);
                    break;
                case DataType::FP8_E5M2:
                    gpuConvertTypeImpl<bool, __nv_fp8_e5m2>((bool *)source_d, (__nv_fp8_e5m2 *)dest_d, numElements, stream);
                    break;
                case DataType::BF16:
                    gpuConvertTypeImpl<bool, __nv_bfloat16>((bool *)source_d, (__nv_bfloat16 *)dest_d, numElements, stream);
                    break;
                case DataType::FP16:
                    gpuConvertTypeImpl<bool, half>((bool *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case DataType::FP32:
                    gpuConvertTypeImpl<bool, float>((bool *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case DataType::FP64:
                    gpuConvertTypeImpl<bool, double>((bool *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case DataType::INT8:
                    gpuConvertTypeImpl<bool, int8_t>((bool *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT16:
                    gpuConvertTypeImpl<bool, int16_t>((bool *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT32:
                    gpuConvertTypeImpl<bool, int32_t>((bool *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case DataType::INT64:
                    gpuConvertTypeImpl<bool, int64_t>((bool *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT8:
                    gpuConvertTypeImpl<bool, uint8_t>((bool *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT16:
                    gpuConvertTypeImpl<bool, uint16_t>((bool *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT32:
                    gpuConvertTypeImpl<bool, uint32_t>((bool *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case DataType::UINT64:
                    gpuConvertTypeImpl<bool, uint64_t>((bool *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case DataType::BOOLEAN:
                    gpuConvertTypeImpl<bool, bool>((bool *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                default:
                    THOR_UNREACHABLE();
            }
            break;
        default:
            THOR_UNREACHABLE();
    }
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelNoOvershoot(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index >= numElements)
            return;
        dest_d[index] = ElementConverter<FROM_TYPE, TO_TYPE>{}(((FROM_TYPE *)source_d)[index]);
    }
}

template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernelNoOvershoot(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    convertOutOfPlaceKernelNoOvershoot<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

// To Bigger
template <typename FROM_TYPE, typename TO_TYPE>
void TypeConverter::convertToBiggerElementsInPlaceOnGpu(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    THOR_THROW_IF_FALSE(numElements >= 0);
    if (numElements == 0)
        return;

    // I have empty space in the allocated memory since I have the smaller element occupying enough memory to hold the bigger elements
    // So convert some number of the trailing elements and put them in the back of the memory space, the trick here is that my writes cannot
    // overlap my reads, so I choose a number of elements such that there is enough empty space existing that I can write all converted
    // elements to it, after this the elements I just read and converted become empty space, so repeat until all elements are converted

    long numElementsLeft = numElements;
    long numEmptyBytes = (sizeof(TO_TYPE) - sizeof(FROM_TYPE)) * numElements;

    while (numElementsLeft > 8 * 256) {
        long chunkSize = numEmptyBytes / sizeof(TO_TYPE);
        long startingElement = numElementsLeft - chunkSize;

        launchOutOfPlaceConvertKernelNoOvershoot<FROM_TYPE, TO_TYPE>(
            source_d + startingElement, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes += (sizeof(FROM_TYPE) - sizeof(TO_TYPE)) * chunkSize;
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElementsLeft, stream);
    }
}

template <typename FROM_TYPE, typename TO_TYPE>
void TypeConverter::gpuConvertTypeImpl(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    // A conversion to the same type is refused (Tensor copies those as bytes), so it instantiates no kernel.
    THOR_THROW_IF_FALSE(!(is_same<FROM_TYPE, TO_TYPE>::value));
    if constexpr (!is_same_v<FROM_TYPE, TO_TYPE>) {
        THOR_THROW_IF_FALSE(numElements >= 0);
        if (numElements == 0)
            return;

        ScopedGpu scopedGpu(stream.getGpuNum());

        bool inPlaceConversion = ((void *)source_d == (void *)dest_d);

        // When not doing an inplace operation, the memory regions must not overlap
        if (!inPlaceConversion) {
            void *sourceStart = source_d;
            void *sourceEnd = (void *)&(source_d[numElements - 1]);
            void *destStart = dest_d;
            void *destEnd = (void *)&(dest_d[numElements - 1]);
            THOR_THROW_IF_FALSE(sourceEnd < destStart || sourceStart > destEnd);
        }

        // Every pair converts out of place with one kernel, which also runs in place between same-sized types. Only
        // widening pairs instantiate the in-place kernels; convertType refuses in-place narrowing.
        if constexpr (sizeof(FROM_TYPE) < sizeof(TO_TYPE)) {
            if (inPlaceConversion) {
                convertToBiggerElementsInPlaceOnGpu<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElements, stream);
                return;
            }
        }
        launchOutOfPlaceConvertKernel<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElements, stream);
    }
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertReadWholeChunkThenWriteWholeChunkKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    __shared__ FROM_TYPE buffer_shared[8 * 256];

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index < numElements)
            buffer_shared[i * 256 + threadIdx.x] = source_d[index];
    }

    __syncthreads();

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index < numElements)
            dest_d[index] = ElementConverter<FROM_TYPE, TO_TYPE>{}(((FROM_TYPE *)buffer_shared)[i * 256 + threadIdx.x]);
    }
}

// Out of place: each thread converts E = W / max(sizeof(FROM_TYPE), sizeof(TO_TYPE)) elements, that is W bytes of
// the wider side and its share of the narrower side, each moved in the widest accesses that fit (16 bytes, or 32 where
// nvcc merges a pair on sm_100+). W is 16 bytes when the destination is at least as wide as the source and 32 when it
// is narrower: the fastest of 16 to 128 in the 1518 probe (RTX 5090; 16 beat 32 by 1.5% to 5% when widening, 32 beat
// 16 by up to 1% when narrowing). A thread whose elements run past the end converts them one at a time, so nothing is
// read or written past the last element.
template <typename FROM_TYPE, typename TO_TYPE>
__host__ __device__ constexpr int outOfPlaceElementsPerThread() {
    constexpr int wide = sizeof(FROM_TYPE) > sizeof(TO_TYPE) ? static_cast<int>(sizeof(FROM_TYPE)) : static_cast<int>(sizeof(TO_TYPE));
    constexpr int bytesOfTheWiderSide = sizeof(FROM_TYPE) > sizeof(TO_TYPE) ? 32 : 16;
    static_assert(wide == 1 || wide == 2 || wide == 4 || wide == 8, "element sizes of 1, 2, 4 and 8 bytes");
    return bytesOfTheWiderSide / wide;
}

template <typename FROM_TYPE, typename TO_TYPE>
__global__ void __launch_bounds__(256) convertOutOfPlaceKernel(const FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    constexpr int E = outOfPlaceElementsPerThread<FROM_TYPE, TO_TYPE>();
    const long first = (static_cast<long>(blockIdx.x) * 256 + threadIdx.x) * E;
    if (first >= numElements)
        return;
    if (first + E > numElements) {
        // At most one thread per launch runs this. The loop stays rolled, so it adds one copy of the conversion to
        // the kernel, not one per element.
#pragma unroll 1
        for (long i = first; i < numElements; ++i)
            dest_d[i] = ElementConverter<FROM_TYPE, TO_TYPE>{}(source_d[i]);
        return;
    }

    alignas(16) FROM_TYPE in[E];
    loadConvertBytes<E * static_cast<int>(sizeof(FROM_TYPE))>(in, source_d + first);
    alignas(16) TO_TYPE out[E];
#pragma unroll
    for (int i = 0; i < E; ++i)
        out[i] = ElementConverter<FROM_TYPE, TO_TYPE>{}(in[i]);
    storeConvertBytes<E * static_cast<int>(sizeof(TO_TYPE))>(dest_d + first, out);
}

// Also converts in place between same-sized types: each thread reads its elements before it writes them.
template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    constexpr long E = outOfPlaceElementsPerThread<FROM_TYPE, TO_TYPE>();
    const long blocks = ((numElements + E - 1) / E + 255) / 256;
    THOR_THROW_IF_FALSE(blocks <= static_cast<long>(std::numeric_limits<int>::max()));
    convertOutOfPlaceKernel<FROM_TYPE, TO_TYPE>
        <<<static_cast<unsigned int>(blocks), 256, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE, typename TO_TYPE>
void launchReadConvertSyncWriteKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel<FROM_TYPE, TO_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}
