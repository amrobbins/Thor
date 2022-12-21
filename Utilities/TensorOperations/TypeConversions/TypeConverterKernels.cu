#include "TypeConverter.h"

#include <stdio.h>

using namespace ThorImplementation;
using namespace std;

// Launch out-of-place kernels:
template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
template <typename TO_TYPE>
void launchOutOfPlaceConvertKernel_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
template <typename FROM_TYPE>
void launchOutOfPlaceConvertKernel_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements, Stream stream);
template <typename FROM_TYPE>
void launchOutOfPlaceConvertKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream);
void launchOutOfPlaceConvertKernel_halfToPackedBoolean(half *source_d, uint8_t *dest_d, long numElements, Stream stream);
template <typename TO_TYPE>
void launchOutOfPlaceConvertKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
void launchOutOfPlaceConvertKernel_packedBooleanToHalf(uint8_t *source_d, half *dest_d, long numElements, Stream stream);

// Launch in-place kernels:
template <typename FROM_TYPE, typename TO_TYPE>
void launchReadConvertSyncWriteKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
template <typename TO_TYPE>
void launchReadConvertSyncWriteKernel_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
template <typename FROM_TYPE>
void launchReadConvertSyncWriteKernel_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements, Stream stream);
template <typename FROM_TYPE>
void launchReadConvertSyncWriteKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream);
void launchReadConvertSyncWriteKernel_halfToPackedBoolean(half *source_d, uint8_t *dest_d, long numElements, Stream stream);
template <typename TO_TYPE>
void launchReadConvertSyncWriteKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
void launchReadConvertSyncWriteKernel_packedBooleanToHalf(uint8_t *source_d, half *dest_d, long numElements, Stream stream);

void TypeConverter::gpuConvertType(void *source_d,
                                   void *dest_d,
                                   TensorDescriptor::DataType sourceDataType,
                                   TensorDescriptor::DataType destDataType,
                                   long numElements,
                                   Stream stream) {
    switch (sourceDataType) {
        case TensorDescriptor::DataType::FP16:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeImpl<half, half>((half *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<half, float>((half *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<half, double>((half *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeFromHalfToIntegralImpl<int8_t>((half *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeFromHalfToIntegralImpl<int16_t>((half *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeFromHalfToIntegralImpl<int32_t>((half *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeFromHalfToIntegralImpl<int64_t>((half *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeFromHalfToIntegralImpl<uint8_t>((half *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeFromHalfToIntegralImpl<uint16_t>((half *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeFromHalfToIntegralImpl<uint32_t>((half *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeFromHalfToIntegralImpl<uint64_t>((half *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeFromHalfToIntegralImpl<bool>((half *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeFromHalfToPackedBooleanImpl((half *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::FP32:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeImpl<float, half>((float *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<float, float>((float *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<float, double>((float *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<float, int8_t>((float *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<float, int16_t>((float *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<float, int32_t>((float *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<float, int64_t>((float *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<float, uint8_t>((float *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<float, uint16_t>((float *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<float, uint32_t>((float *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<float, uint64_t>((float *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<float, bool>((float *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<float>((float *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::FP64:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeImpl<double, half>((double *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<double, float>((double *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<double, double>((double *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<double, int8_t>((double *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<double, int16_t>((double *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<double, int32_t>((double *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<double, int64_t>((double *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<double, uint8_t>((double *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<double, uint16_t>((double *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<double, uint32_t>((double *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<double, uint64_t>((double *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<double, bool>((double *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<double>((double *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT8:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<int8_t>((int8_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<int8_t, float>((int8_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<int8_t, double>((int8_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<int8_t, int8_t>((int8_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<int8_t, int16_t>((int8_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<int8_t, int32_t>((int8_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<int8_t, int64_t>((int8_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<int8_t, uint8_t>((int8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<int8_t, uint16_t>((int8_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<int8_t, uint32_t>((int8_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<int8_t, uint64_t>((int8_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<int8_t, bool>((int8_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<int8_t>((int8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT16:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<int16_t>((int16_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<int16_t, float>((int16_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<int16_t, double>((int16_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<int16_t, int8_t>((int16_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<int16_t, int16_t>((int16_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<int16_t, int32_t>((int16_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<int16_t, int64_t>((int16_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<int16_t, uint8_t>((int16_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<int16_t, uint16_t>((int16_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<int16_t, uint32_t>((int16_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<int16_t, uint64_t>((int16_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<int16_t, bool>((int16_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<int16_t>((int16_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT32:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<int32_t>((int32_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<int32_t, float>((int32_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<int32_t, double>((int32_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<int32_t, int8_t>((int32_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<int32_t, int16_t>((int32_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<int32_t, int32_t>((int32_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<int32_t, int64_t>((int32_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<int32_t, uint8_t>((int32_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<int32_t, uint16_t>((int32_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<int32_t, uint32_t>((int32_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<int32_t, uint64_t>((int32_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<int32_t, bool>((int32_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<int32_t>((int32_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT64:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<int64_t>((int64_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<int64_t, float>((int64_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<int64_t, double>((int64_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<int64_t, int8_t>((int64_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<int64_t, int16_t>((int64_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<int64_t, int32_t>((int64_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<int64_t, int64_t>((int64_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<int64_t, uint8_t>((int64_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<int64_t, uint16_t>((int64_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<int64_t, uint32_t>((int64_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<int64_t, uint64_t>((int64_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<int64_t, bool>((int64_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<int64_t>((int64_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT8:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<uint8_t>((uint8_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<uint8_t, float>((uint8_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<uint8_t, double>((uint8_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<uint8_t, int8_t>((uint8_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<uint8_t, int16_t>((uint8_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<uint8_t, int32_t>((uint8_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<uint8_t, int64_t>((uint8_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<uint8_t, uint8_t>((uint8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<uint8_t, uint16_t>((uint8_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<uint8_t, uint32_t>((uint8_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<uint8_t, uint64_t>((uint8_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint8_t, bool>((uint8_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<uint8_t>((uint8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT16:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<uint16_t>((uint16_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<uint16_t, float>((uint16_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<uint16_t, double>((uint16_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<uint16_t, int8_t>((uint16_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<uint16_t, int16_t>((uint16_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<uint16_t, int32_t>((uint16_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<uint16_t, int64_t>((uint16_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<uint16_t, uint8_t>((uint16_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<uint16_t, uint16_t>((uint16_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<uint16_t, uint32_t>((uint16_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<uint16_t, uint64_t>((uint16_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint16_t, bool>((uint16_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<uint16_t>((uint16_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT32:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<uint32_t>((uint32_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<uint32_t, float>((uint32_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<uint32_t, double>((uint32_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<uint32_t, int8_t>((uint32_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<uint32_t, int16_t>((uint32_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<uint32_t, int32_t>((uint32_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<uint32_t, int64_t>((uint32_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<uint32_t, uint8_t>((uint32_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<uint32_t, uint16_t>((uint32_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<uint32_t, uint32_t>((uint32_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<uint32_t, uint64_t>((uint32_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint32_t, bool>((uint32_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<uint32_t>((uint32_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT64:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<uint64_t>((uint64_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<uint64_t, float>((uint64_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<uint64_t, double>((uint64_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<uint64_t, int8_t>((uint64_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<uint64_t, int16_t>((uint64_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<uint64_t, int32_t>((uint64_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<uint64_t, int64_t>((uint64_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<uint64_t, uint8_t>((uint64_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<uint64_t, uint16_t>((uint64_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<uint64_t, uint32_t>((uint64_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<uint64_t, uint64_t>((uint64_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<uint64_t, bool>((uint64_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<uint64_t>((uint64_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::BOOLEAN:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromIntegralToHalfImpl<bool>((bool *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeImpl<bool, float>((bool *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeImpl<bool, double>((bool *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeImpl<bool, int8_t>((bool *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<bool, int16_t>((bool *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<bool, int32_t>((bool *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<bool, int64_t>((bool *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<bool, uint8_t>((bool *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<bool, uint16_t>((bool *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<bool, uint32_t>((bool *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<bool, uint64_t>((bool *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<bool, bool>((bool *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<bool>((bool *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::PACKED_BOOLEAN:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    gpuConvertTypeFromPackedBooleanToHalfImpl((uint8_t *)source_d, (half *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP32:
                    gpuConvertTypeFromPackedBooleanImpl<float>((uint8_t *)source_d, (float *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::FP64:
                    gpuConvertTypeFromPackedBooleanImpl<double>((uint8_t *)source_d, (double *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT8:
                    gpuConvertTypeFromPackedBooleanImpl<int8_t>((uint8_t *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeFromPackedBooleanImpl<int16_t>((uint8_t *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeFromPackedBooleanImpl<int32_t>((uint8_t *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeFromPackedBooleanImpl<int64_t>((uint8_t *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeFromPackedBooleanImpl<uint8_t>((uint8_t *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeFromPackedBooleanImpl<uint16_t>((uint8_t *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeFromPackedBooleanImpl<uint32_t>((uint8_t *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeFromPackedBooleanImpl<uint64_t>((uint8_t *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeFromPackedBooleanImpl<bool>((uint8_t *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    assert(false);
                default:
                    assert(false);
            }
            break;
        default:
            assert(false);
    }
}

// To Smaller
template <typename FROM_TYPE, typename TO_TYPE>
void TypeConverter::convertToSmallerElementsInPlaceOnGpu(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert(numElements > 0);
    if (numElements == 0)
        return;

    // First, make some empty space in the front by converting some number of the front-most elements to smaller elements
    long availableBytes = 0;
    long chunkSize = 8 * 256;

    launchReadConvertSyncWriteKernel<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElements < chunkSize ? numElements : chunkSize, stream);

    long numElementsLeft = numElements - chunkSize;
    long startingElement = chunkSize;
    availableBytes = (sizeof(FROM_TYPE) - sizeof(TO_TYPE)) * chunkSize;

    // Then convert elements into the empty space, thereby freeing up more empty space, and repeat untill all elements are converted.
    while (numElementsLeft > 0) {
        chunkSize = availableBytes / sizeof(TO_TYPE);

        launchOutOfPlaceConvertKernel<FROM_TYPE, TO_TYPE>(
            source_d + startingElement, dest_d + startingElement, chunkSize < numElementsLeft ? chunkSize : numElementsLeft, stream);

        numElementsLeft -= chunkSize;
        startingElement += chunkSize;
        availableBytes += (sizeof(FROM_TYPE) - sizeof(TO_TYPE)) * chunkSize;
    }
}

template <typename TO_TYPE>
void TypeConverter::convertToSmallerElementsInPlaceOnGpu_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert((is_integral<TO_TYPE>::value));

    assert(numElements > 0);
    if (numElements == 0)
        return;

    // First, make some empty space in the front by converting some number of the front-most elements to smaller elements
    long availableBytes = 0;
    long chunkSize = 8 * 256;

    launchReadConvertSyncWriteKernel_halfToIntegral<TO_TYPE>(source_d, dest_d, numElements < chunkSize ? numElements : chunkSize, stream);

    long numElementsLeft = numElements - chunkSize;
    long startingElement = chunkSize;
    availableBytes = (sizeof(half) - sizeof(TO_TYPE)) * chunkSize;

    // Then convert elements into the empty space, thereby freeing up more empty space, and repeat untill all elements are converted.
    while (numElementsLeft > 0) {
        chunkSize = availableBytes / sizeof(TO_TYPE);

        launchOutOfPlaceConvertKernel_halfToIntegral<TO_TYPE>(
            source_d + startingElement, dest_d + startingElement, chunkSize < numElementsLeft ? chunkSize : numElementsLeft, stream);

        numElementsLeft -= chunkSize;
        startingElement += chunkSize;
        availableBytes += (sizeof(half) - sizeof(TO_TYPE)) * chunkSize;
    }
}

template <typename FROM_TYPE>
void TypeConverter::convertToSmallerElementsInPlaceOnGpu_integralToHalf(FROM_TYPE *source_d,
                                                                        half *dest_d,
                                                                        long numElements,
                                                                        Stream stream) {
    assert((is_integral<FROM_TYPE>::value));

    assert(numElements > 0);
    if (numElements == 0)
        return;

    // First, make some empty space in the front by converting some number of the front-most elements to smaller elements
    long availableBytes = 0;
    long chunkSize = 8 * 256;

    launchReadConvertSyncWriteKernel_integralToHalf<FROM_TYPE>(source_d, dest_d, numElements < chunkSize ? numElements : chunkSize, stream);

    long numElementsLeft = numElements - chunkSize;
    long startingElement = chunkSize;
    availableBytes = (sizeof(FROM_TYPE) - sizeof(half)) * chunkSize;

    // Then convert elements into the empty space, thereby freeing up more empty space, and repeat untill all elements are converted.
    while (numElementsLeft > 0) {
        chunkSize = availableBytes / sizeof(half);

        launchOutOfPlaceConvertKernel_integralToHalf<FROM_TYPE>(
            source_d + startingElement, dest_d + startingElement, chunkSize < numElementsLeft ? chunkSize : numElementsLeft, stream);

        numElementsLeft -= chunkSize;
        startingElement += chunkSize;
        availableBytes += (sizeof(FROM_TYPE) - sizeof(half)) * chunkSize;
    }
}

template <typename FROM_TYPE>
void TypeConverter::convertToSmallerElementsInPlaceOnGpu_toPackedBoolean(FROM_TYPE *source_d,
                                                                         uint8_t *dest_d,
                                                                         long numElements,
                                                                         Stream stream) {
    assert(numElements > 0);
    if (numElements == 0)
        return;

    // First, make some empty space in the front by converting some number of the front-most elements to smaller elements
    long availableBytes = 0;
    long chunkSize = 8 * 256;

    launchReadConvertSyncWriteKernel_toPackedBoolean<FROM_TYPE>(
        source_d, dest_d, numElements < chunkSize ? numElements : chunkSize, stream);

    long numElementsLeft = numElements - chunkSize;
    long startingElement = chunkSize;
    availableBytes =
        (sizeof(FROM_TYPE) * chunkSize) - TensorDescriptor::getArraySizeInBytes(chunkSize, TensorDescriptor::DataType::PACKED_BOOLEAN);

    // Then convert elements into the empty space, thereby freeing up more empty space, and repeat untill all elements are converted.
    while (numElementsLeft > 0) {
        chunkSize = availableBytes * 8;
        if (chunkSize > numElementsLeft)
            chunkSize = numElementsLeft;

        launchOutOfPlaceConvertKernel_toPackedBoolean<FROM_TYPE>(
            source_d + startingElement, dest_d + startingElement / 8, chunkSize, stream);

        numElementsLeft -= chunkSize;
        startingElement += chunkSize;
        availableBytes +=
            (sizeof(FROM_TYPE) * chunkSize) - TensorDescriptor::getArraySizeInBytes(chunkSize, TensorDescriptor::DataType::PACKED_BOOLEAN);
    }
}

void TypeConverter::convertToSmallerElementsInPlaceOnGpu_halfToPackedBoolean(half *source_d,
                                                                             uint8_t *dest_d,
                                                                             long numElements,
                                                                             Stream stream) {
    assert(numElements > 0);
    if (numElements == 0)
        return;

    // First, make some empty space in the front by converting some number of the front-most elements to smaller elements
    long availableBytes = 0;
    long chunkSize = 8 * 256;

    launchReadConvertSyncWriteKernel_halfToPackedBoolean(source_d, dest_d, numElements < chunkSize ? numElements : chunkSize, stream);

    long numElementsLeft = numElements - chunkSize;
    long startingElement = chunkSize;
    availableBytes =
        (sizeof(half) * chunkSize) - TensorDescriptor::getArraySizeInBytes(chunkSize, TensorDescriptor::DataType::PACKED_BOOLEAN);

    // Then convert elements into the empty space, thereby freeing up more empty space, and repeat untill all elements are converted.
    while (numElementsLeft > 0) {
        chunkSize = availableBytes * 8;
        if (chunkSize > numElementsLeft)
            chunkSize = numElementsLeft;

        launchOutOfPlaceConvertKernel_halfToPackedBoolean(source_d + startingElement, dest_d + startingElement / 8, chunkSize, stream);

        numElementsLeft -= chunkSize;
        startingElement += chunkSize;
        availableBytes +=
            (sizeof(half) * chunkSize) - TensorDescriptor::getArraySizeInBytes(chunkSize, TensorDescriptor::DataType::PACKED_BOOLEAN);
    }
}

// To Bigger
template <typename FROM_TYPE, typename TO_TYPE>
void TypeConverter::convertToBiggerElementsInPlaceOnGpu(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert(numElements >= 0);
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

        launchOutOfPlaceConvertKernel<FROM_TYPE, TO_TYPE>(source_d + startingElement, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes += (sizeof(FROM_TYPE) - sizeof(TO_TYPE)) * chunkSize;
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElementsLeft, stream);
    }
}

template <typename TO_TYPE>
void TypeConverter::convertToBiggerElementsInPlaceOnGpu_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert((is_integral<TO_TYPE>::value));

    assert(numElements >= 0);
    if (numElements == 0)
        return;

    // I have empty space in the allocated memory since I have the smaller element occupying enough memory to hold the bigger elements
    // So convert some number of the trailing elements and put them in the back of the memory space, the trick here is that my writes cannot
    // overlap my reads, so I choose a number of elements such that there is enough empty space existing that I can write all converted
    // elements to it, after this the elements I just read and converted become empty space, so repeat until all elements are converted

    long numElementsLeft = numElements;
    long numEmptyBytes = (sizeof(TO_TYPE) - sizeof(half)) * numElements;

    while (numElementsLeft > 8 * 256) {
        long chunkSize = numEmptyBytes / sizeof(TO_TYPE);
        long startingElement = numElementsLeft - chunkSize;

        launchOutOfPlaceConvertKernel_halfToIntegral<TO_TYPE>(source_d + startingElement, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes += (sizeof(half) - sizeof(TO_TYPE)) * chunkSize;
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel_halfToIntegral<TO_TYPE>(source_d, dest_d, numElementsLeft, stream);
    }
}

template <typename FROM_TYPE>
void TypeConverter::convertToBiggerElementsInPlaceOnGpu_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements, Stream stream) {
    assert((is_integral<FROM_TYPE>::value));

    assert(numElements >= 0);
    if (numElements == 0)
        return;

    // I have empty space in the allocated memory since I have the smaller element occupying enough memory to hold the bigger elements
    // So convert some number of the trailing elements and put them in the back of the memory space, the trick here is that my writes cannot
    // overlap my reads, so I choose a number of elements such that there is enough empty space existing that I can write all converted
    // elements to it, after this the elements I just read and converted become empty space, so repeat until all elements are converted

    long numElementsLeft = numElements;
    long numEmptyBytes = (sizeof(half) - sizeof(FROM_TYPE)) * numElements;

    while (numElementsLeft > 8 * 256) {
        long chunkSize = numEmptyBytes / sizeof(half);
        long startingElement = numElementsLeft - chunkSize;

        launchOutOfPlaceConvertKernel_integralToHalf<FROM_TYPE>(source_d + startingElement, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes += (sizeof(FROM_TYPE) - sizeof(half)) * chunkSize;
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel_integralToHalf<FROM_TYPE>(source_d, dest_d, numElementsLeft, stream);
    }
}

template <typename TO_TYPE>
void TypeConverter::convertToBiggerElementsInPlaceOnGpu_fromPackedBoolean(uint8_t *source_d,
                                                                          TO_TYPE *dest_d,
                                                                          long numElements,
                                                                          Stream stream) {
    assert(numElements >= 0);
    if (numElements == 0)
        return;

    // I have empty space in the allocated memory since I have the smaller element occupying enough memory to hold the bigger elements
    // So convert some number of the trailing elements and put them in the back of the memory space, the trick here is that my writes cannot
    // overlap my reads, so I choose a number of elements such that there is enough empty space existing that I can write all converted
    // elements to it, after this the elements I just read and converted become empty space, so repeat until all elements are converted

    long numElementsLeft = numElements;
    long numEmptyBytes =
        (sizeof(TO_TYPE) * numElements) - TensorDescriptor::getArraySizeInBytes(numElements, TensorDescriptor::DataType::PACKED_BOOLEAN);

    while (numElementsLeft > 8 * 256) {
        long chunkSize = numEmptyBytes / sizeof(TO_TYPE);
        long startingElement = numElementsLeft - chunkSize;
        startingElement = 8 * ((startingElement + 7) / 8);
        chunkSize = numElementsLeft - startingElement;

        launchOutOfPlaceConvertKernel_fromPackedBoolean<TO_TYPE>(
            source_d + startingElement / 8, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes +=
            TensorDescriptor::getArraySizeInBytes(chunkSize, TensorDescriptor::DataType::PACKED_BOOLEAN) - (sizeof(TO_TYPE) * chunkSize);
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel_fromPackedBoolean<TO_TYPE>(source_d, dest_d, numElementsLeft, stream);
    }
}

void TypeConverter::convertToBiggerElementsInPlaceOnGpu_packedBooleanToHalf(uint8_t *source_d,
                                                                            half *dest_d,
                                                                            long numElements,
                                                                            Stream stream) {
    assert(numElements >= 0);
    if (numElements == 0)
        return;

    // I have empty space in the allocated memory since I have the smaller element occupying enough memory to hold the bigger elements
    // So convert some number of the trailing elements and put them in the back of the memory space, the trick here is that my writes cannot
    // overlap my reads, so I choose a number of elements such that there is enough empty space existing that I can write all converted
    // elements to it, after this the elements I just read and converted become empty space, so repeat until all elements are converted

    long numElementsLeft = numElements;
    long numEmptyBytes =
        (sizeof(half) * numElements) - TensorDescriptor::getArraySizeInBytes(numElements, TensorDescriptor::DataType::PACKED_BOOLEAN);

    while (numElementsLeft > 8 * 256) {
        long chunkSize = numEmptyBytes / sizeof(half);
        long startingElement = numElementsLeft - chunkSize;
        startingElement = 8 * ((startingElement + 7) / 8);
        chunkSize = numElementsLeft - startingElement;

        launchOutOfPlaceConvertKernel_packedBooleanToHalf(source_d + startingElement / 8, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes +=
            TensorDescriptor::getArraySizeInBytes(chunkSize, TensorDescriptor::DataType::PACKED_BOOLEAN) - (sizeof(half) * chunkSize);
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel_packedBooleanToHalf(source_d, dest_d, numElementsLeft, stream);
    }
}

template <typename FROM_TYPE, typename TO_TYPE>
void TypeConverter::gpuConvertTypeImpl(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert(!(is_same<FROM_TYPE, TO_TYPE>::value));
    assert((is_convertible<FROM_TYPE, TO_TYPE>::value));

    assert(numElements >= 0);
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
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion || sizeof(FROM_TYPE) == sizeof(TO_TYPE)) {
        launchOutOfPlaceConvertKernel<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElements, stream);
    } else {
        if (sizeof(FROM_TYPE) > sizeof(TO_TYPE))
            convertToSmallerElementsInPlaceOnGpu<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElements, stream);
        else
            convertToBiggerElementsInPlaceOnGpu<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElements, stream);
    }
}

template <typename FROM_TYPE>
void TypeConverter::gpuConvertTypeFromIntegralToHalfImpl(FROM_TYPE *source_d, half *dest_d, long numElements, Stream stream) {
    assert(!(is_same<FROM_TYPE, half>::value));

    assert(numElements >= 0);
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
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion || sizeof(FROM_TYPE) == sizeof(half)) {
        launchOutOfPlaceConvertKernel_integralToHalf<FROM_TYPE>(source_d, dest_d, numElements, stream);
    } else {
        if (sizeof(FROM_TYPE) > sizeof(half))
            convertToSmallerElementsInPlaceOnGpu_integralToHalf<FROM_TYPE>(source_d, dest_d, numElements, stream);
        else
            convertToBiggerElementsInPlaceOnGpu_integralToHalf<FROM_TYPE>(source_d, dest_d, numElements, stream);
    }
}

template <typename TO_TYPE>
void TypeConverter::gpuConvertTypeFromHalfToIntegralImpl(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert(!(is_same<half, TO_TYPE>::value));

    assert(numElements >= 0);
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
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion || sizeof(half) == sizeof(TO_TYPE)) {
        launchOutOfPlaceConvertKernel_halfToIntegral<TO_TYPE>(source_d, dest_d, numElements, stream);
    } else {
        if (sizeof(half) > sizeof(TO_TYPE))
            convertToSmallerElementsInPlaceOnGpu_halfToIntegral<TO_TYPE>(source_d, dest_d, numElements, stream);
        else
            convertToBiggerElementsInPlaceOnGpu_halfToIntegral<TO_TYPE>(source_d, dest_d, numElements, stream);
    }
}

template <typename FROM_TYPE>
void TypeConverter::gpuConvertTypeToPackedBooleanImpl(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    assert((is_convertible<FROM_TYPE, bool>::value));

    assert(numElements >= 0);
    if (numElements == 0)
        return;

    ScopedGpu scopedGpu(stream.getGpuNum());

    bool inPlaceConversion = ((void *)source_d == (void *)dest_d);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source_d;
        void *sourceEnd = (void *)&(source_d[numElements - 1]);
        void *destStart = dest_d;
        void *destEnd = (void *)&(dest_d[((numElements + 7) / 8) - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion) {
        launchOutOfPlaceConvertKernel_toPackedBoolean<FROM_TYPE>(source_d, dest_d, numElements, stream);
    } else {
        convertToSmallerElementsInPlaceOnGpu_toPackedBoolean<FROM_TYPE>(source_d, dest_d, numElements, stream);
    }
}

void TypeConverter::gpuConvertTypeFromHalfToPackedBooleanImpl(half *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    assert(numElements >= 0);
    if (numElements == 0)
        return;

    ScopedGpu scopedGpu(stream.getGpuNum());

    bool inPlaceConversion = ((void *)source_d == (void *)dest_d);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source_d;
        void *sourceEnd = (void *)&(source_d[numElements - 1]);
        void *destStart = dest_d;
        void *destEnd = (void *)&(dest_d[((numElements + 7) / 8) - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion) {
        launchOutOfPlaceConvertKernel_halfToPackedBoolean(source_d, dest_d, numElements, stream);
    } else {
        convertToSmallerElementsInPlaceOnGpu_halfToPackedBoolean(source_d, dest_d, numElements, stream);
    }
}

template <typename TO_TYPE>
void TypeConverter::gpuConvertTypeFromPackedBooleanImpl(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    assert((is_convertible<bool, TO_TYPE>::value));

    assert(numElements >= 0);
    if (numElements == 0)
        return;

    ScopedGpu scopedGpu(stream.getGpuNum());

    bool inPlaceConversion = ((void *)source_d == (void *)dest_d);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source_d;
        void *sourceEnd = (void *)&(source_d[((numElements + 7) / 8) - 1]);
        void *destStart = dest_d;
        void *destEnd = (void *)&(dest_d[numElements - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion) {
        launchOutOfPlaceConvertKernel_fromPackedBoolean<TO_TYPE>(source_d, dest_d, numElements, stream);
    } else {
        convertToBiggerElementsInPlaceOnGpu_fromPackedBoolean<TO_TYPE>(source_d, dest_d, numElements, stream);
    }
}

void TypeConverter::gpuConvertTypeFromPackedBooleanToHalfImpl(uint8_t *source_d, half *dest_d, long numElements, Stream stream) {
    assert(numElements >= 0);
    if (numElements == 0)
        return;

    ScopedGpu scopedGpu(stream.getGpuNum());

    bool inPlaceConversion = ((void *)source_d == (void *)dest_d);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source_d;
        void *sourceEnd = (void *)&(source_d[((numElements + 7) / 8) - 1]);
        void *destStart = dest_d;
        void *destEnd = (void *)&(dest_d[numElements - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    // All conversions to and from all element sizes of 1, 2, 4, and 8 bytes are supported.
    // Also, PACKED_BOOLEAN is supported
    // InPlace and out of place conversions are supported in all cases
    if (!inPlaceConversion) {
        launchOutOfPlaceConvertKernel_packedBooleanToHalf(source_d, dest_d, numElements, stream);
    } else {
        convertToBiggerElementsInPlaceOnGpu_packedBooleanToHalf(source_d, dest_d, numElements, stream);
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
            dest_d[index] = (TO_TYPE)(buffer_shared[i * 256 + threadIdx.x]);
    }
}

// Reads 8 writes 8
template <typename TO_TYPE>
__global__ void convertReadWholeChunkThenWriteWholeChunkKernel_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements) {
    __shared__ half buffer_shared[8 * 256];

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index < numElements)
            buffer_shared[i * 256 + threadIdx.x] = source_d[index];

        buffer_shared[i * 256 + threadIdx.x] = source_d[index];
    }

    __syncthreads();

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index < numElements)
            dest_d[index] = (TO_TYPE)(float)(buffer_shared[i * 256 + threadIdx.x]);
    }
}

// Reads 8 writes 8
template <typename FROM_TYPE>
__global__ void convertReadWholeChunkThenWriteWholeChunkKernel_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements) {
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
            dest_d[index] = (half)(float)(buffer_shared[i * 256 + threadIdx.x]);
    }
}

// Reads 8 writes 1
template <typename FROM_TYPE>
__global__ void convertReadWholeChunkThenWriteWholeChunkKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements) {
    __shared__ FROM_TYPE buffer_shared[8 * 256];

    long blockElementChunkStart = blockIdx.x * 256 * 8;
    long threadInputElementIndex = threadIdx.x + blockElementChunkStart;
    long threadOutputElementChunkStart = blockElementChunkStart + threadIdx.x * 8;

    // So this thread has at least one element of an 8 element packed boolean uint8_t
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        if (threadInputElementIndex < numElements)
            buffer_shared[threadIdx.x + i * 256] = source_d[threadInputElementIndex + i * 256];
    }

    __syncthreads();

    uint8_t toBuff = 0;
    for (int j = 0; j < 8; ++j) {
        if (threadOutputElementChunkStart + j < numElements) {
            FROM_TYPE fromBuff = buffer_shared[threadIdx.x * 8 + j];
            bool toBuffRaw = (bool)fromBuff;
            toBuff |= (toBuffRaw) << j;
        }
    }
    if (threadOutputElementChunkStart < numElements) {
        long destMemIndex = threadOutputElementChunkStart / 8;
        dest_d[destMemIndex] = toBuff;
    }
}

// Reads 8 writes 1
__global__ void convertReadWholeChunkThenWriteWholeChunkKernel_halfToPackedBoolean(half *source_d, uint8_t *dest_d, long numElements) {
    __shared__ half buffer_shared[8 * 256];

    long blockElementChunkStart = blockIdx.x * 256 * 8;
    long threadInputElementIndex = threadIdx.x + blockElementChunkStart;
    long threadOutputElementChunkStart = blockElementChunkStart + threadIdx.x * 8;

    // So this thread has at least one element of an 8 element packed boolean uint8_t
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        if (threadInputElementIndex < numElements)
            buffer_shared[threadIdx.x + i * 256] = source_d[threadInputElementIndex + i * 256];
    }

    __syncthreads();

    uint8_t toBuff = 0;
    for (int j = 0; j < 8; ++j) {
        if (threadOutputElementChunkStart + j < numElements) {
            half fromBuff = buffer_shared[threadIdx.x * 8 + j];
            bool toBuffRaw = (bool)(float)fromBuff;
            toBuff |= (toBuffRaw) << j;
        }
    }
    if (threadOutputElementChunkStart < numElements) {
        long destMemIndex = threadOutputElementChunkStart / 8;
        dest_d[destMemIndex] = toBuff;
    }
}

// Reads 1 writes 8
template <typename TO_TYPE>
__global__ void convertReadWholeChunkThenWriteWholeChunkKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements) {
    __shared__ uint8_t buffer_shared[256];

    long blockElementChunkStart = blockIdx.x * 256 * 8;
    long threadInputElementIndex = blockElementChunkStart + threadIdx.x * 8;
    long threadOutputElementIndex = blockElementChunkStart + threadIdx.x;

    if (threadInputElementIndex < numElements)
        buffer_shared[threadIdx.x] = source_d[threadInputElementIndex / 8];
    __syncthreads();

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        uint8_t buffRaw = buffer_shared[(threadIdx.x + i * (256)) / 8];
        TO_TYPE buff = (TO_TYPE)((buffRaw >> (threadIdx.x % 8)) & 0x1);
        if (threadOutputElementIndex + i * 256 < numElements)
            dest_d[threadOutputElementIndex + i * 256] = buff;
    }
}

__global__ void convertReadWholeChunkThenWriteWholeChunkKernel_packedBooleanToHalf(uint8_t *source_d, half *dest_d, long numElements) {
    __shared__ uint8_t buffer_shared[256];

    long blockElementChunkStart = blockIdx.x * 256 * 8;
    long threadInputElementIndex = blockElementChunkStart + threadIdx.x * 8;
    long threadOutputElementIndex = blockElementChunkStart + threadIdx.x;

    if (threadInputElementIndex < numElements)
        buffer_shared[threadIdx.x] = source_d[threadInputElementIndex / 8];
    __syncthreads();

#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        uint8_t buffRaw = buffer_shared[(threadIdx.x + i * (256)) / 8];
        half buff = (half)(float)((buffRaw >> (threadIdx.x % 8)) & 0x1);
        if (threadOutputElementIndex + i * 256 < numElements)
            dest_d[threadOutputElementIndex + i * 256] = buff;
    }
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index >= numElements)
            return;

        dest_d[index] = (TO_TYPE)(source_d[index]);
    }
}

// Reads 8 writes 8
template <typename TO_TYPE>
__global__ void convertOutOfPlaceKernel_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements) {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index >= numElements)
            return;

        dest_d[index] = (TO_TYPE)(float)(source_d[index]);
    }
}

// Reads 8 writes 8
template <typename FROM_TYPE>
__global__ void convertOutOfPlaceKernel_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements) {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index >= numElements)
            return;

        dest_d[index] = (half)(float)(source_d[index]);
    }
}

/*
// Reads 8 writes 1
template <typename FROM_TYPE>
__global__ void convertOutOfPlaceKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements) {
    __shared__ FROM_TYPE buffer_shared[256];

    long blockElementChunkStart = blockIdx.x * 256 * 8;
    long threadInputElementIndex = threadIdx.x + blockElementChunkStart;
    long threadOutputElementChunkStart = blockElementChunkStart + threadIdx.x * 8;

    // So this thread has at least one element of an 8 element packed boolean uint8_t
    uint8_t toBuff = 0;
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        __syncthreads();
        if (threadInputElementIndex < numElements)
            buffer_shared[threadIdx.x] = source_d[threadInputElementIndex];
        __syncthreads();

        // Now all of these elements belong to 1/8 of the threads.
        // Each of those threads will read its elements out of shared and shift them into the toBuff;
        if (threadIdx.x / (256 / 8) == i) {
#pragma unroll 8
            for (int j = 0; j < 8; ++j) {
                if (threadOutputElementChunkStart + j < numElements) {
                    FROM_TYPE fromBuff = buffer_shared[((threadIdx.x * 8) % 256) + j];
                    bool toBuffRaw = (bool)fromBuff;
                    toBuff |= (toBuffRaw) << j;
                }
            }
        }
        threadInputElementIndex += 256;
    }
    if (threadOutputElementChunkStart < numElements) {
        long destMemIndex = threadOutputElementChunkStart / 8;
        dest_d[destMemIndex] = toBuff;
    }
}
*/

// Reads 8 writes 1
__global__ void convertOutOfPlaceKernel_halfToPackedBoolean(half *source_d, uint8_t *dest_d, long numElements) {
    __shared__ half buffer_shared[256];

    long blockElementChunkStart = blockIdx.x * 256 * 8;
    long threadInputElementIndex = threadIdx.x + blockElementChunkStart;
    long threadOutputElementChunkStart = blockElementChunkStart + threadIdx.x * 8;

    // So this thread has at least one element of an 8 element packed boolean uint8_t
    uint8_t toBuff = 0;
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        __syncthreads();
        if (threadInputElementIndex < numElements)
            buffer_shared[threadIdx.x] = source_d[threadInputElementIndex];
        __syncthreads();

        // Now all of these elements belong to 1/8 of the threads.
        // Each of those threads will read its elements out of shared and shift them into the toBuff;
        if (threadIdx.x / (256 / 8) == i) {
#pragma unroll 8
            for (int j = 0; j < 8; ++j) {
                if (threadOutputElementChunkStart + j < numElements) {
                    half fromBuff = buffer_shared[((threadIdx.x * 8) % 256) + j];
                    bool toBuffRaw = (bool)(float)fromBuff;
                    toBuff |= (toBuffRaw) << j;
                }
            }
        }
        threadInputElementIndex += 256;
    }
    if (threadOutputElementChunkStart < numElements) {
        long destMemIndex = threadOutputElementChunkStart / 8;
        dest_d[destMemIndex] = toBuff;
    }
}

template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    convertOutOfPlaceKernel<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename TO_TYPE>
void launchOutOfPlaceConvertKernel_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    convertOutOfPlaceKernel_halfToIntegral<TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE>
void launchOutOfPlaceConvertKernel_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    convertOutOfPlaceKernel_integralToHalf<FROM_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE>
void launchOutOfPlaceConvertKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    // This one needs syncing anyway so it the synced kernel is used here.
    convertReadWholeChunkThenWriteWholeChunkKernel_toPackedBoolean<FROM_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

void launchOutOfPlaceConvertKernel_halfToPackedBoolean(half *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    // This one needs syncing anyway so it the synced kernel is used here.
    convertReadWholeChunkThenWriteWholeChunkKernel_halfToPackedBoolean<<<gridSize, blockSize, 0, stream.getStream()>>>(
        source_d, dest_d, numElements);
}

template <typename TO_TYPE>
void launchOutOfPlaceConvertKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    // This one needs syncing anyway so it the synced kernel is used here.
    convertReadWholeChunkThenWriteWholeChunkKernel_fromPackedBoolean<TO_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

void launchOutOfPlaceConvertKernel_packedBooleanToHalf(uint8_t *source_d, half *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    // This one needs syncing anyway so it the synced kernel is used here.
    convertReadWholeChunkThenWriteWholeChunkKernel_packedBooleanToHalf<<<gridSize, blockSize, 0, stream.getStream()>>>(
        source_d, dest_d, numElements);
}

template <typename FROM_TYPE, typename TO_TYPE>
void launchReadConvertSyncWriteKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel<FROM_TYPE, TO_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename TO_TYPE>
void launchReadConvertSyncWriteKernel_halfToIntegral(half *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel_halfToIntegral<TO_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE>
void launchReadConvertSyncWriteKernel_integralToHalf(FROM_TYPE *source_d, half *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel_integralToHalf<FROM_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE>
void launchReadConvertSyncWriteKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel_toPackedBoolean<FROM_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

void launchReadConvertSyncWriteKernel_halfToPackedBoolean(half *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel_halfToPackedBoolean<<<gridSize, blockSize, 0, stream.getStream()>>>(
        source_d, dest_d, numElements);
}

template <typename TO_TYPE>
void launchReadConvertSyncWriteKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel_fromPackedBoolean<TO_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

void launchReadConvertSyncWriteKernel_packedBooleanToHalf(uint8_t *source_d, half *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel_packedBooleanToHalf<<<gridSize, blockSize, 0, stream.getStream()>>>(
        source_d, dest_d, numElements);
}
