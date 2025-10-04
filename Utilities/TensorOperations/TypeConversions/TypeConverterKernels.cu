#include "TypeConverter.h"

#include <stdio.h>

using namespace ThorImplementation;
using namespace std;

// FIXME: get rid of in-place variants - can't free the back half of a tensor, there is no use for that and it is inefficient.

// Launch out-of-place kernels:
template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
template <typename FROM_TYPE>
void launchOutOfPlaceConvertKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream);
template <typename TO_TYPE>
void launchOutOfPlaceConvertKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream);

// Launch in-place kernels:
template <typename FROM_TYPE, typename TO_TYPE>
void launchReadConvertSyncWriteKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
template <typename FROM_TYPE>
void launchReadConvertSyncWriteKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream);
template <typename TO_TYPE>
void launchReadConvertSyncWriteKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream);

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
                    gpuConvertTypeImpl<half, int8_t>((half *)source_d, (int8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT16:
                    gpuConvertTypeImpl<half, int16_t>((half *)source_d, (int16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT32:
                    gpuConvertTypeImpl<half, int32_t>((half *)source_d, (int32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::INT64:
                    gpuConvertTypeImpl<half, int64_t>((half *)source_d, (int64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    gpuConvertTypeImpl<half, uint8_t>((half *)source_d, (uint8_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    gpuConvertTypeImpl<half, uint16_t>((half *)source_d, (uint16_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    gpuConvertTypeImpl<half, uint32_t>((half *)source_d, (uint32_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    gpuConvertTypeImpl<half, uint64_t>((half *)source_d, (uint64_t *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    gpuConvertTypeImpl<half, bool>((half *)source_d, (bool *)dest_d, numElements, stream);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    gpuConvertTypeToPackedBooleanImpl<half>((half *)source_d, (uint8_t *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<int8_t, half>((int8_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<int16_t, half>((int16_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<int32_t, half>((int32_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<int64_t, half>((int64_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<uint8_t, half>((uint8_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<uint16_t, half>((uint16_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<uint32_t, half>((uint32_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<uint64_t, half>((uint64_t *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeImpl<bool, half>((bool *)source_d, (half *)dest_d, numElements, stream);
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
                    gpuConvertTypeFromPackedBooleanImpl((uint8_t *)source_d, (half *)dest_d, numElements, stream);
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

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelNoOvershoot(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        long index = threadIdx.x + blockIdx.x * 256 * 8 + i * 256;
        if (index >= numElements)
            return;
        dest_d[index] = (TO_TYPE)(source_d[index]);
    }
}

template <typename FROM_TYPE, typename TO_TYPE>
void launchOutOfPlaceConvertKernelNoOvershoot(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
    convertOutOfPlaceKernelNoOvershoot<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
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

        launchOutOfPlaceConvertKernelNoOvershoot<FROM_TYPE, TO_TYPE>(
            source_d + startingElement, dest_d + startingElement, chunkSize < numElementsLeft ? chunkSize : numElementsLeft, stream);

        numElementsLeft -= chunkSize;
        startingElement += chunkSize;
        availableBytes += (sizeof(FROM_TYPE) - sizeof(TO_TYPE)) * chunkSize;
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

        launchOutOfPlaceConvertKernelNoOvershoot<FROM_TYPE, TO_TYPE>(
            source_d + startingElement, dest_d + startingElement, chunkSize, stream);

        numEmptyBytes += (sizeof(FROM_TYPE) - sizeof(TO_TYPE)) * chunkSize;
        numElementsLeft = startingElement;
    }

    if (numElementsLeft > 0) {
        launchReadConvertSyncWriteKernel<FROM_TYPE, TO_TYPE>(source_d, dest_d, numElementsLeft, stream);
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

// Reads 16 writes 16
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS1D1(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 16;
    if (index >= numElements)
        return;

    char inBuff[16];
    long index16Elements = index >> 4;
    ((float4 *)inBuff)[0] = ((float4 *)source_d)[index16Elements];

    char outBuff[16];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];
    ((TO_TYPE *)outBuff)[8] = (TO_TYPE)((FROM_TYPE *)inBuff)[8];
    ((TO_TYPE *)outBuff)[9] = (TO_TYPE)((FROM_TYPE *)inBuff)[9];
    ((TO_TYPE *)outBuff)[10] = (TO_TYPE)((FROM_TYPE *)inBuff)[10];
    ((TO_TYPE *)outBuff)[11] = (TO_TYPE)((FROM_TYPE *)inBuff)[11];
    ((TO_TYPE *)outBuff)[12] = (TO_TYPE)((FROM_TYPE *)inBuff)[12];
    ((TO_TYPE *)outBuff)[13] = (TO_TYPE)((FROM_TYPE *)inBuff)[13];
    ((TO_TYPE *)outBuff)[14] = (TO_TYPE)((FROM_TYPE *)inBuff)[14];
    ((TO_TYPE *)outBuff)[15] = (TO_TYPE)((FROM_TYPE *)inBuff)[15];

    ((float4 *)dest_d)[index16Elements] = ((float4 *)outBuff)[0];
}

// Reads 16 writes 16
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS1D2(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 16;
    if (index >= numElements)
        return;

    char inBuff[16];
    long index16Elements = index >> 4;
    ((float4 *)inBuff)[0] = ((float4 *)source_d)[index16Elements];

    half outBuff[16];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];
    ((TO_TYPE *)outBuff)[8] = (TO_TYPE)((FROM_TYPE *)inBuff)[8];
    ((TO_TYPE *)outBuff)[9] = (TO_TYPE)((FROM_TYPE *)inBuff)[9];
    ((TO_TYPE *)outBuff)[10] = (TO_TYPE)((FROM_TYPE *)inBuff)[10];
    ((TO_TYPE *)outBuff)[11] = (TO_TYPE)((FROM_TYPE *)inBuff)[11];
    ((TO_TYPE *)outBuff)[12] = (TO_TYPE)((FROM_TYPE *)inBuff)[12];
    ((TO_TYPE *)outBuff)[13] = (TO_TYPE)((FROM_TYPE *)inBuff)[13];
    ((TO_TYPE *)outBuff)[14] = (TO_TYPE)((FROM_TYPE *)inBuff)[14];
    ((TO_TYPE *)outBuff)[15] = (TO_TYPE)((FROM_TYPE *)inBuff)[15];

    ((double4 *)dest_d)[index16Elements] = ((double4 *)outBuff)[0];
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS1D4(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    if (index >= numElements)
        return;

    char inBuff[8];
    long index8Elements = index >> 3;
    ((float2 *)inBuff)[0] = ((float2 *)source_d)[index8Elements];

    float outBuff[8];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];

    ((double4 *)dest_d)[index8Elements] = ((double4 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS1D8(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    char inBuff[4];
    long index4Elements = index >> 2;
    ((float *)inBuff)[0] = ((float *)source_d)[index4Elements];

    double outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((double4 *)dest_d)[index4Elements] = ((double4 *)outBuff)[0];
}

// Reads 16 writes 16
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS2D1(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 16;
    if (index >= numElements)
        return;

    half inBuff[16];
    long index16Elements = index >> 4;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index16Elements];

    char outBuff[16];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];
    ((TO_TYPE *)outBuff)[8] = (TO_TYPE)((FROM_TYPE *)inBuff)[8];
    ((TO_TYPE *)outBuff)[9] = (TO_TYPE)((FROM_TYPE *)inBuff)[9];
    ((TO_TYPE *)outBuff)[10] = (TO_TYPE)((FROM_TYPE *)inBuff)[10];
    ((TO_TYPE *)outBuff)[11] = (TO_TYPE)((FROM_TYPE *)inBuff)[11];
    ((TO_TYPE *)outBuff)[12] = (TO_TYPE)((FROM_TYPE *)inBuff)[12];
    ((TO_TYPE *)outBuff)[13] = (TO_TYPE)((FROM_TYPE *)inBuff)[13];
    ((TO_TYPE *)outBuff)[14] = (TO_TYPE)((FROM_TYPE *)inBuff)[14];
    ((TO_TYPE *)outBuff)[15] = (TO_TYPE)((FROM_TYPE *)inBuff)[15];

    ((float4 *)dest_d)[index16Elements] = ((float4 *)outBuff)[0];
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS2D2(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    if (index >= numElements)
        return;

    half inBuff[8];
    long index8Elements = index >> 3;
    ((float4 *)inBuff)[0] = ((float4 *)source_d)[index8Elements];

    half outBuff[8];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];

    ((float4 *)dest_d)[index8Elements] = ((float4 *)outBuff)[0];
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS2D4(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    if (index >= numElements)
        return;

    half inBuff[8];
    long index8Elements = index >> 3;
    ((float4 *)inBuff)[0] = ((float4 *)source_d)[index8Elements];

    float outBuff[8];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];

    ((double4 *)dest_d)[index8Elements] = ((double4 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS2D8(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    half inBuff[4];
    long index4Elements = index >> 2;
    ((float2 *)inBuff)[0] = ((float2 *)source_d)[index4Elements];

    double outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((double4 *)dest_d)[index4Elements] = ((double4 *)outBuff)[0];
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS4D1(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    if (index >= numElements)
        return;

    float inBuff[8];
    long index8Elements = index >> 3;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index8Elements];

    char outBuff[8];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];

    ((float2 *)dest_d)[index8Elements] = ((float2 *)outBuff)[0];
}

// Reads 8 writes 8
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS4D2(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    if (index >= numElements)
        return;

    float inBuff[8];
    long index8Elements = index >> 3;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index8Elements];

    half outBuff[8];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];
    ((TO_TYPE *)outBuff)[4] = (TO_TYPE)((FROM_TYPE *)inBuff)[4];
    ((TO_TYPE *)outBuff)[5] = (TO_TYPE)((FROM_TYPE *)inBuff)[5];
    ((TO_TYPE *)outBuff)[6] = (TO_TYPE)((FROM_TYPE *)inBuff)[6];
    ((TO_TYPE *)outBuff)[7] = (TO_TYPE)((FROM_TYPE *)inBuff)[7];

    ((float4 *)dest_d)[index8Elements] = ((float4 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS4D4(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    float inBuff[4];
    long index4Elements = index >> 2;
    ((float4 *)inBuff)[0] = ((float4 *)source_d)[index4Elements];

    float outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((float4 *)dest_d)[index4Elements] = ((float4 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS4D8(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    float inBuff[4];
    long index4Elements = index >> 2;
    ((float4 *)inBuff)[0] = ((float4 *)source_d)[index4Elements];

    double outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((double4 *)dest_d)[index4Elements] = ((double4 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS8D1(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    double inBuff[4];
    long index4Elements = index >> 2;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index4Elements];

    char outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((float *)dest_d)[index4Elements] = ((float *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS8D2(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    double inBuff[4];
    long index4Elements = index >> 2;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index4Elements];

    half outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((float2 *)dest_d)[index4Elements] = ((float2 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS8D4(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    double inBuff[4];
    long index4Elements = index >> 2;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index4Elements];

    float outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((float4 *)dest_d)[index4Elements] = ((float4 *)outBuff)[0];
}

// Reads 4 writes 4
template <typename FROM_TYPE, typename TO_TYPE>
__global__ void convertOutOfPlaceKernelS8D8(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements) {
    long index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index >= numElements)
        return;

    double inBuff[4];
    long index4Elements = index >> 2;
    ((double4 *)inBuff)[0] = ((double4 *)source_d)[index4Elements];

    double outBuff[4];
    ((TO_TYPE *)outBuff)[0] = (TO_TYPE)((FROM_TYPE *)inBuff)[0];
    ((TO_TYPE *)outBuff)[1] = (TO_TYPE)((FROM_TYPE *)inBuff)[1];
    ((TO_TYPE *)outBuff)[2] = (TO_TYPE)((FROM_TYPE *)inBuff)[2];
    ((TO_TYPE *)outBuff)[3] = (TO_TYPE)((FROM_TYPE *)inBuff)[3];

    ((double4 *)dest_d)[index4Elements] = ((double4 *)outBuff)[0];
}

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
    if (sizeof(FROM_TYPE) == 1) {
        if (sizeof(TO_TYPE) == 1) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 16;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS1D1<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 2) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 16;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS1D2<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 4) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 8;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS1D4<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 8) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS1D8<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        }
    } else if (sizeof(FROM_TYPE) == 2) {
        if (sizeof(TO_TYPE) == 1) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 16;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS2D1<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 2) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 8;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS2D2<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 4) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 8;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS2D4<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 8) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS2D8<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        }
    } else if (sizeof(FROM_TYPE) == 4) {
        if (sizeof(TO_TYPE) == 1) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 8;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS4D1<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 2) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 8;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS4D2<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 4) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS4D4<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 8) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS4D8<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        }
    } else if (sizeof(FROM_TYPE) == 8) {
        if (sizeof(TO_TYPE) == 1) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS8D1<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 2) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS8D2<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 4) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS8D4<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        } else if (sizeof(TO_TYPE) == 8) {
            dim3 blockSize(256);
            constexpr int elementsPerBlock = 256 * 4;
            dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);
            convertOutOfPlaceKernelS8D8<FROM_TYPE, TO_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
        }
    } else {
        assert(false);
    }
}

template <typename FROM_TYPE>
void launchOutOfPlaceConvertKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);

    if (is_same<FROM_TYPE, half>::value)
        convertReadWholeChunkThenWriteWholeChunkKernel_halfToPackedBoolean<<<gridSize, blockSize, 0, stream.getStream()>>>(
            (half *)source_d, dest_d, numElements);
    else
        convertReadWholeChunkThenWriteWholeChunkKernel_toPackedBoolean<FROM_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename TO_TYPE>
void launchOutOfPlaceConvertKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    constexpr int elementsPerBlock = 256 * 8;
    dim3 gridSize((numElements + (elementsPerBlock - 1)) / elementsPerBlock);

    if (is_same<TO_TYPE, half>::value)
        convertReadWholeChunkThenWriteWholeChunkKernel_packedBooleanToHalf<<<gridSize, blockSize, 0, stream.getStream()>>>(
            source_d, (half *)dest_d, numElements);
    else
        convertReadWholeChunkThenWriteWholeChunkKernel_fromPackedBoolean<TO_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE, typename TO_TYPE>
void launchReadConvertSyncWriteKernel(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    convertReadWholeChunkThenWriteWholeChunkKernel<FROM_TYPE, TO_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename FROM_TYPE>
void launchReadConvertSyncWriteKernel_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    if (is_same<FROM_TYPE, half>::value)
        convertReadWholeChunkThenWriteWholeChunkKernel_halfToPackedBoolean<<<gridSize, blockSize, 0, stream.getStream()>>>(
            (half *)source_d, dest_d, numElements);
    else
        convertReadWholeChunkThenWriteWholeChunkKernel_toPackedBoolean<FROM_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template <typename TO_TYPE>
void launchReadConvertSyncWriteKernel_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(1);
    if (is_same<TO_TYPE, half>::value)
        convertReadWholeChunkThenWriteWholeChunkKernel_packedBooleanToHalf<<<gridSize, blockSize, 0, stream.getStream()>>>(
            source_d, (half *)dest_d, numElements);
    else
        convertReadWholeChunkThenWriteWholeChunkKernel_fromPackedBoolean<TO_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}
