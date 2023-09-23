#include "TypeConverter.h"

using namespace ThorImplementation;
using namespace std;

void TypeConverter::convertType(void *source,
                                void *dest,
                                TensorDescriptor::DataType sourceDataType,
                                TensorDescriptor::DataType destDataType,
                                long numElements,
                                Stream stream,
                                int deviceNum) {
    // They should not be the same type, if they are then convertType should not have been invoked
    assert(sourceDataType != destDataType);

    assert(numElements >= 0);
    if (numElements == 0)
        return;

    if (deviceNum == -1) {
        // CPU
        cudaError_t cudaStatus;
        Args *args = new Args(source, dest, sourceDataType, destDataType, numElements);
        cudaStatus = cudaLaunchHostFunc(stream.getStream(), cpuConvertType, args);
        assert(cudaStatus == cudaSuccess);
    } else {
        // GPU
        assert(stream.getGpuNum() == deviceNum);
        gpuConvertType(source, dest, sourceDataType, destDataType, numElements, stream);
    }
}

//----------------------------------
//
// CPU
//
//----------------------------------

// Convert on CPU between two types. In place or out of place is supported.
void CUDART_CB TypeConverter::cpuConvertType(void *data) {
    Args *args = (Args *)data;
    // FROM_TYPE *source = args->source;
    // TO_TYPE *dest = args->dest;
    long numElements = args->numElements;
    TensorDescriptor::DataType sourceDataType = args->sourceDataType;
    TensorDescriptor::DataType destDataType = args->destDataType;

    switch (sourceDataType) {
        case TensorDescriptor::DataType::FP16:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeImpl<half, half>((half *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<half, float>((half *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<half, double>((half *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<half, int8_t>((half *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<half, int16_t>((half *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<half, int32_t>((half *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<half, int64_t>((half *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<half, uint8_t>((half *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<half, uint16_t>((half *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<half, uint32_t>((half *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<half, uint64_t>((half *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<half, bool>((half *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<half>((half *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::FP32:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeImpl<float, half>((float *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<float, float>((float *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<float, double>((float *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<float, int8_t>((float *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<float, int16_t>((float *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<float, int32_t>((float *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<float, int64_t>((float *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<float, uint8_t>((float *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<float, uint16_t>((float *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<float, uint32_t>((float *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<float, uint64_t>((float *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<float, bool>((float *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<float>((float *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::FP64:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeImpl<double, half>((double *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<double, float>((double *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<double, double>((double *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<double, int8_t>((double *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<double, int16_t>((double *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<double, int32_t>((double *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<double, int64_t>((double *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<double, uint8_t>((double *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<double, uint16_t>((double *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<double, uint32_t>((double *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<double, uint64_t>((double *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<double, bool>((double *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<double>((double *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT8:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<int8_t>((int8_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<int8_t, float>((int8_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<int8_t, double>((int8_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<int8_t, int8_t>((int8_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<int8_t, int16_t>((int8_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<int8_t, int32_t>((int8_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<int8_t, int64_t>((int8_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<int8_t, uint8_t>((int8_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<int8_t, uint16_t>((int8_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<int8_t, uint32_t>((int8_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<int8_t, uint64_t>((int8_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<int8_t, bool>((int8_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<int8_t>((int8_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT16:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<int16_t>((int16_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<int16_t, float>((int16_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<int16_t, double>((int16_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<int16_t, int8_t>((int16_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<int16_t, int16_t>((int16_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<int16_t, int32_t>((int16_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<int16_t, int64_t>((int16_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<int16_t, uint8_t>((int16_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<int16_t, uint16_t>((int16_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<int16_t, uint32_t>((int16_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<int16_t, uint64_t>((int16_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<int16_t, bool>((int16_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<int16_t>((int16_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT32:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<int32_t>((int32_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<int32_t, float>((int32_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<int32_t, double>((int32_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<int32_t, int8_t>((int32_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<int32_t, int16_t>((int32_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<int32_t, int32_t>((int32_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<int32_t, int64_t>((int32_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<int32_t, uint8_t>((int32_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<int32_t, uint16_t>((int32_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<int32_t, uint32_t>((int32_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<int32_t, uint64_t>((int32_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<int32_t, bool>((int32_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<int32_t>((int32_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::INT64:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<int64_t>((int64_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<int64_t, float>((int64_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<int64_t, double>((int64_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<int64_t, int8_t>((int64_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<int64_t, int16_t>((int64_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<int64_t, int32_t>((int64_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<int64_t, int64_t>((int64_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<int64_t, uint8_t>((int64_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<int64_t, uint16_t>((int64_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<int64_t, uint32_t>((int64_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<int64_t, uint64_t>((int64_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<int64_t, bool>((int64_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<int64_t>((int64_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT8:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<uint8_t>((uint8_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<uint8_t, float>((uint8_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<uint8_t, double>((uint8_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<uint8_t, int8_t>((uint8_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<uint8_t, int16_t>((uint8_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<uint8_t, int32_t>((uint8_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<uint8_t, int64_t>((uint8_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<uint8_t, uint8_t>((uint8_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<uint8_t, uint16_t>((uint8_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<uint8_t, uint32_t>((uint8_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<uint8_t, uint64_t>((uint8_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<uint8_t, bool>((uint8_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<uint8_t>((uint8_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT16:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<uint16_t>((uint16_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<uint16_t, float>((uint16_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<uint16_t, double>((uint16_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<uint16_t, int8_t>((uint16_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<uint16_t, int16_t>((uint16_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<uint16_t, int32_t>((uint16_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<uint16_t, int64_t>((uint16_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<uint16_t, uint8_t>((uint16_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<uint16_t, uint16_t>((uint16_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<uint16_t, uint32_t>((uint16_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<uint16_t, uint64_t>((uint16_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<uint16_t, bool>((uint16_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<uint16_t>((uint16_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT32:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<uint32_t>((uint32_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<uint32_t, float>((uint32_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<uint32_t, double>((uint32_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<uint32_t, int8_t>((uint32_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<uint32_t, int16_t>((uint32_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<uint32_t, int32_t>((uint32_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<uint32_t, int64_t>((uint32_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<uint32_t, uint8_t>((uint32_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<uint32_t, uint16_t>((uint32_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<uint32_t, uint32_t>((uint32_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<uint32_t, uint64_t>((uint32_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<uint32_t, bool>((uint32_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<uint32_t>((uint32_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::UINT64:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<uint64_t>((uint64_t *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<uint64_t, float>((uint64_t *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<uint64_t, double>((uint64_t *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<uint64_t, int8_t>((uint64_t *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<uint64_t, int16_t>((uint64_t *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<uint64_t, int32_t>((uint64_t *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<uint64_t, int64_t>((uint64_t *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<uint64_t, uint8_t>((uint64_t *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<uint64_t, uint16_t>((uint64_t *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<uint64_t, uint32_t>((uint64_t *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<uint64_t, uint64_t>((uint64_t *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<uint64_t, bool>((uint64_t *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<uint64_t>((uint64_t *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::BOOLEAN:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromIntegralToHalfImpl<bool>((bool *)args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeImpl<bool, float>((bool *)args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeImpl<bool, double>((bool *)args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeImpl<bool, int8_t>((bool *)args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeImpl<bool, int16_t>((bool *)args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeImpl<bool, int32_t>((bool *)args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeImpl<bool, int64_t>((bool *)args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeImpl<bool, uint8_t>((bool *)args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeImpl<bool, uint16_t>((bool *)args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeImpl<bool, uint32_t>((bool *)args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeImpl<bool, uint64_t>((bool *)args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeImpl<bool, bool>((bool *)args->source, (bool *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::PACKED_BOOLEAN:
                    cpuConvertTypeToPackedBooleanImpl<bool>((bool *)args->source, args->dest, numElements);
                    break;
                default:
                    assert(false);
            }
            break;
        case TensorDescriptor::DataType::PACKED_BOOLEAN:
            switch (destDataType) {
                case TensorDescriptor::DataType::FP16:
                    cpuConvertTypeFromPackedBooleanToHalfImpl(args->source, (half *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP32:
                    cpuConvertTypeFromPackedBooleanImpl<float>(args->source, (float *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::FP64:
                    cpuConvertTypeFromPackedBooleanImpl<double>(args->source, (double *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT8:
                    cpuConvertTypeFromPackedBooleanImpl<int8_t>(args->source, (int8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT16:
                    cpuConvertTypeFromPackedBooleanImpl<int16_t>(args->source, (int16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT32:
                    cpuConvertTypeFromPackedBooleanImpl<int32_t>(args->source, (int32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::INT64:
                    cpuConvertTypeFromPackedBooleanImpl<int64_t>(args->source, (int64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT8:
                    cpuConvertTypeFromPackedBooleanImpl<uint8_t>(args->source, (uint8_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT16:
                    cpuConvertTypeFromPackedBooleanImpl<uint16_t>(args->source, (uint16_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT32:
                    cpuConvertTypeFromPackedBooleanImpl<uint32_t>(args->source, (uint32_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::UINT64:
                    cpuConvertTypeFromPackedBooleanImpl<uint64_t>(args->source, (uint64_t *)args->dest, numElements);
                    break;
                case TensorDescriptor::DataType::BOOLEAN:
                    cpuConvertTypeFromPackedBooleanImpl<bool>(args->source, (bool *)args->dest, numElements);
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

    delete args;
}

template <typename FROM_TYPE, typename TO_TYPE>
void TypeConverter::cpuConvertTypeImpl(FROM_TYPE *source, TO_TYPE *dest, long numElements) {
    assert(!(is_same<FROM_TYPE, TO_TYPE>::value));
    assert((is_convertible<FROM_TYPE, TO_TYPE>::value));

    assert(numElements >= 0);
    if (numElements == 0)
        return;

    bool inPlaceConversion = ((void *)source == (void *)dest);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source;
        void *sourceEnd = (void *)&(source[numElements - 1]);
        void *destStart = dest;
        void *destEnd = (void *)&(dest[numElements - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    if (!inPlaceConversion || sizeof(TO_TYPE) == sizeof(FROM_TYPE)) {
        // Out of place or in place or conversion to a same sized type
        const uint32_t numProcs = max(min((long)omp_get_num_procs(), numElements / 500000), 1L);
        if (numProcs > 1) {
            const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;
#pragma omp parallel for schedule(static, elementsPerThread) shared(dest, source, elementsPerThread, numElements) default(none)
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (TO_TYPE)(source[i]);
            }
        } else {
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (TO_TYPE)(source[i]);
            }
        }
    } else if (sizeof(TO_TYPE) < sizeof(FROM_TYPE)) {
        // In place and converting to a smaller type
        for (long i = 0; i < numElements; ++i) {
            dest[i] = (TO_TYPE)(source[i]);
        }
    } else {
        // In place and converting to a larger type
        for (long i = numElements - 1; i >= 0; --i) {
            dest[i] = (TO_TYPE)(source[i]);
        }
    }
}

// when converting from an integeral type to half, first convert to float to remove the abiguity between calling
// __half(float) and __half(double)
template <typename FROM_TYPE>
void TypeConverter::cpuConvertTypeFromIntegralToHalfImpl(FROM_TYPE *source, half *dest, long numElements) {
    assert((is_integral<FROM_TYPE>::value));
    assert((is_convertible<FROM_TYPE, float>::value));
    assert((is_convertible<float, half>::value));

    if (numElements == 0)
        return;

    bool inPlaceConversion = ((void *)source == (void *)dest);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source;
        void *sourceEnd = (void *)&(source[numElements - 1]);
        void *destStart = dest;
        void *destEnd = (void *)&(dest[numElements - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    if (!inPlaceConversion || sizeof(half) == sizeof(FROM_TYPE)) {
        // Out of place or in place or conversion to a same sized type
        const uint32_t numProcs = max(min((long)omp_get_num_procs(), numElements / 500000), 1L);
        if (numProcs > 1) {
            const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;
#pragma omp parallel for schedule(static, elementsPerThread) shared(dest, source, elementsPerThread, numElements) default(none)
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (half)(float)(source[i]);
            }
        } else {
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (half)(float)(source[i]);
            }
        }
    } else if (sizeof(half) < sizeof(FROM_TYPE)) {
        // In place and converting to a smaller type
        for (long i = 0; i < numElements; ++i) {
            dest[i] = (half)(float)(source[i]);
        }
    } else {
        // In place and converting to a larger type
        for (long i = numElements - 1; i >= 0; --i) {
            dest[i] = (half)(float)(source[i]);
        }
    }
}

template <typename TO_TYPE>
void TypeConverter::cpuConvertTypeFromPackedBooleanImpl(void *source, TO_TYPE *dest, long numElements) {
    assert((is_convertible<bool, TO_TYPE>::value));

    if (numElements == 0)
        return;

    bool inPlaceConversion = ((void *)source == (void *)dest);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source;
        void *sourceEnd = (void *)&(((uint8_t *)source)[((numElements - 1) + 7) / 8]);
        void *destStart = dest;
        void *destEnd = (void *)&(dest[numElements - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    if (!inPlaceConversion) {
        // Out of place
        const uint32_t numProcs = max(min((long)omp_get_num_procs(), numElements / 500000), 1L);
        if (numProcs > 1) {
            const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;
#pragma omp parallel for schedule(static, elementsPerThread) shared(dest, source, elementsPerThread, numElements) default(none)
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (TO_TYPE)PackedBoolean::getElement(i, source);
            }
        } else {
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (TO_TYPE)PackedBoolean::getElement(i, source);
            }
        }
    } else {
        // In place and to a larger type
        for (long i = numElements - 1; i >= 0; --i) {
            dest[i] = (TO_TYPE)PackedBoolean::getElement(i, source);
        }
    }
}

template <typename FROM_TYPE>
void TypeConverter::cpuConvertTypeToPackedBooleanImpl(FROM_TYPE *source, void *dest, long numElements) {
    assert((is_convertible<FROM_TYPE, bool>::value));

    if (numElements == 0)
        return;

    bool inPlaceConversion = ((void *)source == (void *)dest);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source;
        void *sourceEnd = (void *)&(source[numElements - 1]);
        void *destStart = dest;
        void *destEnd = (void *)&(((uint8_t *)dest)[((numElements - 1) + 7) / 8]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    if (!inPlaceConversion) {
        // Out of place
        const uint32_t numProcs = max(min((long)omp_get_num_procs(), numElements / 500000), 1L);
        if (numProcs > 1) {
            const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;
#pragma omp parallel for schedule(static, elementsPerThread) shared(dest, source, elementsPerThread, numElements) default(none)
            for (long i = 0; i < numElements; ++i) {
                PackedBoolean::setElement((bool)source[i], i, dest);
            }
        } else {
            for (long i = 0; i < numElements; ++i) {
                PackedBoolean::setElement((bool)source[i], i, dest);
            }
        }
    } else {
        // In place and to a smaller type
        for (long i = 0; i < numElements; ++i) {
            PackedBoolean::setElement((bool)source[i], i, dest);
        }
    }
}

void TypeConverter::cpuConvertTypeFromPackedBooleanToHalfImpl(void *source, half *dest, long numElements) {
    assert((is_convertible<bool, float>::value));
    assert((is_convertible<float, half>::value));

    if (numElements == 0)
        return;

    bool inPlaceConversion = ((void *)source == (void *)dest);

    // When not doing an inplace operation, the memory regions must not overlap
    if (!inPlaceConversion) {
        void *sourceStart = source;
        void *sourceEnd = (void *)&(((uint8_t *)source)[((numElements - 1) + 7) / 8]);
        void *destStart = dest;
        void *destEnd = (void *)&(dest[numElements - 1]);
        assert(sourceEnd < destStart || sourceStart > destEnd);
    }

    if (!inPlaceConversion) {
        // Out of place
        const uint32_t numProcs = max(min((long)omp_get_num_procs(), numElements / 500000), 1L);
        if (numProcs > 1) {
            const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;
#pragma omp parallel for schedule(static, elementsPerThread) shared(dest, source, elementsPerThread, numElements) default(none)
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (half)((float)PackedBoolean::getElement(i, source));
            }
        } else {
            for (long i = 0; i < numElements; ++i) {
                dest[i] = (half)((float)PackedBoolean::getElement(i, source));
            }
        }
    } else {
        // In place
        for (long i = numElements - 1; i >= 0; --i) {
            dest[i] = (half)((float)PackedBoolean::getElement(i, source));
        }
    }
}
