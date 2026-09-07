#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include <stdio.h>

#include <cmath>
#include <limits>

#include "gtest/gtest.h"

using namespace ThorImplementation;
using namespace std;

TEST(RandomTestConversions, CpuAllConversionsOutOfPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (DataType sourceDataType = DataType::FP16; sourceDataType <= DataType::BF16; sourceDataType = (DataType)((int)sourceDataType + 1)) {
        for (DataType destDataType = DataType::FP16; destDataType <= DataType::BF16; destDataType = (DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            int NUM_ELEMENTS = 1 + (rand() % 2000);
            if (rand() % 20 == 0)
                NUM_ELEMENTS = 10000 + rand() % 10000;

            // printf("%s -> %s\n",
            //        TensorDescriptor::getElementTypeName(sourceDataType).c_str(),
            //        TensorDescriptor::getElementTypeName(destDataType).c_str());
            //
            // printf("source data type %d dest data type %d numElements %d\n", (int)sourceDataType, (int)destDataType, NUM_ELEMENTS);

            void *source;
            void *dest;
            int numSourceBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType);
            int numDestBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType);
            cudaStatus = cudaHostAlloc(&source, numSourceBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&dest, numDestBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);

            bool destDataTypeIsSigned = TensorDescriptor::isSignedType(destDataType);
            if (sourceDataType == DataType::FP8_E4M3) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP8_E5M2) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::BF16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((rand() % 100) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (4 - (rand() % 8)) * 0.25;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 8) * 0.25;
                    }
                }
            } else if (sourceDataType == DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 8) * 0.25;
                }
            } else if (sourceDataType == DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 8) * 0.25;
                }
            } else if (sourceDataType == DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else {
                assert(false);
            }

            TypeConverter::convertType(source, dest, sourceDataType, destDataType, NUM_ELEMENTS, stream, -1);
            cudaStatus = cudaStreamSynchronize(stream.getStream());
            assert(cudaStatus == cudaSuccess);

            for (int i = 0; i < NUM_ELEMENTS; ++i) {
                string sourceStringVal = TensorDescriptor::getValueAsString(source, i, sourceDataType);
                string destStringVal = TensorDescriptor::getValueAsString(dest, i, destDataType);
                double sourceVal = std::stod(sourceStringVal);
                double destVal = std::stod(destStringVal);
                if (TensorDescriptor::isBooleanType(sourceDataType))
                    destVal = (bool)destVal;
                if (TensorDescriptor::isBooleanType(destDataType))
                    sourceVal = (bool)sourceVal;

                if (TensorDescriptor::isIntegralType(sourceDataType)) {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        ASSERT_EQ(sourceVal, destVal);
                    } else {
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                } else {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        ASSERT_LT(abs(sourceVal - destVal), 1.0);
                    } else {
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                }
            }

            cudaStatus = cudaFreeHost(source);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(dest);
            assert(cudaStatus == cudaSuccess);
        }
    }
}

TEST(RandomTestConversions, CpuAllConversionsInPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (DataType sourceDataType = DataType::FP16; sourceDataType <= DataType::BF16; sourceDataType = (DataType)((int)sourceDataType + 1)) {
        for (DataType destDataType = DataType::FP16; destDataType <= DataType::BF16; destDataType = (DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            int NUM_ELEMENTS = 1 + (rand() % 2000);
            if (rand() % 30 == 0)
                NUM_ELEMENTS = 10000 + rand() % 10000;

            void *source;
            void *dest;
            int numSourceBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType);
            int numDestBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType);
            int numBytes = numSourceBytes > numDestBytes ? numSourceBytes : numDestBytes;
            cudaStatus = cudaHostAlloc(&source, numBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&dest, numBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);

            bool destDataTypeIsSigned = TensorDescriptor::isSignedType(destDataType);
            if (sourceDataType == DataType::FP8_E4M3) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP8_E5M2) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::BF16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((rand() % 100) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (4 - (rand() % 8)) * 0.25;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 8) * 0.25;
                    }
                }
            } else if (sourceDataType == DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 8) * 0.25;
                }
            } else if (sourceDataType == DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 8) * 0.25;
                }
            } else if (sourceDataType == DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else {
                assert(false);
            }

            memcpy(dest, source, numBytes);

            // in-place
            TypeConverter::convertType(dest, dest, sourceDataType, destDataType, NUM_ELEMENTS, stream, -1);
            cudaStatus = cudaStreamSynchronize(stream.getStream());
            assert(cudaStatus == cudaSuccess);

            for (int i = 0; i < NUM_ELEMENTS; ++i) {
                string sourceStringVal = TensorDescriptor::getValueAsString(source, i, sourceDataType);
                string destStringVal = TensorDescriptor::getValueAsString(dest, i, destDataType);
                double sourceVal = std::stod(sourceStringVal);
                double destVal = std::stod(destStringVal);
                if (TensorDescriptor::isBooleanType(sourceDataType))
                    destVal = (bool)destVal;
                if (TensorDescriptor::isBooleanType(destDataType))
                    sourceVal = (bool)sourceVal;

                if (TensorDescriptor::isIntegralType(sourceDataType)) {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        ASSERT_EQ(sourceVal, destVal);
                    } else {
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                } else {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        ASSERT_LT(abs(sourceVal - destVal), 1.0);
                    } else {
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                }
            }

            cudaStatus = cudaFreeHost(source);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(dest);
            assert(cudaStatus == cudaSuccess);
        }
    }
}

TEST(RandomTestConversions, GpuAllConversionsOutOfPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (DataType sourceDataType = DataType::FP16; sourceDataType <= DataType::BF16; sourceDataType = (DataType)((int)sourceDataType + 1)) {
        for (DataType destDataType = DataType::FP16; destDataType <= DataType::BF16; destDataType = (DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            // printf("%s -> %s\n",
            //        TensorDescriptor::getElementTypeName(sourceDataType).c_str(),
            //        TensorDescriptor::getElementTypeName(destDataType).c_str());

            const int NUM_ELEMENTS = 1 + (rand() % 25000);

            // printf("source data type %d dest data type %d numElements %d\n", (int)sourceDataType, (int)destDataType, NUM_ELEMENTS);

            void *source;
            void *dest;
            int numSourceBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType);
            int numDestBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType);
            cudaStatus = cudaHostAlloc(&source, numSourceBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&dest, numDestBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);

            bool destDataTypeIsSigned = TensorDescriptor::isSignedType(destDataType);
            if (sourceDataType == DataType::FP8_E4M3) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP8_E5M2) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::BF16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((rand() % 100) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (4 - (rand() % 8)) * 0.25;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 10) * 0.25;
                    }
                }
            } else if (sourceDataType == DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 10) * 0.25;
                }
            } else if (sourceDataType == DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 10) * 0.25;
                }
            } else if (sourceDataType == DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 10;
                }
            } else if (sourceDataType == DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 10;
                }
            } else if (sourceDataType == DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 10;
                }
            } else if (sourceDataType == DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 10;
                }
            } else if (sourceDataType == DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else {
                assert(false);
            }

            void *source_d;
            void *dest_d;
            cudaStatus = cudaMalloc(&source_d, TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType));
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaMalloc(&dest_d, TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType));
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaMemcpyAsync(source_d,
                                         source,
                                         TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType),
                                         cudaMemcpyHostToDevice,
                                         stream.getStream());
            assert(cudaStatus == cudaSuccess);

            TypeConverter::convertType(source_d, dest_d, sourceDataType, destDataType, NUM_ELEMENTS, stream, 0);

            cudaStatus = cudaMemcpyAsync(dest,
                                         dest_d,
                                         TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType),
                                         cudaMemcpyDeviceToHost,
                                         stream.getStream());
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaStreamSynchronize(stream.getStream());
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaFree(source_d);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFree(dest_d);
            assert(cudaStatus == cudaSuccess);

            for (int i = 0; i < NUM_ELEMENTS; ++i) {
                string sourceStringVal = TensorDescriptor::getValueAsString(source, i, sourceDataType);
                string destStringVal = TensorDescriptor::getValueAsString(dest, i, destDataType);
                double sourceVal = std::stod(sourceStringVal);
                double destVal = std::stod(destStringVal);
                if (TensorDescriptor::isBooleanType(sourceDataType))
                    destVal = (bool)destVal;
                if (TensorDescriptor::isBooleanType(destDataType))
                    sourceVal = (bool)sourceVal;

                //    printf("[%5d] s:%lf d:%lf\n", i, sourceVal, destVal);

                if (TensorDescriptor::isIntegralType(sourceDataType)) {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        if (sourceVal != destVal)
                            printf("[%d] source %f dest %f\n", i, sourceVal, destVal);
                        ASSERT_EQ(sourceVal, destVal);
                    } else {
                        if (!(abs(sourceVal - destVal) < 0.1))
                            printf("[%d] source %f dest %f\n", i, sourceVal, destVal);
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                } else {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        if (abs(sourceVal - destVal) >= 1.0)
                            printf("[%d] source %f dest %f\n", i, sourceVal, destVal);
                        ASSERT_LT(abs(sourceVal - destVal), 1.0);
                    } else {
                        if (abs(sourceVal - destVal) >= 0.1)
                            printf("[%d] source %f dest %f\n", i, sourceVal, destVal);
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                }
            }

            cudaStatus = cudaFreeHost(source);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(dest);
            assert(cudaStatus == cudaSuccess);
        }
    }
}

TEST(RandomTestConversions, GpuAllConversionsInPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (DataType sourceDataType = DataType::FP16; sourceDataType <= DataType::BF16; sourceDataType = (DataType)((int)sourceDataType + 1)) {
        for (DataType destDataType = DataType::FP16; destDataType <= DataType::BF16; destDataType = (DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            const int NUM_ELEMENTS = 1 + (rand() % 25000);

            // printf("%s <- %s   %d\n",
            //       TensorDescriptor::getElementTypeName(destDataType).c_str(),
            //       TensorDescriptor::getElementTypeName(sourceDataType).c_str(),
            //       NUM_ELEMENTS);

            void *source;
            void *dest;
            int numSourceBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType);
            int numDestBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType);
            cudaStatus = cudaHostAlloc(&source, numSourceBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&dest, numDestBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);

            bool destDataTypeIsSigned = TensorDescriptor::isSignedType(destDataType);
            if (sourceDataType == DataType::FP8_E4M3) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e4m3 *)source)[i] = __nv_fp8_e4m3((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP8_E5M2) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_fp8_e5m2 *)source)[i] = __nv_fp8_e5m2((rand() % 10) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::BF16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((4 - (rand() % 8)) * 0.25);
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((__nv_bfloat16 *)source)[i] = __nv_bfloat16((rand() % 100) * 0.25);
                    }
                }
            } else if (sourceDataType == DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (4 - (rand() % 8)) * 0.25;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 100) * 0.1;
                    }
                }
            } else if (sourceDataType == DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (4 - (rand() % 8)) * 0.25;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 4 - (rand() % 8);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 8;
                }
            } else if (sourceDataType == DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 8);
            } else if (sourceDataType == DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else {
                assert(false);
            }

            void *inPlaceMem_d;
            int numBytes = numDestBytes > numSourceBytes ? numDestBytes : numSourceBytes;
            cudaStatus = cudaMalloc(&inPlaceMem_d, numBytes);
            assert(cudaStatus == cudaSuccess);

            cudaStatus = cudaMemcpyAsync(inPlaceMem_d,
                                         source,
                                         TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType),
                                         cudaMemcpyHostToDevice,
                                         stream.getStream());
            assert(cudaStatus == cudaSuccess);

            TypeConverter::convertType(inPlaceMem_d, inPlaceMem_d, sourceDataType, destDataType, NUM_ELEMENTS, stream, 0);

            cudaStatus = cudaMemcpyAsync(dest,
                                         inPlaceMem_d,
                                         TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType),
                                         cudaMemcpyDeviceToHost,
                                         stream.getStream());
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaStreamSynchronize(stream.getStream());
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFree(inPlaceMem_d);
            assert(cudaStatus == cudaSuccess);

            for (int i = 0; i < NUM_ELEMENTS; ++i) {
                string sourceStringVal = TensorDescriptor::getValueAsString(source, i, sourceDataType);
                string destStringVal = TensorDescriptor::getValueAsString(dest, i, destDataType);
                double sourceVal = std::stod(sourceStringVal);
                double destVal = std::stod(destStringVal);
                if (TensorDescriptor::isBooleanType(sourceDataType))
                    destVal = (bool)destVal;
                if (TensorDescriptor::isBooleanType(destDataType))
                    sourceVal = (bool)sourceVal;

                // if(sourceDataType == DataType::FP16 && destDataType == DataType::FP32)
                //    printf("[%5d] s:%lf d:%lf\n", i, sourceVal, destVal);

                if (TensorDescriptor::isIntegralType(sourceDataType)) {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        ASSERT_EQ(sourceVal, destVal);
                    } else {
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                } else {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        ASSERT_LT(abs(sourceVal - destVal), 1.0);
                    } else {
                        ASSERT_LT(abs(sourceVal - destVal), 0.1);
                    }
                }
            }

            cudaStatus = cudaFreeHost(source);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaFreeHost(dest);
            assert(cudaStatus == cudaSuccess);
        }
    }
}

TEST(TypeConverter, CpuFp32ToE5M2UsesNativeInfinityOverflowSemantics) {
    Stream stream(0);
    constexpr int numElements = 6;
    float source[numElements] = {57344.0f,
                                 60000.0f,
                                 100000.0f,
                                 -100000.0f,
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity()};
    __nv_fp8_e5m2 dest[numElements]{};

    TypeConverter::convertType(source, dest, DataType::FP32, DataType::FP8_E5M2, numElements, stream, -1);
    ASSERT_EQ(cudaStreamSynchronize(stream.getStream()), cudaSuccess);

    EXPECT_FLOAT_EQ(static_cast<float>(dest[0]), 57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(dest[1]), 57344.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(dest[2])));
    EXPECT_FALSE(std::signbit(static_cast<float>(dest[2])));
    EXPECT_TRUE(std::isinf(static_cast<float>(dest[3])));
    EXPECT_TRUE(std::signbit(static_cast<float>(dest[3])));
    EXPECT_TRUE(std::isinf(static_cast<float>(dest[4])));
    EXPECT_FALSE(std::signbit(static_cast<float>(dest[4])));
    EXPECT_TRUE(std::isinf(static_cast<float>(dest[5])));
    EXPECT_TRUE(std::signbit(static_cast<float>(dest[5])));
}

TEST(TypeConverter, GpuFp32ToE5M2UsesNativeInfinityOverflowSemantics) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        GTEST_SKIP() << "CUDA device is required for GPU TypeConverter semantics test.";
    }

    Stream stream(0);
    constexpr int numElements = 4;
    float source[numElements] = {60000.0f, 100000.0f, -100000.0f, -60000.0f};
    __nv_fp8_e5m2 dest[numElements]{};
    float* sourceGpu = nullptr;
    __nv_fp8_e5m2* destGpu = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&sourceGpu), sizeof(source)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&destGpu), sizeof(dest)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(sourceGpu, source, sizeof(source), cudaMemcpyHostToDevice, stream.getStream()), cudaSuccess);

    TypeConverter::convertType(sourceGpu, destGpu, DataType::FP32, DataType::FP8_E5M2, numElements, stream, 0);
    ASSERT_EQ(cudaMemcpyAsync(dest, destGpu, sizeof(dest), cudaMemcpyDeviceToHost, stream.getStream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.getStream()), cudaSuccess);

    EXPECT_FLOAT_EQ(static_cast<float>(dest[0]), 57344.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(dest[1])));
    EXPECT_FALSE(std::signbit(static_cast<float>(dest[1])));
    EXPECT_TRUE(std::isinf(static_cast<float>(dest[2])));
    EXPECT_TRUE(std::signbit(static_cast<float>(dest[2])));
    EXPECT_FLOAT_EQ(static_cast<float>(dest[3]), -57344.0f);

    EXPECT_EQ(cudaFree(sourceGpu), cudaSuccess);
    EXPECT_EQ(cudaFree(destGpu), cudaSuccess);
}
