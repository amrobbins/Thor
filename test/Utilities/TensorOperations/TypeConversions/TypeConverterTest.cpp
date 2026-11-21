#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"

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

TEST(RandomTestConversions, CpuWideningAndSameSizeConversionsInPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (DataType sourceDataType = DataType::FP16; sourceDataType <= DataType::BF16; sourceDataType = (DataType)((int)sourceDataType + 1)) {
        for (DataType destDataType = DataType::FP16; destDataType <= DataType::BF16; destDataType = (DataType)((int)destDataType + 1)) {
            // In-place narrowing is refused; RefusesInPlaceNarrowing covers it.
            if (sourceDataType == destDataType ||
                TensorDescriptor::getElementSizeInBytes(destDataType) < TensorDescriptor::getElementSizeInBytes(sourceDataType))
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

TEST(RandomTestConversions, GpuWideningAndSameSizeConversionsInPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (DataType sourceDataType = DataType::FP16; sourceDataType <= DataType::BF16; sourceDataType = (DataType)((int)sourceDataType + 1)) {
        for (DataType destDataType = DataType::FP16; destDataType <= DataType::BF16; destDataType = (DataType)((int)destDataType + 1)) {
            // In-place narrowing is refused; RefusesInPlaceNarrowing covers it.
            if (sourceDataType == destDataType ||
                TensorDescriptor::getElementSizeInBytes(destDataType) < TensorDescriptor::getElementSizeInBytes(sourceDataType))
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

TEST(TypeConverter, CpuFp32ToE5M2UsesCanonicalSatfiniteOverflowSemantics) {
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
    EXPECT_FLOAT_EQ(static_cast<float>(dest[2]), 57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(dest[3]), -57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(dest[4]), 57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(dest[5]), -57344.0f);
}

TEST(TypeConverter, GpuFp32ToE5M2UsesCanonicalSatfiniteOverflowSemantics) {
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
    EXPECT_FLOAT_EQ(static_cast<float>(dest[1]), 57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(dest[2]), -57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(dest[3]), -57344.0f);

    EXPECT_EQ(cudaFree(sourceGpu), cudaSuccess);
    EXPECT_EQ(cudaFree(destGpu), cudaSuccess);
}

TEST(TypeConverter, RefusesInPlaceNarrowing) {
    Stream stream(0);
    constexpr int numElements = 4;

    float hostMem[numElements] = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(TypeConverter::convertType(hostMem, hostMem, DataType::FP32, DataType::FP16, numElements, stream, -1),
                 std::runtime_error);
    EXPECT_THROW(TypeConverter::convertType(hostMem, hostMem, DataType::FP32, DataType::FP8_E5M2, 0, stream, -1),
                 std::runtime_error);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor cpuTensor(cpuPlacement, TensorDescriptor(DataType::FP64, {numElements}));
    EXPECT_THROW(TypeConverter::convertType(cpuTensor, cpuTensor, DataType::FP64, DataType::INT32, numElements, stream),
                 std::runtime_error);

    void *gpuMem = nullptr;
    ASSERT_EQ(cudaMalloc(&gpuMem, numElements * sizeof(double)), cudaSuccess);
    EXPECT_THROW(TypeConverter::convertType(gpuMem, gpuMem, DataType::FP64, DataType::FP8_E4M3, numElements, stream, 0),
                 std::runtime_error);
    EXPECT_THROW(TypeConverter::convertType(gpuMem, gpuMem, DataType::INT16, DataType::BOOLEAN, numElements, stream, 0),
                 std::runtime_error);

    // Widening and same-size conversions still run in place.
    TypeConverter::convertType(gpuMem, gpuMem, DataType::FP16, DataType::FP32, numElements, stream, 0);
    TypeConverter::convertType(gpuMem, gpuMem, DataType::FP32, DataType::INT32, numElements, stream, 0);
    ASSERT_EQ(cudaStreamSynchronize(stream.getStream()), cudaSuccess);
    EXPECT_EQ(cudaFree(gpuMem), cudaSuccess);
}

namespace {

bool isFp8Nan(uint8_t bits, __nv_fp8_interpretation_t kind) {
    return kind == __NV_E4M3 ? (bits & 0x7F) == 0x7F : (bits & 0x7F) > 0x7C;
}

// Converts count elements of source to both FP8 kinds on the CPU and the GPU, and expects the reference's bits (any
// NaN where the reference is a NaN).
void expectFp8ConversionsMatch(DataType sourceType,
                               const void *source,
                               size_t count,
                               const vector<uint8_t> &e4m3Reference,
                               const vector<uint8_t> &e5m2Reference) {
    Stream stream(0);
    const size_t sourceBytes = count * static_cast<size_t>(TensorDescriptor::getElementSizeInBytes(sourceType));
    constexpr size_t padding = 256;  // the out-of-place kernels may run past the last element
    void *source_d = nullptr;
    void *dest_d = nullptr;
    ASSERT_EQ(cudaMalloc(&source_d, sourceBytes + padding), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dest_d, count + padding), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(source_d, source, sourceBytes, cudaMemcpyHostToDevice, stream.getStream()), cudaSuccess);

    for (const DataType destType : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const __nv_fp8_interpretation_t kind = destType == DataType::FP8_E4M3 ? __NV_E4M3 : __NV_E5M2;
        const vector<uint8_t> &reference = destType == DataType::FP8_E4M3 ? e4m3Reference : e5m2Reference;
        vector<uint8_t> cpu(count);
        vector<uint8_t> gpu(count);
        TypeConverter::convertType(const_cast<void *>(source), cpu.data(), sourceType, destType, count, stream, -1);
        TypeConverter::convertType(source_d, dest_d, sourceType, destType, count, stream, 0);
        ASSERT_EQ(cudaMemcpyAsync(gpu.data(), dest_d, count, cudaMemcpyDeviceToHost, stream.getStream()), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream.getStream()), cudaSuccess);

        size_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) {
            const uint8_t results[2] = {cpu[i], gpu[i]};
            for (int side = 0; side < 2; ++side) {
                const uint8_t bits = results[side];
                const bool match = isFp8Nan(reference[i], kind) ? isFp8Nan(bits, kind) : bits == reference[i];
                if (!match && mismatches++ < 8) {
                    ADD_FAILURE() << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                  << TensorDescriptor::getElementTypeName(destType) << (side == 0 ? " cpu" : " gpu")
                                  << " element " << i << ": got 0x" << hex << int(bits) << ", expected 0x" << int(reference[i])
                                  << dec;
                }
            }
        }
        EXPECT_EQ(mismatches, 0u) << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                  << TensorDescriptor::getElementTypeName(destType);
    }

    EXPECT_EQ(cudaFree(source_d), cudaSuccess);
    EXPECT_EQ(cudaFree(dest_d), cudaSuccess);
}

// Every value of an 8- or 16-bit integer type; for wider types every value in [-70000, 70000] (both FP8 kinds' whole
// finite range and their rounding), and the values around 2^24, 2^31, 2^53 and the type's limits.
template <typename T>
vector<T> integerSweep() {
    vector<T> values;
    if constexpr (sizeof(T) <= 2) {
        for (int64_t v = numeric_limits<T>::min(); v <= numeric_limits<T>::max(); ++v)
            values.push_back(static_cast<T>(v));
    } else {
        const int64_t low = numeric_limits<T>::is_signed ? -70000 : 0;
        for (int64_t v = low; v <= 70000; ++v)
            values.push_back(static_cast<T>(v));
        for (const double edge : {16777216.0, 2147483648.0, 9007199254740992.0}) {
            for (int64_t delta = -1; delta <= 1; ++delta) {
                for (const double sign : {1.0, -1.0}) {
                    const long double v = static_cast<long double>(sign * edge) + delta;
                    if (v >= static_cast<long double>(numeric_limits<T>::min()) && v <= static_cast<long double>(numeric_limits<T>::max()))
                        values.push_back(static_cast<T>(v));
                }
            }
        }
        for (const T edge : {numeric_limits<T>::min(), static_cast<T>(numeric_limits<T>::min() + 1), numeric_limits<T>::max(),
                             static_cast<T>(numeric_limits<T>::max() - 1)})
            values.push_back(edge);
    }
    return values;
}

template <typename T>
void expectIntegerToFp8MatchesTheConversionThroughDouble(DataType sourceType) {
    const vector<T> values = integerSweep<T>();
    vector<uint8_t> e4m3(values.size());
    vector<uint8_t> e5m2(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        e4m3[i] = __nv_cvt_double_to_fp8(static_cast<double>(values[i]), __NV_SATFINITE, __NV_E4M3);
        e5m2[i] = __nv_cvt_double_to_fp8(static_cast<double>(values[i]), __NV_SATFINITE, __NV_E5M2);
    }
    expectFp8ConversionsMatch(sourceType, values.data(), values.size(), e4m3, e5m2);
}

}  // namespace

// Integers and bool convert to FP8 through FP32; the bits must be those of the conversion through double.
TEST(TypeConverter, IntegersAndBoolToFp8MatchTheConversionThroughDouble) {
    expectIntegerToFp8MatchesTheConversionThroughDouble<int8_t>(DataType::INT8);
    expectIntegerToFp8MatchesTheConversionThroughDouble<uint8_t>(DataType::UINT8);
    expectIntegerToFp8MatchesTheConversionThroughDouble<int16_t>(DataType::INT16);
    expectIntegerToFp8MatchesTheConversionThroughDouble<uint16_t>(DataType::UINT16);
    expectIntegerToFp8MatchesTheConversionThroughDouble<int32_t>(DataType::INT32);
    expectIntegerToFp8MatchesTheConversionThroughDouble<uint32_t>(DataType::UINT32);
    expectIntegerToFp8MatchesTheConversionThroughDouble<int64_t>(DataType::INT64);
    expectIntegerToFp8MatchesTheConversionThroughDouble<uint64_t>(DataType::UINT64);

    const vector<uint8_t> bools = {0, 1, 1, 0};
    const vector<uint8_t> e4m3 = {__nv_cvt_double_to_fp8(0.0, __NV_SATFINITE, __NV_E4M3),
                                  __nv_cvt_double_to_fp8(1.0, __NV_SATFINITE, __NV_E4M3),
                                  __nv_cvt_double_to_fp8(1.0, __NV_SATFINITE, __NV_E4M3),
                                  __nv_cvt_double_to_fp8(0.0, __NV_SATFINITE, __NV_E4M3)};
    const vector<uint8_t> e5m2 = {__nv_cvt_double_to_fp8(0.0, __NV_SATFINITE, __NV_E5M2),
                                  __nv_cvt_double_to_fp8(1.0, __NV_SATFINITE, __NV_E5M2),
                                  __nv_cvt_double_to_fp8(1.0, __NV_SATFINITE, __NV_E5M2),
                                  __nv_cvt_double_to_fp8(0.0, __NV_SATFINITE, __NV_E5M2)};
    expectFp8ConversionsMatch(DataType::BOOLEAN, bools.data(), bools.size(), e4m3, e5m2);
}

// FP16 converts to FP8 directly; for every FP16 bit pattern the bits must be those of the conversion through FP32.
TEST(TypeConverter, Fp16ToFp8MatchesTheConversionThroughFp32ForEveryValue) {
    vector<uint16_t> bits(65536);
    vector<uint8_t> e4m3(bits.size());
    vector<uint8_t> e5m2(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        bits[i] = static_cast<uint16_t>(i);
        __half_raw raw;
        raw.x = bits[i];
        const float value = __half2float(__half(raw));
        e4m3[i] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
        e5m2[i] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
    }
    expectFp8ConversionsMatch(DataType::FP16, bits.data(), bits.size(), e4m3, e5m2);
}

// FP8 to int8 and uint8, for every FP8 bit pattern of both kinds, on the CPU and the GPU: NaN to 0, otherwise
// truncated toward zero and saturated to the destination's range (infinities included).
TEST(TypeConverter, Fp8ToInt8AndUint8TruncateSaturateAndMapNanToZeroForEveryValue) {
    Stream stream(0);
    constexpr size_t count = 256;
    constexpr size_t padding = 256;
    vector<uint8_t> source(count);
    for (size_t i = 0; i < count; ++i)
        source[i] = static_cast<uint8_t>(i);

    void *source_d = nullptr;
    void *dest_d = nullptr;
    ASSERT_EQ(cudaMalloc(&source_d, count + padding), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dest_d, count + padding), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(source_d, source.data(), count, cudaMemcpyHostToDevice, stream.getStream()), cudaSuccess);

    for (const DataType sourceType : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const __nv_fp8_interpretation_t kind = sourceType == DataType::FP8_E4M3 ? __NV_E4M3 : __NV_E5M2;
        for (const DataType destType : {DataType::INT8, DataType::UINT8}) {
            const float low = destType == DataType::INT8 ? -128.0f : 0.0f;
            const float high = destType == DataType::INT8 ? 127.0f : 255.0f;
            vector<uint8_t> reference(count);
            for (size_t i = 0; i < count; ++i) {
                const float value = __half2float(__half(__nv_cvt_fp8_to_halfraw(source[i], kind)));
                const float converted = std::isnan(value) ? 0.0f : std::min(std::max(std::trunc(value), low), high);
                reference[i] = destType == DataType::INT8 ? static_cast<uint8_t>(static_cast<int8_t>(converted))
                                                          : static_cast<uint8_t>(converted);
            }

            vector<uint8_t> cpu(count);
            vector<uint8_t> gpu(count);
            TypeConverter::convertType(source.data(), cpu.data(), sourceType, destType, count, stream, -1);
            TypeConverter::convertType(source_d, dest_d, sourceType, destType, count, stream, 0);
            ASSERT_EQ(cudaMemcpyAsync(gpu.data(), dest_d, count, cudaMemcpyDeviceToHost, stream.getStream()), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream.getStream()), cudaSuccess);

            for (size_t i = 0; i < count; ++i) {
                EXPECT_EQ(int(cpu[i]), int(reference[i])) << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                                          << TensorDescriptor::getElementTypeName(destType) << " cpu, bits 0x" << hex
                                                          << i << dec;
                EXPECT_EQ(int(gpu[i]), int(reference[i])) << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                                          << TensorDescriptor::getElementTypeName(destType) << " gpu, bits 0x" << hex
                                                          << i << dec;
            }
        }
    }

    EXPECT_EQ(cudaFree(source_d), cudaSuccess);
    EXPECT_EQ(cudaFree(dest_d), cudaSuccess);
}

namespace {

double decodeFloatingSource(DataType type, const uint8_t *element) {
    switch (type) {
        case DataType::FP32: {
            float value;
            memcpy(&value, element, sizeof(value));
            return value;
        }
        case DataType::FP64: {
            double value;
            memcpy(&value, element, sizeof(value));
            return value;
        }
        case DataType::FP16: {
            __half_raw raw;
            memcpy(&raw.x, element, sizeof(raw.x));
            return __half2float(__half(raw));
        }
        case DataType::BF16: {
            __nv_bfloat16_raw raw;
            memcpy(&raw.x, element, sizeof(raw.x));
            return __bfloat162float(__nv_bfloat16(raw));
        }
        case DataType::FP8_E4M3:
            return __half2float(__half(__nv_cvt_fp8_to_halfraw(*element, __NV_E4M3)));
        case DataType::FP8_E5M2:
            return __half2float(__half(__nv_cvt_fp8_to_halfraw(*element, __NV_E5M2)));
        default:
            THOR_UNREACHABLE();
    }
    return 0;
}

struct IntegerTarget {
    DataType type;
    long double low;
    long double high;
};

long double readInteger(DataType type, const uint8_t *element) {
    switch (type) {
#define THOR_READ_INTEGER(DTYPE, T)          \
    case DataType::DTYPE: {                  \
        T value;                             \
        memcpy(&value, element, sizeof(T));  \
        return static_cast<long double>(value); \
    }
        THOR_READ_INTEGER(INT8, int8_t)
        THOR_READ_INTEGER(INT16, int16_t)
        THOR_READ_INTEGER(INT32, int32_t)
        THOR_READ_INTEGER(INT64, int64_t)
        THOR_READ_INTEGER(UINT8, uint8_t)
        THOR_READ_INTEGER(UINT16, uint16_t)
        THOR_READ_INTEGER(UINT32, uint32_t)
        THOR_READ_INTEGER(UINT64, uint64_t)
#undef THOR_READ_INTEGER
        default:
            THOR_UNREACHABLE();
    }
    return 0;
}

// Converts every element of source to every integer type on the CPU and the GPU, and expects Thor's rule: truncate
// toward zero, saturate (computed here in long double, which holds every 64-bit bound exactly on x86-64). NaN is what
// PTX (ISA 9.0) documents for cvt: 0 unless the source is FP64 or the destination 64-bit, and otherwise 1 << (N - 1)
// of the conversion's N bits (the minimum when signed, 2^(N-1) when unsigned). The device converts to 8- to 32-bit
// through the 32-bit cvt and a clamp, so that value is clamped to the destination's range. The CPU gives the same.
void expectFloatToIntegerRule(DataType sourceType, const vector<uint8_t> &source) {
    const vector<IntegerTarget> targets = {
        {DataType::INT8, -128.0L, 127.0L},
        {DataType::INT16, -32768.0L, 32767.0L},
        {DataType::INT32, -2147483648.0L, 2147483647.0L},
        {DataType::INT64, -9223372036854775808.0L, 9223372036854775807.0L},
        {DataType::UINT8, 0.0L, 255.0L},
        {DataType::UINT16, 0.0L, 65535.0L},
        {DataType::UINT32, 0.0L, 4294967295.0L},
        {DataType::UINT64, 0.0L, 18446744073709551615.0L},
    };
    Stream stream(0);
    const size_t sourceBytes = static_cast<size_t>(TensorDescriptor::getElementSizeInBytes(sourceType));
    const size_t count = source.size() / sourceBytes;
    constexpr size_t padding = 256;
    void *source_d = nullptr;
    void *dest_d = nullptr;
    ASSERT_EQ(cudaMalloc(&source_d, source.size() + padding), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dest_d, count * 8 + padding), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(source_d, source.data(), source.size(), cudaMemcpyHostToDevice, stream.getStream()), cudaSuccess);

    for (const IntegerTarget &target : targets) {
        const size_t destBytes = static_cast<size_t>(TensorDescriptor::getElementSizeInBytes(target.type));
        vector<uint8_t> cpu(count * destBytes);
        vector<uint8_t> gpu(count * destBytes);
        TypeConverter::convertType(const_cast<uint8_t *>(source.data()), cpu.data(), sourceType, target.type, count, stream, -1);
        TypeConverter::convertType(source_d, dest_d, sourceType, target.type, count, stream, 0);
        ASSERT_EQ(cudaMemcpyAsync(gpu.data(), dest_d, gpu.size(), cudaMemcpyDeviceToHost, stream.getStream()), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream.getStream()), cudaSuccess);

        size_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) {
            const double value = decodeFloatingSource(sourceType, source.data() + i * sourceBytes);
            long double expected = 0.0L;
            if (!std::isnan(value)) {
                expected = std::min(std::max(std::trunc(static_cast<long double>(value)), target.low), target.high);
            } else if (sourceType == DataType::FP64 || destBytes == 8) {
                const long double documented = std::ldexp(target.low < 0 ? -1.0L : 1.0L, destBytes == 8 ? 63 : 31);
                expected = std::min(std::max(documented, target.low), target.high);
            }
            const long double results[2] = {readInteger(target.type, cpu.data() + i * destBytes),
                                            readInteger(target.type, gpu.data() + i * destBytes)};
            for (int side = 0; side < 2; ++side) {
                if (results[side] != expected && mismatches++ < 8) {
                    ADD_FAILURE() << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                  << TensorDescriptor::getElementTypeName(target.type) << (side == 0 ? " cpu" : " gpu") << " of "
                                  << value << ": got " << results[side] << ", expected " << expected;
                }
            }
        }
        EXPECT_EQ(mismatches, 0u) << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                  << TensorDescriptor::getElementTypeName(target.type);
    }

    EXPECT_EQ(cudaFree(source_d), cudaSuccess);
    EXPECT_EQ(cudaFree(dest_d), cudaSuccess);
}

// NaN, infinities, signed zeros, subnormals, fractions, every power of two from 2^7 to 2^64 with its neighbors
// (+-0.5, +-1 and the adjacent representable values), and the format's largest values, both signs.
template <typename Float>
vector<uint8_t> floatToIntegerSpecials() {
    vector<Float> values = {std::numeric_limits<Float>::quiet_NaN(),
                            std::numeric_limits<Float>::infinity(),
                            std::numeric_limits<Float>::denorm_min(),
                            std::numeric_limits<Float>::max(),
                            std::numeric_limits<Float>::lowest(),
                            Float(0.0),
                            Float(-0.0),
                            Float(1e30)};
    for (const double fraction : {0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 99.9})
        values.push_back(static_cast<Float>(fraction));
    for (const int exponent : {7, 8, 15, 16, 31, 32, 53, 63, 64}) {
        const Float power = std::ldexp(Float(1), exponent);
        for (const Float offset : {Float(-1), Float(-0.5), Float(0), Float(0.5), Float(1)})
            values.push_back(power + offset);
        values.push_back(std::nextafter(power, Float(0)));
        values.push_back(std::nextafter(power, std::numeric_limits<Float>::infinity()));
    }
    const size_t positives = values.size();
    for (size_t i = 0; i < positives; ++i)
        values.push_back(-values[i]);
    vector<uint8_t> bytes(values.size() * sizeof(Float));
    memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

vector<uint8_t> everyBitPattern(size_t bytesPerElement) {
    const size_t count = size_t(1) << (8 * bytesPerElement);
    vector<uint8_t> bytes(count * bytesPerElement);
    for (size_t i = 0; i < count; ++i)
        memcpy(bytes.data() + i * bytesPerElement, &i, bytesPerElement);
    return bytes;
}

}  // namespace

// Every float -> integer conversion truncates toward zero and saturates (infinities included) on the CPU and the GPU;
// NaN gives what PTX documents on both (see expectFloatToIntegerRule). FP16, BF16 and both FP8
// kinds take every bit pattern; FP32 and FP64 take the special values.
TEST(TypeConverter, FloatToIntegerTruncatesAndSaturatesOnCpuAndGpu) {
    expectFloatToIntegerRule(DataType::FP32, floatToIntegerSpecials<float>());
    expectFloatToIntegerRule(DataType::FP64, floatToIntegerSpecials<double>());
    expectFloatToIntegerRule(DataType::FP16, everyBitPattern(2));
    expectFloatToIntegerRule(DataType::BF16, everyBitPattern(2));
    expectFloatToIntegerRule(DataType::FP8_E4M3, everyBitPattern(1));
    expectFloatToIntegerRule(DataType::FP8_E5M2, everyBitPattern(1));
}
