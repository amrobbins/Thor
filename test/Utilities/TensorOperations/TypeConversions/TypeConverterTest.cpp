#include "Thor.h"

#include <stdio.h>

#include "gtest/gtest.h"

using namespace ThorImplementation;
using namespace std;

TEST(RandomTestConversions, CpuAllConversionsOutOfPlace) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    for (TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP16;
         sourceDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
         sourceDataType = (TensorDescriptor::DataType)((int)sourceDataType + 1)) {
        for (TensorDescriptor::DataType destDataType = TensorDescriptor::DataType::FP16;
             destDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
             destDataType = (TensorDescriptor::DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            int NUM_ELEMENTS = 1 + (rand() % 2000);
            if (rand() % 20 == 0)
                NUM_ELEMENTS = 1000000 + rand() % 1000000;

            void *source;
            void *dest;
            int numSourceBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, sourceDataType);
            int numDestBytes = TensorDescriptor::getArraySizeInBytes(NUM_ELEMENTS, destDataType);
            cudaStatus = cudaHostAlloc(&source, numSourceBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);
            cudaStatus = cudaHostAlloc(&dest, numDestBytes, cudaHostAllocWriteCombined);
            assert(cudaStatus == cudaSuccess);

            bool destDataTypeIsSigned = TensorDescriptor::isSignedType(destDataType);
            if (sourceDataType == TensorDescriptor::DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (50 - (rand() % 100)) * 0.1;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 100) * 0.1;
                    }
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else if (sourceDataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    PackedBoolean::setElement(rand() % 2 ? true : false, i, source);
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

    for (TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP16;
         sourceDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
         sourceDataType = (TensorDescriptor::DataType)((int)sourceDataType + 1)) {
        for (TensorDescriptor::DataType destDataType = TensorDescriptor::DataType::FP16;
             destDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
             destDataType = (TensorDescriptor::DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            int NUM_ELEMENTS = 1 + (rand() % 2000);
            if (rand() % 30 == 0)
                NUM_ELEMENTS = 1000000 + rand() % 1000000;

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
            if (sourceDataType == TensorDescriptor::DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (50 - (rand() % 100)) * 0.1;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 100) * 0.1;
                    }
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else if (sourceDataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    PackedBoolean::setElement(rand() % 2 ? true : false, i, source);
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

    for (TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP16;
         sourceDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
         sourceDataType = (TensorDescriptor::DataType)((int)sourceDataType + 1)) {
        for (TensorDescriptor::DataType destDataType = TensorDescriptor::DataType::FP16;
             destDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
             destDataType = (TensorDescriptor::DataType)((int)destDataType + 1)) {
            if (sourceDataType == destDataType)
                continue;

            // printf("%s <- %s\n",
            //       TensorDescriptor::getElementTypeName(destDataType).c_str(),
            //       TensorDescriptor::getElementTypeName(sourceDataType).c_str());

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
            if (sourceDataType == TensorDescriptor::DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (50 - (rand() % 100)) * 0.1;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 100) * 0.1;
                    }
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else if (sourceDataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    PackedBoolean::setElement(rand() % 2 ? true : false, i, source);
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

                // if(sourceDataType == TensorDescriptor::DataType::PACKED_BOOLEAN && destDataType == TensorDescriptor::DataType::FP16)
                //    printf("[%5d] s:%lf d:%lf\n", i, sourceVal, destVal);

                if (TensorDescriptor::isIntegralType(sourceDataType)) {
                    if (TensorDescriptor::isIntegralType(destDataType)) {
                        if (sourceVal != destVal)
                            printf("[%d] source %f dest %f\n", i, sourceVal, destVal);
                        ASSERT_EQ(sourceVal, destVal);
                    } else {
                        if (abs(sourceVal - destVal) >= 0.1)
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

    for (TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP16;
         sourceDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
         sourceDataType = (TensorDescriptor::DataType)((int)sourceDataType + 1)) {
        for (TensorDescriptor::DataType destDataType = TensorDescriptor::DataType::FP16;
             destDataType <= TensorDescriptor::DataType::PACKED_BOOLEAN;
             destDataType = (TensorDescriptor::DataType)((int)destDataType + 1)) {
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
            if (sourceDataType == TensorDescriptor::DataType::FP16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (50 - (rand() % 100)) * 0.1;
                    }
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i) {
                        ((half *)source)[i] = (rand() % 100) * 0.1;
                    }
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((float *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::FP64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (50 - (rand() % 100)) * 0.1;
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((double *)source)[i] = (rand() % 100) * 0.1;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT8) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int8_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT16) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int16_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT32) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int32_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::INT64) {
                if (destDataTypeIsSigned) {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = 50 - (rand() % 100);
                } else {
                    for (int i = 0; i < NUM_ELEMENTS; ++i)
                        ((int64_t *)source)[i] = rand() % 100;
                }
            } else if (sourceDataType == TensorDescriptor::DataType::UINT8) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint8_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT16) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint16_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT32) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint32_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::UINT64) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((uint64_t *)source)[i] = (rand() % 100);
            } else if (sourceDataType == TensorDescriptor::DataType::BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    ((bool *)source)[i] = rand() % 2 ? true : false;
            } else if (sourceDataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
                for (int i = 0; i < NUM_ELEMENTS; ++i)
                    PackedBoolean::setElement(rand() % 2 ? true : false, i, source);
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

                // if(sourceDataType == TensorDescriptor::DataType::FP16 && destDataType == TensorDescriptor::DataType::FP32)
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
