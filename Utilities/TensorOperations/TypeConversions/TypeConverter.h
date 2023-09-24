#pragma once

#include "DeepLearning/Implementation/Tensor/PackedBoolean.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <type_traits>

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace ThorImplementation {

// In-place conversions are supported and are about 20% slower than non-in-place conversions
// To do an inplace conversion, just send the same pointer to the source and dest.
class TypeConverter {
   public:
    // Schedules a type conversion on the specified device at the back of the stream.
    // Conversion will be performed on the specified device, both memory pointers must be local to the device.
    // If the device is a GPU, then the stream must belong to the device.
    // All tensor types are convertible to all tensor types, the conversion rules follow the C++ rule for the base type
    static void convertType(void *source,
                            void *dest,
                            TensorDescriptor::DataType sourceDataType,
                            TensorDescriptor::DataType destDataType,
                            long numElements,
                            Stream stream,
                            int deviceNum);

    static constexpr int THREADS_PER_SYNCED_BLOCK = 1024;
    static constexpr int THREADS_PER_UNSYNCED_BLOCK = 256;

   private:
    struct Args {
        void *source;
        void *dest;
        long numElements;
        TensorDescriptor::DataType sourceDataType;
        TensorDescriptor::DataType destDataType;

        Args(void *source,
             void *dest,
             TensorDescriptor::DataType sourceDataType,
             TensorDescriptor::DataType destDataType,
             long numElements) {
            this->source = source;
            this->dest = dest;
            this->numElements = numElements;
            this->sourceDataType = sourceDataType;
            this->destDataType = destDataType;
        }
    };

    // Convert on CPU between two types. In place or out of place is supported.
    static void CUDART_CB cpuConvertType(void *data);

    template <typename FROM_TYPE, typename TO_TYPE>
    static void cpuConvertTypeImpl(FROM_TYPE *source, TO_TYPE *dest, long numElements);
    template <typename TO_TYPE>
    static void cpuConvertTypeFromPackedBooleanImpl(void *source, TO_TYPE *dest, long numElements);
    template <typename FROM_TYPE>
    static void cpuConvertTypeToPackedBooleanImpl(FROM_TYPE *source, void *dest, long numElements);
    template <typename FROM_TYPE>
    static void cpuConvertTypeFromIntegralToHalfImpl(FROM_TYPE *source, half *dest, long numElements);
    static void cpuConvertTypeFromPackedBooleanToHalfImpl(void *source, half *dest, long numElements);

    // Convert on GPU between two types. In place or out of place is supported.
    static void gpuConvertType(void *source_d,
                               void *dest_d,
                               TensorDescriptor::DataType sourceDataType,
                               TensorDescriptor::DataType destDataType,
                               long numElements,
                               Stream stream);

    template <typename FROM_TYPE, typename TO_TYPE>
    static void gpuConvertTypeImpl(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
    template <typename FROM_TYPE>
    static void gpuConvertTypeToPackedBooleanImpl(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream);
    template <typename TO_TYPE>
    static void gpuConvertTypeFromPackedBooleanImpl(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream);

    template <typename FROM_TYPE, typename TO_TYPE>
    static void convertToSmallerElementsInPlaceOnGpu(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
    template <typename FROM_TYPE, typename TO_TYPE>
    static void convertToBiggerElementsInPlaceOnGpu(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
    template <typename FROM_TYPE>
    static void convertToSmallerElementsInPlaceOnGpu_toPackedBoolean(FROM_TYPE *source_d, uint8_t *dest_d, long numElements, Stream stream);
    template <typename TO_TYPE>
    static void convertToBiggerElementsInPlaceOnGpu_fromPackedBoolean(uint8_t *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
};

}  // namespace ThorImplementation
