#pragma once

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/Stream.h"

#include <omp.h>
#include <algorithm>
#include <type_traits>

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace ThorImplementation {

class Tensor;

// An in-place conversion passes the same pointer as source and dest. It is supported when the destination element is
// at least as large as the source element (widening, or the same size). In-place narrowing is refused: narrow out of
// place into a separate destination, which is faster.
class TypeConverter {
   public:
    // Schedules a type conversion on the specified device at the back of the stream.
    // Conversion will be performed on the specified device, both memory pointers must be local to the device.
    // If the device is a GPU, then the stream must belong to the device.
    // All tensor types are convertible to all tensor types, the conversion rules follow the C++ rule for the base type
    // Low-level raw-memory conversion. For asynchronous CPU conversion, callers
    // must keep source and destination memory alive until the stream reaches the
    // callback. Tensor-backed callers should use the Tensor overload below.
    static void convertType(void *source,
                            void *dest,
                            DataType sourceDataType,
                            DataType destDataType,
                            long numElements,
                            Stream stream,
                            int deviceNum);

    // Tensor-backed CPU conversion. Callback state retains both tensors until
    // the callback has completed, so the asynchronous operation owns the
    // storage lifetime it depends on without adding ownership traffic to GPU
    // conversion launch paths.
    static void convertType(const Tensor &source,
                            const Tensor &dest,
                            DataType sourceDataType,
                            DataType destDataType,
                            long numElements,
                            Stream stream);

    static constexpr int THREADS_PER_SYNCED_BLOCK = 1024;
    static constexpr int THREADS_PER_UNSYNCED_BLOCK = 256;

   private:
    struct Args;
    struct TensorArgs;

    // Convert on CPU between two types, out of place or in place when not narrowing.
    static void CUDART_CB cpuConvertType(void *data);

    template <typename FROM_TYPE, typename TO_TYPE>
    static void cpuConvertTypeImpl(FROM_TYPE *source, TO_TYPE *dest, long numElements);

    // Convert on GPU between two types, out of place or in place when not narrowing.
    static void gpuConvertType(void *source_d,
                               void *dest_d,
                               DataType sourceDataType,
                               DataType destDataType,
                               long numElements,
                               Stream stream);

    template <typename FROM_TYPE, typename TO_TYPE>
    static void gpuConvertTypeImpl(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);

    template <typename FROM_TYPE, typename TO_TYPE>
    static void convertToBiggerElementsInPlaceOnGpu(FROM_TYPE *source_d, TO_TYPE *dest_d, long numElements, Stream stream);
};

}  // namespace ThorImplementation
