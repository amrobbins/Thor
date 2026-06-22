#include "Map.h"
#include "DataTypeDispatch.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <type_traits>

namespace {

template <typename T>
__device__ double toAccum(T value) {
    if constexpr (std::is_same<T, double>::value) {
        return value;
    } else if constexpr (std::is_integral<T>::value || std::is_same<T, bool>::value) {
        return static_cast<double>(value);
    } else {
        return static_cast<double>(static_cast<float>(value));
    }
}

template <typename T>
__device__ T fromAccum(double value) {
    if constexpr (std::is_same<T, double>::value) {
        return value;
    } else if constexpr (std::is_integral<T>::value || std::is_same<T, bool>::value) {
        return static_cast<T>(value);
    } else {
        return T(static_cast<float>(value));
    }
}

template <typename INDEX_TYPE, typename ELEMENT_TYPE>
__global__ void mapKernel(ELEMENT_TYPE *dest, ELEMENT_TYPE *source, INDEX_TYPE *mapping, INDEX_TYPE numDestElements) {
    INDEX_TYPE index = blockIdx.x * 512 + threadIdx.x;

    if (index >= numDestElements)
        return;
    dest[index] = source[mapping[index]];
    index += 256;
    if (index >= numDestElements)
        return;
    dest[index] = source[mapping[index]];
}

template <typename INDEX_TYPE, typename ELEMENT_TYPE>
void launchMapTyped(void *dest_d, void *source_d, INDEX_TYPE *mapping_d, INDEX_TYPE numDestElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numDestElements + 511) / 512);
    // in place is not supported
    THOR_THROW_IF_FALSE(dest_d != source_d);
    mapKernel<INDEX_TYPE, ELEMENT_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(
        reinterpret_cast<ELEMENT_TYPE *>(dest_d), reinterpret_cast<ELEMENT_TYPE *>(source_d), mapping_d, numDestElements);
}

template <typename INDEX_TYPE, typename ELEMENT_TYPE>
__global__ void mapNInto1Kernel(unsigned int N, ELEMENT_TYPE *dest, ELEMENT_TYPE *source, INDEX_TYPE *mapping, INDEX_TYPE numMapElements) {
    INDEX_TYPE mapIndex = (N + 1) * (blockIdx.x * 512 + threadIdx.x);
    if (mapIndex / (N + 1) >= numMapElements)
        return;
    INDEX_TYPE destIndex = mapping[mapIndex];

    double accum = 0.0;
    for (unsigned int i = 1; i <= N; ++i) {
        accum += toAccum(source[mapping[mapIndex + i]]);
    }
    dest[destIndex] = fromAccum<ELEMENT_TYPE>(accum);

    mapIndex += (N + 1) * 256;
    if (mapIndex / (N + 1) >= numMapElements)
        return;
    destIndex = mapping[mapIndex];
    accum = 0.0;
    for (unsigned int i = 1; i <= N; ++i) {
        accum += toAccum(source[mapping[mapIndex + i]]);
    }
    dest[destIndex] = fromAccum<ELEMENT_TYPE>(accum);
}

template <typename INDEX_TYPE, typename ELEMENT_TYPE>
void launchMapNInto1Typed(unsigned int N, void *dest_d, void *source_d, INDEX_TYPE *mapping_d, INDEX_TYPE numMapElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numMapElements + 511) / 512);
    // in place is not supported
    THOR_THROW_IF_FALSE(dest_d != source_d);
    mapNInto1Kernel<INDEX_TYPE, ELEMENT_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(
        N, reinterpret_cast<ELEMENT_TYPE *>(dest_d), reinterpret_cast<ELEMENT_TYPE *>(source_d), mapping_d, numMapElements);
}

template <typename INDEX_TYPE>
struct LaunchMapFunctor {
    void *dest_d;
    void *source_d;
    INDEX_TYPE *mapping_d;
    INDEX_TYPE numDestElements;
    Stream stream;

    template <typename ELEMENT_TYPE>
    void operator()() const {
        launchMapTyped<INDEX_TYPE, ELEMENT_TYPE>(dest_d, source_d, mapping_d, numDestElements, stream);
    }
};

template <typename INDEX_TYPE>
struct LaunchMapNInto1Functor {
    unsigned int N;
    void *dest_d;
    void *source_d;
    INDEX_TYPE *mapping_d;
    INDEX_TYPE numMapElements;
    Stream stream;

    template <typename ELEMENT_TYPE>
    void operator()() const {
        launchMapNInto1Typed<INDEX_TYPE, ELEMENT_TYPE>(N, dest_d, source_d, mapping_d, numMapElements, stream);
    }
};

}  // namespace

template <typename INDEX_TYPE>
void launchMap(void *dest_d,
               void *source_d,
               INDEX_TYPE *mapping_d,
               INDEX_TYPE numDestElements,
               ThorImplementation::DataType dataType,
               Stream stream) {
    ThorImplementation::MiscTensorOperationSupport::dispatchTensorDataType(
        dataType, LaunchMapFunctor<INDEX_TYPE>{dest_d, source_d, mapping_d, numDestElements, stream});
}

template void launchMap<uint8_t>(void *dest,
                                 void *source,
                                 uint8_t *mapping,
                                 uint8_t numDestElements,
                                 ThorImplementation::DataType dataType,
                                 Stream stream);
template void launchMap<uint16_t>(void *dest,
                                  void *source,
                                  uint16_t *mapping,
                                  uint16_t numDestElements,
                                  ThorImplementation::DataType dataType,
                                  Stream stream);
template void launchMap<uint32_t>(void *dest,
                                  void *source,
                                  uint32_t *mapping,
                                  uint32_t numDestElements,
                                  ThorImplementation::DataType dataType,
                                  Stream stream);
template void launchMap<uint64_t>(void *dest,
                                  void *source,
                                  uint64_t *mapping,
                                  uint64_t numDestElements,
                                  ThorImplementation::DataType dataType,
                                  Stream stream);

template <typename INDEX_TYPE>
void launchMapNInto1(unsigned int N,
                     void *dest_d,
                     void *source_d,
                     INDEX_TYPE *mapping_d,
                     INDEX_TYPE numMapElements,
                     ThorImplementation::DataType dataType,
                     Stream stream) {
    ThorImplementation::MiscTensorOperationSupport::dispatchTensorDataType(
        dataType, LaunchMapNInto1Functor<INDEX_TYPE>{N, dest_d, source_d, mapping_d, numMapElements, stream});
}

template void launchMapNInto1<uint8_t>(unsigned int N,
                                       void *dest,
                                       void *source,
                                       uint8_t *mapping,
                                       uint8_t numDestElements,
                                       ThorImplementation::DataType dataType,
                                       Stream stream);
template void launchMapNInto1<uint16_t>(unsigned int N,
                                        void *dest,
                                        void *source,
                                        uint16_t *mapping,
                                        uint16_t numDestElements,
                                        ThorImplementation::DataType dataType,
                                        Stream stream);
template void launchMapNInto1<uint32_t>(unsigned int N,
                                        void *dest,
                                        void *source,
                                        uint32_t *mapping,
                                        uint32_t numDestElements,
                                        ThorImplementation::DataType dataType,
                                        Stream stream);
template void launchMapNInto1<uint64_t>(unsigned int N,
                                        void *dest,
                                        void *source,
                                        uint64_t *mapping,
                                        uint64_t numDestElements,
                                        ThorImplementation::DataType dataType,
                                        Stream stream);
