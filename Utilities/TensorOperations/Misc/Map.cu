#include "Map.h"

template <typename INDEX_TYPE>
__global__ void map(half *dest, half *source, INDEX_TYPE *mapping, INDEX_TYPE numDestElements) {
    INDEX_TYPE index = blockIdx.x * 512 + threadIdx.x;

    if (index >= numDestElements)
        return;
    dest[index] = source[mapping[index]];
    index += 256;
    if (index >= numDestElements)
        return;
    dest[index] = source[mapping[index]];
}

template <typename INDEX_TYPE>
void launchMap(half *dest_d, half *source_d, INDEX_TYPE *mapping_d, INDEX_TYPE numDestElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numDestElements + 511) / 512);
    // in place is not supported
    assert(dest_d != source_d);
    map<INDEX_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(dest_d, source_d, mapping_d, numDestElements);
}

template void launchMap<uint8_t>(half *dest, half *source, uint8_t *mapping, uint8_t numDestElements, Stream stream);
template void launchMap<uint16_t>(half *dest, half *source, uint16_t *mapping, uint16_t numDestElements, Stream stream);
template void launchMap<uint32_t>(half *dest, half *source, uint32_t *mapping, uint32_t numDestElements, Stream stream);
template void launchMap<uint64_t>(half *dest, half *source, uint64_t *mapping, uint64_t numDestElements, Stream stream);

template <typename INDEX_TYPE>
__global__ void mapNInto1(unsigned int N, half *dest, half *source, INDEX_TYPE *mapping, INDEX_TYPE numMapElements) {
    INDEX_TYPE mapIndex = (N + 1) * (blockIdx.x * 512 + threadIdx.x);
    if (mapIndex / (N + 1) >= numMapElements)
        return;
    INDEX_TYPE destIndex = mapping[mapIndex];

    float accum = 0.0f;
    for (unsigned int i = 1; i <= N; ++i) {
        accum += (float)source[mapping[mapIndex + i]];
    }
    dest[destIndex] = (half)accum;

    mapIndex += (N + 1) * 256;
    if (mapIndex / (N + 1) >= numMapElements)
        return;
    destIndex = mapping[mapIndex];
    accum = 0.0f;
    for (unsigned int i = 1; i <= N; ++i) {
        accum += (float)source[mapping[mapIndex + i]];
    }
    dest[destIndex] = (half)accum;
}

template <typename INDEX_TYPE>
void launchMapNInto1(unsigned int N, half *dest_d, half *source_d, INDEX_TYPE *mapping_d, INDEX_TYPE numMapElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numMapElements + 511) / 512);
    // in place is not supported
    assert(dest_d != source_d);
    mapNInto1<INDEX_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(N, dest_d, source_d, mapping_d, numMapElements);
}

template void launchMapNInto1<uint8_t>(unsigned int N, half *dest, half *source, uint8_t *mapping, uint8_t numDestElements, Stream stream);
template void launchMapNInto1<uint16_t>(
    unsigned int N, half *dest, half *source, uint16_t *mapping, uint16_t numDestElements, Stream stream);
template void launchMapNInto1<uint32_t>(
    unsigned int N, half *dest, half *source, uint32_t *mapping, uint32_t numDestElements, Stream stream);
template void launchMapNInto1<uint64_t>(
    unsigned int N, half *dest, half *source, uint64_t *mapping, uint64_t numDestElements, Stream stream);
