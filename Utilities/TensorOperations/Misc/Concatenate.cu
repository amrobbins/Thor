#include "Concatenate.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Misc/ConcatenateSpanCopy.cuh"

#include <limits>
#include <stdexcept>

namespace {

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        throw std::invalid_argument(what);
    return lhs * rhs;
}

}  // namespace

std::vector<ConcatenateSpanGeometry> buildConcatenateSpanGeometry(
    std::size_t elementSizeBytes,
    uint64_t innerElements,
    const std::vector<uint64_t>& axisElementsPerArray) {
    if (elementSizeBytes == 0 || innerElements == 0)
        throw std::invalid_argument("Dense concatenate requires non-zero static span geometry.");
    if (axisElementsPerArray.empty())
        throw std::invalid_argument("Dense concatenate requires at least one array.");

    std::vector<ConcatenateSpanGeometry> geometry(axisElementsPerArray.size());
    uint64_t packedOffsetBytes = 0;
    for (size_t i = 0; i < axisElementsPerArray.size(); ++i) {
        const uint64_t spanElements = checkedMultiply(axisElementsPerArray[i], innerElements,
                                                      "Dense concatenate span element count overflow.");
        const uint64_t spanBytes = checkedMultiply(spanElements, static_cast<uint64_t>(elementSizeBytes),
                                                   "Dense concatenate span byte count overflow.");
        geometry[i] = ConcatenateSpanGeometry{spanBytes, packedOffsetBytes};
        if (spanBytes > std::numeric_limits<uint64_t>::max() - packedOffsetBytes)
            throw std::invalid_argument("Dense concatenate packed slice byte count overflow.");
        packedOffsetBytes += spanBytes;
    }
    return geometry;
}

void launchConcatenate(void *dest,
                       void *source[],
                       uint64_t outerSlices,
                       uint32_t numSourceArrays,
                       uint64_t packedSliceBytes,
                       const ConcatenateSpanGeometry spanGeometry[],
                       Stream stream) {
    ScopedGpu scopedGpu(stream.getGpuNum());
    ThorConcatenateSpanCopy::launch<true>(
        dest, source, outerSlices, numSourceArrays, packedSliceBytes, spanGeometry, stream);
}

void launchConcatenateWithSpansPerCtaForBenchmark(void *dest,
                                                  void *source[],
                                                  uint64_t outerSlices,
                                                  uint32_t numSourceArrays,
                                                  uint64_t packedSliceBytes,
                                                  const ConcatenateSpanGeometry spanGeometry[],
                                                  uint32_t spansPerCta,
                                                  Stream stream) {
    ScopedGpu scopedGpu(stream.getGpuNum());
    ThorConcatenateSpanCopy::launchWithSpansPerCtaForBenchmark<true>(
        dest, source, outerSlices, numSourceArrays, packedSliceBytes, spanGeometry, spansPerCta, stream);
}
