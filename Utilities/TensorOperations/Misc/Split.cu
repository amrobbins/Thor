#include "Split.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Misc/ConcatenateSpanCopy.cuh"

void launchSplit(void *dest[],
                 void *source,
                 uint64_t outerSlices,
                 uint32_t numDestArrays,
                 uint64_t packedSliceBytes,
                 const ConcatenateSpanGeometry spanGeometry[],
                 Stream stream) {
    ScopedGpu scopedGpu(stream.getGpuNum());
    ThorConcatenateSpanCopy::launch<false>(
        source, dest, outerSlices, numDestArrays, packedSliceBytes, spanGeometry, stream);
}
