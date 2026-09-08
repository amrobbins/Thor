#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"

#include <cstdint>

void launchSplit(void *dest[],
                 void *source,
                 uint64_t outerSlices,
                 uint32_t numDestArrays,
                 uint64_t packedSliceBytes,
                 const ConcatenateSpanGeometry spanGeometry[],
                 Stream stream);
