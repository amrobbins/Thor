#pragma once

#include "Utilities/TensorOperations/Misc/Concatenate.h"

// The reverse of launchConcatenate: for every outer slice, array i takes its span of spanElements[i] elements of
// source, converted from sourceDataType to destDataTypes[i] as it moves. A null dests[i] skips array i (its elements
// are not read); a span of no elements is skipped too.
void launchSplit(const std::vector<void *> &dests,
                 const std::vector<ThorImplementation::DataType> &destDataTypes,
                 const void *source,
                 ThorImplementation::DataType sourceDataType,
                 uint64_t outerSlices,
                 const std::vector<uint64_t> &spanElements,
                 Stream stream);
