#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Concatenate the arrays into dest: for every outer slice, array i holds spanElements[i] elements of
// sourceDataTypes[i], and dest holds the spans in order as elements of destDataType. outerSlices is the number of
// valid coordinates before the concatenation axis; for a non-batch physical axis this can be restricted to the valid
// batch prefix. An array of another type than dest's is converted as it moves, element by element as TypeConverter
// converts (the same bits), by the one kernel that reads every array and writes dest: no array is converted ahead of
// the concatenate and nothing is staged. Arrays of dest's type are copied; when every array has it, the launches are
// byte copies. sources[i] may be null only for a span of no elements. Every base is on a boundary of its element type.
void launchConcatenate(void *dest,
                       ThorImplementation::DataType destDataType,
                       const std::vector<void *> &sources,
                       const std::vector<ThorImplementation::DataType> &sourceDataTypes,
                       uint64_t outerSlices,
                       const std::vector<uint64_t> &spanElements,
                       Stream stream);
