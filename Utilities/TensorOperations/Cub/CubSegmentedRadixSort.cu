#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

// Segmented radix sort is split by key/pair and plan/run operation families so
// Ninja can compile the especially heavy CUB key/value/offset template
// instantiations in parallel.
