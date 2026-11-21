#pragma once

// Build switches for storage dtypes, in a small header so callers that only need them do not parse the
// CubDevicePrimitives API.

#ifndef THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
// UINT64 segment offsets for the segmented reducers (canonical ragged row partitions are UINT32 or UINT64).
#define THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS 1
#endif

// FP8 storage types (E4M3, E5M2), for the reducer's FP8 kernels.
#ifndef THOR_CUB_ENABLE_FP8_TYPES
#define THOR_CUB_ENABLE_FP8_TYPES 1
#endif
