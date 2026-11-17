#pragma once

// Build switches for storage dtypes, in a small header so callers that only need them do not parse the
// CubDevicePrimitives API.

// 64-bit scan value types (UINT64, FP64). Off: Thor does not instantiate them.
#ifndef THOR_CUB_ENABLE_64BIT_TYPES
#define THOR_CUB_ENABLE_64BIT_TYPES 0
#endif

#ifndef THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
// Segment offsets are structural indices, not scan values. CUB accepts a
// 64-bit offset iterator even when Thor intentionally avoids instantiating
// 64-bit scan/reduction value types, and canonical ragged row partitions
// support both UINT32 and UINT64. Keep this capability independent from
// THOR_CUB_ENABLE_64BIT_TYPES.
#define THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS 1
#endif

// FP8 storage types (E4M3, E5M2), for the reducer's FP8 kernels.
#ifndef THOR_CUB_ENABLE_FP8_TYPES
#define THOR_CUB_ENABLE_FP8_TYPES 1
#endif
