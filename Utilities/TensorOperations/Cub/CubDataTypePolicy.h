#pragma once

// Shared build policy for CUB storage dtypes. Keep this header intentionally
// small so callers that only need the policy do not have to parse the complete
// CubDevicePrimitives API.
#ifndef THOR_CUB_ENABLE_64BIT_TYPES
#ifdef THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS
#define THOR_CUB_ENABLE_64BIT_TYPES THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS
#else
#define THOR_CUB_ENABLE_64BIT_TYPES 0
#endif
#endif

#ifndef THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
// Segment offsets are structural indices, not scan values. CUB accepts a
// 64-bit offset iterator even when Thor intentionally avoids instantiating
// 64-bit scan/reduction value types, and canonical ragged row partitions
// support both UINT32 and UINT64. Keep this capability independent from
// THOR_CUB_ENABLE_64BIT_TYPES.
#define THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS 1
#endif

#ifndef THOR_CUB_ENABLE_FP8_TYPES
#ifdef THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS
#define THOR_CUB_ENABLE_FP8_TYPES THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS
#else
#define THOR_CUB_ENABLE_FP8_TYPES 1
#endif
#endif

#ifndef THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS
#define THOR_CUB_RADIX_SORT_ENABLE_64BIT_KEYS THOR_CUB_ENABLE_64BIT_TYPES
#endif

#ifndef THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS
#define THOR_CUB_RADIX_SORT_ENABLE_FP8_KEYS THOR_CUB_ENABLE_FP8_TYPES
#endif
