#pragma once

// Shared build policy for CUB storage dtypes. Keep this header intentionally
// small so callers that only need the policy do not have to parse the complete
// CubDevicePrimitives API.
// Generic CUB wrappers that are not used by Thor production code are kept behind this
// build-time switch. Production currently uses the generic scan/arg-scan families;
// embedding radix sort/RLE uses its own deliberately narrow direct-CUB instantiations
// and is not controlled by this switch. Leave the future wrapper code in-tree so the
// surface can be restored without reconstructing it, but do not instantiate its CUB
// template cartesian in normal builds.
#ifndef THOR_FUTURE_CUB_OPS
#define THOR_FUTURE_CUB_OPS false
#endif

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
