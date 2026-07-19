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
