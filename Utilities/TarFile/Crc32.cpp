#include "Crc32.h"

#ifndef THOR_REQUIRE_ZLIBNG
#error "This build requires zlib-ng (<zlib-ng.h>)."
#endif

#include <zlib-ng.h>

#ifndef THOR_ZLIBNG_HAS_CRC32_Z
#error "THOR_ZLIBNG_HAS_CRC32_Z is not defined. CMake must set it to 1."
#endif

#if THOR_ZLIBNG_HAS_CRC32_Z != 1
#error "This build requires zlib-ng's zng_crc32_z (size_t length). Configure/build zlib-ng appropriately."
#endif

uint32_t crc32_ieee(uint32_t crc_accum, const uint8_t* chunk, size_t len) { return zng_crc32_z(crc_accum, chunk, len); }
