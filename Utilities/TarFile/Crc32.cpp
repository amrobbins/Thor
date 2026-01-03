#include "Crc32.h"

// For first chunk, set crc_accum to 0.

#if __has_include(<zlib-ng.h>)
#include <zlib-ng.h>
uint32_t crc32_ieee(const uint32_t crc_accum, const uint8_t* chunk, const size_t len) { return zng_crc32_z(crc_accum, chunk, len); }
#elif __has_include(<zlib.h>)
#include <zlib.h>
uint32_t crc32_ieee(const uint32_t crc_accum, const uint8_t* chunk, const size_t len) { return crc32_z(crc_accum, chunk, len); }
#else
#error "No zlib-ng.h or zlib.h found"
#endif
