#pragma once

#include <cstddef>
#include <cstdint>

uint32_t crc32_ieee(const uint32_t crc_accum, const uint8_t* chunk, const size_t len);
