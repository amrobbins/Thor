#include "Crc32c.h"

namespace thor_file {

uint32_t Crc32c::update(uint32_t crc, const uint8_t* data, size_t len) {
    // const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint8_t* p = data;
    uint64_t c = static_cast<uint64_t>(crc);

    // Handle unaligned bytes until 8-byte aligned
    while (len && (reinterpret_cast<std::uintptr_t>(p) & 7u)) {
        c = _mm_crc32_u8(static_cast<uint32_t>(c), *p);
        ++p;
        --len;
    }

    // Process 8 bytes at a time
    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(p);
    while (len >= 8) {
        // If any issue with alignment, there shouldn't be, use memcpy instead:
        // std::memcpy(&v, p, sizeof(v));  // safe for unaligned
        const uint64_t v = *p64;
        c = _mm_crc32_u64(c, v);
        ++p64;
        len -= 8;
    }

    // Handle any trailing bytes after exhausting the 8 byte aligned section
    p = reinterpret_cast<const uint8_t*>(p64);
    while (len) {
        c = _mm_crc32_u8(static_cast<uint32_t>(c), *p);
        ++p;
        --len;
    }

    return static_cast<uint32_t>(c);
}

}  // namespace thor_file
