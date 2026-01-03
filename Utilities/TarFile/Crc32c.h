#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <nmmintrin.h>

namespace thor_file {

struct Crc32c {
    // Start value for streaming CRC32C
    static constexpr uint32_t kInit = 0xFFFFFFFFu;

    // Update a running CRC with new bytes
    static uint32_t update(uint32_t crc, const void* data, size_t len);

    // Convenience: compute CRC32C(data)
    static uint32_t compute(const void* data, size_t len) { return finalize(update(kInit, data, len)); }

    // Finalize a streaming CRC
    static uint32_t finalize(uint32_t crc) { return crc ^ 0xFFFFFFFFu; }
};

}  // namespace thor_file
