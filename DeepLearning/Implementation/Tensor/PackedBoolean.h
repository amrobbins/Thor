#pragma once

#include <cstdint>

class PackedBoolean {
   public:
    static inline bool getElement(unsigned long index, void *data) { return (((uint8_t *)data)[index >> 3] >> (index & 0x7)) & 0x1; }

    static inline void setElement(bool value, unsigned long index, void *data) {
        if (value)
            ((uint8_t *)data)[index >> 3] |= 0x1 << (index & 0x7);
        else
            ((uint8_t *)data)[index >> 3] &= ~(0x1 << (index & 0x7));
    }
};
